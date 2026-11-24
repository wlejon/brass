#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/target/msl_target.hpp>

using namespace brass;
using namespace brass::codegen;
using namespace brass::target;

namespace {

bool has(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

size_t count_of(const std::string& haystack, const std::string& needle) {
    size_t n = 0;
    for (size_t pos = haystack.find(needle); pos != std::string::npos;
         pos = haystack.find(needle, pos + 1)) {
        ++n;
    }
    return n;
}

Function* build_vec4_saxpy(Module& mod, const char* name) {
    Function* fn = mod.create_function(
        name,
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::f32(), Type::i32()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* in = b.add_block_param(entry, Type::ptr());
    Value* alpha = b.add_block_param(entry, Type::f32());
    Value* n_vec = b.add_block_param(entry, Type::i32());

    KernelBuilder kb(b);
    Value* gid = kb.global_tid_x();
    Value* stride = kb.mul(kb.nctaid_x(), kb.ntid_x());

    kb.for_range(gid, n_vec, stride, [&](Value* idx) {
        Value* byte_off = b.build_sext_i64(b.build_shl(idx, kb.const_i32(4))); // * 16 bytes
        Value* pin = kb.add(in, byte_off);
        Value* pout = kb.add(out, byte_off);
        Value* vx = kb.vload_f32x4(pin);
        Value* va = b.build_vbroadcast(Type::f32x4(), alpha);
        Value* vr = kb.vfma(vx, va, vx);
        kb.vstore_f32x4(pout, vr);
    });

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

} // namespace

TEST_CASE("MSL Target - Vector Elementwise f32x4 Kernel and Packed Params") {
    Module mod("msl_vec4");
    Function* fn = build_vec4_saxpy(mod, "saxpy_vec4");

    MslOptions opts;
    opts.workgroup_x = 128;
    MslKernel k = MslTarget::compile(*fn, opts);

    CHECK_EQ(k.entry_name, std::string("saxpy_vec4"));
    CHECK_EQ(k.num_buffers, 2u);
    CHECK_EQ(k.scalar_param_offsets.size(), 2u);
    CHECK_EQ(k.scalar_param_offsets[0], 0u);
    CHECK_EQ(k.scalar_param_offsets[1], 4u);
    CHECK_EQ(k.scalar_params_bytes, 8u);
    CHECK_EQ(k.workgroup_x, 128u);
    CHECK(!k.uses_shared_memory);
    CHECK_EQ(k.shared_memory_bytes, 0u);

    CHECK(has(k.source, "#include <metal_stdlib>"));
    CHECK(has(k.source, "using namespace metal;"));
    CHECK(has(k.source, "struct saxpy_vec4_Params"));
    CHECK(has(k.source, "device uint8_t* _b0 [[buffer(0)]]"));
    CHECK(has(k.source, "device uint8_t* _b1 [[buffer(1)]]"));
    CHECK(has(k.source, "constant saxpy_vec4_Params& _params [[buffer(2)]]"));
    CHECK(has(k.source, "[[thread_position_in_threadgroup]]"));
    CHECK(has(k.source, "[[threads_per_threadgroup]]"));
    CHECK(has(k.source, "[[threadgroup_position_in_grid]]"));
    CHECK(has(k.source, "[[threadgroups_per_grid]]"));
    CHECK(has(k.source, "float4"));
    CHECK(has(k.source, "fma("));
    CHECK(has(k.source, "while (true)"));
    CHECK(has(k.source, "if ("));
    CHECK(!has(k.source, "goto"));
}

TEST_CASE("MSL Target - RowNorm Reduction with Shared Memory, Barrier, and SIMD Shuffle") {
    Module mod("msl_rownorm");
    Function* fn = mod.create_function(
        "rownorm_row",
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::f32()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* in = b.add_block_param(entry, Type::ptr());
    Value* dim = b.add_block_param(entry, Type::i32());
    Value* eps = b.add_block_param(entry, Type::f32());

    KernelBuilder kb(b);
    Value* row = kb.ctaid_x();
    Value* tid = kb.tid_x();
    Value* ntid = kb.ntid_x();
    Value* row_off = b.build_sext_i64(kb.mul(row, b.build_shl(dim, kb.const_i32(2))));
    Value* x_row = kb.add(in, row_off);
    Value* y_row = kb.add(out, row_off);

    Value* smem = kb.shared_alloc_f32(32);

    Value* sq_sum = kb.for_range_reduce(
        tid, dim, ntid, kb.const_f32(0.0f),
        [&](Value* i, Value* acc) {
            Value* px = kb.add(x_row, b.build_sext_i64(b.build_shl(i, kb.const_i32(2))));
            Value* x = kb.load_f32(px);
            return b.build_fma_f32(x, x, acc);
        });

    Value* warp_sq = kb.warp_reduce_sum_f32(sq_sum);
    kb.shared_store_f32_indexed(smem, kb.warp_id(), warp_sq);
    kb.sync();

    Value* lane = kb.lane_id();
    Value* s_val = kb.shared_load_f32_indexed(smem, b.build_and(lane, kb.const_i32(7)));
    Value* bfly = kb.shfl_bfly_f32(s_val, 1u);
    Value* rstd = kb.rsqrt_approx(
        b.build_add(b.build_mul(bfly, kb.const_f32(1.0f)), eps));

    kb.for_range(tid, dim, ntid, [&](Value* i) {
        Value* off = b.build_sext_i64(b.build_shl(i, kb.const_i32(2)));
        Value* x = kb.load_f32(kb.add(x_row, off));
        kb.store_f32(kb.add(y_row, off), b.build_mul(x, rstd));
    });

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    MslKernel k = MslTarget::compile(*fn);
    CHECK(k.uses_shared_memory);
    CHECK_EQ(k.shared_memory_bytes, 128u);
    CHECK(has(k.source, "threadgroup float _smem_0[32];"));
    CHECK(has(k.source, "threadgroup_barrier(mem_flags::mem_threadgroup);"));
    CHECK(has(k.source, "simd_shuffle_xor("));
    CHECK(has(k.source, "rsqrt("));
    CHECK(has(k.source, "while (true)"));
    CHECK(!has(k.source, "goto"));
}

TEST_CASE("MSL Target - Intrinsics Lowering and BF16 Helper") {
    Module mod("msl_intrinsics");
    Function* fn = mod.create_function(
        "intrinsics_kernel",
        Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::i32(), Type::i32()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* fptr = b.add_block_param(entry, Type::ptr());
    Value* iptr = b.add_block_param(entry, Type::ptr());
    Value* ua = b.add_block_param(entry, Type::i32());
    Value* ub = b.add_block_param(entry, Type::i32());

    KernelBuilder kb(b);
    Value* x = kb.load_f32(fptr);
    Value* t = kb.tanh_approx(x);
    Value* r = kb.rsqrt_approx(t);
    Value* bf16 = kb.f32_to_bf16(r);
    Value* bf_back = kb.bf16_to_f32(bf16);
    Value* f16 = kb.f32_to_f16(bf_back);
    Value* f16_back = kb.f16_to_f32(f16);
    Value* hi = kb.mul_hi_u32(ua, ub);

    kb.store_f32(fptr, f16_back);
    kb.store_i32(iptr, b.build_add(bf16, hi));
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    MslKernel k = MslTarget::compile(*fn);
    CHECK(has(k.source, "tanh("));
    CHECK(has(k.source, "rsqrt("));
    CHECK(has(k.source, "_brass_f32_to_bf16("));
    CHECK(has(k.source, "inline int32_t _brass_f32_to_bf16(float x)"));
    CHECK(has(k.source, "as_type<half>"));
    CHECK(has(k.source, "static_cast<half>("));
    CHECK(has(k.source, "mulhi("));
}

TEST_CASE("MSL Target - 128-bit and 16-bit Shared Memory, Subgroup Barrier and ID") {
    Module mod("msl_shared_ext");
    Function* fn = mod.create_function("shared_ext", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* in = b.add_block_param(entry, Type::ptr());
    Value* idx = b.add_block_param(entry, Type::i32());

    KernelBuilder kb(b);
    Value* sg = kb.subgroup_id();
    Value* sm_v4 = kb.shared_alloc_i32x4(16);
    Value* sm_u16 = kb.shared_alloc_u16(32);
    Value* vec = kb.vload(Type::i32x4(), in);
    kb.shared_store_i32x4_indexed(sm_v4, sg, vec);
    kb.shared_store_u16_indexed(sm_u16, idx, sg);
    kb.subgroup_barrier();
    Value* loaded_v4 = kb.shared_load_i32x4_indexed(sm_v4, sg);
    Value* loaded_f = kb.shared_load_f32_from_v4(sm_v4, idx);
    Value* loaded_u16 = kb.shared_load_u16_indexed(sm_u16, idx);
    kb.vstore(Type::i32x4(), out, loaded_v4);
    kb.store_f32(kb.add(out, kb.const_i64(16)), loaded_f);
    kb.store_i32(kb.add(out, kb.const_i64(20)), loaded_u16);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    MslKernel k = MslTarget::compile(*fn);
    CHECK(k.uses_shared_memory);
    CHECK_EQ(k.shared_memory_bytes, 16u * 16u + 32u * 2u);
    CHECK(has(k.source, "[[simdgroup_index_in_threadgroup]]"));
    CHECK(has(k.source, "simdgroup_barrier(mem_flags::mem_threadgroup);"));
    CHECK(has(k.source, "threadgroup int4 _smem_0[16];"));
    CHECK(has(k.source, "threadgroup uint16_t _smem_1[32];"));
    CHECK(has(k.source, "reinterpret_cast<const threadgroup float*>"));
}

TEST_CASE("MSL Target - Packed Dot, Half2 Pack/Hi, and Ldexp") {
    Module mod("msl_quant_stage");
    Function* fn = mod.create_function("quant_stage", Type::void_type(),
                                       {Type::ptr(), Type::i32(), Type::i32(), Type::f32()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* qa = b.add_block_param(entry, Type::i32());
    Value* qb = b.add_block_param(entry, Type::i32());
    Value* xf = b.add_block_param(entry, Type::f32());

    KernelBuilder kb(b);
    Value* dot = kb.sdot4(qa, qb, kb.const_i32(0));
    Value* packed = kb.pack_f16x2(xf, kb.ldexp_f32(xf, kb.const_i32(-2)));
    Value* hi = kb.f16_hi_to_f32(packed);
    kb.store_i32(out, dot);
    kb.store_f32(kb.add(out, kb.const_i64(4)), hi);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    MslKernel k = MslTarget::compile(*fn);
    CHECK(has(k.source, "_brass_sdot4("));
    CHECK(has(k.source, "half2("));
    CHECK(has(k.source, ".y"));
    CHECK(has(k.source, "ldexp("));
}

TEST_CASE("MSL Target - 16x16x16 Cooperative Matrix GEMM via simdgroup_matrix") {
    Module mod("msl_coop");
    Function* fn = mod.create_function("coop_gemm_16", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});

    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* pa = b.add_block_param(entry, Type::ptr());
    Value* pb = b.add_block_param(entry, Type::ptr());
    Value* pc = b.add_block_param(entry, Type::ptr());
    Value* k_stride = b.add_block_param(entry, Type::i32());

    KernelBuilder kb(b);
    Value* a_mat = kb.coop_alloc_a();
    Value* b_mat = kb.coop_alloc_b();
    Value* acc   = kb.coop_alloc_acc();
    kb.coop_fill_acc(acc, kb.const_f32(0.0f));
    kb.coop_load_a(a_mat, pa, k_stride, false);
    kb.coop_load_b(b_mat, pb, k_stride, true);
    kb.coop_mma(acc, a_mat, b_mat);
    kb.coop_scale_acc(acc, kb.const_f32(2.0f));
    kb.coop_store_acc(acc, pc, kb.const_i32(16), false, true);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    MslKernel k = MslTarget::compile(*fn);
    CHECK(has(k.source, "#include <metal_simdgroup_matrix>"));
    CHECK(has(k.source, "simdgroup_half8x8"));
    CHECK(has(k.source, "simdgroup_float8x8"));
    CHECK(has(k.source, "simdgroup_load("));
    CHECK(has(k.source, "simdgroup_multiply_accumulate("));
    CHECK(has(k.source, "simdgroup_store("));
    CHECK(has(k.source, "true);"));
}

TEST_CASE("MSL Target - Exhaustive Intrinsic Parity (Math, Sync Shuffle, Shared I64, Atomics)") {
    Module mod("msl_parity");
    Function* fn = mod.create_function("parity_kernel", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::f32(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* out_i = b.add_block_param(entry, Type::ptr());
    Value* out_f = b.add_block_param(entry, Type::ptr());
    Value* xf = b.add_block_param(entry, Type::f32());
    Value* i64v = b.add_block_param(entry, Type::i64());

    KernelBuilder kb(b);
    Value* smem_i64 = b.build_call("ptx_shared_alloc_i64", Type::ptr(), {kb.const_i32(4)});
    Value* smem_i32 = kb.shared_alloc_i32(4);
    Value* smem_f32 = kb.shared_alloc_f32(4);
    b.build_call("ptx_shared_store_i64", Type::void_type(), {smem_i64, i64v});
    b.build_call("ptx_shared_store_i64_indexed", Type::void_type(), {smem_i64, kb.const_i32(1), i64v});
    Value* l0 = b.build_call("ptx_shared_load_i64", Type::i64(), {smem_i64});
    Value* l1 = b.build_call("ptx_shared_load_i64_indexed", Type::i64(), {smem_i64, kb.const_i32(1)});
    Value* f_u64 = b.build_call("ptx_u64_to_f32", Type::f32(), {l0});
    Value* f_i64 = b.build_call("ptx_i64_to_f32", Type::f32(), {l1});

    Value* yf = kb.add(f_u64, f_i64);
    for (const char* ufn : {"tanf", "asinf", "acosf", "atanf", "sinhf", "coshf", "signf", "fractf"}) {
        yf = kb.add(yf, b.build_call(ufn, Type::f32(), {xf}));
    }
    yf = kb.add(yf, b.build_call("atan2f", Type::f32(), {xf, kb.const_f32(1.0f)}));
    yf = kb.add(yf, b.build_call("powf", Type::f32(), {xf, kb.const_f32(2.0f)}));
    yf = kb.add(yf, b.build_call("stepf", Type::f32(), {kb.const_f32(0.5f), xf}));
    yf = kb.add(yf, b.build_call("smoothstepf", Type::f32(), {kb.const_f32(0.0f), kb.const_f32(1.0f), xf}));
    yf = kb.add(yf, b.build_call("ptx_shfl_down_sync_f32", Type::f32(), {kb.const_i32(-1), yf, kb.const_i32(4)}));

    Value* a0 = b.build_call("ptx_atom_add_i32", Type::i32(), {out_i, kb.const_i32(1)});
    Value* a1 = b.build_call("ptx_atom_min_u32", Type::i32(), {out_i, kb.const_i32(9)});
    Value* a2 = b.build_call("ptx_atom_max_i32", Type::i32(), {out_i, kb.const_i32(2)});
    Value* a3 = b.build_call("ptx_atom_and_u32", Type::i32(), {out_i, kb.const_i32(7)});
    Value* a4 = b.build_call("ptx_atom_or_i32",  Type::i32(), {out_i, kb.const_i32(3)});
    Value* a5 = b.build_call("ptx_atom_xor_u32", Type::i32(), {out_i, kb.const_i32(5)});
    Value* a6 = b.build_call("ptx_atom_exch_i32",Type::i32(), {out_i, kb.const_i32(8)});
    Value* a7 = b.build_call("ptx_atom_cas_u32", Type::i32(), {out_i, a0, kb.const_i32(4)});
    Value* s0 = b.build_call("ptx_atom_shared_add_i32", Type::i32(), {smem_i32, kb.const_i32(1)});
    Value* s1 = b.build_call("ptx_atom_shared_cas_i32", Type::i32(), {smem_i32, s0, kb.const_i32(2)});
    Value* fa0 = b.build_call("ptx_atom_add_f32", Type::f32(), {out_f, yf});
    Value* fa1 = b.build_call("ptx_atom_shared_add_f32", Type::f32(), {smem_f32, fa0});
    Value* fa2 = b.build_call("ptx_atom_exch_f32", Type::f32(), {out_f, fa1});

    Value* sum_i = kb.add(kb.add(kb.add(a1, a2), kb.add(a3, a4)), kb.add(kb.add(a5, a6), kb.add(a7, s1)));
    kb.store_i32(out_i, sum_i);
    kb.store_f32(out_f, fa2);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    MslKernel k = MslTarget::compile(*fn);
    CHECK(has(k.source, "tan("));
    CHECK(has(k.source, "atan2("));
    CHECK(has(k.source, "smoothstep("));
    CHECK(has(k.source, "simd_shuffle_down("));
    CHECK(has(k.source, "threadgroup int64_t"));
    CHECK(has(k.source, "atomic_fetch_add_explicit("));
    CHECK(has(k.source, "atomic_fetch_min_explicit("));
    CHECK(has(k.source, "_brass_atom_add_f32("));
    CHECK(has(k.source, "_brass_atom_cas_i32("));

    // 64-bit atomics are rejected cleanly on MSL rather than emitting invalid syntax.
    Module mod64("msl_atom64");
    Function* fn64 = mod64.create_function("bad64", Type::void_type(), {Type::ptr(), Type::i64()});
    Builder b64(mod64);
    b64.set_function(fn64);
    BasicBlock* e64 = b64.append_block("entry");
    b64.position_at_end(e64);
    Value* p64 = b64.add_block_param(e64, Type::ptr());
    Value* v64 = b64.add_block_param(e64, Type::i64());
    b64.build_call("ptx_atom_add_i64", Type::i64(), {p64, v64});
    b64.build_ret_void();
    fn64->rebuild_cfg_predecessors();
    bool threw64 = false;
    try { (void)MslTarget::compile(*fn64); } catch (const std::runtime_error&) { threw64 = true; }
    CHECK(threw64);
}

TEST_CASE("MSL Target - Multi-Kernel Module Emission") {
    Module mod("msl_multi");
    build_vec4_saxpy(mod, "kernel_one");
    build_vec4_saxpy(mod, "kernel_two");

    std::string msl = MslTarget::compile_module(mod);
    CHECK_EQ(count_of(msl, "#include <metal_stdlib>"), size_t(1));
    CHECK_EQ(count_of(msl, "using namespace metal;"), size_t(1));
    CHECK(has(msl, "kernel void kernel_one("));
    CHECK(has(msl, "kernel void kernel_two("));
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
