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
