#include "test_framework.hpp"

#include <brass/mir/builder.hpp>
#include <brass/mir/gpu_opt.hpp>
#include <brass/mir/gpu_portability.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/msl_target.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/ptx/ptx_printer.hpp>
#include <brass/target/ptx_target.hpp>
#if BRASS_WITH_SPIRV
#include <brass/target/spirv_target.hpp>
#endif
#include <brass/verify/smt_equiv.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brass;

namespace {

bool ptxas_available() {
    static const bool ok = (
#if defined(_WIN32)
        std::system("ptxas --version >NUL 2>&1") == 0
#else
        std::system("ptxas --version >/dev/null 2>&1") == 0
#endif
    );
    return ok;
}

bool ptxas_assembles_sm80(const std::string& ptx) {
    const std::filesystem::path dir = brass::test::scratch_dir();
    const std::filesystem::path in = dir / "ampere_test.ptx";
    const std::filesystem::path out = dir / "ampere_test.cubin";
    {
        std::ofstream f(in, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f << ptx;
    }
    const std::string cmd = "ptxas -arch=sm_80 \"" + in.string() + "\" -o \"" + out.string() + "\""
#if defined(_WIN32)
        " >NUL 2>&1"
#else
        " >/dev/null 2>&1"
#endif
    ;
    const bool ok = (std::system(cmd.c_str()) == 0);
    std::error_code ec;
    std::filesystem::remove(in, ec);
    std::filesystem::remove(out, ec);
    return ok;
}

} // namespace

TEST_CASE("PTX Ampere - Launch Bounds + Read-Only .nc Cache Loads + Cross-Backend Parity") {
    Module mod("ldg_mod");
    Function* fn = mod.create_function("ldg_kernel", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});
    {
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* work = b.append_block("work");
        BasicBlock* done = b.append_block("done");

        b.position_at_end(entry);
        Value* out = b.add_block_param(entry, Type::ptr());
        Value* in1 = b.add_block_param(entry, Type::ptr());
        Value* in4 = b.add_block_param(entry, Type::ptr());
        Value* n = b.add_block_param(entry, Type::i32());
        Value* gid = b.build_call("gpu_global_tid_x", Type::i32());
        b.build_br_if(b.build_slt(gid, n), work, {}, done, {});

        b.position_at_end(work);
        Value* off4 = b.build_mul(gid, b.build_iconst_i32(4));
        Value* s = b.build_call("gpu_ldg_f32", Type::f32(), {in1, off4});
        Value* off16 = b.build_mul(gid, b.build_iconst_i32(16));
        Value* v = b.build_call("gpu_ldg_f32x4", Type::f32x4(), {in4, off16});
        Value* v_plus = b.build_vadd(v, b.build_vbroadcast(Type::f32x4(), s));
        b.build_vstore(Type::f32x4(), b.build_add(out, b.build_sext_i64(off16)), 0, v_plus);
        b.build_br(done);

        b.position_at_end(done);
        b.build_ret_void();
    }
    fn->rebuild_cfg_predecessors();

    CHECK(verify_gpu_portability(*fn));

    target::PtxOptions popts;
    popts.sm_arch = "sm_80";
    popts.maxntid[0] = 128;
    popts.minnctapersm = 2;
    popts.maxnreg = 64;

    const std::string ptx = target::PtxTarget::emit_function(*fn, popts);
    CHECK(ptx.find(".maxntid 128, 1, 1") != std::string::npos);
    CHECK(ptx.find(".minnctapersm 2") != std::string::npos);
    CHECK(ptx.find(".maxnreg 64") != std::string::npos);
    CHECK(ptx.find("ld.global.nc.f32") != std::string::npos);
    CHECK(ptx.find("ld.global.nc.v4.f32") != std::string::npos);
    if (ptxas_available()) CHECK(ptxas_assembles_sm80(ptx));

    // Also portable to Metal MSL and SPIR-V
    const target::MslKernel msl = target::MslTarget::compile(*fn);
    CHECK(!msl.source.empty());
#if BRASS_WITH_SPIRV
    const target::SpirvKernel spv = target::SpirvTarget::compile(*fn);
    CHECK(!spv.words.empty());
#endif
}

TEST_CASE("PTX Ampere - cp.async, cvta, ldmatrix, and mma.sync.m16n8k16") {
    Module mod("ampere_mma");
    Function* fn = mod.create_function("mma_async_k", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr()});
    {
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* g_a = b.add_block_param(entry, Type::ptr());
        Value* g_b = b.add_block_param(entry, Type::ptr());
        Value* g_c = b.add_block_param(entry, Type::ptr());

        Value* smem = b.build_call("ptx_shared_alloc_i32", Type::ptr(), {b.build_iconst_i32(256)});
        Value* smem_u32 = b.build_call("ptx_cvta_to_shared_u32", Type::i32(), {smem});
        Value* tid = b.build_call("ptx_tid_x", Type::i32());
        Value* s_off = b.build_add(smem_u32, b.build_mul(tid, b.build_iconst_i32(16)));

        b.build_call("ptx_cp_async_cg_16", Type::void_type(), {s_off, g_a});
        b.build_call("ptx_cp_async_ca_16", Type::void_type(), {smem, g_b});
        b.build_call("ptx_cp_async_commit", Type::void_type());
        b.build_call("ptx_cp_async_wait_group", Type::void_type(), {b.build_iconst_i32(0)});
        b.build_call("ptx_cp_async_wait_all", Type::void_type());
        b.build_call("ptx_sync", Type::void_type());

        Value* a_frag = b.build_call("ptx_ldmatrix_x4", Type::i32x4(), {s_off});
        Value* b_pair = b.build_call("ptx_ldmatrix_x2_trans", Type::i64(), {s_off});
        Value* b0 = b.build_trunc_i32(b_pair);
        Value* b1 = b.build_trunc_i32(b.build_lshr(b_pair, b.build_iconst_i64(32)));

        Value* c_zero = b.build_vzero(Type::f32x4());
        Value* d_f16 = b.build_call("ptx_mma_m16n8k16_f16_f32", Type::f32x4(), {a_frag, b0, b1, c_zero});
        Value* d_bf16 = b.build_call("ptx_mma_m16n8k16_bf16_f32", Type::f32x4(), {a_frag, b0, b1, d_f16});
        b.build_vstore(Type::f32x4(), g_c, 0, d_bf16);
        b.build_ret_void();
    }

    target::PtxOptions sm80;
    sm80.sm_arch = "sm_80";
    const std::string ptx = target::PtxTarget::emit_function(*fn, sm80);

    CHECK(ptx.find("cvta.to.shared.u64") != std::string::npos);
    CHECK(ptx.find("cvt.u32.u64") != std::string::npos);
    CHECK(ptx.find("cp.async.cg.shared.global") != std::string::npos);
    CHECK(ptx.find("cp.async.ca.shared.global") != std::string::npos);
    CHECK(ptx.find("cp.async.commit_group;") != std::string::npos);
    CHECK(ptx.find("cp.async.wait_group 0;") != std::string::npos);
    CHECK(ptx.find("cp.async.wait_all;") != std::string::npos);
    CHECK(ptx.find("ldmatrix.sync.aligned.m8n8.x4.shared.b16") != std::string::npos);
    CHECK(ptx.find("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16") != std::string::npos);
    CHECK(ptx.find("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32") != std::string::npos);
    CHECK(ptx.find("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32") != std::string::npos);
    if (ptxas_available()) CHECK(ptxas_assembles_sm80(ptx));

    // PTX-only intrinsics must be rejected by GPU portability checks.
    CHECK(!verify_gpu_portability(*fn));
}

TEST_CASE("PTX Ampere - Native bf16 cvt on sm_80 vs Software Fallback on sm_70") {
    Module mod("bf16_cvt");
    Function* fn = mod.create_function("bf16_roundtrip", Type::void_type(), {Type::ptr(), Type::ptr()});
    {
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* in = b.add_block_param(entry, Type::ptr());
        Value* out = b.add_block_param(entry, Type::ptr());
        Value* tid = b.build_call("ptx_global_tid_x", Type::i32());
        Value* x = b.build_load_indexed(Type::f32(), in, tid, 4);
        Value* bf = b.build_call("ptx_f32_to_bf16", Type::i32(), {x});
        Value* back = b.build_call("ptx_bf16_to_f32", Type::f32(), {bf});
        b.build_store_indexed(Type::f32(), out, tid, 4, back);
        b.build_ret_void();
    }

    target::PtxOptions sm70;
    sm70.sm_arch = "sm_70";
    const std::string ptx70 = target::PtxTarget::emit_function(*fn, sm70);
    CHECK(ptx70.find("cvt.rn.bf16.f32") == std::string::npos);
    CHECK(ptx70.find("selp.b32") != std::string::npos);

    target::PtxOptions sm80;
    sm80.sm_arch = "sm_80";
    const std::string ptx80 = target::PtxTarget::emit_function(*fn, sm80);
    CHECK(ptx80.find("cvt.rn.bf16.f32") != std::string::npos);
    CHECK(ptx80.find("shl.b32") != std::string::npos);
    if (ptxas_available()) CHECK(ptxas_assembles_sm80(ptx80));
}

TEST_CASE("GPU Opt - Small Loop Unroll + 4x32-bit v4 Coalescing") {
    Module mod("gpu_vec_opt");
    Function* fn = mod.create_function("scale4", Type::void_type(), {Type::ptr(), Type::ptr(), Type::f32()});
    {
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* loop = b.append_block("loop");
        BasicBlock* body = b.append_block("body");
        BasicBlock* exit = b.append_block("exit");

        b.position_at_end(entry);
        Value* out = b.add_block_param(entry, Type::ptr());
        Value* in = b.add_block_param(entry, Type::ptr());
        Value* alpha = b.add_block_param(entry, Type::f32());
        Value* gid = b.build_call("gpu_global_tid_x", Type::i32());
        Value* base_idx = b.build_shl(gid, b.build_iconst_i32(2));
        b.build_br(loop, {b.build_iconst_i32(0)});

        b.position_at_end(loop);
        Value* k = b.add_block_param(loop, Type::i32());
        b.build_br_if(b.build_slt(k, b.build_iconst_i32(4)), body, {}, exit, {});

        b.position_at_end(body);
        Value* idx = b.build_add(base_idx, k);
        Value* v = b.build_load_indexed(Type::f32(), in, idx, 4);
        Value* scaled = b.build_mul(v, alpha);
        b.build_store_indexed(Type::f32(), out, idx, 4, scaled);
        b.build_br(loop, {b.build_add(k, b.build_iconst_i32(1))});

        b.position_at_end(exit);
        b.build_ret_void();
    }
    fn->rebuild_cfg_predecessors();

    // Clone reference before optimizing
    Module ref_mod("ref_mod");
    Function* ref_fn = clone_function(*fn, ref_mod, "scale4_ref");
    CHECK(ref_fn != nullptr);

    GpuOptOptions gopts;
    gopts.enable_small_loop_unroll = true;
    gopts.enable_load_coalesce = true;
    gopts.max_unroll_trip_count = 8;
    const bool changed = optimize_gpu_function(*fn, mod, gopts);
    CHECK(changed);
    CHECK(fn->block_count() <= 2);

    size_t vload_count = 0;
    size_t vstore_count = 0;
    for (const BasicBlock* bb : fn->blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::vload) ++vload_count;
            if (inst->opcode() == Opcode::vstore) ++vstore_count;
        }
    }
    CHECK(vload_count == 1);
    CHECK(vstore_count == 1);

    const std::string ptx = target::PtxTarget::emit_function(*fn);
    CHECK(ptx.find("ld.global.v4.f32") != std::string::npos);
    CHECK(ptx.find("st.global.v4.f32") != std::string::npos);

    // Verify alias-safe unroll + v4 load coalescing equivalence on a 4-load reduction kernel.
    Function* sum_fn = mod.create_function("sum4", Type::void_type(), {Type::ptr(), Type::ptr(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(sum_fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* loop = b.append_block("loop");
        BasicBlock* body = b.append_block("body");
        BasicBlock* exit = b.append_block("exit");

        b.position_at_end(entry);
        Value* dst = b.add_block_param(entry, Type::ptr());
        Value* src = b.add_block_param(entry, Type::ptr());
        Value* gid64 = b.add_block_param(entry, Type::i64());
        Value* base_idx = b.build_shl(gid64, b.build_iconst_i64(2));
        b.build_br(loop, {b.build_iconst_i64(0), b.build_iconst_i32(0)});

        b.position_at_end(loop);
        Value* k = b.add_block_param(loop, Type::i64());
        Value* acc = b.add_block_param(loop, Type::i32());
        b.build_br_if(b.build_slt(k, b.build_iconst_i64(4)), body, {}, exit, {});

        b.position_at_end(body);
        Value* v = b.build_load_indexed(Type::i32(), src, b.build_add(base_idx, k), 4);
        b.build_br(loop, {b.build_add(k, b.build_iconst_i64(1)), b.build_add(acc, v)});

        b.position_at_end(exit);
        b.build_store_indexed(Type::i32(), dst, gid64, 4, acc);
        b.build_ret_void();
    }
    sum_fn->rebuild_cfg_predecessors();

    Module sum_ref_mod("sum_ref");
    Function* sum_ref = clone_function(*sum_fn, sum_ref_mod, "sum4_ref");
    GpuOptOptions s_opts;
    s_opts.enable_small_loop_unroll = true;
    s_opts.enable_load_coalesce = true;
    s_opts.assume_disjoint_buffers = false;
    s_opts.max_unroll_trip_count = 8;
    CHECK(optimize_gpu_function(*sum_fn, mod, s_opts));

    EquivOptions eopts;
    eopts.loop_unroll_bound = 8;
    const EquivResult eq = verify_mir_equivalence(*sum_ref, *sum_fn, eopts);
    CHECK(eq.is_equivalent());
}

TEST_CASE("PTX Ampere - required_isa reports minimal SM and PTX version") {
    auto isa_for = [](auto&& build_body) {
        Module mod("isa_mod");
        Function* fn = mod.create_function("k", Type::void_type(), {Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* e = b.append_block("entry");
        b.position_at_end(e);
        Value* p = b.add_block_param(e, Type::ptr());
        build_body(b, p);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();
        ptx::PtxISel isel;
        ptx::Function ptx_fn = isel.lower(*fn);
        return ptx::required_isa(ptx_fn);
    };

    const ptx::PtxIsaRequirement base = isa_for([](Builder& b, Value* p) {
        b.build_store(Type::i32(), p, 0, b.build_iconst_i32(42));
    });
    CHECK(base.min_sm == 70);
    CHECK(base.min_ptx_major == 7);
    CHECK(base.min_ptx_minor == 0);

    const ptx::PtxIsaRequirement ampere = isa_for([](Builder& b, Value* p) {
        Value* s = b.build_call("ptx_shared_alloc_i32", Type::ptr(), {b.build_iconst_i32(16)});
        b.build_call("ptx_cp_async_cg_16", Type::void_type(), {s, p});
        b.build_call("ptx_cp_async_wait_all", Type::void_type());
    });
    CHECK(ampere.min_sm == 80);
    CHECK(ampere.min_ptx_major == 7);
    CHECK(ampere.min_ptx_minor == 0);

    const ptx::PtxIsaRequirement l2 = isa_for([](Builder& b, Value* p) {
        Value* pol = b.build_call("ptx_l2_evict_first_policy", Type::i64());
        b.build_store(Type::i64(), p, 0, pol);
    });
    CHECK(l2.min_sm == 80);
    CHECK(l2.min_ptx_major == 7);
    CHECK(l2.min_ptx_minor == 4);
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
