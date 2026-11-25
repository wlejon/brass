#include "test_framework.hpp"

#include <brass/mir/builder.hpp>
#include <brass/mir/gpu_opt.hpp>
#include <brass/mir/gpu_portability.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/msl_target.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv_target.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brass;

namespace {

Function* build_intrinsic_parity_kernel(Module& mod, const char* fn_name, bool use_gpu_prefix) {
    Function* fn = mod.create_function(fn_name, Type::void_type(), {Type::ptr(), Type::ptr(), Type::f32(), Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* aux = b.add_block_param(entry, Type::ptr());
    Value* x   = b.add_block_param(entry, Type::f32());
    Value* n   = b.add_block_param(entry, Type::i32());

    auto sym = [use_gpu_prefix](const char* stem) -> std::string {
        return std::string(use_gpu_prefix ? "gpu_" : "ptx_") + stem;
    };

    Value* gid  = b.build_call(sym("global_tid_x"), Type::i32());
    Value* lane = b.build_call(sym("laneid"), Type::i32());
    Value* smem = b.build_call(sym("shared_alloc_f32"), Type::ptr(), {b.build_iconst_i32(32)});
    Value* ex   = b.build_call(sym("exp"), Type::f32(), {x});
    Value* sh   = b.build_call(sym("shfl_down_f32"), Type::f32(), {ex, b.build_iconst_i32(1)});

    b.build_call(sym("shared_store_f32_indexed"), Type::void_type(), {smem, lane, sh});
    b.build_call(sym("sync"), Type::void_type());
    Value* ld = b.build_call(sym("shared_load_f32_indexed"), Type::f32(), {smem, lane});
    b.build_call(sym("atom_add_f32"), Type::f32(), {aux, ld});
    b.build_store_indexed(Type::f32(), out, b.build_add(gid, n), 4, ld);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

Function* build_coop_parity_kernel(Module& mod, const char* fn_name, bool use_gpu_prefix) {
    Function* fn = mod.create_function(fn_name, Type::void_type(), {Type::ptr(), Type::ptr(), Type::f32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* a_ptr = b.add_block_param(entry, Type::ptr());
    Value* c_ptr = b.add_block_param(entry, Type::ptr());
    Value* scale = b.add_block_param(entry, Type::f32());

    auto sym = [use_gpu_prefix](const char* stem) -> std::string {
        return std::string(use_gpu_prefix ? "gpu_" : "ptx_") + stem;
    };

    Value* ma = b.build_call(sym("coop_alloc_a"), Type::ptr());
    Value* mb = b.build_call(sym("coop_alloc_b"), Type::ptr());
    Value* mc = b.build_call(sym("coop_alloc_acc"), Type::ptr());
    b.build_call(sym("coop_fill_acc"), Type::void_type(), {mc, scale});
    Value* s16 = b.build_iconst_i32(16);
    Value* c0  = b.build_iconst_i32(0);
    b.build_call(sym("coop_load_a"), Type::void_type(), {ma, a_ptr, s16, c0});
    b.build_call(sym("coop_load_b"), Type::void_type(), {mb, a_ptr, s16, c0});
    b.build_call(sym("coop_mma"), Type::void_type(), {mc, ma, mb});
    b.build_call(sym("coop_scale_acc"), Type::void_type(), {mc, scale});
    b.build_call(sym("coop_store_acc"), Type::void_type(), {mc, c_ptr, s16, c0, c0});
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

size_t count_in_func(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == op) ++n;
        }
    }
    return n;
}

} // namespace

TEST_CASE("GPU Intrinsics - gpu_* Aliases Match ptx_* Across PTX, SPIR-V, and MSL") {
    Module mod_ptx("alias_ptx");
    Module mod_gpu("alias_gpu");
    Function* fn_ptx = build_intrinsic_parity_kernel(mod_ptx, "k", false);
    Function* fn_gpu = build_intrinsic_parity_kernel(mod_gpu, "k", true);

    CHECK_EQ(target::PtxTarget::emit_function(*fn_gpu), target::PtxTarget::emit_function(*fn_ptx));

    target::SpirvOptions spv_opts;
    spv_opts.stage = target::SpirvStage::Compute;
    spv_opts.local_size_x = 32;
    CHECK(target::SpirvTarget::compile(*fn_gpu, spv_opts).words ==
          target::SpirvTarget::compile(*fn_ptx, spv_opts).words);

    target::MslOptions msl_opts;
    msl_opts.workgroup_x = 32;
    CHECK_EQ(target::MslTarget::compile(*fn_gpu, msl_opts).source,
             target::MslTarget::compile(*fn_ptx, msl_opts).source);

    Module cm_ptx("cm_ptx");
    Module cm_gpu("cm_gpu");
    Function* fcm_ptx = build_coop_parity_kernel(cm_ptx, "cm", false);
    Function* fcm_gpu = build_coop_parity_kernel(cm_gpu, "cm", true);
    CHECK_EQ(target::PtxTarget::emit_function(*fcm_gpu), target::PtxTarget::emit_function(*fcm_ptx));
    CHECK(target::SpirvTarget::compile(*fcm_gpu, spv_opts).words ==
          target::SpirvTarget::compile(*fcm_ptx, spv_opts).words);
    CHECK_EQ(target::MslTarget::compile(*fcm_gpu, msl_opts).source,
             target::MslTarget::compile(*fcm_ptx, msl_opts).source);
}

TEST_CASE("GPU Opt - FMA Fusion and Pure Loop-Invariant Hoisting Lower Cleanly") {
    Module mod("gpu_fma_licm");
    Function* fn = mod.create_function("fma_licm", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::f32(), Type::f32(), Type::i32(), Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop  = b.append_block("loop");
    BasicBlock* body  = b.append_block("body");
    BasicBlock* exit  = b.append_block("exit");

    b.position_at_end(entry);
    Value* in_ptr  = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    Value* a       = b.add_block_param(entry, Type::f32());
    Value* c       = b.add_block_param(entry, Type::f32());
    Value* row     = b.add_block_param(entry, Type::i32());
    Value* n       = b.add_block_param(entry, Type::i32());
    Value* c0      = b.build_iconst_i32(0);
    Value* c1      = b.build_iconst_i32(1);
    Value* stride  = b.build_iconst_i32(16);
    b.build_br(loop, {c0, b.build_fconst_f32(0.0f)});

    b.position_at_end(loop);
    Value* iv  = b.add_block_param(loop, Type::i32());
    Value* acc = b.add_block_param(loop, Type::f32());
    b.build_br_if(b.build_slt(iv, n), body, {}, exit, {});

    b.position_at_end(body);
    // Pure loop-invariant index base: should be hoisted to preheader.
    Value* base_idx = b.build_mul(row, stride);
    Value* elem_idx = b.build_add(base_idx, iv);
    Value* x        = b.build_load_indexed(Type::f32(), in_ptr, elem_idx, 4);
    // Single-use mul + add: should fuse into fma_f32.
    Value* prod     = b.build_mul(a, x);
    Value* next_acc = b.build_add(prod, acc);
    Value* inv_bias = b.build_add(a, c); // pure float loop invariant -> hoisted
    b.build_store_indexed(Type::f32(), out_ptr, elem_idx, 4, inv_bias);
    b.build_br(loop, {b.build_add(iv, c1), next_acc});

    b.position_at_end(exit);
    b.build_store(Type::f32(), out_ptr, 0, acc);
    b.build_ret_void();

    GpuOptOptions opt;
    opt.opt_level = 1;
    opt.enable_fma = true;
    CHECK(optimize_gpu_function(*fn, mod, opt));

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));
    CHECK_EQ(count_in_func(*fn, Opcode::fma_f32), 1u);

    // Loop body should no longer hold base_idx (mul(row, stride)) or inv_bias.
    size_t body_muls = 0;
    for (const Instruction* inst : *body) {
        if (inst->opcode() == Opcode::mul) ++body_muls;
    }
    CHECK_EQ(body_muls, 0u);

    // Must lower cleanly across all three GPU targets.
    std::string ptx = target::PtxTarget::emit_function(*fn);
    CHECK(ptx.find("fma.rn.f32") != std::string::npos);

    target::SpirvOptions spv_opts;
    spv_opts.stage = target::SpirvStage::Compute;
    spv_opts.local_size_x = 32;
    CHECK(!target::SpirvTarget::compile(*fn, spv_opts).words.empty());

    target::MslOptions msl_opts;
    msl_opts.workgroup_x = 32;
    CHECK(target::MslTarget::compile(*fn, msl_opts).source.find("fma(") != std::string::npos);
}

TEST_CASE("GPU Opt - LICM Hoists In-Loop Constant Operands Before Hoisted Users") {
    Module mod("gpu_licm_consts");
    Function* fn = mod.create_function("licm_consts", Type::void_type(),
                                       {Type::ptr(), Type::f32(), Type::i32(), Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop  = b.append_block("loop");
    BasicBlock* body  = b.append_block("body");
    BasicBlock* exit  = b.append_block("exit");

    b.position_at_end(entry);
    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    Value* a       = b.add_block_param(entry, Type::f32());
    Value* row     = b.add_block_param(entry, Type::i32());
    Value* n       = b.add_block_param(entry, Type::i32());
    Value* c0      = b.build_iconst_i32(0);
    Value* c1      = b.build_iconst_i32(1);
    b.build_br(loop, {c0});

    b.position_at_end(loop);
    Value* iv = b.add_block_param(loop, Type::i32());
    b.build_br_if(b.build_slt(iv, n), body, {}, exit, {});

    b.position_at_end(body);
    // Constants materialized inside the loop body and consumed by loop-invariant ops.
    Value* c33      = b.build_iconst_i32(33);
    Value* chalf    = b.build_fconst_f32(0.5f);
    Value* base_idx = b.build_mul(row, c33);
    Value* elem_idx = b.build_add(base_idx, iv);
    Value* shifted  = b.build_add(a, chalf);
    b.build_store_indexed(Type::f32(), out_ptr, elem_idx, 4, shifted);
    b.build_br(loop, {b.build_add(iv, c1)});

    b.position_at_end(exit);
    b.build_ret_void();

    GpuOptOptions opt;
    opt.opt_level = 1;
    CHECK(optimize_gpu_function(*fn, mod, opt));

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));

    size_t body_muls = 0;
    for (const Instruction* inst : *body) {
        if (inst->opcode() == Opcode::mul) ++body_muls;
    }
    CHECK_EQ(body_muls, 0u);

    // Every non-loop block's instructions must only use values defined outside the loop.
    for (const BasicBlock* bb : fn->blocks()) {
        if (bb == loop || bb == body) continue;
        for (const Instruction* inst : *bb) {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                const Value* op = inst->operand(i);
                if (op && op->is_instruction()) {
                    const BasicBlock* def_bb = op->defining_instruction()->parent();
                    CHECK(def_bb != loop);
                    CHECK(def_bb != body);
                }
            }
        }
    }

    target::MslOptions msl_opts;
    msl_opts.workgroup_x = 32;
    std::string msl_src = target::MslTarget::compile(*fn, msl_opts).source;
    size_t pos_33 = msl_src.find("= 33;");
    size_t pos_while = msl_src.find("while (true)");
    CHECK(pos_33 != std::string::npos);
    CHECK(pos_while != std::string::npos);
    CHECK(pos_33 < pos_while);

    // Constant 33 must appear before any preheader multiply that uses it.
    size_t pos_mul = msl_src.find(" * ", pos_33);
    CHECK(pos_mul != std::string::npos);
    CHECK(pos_mul < pos_while);

    CHECK(!target::PtxTarget::emit_function(*fn).empty());
    target::SpirvOptions spv_opts;
    spv_opts.stage = target::SpirvStage::Compute;
    spv_opts.local_size_x = 32;
    CHECK(!target::SpirvTarget::compile(*fn, spv_opts).words.empty());
}

TEST_CASE("GPU Opt - Pow2 Float Division Rewrites and Fuses Into FMA") {
    Module mod("gpu_fdiv_pow2");
    Function* fn = mod.create_function("fdiv_pow2_k", Type::void_type(),
                                       {Type::ptr(), Type::f32(), Type::f32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* x   = b.add_block_param(entry, Type::f32());
    Value* y   = b.add_block_param(entry, Type::f32());
    Value* two = b.build_fconst_f32(2.0f);
    Value* q   = b.build_sdiv(x, two);
    Value* res = b.build_add(q, y);
    b.build_store(Type::f32(), out, 0, res);
    b.build_ret_void();

    GpuOptOptions opt;
    opt.opt_level = 1;
    opt.enable_fdiv_pow2 = true;
    opt.enable_fma = true;
    CHECK(optimize_gpu_function(*fn, mod, opt));
    CHECK(verify_gpu_portability(*fn));
    CHECK_EQ(count_in_func(*fn, Opcode::sdiv), 0u);
    CHECK_EQ(count_in_func(*fn, Opcode::fma_f32), 1u);
}

TEST_CASE("GPU Portability - Divergence, Barriers, Provenance, FP64, and i64 Atomics") {
    GpuPortabilityOptions strict;

    // 1. Portable kernel with uniform conditional barrier and shared memory passes.
    {
        Module mod("port_ok");
        Function* fn = mod.create_function("port_ok", Type::void_type(), {Type::ptr(), Type::i32()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* then_b = b.append_block("then");
        BasicBlock* merge = b.append_block("merge");
        b.position_at_end(entry);
        Value* ptr  = b.add_block_param(entry, Type::ptr());
        Value* flag = b.add_block_param(entry, Type::i32());
        Value* c0   = b.build_iconst_i32(0);
        Value* c16  = b.build_iconst_i32(16);
        Value* tid  = b.build_call("gpu_tid_x", Type::i32());
        Value* smem = b.build_call("gpu_shared_alloc_f32", Type::ptr(), {c16});
        Value* cond = b.build_ne(flag, c0);
        b.build_br_if(cond, then_b, {}, merge, {});

        b.position_at_end(then_b);
        b.build_call("gpu_shared_store_f32_indexed", Type::void_type(), {smem, tid, b.build_fconst_f32(1.0f)});
        b.build_call("gpu_sync", Type::void_type());
        b.build_br(merge, {});

        b.position_at_end(merge);
        Value* v = b.build_call("gpu_shared_load_f32_indexed", Type::f32(), {smem, tid});
        Value* sh = b.build_call("gpu_shfl_xor_f32", Type::f32(), {v, b.build_iconst_i32(1)});
        b.build_store(Type::f32(), ptr, 0, sh);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();

        DiagnosticReporter diag;
        CHECK(verify_gpu_portability(*fn, strict, &diag));
        CHECK(!diag.has_errors());
    }

    // 2. Non-portable warp/timer intrinsics rejected.
    {
        Module mod("port_warpid");
        Function* fn = mod.create_function("port_warpid", Type::void_type(), {Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* ptr = b.add_block_param(entry, Type::ptr());
        Value* w = b.build_call("gpu_warpid", Type::i32());
        b.build_store(Type::i32(), ptr, 0, w);
        b.build_ret_void();

        DiagnosticReporter diag;
        CHECK(!verify_gpu_portability(*fn, strict, &diag));
        CHECK(diag.has_errors());
    }

    // 3. FP64 rejected by default, allowed when allow_fp64=true.
    {
        Module mod("port_fp64");
        Function* fn = mod.create_function("port_fp64", Type::void_type(), {Type::ptr(), Type::f32()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* ptr = b.add_block_param(entry, Type::ptr());
        Value* x   = b.add_block_param(entry, Type::f32());
        Value* d   = b.build_call("gpu_f32_to_f64", Type::f64(), {x});
        b.build_store(Type::f64(), ptr, 0, d);
        b.build_ret_void();

        DiagnosticReporter diag_off;
        CHECK(!verify_gpu_portability(*fn, strict, &diag_off));
        GpuPortabilityOptions fp64_ok = strict;
        fp64_ok.allow_fp64 = true;
        DiagnosticReporter diag_on;
        CHECK(verify_gpu_portability(*fn, fp64_ok, &diag_on));
    }

    // 4. i64 atomics rejected by default, allowed when allow_i64_atomics=true.
    {
        Module mod("port_i64_atom");
        Function* fn = mod.create_function("port_i64_atom", Type::void_type(), {Type::ptr(), Type::i64()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* ptr = b.add_block_param(entry, Type::ptr());
        Value* v   = b.add_block_param(entry, Type::i64());
        Value* old = b.build_call("gpu_atom_add_i64", Type::i64(), {ptr, v});
        b.build_store(Type::i64(), ptr, 0, old);
        b.build_ret_void();

        DiagnosticReporter diag_off;
        CHECK(!verify_gpu_portability(*fn, strict, &diag_off));
        GpuPortabilityOptions i64_ok = strict;
        i64_ok.allow_i64_atomics = true;
        DiagnosticReporter diag_on;
        CHECK(verify_gpu_portability(*fn, i64_ok, &diag_on));
    }

    // 5. Mixing global and shared pointers via select violates provenance.
    {
        Module mod("port_mix_ptr");
        Function* fn = mod.create_function("port_mix_ptr", Type::void_type(), {Type::ptr(), Type::i32()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* gptr = b.add_block_param(entry, Type::ptr());
        Value* flag = b.add_block_param(entry, Type::i32());
        Value* sptr = b.build_call("gpu_shared_alloc_f32", Type::ptr(), {b.build_iconst_i32(8)});
        Value* c0   = b.build_iconst_i32(0);
        Value* mixed = b.build_select(b.build_ne(flag, c0), gptr, sptr);
        b.build_store(Type::f32(), mixed, 0, b.build_fconst_f32(1.0f));
        b.build_ret_void();

        DiagnosticReporter diag;
        CHECK(!verify_gpu_portability(*fn, strict, &diag));
        CHECK(diag.has_errors());
    }

    // 6. Barrier under thread-divergent branch violates barrier uniformity.
    {
        Module mod("port_div_bar");
        Function* fn = mod.create_function("port_div_bar", Type::void_type(), {Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* then_b = b.append_block("then");
        BasicBlock* merge = b.append_block("merge");
        b.position_at_end(entry);
        (void)b.add_block_param(entry, Type::ptr());
        Value* tid = b.build_call("gpu_tid_x", Type::i32());
        Value* cond = b.build_slt(tid, b.build_iconst_i32(16));
        b.build_br_if(cond, then_b, {}, merge, {});

        b.position_at_end(then_b);
        b.build_call("gpu_sync", Type::void_type());
        b.build_br(merge, {});

        b.position_at_end(merge);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();

        GpuDivergenceAnalysis div(*fn);
        CHECK(div.is_divergent(tid));
        CHECK(div.is_divergent_branch(entry));
        CHECK(div.is_under_divergent_control(then_b));
        CHECK(!div.is_under_divergent_control(merge));

        DiagnosticReporter diag;
        CHECK(!verify_gpu_portability(*fn, strict, &diag));
        CHECK(diag.has_errors());
    }
}

TEST_CASE("GPU Targets - Native spirv-val and xcrun metal Validation") {
    Module mod("native_val");
    Function* fn = mod.create_function("native_val_k", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::f32(), Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop  = b.append_block("loop");
    BasicBlock* body  = b.append_block("body");
    BasicBlock* exit  = b.append_block("exit");

    b.position_at_end(entry);
    Value* in_ptr  = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    Value* alpha   = b.add_block_param(entry, Type::f32());
    Value* n       = b.add_block_param(entry, Type::i32());
    Value* tid     = b.build_call("gpu_tid_x", Type::i32());
    Value* smem    = b.build_call("gpu_shared_alloc_f32", Type::ptr(), {b.build_iconst_i32(32)});
    Value* smem_off = b.build_add(smem, b.build_shl(tid, b.build_iconst_i32(2)));
    Value* c0      = b.build_iconst_i32(0);
    Value* c1      = b.build_iconst_i32(1);
    b.build_br(loop, {c0, b.build_fconst_f32(0.0f)});

    b.position_at_end(loop);
    Value* iv  = b.add_block_param(loop, Type::i32());
    Value* acc = b.add_block_param(loop, Type::f32());
    b.build_br_if(b.build_slt(iv, n), body, {}, exit, {});

    b.position_at_end(body);
    Value* idx = b.build_add(tid, iv);
    Value* x   = b.build_load_indexed(Type::f32(), in_ptr, idx, 4);
    Value* h   = b.build_sdiv(x, b.build_fconst_f32(2.0f));
    Value* nxt = b.build_add(b.build_mul(alpha, h), acc);
    b.build_br(loop, {b.build_add(iv, c1), nxt});

    b.position_at_end(exit);
    b.build_call("gpu_shared_store_f32_indexed", Type::void_type(), {smem, tid, acc});
    b.build_call("gpu_sync", Type::void_type());
    Value* partner = b.build_call("gpu_shfl_xor_f32", Type::f32(), {acc, c1});
    Value* sv = b.build_call("gpu_shared_load_f32", Type::f32(), {smem_off});
    b.build_store_indexed(Type::f32(), out_ptr, tid, 4, b.build_add(partner, sv));
    b.build_ret_void();

    GpuOptOptions gopt;
    gopt.opt_level = 1;
    gopt.enable_fdiv_pow2 = true;
    gopt.enable_fma = true;
    CHECK(optimize_gpu_function(*fn, mod, gopt));
    DiagnosticReporter diag;
    CHECK(verify_gpu_portability(*fn, {}, &diag));

    target::SpirvOptions spv_opts;
    spv_opts.stage = target::SpirvStage::Compute;
    spv_opts.local_size_x = 32;
    target::SpirvKernel spv = target::SpirvTarget::compile(*fn, spv_opts);
    CHECK(!spv.words.empty());

    target::MslOptions msl_opts;
    msl_opts.workgroup_x = 32;
    target::MslKernel msl = target::MslTarget::compile(*fn, msl_opts);
    CHECK(!msl.source.empty());
    CHECK(!target::PtxTarget::emit_function(*fn).empty());

#if !defined(_WIN32)
    const std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
    if (std::system("command -v spirv-val >/dev/null 2>&1") == 0) {
        const std::filesystem::path spv_path = tmp_dir / "brass_native_val_test.spv";
        {
            std::ofstream os(spv_path, std::ios::binary | std::ios::trunc);
            os.write(reinterpret_cast<const char*>(spv.words.data()),
                     static_cast<std::streamsize>(spv.words.size() * sizeof(uint32_t)));
        }
        const std::string cmd = "spirv-val --target-env vulkan1.2 \"" + spv_path.string() + "\"";
        const int rc = std::system(cmd.c_str());
        std::error_code ec;
        std::filesystem::remove(spv_path, ec);
        CHECK_EQ(rc, 0);
    }
#if defined(__APPLE__)
    if (std::system("xcrun -sdk macosx metal --version >/dev/null 2>&1") == 0) {
        const std::filesystem::path mtl_path = tmp_dir / "brass_native_val_test.metal";
        const std::filesystem::path air_path = tmp_dir / "brass_native_val_test.air";
        {
            std::ofstream os(mtl_path, std::ios::trunc);
            os << msl.source;
        }
        const std::string cmd = "xcrun -sdk macosx metal -std=metal3.1 -c \"" +
                                mtl_path.string() + "\" -o \"" + air_path.string() + "\"";
        const int rc = std::system(cmd.c_str());
        std::error_code ec;
        std::filesystem::remove(mtl_path, ec);
        std::filesystem::remove(air_path, ec);
        CHECK_EQ(rc, 0);
    }
#endif
#endif
}
