#include "test_framework.hpp"

#include <brass/mir/builder.hpp>
#include <brass/mir/gpu_opt.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/msl_target.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv_target.hpp>

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
