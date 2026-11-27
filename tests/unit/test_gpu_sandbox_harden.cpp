#include "test_framework.hpp"

#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/verify/gpu_sandbox_instrument.hpp>

#include <cstdint>
#include <vector>

using namespace brass;

namespace {

Function* build_safe_saxpy_kernel(Module& mod, const char* name) {
    // params: (ptr out, ptr in, i32 n)
    Function* fn = mod.create_function(name, Type::void_type(), {Type::ptr(), Type::ptr(), Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* work = b.append_block("work");
    BasicBlock* done = b.append_block("done");

    b.position_at_end(entry);
    Value* out = b.add_block_param(entry, Type::ptr());
    Value* in = b.add_block_param(entry, Type::ptr());
    Value* n = b.add_block_param(entry, Type::i32());
    Value* gid = b.build_call("gpu_global_tid_x", Type::i32());
    Value* ok = b.build_and(b.build_sge(gid, b.build_iconst_i32(0)), b.build_slt(gid, n));
    b.build_br_if(ok, work, {}, done, {});

    b.position_at_end(work);
    Value* x = b.build_load_indexed(Type::f32(), in, gid, 4);
    Value* y = b.build_fadd(b.build_mul(x, b.build_fconst_f32(2.0f)), b.build_fconst_f32(1.0f));
    b.build_store_indexed(Type::f32(), out, gid, 4, y);
    b.build_br(done);

    b.position_at_end(done);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();
    return fn;
}

} // namespace

TEST_CASE("GpuSandboxHarden - Pure SMT Zero-Overhead Fast Path + PTX/MSL Parity") {
    Module mod("safe_gpu");
    Function* fn = build_safe_saxpy_kernel(mod, "safe_saxpy");

    GpuHardenOptions hopts;
    hopts.sandbox_opts.verify_spirv = true;
    hopts.sandbox_opts.run_smt_prover = true;
    hopts.sandbox_opts.verify_ptx_compile = true;
    hopts.sandbox_opts.verify_ptx_equiv = true;
    hopts.sandbox_opts.verify_msl_compile = true;
    hopts.sandbox_opts.prover_opts.local_size_x = 16;
    hopts.sandbox_opts.prover_opts.grid_size_x = 4;

    GpuBufferSpec b0{};
    b0.param_index = 0;
    b0.length_param_index = 2;
    b0.elem_bytes = 4;
    GpuBufferSpec b1{};
    b1.param_index = 1;
    b1.length_param_index = 2;
    b1.elem_bytes = 4;
    hopts.sandbox_opts.prover_opts.buffers = {b0, b1};

    const GpuHardenResult hr = harden_and_verify_gpu_sandbox(*fn, hopts);
    CHECK(hr.passed());
    CHECK(hr.initial_sandbox.ptx_compiled);
    CHECK(hr.initial_sandbox.ptx_equiv_ok);
    CHECK(hr.initial_sandbox.msl_compiled);
#if BRASS_WITH_SPIRV
    CHECK(hr.tier == HardeningTier::PureSmtZeroOverhead);
#endif
    CHECK(hr.clamp.clamped_global_loads == 0);
    CHECK(hr.clamp.clamped_global_stores == 0);
    CHECK(hr.fuel.loops_fuel_instrumented == 0);
}

TEST_CASE("GpuSandboxHarden - Indirect Gather/Scatter Clamping + Runtime OOB Check") {
    Module mod("indirect_gpu");
    // params: (ptr out, ptr in, ptr idx_buf, i32 n)
    Function* fn = mod.create_function("gather_scatter", Type::void_type(),
                                       {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});
    {
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* work = b.append_block("work");
        BasicBlock* done = b.append_block("done");

        b.position_at_end(entry);
        Value* out = b.add_block_param(entry, Type::ptr());
        Value* in = b.add_block_param(entry, Type::ptr());
        Value* idx_buf = b.add_block_param(entry, Type::ptr());
        Value* n = b.add_block_param(entry, Type::i32());
        Value* tid = b.build_call("gpu_global_tid_x", Type::i32());
        Value* ok = b.build_and(b.build_sge(tid, b.build_iconst_i32(0)), b.build_slt(tid, n));
        b.build_br_if(ok, work, {}, done, {});

        b.position_at_end(work);
        Value* target_i = b.build_load_indexed(Type::i32(), idx_buf, tid, 4);
        // Indirect load & indirect store with data-dependent `target_i`
        Value* val = b.build_load_indexed(Type::i32(), in, target_i, 4);
        Value* val_plus_1 = b.build_add(val, b.build_iconst_i32(1));
        b.build_store_indexed(Type::i32(), out, target_i, 4, val_plus_1);
        b.build_br(done);

        b.position_at_end(done);
        b.build_ret_void();
    }
    fn->rebuild_cfg_predecessors();

    GpuHardenOptions hopts;
    hopts.sandbox_opts.verify_spirv = true;
    hopts.sandbox_opts.run_smt_prover = true;
    hopts.sandbox_opts.prover_opts.local_size_x = 8;
    hopts.sandbox_opts.prover_opts.grid_size_x = 1;

    GpuBufferSpec bout{0, 0, 3, 4};
    GpuBufferSpec bin{1, 0, 3, 4};
    GpuBufferSpec bidx{2, 0, 3, 4};
    hopts.sandbox_opts.prover_opts.buffers = {bout, bin, bidx};

    const GpuHardenResult hr = harden_and_verify_gpu_sandbox(*fn, hopts);
    CHECK(hr.passed());
    CHECK(hr.tier == HardeningTier::HybridSmtClamped);
    CHECK(hr.clamp.clamped_global_loads >= 2);
    CHECK(hr.clamp.clamped_global_stores >= 1);
    CHECK(hr.final_sandbox.portability_ok);
#if BRASS_WITH_SPIRV
    CHECK(hr.final_sandbox.spirv_compiled);
    CHECK(hr.final_sandbox.spirv_parsed);
#endif

    // Execute the hardened MIR on host buffers via Interpreter with malicious indices
    // Alloc buffers with canary guard sentinels before/after the declared length (4 elems)
    std::vector<int32_t> out_backing(8, -777);
    std::vector<int32_t> in_backing(8, 999);
    // Active 4 elements start at index +2
    int32_t* out_ptr = out_backing.data() + 2;
    int32_t* in_ptr = in_backing.data() + 2;
    for (int32_t i = 0; i < 4; ++i) {
        out_ptr[i] = 0;
        in_ptr[i] = (i + 1) * 10; // {10, 20, 30, 40}
    }
    // Thread indices: 0 -> idx 2 (in-bounds: 30+1=31), 1 -> idx 10 (OOB store!),
    //                 2 -> idx -1 (OOB!), 3 -> idx 4 (OOB!)
    int32_t idx_data[4] = {2, 10, -1, 4};

    for (int32_t tid_v = 0; tid_v < 4; ++tid_v) {
        Interpreter interp;
        interp.set_module(&mod);
        interp.register_external_function("gpu_global_tid_x",
            [tid_v](Interpreter&, const std::vector<RuntimeValue>&) {
                return RuntimeValue::from_i32(tid_v);
            });
        interp.run(*fn, {
            RuntimeValue::from_ptr(out_ptr),
            RuntimeValue::from_ptr(in_ptr),
            RuntimeValue::from_ptr(idx_data),
            RuntimeValue::from_i32(4)
        });
    }

    // Valid element updated
    CHECK(out_ptr[2] == 31);
    // Out-of-bounds canary sentinels completely untouched!
    CHECK(out_backing[0] == -777);
    CHECK(out_backing[1] == -777);
    CHECK(out_backing[6] == -777);
    CHECK(out_backing[7] == -777);
}

TEST_CASE("GpuSandboxHarden - Loop Fuel Bounds Unbounded Loop + Terminates") {
    Module mod("fuel_gpu");
    // Kernel with an unbounded loop (`while (true) { acc++; }`) that stores result to out[0]
    Function* fn = mod.create_function("unbounded_loop", Type::void_type(), {Type::ptr()});
    {
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* loop = b.append_block("loop");
        BasicBlock* exit = b.append_block("exit");

        b.position_at_end(entry);
        Value* out = b.add_block_param(entry, Type::ptr());
        b.build_br(loop, {b.build_iconst_i32(0)});

        b.position_at_end(loop);
        Value* cnt = b.add_block_param(loop, Type::i32());
        Value* always_true = b.build_iconst_i32(1);
        Value* next_cnt = b.build_add(cnt, b.build_iconst_i32(1));
        b.build_store_indexed(Type::i32(), out, b.build_iconst_i32(0), 4, next_cnt);
        b.build_br_if(always_true, loop, {next_cnt}, exit, {});

        b.position_at_end(exit);
        b.build_ret_void();
    }
    fn->rebuild_cfg_predecessors();

    GpuHardenOptions hopts;
    hopts.fuel_opts.max_loop_iterations = 25;
    GpuBufferSpec b0{0, 16, -1, 4};
    hopts.sandbox_opts.prover_opts.buffers = {b0};

    const GpuHardenResult hr = harden_and_verify_gpu_sandbox(*fn, hopts);
    CHECK(hr.passed());
    CHECK(hr.fuel.loops_fuel_instrumented == 1);
#if BRASS_WITH_SPIRV
    CHECK(hr.final_sandbox.spirv_compiled);
    CHECK(hr.final_sandbox.spirv_parsed);
#endif

    int32_t out_val = 0;
    Interpreter interp;
    interp.set_module(&mod);
    interp.run(*fn, {RuntimeValue::from_ptr(&out_val)});
    CHECK(out_val == 25);
}

TEST_CASE("GpuSandboxHarden - Reject Untracked Pointer and Divergent Unbounded Barrier Loop") {
    // 1. Untracked derived pointer loaded from buffer `ptrs[0]`
    {
        Module mod("untracked_gpu");
        const uint32_t f_id = mod.debug_context().get_or_add_file("user_shader.bk");
        Function* fn = mod.create_function("bad_ptr", Type::void_type(), {Type::ptr()});
        {
            Builder b(mod);
            b.set_function(fn);
            BasicBlock* entry = b.append_block("entry");
            b.position_at_end(entry);
            Value* pbuf = b.add_block_param(entry, Type::ptr());
            b.set_current_loc(f_id, 19, 3);
            Value* leaked_ptr = b.build_load(Type::ptr(), pbuf);
            Value* v = b.build_load(Type::i32(), leaked_ptr);
            b.build_store(Type::i32(), pbuf, v);
            b.build_ret_void();
        }
        GpuHardenOptions hopts;
        hopts.sandbox_opts.verify_spirv = false;
        hopts.clamp_opts.debug_ctx = &mod.debug_context();
        GpuBufferSpec b0{0, 64, -1, 8};
        hopts.sandbox_opts.prover_opts.buffers = {b0};

        const GpuHardenResult hr = harden_and_verify_gpu_sandbox(*fn, hopts);
        CHECK(!hr.passed());
        CHECK(hr.tier == HardeningTier::Rejected);
        CHECK(hr.clamp.untracked_accesses >= 1);
        CHECK(hr.diagnostic.find("user_shader.bk:19") != std::string::npos);
    }

    // 2. Divergent unbounded barrier loop
    {
        Module mod("div_loop_gpu");
        Function* fn = mod.create_function("div_bar_loop", Type::void_type(), {Type::ptr(), Type::i32()});
        {
            Builder b(mod);
            b.set_function(fn);
            BasicBlock* entry = b.append_block("entry");
            BasicBlock* loop = b.append_block("loop");
            BasicBlock* exit = b.append_block("exit");

            b.position_at_end(entry);
            Value* out = b.add_block_param(entry, Type::ptr());
            Value* n = b.add_block_param(entry, Type::i32());
            Value* tid = b.build_call("gpu_tid_x", Type::i32());
            b.build_br_if(b.build_slt(tid, b.build_iconst_i32(4)), loop, {b.build_iconst_i32(0)}, exit, {});

            b.position_at_end(loop);
            Value* i = b.add_block_param(loop, Type::i32());
            b.build_call("gpu_sync", Type::void_type());
            Value* ni = b.build_add(i, b.build_iconst_i32(1));
            b.build_store_indexed(Type::i32(), out, tid, 4, ni);
            b.build_br_if(b.build_slt(ni, n), loop, {ni}, exit, {});

            b.position_at_end(exit);
            b.build_ret_void();
        }
        fn->rebuild_cfg_predecessors();

        GpuHardenOptions hopts;
        hopts.sandbox_opts.verify_spirv = false;
        GpuBufferSpec b0{0, 64, -1, 4};
        hopts.sandbox_opts.prover_opts.buffers = {b0};

        const GpuHardenResult hr = harden_and_verify_gpu_sandbox(*fn, hopts);
        CHECK(!hr.passed());
        CHECK(hr.tier == HardeningTier::Rejected);

        const GpuLoopFuelResult direct_fuel = instrument_gpu_loop_fuel(*fn);
        CHECK(!direct_fuel.ok);
        CHECK(!direct_fuel.violations.empty());
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
