#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_parallel.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <vector>
#include <random>

using namespace brass;

namespace {

std::unique_ptr<Module> build_doall_module(std::string_view name) {
    auto mod = std::make_unique<Module>(name);
    Function* fn = mod->create_function("transform_array", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* three = b.build_iconst_i64(3);
    Value* seven = b.build_iconst_i64(7);
    b.build_br(hdr, {zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, N);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);
    Value* elem = b.build_load_indexed(Type::i64(), in_ptr, iv, 8, 0);
    Value* mul3 = b.build_mul(elem, three);
    Value* add7 = b.build_add(mul3, seven);
    b.build_store_indexed(Type::i64(), out_ptr, iv, 8, 0, add7);
    Value* next_iv = b.build_add(iv, one);
    b.build_br(hdr, {next_iv});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_sum_reduce_module(std::string_view name) {
    auto mod = std::make_unique<Module>(name);
    Function* fn = mod->create_function("sum_array", Type::i64(), {
        Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(hdr, {zero, zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* acc = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, N);
    b.build_br_if(cond, body, {}, exit, {acc});

    fn->append_block(body);
    b.position_at_end(body);
    Value* elem = b.build_load_indexed(Type::i64(), in_ptr, iv, 8, 0);
    Value* next_acc = b.build_add(acc, elem);
    Value* next_iv = b.build_add(iv, one);
    b.build_br(hdr, {next_iv, next_acc});

    fn->append_block(exit);
    b.position_at_end(exit);
    Value* res = b.add_block_param(exit, Type::i64());
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

TEST_CASE("Differential: DOALL Parallel JIT vs Sequential Interpreter") {
    const std::vector<int64_t> trip_counts = {1, 10, 100, 1000, 10000, 50000};
    const std::vector<uint32_t> worker_counts = {1, 2, 4};

    for (int64_t n : trip_counts) {
        std::vector<int64_t> input(static_cast<size_t>(n));
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<int64_t>(i * 2 + 5);
        }

        // 1. Sequential Interpreter Baseline
        auto seq_mod = build_doall_module("seq_mod");
        std::vector<int64_t> seq_output(static_cast<size_t>(n), 0);
        Interpreter interp;
        interp.run(*seq_mod->get_function("transform_array"), {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(input.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(seq_output.data())),
            RuntimeValue::from_i64(n)
        });

        // 2. Parallel JIT Execution
        for (uint32_t workers : worker_counts) {
            brass_set_parallel_workers(workers);

            auto par_mod = build_doall_module("par_mod");
            Function* par_fn = par_mod->get_function("transform_array");
            DominatorTree dom(*par_fn);
            ParallelLoopOptions opts;
            opts.parallel_threshold = 10; // ensure parallelization triggers for n >= 10
            auto_parallelize_function(*par_fn, dom, opts);

            codegen::JitExecutionEngine jit(Target::host());
            jit.register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
            jit.register_external_symbol("brass_set_parallel_workers", reinterpret_cast<void*>(&brass_set_parallel_workers));
            jit.register_external_symbol("brass_get_parallel_workers", reinterpret_cast<void*>(&brass_get_parallel_workers));
            jit.register_external_symbol("brass_parallel_reduce_i64", reinterpret_cast<void*>(&brass_parallel_reduce_i64));
            jit.register_external_symbol("brass_parallel_reduce_f64", reinterpret_cast<void*>(&brass_parallel_reduce_f64));
            jit.register_external_symbol("brass_parallel_alloc_context", reinterpret_cast<void*>(&brass_parallel_alloc_context));
            jit.register_external_symbol("brass_parallel_free_context", reinterpret_cast<void*>(&brass_parallel_free_context));

            REQUIRE(jit.compile_and_load(*par_mod));

            auto par_exec = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t)>("transform_array");
            REQUIRE(par_exec != nullptr);

            std::vector<int64_t> par_output(static_cast<size_t>(n), 0);
            par_exec(input.data(), par_output.data(), n);

            CHECK(par_output == seq_output);
        }
    }
}

TEST_CASE("Differential: Reduction Parallel JIT vs Sequential Interpreter") {
    const std::vector<int64_t> trip_counts = {1, 10, 100, 1000, 10000, 50000};
    const std::vector<uint32_t> worker_counts = {1, 2, 4};

    for (int64_t n : trip_counts) {
        std::vector<int64_t> input(static_cast<size_t>(n));
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<int64_t>((i % 17) + 1);
        }

        // 1. Sequential Interpreter Baseline
        auto seq_mod = build_sum_reduce_module("seq_red_mod");
        Interpreter interp;
        RuntimeValue seq_res = interp.run(*seq_mod->get_function("sum_array"), {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(input.data())),
            RuntimeValue::from_i64(n)
        });
        int64_t expected_sum = seq_res.as_i64();

        // 2. Parallel JIT Execution
        for (uint32_t workers : worker_counts) {
            brass_set_parallel_workers(workers);

            auto par_mod = build_sum_reduce_module("par_red_mod");
            Function* par_fn = par_mod->get_function("sum_array");
            DominatorTree dom(*par_fn);
            ParallelLoopOptions opts;
            opts.parallel_threshold = 10;
            auto_parallelize_function(*par_fn, dom, opts);

            codegen::JitExecutionEngine jit(Target::host());
            jit.register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
            jit.register_external_symbol("brass_set_parallel_workers", reinterpret_cast<void*>(&brass_set_parallel_workers));
            jit.register_external_symbol("brass_get_parallel_workers", reinterpret_cast<void*>(&brass_get_parallel_workers));
            jit.register_external_symbol("brass_parallel_reduce_i64", reinterpret_cast<void*>(&brass_parallel_reduce_i64));
            jit.register_external_symbol("brass_parallel_reduce_f64", reinterpret_cast<void*>(&brass_parallel_reduce_f64));
            jit.register_external_symbol("brass_parallel_alloc_context", reinterpret_cast<void*>(&brass_parallel_alloc_context));
            jit.register_external_symbol("brass_parallel_free_context", reinterpret_cast<void*>(&brass_parallel_free_context));

            REQUIRE(jit.compile_and_load(*par_mod));

            auto par_exec = jit.get_function_ptr<int64_t(*)(int64_t*, int64_t)>("sum_array");
            REQUIRE(par_exec != nullptr);

            int64_t actual_sum = par_exec(input.data(), n);
            CHECK_EQ(actual_sum, expected_sum);
        }
    }
}
