#include "test_framework.hpp"
#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <vector>

using namespace brass;

TEST_CASE("Auto-Parallelization Pass: DOALL Vector Scale") {
    Module mod("auto_par_mod");
    Function* fn = mod.create_function("vec_scale", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::i64(), Type::i64()
    });
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_arr = b.add_block_param(entry, Type::ptr());
    Value* out_arr = b.add_block_param(entry, Type::ptr());
    Value* factor = b.add_block_param(entry, Type::i64());
    Value* n = b.add_block_param(entry, Type::i64());
    b.position_at_end(entry);

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(hdr, {zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, n);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);
    Value* elem = b.build_load_indexed(Type::i64(), in_arr, iv, 8, 0);
    Value* scaled = b.build_mul(elem, factor);
    b.build_store_indexed(Type::i64(), out_arr, iv, 8, 0, scaled);
    Value* next_iv = b.build_add(iv, one);
    b.build_br(hdr, {next_iv});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DominatorTree dom(*fn);
    ParallelLoopStats stats;
    ParallelLoopOptions opts;
    opts.stats = &stats;
    opts.parallel_threshold = 100; // Low threshold for testing

    bool changed = auto_parallelize_function(*fn, dom, opts);
    CHECK(changed);
    CHECK_EQ(stats.parallel_loops_transformed, 1u);

    DiagnosticReporter diag;
    if (!verify_module(mod, &diag)) {
        std::cout << "DIAGNOSTIC ERROR: " << diag.format_all() << "\n";
    }
    REQUIRE(verify_module(mod, &diag));

    // JIT execute
    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
    jit.register_external_symbol("brass_set_parallel_workers", reinterpret_cast<void*>(&brass_set_parallel_workers));
    jit.register_external_symbol("brass_get_parallel_workers", reinterpret_cast<void*>(&brass_get_parallel_workers));
    jit.register_external_symbol("brass_parallel_reduce_i64", reinterpret_cast<void*>(&brass_parallel_reduce_i64));
    jit.register_external_symbol("brass_parallel_reduce_f64", reinterpret_cast<void*>(&brass_parallel_reduce_f64));
    jit.register_external_symbol("brass_parallel_alloc_context", reinterpret_cast<void*>(&brass_parallel_alloc_context));
    jit.register_external_symbol("brass_parallel_free_context", reinterpret_cast<void*>(&brass_parallel_free_context));

    REQUIRE(jit.compile_and_load(mod));

    auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t, int64_t)>("vec_scale");
    REQUIRE(fn_ptr != nullptr);

    const int64_t N = 1000;
    std::vector<int64_t> in_data(N);
    std::vector<int64_t> out_data(N, 0);
    for (int64_t i = 0; i < N; ++i) {
        in_data[static_cast<size_t>(i)] = i + 1;
    }

    fn_ptr(in_data.data(), out_data.data(), 5, N);

    for (int64_t i = 0; i < N; ++i) {
        CHECK_EQ(out_data[static_cast<size_t>(i)], (i + 1) * 5);
    }
}

TEST_CASE("Auto-Parallelization Pass: Sum Reduction Loop") {
    Module mod("auto_par_red");
    Function* fn = mod.create_function("parallel_sum", Type::i64(), {
        Type::ptr(), Type::i64()
    });
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_arr = b.add_block_param(entry, Type::ptr());
    Value* n = b.add_block_param(entry, Type::i64());
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
    Value* cond = b.build_slt(iv, n);
    b.build_br_if(cond, body, {}, exit, {acc});

    fn->append_block(body);
    b.position_at_end(body);
    Value* elem = b.build_load_indexed(Type::i64(), in_arr, iv, 8, 0);
    Value* next_acc = b.build_add(acc, elem);
    Value* next_iv = b.build_add(iv, one);
    b.build_br(hdr, {next_iv, next_acc});

    fn->append_block(exit);
    b.position_at_end(exit);
    Value* res = b.add_block_param(exit, Type::i64());
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();

    DominatorTree dom(*fn);
    ParallelLoopStats stats;
    ParallelLoopOptions opts;
    opts.stats = &stats;
    opts.parallel_threshold = 100;

    bool changed = auto_parallelize_function(*fn, dom, opts);
    CHECK(changed);
    CHECK_EQ(stats.parallel_loops_transformed, 1u);

    DiagnosticReporter diag;
    if (!verify_module(mod, &diag)) {
        std::cout << "DIAGNOSTIC ERROR RED: " << diag.format_all() << "\n";
    }
    REQUIRE(verify_module(mod, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));
    jit.register_external_symbol("brass_set_parallel_workers", reinterpret_cast<void*>(&brass_set_parallel_workers));
    jit.register_external_symbol("brass_get_parallel_workers", reinterpret_cast<void*>(&brass_get_parallel_workers));
    jit.register_external_symbol("brass_parallel_reduce_i64", reinterpret_cast<void*>(&brass_parallel_reduce_i64));
    jit.register_external_symbol("brass_parallel_reduce_f64", reinterpret_cast<void*>(&brass_parallel_reduce_f64));
    jit.register_external_symbol("brass_parallel_alloc_context", reinterpret_cast<void*>(&brass_parallel_alloc_context));
    jit.register_external_symbol("brass_parallel_free_context", reinterpret_cast<void*>(&brass_parallel_free_context));

    REQUIRE(jit.compile_and_load(mod));

    auto fn_ptr = jit.get_function_ptr<int64_t(*)(int64_t*, int64_t)>("parallel_sum");
    REQUIRE(fn_ptr != nullptr);

    const int64_t N = 2000;
    std::vector<int64_t> in_data(N);
    int64_t expected_sum = 0;
    for (int64_t i = 0; i < N; ++i) {
        in_data[static_cast<size_t>(i)] = i + 1;
        expected_sum += (i + 1);
    }

    int64_t actual_sum = fn_ptr(in_data.data(), N);
    CHECK_EQ(actual_sum, expected_sum);
}

TEST_CASE("Auto-Parallelization: Cost Model Threshold") {
    Module mod("cost_mod");
    Function* fn = mod.create_function("tiny_loop", Type::void_type(), {Type::ptr()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ptr = b.add_block_param(entry, Type::ptr());
    b.position_at_end(entry);

    BasicBlock* hdr = b.create_block("hdr");
    BasicBlock* body = b.create_block("body");
    BasicBlock* exit = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* const_ten = b.build_iconst_i64(10); // Constant trip count = 10
    b.build_br(hdr, {zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, const_ten);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);
    b.build_store_indexed(Type::i64(), ptr, iv, 8, 0, iv);
    Value* next_iv = b.build_add(iv, one);
    b.build_br(hdr, {next_iv});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DominatorTree dom(*fn);
    ParallelLoopStats stats;
    ParallelLoopOptions opts;
    opts.stats = &stats;
    opts.parallel_threshold = 10000; // Threshold higher than 10 * ~3 instructions

    bool changed = auto_parallelize_function(*fn, dom, opts);
    CHECK(!changed);
    CHECK(stats.loops_rejected_cost > 0u);
}
