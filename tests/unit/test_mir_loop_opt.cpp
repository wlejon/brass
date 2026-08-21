#include "test_framework.hpp"
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/lir.hpp>
#include "../benchmarks/bench_numeric_modules.hpp"
#include <brass/brass.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <vector>

using namespace brass;

TEST_CASE("MIR Loop Opt - Dominators and Natural Loop Detection") {
    Module mod("test_loop_dom");
    Builder b(mod);

    Function* fn = mod.create_function("simple_loop", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_hdr, {zero, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* next_i = b.build_add(i, one);
    Value* next_acc = b.build_add(acc, i);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* ret_val = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(ret_val);

    fn->rebuild_cfg_predecessors();

    DominatorTree dom(*fn);
    CHECK(dom.dominates(entry, loop_hdr));
    CHECK(dom.dominates(loop_hdr, loop_body));
    CHECK(dom.dominates(loop_hdr, exit_bb));

    LoopAnalysis loops(*fn, dom);
    CHECK_EQ(loops.top_level_loops().size(), 1);
    LoopInfo* loop = loops.top_level_loops()[0].get();
    CHECK_EQ(loop->header(), loop_hdr);
    CHECK(loop->contains(loop_hdr));
    CHECK(loop->contains(loop_body));
    CHECK(!loop->contains(entry));
    CHECK(!loop->contains(exit_bb));
}

TEST_CASE("MIR Loop Opt - LICM Hoisting Pure Invariant Computations") {
    Module mod("test_licm");
    Builder b(mod);

    Function* fn = mod.create_function("licm_fn", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_hdr, {zero, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* eight = b.build_iconst_i64(8);
    Value* inv_mul = b.build_mul(c, eight);
    Value* next_acc = b.build_add(acc, inv_mul);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* ret_val = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(ret_val);

    fn->rebuild_cfg_predecessors();

    LoopOptOptions options;
    options.enable_unroll = false;
    bool changed = optimize_function_loops(*fn, options);
    CHECK(changed);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));

    // Check that inv_mul was hoisted out of loop_body into entry / preheader
    BasicBlock* body_opt = fn->get_block_by_name("loop_body");
    REQUIRE(body_opt != nullptr);
    for (const Instruction* inst : *body_opt) {
        if (inst && inst->opcode() == Opcode::mul) {
            CHECK(false); // mul must NOT be in loop_body!
        }
    }
}

TEST_CASE("MIR Loop Opt - MatMul i64 Naive Optimization & JIT Execution") {
    auto mod = bench::build_matmul_i64_naive_module();
    Function* fn = mod->get_function("matmul_i64_naive");
    REQUIRE(fn != nullptr);

    bool changed = optimize_function_loops(*fn);
    CHECK(changed);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));

    // Test JIT execution across sizes
    codegen::JitExecutionEngine jit;
    bool compiled = jit.compile_and_load(*mod);
    CHECK(compiled);
    using MatMulFn = void(*)(const int64_t*, const int64_t*, int64_t*, int64_t);
    auto fn_ptr = jit.get_function_ptr<MatMulFn>("matmul_i64_naive");
    REQUIRE(fn_ptr != nullptr);

    for (int64_t size_n : {1, 2, 4, 8, 32}) {
        std::vector<int64_t> A_test(static_cast<size_t>(size_n * size_n), 2);
        std::vector<int64_t> B_test(static_cast<size_t>(size_n * size_n), 3);
        std::vector<int64_t> C_test(static_cast<size_t>(size_n * size_n), 0);
        fn_ptr(A_test.data(), B_test.data(), C_test.data(), size_n);
        int64_t expected = 2 * 3 * size_n;
        for (int64_t val : C_test) {
            CHECK_EQ(val, expected);
        }
    }
}

TEST_CASE("MIR Loop Opt - MatMul i64 Preopt Optimization & JIT Execution") {
    auto mod = bench::build_matmul_i64_preopt_module();
    Function* fn = mod->get_function("matmul_i64_preopt");
    REQUIRE(fn != nullptr);

    bool changed = optimize_function_loops(*fn);
    (void)changed;

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));

    // Test JIT execution across sizes
    codegen::JitExecutionEngine jit;
    bool compiled = jit.compile_and_load(*mod);
    CHECK(compiled);
    using MatMulFn = void(*)(const int64_t*, const int64_t*, int64_t*, int64_t);
    auto fn_ptr = jit.get_function_ptr<MatMulFn>("matmul_i64_preopt");
    REQUIRE(fn_ptr != nullptr);

    for (int64_t size_n : {1, 2, 4, 8, 32}) {
        std::vector<int64_t> A_test(static_cast<size_t>(size_n * size_n), 2);
        std::vector<int64_t> B_test(static_cast<size_t>(size_n * size_n), 3);
        std::vector<int64_t> C_test(static_cast<size_t>(size_n * size_n), 0);
        fn_ptr(A_test.data(), B_test.data(), C_test.data(), size_n);
        int64_t expected = 2 * 3 * size_n;
        for (int64_t val : C_test) {
            CHECK_EQ(val, expected);
        }
    }
}

TEST_CASE("MIR Loop Opt - MatMul f64 Naive Optimization & JIT Execution") {
    auto mod = bench::build_matmul_f64_naive_module();
    Function* fn = mod->get_function("matmul_f64_naive");
    REQUIRE(fn != nullptr);

    // Optimize function loops is performed by compilation pipeline in compile_and_load
    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));

    codegen::JitExecutionEngine jit;
    bool compiled = jit.compile_and_load(*mod);
    CHECK(compiled);
    using MatMulF64Fn = void(*)(const double*, const double*, double*, int64_t);
    auto fn_ptr = jit.get_function_ptr<MatMulF64Fn>("matmul_f64_naive");
    REQUIRE(fn_ptr != nullptr);

    for (int64_t size_n : {1, 2, 4, 8, 32}) {
        std::vector<double> A_test(static_cast<size_t>(size_n * size_n), 2.5);
        std::vector<double> B_test(static_cast<size_t>(size_n * size_n), 4.0);
        std::vector<double> C_test(static_cast<size_t>(size_n * size_n), 0.0);
        fn_ptr(A_test.data(), B_test.data(), C_test.data(), size_n);
        double expected = 2.5 * 4.0 * static_cast<double>(size_n);
        for (double val : C_test) {
            CHECK(std::abs(val - expected) < 1e-9);
        }
    }
}
