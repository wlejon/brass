#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>
#include <numeric>

using namespace brass;

TEST_CASE("BCE - Local Constant Bounds Check Elimination (In Bounds)") {
    Module mod("test_bce_local_in_bounds");
    Builder b(mod);

    Function* fn = mod.create_function("test_in_bounds", Type::i32(), {});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* bb_then = b.create_block("bb_then");
    BasicBlock* bb_else = b.create_block("bb_else");

    // len = 10, idx = 3 => ult 3, 10 is true
    Value* len = b.build_iconst_i64(10);
    Value* idx = b.build_iconst_i64(3);
    Value* cond = b.build_ult(idx, len);
    b.build_br_if(cond, bb_then, {}, bb_else, {});

    fn->append_block(bb_then);
    b.position_at_end(bb_then);
    Value* ret_then = b.build_iconst_i32(42);
    b.build_ret(ret_then);

    fn->append_block(bb_else);
    b.position_at_end(bb_else);
    Value* ret_else = b.build_iconst_i32(0);
    b.build_ret(ret_else);

    fn->rebuild_cfg_predecessors();

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;

    bool changed = run_bounds_check_elimination(*fn, mod, opts);
    CHECK(changed);
    CHECK(stats.bounds_checks_eliminated > 0);
    CHECK(stats.branches_folded > 0);

    // After folding and CFG simplification, bb_then is merged into entry with ret 42
    Instruction* term = entry->terminator();
    REQUIRE(term != nullptr);
    CHECK_EQ(term->opcode(), Opcode::ret);
    REQUIRE(term->operand(0) != nullptr);
    REQUIRE(term->operand(0)->defining_instruction() != nullptr);
    CHECK_EQ(term->operand(0)->defining_instruction()->imm_i32(), 42);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
}

TEST_CASE("BCE - Local Constant Bounds Check Elimination (Out of Bounds)") {
    Module mod("test_bce_local_oob");
    Builder b(mod);

    Function* fn = mod.create_function("test_oob", Type::i32(), {});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* bb_then = b.create_block("bb_then");
    BasicBlock* bb_else = b.create_block("bb_else");

    // len = 5, idx = 10 => ult 10, 5 is false
    Value* len = b.build_iconst_i64(5);
    Value* idx = b.build_iconst_i64(10);
    Value* cond = b.build_ult(idx, len);
    b.build_br_if(cond, bb_then, {}, bb_else, {});

    fn->append_block(bb_then);
    b.position_at_end(bb_then);
    Value* ret_then = b.build_iconst_i32(1);
    b.build_ret(ret_then);

    fn->append_block(bb_else);
    b.position_at_end(bb_else);
    Value* ret_else = b.build_iconst_i32(99);
    b.build_ret(ret_else);

    fn->rebuild_cfg_predecessors();

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;

    bool changed = run_bounds_check_elimination(*fn, mod, opts);
    CHECK(changed);
    CHECK(stats.bounds_checks_eliminated > 0);
    CHECK(stats.branches_folded > 0);

    // After folding and CFG simplification, bb_else is merged into entry with ret 99
    Instruction* term = entry->terminator();
    REQUIRE(term != nullptr);
    CHECK_EQ(term->opcode(), Opcode::ret);
    REQUIRE(term->operand(0) != nullptr);
    REQUIRE(term->operand(0)->defining_instruction() != nullptr);
    CHECK_EQ(term->operand(0)->defining_instruction()->imm_i32(), 99);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
}

TEST_CASE("BCE - Redundant Guard Elimination") {
    Module mod("test_bce_guard");
    Builder b(mod);

    Function* fn = mod.create_function("test_guard", Type::i32(), {});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* true_cond = b.build_iconst_i32(1);
    b.build_guard(true_cond, "@test_stub");
    Value* ret_val = b.build_iconst_i32(7);
    b.build_ret(ret_val);

    fn->rebuild_cfg_predecessors();

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;

    bool changed = run_bounds_check_elimination(*fn, mod, opts);
    CHECK(changed);
    CHECK(stats.guards_eliminated > 0);

    // Verify guard instruction was removed
    for (Instruction* inst : *entry) {
        CHECK_NE(inst->opcode(), Opcode::guard);
    }

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
}

TEST_CASE("BCE - Loop Bounds Check Hoisting") {
    Module mod("test_bce_loop_hoist");
    Builder b(mod);

    // Function: int64 sum_loop(int64* arr, int64 len, int64 N)
    Function* fn = mod.create_function("sum_loop", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* arr = b.add_block_param(entry, Type::i64());
    Value* len = b.add_block_param(entry, Type::i64());
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* preheader = b.create_block("preheader");
    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_check = b.create_block("loop_check");
    BasicBlock* loop_fast = b.create_block("loop_fast");
    BasicBlock* loop_fallback = b.create_block("loop_fallback");
    BasicBlock* latch = b.create_block("latch");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(preheader, {});

    fn->append_block(preheader);
    b.position_at_end(preheader);
    b.build_br(loop_hdr, {zero, zero});

    // loop_hdr: (i, acc)
    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_check, {}, exit_bb, {acc});

    // loop_check: bounds check ult i, len
    fn->append_block(loop_check);
    b.position_at_end(loop_check);
    Value* in_bounds = b.build_ult(i, len);
    b.build_br_if(in_bounds, loop_fast, {}, loop_fallback, {});

    // loop_fast: load arr[i] and add to acc
    fn->append_block(loop_fast);
    b.position_at_end(loop_fast);
    Value* elem = b.build_load_indexed(Type::i64(), arr, i, 8, 0);
    Value* next_acc_fast = b.build_add(acc, elem);
    b.build_br(latch, {next_acc_fast});

    // loop_fallback: return fallback / dummy
    fn->append_block(loop_fallback);
    b.position_at_end(loop_fallback);
    b.build_br(latch, {acc});

    // latch: i + 1 -> loop_hdr
    fn->append_block(latch);
    b.position_at_end(latch);
    Value* cur_acc = b.add_block_param(latch, Type::i64());
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, cur_acc});

    // exit
    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_sum = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;

    bool changed = run_bounds_check_elimination(*fn, mod, opts);
    CHECK(changed);
    CHECK(stats.bounds_checks_hoisted > 0);

    // Verify hoisted guard exists in loop preheader / entry
    bool found_guard = false;
    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::guard) {
                found_guard = true;
                CHECK_EQ(inst->symbol(), "@exit_stub");
            }
        }
    }
    CHECK(found_guard);

    // loop_fast is merged into loop_check by cfg_simplify, branching directly to latch
    bool found_load = false;
    for (Instruction* inst : *loop_check) {
        if (inst->opcode() == Opcode::load_indexed) {
            found_load = true;
        }
    }
    CHECK(found_load);

    Instruction* check_term = loop_check->terminator();
    REQUIRE(check_term != nullptr);
    CHECK_EQ(check_term->opcode(), Opcode::br);
    // cfg_simplify collapses loop_fast and latch into loop_check, branching straight to loop_hdr
    CHECK(check_term->branch_target().block == loop_hdr || check_term->branch_target().block == latch);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
}

TEST_CASE("BCE - JIT Execution of Hoisted Bounds Checked Loop") {
    Module mod("test_bce_jit");
    Builder b(mod);

    // Function: int64 sum_array(int64* arr, int64 len, int64 N)
    Function* fn = mod.create_function("sum_array", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* arr = b.add_block_param(entry, Type::i64());
    Value* len = b.add_block_param(entry, Type::i64());
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* preheader = b.create_block("preheader");
    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_check = b.create_block("loop_check");
    BasicBlock* loop_fast = b.create_block("loop_fast");
    BasicBlock* loop_fallback = b.create_block("loop_fallback");
    BasicBlock* latch = b.create_block("latch");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(preheader, {});

    fn->append_block(preheader);
    b.position_at_end(preheader);
    b.build_br(loop_hdr, {zero, zero});

    // loop_hdr: (i, acc)
    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_check, {}, exit_bb, {acc});

    // loop_check: bounds check ult i, len
    fn->append_block(loop_check);
    b.position_at_end(loop_check);
    Value* in_bounds = b.build_ult(i, len);
    b.build_br_if(in_bounds, loop_fast, {}, loop_fallback, {});

    // loop_fast: load arr[i] and add to acc
    fn->append_block(loop_fast);
    b.position_at_end(loop_fast);
    Value* elem = b.build_load_indexed(Type::i64(), arr, i, 8, 0);
    Value* next_acc_fast = b.build_add(acc, elem);
    b.build_br(latch, {next_acc_fast});

    // loop_fallback: return acc
    fn->append_block(loop_fallback);
    b.position_at_end(loop_fallback);
    b.build_br(latch, {acc});

    // latch: i + 1 -> loop_hdr
    fn->append_block(latch);
    b.position_at_end(latch);
    Value* cur_acc = b.add_block_param(latch, Type::i64());
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, cur_acc});

    // exit
    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_sum = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();

    // Optimize loop using LoopOptOptions with enable_bce = true
    LoopOptOptions loop_opts;
    loop_opts.enable_bce = true;
    loop_opts.enable_unroll = false;
    loop_opts.enable_vectorize = false;
    bool changed = optimize_function_loops(*fn, loop_opts);
    CHECK(changed);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));

    // JIT compile and execute
    codegen::JitExecutionEngine jit;
    bool compiled = jit.compile_and_load(mod);
    CHECK(compiled);

    using SumFn = int64_t(*)(const int64_t*, int64_t, int64_t);
    auto sum_ptr = jit.get_function_ptr<SumFn>("sum_array");
    REQUIRE(sum_ptr != nullptr);

    std::vector<int64_t> test_data(100);
    for (size_t idx = 0; idx < test_data.size(); ++idx) {
        test_data[idx] = static_cast<int64_t>(idx + 1);
    }
    int64_t expected_sum = 100 * 101 / 2; // 5050
    int64_t actual_sum = sum_ptr(test_data.data(), 100, 100);
    CHECK_EQ(actual_sum, expected_sum);

    // Partial sum test
    int64_t partial_expected = 50 * 51 / 2; // 1275
    int64_t partial_actual = sum_ptr(test_data.data(), 100, 50);
    CHECK_EQ(partial_actual, partial_expected);
}
