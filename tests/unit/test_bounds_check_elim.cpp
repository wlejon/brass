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

namespace {

// int64 sum(int64* arr, int64 len, int64 n):
//   for (i = 0; i < n; ++i) acc += (i <u bound) ? arr[i] : 0
// where `bound` is either `n` itself (the check repeats the loop test) or the
// independent `len` (nothing in the function relates it to `n`).
struct SumLoop {
    Function* fn = nullptr;
    BasicBlock* loop_check = nullptr;
    Instruction* check = nullptr;
};

SumLoop build_sum_loop(Module& mod, bool check_against_n) {
    Builder b(mod);
    SumLoop out;
    Function* fn = mod.create_function("sum_array", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    out.fn = fn;
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

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_check, {}, exit_bb, {acc});

    fn->append_block(loop_check);
    b.position_at_end(loop_check);
    Value* in_bounds = b.build_ult(i, check_against_n ? n : len);
    out.check = in_bounds->defining_instruction();
    out.loop_check = loop_check;
    b.build_br_if(in_bounds, loop_fast, {}, loop_fallback, {});

    fn->append_block(loop_fast);
    b.position_at_end(loop_fast);
    Value* elem = b.build_load_indexed(Type::i64(), arr, i, 8, 0);
    Value* next_acc_fast = b.build_add(acc, elem);
    b.build_br(latch, {next_acc_fast});

    fn->append_block(loop_fallback);
    b.position_at_end(loop_fallback);
    b.build_br(latch, {acc});

    fn->append_block(latch);
    b.position_at_end(latch);
    Value* cur_acc = b.add_block_param(latch, Type::i64());
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, cur_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_sum = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();
    return out;
}

size_t count_opcode(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == op) ++n;
        }
    }
    return n;
}

} // namespace

TEST_CASE("BCE - Loop check implied by the loop's own exit test is folded") {
    Module mod("test_bce_implied");
    SumLoop loop = build_sum_loop(mod, /*check_against_n=*/true);

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;
    CHECK(run_bounds_check_elimination(*loop.fn, mod, opts));
    CHECK(stats.checks_implied > 0);

    // `i <u n` inside a body entered on `i <s n` from i = 0 always holds: the
    // fallback path is gone, and nothing was speculated to get there.
    Instruction* term = loop.loop_check->terminator();
    REQUIRE(term != nullptr);
    CHECK_EQ(term->opcode(), Opcode::br);
    CHECK_EQ(count_opcode(*loop.fn, Opcode::guard), size_t(0));
    CHECK(verify_function(*loop.fn));
}

TEST_CASE("BCE - Check against an unrelated bound is not folded or guarded") {
    Module mod("test_bce_unrelated");
    SumLoop loop = build_sum_loop(mod, /*check_against_n=*/false);

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;
    run_bounds_check_elimination(*loop.fn, mod, opts);

    // Nothing relates `len` to `n`, so the per-iteration check has to stay,
    // and no deopt guard may stand in for it (AOT code has nowhere to go).
    CHECK_EQ(stats.checks_implied, size_t(0));
    CHECK_EQ(count_opcode(*loop.fn, Opcode::guard), size_t(0));
    CHECK_EQ(count_opcode(*loop.fn, Opcode::ult), size_t(1));
    Instruction* term = loop.loop_check->terminator();
    REQUIRE(term != nullptr);
    CHECK_EQ(term->opcode(), Opcode::br_if);
    CHECK(verify_function(*loop.fn));
}

TEST_CASE("BCE - JIT Execution of Bounds Checked Loops") {
    std::vector<int64_t> data(100);
    std::iota(data.begin(), data.end(), int64_t{1});
    using SumFn = int64_t(*)(const int64_t*, int64_t, int64_t);

    for (bool against_n : {true, false}) {
        Module mod("test_bce_jit");
        SumLoop loop = build_sum_loop(mod, against_n);

        LoopOptOptions loop_opts;
        loop_opts.enable_bce = true;
        loop_opts.enable_unroll = false;
        loop_opts.enable_vectorize = false;
        optimize_function_loops(*loop.fn, loop_opts);
        CHECK(verify_function(*loop.fn));

        codegen::JitExecutionEngine jit;
        REQUIRE(jit.compile_and_load(mod));
        auto sum_ptr = jit.get_function_ptr<SumFn>("sum_array");
        REQUIRE(sum_ptr != nullptr);

        CHECK_EQ(sum_ptr(data.data(), 100, 100), int64_t{5050});
        CHECK_EQ(sum_ptr(data.data(), 100, 50), int64_t{1275});
        if (!against_n) {
            // n beyond len: the elements past len fall back to 0 every time.
            CHECK_EQ(sum_ptr(data.data(), 10, 100), int64_t{55});
            CHECK_EQ(sum_ptr(data.data(), 0, 100), int64_t{0});
        }
    }
}
