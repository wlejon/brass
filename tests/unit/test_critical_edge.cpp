#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/critical_edge.hpp>
#include <brass/mir/parser.hpp>
#include <brass/interpreter/interpreter.hpp>

using namespace brass;

TEST_CASE("Critical Edge - Detection on Branching Graphs") {
    Module mod("test_crit_edge_detect");
    Builder b(mod);

    // func @test_detect(%cond1: i32, %cond2: i32) -> i32
    Function* fn = mod.create_function("test_detect", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn);

    // b0 -> b1, b2
    // b1 -> b2, b3
    // b2 -> b3
    // b3 -> ret
    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");

    b.position_at_end(b0);
    Value* c1 = b.add_block_param(b0, Type::i32());
    Value* c2 = b.add_block_param(b0, Type::i32());
    b.build_br_if(c1, b1, b2);

    b.position_at_end(b1);
    b.build_br_if(c2, b2, b3);

    b.position_at_end(b2);
    b.build_br(b3);

    b.position_at_end(b3);
    b.build_ret(b.build_iconst_i32(42));

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // In this graph:
    // b0 has successors: b1, b2 (2 succs)
    // b1 has successors: b2, b3 (2 succs)
    // b2 has predecessors: b0, b1 (2 preds)
    // b3 has predecessors: b1, b2 (2 preds)

    // Edge (b1 -> b2): b1 has 2 successors, b2 has 2 predecessors => CRITICAL!
    CHECK(is_critical_edge(b1, b2));

    // Edge (b0 -> b1): b0 has 2 successors, but b1 only has 1 predecessor (b0) => NOT critical
    CHECK_FALSE(is_critical_edge(b0, b1));

    // Edge (b2 -> b3): b2 has only 1 successor (b3), so NOT critical even though b3 has 2 preds
    CHECK_FALSE(is_critical_edge(b2, b3));
}

TEST_CASE("Critical Edge - Splitting Preserves Dominance and Verification") {
    Module mod("test_crit_edge_split");
    Builder b(mod);

    Function* fn = mod.create_function("test_split", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");

    b.position_at_end(b0);
    Value* c1 = b.add_block_param(b0, Type::i32());
    Value* c2 = b.add_block_param(b0, Type::i32());
    b.build_br_if(c1, b1, b2);

    b.position_at_end(b1);
    b.build_br_if(c2, b2, b3);

    b.position_at_end(b2);
    b.build_br(b3);

    b.position_at_end(b3);
    b.build_ret(b.build_iconst_i32(100));

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CriticalEdgeStats stats;
    BasicBlock* split_bb = split_critical_edge(*fn, b1, b2, &stats);
    REQUIRE(split_bb != nullptr);
    CHECK_EQ(stats.critical_edges_split, 1ULL);

    // After split, b1 -> split_bb -> b2
    CHECK_FALSE(is_critical_edge(b1, b2));
    CHECK_FALSE(is_critical_edge(b1, split_bb));
    CHECK_FALSE(is_critical_edge(split_bb, b2));

    // Function must verify cleanly
    REQUIRE(verify_function(*fn));

    // Dominator tree checks
    DominatorTree dom(*fn);
    CHECK(dom.is_reachable(split_bb));
    CHECK(dom.dominates(b0, split_bb));
    CHECK(dom.dominates(b1, split_bb));
    CHECK_FALSE(dom.dominates(split_bb, b2));
}

TEST_CASE("Critical Edge - Splitting Preserves Block Parameter Passing and Runtime Values") {
    Module mod("test_crit_edge_params");
    Builder b(mod);

    // func @calc(%c1: i32, %c2: i32) -> i64
    Function* fn = mod.create_function("calc", Type::i64(), {Type::i32(), Type::i32()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");

    b.position_at_end(b0);
    Value* c1 = b.add_block_param(b0, Type::i32());
    Value* c2 = b.add_block_param(b0, Type::i32());
    Value* v0 = b.build_iconst_i64(10);
    b.build_br_if(c1, b1, {}, b2, {v0});

    b.position_at_end(b1);
    Value* v1 = b.build_iconst_i64(25);
    b.build_br_if(c2, b2, {v1}, b3, {});

    // b2 expects one block param (%p: i64)
    b.position_at_end(b2);
    Value* p = b.add_block_param(b2, Type::i64());
    Value* v2 = b.build_add(p, b.build_iconst_i64(5));
    b.build_ret(v2);

    b.position_at_end(b3);
    b.build_ret(b.build_iconst_i64(0));

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Edge (b1 -> b2) is critical and passes parameter {v1} to b2
    CHECK(is_critical_edge(b1, b2));

    CriticalEdgeStats stats;
    bool any_split = split_critical_edges(*fn, &stats);
    CHECK(any_split);
    CHECK_EQ(stats.critical_edges_split, 2ULL);
    REQUIRE(verify_function(*fn));

    // Verify evaluation correctness via Interpreter
    Interpreter interp;

    // Path 1: c1=1, c2=1 -> b0 -> b1 -> split_bb(25) -> b2(25) -> returns 25+5 = 30
    RuntimeValue r1 = interp.run(mod, "calc", {RuntimeValue::from_i32(1), RuntimeValue::from_i32(1)});
    CHECK_EQ(r1.as_i64(), 30LL);

    // Path 2: c1=0, c2=ignored -> b0 -> b2(10) -> returns 10+5 = 15
    RuntimeValue r2 = interp.run(mod, "calc", {RuntimeValue::from_i32(0), RuntimeValue::from_i32(0)});
    CHECK_EQ(r2.as_i64(), 15LL);

    // Path 3: c1=1, c2=0 -> b0 -> b1 -> b3 -> returns 0
    RuntimeValue r3 = interp.run(mod, "calc", {RuntimeValue::from_i32(1), RuntimeValue::from_i32(0)});
    CHECK_EQ(r3.as_i64(), 0LL);
}

TEST_CASE("Critical Edge - edges into a landing pad two invokes share stay whole, and the split ends") {
    // Both unwind edges are critical by shape (each invoke has two
    // successors, the pad two predecessors), and neither may be split: the
    // pass must finish with the pad still their shared target.
    const char* src = R"(
func @work(%0: i64) -> i64 {
bb0:
  ret %0
}

func @two(%0: i64) -> i64 {
bb0:
  %1 = invoke.i64 @work(%0), bb1, bb3

bb1:
  %2 = invoke.i64 @work(%1), bb2, bb3

bb2:
  ret %2

bb3:
  %3 = landing_pad
  ret %3
}
)";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    Function* fn = mod->get_function("two");
    REQUIRE(fn != nullptr);
    const size_t blocks_before = fn->blocks().size();
    CHECK_FALSE(split_critical_edges(*fn));
    CHECK_EQ(fn->blocks().size(), blocks_before);
    CHECK(verify_function(*fn));
}
