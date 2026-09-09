#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <vector>

using namespace brass;

namespace {

size_t count_opcodes(const Function& fn, Opcode op) {
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == op) count++;
        }
    }
    return count;
}

} // namespace

TEST_CASE("SCCP - Constant Propagation Through Arithmetic Chains Across Basic Blocks") {
    Module mod("test_sccp_arithmetic");
    Builder b(mod);

    // func @chain() -> i64
    // b0:
    //   %c10 = 10
    //   %c20 = 20
    //   %sum = add 10, 20  (30)
    //   br b1(%sum)
    // b1(%p: i64):
    //   %mul = mul %p, 2   (60)
    //   br b2(%mul)
    // b2(%q: i64):
    //   %res = sub %q, 5   (55)
    //   ret %res
    Function* fn = mod.create_function("chain", Type::i64(), {});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");

    b.position_at_end(b0);
    Value* c10 = b.build_iconst_i64(10);
    Value* c20 = b.build_iconst_i64(20);
    Value* sum = b.build_add(c10, c20);
    b.build_br(b1, {sum});

    b.position_at_end(b1);
    Value* p = b.add_block_param(b1, Type::i64());
    Value* mul = b.build_mul(p, b.build_iconst_i64(2));
    b.build_br(b2, {mul});

    b.position_at_end(b2);
    Value* q = b.add_block_param(b2, Type::i64());
    Value* res = b.build_sub(q, b.build_iconst_i64(5));
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    SccpStats stats;
    SccpOptions opts;
    opts.stats = &stats;
    bool sccp_changed = sccp_function(*fn, opts);
    CHECK(sccp_changed);
    CHECK(stats.constants_propagated > 0);
    REQUIRE(verify_function(*fn));

    bool cfg_changed = cfg_simplify_function(*fn);
    CHECK(cfg_changed);
    REQUIRE(verify_function(*fn));

    // Linear block merge should collapse the 3 linear blocks into 1
    CHECK_EQ(fn->block_count(), 1ULL);

    Interpreter interp;
    RuntimeValue iv = interp.run(mod, "chain", {});
    CHECK_EQ(iv.as_i64(), 55);
}

TEST_CASE("SCCP - Branch Folding: br_if 1 Folded to br, False Block Pruned") {
    Module mod("test_sccp_branch_fold");
    Builder b(mod);

    // func @fold_branch() -> i32
    // entry:
    //   %cond = iconst_i32 1
    //   br_if %cond, b_true, b_false
    // b_true:
    //   ret 100
    // b_false:
    //   ret 200
    Function* fn = mod.create_function("fold_branch", Type::i32(), {});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* b_true = b.append_block("b_true");
    BasicBlock* b_false = b.append_block("b_false");

    b.position_at_end(entry);
    Value* cond = b.build_iconst_i32(1);
    b.build_br_if(cond, b_true, b_false);

    b.position_at_end(b_true);
    b.build_ret(b.build_iconst_i32(100));

    b.position_at_end(b_false);
    b.build_ret(b.build_iconst_i32(200));

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    SccpStats stats;
    SccpOptions opts;
    opts.stats = &stats;
    bool sccp_changed = sccp_function(*fn, opts);
    CHECK(sccp_changed);
    CHECK_EQ(stats.branches_folded, 1ULL);
    REQUIRE(verify_function(*fn));

    // After SCCP, br_if is replaced by br b_true. b_false is dead.
    CHECK_EQ(count_opcodes(*fn, Opcode::br_if), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::br), 1ULL);

    // CFG simplification removes unreachable b_false and merges entry with b_true
    bool cfg_changed = cfg_simplify_function(*fn);
    CHECK(cfg_changed);
    REQUIRE(verify_function(*fn));

    CHECK_EQ(fn->block_count(), 1ULL);

    Interpreter interp;
    RuntimeValue iv = interp.run(mod, "fold_branch", {});
    CHECK_EQ(iv.as_i32(), 100);
}

TEST_CASE("SCCP - Block Parameter Phi Constant Resolution with One Unreachable Edge") {
    Module mod("test_sccp_phi_unreachable");
    Builder b(mod);

    // Wegman-Zadeck Classic:
    // entry:
    //   %cond = iconst_i32 1
    //   br_if %cond, b_taken, b_dead
    // b_taken:
    //   br merge_bb(42)
    // b_dead:
    //   br merge_bb(99)
    // merge_bb(%phi: i32):
    //   ret %phi
    Function* fn = mod.create_function("phi_resolution", Type::i32(), {});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* b_taken = b.append_block("b_taken");
    BasicBlock* b_dead = b.append_block("b_dead");
    BasicBlock* merge_bb = b.append_block("merge_bb");

    b.position_at_end(entry);
    Value* cond = b.build_iconst_i32(1);
    b.build_br_if(cond, b_taken, b_dead);

    b.position_at_end(b_taken);
    Value* c42 = b.build_iconst_i32(42);
    b.build_br(merge_bb, {c42});

    b.position_at_end(b_dead);
    Value* c99 = b.build_iconst_i32(99);
    b.build_br(merge_bb, {c99});

    b.position_at_end(merge_bb);
    Value* phi = b.add_block_param(merge_bb, Type::i32());
    b.build_ret(phi);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    SccpStats stats;
    SccpOptions opts;
    opts.stats = &stats;
    bool sccp_changed = sccp_function(*fn, opts);
    CHECK(sccp_changed);
    CHECK(stats.constants_propagated >= 1);
    REQUIRE(verify_function(*fn));

    bool cfg_changed = cfg_simplify_function(*fn);
    CHECK(cfg_changed);
    REQUIRE(verify_function(*fn));

    // The whole diamond collapses into 1 block returning 42
    CHECK_EQ(fn->block_count(), 1ULL);

    Interpreter interp;
    RuntimeValue iv = interp.run(mod, "phi_resolution", {});
    CHECK_EQ(iv.as_i32(), 42);
}

TEST_CASE("SCCP - Speculation Guard Elimination: Verified guard 1 is Removed") {
    Module mod("test_sccp_guard_elim");
    Builder b(mod);

    // func @guard_pass(%x: i64) -> i64
    //   %cond = iconst_i32 1
    //   guard %cond, "stub_exit", [%x]
    //   %res = add %x, 10
    //   ret %res
    Function* fn = mod.create_function("guard_pass", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Value* cond = b.build_iconst_i32(1);
    b.build_guard(cond, "stub_exit", {x});
    Value* res = b.build_add(x, b.build_iconst_i64(10));
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(count_opcodes(*fn, Opcode::guard), 1ULL);

    SccpStats stats;
    SccpOptions opts;
    opts.enable_guard_elim = true;
    opts.stats = &stats;
    bool changed = sccp_function(*fn, opts);
    CHECK(changed);
    CHECK_EQ(stats.guards_eliminated, 1ULL);
    REQUIRE(verify_function(*fn));

    // The guard instruction MUST be completely removed!
    CHECK_EQ(count_opcodes(*fn, Opcode::guard), 0ULL);

    Interpreter interp;
    RuntimeValue iv = interp.run(mod, "guard_pass", {RuntimeValue::from_i64(32)});
    CHECK_EQ(iv.as_i64(), 42);
}

TEST_CASE("SCCP - Speculation Guard Provably Always Failing (guard 0)") {
    Module mod("test_sccp_guard_fail");
    Builder b(mod);

    // func @guard_always_fail(%x: i64) -> i64
    //   %cond = iconst_i32 0
    //   guard %cond, "stub_exit", [%x]
    //   %res = add %x, 10
    //   ret %res
    Function* fn = mod.create_function("guard_always_fail", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Value* cond = b.build_iconst_i32(0);
    b.build_guard(cond, "stub_exit", {x});
    Value* res = b.build_add(x, b.build_iconst_i64(10));
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    SccpStats stats;
    SccpOptions opts;
    opts.enable_guard_elim = true;
    opts.stats = &stats;
    bool changed = sccp_function(*fn, opts);
    CHECK(changed);
    CHECK_EQ(stats.guards_always_failing, 1ULL);
    REQUIRE(verify_function(*fn));

    // Guard is retained as deopt trigger, and trailing instructions are replaced with unreachable
    CHECK_EQ(count_opcodes(*fn, Opcode::guard), 1ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::add), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::unreachable), 1ULL);
}

TEST_CASE("SCCP - Unreachable Block Removal and Linear Block Merging") {
    Module mod("test_sccp_dead_block_merge");
    Builder b(mod);

    // func @dead_blocks() -> i64
    // entry:
    //   br b1
    // b1:
    //   %v1 = iconst_i64 7
    //   br b2(%v1)
    // b2(%p: i64):
    //   %v2 = add %p, 3
    //   br b3(%v2)
    // b3(%q: i64):
    //   ret %q
    // dead_bb:
    //   ret 999
    Function* fn = mod.create_function("dead_blocks", Type::i64(), {});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");
    BasicBlock* dead_bb = b.append_block("dead_bb");

    b.position_at_end(entry);
    b.build_br(b1);

    b.position_at_end(b1);
    Value* v1 = b.build_iconst_i64(7);
    b.build_br(b2, {v1});

    b.position_at_end(b2);
    Value* p = b.add_block_param(b2, Type::i64());
    Value* v2 = b.build_add(p, b.build_iconst_i64(3));
    b.build_br(b3, {v2});

    b.position_at_end(b3);
    Value* q = b.add_block_param(b3, Type::i64());
    b.build_ret(q);

    b.position_at_end(dead_bb);
    b.build_ret(b.build_iconst_i64(999));

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(fn->block_count(), 5ULL);

    CfgSimplifyStats stats;
    CfgSimplifyOptions opts;
    opts.stats = &stats;
    bool changed = cfg_simplify_function(*fn, opts);
    CHECK(changed);
    CHECK_EQ(stats.blocks_removed, 1ULL); // dead_bb removed
    CHECK(stats.blocks_merged >= 3ULL);
    REQUIRE(verify_function(*fn));

    CHECK_EQ(fn->block_count(), 1ULL);

    Interpreter interp;
    RuntimeValue iv = interp.run(mod, "dead_blocks", {});
    CHECK_EQ(iv.as_i64(), 10);
}

TEST_CASE("SCCP - Negative Test: Dynamic Condition Transitions to Bottom, Preserves Both Paths") {
    Module mod("test_sccp_dynamic_cond");
    Builder b(mod);

    // func @dynamic_cond(%cond: i32) -> i32
    // entry:
    //   br_if %cond, b_true, b_false
    // b_true:
    //   br merge(42)
    // b_false:
    //   br merge(99)
    // merge(%phi: i32):
    //   ret %phi
    Function* fn = mod.create_function("dynamic_cond", Type::i32(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* b_true = b.append_block("b_true");
    BasicBlock* b_false = b.append_block("b_false");
    BasicBlock* merge = b.append_block("merge");

    b.position_at_end(entry);
    Value* cond = b.add_block_param(entry, Type::i32());
    b.build_br_if(cond, b_true, b_false);

    b.position_at_end(b_true);
    b.build_br(merge, {b.build_iconst_i32(42)});

    b.position_at_end(b_false);
    b.build_br(merge, {b.build_iconst_i32(99)});

    b.position_at_end(merge);
    Value* phi = b.add_block_param(merge, Type::i32());
    b.build_ret(phi);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    SccpStats stats;
    SccpOptions opts;
    opts.stats = &stats;
    sccp_function(*fn, opts);
    REQUIRE(verify_function(*fn));

    // Condition is dynamic: neither branch should be folded!
    CHECK_EQ(stats.branches_folded, 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::br_if), 1ULL);

    // Both paths must be preserved
    Interpreter interp;
    RuntimeValue iv_true = interp.run(mod, "dynamic_cond", {RuntimeValue::from_i32(1)});
    CHECK_EQ(iv_true.as_i32(), 42);

    RuntimeValue iv_false = interp.run(mod, "dynamic_cond", {RuntimeValue::from_i32(0)});
    CHECK_EQ(iv_false.as_i32(), 99);
}

TEST_CASE("CFG Simplify - Empty Trampoline Block Elimination") {
    Module mod("test_trampoline");
    Builder b(mod);

    // entry:
    //   br_if %cond, tramp, b_other
    // tramp:
    //   br real_dest
    // b_other:
    //   ret 20
    // real_dest:
    //   ret 10
    Function* fn = mod.create_function("trampoline_fn", Type::i32(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* tramp = b.append_block("tramp");
    BasicBlock* b_other = b.append_block("b_other");
    BasicBlock* real_dest = b.append_block("real_dest");

    b.position_at_end(entry);
    Value* cond = b.add_block_param(entry, Type::i32());
    b.build_br_if(cond, tramp, b_other);

    b.position_at_end(tramp);
    b.build_br(real_dest);

    b.position_at_end(b_other);
    b.build_ret(b.build_iconst_i32(20));

    b.position_at_end(real_dest);
    b.build_ret(b.build_iconst_i32(10));

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(fn->block_count(), 4ULL);

    CfgSimplifyStats stats;
    CfgSimplifyOptions opts;
    opts.stats = &stats;
    bool changed = cfg_simplify_function(*fn, opts);
    CHECK(changed);
    CHECK_EQ(stats.trampolines_eliminated, 1ULL);
    REQUIRE(verify_function(*fn));

    // tramp block is removed, entry branches directly to real_dest
    CHECK_EQ(entry->terminator()->true_target().block, real_dest);

    Interpreter interp;
    RuntimeValue iv1 = interp.run(mod, "trampoline_fn", {RuntimeValue::from_i32(1)});
    CHECK_EQ(iv1.as_i32(), 10);
    RuntimeValue iv0 = interp.run(mod, "trampoline_fn", {RuntimeValue::from_i32(0)});
    CHECK_EQ(iv0.as_i32(), 20);
}

TEST_CASE("SCCP - Constant Select Optimization with Dynamic Condition") {
    Module mod("test_sccp_select");
    Builder b(mod);

    // func @select_const(%cond: i32) -> i64
    // entry:
    //   %t = iconst_i64 100
    //   %f = iconst_i64 100
    //   %res = select %cond, %t, %f  (Both arms are 100! Result is 100 even if cond is unknown!)
    //   ret %res
    Function* fn = mod.create_function("select_const", Type::i64(), {Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* cond = b.add_block_param(entry, Type::i32());
    Value* t = b.build_iconst_i64(100);
    Value* f = b.build_iconst_i64(100);
    Value* res = b.build_select(cond, t, f);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    SccpStats stats;
    SccpOptions opts;
    opts.stats = &stats;
    bool changed = sccp_function(*fn, opts);
    CHECK(changed);
    CHECK_EQ(stats.constants_propagated, 1ULL);
    REQUIRE(verify_function(*fn));

    // The select instruction should be eliminated and folded into constant 100
    CHECK_EQ(count_opcodes(*fn, Opcode::select), 0ULL);

    Interpreter interp;
    RuntimeValue iv1 = interp.run(mod, "select_const", {RuntimeValue::from_i32(1)});
    CHECK_EQ(iv1.as_i64(), 100);
    RuntimeValue iv0 = interp.run(mod, "select_const", {RuntimeValue::from_i32(0)});
    CHECK_EQ(iv0.as_i64(), 100);
}
