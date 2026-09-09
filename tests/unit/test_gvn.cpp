#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/memory_ssa.hpp>
#include <brass/mir/gvn.hpp>
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

TEST_CASE("GVN - Commutative Expression Canonicalization & Dominance Elimination") {
    Module mod("test_commutative_gvn");
    Builder b(mod);

    // func @commutative(%a: i64, %b: i64) -> i64
    Function* fn = mod.create_function("commutative", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");

    // Block 0: sum1 = a + b; mul1 = a * b
    b.position_at_end(b0);
    Value* a = b.add_block_param(b0, Type::i64());
    Value* b_val = b.add_block_param(b0, Type::i64());

    Value* sum1 = b.build_add(a, b_val);
    Value* mul1 = b.build_mul(a, b_val);
    (void)sum1;
    (void)mul1;
    b.build_br(b1);

    // Block 1 (dominated by b0): sum2 = b + a; mul2 = b * a
    b.position_at_end(b1);
    Value* sum2 = b.build_add(b_val, a); // reversed operands!
    Value* mul2 = b.build_mul(b_val, a); // reversed operands!
    Value* res = b.build_add(sum2, mul2);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(count_opcodes(*fn, Opcode::add), 3ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::mul), 2ULL);

    GvnStats stats;
    GvnOptions opts;
    opts.stats = &stats;
    bool changed = gvn_function(*fn, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    // sum2 and mul2 should be eliminated as redundant expressions
    CHECK_EQ(stats.expressions_eliminated, 2ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::add), 2ULL); // sum1 and res
    CHECK_EQ(count_opcodes(*fn, Opcode::mul), 1ULL); // mul1

    // Verify evaluation correctness with interpreter
    Interpreter interp;
    RuntimeValue iv = interp.run(mod, "commutative", {RuntimeValue::from_i64(7), RuntimeValue::from_i64(13)});
    CHECK_EQ(iv.as_i64(), (7 + 13) + (7 * 13));
}

TEST_CASE("GVN - Global CSE Across Diamond Branching and Multi-level Dominance") {
    Module mod("test_diamond_gvn");
    Builder b(mod);

    // Diamond CFG:
    //      entry: (x * y)
    //     /      |
    //   left   right
    //     \      /
    //      merge: (x * y) redundant!
    Function* fn = mod.create_function("diamond_cse", Type::i64(), {Type::i64(), Type::i64(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* left = b.append_block("left");
    BasicBlock* right = b.append_block("right");
    BasicBlock* merge = b.append_block("merge");

    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());
    Value* cond = b.add_block_param(entry, Type::i32());

    Value* xy_entry = b.build_mul(x, y);
    b.build_br_if(cond, left, right);

    b.position_at_end(left);
    Value* left_val = b.build_add(xy_entry, b.build_iconst_i64(1));
    (void)left_val;
    b.build_br(merge);

    b.position_at_end(right);
    Value* right_val = b.build_add(xy_entry, b.build_iconst_i64(2));
    (void)right_val;
    b.build_br(merge);

    b.position_at_end(merge);
    // In merge block, compute (x * y) again. Since entry dominates merge, this is redundant!
    Value* xy_merge = b.build_mul(x, y);
    Value* final_res = b.build_add(xy_merge, b.build_iconst_i64(10));
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(count_opcodes(*fn, Opcode::mul), 2ULL);

    GvnStats stats;
    GvnOptions opts;
    opts.stats = &stats;
    bool changed = gvn_function(*fn, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    CHECK_EQ(stats.expressions_eliminated, 1ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::mul), 1ULL);

    Interpreter interp;
    RuntimeValue r1 = interp.run(mod, "diamond_cse", {RuntimeValue::from_i64(5), RuntimeValue::from_i64(6), RuntimeValue::from_i32(1)});
    CHECK_EQ(r1.as_i64(), 30 + 10);
}

TEST_CASE("GVN - Store-to-Load Forwarding Across Basic Block Boundaries") {
    Module mod("test_store_to_load");
    Builder b(mod);

    // func @store_load_fwd(%x: i64) -> i64
    Function* fn = mod.create_function("store_load_fwd", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");

    b.position_at_end(b0);
    Value* x = b.add_block_param(b0, Type::i64());

    // Allocate an object
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});

    // Store value into object field 0
    b.build_store(Type::i64(), obj, 0, x);
    b.build_br(b1);

    b.position_at_end(b1);
    // Load from field 0 without any intervening clobber
    Value* loaded = b.build_load(Type::i64(), obj, 0);
    Value* ret_val = b.build_add(loaded, b.build_iconst_i64(100));
    b.build_ret(ret_val);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(count_opcodes(*fn, Opcode::load), 1ULL);

    GvnStats stats;
    GvnOptions opts;
    opts.stats = &stats;
    bool changed = gvn_function(*fn, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    // Load should be forwarded and eliminated
    CHECK_EQ(stats.loads_forwarded, 1ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::load), 0ULL);

    Interpreter interp;
    RuntimeValue res = interp.run(mod, "store_load_fwd", {RuntimeValue::from_i64(42)});
    CHECK_EQ(res.as_i64(), 142);
}

TEST_CASE("GVN - Load-to-Load Elimination Across Basic Block Boundaries with Intervening Non-Aliasing Store") {
    Module mod("test_load_to_load");
    Builder b(mod);

    // func @load_load_elim(%ptr: ptr, %other_ptr: ptr) -> i64
    Function* fn = mod.create_function("load_load_elim", Type::i64(), {Type::ptr(), Type::ptr()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");

    b.position_at_end(b0);
    Value* ptr = b.add_block_param(b0, Type::ptr());
    Value* other_ptr = b.add_block_param(b0, Type::ptr());
    (void)other_ptr;

    // First load from ptr + 0
    Value* l1 = b.build_load(Type::i64(), ptr, 0);

    // Store to distinct offset on same pointer (ptr + 8), provably distinct offset
    b.build_store(Type::i64(), ptr, 8, b.build_iconst_i64(999));
    b.build_br(b1);

    b.position_at_end(b1);
    // Second load from ptr + 0: should be eliminated because store at offset 8 does not clobber offset 0!
    Value* l2 = b.build_load(Type::i64(), ptr, 0);
    Value* sum = b.build_add(l1, l2);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(count_opcodes(*fn, Opcode::load), 2ULL);

    GvnStats stats;
    GvnOptions opts;
    opts.stats = &stats;
    bool changed = gvn_function(*fn, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    // One of the loads must be eliminated
    CHECK_EQ(stats.loads_eliminated, 1ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::load), 1ULL);
}

TEST_CASE("GVN - Clobber Safety: Aliasing Store & Non-Pure Call Prevent Load Elimination") {
    Module mod("test_clobber_safety");
    Builder b(mod);

    // func @clobber_test(%p: ptr, %val: i64) -> i64
    Function* fn = mod.create_function("clobber_test", Type::i64(), {Type::ptr(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* p = b.add_block_param(entry, Type::ptr());
    Value* val = b.add_block_param(entry, Type::i64());

    // 1. Initial load
    Value* v1 = b.build_load(Type::i64(), p, 0);

    // 2. An unhandled external call that might modify p
    b.build_call("external_mutate_side_effect", Type::void_type(), {p, val});

    // 3. Second load from p: MUST NOT be eliminated using v1 because external call could clobber p!
    Value* v2 = b.build_load(Type::i64(), p, 0);

    Value* sum = b.build_add(v1, v2);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(count_opcodes(*fn, Opcode::load), 2ULL);

    GvnStats stats;
    GvnOptions opts;
    opts.stats = &stats;
    bool changed = gvn_function(*fn, opts);
    (void)changed;

    // Neither load can be eliminated because of the clobbering call
    CHECK_EQ(stats.loads_eliminated, 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::load), 2ULL);
}

TEST_CASE("GVN - Dead Store Elimination Within a Block and Across Simple CFGs") {
    Module mod("test_dse");
    Builder b(mod);

    // Subcase 1: Intra-block DSE
    Function* fn1 = mod.create_function("intra_block_dse", Type::i64(), {Type::ptr(), Type::i64(), Type::i64()});
    b.set_function(fn1);
    BasicBlock* eb = b.append_block("entry");
    b.position_at_end(eb);
    Value* p1 = b.add_block_param(eb, Type::ptr());
    Value* v1 = b.add_block_param(eb, Type::i64());
    Value* v2 = b.add_block_param(eb, Type::i64());

    // Store v1 to p1 + 0 (DEAD, overwritten by v2 before any read)
    b.build_store(Type::i64(), p1, 0, v1);
    b.build_add(v1, v2); // unrelated pure computation
    // Store v2 to p1 + 0
    b.build_store(Type::i64(), p1, 0, v2);
    b.build_ret(v2);

    fn1->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn1));
    CHECK_EQ(count_opcodes(*fn1, Opcode::store), 2ULL);

    GvnStats stats1;
    GvnOptions opts;
    opts.stats = &stats1;
    bool changed1 = gvn_function(*fn1, opts);
    CHECK(changed1);
    REQUIRE(verify_function(*fn1));

    CHECK_EQ(stats1.dead_stores_eliminated, 1ULL);
    CHECK_EQ(count_opcodes(*fn1, Opcode::store), 1ULL);

    // Subcase 2: Inter-block simple CFG DSE
    Function* fn2 = mod.create_function("inter_block_dse", Type::i64(), {Type::ptr(), Type::i64(), Type::i64()});
    b.set_function(fn2);
    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");

    b.position_at_end(b0);
    Value* p2 = b.add_block_param(b0, Type::ptr());
    Value* val_a = b.add_block_param(b0, Type::i64());
    Value* val_b = b.add_block_param(b0, Type::i64());

    // Store val_a into p2 + 0 in block 0
    b.build_store(Type::i64(), p2, 0, val_a);
    b.build_br(b1);

    b.position_at_end(b1);
    // Block 1 unconditionally overwrites p2 + 0 before any read
    b.build_store(Type::i64(), p2, 0, val_b);
    b.build_ret(val_b);

    fn2->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn2));
    CHECK_EQ(count_opcodes(*fn2, Opcode::store), 2ULL);

    GvnStats stats2;
    opts.stats = &stats2;
    bool changed2 = gvn_function(*fn2, opts);
    CHECK(changed2);
    REQUIRE(verify_function(*fn2));

    CHECK_EQ(stats2.dead_stores_eliminated, 1ULL);
    CHECK_EQ(count_opcodes(*fn2, Opcode::store), 1ULL);
}
