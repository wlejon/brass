#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/gvn_pre.hpp>
#include <brass/interpreter/interpreter.hpp>

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

size_t count_opcodes(const BasicBlock& bb, Opcode op) {
    size_t count = 0;
    for (const Instruction* inst : bb) {
        if (inst && inst->opcode() == op) count++;
    }
    return count;
}

} // namespace

TEST_CASE("GVN-PRE - Diamond CFG Partial Redundancy Elimination") {
    Module mod("test_diamond_pre");
    Builder b(mod);

    // func @diamond_pre(%cond: i32, %a: i64, %b: i64) -> i64
    Function* fn = mod.create_function("diamond_pre", Type::i64(),
        {Type::i32(), Type::i64(), Type::i64()});
    b.set_function(fn);

    /*
     * Graph:
     *         entry
     *        /     \
     *      left    right
     *     (a + b)  (no add)
     *        \     /
     *         merge
     *        (a + b) partially redundant!
     */
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* left = b.append_block("left");
    BasicBlock* right = b.append_block("right");
    BasicBlock* merge = b.append_block("merge");

    b.position_at_end(entry);
    Value* cond = b.add_block_param(entry, Type::i32());
    Value* a = b.add_block_param(entry, Type::i64());
    Value* b_val = b.add_block_param(entry, Type::i64());
    b.build_br_if(cond, left, right);

    b.position_at_end(left);
    Value* left_sum = b.build_add(a, b_val);
    Value* left_res = b.build_mul(left_sum, b.build_iconst_i64(2));
    b.build_br(merge, {left_res});

    b.position_at_end(right);
    Value* right_res = b.build_iconst_i64(1);
    b.build_br(merge, {right_res});

    b.position_at_end(merge);
    Value* branch_res = b.add_block_param(merge, Type::i64());
    Value* merge_sum = b.build_add(a, b_val); // Partially redundant computation!
    Value* final_res = b.build_add(branch_res, merge_sum);
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Initially 3 additions: left_sum, merge_sum, final_res
    CHECK_EQ(count_opcodes(*fn, Opcode::add), 3ULL);

    GvnPreStats stats;
    GvnPreOptions opts;
    opts.stats = &stats;
    bool changed = gvn_pre_function(*fn, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    // One expression should be hoisted to right, one eliminated from merge
    CHECK_EQ(stats.expressions_hoisted, 1ULL);
    CHECK_EQ(stats.expressions_eliminated, 1ULL);
    CHECK_EQ(stats.block_params_inserted, 1ULL);

    // Merge block should no longer contain merge_sum!
    // Remaining additions: left_sum, hoisted addition in right, and final_res
    CHECK_EQ(count_opcodes(*merge, Opcode::add), 1ULL); // only final_res

    // Verify interpreter correctness across both branches
    Interpreter interp;

    // Path 1 (cond=1, a=7, b=13):
    // left: left_sum = 20, left_res = 40
    // merge: merge_sum = 20
    // final_res = 40 + 20 = 60
    RuntimeValue r1 = interp.run(mod, "diamond_pre", {
        RuntimeValue::from_i32(1), RuntimeValue::from_i64(7), RuntimeValue::from_i64(13)
    });
    CHECK_EQ(r1.as_i64(), 60LL);

    // Path 2 (cond=0, a=7, b=13):
    // right: right_res = 1
    // merge: merge_sum = 20
    // final_res = 1 + 20 = 21
    RuntimeValue r2 = interp.run(mod, "diamond_pre", {
        RuntimeValue::from_i32(0), RuntimeValue::from_i64(7), RuntimeValue::from_i64(13)
    });
    CHECK_EQ(r2.as_i64(), 21LL);
}

TEST_CASE("GVN-PRE - Loop-Invariant Code Motion Generalization (LICM)") {
    Module mod("test_licm_pre");
    Builder b(mod);

    // func @loop_licm(%n: i64, %scale: i64, %bias: i64) -> i64
    Function* fn = mod.create_function("loop_licm", Type::i64(),
        {Type::i64(), Type::i64(), Type::i64()});
    b.set_function(fn);

    /*
     * Loop CFG:
     *   preheader (entry)
     *       |
     *     header (%i: i64, %acc: i64)
     *     /    \
     *   body   exit
     * (scale * 3 + bias) -> loop invariant!
     */
    BasicBlock* pre = b.append_block("pre");
    BasicBlock* header = b.append_block("header");
    BasicBlock* body = b.append_block("body");
    BasicBlock* exit = b.append_block("exit");

    b.position_at_end(pre);
    Value* n = b.add_block_param(pre, Type::i64());
    Value* scale = b.add_block_param(pre, Type::i64());
    Value* bias = b.add_block_param(pre, Type::i64());

    Value* i0 = b.build_iconst_i64(0);
    Value* acc0 = b.build_iconst_i64(0);
    b.build_br(header, {i0, acc0});

    b.position_at_end(header);
    Value* cur_i = b.add_block_param(header, Type::i64());
    Value* cur_acc = b.add_block_param(header, Type::i64());
    Value* cmp = b.build_slt(cur_i, n);
    b.build_br_if(cmp, body, exit);

    b.position_at_end(body);
    // Loop-invariant computation: scale * 3 + bias
    Value* term1 = b.build_mul(scale, b.build_iconst_i64(3));
    Value* inv = b.build_add(term1, bias);
    Value* next_acc = b.build_add(cur_acc, inv);
    Value* next_i = b.build_add(cur_i, b.build_iconst_i64(1));
    b.build_br(header, {next_i, next_acc});

    b.position_at_end(exit);
    b.build_ret(cur_acc);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(count_opcodes(*body, Opcode::mul), 1ULL);

    GvnPreStats stats;
    GvnPreOptions opts;
    opts.stats = &stats;
    bool changed = gvn_pre_function(*fn, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    // The invariant computations should be hoisted out of body into preheader
    CHECK(stats.expressions_hoisted >= 1ULL);
    CHECK(stats.expressions_eliminated >= 1ULL);

    // Body should now have 0 multiplications!
    CHECK_EQ(count_opcodes(*body, Opcode::mul), 0ULL);

    // Verify correctness with Interpreter
    Interpreter interp;
    // 5 iterations: acc += (4 * 3 + 2) = 14 * 5 = 70
    RuntimeValue res = interp.run(mod, "loop_licm", {
        RuntimeValue::from_i64(5), RuntimeValue::from_i64(4), RuntimeValue::from_i64(2)
    });
    CHECK_EQ(res.as_i64(), 70LL);
}

TEST_CASE("GVN-PRE - Memory Load Partial Redundancy Elimination (Load PRE)") {
    Module mod("test_load_pre");
    Builder b(mod);

    // func @load_pre(%buf: ptr, %cond: i32) -> i64
    Function* fn = mod.create_function("load_pre", Type::i64(),
        {Type::ptr(), Type::i32()});
    b.set_function(fn);

    /*
     * Diamond CFG with partial memory load:
     *       entry
     *      /     \
     *    pathA   pathB
     *  (load)   (no load, no clobber)
     *      \     /
     *       join
     *      (load) partially redundant!
     */
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* pathA = b.append_block("pathA");
    BasicBlock* pathB = b.append_block("pathB");
    BasicBlock* join = b.append_block("join");

    b.position_at_end(entry);
    Value* buf = b.add_block_param(entry, Type::ptr());
    Value* cond = b.add_block_param(entry, Type::i32());
    b.build_br_if(cond, pathA, pathB);

    b.position_at_end(pathA);
    Value* vA = b.build_load(Type::i64(), buf, 0);
    Value* resA = b.build_mul(vA, b.build_iconst_i64(10));
    b.build_br(join, {resA});

    b.position_at_end(pathB);
    Value* resB = b.build_iconst_i64(100);
    b.build_br(join, {resB});

    b.position_at_end(join);
    Value* join_arg = b.add_block_param(join, Type::i64());
    Value* v_join = b.build_load(Type::i64(), buf, 0); // Partially redundant load!
    Value* total = b.build_add(join_arg, v_join);
    b.build_ret(total);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    CHECK_EQ(count_opcodes(*fn, Opcode::load), 2ULL);

    GvnPreStats stats;
    GvnPreOptions opts;
    opts.stats = &stats;
    bool changed = gvn_pre_function(*fn, opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    CHECK_EQ(stats.expressions_hoisted, 1ULL);
    CHECK_EQ(stats.expressions_eliminated, 1ULL);

    // Join block must no longer contain any load instruction
    CHECK_EQ(count_opcodes(*join, Opcode::load), 0ULL);

    // Verify interpreter execution
    int64_t cell = 7;
    uintptr_t buf_addr = reinterpret_cast<uintptr_t>(&cell);
    Interpreter interp;

    // Path A (cond=1): vA=7, resA=70, join v=7 => total = 77
    RuntimeValue r1 = interp.run(mod, "load_pre", {
        RuntimeValue::from_ptr(buf_addr), RuntimeValue::from_i32(1)
    });
    CHECK_EQ(r1.as_i64(), 77LL);

    // Path B (cond=0): resB=100, join v=7 => total = 107
    RuntimeValue r2 = interp.run(mod, "load_pre", {
        RuntimeValue::from_ptr(buf_addr), RuntimeValue::from_i32(0)
    });
    CHECK_EQ(r2.as_i64(), 107LL);
}
