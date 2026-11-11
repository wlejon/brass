// The code sinking pass (code_sinking.hpp): what moves down an edge, what
// stays, and that a chain follows its end.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/code_sinking.hpp>
#include <brass/mir/verifier.hpp>

using namespace brass;
using namespace brass::test;

namespace {

size_t count_in(const BasicBlock* bb, Opcode op) {
    size_t n = 0;
    for (const Instruction* inst : *bb) n += inst->opcode() == op;
    return n;
}

} // namespace

TEST_CASE("Code sinking - a chain only one edge needs moves to that edge") {
    // entry: s = a + b; m = s * a; k = a - b; if (c) return k; else return m + k
    Module mod("sink_chain");
    Builder b(mod);
    Function* fn = mod.create_function("sink_chain", Type::i64(), {Type::i64(), Type::i64(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* fast = b.append_block("fast");
    BasicBlock* slow = b.append_block("slow");
    b.position_at_end(entry);
    Value* a = b.add_block_param(entry, Type::i64());
    Value* bv = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i32());
    Value* s = b.build_add(a, bv);
    Value* m = b.build_mul(s, a);
    Value* k = b.build_sub(a, bv);
    b.build_br_if(c, fast, slow);
    b.position_at_end(fast);
    b.build_ret(k);
    b.position_at_end(slow);
    b.build_ret(b.build_add(m, k));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    CHECK_EQ(sink_code(*fn), 2ULL);
    REQUIRE(verify_function(*fn));
    // The add and the multiply left; the subtraction both edges read stayed.
    CHECK_EQ(count_in(entry, Opcode::add), 0ULL);
    CHECK_EQ(count_in(entry, Opcode::mul), 0ULL);
    CHECK_EQ(count_in(entry, Opcode::sub), 1ULL);
    CHECK_EQ(count_in(slow, Opcode::mul), 1ULL);
    CHECK_EQ(count_in(slow, Opcode::add), 2ULL);
    CHECK_EQ(slow->head()->opcode(), Opcode::add);
}

TEST_CASE("Code sinking - an edge argument's value moves to the split edge, a load stays") {
    // entry: s = a + 1; l = load p; if (c) goto mid else goto join(s); mid: goto join(a); join(x): return x + l
    Module mod("sink_edge");
    Builder b(mod);
    Function* fn = mod.create_function("sink_edge", Type::i64(), {Type::i64(), Type::ptr(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* mid = b.append_block("mid");
    BasicBlock* join = b.append_block("join");
    b.position_at_end(entry);
    Value* a = b.add_block_param(entry, Type::i64());
    Value* p = b.add_block_param(entry, Type::ptr());
    Value* c = b.add_block_param(entry, Type::i32());
    Value* s = b.build_add(a, b.build_iconst_i64(1));
    Value* l = b.build_load(Type::i64(), p, 0);
    Value* x = b.add_block_param(join, Type::i64());
    b.build_br_if(c, mid, {}, join, {s});
    b.position_at_end(mid);
    b.build_br(join, {a});
    b.position_at_end(join);
    b.build_ret(b.build_add(x, l));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    CHECK_EQ(sink_code(*fn), 1ULL);
    REQUIRE(verify_function(*fn));
    CHECK_EQ(count_in(entry, Opcode::add), 0ULL);
    CHECK_EQ(count_in(entry, Opcode::load), 1ULL);
}

TEST_CASE("Code sinking - uses down two edges each get a copy, and a copy sinks again") {
    // entry: s = a + b; if (c) goto left else goto right
    // left: if (d) return a; else return s        right: return s * 2
    Module mod("sink_copies");
    Builder b(mod);
    Function* fn = mod.create_function("sink_copies", Type::i64(), {Type::i64(), Type::i64(), Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* left = b.append_block("left");
    BasicBlock* left_hot = b.append_block("left_hot");
    BasicBlock* left_exit = b.append_block("left_exit");
    BasicBlock* right = b.append_block("right");
    b.position_at_end(entry);
    Value* a = b.add_block_param(entry, Type::i64());
    Value* bv = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i32());
    Value* d = b.add_block_param(entry, Type::i32());
    Value* s = b.build_add(a, bv);
    b.build_br_if(c, left, right);
    b.position_at_end(left);
    b.build_br_if(d, left_hot, left_exit);
    b.position_at_end(left_hot);
    b.build_ret(a);
    b.position_at_end(left_exit);
    b.build_ret(s);
    b.position_at_end(right);
    b.build_ret(b.build_mul(s, b.build_iconst_i64(2)));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    CHECK(sink_code(*fn) >= 2ULL);
    REQUIRE(verify_function(*fn));
    CHECK_EQ(count_in(entry, Opcode::add), 0ULL);
    CHECK_EQ(count_in(left, Opcode::add), 0ULL);
    CHECK_EQ(count_in(left_hot, Opcode::add), 0ULL);
    CHECK_EQ(count_in(left_exit, Opcode::add), 1ULL);
    CHECK_EQ(count_in(right, Opcode::add), 1ULL);
}

TEST_CASE("Code sinking - bits boxed as a tagged value move only when no GC point follows") {
    auto boxes_left = [](bool call_after) {
        Module mod("sink_box");
        Builder b(mod);
        Function* fn = mod.create_function("sink_box", Type::tagged(), {Type::i64(), Type::tagged(), Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* fast = b.append_block("fast");
        BasicBlock* slow = b.append_block("slow");
        b.position_at_end(entry);
        Value* bits = b.add_block_param(entry, Type::i64());
        Value* other = b.add_block_param(entry, Type::tagged());
        Value* c = b.add_block_param(entry, Type::i32());
        Value* boxed = b.build_bitcast_tagged_i64(bits);
        if (call_after) b.build_call("external_may_collect", Type::void_type(), {});
        b.build_br_if(c, fast, slow);
        b.position_at_end(fast);
        b.build_ret(other);
        b.position_at_end(slow);
        b.build_ret(boxed);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        sink_code(*fn);
        REQUIRE(verify_function(*fn));
        return count_in(entry, Opcode::bitcast_tagged_i64);
    };
    CHECK_EQ(boxes_left(false), 0ULL);
    CHECK_EQ(boxes_left(true), 1ULL);
}
