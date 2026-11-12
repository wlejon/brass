// Untagged addresses carried through joins (untag_reuse.hpp): the arm that
// holds the address passes it, the arm that called computes it again, and a
// read a GC point separates from every earlier one stays.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/untag_reuse.hpp>
#include <brass/mir/verifier.hpp>

using namespace brass;
using namespace brass::test;

namespace {

constexpr int64_t kMask = -4222124650659840;  // a NaN-boxing front end's object tag

size_t count_in(const BasicBlock* bb, Opcode op) {
    size_t n = 0;
    for (const Instruction* inst : *bb) n += inst->opcode() == op;
    return n;
}

Value* untag(Builder& b, Value* tagged) {
    return b.build_xor(b.build_bitcast_i64_tagged(tagged), b.build_iconst_i64(kMask));
}

struct Diamond {
    Module mod{"untag_reuse"};
    Function* fn = nullptr;
    BasicBlock* entry = nullptr;
    BasicBlock* fast = nullptr;
    BasicBlock* slow = nullptr;
    BasicBlock* join = nullptr;
};

// entry: read x's field; if (c) goto fast else goto slow
// fast: goto join        slow: call; goto join        join: [call;] read x's field again; return the sum
void build(Diamond& d, bool read_above, bool call_in_join) {
    Builder b(d.mod);
    d.fn = d.mod.create_function("untag_reuse", Type::i64(), {Type::tagged(), Type::i32()});
    b.set_function(d.fn);
    d.entry = b.append_block("entry");
    d.fast = b.append_block("fast");
    d.slow = b.append_block("slow");
    d.join = b.append_block("join");
    b.position_at_end(d.entry);
    Value* x = b.add_block_param(d.entry, Type::tagged());
    Value* c = b.add_block_param(d.entry, Type::i32());
    Value* first = read_above ? b.build_load(Type::i64(), untag(b, x), 8) : b.build_iconst_i64(0);
    b.build_br_if(c, d.fast, d.slow);
    b.position_at_end(d.fast);
    b.build_br(d.join);
    b.position_at_end(d.slow);
    b.build_call("external_may_collect", Type::void_type(), {});
    b.build_br(d.join);
    b.position_at_end(d.join);
    if (call_in_join) b.build_call("external_may_collect", Type::void_type(), {});
    Value* second = b.build_load(Type::i64(), untag(b, x), 16);
    b.build_ret(b.build_add(first, second));
    d.fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*d.fn));
}

} // namespace

TEST_CASE("Untag reuse - the arm that holds the address passes it, the arm that called computes it") {
    Diamond d;
    build(d, true, false);
    CHECK_EQ(reuse_untagged_addresses(*d.fn), 1ULL);
    REQUIRE(verify_function(*d.fn));
    CHECK_EQ(d.join->param_count(), 1ULL);
    CHECK_EQ(count_in(d.join, Opcode::xor_), 0ULL);
    CHECK_EQ(count_in(d.join, Opcode::bitcast_i64_tagged), 0ULL);
    CHECK_EQ(count_in(d.fast, Opcode::xor_), 0ULL);
    // After its call, before its branch.
    CHECK_EQ(count_in(d.slow, Opcode::xor_), 1ULL);
    CHECK_EQ(count_in(d.slow, Opcode::bitcast_i64_tagged), 1ULL);
    CHECK_EQ(d.slow->head()->opcode(), Opcode::call);
    CHECK_EQ(count_in(d.entry, Opcode::xor_), 1ULL);
}

TEST_CASE("Untag reuse - a read past a GC point in the join stays") {
    Diamond d;
    build(d, true, true);
    CHECK_EQ(reuse_untagged_addresses(*d.fn), 0ULL);
    CHECK_EQ(d.join->param_count(), 0ULL);
    CHECK_EQ(count_in(d.join, Opcode::xor_), 1ULL);
}

TEST_CASE("Untag reuse - an address no arm holds is not computed on them") {
    Diamond d;
    build(d, false, false);
    CHECK_EQ(reuse_untagged_addresses(*d.fn), 0ULL);
    CHECK_EQ(d.join->param_count(), 0ULL);
    CHECK_EQ(count_in(d.join, Opcode::xor_), 1ULL);
    CHECK_EQ(count_in(d.fast, Opcode::xor_) + count_in(d.slow, Opcode::xor_), 0ULL);
}

TEST_CASE("Untag reuse - a loop header starts with nothing") {
    // entry: read x; goto head        head: read x; if (c) goto head else return
    Module mod("untag_loop");
    Builder b(mod);
    Function* fn = mod.create_function("untag_loop", Type::i64(), {Type::tagged(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* head = b.append_block("head");
    BasicBlock* exit = b.append_block("exit");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::tagged());
    Value* c = b.add_block_param(entry, Type::i32());
    b.build_load(Type::i64(), untag(b, x), 8);
    b.build_br(head);
    b.position_at_end(head);
    Value* v = b.build_load(Type::i64(), untag(b, x), 8);
    b.build_call("external_may_collect", Type::void_type(), {});
    b.build_br_if(c, head, exit);
    b.position_at_end(exit);
    b.build_ret(v);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    CHECK_EQ(reuse_untagged_addresses(*fn), 0ULL);
    REQUIRE(verify_function(*fn));
    CHECK_EQ(count_in(head, Opcode::xor_), 1ULL);
}
