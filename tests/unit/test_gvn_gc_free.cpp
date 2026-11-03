#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/runtime_symbols.hpp>

using namespace brass;

// GVN's merge of re-reads of a tagged value's bits (gvn_gc_free.hpp), the
// front end's alias classes, and pure calls as non-clobbers.

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

// fn(t: tagged) -> i64: bitcast t, optionally `call @gc_point`, bitcast t
// again, add the two.
size_t bitcasts_after_gvn_straight(bool gc_between) {
    Module mod("gc_free_straight");
    Builder b(mod);
    Function* fn = mod.create_function("f", Type::i64(), {Type::tagged()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* t = b.add_block_param(entry, Type::tagged());
    Value* a = b.build_bitcast_i64_tagged(t);
    if (gc_between) b.build_call("gc_point", Type::void_type());
    Value* c = b.build_bitcast_i64_tagged(t);
    b.build_ret(b.build_add(a, c));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    gvn_function(*fn);
    REQUIRE(verify_function(*fn));
    return count_opcodes(*fn, Opcode::bitcast_i64_tagged);
}

// A bitcast before a loop and one in its body; `call @gc_point` either in
// the body after the read (reached again around the back edge) or nowhere.
size_t bitcasts_after_gvn_loop(bool gc_in_body) {
    Module mod("gc_free_loop");
    Builder b(mod);
    Function* fn = mod.create_function("f", Type::i64(), {Type::tagged(), Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* head = b.append_block("head");
    BasicBlock* body = b.append_block("body");
    BasicBlock* exit = b.append_block("exit");
    b.position_at_end(entry);
    Value* t = b.add_block_param(entry, Type::tagged());
    Value* n = b.add_block_param(entry, Type::i64());
    Value* a = b.build_bitcast_i64_tagged(t);
    Value* zero = b.build_iconst_i64(0);
    b.build_br(head, {zero, a});

    b.position_at_end(head);
    Value* i = b.add_block_param(head, Type::i64());
    Value* acc = b.add_block_param(head, Type::i64());
    b.build_br_if(b.build_ult(i, n), body, exit);

    b.position_at_end(body);
    Value* c = b.build_bitcast_i64_tagged(t);
    if (gc_in_body) b.build_call("gc_point", Type::void_type());
    Value* one = b.build_iconst_i64(1);
    b.build_br(head, {b.build_add(i, one), b.build_add(acc, c)});

    b.position_at_end(exit);
    b.build_ret(acc);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    gvn_function(*fn);
    REQUIRE(verify_function(*fn));
    return count_opcodes(*fn, Opcode::bitcast_i64_tagged);
}

} // namespace

TEST_CASE("GVN - tagged bitcasts merge with no GC point between, not across one") {
    CHECK_EQ(bitcasts_after_gvn_straight(false), 1ULL);
    CHECK_EQ(bitcasts_after_gvn_straight(true), 2ULL);
}

TEST_CASE("GVN - a tagged bitcast in a loop merges only when no GC point is on the back edge") {
    CHECK_EQ(bitcasts_after_gvn_loop(false), 1ULL);
    CHECK_EQ(bitcasts_after_gvn_loop(true), 2ULL);
}

TEST_CASE("GVN - a store of another alias class and a pure call do not clobber a load") {
    Module mod("alias_class_pure");
    mod.add_symbol_role("pure_fn", SymbolRole::Pure);
    Builder b(mod);
    Function* fn = mod.create_function("f", Type::i64(), {Type::ptr(), Type::ptr(), Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* p = b.add_block_param(entry, Type::ptr());
    Value* q = b.add_block_param(entry, Type::ptr());
    Value* v = b.add_block_param(entry, Type::i64());

    Value* l1 = b.build_load(Type::i64(), p, 8);
    l1->defining_instruction()->set_alias_class(2);
    Instruction* st = b.build_store(Type::i64(), q, 0, v);
    st->set_alias_class(3);
    Value* r = b.build_call("pure_fn", Type::i64(), {v});
    Value* l2 = b.build_load(Type::i64(), p, 8);
    l2->defining_instruction()->set_alias_class(2);
    // Same class: this store may overlap the load, which must stay.
    Instruction* st2 = b.build_store(Type::i64(), q, 0, r);
    st2->set_alias_class(2);
    Value* l3 = b.build_load(Type::i64(), p, 8);
    l3->defining_instruction()->set_alias_class(2);
    b.build_ret(b.build_add(b.build_add(l1, l2), l3));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    GvnStats stats;
    GvnOptions opts;
    opts.stats = &stats;
    gvn_function(*fn, opts);
    REQUIRE(verify_function(*fn));
    CHECK_EQ(count_opcodes(*fn, Opcode::load), 2ULL);
}
