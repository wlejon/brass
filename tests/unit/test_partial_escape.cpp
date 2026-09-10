#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/partial_escape.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>

using namespace brass;

TEST_CASE("PEA - Path-Sensitive Escape Analysis and Materialization Frontier") {
    Module mod("test_pea_path_sensitive");
    Builder b(mod);

    // func @test_cold_exit(%cond: i32, %val: i64) -> i64
    Function* fn = mod.create_function("test_cold_exit", Type::i64(), {Type::i32(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* hot_path = b.append_block("hot_path");
    BasicBlock* cold_path = b.append_block("cold_path");

    Value* cond = b.add_block_param(entry, Type::i32());
    Value* val = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});
    b.build_store(Type::i64(), alloc, 0, val);
    b.build_store(Type::i64(), alloc, 8, val);

    Value* zero = b.build_iconst_i32(0);
    Value* is_zero = b.build_eq(cond, zero);
    b.build_br_if(is_zero, hot_path, cold_path);

    // Hot path: reads fields, returns scalar sum
    b.position_at_end(hot_path);
    Value* v0 = b.build_load(Type::i64(), alloc, 0);
    Value* v8 = b.build_load(Type::i64(), alloc, 8);
    Value* sum = b.build_add(v0, v8);
    b.build_ret(sum);

    // Cold path: escapes object via external call
    b.position_at_end(cold_path);
    b.build_call("escape_sink", Type::void_type(), {alloc});
    b.build_ret(val);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    PartialEscapeAnalysis pea(*fn);
    CHECK(pea.is_candidate(alloc));
    CHECK_EQ(pea.get_block_state(entry, alloc), ObjectState::Virtual);
    CHECK_EQ(pea.get_block_state(hot_path, alloc), ObjectState::Virtual);
    CHECK_EQ(pea.get_block_state(cold_path, alloc), ObjectState::Materialized);

    auto frontier = pea.get_materialization_frontier(alloc);
    CHECK_EQ(frontier.size(), 1ULL);
    if (!frontier.empty()) {
        CHECK_EQ(frontier[0].from, entry);
        CHECK_EQ(frontier[0].to, cold_path);
    }
}

TEST_CASE("PEA - Virtual Object Slot and Field Tracking") {
    Module mod("test_pea_virtual_fields");
    Builder b(mod);

    Function* fn = mod.create_function("test_fields", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* ret_block = b.append_block("ret_block");

    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* sz24 = b.build_iconst_i64(24);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag});

    // Write initial fields
    b.build_store(Type::i64(), alloc, 0, a);
    b.build_store(Type::i64(), alloc, 8, a);
    b.build_store(Type::i64(), alloc, 16, c);

    // Overwrite field 8
    Value* new_b = b.build_add(a, c);
    b.build_store(Type::i64(), alloc, 8, new_b);

    b.build_br(ret_block);

    b.position_at_end(ret_block);
    Value* res = b.build_load(Type::i64(), alloc, 8);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    PartialEscapeAnalysis pea(*fn);
    CHECK(pea.is_candidate(alloc));

    const VirtualObject* vobj_entry = pea.get_virtual_object(entry, alloc);
    REQUIRE(vobj_entry != nullptr);
    CHECK_EQ(vobj_entry->state(), ObjectState::Virtual);
    CHECK(vobj_entry->has_field(0));
    CHECK(vobj_entry->has_field(8));
    CHECK(vobj_entry->has_field(16));
    CHECK_EQ(vobj_entry->get_field(0), a);
    CHECK_EQ(vobj_entry->get_field(8), new_b);
    CHECK_EQ(vobj_entry->get_field(16), c);
}

TEST_CASE("PEA - Merge Handling at Control Flow Join") {
    Module mod("test_pea_merge");
    Builder b(mod);

    Function* fn = mod.create_function("test_merge", Type::i64(), {Type::i32(), Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* left = b.append_block("left");
    BasicBlock* right = b.append_block("right");
    BasicBlock* merge = b.append_block("merge");
    BasicBlock* exit_hot = b.append_block("exit_hot");
    BasicBlock* exit_cold = b.append_block("exit_cold");

    Value* cond = b.add_block_param(entry, Type::i32());
    Value* v1 = b.add_block_param(entry, Type::i64());
    Value* v2 = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});

    Value* zero = b.build_iconst_i32(0);
    Value* is_zero = b.build_eq(cond, zero);
    b.build_br_if(is_zero, left, right);

    // Left branch
    b.position_at_end(left);
    b.build_store(Type::i64(), alloc, 0, v1);
    b.build_br(merge);

    // Right branch
    b.position_at_end(right);
    b.build_store(Type::i64(), alloc, 0, v2);
    b.build_br(merge);

    // Merge block: object is virtual across both branches
    b.position_at_end(merge);
    Value* read_v = b.build_load(Type::i64(), alloc, 0);
    Value* threshold = b.build_iconst_i64(100);
    Value* cmp = b.build_slt(read_v, threshold);
    b.build_br_if(cmp, exit_hot, exit_cold);

    b.position_at_end(exit_hot);
    b.build_ret(read_v);

    b.position_at_end(exit_cold);
    b.build_call("escape_sink", Type::void_type(), {alloc});
    b.build_ret(read_v);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    PartialEscapeAnalysis pea(*fn);
    CHECK(pea.is_candidate(alloc));
    CHECK_EQ(pea.get_block_state(entry, alloc), ObjectState::Virtual);
    CHECK_EQ(pea.get_block_state(left, alloc), ObjectState::Virtual);
    CHECK_EQ(pea.get_block_state(right, alloc), ObjectState::Virtual);
    CHECK_EQ(pea.get_block_state(merge, alloc), ObjectState::Virtual);
    CHECK_EQ(pea.get_block_state(exit_hot, alloc), ObjectState::Virtual);
    CHECK_EQ(pea.get_block_state(exit_cold, alloc), ObjectState::Materialized);

    auto frontier = pea.get_materialization_frontier(alloc);
    CHECK_EQ(frontier.size(), 1ULL);
    if (!frontier.empty()) {
        CHECK_EQ(frontier[0].from, merge);
        CHECK_EQ(frontier[0].to, exit_cold);
    }
}

TEST_CASE("PEA - Loop Escape Detection and Frontier Computation") {
    Module mod("test_pea_loop");
    Builder b(mod);

    // func @loop_test(%n: i64) -> gcref
    Function* fn = mod.create_function("loop_test", Type::gcref(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* header = b.append_block("header");
    BasicBlock* body = b.append_block("body");
    BasicBlock* exit = b.append_block("exit");

    Value* n = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    b.build_br(header, {zero});

    // header(%i: i64)
    Value* i_param = b.add_block_param(header, Type::i64());
    b.position_at_end(header);

    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});
    b.build_store(Type::i64(), alloc, 0, i_param);

    Value* cond = b.build_slt(i_param, n);
    b.build_br_if(cond, body, exit);

    // body
    b.position_at_end(body);
    Value* v0 = b.build_load(Type::i64(), alloc, 0);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(v0, one);
    b.build_br(header, {next_i});

    // exit: returns escaping object
    b.position_at_end(exit);
    b.build_ret(alloc);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    DominatorTree dom(*fn);
    LoopAnalysis la(*fn, dom);
    REQUIRE(!la.top_level_loops().empty());
    const LoopInfo& loop = *la.top_level_loops().front();

    PartialEscapeAnalysis pea(*fn);
    CHECK(pea.is_candidate(alloc));
    CHECK(pea.is_virtual_in_loop(alloc, loop));
    CHECK(!pea.escapes_in_loop(alloc, loop));

    auto frontier = pea.get_materialization_frontier(alloc);
    CHECK_EQ(frontier.size(), 1ULL);
    if (!frontier.empty()) {
        CHECK_EQ(frontier[0].from, header);
        CHECK_EQ(frontier[0].to, exit);
    }
}

TEST_CASE("PEA - Unconditional Immediate Escape Not Candidate") {
    Module mod("test_pea_unconditional");
    Builder b(mod);

    Function* fn = mod.create_function("test_uncond", Type::gcref(), {});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});
    b.build_ret(alloc);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    PartialEscapeAnalysis pea(*fn);
    CHECK(!pea.is_candidate(alloc));
    CHECK_EQ(pea.get_block_state(entry, alloc), ObjectState::Materialized);
    CHECK(pea.get_materialization_frontier(alloc).empty());
}
