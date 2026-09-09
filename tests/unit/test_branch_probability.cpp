#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/branch_probability.hpp>
#include <brass/pgo/instrument.hpp>
#include <brass/pgo/profile_data.hpp>
#include <cmath>

using namespace brass;
using namespace brass::mir;
using namespace brass::pgo;

TEST_CASE("Flow Conservation - Diamond CFG") {
    Module mod("test_diamond");
    Function* fn = mod.create_function("diamond", Type::i32(), {Type::i32()});
    Builder b(*fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* then_bb = b.append_block("then_bb");
    BasicBlock* else_bb = b.append_block("else_bb");
    BasicBlock* merge_bb = b.append_block("merge_bb");

    Value* arg = b.add_block_param(entry, Type::i32());
    Value* zero = b.build_iconst_i32(0);
    Value* cond = b.build_sgt(arg, zero);

    b.position_at_end(entry);
    b.build_br_if(cond, then_bb, else_bb);

    b.position_at_end(then_bb);
    b.build_br(merge_bb, {b.build_iconst_i32(1)});

    b.position_at_end(else_bb);
    b.build_br(merge_bb, {b.build_iconst_i32(2)});

    b.position_at_end(merge_bb);
    Value* p = b.add_block_param(merge_bb, Type::i32());
    b.build_ret(p);

    fn->rebuild_cfg_predecessors();

    // Profile: 1000 entries, chord counter = 300 for taken branch
    FunctionProfile prof;
    prof.name = "diamond";
    prof.entry_count = 1000;
    prof.edge_counters = {300};

    BranchProbabilityAnalysis bpa(*fn, prof);
    const auto& bfi = bpa.block_frequency_info();
    const auto& bpi = bpa.branch_probability_info();

    CHECK_EQ(bfi.get_block_count(entry), 1000ULL);
    CHECK_EQ(bfi.get_block_count(merge_bb), 1000ULL);
    CHECK_EQ(bfi.get_block_count(then_bb) + bfi.get_block_count(else_bb), 1000ULL);

    // Probabilities sum to 1.0
    double p_then = bpi.get_edge_probability(entry, then_bb);
    double p_else = bpi.get_edge_probability(entry, else_bb);
    CHECK(std::abs(p_then + p_else - 1.0) < 1e-6);

    CHECK(bfi.is_hot_block(entry, 0.50));
    CHECK(bfi.is_hot_block(merge_bb, 0.50));
}

TEST_CASE("Flow Conservation - Loop with Back-Edge") {
    Module mod("test_loop");
    Function* fn = mod.create_function("loop_fn", Type::i32(), {Type::i32()});
    Builder b(*fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_hdr = b.append_block("loop_hdr");
    BasicBlock* loop_body = b.append_block("loop_body");
    BasicBlock* loop_exit = b.append_block("loop_exit");

    Value* n = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);
    b.build_br(loop_hdr, {b.build_iconst_i32(0)});

    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i32());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_body, loop_exit);

    b.position_at_end(loop_body);
    Value* next_i = b.build_add(i, b.build_iconst_i32(1));
    b.build_br(loop_hdr, {next_i});

    b.position_at_end(loop_exit);
    b.build_ret(i);

    fn->rebuild_cfg_predecessors();

    // 100 calls, back-edge taken 900 times (10 iterations per call average)
    FunctionProfile prof;
    prof.name = "loop_fn";
    prof.entry_count = 100;
    prof.edge_counters = {900};

    BranchProbabilityAnalysis bpa(*fn, prof);
    const auto& bfi = bpa.block_frequency_info();
    const auto& bpi = bpa.branch_probability_info();

    CHECK_EQ(bfi.get_block_count(entry), 100ULL);
    CHECK_EQ(bfi.get_block_count(loop_exit), 100ULL);

    // Header count = 1000, body count = 900
    CHECK_EQ(bfi.get_block_count(loop_hdr), 1000ULL);
    CHECK_EQ(bfi.get_block_count(loop_body), 900ULL);

    // Probability loop_hdr -> loop_body: 900/1000 = 0.90
    CHECK(std::abs(bpi.get_edge_probability(loop_hdr, loop_body) - 0.90) < 1e-4);
    // Probability loop_hdr -> loop_exit: 100/1000 = 0.10
    CHECK(std::abs(bpi.get_edge_probability(loop_hdr, loop_exit) - 0.10) < 1e-4);

    CHECK(bfi.get_block_frequency(loop_hdr) >= 9.9);
}

TEST_CASE("Flow Conservation - Nested Conditionals") {
    Module mod("test_nested");
    Function* fn = mod.create_function("nested", Type::i32(), {Type::i32(), Type::i32()});
    Builder b(*fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* b_outer_then = b.append_block("outer_then");
    BasicBlock* b_inner_then = b.append_block("inner_then");
    BasicBlock* b_inner_else = b.append_block("inner_else");
    BasicBlock* b_outer_else = b.append_block("outer_else");
    BasicBlock* merge = b.append_block("merge");

    Value* a = b.add_block_param(entry, Type::i32());
    Value* b_val = b.add_block_param(entry, Type::i32());

    b.position_at_end(entry);
    b.build_br_if(b.build_sgt(a, b.build_iconst_i32(0)), b_outer_then, b_outer_else);

    b.position_at_end(b_outer_then);
    b.build_br_if(b.build_sgt(b_val, b.build_iconst_i32(0)), b_inner_then, b_inner_else);

    b.position_at_end(b_inner_then);
    b.build_br(merge, {b.build_iconst_i32(1)});

    b.position_at_end(b_inner_else);
    b.build_br(merge, {b.build_iconst_i32(2)});

    b.position_at_end(b_outer_else);
    b.build_br(merge, {b.build_iconst_i32(3)});

    b.position_at_end(merge);
    b.build_ret(b.add_block_param(merge, Type::i32()));

    fn->rebuild_cfg_predecessors();

    FunctionProfile prof;
    prof.name = "nested";
    prof.entry_count = 1000;
    prof.edge_counters = {200, 150};

    BranchProbabilityAnalysis bpa(*fn, prof);
    const auto& bfi = bpa.block_frequency_info();

    // Verify entry and merge counts equal entry_count
    CHECK_EQ(bfi.get_block_count(entry), 1000ULL);
    CHECK_EQ(bfi.get_block_count(merge), 1000ULL);

    // Sum of interior paths must equal 1000
    uint64_t path_sum = bfi.get_block_count(b_inner_then) +
                        bfi.get_block_count(b_inner_else) +
                        bfi.get_block_count(b_outer_else);
    CHECK_EQ(path_sum, 1000ULL);
}

TEST_CASE("Flow Conservation - Multi-Exit Function") {
    Module mod("test_multi_exit");
    Function* fn = mod.create_function("multi_ret", Type::i32(), {Type::i32()});
    Builder b(*fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* ret_early = b.append_block("ret_early");
    BasicBlock* ret_normal = b.append_block("ret_normal");

    Value* x = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);
    b.build_br_if(b.build_slt(x, b.build_iconst_i32(0)), ret_early, ret_normal);

    b.position_at_end(ret_early);
    b.build_ret(b.build_iconst_i32(-1));

    b.position_at_end(ret_normal);
    b.build_ret(b.build_iconst_i32(42));

    fn->rebuild_cfg_predecessors();

    FunctionProfile prof;
    prof.name = "multi_ret";
    prof.entry_count = 500;
    prof.edge_counters = {50}; // 50 early returns, 450 normal

    BranchProbabilityAnalysis bpa(*fn, prof);
    const auto& bfi = bpa.block_frequency_info();
    const auto& bpi = bpa.branch_probability_info();

    CHECK_EQ(bfi.get_block_count(entry), 500ULL);
    uint64_t c_early = bfi.get_block_count(ret_early);
    uint64_t c_normal = bfi.get_block_count(ret_normal);
    CHECK_EQ(c_early + c_normal, 500ULL);

    double p_early = bpi.get_edge_probability(entry, ret_early);
    double p_normal = bpi.get_edge_probability(entry, ret_normal);
    CHECK(std::abs(p_early + p_normal - 1.0) < 1e-6);
}
