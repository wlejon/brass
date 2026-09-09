#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/pgo/instrument.hpp>
#include <brass/pgo/profile_data.hpp>

using namespace brass;
using namespace brass::pgo;

TEST_CASE("PGO Instrument - Diamond CFG and Critical Edge Splitting") {
    Module mod("test_mod_diamond");
    Function* fn = mod.create_function("diamond", Type::i32(), {Type::i32()});
    Builder b(*fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* then_bb = b.append_block("then_bb");
    BasicBlock* else_bb = b.append_block("else_bb");
    BasicBlock* merge_bb = b.append_block("merge_bb");

    Value* arg0 = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);
    Value* zero = b.build_iconst_i32(0);
    Value* cond = b.build_sgt(arg0, zero);
    b.build_br_if(cond, then_bb, else_bb);

    b.position_at_end(then_bb);
    Value* v1 = b.build_iconst_i32(42);
    b.build_br(merge_bb, {v1});

    b.position_at_end(else_bb);
    Value* v2 = b.build_iconst_i32(100);
    b.build_br(merge_bb, {v2});

    b.position_at_end(merge_bb);
    Value* phi = b.add_block_param(merge_bb, Type::i32());
    b.build_ret(phi);

    fn->rebuild_cfg_predecessors();

    PgoInstrumentResult res = instrument_function(*fn, 10);
    CHECK_EQ(res.counter_base_index, 10U);
    CHECK_EQ(res.entry_counter_index, 10U);
    CHECK(res.total_counters >= 2U);
    CHECK(!res.chords.empty());
    CHECK(!res.spanning_tree_edges.empty());

    // Verify counter indices are strictly increasing
    for (size_t i = 0; i < res.chords.size(); ++i) {
        CHECK_EQ(res.chords[i].counter_index, 10U + static_cast<uint32_t>(i));
    }

    // Verify critical edge splitting occurred for split chords
    bool found_split = false;
    for (const auto& chord : res.chords) {
        if (chord.is_critical) {
            found_split = true;
            REQUIRE(chord.split_block != nullptr);
            CHECK(chord.split_block->name().find("pgo_chord") != std::string_view::npos);
        }
    }
    CHECK(found_split);

    // Verify CFG validity after instrumentation
    DiagnosticReporter diag;
    bool valid = verify_function(*fn, &diag);
    if (!valid) {
        std::cerr << "Verification failed: " << diag.format_all() << "\n";
    }
    CHECK(valid);
}

TEST_CASE("PGO Instrument - Loop CFG and Back-Edge Chord Selection") {
    Module mod("test_mod_loop");
    Function* fn = mod.create_function("sum_loop", Type::i32(), {Type::i32()});
    Builder b(*fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_hdr = b.append_block("loop_hdr");
    BasicBlock* loop_body = b.append_block("loop_body");
    BasicBlock* loop_exit = b.append_block("loop_exit");

    Value* n_arg = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);
    Value* zero = b.build_iconst_i32(0);
    b.build_br(loop_hdr, {zero, zero});

    b.position_at_end(loop_hdr);
    Value* i_val = b.add_block_param(loop_hdr, Type::i32());
    Value* acc_val = b.add_block_param(loop_hdr, Type::i32());
    Value* loop_cond = b.build_slt(i_val, n_arg);
    b.build_br_if(loop_cond, loop_body, loop_exit);

    b.position_at_end(loop_body);
    Value* one = b.build_iconst_i32(1);
    Value* next_i = b.build_add(i_val, one);
    Value* next_acc = b.build_add(acc_val, i_val);
    b.build_br(loop_hdr, {next_i, next_acc});

    b.position_at_end(loop_exit);
    b.build_ret(acc_val);

    fn->rebuild_cfg_predecessors();

    PgoInstrumentResult res = instrument_function(*fn, 0);
    CHECK_EQ(res.counter_base_index, 0U);
    CHECK_EQ(res.entry_counter_index, 0U);
    CHECK(res.total_counters >= 2U);

    // Verify back-edge (loop_body -> loop_hdr) or taken branch is selected as chord
    bool has_back_edge_chord = false;
    for (const auto& chord : res.chords) {
        if (chord.src == loop_body && chord.dst == loop_hdr) {
            has_back_edge_chord = true;
        }
    }
    CHECK(has_back_edge_chord);

    // Verify MIR remains valid
    DiagnosticReporter diag;
    bool valid = verify_function(*fn, &diag);
    if (!valid) {
        std::cerr << "Verification failed: " << diag.format_all() << "\n";
    }
    CHECK(valid);
}

TEST_CASE("PGO Instrument - Module Instrumentation and Dump") {
    Module mod("test_mod_multi");
    Function* fn1 = mod.create_function("f1", Type::i32(), {});
    {
        Builder b(*fn1);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        b.build_ret(b.build_iconst_i32(42));
    }

    Function* fn2 = mod.create_function("f2", Type::i32(), {Type::i32()});
    {
        Builder b(*fn2);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* exit_bb = b.append_block("exit_bb");
        b.position_at_end(entry);
        b.build_br(exit_bb);
        b.position_at_end(exit_bb);
        b.build_ret(b.build_iconst_i32(100));
    }

    PgoInstrumentResult mod_res = instrument_module(mod);
    CHECK(mod_res.total_counters >= 2U);
    CHECK_EQ(mod_res.metadata.functions.size(), 2ULL);

    // Verify function 2 counter base is offset from function 1
    CHECK_EQ(mod_res.metadata.functions[0].counter_base_index, 0U);
    CHECK(mod_res.metadata.functions[1].counter_base_index >= mod_res.metadata.functions[0].num_edge_counters + 1);

    // Test runtime counters and dump
    std::vector<uint64_t> counters(mod_res.total_counters, 0);
    counters[mod_res.metadata.functions[0].entry_counter_index] = 500;
    counters[mod_res.metadata.functions[1].entry_counter_index] = 1200;

    std::string dump_path = "test_mod_dump.bprof";
    bool dump_ok = brass_pgo_dump(dump_path.c_str(), counters.data(), counters.size(), mod_res.metadata);
    REQUIRE(dump_ok);

    std::string err;
    auto prof_read = ProfileData::read_from_file(dump_path, &err);
    REQUIRE(prof_read != nullptr);
    CHECK_EQ(prof_read->module_name(), "test_mod_multi");
    CHECK_EQ(prof_read->function_count(), 2ULL);

    const FunctionProfile* fp1 = prof_read->find_function("f1");
    REQUIRE(fp1 != nullptr);
    CHECK_EQ(fp1->entry_count, 500ULL);

    const FunctionProfile* fp2 = prof_read->find_function("f2");
    REQUIRE(fp2 != nullptr);
    CHECK_EQ(fp2->entry_count, 1200ULL);

    std::remove(dump_path.c_str());
}
