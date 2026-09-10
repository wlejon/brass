#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/osr.hpp>
#include <brass/embedding/embedding.hpp>

using namespace brass;

TEST_CASE("OSR Entry - Migration Frame ABI and Slots") {
    OsrMigrationFrame frame;
    frame.loop_header_id = 42;
    CHECK_EQ(frame.count, 0);
    CHECK_EQ(frame.loop_header_id, 42);

    frame.add_slot(0, 100ULL);
    frame.add_slot(1, 200ULL);
    frame.add_slot(2, 300ULL);

    CHECK_EQ(frame.count, 3);
    CHECK_EQ(frame.get_raw_value(0), 100ULL);
    CHECK_EQ(frame.get_raw_value(1), 200ULL);
    CHECK_EQ(frame.get_raw_value(2), 300ULL);

    // HostValue checks
    CHECK_EQ(frame.slots[0].slot_idx, 0);
    CHECK_EQ(frame.slots[1].slot_idx, 1);
    CHECK_EQ(frame.slots[2].slot_idx, 2);
}

TEST_CASE("OSR Entry - Target Live-in Analysis") {
    Module mod("osr_test_mod");
    // func @loop_fn(%n: i64, %step: i64) -> i64
    // b0:
    //   jump b1(0, 0)
    // b1(%i: i64, %acc: i64):
    //   %cond = cmp.slt %i, %n
    //   br %cond, b2, b3
    // b2:
    //   %acc_next = add %acc, %step
    //   %i_next = add %i, 1
    //   jump b1(%i_next, %acc_next)
    // b3:
    //   ret %acc
    Function* fn = mod.create_function("loop_fn", Type::i64(), {Type::i64(), Type::i64()});
    Builder b(*fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");

    b.position_at_end(b0);
    Value* n = b.add_param(Type::i64());
    Value* step = b.add_param(Type::i64());
    b.build_br(b1, {b.build_iconst_i64(0), b.build_iconst_i64(0)});

    b.position_at_end(b1);
    Value* i_val = b.add_param(Type::i64());
    Value* acc_val = b.add_param(Type::i64());
    Value* cond = b.build_slt(i_val, n);
    b.build_br_if(cond, b2, {}, b3, {});

    b.position_at_end(b2);
    Value* acc_next = b.build_add(acc_val, step);
    Value* i_next = b.build_add(i_val, b.build_iconst_i64(1));
    b.build_br(b1, {i_next, acc_next});

    b.position_at_end(b3);
    b.build_ret(acc_val);

    OsrTarget target = analyze_osr_target(*fn, b1);
    REQUIRE(target.is_valid());
    CHECK_EQ(target.loop_header, b1);
    CHECK_EQ(target.loop_header_id, b1->id());

    // Should include loop parameters (i_val, acc_val) and external live-ins (n, step)
    CHECK(target.live_in_count() >= 4);
    CHECK(target.get_slot_index(i_val) != UINT32_MAX);
    CHECK(target.get_slot_index(acc_val) != UINT32_MAX);
    CHECK(target.get_slot_index(n) != UINT32_MAX);
    CHECK(target.get_slot_index(step) != UINT32_MAX);

    auto all_targets = find_all_osr_targets(*fn);
    CHECK_EQ(all_targets.size(), 1);
    CHECK_EQ(all_targets[0].loop_header_id, b1->id());
}

TEST_CASE("OSR Entry - Compilation and Secondary Prologue Emission") {
    Module mod("osr_compile_mod");
    Function* fn = mod.create_function("counter_loop", Type::i64(), {Type::i64()});
    Builder b(*fn);

    BasicBlock* b0 = b.append_block("b0");
    BasicBlock* b1 = b.append_block("b1");
    BasicBlock* b2 = b.append_block("b2");
    BasicBlock* b3 = b.append_block("b3");

    b.position_at_end(b0);
    Value* max_val = b.add_param(Type::i64());
    b.build_br(b1, {b.build_iconst_i64(0), b.build_iconst_i64(0)});

    b.position_at_end(b1);
    Value* i_val = b.add_param(Type::i64());
    Value* sum_val = b.add_param(Type::i64());
    Value* cond = b.build_slt(i_val, max_val);
    b.build_br_if(cond, b2, {}, b3, {});

    b.position_at_end(b2);
    Value* sum_next = b.build_add(sum_val, i_val);
    Value* i_next = b.build_add(i_val, b.build_iconst_i64(1));
    b.build_br(b1, {i_next, sum_next});

    b.position_at_end(b3);
    b.build_ret(sum_val);

    HostEngine engine;
    auto compiled = engine.compile_with_osr(mod, "counter_loop", b1->id());
    REQUIRE(compiled != nullptr);

    size_t osr_offset = compiled->get_osr_entry_offset("counter_loop");
    CHECK(osr_offset > 0);

    void* osr_addr = compiled->get_osr_entry_address("counter_loop");
    CHECK(osr_addr != nullptr);
}
