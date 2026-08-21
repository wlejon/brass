#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <iostream>

using namespace brass;
using namespace brass::runtime;
using namespace brass::codegen;

TEST_CASE("Deopt - Metadata and DeoptValue conversions") {
    DeoptValue v_i32 = DeoptValue::i32(42);
    CHECK_EQ(v_i32.kind, DeoptValueKind::Int32);
    CHECK_EQ(v_i32.as_i32(), 42);

    DeoptValue v_i64 = DeoptValue::i64(0x123456789ABCDEF0LL);
    CHECK_EQ(v_i64.kind, DeoptValueKind::Int64);
    CHECK_EQ(v_i64.as_i64(), 0x123456789ABCDEF0LL);

    DeoptValue v_f64 = DeoptValue::f64(3.141592653589793);
    CHECK_EQ(v_f64.kind, DeoptValueKind::Float64);
    CHECK(std::abs(v_f64.as_f64() - 3.141592653589793) < 1e-12);

    DeoptValue v_ptr = DeoptValue::ptr(0xDEADBEEFULL);
    CHECK_EQ(v_ptr.kind, DeoptValueKind::Pointer);
    CHECK_EQ(v_ptr.as_ptr(), 0xDEADBEEFULL);

    DeoptValue v_gc = DeoptValue::gcref(0xCAFEBABEU);
    CHECK_EQ(v_gc.kind, DeoptValueKind::GcRef);
    CHECK_EQ(v_gc.as_gcref(), 0xCAFEBABEU);

    RuntimeValue rv = v_i64.to_runtime_value();
    CHECK_EQ(rv.as_i64(), 0x123456789ABCDEF0LL);

    DeoptValue roundtrip = DeoptValue::from_runtime_value(rv);
    CHECK_EQ(roundtrip.as_i64(), 0x123456789ABCDEF0LL);
}

TEST_CASE("Deopt - DeoptFrame and Thread-Local Storage") {
    DeoptFrame frame;
    frame.resume_id = 7;
    frame.reason = DeoptReason::TypeCheckFailed;
    frame.push_deopt_value(DeoptValue::i32(100));
    frame.push_deopt_value(DeoptValue::i64(200));
    frame.push_deopt_value(DeoptValue::f64(1.25));

    CHECK_EQ(frame.count, size_t(3));
    CHECK_EQ(frame.resume_id, 7u);
    CHECK_EQ(frame.reason, DeoptReason::TypeCheckFailed);
    CHECK_EQ(frame.get_value(0).as_i32(), 100);
    CHECK_EQ(frame.get_value(1).as_i64(), 200);
    CHECK(std::abs(frame.get_value(2).as_f64() - 1.25) < 1e-12);

    set_thread_deopt_frame(&frame);
    DeoptFrame* tl_frame = get_thread_deopt_frame();
    REQUIRE(tl_frame != nullptr);
    CHECK_EQ(tl_frame->resume_id, 7u);
    CHECK_EQ(tl_frame->count, size_t(3));
    CHECK_EQ(tl_frame->get_value(1).as_i64(), 200);

    set_thread_deopt_frame(nullptr);
    CHECK_EQ(tl_frame->count, size_t(0));
}

TEST_CASE("Resume Table - Metadata Registry and Entry Point Resolution") {
    FunctionResumeTable table;
    table.add_entry(0, 16, "bb_resume_0");
    table.add_entry(1, 48, "bb_resume_1");
    table.add_entry(2, 96, "bb_resume_2");

    CHECK_EQ(table.size(), size_t(3));
    CHECK(table.has_entry(0));
    CHECK(table.has_entry(1));
    CHECK(table.has_entry(2));
    CHECK(!table.has_entry(3));

    CHECK_EQ(table.get_offset(0), size_t(16));
    CHECK_EQ(table.get_offset(1), size_t(48));
    CHECK_EQ(table.get_offset(2), size_t(96));
    CHECK_EQ(table.get_block_name(1), "bb_resume_1");

    uint8_t dummy_base[128];
    void* addr_1 = table.get_target_address(dummy_base, 1);
    CHECK_EQ(addr_1, dummy_base + 48);

    ResumeTableRegistry reg;
    reg.register_table("twin_fn", table);
    CHECK(reg.has_table("twin_fn"));
    CHECK(!reg.has_table("other_fn"));

    const FunctionResumeTable* looked_up = reg.get_table("twin_fn");
    REQUIRE(looked_up != nullptr);
    CHECK_EQ(looked_up->get_offset(2), size_t(96));
}

TEST_CASE("Speculation - Native JIT Guard Fast Path (Condition True)") {
    // func @spec_fast(%x: i64, %cond: i32) -> i64
    //   guard %cond, @fallback, [%x]
    //   %res = add %x, 10
    //   ret %res
    Module mod("spec_fast_mod");
    Function* fn = mod.create_function("spec_fast", Type::i64(), {Type::i64(), Type::i32()});
    Builder b(*fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x = b.add_param(Type::i64());
    Value* cond = b.add_param(Type::i32());

    b.build_guard(cond, "", {x});
    Value* c10 = b.build_iconst_i64(10);
    Value* res = b.build_add(x, c10);
    b.build_ret(res);

    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    // Clear deopt frame
    get_thread_deopt_frame()->clear();

    // Invoke with cond = 1 (Guard passes)
    RuntimeValue result = engine.invoke("spec_fast", {RuntimeValue::from_i64(32), RuntimeValue::from_i32(1)});
    CHECK_EQ(result.as_i64(), 42);

    // Assert that deoptimization was NOT triggered
    CHECK_EQ(get_thread_deopt_frame()->count, size_t(0));
}

TEST_CASE("Speculation - Native JIT Guard Slow Path & State Map Capture") {
    // func @spec_slow(%a: i64, %b: i64, %cond: i32) -> i64
    //   guard %cond, @fallback_exit, [%a, %b]
    //   %res = add %a, %b
    //   ret %res
    Module mod("spec_slow_mod");
    Function* fn = mod.create_function("spec_slow", Type::i64(), {Type::i64(), Type::i64(), Type::i32()});
    Builder b(*fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* a = b.add_param(Type::i64());
    Value* b_param = b.add_param(Type::i64());
    Value* cond = b.add_param(Type::i32());

    Instruction* guard_inst = b.build_guard(cond, "", {a, b_param});
    guard_inst->set_resume_id(42);
    Value* res = b.build_add(a, b_param);
    b.build_ret(res);

    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    get_thread_deopt_frame()->clear();

    // Register a deopt handler to record when exit stub is triggered
    bool handler_called = false;
    register_deopt_handler([&](const DeoptFrame& frame) -> void* {
        handler_called = true;
        CHECK_EQ(frame.resume_id, 42);
        CHECK_EQ(frame.count, 2);
        CHECK_EQ(frame.get_value(0).as_i64(), 100);
        CHECK_EQ(frame.get_value(1).as_i64(), 200);
        return nullptr;
    });

    // Invoke with cond = 0 (Guard fails -> deoptimization exit stub executes)
    engine.invoke("spec_slow", {RuntimeValue::from_i64(100), RuntimeValue::from_i64(200), RuntimeValue::from_i32(0)});

    CHECK(handler_called);

    // Verify thread-local deopt frame captured values
    DeoptFrame* frame = get_thread_deopt_frame();
    REQUIRE(frame != nullptr);
    CHECK_EQ(frame->resume_id, 42);
    CHECK_EQ(frame->count, 2);
    CHECK_EQ(frame->get_value(0).as_i64(), 100);
    CHECK_EQ(frame->get_value(1).as_i64(), 200);

    register_deopt_handler(nullptr);
}

TEST_CASE("Speculation - Native JIT Deoptimization into Generic Twin at Interior Resume Point") {
    Module mod("deopt_twin_mod");

    // Generic twin function with interior resume table:
    // func @generic_twin(%resume_id: i32, %state_buf: ptr) -> i64
    // resume_table {
    //   entry 0 -> bb_resume_0
    //   entry 1 -> bb_resume_1
    // }
    // bb0:
    //   ret 0
    // bb_resume_0:
    //   %v0 = load.i64 %state_buf, 0
    //   %v1 = load.i64 %state_buf, 8
    //   %r0 = add %v0, %v1
    //   %r1 = mul %r0, 2
    //   ret %r1
    // bb_resume_1:
    //   %w0 = load.i64 %state_buf, 0
    //   %w1 = load.i64 %state_buf, 8
    //   %r2 = sub %w0, %w1
    //   ret %r2
    Function* twin = mod.create_function("generic_twin", Type::i64(), {Type::i32(), Type::ptr()});
    Builder tb(*twin);

    BasicBlock* tb_entry = tb.append_block("bb0");
    BasicBlock* tb_res0 = tb.append_block("bb_resume_0");
    BasicBlock* tb_res1 = tb.append_block("bb_resume_1");

    twin->add_resume_point(0, tb_res0);
    twin->add_resume_point(1, tb_res1);

    tb.position_at_end(tb_entry);
    (void)tb.add_param(Type::i32());
    Value* state_buf = tb.add_param(Type::ptr());
    tb.build_ret(tb.build_iconst_i64(0));

    tb.position_at_end(tb_res0);
    Value* v0 = tb.build_load(Type::i64(), state_buf, 0);
    Value* v1 = tb.build_load(Type::i64(), state_buf, 8);
    Value* sum = tb.build_add(v0, v1);
    Value* doubled = tb.build_mul(sum, tb.build_iconst_i64(2));
    tb.build_ret(doubled);

    tb.position_at_end(tb_res1);
    Value* w0 = tb.build_load(Type::i64(), state_buf, 0);
    Value* w1 = tb.build_load(Type::i64(), state_buf, 8);
    Value* diff = tb.build_sub(w0, w1);
    tb.build_ret(diff);

    // Speculatively optimized function:
    // func @spec_opt(%x: i64, %y: i64, %cond: i32) -> i64
    //   guard %cond, @generic_twin, [%x, %y]  (with resume_id = 0)
    //   %fast_res = mul %x, %y
    //   ret %fast_res
    Function* opt_fn = mod.create_function("spec_opt", Type::i64(), {Type::i64(), Type::i64(), Type::i32()});
    Builder ob(*opt_fn);
    BasicBlock* opt_entry = ob.append_block("entry");
    ob.position_at_end(opt_entry);
    Value* x = ob.add_param(Type::i64());
    Value* y = ob.add_param(Type::i64());
    Value* cond = ob.add_param(Type::i32());

    Instruction* g = ob.build_guard(cond, "generic_twin", {x, y});
    g->set_resume_id(0);
    Value* fast_res = ob.build_mul(x, y);
    ob.build_ret(fast_res);

    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    // 1. Guard passes (cond = 1): executes fast path (x * y = 5 * 6 = 30)
    RuntimeValue fast_val = engine.invoke("spec_opt", {RuntimeValue::from_i64(5), RuntimeValue::from_i64(6), RuntimeValue::from_i32(1)});
    CHECK_EQ(fast_val.as_i64(), 30);

    // 2. Guard fails (cond = 0): deopts into @generic_twin at resume point 0
    // State [5, 6] captured -> generic_twin at bb_resume_0 computes (5 + 6) * 2 = 22
    RuntimeValue deopt_val = engine.invoke("spec_opt", {RuntimeValue::from_i64(5), RuntimeValue::from_i64(6), RuntimeValue::from_i32(0)});
    CHECK_EQ(deopt_val.as_i64(), 22);

    // 3. Directly resume twin at entry 1 with state buffer
    uint64_t manual_state[2] = {50, 20};
    RuntimeValue res1_val = engine.resume("generic_twin", 1, {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(manual_state))});
    CHECK_EQ(res1_val.as_i64(), 30); // 50 - 20 = 30
}

TEST_CASE("Speculation - Multiple Guards and Interior Resume Points") {
    Module mod("multi_guard_mod");

    // Generic twin function:
    // func @twin_multi(%resume_id: i32, %buf: ptr) -> i64
    // resume_table {
    //   entry 10 -> bb_res_10
    //   entry 20 -> bb_res_20
    // }
    Function* twin = mod.create_function("twin_multi", Type::i64(), {Type::i32(), Type::ptr()});
    Builder tb(*twin);

    BasicBlock* tb_0 = tb.append_block("bb0");
    BasicBlock* tb_10 = tb.append_block("bb_res_10");
    BasicBlock* tb_20 = tb.append_block("bb_res_20");

    twin->add_resume_point(10, tb_10);
    twin->add_resume_point(20, tb_20);

    tb.position_at_end(tb_0);
    tb.add_param(Type::i32());
    tb.add_param(Type::ptr());
    tb.build_ret(tb.build_iconst_i64(0));

    tb.position_at_end(tb_10);
    Value* a10 = tb.build_load(Type::i64(), twin->entry_block()->param(1), 0);
    Value* r10 = tb.build_add(a10, tb.build_iconst_i64(1000));
    tb.build_ret(r10);

    tb.position_at_end(tb_20);
    Value* a20 = tb.build_load(Type::i64(), twin->entry_block()->param(1), 0);
    Value* r20 = tb.build_add(a20, tb.build_iconst_i64(2000));
    tb.build_ret(r20);

    // Optimized function with 2 guards:
    // func @multi_guard(%val: i64, %g1: i32, %g2: i32) -> i64
    Function* opt = mod.create_function("multi_guard", Type::i64(), {Type::i64(), Type::i32(), Type::i32()});
    Builder ob(*opt);
    BasicBlock* opt_entry = ob.append_block("entry");
    ob.position_at_end(opt_entry);
    Value* val = ob.add_param(Type::i64());
    Value* g1 = ob.add_param(Type::i32());
    Value* g2 = ob.add_param(Type::i32());

    Instruction* guard1 = ob.build_guard(g1, "twin_multi", {val});
    guard1->set_resume_id(10);

    Value* step1 = ob.build_add(val, ob.build_iconst_i64(5));

    Instruction* guard2 = ob.build_guard(g2, "twin_multi", {step1});
    guard2->set_resume_id(20);

    Value* step2 = ob.build_add(step1, ob.build_iconst_i64(10));
    ob.build_ret(step2);

    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    // Case 1: Both guards pass (g1=1, g2=1) -> 7 + 5 + 10 = 22
    RuntimeValue r_all_pass = engine.invoke("multi_guard", {RuntimeValue::from_i64(7), RuntimeValue::from_i32(1), RuntimeValue::from_i32(1)});
    CHECK_EQ(r_all_pass.as_i64(), 22);

    // Case 2: Guard 1 fails (g1=0, g2=1) -> deopts at resume_id=10 with val=7 -> 7 + 1000 = 1007
    RuntimeValue r_g1_fail = engine.invoke("multi_guard", {RuntimeValue::from_i64(7), RuntimeValue::from_i32(0), RuntimeValue::from_i32(1)});
    CHECK_EQ(r_g1_fail.as_i64(), 1007);

    // Case 3: Guard 1 passes, Guard 2 fails (g1=1, g2=0) -> deopts at resume_id=20 with step1=12 -> 12 + 2000 = 2012
    RuntimeValue r_g2_fail = engine.invoke("multi_guard", {RuntimeValue::from_i64(7), RuntimeValue::from_i32(1), RuntimeValue::from_i32(0)});
    CHECK_EQ(r_g2_fail.as_i64(), 2012);
}
