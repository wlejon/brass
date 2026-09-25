#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/vm/bytecode.hpp>
#include <brass/vm/bytecode_compiler.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/gc/heap.hpp>
#include <vector>
#include <cstring>

using namespace brass;

// ============================================================================
// 1. Exception Handling & Unwinding Tests
// ============================================================================

TEST_CASE("Fast Interpreter - Simple Try Catch with Landing Pad") {
    Module mod("eh_simple");
    Builder b(mod);

    // callee: func @fail(%err_code: i64) -> i64
    Function* callee = mod.create_function("fail", Type::i64(), {Type::i64()});
    b.set_function(callee);
    BasicBlock* c_entry = b.append_block("entry");
    b.position_at_end(c_entry);
    Value* err_code = b.add_block_param(c_entry, Type::i64());
    b.build_throw(err_code);
    callee->rebuild_cfg_predecessors();

    // caller: func @try_caller(%arg: i64) -> i64
    Function* caller = mod.create_function("try_caller", Type::i64(), {Type::i64()});
    b.set_function(caller);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* normal_bb = b.append_block("normal_bb");
    BasicBlock* unwind_bb = b.append_block("unwind_bb");

    b.position_at_end(entry);
    Value* arg = b.add_block_param(entry, Type::i64());
    Instruction* inv = b.build_invoke("fail", Type::i64(), {arg}, normal_bb, unwind_bb);

    b.position_at_end(normal_bb);
    b.build_ret(inv->result());

    b.position_at_end(unwind_bb);
    Value* caught = b.build_landing_pad(Type::i64());
    Value* one = b.build_iconst_i64(1);
    Value* res = b.build_add(caught, one);
    b.build_ret(res);
    caller->rebuild_cfg_predecessors();

    // 1. Oracle: Interpreter
    Interpreter oracle;
    RuntimeValue o_res = oracle.run(*caller, {RuntimeValue::from_i64(41)});
    CHECK_EQ(o_res.as_i64(), 42);

    // 2. FastInterpreter
    FastInterpreter fast_interp;
    RuntimeValue f_res = fast_interp.run(*caller, {RuntimeValue::from_i64(41)});
    CHECK_EQ(f_res.as_i64(), 42);
    CHECK_EQ(f_res.as_i64(), o_res.as_i64());
}

TEST_CASE("Fast Interpreter - Invoke Normal Path Execution") {
    Module mod("eh_normal");
    Builder b(mod);

    // callee: func @double(%val: i64) -> i64
    Function* callee = mod.create_function("double", Type::i64(), {Type::i64()});
    b.set_function(callee);
    BasicBlock* c_entry = b.append_block("entry");
    b.position_at_end(c_entry);
    Value* val = b.add_block_param(c_entry, Type::i64());
    Value* two = b.build_iconst_i64(2);
    b.build_ret(b.build_mul(val, two));
    callee->rebuild_cfg_predecessors();

    // caller: invokes @double, if normal returns res + 5, if unwind returns -1
    Function* caller = mod.create_function("caller", Type::i64(), {Type::i64()});
    b.set_function(caller);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* norm_bb = b.append_block("norm_bb");
    BasicBlock* unw_bb = b.append_block("unw_bb");

    b.position_at_end(entry);
    Value* arg = b.add_block_param(entry, Type::i64());
    Instruction* inv = b.build_invoke("double", Type::i64(), {arg}, norm_bb, unw_bb);

    b.position_at_end(norm_bb);
    Value* five = b.build_iconst_i64(5);
    b.build_ret(b.build_add(inv->result(), five));

    b.position_at_end(unw_bb);
    (void)b.build_landing_pad(Type::i64());
    b.build_ret(b.build_iconst_i64(-1));
    caller->rebuild_cfg_predecessors();

    FastInterpreter fast_interp;
    RuntimeValue res = fast_interp.run(*caller, {RuntimeValue::from_i64(10)});
    CHECK_EQ(res.as_i64(), 25);
}

TEST_CASE("Fast Interpreter - Nested Exceptions and Resume Rethrow") {
    Module mod("eh_nested");
    Builder b(mod);

    // level3: func @level3(%code: i64) -> i64 (throws %code)
    Function* f3 = mod.create_function("level3", Type::i64(), {Type::i64()});
    b.set_function(f3);
    BasicBlock* f3_entry = b.append_block("entry");
    b.position_at_end(f3_entry);
    Value* code = b.add_block_param(f3_entry, Type::i64());
    b.build_throw(code);
    f3->rebuild_cfg_predecessors();

    // level2: func @level2(%arg: i64) -> i64
    // invokes @level3. On unwind, catches and rethrows using resume
    Function* f2 = mod.create_function("level2", Type::i64(), {Type::i64()});
    b.set_function(f2);
    BasicBlock* f2_entry = b.append_block("entry");
    BasicBlock* f2_norm = b.append_block("norm");
    BasicBlock* f2_unw = b.append_block("unw");

    b.position_at_end(f2_entry);
    Value* f2_arg = b.add_block_param(f2_entry, Type::i64());
    Instruction* inv2 = b.build_invoke("level3", Type::i64(), {f2_arg}, f2_norm, f2_unw);

    b.position_at_end(f2_norm);
    b.build_ret(inv2->result());

    b.position_at_end(f2_unw);
    Value* caught2 = b.build_landing_pad(Type::i64());
    b.build_resume(caught2); // rethrow to level1
    f2->rebuild_cfg_predecessors();

    // level1: func @level1(%arg: i64) -> i64
    // invokes @level2. On unwind, adds 1000 to caught value
    Function* f1 = mod.create_function("level1", Type::i64(), {Type::i64()});
    b.set_function(f1);
    BasicBlock* f1_entry = b.append_block("entry");
    BasicBlock* f1_norm = b.append_block("norm");
    BasicBlock* f1_unw = b.append_block("unw");

    b.position_at_end(f1_entry);
    Value* f1_arg = b.add_block_param(f1_entry, Type::i64());
    Instruction* inv1 = b.build_invoke("level2", Type::i64(), {f1_arg}, f1_norm, f1_unw);

    b.position_at_end(f1_norm);
    b.build_ret(inv1->result());

    b.position_at_end(f1_unw);
    Value* caught1 = b.build_landing_pad(Type::i64());
    Value* c1000 = b.build_iconst_i64(1000);
    b.build_ret(b.build_add(caught1, c1000));
    f1->rebuild_cfg_predecessors();

    // Test both Oracle and FastInterpreter
    Interpreter oracle;
    RuntimeValue o_val = oracle.run(*f1, {RuntimeValue::from_i64(42)});
    CHECK_EQ(o_val.as_i64(), 1042);

    FastInterpreter fast_interp;
    RuntimeValue f_val = fast_interp.run(*f1, {RuntimeValue::from_i64(42)});
    CHECK_EQ(f_val.as_i64(), 1042);
    CHECK_EQ(f_val.as_i64(), o_val.as_i64());
}

TEST_CASE("Fast Interpreter - Uncaught Exception Throws C++ Envelope") {
    Module mod("eh_uncaught");
    Builder b(mod);

    Function* fn = mod.create_function("throw_directly", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    b.build_throw(x);
    fn->rebuild_cfg_predecessors();

    FastInterpreter fast_interp;
    bool caught = false;
    try {
        fast_interp.run(*fn, {RuntimeValue::from_i64(777)});
    } catch (const InterpreterThrownException& ex) {
        caught = true;
        CHECK_EQ(ex.value().as_i64(), 777);
    }
    CHECK(caught);
}

// ============================================================================
// 2. Coroutine Lifecycle & State Tests
// ============================================================================

TEST_CASE("Fast Interpreter - Direct Coroutine Lifecycle (Generator)") {
    Module mod("coro_gen");
    Builder b(mod);

    // Generator function: yields 10, then 20, then returns 30
    Function* gen = mod.create_function("num_gen", Type::i64(), {});
    b.set_function(gen);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* v10 = b.build_iconst_i64(10);
    b.build_coro_suspend(v10, 1, Type::i64());

    Value* v20 = b.build_iconst_i64(20);
    b.build_coro_suspend(v20, 2, Type::i64());

    Value* v30 = b.build_iconst_i64(30);
    b.build_ret(v30);
    gen->rebuild_cfg_predecessors();

    FastInterpreter interp;
    uintptr_t handle = interp.coro_create(*gen, {});
    REQUIRE(handle != 0);
    CHECK(!interp.coro_is_done(handle));

    // First resume: yields 10
    uint64_t y1 = interp.coro_resume(handle, 0);
    CHECK_EQ(y1, 10ULL);
    CHECK(!interp.coro_is_done(handle));

    // Second resume: yields 20
    uint64_t y2 = interp.coro_resume(handle, 0);
    CHECK_EQ(y2, 20ULL);
    CHECK(!interp.coro_is_done(handle));

    // Third resume: completes and returns 30
    uint64_t y3 = interp.coro_resume(handle, 0);
    CHECK_EQ(y3, 30ULL);
    CHECK(interp.coro_is_done(handle));

    interp.coro_destroy(handle);
}

TEST_CASE("Fast Interpreter - Coroutine Resume Input Values (Accumulator)") {
    Module mod("coro_accum");
    Builder b(mod);

    // Coroutine: takes initial value %base, suspends yielding %base.
    // Receives %in1, suspends yielding %base + %in1.
    // Receives %in2, returns %base + %in1 + %in2.
    Function* accum = mod.create_function("accum", Type::i64(), {Type::i64()});
    b.set_function(accum);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* base = b.add_block_param(entry, Type::i64());

    Value* in1 = b.build_coro_suspend(base, 1, Type::i64());
    Value* sum1 = b.build_add(base, in1);

    Value* in2 = b.build_coro_suspend(sum1, 2, Type::i64());
    Value* sum2 = b.build_add(sum1, in2);
    b.build_ret(sum2);
    accum->rebuild_cfg_predecessors();

    FastInterpreter interp;
    uintptr_t handle = interp.coro_create(*accum, {RuntimeValue::from_i64(100)});
    REQUIRE(handle != 0);

    // First resume: yields initial base = 100
    uint64_t y1 = interp.coro_resume(handle, 0);
    CHECK_EQ(y1, 100ULL);
    CHECK(!interp.coro_is_done(handle));

    // Second resume: sends 25 -> yields 100 + 25 = 125
    uint64_t y2 = interp.coro_resume(handle, 25);
    CHECK_EQ(y2, 125ULL);
    CHECK(!interp.coro_is_done(handle));

    // Third resume: sends 75 -> completes with 125 + 75 = 200
    uint64_t y3 = interp.coro_resume(handle, 75);
    CHECK_EQ(y3, 200ULL);
    CHECK(interp.coro_is_done(handle));

    interp.coro_destroy(handle);
}

TEST_CASE("Fast Interpreter - Transformed Coroutine MIR Parity") {
    Module mod("coro_transformed");
    Builder b(mod);

    Function* fn = mod.create_function("t_coro", Type::i64(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::gcref());

    Value* v1 = b.build_iconst_i64(5);
    Value* r1 = b.build_coro_suspend(v1, 1, Type::i64());

    Value* v2 = b.build_add(v1, r1);
    Value* r2 = b.build_coro_suspend(v2, 2, Type::i64());

    Value* v3 = b.build_add(v2, r2);
    b.build_ret(v3);

    CoroTransformPass pass;
    REQUIRE(pass.run_on_module(mod));

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    // Test with FastInterpreter
    FastInterpreter interp;
    Function* t_fn = mod.get_function("t_coro");
    REQUIRE(t_fn != nullptr);

    uintptr_t c_frame = brass_coro_create(nullptr, 16, 0);
    REQUIRE(c_frame != 0);
    auto* f_ptr = reinterpret_cast<runtime::BrassCoroFrame*>(c_frame);

    // Resume 1
    RuntimeValue v_out1 = interp.run(*t_fn, {RuntimeValue::from_ptr(c_frame)});
    CHECK_EQ(v_out1.as_i64(), 5);
    CHECK_EQ(f_ptr->is_done, 0u);

    // Resume 2
    f_ptr->resume_arg = 15;
    RuntimeValue v_out2 = interp.run(*t_fn, {RuntimeValue::from_ptr(c_frame)});
    CHECK_EQ(v_out2.as_i64(), 20);
    CHECK_EQ(f_ptr->is_done, 0u);

    // Resume 3
    f_ptr->resume_arg = 30;
    RuntimeValue v_out3 = interp.run(*t_fn, {RuntimeValue::from_ptr(c_frame)});
    CHECK_EQ(v_out3.as_i64(), 50);
    CHECK_EQ(f_ptr->is_done, 1u);

    brass_coro_destroy(c_frame);
}

// ============================================================================
// 3. Speculation, Deoptimization & OSR Tests
// ============================================================================

TEST_CASE("Fast Interpreter - Guard Fast Path (Condition True)") {
    Module mod("guard_fast");
    Builder b(mod);

    Function* fn = mod.create_function("test_guard_pass", Type::i64(), {Type::i64(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Value* cond = b.add_block_param(entry, Type::i32());

    b.build_guard(cond, "fallback", {x});
    Value* c10 = b.build_iconst_i64(10);
    b.build_ret(b.build_add(x, c10));
    fn->rebuild_cfg_predecessors();

    FastInterpreter interp;
    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i64(50), RuntimeValue::from_i32(1)});
    CHECK_EQ(res.as_i64(), 60);
    CHECK(!interp.last_deopt().deoptimized);
}

TEST_CASE("Fast Interpreter - Guard Failure State Map Capture & DeoptException") {
    Module mod("guard_fail");
    Builder b(mod);

    Function* fn = mod.create_function("test_guard_fail", Type::i64(), {Type::i64(), Type::i64(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* cond = b.add_block_param(entry, Type::i32());

    Instruction* g = b.build_guard(cond, "exit_stub_42", {a, c});
    g->set_resume_id(42);
    b.build_ret(b.build_add(a, c));
    fn->rebuild_cfg_predecessors();

    FastInterpreter interp;
    bool caught = false;
    try {
        interp.run(*fn, {RuntimeValue::from_i64(100), RuntimeValue::from_i64(200), RuntimeValue::from_i32(0)});
    } catch (const DeoptException& de) {
        caught = true;
        CHECK(de.result().deoptimized);
        CHECK_EQ(de.result().resume_id, 42u);
        CHECK_EQ(de.result().exit_stub, "exit_stub_42");
        REQUIRE_EQ(de.result().state_map.size(), 2u);
        CHECK_EQ(de.result().state_map[0].as_i64(), 100);
        CHECK_EQ(de.result().state_map[1].as_i64(), 200);
    }
    CHECK(caught);
    CHECK(interp.last_deopt().deoptimized);
    CHECK_EQ(interp.last_deopt().resume_id, 42u);
}

TEST_CASE("Fast Interpreter - Custom Deopt Handler") {
    Module mod("guard_handler");
    Builder b(mod);

    Function* fn = mod.create_function("test_deopt_h", Type::i64(), {Type::i64(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Value* cond = b.add_block_param(entry, Type::i32());

    Instruction* g = b.build_guard(cond, "stub", {x});
    g->set_resume_id(99);
    b.build_ret(x);
    fn->rebuild_cfg_predecessors();

    FastInterpreter interp;
    interp.set_deopt_handler([](FastInterpreter&, const DeoptResult& deopt) -> RuntimeValue {
        int64_t captured = deopt.state_map[0].as_i64();
        return RuntimeValue::from_i64(captured * 10);
    });

    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i64(7), RuntimeValue::from_i32(0)});
    CHECK_EQ(res.as_i64(), 70);
}

TEST_CASE("Fast Interpreter - Guard Fallback to Stub Function") {
    Module mod("guard_stub");
    Builder b(mod);

    // fallback_stub: func @fallback_stub(%x: i64, %y: i64) -> i64
    Function* stub = mod.create_function("fallback_stub", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(stub);
    BasicBlock* s_entry = b.append_block("entry");
    b.position_at_end(s_entry);
    Value* sx = b.add_block_param(s_entry, Type::i64());
    Value* sy = b.add_block_param(s_entry, Type::i64());
    b.build_ret(b.build_mul(sx, sy));
    stub->rebuild_cfg_predecessors();

    // spec_fn: func @spec_fn(%x: i64, %y: i64, %cond: i32) -> i64
    Function* fn = mod.create_function("spec_fn", Type::i64(), {Type::i64(), Type::i64(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());
    Value* cond = b.add_block_param(entry, Type::i32());

    b.build_guard(cond, "fallback_stub", {x, y});
    b.build_ret(b.build_add(x, y));
    fn->rebuild_cfg_predecessors();

    FastInterpreter interp;
    // When cond = 0, guard triggers fallback_stub(x=6, y=7) -> 6 * 7 = 42
    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i64(6), RuntimeValue::from_i64(7), RuntimeValue::from_i32(0)});
    CHECK_EQ(res.as_i64(), 42);
}

TEST_CASE("Fast Interpreter - Resume from Resume ID") {
    Module mod("resume_test");
    Builder b(mod);

    Function* fn = mod.create_function("resume_fn", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* resume_target = b.append_block("resume_target");

    b.position_at_end(entry);
    (void)b.add_block_param(entry, Type::i64());
    b.build_br(resume_target, {b.build_iconst_i64(0)});

    b.position_at_end(resume_target);
    Value* r_param = b.add_block_param(resume_target, Type::i64());
    b.build_resume_point(55);
    Value* c10 = b.build_iconst_i64(10);
    b.build_ret(b.build_add(r_param, c10));

    fn->add_resume_point(55, resume_target);
    fn->rebuild_cfg_predecessors();

    FastInterpreter interp;
    // Resume with resume_id = 55, state value = 32 -> 32 + 10 = 42
    RuntimeValue res = interp.resume(*fn, 55, {RuntimeValue::from_i64(32)});
    CHECK_EQ(res.as_i64(), 42);
}

TEST_CASE("Fast Interpreter - OSR Loop Backedge Trigger Hook") {
    Module mod("osr_loop");
    Builder b(mod);

    // Sum loop 1 to 100
    Function* fn = mod.create_function("sum_loop", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_header = b.append_block("loop_header");
    BasicBlock* loop_body = b.append_block("loop_body");
    BasicBlock* exit_bb = b.append_block("exit_bb");

    b.position_at_end(entry);
    Value* n = b.add_block_param(entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_header, {one, zero});

    b.position_at_end(loop_header);
    Value* i = b.add_block_param(loop_header, Type::i64());
    Value* acc = b.add_block_param(loop_header, Type::i64());
    Value* cond = b.build_sle(i, n);
    b.build_br_if(cond, loop_body, exit_bb);

    b.position_at_end(loop_body);
    Value* new_acc = b.build_add(acc, i);
    Value* new_i = b.build_add(i, one);
    b.build_br(loop_header, {new_i, new_acc});

    b.position_at_end(exit_bb);
    b.build_ret(acc);
    fn->rebuild_cfg_predecessors();

    FastInterpreter interp;
    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i64(100)});
    CHECK_EQ(res.as_i64(), 5050);
}

// ============================================================================
// 4. Precise GC Root Scanning Tests
// ============================================================================

TEST_CASE("Fast Interpreter - GC Root Scanning during Active Frame") {
    Module mod("gc_roots");
    Builder b(mod);

    Function* fn = mod.create_function("gc_test_fn", Type::i64(), {});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    // Call external host function that triggers Cheney scavenge
    Value* call_inst = b.build_call("trigger_gc_scavenge", Type::i64(), {});
    b.build_ret(call_inst);
    fn->rebuild_cfg_predecessors();

    FastInterpreter interp(gc::HeapConfig{});

    // Allocate an object in GC heap before running
    uintptr_t obj = interp.heap().allocate_masked(24, 0, 1);
    REQUIRE(obj != 0);
    interp.heap().store(obj, 0, 0x1122334455667788ULL);
    interp.heap().store(obj, 1, 0xAABBCCDDEEFF0011ULL);

    uintptr_t old_addr = obj;
    uintptr_t new_addr = 0;

    interp.register_external_function("trigger_gc_scavenge", [&](FastInterpreter& in, const std::vector<RuntimeValue>&) -> RuntimeValue {
        uint64_t local_root = obj;
        in.heap().add_root(&local_root);
        // A full collection: the young object is promoted (moved).
        in.heap().collect(gc::CollectionKind::Full);
        in.heap().remove_root(&local_root);

        new_addr = static_cast<uintptr_t>(local_root);
        return RuntimeValue::from_i64(static_cast<int64_t>(gc::Heap::load(new_addr, 0)));
    });

    RuntimeValue res = interp.run(*fn, {});
    CHECK_EQ(static_cast<uint64_t>(res.as_i64()), 0x1122334455667788ULL);
    CHECK_NE(old_addr, new_addr);
    CHECK(interp.heap().is_valid_object(new_addr));
    CHECK(interp.heap().is_old(new_addr));
    CHECK(!interp.heap().is_valid_object(old_addr));
}

TEST_CASE("Fast Interpreter - Young Generation Scavenge with Active Frame") {
    gc::Heap gen_gc;
    FastInterpreter interp(&gen_gc);

    Module mod("gen_gc_mod");
    Builder b(mod);

    Function* fn = mod.create_function("gen_gc_run", Type::i64(), {});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* call_inst = b.build_call("gen_gc_hook", Type::i64(), {});
    b.build_ret(call_inst);
    fn->rebuild_cfg_predecessors();

    uintptr_t nursery_obj = gen_gc.allocate_masked(16, 0, 9);
    REQUIRE(nursery_obj != 0);
    gen_gc.store(nursery_obj, 0, 0xCAFEBABE12345678ULL);

    uintptr_t old_addr = nursery_obj;
    uintptr_t updated_addr = 0;

    interp.register_external_function("gen_gc_hook", [&](FastInterpreter&, const std::vector<RuntimeValue>&) -> RuntimeValue {
        uint64_t root = nursery_obj;
        gen_gc.add_root(&root);
        gen_gc.collect(gc::CollectionKind::Minor);
        gen_gc.remove_root(&root);

        updated_addr = static_cast<uintptr_t>(root);
        return RuntimeValue::from_i64(static_cast<int64_t>(gc::Heap::load(updated_addr, 0)));
    });

    RuntimeValue res = interp.run(*fn, {});
    CHECK_EQ(static_cast<uint64_t>(res.as_i64()), 0xCAFEBABE12345678ULL);
    CHECK_NE(old_addr, updated_addr);
    CHECK(gen_gc.is_valid_object(updated_addr));
    CHECK(gen_gc.is_young(updated_addr));  // copied to a survivor space, not yet promoted
}

TEST_CASE("Fast Interpreter - GC Root Scanning of Suspended Coroutine") {
    Module mod("coro_gc");
    Builder b(mod);

    // Coroutine holds GC object pointer in a register across suspend
    Function* gen = mod.create_function("gc_coro", Type::i64(), {Type::gcref()});
    b.set_function(gen);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* obj_ref = b.add_block_param(entry, Type::gcref());

    // Suspend yielding 1
    Value* v1 = b.build_iconst_i64(1);
    b.build_coro_suspend(v1, 1, Type::i64());

    // After resume, call host function to read and return field
    Value* read_call = b.build_call("read_coro_gc_field", Type::i64(), {obj_ref});
    b.build_ret(read_call);
    gen->rebuild_cfg_predecessors();

    FastInterpreter interp(gc::HeapConfig{});

    uintptr_t obj = interp.heap().allocate_masked(16, 0, 10);
    interp.heap().store(obj, 0, 0xDEADBEEFCAFEULL);
    uintptr_t old_addr = obj;
    uintptr_t updated_obj = 0;

    interp.register_external_function("read_coro_gc_field", [&](FastInterpreter&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        updated_obj = args[0].as_gcref();
        return RuntimeValue::from_i64(static_cast<int64_t>(gc::Heap::load(updated_obj, 0)));
    });

    uintptr_t handle = interp.coro_create(*gen, {RuntimeValue::from_gcref(obj)});
    REQUIRE(handle != 0);

    // 1. Run until first suspend: coroutine yields 1
    uint64_t y1 = interp.coro_resume(handle, 0);
    CHECK_EQ(y1, 1ULL);
    CHECK(!interp.coro_is_done(handle));

    // 2. Coroutine is now suspended. Trigger GC collection from host: the
    // interpreter's roots (the suspended coroutine's registers among them)
    // are a root source of its heap.
    std::vector<uintptr_t*> roots;
    interp.collect_all_roots(roots);
    REQUIRE(!roots.empty());
    interp.heap().collect(gc::CollectionKind::Minor);

    // 3. Resume coroutine: it reads from the updated object pointer
    uint64_t final_res = interp.coro_resume(handle, 0);
    CHECK_EQ(final_res, 0xDEADBEEFCAFEULL);
    CHECK_NE(old_addr, updated_obj);
    CHECK(interp.heap().is_valid_object(updated_obj));
    CHECK(interp.coro_is_done(handle));

    interp.coro_destroy(handle);
}
