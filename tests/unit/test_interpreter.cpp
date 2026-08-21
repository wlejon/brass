#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <vector>
#include <cmath>

using namespace brass;

TEST_CASE("Interpreter - Arithmetic & Logic") {
    Module mod("arith_test");
    Builder b(mod);

    // func @test_i32(%a: i32, %b: i32) -> i32
    Function* fn = mod.create_function("test_i32", Type::i32(), {Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");

    Value* a = b.add_block_param(bb, Type::i32());
    Value* b_param = b.add_block_param(bb, Type::i32());

    Value* sum = b.build_add(a, b_param);
    Value* diff = b.build_sub(sum, b_param);
    Value* prod = b.build_mul(diff, b_param);
    Value* quot = b.build_sdiv(prod, a);
    b.build_ret(quot);

    Interpreter interp;
    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i32(10), RuntimeValue::from_i32(5)});
    CHECK_EQ(res.as_i32(), 5);
}

TEST_CASE("Interpreter - Bitwise, CLZ, CTZ, Popcnt") {
    Module mod("bit_test");
    Builder b(mod);

    Function* fn = mod.create_function("test_bits", Type::i32(), {Type::i32()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");

    Value* x = b.add_block_param(bb, Type::i32());
    Value* c1 = b.build_clz(x);
    Value* c2 = b.build_ctz(x);
    Value* p = b.build_popcnt(x);
    Value* sum1 = b.build_add(c1, c2);
    Value* total = b.build_add(sum1, p);
    b.build_ret(total);

    Interpreter interp;
    // x = 0b0000_0000_0000_0000_0000_0000_0000_1000 = 8
    // clz(8) = 28, ctz(8) = 3, popcnt(8) = 1 -> 28 + 3 + 1 = 32
    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i32(8)});
    CHECK_EQ(res.as_i32(), 32);
}

TEST_CASE("Interpreter - Bitcasts & Float Conversions") {
    Module mod("float_test");
    Builder b(mod);

    Function* fn = mod.create_function("test_float_ops", Type::i64(), {Type::i32()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");

    Value* x = b.add_block_param(bb, Type::i32());
    Value* f = b.build_sitofp_f64_i32(x);
    Value* half = b.build_fconst_f64(0.5);
    Value* prod = b.build_mul(f, half);
    Value* as_i64 = b.build_bitcast_i64_f64(prod);
    b.build_ret(as_i64);

    Interpreter interp;
    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i32(10)});
    // 10 * 0.5 = 5.0. Bitcast 5.0 to i64
    double d = res.as_f64();
    CHECK_EQ(d, 5.0);
}

TEST_CASE("Interpreter - Recursive Fibonacci") {
    Module mod("fib_rec_mod");
    Builder b(mod);

    // func @fib(%n: i32) -> i32
    Function* fn = mod.create_function("fib", Type::i32(), {Type::i32()});
    b.set_function(fn);
    BasicBlock* bb_entry = b.append_block("entry");
    BasicBlock* bb_base = b.append_block("base");
    BasicBlock* bb_recurse = b.append_block("recurse");

    b.position_at_end(bb_entry);
    Value* n = b.add_block_param(bb_entry, Type::i32());
    Value* c2 = b.build_iconst_i32(2);
    Value* cond = b.build_slt(n, c2);
    b.build_br_if(cond, bb_base, {}, bb_recurse, {});

    b.position_at_end(bb_base);
    b.build_ret(n);

    b.position_at_end(bb_recurse);
    Value* c1 = b.build_iconst_i32(1);
    Value* n_sub_1 = b.build_sub(n, c1);
    Value* r1 = b.build_call("fib", Type::i32(), {n_sub_1});

    Value* n_sub_2 = b.build_sub(n, c2);
    Value* r2 = b.build_call("fib", Type::i32(), {n_sub_2});

    Value* sum = b.build_add(r1, r2);
    b.build_ret(sum);

    Interpreter interp;
    interp.set_module(&mod);

    RuntimeValue res0 = interp.run(*fn, {RuntimeValue::from_i32(0)});
    CHECK_EQ(res0.as_i32(), 0);

    RuntimeValue res1 = interp.run(*fn, {RuntimeValue::from_i32(1)});
    CHECK_EQ(res1.as_i32(), 1);

    RuntimeValue res10 = interp.run(*fn, {RuntimeValue::from_i32(10)});
    CHECK_EQ(res10.as_i32(), 55);

    RuntimeValue res15 = interp.run(*fn, {RuntimeValue::from_i32(15)});
    CHECK_EQ(res15.as_i32(), 610);
}

TEST_CASE("Interpreter - Iterative Loop with Block Arguments (Sum 1 to N)") {
    Module mod("loop_sum_mod");
    Builder b(mod);

    // func @sum_1_to_n(%n: i64) -> i64
    Function* fn = mod.create_function("sum_1_to_n", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* bb_entry = b.append_block("entry");
    BasicBlock* bb_loop = b.append_block("loop");
    BasicBlock* bb_body = b.append_block("body");
    BasicBlock* bb_exit = b.append_block("exit");

    b.position_at_end(bb_entry);
    Value* n = b.add_block_param(bb_entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(bb_loop, {one, zero}); // loop(i = 1, acc = 0)

    b.position_at_end(bb_loop);
    Value* i = b.add_block_param(bb_loop, Type::i64());
    Value* acc = b.add_block_param(bb_loop, Type::i64());

    Value* cond = b.build_sle(i, n);
    b.build_br_if(cond, bb_body, {}, bb_exit, {acc});

    b.position_at_end(bb_body);
    Value* next_acc = b.build_add(acc, i);
    Value* next_i = b.build_add(i, one);
    b.build_br(bb_loop, {next_i, next_acc});

    b.position_at_end(bb_exit);
    Value* final_acc = b.add_block_param(bb_exit, Type::i64());
    b.build_ret(final_acc);

    Interpreter interp;
    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_i64(100)});
    CHECK_EQ(res.as_i64(), 5050);
}

TEST_CASE("Interpreter - Collatz Sequence Steps") {
    Module mod("collatz_mod");
    Builder b(mod);

    // func @collatz_steps(%n: i64) -> i64
    Function* fn = mod.create_function("collatz_steps", Type::i64(), {Type::i64()});
    b.set_function(fn);
    BasicBlock* bb_entry = b.append_block("entry");
    BasicBlock* bb_loop = b.append_block("loop");
    BasicBlock* bb_step = b.append_block("step");
    BasicBlock* bb_even = b.append_block("even");
    BasicBlock* bb_odd = b.append_block("odd");
    BasicBlock* bb_done = b.append_block("done");

    b.position_at_end(bb_entry);
    Value* n = b.add_block_param(bb_entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    b.build_br(bb_loop, {n, zero});

    b.position_at_end(bb_loop);
    Value* curr_n = b.add_block_param(bb_loop, Type::i64());
    Value* steps = b.add_block_param(bb_loop, Type::i64());
    Value* one = b.build_iconst_i64(1);
    Value* is_one = b.build_eq(curr_n, one);
    b.build_br_if(is_one, bb_done, {steps}, bb_step, {});

    b.position_at_end(bb_step);
    Value* two = b.build_iconst_i64(2);
    Value* rem = b.build_smod(curr_n, two);
    Value* is_even = b.build_eq(rem, zero);
    b.build_br_if(is_even, bb_even, {}, bb_odd, {});

    b.position_at_end(bb_even);
    Value* next_even_n = b.build_sdiv(curr_n, two);
    Value* next_even_steps = b.build_add(steps, one);
    b.build_br(bb_loop, {next_even_n, next_even_steps});

    b.position_at_end(bb_odd);
    Value* three = b.build_iconst_i64(3);
    Value* mul3 = b.build_mul(curr_n, three);
    Value* next_odd_n = b.build_add(mul3, one);
    Value* next_odd_steps = b.build_add(steps, one);
    b.build_br(bb_loop, {next_odd_n, next_odd_steps});

    b.position_at_end(bb_done);
    Value* final_steps = b.add_block_param(bb_done, Type::i64());
    b.build_ret(final_steps);

    Interpreter interp;
    // Collatz steps for 6: 6 -> 3 -> 10 -> 5 -> 16 -> 8 -> 4 -> 2 -> 1 (8 steps)
    RuntimeValue res6 = interp.run(*fn, {RuntimeValue::from_i64(6)});
    CHECK_EQ(res6.as_i64(), 8);

    // Collatz steps for 27: 111 steps
    RuntimeValue res27 = interp.run(*fn, {RuntimeValue::from_i64(27)});
    CHECK_EQ(res27.as_i64(), 111);
}

TEST_CASE("Interpreter - Raw Pointer Array Read / Write") {
    Module mod("ptr_test");
    Builder b(mod);

    // func @fill_and_sum(%arr: ptr, %len: i64) -> i64
    Function* fn = mod.create_function("fill_and_sum", Type::i64(), {Type::ptr(), Type::i64()});
    b.set_function(fn);
    BasicBlock* bb_entry = b.append_block("entry");
    BasicBlock* bb_fill_loop = b.append_block("fill_loop");
    BasicBlock* bb_fill_body = b.append_block("fill_body");
    BasicBlock* bb_sum_init = b.append_block("sum_init");
    BasicBlock* bb_sum_loop = b.append_block("sum_loop");
    BasicBlock* bb_sum_body = b.append_block("sum_body");
    BasicBlock* bb_done = b.append_block("done");

    b.position_at_end(bb_entry);
    Value* arr = b.add_block_param(bb_entry, Type::ptr());
    Value* len = b.add_block_param(bb_entry, Type::i64());
    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(bb_fill_loop, {zero});

    // Fill loop: arr[i] = i * 10
    b.position_at_end(bb_fill_loop);
    Value* i_fill = b.add_block_param(bb_fill_loop, Type::i64());
    Value* cond_fill = b.build_slt(i_fill, len);
    b.build_br_if(cond_fill, bb_fill_body, {}, bb_sum_init, {});

    b.position_at_end(bb_fill_body);
    Value* c10 = b.build_iconst_i64(10);
    Value* elem_val = b.build_mul(i_fill, c10);
    b.build_store_indexed(Type::i64(), arr, i_fill, 8, 0, elem_val);
    Value* next_i_fill = b.build_add(i_fill, one);
    b.build_br(bb_fill_loop, {next_i_fill});

    // Sum loop: sum += arr[i]
    b.position_at_end(bb_sum_init);
    b.build_br(bb_sum_loop, {zero, zero});

    b.position_at_end(bb_sum_loop);
    Value* i_sum = b.add_block_param(bb_sum_loop, Type::i64());
    Value* total = b.add_block_param(bb_sum_loop, Type::i64());
    Value* cond_sum = b.build_slt(i_sum, len);
    b.build_br_if(cond_sum, bb_sum_body, {}, bb_done, {total});

    b.position_at_end(bb_sum_body);
    Value* loaded_val = b.build_load_indexed(Type::i64(), arr, i_sum, 8, 0);
    Value* next_total = b.build_add(total, loaded_val);
    Value* next_i_sum = b.build_add(i_sum, one);
    b.build_br(bb_sum_loop, {next_i_sum, next_total});

    b.position_at_end(bb_done);
    Value* final_sum = b.add_block_param(bb_done, Type::i64());
    b.build_ret(final_sum);

    int64_t buffer[10] = {0};
    Interpreter interp;
    RuntimeValue res = interp.run(*fn, {RuntimeValue::from_ptr(buffer), RuntimeValue::from_i64(10)});

    // Sum of 0*10 + 1*10 + ... + 9*10 = 10 * (9*10/2) = 450
    CHECK_EQ(res.as_i64(), 450);
    for (int64_t k = 0; k < 10; ++k) {
        CHECK_EQ(buffer[k], k * 10);
    }
}

TEST_CASE("Interpreter - Moving GC Heap Execution with Safepoint") {
    Module mod("gc_exec_mod");
    Builder b(mod);

    // Build function that:
    // 1. Allocates GC object 1 (val = 100)
    // 2. Allocates GC object 2 (val = 200, next = obj1)
    // 3. Executes explicit safepoint (GC runs in stress mode, moving both objects!)
    // 4. Reads obj2.val and obj2.next.val, adds them and returns sum
    Function* fn = mod.create_function("gc_test_fn", Type::i64(), {});
    b.set_function(fn);
    b.append_block("entry");

    Value* size16 = b.build_iconst_i64(16);
    Value* mask2 = b.build_iconst_i64(2); // bit 1 is gcref
    Value* zero_mask = b.build_iconst_i64(0);
    Value* tag1 = b.build_iconst_i32(1);

    // obj1 = brass_gc_alloc(16, 0, 1)
    Value* obj1 = b.build_call("brass_gc_alloc", Type::gcref(), {size16, zero_mask, tag1});
    Value* val100 = b.build_iconst_i64(100);
    b.build_store(Type::i64(), obj1, 0, val100);

    // obj2 = brass_gc_alloc(16, 2, 1)
    Value* obj2 = b.build_call("brass_gc_alloc", Type::gcref(), {size16, mask2, tag1});
    Value* val200 = b.build_iconst_i64(200);
    b.build_store(Type::i64(), obj2, 0, val200);
    b.build_store(Type::gcref(), obj2, 8, obj1);

    // Safepoint!
    b.build_safepoint();

    // Read after moving collection
    Value* read_v2 = b.build_load(Type::i64(), obj2, 0);
    Value* read_obj1 = b.build_load(Type::gcref(), obj2, 8);
    Value* read_v1 = b.build_load(Type::i64(), read_obj1, 0);
    Value* sum = b.build_add(read_v1, read_v2);
    b.build_ret(sum);

    Interpreter interp(128 * 1024);
    interp.gc().set_stress_mode(true); // Force moving GC at allocation and safepoints

    RuntimeValue res = interp.run(*fn);
    CHECK_EQ(res.as_i64(), 300);
    CHECK(interp.gc().collection_count() >= 3); // 2 allocations + 1 safepoint
}

TEST_CASE("Interpreter - Speculation, Guard Failure, and State Map Capture") {
    Module mod("spec_mod");
    Builder b(mod);

    // func @speculative_add(%a: i32, %b: i32, %expected_type: i32) -> i32
    Function* fn = mod.create_function("speculative_add", Type::i32(), {Type::i32(), Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");

    Value* a = b.add_block_param(bb, Type::i32());
    Value* b_param = b.add_block_param(bb, Type::i32());
    Value* type_tag = b.add_block_param(bb, Type::i32());

    Value* expected = b.build_iconst_i32(1);
    Value* is_fast = b.build_eq(type_tag, expected);

    // Guard: if type_tag == 1, continue fast path; else deoptimize with state [a, b_param]
    b.build_guard(is_fast, "fallback_handler", {a, b_param});

    Value* fast_sum = b.build_add(a, b_param);
    b.build_ret(fast_sum);

    Interpreter interp;

    // 1. Guard passes (type_tag = 1)
    RuntimeValue fast_res = interp.run(*fn, {RuntimeValue::from_i32(20), RuntimeValue::from_i32(30), RuntimeValue::from_i32(1)});
    CHECK_EQ(fast_res.as_i32(), 50);
    CHECK(!interp.last_deopt().deoptimized);

    // 2. Guard fails (type_tag = 0) with custom deopt handler
    interp.set_deopt_handler([](Interpreter&, const DeoptResult& deopt) -> RuntimeValue {
        REQUIRE_EQ(deopt.exit_stub, "fallback_handler");
        REQUIRE_EQ(static_cast<int>(deopt.state_map.size()), 2);
        int32_t v1 = deopt.state_map[0].as_i32();
        int32_t v2 = deopt.state_map[1].as_i32();
        // Fallback computes sum + 1000
        return RuntimeValue::from_i32(v1 + v2 + 1000);
    });

    RuntimeValue slow_res = interp.run(*fn, {RuntimeValue::from_i32(20), RuntimeValue::from_i32(30), RuntimeValue::from_i32(0)});
    CHECK_EQ(slow_res.as_i32(), 1050);
    CHECK(interp.last_deopt().deoptimized);
}

TEST_CASE("Interpreter - Dynamic Patching (Const and Call)") {
    Module mod("patch_mod");
    Builder b(mod);

    // func @target_add(%x: i32) -> i32
    Function* fn_add = mod.create_function("target_add", Type::i32(), {Type::i32()});
    b.set_function(fn_add);
    BasicBlock* bb_add = b.append_block("entry");
    Value* x1 = b.add_block_param(bb_add, Type::i32());
    Value* c10 = b.build_iconst_i32(10);
    Value* r1 = b.build_add(x1, c10);
    b.build_ret(r1);

    // func @target_mul(%x: i32) -> i32
    Function* fn_mul = mod.create_function("target_mul", Type::i32(), {Type::i32()});
    b.set_function(fn_mul);
    BasicBlock* bb_mul = b.append_block("entry");
    Value* x2 = b.add_block_param(bb_mul, Type::i32());
    Value* c2 = b.build_iconst_i32(2);
    Value* r2 = b.build_mul(x2, c2);
    b.build_ret(r2);

    // func @caller_fn(%val: i32) -> i32
    Function* fn_caller = mod.create_function("caller_fn", Type::i32(), {Type::i32()});
    b.set_function(fn_caller);
    BasicBlock* bb_caller = b.append_block("entry");
    Value* val = b.add_block_param(bb_caller, Type::i32());

    // Patchable const: default initial value = 5
    Value* bias = b.build_patchable_const_i32("bias_slot_1", 5);
    Value* biased_val = b.build_add(val, bias);

    // Patchable call: default target = target_add
    Value* call_res = b.build_patchable_call("call_site_1", "target_add", Type::i32(), {biased_val});
    b.build_ret(call_res);

    Interpreter interp;
    interp.set_module(&mod);

    // Initial run: val = 10, bias = 5 -> biased = 15 -> target_add(15) -> 25
    RuntimeValue initial_res = interp.run(*fn_caller, {RuntimeValue::from_i32(10)});
    CHECK_EQ(initial_res.as_i32(), 25);

    // Dynamically patch const: bias = 20
    interp.patch_const("bias_slot_1", 20);
    // val = 10, bias = 20 -> biased = 30 -> target_add(30) -> 40
    RuntimeValue patched_const_res = interp.run(*fn_caller, {RuntimeValue::from_i32(10)});
    CHECK_EQ(patched_const_res.as_i32(), 40);

    // Dynamically patch call site: call_site_1 -> target_mul
    interp.patch_call("call_site_1", "target_mul");
    // val = 10, bias = 20 -> biased = 30 -> target_mul(30) -> 60
    RuntimeValue patched_call_res = interp.run(*fn_caller, {RuntimeValue::from_i32(10)});
    CHECK_EQ(patched_call_res.as_i32(), 60);
}
