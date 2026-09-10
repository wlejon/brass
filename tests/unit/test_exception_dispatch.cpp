#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/mini_cheney.hpp>

using namespace brass;
using namespace brass::runtime;
using namespace brass::codegen;

TEST_CASE("Exception Dispatch - Thread Local APIs") {
    brass_clear_current_exception();
    CHECK(!brass_has_current_exception());

    HostValue v = HostValue::from_i32(42);
    brass_set_current_exception(v);
    CHECK(brass_has_current_exception());
    CHECK_EQ(brass_get_current_exception().as_i32(), 42);

    HostValue taken = brass_take_current_exception();
    CHECK_EQ(taken.as_i32(), 42);
    CHECK(!brass_has_current_exception());
}

TEST_CASE("Exception Dispatch - Unhandled Exception Throws C++ Envelope") {
    // When brass_throw is called from outside any protected JIT landing pad,
    // it must throw a C++ BrassException envelope
    HostValue exc_val = HostValue::from_i32(99);
    bool caught = false;
    try {
        brass_throw(exc_val);
    } catch (const BrassException& e) {
        caught = true;
        CHECK_EQ(e.value().as_i32(), 99);
    }
    CHECK(caught);
}

TEST_CASE("Exception Dispatch - JIT Throw and Catch") {
    Module mod("jit_eh_test");
    Builder b(mod);

    // callee: func @fail(%val: i64) -> i64
    Function* callee = mod.create_function("fail", Type::i64(), {Type::i64()});
    b.set_function(callee);
    BasicBlock* c_bb = b.append_block("entry");
    b.position_at_end(c_bb);
    Value* throw_val = b.add_block_param(c_bb, Type::i64());
    b.build_throw(throw_val);
    callee->rebuild_cfg_predecessors();

    // caller: func @try_catch(%x: i64) -> i64
    Function* caller = mod.create_function("try_catch", Type::i64(), {Type::i64()});
    b.set_function(caller);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* normal_bb = b.append_block("normal_bb");
    BasicBlock* unwind_bb = b.append_block("unwind_bb");

    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Instruction* inv = b.build_invoke("fail", Type::i64(), {x}, normal_bb, unwind_bb);

    b.position_at_end(normal_bb);
    b.build_ret(inv->result());

    b.position_at_end(unwind_bb);
    Value* caught = b.build_landing_pad(Type::i64());
    Value* one = b.build_iconst_i64(1);
    Value* res = b.build_add(caught, one);
    b.build_ret(res);
    caller->rebuild_cfg_predecessors();

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));

    // Invoke JIT caller: pass 41 -> callee throws 41 -> caught in unwind_bb -> 41 + 1 = 42
    typedef int64_t (*FnPtr)(int64_t);
    FnPtr fn = jit.get_function_ptr<FnPtr>("try_catch");
    REQUIRE(fn != nullptr);

    int64_t ret = fn(41);
    CHECK_EQ(ret, 42);
}

TEST_CASE("Exception Dispatch - JIT Nested Try Catch & Rethrow") {
    Module mod("jit_nested_eh");
    Builder b(mod);

    // callee: func @fail_nested(%code: i64) -> i64
    Function* callee = mod.create_function("fail_nested", Type::i64(), {Type::i64()});
    b.set_function(callee);
    BasicBlock* c_bb = b.append_block("entry");
    b.position_at_end(c_bb);
    Value* code = b.add_block_param(c_bb, Type::i64());
    b.build_throw(code);
    callee->rebuild_cfg_predecessors();

    // inner_wrapper: calls @fail_nested, if caught == 10 returns 100, else rethrows
    Function* inner = mod.create_function("inner", Type::i64(), {Type::i64()});
    b.set_function(inner);
    BasicBlock* in_entry = b.append_block("entry");
    BasicBlock* in_norm = b.append_block("in_norm");
    BasicBlock* in_unw = b.append_block("in_unw");
    BasicBlock* in_handle = b.append_block("in_handle");
    BasicBlock* in_rethrow = b.append_block("in_rethrow");

    b.position_at_end(in_entry);
    Value* in_arg = b.add_block_param(in_entry, Type::i64());
    Instruction* inv_in = b.build_invoke("fail_nested", Type::i64(), {in_arg}, in_norm, in_unw);

    b.position_at_end(in_norm);
    b.build_ret(inv_in->result());

    b.position_at_end(in_unw);
    Value* in_exc = b.build_landing_pad(Type::i64());
    Value* ten = b.build_iconst_i64(10);
    Value* is_ten = b.build_eq(in_exc, ten);
    b.build_br_if(is_ten, in_handle, in_rethrow);

    b.position_at_end(in_handle);
    b.build_ret(b.build_iconst_i64(100));

    b.position_at_end(in_rethrow);
    b.build_resume(in_exc);
    inner->rebuild_cfg_predecessors();

    // outer: calls @inner, if caught returns caught * 2
    Function* outer = mod.create_function("outer", Type::i64(), {Type::i64()});
    b.set_function(outer);
    BasicBlock* out_entry = b.append_block("entry");
    BasicBlock* out_norm = b.append_block("out_norm");
    BasicBlock* out_unw = b.append_block("out_unw");

    b.position_at_end(out_entry);
    Value* out_arg = b.add_block_param(out_entry, Type::i64());
    Instruction* inv_out = b.build_invoke("inner", Type::i64(), {out_arg}, out_norm, out_unw);

    b.position_at_end(out_norm);
    b.build_ret(inv_out->result());

    b.position_at_end(out_unw);
    Value* out_exc = b.build_landing_pad(Type::i64());
    Value* two = b.build_iconst_i64(2);
    Value* out_res = b.build_mul(out_exc, two);
    b.build_ret(out_res);
    outer->rebuild_cfg_predecessors();

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(mod));

    typedef int64_t (*FnPtr)(int64_t);
    FnPtr outer_fn = jit.get_function_ptr<FnPtr>("outer");
    REQUIRE(outer_fn != nullptr);

    // 1. in_arg = 10: inner catches and returns 100
    int64_t r1 = outer_fn(10);
    CHECK_EQ(r1, 100);

    // 2. in_arg = 20: inner rethrows to outer, outer catches and returns 20 * 2 = 40
    int64_t r2 = outer_fn(20);
    CHECK_EQ(r2, 40);
}
