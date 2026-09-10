#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/tiering.hpp>
#include <vector>

using namespace brass;
using namespace brass::runtime;

namespace {

std::unique_ptr<Module> build_fibonacci_module() {
    auto mod = std::make_unique<Module>("fib_module");
    Function* fn = mod->create_function("fib", Type::i64(), {Type::i64()});
    Builder b(*fn);
    BasicBlock* b_entry = b.append_block("entry");
    BasicBlock* b_base = b.append_block("base");
    BasicBlock* b_rec = b.append_block("rec");

    b.position_at_end(b_entry);
    Value* n = b.add_block_param(b_entry, Type::i64());
    Value* c2 = b.build_iconst_i64(2);
    Value* cmp = b.build_slt(n, c2);
    b.build_br_if(cmp, b_base, b_rec);

    b.position_at_end(b_base);
    b.build_ret(n);

    b.position_at_end(b_rec);
    Value* c1 = b.build_iconst_i64(1);
    Value* n1 = b.build_sub(n, c1);
    Value* f1 = b.build_call("fib", Type::i64(), {n1});
    Value* n2 = b.build_sub(n, c2);
    Value* f2 = b.build_call("fib", Type::i64(), {n2});
    Value* sum = b.build_add(f1, f2);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_loop_sum_module() {
    auto mod = std::make_unique<Module>("loop_sum_mod");
    Function* fn = mod->create_function("loop_sum", Type::i64(), {Type::i64()});
    Builder b(*fn);
    BasicBlock* b_entry = b.append_block("entry");
    BasicBlock* b_loop = b.append_block("loop");
    BasicBlock* b_body = b.append_block("body");
    BasicBlock* b_exit = b.append_block("exit");

    b.position_at_end(b_entry);
    Value* n = b.add_block_param(b_entry, Type::i64());
    Value* init_i = b.build_iconst_i64(0);
    Value* init_acc = b.build_iconst_i64(0);
    b.build_br(b_loop, {init_i, init_acc});

    b.position_at_end(b_loop);
    Value* cur_i = b.add_block_param(b_loop, Type::i64());
    Value* cur_acc = b.add_block_param(b_loop, Type::i64());
    Value* cond = b.build_slt(cur_i, n);
    b.build_br_if(cond, b_body, b_exit);

    b.position_at_end(b_body);
    Value* c1 = b.build_iconst_i64(1);
    Value* next_i = b.build_add(cur_i, c1);
    Value* next_acc = b.build_add(cur_acc, cur_i);
    b.build_br(b_loop, {next_i, next_acc});

    b.position_at_end(b_exit);
    b.build_ret(cur_acc);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_float_poly_module() {
    auto mod = std::make_unique<Module>("float_poly_mod");
    Function* fn = mod->create_function("poly", Type::f64(), {Type::f64()});
    Builder b(*fn);
    BasicBlock* b_entry = b.append_block("entry");
    b.position_at_end(b_entry);
    Value* x = b.add_block_param(b_entry, Type::f64());
    // 3.5 * x^2 + 2.0 * x - 1.5
    Value* x2 = b.build_mul(x, x);
    Value* c35 = b.build_fconst_f64(3.5);
    Value* t1 = b.build_mul(x2, c35);
    Value* c20 = b.build_fconst_f64(2.0);
    Value* t2 = b.build_mul(x, c20);
    Value* sum1 = b.build_add(t1, t2);
    Value* c15 = b.build_fconst_f64(1.5);
    Value* res = b.build_sub(sum1, c15);
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

TEST_CASE("Async JIT Differential - Background vs Synchronous Recursive Fibonacci") {
    auto mod = build_fibonacci_module();

    // 1. Synchronous verification baseline
    test::assert_diff(*mod, "fib", {RuntimeValue::from_i64(10)});
    test::assert_diff(*mod, "fib", {RuntimeValue::from_i64(12)});

    // 2. Background JIT compiler execution
    BackgroundCompiler compiler(2);
    FunctionHandle* handle = FunctionDispatchTable::instance().get_or_create("fib", mod->get_function("fib"));
    compiler.enqueue("fib", *mod, handle, CompilePriority::Normal);

    Interpreter interp;
    // Call during background compilation (begins in interpreter)
    RuntimeValue res_during = handle->call(interp, {RuntimeValue::from_i64(10)});
    CHECK_EQ(res_during.as_i64(), 55);

    compiler.wait_idle();
    CHECK(handle->has_native_entry());

    // Call after background compilation (enters native Tier-2 code)
    RuntimeValue res_native = handle->call(interp, {RuntimeValue::from_i64(10)});
    CHECK_EQ(res_native.as_i64(), 55);

    RuntimeValue res_native14 = handle->call(interp, {RuntimeValue::from_i64(14)});
    CHECK_EQ(res_native14.as_i64(), 377);

    // Bit-exact equivalence
    CHECK_EQ(res_during.raw_bits(), res_native.raw_bits());
}

TEST_CASE("Async JIT Differential - Background vs Synchronous Iterative Loop Sum") {
    auto mod = build_loop_sum_module();

    // 1. Synchronous verification baseline
    test::assert_diff(*mod, "loop_sum", {RuntimeValue::from_i64(100)});

    // 2. Background compilation
    BackgroundCompiler compiler(2);
    FunctionHandle* handle = FunctionDispatchTable::instance().get_or_create("loop_sum", mod->get_function("loop_sum"));
    compiler.enqueue("loop_sum", *mod, handle, CompilePriority::High);

    Interpreter interp;
    RuntimeValue res_early = handle->call(interp, {RuntimeValue::from_i64(50)});
    CHECK_EQ(res_early.as_i64(), 1225);

    compiler.wait_idle();
    CHECK(handle->has_native_entry());

    RuntimeValue res_late = handle->call(interp, {RuntimeValue::from_i64(100)});
    CHECK_EQ(res_late.as_i64(), 4950);
}

TEST_CASE("Async JIT Differential - Background vs Synchronous Floating Point Math") {
    auto mod = build_float_poly_module();

    test::assert_diff(*mod, "poly", {RuntimeValue::from_f64(2.5)});
    test::assert_diff(*mod, "poly", {RuntimeValue::from_f64(-1.2)});

    BackgroundCompiler compiler(2);
    FunctionHandle* handle = FunctionDispatchTable::instance().get_or_create("poly", mod->get_function("poly"));
    compiler.enqueue("poly", *mod, handle, CompilePriority::Normal);

    compiler.wait_idle();
    CHECK(handle->has_native_entry());

    Interpreter interp;
    RuntimeValue res = handle->call(interp, {RuntimeValue::from_f64(3.0)});
    // 3.5 * 9 + 2.0 * 3 - 1.5 = 31.5 + 6.0 - 1.5 = 36.0
    CHECK_EQ(res.as_f64(), 36.0);
}
