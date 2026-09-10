#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/coroutine.hpp>
#include <vector>

using namespace brass;

TEST_CASE("Coroutine Differential - Interpreter vs JIT") {
    Module mod("coro_diff_mod");
    Builder b(mod);

    // Coroutine accumulating sum across suspends:
    // yields 1 (sum=1)
    // receives arg + 10 (sum = 1 + 10 = 11), yields 11
    // receives arg + 20 (sum = 11 + 20 = 31), yields 31
    // returns sum
    Function* fn = mod.create_function("diff_coro", Type::i64(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::gcref());

    Value* sum0 = b.build_iconst_i64(1);
    Value* r1 = b.build_coro_suspend(sum0, 1, Type::i64());

    Value* sum1 = b.build_add(sum0, r1);
    Value* r2 = b.build_coro_suspend(sum1, 2, Type::i64());

    Value* sum2 = b.build_add(sum1, r2);
    b.build_ret(sum2);

    CoroTransformPass pass;
    CHECK(pass.run_on_module(mod));

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));

    // 1. Run in JIT
    codegen::JitExecutionEngine jit;
    jit.compile_and_load(mod);
    void* jit_fn = jit.get_symbol_address("diff_coro");
    CHECK(jit_fn != nullptr);

    uintptr_t jit_frame = brass_coro_create(jit_fn, 16, 0);
    CHECK(jit_frame != 0);

    uint64_t jit_y1 = brass_coro_resume(jit_frame, 0);
    uint64_t jit_y2 = brass_coro_resume(jit_frame, 10);
    uint64_t jit_y3 = brass_coro_resume(jit_frame, 20);
    uint32_t jit_done = brass_coro_is_done(jit_frame);
    brass_coro_destroy(jit_frame);

    // 2. Run in Interpreter
    Interpreter interp;
    Function* interp_fn = mod.get_function("diff_coro");
    CHECK(interp_fn != nullptr);

    uintptr_t interp_frame = brass_coro_create(nullptr, 16, 0);
    CHECK(interp_frame != 0);
    auto* f_ptr = reinterpret_cast<runtime::BrassCoroFrame*>(interp_frame);

    RuntimeValue iv1 = interp.run(*interp_fn, {RuntimeValue::from_ptr(interp_frame)});
    uint64_t interp_y1 = static_cast<uint64_t>(iv1.as_i64());

    f_ptr->resume_arg = 10;
    RuntimeValue iv2 = interp.run(*interp_fn, {RuntimeValue::from_ptr(interp_frame)});
    uint64_t interp_y2 = static_cast<uint64_t>(iv2.as_i64());

    f_ptr->resume_arg = 20;
    RuntimeValue iv3 = interp.run(*interp_fn, {RuntimeValue::from_ptr(interp_frame)});
    uint64_t interp_y3 = static_cast<uint64_t>(iv3.as_i64());

    uint32_t interp_done = f_ptr->is_done;
    brass_coro_destroy(interp_frame);

    // 3. Differential assertions
    CHECK(jit_y1 == 1);
    CHECK(interp_y1 == jit_y1);

    CHECK(jit_y2 == 11);
    CHECK(interp_y2 == jit_y2);

    CHECK(jit_y3 == 31);
    CHECK(interp_y3 == jit_y3);

    CHECK(jit_done == 1);
    CHECK(interp_done == jit_done);
}
