#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/gc/heap.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <memory>

using namespace brass;

TEST_CASE("Coroutine Runtime - JIT Generator Execution") {
    Module mod("jit_gen_mod");
    Builder b(mod);

    // Generator counting sequence: 10, 20, 30
    Function* gen = mod.create_function("counter_gen", Type::i64(), {Type::gcref()});
    b.set_function(gen);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::gcref());

    Value* v1 = b.build_iconst_i64(10);
    b.build_coro_suspend(v1, 1, Type::i64());

    Value* v2 = b.build_add(v1, b.build_iconst_i64(10));
    b.build_coro_suspend(v2, 2, Type::i64());

    Value* v3 = b.build_add(v2, b.build_iconst_i64(10));
    b.build_ret(v3);

    CoroTransformPass pass;
    CHECK(pass.run_on_module(mod));

    DiagnosticReporter diag;
    CHECK(verify_module(mod, &diag));

    codegen::JitExecutionEngine jit;
    jit.compile_and_load(mod);

    void* fn_ptr = jit.get_symbol_address("counter_gen");
    CHECK(fn_ptr != nullptr);

    uintptr_t frame_addr = brass_coro_create(fn_ptr, 16, 0);
    CHECK(frame_addr != 0);

    // Yield 1: 10
    uint64_t y1 = brass_coro_resume(frame_addr, 0);
    CHECK(y1 == 10);
    CHECK(brass_coro_is_done(frame_addr) == 0);

    // Yield 2: 20
    uint64_t y2 = brass_coro_resume(frame_addr, 0);
    CHECK(y2 == 20);
    CHECK(brass_coro_is_done(frame_addr) == 0);

    // Final return: 30
    uint64_t y3 = brass_coro_resume(frame_addr, 0);
    CHECK(y3 == 30);
    CHECK(brass_coro_is_done(frame_addr) == 1);

    brass_coro_destroy(frame_addr);
}

namespace {

// Yields 10, then 30, then returns 50 — after CoroTransformPass, whose entry
// dispatches on the frame's i32 state_id.
std::unique_ptr<Module> build_fib_step_module(std::string_view fn_name) {
    auto mod = std::make_unique<Module>("fib_step_mod");
    Builder b(*mod);
    Function* fn = mod->create_function(fn_name, Type::i64(), {Type::gcref()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    b.add_block_param(entry, Type::gcref());

    Value* v1 = b.build_iconst_i64(10);
    b.build_coro_suspend(v1, 1, Type::i64());
    Value* v2 = b.build_add(v1, b.build_iconst_i64(20));
    b.build_coro_suspend(v2, 2, Type::i64());
    Value* v3 = b.build_add(v2, b.build_iconst_i64(20));
    b.build_ret(v3);

    CoroTransformPass pass;
    REQUIRE(pass.run_on_module(*mod));
    DiagnosticReporter diag;
    REQUIRE(verify_module(*mod, &diag));
    return mod;
}

void step_fib_in_fast_interpreter(FastInterpreter& interp, const Function& fn) {
    uintptr_t c_frame = brass_coro_create(nullptr, 16, 0);
    REQUIRE(c_frame != 0);
    auto* f_ptr = reinterpret_cast<runtime::BrassCoroFrame*>(c_frame);

    RuntimeValue o1 = interp.run(fn, {RuntimeValue::from_ptr(c_frame)});
    CHECK_EQ(o1.as_u64(), 10u);
    CHECK_EQ(f_ptr->is_done, 0u);
    RuntimeValue o2 = interp.run(fn, {RuntimeValue::from_ptr(c_frame)});
    CHECK_EQ(o2.as_u64(), 30u);
    CHECK_EQ(f_ptr->is_done, 0u);
    RuntimeValue o3 = interp.run(fn, {RuntimeValue::from_ptr(c_frame)});
    CHECK_EQ(o3.as_u64(), 50u);
    CHECK_EQ(f_ptr->is_done, 1u);

    brass_coro_destroy(c_frame);
}

} // namespace

TEST_CASE("Coroutine Runtime - Baseline JIT resumes a transformed coroutine on every frame") {
    // The baseline switch on the i32 state_id used to read the whole 8-byte
    // slot, so whether a resume reached its state depended on stack garbage:
    // runs repeated the first yield. Several frames make a lucky pass unlikely.
    auto mod = build_fib_step_module("fib_step_baseline");
    codegen::BaselineJitCompiler compiler(Target::host());
    auto compiled = compiler.compile_module(*mod);
    REQUIRE(compiled.size() == 1);
    void* entry = compiled[0].entry_point();
    REQUIRE(entry != nullptr);

    for (int round = 0; round < 4; ++round) {
        uintptr_t frame = brass_coro_create(entry, 16, 0);
        REQUIRE(frame != 0);
        CHECK_EQ(brass_coro_resume(frame, 0), 10u);
        CHECK_EQ(brass_coro_resume(frame, 0), 30u);
        CHECK_EQ(brass_coro_resume(frame, 0), 50u);
        CHECK_EQ(brass_coro_is_done(frame), 1u);
        brass_coro_destroy(frame);
    }
}

TEST_CASE("Coroutine Runtime - FastInterpreter ignores a destroyed module's dispatch handle") {
    // A baseline-compiled module registers its functions by name in the global
    // dispatch table. Once the module dies, a new Function allocated at the same
    // address must not match `handle->mir_function() == &fn`, or
    // FastInterpreter::run silently runs the dead module's native code.
    const char* name = "fib_step_aba";
    {
        auto dead = build_fib_step_module(name);
        codegen::BaselineJitCompiler compiler(Target::host());
        auto compiled = compiler.compile_module(*dead);
        auto* handle = runtime::FunctionDispatchTable::instance().find(name);
        REQUIRE(handle != nullptr);
        CHECK(handle->mir_function() == dead->get_function(name));
    }
    auto* handle = runtime::FunctionDispatchTable::instance().find(name);
    REQUIRE(handle != nullptr);
    CHECK(handle->mir_function() == nullptr);

    auto live = build_fib_step_module(name);
    FastInterpreter interp;
    step_fib_in_fast_interpreter(interp, *live->get_function(name));
}

TEST_CASE("Coroutine Runtime - Cheney Moving GC Active Frame Tracking") {
    // Test that suspended coroutine frames are tracked in g_active_coro_frames
    // and correctly visited during GC cycles
    runtime::BrassCoroFrame dummy_frame;
    dummy_frame.fn_ptr = nullptr;
    dummy_frame.state_id = 1;
    dummy_frame.is_done = 0;
    dummy_frame.slot_count = 2;
    dummy_frame.slots[0] = 0x12345678;

    runtime::register_active_coro_frame(&dummy_frame);

    size_t visited_count = 0;
    runtime::visit_active_coro_frames([&](uintptr_t* frame_ptr) {
        if (*frame_ptr == reinterpret_cast<uintptr_t>(&dummy_frame)) {
            visited_count++;
        }
    });
    CHECK(visited_count == 1);

    runtime::unregister_active_coro_frame(&dummy_frame);

    visited_count = 0;
    runtime::visit_active_coro_frames([&](uintptr_t* frame_ptr) {
        if (*frame_ptr == reinterpret_cast<uintptr_t>(&dummy_frame)) {
            visited_count++;
        }
    });
    CHECK(visited_count == 0);
}

TEST_CASE("Coroutine Runtime - Microtask Queue & Promises") {
    auto& mq = runtime::get_global_microtask_queue();
    (void)mq;

    runtime::Promise p;
    CHECK(p.state() == runtime::PromiseState::Pending);

    uint64_t received = 0;
    p.then([&](uint64_t val) {
        received = val;
    });

    p.fulfill(999);
    CHECK(p.state() == runtime::PromiseState::Fulfilled);
    CHECK(received == 999);
}
