#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/gc/mini_cheney.hpp>

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
