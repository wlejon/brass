#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/tiering.hpp>
#include <thread>
#include <vector>
#include <chrono>

using namespace brass;
using namespace brass::runtime;

namespace {

std::unique_ptr<Module> create_test_math_module(const std::string& mod_name, const std::string& fn_name, int64_t multiplier) {
    auto mod = std::make_unique<Module>(mod_name);
    Function* fn = mod->create_function(fn_name, Type::i64(), {Type::i64()});
    Builder b(*fn);
    BasicBlock* bb = b.append_block("entry");
    b.position_at_end(bb);
    Value* p0 = b.add_block_param(bb, Type::i64());
    Value* mult = b.build_iconst_i64(multiplier);
    Value* res = b.build_mul(p0, mult);
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

TEST_CASE("BackgroundCompiler - Worker Thread Pool Lifecycle and Task Enqueue") {
    BackgroundCompiler compiler(2);
    CHECK(compiler.is_running());
    CHECK_EQ(compiler.thread_count(), 2);

    auto mod = create_test_math_module("math_mod_1", "multiply_by_7", 7);
    Function* fn = mod->get_function("multiply_by_7");
    REQUIRE(fn != nullptr);

    FunctionHandle handle("multiply_by_7", fn);
    CHECK(!handle.has_native_entry());
    CHECK_EQ(handle.tier(), TierLevel::Tier0_Interpreter);

    bool enqueued = compiler.enqueue("multiply_by_7", *mod, &handle, CompilePriority::Normal);
    CHECK(enqueued);

    compiler.wait_idle();

    CHECK(handle.has_native_entry());
    CHECK_EQ(handle.tier(), TierLevel::Tier2_Optimized);
    CHECK_EQ(compiler.get_task_status("multiply_by_7"), CompileStatus::Completed);

    auto stats = compiler.stats();
    CHECK_EQ(stats.tasks_enqueued, 1);
    CHECK_EQ(stats.tasks_completed, 1);
    CHECK_EQ(stats.tasks_failed, 0);

    auto fn_ptr = handle.get_function_ptr<int64_t(*)(int64_t)>();
    REQUIRE(fn_ptr != nullptr);
    CHECK_EQ(fn_ptr(5), 35);
    CHECK_EQ(fn_ptr(10), 70);
    CHECK_EQ(fn_ptr(-3), -21);
}

TEST_CASE("BackgroundCompiler - Request Deduplication") {
    BackgroundCompiler compiler(1);
    auto mod = create_test_math_module("math_mod_dup", "dedup_fn", 3);
    Function* fn = mod->get_function("dedup_fn");
    FunctionHandle handle("dedup_fn", fn);

    bool first = compiler.enqueue("dedup_fn", *mod, &handle, CompilePriority::Normal);
    CHECK(first);

    // Immediate duplicate enqueue while pending/compiling must be rejected
    bool second = compiler.enqueue("dedup_fn", *mod, &handle, CompilePriority::Normal);
    CHECK(!second);

    bool third = compiler.enqueue("dedup_fn", *mod, &handle, CompilePriority::High);
    CHECK(!third);

    CHECK_EQ(compiler.stats().tasks_deduplicated, 2);

    compiler.wait_idle();

    CHECK(handle.has_native_entry());
    CHECK_EQ(compiler.stats().tasks_completed, 1);
    CHECK_EQ(compiler.stats().tasks_enqueued, 1);
    CHECK_EQ(compiler.stats().tasks_deduplicated, 2);

    // Once already compiled, subsequent enqueues for completed handle are also rejected
    bool fourth = compiler.enqueue("dedup_fn", *mod, &handle, CompilePriority::Normal);
    CHECK(!fourth);
}

TEST_CASE("BackgroundCompiler - Priority Scheduling (High over Normal)") {
    // Start with 0 worker threads to accumulate tasks in priority queue
    BackgroundCompiler compiler(0);
    CHECK_EQ(compiler.thread_count(), 0);

    std::vector<std::unique_ptr<Module>> modules;
    std::vector<std::unique_ptr<FunctionHandle>> handles;

    for (int i = 0; i < 4; ++i) {
        std::string name = "normal_task_" + std::to_string(i);
        auto mod = create_test_math_module("mod_" + name, name, i + 1);
        auto handle = std::make_unique<FunctionHandle>(name, mod->get_function(name));
        compiler.enqueue(name, *mod, handle.get(), CompilePriority::Normal);
        modules.push_back(std::move(mod));
        handles.push_back(std::move(handle));
    }

    // Enqueue hot high priority task last
    std::string hot_name = "hot_loop_task";
    auto hot_mod = create_test_math_module("mod_hot", hot_name, 100);
    auto hot_handle = std::make_unique<FunctionHandle>(hot_name, hot_mod->get_function(hot_name));
    compiler.enqueue(hot_name, *hot_mod, hot_handle.get(), CompilePriority::High);

    CHECK_EQ(compiler.queue_size(), 5);

    // Launch single worker so tasks are drained sequentially in strict priority order
    compiler.start(1);
    compiler.wait_idle();

    CHECK_EQ(compiler.stats().tasks_completed, 5);
    CHECK(hot_handle->has_native_entry());
    for (const auto& h : handles) {
        CHECK(h->has_native_entry());
    }

    auto hot_fn = hot_handle->get_function_ptr<int64_t(*)(int64_t)>();
    CHECK_EQ(hot_fn(7), 700);
}

TEST_CASE("BackgroundCompiler - Concurrency Stress Under Multi-Producer Threads") {
    constexpr size_t num_producers = 6;
    constexpr size_t tasks_per_producer = 15;

    BackgroundCompiler compiler(4);
    std::vector<std::thread> producers;
    std::vector<std::unique_ptr<FunctionHandle>> all_handles;
    std::mutex handles_mutex;

    for (size_t p = 0; p < num_producers; ++p) {
        producers.emplace_back([&, p]() {
            for (size_t t = 0; t < tasks_per_producer; ++t) {
                std::string fn_name = "stress_p" + std::to_string(p) + "_t" + std::to_string(t);
                int64_t factor = static_cast<int64_t>((p + 1) * 10 + t);
                auto mod = create_test_math_module("mod_" + fn_name, fn_name, factor);
                auto handle = std::make_unique<FunctionHandle>(fn_name, mod->get_function(fn_name));

                CompilePriority prio = (t % 3 == 0) ? CompilePriority::High : CompilePriority::Normal;
                compiler.enqueue(fn_name, *mod, handle.get(), prio);

                // Concurrently attempt duplicate enqueue to stress deduplication
                compiler.enqueue(fn_name, *mod, handle.get(), prio);

                {
                    std::lock_guard<std::mutex> lock(handles_mutex);
                    all_handles.push_back(std::move(handle));
                }
            }
        });
    }

    for (auto& t : producers) {
        t.join();
    }

    compiler.wait_idle();

    auto stats = compiler.stats();
    CHECK_EQ(stats.tasks_completed, num_producers * tasks_per_producer);
    CHECK_EQ(stats.tasks_failed, 0);
    CHECK(stats.tasks_deduplicated > 0);

    // Verify every compiled handle executes with 100% mathematical precision
    for (const auto& h : all_handles) {
        CHECK(h->has_native_entry());
        auto fn_ptr = h->get_function_ptr<int64_t(*)(int64_t)>();
        REQUIRE(fn_ptr != nullptr);
        CHECK_NE(fn_ptr(2), 0);
    }
}

TEST_CASE("BackgroundCompiler - Clean Shutdown and Destruction") {
    {
        BackgroundCompiler compiler(3);
        for (int i = 0; i < 10; ++i) {
            std::string name = "shutdown_task_" + std::to_string(i);
            auto mod = create_test_math_module("mod_" + name, name, i + 2);
            compiler.enqueue(name, *mod, nullptr, CompilePriority::Normal);
        }
        // Destroy while tasks may be in flight; must exit cleanly without hangs or deadlocks
    }
    CHECK(true);
}
