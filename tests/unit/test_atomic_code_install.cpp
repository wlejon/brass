#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/runtime/tiering.hpp>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>

using namespace brass;
using namespace brass::runtime;

namespace {

std::unique_ptr<Module> create_collatz_module(const std::string& name) {
    auto mod = std::make_unique<Module>("mod_" + name);
    // func collatz_step(%n: i64) -> i64
    //   %rem = umod %n, 2
    //   %is_zero = cmp.eq %rem, 0
    //   br %is_zero, b_even, b_odd
    // b_even:
    //   %half = udiv %n, 2
    //   ret %half
    // b_odd:
    //   %m = mul %n, 3
    //   %p1 = add %m, 1
    //   ret %p1
    Function* fn = mod->create_function(name, Type::i64(), {Type::i64()});
    Builder b(*fn);
    BasicBlock* b_entry = b.append_block("entry");
    BasicBlock* b_even = b.append_block("even");
    BasicBlock* b_odd = b.append_block("odd");

    b.position_at_end(b_entry);
    Value* p0 = b.add_block_param(b_entry, Type::i64());
    Value* c2 = b.build_iconst_i64(2);
    Value* rem = b.build_umod(p0, c2);
    Value* c0 = b.build_iconst_i64(0);
    Value* is_even = b.build_eq(rem, c0);
    b.build_br_if(is_even, b_even, b_odd);

    b.position_at_end(b_even);
    Value* half = b.build_udiv(p0, c2);
    b.build_ret(half);

    b.position_at_end(b_odd);
    Value* c3 = b.build_iconst_i64(3);
    Value* mul3 = b.build_mul(p0, c3);
    Value* c1 = b.build_iconst_i64(1);
    Value* add1 = b.build_add(mul3, c1);
    b.build_ret(add1);

    fn->rebuild_cfg_predecessors();
    return mod;
}

int64_t oracle_collatz_step(int64_t n) {
    return (n % 2 == 0) ? (n / 2) : (n * 3 + 1);
}

} // namespace

TEST_CASE("Atomic Code Install - Seamless Entry Point Swapping") {
    auto mod = create_collatz_module("collatz_step_single");
    Function* fn = mod->get_function("collatz_step_single");
    REQUIRE(fn != nullptr);

    FunctionHandle handle("collatz_step_single", fn);
    CHECK(!handle.has_native_entry());
    CHECK_EQ(handle.native_entry(), nullptr);
    CHECK_EQ(handle.tier(), TierLevel::Tier0_Interpreter);

    std::atomic<bool> start_flag{false};
    std::atomic<bool> mutator_saw_native{false};
    std::atomic<uint64_t> total_invocations{0};
    std::atomic<bool> stop_mutator{false};

    // Mutator thread repeatedly calls function
    std::thread mutator([&]() {
        Interpreter interp;
        while (!start_flag.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }

        int64_t val = 27;
        while (!stop_mutator.load(std::memory_order_relaxed)) {
            RuntimeValue res = handle.call(interp, {RuntimeValue::from_i64(val)});
            int64_t expected = oracle_collatz_step(val);
            CHECK_EQ(res.as_i64(), expected);

            if (handle.has_native_entry()) {
                mutator_saw_native.store(true, std::memory_order_release);
            }

            val = (val % 1000) + 1;
            total_invocations.fetch_add(1, std::memory_order_relaxed);
        }
    });

    start_flag.store(true, std::memory_order_release);
    // Allow mutator to execute in interpreter first
    std::this_thread::sleep_for(std::chrono::milliseconds(5));

    // Install Tier-2 code concurrently
    CodeInstaller installer;
    CodeInstallResult res = installer.install_tier2(handle, *mod, "collatz_step_single");
    CHECK(res.success);
    CHECK(res.entry_point != nullptr);
    CHECK(handle.has_native_entry());
    CHECK_EQ(handle.tier(), TierLevel::Tier2_Optimized);

    // Wait until mutator observes native entry
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!mutator_saw_native.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::yield();
    }

    CHECK(mutator_saw_native.load());
    stop_mutator.store(true, std::memory_order_release);
    mutator.join();

    CHECK(total_invocations.load() > 100);
}

TEST_CASE("Atomic Code Install - Multi-Threaded Mutators Under Concurrent Compilation") {
    constexpr size_t num_mutators = 6;
    constexpr int64_t iterations_per_mutator = 500;

    auto mod = create_collatz_module("collatz_multi");
    Function* fn = mod->get_function("collatz_multi");
    REQUIRE(fn != nullptr);

    FunctionHandle handle("collatz_multi", fn);
    std::atomic<bool> ready{false};
    std::vector<std::thread> mutators;
    std::atomic<uint64_t> errors{0};

    for (size_t i = 0; i < num_mutators; ++i) {
        mutators.emplace_back([&, i]() {
            Interpreter interp;
            while (!ready.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            int64_t state = static_cast<int64_t>(i * 10 + 3);
            for (int64_t it = 0; it < iterations_per_mutator; ++it) {
                RuntimeValue res = handle.call(interp, {RuntimeValue::from_i64(state)});
                int64_t exp = oracle_collatz_step(state);
                if (res.as_i64() != exp) {
                    errors.fetch_add(1);
                }
                state = (state % 250) + 1;
            }
        });
    }

    // Launch background compilation on worker thread
    BackgroundCompiler compiler(2);
    compiler.enqueue("collatz_multi", *mod, &handle, CompilePriority::Normal);

    // Start all mutators simultaneously
    ready.store(true, std::memory_order_release);

    for (auto& t : mutators) {
        t.join();
    }

    compiler.wait_idle();

    CHECK_EQ(errors.load(), 0);
    CHECK(handle.has_native_entry());
    CHECK_EQ(handle.tier(), TierLevel::Tier2_Optimized);

    // Verify native call directly
    auto fn_ptr = handle.get_function_ptr<int64_t(*)(int64_t)>();
    REQUIRE(fn_ptr != nullptr);
    CHECK_EQ(fn_ptr(10), 5);
    CHECK_EQ(fn_ptr(7), 22);
}

TEST_CASE("Atomic Code Install - Dispatch Table Registration and Lookup") {
    FunctionDispatchTable& table = FunctionDispatchTable::instance();
    table.clear();

    CHECK_EQ(table.size(), 0);
    CHECK(!table.has("nonexistent_fn"));
    CHECK_EQ(table.find("nonexistent_fn"), nullptr);

    FunctionHandle* h1 = table.get_or_create("fn_alpha");
    REQUIRE(h1 != nullptr);
    CHECK_EQ(h1->name(), "fn_alpha");
    CHECK_EQ(table.size(), 1);
    CHECK(table.has("fn_alpha"));

    FunctionHandle* h2 = table.get_or_create("fn_alpha");
    CHECK_EQ(h1, h2);

    FunctionHandle* h3 = table.find("fn_alpha");
    CHECK_EQ(h1, h3);

    table.clear();
    CHECK_EQ(table.size(), 0);
    CHECK(!table.has("fn_alpha"));
}
