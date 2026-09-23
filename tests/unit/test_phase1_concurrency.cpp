#include "test_framework.hpp"
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/gc/tlab.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/runtime/patcher.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/mir/builder.hpp>
#include <thread>
#include <vector>
#include <atomic>
#include <set>

using namespace brass;
using namespace brass::runtime;

TEST_CASE("Phase 1 - Interpreter Float Conversions and Mixed Arithmetic") {
    // 1. as_f64() and as_f32() on integer kinds
    RuntimeValue v_i32_one = RuntimeValue::from_i32(1);
    CHECK_EQ(v_i32_one.as_f64(), 1.0);
    CHECK_EQ(v_i32_one.as_f32(), 1.0f);

    RuntimeValue v_i64_neg = RuntimeValue::from_i64(-42);
    CHECK_EQ(v_i64_neg.as_f64(), -42.0);
    CHECK_EQ(v_i64_neg.as_f32(), -42.0f);

    // 2. Float + Integer in val_add
    RuntimeValue v_f64 = RuntimeValue::from_f64(2.5);
    RuntimeValue v_add1 = val_add(v_f64, v_i32_one);
    CHECK_EQ(v_add1.as_f64(), 3.5);

    RuntimeValue v_f32 = RuntimeValue::from_f32(1.25f);
    RuntimeValue v_add2 = val_add(v_i32_one, v_f32);
    CHECK_EQ(v_add2.as_f32(), 2.25f);

    // 3. Mixed integer arithmetic with sign extension
    RuntimeValue neg_i32 = RuntimeValue::from_i32(-5);
    RuntimeValue pos_i64 = RuntimeValue::from_i64(10);

    // Add: -5 + 10 = 5
    RuntimeValue res_add = val_add(neg_i32, pos_i64);
    CHECK_EQ(res_add.as_i64(), 5);

    // Sub: 10 - (-5) = 15; -5 - 10 = -15
    RuntimeValue res_sub1 = val_sub(pos_i64, neg_i32);
    CHECK_EQ(res_sub1.as_i64(), 15);
    RuntimeValue res_sub2 = val_sub(neg_i32, pos_i64);
    CHECK_EQ(res_sub2.as_i64(), -15);

    // Mul: -5 * 10 = -50
    RuntimeValue res_mul = val_mul(neg_i32, pos_i64);
    CHECK_EQ(res_mul.as_i64(), -50);

    // Sdiv: -20 / 4 = -5
    RuntimeValue neg20_i64 = RuntimeValue::from_i64(-20);
    RuntimeValue pos4_i32 = RuntimeValue::from_i32(4);
    RuntimeValue res_div = val_sdiv(neg20_i64, pos4_i32);
    CHECK_EQ(res_div.as_i64(), -5);

    // Smod: -21 % 4 = -1
    RuntimeValue neg21_i64 = RuntimeValue::from_i64(-21);
    RuntimeValue res_mod = val_smod(neg21_i64, pos4_i32);
    CHECK_EQ(res_mod.as_i64(), -1);
}

TEST_CASE("Phase 1 - Dynamic Patching 0xE9 Near Jump and Cache Line Safety") {
    // 1. Verify 0xE9 (near JMP) patch layout
    alignas(64) uint8_t jmp_code[16] = {};
    jmp_code[0] = 0xE9; // JMP rel32 opcode
    *reinterpret_cast<int32_t*>(&jmp_code[1]) = 0;

    void* target_dest = &jmp_code[10];
    // x64 encoding stated explicitly: brass_patch_call patches host code,
    // and on an AArch64 host these bytes are not a branch.
    bool patched = patch_call_site(CodeArch::X64, &jmp_code[0], target_dest);
    CHECK(patched);
    // Opcode must remain 0xE9, displacement must be written to bytes 1..4
    CHECK_EQ(jmp_code[0], 0xE9);
    int32_t disp = *reinterpret_cast<int32_t*>(&jmp_code[1]);
    intptr_t next_ip = reinterpret_cast<intptr_t>(&jmp_code[5]);
    CHECK_EQ(reinterpret_cast<intptr_t>(target_dest) - next_ip, disp);

    // 2. Verify cache line safety check
    // An address at byte 62 of a 64-byte aligned chunk crosses into the next cache line
    alignas(64) uint8_t buffer[128] = {};
    void* crossing_addr = &buffer[62];
    CHECK(!is_cache_line_safe(crossing_addr, sizeof(int32_t)));
    CHECK(!brass_patch_const32(crossing_addr, 42));
    CHECK(!brass_patch_const64(crossing_addr, 42));
}

TEST_CASE("Phase 1 - MultiTierPipeline Concurrency and Deadlock Prevention") {
    Module mod("concurrent_tiering");
    Function* fn = mod.create_function("fn_concurrent", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* c1 = b.build_iconst_i64(1);
    Value* res = b.build_add(x, c1);
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();

    TieringConfig config;
    config.invocation_tier1_threshold = 5;
    config.enable_background_compile = false;

    auto& pipeline = MultiTierPipeline::instance();
    pipeline.initialize(config);
    pipeline.reset_stats();

    TieringRegistry::instance().set_active_module(&mod);
    auto* handle = FunctionDispatchTable::instance().get_or_create("fn_concurrent", fn);
    handle->set_tier(TierLevel::Tier0_Interpreter);

    // Invoke concurrently from multiple threads without deadlock
    constexpr int NUM_THREADS = 4;
    constexpr int INVS_PER_THREAD = 20;
    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&pipeline]() {
            for (int i = 0; i < INVS_PER_THREAD; ++i) {
                pipeline.on_invocation("fn_concurrent");
            }
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    MultiTierStats stats = pipeline.stats();
    CHECK(stats.tier0_invocations >= 5);
    CHECK(stats.tier1_compilations >= 1);
    CHECK(handle->has_native_entry());
    CHECK_EQ(handle->tier(), TierLevel::Tier1_Baseline);

    pipeline.shutdown();
}

TEST_CASE("Phase 1 - Thread-Safe TLAB Concurrent Refills") {
    HostGC gc(1024 * 1024); // 1 MB semispace
    constexpr int NUM_THREADS = 8;
    constexpr int ALLOCS_PER_THREAD = 50;

    struct BufferRange {
        uintptr_t start;
        uintptr_t end;
    };
    std::vector<std::vector<BufferRange>> thread_buffers(NUM_THREADS);

    std::vector<std::thread> threads;
    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&gc, &thread_buffers, t]() {
            ThreadLocalAllocBuffer tlab;
            gc.register_tlab(&tlab);
            for (int i = 0; i < ALLOCS_PER_THREAD; ++i) {
                uintptr_t top = 0, end = 0;
                if (gc.allocate_tlab(64, 256, top, end)) {
                    thread_buffers[t].push_back({top, end});
                    gc.retire_tlab(end, end);
                }
            }
            gc.unregister_tlab(&tlab);
        });
    }
    for (auto& t : threads) {
        t.join();
    }

    // Verify no two concurrently active buffer allocations overlapped
    std::vector<BufferRange> all_ranges;
    for (const auto& tb : thread_buffers) {
        all_ranges.insert(all_ranges.end(), tb.begin(), tb.end());
    }
    std::sort(all_ranges.begin(), all_ranges.end(), [](const BufferRange& a, const BufferRange& b) {
        return a.start < b.start;
    });

    for (size_t i = 1; i < all_ranges.size(); ++i) {
        CHECK(all_ranges[i].start >= all_ranges[i - 1].end);
    }
}

TEST_CASE("Phase 1 - HostGC Stack Root Capture During Heap Exhaustion") {
    // Small semispace (4 KB) so we quickly trigger collection on allocation
    HostGC gc(4096);
    CHECK_EQ(gc.collection_count(), 0);

    // Register a root slot
    uintptr_t root_obj = gc.allocate(32, 0, 1);
    gc.write_field(root_obj, 0, 12345);
    gc.register_root(&root_obj);

    // Allocate until heap exhausts and triggers collect(caller_rbp, caller_ip)
    for (int i = 0; i < 200; ++i) {
        gc.allocate(32, 0, 2);
    }

    CHECK(gc.collection_count() > 0);
    // Root must have been relocated and not poisoned
    CHECK(gc.is_valid_object(root_obj));
    CHECK_EQ(gc.read_field(root_obj, 0), 12345);
    gc.unregister_root(&root_obj);
}

TEST_CASE("Phase 1 - ParallelRuntime Algebraic Reduction Identity and Re-entrancy") {
    // 1. Min reduction with positive values must NOT be seeded with 0
    const size_t N = 1000;
    std::vector<int64_t> pos_arr(N, 100);
    pos_arr[42] = 50; // Minimum is +50

    struct Context {
        int64_t* arr;
    } ctx{pos_arr.data()};

    int64_t min_result = 0; // If seeded with 0, min(0, 50) would incorrectly return 0
    brass_parallel_for(
        N,
        32,
        [](uint64_t start, uint64_t end, void* context) {
            auto* c = static_cast<Context*>(context);
            int64_t local_min = std::numeric_limits<int64_t>::max();
            for (uint64_t i = start; i < end; ++i) {
                local_min = std::min(local_min, c->arr[i]);
            }
            brass_parallel_reduce_i64(local_min);
        },
        &ctx,
        ReductionKind::MinI64,
        &min_result
    );
    CHECK_EQ(min_result, 50);

    // 2. Nested parallel_for inside kernel
    std::atomic<int64_t> nested_total{0};
    brass_parallel_for(
        10,
        1,
        [](uint64_t outer_start, uint64_t outer_end, void* context) {
            auto* total = static_cast<std::atomic<int64_t>*>(context);
            for (uint64_t i = outer_start; i < outer_end; ++i) {
                int64_t inner_sum = 0;
                brass_parallel_for(
                    5,
                    1,
                    [](uint64_t inner_start, uint64_t inner_end, void* inner_ctx) {
                        auto* isum = static_cast<int64_t*>(inner_ctx);
                        for (uint64_t j = inner_start; j < inner_end; ++j) {
                            *isum += 1;
                        }
                    },
                    &inner_sum,
                    ReductionKind::None,
                    nullptr
                );
                total->fetch_add(inner_sum);
            }
        },
        &nested_total,
        ReductionKind::None,
        nullptr
    );
    CHECK_EQ(nested_total.load(), 50);

    // 3. Concurrent parallel_for from multiple threads
    std::vector<std::thread> p_threads;
    std::vector<int64_t> results(4, 0);
    for (int t = 0; t < 4; ++t) {
        p_threads.emplace_back([&results, t]() {
            std::atomic<int64_t> sum{0};
            brass_parallel_for(
                500,
                32,
                [](uint64_t s, uint64_t e, void* c) {
                    auto* s_ptr = static_cast<std::atomic<int64_t>*>(c);
                    s_ptr->fetch_add(static_cast<int64_t>(e - s), std::memory_order_relaxed);
                },
                &sum,
                ReductionKind::None,
                nullptr
            );
            results[t] = sum.load();
        });
    }
    for (auto& pt : p_threads) {
        pt.join();
    }
    for (int t = 0; t < 4; ++t) {
        CHECK_EQ(results[t], 500);
    }
}
