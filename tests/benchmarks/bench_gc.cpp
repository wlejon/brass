#include "bench_gc.hpp"
#include "bench_utils.hpp"
#include <brass/brass.hpp>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace brass;
using namespace brass::bench;
using namespace brass::codegen;

extern "C"
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
int64_t bench_leaf_gc_subroutine(int64_t x) {
    DoNotOptimize(x);
    int64_t res = (x * 1664525 + 1013904223) & 0x7FFFFFFF;
    DoNotOptimize(res);
    return res;
}

namespace {

// (a) Shadow-Stack Model: Live GC root bookkeeping on every call
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
int64_t shadow_stack_gc_runner(
    void* r0, void* r1, void* r2, void* r3,
    int64_t iters, ThreadShadowStack& ss
) {
    int64_t acc = 0;
    for (int64_t i = 0; i < iters; ++i) {
        ShadowStackFrame frame;
        frame.roots[0] = r0;
        frame.roots[1] = r1;
        frame.roots[2] = r2;
        frame.roots[3] = r3;
        ss.push(&frame, 4);
        ClobberMemory();

        int64_t val0 = *reinterpret_cast<volatile int64_t*>(frame.roots[0]);
        int64_t val1 = *reinterpret_cast<volatile int64_t*>(frame.roots[1]);
        int64_t val2 = *reinterpret_cast<volatile int64_t*>(frame.roots[2]);
        int64_t val3 = *reinterpret_cast<volatile int64_t*>(frame.roots[3]);

        int64_t arg = val0 + val1 + val2 + val3 + acc;
        int64_t sub_res = bench_leaf_gc_subroutine(arg);
        acc += sub_res;

        // In a moving collector with shadow stack, roots must be re-read after safepoint
        r0 = frame.roots[0];
        r1 = frame.roots[1];
        r2 = frame.roots[2];
        r3 = frame.roots[3];
        DoNotOptimize(r0);
        DoNotOptimize(r1);
        DoNotOptimize(r2);
        DoNotOptimize(r3);

        ss.pop();
        ClobberMemory();
    }
    return acc;
}

std::unique_ptr<Module> build_gc_benchmark_module() {
    auto mod = std::make_unique<Module>("bench_gc_model");
    mod->add_external_symbol("bench_leaf_gc_subroutine");
    Builder b(*mod);

    Function* fn = mod->create_function("brass_gc_caller", Type::i64(), {
        Type::gcref(), Type::gcref(), Type::gcref(), Type::gcref(), Type::i64()
    });
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* r0 = b.add_block_param(entry, Type::gcref());
    Value* r1 = b.add_block_param(entry, Type::gcref());
    Value* r2 = b.add_block_param(entry, Type::gcref());
    Value* r3 = b.add_block_param(entry, Type::gcref());
    Value* iters = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    b.build_br(loop_hdr, {zero, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, iters);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* v0 = b.build_load(Type::i64(), r0, 0);
    Value* v1 = b.build_load(Type::i64(), r1, 0);
    Value* v2 = b.build_load(Type::i64(), r2, 0);
    Value* v3 = b.build_load(Type::i64(), r3, 0);

    Value* sum4 = b.build_add(b.build_add(v0, v1), b.build_add(v2, v3));
    Value* arg = b.build_add(sum4, acc);

    // Call subroutine while holding 4 live gcrefs in registers/stack map
    Value* sub_res = b.build_call("bench_leaf_gc_subroutine", Type::i64(), {arg});
    Value* next_acc = b.build_add(acc, sub_res);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

namespace brass::bench {

void run_gc_benchmark(std::vector<BenchmarkResult>& results) {
    size_t iters = 500000;

    int64_t heap_obj0 = 10;
    int64_t heap_obj1 = 20;
    int64_t heap_obj2 = 30;
    int64_t heap_obj3 = 40;

    // (a) Shadow-Stack Model
    ThreadShadowStack ss;

    // (b) Brass Model (Live gcref registers & stack maps, zero shadow-stack pushing)
    auto mod = build_gc_benchmark_module();
    JitExecutionEngine jit;
    jit.register_external_symbol("bench_leaf_gc_subroutine", reinterpret_cast<void*>(&bench_leaf_gc_subroutine));
    jit.compile_and_load(*mod);
    auto gc_fn = jit.get_function_ptr<int64_t(*)(void*, void*, void*, void*, int64_t)>("brass_gc_caller");
    if (!gc_fn) {
        std::cerr << "FATAL: brass_gc_caller function pointer is null!\n";
        std::abort();
    }

    int64_t shadow_res = 0;
    int64_t brass_res = 0;

    std::vector<double> shadow_samples, brass_samples, speedup_samples;
    shadow_samples.reserve(DEFAULT_BENCH_REPETITIONS);
    brass_samples.reserve(DEFAULT_BENCH_REPETITIONS);
    speedup_samples.reserve(DEFAULT_BENCH_REPETITIONS);

    Stopwatch sw;
    for (size_t r = 0; r < DEFAULT_BENCH_REPETITIONS; ++r) {
        sw.start();
        shadow_res = shadow_stack_gc_runner(
            &heap_obj0, &heap_obj1, &heap_obj2, &heap_obj3,
            static_cast<int64_t>(iters), ss
        );
        DoNotOptimize(shadow_res);
        double s_ms = sw.stop_ms();

        sw.start();
        brass_res = gc_fn(
            &heap_obj0, &heap_obj1, &heap_obj2, &heap_obj3,
            static_cast<int64_t>(iters)
        );
        DoNotOptimize(brass_res);
        double b_ms = sw.stop_ms();

        shadow_samples.push_back(s_ms);
        brass_samples.push_back(b_ms);
        double sp = (b_ms > 0.0) ? (s_ms / b_ms) : 1.0;
        speedup_samples.push_back(sp);
    }

    TimingStats shadow_stats(std::move(shadow_samples));
    TimingStats brass_stats(std::move(brass_samples));
    TimingStats speedup_stats(std::move(speedup_samples));

    if (shadow_res != brass_res) {
        std::cerr << "FATAL: GC Model result mismatch: shadow=" << shadow_res << ", Brass=" << brass_res << "\n";
        std::abort();
    }
    double speedup = speedup_stats.median;
    bool passes_gc_bar = (speedup >= 1.25);

    BenchmarkReporter::print_gc_comparison(shadow_stats, brass_stats, speedup_stats, passes_gc_bar);

    BenchmarkResult res;
    res.key = "gc_model_speedup";
    res.name = "GC Stack-Map Model (vs Shadow-Stack)";
    res.iterations = iters;
    res.repetitions = DEFAULT_BENCH_REPETITIONS;
    res.native_ms = shadow_stats.median;
    res.native_min_ms = shadow_stats.min;
    res.native_max_ms = shadow_stats.max;
    res.native_scalar_ms = 0.0;
    res.brass_ms = brass_stats.median;
    res.brass_min_ms = brass_stats.min;
    res.brass_max_ms = brass_stats.max;
    res.ratio = speedup_stats.median;
    res.ratio_min = speedup_stats.min;
    res.ratio_max = speedup_stats.max;
    res.target_ratio = 1.25;
    res.passes_bar = passes_gc_bar;
    res.notes = ">= 1.25x speedup";
    results.push_back(res);
}

} // namespace brass::bench
