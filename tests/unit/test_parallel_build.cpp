// A build's per-function work runs on many threads (pass_manager.hpp,
// "Threads"; ModuleCompiler::set_threads) and must not show it: the optimized
// module and the object's bytes are the same at 1, 2 and many threads, run
// after run, and the same whether a concurrent run of steps is taken function
// by function or (with hooks) step by step.
#include "test_framework.hpp"
#include <brass/core/parallel_for.hpp>
#include <brass/fuzz/program_generator.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/pass_manager.hpp>
#include <brass/mir/pass_pipeline.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/object_writer.hpp>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brass;

namespace {

// Many generated programs (loops, branches, memory, GC allocation, calls to
// small helpers) in one module: thousands of blocks across hundreds of
// functions, enough for every function stage to fan out.
std::unique_ptr<Module> make_module(size_t programs) {
    auto mod = std::make_unique<Module>("parallel_build");
    fuzz::ProgramGeneratorOptions o;
    o.max_depth = 4;
    o.max_statements = 60;
    fuzz::ProgramGenerator gen(o);
    for (size_t k = 0; k < programs; ++k) {
        gen.generate(*mod, "pb" + std::to_string(k), 0x9e3779b97f4a7c15ull * (k + 1));
    }
    DiagnosticReporter diag;
    if (!verify_module(*mod, &diag)) throw std::runtime_error(diag.format_all());
    return mod;
}

// The production pipeline with the module steps (inlining) and the serial
// step (auto-parallelization, which adds functions) switched on, so the run
// has every kind of stage.
Pipeline pipeline() {
    PassPipelineOptions o = production_pass_pipeline_options();
    o.enable_inlining = true;
    o.loop.enable_parallel_loops = true;
    o.loop.parallel_threshold = 64;
    return pass_pipeline(o);
}

struct Built {
    std::string mir;
    std::vector<uint8_t> object;
};

Built build(size_t threads, bool stepwise) {
    std::unique_ptr<Module> mod = make_module(160);
    PipelineRunOptions run;
    run.threads = threads;
    run.min_blocks_per_thread = 1;  // fan out even a small module
    PassPipelineHooks hooks;
    if (stepwise) hooks.after_pass = [](std::string_view) { return true; };
    REQUIRE(run_pipeline(*mod, pipeline(), hooks, run).completed);
    DiagnosticReporter diag;
    const bool valid = verify_module(*mod, &diag);
    if (!valid) std::cerr << diag.format_all();
    REQUIRE(valid);

    object::ModuleCompiler compiler(Target::host());
    codegen::SchedOptions sched;
    sched.enable_post_ra = true;
    sched.enable_software_pipelining = true;
    compiler.set_sched_options(sched);
    compiler.set_enable_trace_layout(true);
    compiler.set_threads(threads);
    compiler.set_min_blocks_per_thread(1);
    object::ObjectFile obj = compiler.compile(*mod);

    Built out;
    out.mir = to_string(*mod);
    out.object = object::emit_elf_object(obj);
    return out;
}

} // namespace

TEST_CASE("parallel build: optimized module and object are byte-identical at any thread count") {
    const Built serial = build(1, false);
    CHECK(serial.object.size() > 100000u);
    for (size_t threads : {size_t{2}, size_t{3}, size_t{8}, size_t{32}}) {
        for (int run = 0; run < 2; ++run) {
            const Built parallel = build(threads, false);
            CHECK(parallel.mir == serial.mir);
            CHECK(parallel.object == serial.object);
        }
    }
}

TEST_CASE("parallel build: a hooked (step by step) run leaves the same module as a fused one") {
    const Built fused = build(8, false);
    const Built stepwise = build(8, true);
    CHECK(stepwise.mir == fused.mir);
    CHECK(stepwise.object == fused.object);
    const Built stepwise_serial = build(1, true);
    CHECK(stepwise_serial.object == fused.object);
}

// Every loop pass walks the loops in LoopAnalysis's order; it was the order of
// a pointer-keyed hash map, so a build depended on where the allocator had put
// the blocks (the same run differed from process to process under ASLR).
TEST_CASE("loop analysis: loops come in block order, not address order") {
    std::string src = "func @f(%0: i64) -> i64 {\nentry:\n  br l0\n\n";
    const int loops = 64;
    for (int k = 0; k < loops; ++k) {
        src += "l" + std::to_string(k) + ":\n";
        src += "  %c" + std::to_string(k) + " = iconst.i64 " + std::to_string(k) + "\n";
        src += "  %t" + std::to_string(k) + " = sgt.i64 %0, %c" + std::to_string(k) + "\n";
        src += "  br_if %t" + std::to_string(k) + ", l" + std::to_string(k) + ", l" + std::to_string(k + 1) + "\n\n";
    }
    src += "l" + std::to_string(loops) + ":\n  ret %0\n}\n";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << diag.format_all();
    REQUIRE(mod != nullptr);
    Function* fn = mod->get_function("f");
    REQUIRE(fn != nullptr);
    DominatorTree dom(*fn);
    LoopAnalysis analysis(*fn, dom);
    const std::vector<LoopInfo*> order = analysis.post_order_loops();
    REQUIRE_EQ(order.size(), size_t{loops});
    for (int k = 0; k < loops; ++k) CHECK(order[k]->header()->name() == "l" + std::to_string(k));
}

TEST_CASE("parallel_for_each: every item once, and the smallest failing item's error") {
    std::vector<std::atomic<int>> seen(1000);
    std::vector<size_t> weights(seen.size());
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = i % 7;
    parallel_for_each(heaviest_first(weights), 8, [&](size_t i) { seen[i].fetch_add(1); });
    for (auto& s : seen) CHECK_EQ(s.load(), 1);

    for (size_t threads : {size_t{1}, size_t{8}}) {
        std::string caught;
        try {
            parallel_for_each(heaviest_first(weights), threads, [](size_t i) {
                if (i == 700 || i == 41 || i == 999) throw std::runtime_error("item " + std::to_string(i));
            });
        } catch (const std::runtime_error& e) {
            caught = e.what();
        }
        CHECK(caught == "item 41");
    }
    CHECK_EQ(threads_for_work(32, 10, 1000000, 2000), 10u);
    CHECK_EQ(threads_for_work(32, 5000, 3000, 2000), 1u);
    CHECK_EQ(threads_for_work(4, 5000, 1000000, 2000), 4u);
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
