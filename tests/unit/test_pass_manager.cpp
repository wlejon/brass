// The pass manager (brass/mir/pass_manager.hpp) and the declared pipelines
// built on it: duplicate steps are refused, repeats are explicit, filters and
// follow-ups apply, and the loop pipeline orders IVSR after the vectorizer so
// a bronze-shaped loop actually vectorizes.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/pass_pipeline.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/inliner.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/pass_catalog.hpp>
#include <brass/mir/pass_manager.hpp>
#include <brass/mir/pass_pipeline.hpp>
#include <brass/mir/verifier.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brass;

namespace {

std::unique_ptr<Module> parse(std::string_view text) {
    DiagnosticReporter diag;
    auto mod = parse_module(text, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    return mod;
}

PassStep counting_step(std::string name, int& runs, bool changes, FunctionFilter filter = FunctionFilter::All) {
    PassStep s;
    s.name = std::move(name);
    s.filter = filter;
    s.run_function = [&runs, changes](Function&) {
        ++runs;
        return changes;
    };
    return s;
}

bool throws_logic_error(const std::function<void()>& f) {
    try {
        f();
    } catch (const std::logic_error&) {
        return true;
    }
    return false;
}

size_t index_of(const std::vector<std::string>& names, std::string_view name) {
    auto it = std::find(names.begin(), names.end(), name);
    REQUIRE(it != names.end());
    return static_cast<size_t>(it - names.begin());
}

size_t count_opcode(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) n += inst->opcode() == op ? 1 : 0;
    }
    return n;
}

constexpr std::string_view kTwoFunctions = R"(
func @f(%0: i64) -> i64 {
bb0:
  ret %0
}

func @__wrapper_f(%0: i64) -> i64 {
bb0:
  ret %0
}
)";

// A loop shaped like Bronze's array loops once BCE removed their checks:
// an invariant base, `i < n`, and 8-byte elements past an 8-byte header.
constexpr std::string_view kBronzeLoop = R"(
func @kernel(%arr: ptr, %n: i64, %k: i64) -> void {
bb0:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  br hdr(%zero)

hdr(%i: i64):
  %c = slt.i64 %i, %n
  br_if %c, body, exit

body:
  %e = load_indexed.i64 %arr, %i, 8, 8
  %s = add.i64 %e, %k
  store_indexed.i64 %arr, %i, 8, 8, %s
  %i1 = add.i64 %i, %one
  br hdr(%i1)

exit:
  ret
}

func @main(%n: i64, %k: i64) -> i64 {
bb0:
  %buf = alloca 256, 8
  %zero = iconst.i64 0
  %one = iconst.i64 1
  br fill(%zero)

fill(%i: i64):
  %c = slt.i64 %i, %n
  br_if %c, fill_body, run

fill_body:
  store_indexed.i64 %buf, %i, 8, 8, %i
  %i1 = add.i64 %i, %one
  br fill(%i1)

run:
  call @kernel(%buf, %n, %k)
  br sum(%zero, %zero)

sum(%j: i64, %acc: i64):
  %d = slt.i64 %j, %n
  br_if %d, sum_body, done

sum_body:
  %e = load_indexed.i64 %buf, %j, 8, 8
  %acc2 = add.i64 %acc, %e
  %j1 = add.i64 %j, %one
  br sum(%j1, %acc2)

done:
  ret %acc
}
)";

int64_t run_main(const Module& mod, int64_t n, int64_t k) {
    Interpreter interp;
    return interp.run(mod, "main", {RuntimeValue::from_i64(n), RuntimeValue::from_i64(k)}).as_i64();
}

} // namespace

TEST_CASE("Pass manager - a step name runs once and repeats are named") {
    int runs = 0;
    Pipeline p;
    p.add(counting_step("cfg_simplify", runs, false));
    CHECK(throws_logic_error([&] { p.add(counting_step("cfg_simplify", runs, false)); }));
    p.add(counting_step("cfg_simplify 2", runs, false));
    p.add(counting_step("gvn", runs, false));
    CHECK(p.names() == (std::vector<std::string>{"cfg_simplify", "cfg_simplify 2", "gvn"}));

    // A base name drops its numbered repeats, and only those.
    CHECK(p.without({"cfg_simplify"}).names() == std::vector<std::string>{"gvn"});
    CHECK(p.without({"cfg"}).names().size() == 3u);

    PassStep empty;
    empty.name = "nothing";
    CHECK(throws_logic_error([&] { p.add(empty); }));

    Pipeline q;
    q.add(counting_step("gvn", runs, false));
    CHECK(throws_logic_error([&] { p.append(q); }));
}

TEST_CASE("Pass manager - filters, follow-ups and hooks") {
    auto mod = parse(kTwoFunctions);
    int all_runs = 0, eligible_runs = 0, follow_runs = 0, quiet_follow_runs = 0;
    Pipeline p;
    p.add(counting_step("all", all_runs, false));
    p.add(counting_step("eligible", eligible_runs, false, FunctionFilter::LoopEligible));
    PassStep changing = counting_step("changing", all_runs, true, FunctionFilter::NonWrapper);
    changing.follow_up = [&](Function&) { return ++follow_runs, false; };
    p.add(changing);
    PassStep quiet = counting_step("quiet", all_runs, false);
    quiet.follow_up = [&](Function&) { return ++quiet_follow_runs, false; };
    p.add(quiet);

    std::vector<std::string> seen;
    PassPipelineHooks hooks;
    hooks.before_pass = [&](std::string_view name) { seen.emplace_back(name); };
    const PipelineResult r = run_pipeline(*mod, p, hooks);
    CHECK(r.completed);
    CHECK(r.changed);
    CHECK_EQ(all_runs, 2 + 1 + 2);  // "all" x2, "changing" x1, "quiet" x2
    CHECK_EQ(eligible_runs, 1);      // not the wrapper
    CHECK_EQ(follow_runs, 1);        // only after a change
    CHECK_EQ(quiet_follow_runs, 0);
    CHECK(seen == p.names());
    CHECK(!mod->has_loop_optimizations());

    // after_pass returning false stops the pipeline.
    hooks.after_pass = [](std::string_view name) { return name != "eligible"; };
    all_runs = 0;
    CHECK(!run_pipeline(*mod, p, hooks).completed);
    CHECK_EQ(all_runs, 2);

    // Module steps cannot run on one function.
    Pipeline with_module;
    with_module.add(passes::write_barrier_elim());
    CHECK(throws_logic_error([&] { run_pipeline(*mod->get_function("f"), with_module); }));

    Pipeline loops = loop_pipeline(LoopOptOptions{});
    CHECK(loops.marks_loop_optimized());
    run_pipeline(*mod, loops);
    CHECK(mod->has_loop_optimizations());
}

TEST_CASE("Pass manager - the declared pipelines have no accidental repeats") {
    // Production (inlining on): SROA runs before and after inlining, never
    // again in the loop stage; BCE runs early and late.
    PassPipelineOptions t = production_pass_pipeline_options();
    t.enable_inlining = true;
    const std::vector<std::string> prod = pass_pipeline(t).names();
    CHECK_EQ(std::count(prod.begin(), prod.end(), "sroa"), 1);
    CHECK_EQ(std::count(prod.begin(), prod.end(), "sroa 2"), 1);
    CHECK(index_of(prod, "inline") < index_of(prod, "sroa 2"));
    CHECK(index_of(prod, "bce") < index_of(prod, "loop_cleanup"));
    CHECK(index_of(prod, "loop_cleanup") < index_of(prod, "bce 2"));
    CHECK_EQ(index_of(prod, "wbe"), prod.size() - 1);

    // optimize_function: SROA and allocation sinking once each.
    LoopOptOptions o;
    o.enable_sroa = true;
    o.enable_allocation_sinking = true;
    const std::vector<std::string> fn = function_pipeline(o).names();
    CHECK_EQ(std::count(fn.begin(), fn.end(), "sroa"), 1);
    CHECK_EQ(std::count(fn.begin(), fn.end(), "allocation_sinking"), 1);

    // IVSR after the transforms whose input form it would destroy.
    const std::vector<std::string> loops = loop_pipeline(LoopOptOptions{}).names();
    CHECK(index_of(loops, "loop_vectorize") < index_of(loops, "ivsr"));
    CHECK(index_of(loops, "loop_unroll") < index_of(loops, "ivsr"));

    // IPO devirtualizes in its own step only.
    InlinerOptions io;
    io.enable_devirtualization = true;
    const std::vector<std::string> ipo = ipo_pipeline(io).names();
    CHECK(ipo == (std::vector<std::string>{"devirtualize", "inline", "sroa"}));
}

TEST_CASE("Pass manager - the inline step never devirtualizes") {
    auto mod = parse(R"(
extern @target

func @callee() -> i64 {
bb0:
  %0 = iconst.i64 5
  ret %0
}

func @f() -> i64 {
bb0:
  %0 = patchable_call.i64 @site, @callee()
  ret %0
}
)");
    InlinerOptions io;
    io.enable_devirtualization = true;
    Pipeline p;
    p.add(passes::inline_calls(io));
    run_pipeline(*mod, p);
    CHECK_EQ(count_opcode(*mod->get_function("f"), Opcode::patchable_call), size_t{1});
}

TEST_CASE("Pass manager - production pipeline vectorizes a bronze-shaped loop") {
    auto mod = parse(kBronzeLoop);
    const int64_t expected = run_main(*mod, 13, 100);
    CHECK_EQ(expected, 78 + 1300);

    PassPipelineOptions opts = production_pass_pipeline_options();
    opts.loop.enable_avx2 = true;  // the transform is target-independent
    opts.loop.vector_width = 256;
    run_pipeline(*mod, pass_pipeline(opts));
    DiagnosticReporter diag;
    const bool ok = verify_module(*mod, &diag);
    if (!ok) std::cerr << diag.format_all() << "\n";
    REQUIRE(ok);
    const Function& kernel = *mod->get_function("kernel");
    CHECK(count_opcode(kernel, Opcode::vload) >= 1u);
    CHECK(count_opcode(kernel, Opcode::vstore) >= 1u);
    for (int64_t n : {0, 1, 3, 4, 13, 28}) {
        auto fresh = parse(kBronzeLoop);
        CHECK_EQ(run_main(*mod, n, 100), run_main(*fresh, n, 100));
    }

    // The old order - IVSR first - leaves the vectorizer nothing to match.
    auto old_order = parse(kBronzeLoop);
    Pipeline p;
    p.add(passes::ivsr(opts.loop));
    p.add(passes::loop_vectorize(opts.loop));
    run_pipeline(*old_order, p);
    CHECK_EQ(count_opcode(*old_order->get_function("kernel"), Opcode::vload), size_t{0});
}
