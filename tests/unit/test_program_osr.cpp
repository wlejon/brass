// OSR in a program whose pipeline runs it: a hot loop's frame moves into
// an OSR entry function compiled in the background, mid-loop, and finishes
// there with the result the interpreter would have given.
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/osr_entry.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/tiering.hpp>
#include <iostream>
#include <memory>
#include <string>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

std::unique_ptr<Module> parse_or_fail(const std::string& src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << "parse failed:\n" << diag.format_all() << "\n" << src << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n" << src << "\n";
    REQUIRE(ok);
    return mod;
}

// Interpreted only (no tier-up by invocation), with OSR after `threshold`
// backedges.
void init_program(FunctionDispatchTable& prog, uint64_t threshold) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);
    prog.osr().set_enabled(true);
    prog.osr().set_threshold(threshold);
}

// sum over i < n of (step(i) + k), k computed before the loop.
const char* kSimpleLoop = R"(module @osr_simple
func @osr_step(%0: i64) -> i64 {
entry:
  %c3 = iconst.i64 3
  %r = mul.i64 %0, %c3
  ret %r
}
func @osr_sum(%n: i64) -> i64 {
entry:
  %c7 = iconst.i64 7
  %k = add.i64 %n, %c7
  %c0 = iconst.i64 0
  br loop(%c0, %c0)
loop(%i: i64, %acc: i64):
  %s = call.i64 @osr_step(%i)
  %t = add.i64 %s, %k
  %a = add.i64 %acc, %t
  %c1 = iconst.i64 1
  %j = add.i64 %i, %c1
  %more = slt.i64 %j, %n
  br_if %more, loop(%j, %a), done(%a)
done(%r: i64):
  ret %r
}
)";

int64_t simple_expected(int64_t n) {
    int64_t acc = 0;
    for (int64_t i = 0; i < n; ++i) acc += i * 3 + n + 7;
    return acc;
}

// sum over i < n, j < m of (i * j + i): the inner loop's entry carries the
// outer loop's %i and %acc, which its latch defines again.
const char* kNestedLoop = R"(module @osr_nested
func @osr_nest(%n: i64, %m: i64) -> i64 {
entry:
  %c0 = iconst.i64 0
  br outer(%c0, %c0)
outer(%i: i64, %acc: i64):
  %z = iconst.i64 0
  br inner(%z, %acc)
inner(%j: i64, %acc2: i64):
  %p = mul.i64 %i, %j
  %q = add.i64 %p, %i
  %a = add.i64 %acc2, %q
  %c1 = iconst.i64 1
  %j2 = add.i64 %j, %c1
  %more = slt.i64 %j2, %m
  br_if %more, inner(%j2, %a), latch(%a)
latch(%acc3: i64):
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  %again = slt.i64 %i2, %n
  br_if %again, outer(%i2, %acc3), done(%acc3)
done(%r: i64):
  ret %r
}
)";

int64_t nested_expected(int64_t n, int64_t m) {
    int64_t acc = 0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < m; ++j) acc += i * j + i;
    }
    return acc;
}

} // namespace

TEST_CASE("Program OSR - the entry function of an inner loop carries the outer loop's values") {
    auto mod = parse_or_fail(kNestedLoop);
    const Function& fn = *mod->get_function("osr_nest");
    const BasicBlock* inner = nullptr;
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb->name() == "inner") inner = bb;
    }
    REQUIRE(inner != nullptr);
    std::string why;
    auto plan = plan_osr_entry(fn, *inner, &why);
    REQUIRE(plan.has_value());
    // %j, %acc2 (its parameters), then %n, %m and %i.
    CHECK_EQ(plan->live_ins.size(), 5u);
    Module dst("osr_nested_copy");
    Function* entry = build_osr_entry_function(*plan, dst, "osr_nest.osr", &why);
    if (!entry) std::cerr << why << "\n";
    REQUIRE(entry != nullptr);
    CHECK(entry->param_count() == 1);
    // The entry block itself is not an OSR entry.
    CHECK(!plan_osr_entry(fn, *fn.entry_block(), &why).has_value());
}

TEST_CASE("Program OSR - a hot loop moves into compiled code mid-loop and finishes there") {
    auto mod = parse_or_fail(kSimpleLoop);
    const Function& fn = *mod->get_function("osr_sum");
    FunctionDispatchTable prog;
    init_program(prog, 100);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);

    // The first run asks for the loop's OSR code; the second, once it is
    // compiled, enters it.
    CHECK_EQ(interp.run(fn, {RuntimeValue::from_i64(2000)}).as_i64(), simple_expected(2000));
    CompilePool::shared().wait_owner(&prog.osr());
    const uint64_t before = prog.osr().total_osr_migrations();
    CHECK_EQ(interp.run(fn, {RuntimeValue::from_i64(100000)}).as_i64(), simple_expected(100000));
    CHECK(prog.osr().total_osr_migrations() > before);
}

TEST_CASE("Program OSR - nested loops finish with the interpreter's result") {
    auto mod = parse_or_fail(kNestedLoop);
    const Function& fn = *mod->get_function("osr_nest");
    FunctionDispatchTable prog;
    init_program(prog, 100);
    FastInterpreter interp;
    interp.set_dispatch_table(&prog);
    const std::vector<RuntimeValue> small{RuntimeValue::from_i64(30), RuntimeValue::from_i64(40)};
    CHECK_EQ(interp.run(fn, small).as_i64(), nested_expected(30, 40));
    CompilePool::shared().wait_owner(&prog.osr());
    const uint64_t before = prog.osr().total_osr_migrations();
    const std::vector<RuntimeValue> big{RuntimeValue::from_i64(300), RuntimeValue::from_i64(700)};
    CHECK_EQ(interp.run(fn, big).as_i64(), nested_expected(300, 700));
    CHECK(prog.osr().total_osr_migrations() > before);
}

TEST_CASE("Program OSR - a program destroyed with its OSR compile queued goes cleanly") {
    auto mod = parse_or_fail(kSimpleLoop);
    const Function& fn = *mod->get_function("osr_sum");
    for (int round = 0; round < 4; ++round) {
        auto prog = std::make_unique<FunctionDispatchTable>();
        init_program(*prog, 100);
        FastInterpreter interp;
        interp.set_dispatch_table(prog.get());
        CHECK_EQ(interp.run(fn, {RuntimeValue::from_i64(500)}).as_i64(), simple_expected(500));
        prog.reset();
    }
}

#endif
