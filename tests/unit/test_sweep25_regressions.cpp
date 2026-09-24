// Regressions from bug sweep 25:
// - Tier 2 fused an i8 / i16 load into a 64-bit add/sub/mul/and/or/xor
//   (`op r64, qword [mem]`), reading 6-7 bytes past the value; at the end
//   of a mapping it faulted. A load moves exactly its width
//   (docs/semantics.md), so narrow loads are never fused.
// - The same fusion into a narrow compare left the load without a register
//   and widen_narrow_operands threw logic_error.
// - CodeInstaller::install_tier2 let compile exceptions escape, and
//   BackgroundCompiler::worker_loop had no handler, so an
//   UnsupportedOperation (narrow clz) there terminated the process.
// - Tier 2 was never queued automatically: enqueue refused every function
//   with a native entry, which tier-1 code always has.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

using namespace brass;
using namespace brass::runtime;

namespace {

std::unique_ptr<Module> parse_ok(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::printf("%s", diag.format_all().c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    return mod;
}

// A read/write page followed by a no-access page; returns the address of
// the first byte of the no-access page.
uint8_t* page_end() {
#ifdef _WIN32
    auto* base = static_cast<uint8_t*>(VirtualAlloc(nullptr, 8192, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    REQUIRE(base != nullptr);
    DWORD old = 0;
    REQUIRE(VirtualProtect(base + 4096, 4096, PAGE_NOACCESS, &old));
#else
    void* m = mmap(nullptr, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    REQUIRE(m != MAP_FAILED);
    auto* base = static_cast<uint8_t*>(m);
    REQUIRE(mprotect(base + 4096, 4096, PROT_NONE) == 0);
#endif
    return base + 4096; // leaked: a handful of pages per test run
}

int64_t oracle(Module& mod, const char* fn, std::vector<RuntimeValue> args) {
    Interpreter interp;
    interp.set_module(&mod);
    return interp.run(*mod.get_function(fn), args).as_i64();
}

const char* kPageEnd = R"(module @s25_pageend
func @add8(%p: ptr, %y: i64) -> i64 {
b0:
  %t = trunc_i32 %y
  %a = trunc.i8 %t
  %l = load.i8 %p, 0
  %s = add.i8 %a, %l
  %z = zext_i64 %s
  ret %z
}
func @mul16(%p: ptr, %y: i64) -> i64 {
b0:
  %p1 = load.i16 %p, 0
  %p2 = load.i16 %p, 0
  %s = mul.i16 %p1, %p2
  store.i16 %p, 0, %s
  %r = load.i64 %p, -6
  ret %r
}
func @xor8(%p: ptr, %y: i64) -> i64 {
b0:
  %t = trunc_i32 %y
  %a = trunc.i8 %t
  %l = load.i8 %p, 0
  %s = xor.i8 %a, %l
  %z = zext_i64 %s
  ret %z
}
)";

const char* kNarrowCompare = R"(module @s25_cmp
func @eqld(%x: i64, %y: i64) -> i64 {
b0:
  %p = alloca 64, 16
  store.i64 %p, 8, %x
  %t = trunc_i32 %y
  %a = trunc.i8 %t
  %l = load.i8 %p, 8
  %c = eq.i8 %a, %l
  %z = zext_i64 %c
  ret %z
}
func @sltld(%x: i64, %y: i64) -> i64 {
b0:
  %p = alloca 64, 16
  store.i64 %p, 8, %x
  %t = trunc_i32 %y
  %a = trunc.i8 %t
  %l = load.i8 %p, 8
  %c = slt.i8 %a, %l
  br_if %c, yes, no
yes:
  %r1 = iconst.i64 1
  ret %r1
no:
  %r0 = iconst.i64 0
  ret %r0
}
func @sltld16(%x: i64, %y: i64) -> i64 {
b0:
  %p = alloca 64, 16
  store.i64 %p, 8, %x
  store.i64 %p, 16, %y
  %l = load.i16 %p, 8
  %m = load.i16 %p, 16
  %c = slt.i16 %l, %m
  br_if %c, yes, no
yes:
  %r1 = iconst.i64 1
  ret %r1
no:
  %r0 = iconst.i64 0
  ret %r0
}
)";

const char* kClz8 = R"(module @s25_clz
func @f(%x: i64) -> i64 {
b0:
  %t = trunc_i32 %x
  %a = trunc.i8 %t
  %c = clz.i8 %a
  %z = zext_i64 %c
  ret %z
}
func @main(%n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  %one = iconst.i64 1
  %k = iconst.i64 0x10
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %r = call.i64 @f(%k)
  %acc2 = add %acc, %r
  %i2 = add %i, %one
  %c = slt %i2, %n
  br_if %c, loop(%i2, %acc2), done(%acc2)
done(%res: i64):
  ret %res
}
)";

// f(x) = 3x + 1 while x < 1000; past that its guard fails and the exit stub
// returns 3x - x. main(n, k) sums f(i + k) for i in [0, n).
const char* kTierUp = R"(module @s25_tierup
func @stub(%a: i64, %x: i64) -> i64 {
b0:
  %r = sub %a, %x
  ret %r
}
func @f(%x: i64) -> i64 {
b0:
  %three = iconst.i64 3
  %a = mul %x, %three
  %lim = iconst.i64 1000
  %ok = slt %x, %lim
  guard %ok, @stub, [%a, %x]
  %one = iconst.i64 1
  %r = add %a, %one
  ret %r
}
func @main(%n: i64, %k: i64) -> i64 {
b0:
  %z = iconst.i64 0
  %one = iconst.i64 1
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %x = add %i, %k
  %r = call.i64 @f(%x)
  %acc2 = add %acc, %r
  %i2 = add %i, %one
  %c = slt %i2, %n
  br_if %c, loop(%i2, %acc2), done(%acc2)
done(%res: i64):
  ret %res
}
)";

TieringConfig tierup_config(bool background) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 2;
    cfg.invocation_tier2_threshold = 5;
    cfg.enable_background_compile = background;
    cfg.jit_threads = 1;
    return cfg;
}

void run_tierup_case(bool background) {
    auto mod = parse_ok(kTierUp);
    const std::vector<RuntimeValue> pass = {RuntimeValue::from_i64(50), RuntimeValue::from_i64(0)};
    const std::vector<RuntimeValue> fail = {RuntimeValue::from_i64(50), RuntimeValue::from_i64(2000)};
    const int64_t want_pass = oracle(*mod, "main", pass);
    const int64_t want_fail = oracle(*mod, "main", fail);

    FunctionDispatchTable prog;
    prog.pipeline().initialize(tierup_config(background));
    CHECK_EQ(prog.pipeline().execute(*mod, "main", pass).as_i64(), want_pass);
    if (background) prog.pipeline().background_compiler().wait_idle();

    FunctionHandle* f = prog.find("f");
    REQUIRE(f != nullptr);
    CHECK(f->tier() == TierLevel::Tier2_Optimized);
    CHECK(prog.tiering().get_feedback("f").current_tier() == TierLevel::Tier2_Optimized);
    // The tier-1 code is kept (invalidate_optimized falls back to it).
    CHECK(f->baseline_function() != nullptr);
    if (background) CHECK_EQ(prog.pipeline().background_compiler().stats().tasks_enqueued, 1u);

    // Runs in tier 2 again, then deoptimizes through the exit stub.
    CHECK_EQ(prog.pipeline().execute(*mod, "main", pass).as_i64(), want_pass);
    const uint64_t deopts0 = prog.pipeline().tier2_deopts();
    CHECK_EQ(prog.pipeline().execute(*mod, "main", fail).as_i64(), want_fail);
    CHECK(prog.pipeline().tier2_deopts() > deopts0);
    // The failing guard invalidated the tier-2 code: f is back on its
    // tier-1 code, bailed out, and not compiled to tier 2 again (the exit
    // stub, now hot, may be).
    CHECK(f->tier() == TierLevel::Tier1_Baseline);
    CHECK(prog.tiering().get_feedback("f").is_bailout_set());
    CHECK_EQ(f->retired_engine_count(), 1u);
    CHECK_EQ(prog.pipeline().execute(*mod, "main", pass).as_i64(), want_pass);
    if (background) prog.pipeline().background_compiler().wait_idle();
    CHECK(f->tier() == TierLevel::Tier1_Baseline);
    CHECK(f->jit_engine() == nullptr);
}

} // namespace

TEST_CASE("Sweep25 - tier 2 reads exactly an i8/i16 load's bytes at a page end") {
    auto mod = parse_ok(kPageEnd);
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    uint8_t* end = page_end();
    end[-1] = 0x42;
    end[-2] = 0x42;
    auto add8 = reinterpret_cast<int64_t (*)(uint8_t*, int64_t)>(jit.get_symbol_address("add8"));
    auto xor8 = reinterpret_cast<int64_t (*)(uint8_t*, int64_t)>(jit.get_symbol_address("xor8"));
    auto mul16 = reinterpret_cast<int64_t (*)(uint8_t*, int64_t)>(jit.get_symbol_address("mul16"));
    REQUIRE(add8 && xor8 && mul16);
    CHECK_EQ(add8(end - 1, 1), 0x43);
    CHECK_EQ(xor8(end - 1, 3), 0x41);
    // The low 16 bits of 0x4242 * 0x4242 are stored back, and read as the
    // top bytes of the i64 that ends at the page end.
    const int64_t r = mul16(end - 2, 0);
    CHECK_EQ(static_cast<uint64_t>(r) >> 48, (0x4242u * 0x4242u) & 0xFFFFu);
}

TEST_CASE("Sweep25 - tier 2 compiles a narrow compare of a single-use load") {
    auto mod = parse_ok(kNarrowCompare);
    codegen::JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    using Fn = int64_t (*)(int64_t, int64_t);
    const std::pair<int64_t, int64_t> cases[] = {{0x1105, 5}, {0x1105, 6}, {0x80, 1}, {0x7f, 0x80},
                                                 {0x8000, 1}, {1, 0x8000}, {-1, -2}, {0, 0}};
    for (const char* name : {"eqld", "sltld", "sltld16"}) {
        auto fn = reinterpret_cast<Fn>(jit.get_symbol_address(name));
        REQUIRE(fn != nullptr);
        for (const auto& [x, y] : cases) {
            CHECK_EQ(fn(x, y), oracle(*mod, name, {RuntimeValue::from_i64(x), RuntimeValue::from_i64(y)}));
        }
    }
}

TEST_CASE("Sweep25 - install_tier2 reports an UnsupportedOperation as a failed result") {
    auto mod = parse_ok(kClz8);
    FunctionDispatchTable prog;
    FunctionHandle* h = prog.get_or_create("f", mod->get_function("f"));
    CodeInstaller installer(prog);
    CodeInstallResult res = installer.install_tier2(*h, *mod, "f");
    CHECK(!res.success);
    CHECK(!res.error_message.empty());
    CHECK(!h->has_native_entry());
    CHECK(h->tier2_rejected());
}

TEST_CASE("Sweep25 - a background worker survives an UnsupportedOperation") {
    auto mod = parse_ok(kClz8);
    auto mod2 = parse_ok(R"(module @s25_ok
func @main(%n: i64) -> i64 {
b0:
  %two = iconst.i64 2
  %r = mul %n, %two
  ret %r
}
)");
    FunctionDispatchTable prog;
    BackgroundCompilerConfig cfg;
    cfg.num_threads = 1;
    cfg.table = &prog;
    BackgroundCompiler compiler(cfg);
    FunctionHandle* h = prog.get_or_create("f", mod->get_function("f"));
    REQUIRE(compiler.enqueue("f", *mod, h));
    compiler.wait_idle();
    CHECK_EQ(compiler.get_task_status("f"), CompileStatus::Failed);
    CHECK_EQ(compiler.stats().tasks_failed, 1u);
    CHECK(!h->has_native_entry());
    CHECK(h->tier() == TierLevel::Tier0_Interpreter);
    // Rejected: not queued again.
    CHECK(!compiler.enqueue("f", *mod, h));
    // The worker is still alive and compiles the next task.
    FunctionHandle* m = prog.get_or_create("main", mod2->get_function("main"));
    REQUIRE(compiler.enqueue("main", *mod2, m));
    compiler.wait_idle();
    CHECK_EQ(compiler.get_task_status("main"), CompileStatus::Completed);
    CHECK(m->tier() == TierLevel::Tier2_Optimized);
    auto twice = m->get_function_ptr<int64_t (*)(int64_t)>();
    REQUIRE(twice != nullptr);
    CHECK_EQ(twice(21), 42);
    compiler.stop();
}

TEST_CASE("Sweep25 - automatic tier-up leaves a function tier 2 rejects on tier 1, once") {
    auto mod = parse_ok(kClz8);
    const int64_t want = oracle(*mod, "main", {RuntimeValue::from_i64(40)});
    FunctionDispatchTable prog;
    prog.pipeline().initialize(tierup_config(true));
    for (int round = 0; round < 3; ++round) {
        CHECK_EQ(prog.pipeline().execute(*mod, "main", {RuntimeValue::from_i64(40)}).as_i64(), want);
        prog.pipeline().background_compiler().wait_idle();
    }
    FunctionHandle* f = prog.find("f");
    REQUIRE(f != nullptr);
    CHECK(f->tier() == TierLevel::Tier1_Baseline);
    CHECK(f->tier2_rejected());
    const auto stats = prog.pipeline().background_compiler().stats();
    CHECK_EQ(stats.tasks_enqueued, 1u);
    CHECK_EQ(stats.tasks_failed, 1u);
}

TEST_CASE("Sweep25 - automatic tier-up to tier 2 in the background, then a deopt") {
    run_tierup_case(true);
}

TEST_CASE("Sweep25 - automatic tier-up to tier 2 on the calling thread, then a deopt") {
    run_tierup_case(false);
}

TEST_CASE("Sweep25 - tier-2 code is not published into a handle rebound while it compiled") {
    auto mod = parse_ok(kTierUp);
    auto other = parse_ok(kTierUp);
    FunctionDispatchTable prog;
    FunctionHandle* h = prog.get_or_create("f", mod->get_function("f"));
    // The compile is validated against @f of `mod`; publishing it into a
    // handle now bound to another Function is refused.
    auto engine = std::make_shared<codegen::JitExecutionEngine>();
    CHECK(!h->publish_optimized(engine, reinterpret_cast<void*>(0x1000), other->get_function("f"), Type::i64(),
                                {Type::i64()}, false));
    CHECK(!h->has_native_entry());
    CHECK(h->tier() == TierLevel::Tier0_Interpreter);
    CHECK_EQ(h->retired_engine_count(), 1u);
}
