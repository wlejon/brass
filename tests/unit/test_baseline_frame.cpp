// Baseline frame layout (x64 and AArch64): values whose live ranges do not overlap share a
// slot, so a frame is sized by the values live at once, not by the SSA value
// count (a deep recursion through baseline code once overflowed the native
// stack where tier 2 did not). The sharing must keep every value that is
// still live intact: across loop back edges, blocks laid out before their
// dominators' successors, and calls.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/fuzz/program_generator.hpp>
#include <brass/runtime/code_installer.hpp>
#include <iostream>
#include <memory>
#include <string>

using namespace brass;
using namespace brass::codegen;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

std::unique_ptr<Module> parse_or_fail(const std::string& src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << "parse failed:\n" << diag.format_all() << "\n" << src << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n" << src << "\n";
    REQUIRE(ok);
    return mod;
}

// n + 32 * (1 + 2 + ... ) as a chain of 64 short-lived values, then a
// self-call while n > 0: at most a handful of values are live at once.
std::string chain_source() {
    std::string s = "module @m\nfunc @bl_fr_chain(%0: i64) -> i64 {\nentry:\n  %z = iconst.i64 0\n";
    std::string prev = "%0";
    for (int i = 0; i < 32; ++i) {
        const std::string c = "%c" + std::to_string(i);
        const std::string v = "%v" + std::to_string(i);
        s += "  " + c + " = iconst.i64 " + std::to_string(i + 1) + "\n";
        s += "  " + v + " = add.i64 " + prev + ", " + c + "\n";
        prev = v;
    }
    s += "  %pos = sgt.i64 %0, %z\n  br_if %pos, rec, base\n";
    s += "rec:\n  %one = iconst.i64 1\n  %n = sub.i64 %0, %one\n";
    s += "  %r = call.i64 @bl_fr_chain(%n)\n  %t = add.i64 %r, " + prev + "\n  ret %t\n";
    s += "base:\n  ret " + prev + "\n}\n";
    return s;
}

int64_t chain_expected(int64_t n) {
    // Each level adds n + 528 (1 + ... + 32).
    int64_t total = 0;
    for (int64_t k = n; k >= 0; --k) total += k + 528;
    return total;
}

} // namespace

TEST_CASE("Baseline frame - short-lived values share slots") {
    auto mod = parse_or_fail(chain_source());
    BaselineJitCompiler compiler;
    auto compiled = compiler.compile(*mod->get_function("bl_fr_chain"));
    REQUIRE(compiled.is_valid());
    REQUIRE(!compiled.stack_map().records.empty());
    // 70 values; one slot each was over 560 bytes. Live at once: a few.
    const uint32_t frame = compiled.stack_map().records.front().frame_size;
    CHECK(frame <= 96);

    auto fn = compiled.get_function_ptr<int64_t (*)(int64_t)>();
    CHECK_EQ(fn(0), chain_expected(0));
    CHECK_EQ(fn(7), chain_expected(7));
    // Deep enough that one-slot-per-value frames (~600 bytes) would need
    // well over the default 1 MB stack.
    CHECK_EQ(fn(3000), chain_expected(3000));
}

TEST_CASE("Baseline frame - values live across a loop and a block laid out early") {
    // @done is laid out before @loop but reached only after it: %base and
    // %k (defined in entry) must survive the whole loop, and the loop's
    // temporaries must not reuse their slots.
    auto mod = parse_or_fail(R"(module @m
func @bl_fr_loop(%0: i64, %1: i64) -> i64 {
entry:
  %c7 = iconst.i64 7
  %base = mul.i64 %0, %c7
  %k = add.i64 %1, %c7
  %c0 = iconst.i64 0
  br loop(%c0, %c0)
done(%acc2: i64):
  %x = add.i64 %acc2, %base
  %y = mul.i64 %x, %k
  ret %y
loop(%i: i64, %acc: i64):
  %c3 = iconst.i64 3
  %t1 = mul.i64 %i, %c3
  %t2 = add.i64 %t1, %k
  %t3 = sub.i64 %t2, %i
  %t4 = add.i64 %acc, %t3
  %one = iconst.i64 1
  %n = add.i64 %i, %one
  %more = slt.i64 %n, %0
  br_if %more, loop(%n, %t4), done(%t4)
}
)");
    BaselineJitCompiler compiler;
    auto compiled = compiler.compile(*mod->get_function("bl_fr_loop"));
    REQUIRE(compiled.is_valid());
    auto fn = compiled.get_function_ptr<int64_t (*)(int64_t, int64_t)>();
    auto expected = [](int64_t a, int64_t b) {
        const int64_t k = b + 7;
        int64_t acc = 0;
        int64_t i = 0;
        do {
            acc += i * 3 + k - i;
            ++i;
        } while (i < a);
        return (acc + a * 7) * k;
    };
    for (int64_t a : {1, 2, 5, 40}) {
        for (int64_t b : {-3, 0, 11}) CHECK_EQ(fn(a, b), expected(a, b));
    }
}

namespace {

// %obj is live around a loop with a safepoint; %p = %obj + 8 is an interior
// pointer defined and used inside the loop body. `derived_across` moves the
// load of %p after the safepoint, which the verifier rejects.
std::string derived_source(bool derived_across) {
    std::string s = R"(module @m
func @bl_fr_derived(%0: i64) -> i64 {
entry:
  %sz = iconst.i64 64
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %obj = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %c8 = iconst.i64 8
  %p = add.gcref %obj, %c8
)";
    if (derived_across) s += "  safepoint\n  %v = load.i32 %p\n";
    else s += "  %v = load.i32 %p\n  safepoint\n";
    s += R"(  %w = zext.i64 %v
  %acc2 = add.i64 %acc, %w
  %one = iconst.i64 1
  %n = add.i64 %i, %one
  %more = slt.i64 %n, %0
  br_if %more, loop(%n, %acc2), done
done:
  ret %acc2
}
)";
    return s;
}

} // namespace

// Fuzz seeds 18 and 20049 (brass-fuzz --baseline): the derived pointer's
// slot was a stack-map root, so a collection at the loop safepoint read an
// object header from inside the object and threw from evacuate_object.
TEST_CASE("Baseline frame - a derived gcref is not a stack-map root") {
    auto mod = parse_or_fail(derived_source(false));
    BaselineJitCompiler compiler;
    auto compiled = compiler.compile(*mod->get_function("bl_fr_derived"));
    REQUIRE(compiled.is_valid());
    REQUIRE(!compiled.stack_map().records.empty());
    // One gcref slot: %obj. %p is not rooted.
    for (const auto& rec : compiled.stack_map().records) CHECK_EQ(rec.roots.size(), size_t{1});
}

TEST_CASE("Baseline frame - a derived gcref live across a GC point is rejected") {
    DiagnosticReporter diag;
    auto mod = parse_module(derived_source(true), &diag);
    REQUIRE(mod != nullptr);
    BaselineJitCompiler compiler;
    std::string message;
    try {
        (void)compiler.compile(*mod->get_function("bl_fr_derived"));
    } catch (const std::exception& e) {
        message = e.what();
    }
    CHECK(message.find("derived gcref live across a GC point") != std::string::npos);
}

TEST_CASE("Baseline frame - fuzz seeds with derived gcrefs across loop safepoints") {
    struct Seed { uint64_t seed; uint32_t max_statements; };
    for (const Seed s : {Seed{18, 36}, Seed{20049, 120}}) {
        fuzz::DiffFuzzerOptions opts;
        opts.save_reproducers = false;
        opts.tier6_baseline = true;
        opts.timeout_ms = 5000;
        fuzz::DiffFuzzer fuzzer(opts);
        fuzz::ProgramGeneratorOptions gen_opts;
        gen_opts.max_statements = s.max_statements;
        fuzz::ProgramGenerator gen(gen_opts);
        Module mod("fuzz_mod_" + std::to_string(s.seed));
        REQUIRE(gen.generate(mod, "fuzz_fn", s.seed) != nullptr);
        const std::vector<RuntimeValue> args = {
            RuntimeValue::from_i64(static_cast<int64_t>(s.seed % 100)),
            RuntimeValue::from_i64(static_cast<int64_t>((s.seed >> 8) % 100)),
        };
        fuzz::DiffResult r = fuzzer.run_test(mod, "fuzz_fn", args, s.seed);
        if (!r.passed) std::cerr << "seed " << s.seed << ": " << r.failure_class << " " << r.mismatch_reason << "\n";
        CHECK(r.passed);
        CHECK_FALSE(r.baseline_rejected);
    }
}

#endif
