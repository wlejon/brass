#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/float_rem_inline.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace brass;
using namespace brass::test;

namespace {

int host_calls = 0;
double host_fmod(double a, double b) {
    ++host_calls;
    return std::fmod(a, b);
}

uint64_t bits(double d) {
    uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

// fmod's answer, bit for bit: the integer path must give -0 where fmod
// does, and a NaN only where fmod does (any NaN, which the bits would not
// compare equal).
bool same(double got, double want) {
    if (std::isnan(want)) return std::isnan(got);
    return bits(got) == bits(want);
}

// f(a, b) = frem(a, b) and g(a) = frem(a, `divisor`), expanded.
Module* build(Module& mod, double divisor) {
    mod.add_symbol_role("host_fmod", SymbolRole::FloatRem);
    Builder b(mod);
    Function* f = mod.create_function("f", Type::f64(), {Type::f64(), Type::f64()});
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    Value* x = b.add_block_param(e, Type::f64());
    Value* y = b.add_block_param(e, Type::f64());
    b.build_ret(b.build_call("host_fmod", Type::f64(), {x, y}));
    f->rebuild_cfg_predecessors();

    Function* g = mod.create_function("g", Type::f64(), {Type::f64()});
    b.set_function(g);
    BasicBlock* ge = b.append_block("entry");
    Value* gx = b.add_block_param(ge, Type::f64());
    b.build_ret(b.build_call("host_fmod", Type::f64(), {gx, b.build_fconst_f64(divisor)}));
    g->rebuild_cfg_predecessors();
    return &mod;
}

} // namespace

TEST_CASE("Float rem inline - integer operands take the integer path, fmod's bits everywhere") {
    Module mod("frem_inline");
    build(mod, 10.0);
    CHECK_EQ(inline_float_rem(*mod.get_function("f")), 1u);
    CHECK_EQ(inline_float_rem(*mod.get_function("g")), 1u);
    DiagnosticReporter diag;
    REQUIRE(verify_function(*mod.get_function("f"), &diag));
    REQUIRE(verify_function(*mod.get_function("g"), &diag));

    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("host_fmod", reinterpret_cast<void*>(&host_fmod));
    REQUIRE(jit.compile_and_load(mod));
    auto f = jit.get_function_ptr<double (*)(double, double)>("f");
    auto g = jit.get_function_ptr<double (*)(double)>("g");
    REQUIRE(f != nullptr);
    REQUIRE(g != nullptr);

    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double big = 9223372036854775808.0;  // 2^63
    const double vals[] = {0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 7.0, -7.0, 10.0, -10.0,
                           2147483647.0, -2147483648.0, 2147483648.0, 9007199254740992.0, 1e18, -1e18,
                           big, -big, 1e300, 0.5, -2.5, 5e-324, inf, -inf, nan};
    int mismatches = 0;
    auto expect = [&](double got, double a, double d) {
        if (same(got, std::fmod(a, d))) return;
        ++mismatches;
        std::printf("frem mismatch: %.17g %% %.17g = %.17g, fmod %.17g\n", a, d, got, std::fmod(a, d));
    };
    for (double a : vals) {
        for (double d : vals) expect(f(a, d), a, d);
        expect(g(a), a, 10.0);
    }
    CHECK_EQ(mismatches, 0);

    // Integer pairs never reach the helper; -0, a 0 or -1 divisor, a
    // fraction and an operand past 2^63 do.
    host_calls = 0;
    CHECK_EQ(f(-7.0, 3.0), -1.0);
    CHECK_EQ(bits(f(-6.0, 3.0)), bits(-0.0));
    CHECK_EQ(f(-big, 10.0), -8.0);
    CHECK_EQ(g(123456789.0), 9.0);
    CHECK_EQ(host_calls, 0);
    f(-0.0, 3.0);
    f(5.0, 0.0);
    f(5.0, -1.0);
    f(5.5, 2.0);
    f(big, 10.0);
    CHECK_EQ(host_calls, 5);
}

TEST_CASE("Float rem inline - a constant divisor the integer path never takes keeps the call") {
    for (double d : {0.0, -0.0, -1.0, 2.5}) {
        Module mod("frem_const");
        build(mod, d);
        CHECK_EQ(inline_float_rem(*mod.get_function("g")), 0u);
    }
}

TEST_CASE("Float rem inline - a constant dividend: integral ones take the path, the rest keep the call") {
    const double big = 9223372036854775808.0;  // 2^63
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double dividends[] = {-0.0, 7.0, -7.0, -big, 0.5, nan, big};
    const size_t expanded[] = {1, 1, 1, 1, 0, 0, 0};
    const double divisors[] = {3.0, -3.0, 1.0, 0.0, -0.0, -1.0, 2.5, 10.0, big, -big, nan};
    int mismatches = 0;
    for (size_t k = 0; k < sizeof dividends / sizeof dividends[0]; ++k) {
        Module mod("frem_const_dividend");
        mod.add_symbol_role("host_fmod", SymbolRole::FloatRem);
        Builder b(mod);
        Function* h = mod.create_function("h", Type::f64(), {Type::f64()});
        b.set_function(h);
        BasicBlock* e = b.append_block("entry");
        Value* d = b.add_block_param(e, Type::f64());
        b.build_ret(b.build_call("host_fmod", Type::f64(), {b.build_fconst_f64(dividends[k]), d}));
        h->rebuild_cfg_predecessors();
        CHECK_EQ(inline_float_rem(*h), expanded[k]);
        DiagnosticReporter diag;
        REQUIRE(verify_function(*h, &diag));
        codegen::JitExecutionEngine jit(Target::host());
        jit.register_external_symbol("host_fmod", reinterpret_cast<void*>(&host_fmod));
        REQUIRE(jit.compile_and_load(mod));
        auto fn = jit.get_function_ptr<double (*)(double)>("h");
        REQUIRE(fn != nullptr);
        for (double dv : divisors) {
            const double got = fn(dv);
            if (same(got, std::fmod(dividends[k], dv))) continue;
            ++mismatches;
            std::printf("frem mismatch: %.17g %% %.17g = %.17g\n", dividends[k], dv, got);
        }
    }
    CHECK_EQ(mismatches, 0);
}
