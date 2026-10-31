#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/fdiv_pow2.hpp>
#include <brass/mir/verifier.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::test;

namespace {

uint64_t bits(double d) {
    uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

uint32_t bits32(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

size_t count_op(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == op) ++n;
        }
    }
    return n;
}

// name(x) = x / c, in the type `f32` picks.
Function* divide_by(Module& mod, Builder& b, const std::string& name, double c, bool f32) {
    const Type t = f32 ? Type::f32() : Type::f64();
    Function* f = mod.create_function(name, t, {t});
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    Value* x = b.add_block_param(e, t);
    Value* k = f32 ? b.build_fconst_f32(static_cast<float>(c)) : b.build_fconst_f64(c);
    b.build_ret(b.build_sdiv(x, k));
    f->rebuild_cfg_predecessors();
    return f;
}

const double kInputs[] = {
    0.0, -0.0, 5e-324, -5e-324, 2.2250738585072014e-308, 1e-300, 0.1, 1.0, -1.5, 3.0, 1e300,
    1.7976931348623157e308, -1.7976931348623157e308, 4503599627370497.0, 0.30000000000000004,
    std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::quiet_NaN(),
};

bool same(double got, double want) {
    if (want != want) return got != got;
    return bits(got) == bits(want);
}

bool same32(float got, float want) {
    if (want != want) return got != got;
    return bits32(got) == bits32(want);
}

}  // namespace

TEST_CASE("Fdiv pow2 - which constants have an exact reciprocal") {
    double r = 0.0;
    CHECK(exact_pow2_reciprocal(2.0, false, r));
    CHECK(r == 0.5);
    CHECK(exact_pow2_reciprocal(-4.0, false, r));
    CHECK(r == -0.25);
    CHECK(exact_pow2_reciprocal(0.5, false, r));
    CHECK(r == 2.0);
    CHECK(exact_pow2_reciprocal(1.0, false, r));
    CHECK(exact_pow2_reciprocal(std::ldexp(1.0, -1022), false, r));
    CHECK(r == std::ldexp(1.0, 1022));
    CHECK(exact_pow2_reciprocal(std::ldexp(1.0, 1022), false, r));
    // 2^1023's reciprocal is subnormal, so it is kept. 2^-1023 is subnormal
    // itself but exact, and its reciprocal 2^1023 is normal: both forms are
    // the same real number rounded once, so it is rewritten.
    CHECK(!exact_pow2_reciprocal(std::ldexp(1.0, 1023), false, r));
    CHECK(exact_pow2_reciprocal(std::ldexp(1.0, -1023), false, r));
    CHECK(r == std::ldexp(1.0, 1023));
    CHECK(!exact_pow2_reciprocal(3.0, false, r));
    CHECK(!exact_pow2_reciprocal(0.1, false, r));
    CHECK(!exact_pow2_reciprocal(0.0, false, r));
    CHECK(!exact_pow2_reciprocal(-0.0, false, r));
    CHECK(!exact_pow2_reciprocal(std::numeric_limits<double>::infinity(), false, r));
    CHECK(!exact_pow2_reciprocal(std::numeric_limits<double>::quiet_NaN(), false, r));
    // A float's range is narrower: 2^127's reciprocal is a subnormal float,
    // 2^-127 is not a normal float at all.
    CHECK(exact_pow2_reciprocal(2.0, true, r));
    CHECK(exact_pow2_reciprocal(std::ldexp(1.0, 126), true, r));
    CHECK(!exact_pow2_reciprocal(std::ldexp(1.0, 127), true, r));
    CHECK(!exact_pow2_reciprocal(std::ldexp(1.0, 200), true, r));
}

TEST_CASE("Fdiv pow2 - a division by a power of two becomes a multiply with the division's bits") {
    const std::vector<double> rewritten = {2.0, 0.5, -4.0, 1.0, 1024.0, std::ldexp(1.0, -1022), std::ldexp(1.0, 1022), std::ldexp(1.0, -1023)};
    const std::vector<double> kept = {3.0, 0.1, -0.0, std::ldexp(1.0, 1023), std::numeric_limits<double>::infinity()};
    Module mod("fdiv");
    Builder b(mod);
    std::vector<Function*> fns;
    for (size_t i = 0; i < rewritten.size(); ++i) fns.push_back(divide_by(mod, b, "r" + std::to_string(i), rewritten[i], false));
    for (size_t i = 0; i < kept.size(); ++i) fns.push_back(divide_by(mod, b, "k" + std::to_string(i), kept[i], false));
    Function* f32 = divide_by(mod, b, "half32", 2.0, true);
    Function* f32k = divide_by(mod, b, "big32", std::ldexp(1.0, 127), true);
    for (size_t i = 0; i < fns.size(); ++i) {
        const bool expect = i < rewritten.size();
        CHECK_EQ(rewrite_fdiv_pow2(*fns[i]), expect ? 1u : 0u);
        CHECK_EQ(count_op(*fns[i], Opcode::sdiv), expect ? 0u : 1u);
        CHECK_EQ(count_op(*fns[i], Opcode::mul), expect ? 1u : 0u);
    }
    CHECK_EQ(rewrite_fdiv_pow2(*f32), 1u);
    CHECK_EQ(rewrite_fdiv_pow2(*f32k), 0u);
    DiagnosticReporter diag;
    for (Function* fn : fns) REQUIRE(verify_function(*fn, &diag));
    REQUIRE(verify_function(*f32, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    std::vector<double> all = rewritten;
    all.insert(all.end(), kept.begin(), kept.end());
    for (size_t i = 0; i < all.size(); ++i) {
        const std::string name = i < rewritten.size() ? "r" + std::to_string(i) : "k" + std::to_string(i - rewritten.size());
        auto p = jit.get_function_ptr<double (*)(double)>(name);
        REQUIRE(p);
        for (double x : kInputs) {
            volatile double c = all[i];
            CHECK(same(p(x), x / c));
        }
    }
    auto p32 = jit.get_function_ptr<float (*)(float)>("half32");
    REQUIRE(p32);
    for (double xd : kInputs) {
        const float x = static_cast<float>(xd);
        volatile float two = 2.0f;
        CHECK(same32(p32(x), x / two));
    }
}
