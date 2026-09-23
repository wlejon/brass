// x64 tier 2, vector calls: vector arguments and results travel in XMM
// registers (by position on Win64, the next free XMM on SysV), the same
// convention as the tier-2 entry, the baseline JIT and the invoke thunks.
// Tier 2 calling tier 2 and tier 2 calling baseline code are checked
// against the interpreter; a vector that would go on the stack is a
// compile error.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::codegen;

#if defined(__x86_64__) || defined(_M_X64)

namespace {

std::unique_ptr<Module> parse_t2v(const std::string& src) {
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

// Bit-exact, except that a NaN float lane matches any NaN.
bool same_t2v(const RuntimeValue& a, const RuntimeValue& b, Type t) {
    if (t.is_v128()) {
        const uint8_t* x = static_cast<const uint8_t*>(a.v128_bytes());
        const uint8_t* y = static_cast<const uint8_t*>(b.v128_bytes());
        const size_t esz = t.element_type().size_in_bytes();
        for (size_t off = 0; off < 16; off += esz) {
            if (std::memcmp(x + off, y + off, esz) == 0) continue;
            if (!t.element_type().is_float()) return false;
            double p, q;
            if (esz == 4) {
                float fp, fq;
                std::memcpy(&fp, x + off, 4);
                std::memcpy(&fq, y + off, 4);
                p = fp; q = fq;
            } else {
                std::memcpy(&p, x + off, 8);
                std::memcpy(&q, y + off, 8);
            }
            if (!(std::isnan(p) && std::isnan(q))) return false;
        }
        return true;
    }
    return a.raw_bits() == b.raw_bits();
}

const double kNaN = std::numeric_limits<double>::quiet_NaN();

std::vector<RuntimeValue> f64x2_samples() {
    return {RuntimeValue::from_f64x2(1.5, -0.0), RuntimeValue::from_f64x2(0.0, kNaN),
            RuntimeValue::from_f64x2(-3.25, 2.5), RuntimeValue::from_f64x2(16.0, 1e-300)};
}

std::vector<RuntimeValue> i32x4_samples() {
    return {RuntimeValue::from_i32x4(1, -1, 0, 7), RuntimeValue::from_i32x4(INT32_MIN, -1, 5, 0),
            RuntimeValue::from_i32x4(INT32_MAX, 3, -7, 100000), RuntimeValue::from_i32x4(-1, 0, 2, INT32_MIN)};
}

std::vector<std::vector<RuntimeValue>> pairs(const std::vector<RuntimeValue>& xs) {
    std::vector<std::vector<RuntimeValue>> out;
    for (const auto& a : xs) for (const auto& b : xs) out.push_back({a, b});
    return out;
}

// Runs `entry` of `jit` against the interpreter on `ref` for every argument set.
int diff_t2v(JitExecutionEngine& jit, Module& ref, const std::string& entry,
             const std::vector<std::vector<RuntimeValue>>& arg_sets) {
    const Type rt = ref.get_function(entry)->return_type();
    Interpreter interp;
    int mismatches = 0;
    for (const auto& args : arg_sets) {
        RuntimeValue expect = interp.run(ref, entry, args);
        RuntimeValue got = jit.invoke(entry, args);
        if (!same_t2v(expect, got, rt)) {
            ++mismatches;
            std::cerr << "tier-2/interpreter mismatch in " << entry << "\n";
        }
    }
    return mismatches;
}

// Callees mixing vector and scalar parameters: positions 1 and 3 are XMM
// registers on Win64 (by position) and XMM0/XMM2 on SysV (the f64 takes XMM1).
const char* kF64Callee = R"(
func @t2v_f64_callee(%0: i64, %1: f64x2, %2: f64, %3: f64x2) -> f64x2 {
entry:
  %f = sitofp_f64_i64 %0
  %g = add.f64 %f, %2
  %b: f64x2 = vbroadcast.f64x2 %g
  %m: f64x2 = vmul %1, %b
  %r: f64x2 = vsub %m, %3
  ret %r
}
)";

const char* kF64Caller = R"(
func @t2v_f64_caller(%0: f64x2, %1: f64x2) -> f64x2 {
entry:
  %k = iconst.i64 3
  %h = fconst.f64 0.5
  %r: f64x2 = call.f64x2 @t2v_f64_callee(%k, %1, %h, %0)
  %s: f64x2 = vadd %r, %0
  ret %s
}
)";

// Four vector arguments fill XMM0-3 on both conventions; the result feeds
// a scalar, so a vector result and a scalar result both cross a call.
const char* kI32Callee = R"(
func @t2v_i32_callee(%0: i32x4, %1: i32x4, %2: i32x4, %3: i32x4) -> i32x4 {
entry:
  %a: i32x4 = vadd %0, %1
  %b: i32x4 = vsub %2, %3
  %r: i32x4 = vsub %a, %b
  ret %r
}
)";

const char* kI32Caller = R"(
func @t2v_i32_caller(%0: i32x4, %1: i32x4) -> i64 {
entry:
  %r: i32x4 = call.i32x4 @t2v_i32_callee(%0, %1, %1, %0)
  %q: i32x4 = call.i32x4 @t2v_i32_callee(%r, %0, %1, %r)
  %a: i32 = vextract_lane %q, 0
  %b: i32 = vextract_lane %q, 3
  %c = add.i32 %a, %b
  %d = sext_i64 %c
  ret %d
}
)";

} // namespace

TEST_CASE("Tier-2 vectors - tier 2 calls tier 2 with vector arguments and results") {
    auto mod = parse_t2v(std::string("module @m\n") + kF64Callee + kF64Caller + kI32Callee + kI32Caller);
    JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    CHECK_EQ(diff_t2v(jit, *mod, "t2v_f64_caller", pairs(f64x2_samples())), 0);
    CHECK_EQ(diff_t2v(jit, *mod, "t2v_i32_caller", pairs(i32x4_samples())), 0);
}

TEST_CASE("Tier-2 vectors - tier 2 calls baseline code with vector arguments and results") {
    auto ref = parse_t2v(std::string("module @m\n") + kF64Callee + kF64Caller + kI32Callee + kI32Caller);

    BaselineJitCompiler baseline;
    auto f64_callee = baseline.compile(*ref->get_function("t2v_f64_callee"));
    auto i32_callee = baseline.compile(*ref->get_function("t2v_i32_callee"));

    auto callers = parse_t2v(std::string("module @c\n") + kF64Caller + kI32Caller);
    JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("t2v_f64_callee", f64_callee.entry_point());
    jit.register_external_symbol("t2v_i32_callee", i32_callee.entry_point());
    REQUIRE(jit.compile_and_load(*callers));
    CHECK_EQ(diff_t2v(jit, *ref, "t2v_f64_caller", pairs(f64x2_samples())), 0);
    CHECK_EQ(diff_t2v(jit, *ref, "t2v_i32_caller", pairs(i32x4_samples())), 0);
}

TEST_CASE("Tier-2 vectors - a vector argument or parameter on the stack is a compile error") {
    // A fifth argument is on the stack on Win64, the ninth on SysV.
    const char* sources[] = {
        R"(module @m
func @t2v_stack_caller(%0: f32x4) -> f32x4 {
entry:
  %r: f32x4 = call.f32x4 @t2v_ext(%0, %0, %0, %0, %0, %0, %0, %0, %0)
  ret %r
}
)",
        R"(module @m
func @t2v_stack_param(%0: f32x4, %1: f32x4, %2: f32x4, %3: f32x4, %4: f32x4, %5: f32x4, %6: f32x4, %7: f32x4, %8: f32x4) -> f32x4 {
entry:
  ret %8
}
)",
    };
    for (const char* src : sources) {
        auto mod = parse_t2v(src);
        JitExecutionEngine jit(Target::host());
        jit.register_external_symbol("t2v_ext", reinterpret_cast<void*>(&same_t2v));
        bool threw = false;
        try {
            (void)jit.compile_and_load(*mod);
        } catch (const UnsupportedOperation& e) {
            threw = true;
            CHECK(e.operation().find("on the stack") != std::string::npos);
        }
        CHECK(threw);
    }
}

#endif
