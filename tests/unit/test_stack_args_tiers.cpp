// Arguments past the argument registers, across the native tiers: tier 2 as
// a leaf and as a caller, the baseline tier, and the invoke thunk that calls
// either from C++, checked against the interpreter. On AArch64 a leaf sets up
// no frame, so its incoming stack arguments are SP-relative (they were read
// through the caller's FP), and on Apple they are packed at their natural
// size (the invoke thunk wrote 8-byte words). Also: a native trap inside
// DiffFuzzer::run_protected is reported, not fatal, on every host (AArch64
// traps with brk, which arrives as SIGTRAP).
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/code_buffer.hpp>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::codegen;

namespace {

std::unique_ptr<Module> parse_sa(const std::string& src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << "parse failed:\n" << diag.format_all() << "\n" << src << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n";
    REQUIRE(ok);
    return mod;
}

// Leaves reading their last parameters, of each width, past eight integer
// and eight float registers; and callers passing them.
const char* kModule = R"(module @m
func @sa_i64(%0: i64, %1: i64, %2: i64, %3: i64, %4: i64, %5: i64, %6: i64, %7: i64, %8: i64, %9: i64) -> i64 {
entry:
  %r = sub.i64 %8, %9
  ret %r
}
func @sa_i32(%0: i32, %1: i32, %2: i32, %3: i32, %4: i32, %5: i32, %6: i32, %7: i32, %8: i32, %9: i32, %10: i32) -> i32 {
entry:
  %a = mul.i32 %8, %9
  %r = sub.i32 %a, %10
  ret %r
}
func @sa_mixed(%0: f64, %1: f64, %2: f64, %3: f64, %4: f64, %5: f64, %6: f64, %7: f64, %8: f32, %9: i64, %10: i64, %11: i64, %12: i64, %13: i64, %14: i64, %15: i64, %16: i32, %17: f64, %18: i32) -> f64 {
entry:
  %a = fpext_f64_f32 %8
  %b = sitofp_f64_i32 %16
  %c = sitofp_f64_i32 %18
  %s = add.f64 %a, %b
  %t = mul.f64 %s, %17
  %u = sub.f64 %t, %c
  %w = sitofp_f64_i64 %15
  %r = add.f64 %u, %w
  ret %r
}
func @sa_call_i32(%0: i32, %1: i32) -> i32 {
entry:
  %r = call.i32 @sa_i32(%0, %0, %0, %0, %0, %0, %0, %0, %0, %1, %0)
  ret %r
}
func @sa_call_mixed(%0: f64, %1: i64) -> f64 {
entry:
  %h = fptrunc_f32_f64 %0
  %k = trunc_i32 %1
  %r = call.f64 @sa_mixed(%0, %0, %0, %0, %0, %0, %0, %0, %h, %1, %1, %1, %1, %1, %1, %1, %k, %0, %k)
  ret %r
}
)";

std::vector<RuntimeValue> i32s(std::initializer_list<int32_t> xs) {
    std::vector<RuntimeValue> v;
    for (int32_t x : xs) v.push_back(RuntimeValue::from_i32(x));
    return v;
}

std::vector<RuntimeValue> mixed_args(double d, float f, int64_t i, int32_t k, double e, int32_t m) {
    std::vector<RuntimeValue> v;
    for (int j = 0; j < 8; ++j) v.push_back(RuntimeValue::from_f64(d + j));
    v.push_back(RuntimeValue::from_f32(f));
    for (int j = 0; j < 7; ++j) v.push_back(RuntimeValue::from_i64(i + j));
    v.push_back(RuntimeValue::from_i32(k));
    v.push_back(RuntimeValue::from_f64(e));
    v.push_back(RuntimeValue::from_i32(m));
    return v;
}

bool same(const RuntimeValue& a, const RuntimeValue& b, Type t) {
    if (t.kind() == TypeKind::F64) return a.as_f64() == b.as_f64();
    if (t.kind() == TypeKind::I32) return a.as_i32() == b.as_i32();
    return a.raw_bits() == b.raw_bits();
}

struct Case {
    const char* fn;
    std::vector<RuntimeValue> args;
};

std::vector<Case> cases() {
    std::vector<Case> out;
    std::vector<RuntimeValue> ten;
    for (int j = 0; j < 10; ++j) ten.push_back(RuntimeValue::from_i64(100 * j - 7));
    out.push_back({"sa_i64", ten});
    out.push_back({"sa_i32", i32s({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})});
    out.push_back({"sa_i32", i32s({-1, -2, -3, -4, -5, -6, -7, -8, -9, INT32_MIN, 0x7fffffff})});
    out.push_back({"sa_mixed", mixed_args(0.5, 2.25f, 40, -3, 1.5, 17)});
    out.push_back({"sa_mixed", mixed_args(-8.0, -0.125f, -1, 1 << 20, -2.0, -99)});
    out.push_back({"sa_call_i32", i32s({6, -4})});
    out.push_back({"sa_call_mixed", {RuntimeValue::from_f64(1.75), RuntimeValue::from_i64(-12)}});
    return out;
}

} // namespace

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

TEST_CASE("Stack arguments - tier 2 leaves and callers agree with the interpreter") {
    auto mod = parse_sa(kModule);
    JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    Interpreter interp;
    for (const Case& c : cases()) {
        const Type rt = mod->get_function(c.fn)->return_type();
        const RuntimeValue want = interp.run(*mod, c.fn, c.args);
        const RuntimeValue got = jit.invoke(c.fn, c.args);
        if (!same(want, got, rt)) std::cerr << "tier 2 mismatch in " << c.fn << "\n";
        CHECK(same(want, got, rt));
    }
}

TEST_CASE("Stack arguments - baseline code agrees with the interpreter and calls tier 2") {
    auto mod = parse_sa(kModule);
    Interpreter interp;

    // Baseline leaves, baseline callers of them.
    BaselineJitCompiler baseline;
    std::vector<BaselineCompiledFunction> compiled;
    for (const char* name : {"sa_i64", "sa_i32", "sa_mixed", "sa_call_i32", "sa_call_mixed"}) {
        compiled.push_back(baseline.compile(*mod->get_function(name)));
        baseline.register_external_symbol(name, compiled.back().entry_point());
    }
    for (const Case& c : cases()) {
        const Type rt = mod->get_function(c.fn)->return_type();
        const BaselineCompiledFunction* f = nullptr;
        for (const auto& cf : compiled) if (cf.name() == c.fn) f = &cf;
        REQUIRE(f != nullptr);
        const RuntimeValue want = interp.run(*mod, c.fn, c.args);
        const RuntimeValue got = f->invoke(c.args);
        if (!same(want, got, rt)) std::cerr << "baseline mismatch in " << c.fn << "\n";
        CHECK(same(want, got, rt));
    }

    // Baseline callers of tier-2 leaves: both sides of the stack layout.
    JitExecutionEngine jit(Target::host());
    auto leaves = parse_sa(kModule);
    REQUIRE(jit.compile_and_load(*leaves));
    BaselineJitCompiler callers;
    callers.register_external_symbol("sa_i32", jit.get_symbol_address("sa_i32"));
    callers.register_external_symbol("sa_mixed", jit.get_symbol_address("sa_mixed"));
    auto call_i32 = callers.compile(*mod->get_function("sa_call_i32"));
    auto call_mixed = callers.compile(*mod->get_function("sa_call_mixed"));
    for (const Case& c : cases()) {
        const BaselineCompiledFunction* f = nullptr;
        if (std::string(c.fn) == "sa_call_i32") f = &call_i32;
        if (std::string(c.fn) == "sa_call_mixed") f = &call_mixed;
        if (!f) continue;
        const Type rt = mod->get_function(c.fn)->return_type();
        CHECK(same(interp.run(*mod, c.fn, c.args), f->invoke(c.args), rt));
    }
}

TEST_CASE("Native traps - a division by zero in JIT code is reported by run_protected") {
    // x64 faults with #DE (SIGFPE); AArch64 has no divide fault and traps
    // with brk, which must be caught and classified the same way.
    auto mod = parse_sa(R"(module @m
func @sa_div(%0: i64, %1: i64) -> i64 {
entry:
  %r = sdiv.i64 %0, %1
  ret %r
}
)");
    BaselineJitCompiler baseline;
    auto div = baseline.compile(*mod->get_function("sa_div"));
    auto fn = div.get_function_ptr<int64_t (*)(int64_t, int64_t)>();
    CHECK_EQ(fn(84, 2), int64_t{42});
    std::string fault;
    const bool ok = fuzz::DiffFuzzer::run_protected([fn] { (void)fn(1, 0); }, fault);
    CHECK(!ok);
    CHECK(fault.find("Integer Divide by Zero") != std::string::npos);

    // Still usable after the recovery.
    CHECK_EQ(fn(-9, 3), int64_t{-3});
}

#endif

TEST_CASE("AArch64 Encoder - sign-extending loads keep their sign in every addressing mode") {
    // ldursw / ldursb / ldursh (negative or unaligned offsets) and the
    // register-offset forms were emitted as plain zero-extending loads.
    using namespace brass::aarch64;
    CodeBuffer buf;
    AArch64Encoder enc(buf);
    enc.ldrsw(GPR::X0, ptr(GPR::X1, -4));                                                // ldursw x0, [x1, #-4]
    enc.ldrsb(GPR::X0, ptr(GPR::X1, -1));                                                // ldursb x0, [x1, #-1]
    enc.ldrsh(GPR::X0, ptr(GPR::X1, 3));                                                 // ldursh x0, [x1, #3]
    enc.ldrsw(GPR::X0, MemAddress::base_index(GPR::X1, GPR::X2, ExtendType::UXTX, 2));   // ldrsw x0, [x1, x2, lsl #2]
    enc.ldrsw(GPR::X0, ptr(GPR::X1, 8));                                                 // ldrsw x0, [x1, #8]
    enc.ldr32(GPR::X0, ptr(GPR::X1, -4));                                                // ldur w0, [x1, #-4]
    REQUIRE(buf.size() == 24);
    auto word = [&](size_t i) {
        uint32_t w = 0;
        std::memcpy(&w, buf.data() + 4 * i, 4);
        return w;
    };
    CHECK_EQ(word(0), 0xB89FC020u);
    CHECK_EQ(word(1), 0x389FF020u);
    CHECK_EQ(word(2), 0x78803020u);
    CHECK_EQ(word(3), 0xB8A27820u);
    CHECK_EQ(word(4), 0xB9800820u);
    CHECK_EQ(word(5), 0xB85FC020u);
}
