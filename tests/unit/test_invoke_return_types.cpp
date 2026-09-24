// Invoke paths, scalar signatures: a native function's result is converted by
// its return type in one place (native_return_value), and each argument is
// passed by its parameter's type, whatever the arity. Tier-2 invoke, the
// embedding API's CompiledModule::invoke and the baseline JIT's invoke are
// checked against the interpreter for every scalar return type, 0 to 7
// arguments of mixed types, and negative and boundary values. An i32 result
// was zero-extended into an i64 by the 1- and 2-argument fast paths of
// tier-2 invoke (-999995 came back as 4293967301), and an f32 argument of the
// 1-argument path was passed as a double.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/embedding/embedding.hpp>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::codegen;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

std::unique_ptr<Module> parse_irt(const std::string& src) {
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

const Type kRetTypes[] = {Type::i32(), Type::i64(), Type::f32(), Type::f64(), Type::ptr()};
// Parameter types after the first (which is the return type), in rotation.
const Type kParamRotation[] = {Type::i32(), Type::f32(), Type::i64(), Type::f64(), Type::ptr()};
constexpr size_t kMaxArity = 7;

std::string type_name(Type t) { return to_string(t); }

std::string fn_name(Type ret, size_t arity) {
    return "irt_" + type_name(ret) + "_" + std::to_string(arity);
}

std::vector<Type> param_types(Type ret, size_t arity) {
    std::vector<Type> ps;
    for (size_t k = 0; k < arity; ++k) {
        ps.push_back(k == 0 ? ret : kParamRotation[(k - 1) % 5]);
    }
    return ps;
}

// fn(p0: R, p1, ...) -> R: an integer or float result is a constant plus
// every parameter of type R; a ptr result is p0. With no parameters, the
// result is the constant (a ptr function then takes one parameter).
std::string function_src(Type ret, size_t arity) {
    const std::vector<Type> ps = param_types(ret, arity);
    std::string s = "func @" + fn_name(ret, arity) + "(";
    for (size_t k = 0; k < ps.size(); ++k) {
        if (k) s += ", ";
        s += "%p" + std::to_string(k) + ": " + type_name(ps[k]);
    }
    s += ") -> " + type_name(ret) + " {\nbb0:\n";
    if (ret.is_pointer()) {
        return s + "  ret %p0\n}\n";
    }
    switch (ret.kind()) {
        case TypeKind::I32: s += "  %a0 = iconst.i32 -1000000\n"; break;
        case TypeKind::I64: s += "  %a0 = iconst.i64 -5000000000\n"; break;
        case TypeKind::F32: s += "  %a0 = fconst.f32 -0.5\n"; break;
        default: s += "  %a0 = fconst.f64 -0.25\n"; break;
    }
    size_t acc = 0;
    for (size_t k = 0; k < ps.size(); ++k) {
        if (ps[k] != ret) continue;
        s += "  %a" + std::to_string(acc + 1) + " = add %a" + std::to_string(acc) + ", %p" + std::to_string(k) + "\n";
        ++acc;
    }
    return s + "  ret %a" + std::to_string(acc) + "\n}\n";
}

std::string module_src() {
    std::string s = "module @irt\n";
    for (Type ret : kRetTypes) {
        for (size_t n = ret.is_pointer() ? 1 : 0; n <= kMaxArity; ++n) s += function_src(ret, n);
    }
    return s;
}

std::vector<RuntimeValue> samples(Type t) {
    switch (t.kind()) {
        case TypeKind::I32:
            return {RuntimeValue::from_i32(std::numeric_limits<int32_t>::min()), RuntimeValue::from_i32(-1),
                    RuntimeValue::from_i32(0), RuntimeValue::from_i32(std::numeric_limits<int32_t>::max()),
                    RuntimeValue::from_i32(5)};
        case TypeKind::I64:
            return {RuntimeValue::from_i64(std::numeric_limits<int64_t>::min()), RuntimeValue::from_i64(-1),
                    RuntimeValue::from_i64(std::numeric_limits<int64_t>::max()), RuntimeValue::from_i64(-4294967296LL),
                    RuntimeValue::from_i64(7)};
        case TypeKind::F32:
            return {RuntimeValue::from_f32(-0.0f), RuntimeValue::from_f32(-1.5f),
                    RuntimeValue::from_f32(std::numeric_limits<float>::max()),
                    RuntimeValue::from_f32(std::numeric_limits<float>::denorm_min()), RuntimeValue::from_f32(2.25f)};
        case TypeKind::F64:
            return {RuntimeValue::from_f64(-0.0), RuntimeValue::from_f64(-2.5),
                    RuntimeValue::from_f64(std::numeric_limits<double>::max()),
                    RuntimeValue::from_f64(std::numeric_limits<double>::denorm_min()), RuntimeValue::from_f64(1e-300)};
        default:
            return {RuntimeValue::from_ptr(static_cast<uintptr_t>(0xFFFFFFFF80000010ULL)),
                    RuntimeValue::from_ptr(static_cast<uintptr_t>(0x0000000080000000ULL)),
                    RuntimeValue::from_ptr(static_cast<uintptr_t>(0)),
                    RuntimeValue::from_ptr(static_cast<uintptr_t>(0x7FFFFFFFFFFFFFF8ULL)),
                    RuntimeValue::from_ptr(static_cast<uintptr_t>(0x1000))};
    }
}

// Argument sets for fn: sample i of each parameter's type, i shifted by the
// parameter's position so that values mix across positions.
std::vector<std::vector<RuntimeValue>> arg_sets(Type ret, size_t arity) {
    const std::vector<Type> ps = param_types(ret, arity);
    std::vector<std::vector<RuntimeValue>> sets;
    for (size_t i = 0; i < 5; ++i) {
        std::vector<RuntimeValue> args;
        for (size_t k = 0; k < ps.size(); ++k) args.push_back(samples(ps[k])[(i + k) % 5]);
        sets.push_back(std::move(args));
        if (arity == 0) break;
    }
    return sets;
}

bool same_irt(const RuntimeValue& a, const RuntimeValue& b) {
    return a.kind() == b.kind() && a.raw_bits() == b.raw_bits();
}

std::string show(const RuntimeValue& v) {
    return to_string(v.type()) + " bits 0x" + [&] {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(v.raw_bits()));
        return std::string(buf);
    }();
}

// Runs every function through `call` against the interpreter.
template <typename Call>
int diff_irt(const char* engine, Call&& call) {
    auto ref = parse_irt(module_src());
    Interpreter interp;
    int mismatches = 0;
    for (Type ret : kRetTypes) {
        for (size_t n = ret.is_pointer() ? 1 : 0; n <= kMaxArity; ++n) {
            const std::string name = fn_name(ret, n);
            for (const auto& args : arg_sets(ret, n)) {
                RuntimeValue expect = interp.run(*ref, name, args);
                RuntimeValue got = call(name, args);
                if (!same_irt(expect, got)) {
                    ++mismatches;
                    std::cerr << engine << "/interpreter mismatch in " << name << ": expected "
                              << show(expect) << ", got " << show(got) << "\n";
                }
            }
        }
    }
    return mismatches;
}

} // namespace

TEST_CASE("Invoke return types - tier-2 invoke matches the interpreter") {
    auto mod = parse_irt(module_src());
    JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    CHECK_EQ(diff_irt("tier-2", [&](const std::string& name, const std::vector<RuntimeValue>& args) {
        return jit.invoke(name, args);
    }), 0);
}

TEST_CASE("Invoke return types - the embedding API matches the interpreter") {
    auto mod = parse_irt(module_src());
    HostEngine engine;
    auto compiled = engine.compile(*mod);
    REQUIRE(compiled != nullptr);
    CHECK_EQ(diff_irt("embedding", [&](const std::string& name, const std::vector<RuntimeValue>& args) {
        return compiled->invoke(name, args);
    }), 0);
}

TEST_CASE("Invoke return types - baseline invoke matches the interpreter") {
    auto mod = parse_irt(module_src());
    BaselineJitCompiler baseline;
    std::vector<std::pair<std::string, BaselineCompiledFunction>> fns;
    for (const Function* f : mod->functions()) fns.emplace_back(std::string(f->name()), baseline.compile(*f));
    CHECK_EQ(diff_irt("baseline", [&](const std::string& name, const std::vector<RuntimeValue>& args) {
        for (const auto& [n, f] : fns) {
            if (n == name) return f.invoke(args);
        }
        throw std::runtime_error("no baseline function " + name);
    }), 0);
}

TEST_CASE("Invoke return types - arguments are passed by parameter type") {
    auto mod = parse_irt(R"(module @args
func @sx(%x: i32) -> i64 {
bb0:
  %r = sext.i64 %x
  ret %r
}
func @zx(%x: i32) -> i64 {
bb0:
  %r = zext.i64 %x
  ret %r
}
func @zx5(%a: i64, %b: i64, %c: i64, %d: i64, %x: i32) -> i64 {
bb0:
  %r = zext.i64 %x
  ret %r
}
func @ext(%x: f32) -> f64 {
bb0:
  %r = fpext.f64.f32 %x
  ret %r
}
func @ext2(%y: i64, %x: f32) -> f64 {
bb0:
  %r = fpext.f64.f32 %x
  ret %r
}
func @half(%x: f64) -> f64 {
bb0:
  %h = fconst.f64 0.5
  %r = mul %x, %h
  ret %r
}
)");
    JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    // An i64 value for an i32 parameter: only its low 32 bits are the argument.
    const RuntimeValue wide = RuntimeValue::from_i64(static_cast<int64_t>(0x12345678FFFFFFFBULL));
    CHECK_EQ(jit.invoke("sx", {wide}).as_i64(), -5);
    CHECK_EQ(jit.invoke("zx", {wide}).as_i64(), 4294967291LL);
    CHECK_EQ(jit.invoke("zx5", {RuntimeValue::from_i64(1), RuntimeValue::from_i64(2), RuntimeValue::from_i64(3),
                                RuntimeValue::from_i64(4), wide}).as_i64(), 4294967291LL);
    // An f32 parameter takes an f32 or f64 value as a float.
    CHECK_EQ(jit.invoke("ext", {RuntimeValue::from_f32(1.5f)}).as_f64(), 1.5);
    CHECK_EQ(jit.invoke("ext", {RuntimeValue::from_f64(-2.25)}).as_f64(), -2.25);
    CHECK_EQ(jit.invoke("ext2", {RuntimeValue::from_i64(9), RuntimeValue::from_f64(3.5)}).as_f64(), 3.5);
    // An f64 parameter takes an integer value as its number.
    CHECK_EQ(jit.invoke("half", {RuntimeValue::from_i64(-7)}).as_f64(), -3.5);
    // A float value for an integer parameter is an error.
    bool threw = false;
    try {
        jit.invoke("sx", {RuntimeValue::from_f64(1.0)});
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("Invoke return types - narrow integer results are sign-extended from their width") {
    const uint8_t vec[16] = {0};
    // Garbage above the result's width does not reach the value.
    CHECK_EQ(native_return_value(Type::i8(), 0xDEADBEEF000000FFULL, vec).as_i64(), -1);
    CHECK_EQ(native_return_value(Type::i8(), 0x000000000000017FULL, vec).as_i64(), 127);
    CHECK_EQ(native_return_value(Type::i16(), 0xDEADBEEF00008000ULL, vec).as_i64(), -32768);
    CHECK_EQ(native_return_value(Type::i32(), 0xDEADBEEFFFF0BDC5ULL, vec).as_i64(), -999995);
    CHECK(native_return_value(Type::i32(), 0xDEADBEEFFFF0BDC5ULL, vec).is_i32());
    CHECK_EQ(native_return_value(Type::i64(), 0xFFFFFFFFFFF0BDC5ULL, vec).as_i64(), -999995);
    // A 256-bit vector does not fit the captured registers.
    bool threw = false;
    try {
        native_return_value(Type::f32x8(), 0, vec);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
}

#endif
