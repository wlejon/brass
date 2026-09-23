// x64 baseline JIT, 128-bit vectors: every vector opcode, vector select /
// loads / stores / block parameters, and vector calls between baseline
// functions and into tier 2, checked against the interpreter.
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

std::unique_ptr<Module> parse_vec_or_fail(const std::string& src) {
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
bool same_value(const RuntimeValue& a, const RuntimeValue& b, Type t) {
    if (t.is_v128()) {
        const uint8_t* x = static_cast<const uint8_t*>(a.v128_bytes());
        const uint8_t* y = static_cast<const uint8_t*>(b.v128_bytes());
        const size_t esz = t.element_type().size_in_bytes();
        for (size_t off = 0; off < 16; off += esz) {
            if (std::memcmp(x + off, y + off, esz) == 0) continue;
            if (!t.element_type().is_float()) return false;
            if (esz == 4) {
                float p, q;
                std::memcpy(&p, x + off, 4);
                std::memcpy(&q, y + off, 4);
                if (!(std::isnan(p) && std::isnan(q))) return false;
            } else {
                double p, q;
                std::memcpy(&p, x + off, 8);
                std::memcpy(&q, y + off, 8);
                if (!(std::isnan(p) && std::isnan(q))) return false;
            }
        }
        return true;
    }
    if (t.is_float()) {
        double x = t.kind() == TypeKind::F32 ? a.as_f32() : a.as_f64();
        double y = t.kind() == TypeKind::F32 ? b.as_f32() : b.as_f64();
        if (std::isnan(x) || std::isnan(y)) return std::isnan(x) && std::isnan(y);
        return std::memcmp(&x, &y, 8) == 0;
    }
    if (t.is_integer() && t.size_in_bytes() <= 4) return a.as_i32() == b.as_i32();
    return a.raw_bits() == b.raw_bits();
}

std::string hex_value(const RuntimeValue& v, Type t) {
    static const char* digits = "0123456789abcdef";
    std::string s;
    if (t.is_v128()) {
        const uint8_t* p = static_cast<const uint8_t*>(v.v128_bytes());
        for (int i = 15; i >= 0; --i) { s += digits[p[i] >> 4]; s += digits[p[i] & 15]; }
        return s;
    }
    return std::to_string(static_cast<int64_t>(v.raw_bits()));
}

// Baseline-compiles `order` (callees first, each registered for the next)
// and checks `entry` against the interpreter on every argument set.
int diff_vec(Module& mod, const std::vector<std::string>& order, const std::string& entry,
             const std::vector<std::vector<RuntimeValue>>& arg_sets, BaselineJitCompiler* shared = nullptr) {
    BaselineJitCompiler local;
    BaselineJitCompiler& compiler = shared ? *shared : local;
    std::vector<BaselineCompiledFunction> compiled;
    for (const auto& name : order) {
        const Function* f = mod.get_function(name);
        REQUIRE(f != nullptr);
        compiled.push_back(compiler.compile(*f));
        compiler.register_external_symbol(name, compiled.back().entry_point());
    }
    const BaselineCompiledFunction* target = nullptr;
    for (const auto& c : compiled) if (c.name() == entry) target = &c;
    REQUIRE(target != nullptr);
    const Type rt = mod.get_function(entry)->return_type();
    int mismatches = 0;
    Interpreter interp;
    for (const auto& args : arg_sets) {
        RuntimeValue expect = interp.run(mod, entry, args);
        RuntimeValue got = target->invoke(args);
        if (!same_value(expect, got, rt)) {
            ++mismatches;
            std::cerr << "baseline/interpreter mismatch in " << entry << ": interp=" << hex_value(expect, rt)
                      << " baseline=" << hex_value(got, rt) << "\n";
        }
    }
    return mismatches;
}

const float kNaNf = std::numeric_limits<float>::quiet_NaN();
const float kInff = std::numeric_limits<float>::infinity();
const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

// A handful of vectors of `t` with signed zeros, NaN, infinities, integer
// extremes and zeros (so integer division meets x / 0 and INT_MIN / -1).
std::vector<RuntimeValue> sample_vectors(Type t) {
    switch (t.kind()) {
        case TypeKind::F32x4:
            return {RuntimeValue::from_f32x4(1.5f, -2.25f, 0.0f, -0.0f),
                    RuntimeValue::from_f32x4(-0.0f, 3.0f, kNaNf, 1e30f),
                    RuntimeValue::from_f32x4(kInff, -kInff, 7.0f, 0.5f),
                    RuntimeValue::from_f32x4(4.0f, 0.25f, -1.0f, 9.0f)};
        case TypeKind::F64x2:
            return {RuntimeValue::from_f64x2(1.5, -0.0), RuntimeValue::from_f64x2(0.0, kNaN),
                    RuntimeValue::from_f64x2(-kInf, 2.5), RuntimeValue::from_f64x2(16.0, 1e-300)};
        case TypeKind::I32x4:
            return {RuntimeValue::from_i32x4(1, -1, 0, 7),
                    RuntimeValue::from_i32x4(INT32_MIN, -1, 5, 0),
                    RuntimeValue::from_i32x4(INT32_MAX, 3, -7, 100000),
                    RuntimeValue::from_i32x4(-1, 0, 2, INT32_MIN)};
        default:
            return {RuntimeValue::from_i64x2(1, -1), RuntimeValue::from_i64x2(INT64_MIN, 0),
                    RuntimeValue::from_i64x2(-1, INT64_MAX), RuntimeValue::from_i64x2(0x123456789LL, -3)};
    }
}

std::vector<std::vector<RuntimeValue>> vec_tuples(Type t, int arity) {
    const auto xs = sample_vectors(t);
    std::vector<std::vector<RuntimeValue>> out;
    for (const auto& a : xs) {
        if (arity == 1) { out.push_back({a}); continue; }
        for (const auto& b : xs) {
            if (arity == 2) { out.push_back({a, b}); continue; }
            for (const auto& c : xs) out.push_back({a, b, c});
        }
    }
    return out;
}

const char* kTypes[] = {"f32x4", "f64x2", "i32x4", "i64x2"};

Type type_named(const std::string& n) {
    if (n == "f32x4") return Type::f32x4();
    if (n == "f64x2") return Type::f64x2();
    if (n == "i32x4") return Type::i32x4();
    return Type::i64x2();
}

std::string elem_name(const std::string& vt) {
    if (vt == "f32x4") return "f32";
    if (vt == "f64x2") return "f64";
    if (vt == "i32x4") return "i32";
    return "i64";
}

} // namespace

TEST_CASE("Baseline JIT vectors - arithmetic, bitwise and unary ops agree with the interpreter") {
    const char* binops[] = {"vadd", "vsub", "vmul", "vdiv", "vmin", "vmax", "vand", "vor", "vxor"};
    for (const char* ty : kTypes) {
        const std::string t = ty;
        const bool flt = t[0] == 'f';
        const Type vt = type_named(t);
        auto run = [&](const std::string& op, int arity) {
            const std::string name = "bl_v_" + op + "_" + t;
            std::string params, operands;
            for (int i = 0; i < arity; ++i) {
                params += (i ? ", %" : "%") + std::to_string(i) + ": " + t;
                operands += (i ? ", %" : "%") + std::to_string(i);
            }
            const std::string src = "module @m\nfunc @" + name + "(" + params + ") -> " + t +
                                    " {\nentry:\n  %r: " + t + " = " + op + " " + operands + "\n  ret %r\n}\n";
            auto mod = parse_vec_or_fail(src);
            CHECK_EQ(diff_vec(*mod, {name}, name, vec_tuples(vt, arity)), 0);
        };
        for (const char* op : binops) run(op, 2);
        run("vneg", 1);
        run("vnot", 1);
        if (flt) {
            run("vsqrt", 1);
            run("vfma", 3);
        }
    }
}

TEST_CASE("Baseline JIT vectors - lanes, broadcast, shuffle and zero agree with the interpreter") {
    for (const char* ty : kTypes) {
        const std::string t = ty;
        const std::string e = elem_name(t);
        const Type vt = type_named(t);
        const uint32_t lanes = vt.vector_lanes();
        for (uint32_t lane = 0; lane < lanes; ++lane) {
            const std::string l = std::to_string(lane);
            // extract lane, broadcast it, insert it into lane (lane+1) % n.
            const std::string name = "bl_v_lane" + l + "_" + t;
            const std::string src =
                "module @m\nfunc @" + name + "(%0: " + t + ", %1: " + t + ") -> " + t + " {\nentry:\n"
                "  %x: " + e + " = vextract_lane %0, " + l + "\n"
                "  %b: " + t + " = vbroadcast." + t + " %x\n"
                "  %y: " + e + " = vextract_lane %1, " + l + "\n"
                "  %i: " + t + " = vinsert_lane %b, %y, " + std::to_string((lane + 1) % lanes) + "\n"
                "  %z: " + t + " = vzero." + t + "\n"
                "  %r: " + t + " = vor %i, %z\n"
                "  ret %r\n}\n";
            auto mod = parse_vec_or_fail(src);
            CHECK_EQ(diff_vec(*mod, {name}, name, vec_tuples(vt, 2)), 0);
        }
        for (uint32_t mask : {0x00u, 0x1Bu, 0xE4u, 0x4Eu, 0x93u, 0x01u, 0x02u, 0x03u}) {
            const std::string name = "bl_v_shuf" + std::to_string(mask) + "_" + t;
            const std::string src =
                "module @m\nfunc @" + name + "(%0: " + t + ", %1: " + t + ") -> " + t + " {\nentry:\n"
                "  %r: " + t + " = vshuffle %0, %1, " + std::to_string(mask) + "\n  ret %r\n}\n";
            auto mod = parse_vec_or_fail(src);
            CHECK_EQ(diff_vec(*mod, {name}, name, vec_tuples(vt, 2)), 0);
        }
    }
}

TEST_CASE("Baseline JIT vectors - memory, select, loops over vector block parameters, mixed scalar code") {
    auto mod = parse_vec_or_fail(R"(module @m
func @bl_v_mix(%0: f32x4, %1: i64, %2: f64, %3: f32x4) -> f64 {
entry:
  %buf = alloca 64, 16
  vstore.f32x4 %buf, 16, %0
  %v: f32x4 = vload.f32x4 %buf, 16
  %z: f32x4 = vzero.f32x4
  %c0 = iconst.i64 0
  br loop(%z, %c0, %v)
loop(%acc: f32x4, %i: i64, %w: f32x4):
  %s: f32x4 = vadd %acc, %w
  %m: f32x4 = vmul %w, %3
  %one = iconst.i64 1
  %n = add.i64 %i, %one
  %more = slt.i64 %n, %1
  br_if %more, loop(%s, %n, %m), done(%s, %m)
done(%a: f32x4, %b: f32x4):
  %odd64 = and.i64 %1, %one
  %odd = trunc_i32 %odd64
  %sel: f32x4 = select %odd, %a, %b
  vstore.f32x4 %buf, 32, %sel
  %lane = load.f32 %buf, 36
  %e = fpext_f64_f32 %lane
  %x: f32 = vextract_lane %sel, 3
  %xe = fpext_f64_f32 %x
  %t = add.f64 %e, %xe
  %u = mul.f64 %t, %2
  ret %u
}
)");
    std::vector<std::vector<RuntimeValue>> args;
    for (const auto& v : sample_vectors(Type::f32x4())) {
        for (int64_t n : {1LL, 2LL, 5LL}) {
            args.push_back({v, RuntimeValue::from_i64(n), RuntimeValue::from_f64(0.5),
                            RuntimeValue::from_f32x4(0.5f, -1.0f, 2.0f, 1.0f)});
        }
    }
    CHECK_EQ(diff_vec(*mod, {"bl_v_mix"}, "bl_v_mix", args), 0);
}

TEST_CASE("Baseline JIT vectors - calls between baseline functions mix vector and scalar arguments") {
    // Win64: vectors in XMM0-3 by position; SysV: the next argument XMMs.
    auto mod = parse_vec_or_fail(R"(module @m
func @bl_v_callee(%0: i64, %1: i32x4, %2: f64, %3: i32x4) -> i32x4 {
entry:
  %t = trunc_i32 %0
  %b: i32x4 = vbroadcast.i32x4 %t
  %s: i32x4 = vsub %1, %3
  %r: i32x4 = vmul %s, %b
  ret %r
}
func @bl_v_caller(%0: i32x4, %1: i32x4, %2: i64) -> i64 {
entry:
  %h = iconst.i64 3
  %f = fconst.f64 1.5
  %r: i32x4 = call.i32x4 @bl_v_callee(%h, %0, %f, %1)
  %q: i32x4 = call.i32x4 @bl_v_callee(%2, %r, %f, %0)
  %a: i32 = vextract_lane %q, 0
  %b: i32 = vextract_lane %q, 3
  %c = add.i32 %a, %b
  %d = sext_i64 %c
  ret %d
}
)");
    std::vector<std::vector<RuntimeValue>> args;
    for (const auto& ab : vec_tuples(Type::i32x4(), 2)) {
        for (int64_t k : {0LL, -2LL, 77LL}) args.push_back({ab[0], ab[1], RuntimeValue::from_i64(k)});
    }
    CHECK_EQ(diff_vec(*mod, {"bl_v_callee", "bl_v_caller"}, "bl_v_caller", args), 0);
}

TEST_CASE("Baseline JIT vectors - baseline code calls tier-2 code with vector arguments and result") {
    const char* callee = R"(
func @bl_v_t2(%0: f64x2, %1: i64, %2: f64x2) -> f64x2 {
entry:
  %f = sitofp_f64_i64 %1
  %b: f64x2 = vbroadcast.f64x2 %f
  %m: f64x2 = vmul %0, %b
  %r: f64x2 = vsub %m, %2
  ret %r
}
)";
    const std::string caller = R"(
func @bl_v_t2_caller(%0: f64x2, %1: f64x2) -> f64x2 {
entry:
  %k = iconst.i64 3
  %r: f64x2 = call.f64x2 @bl_v_t2(%0, %k, %1)
  %s: f64x2 = vadd %r, %0
  ret %s
}
)";
    auto t2_mod = parse_vec_or_fail(std::string("module @t2\n") + callee);
    JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*t2_mod));
    void* t2_entry = jit.get_symbol_address("bl_v_t2");
    REQUIRE(t2_entry != nullptr);

    auto mod = parse_vec_or_fail(std::string("module @m\n") + callee + caller);
    BaselineJitCompiler compiler;
    compiler.register_external_symbol("bl_v_t2", t2_entry);
    CHECK_EQ(diff_vec(*mod, {"bl_v_t2_caller"}, "bl_v_t2_caller", vec_tuples(Type::f64x2(), 2), &compiler), 0);
}

TEST_CASE("Baseline JIT vectors - what stays in the interpreter is rejected at compile time") {
    const char* sources[] = {
        // 256-bit vectors.
        R"(module @m
func @bl_vrej_256(%0: i64) -> i64 {
entry:
  %1: f64x4 = vzero.f64x4
  ret %0
}
)",
        R"(module @m
func @bl_vrej_256_param(%0: i32x8, %1: i32x8) -> i32x8 {
entry:
  %2: i32x8 = vadd %0, %1
  ret %2
}
)",
        // A fifth argument is on the stack on Win64 (the ninth on SysV).
        R"(module @m
func @bl_vrej_stack(%0: f32x4, %1: f32x4, %2: f32x4, %3: f32x4, %4: f32x4, %5: f32x4, %6: f32x4, %7: f32x4, %8: f32x4) -> f32x4 {
entry:
  ret %8
}
)",
    };
    for (const char* src : sources) {
        auto mod = parse_vec_or_fail(src);
        const Function* f = *mod->functions().begin();
        BaselineJitCompiler compiler;
        bool threw = false;
        try {
            (void)compiler.compile(*f);
        } catch (const UnsupportedOperation& e) {
            threw = true;
            CHECK_EQ(e.stage(), std::string("x64 baseline"));
        }
        CHECK(threw);
    }
}

#endif
