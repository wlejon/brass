// Baseline JIT opcode coverage (x64 and AArch64): every MIR opcode is either compiled or
// rejected at compile time, and what is compiled agrees with the interpreter.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/code_installer.hpp>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::codegen;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

// The stage name of the host's baseline tier.
#if defined(__aarch64__) || defined(_M_ARM64)
constexpr const char* kHostBaselineStage = "aarch64 baseline";
#else
constexpr const char* kHostBaselineStage = "x64 baseline";
#endif

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

bool same_result(const RuntimeValue& a, const RuntimeValue& b, Type t) {
    if (t.is_float()) {
        if (t.kind() == TypeKind::F32) {
            float x = a.as_f32(), y = b.as_f32();
            if (std::isnan(x) || std::isnan(y)) return std::isnan(x) && std::isnan(y);
            return std::memcmp(&x, &y, 4) == 0;
        }
        double x = a.as_f64(), y = b.as_f64();
        if (std::isnan(x) || std::isnan(y)) return std::isnan(x) && std::isnan(y);
        return std::memcmp(&x, &y, 8) == 0;
    }
    if (t.is_integer() && t.size_in_bytes() <= 4) return a.as_i32() == b.as_i32();
    return a.raw_bits() == b.raw_bits();
}

std::string describe(const RuntimeValue& v, Type t) {
    if (t.is_float()) return std::to_string(t.kind() == TypeKind::F32 ? v.as_f32() : v.as_f64());
    if (t.is_integer() && t.size_in_bytes() <= 4) return std::to_string(v.as_i32());
    return std::to_string(static_cast<int64_t>(v.raw_bits()));
}

// Compiles every function of `mod` with the baseline tier, `order` first
// (callees before callers, so their entries resolve), and checks `entry`
// against the interpreter on each argument set. Returns the mismatch count.
int diff_module(Module& mod, const std::vector<std::string>& order, const std::string& entry,
                const std::vector<std::vector<RuntimeValue>>& arg_sets) {
    BaselineJitCompiler compiler;
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
    Type rt = mod.get_function(entry)->return_type();

    int mismatches = 0;
    Interpreter interp;
    for (const auto& args : arg_sets) {
        RuntimeValue expect = interp.run(mod, entry, args);
        RuntimeValue got = target->invoke(args);
        if (!same_result(expect, got, rt)) {
            ++mismatches;
            std::cerr << "baseline/interpreter mismatch in " << entry << ": interp=" << describe(expect, rt)
                      << " baseline=" << describe(got, rt) << " args:";
            for (const auto& a : args) std::cerr << " " << a.raw_bits();
            std::cerr << "\n";
        }
    }
    return mismatches;
}

std::vector<std::vector<RuntimeValue>> pairs_f64(const std::vector<double>& xs) {
    std::vector<std::vector<RuntimeValue>> out;
    for (double a : xs) for (double b : xs) out.push_back({RuntimeValue::from_f64(a), RuntimeValue::from_f64(b)});
    return out;
}

// Comparisons and overflow checks yield an i32 flag.
bool is_flag_op(const std::string& op) {
    static const char* flags[] = {"eq", "ne", "slt", "ult", "sle", "ule", "sgt", "ugt", "sge", "uge"};
    for (const char* f : flags) if (op == f) return true;
    return op.find("overflow") != std::string::npos;
}

const std::vector<double> kDoubles = {
    0.0, -0.0, 0.5, -0.5, 1.0, 2.5, -2.5, 3.7, -3.7, 1e10, -1e-10,
    std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::quiet_NaN()};

const std::vector<int64_t> kInts = {
    0, 1, -1, 2, 7, -7, 31, 32, 63, 64, 255, 256, 0x7fffffff, -0x7fffffff - 1,
    0x100000000LL, INT64_MAX, INT64_MIN, 0x123456789abcdefLL};

} // namespace

TEST_CASE("Baseline JIT opcodes - every opcode is compiled or deliberately rejected") {
    const auto last = static_cast<uint16_t>(Opcode::coro_destroy);
    size_t supported = 0, rejected = 0;
    for (uint16_t i = 0; i <= last; ++i) {
        const Opcode op = static_cast<Opcode>(i);
        CHECK(!opcode_name(op).empty());
        // Vector opcodes are compiled for 128-bit types (the pre-scan
        // rejects 256-bit ones: test_baseline_jit_vector.cpp).
        // (The AArch64 pre-scan also rejects the exception and coroutine
        // opcodes.) coro_suspend exists only in an unlowered body.
        const bool deliberately_rejected = op == Opcode::coro_suspend;
        CHECK_EQ(BaselineJitCompiler::supports_opcode(op), !deliberately_rejected);
        (deliberately_rejected ? rejected : supported)++;
    }
    CHECK_EQ(rejected, size_t{1});
    CHECK_EQ(supported + rejected, size_t{last} + 1);
}

TEST_CASE("Baseline JIT opcodes - rejected opcodes fail at compile time, not at run time") {
    const char* sources[] = {
        R"(module @m
func @bl_rej_vec(%0: i64) -> i64 {
entry:
  %1 = vzero.f64x4
  ret %0
}
)",
    };
    for (const char* src : sources) {
        auto mod = parse_or_fail(src);
        const Function* f = *mod->functions().begin();
        BaselineJitCompiler compiler;
        bool threw = false;
        try {
            (void)compiler.compile(*f);
        } catch (const UnsupportedOperation& e) {
            threw = true;
            CHECK_EQ(e.stage(), std::string(kHostBaselineStage));
        }
        CHECK(threw);
    }
}

TEST_CASE("Baseline JIT opcodes - the pipeline keeps a rejected function in the interpreter") {
    auto mod = parse_or_fail(R"(module @m
func @bl_rej_pipeline(%0: i64) -> i64 {
entry:
  %1 = vzero.f64x4
  ret %0
}
)");
    const Function* f = mod->get_function("bl_rej_pipeline");
    TieringConfig config;
    config.invocation_tier1_threshold = 1;
    config.enable_background_compile = false;
    auto& pipeline = MultiTierPipeline::instance();
    pipeline.initialize(config);
    TieringRegistry::instance().set_active_module(mod.get());
    auto* handle = FunctionDispatchTable::instance().get_or_create("bl_rej_pipeline", f);
    handle->set_tier(TierLevel::Tier0_Interpreter);

    CHECK(!pipeline.compile_and_install_tier1("bl_rej_pipeline", f));
    CHECK(pipeline.is_baseline_rejected("bl_rej_pipeline"));
    CHECK_EQ(handle->tier(), TierLevel::Tier0_Interpreter);
    // Not retried.
    CHECK(!pipeline.compile_and_install_tier1("bl_rej_pipeline", f));
    pipeline.shutdown();
    TieringRegistry::instance().set_active_module(nullptr);
}

TEST_CASE("Baseline JIT opcodes - integer ops agree with the interpreter (i32 and i64)") {
    const char* binops[] = {"add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr",
                            "sadd_overflow", "ssub_overflow", "smul_overflow",
                            "uadd_overflow", "usub_overflow", "umul_overflow",
                            "eq", "ne", "slt", "ult", "sle", "ule", "sgt", "ugt", "sge", "uge"};
    const char* divops[] = {"sdiv", "smod", "udiv", "umod"};
    const char* unops[] = {"neg", "not", "clz", "ctz", "popcnt"};
    for (const char* ty : {"i32", "i64"}) {
        const bool w32 = std::string(ty) == "i32";
        auto arg = [&](int64_t v) { return w32 ? RuntimeValue::from_i32(static_cast<int32_t>(v)) : RuntimeValue::from_i64(v); };
        std::vector<std::vector<RuntimeValue>> bin_args, div_args, un_args;
        for (int64_t a : kInts) {
            un_args.push_back({arg(a)});
            for (int64_t b : kInts) {
                bin_args.push_back({arg(a), arg(b)});
                const int64_t bb = w32 ? static_cast<int32_t>(b) : b;
                if (bb != 0) div_args.push_back({arg(a), arg(b)});
            }
        }
        auto run_bin = [&](const char* op, const std::vector<std::vector<RuntimeValue>>& args) {
            const bool is_flag = is_flag_op(op);
            std::string name = std::string("bl_int_") + op + "_" + ty;
            std::string ret = is_flag ? "i32" : ty;
            std::string src = "module @m\nfunc @" + name + "(%0: " + ty + ", %1: " + ty + ") -> " + ret +
                              " {\nentry:\n  %2 = " + op + "." + ty + " %0, %1\n  ret %2\n}\n";
            auto mod = parse_or_fail(src);
            CHECK_EQ(diff_module(*mod, {name}, name, args), 0);
        };
        for (const char* op : binops) run_bin(op, bin_args);
        for (const char* op : divops) run_bin(op, div_args);
        for (const char* op : unops) {
            std::string name = std::string("bl_int_") + op + "_" + ty;
            std::string src = "module @m\nfunc @" + name + "(%0: " + ty + ") -> " + ty +
                              " {\nentry:\n  %1 = " + op + "." + ty + " %0\n  ret %1\n}\n";
            auto mod = parse_or_fail(src);
            CHECK_EQ(diff_module(*mod, {name}, name, un_args), 0);
        }
    }
}

TEST_CASE("Baseline JIT opcodes - float arithmetic, math and comparisons agree with the interpreter") {
    const char* binops[] = {"add", "sub", "mul", "sdiv", "smod",
                            "eq", "ne", "slt", "sle", "sgt", "sge"};
    const char* fbin[] = {"fmin", "fmax"};
    const char* fun[] = {"sqrt", "floor", "ceil", "round", "fabs"};
    const auto args2 = pairs_f64(kDoubles);
    std::vector<std::vector<RuntimeValue>> args1, args3;
    for (double a : kDoubles) {
        args1.push_back({RuntimeValue::from_f64(a)});
        for (double b : {2.0, -0.5, 1e300}) {
            args3.push_back({RuntimeValue::from_f64(a), RuntimeValue::from_f64(b), RuntimeValue::from_f64(0.25)});
        }
    }
    for (const char* ty : {"f32", "f64"}) {
        const bool f32 = std::string(ty) == "f32";
        // Parameters are f64; an f32 variant narrows them first and widens
        // the result, so both widths run through the same harness.
        auto prologue = [&](int n) {
            std::string s;
            for (int i = 0; i < n; ++i) {
                if (f32) s += "  %n" + std::to_string(i) + " = fptrunc_f32_f64 %" + std::to_string(i) + "\n";
            }
            return s;
        };
        auto opnd = [&](int i) { return (f32 ? "%n" : "%") + std::to_string(i); };
        auto epilogue = [&](const std::string& v, bool is_flag) {
            if (is_flag) return "  ret " + v + "\n";
            if (f32) return "  %w = fpext_f64_f32 " + v + "\n  ret %w\n";
            return "  ret " + v + "\n";
        };
        auto emit = [&](const std::string& name, int arity, const std::string& op_text, bool is_flag,
                        const std::vector<std::vector<RuntimeValue>>& args) {
            std::string params;
            for (int i = 0; i < arity; ++i) params += (i ? ", %" : "%") + std::to_string(i) + ": f64";
            std::string src = "module @m\nfunc @" + name + "(" + params + ") -> " + (is_flag ? "i32" : "f64") +
                              " {\nentry:\n" + prologue(arity) + "  %r = " + op_text + "\n" +
                              epilogue("%r", is_flag) + "}\n";
            auto mod = parse_or_fail(src);
            CHECK_EQ(diff_module(*mod, {name}, name, args), 0);
        };
        for (const char* op : binops) {
            const bool is_flag = is_flag_op(op);
            emit(std::string("bl_fp_") + op + "_" + ty, 2,
                 std::string(op) + "." + ty + " " + opnd(0) + ", " + opnd(1), is_flag, args2);
        }
        for (const char* op : fbin) {
            emit(std::string("bl_fp_") + op + "_" + ty, 2,
                 std::string(op) + "_" + ty + " " + opnd(0) + ", " + opnd(1), false, args2);
        }
        for (const char* op : fun) {
            emit(std::string("bl_fp_") + op + "_" + ty, 1, std::string(op) + "_" + ty + " " + opnd(0), false, args1);
        }
        emit(std::string("bl_fp_neg_") + ty, 1, std::string("neg.") + ty + " " + opnd(0), false, args1);
        emit(std::string("bl_fp_fma_") + ty, 3,
             std::string("fma_") + ty + " " + opnd(0) + ", " + opnd(1) + ", " + opnd(2), false, args3);
    }
}

TEST_CASE("Baseline JIT opcodes - conversions agree with the interpreter") {
    // In-range values only: an out-of-range float -> int cast is undefined in
    // the interpreter's C++.
    const std::vector<double> in_range = {0.0, -0.0, 0.5, -0.5, 2.5, -2.5, 3.7, -3.7, 1e9, -1e9, 12345.678};
    std::vector<std::vector<RuntimeValue>> fargs;
    for (double d : in_range) fargs.push_back({RuntimeValue::from_f64(d)});
    std::vector<std::vector<RuntimeValue>> iargs;
    for (int64_t v : kInts) iargs.push_back({RuntimeValue::from_i64(v)});

    struct Case { const char* name; const char* param; const char* ret; const char* body; };
    const Case cases[] = {
        {"bl_cv_fptosi_i32", "f64", "i32", "  %r = fptosi_i32 %0\n  ret %r\n"},
        {"bl_cv_fptosi_i64", "f64", "i64", "  %r = fptosi_i64 %0\n  ret %r\n"},
        {"bl_cv_fptosi_i32_f32", "f64", "i32", "  %n = fptrunc_f32_f64 %0\n  %r = fptosi_i32_f32 %n\n  ret %r\n"},
        {"bl_cv_fptosi_i64_f32", "f64", "i64", "  %n = fptrunc_f32_f64 %0\n  %r = fptosi_i64_f32 %n\n  ret %r\n"},
        {"bl_cv_sitofp_f64_i32", "i64", "f64", "  %t = trunc_i32 %0\n  %r = sitofp_f64_i32 %t\n  ret %r\n"},
        {"bl_cv_sitofp_f64_i64", "i64", "f64", "  %r = sitofp_f64_i64 %0\n  ret %r\n"},
        {"bl_cv_sitofp_f32_i32", "i64", "f32", "  %t = trunc_i32 %0\n  %r = sitofp_f32_i32 %t\n  ret %r\n"},
        {"bl_cv_sitofp_f32_i64", "i64", "f32", "  %r = sitofp_f32_i64 %0\n  ret %r\n"},
        {"bl_cv_fptrunc", "f64", "f32", "  %r = fptrunc_f32_f64 %0\n  ret %r\n"},
        {"bl_cv_fpext", "f64", "f64", "  %n = fptrunc_f32_f64 %0\n  %r = fpext_f64_f32 %n\n  ret %r\n"},
        {"bl_cv_bitcast", "f64", "f64",
         "  %b = bitcast_i64_f64 %0\n  %c = iconst.i64 1\n  %x = xor.i64 %b, %c\n  %r = bitcast_f64_i64 %x\n  ret %r\n"},
        {"bl_cv_sext", "i64", "i64", "  %t = trunc_i32 %0\n  %r = sext_i64 %t\n  ret %r\n"},
        {"bl_cv_zext", "i64", "i64", "  %t = trunc_i32 %0\n  %r = zext_i64 %t\n  ret %r\n"},
        // i8 arithmetic runs at 32 bits, as in the interpreter; the slot
        // written by trunc_i8 must hold exactly 0..255.
        {"bl_cv_trunc_i8", "i64", "i64",
         "  %t = trunc_i32 %0\n  %b = trunc.i8 %t\n  %s = add.i8 %b, %b\n  %q = slt.i8 %b, %s\n"
         "  %y = zext_i64 %s\n  %v = zext_i64 %q\n  %r = shl.i64 %y, %v\n  ret %r\n"},
        {"bl_cv_trunc_i8_zext", "i64", "i64", "  %t = trunc_i32 %0\n  %b = trunc.i8 %t\n  %r = zext_i64 %b\n  ret %r\n"},
    };
    for (const Case& c : cases) {
        std::string src = std::string("module @m\nfunc @") + c.name + "(%0: " + c.param + ") -> " + c.ret +
                          " {\nentry:\n" + c.body + "}\n";
        auto mod = parse_or_fail(src);
        CHECK_EQ(diff_module(*mod, {c.name}, c.name, std::string(c.param) == "f64" ? fargs : iargs), 0);
    }
}

TEST_CASE("Baseline JIT opcodes - calls with stack arguments, indirect calls and select") {
    // Seven integer and three float arguments: stack arguments on both
    // conventions, f32 through a call, an i32 result.
    auto mod = parse_or_fail(R"(module @m
func @bl_call_callee(%0: i64, %1: f64, %2: i32, %3: i64, %4: f32, %5: i64, %6: i64, %7: f64, %8: i64, %9: i64) -> i32 {
entry:
  %a = add.i64 %0, %3
  %b = add.i64 %5, %6
  %c = sub.i64 %8, %9
  %d = mul.i64 %a, %b
  %e = add.i64 %d, %c
  %f = fpext_f64_f32 %4
  %g = add.f64 %1, %f
  %h = mul.f64 %g, %7
  %i = fptosi_i64 %h
  %j = add.i64 %e, %i
  %k = trunc_i32 %j
  %l = add.i32 %k, %2
  ret %l
}
func @bl_call_twice(%0: i64) -> i64 {
entry:
  %1 = add.i64 %0, %0
  ret %1
}
func @bl_call_main(%0: i64, %1: f64) -> i64 {
entry:
  %c3 = iconst.i32 3
  %c4 = iconst.i64 4
  %f = fptrunc_f32_f64 %1
  %c5 = iconst.i64 5
  %c6 = iconst.i64 6
  %c8 = iconst.i64 8
  %c9 = iconst.i64 9
  %r = call.i32 @bl_call_callee(%0, %1, %c3, %c4, %f, %c5, %c6, %1, %c8, %c9)
  %r64 = sext_i64 %r
  %fp = func_addr @bl_call_twice
  %t = call_indirect.i64 %fp(%r64)
  %z = iconst.i64 0
  %neg = slt.i64 %t, %z
  %n = neg.i64 %t
  %s = select %neg, %n, %t
  safepoint
  ret %s
}
)");
    std::vector<std::vector<RuntimeValue>> args;
    for (int64_t a : {0LL, 1LL, -5LL, 1000000LL}) {
        for (double d : {0.0, 1.5, -2.25, 100.0}) args.push_back({RuntimeValue::from_i64(a), RuntimeValue::from_f64(d)});
    }
    CHECK_EQ(diff_module(*mod, {"bl_call_callee", "bl_call_twice", "bl_call_main"}, "bl_call_main", args), 0);
}

TEST_CASE("Baseline JIT opcodes - memory: widths, indexed access with a negative i32 index") {
    auto mod = parse_or_fail(R"(module @m
func @bl_mem(%0: i64, %1: f64) -> i64 {
entry:
  %buf = alloca 64, 16
  %t = trunc_i32 %0
  store.i32 %buf, 0, %t
  %sentinel = iconst.i32 -1
  store.i32 %buf, 4, %sentinel
  %l = load.i32 %buf
  %f = fptrunc_f32_f64 %1
  store.f32 %buf, 8, %f
  %lf = load.f32 %buf, 8
  store.f64 %buf, 16, %1
  %m = iconst.i32 -2
  %base = load.i64 %buf, 0
  store_indexed.i64 %buf, %m, 8, 40, %0
  %li = load_indexed.i64 %buf, %m, 8, 40
  %hi = load.i32 %buf, 4
  %a = sext_i64 %l
  %b = sext_i64 %hi
  %c = fpext_f64_f32 %lf
  %d = fptosi_i64 %c
  %e = add.i64 %a, %b
  %g = add.i64 %e, %d
  %h = add.i64 %g, %li
  %i = xor.i64 %h, %base
  ret %i
}
)");
    std::vector<std::vector<RuntimeValue>> args;
    for (int64_t a : {0LL, 7LL, -1LL, 0x1234567890LL}) {
        for (double d : {0.0, 3.5, -1e6}) args.push_back({RuntimeValue::from_i64(a), RuntimeValue::from_f64(d)});
    }
    CHECK_EQ(diff_module(*mod, {"bl_mem"}, "bl_mem", args), 0);
}

TEST_CASE("Baseline JIT opcodes - narrow stores write only their width") {
    auto mod = parse_or_fail(R"(module @m
func @bl_mem_narrow(%0: i64) -> i64 {
entry:
  %buf = alloca 16, 16
  %ones = iconst.i64 -1
  store.i64 %buf, 0, %ones
  %t = trunc_i32 %0
  %b = trunc.i8 %t
  store.i8 %buf, 1, %b
  %r = load.i64 %buf
  ret %r
}
)");
    BaselineJitCompiler compiler;
    auto compiled = compiler.compile(*mod->get_function("bl_mem_narrow"));
    auto fnp = compiled.get_function_ptr<int64_t (*)(int64_t)>();
    CHECK_EQ(static_cast<uint64_t>(fnp(0x42)), 0xffffffffffff42ffULL);
    auto l8 = parse_or_fail(R"(module @m
func @bl_mem_load8(%0: i64) -> i64 {
entry:
  %buf = alloca 16, 16
  store.i64 %buf, 0, %0
  %b = load.i8 %buf, 2
  %r = zext_i64 %b
  ret %r
}
)");
    auto c8 = compiler.compile(*l8->get_function("bl_mem_load8"));
    CHECK_EQ(c8.get_function_ptr<int64_t (*)(int64_t)>()(0x7766554433221100LL), 0x22);
}

TEST_CASE("Baseline JIT opcodes - a failed guard takes its exit stub") {
    auto mod = parse_or_fail(R"(module @m
func @bl_guard_stub(%0: i64, %1: i64) -> i64 {
entry:
  %c = iconst.i64 1000
  %r = add.i64 %0, %c
  %s = mul.i64 %r, %1
  ret %s
}
func @bl_guard(%0: i64, %1: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %ok = sgt.i64 %0, %z
  guard %ok, @bl_guard_stub, [%0, %1]
  %r = add.i64 %0, %1
  ret %r
}
)");
    std::vector<std::vector<RuntimeValue>> args;
    for (int64_t a : {-3LL, 0LL, 5LL}) args.push_back({RuntimeValue::from_i64(a), RuntimeValue::from_i64(7)});
    CHECK_EQ(diff_module(*mod, {"bl_guard_stub", "bl_guard"}, "bl_guard", args), 0);
}

TEST_CASE("Baseline JIT opcodes - patchable constants and calls, read_sp, unreachable") {
    auto mod = parse_or_fail(R"(module @m
func @bl_patch_callee(%0: i64) -> i64 {
entry:
  %c = iconst.i64 3
  %r = mul.i64 %0, %c
  ret %r
}
func @bl_patch(%0: i64) -> i64 {
entry:
  %a = patchable_const.i32 @bl_site_a, 40
  %b = patchable_const.i64 @bl_site_b, 2
  %sp = read_sp
  %z = iconst.i64 0
  %ok = ne.i64 %sp, %z
  br_if %ok, go, dead
go:
  %r = patchable_call.i64 @bl_site_c, @bl_patch_callee(%0)
  %a64 = sext_i64 %a
  %s = add.i64 %r, %a64
  %t = add.i64 %s, %b
  ret %t
dead:
  unreachable
}
)");
    BaselineJitCompiler compiler;
    auto callee = compiler.compile(*mod->get_function("bl_patch_callee"));
    compiler.register_external_symbol("bl_patch_callee", callee.entry_point());
    auto compiled = compiler.compile(*mod->get_function("bl_patch"));
    CHECK_EQ(compiled.get_function_ptr<int64_t (*)(int64_t)>()(5), 57);
}

TEST_CASE("Baseline JIT opcodes - write_barrier passes (obj, val) in the C convention") {
    static uint64_t seen_obj = 0, seen_val = 0;
    struct Hook {
        static void barrier(uint64_t obj, uint64_t val) { seen_obj = obj; seen_val = val; }
    };
    auto mod = parse_or_fail(R"(module @m
func @bl_wb(%0: i64, %1: i64) -> i64 {
entry:
  write_barrier %0, %1
  ret %0
}
)");
    BaselineJitCompiler compiler;
    compiler.register_external_symbol("brass_gc_write_barrier", reinterpret_cast<void*>(&Hook::barrier));
    auto compiled = compiler.compile(*mod->get_function("bl_wb"));
    CHECK_EQ(compiled.get_function_ptr<int64_t (*)(int64_t, int64_t)>()(0x1111, 0x2222), 0x1111);
    CHECK_EQ(seen_obj, 0x1111u);
    CHECK_EQ(seen_val, 0x2222u);
}

#endif
