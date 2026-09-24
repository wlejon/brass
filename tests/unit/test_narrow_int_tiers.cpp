// i8 / i16 integer semantics across the four tiers (docs/semantics.md,
// "Narrow integers"): the Interpreter, the FastInterpreter, the tier-2 JIT
// and the baseline JIT each run small programs on narrow values and must
// produce the value a C++ reference computes. Operands are themselves
// narrow arithmetic results (a wrapped add), so a tier that leaves bits
// above the width in a register must not let a consumer see them.
// Regressions: a tier-2 store.i8 wrote the register's full width; switch on
// an i8 never took a negative case (or compared unmasked bits) on three
// tiers; the Interpreter could not store.i8; i8 arithmetic was normalised
// on one tier only and slt.i8 compared unsigned everywhere.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::codegen;

namespace {

std::unique_ptr<Module> parse_narrow(const std::string& src) {
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

uint64_t zmask(unsigned bits, int64_t v) { return static_cast<uint64_t>(v) & ((uint64_t{1} << bits) - 1); }
int64_t smask(unsigned bits, int64_t v) {
    const int64_t z = static_cast<int64_t>(zmask(bits, v));
    const int64_t sign = int64_t{1} << (bits - 1);
    return (z ^ sign) - sign;
}

// The reference: `op` on N-bit operands a, b (any bits above N ignored).
// Returns false for an input the case does not test.
bool reference(const std::string& op, unsigned bits, int64_t a_in, int64_t b_in, uint64_t& out) {
    const uint64_t a = zmask(bits, a_in), b = zmask(bits, b_in);
    const int64_t sa = smask(bits, a_in), sb = smask(bits, b_in);
    auto val = [&](int64_t r) { out = zmask(bits, r); return true; };
    auto flag = [&](bool f) { out = f ? 1 : 0; return true; };
    const int64_t smin = -(int64_t{1} << (bits - 1)), smax = (int64_t{1} << (bits - 1)) - 1;
    const int64_t umax = (int64_t{1} << bits) - 1;
    if (op == "add") return val(static_cast<int64_t>(a + b));
    if (op == "sub") return val(static_cast<int64_t>(a - b));
    if (op == "mul") return val(static_cast<int64_t>(a * b));
    if (op == "and") return val(static_cast<int64_t>(a & b));
    if (op == "or") return val(static_cast<int64_t>(a | b));
    if (op == "xor") return val(static_cast<int64_t>(a ^ b));
    if (op == "shl") return b < bits && val(static_cast<int64_t>(a << b));
    if (op == "lshr") return b < bits && val(static_cast<int64_t>(a >> b));
    if (op == "ashr") return b < bits && val(sa >> b);
    if (op == "sdiv") return b != 0 && val(sa / sb);
    if (op == "smod") return b != 0 && val(sa % sb);
    if (op == "udiv") return b != 0 && val(static_cast<int64_t>(a / b));
    if (op == "umod") return b != 0 && val(static_cast<int64_t>(a % b));
    if (op == "eq") return flag(a == b);
    if (op == "ne") return flag(a != b);
    if (op == "slt") return flag(sa < sb);
    if (op == "sle") return flag(sa <= sb);
    if (op == "sgt") return flag(sa > sb);
    if (op == "sge") return flag(sa >= sb);
    if (op == "ult") return flag(a < b);
    if (op == "ule") return flag(a <= b);
    if (op == "ugt") return flag(a > b);
    if (op == "uge") return flag(a >= b);
    if (op == "sadd_overflow") return flag(sa + sb < smin || sa + sb > smax);
    if (op == "ssub_overflow") return flag(sa - sb < smin || sa - sb > smax);
    if (op == "smul_overflow") return flag(sa * sb < smin || sa * sb > smax);
    if (op == "uadd_overflow") return flag(static_cast<int64_t>(a + b) > umax);
    if (op == "usub_overflow") return flag(a < b);
    if (op == "umul_overflow") return flag(static_cast<int64_t>(a * b) > umax);
    if (op == "neg") return val(-static_cast<int64_t>(a));
    if (op == "not") return val(~static_cast<int64_t>(a));
    if (op == "popcnt") { unsigned n = 0; for (uint64_t x = a; x; x &= x - 1) ++n; out = n; return true; }
    if (op == "clz") { unsigned n = 0; for (int i = static_cast<int>(bits) - 1; i >= 0 && !((a >> i) & 1); --i) ++n; out = n; return true; }
    if (op == "ctz") { unsigned n = 0; while (n < bits && !((a >> n) & 1)) ++n; out = n; return true; }
    return false;
}

bool is_flag_op(const std::string& op) {
    static const char* const flags[] = {"eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge",
                                        "sadd_overflow", "ssub_overflow", "smul_overflow",
                                        "uadd_overflow", "usub_overflow", "umul_overflow"};
    for (const char* f : flags) if (op == f) return true;
    return false;
}
bool is_unary_op(const std::string& op) {
    return op == "neg" || op == "not" || op == "popcnt" || op == "clz" || op == "ctz";
}

// The N-bit value (x1 + x2) mod 2^N as %<name>: i8 by trunc.i8, i16 by a
// load.i16 of the i32 stored in the alloca %p.
std::string narrow_of(unsigned bits, const std::string& name, const std::string& x1, const std::string& x2,
                      int off) {
    const std::string t = bits == 8 ? "i8" : "i16";
    std::string s;
    for (const std::string& x : {x1, x2}) {
        s += "  %" + name + x + "w = trunc_i32 %" + x + "\n";
        if (bits == 8) {
            s += "  %" + name + x + " = trunc.i8 %" + name + x + "w\n";
        } else {
            s += "  store.i32 %p, " + std::to_string(off) + ", %" + name + x + "w\n";
            s += "  %" + name + x + " = load.i16 %p, " + std::to_string(off) + "\n";
        }
    }
    s += "  %" + name + " = add." + t + " %" + name + x1 + ", %" + name + x2 + "\n";
    return s;
}

// func @f(%x1, %x2, %y1, %y2) -> i64: `op` on a = x1 + x2, b = y1 + y2 at
// N bits. A narrow result is stored (N bytes) into a zeroed i64 and that
// i64 returned, so a store wider than N shows too.
std::string op_module(const std::string& op, unsigned bits) {
    const std::string t = bits == 8 ? "i8" : "i16";
    std::string s = "module @m\nfunc @f(%x1: i64, %x2: i64, %y1: i64, %y2: i64) -> i64 {\nentry:\n";
    s += "  %p = alloca 64, 16\n  %z = iconst.i64 0\n  %eight = iconst.i64 8\n";
    s += "  store.i64 %p, 16, %z\n  store.i64 %p, 24, %z\n";
    s += narrow_of(bits, "a", "x1", "x2", 0);
    if (!is_unary_op(op)) s += narrow_of(bits, "b", "y1", "y2", 8);
    s += "  %r = " + op + "." + t + " %a" + (is_unary_op(op) ? "" : ", %b") + "\n";
    if (is_flag_op(op)) {
        s += "  %rz = zext_i64 %r\n  ret %rz\n}\n";
    } else {
        // Byte 16 stays 0; bytes above the result must stay 0 as well.
        s += "  store." + t + " %p, 17, %r\n  %v = load.i64 %p, 16\n  %w = lshr.i64 %v, %eight\n  ret %w\n}\n";
    }
    return s;
}

using Fn4 = int64_t (*)(int64_t, int64_t, int64_t, int64_t);

struct Runner {
    std::unique_ptr<Module> mod;
    JitExecutionEngine jit;
    BaselineJitCompiler baseline;
    BaselineCompiledFunction bfn;
    Fn4 jit_fn = nullptr;
    Fn4 base_fn = nullptr;

    explicit Runner(const std::string& src, bool expect_tier2 = true) : mod(parse_narrow(src)) {
        bool jit_ok = false;
        std::string jit_error;
        try {
            jit_ok = jit.compile_and_load(*mod);
        } catch (const UnsupportedOperation& e) {
            jit_error = e.what();
        }
        if (!expect_tier2) CHECK(jit_error.find("unsupported operation") != std::string::npos);
        if (expect_tier2) {
            REQUIRE(jit_ok);
            jit_fn = reinterpret_cast<Fn4>(jit.get_symbol_address("f"));
            REQUIRE(jit_fn != nullptr);
        } else {
            CHECK(!jit_ok);
        }
        bfn = baseline.compile(*mod->get_function("f"));
        REQUIRE(bfn.is_valid());
        base_fn = bfn.get_function_ptr<Fn4>();
    }

    // Every tier's result for the arguments; false (and a report) on the
    // first disagreement with `want`.
    bool agree(const std::vector<int64_t>& a, uint64_t want, const std::string& what) {
        std::vector<RuntimeValue> args;
        for (int64_t v : a) args.push_back(RuntimeValue::from_i64(v));
        const Function& f = *mod->get_function("f");
        uint64_t got[4] = {};
        Interpreter in;
        in.set_module(mod.get());
        got[0] = static_cast<uint64_t>(in.run(f, args).as_i64());
        FastInterpreter fi;
        got[1] = static_cast<uint64_t>(fi.run(f, args).as_i64());
        got[2] = jit_fn ? static_cast<uint64_t>(jit_fn(a[0], a[1], a[2], a[3])) : want;
        got[3] = static_cast<uint64_t>(base_fn(a[0], a[1], a[2], a[3]));
        static const char* const tiers[] = {"interp", "fast", "jit", "baseline"};
        for (int t = 0; t < 4; ++t) {
            if (got[t] != want) {
                std::cerr << what << " (" << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << "): "
                          << tiers[t] << " gave 0x" << std::hex << got[t] << ", want 0x" << want << std::dec
                          << "\n";
                return false;
            }
        }
        return true;
    }
};

const int64_t kInputs[][2] = {{0, 0},   {1, 0},     {-1, 0},    {127, 0},  {-128, 0}, {200, 61},
                              {255, 6}, {-7, 0},    {49, 0},    {3, 5},    {100, 161}, {0x7fff, 0},
                              {-0x8000, 0}, {0x1234, 0xf00}, {0xfff9, 0}, {0x18000, 0x8001}};

void check_op(const std::string& op, unsigned bits) {
    // Every op compiles on every tier, clz / ctz and the overflow checks,
    // whose results depend on the width itself, among them.
    Runner r(op_module(op, bits));
    int bad = 0;
    for (const auto& x : kInputs) {
        for (const auto& y : kInputs) {
            uint64_t want = 0;
            if (!reference(op, bits, x[0] + x[1], y[0] + y[1], want)) continue;
            if (!r.agree({x[0], x[1], y[0], y[1]}, want, op + ".i" + std::to_string(bits))) ++bad;
            if (bad > 3) break;
        }
        if (bad > 3) break;
    }
    CHECK_EQ(bad, 0);
}

const char* const kOps[] = {"add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr",
                            "sdiv", "smod", "udiv", "umod", "neg", "not", "popcnt", "clz", "ctz",
                            "eq", "ne", "slt", "sle", "sgt", "sge", "ult", "ule", "ugt", "uge",
                            "sadd_overflow", "ssub_overflow", "smul_overflow",
                            "uadd_overflow", "usub_overflow", "umul_overflow"};

} // namespace

TEST_CASE("Narrow integers - i8 operations agree on every tier") {
    for (const char* op : kOps) check_op(op, 8);
}

TEST_CASE("Narrow integers - i16 operations agree on every tier") {
    for (const char* op : kOps) check_op(op, 16);
}

TEST_CASE("Narrow integers - switch on i8 / i16 takes negative and wrapped cases") {
    for (unsigned bits : {8u, 16u}) {
        const std::string t = bits == 8 ? "i8" : "i16";
        const int64_t lo = bits == 8 ? -128 : -32768, hi = bits == 8 ? 127 : 32767;
        std::string s = "module @m\nfunc @f(%x1: i64, %x2: i64, %y1: i64, %y2: i64) -> i64 {\nentry:\n";
        s += "  %p = alloca 64, 16\n" + narrow_of(bits, "a", "x1", "x2", 0);
        s += "  switch %a, default: dd, [" + std::to_string(lo) + ": c0, -1: c1, 0: c2, " + std::to_string(hi) +
             ": c3, 5: c4, -100: c5]\n";
        for (int i = 0; i < 6; ++i) {
            s += "c" + std::to_string(i) + ":\n  %r" + std::to_string(i) + " = iconst.i64 " +
                 std::to_string(1000 + i) + "\n  ret %r" + std::to_string(i) + "\n";
        }
        s += "dd:\n  %d = iconst.i64 7\n  ret %d\n}\n";
        Runner r(s);
        const int64_t cases[] = {lo, -1, 0, hi, 5, -100};
        int bad = 0;
        for (const auto& x : kInputs) {
            const int64_t v = smask(bits, x[0] + x[1]);
            uint64_t want = 7;
            for (int i = 0; i < 6; ++i) if (cases[i] == v) want = 1000 + i;
            if (!r.agree({x[0], x[1], 0, 0}, want, "switch." + t)) ++bad;
        }
        // Values that wrap onto a case: 261 is 5 in i8, 0x10005 in i16.
        const int64_t wrap = bits == 8 ? 261 : 0x10005;
        if (!r.agree({wrap, 0, 0, 0}, 1004, "switch." + t + " wrapped")) ++bad;
        if (!r.agree({wrap - 6, 0, 0, 0}, 1001, "switch." + t + " wrapped to -1")) ++bad;
        CHECK_EQ(bad, 0);
    }
}

TEST_CASE("Narrow integers - stores to an alloca write exactly their width") {
    // Bytes 0..15 of the buffer are 0xAA; store.i8 at 3 and store.i16 at 9
    // of trunc / wrapped values, then both words are returned combined.
    const std::string body =
        "  %c = iconst.i64 -6148914691236517206\n"
        "  store.i64 %q, 0, %c\n  store.i64 %q, 8, %c\n"
        "  %xw = trunc_i32 %x1\n  %b = trunc.i8 %xw\n  %b2 = add.i8 %b, %b\n  store.i8 %q, 3, %b2\n"
        "  store.i32 %q, 12, %xw\n  %h = load.i16 %q, 12\n  %h2 = add.i16 %h, %h\n  store.i16 %q, 9, %h2\n"
        "  store.i64 %q, 12, %c\n"
        "  %v0 = load.i64 %q, 0\n  %v1 = load.i64 %q, 8\n  %r = xor.i64 %v0, %v1\n  ret %r\n}\n";
    const std::string alloca_src =
        "module @m\nfunc @f(%x1: i64, %x2: i64, %y1: i64, %y2: i64) -> i64 {\nentry:\n  %q = alloca 64, 16\n" + body;
    Runner r(alloca_src);
    int bad = 0;
    for (const auto& x : kInputs) {
        const int64_t v = x[0] + x[1];
        uint8_t buf[16];
        for (auto& c : buf) c = 0xAA;
        buf[3] = static_cast<uint8_t>(2 * v);
        const uint16_t h = static_cast<uint16_t>(2 * v);
        buf[9] = static_cast<uint8_t>(h);
        buf[10] = static_cast<uint8_t>(h >> 8);
        uint64_t w0 = 0, w1 = 0;
        std::memcpy(&w0, buf, 8);
        std::memcpy(&w1, buf + 8, 8);
        if (!r.agree({v, 0, 0, 0}, w0 ^ w1, "store alloca")) ++bad;
    }
    CHECK_EQ(bad, 0);
}

TEST_CASE("Narrow integers - stores through a host pointer write exactly their width") {
    const std::string src =
        "module @m\nfunc @f(%q: ptr, %x1: i64) -> i64 {\nentry:\n"
        "  %xw = trunc_i32 %x1\n  %b = trunc.i8 %xw\n  %b2 = mul.i8 %b, %b\n  store.i8 %q, 1, %b2\n"
        "  store.i32 %q, 8, %xw\n  %h = load.i16 %q, 8\n  %h2 = mul.i16 %h, %h\n  store.i16 %q, 4, %h2\n"
        "  %z = iconst.i64 0\n  ret %z\n}\n";
    auto mod = parse_narrow(src);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto jit_fn = reinterpret_cast<int64_t (*)(void*, int64_t)>(jit.get_symbol_address("f"));
    BaselineJitCompiler baseline;
    BaselineCompiledFunction bfn = baseline.compile(*mod->get_function("f"));
    REQUIRE(bfn.is_valid());
    auto base_fn = bfn.get_function_ptr<int64_t (*)(void*, int64_t)>();
    for (const auto& x : kInputs) {
        const int64_t v = x[0] + x[1];
        uint8_t want[16];
        for (auto& c : want) c = 0x55;
        want[1] = static_cast<uint8_t>(v * v);
        const uint16_t hv = static_cast<uint16_t>(static_cast<uint16_t>(v) * static_cast<uint16_t>(v));
        want[4] = static_cast<uint8_t>(hv);
        want[5] = static_cast<uint8_t>(hv >> 8);
        std::memcpy(want + 8, &v, 4);
        for (int t = 0; t < 4; ++t) {
            alignas(16) uint8_t buf[16];
            for (auto& c : buf) c = 0x55;
            if (t == 0) {
                Interpreter in;
                in.set_module(mod.get());
                in.run(*mod->get_function("f"), {RuntimeValue::from_ptr(buf), RuntimeValue::from_i64(v)});
            } else if (t == 1) {
                FastInterpreter fi;
                fi.run(*mod->get_function("f"), {RuntimeValue::from_ptr(buf), RuntimeValue::from_i64(v)});
            } else if (t == 2) {
                jit_fn(buf, v);
            } else {
                base_fn(buf, v);
            }
            const bool same = std::memcmp(buf, want, 16) == 0;
            if (!same) std::cerr << "host store, tier " << t << ", x = " << v << "\n";
            CHECK(same);
        }
    }
}
