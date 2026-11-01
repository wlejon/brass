// The trig runtime (fast_trig.hpp) and the pass that pairs its sine and
// cosine (sincos_fuse.hpp): V8's bits for sin and cos, sincos as the two
// bit for bit, the fusion's rewrite, and the fused code giving the same bits
// in the MIR interpreter, the bytecode VM, the baseline tier and the
// optimizing tier.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/sincos_fuse.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/fast_trig.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace brass;
using namespace brass::test;

namespace {

int sin_calls = 0, cos_calls = 0, sincos_calls = 0;
double host_sin(double x) { ++sin_calls; return math::sin_f64(x); }
double host_cos(double x) { ++cos_calls; return math::cos_f64(x); }
void host_sincos(double x, double* out) { ++sincos_calls; math::sincos_f64(x, out); }

uint64_t bits(double d) {
    uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

void declare(Module& mod, bool with_fused = true, const char* sin = "host_sin", const char* cos = "host_cos",
             const char* sincos = "host_sincos") {
    mod.add_symbol_role(sin, SymbolRole::Pure);
    mod.add_symbol_role(sin, SymbolRole::SinF64);
    mod.add_symbol_role(cos, SymbolRole::Pure);
    mod.add_symbol_role(cos, SymbolRole::CosF64);
    if (with_fused) {
        mod.add_symbol_role(sincos, SymbolRole::SinCosF64);
        mod.add_symbol_role(sincos, SymbolRole::NoCollect);
    }
}

size_t count_calls(const Function& fn, std::string_view sym) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call && inst->symbol() == sym) ++n;
        }
    }
    return n;
}

// f(x) = sin(x) * 3 - cos(x) in one block (cos first when `cos_first`).
Function* one_block(Module& mod, Builder& b, const char* name, bool cos_first, const char* sin = "host_sin",
                    const char* cos = "host_cos") {
    Function* f = mod.create_function(name, Type::f64(), {Type::f64()});
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    Value* x = b.add_block_param(e, Type::f64());
    Value* c = cos_first ? b.build_call(cos, Type::f64(), {x}) : nullptr;
    Value* s = b.build_call(sin, Type::f64(), {x});
    if (!c) c = b.build_call(cos, Type::f64(), {x});
    b.build_ret(b.build_sub(b.build_mul(s, b.build_fconst_f64(3.0)), c));
    f->rebuild_cfg_predecessors();
    return f;
}

// g(x, k): sin(x) in the entry, then cos(x) in the block both arms of a
// branch on k join at: the entry dominates it.
Function* dominated(Module& mod, Builder& b) {
    Function* g = mod.create_function("g", Type::f64(), {Type::f64(), Type::i64()});
    b.set_function(g);
    BasicBlock* e = b.append_block("entry");
    BasicBlock* t = b.append_block("then");
    BasicBlock* el = b.append_block("else");
    BasicBlock* j = b.append_block("join");
    Value* x = b.add_block_param(e, Type::f64());
    Value* k = b.add_block_param(e, Type::i64());
    Value* jp = b.add_block_param(j, Type::f64());
    b.position_at_end(e);
    Value* s = b.build_call("host_sin", Type::f64(), {x});
    b.build_br_if(b.build_eq(k, b.build_iconst_i64(0)), t, el);
    b.position_at_end(t);
    b.build_br(j, {b.build_fconst_f64(1.0)});
    b.position_at_end(el);
    b.build_br(j, {b.build_fconst_f64(2.0)});
    b.position_at_end(j);
    Value* c = b.build_call("host_cos", Type::f64(), {x});
    b.build_ret(b.build_add(b.build_mul(s, jp), c));
    g->rebuild_cfg_predecessors();
    return g;
}

// h(x, k): sin(x) on one arm and cos(x) on the other; neither dominates.
Function* apart(Module& mod, Builder& b) {
    Function* h = mod.create_function("h", Type::f64(), {Type::f64(), Type::i64()});
    b.set_function(h);
    BasicBlock* e = b.append_block("entry");
    BasicBlock* t = b.append_block("then");
    BasicBlock* el = b.append_block("else");
    Value* x = b.add_block_param(e, Type::f64());
    Value* k = b.add_block_param(e, Type::i64());
    b.position_at_end(e);
    b.build_br_if(b.build_eq(k, b.build_iconst_i64(0)), t, el);
    b.position_at_end(t);
    b.build_ret(b.build_call("host_sin", Type::f64(), {x}));
    b.position_at_end(el);
    b.build_ret(b.build_call("host_cos", Type::f64(), {x}));
    h->rebuild_cfg_predecessors();
    return h;
}

const double kInputs[] = {
    0.0, -0.0, 5e-324, 1e-20, 0.5, 0.7853981633974483, 1.5707963267948966, 2.356194490192346, 3.0,
    -3.141592653589793, 100.0, 823549.6354338070, 1e7, -1e22, 1.7976931348623157e308,
    std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
    std::numeric_limits<double>::quiet_NaN(),
};

bool same(double got, double want) {
    if (want != want) return got != got;
    return bits(got) == bits(want);
}

struct TrigGolden {
    double x;
    uint64_t sin_bits;
    uint64_t cos_bits;
};

// V8's Math.sin / Math.cos bits (node). One argument down each of fdlibm's
// paths, which the fast path takes without its branches: kernel_cos's three
// qx, the +-1 step through the medium one, both signs of every quadrant, x
// by pi/2's own high word and the cancellation steps, the medium range's top,
// and past it the Payne-Hanek reduction, which needs 2/pi to beyond a
// thousand bits.
constexpr TrigGolden kTrigGoldens[] = {
    {-100.0, 0x3fe03425b78c4db8ULL, 0x3feb981dbf665fdfULL},
    {-3.141592653589793, 0xbca1a62633145c07ULL, 0xbff0000000000000ULL},
    {-0.7853981633974483, 0xbfe6a09e667f3bccULL, 0x3fe6a09e667f3bcdULL},
    {-1e-7, 0xbe7ad7f29abcaf3bULL, 0x3fefffffffffffd3ULL},
    {0.1234567890123456, 0x3fbf8653acfa781fULL, 0x3fefc1a655cf1085ULL},
    {0.7853981633974483, 0x3fe6a09e667f3bccULL, 0x3fe6a09e667f3bcdULL},
    {1.5707963267948966, 0x3ff0000000000000ULL, 0x3c91a62633145c07ULL},
    {42.5, 0xbfefdff0cd5c5750ULL, 0x3fb6a0506a04973eULL},
    {0.2, 0x3fc96dff233dd2bcULL, 0x3fef5cb49577627aULL},
    {0.5, 0x3fdeaee8744b05f0ULL, 0x3fec1528065b7d50ULL},
    {0.785, 0x3fe69e4fd79ac743ULL, 0x3fe6a2ecb934b59aULL},
    {1.0, 0x3feaed548f090ceeULL, 0x3fe14a280fb5068cULL},
    {-1.0, 0xbfeaed548f090ceeULL, 0x3fe14a280fb5068cULL},
    {2.0, 0x3fed18f6ead1b446ULL, 0xbfdaa22657537205ULL},
    {-2.3, 0xbfe7dcd12d582f24ULL, 0xbfe5522217302fe0ULL},
    {2.356, 0x3fe6a1bec8ab043dULL, 0xbfe69f7df5f75366ULL},
    {1.5707963, 0x3feffffffffffffdULL, 0x3e5cc553391a6262ULL},
    {-1.5707964, 0xbfefffffffffffe8ULL, 0xbe73a69dccb96762ULL},
    {3.0, 0x3fc210386db6d55bULL, 0xbfefae04be85e5d2ULL},
    {-3.0, 0xbfc210386db6d55bULL, 0xbfefae04be85e5d2ULL},
    {4.0, 0xbfe837b9dddc1eaeULL, 0xbfe4eaa606db24c1ULL},
    {-4.0, 0x3fe837b9dddc1eaeULL, 0xbfe4eaa606db24c1ULL},
    {5.0, 0xbfeeaf81f5e09933ULL, 0x3fd22785706b4ad9ULL},
    {-5.0, 0x3feeaf81f5e09933ULL, 0x3fd22785706b4ad9ULL},
    {6.0, 0xbfd1e1f18ab0a2c0ULL, 0x3feeb9b7097822f6ULL},
    {-6.0, 0x3fd1e1f18ab0a2c0ULL, 0x3feeb9b7097822f6ULL},
    {7.0, 0x3fe50608c26d0a08ULL, 0x3fe81ff79ed92017ULL},
    {250.41, 0xbfe968b443c61358ULL, 0x3fe373ba497791d0ULL},
    {4.71238898038469, 0xbff0000000000000ULL, 0xbcaa79394c9e8a0aULL},
    {-7.853981633974483, 0xbff0000000000000ULL, 0x3cb60fafbfd97309ULL},
    {6.283185307179586, 0xbcb1a62633145c07ULL, 0x3ff0000000000000ULL},
    {157079.63267948966, 0x3daaa8ecfb27878aULL, 0x3ff0000000000000ULL},
    {823549.0, 0xbfe3bc41ec71c06dULL, 0x3fe930880dddf826ULL},
    {-823551.0, 0xbfef1e1d322aef27ULL, 0x3fcdd9e00a413200ULL},
    {1e10, 0xbfdf334c7896a4e3ULL, 0x3febf098901c931aULL},
    {1e22, 0xbfeb453ab76bf397ULL, 0x3fe0be2cef01c8f4ULL},
    {1e300, 0xbfea2c16b010e385ULL, 0xbfe2699022adc4c1ULL},
};

} // namespace

TEST_CASE("Fast trig - sin and cos are V8's bits") {
    for (const auto& g : kTrigGoldens) {
        CHECK_EQ(bits(math::sin_f64(g.x)), g.sin_bits);
        CHECK_EQ(bits(math::cos_f64(g.x)), g.cos_bits);
        CHECK_EQ(bits(brass_sin_f64(g.x)), g.sin_bits);
        CHECK_EQ(bits(brass_cos_f64(g.x)), g.cos_bits);
    }
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    constexpr uint64_t kQNaN = 0x7ff8000000000000ULL;
    CHECK_EQ(bits(math::sin_f64(qnan)), kQNaN);
    CHECK_EQ(bits(math::cos_f64(qnan)), kQNaN);
    CHECK_EQ(bits(math::sin_f64(inf)), kQNaN);
    CHECK_EQ(bits(math::cos_f64(-inf)), kQNaN);
    CHECK_EQ(bits(math::sin_f64(0.0)), bits(0.0));
    CHECK_EQ(bits(math::sin_f64(-0.0)), bits(-0.0));
    CHECK_EQ(bits(math::cos_f64(0.0)), bits(1.0));
    CHECK_EQ(bits(math::cos_f64(-0.0)), bits(1.0));
}

TEST_CASE("Fast trig - sincos_f64 is sin_f64 and cos_f64 bit for bit") {
    uint64_t state = 0x9e3779b97f4a7c15ULL;
    auto next = [&] {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    };
    int mismatches = 0;
    for (int i = 0; i < 200000; ++i) {
        // Random bit patterns (every exponent, NaNs and infinities included)
        // and random small multiples of pi/2 nudged by a few ulps.
        double x;
        const uint64_t r = next();
        if (i & 1) {
            std::memcpy(&x, &r, sizeof x);
        } else {
            uint64_t u;
            const double m = static_cast<double>(static_cast<int64_t>(r % 2000000) - 1000000) * 1.5707963267948966;
            std::memcpy(&u, &m, sizeof u);
            u += (r >> 40) % 9;
            u -= 4;
            std::memcpy(&x, &u, sizeof x);
        }
        double out[2];
        math::sincos_f64(x, out);
        if (!same(out[0], math::sin_f64(x)) || !same(out[1], math::cos_f64(x))) ++mismatches;
    }
    CHECK_EQ(mismatches, 0);
}

TEST_CASE("Sincos fuse - a sine and a cosine of one value become one call, same bits") {
    Module mod("sincos");
    declare(mod);
    Builder b(mod);
    Function* f = one_block(mod, b, "f", false);
    Function* fc = one_block(mod, b, "fc", true);
    Function* g = dominated(mod, b);
    Function* h = apart(mod, b);
    CHECK_EQ(fuse_sincos(*f), 1u);
    CHECK_EQ(fuse_sincos(*fc), 1u);
    CHECK_EQ(fuse_sincos(*g), 1u);
    CHECK_EQ(fuse_sincos(*h), 0u);
    for (Function* fn : {f, fc, g}) {
        CHECK_EQ(count_calls(*fn, "host_sin"), 0u);
        CHECK_EQ(count_calls(*fn, "host_cos"), 0u);
        CHECK_EQ(count_calls(*fn, "host_sincos"), 1u);
    }
    CHECK_EQ(count_calls(*h, "host_sincos"), 0u);
    DiagnosticReporter diag;
    for (Function* fn : {f, fc, g, h}) REQUIRE(verify_function(*fn, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("host_sin", reinterpret_cast<void*>(&host_sin));
    jit.register_external_symbol("host_cos", reinterpret_cast<void*>(&host_cos));
    jit.register_external_symbol("host_sincos", reinterpret_cast<void*>(&host_sincos));
    REQUIRE(jit.compile_and_load(mod));
    auto pf = jit.get_function_ptr<double (*)(double)>("f");
    auto pfc = jit.get_function_ptr<double (*)(double)>("fc");
    auto pg = jit.get_function_ptr<double (*)(double, int64_t)>("g");
    auto ph = jit.get_function_ptr<double (*)(double, int64_t)>("h");
    REQUIRE(pf && pfc && pg && ph);
    int n = 0;
    for (double x : kInputs) {
        const double s = math::sin_f64(x), c = math::cos_f64(x);
        sin_calls = cos_calls = sincos_calls = 0;
        CHECK(same(pf(x), s * 3.0 - c));
        CHECK(same(pfc(x), s * 3.0 - c));
        CHECK(same(pg(x, 0), s * 1.0 + c));
        CHECK(same(pg(x, 1), s * 2.0 + c));
        CHECK_EQ(sincos_calls, 4);
        CHECK_EQ(sin_calls + cos_calls, 0);
        CHECK(same(ph(x, 0), s));
        CHECK(same(ph(x, 1), c));
        CHECK_EQ(sin_calls, 1);
        CHECK_EQ(cos_calls, 1);
        ++n;
    }
    CHECK_EQ(n, static_cast<int>(sizeof(kInputs) / sizeof(kInputs[0])));
}

TEST_CASE("Sincos fuse - the fused runtime calls give the same bits in every tier") {
    Module mod("tiers");
    declare(mod, true, "brass_sin_f64", "brass_cos_f64", "brass_sincos_f64");
    Builder b(mod);
    Function* f = one_block(mod, b, "f", false, "brass_sin_f64", "brass_cos_f64");
    CHECK_EQ(fuse_sincos(*f), 1u);
    CHECK_EQ(count_calls(*f, "brass_sincos_f64"), 1u);
    DiagnosticReporter diag;
    REQUIRE(verify_function(*f, &diag));

    Interpreter interp;
    FastInterpreter vm;
    vm.set_module(&mod);
    codegen::BaselineJitCompiler baseline;
    auto base = baseline.compile(*f);
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    auto pf = jit.get_function_ptr<double (*)(double)>("f");
    REQUIRE(pf);
    for (double x : kInputs) {
        const double want = math::sin_f64(x) * 3.0 - math::cos_f64(x);
        CHECK(same(interp.run(mod, "f", {RuntimeValue::from_f64(x)}).as_f64(), want));
        CHECK(same(vm.run(*f, {RuntimeValue::from_f64(x)}).as_f64(), want));
        CHECK(same(base.invoke({RuntimeValue::from_f64(x)}).as_f64(), want));
        CHECK(same(pf(x), want));
    }
}

TEST_CASE("Sincos fuse - nothing fused without the fused role, across operands, or for an ambiguous symbol") {
    {
        Module mod("no_fused");
        declare(mod, false);
        Builder b(mod);
        Function* f = one_block(mod, b, "f", false);
        CHECK_EQ(fuse_sincos(*f), 0u);
    }
    {
        Module mod("two_operands");
        declare(mod);
        Builder b(mod);
        Function* f = mod.create_function("f", Type::f64(), {Type::f64(), Type::f64()});
        b.set_function(f);
        BasicBlock* e = b.append_block("entry");
        Value* x = b.add_block_param(e, Type::f64());
        Value* y = b.add_block_param(e, Type::f64());
        Value* s = b.build_call("host_sin", Type::f64(), {x});
        Value* c = b.build_call("host_cos", Type::f64(), {y});
        b.build_ret(b.build_add(s, c));
        f->rebuild_cfg_predecessors();
        CHECK_EQ(fuse_sincos(*f), 0u);
    }
    {
        // A symbol declared both a sine and a cosine names neither.
        Module mod("ambiguous");
        declare(mod);
        mod.add_symbol_role("host_sin", SymbolRole::CosF64);
        Builder b(mod);
        Function* f = one_block(mod, b, "f", false);
        CHECK_EQ(fuse_sincos(*f), 0u);
    }
}

TEST_CASE("Sincos fuse - a nocollect call is no GC point to the MIR passes") {
    Module mod("nocollect");
    mod.add_symbol_role("quiet", SymbolRole::NoCollect);
    Builder b(mod);
    Function* f = mod.create_function("f", Type::void_type(), {Type::f64()});
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    Value* x = b.add_block_param(e, Type::f64());
    b.build_call("quiet", Type::void_type(), {x});
    b.build_call("loud", Type::void_type(), {x});
    b.build_ret_void();
    f->rebuild_cfg_predecessors();
    std::vector<const Instruction*> calls;
    for (const Instruction* inst : *e) {
        if (inst->opcode() == Opcode::call) calls.push_back(inst);
    }
    REQUIRE(calls.size() == 2u);
    CHECK(!may_trigger_gc(*calls[0]));
    CHECK(may_trigger_gc(*calls[1]));
}
