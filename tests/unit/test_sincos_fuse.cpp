#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/sincos_fuse.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/deterministic_math.hpp>
#include <cstdint>
#include <cstring>
#include <limits>

using namespace brass;
using namespace brass::test;

namespace {

int sin_calls = 0, cos_calls = 0, sincos_calls = 0;
double host_sin(double x) { ++sin_calls; return math::det_sin(x); }
double host_cos(double x) { ++cos_calls; return math::det_cos(x); }
void host_sincos(double x, double* out) { ++sincos_calls; math::det_sincos(x, out); }

uint64_t bits(double d) {
    uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

void declare(Module& mod, bool with_fused = true) {
    mod.add_symbol_role("host_sin", SymbolRole::Pure);
    mod.add_symbol_role("host_sin", SymbolRole::SinF64);
    mod.add_symbol_role("host_cos", SymbolRole::Pure);
    mod.add_symbol_role("host_cos", SymbolRole::CosF64);
    if (with_fused) {
        mod.add_symbol_role("host_sincos", SymbolRole::SinCosF64);
        mod.add_symbol_role("host_sincos", SymbolRole::NoCollect);
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
Function* one_block(Module& mod, Builder& b, const char* name, bool cos_first) {
    Function* f = mod.create_function(name, Type::f64(), {Type::f64()});
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    Value* x = b.add_block_param(e, Type::f64());
    Value* c = cos_first ? b.build_call("host_cos", Type::f64(), {x}) : nullptr;
    Value* s = b.build_call("host_sin", Type::f64(), {x});
    if (!c) c = b.build_call("host_cos", Type::f64(), {x});
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

} // namespace

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
        const double s = math::det_sin(x), c = math::det_cos(x);
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

TEST_CASE("Sincos fuse - det_sincos is det_sin and det_cos bit for bit") {
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
        math::det_sincos(x, out);
        if (!same(out[0], math::det_sin(x)) || !same(out[1], math::det_cos(x))) ++mismatches;
    }
    CHECK_EQ(mismatches, 0);
}
