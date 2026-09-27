#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <string>

using namespace brass;
using namespace brass::test;

// The high lane of an f64 vector taken into an f64 register that is spilled.
// Lane 1 was selected as a copy of the whole vector into the f64 register
// and a shufpd reading it back, so a spill between the two (an f64 slot, 8
// bytes, reloaded with movsd) zeroed the lane: in an OSR entry of tats's
// DSP code the two coefficients a biquad loaded as one vector came out as
// (a1, 0) and the filter blew up to Infinity.

namespace {

// fn(a, b) = sum_i (hi_i * (i + 1) + lo_i) + opaque(a), where lane 0 of
// vector i is a + i and lane 1 is b - i: every lane is live across the call,
// more of them than there are registers, so some are spilled.
void build_lanes_across_call(Module& mod, int n) {
    {
        Function* opaque = mod.create_function("opaque", Type::f64(), {Type::f64()});
        Builder b(mod);
        b.set_function(opaque);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::f64());
        b.build_ret(b.build_mul(x, b.build_fconst_f64(0.5)));
        opaque->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*opaque));
    }
    Function* fn = mod.create_function("lanes", Type::f64(), {Type::f64(), Type::f64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::f64());
    Value* bv = b.add_block_param(entry, Type::f64());
    std::vector<Value*> lo, hi;
    for (int i = 0; i < n; ++i) {
        Value* l = b.build_fadd(a, b.build_fconst_f64(i));
        Value* h = b.build_sub(bv, b.build_fconst_f64(i));
        Value* v = b.build_vinsert_lane(b.build_vbroadcast(Type::f64x2(), l), h, 1);
        lo.push_back(b.build_vextract_lane(v, 0));
        hi.push_back(b.build_vextract_lane(v, 1));
    }
    Value* sum = b.build_call("opaque", Type::f64(), {a});
    for (int i = 0; i < n; ++i) {
        sum = b.build_fadd(sum, b.build_mul(hi[i], b.build_fconst_f64(i + 1)));
        sum = b.build_fadd(sum, lo[i]);
    }
    b.build_ret(sum);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
}

} // namespace

TEST_CASE("vextract_lane - the high f64 lane survives its register being spilled") {
    for (int n : {4, 12, 24}) {
        Module mod("vextract_spill_" + std::to_string(n));
        build_lanes_across_call(mod, n);
        const double a = 1.25, bv = 1000.0;
        double expect = a * 0.5;
        for (int i = 0; i < n; ++i) expect += (bv - i) * (i + 1) + (a + i);
        assert_diff(mod, "lanes", {RuntimeValue::from_f64(a), RuntimeValue::from_f64(bv)});

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(mod));
        CHECK(jit.invoke("lanes", {RuntimeValue::from_f64(a), RuntimeValue::from_f64(bv)}).as_f64() == expect);
    }
}
