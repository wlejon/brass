// A tier-2 copy carries the bodies of the functions it calls and calls those
// copies directly. A guard failing in a carried body finishes that callee in
// Tier 0 and returns into the caller's code, which calls the same copy again
// on its next run. The failures must count against the caller's code too:
// otherwise it is never invalidated and every later call deoptimizes, which
// in bronze made a mesh build that wrote a second object shape into its
// batches take minutes instead of a fraction of a second.
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <iostream>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::runtime;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

// @helper(%x) = %x * 100
// @callee(%x, %tag): %a = %x * 3; guard %tag == 1 (resume 7, state [%a, %x]);
//   ret %a + 1; resume 7: ret %a + @helper(%x)
// @caller(%x, %tag) = @callee(%x, %tag) + 1000
void build_program(Module& mod) {
    Function* h = mod.create_function("cc_helper", Type::i64(), {Type::i64()});
    {
        Builder hb(*h);
        hb.position_at_end(hb.append_block("entry"));
        Value* x = hb.add_param(Type::i64());
        hb.build_ret(hb.build_mul(x, hb.build_iconst_i64(100)));
    }
    Function* callee = mod.create_function("cc_callee", Type::i64(), {Type::i64(), Type::i32()});
    {
        Builder b(*callee);
        b.position_at_end(b.append_block("entry"));
        Value* x = b.add_param(Type::i64());
        Value* tag = b.add_param(Type::i32());
        Value* a = b.build_mul(x, b.build_iconst_i64(3));
        Value* ok = b.build_eq(tag, b.build_iconst_i32(1));
        Instruction* g = b.build_guard(ok, "cc_no_stub", {a, x});
        g->set_resume_id(7);
        b.build_ret(b.build_add(a, b.build_iconst_i64(1)));
        BasicBlock* slow = b.append_block("resume7");
        b.position_at_end(slow);
        Value* ra = b.add_param(Type::i64());
        Value* rx = b.add_param(Type::i64());
        b.build_ret(b.build_add(ra, b.build_call("cc_helper", Type::i64(), {rx})));
        callee->add_resume_point(7, slow);
    }
    Function* caller = mod.create_function("cc_caller", Type::i64(), {Type::i64(), Type::i32()});
    {
        Builder b(*caller);
        b.position_at_end(b.append_block("entry"));
        Value* x = b.add_param(Type::i64());
        Value* tag = b.add_param(Type::i32());
        Value* r = b.build_call("cc_callee", Type::i64(), {x, tag});
        b.build_ret(b.build_add(r, b.build_iconst_i64(1000)));
    }
}

} // namespace

TEST_CASE("Carried callee deopt - failures in a callee's copy invalidate the caller's tier-2 code") {
    Module mod("cc_carried");
    build_program(mod);
    FunctionDispatchTable prog;
    FunctionHandle* caller = prog.get_or_create("cc_caller", mod.get_function("cc_caller"));
    FunctionHandle* callee = prog.get_or_create("cc_callee", mod.get_function("cc_callee"));
    prog.get_or_create("cc_helper", mod.get_function("cc_helper"));

    CodeInstaller inst(prog);
    CodeInstallResult r = inst.install_tier2(*caller, mod, "cc_caller");
    if (!r.success) std::cerr << r.error_message << "\n";
    REQUIRE(r.success);
    REQUIRE(caller->tier() == TierLevel::Tier2_Optimized);

    FastInterpreter fi;
    fi.set_dispatch_table(&prog);
    fi.set_module(&mod);
    const std::vector<RuntimeValue> pass = {RuntimeValue::from_i64(5), RuntimeValue::from_i32(1)};
    const std::vector<RuntimeValue> fail = {RuntimeValue::from_i64(5), RuntimeValue::from_i32(0)};
    CHECK_EQ(caller->call(fi, pass).as_i64(), 16 + 1000);
    CHECK_EQ(prog.pipeline().tier2_deopts(), 0u);

    // Every call fails the carried guard until the caller's code goes: at
    // most the threshold's worth of deopts, however many calls follow.
    const uint64_t threshold = prog.tiering().get_feedback("cc_caller").deopt_threshold();
    constexpr int kCalls = 200;
    for (int i = 0; i < kCalls; ++i) CHECK_EQ(caller->call(fi, fail).as_i64(), 15 + 500 + 1000);
    CHECK(prog.pipeline().tier2_deopts() <= threshold);
    CHECK(caller->tier() != TierLevel::Tier2_Optimized);
    CHECK(callee->tier() != TierLevel::Tier2_Optimized);

    // The program still answers both ways.
    CHECK_EQ(caller->call(fi, pass).as_i64(), 16 + 1000);
    CHECK_EQ(caller->call(fi, fail).as_i64(), 15 + 500 + 1000);
}

TEST_CASE("Carried callee deopt - a callee whose own code is not tier 2 still charges the caller") {
    Module mod("cc_carried_bailed");
    build_program(mod);
    FunctionDispatchTable prog;
    FunctionHandle* caller = prog.get_or_create("cc_caller", mod.get_function("cc_caller"));
    FunctionHandle* callee = prog.get_or_create("cc_callee", mod.get_function("cc_callee"));
    prog.get_or_create("cc_helper", mod.get_function("cc_helper"));

    // The callee's own tier-2 code is compiled and then dropped, as a
    // callee that bailed out of tier 2 is: the caller's copy of it is then
    // the only tier-2 code of it left running.
    CodeInstaller inst(prog);
    REQUIRE(inst.install_tier2(*callee, mod, "cc_callee").success);
    callee->invalidate_optimized();
    prog.tiering().get_feedback("cc_callee").trigger_bailout("dropped by the test");
    REQUIRE(callee->tier() != TierLevel::Tier2_Optimized);
    REQUIRE(inst.install_tier2(*caller, mod, "cc_caller").success);
    REQUIRE(caller->tier() == TierLevel::Tier2_Optimized);

    FastInterpreter fi;
    fi.set_dispatch_table(&prog);
    fi.set_module(&mod);
    const std::vector<RuntimeValue> fail = {RuntimeValue::from_i64(5), RuntimeValue::from_i32(0)};
    const uint64_t threshold = prog.tiering().get_feedback("cc_caller").deopt_threshold();
    for (int i = 0; i < 200; ++i) CHECK_EQ(caller->call(fi, fail).as_i64(), 15 + 500 + 1000);
    CHECK(prog.pipeline().tier2_deopts() <= threshold);
    CHECK(caller->tier() != TierLevel::Tier2_Optimized);
}

TEST_CASE("Carried callee deopt - tier-1 callers stop entering tier-2 code once it is dropped") {
    Module mod("cc_tier1_caller");
    build_program(mod);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(true);
    prog.pipeline().initialize(cfg);
    FunctionHandle* callee = prog.get_or_create("cc_callee", mod.get_function("cc_callee"));
    prog.get_or_create("cc_helper", mod.get_function("cc_helper"));

    // The callee has tier-2 code when its caller is baseline-compiled.
    CodeInstaller inst(prog);
    REQUIRE(inst.install_tier2(*callee, mod, "cc_callee").success);
    REQUIRE(callee->tier() == TierLevel::Tier2_Optimized);

    FastInterpreter fi;
    fi.set_dispatch_table(&prog);
    fi.set_module(&mod);
    const std::vector<RuntimeValue> pass = {RuntimeValue::from_i64(5), RuntimeValue::from_i32(1)};
    const std::vector<RuntimeValue> fail = {RuntimeValue::from_i64(5), RuntimeValue::from_i32(0)};
    for (int i = 0; i < 4; ++i) CHECK_EQ(fi.run(*mod.get_function("cc_caller"), pass).as_i64(), 16 + 1000);
    FunctionHandle* caller = prog.find("cc_caller");
    REQUIRE(caller != nullptr);
    REQUIRE(caller->tier() == TierLevel::Tier1_Baseline);
    REQUIRE(prog.pipeline().tier2_deopts() == 0u);

    // The callee's guard fails until its code is dropped; the caller's
    // baseline code must then reach the callee's lower tier, not keep
    // entering the dropped code and deoptimizing on every call.
    const uint64_t threshold = prog.tiering().get_feedback("cc_callee").deopt_threshold();
    for (int i = 0; i < 200; ++i) CHECK_EQ(caller->call(fi, fail).as_i64(), 15 + 500 + 1000);
    CHECK(callee->tier() != TierLevel::Tier2_Optimized);
    CHECK(prog.pipeline().tier2_deopts() <= threshold);
}

#endif
