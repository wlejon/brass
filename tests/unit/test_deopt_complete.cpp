// Deoptimization completeness: unbounded state maps through the x64 guard
// exit, and tier-2 guard failures resuming in Tier 0 end to end.
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::runtime;
using namespace brass::codegen;

namespace {

constexpr size_t kBigState = 200;
constexpr uintptr_t kPtrVal = 0x7ff0'1234'5670ull;
constexpr uintptr_t kRefVal = 0x0000'2000'0040ull;

// func @name(%p64: i64, %p32: i32, %pf: f64, %pp: ptr, %pr: gcref, %cond: i32) -> i64
//   200 state values cycling i64 / i32 / f64 / ptr / gcref
//   guard %cond, "", [...200] (resume id 77)
//   ret %p64
// When `with_resume` is set, resume block 77 takes all 200 values and
// returns a checksum over the integer and float ones.
Function* build_big_state_fn(Module& mod, const std::string& name, bool with_resume) {
    Function* fn = mod.create_function(name, Type::i64(),
        {Type::i64(), Type::i32(), Type::f64(), Type::ptr(), Type::gcref(), Type::i32()});
    Builder b(*fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* p64 = b.add_param(Type::i64());
    Value* p32 = b.add_param(Type::i32());
    Value* pf = b.add_param(Type::f64());
    Value* pp = b.add_param(Type::ptr());
    Value* pr = b.add_param(Type::gcref());
    Value* cond = b.add_param(Type::i32());

    std::vector<Value*> state;
    std::vector<Type> types;
    for (size_t i = 0; i < kBigState; ++i) {
        switch (i % 5) {
            case 0: state.push_back(b.build_add(p64, b.build_iconst_i64(static_cast<int64_t>(i) * 1000003))); break;
            case 1: state.push_back(b.build_add(p32, b.build_iconst_i32(-static_cast<int32_t>(i) * 7))); break;
            case 2: state.push_back(b.build_fadd(pf, b.build_fconst_f64(static_cast<double>(i) + 0.25))); break;
            case 3: state.push_back(pp); break;
            default: state.push_back(pr); break;
        }
        types.push_back(state.back()->type());
    }
    Instruction* g = b.build_guard(cond, "exit_stub", Span<Value* const>(state.data(), state.size()));
    g->set_resume_id(77);
    b.build_ret(p64);

    if (with_resume) {
        BasicBlock* slow = b.append_block("resume77");
        b.position_at_end(slow);
        std::vector<Value*> params;
        for (Type t : types) params.push_back(b.add_param(t));
        Value* acc = b.build_iconst_i64(0);
        for (size_t i = 0; i < kBigState; ++i) {
            Value* v = params[i];
            if (i % 5 == 1) v = b.build_sext_i64(v);
            else if (i % 5 == 2) v = b.build_fptosi_i64(v);
            else if (i % 5 != 0) continue;
            acc = b.build_add(b.build_mul(acc, b.build_iconst_i64(31)), v);
        }
        b.build_ret(acc);
        fn->add_resume_point(77, slow);
    }
    return fn;
}

std::vector<RuntimeValue> big_args(int32_t cond) {
    return {RuntimeValue::from_i64(-5000), RuntimeValue::from_i32(12345), RuntimeValue::from_f64(-3.5),
            RuntimeValue::from_ptr(kPtrVal), RuntimeValue::from_gcref(kRefVal), RuntimeValue::from_i32(cond)};
}

} // namespace

TEST_CASE("Deopt - DeoptFrame state is not capped") {
    DeoptFrame frame;
    for (size_t i = 0; i < 1000; ++i) frame.push_deopt_value(DeoptValue::i64(static_cast<int64_t>(i) * 3));
    CHECK_EQ(frame.count, 1000u);
    CHECK_EQ(frame.get_value(999).as_i64(), 2997);
    frame.set_value(1500, DeoptValue::f64(2.5));
    CHECK_EQ(frame.count, 1501u);
    CHECK_EQ(frame.get_value(1500).kind, DeoptValueKind::Float64);
    CHECK_EQ(frame.get_value(1200).as_i64(), 0);
    set_thread_deopt_frame(&frame);
    CHECK_EQ(get_thread_deopt_frame()->count, 1501u);
    CHECK_EQ(brass_get_thread_deopt_slots()[999], 2997u);
    set_thread_deopt_frame(nullptr);
}

TEST_CASE("Deopt - x64 guard exit materializes 200 mixed-type live values") {
    Module mod("deopt_big_state_mod");
    build_big_state_fn(mod, "big_state", false);

    // The oracle's view of the state map.
    Interpreter interp;
    std::vector<RuntimeValue> oracle;
    interp.set_deopt_handler([&](Interpreter&, const DeoptResult& d) -> RuntimeValue {
        oracle = d.state_map;
        return RuntimeValue::from_i64(0);
    });
    interp.run(*mod.get_function("big_state"), big_args(0));
    REQUIRE_EQ(oracle.size(), kBigState);

    JitExecutionEngine engine;
    REQUIRE(engine.compile_and_load(mod));

    DeoptFrame captured;
    bool called = false;
    register_deopt_handler([&](const DeoptFrame& f) -> void* {
        called = true;
        captured = f;
        return reinterpret_cast<void*>(uintptr_t{4242});
    });
    RuntimeValue fast = engine.invoke("big_state", big_args(1));
    RuntimeValue slow = engine.invoke("big_state", big_args(0));
    register_deopt_handler(nullptr);

    CHECK_EQ(fast.as_i64(), -5000);
    REQUIRE(called);
    CHECK_EQ(slow.as_i64(), 4242); // the handler's continuation result
    CHECK_EQ(captured.resume_id, 77u);
    REQUIRE_EQ(captured.count, kBigState);
    CHECK(captured.code_entry != nullptr);

    size_t mismatches = 0;
    for (size_t i = 0; i < kBigState; ++i) {
        DeoptValue v = captured.get_value(i);
        const RuntimeValue& o = oracle[i];
        bool ok = false;
        switch (i % 5) {
            case 0:
                ok = v.kind == DeoptValueKind::Int64 && v.as_i64() == -5000 + static_cast<int64_t>(i) * 1000003 &&
                     o.as_i64() == v.as_i64();
                break;
            case 1:
                ok = v.kind == DeoptValueKind::Int32 && v.as_i32() == 12345 - static_cast<int32_t>(i) * 7 &&
                     o.as_i32() == v.as_i32();
                break;
            case 2:
                ok = v.kind == DeoptValueKind::Float64 && v.as_f64() == -3.5 + (static_cast<double>(i) + 0.25) &&
                     o.as_f64() == v.as_f64();
                break;
            case 3:
                ok = v.kind == DeoptValueKind::Pointer && v.as_ptr() == kPtrVal && o.as_ptr() == kPtrVal;
                break;
            default:
                ok = v.kind == DeoptValueKind::GcRef && v.as_gcref() == kRefVal && o.as_gcref() == kRefVal;
                break;
        }
        if (!ok) {
            if (mismatches < 5) std::cerr << "state value " << i << " mismatched\n";
            ++mismatches;
        }
    }
    CHECK_EQ(mismatches, 0u);
}

namespace {

// func @name(%x: i64, %tag: i32) -> i64
//   %a = mul %x, 3
//   guard (eq %tag, 1), <exit>, [%a, %x]   (resume id 7)
//   ret %a + 1
// resume 7 (%a, %x): ret %a + %x * 100      (when exit is "")
// With `exit_stub`: @<exit_stub>(%a, %x) = %a - %x.
Function* build_spec_fn(Module& mod, const std::string& name, const std::string& exit_stub) {
    if (!exit_stub.empty()) {
        Function* stub = mod.create_function(exit_stub, Type::i64(), {Type::i64(), Type::i64()});
        Builder sb(*stub);
        sb.position_at_end(sb.append_block("entry"));
        Value* a = sb.add_param(Type::i64());
        Value* x = sb.add_param(Type::i64());
        sb.build_ret(sb.build_sub(a, x));
    }
    Function* fn = mod.create_function(name, Type::i64(), {Type::i64(), Type::i32()});
    Builder b(*fn);
    b.position_at_end(b.append_block("entry"));
    Value* x = b.add_param(Type::i64());
    Value* tag = b.add_param(Type::i32());
    Value* a = b.build_mul(x, b.build_iconst_i64(3));
    Value* c = b.build_eq(tag, b.build_iconst_i32(1));
    // "exit_stub" names no function: the guard resumes at its resume target.
    Instruction* g = b.build_guard(c, exit_stub.empty() ? "exit_stub" : exit_stub, {a, x});
    g->set_resume_id(7);
    b.build_ret(b.build_add(a, b.build_iconst_i64(1)));
    if (exit_stub.empty()) {
        BasicBlock* slow = b.append_block("resume7");
        b.position_at_end(slow);
        Value* ra = b.add_param(Type::i64());
        Value* rx = b.add_param(Type::i64());
        b.build_ret(b.build_add(ra, b.build_mul(rx, b.build_iconst_i64(100))));
        fn->add_resume_point(7, slow);
    }
    return fn;
}

// Installs `name` at tier 2 and checks that every call matches the oracle
// interpreter, that the failing guard invalidates the optimized code after
// the threshold, and that nothing recompiles or leaks afterwards.
void run_tier2_deopt_case(Module& mod, const std::string& name) {
    Function* fn = mod.get_function(name);
    REQUIRE(fn != nullptr);
    auto& fb = TieringRegistry::instance().get_feedback(name);
    fb.reset();
    fb.set_deopt_threshold(5);

    FunctionHandle* handle = FunctionDispatchTable::instance().get_or_create(name, fn);
    REQUIRE(handle != nullptr);
    CodeInstaller installer;
    CodeInstallResult res = installer.install_tier2(*handle, mod, name);
    if (!res.success) std::cerr << res.error_message << "\n";
    REQUIRE(res.success);
    REQUIRE(handle->tier() == TierLevel::Tier2_Optimized);

    auto& pipeline = MultiTierPipeline::instance();
    const uint64_t deopts0 = pipeline.tier2_deopts();
    const uint64_t inval0 = pipeline.tier2_invalidations();

    Interpreter oracle;
    FastInterpreter fast;
    fast.set_module(&mod);
    for (int iter = 0; iter < 20; ++iter) {
        const int64_t x = 5 + iter;
        std::vector<RuntimeValue> pass = {RuntimeValue::from_i64(x), RuntimeValue::from_i32(1)};
        std::vector<RuntimeValue> fail = {RuntimeValue::from_i64(x), RuntimeValue::from_i32(0)};
        CHECK_EQ(handle->call(fast, pass).as_i64(), oracle.run(*fn, pass).as_i64());
        CHECK_EQ(handle->call(fast, fail).as_i64(), oracle.run(*fn, fail).as_i64());
    }

    // Five deopts through the optimized code, then it was dropped for good.
    CHECK_EQ(pipeline.tier2_deopts() - deopts0, 5u);
    CHECK_EQ(pipeline.tier2_invalidations() - inval0, 1u);
    CHECK(handle->tier() != TierLevel::Tier2_Optimized);
    CHECK(!handle->has_native_entry());
    CHECK(fb.is_bailout_set());
    CHECK_EQ(handle->retired_engine_count(), 1u);

    // Tier 2 is never requested again for a bailed-out function.
    pipeline.on_invocation(name);
    CHECK(handle->tier() != TierLevel::Tier2_Optimized);
    FunctionDispatchTable::instance().forget_module(mod);
}

} // namespace

TEST_CASE("Deopt - tier-2 guard failure resumes in Tier 0 at the resume target") {
    Module mod("tier2_deopt_resume_mod");
    build_spec_fn(mod, "tier2_deopt_resume_fn", "");
    Interpreter oracle;
    CHECK_EQ(oracle.run(*mod.get_function("tier2_deopt_resume_fn"),
                        {RuntimeValue::from_i64(5), RuntimeValue::from_i32(0)}).as_i64(), 515);
    run_tier2_deopt_case(mod, "tier2_deopt_resume_fn");
}

TEST_CASE("Deopt - tier-2 guard failure calls the exit stub like the interpreter") {
    Module mod("tier2_deopt_stub_mod");
    build_spec_fn(mod, "tier2_deopt_stub_fn", "tier2_deopt_stub_exit");
    Interpreter oracle;
    CHECK_EQ(oracle.run(*mod.get_function("tier2_deopt_stub_fn"),
                        {RuntimeValue::from_i64(5), RuntimeValue::from_i32(0)}).as_i64(), 10);
    run_tier2_deopt_case(mod, "tier2_deopt_stub_fn");
}

TEST_CASE("Deopt - tier-2 resumes 200 mixed-type values into the interpreter") {
    Module mod("tier2_big_state_mod");
    Function* fn = build_big_state_fn(mod, "tier2_big_state_fn", true);
    auto& fb = TieringRegistry::instance().get_feedback("tier2_big_state_fn");
    fb.reset();
    fb.set_deopt_threshold(5);
    FunctionHandle* handle = FunctionDispatchTable::instance().get_or_create("tier2_big_state_fn", fn);
    CodeInstaller installer;
    CodeInstallResult res = installer.install_tier2(*handle, mod, "tier2_big_state_fn");
    if (!res.success) std::cerr << res.error_message << "\n";
    REQUIRE(res.success);

    Interpreter oracle;
    const int64_t expected = oracle.run(*fn, big_args(0)).as_i64();
    FastInterpreter fast;
    fast.set_module(&mod);
    CHECK_EQ(handle->call(fast, big_args(0)).as_i64(), expected);
    CHECK_EQ(handle->call(fast, big_args(1)).as_i64(), -5000);
    CHECK(handle->tier() == TierLevel::Tier2_Optimized);
    FunctionDispatchTable::instance().forget_module(mod);
}

TEST_CASE("Deopt - tier-2 install refuses a guard with nowhere to deoptimize") {
    Module mod("tier2_deopt_orphan_mod");
    Function* fn = mod.create_function("tier2_deopt_orphan_fn", Type::i64(), {Type::i64(), Type::i32()});
    Builder b(*fn);
    b.position_at_end(b.append_block("entry"));
    Value* x = b.add_param(Type::i64());
    Value* c = b.add_param(Type::i32());
    b.build_guard(c, "exit_stub", {x})->set_resume_id(9);
    b.build_ret(x);

    FunctionHandle* handle = FunctionDispatchTable::instance().get_or_create("tier2_deopt_orphan_fn", fn);
    CodeInstaller installer;
    CodeInstallResult res = installer.install_tier2(*handle, mod, "tier2_deopt_orphan_fn");
    CHECK(!res.success);
    CHECK(res.error_message.find("cannot deoptimize") != std::string::npos);
    CHECK(!handle->has_native_entry());
    FunctionDispatchTable::instance().forget_module(mod);
}
