// Per-program dispatch tables: an owned FunctionDispatchTable keys handles
// within its program, so two programs with a function of the same name each
// call their own; destroying one releases its handles and code, and a new
// program built at the same address is never confused with it.
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::runtime;
using namespace brass::codegen;

#if defined(__x86_64__) || defined(_M_X64)

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

// @<f>() = k; @<main>() = @<f>() + 1000.
std::string program_src(const std::string& mod, const std::string& f, const std::string& main, int k) {
    return "module @" + mod + "\n"
           "func @" + f + "() -> i64 {\nentry:\n  %r = iconst.i64 " + std::to_string(k) + "\n  ret %r\n}\n"
           "func @" + main + "() -> i64 {\nentry:\n  %v = call.i64 @" + f + "()\n"
           "  %c = iconst.i64 1000\n  %r = add.i64 %v, %c\n  ret %r\n}\n";
}

// Baseline code for a function returning `k`, owned only by the returned
// pointer (the caller hands it to a handle).
std::shared_ptr<BaselineCompiledFunction> native_const(BaselineJitCompiler& compiler, Module& holder,
                                                       int k) {
    const std::string name = "pd_native_" + std::to_string(k);
    Function* fn = holder.create_function(name, Type::i64(), {});
    Builder b(*fn);
    b.position_at_end(b.append_block("entry"));
    b.build_ret(b.build_iconst_i64(k));
    auto compiled = std::make_shared<BaselineCompiledFunction>(compiler.compile(*fn));
    REQUIRE(compiled->is_valid());
    return compiled;
}

void route_native(FunctionDispatchTable& table, const Function& fn, std::shared_ptr<BaselineCompiledFunction> code) {
    FunctionHandle* h = table.get_or_create(fn.name(), &fn);
    h->set_native_entry(code->entry_point());
    h->set_tier(TierLevel::Tier1_Baseline);
    h->set_baseline_function(std::move(code));
}

} // namespace

TEST_CASE("Program dispatch - same-named functions in two programs each call their own") {
    auto mod_a = parse_or_fail(program_src("pd_a", "pd_f", "pd_main", 1));
    auto mod_b = parse_or_fail(program_src("pd_b", "pd_f", "pd_main", 2));
    Module natives("pd_natives");
    BaselineJitCompiler compiler;
    {
        FunctionDispatchTable prog_a;
        FunctionDispatchTable prog_b;
        CHECK(!prog_a.is_default());
        route_native(prog_a, *mod_a->get_function("pd_f"), native_const(compiler, natives, 100));
        route_native(prog_b, *mod_b->get_function("pd_f"), native_const(compiler, natives, 200));
        CHECK(!FunctionDispatchTable::instance().has("pd_f"));

        // The fast interpreter calls each program's native pd_f.
        FastInterpreter fa;
        fa.set_dispatch_table(&prog_a);
        FastInterpreter fb;
        fb.set_dispatch_table(&prog_b);
        CHECK_EQ(fa.run(*mod_a->get_function("pd_main")).as_i64(), 1100);
        CHECK_EQ(fb.run(*mod_b->get_function("pd_main")).as_i64(), 1200);
        CHECK_EQ(fa.run(*mod_a->get_function("pd_main")).as_i64(), 1100);

        // So does the oracle interpreter.
        Interpreter ia;
        ia.set_dispatch_table(&prog_a);
        Interpreter ib;
        ib.set_dispatch_table(&prog_b);
        CHECK_EQ(ia.run(*mod_a->get_function("pd_main")).as_i64(), 1100);
        CHECK_EQ(ib.run(*mod_b->get_function("pd_main")).as_i64(), 1200);

        // The default program never saw either.
        FastInterpreter fd;
        CHECK_EQ(fd.run(*mod_a->get_function("pd_main")).as_i64(), 1001);
        CHECK(!FunctionDispatchTable::instance().has("pd_f"));
        CHECK(!FunctionDispatchTable::instance().has("pd_main"));
    }
}

TEST_CASE("Program dispatch - baseline compile_module publishes into and links through its program") {
    auto mod_a = parse_or_fail(program_src("pd_bl_a", "pd_bl_f", "pd_bl_main", 3));
    auto mod_b = parse_or_fail(program_src("pd_bl_b", "pd_bl_f", "pd_bl_main", 4));
    FunctionDispatchTable prog_a;
    FunctionDispatchTable prog_b;
    BaselineJitCompiler ca;
    ca.set_dispatch_table(&prog_a);
    BaselineJitCompiler cb;
    cb.set_dispatch_table(&prog_b);
    ca.compile_module(*mod_a);
    cb.compile_module(*mod_b);
    CHECK(!FunctionDispatchTable::instance().has("pd_bl_f"));

    FunctionHandle* ma = prog_a.find("pd_bl_main");
    FunctionHandle* mb = prog_b.find("pd_bl_main");
    REQUIRE(ma != nullptr);
    REQUIRE(mb != nullptr);
    REQUIRE(ma->has_native_entry());
    REQUIRE(mb->has_native_entry());
    // pd_bl_main's call to pd_bl_f links lazily through its own program.
    CHECK_EQ(ma->call_native().as_i64(), 1003);
    CHECK_EQ(mb->call_native().as_i64(), 1004);
}

TEST_CASE("Program dispatch - destroying a program releases its code and a new one at the same address starts clean") {
    auto mod = parse_or_fail(program_src("pd_life", "pd_life_f", "pd_life_main", 5));
    const Function& f = *mod->get_function("pd_life_f");
    const Function& main_fn = *mod->get_function("pd_life_main");
    Module natives("pd_life_natives");
    BaselineJitCompiler compiler;

    std::optional<FunctionDispatchTable> prog;
    prog.emplace();
    FunctionDispatchTable* const addr = &*prog;
    std::weak_ptr<BaselineCompiledFunction> code;
    {
        auto c = native_const(compiler, natives, 300);
        code = c;
        route_native(*prog, f, std::move(c));
    }
    FunctionHandle* h = prog->find("pd_life_f");
    REQUIRE(h != nullptr);
    CHECK(!code.expired());

    FastInterpreter fast;
    fast.set_dispatch_table(addr);
    CHECK_EQ(fast.run(main_fn).as_i64(), 1300);

    prog.reset();
    CHECK(code.expired());

    // Same storage, a program with no native pd_life_f: the interpreter's
    // cached call target must not reach the dead handle.
    prog.emplace();
    REQUIRE(&*prog == addr);
    CHECK(!prog->has("pd_life_f"));
    CHECK_EQ(fast.run(main_fn).as_i64(), 1005);

    route_native(*prog, f, native_const(compiler, natives, 400));
    CHECK_EQ(fast.run(main_fn).as_i64(), 1400);
    prog.reset();
    CHECK_EQ(FunctionDispatchTable::instance().has("pd_life_f"), false);
}

namespace {

// func @name(%x: i64, %tag: i32) -> i64
//   %a = mul %x, 3
//   guard (eq %tag, 1), "exit_stub", [%a, %x]   (resume id 7)
//   ret %a + 1
// resume 7 (%a, %x): ret %a + @<helper>(%x)
// @<helper>(%x) = %x * mult.
Function* build_spec_program(Module& mod, const std::string& name, const std::string& helper, int64_t mult) {
    Function* h = mod.create_function(helper, Type::i64(), {Type::i64()});
    {
        Builder hb(*h);
        hb.position_at_end(hb.append_block("entry"));
        Value* x = hb.add_param(Type::i64());
        hb.build_ret(hb.build_mul(x, hb.build_iconst_i64(mult)));
    }
    Function* fn = mod.create_function(name, Type::i64(), {Type::i64(), Type::i32()});
    Builder b(*fn);
    b.position_at_end(b.append_block("entry"));
    Value* x = b.add_param(Type::i64());
    Value* tag = b.add_param(Type::i32());
    Value* a = b.build_mul(x, b.build_iconst_i64(3));
    Value* c = b.build_eq(tag, b.build_iconst_i32(1));
    Instruction* g = b.build_guard(c, "exit_stub", {a, x});
    g->set_resume_id(7);
    b.build_ret(b.build_add(a, b.build_iconst_i64(1)));
    BasicBlock* slow = b.append_block("resume7");
    b.position_at_end(slow);
    Value* ra = b.add_param(Type::i64());
    Value* rx = b.add_param(Type::i64());
    b.build_ret(b.build_add(ra, b.build_call(helper, Type::i64(), {rx})));
    fn->add_resume_point(7, slow);
    return fn;
}

} // namespace

TEST_CASE("Program dispatch - tier-2 install and deopt resume stay within their program") {
    Module mod_a("pd_spec_a");
    Module mod_b("pd_spec_b");
    Function* fa = build_spec_program(mod_a, "pd_spec", "pd_spec_helper", 100);
    Function* fb = build_spec_program(mod_b, "pd_spec", "pd_spec_helper", 1000);
    Module natives("pd_spec_natives");
    BaselineJitCompiler compiler;

    void* entry_a = nullptr;
    void* entry_b = nullptr;
    {
        FunctionDispatchTable prog_a;
        FunctionDispatchTable prog_b;
        // Program A's helper runs native code that returns 7 (not x * 100),
        // so a resume routed through A is visible in the result.
        route_native(prog_a, *mod_a.get_function("pd_spec_helper"), native_const(compiler, natives, 7));
        FunctionHandle* ha = prog_a.get_or_create("pd_spec", fa);
        FunctionHandle* hb = prog_b.get_or_create("pd_spec", fb);
        // The other functions of B's module get B's tier-2 code; the default
        // program's same-named handle is not touched.
        FunctionHandle* b_helper = prog_b.get_or_create("pd_spec_helper", mod_b.get_function("pd_spec_helper"));
        FunctionHandle* default_helper = FunctionDispatchTable::instance().get_or_create("pd_spec_helper");

        CodeInstaller inst_a(prog_a);
        CodeInstaller inst_b(prog_b);
        CHECK(&inst_a.dispatch_table() == &prog_a);
        CodeInstallResult ra = inst_a.install_tier2(*ha, mod_a, "pd_spec");
        if (!ra.success) std::cerr << ra.error_message << "\n";
        REQUIRE(ra.success);
        CodeInstallResult rb = inst_b.install_tier2(*hb, mod_b, "pd_spec");
        if (!rb.success) std::cerr << rb.error_message << "\n";
        REQUIRE(rb.success);
        entry_a = ra.entry_point;
        entry_b = rb.entry_point;
        CHECK(entry_a != entry_b);
        CHECK(ha->tier() == TierLevel::Tier2_Optimized);
        CHECK(hb->tier() == TierLevel::Tier2_Optimized);
        CHECK(b_helper->tier() == TierLevel::Tier2_Optimized);
        CHECK(!default_helper->has_native_entry());
        CHECK(!FunctionDispatchTable::instance().has("pd_spec"));
        CHECK(has_deopt_resumer(entry_a));
        CHECK(has_deopt_resumer(entry_b));

        const std::vector<RuntimeValue> pass = {RuntimeValue::from_i64(5), RuntimeValue::from_i32(1)};
        const std::vector<RuntimeValue> fail = {RuntimeValue::from_i64(5), RuntimeValue::from_i32(0)};
        for (bool fast_tier0 : {false, true}) {
            // Each program resumes with its own pipeline's Tier-0 choice.
            prog_a.pipeline().set_use_fast_interpreter(fast_tier0);
            prog_b.pipeline().set_use_fast_interpreter(fast_tier0);
            FastInterpreter fia;
            fia.set_dispatch_table(&prog_a);
            fia.set_module(&mod_a);
            FastInterpreter fib;
            fib.set_dispatch_table(&prog_b);
            fib.set_module(&mod_b);
            CHECK_EQ(ha->call(fia, pass).as_i64(), 16);
            CHECK_EQ(hb->call(fib, pass).as_i64(), 16);
            // Guard failure: resume computes a + helper(x) in the handle's program.
            CHECK_EQ(ha->call(fia, fail).as_i64(), 15 + 7);
            CHECK_EQ(hb->call(fib, fail).as_i64(), 15 + 5000);
        }
        // Each program counted its own two deopts; the default program none.
        CHECK_EQ(prog_a.tiering().get_feedback("pd_spec").deopt_count(), 2u);
        CHECK_EQ(prog_b.tiering().get_feedback("pd_spec").deopt_count(), 2u);
        CHECK_EQ(prog_a.pipeline().tier2_deopts(), 2u);
        CHECK_EQ(prog_b.pipeline().tier2_deopts(), 2u);
        CHECK(!TieringRegistry::instance().has("pd_spec"));
    }
    // Destroying the programs unregistered their tier-2 deopt resumers.
    CHECK(!has_deopt_resumer(entry_a));
    CHECK(!has_deopt_resumer(entry_b));
}

// ---------------------------------------------------------------------------
// Per-program tiering: each program's TieringRegistry and MultiTierPipeline.
// ---------------------------------------------------------------------------

namespace {

TieringConfig tier1_config(uint64_t threshold) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = threshold;
    cfg.invocation_tier2_threshold = 1000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(true);
    return cfg;
}

uint64_t invocations(const FunctionDispatchTable& prog, std::string_view name) {
    const TieringFeedback* fb = prog.tiering().find_feedback(name);
    return fb ? fb->invocation_count() : 0;
}

} // namespace

TEST_CASE("Program tiering - same-named functions tier up independently in each program") {
    auto mod_a = parse_or_fail(program_src("pt_a", "pt_f", "pt_main", 1));
    auto mod_b = parse_or_fail(program_src("pt_b", "pt_f", "pt_main", 2));
    {
        FunctionDispatchTable prog_a;
        FunctionDispatchTable prog_b;
        CHECK(&prog_a.tiering() != &prog_b.tiering());
        CHECK(&prog_a.tiering() != &TieringRegistry::instance());
        CHECK(&prog_a.pipeline() != &MultiTierPipeline::instance());
        CHECK(&prog_a.pipeline().dispatch_table() == &prog_a);
        CHECK(&FunctionDispatchTable::instance().tiering() == &TieringRegistry::instance());
        CHECK(&FunctionDispatchTable::instance().pipeline() == &MultiTierPipeline::instance());
        prog_a.pipeline().initialize(tier1_config(3));
        prog_b.pipeline().initialize(tier1_config(1000));

        FastInterpreter fa;
        fa.set_dispatch_table(&prog_a);
        FastInterpreter fb;
        fb.set_dispatch_table(&prog_b);
        for (int i = 0; i < 10; ++i) CHECK_EQ(fa.run(*mod_a->get_function("pt_main")).as_i64(), 1001);
        for (int i = 0; i < 2; ++i) CHECK_EQ(fb.run(*mod_b->get_function("pt_main")).as_i64(), 1002);

        // A crossed its threshold and runs baseline code; B did not.
        FunctionHandle* fa_h = prog_a.find("pt_f");
        REQUIRE(fa_h != nullptr);
        CHECK(fa_h->tier() == TierLevel::Tier1_Baseline);
        CHECK(fa_h->has_native_entry());
        FunctionHandle* fb_h = prog_b.find("pt_f");
        CHECK((fb_h == nullptr || !fb_h->has_native_entry()));
        CHECK(prog_a.tiering().get_feedback("pt_f").current_tier() == TierLevel::Tier1_Baseline);
        CHECK(prog_b.tiering().get_feedback("pt_f").current_tier() == TierLevel::Tier0_Interpreter);
        CHECK(invocations(prog_a, "pt_main") >= 10);
        CHECK_EQ(invocations(prog_b, "pt_main"), 2u);
        CHECK_EQ(invocations(prog_b, "pt_f"), 2u);
        CHECK(prog_a.pipeline().find_baseline_compiled("pt_f") != nullptr);
        CHECK(prog_b.pipeline().find_baseline_compiled("pt_f") == nullptr);
        CHECK(prog_a.pipeline().active_stack_maps().find_function_by_name("pt_f") != nullptr);
        CHECK(prog_b.pipeline().active_stack_maps().find_function_by_name("pt_f") == nullptr);
        // The default program saw none of it.
        CHECK(!TieringRegistry::instance().has("pt_f"));
        CHECK(!FunctionDispatchTable::instance().has("pt_f"));
        CHECK(MultiTierPipeline::instance().find_baseline_compiled("pt_f") == nullptr);

        // Baseline code counts through the feedback baked into it, which
        // reaches its own program's pipeline: B's count does not move.
        const uint64_t a0 = invocations(prog_a, "pt_f");
        for (int i = 0; i < 5; ++i) CHECK_EQ(fa_h->call_native().as_i64(), 1);
        CHECK_EQ(invocations(prog_a, "pt_f"), a0 + 5);
        CHECK_EQ(invocations(prog_b, "pt_f"), 2u);
        CHECK(!TieringRegistry::instance().has("pt_f"));

        // Tier B up too: two same-named baseline functions, each counting
        // into its own program.
        prog_b.pipeline().set_config(tier1_config(3));
        prog_b.tiering().get_feedback("pt_f").set_invocation_tier1_threshold(3);
        prog_b.tiering().get_feedback("pt_main").set_invocation_tier1_threshold(3);
        for (int i = 0; i < 3; ++i) CHECK_EQ(fb.run(*mod_b->get_function("pt_main")).as_i64(), 1002);
        fb_h = prog_b.find("pt_f");
        REQUIRE(fb_h != nullptr);
        REQUIRE(fb_h->tier() == TierLevel::Tier1_Baseline);
        CHECK(fb_h->native_entry() != fa_h->native_entry());
        const uint64_t a1 = invocations(prog_a, "pt_f");
        const uint64_t b1 = invocations(prog_b, "pt_f");
        for (int i = 0; i < 4; ++i) CHECK_EQ(fb_h->call_native().as_i64(), 2);
        CHECK_EQ(invocations(prog_b, "pt_f"), b1 + 4);
        CHECK_EQ(invocations(prog_a, "pt_f"), a1);
    }
}

TEST_CASE("Program tiering - tier-2 deopt counts and bailouts are per program") {
    Module mod_a("pt_bail_a");
    Module mod_b("pt_bail_b");
    Function* fa = build_spec_program(mod_a, "pt_bail", "pt_bail_helper", 100);
    Function* fb = build_spec_program(mod_b, "pt_bail", "pt_bail_helper", 100);
    {
        FunctionDispatchTable prog_a;
        FunctionDispatchTable prog_b;
        prog_a.tiering().get_feedback("pt_bail").set_deopt_threshold(2);
        prog_b.tiering().get_feedback("pt_bail").set_deopt_threshold(2);
        FunctionHandle* ha = prog_a.get_or_create("pt_bail", fa);
        FunctionHandle* hb = prog_b.get_or_create("pt_bail", fb);
        CodeInstaller inst_a(prog_a);
        CodeInstaller inst_b(prog_b);
        REQUIRE(inst_a.install_tier2(*ha, mod_a, "pt_bail").success);
        REQUIRE(inst_b.install_tier2(*hb, mod_b, "pt_bail").success);

        const std::vector<RuntimeValue> fail = {RuntimeValue::from_i64(5), RuntimeValue::from_i32(0)};
        FastInterpreter fia;
        fia.set_dispatch_table(&prog_a);
        fia.set_module(&mod_a);
        FastInterpreter fib;
        fib.set_dispatch_table(&prog_b);
        fib.set_module(&mod_b);
        // A's guard fails until its speculation is judged wrong; B's once.
        for (int i = 0; i < 2; ++i) CHECK_EQ(ha->call(fia, fail).as_i64(), 15 + 500);
        CHECK_EQ(hb->call(fib, fail).as_i64(), 15 + 500);

        const TieringFeedback& fba = prog_a.tiering().get_feedback("pt_bail");
        const TieringFeedback& fbb = prog_b.tiering().get_feedback("pt_bail");
        CHECK(&fba != &fbb);
        CHECK_EQ(fba.deopt_count(), 2u);
        CHECK(fba.is_bailout_set());
        CHECK(ha->tier() != TierLevel::Tier2_Optimized);
        CHECK_EQ(prog_a.pipeline().tier2_invalidations(), 1u);
        CHECK_EQ(fbb.deopt_count(), 1u);
        CHECK(!fbb.is_bailout_set());
        CHECK(hb->tier() == TierLevel::Tier2_Optimized);
        CHECK_EQ(prog_b.pipeline().tier2_invalidations(), 0u);
        CHECK(!TieringRegistry::instance().has("pt_bail"));
    }
}

TEST_CASE("Program tiering - destroying a program drops its stack maps and background compiles") {
    std::string src = "module @pt_life\n";
    for (int i = 0; i < 24; ++i) {
        src += "func @pt_bg_" + std::to_string(i) + "(%x: i64) -> i64 {\nentry:\n"
               "  %c = iconst.i64 " + std::to_string(i) + "\n  %r = add.i64 %x, %c\n  ret %r\n}\n";
    }
    auto mod = parse_or_fail(src);
    const ModuleStackMap* before = brass_get_active_stack_maps();
    const bool default_live = MultiTierPipeline::instance().is_initialized();

    std::optional<FunctionDispatchTable> prog;
    prog.emplace();
    TieringConfig cfg = tier1_config(2);
    cfg.enable_background_compile = true;
    cfg.jit_threads = 1;
    prog->pipeline().initialize(cfg);
    const ModuleStackMap* maps = &prog->pipeline().active_stack_maps();
    CHECK(brass_get_active_stack_maps() == maps);

    // Tier pt_bg_0 up to baseline code: its stack map is the program's.
    FastInterpreter fast;
    fast.set_dispatch_table(&*prog);
    const std::vector<RuntimeValue> args = {RuntimeValue::from_i64(40)};
    for (int i = 0; i < 3; ++i) CHECK_EQ(fast.run(*mod->get_function("pt_bg_0"), args).as_i64(), 40);
    REQUIRE(prog->find("pt_bg_0")->tier() == TierLevel::Tier1_Baseline);
    CHECK(maps->find_function_by_name("pt_bg_0") != nullptr);
    std::weak_ptr<BaselineCompiledFunction> code = prog->pipeline().find_baseline_compiled("pt_bg_0");
    CHECK(!code.expired());

    // Queue tier-2 compiles for the rest on the program's own background
    // compiler, then destroy the program while they are pending/running.
    BackgroundCompiler& bg = prog->pipeline().background_compiler();
    CHECK(&bg != &BackgroundCompiler::instance());
    size_t queued = 0;
    for (int i = 1; i < 24; ++i) {
        if (prog->pipeline().enqueue_tier2("pt_bg_" + std::to_string(i), mod.get())) ++queued;
    }
    CHECK(queued > 0);
    CHECK(!BackgroundCompiler::instance().is_queued_or_compiling("pt_bg_1"));
    prog.reset();

    // The code and its stack maps went with the program, and the GC no
    // longer points at them.
    CHECK(code.expired());
    CHECK(brass_get_active_stack_maps() != maps);
    CHECK(brass_get_active_stack_maps() ==
          (default_live ? &MultiTierPipeline::instance().active_stack_maps() : nullptr));
    CHECK(!FunctionDispatchTable::instance().has("pt_bg_1"));
    CHECK(!TieringRegistry::instance().has("pt_bg_0"));
    brass_set_active_stack_maps(before);

    // A new program at the same address starts clean: the interpreter's
    // cached feedback is re-resolved into it.
    prog.emplace();
    fast.set_dispatch_table(&*prog);
    CHECK_EQ(fast.run(*mod->get_function("pt_bg_0"), args).as_i64(), 40);
    CHECK_EQ(prog->tiering().get_feedback("pt_bg_0").invocation_count(), 1u);
    FunctionHandle* fresh = prog->find("pt_bg_0");
    CHECK((fresh == nullptr || !fresh->has_native_entry()));
    prog.reset();
}

#endif
