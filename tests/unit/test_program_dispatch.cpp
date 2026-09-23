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

    auto& feedback = TieringRegistry::instance().get_feedback("pd_spec");
    feedback.reset();
    feedback.set_deopt_threshold(1000000);

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
            auto& pipeline = MultiTierPipeline::instance();
            const auto saved = pipeline.tier0_interpreter();
            pipeline.set_use_fast_interpreter(fast_tier0);
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
            pipeline.set_tier0_interpreter(saved);
        }
    }
    // Destroying the programs unregistered their tier-2 deopt resumers.
    CHECK(!has_deopt_resumer(entry_a));
    CHECK(!has_deopt_resumer(entry_b));
    feedback.reset();
}

#endif
