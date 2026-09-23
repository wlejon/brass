// x64 baseline JIT lazy linking: a call or func_addr whose symbol is not
// resolvable at compile time goes through a stub that resolves when the
// symbol is registered later, and is a hard error (not a null call) if it is
// still unresolved when it runs.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/lazy_symbols.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/runtime/code_installer.hpp>
#include <iostream>
#include <memory>
#include <string>

using namespace brass;
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

// @ls_callee(x) = 3x + 1 lives in its own module, so compiling the callers
// cannot resolve it.
const char* kCallee = R"(module @callee
func @ls_callee(%0: i64) -> i64 {
entry:
  %c3 = iconst.i64 3
  %m = mul.i64 %0, %c3
  %c1 = iconst.i64 1
  %r = add.i64 %m, %c1
  ret %r
}
)";

const char* kCallers = R"(module @callers
func @ls_direct(%0: i64) -> i64 {
entry:
  %r = call.i64 @ls_callee(%0)
  %s = call.i64 @ls_callee(%r)
  ret %s
}
func @ls_addr() -> ptr {
entry:
  %fp = func_addr @ls_callee
  ret %fp
}
func @ls_indirect(%0: i64) -> i64 {
entry:
  %fp = func_addr @ls_callee
  %r = call_indirect.i64 %fp(%0)
  ret %r
}
)";

using I64Fn = int64_t (*)(int64_t);
using PtrFn = void* (*)();

} // namespace

TEST_CASE("Baseline lazy symbols - a callee registered after compile is called through its stub") {
    auto callers = parse_or_fail(kCallers);
    auto callee_mod = parse_or_fail(kCallee);
    BaselineJitCompiler compiler;

    auto direct = compiler.compile(*callers->get_function("ls_direct"));
    auto addr = compiler.compile(*callers->get_function("ls_addr"));
    auto indirect = compiler.compile(*callers->get_function("ls_indirect"));

    // func_addr yields the stub: stable, the same before and after the
    // symbol is registered.
    void* stub = compiler.lazy_stub("ls_callee");
    REQUIRE(stub != nullptr);
    CHECK(addr.get_function_ptr<PtrFn>()() == stub);
    CHECK(compiler.lazy_symbols()->resolved_target("ls_callee") == nullptr);

    auto callee = compiler.compile(*callee_mod->get_function("ls_callee"));
    compiler.register_external_symbol("ls_callee", callee.entry_point());
    CHECK(compiler.lazy_symbols()->resolved_target("ls_callee") == callee.entry_point());

    CHECK_EQ(direct.get_function_ptr<I64Fn>()(2), int64_t{22});   // 3*(3*2+1)+1
    CHECK_EQ(indirect.get_function_ptr<I64Fn>()(-5), int64_t{-14});
    CHECK(addr.get_function_ptr<PtrFn>()() == stub);
    CHECK_EQ(reinterpret_cast<I64Fn>(stub)(10), int64_t{31});
}

TEST_CASE("Baseline lazy symbols - the resolving call keeps register, float and stack arguments") {
    auto callers = parse_or_fail(R"(module @callers
func @ls_many_caller(%0: i64, %1: f64) -> f64 {
entry:
  %c2 = iconst.i64 2
  %c7 = iconst.i64 7
  %f = fconst.f64 0.5
  %r = call.f64 @ls_many(%0, %1, %c2, %f, %c7, %1)
  ret %r
}
)");
    auto callee_mod = parse_or_fail(R"(module @callee
func @ls_many(%0: i64, %1: f64, %2: i64, %3: f64, %4: i64, %5: f64) -> f64 {
entry:
  %a = add.i64 %0, %2
  %b = mul.i64 %a, %4
  %bf = sitofp.f64.i64 %b
  %c = add.f64 %bf, %1
  %d = mul.f64 %3, %5
  %e = add.f64 %c, %d
  ret %e
}
)");
    BaselineJitCompiler compiler;
    auto caller = compiler.compile(*callers->get_function("ls_many_caller"));
    auto callee = compiler.compile(*callee_mod->get_function("ls_many"));
    compiler.register_external_symbol("ls_many", callee.entry_point());
    using Fn = double (*)(int64_t, double);
    // (3+2)*7 + 1.5 + 0.5*1.5; the first call goes through the resolution path
    // only if registration had not filled the cell, so re-arm it.
    compiler.lazy_symbols()->define("ls_many", nullptr);
    CHECK_EQ(caller.get_function_ptr<Fn>()(3, 1.5), 35.0 + 1.5 + 0.75);
    CHECK_EQ(caller.get_function_ptr<Fn>()(3, 1.5), 35.0 + 1.5 + 0.75);
}

TEST_CASE("Baseline lazy symbols - first call resolves through the compiler's resolvers") {
    auto callers = parse_or_fail(kCallers);
    auto callee_mod = parse_or_fail(kCallee);
    BaselineJitCompiler compiler;
    auto direct = compiler.compile(*callers->get_function("ls_direct"));

    // Not registered, but the custom resolver knows it by the first call.
    BaselineJitCompiler other;
    auto callee = other.compile(*callee_mod->get_function("ls_callee"));
    void* entry = callee.entry_point();
    compiler.set_symbol_resolver([entry](std::string_view name) -> void* {
        return name == "ls_callee" ? entry : nullptr;
    });
    CHECK(compiler.lazy_symbols()->resolved_target("ls_callee") == nullptr);
    CHECK_EQ(direct.get_function_ptr<I64Fn>()(1), int64_t{13});
    CHECK(compiler.lazy_symbols()->resolved_target("ls_callee") == entry);
}

TEST_CASE("Baseline lazy symbols - mutually recursive siblings in compile_module") {
    auto mod = parse_or_fail(R"(module @m
func @ls_even(%0: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %is0 = eq.i64 %0, %z
  br_if %is0, yes, no
yes:
  %one = iconst.i64 1
  ret %one
no:
  %c1 = iconst.i64 1
  %n = sub.i64 %0, %c1
  %r = call.i64 @ls_odd(%n)
  ret %r
}
func @ls_odd(%0: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %is0 = eq.i64 %0, %z
  br_if %is0, yes, no
yes:
  ret %z
no:
  %c1 = iconst.i64 1
  %n = sub.i64 %0, %c1
  %r = call.i64 @ls_even(%n)
  ret %r
}
)");
    BaselineJitCompiler compiler;
    auto fns = compiler.compile_module(*mod);
    I64Fn even = nullptr;
    for (const auto& f : fns) {
        if (f.name() == "ls_even") even = f.get_function_ptr<I64Fn>();
    }
    REQUIRE(even != nullptr);
    CHECK_EQ(even(10), int64_t{1});
    CHECK_EQ(even(7), int64_t{0});
    runtime::FunctionDispatchTable::instance().forget_module(*mod);
}

namespace {
int64_t ls_host_shadowed(int64_t) { return -1; }
} // namespace

TEST_CASE("Baseline lazy symbols - a module function shadows a registered symbol of its name") {
    // The shape of bronze's `function sqrt(x)` beside the registered libm
    // `sqrt`: a direct call must reach the module's function, whether the
    // callee was compiled before the caller (direct call) or after it
    // (through the stub), as tier 2's linker resolves it. compile_module
    // goes last to first: @ls_sh_caller_late is compiled before its callee
    // (stub), @ls_sh_caller_early after its callee (direct call).
    auto mod = parse_or_fail(R"(module @shadow
func @ls_sh_callee_first(%0: i64) -> i64 {
entry:
  %c2 = iconst.i64 2
  %r = mul.i64 %0, %c2
  ret %r
}
func @ls_sh_caller_late(%0: i64) -> i64 {
entry:
  %r = call.i64 @ls_sh_callee_first(%0)
  ret %r
}
func @ls_sh_caller_early(%0: i64) -> i64 {
entry:
  %r = call.i64 @ls_sh_callee_last(%0)
  ret %r
}
func @ls_sh_callee_last(%0: i64) -> i64 {
entry:
  %c3 = iconst.i64 3
  %r = mul.i64 %0, %c3
  ret %r
}
)");
    BaselineJitCompiler compiler;
    compiler.register_external_symbol("ls_sh_callee_first", reinterpret_cast<void*>(&ls_host_shadowed));
    compiler.register_external_symbol("ls_sh_callee_last", reinterpret_cast<void*>(&ls_host_shadowed));
    auto fns = compiler.compile_module(*mod);
    I64Fn late = nullptr, early = nullptr;
    for (const auto& f : fns) {
        if (f.name() == "ls_sh_caller_late") late = f.get_function_ptr<I64Fn>();
        if (f.name() == "ls_sh_caller_early") early = f.get_function_ptr<I64Fn>();
    }
    REQUIRE(late != nullptr);
    REQUIRE(early != nullptr);
    CHECK_EQ(late(21), int64_t{42});
    CHECK_EQ(early(5), int64_t{15});
    runtime::FunctionDispatchTable::instance().forget_module(*mod);
}

TEST_CASE("Baseline lazy symbols - a call still unresolved at run time is a hard error") {
    auto callers = parse_or_fail(kCallers);
    auto callee_mod = parse_or_fail(kCallee);
    BaselineJitCompiler compiler;
    auto direct = compiler.compile(*callers->get_function("ls_direct"));
    auto indirect = compiler.compile(*callers->get_function("ls_indirect"));

    for (I64Fn fn : {direct.get_function_ptr<I64Fn>(), indirect.get_function_ptr<I64Fn>()}) {
        LazySymbolTable::clear_last_unresolved_symbol();
        std::string fault;
        bool ok = fuzz::DiffFuzzer::run_protected([fn] { (void)fn(4); }, fault);
        CHECK(!ok);
        CHECK(fault.find("Illegal Instruction") != std::string::npos || fault.find("SIGILL") != std::string::npos);
        CHECK_EQ(LazySymbolTable::last_unresolved_symbol(), std::string("ls_callee"));
    }

    // Registering it afterwards still links the same code.
    auto callee = compiler.compile(*callee_mod->get_function("ls_callee"));
    compiler.register_external_symbol("ls_callee", callee.entry_point());
    CHECK_EQ(direct.get_function_ptr<I64Fn>()(0), int64_t{4});
}

TEST_CASE("Baseline lazy symbols - compile_module rejects a symbol nothing resolves") {
    // The module is the whole program: @ls_callee is not one of its
    // functions and nobody registered it, so no stub could ever resolve.
    auto callers = parse_or_fail(kCallers);
    BaselineJitCompiler compiler;
    std::string error;
    try {
        (void)compiler.compile_module(*callers);
    } catch (const std::runtime_error& e) {
        error = e.what();
    }
    CHECK(error.find("'ls_callee'") != std::string::npos);
    runtime::FunctionDispatchTable::instance().forget_module(*callers);

    // Registered before compile_module: linked, and it runs.
    auto callee_mod = parse_or_fail(kCallee);
    BaselineJitCompiler linked;
    auto callee = linked.compile(*callee_mod->get_function("ls_callee"));
    linked.register_external_symbol("ls_callee", callee.entry_point());
    auto fns = linked.compile_module(*callers);
    I64Fn direct = nullptr;
    for (const auto& f : fns) {
        if (f.name() == "ls_direct") direct = f.get_function_ptr<I64Fn>();
    }
    REQUIRE(direct != nullptr);
    CHECK_EQ(direct(2), int64_t{22});
    runtime::FunctionDispatchTable::instance().forget_module(*callers);
}

TEST_CASE("Baseline lazy symbols - stubs outlive the compiler and fail cleanly") {
    auto callers = parse_or_fail(kCallers);
    BaselineCompiledFunction direct;
    {
        BaselineJitCompiler compiler;
        direct = compiler.compile(*callers->get_function("ls_direct"));
    }
    LazySymbolTable::clear_last_unresolved_symbol();
    std::string fault;
    I64Fn fn = direct.get_function_ptr<I64Fn>();
    CHECK(!fuzz::DiffFuzzer::run_protected([fn] { (void)fn(1); }, fault));
    CHECK_EQ(LazySymbolTable::last_unresolved_symbol(), std::string("ls_callee"));
}

#endif
