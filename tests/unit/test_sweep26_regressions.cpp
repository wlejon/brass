// Regressions from bug sweep 26:
// - A tier-2 task cloned its module at enqueue time but read the handle's
//   bound Function only when the worker started compiling. A handle rebound
//   in between (get_or_create with another module's Function) was checked
//   against the new Function and got the old module's code as its tier-2
//   entry. The module's other functions were published to whatever their
//   handles were bound to at publish time. The bindings are now recorded
//   when the module is cloned, and code is published only to handles still
//   bound to the Function it was compiled from.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/mir/loop_opt.hpp>
#include <cstdint>
#include <memory>
#include <string>

using namespace brass;
using namespace brass::runtime;

namespace {

std::unique_ptr<Module> parse_ok(const std::string& src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    return mod;
}

// @f(x) = x + k, @g(x) = x + k + 1.
std::unique_ptr<Module> module_k(int k) {
    const std::string ks = std::to_string(k), k1 = std::to_string(k + 1);
    return parse_ok("module @s26_" + ks +
                    "\n"
                    "func @f(%x: i64) -> i64 {\n"
                    "b0:\n"
                    "  %k = iconst.i64 " + ks + "\n"
                    "  %r = add %x, %k\n"
                    "  ret %r\n"
                    "}\n"
                    "func @g(%x: i64) -> i64 {\n"
                    "b0:\n"
                    "  %k = iconst.i64 " + k1 + "\n"
                    "  %r = add %x, %k\n"
                    "  ret %r\n"
                    "}\n");
}

int64_t call0(FunctionHandle& h) {
    return h.call_native({RuntimeValue::from_i64(0)}).as_i64();
}

} // namespace

TEST_CASE("Sweep26 - a queued tier-2 task is not published to a handle rebound before it ran") {
    auto A = module_k(1000);
    auto B = module_k(2000);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.enable_background_compile = false;
    prog.pipeline().initialize(cfg);
    FunctionHandle* hf = prog.get_or_create("f", A->get_function("f"));
    FunctionHandle* hg = prog.get_or_create("g", A->get_function("g"));

    BackgroundCompilerConfig bcfg;
    bcfg.num_threads = 0;  // the task waits in the queue
    bcfg.table = &prog;
    BackgroundCompiler bc(bcfg);
    REQUIRE(bc.enqueue("f", *A, hf));
    // The host replaces module A with B before the worker starts.
    prog.get_or_create("f", B->get_function("f"));
    prog.get_or_create("g", B->get_function("g"));
    REQUIRE(hf->mir_function() == B->get_function("f"));
    bc.start(1);
    bc.wait_idle();

    // Neither the target nor the sibling runs A's code.
    CHECK(!hf->has_native_entry());
    CHECK(hf->tier() == TierLevel::Tier0_Interpreter);
    CHECK(!hg->has_native_entry());
    // The failure is not held against B's Function.
    CHECK(!hf->tier2_rejected());

    // B's own compile installs B's code.
    REQUIRE(bc.enqueue("f", *B, hf));
    bc.wait_idle();
    REQUIRE(hf->has_native_entry());
    CHECK_EQ(call0(*hf), 2000);
    REQUIRE(hg->has_native_entry());
    CHECK_EQ(call0(*hg), 2001);
    bc.stop();
}

TEST_CASE("Sweep26 - a queued task's sibling rebound before it ran keeps its own Function") {
    auto A = module_k(1000);
    auto B = module_k(2000);
    FunctionDispatchTable prog;
    TieringConfig cfg;
    cfg.enable_background_compile = false;
    prog.pipeline().initialize(cfg);
    FunctionHandle* hf = prog.get_or_create("f", A->get_function("f"));
    FunctionHandle* hg = prog.get_or_create("g", A->get_function("g"));

    BackgroundCompilerConfig bcfg;
    bcfg.num_threads = 0;
    bcfg.table = &prog;
    BackgroundCompiler bc(bcfg);
    REQUIRE(bc.enqueue("f", *A, hf));
    prog.get_or_create("g", B->get_function("g"));  // only the sibling moves
    bc.start(1);
    bc.wait_idle();

    REQUIRE(hf->has_native_entry());
    CHECK_EQ(call0(*hf), 1000);
    CHECK(!hg->has_native_entry());
    CHECK(hg->mir_function() == B->get_function("g"));
    bc.stop();
}

TEST_CASE("Sweep26 - install_tier2 publishes a sibling only to a handle bound to the source's Function") {
    auto A = module_k(1000);
    auto B = module_k(2000);
    FunctionDispatchTable prog;
    FunctionHandle* hf = prog.get_or_create("f", A->get_function("f"));
    FunctionHandle* hg = prog.get_or_create("g", B->get_function("g"));
    CodeInstaller installer(prog);
    REQUIRE(installer.install_tier2(*hf, *A, "f").success);
    CHECK_EQ(call0(*hf), 1000);
    CHECK(!hg->has_native_entry());
}

TEST_CASE("Sweep26 - install_tier2 with bindings refuses a handle rebound after the clone") {
    auto A = module_k(1000);
    auto B = module_k(2000);
    FunctionDispatchTable prog;
    FunctionHandle* hf = prog.get_or_create("f", A->get_function("f"));
    prog.get_or_create("g", A->get_function("g"));
    CodeInstaller installer(prog);
    Tier2Bindings bindings = installer.capture_tier2_bindings(*hf, *A, "f", A.get());
    auto copy = clone_module(*A);
    REQUIRE(copy != nullptr);
    prog.get_or_create("f", B->get_function("f"));
    CodeInstallResult res = installer.install_tier2(*hf, std::move(copy), "f", bindings);
    CHECK(!res.success);
    CHECK(!hf->has_native_entry());
    CHECK(!hf->tier2_rejected());
}
