// Regressions from bug sweep 27:
// - install_tier2 (and the queued tier-2 paths) checked the module's other
//   functions' handles against the source module, but recorded the target
//   handle's binding as-is. A handle bound to module B's @f given module A
//   compiled A's @f, published it on the handle and deoptimized into B's @f.
//   Such an install is now refused before compiling, without marking the
//   handle rejected, and automatic tier-up compiles from the module owning
//   the Function the handle is bound to rather than the active module.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
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

// @f(x) = x + k.
std::unique_ptr<Module> module_k(int k) {
    const std::string ks = std::to_string(k);
    return parse_ok("module @s27_" + ks +
                    "\n"
                    "func @f(%x: i64) -> i64 {\n"
                    "b0:\n"
                    "  %k = iconst.i64 " + ks + "\n"
                    "  %r = add %x, %k\n"
                    "  ret %r\n"
                    "}\n");
}

int64_t call0(FunctionHandle& h) {
    return h.call_native({RuntimeValue::from_i64(0)}).as_i64();
}

void init_no_tierup(FunctionDispatchTable& prog) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(false);
    prog.pipeline().initialize(cfg);
}

} // namespace

TEST_CASE("Sweep27 - install_tier2 refuses a handle bound to another module's function") {
    auto A = module_k(1000);
    auto B = module_k(2000);
    FunctionDispatchTable prog;
    init_no_tierup(prog);
    FunctionHandle* h = prog.get_or_create("f", B->get_function("f"));
    CodeInstaller installer(prog);

    CodeInstallResult res = installer.install_tier2(*h, *A, "f");
    CHECK(!res.success);
    CHECK(res.error_message.find("not bound") != std::string::npos);
    CHECK(!h->has_native_entry());
    CHECK(!h->tier2_rejected());
    CHECK(h->mir_function() == B->get_function("f"));

    // The clone overload, with no source: the module names differ.
    res = installer.install_tier2(*h, clone_module(*A), "f");
    CHECK(!res.success);
    CHECK(!h->has_native_entry());
    CHECK(!h->tier2_rejected());

    // Its own module still compiles it.
    REQUIRE(installer.install_tier2(*h, *B, "f").success);
    CHECK_EQ(call0(*h), 2000);
}

TEST_CASE("Sweep27 - queued tier-2 requests refuse a handle bound to another module's function") {
    auto A = module_k(1000);
    auto B = module_k(2000);
    FunctionDispatchTable prog;
    init_no_tierup(prog);
    FunctionHandle* h = prog.get_or_create("f", B->get_function("f"));

    BackgroundCompilerConfig bcfg;
    bcfg.num_threads = 0;
    bcfg.table = &prog;
    BackgroundCompiler bc(bcfg);
    CHECK(!bc.enqueue("f", *A, h));
    CHECK(!bc.enqueue("f", clone_module(*A), h));
    CHECK(!bc.is_queued_or_compiling("f"));
    CHECK(!prog.pipeline().enqueue_tier2("f", A.get(), h));
    CHECK(!prog.tiering().enqueue_compilation("f", A.get(), h));
    CHECK(!prog.pipeline().background_compiler().is_queued_or_compiling("f"));
    CHECK(!h->has_native_entry());
    CHECK(!h->tier2_rejected());
    bc.stop();
}

TEST_CASE("Sweep27 - automatic tier 2 compiles the bound function's module, not the active one") {
    auto A = module_k(1000);
    auto B = module_k(2000);
    FunctionDispatchTable prog;
    init_no_tierup(prog);
    prog.tiering().set_active_module(A.get());
    FunctionHandle* h = prog.get_or_create("f", B->get_function("f"));

    REQUIRE(prog.pipeline().compile_tier2_now("f", h));
    REQUIRE(h->has_native_entry());
    CHECK_EQ(call0(*h), 2000);
    prog.tiering().set_active_module(nullptr);
}

TEST_CASE("Sweep27 - automatic background tier 2 compiles the bound function's module") {
    auto A = module_k(1000);
    auto B = module_k(2000);
    FunctionDispatchTable prog;
    init_no_tierup(prog);
    prog.tiering().set_active_module(A.get());
    FunctionHandle* h = prog.get_or_create("f", B->get_function("f"));

    BackgroundCompiler& bc = prog.pipeline().background_compiler();
    bc.start(1);
    REQUIRE(prog.pipeline().enqueue_tier2("f", nullptr, h));
    bc.wait_idle();
    REQUIRE(h->has_native_entry());
    CHECK_EQ(call0(*h), 2000);
    bc.stop();
    prog.tiering().set_active_module(nullptr);
}
