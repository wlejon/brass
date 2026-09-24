// Baseline JIT unwind data (x64 and AArch64): a C++ exception thrown by a helper that
// baseline code calls unwinds through the baseline frames (and the invoke
// thunk) to a C++ catch in the host. Without registered unwind data the
// process dies with an unhandled 0xE06D7363 instead.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/runtime/code_installer.hpp>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace brass;
using namespace brass::codegen;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

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

struct HelperError : std::runtime_error {
    explicit HelperError(int64_t v) : std::runtime_error("helper threw"), value(v) {}
    int64_t value;
};

int64_t uw_helper(int64_t x) {
    if (x == 7) throw HelperError(x);
    return x * 2;
}

// @uw_outer -> @uw_mid (a big frame) -> @uw_helper, which throws for 7.
const char* kModule = R"(module @uw
func @uw_mid(%0: i64) -> i64 {
entry:
  %buf = alloca 4096, 16
  %r = call.i64 @uw_helper(%0)
  %c1 = iconst.i64 1
  %s = add.i64 %r, %c1
  ret %s
}
func @uw_outer(%0: i64) -> i64 {
entry:
  %r = call.i64 @uw_mid(%0)
  %c10 = iconst.i64 10
  %s = add.i64 %r, %c10
  ret %s
}
)";

using I64Fn = int64_t (*)(int64_t);

I64Fn find_fn(const std::vector<BaselineCompiledFunction>& fns, std::string_view name) {
    for (const auto& f : fns) {
        if (f.name() == name) return f.get_function_ptr<I64Fn>();
    }
    return nullptr;
}

void check_unwinds(bool pinned_tls) {
    auto mod = parse_or_fail(kModule);
    mod->set_pinned_tls_register(pinned_tls);
    BaselineJitCompiler compiler;
    compiler.register_external_symbol("uw_helper", reinterpret_cast<void*>(&uw_helper));
    auto fns = compiler.compile_module(*mod);
    I64Fn outer = find_fn(fns, "uw_outer");
    REQUIRE(outer != nullptr);

    CHECK_EQ(outer(3), int64_t{17});  // 3*2 + 1 + 10

    // Directly from C++, then again: the frames unwound cleanly the first time.
    for (int round = 0; round < 2; ++round) {
        bool caught = false;
        try {
            (void)outer(7);
        } catch (const HelperError& e) {
            caught = true;
            CHECK_EQ(e.value, int64_t{7});
        }
        CHECK(caught);
    }

    // Through BaselineCompiledFunction::invoke (the argument thunk).
    const BaselineCompiledFunction* outer_fn = nullptr;
    for (const auto& f : fns) if (f.name() == "uw_outer") outer_fn = &f;
    REQUIRE(outer_fn != nullptr);
    bool caught = false;
    try {
        (void)outer_fn->invoke({RuntimeValue::from_i64(7)});
    } catch (const HelperError&) {
        caught = true;
    }
    CHECK(caught);
    CHECK_EQ(outer_fn->invoke({RuntimeValue::from_i64(4)}).as_i64(), int64_t{19});

    CHECK_EQ(outer(5), int64_t{21});
    runtime::FunctionDispatchTable::instance().forget_module(*mod);
}

} // namespace

TEST_CASE("Baseline unwind - a helper's C++ exception reaches the host catch") {
    check_unwinds(false);
}

TEST_CASE("Baseline unwind - frames that save R13 (pinned TLS) unwind too") {
    check_unwinds(true);
}

#endif
