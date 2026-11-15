// F64 demotion of a loop left by `i != limit` whose limit and index offset
// are read from memory (canonicalize_loop_bounds, f64_demote_bounds.cpp):
//
//     for (i = 0; i != stride; i++) buf[offset + i] = buf[i]
//
// Nothing says stride is a whole number the counter reaches, or that offset
// is one at all, so the preheader tests both and the integer loop runs only
// where they are; a copy of the loop as written runs every other pair. The
// loop is capped at 40 trips so a limit the counter never meets returns.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/f64_demote.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>
#include <vector>

using namespace brass;

namespace {

constexpr const char* kCopyLoop = R"(module @m

func @copy(%p: ptr, %buf: ptr, %len: i64) -> f64 {
entry:
  %stride = load.f64 %p, 0
  %offset = load.f64 %p, 8
  %zero = fconst.f64 0.0
  br head(%zero)
head(%i: f64):
  %more = ne.f64 %i, %stride
  br_if %more, body, done
body:
  %cap = fconst.f64 40.0
  %capped = sge.f64 %i, %cap
  br_if %capped, done, read
read:
  %ii = fptosi.i64 %i
  %in = ult.i64 %ii, %len
  br_if %in, sum, latch
sum:
  %v = load_indexed.f64 %buf, %ii, 8
  %at = add.f64 %offset, %i
  %ai = fptosi.i64 %at
  %back = sitofp.f64.i64 %ai
  %whole = eq.f64 %back, %at
  br_if %whole, bound, latch
bound:
  %inside = ult.i64 %ai, %len
  br_if %inside, put, latch
put:
  store_indexed.f64 %buf, %ai, 8, %v
  br latch
latch:
  %one = fconst.f64 1.0
  %i2 = add.f64 %i, %one
  br head(%i2)
done:
  ret %i
}
)";

std::unique_ptr<Module> parse(std::string_view text) {
    DiagnosticReporter diag;
    auto mod = parse_module(text, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    return mod;
}

} // namespace

TEST_CASE("F64 Demote - a loop left by != with its limit and index offset read from memory runs as integers where both are whole") {
    auto plain = parse(kCopyLoop);
    auto demoted = parse(kCopyLoop);
    Function* fn = demoted->get_function("copy");
    REQUIRE(fn != nullptr);
    CHECK(f64_demote_pass(*fn, {}));
    DiagnosticReporter diag;
    const bool verified = verify_function(*fn, &diag);
    if (!verified) std::cerr << diag.format_all() << "\n" << to_string(*fn) << "\n";
    REQUIRE(verified);

    bool has_copy = false;
    for (const BasicBlock* bb : fn->blocks()) {
        if (!bb) continue;
        const std::string_view name = bb->name();
        if (name.find("_f64c") != std::string_view::npos) has_copy = true;
        // The loop the test passes runs its counter as an integer and adds
        // the offset to it as one.
        if (name == "head") CHECK_EQ(bb->param(0)->type(), Type::i64());
        if (name == "sum") {
            for (const Instruction* inst : *bb) {
                if (inst->opcode() == Opcode::add) CHECK_EQ(inst->type(), Type::i64());
            }
        }
    }
    CHECK(has_copy);

    codegen::JitExecutionEngine as_written(Target::host());
    REQUIRE(as_written.compile_and_load(*plain));
    codegen::JitExecutionEngine as_demoted(Target::host());
    REQUIRE(as_demoted.compile_and_load(*demoted));
    using Copy = double (*)(const double*, double*, int64_t);
    auto want = as_written.get_function_ptr<Copy>("copy");
    auto got = as_demoted.get_function_ptr<Copy>("copy");
    REQUIRE(want != nullptr);
    REQUIRE(got != nullptr);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double strides[] = {4.0, 0.0, -0.0, 1.0, 16.0, 20.0, 2.5, -3.0, 1e9, 3e9, nan, 39.0};
    const double offsets[] = {8.0, 0.0, -0.0, 14.0, 8.5, -2.0, 1e10, -1e10, nan, 2147483647.0, 2147483648.0, -2147483648.0};
    for (double stride : strides) {
        for (double offset : offsets) {
            const double args[2] = {stride, offset};
            std::vector<double> a(16), b(16);
            for (size_t i = 0; i < 16; ++i) a[i] = b[i] = static_cast<double>(i) + 0.25;
            const double ra = want(args, a.data(), 16);
            const double rb = got(args, b.data(), 16);
            CHECK_EQ(ra, rb);
            CHECK(a == b);
        }
    }
}
