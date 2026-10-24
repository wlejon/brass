// Range-check loop versioning (version_loops_on_range_checks, run by BCE): a
// counted loop whose checks the preheader decides from the first and last
// iteration runs a copy without them, and its result is the loop's own for
// every placement of the range against the array, in bounds or not.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace brass;

namespace {

std::unique_ptr<Module> parse(std::string_view text) {
    DiagnosticReporter diag;
    auto mod = parse_module(text, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    return mod;
}

bool has_block_named(const Function& fn, std::string_view part) {
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb->name().find(part) != std::string_view::npos) return true;
    }
    return false;
}

void version(Module& mod, Function& fn) {
    REQUIRE(run_bounds_check_elimination(fn, mod));
    DiagnosticReporter diag;
    const bool verified = verify_module(mod, &diag);
    if (!verified) std::cerr << diag.format_all() << "\n" << to_string(fn) << "\n";
    REQUIRE(verified);
    CHECK(has_block_named(fn, "_rcv"));
}

} // namespace

TEST_CASE("range check versioning - a checked read's sum is the loop's own wherever the range lies") {
    // s += idx in [0, len) ? p[idx] : 1000 for i in [lo, n), idx = i + off;
    // the sum leaves through the header, so the exit takes it from either copy.
    constexpr std::string_view kSum = R"(
func @f(%p: i64, %len: i64, %off: i64, %lo: i64, %n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br b1(%z, %lo)

b1(%s: i64, %i: i64):
  %c = slt.i64 %i, %n
  br_if %c, b2, b_exit

b2:
  %idx = add.i64 %i, %off
  %in = ult.i64 %idx, %len
  %miss = iconst.i64 1000
  br_if %in, b_load, b_latch(%miss)

b_load:
  %v = load_indexed.i64 %p, %idx, 8
  br b_latch(%v)

b_latch(%x: i64):
  %s2 = add.i64 %s, %x
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  br b1(%s2, %i2)

b_exit:
  ret %s
}
)";
    auto mod = parse(kSum);
    version(*mod, *mod->get_function("f"));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    auto f = jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t, int64_t, int64_t)>("f");
    REQUIRE(f != nullptr);
    std::vector<int64_t> data(64);
    for (size_t k = 0; k < data.size(); ++k) data[k] = static_cast<int64_t>(k * 7 + 3);
    const auto p = static_cast<int64_t>(reinterpret_cast<intptr_t>(data.data()));
    auto expected = [&](int64_t len, int64_t off, int64_t lo, int64_t n) {
        int64_t s = 0;
        for (int64_t i = lo; i < n; ++i) {
            const int64_t idx = i + off;
            s += (static_cast<uint64_t>(idx) < static_cast<uint64_t>(len)) ? data[static_cast<size_t>(idx)] : 1000;
        }
        return s;
    };
    for (int64_t len : {int64_t{0}, int64_t{1}, int64_t{17}, int64_t{64}}) {
        for (int64_t off : {int64_t{-5}, int64_t{-1}, int64_t{0}, int64_t{3}, int64_t{60}}) {
            for (int64_t lo : {int64_t{0}, int64_t{2}, int64_t{-3}}) {
                for (int64_t n : {int64_t{-1}, int64_t{0}, int64_t{1}, int64_t{16}, int64_t{17}, int64_t{63}, int64_t{70}}) {
                    CHECK_EQ(f(p, len, off, lo, n), expected(len, off, lo, n));
                }
            }
        }
    }
    // Terms past the bound the preheader proves exact take the checked loop.
    CHECK_EQ(f(p, 64, int64_t{1} << 60, 0, 4), 4000);
    CHECK_EQ(f(p, 64, 0, -(int64_t{1} << 40), -(int64_t{1} << 40) + 3), 3000);
}

TEST_CASE("range check versioning - a column walk scaled by a run-time stride stores only in bounds") {
    // for k in [0, n): idx = k * stride + col; if idx in [0, len) q[idx] = k.
    constexpr std::string_view kColumn = R"(
func @f(%q: i64, %len: i64, %stride: i64, %col: i64, %n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br b1(%z)

b1(%k: i64):
  %c = slt.i64 %k, %n
  br_if %c, b2, b_exit

b2:
  %row = mul.i64 %k, %stride
  %idx = add.i64 %row, %col
  %in = ult.i64 %idx, %len
  br_if %in, b_store, b_latch

b_store:
  store_indexed.i64 %q, %idx, 8, %k
  br b_latch

b_latch:
  %one = iconst.i64 1
  %k2 = add.i64 %k, %one
  br b1(%k2)

b_exit:
  ret %z
}
)";
    auto mod = parse(kColumn);
    version(*mod, *mod->get_function("f"));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*mod));
    auto f = jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t, int64_t, int64_t)>("f");
    REQUIRE(f != nullptr);
    for (int64_t len : {int64_t{0}, int64_t{30}, int64_t{48}}) {
        for (int64_t stride : {int64_t{-4}, int64_t{0}, int64_t{3}, int64_t{8}}) {
            for (int64_t col : {int64_t{-1}, int64_t{0}, int64_t{2}, int64_t{7}}) {
                for (int64_t n : {int64_t{0}, int64_t{1}, int64_t{5}, int64_t{6}, int64_t{12}}) {
                    std::vector<int64_t> got(64, -1), want(64, -1);
                    for (int64_t k = 0; k < n; ++k) {
                        const int64_t idx = k * stride + col;
                        if (static_cast<uint64_t>(idx) < static_cast<uint64_t>(len)) want[static_cast<size_t>(idx)] = k;
                    }
                    f(static_cast<int64_t>(reinterpret_cast<intptr_t>(got.data())), len, stride, col, n);
                    CHECK(got == want);
                }
            }
        }
    }
}
