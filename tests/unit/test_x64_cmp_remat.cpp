#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <cstdint>
#include <cstdlib>

// Compare rematerialization (x64_isel_cmp.cpp, find_remat_compares): a
// comparison whose every use is a branch or select condition, some of them in
// other blocks than the comparison's own, is re-emitted at each use instead of
// being materialized once with setcc and tested everywhere. The cases here pin
// both halves: no setcc is left, and the function still computes the right
// answer through the JIT, including a use past a block that redefines the
// flags and a comparison against a constant on either side.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

// %c is used by a select in entry, a branch in entry's successor and a select
// in a third block; %k (a constant on the left) by two branches.
const char* kShared = R"(module @remat

func @f(%x: i64, %y: i64, %z: i64) -> i64 {
entry:
  %c = ult.i64 %x, %y
  %seven = iconst.i64 7
  %k = slt.i64 %seven, %z
  %s = select.i64 %c, %x, %z
  %d = slt.i64 %z, %s
  br_if %d, p, q
p:
  %t = select.i64 %c, %y, %z
  br_if %k, out(%t), out(%seven)
q:
  %w = add.i64 %x, %z
  br_if %c, out(%w), r
r:
  br_if %k, out(%y), out(%s)
out(%v: i64):
  ret %v
}
)";

int64_t reference(int64_t x, int64_t y, int64_t z) {
    const bool c = static_cast<uint64_t>(x) < static_cast<uint64_t>(y);
    const bool k = 7 < z;
    const int64_t s = c ? x : z;
    if (z < s) {
        const int64_t t = c ? y : z;
        return k ? t : 7;
    }
    // add.i64 wraps; a signed C++ add that overflows (INT64_MIN + INT64_MIN)
    // is undefined, and clang -O3 on arm64 folded it to INT64_MIN.
    if (c) return static_cast<int64_t>(static_cast<uint64_t>(x) + static_cast<uint64_t>(z));
    return k ? y : s;
}

}  // namespace

TEST_CASE("x64 compare remat - a condition used across blocks leaves no setcc") {
    if (std::getenv("BRASS_NO_CMP_REMAT")) return;
    auto mod = parse_module(kShared);
    REQUIRE(mod != nullptr);
    Function* fn = mod->functions().front();
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    size_t setcc = 0;
    for (const auto& bb : lir->blocks) {
        for (const auto& inst : bb->instructions) {
            if (inst->opcode == LirOpcode::Setcc) ++setcc;
        }
    }
    CHECK_EQ(setcc, size_t(0));
}

TEST_CASE("x64 compare remat - the shared conditions compute the same answers through the JIT") {
    auto mod = parse_module(kShared);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t)>("f");
    REQUIRE(fn != nullptr);
    const int64_t vals[] = {-9, -1, 0, 1, 6, 7, 8, 40, INT64_MIN, INT64_MAX};
    for (int64_t x : vals) {
        for (int64_t y : vals) {
            for (int64_t z : vals) {
                CHECK_EQ(fn(x, y, z), reference(x, y, z));
            }
        }
    }
}
