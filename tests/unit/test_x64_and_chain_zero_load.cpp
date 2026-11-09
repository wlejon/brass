#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <cstdint>

// An and-chain leaf that compares a loaded value with zero, relationally:
// `ult 0, len` is the bounds check of a constant index 0 (bronze emits it for
// `a[0]`), and `ugt len, 0` its mirror. append_fused_compare lowers any
// comparison with a zero side as a `test` of the other side's register, so
// the analysis must leave that side's load unfolded; it used to fold it for
// every comparison but eq / ne, and the leaf's test then named a register the
// load never got ("operand has no virtual register ... in block and_chain").

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

// (i < len) && (i != 7) && (0 <u len): the zero comparison is the chain's
// last leaf, where the failure was; g puts the zero on the right. Each zero
// comparison reads its own load, used once, as bronze's per-check length
// reads are: only a single-use load in the block is ever folded. The branch
// takes no block arguments, as a chain requires.
const char* kSrc = R"(module @m
func @f(%p: i64, %i: i32) -> i64 {
entry:
  %len = load.i32 %p, 0
  %lt = slt.i32 %i, %len
  %seven = iconst.i32 7
  %n7 = ne.i32 %i, %seven
  %zero = iconst.i32 0
  %len0 = load.i32 %p, 0
  %nz = ult.i32 %zero, %len0
  %a = and.i32 %lt, %n7
  %b = and.i32 %a, %nz
  br_if %b, yes, no
yes:
  %one = iconst.i64 1
  ret %one
no:
  %none = iconst.i64 0
  ret %none
}
func @g(%p: i64, %i: i32) -> i64 {
entry:
  %len = load.i32 %p, 0
  %zero = iconst.i32 0
  %len0 = load.i32 %p, 0
  %pos = ugt.i32 %len0, %zero
  %lt = slt.i32 %i, %len
  %b = and.i32 %lt, %pos
  br_if %b, yes, no
yes:
  %one = iconst.i64 1
  ret %one
no:
  %none = iconst.i64 0
  ret %none
}
)";

}  // namespace

TEST_CASE("x64 and-chain - a leaf comparing a loaded value with zero keeps its load") {
    auto mod = parse_module(kSrc);
    REQUIRE(mod != nullptr);
    for (Function* fn : mod->functions()) {
        fn->rebuild_cfg_predecessors();
        X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
        auto lir = isel.lower(*fn);
        REQUIRE(lir != nullptr);
        for (const auto& bb : lir->blocks) {
            for (const auto& inst : bb->instructions) {
                for (const auto& u : inst->uses) {
                    CHECK(!(u.is_vreg() && !u.vreg_val.is_valid()));
                }
            }
        }
    }
}

TEST_CASE("x64 and-chain - zero comparisons on a load answer through the JIT") {
    auto mod = parse_module(kSrc);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto f = jit.get_function_ptr<int64_t (*)(const int32_t*, int32_t)>("f");
    auto g = jit.get_function_ptr<int64_t (*)(const int32_t*, int32_t)>("g");
    REQUIRE(f != nullptr);
    REQUIRE(g != nullptr);
    const int32_t lens[] = {0, 1, 5, 9, -3};
    const int32_t idx[] = {-1, 0, 3, 7, 8};
    for (int32_t len : lens) {
        for (int32_t i : idx) {
            const bool nz = static_cast<uint32_t>(len) > 0u;
            CHECK_EQ(f(&len, i), (i < len && i != 7 && nz) ? 1 : 0);
            CHECK_EQ(g(&len, i), (i < len && nz) ? 1 : 0);
        }
    }
}
