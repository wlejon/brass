#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <cstdint>
#include <cstdlib>

// Compare rematerialization on AArch64 (aarch64_isel_fused_cmp.cpp,
// find_remat_compares), the port of x64_isel_cmp.cpp's: a comparison whose
// every use is a br_if or select condition, some of them in other blocks than
// the comparison's own, is re-emitted at each use (cmp + b.cond / csel)
// instead of being materialized once with cset and tested everywhere. The
// lowering case runs the AArch64 selector explicitly, so it holds on any host;
// the JIT case runs on the host's own target (AArch64 on an arm64 machine).

using namespace brass;
using namespace brass::codegen;
using namespace brass::aarch64;

namespace {

// %c (unsigned, two registers) is used by a select in entry, a select in p
// and a branch in q; %k (a constant on the left) by two branches; %z0 (an
// `and` tested against zero) by branches in two blocks.
const char* kShared = R"(module @remat

func @f(%x: i64, %y: i64, %z: i64) -> i64 {
entry:
  %c = ult.i64 %x, %y
  %seven = iconst.i64 7
  %three = iconst.i64 3
  %zero = iconst.i64 0
  %k = slt.i64 %seven, %z
  %a = and.i64 %x, %three
  %z0 = eq.i64 %a, %zero
  %s = select.i64 %c, %x, %z
  %d = slt.i64 %z, %s
  br_if %d, p, q
p:
  %t = select.i64 %c, %y, %z
  br_if %k, out(%t), u
u:
  br_if %z0, out(%seven), out(%a)
q:
  %w = add.i64 %x, %z
  br_if %c, out(%w), r
r:
  br_if %k, out(%y), v
v:
  br_if %z0, out(%s), out(%three)
out(%v: i64):
  ret %v
}
)";

int64_t reference(int64_t x, int64_t y, int64_t z) {
    const bool c = static_cast<uint64_t>(x) < static_cast<uint64_t>(y);
    const bool k = 7 < z;
    const int64_t a = static_cast<int64_t>(static_cast<uint64_t>(x) & 3u);
    const bool z0 = a == 0;
    const int64_t s = c ? x : z;
    if (z < s) {
        const int64_t t = c ? y : z;
        if (k) return t;
        return z0 ? 7 : a;
    }
    // add.i64 wraps: add as unsigned (a signed overflow is undefined in C++).
    if (c) return static_cast<int64_t>(static_cast<uint64_t>(x) + static_cast<uint64_t>(z));
    if (k) return y;
    return z0 ? s : 3;
}

size_t count_setcc(const LirFunction& lir) {
    size_t n = 0;
    for (const auto& bb : lir.blocks) {
        for (const auto& inst : bb->instructions) {
            if (inst && inst->opcode == LirOpcode::Setcc) ++n;
        }
    }
    return n;
}

}  // namespace

TEST_CASE("AArch64 compare remat - a condition used across blocks leaves no cset") {
    // Without the remat the three shared conditions each leave a cset.
    if (std::getenv("BRASS_NO_CMP_REMAT")) return;
    auto mod = parse_module(kShared);
    REQUIRE(mod != nullptr);
    Function* fn = mod->functions().front();
    fn->rebuild_cfg_predecessors();
    AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    CHECK_EQ(count_setcc(*lir), size_t(0));
}

TEST_CASE("AArch64 compare remat - the shared conditions compute the same answers through the JIT") {
    auto mod = parse_module(kShared);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t)>("f");
    REQUIRE(fn != nullptr);
    const int64_t vals[] = {-9, -1, 0, 1, 3, 4, 6, 7, 8, 40, INT64_MIN, INT64_MAX};
    for (int64_t x : vals) {
        for (int64_t y : vals) {
            for (int64_t z : vals) {
                CHECK_EQ(fn(x, y, z), reference(x, y, z));
            }
        }
    }
}
