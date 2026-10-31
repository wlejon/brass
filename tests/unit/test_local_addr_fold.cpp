#include "brass/core/env.hpp"
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <cstdint>
#include <cstdlib>
#include <string>

// Register + register address folding (x64_isel_mem.cpp / aarch64_isel_mem.cpp
// fold_two_register_sum): a sum computed in another block than the memory
// operation, which is a value of its own anyway (here it is also added to an
// index), is read whole at the load ([t + 8]), not refolded as [p + d + 8],
// which kept both addends live to the loop as well as the sum. A sum used only
// as an address (%s) still folds everywhere, and a sum made in the load's own
// block (%u) too. Both targets' selectors are run explicitly, so the AArch64
// half runs on any host; the JIT half checks the answers on this one.

using namespace brass;
using namespace brass::codegen;

namespace {

const char* kFold = R"(module @fold

func @f(%p: i64, %d: i64, %n: i64) -> i64 {
entry:
  %t = add.i64 %p, %d
  %s = add.i64 %p, %n
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %a = load.i64 %t, 8
  %b = load.i64 %s
  %u = add.i64 %t, %i
  %c = load.i64 %u
  %s1 = add.i64 %acc, %a
  %s2 = add.i64 %s1, %b
  %s3 = add.i64 %s2, %c
  %eight = iconst.i64 8
  %i2 = add.i64 %i, %eight
  %lim = iconst.i64 32
  %k = slt.i64 %i2, %lim
  br_if %k, loop(%i2, %s3), out(%s3)
out(%r: i64):
  ret %r
}
)";

// Loads in the loop block whose address has an index register.
template <typename Lir>
size_t indexed_loads_in_loop(const Lir& lir) {
    size_t n = 0;
    for (const auto& bb : lir.blocks) {
        if (bb->name.find("loop") == std::string::npos) continue;
        for (const auto& inst : bb->instructions) {
            for (const auto& u : inst->uses) {
                if (u.is_mem() && u.mem_val.index_vreg.is_valid()) ++n;
            }
        }
    }
    return n;
}

bool seam_off() { return !brass::env_value("BRASS_NO_LOCAL_ADDR_FOLD").empty(); }

}  // namespace

TEST_CASE("Local address fold - x64: a sum that is also a value is read whole in another block") {
    auto mod = parse_module(kFold);
    REQUIRE(mod != nullptr);
    Function* fn = mod->functions().front();
    fn->rebuild_cfg_predecessors();
    x64::X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    // [p + n] (address-only %s) and [t + i] (%u, same block) keep their
    // index; [t + 8] has none. With the seam, [p + d + 8] has one as well.
    CHECK_EQ(indexed_loads_in_loop(*lir), seam_off() ? size_t(3) : size_t(2));
}

TEST_CASE("Local address fold - AArch64: the same rule, lowered with the AArch64 selector") {
    auto mod = parse_module(kFold);
    REQUIRE(mod != nullptr);
    Function* fn = mod->functions().front();
    fn->rebuild_cfg_predecessors();
    aarch64::AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    CHECK_EQ(indexed_loads_in_loop(*lir), seam_off() ? size_t(3) : size_t(2));
}

TEST_CASE("Local address fold - the loads read the same words through the JIT") {
    auto mod = parse_module(kFold);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t)>("f");
    REQUIRE(fn != nullptr);
    int64_t words[16];
    for (int k = 0; k < 16; ++k) words[k] = static_cast<int64_t>(uint64_t{0x9E3779B97F4A7C15ull} * uint64_t(k + 1));
    const auto base = static_cast<int64_t>(reinterpret_cast<intptr_t>(&words[0]));
    for (int64_t dw = 0; dw <= 4; ++dw) {
        for (int64_t nw = 0; nw <= 8; ++nw) {
            // uint64_t arithmetic: the sums wrap as add.i64 does.
            uint64_t want = 0;
            for (int64_t i = 0; i < 32; i += 8) {
                want += static_cast<uint64_t>(words[dw + 1]);
                want += static_cast<uint64_t>(words[nw]);
                want += static_cast<uint64_t>(words[dw + i / 8]);
            }
            CHECK_EQ(static_cast<uint64_t>(fn(base, dw * 8, nw * 8)), want);
        }
    }
}
