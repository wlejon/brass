#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <cstdint>

// A select's cmov starts from one operand and moves the other over it
// (x64_isel_select.cpp). Started from a value read again after the select,
// that value is copied first; when only the other operand dies at the select
// (its one use), the select starts from that one, with the condition
// inverted, and the copy is gone. Pinned: the moved operand is the value
// still live, and the function computes the same answers through the JIT.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

// %p is read again after the select; %q (p + 168) only by it.
const char* kSelect = R"(module @sel

func @f(%p: i64, %n: i64) -> i64 {
entry:
  %k = iconst.i64 168
  %q = add.i64 %p, %k
  %lim = iconst.i64 20
  %c = ugt.i64 %n, %lim
  %s = select.i64 %c, %q, %p
  %r = xor.i64 %s, %p
  ret %r
}
)";

int64_t reference(int64_t p, int64_t n) {
    const uint64_t q = static_cast<uint64_t>(p) + 168;
    const uint64_t s = static_cast<uint64_t>(n) > 20 ? q : static_cast<uint64_t>(p);
    return static_cast<int64_t>(s ^ static_cast<uint64_t>(p));
}

}  // namespace

TEST_CASE("x64 select - starts from the operand that dies there") {
    auto mod = parse_module(kSelect);
    REQUIRE(mod != nullptr);
    Function* fn = mod->functions().front();
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    // The parameter %p arrives in a vreg copied out of RDI; the cmov must
    // move that vreg (the live one), not start from it.
    const LirInst* cmov = nullptr;
    VReg p_vreg;
    for (const auto& bb : lir->blocks) {
        for (const auto& inst : bb->instructions) {
            if (inst->opcode == LirOpcode::Cmovcc) cmov = inst.get();
            if (inst->opcode == LirOpcode::ParallelCopy && !inst->defs.empty() && inst->defs[0].is_vreg()) {
                p_vreg = inst->defs[0].vreg_val;
            }
        }
    }
    REQUIRE(cmov != nullptr);
    REQUIRE(p_vreg.is_valid());
    REQUIRE(cmov->uses.size() == 2);
    CHECK(cmov->uses[1].is_vreg() && cmov->uses[1].vreg_val == p_vreg);
}

TEST_CASE("x64 select - the swapped select computes the same answers through the JIT") {
    auto mod = parse_module(kSelect);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("f");
    REQUIRE(fn != nullptr);
    const int64_t vals[] = {-9, -1, 0, 1, 19, 20, 21, 40, INT64_MIN, INT64_MAX};
    for (int64_t p : vals) {
        for (int64_t n : vals) {
            CHECK_EQ(fn(p, n), reference(p, n));
        }
    }
}
