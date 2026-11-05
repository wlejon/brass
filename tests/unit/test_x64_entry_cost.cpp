#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <memory>
#include <unordered_set>

// What a small x64 function pays to be entered and left: a spilled 64-bit
// constant is read from the function's constant pool and never stored to its
// slot (linear_scan_constants.cpp), the rewrite's scratches R14 and R15 are
// saved only where named, and the stack check reads rsp and the pinned TLS
// register in place (fold_reserved_register_copies).

using namespace brass;
using namespace brass::codegen;

namespace {

// Six 64-bit constants live across two calls: more than the callee-saved
// registers hold, so some are spilled.
const char* kConstants = R"(module @m

func @g(%a: i64) -> i64 {
entry:
  %one = iconst.i64 1
  %r = add.i64 %a, %one
  ret %r
}

func @f(%x: i64) -> i64 {
entry:
  %k1 = iconst.i64 0xfff1000000000000
  %k2 = iconst.i64 0x0001000000000000
  %k3 = iconst.i64 0x0000ffffffffffff
  %k4 = iconst.i64 0x7ff8000000000000
  %k5 = iconst.i64 0x9e3779b97f4a7c15
  %k6 = iconst.i64 0xbf58476d1ce4e5b9
  %a = xor.i64 %x, %k1
  %b = call.i64 @g(%a)
  %c = call.i64 @g(%b)
  %d = xor.i64 %c, %k1
  %e = add.i64 %d, %k2
  %f = and.i64 %e, %k3
  %h = xor.i64 %f, %k4
  %i = add.i64 %h, %k5
  %j = xor.i64 %i, %k6
  ret %j
}
)";

int64_t expected(int64_t x) {
    auto u = [](uint64_t v) { return v; };
    uint64_t a = u(x) ^ 0xfff1000000000000ull;
    uint64_t c = a + 2;
    uint64_t d = c ^ 0xfff1000000000000ull;
    uint64_t e = d + 0x0001000000000000ull;
    uint64_t f = e & 0x0000ffffffffffffull;
    uint64_t h = f ^ 0x7ff8000000000000ull;
    uint64_t i = h + 0x9e3779b97f4a7c15ull;
    return static_cast<int64_t>(i ^ 0xbf58476d1ce4e5b9ull);
}

std::unique_ptr<LirFunction> allocate_f(Module& mod) {
    Function* fn = mod.get_function("f");
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    brass::x64::X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    lir->sort_blocks_rpo();
    LivenessAnalysis liveness(*lir);
    liveness.run();
    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::sysv64());
    regalloc.allocate();
    return lir;
}

bool names(const LirFunction& fn, x64::GPR g) {
    const PReg r = PReg::gpr(g);
    for (const auto& b : fn.blocks) {
        for (const auto& inst : b->instructions) {
            for (const auto* ops : {&inst->defs, &inst->uses}) {
                for (const auto& op : *ops) {
                    if (op.is_preg() && op.preg_val == r) return true;
                    if (op.is_mem() && (op.mem_val.base_preg == r || op.mem_val.index_preg == r)) return true;
                }
            }
        }
    }
    return false;
}

} // namespace

TEST_CASE("x64 entry - a spilled constant is read from the constant pool and never stored") {
    auto mod = parse_module(kConstants);
    REQUIRE(mod != nullptr);
    auto lir = allocate_f(*mod);
    REQUIRE(!lir->frame.constant_slots.empty());
    std::unordered_set<int32_t> pooled;
    for (const auto& [slot, value] : lir->frame.constant_slots) pooled.insert(slot);
    size_t reads = 0;
    for (const auto& b : lir->blocks) {
        for (const auto& inst : b->instructions) {
            for (const auto& d : inst->defs) CHECK_FALSE(d.is_spill_slot() && pooled.count(d.spill_slot));
            for (const auto& u : inst->uses) reads += u.is_spill_slot() && pooled.count(u.spill_slot);
        }
    }
    CHECK(reads >= 1u);

    auto run = parse_module(kConstants);
    REQUIRE(run != nullptr);
    REQUIRE(verify_module(*run));
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*run));
    auto f = jit.get_function_ptr<int64_t (*)(int64_t)>("f");
    REQUIRE(f != nullptr);
    for (int64_t x : {int64_t{0}, int64_t{7}, int64_t{-1}, int64_t{1} << 47, int64_t{0x123456789abcdef}}) {
        CHECK_EQ(f(x), expected(x));
    }
}

TEST_CASE("x64 entry - R14 and R15 are saved only where the rewrite named them") {
    auto mod = parse_module(kConstants);
    REQUIRE(mod != nullptr);
    auto lir = allocate_f(*mod);
    for (x64::GPR g : {x64::GPR::R14, x64::GPR::R15}) {
        const bool saved = (lir->frame.saved_callee_gprs & x64::reg_mask(g)) != 0;
        CHECK_EQ(saved, names(*lir, g));
    }
}

TEST_CASE("x64 entry - the stack check compares rsp with the limit at the pinned register in place") {
    Module mod("sp_check");
    mod.set_pinned_tls_register(true);
    // f() = sp < [tls + 0x38] ? 1 : 0
    Function* fn = mod.create_function("f", Type::i64(), {});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* tls = b.build_pinned_tls_read();
    Value* limit = b.build_load(Type::i64(), tls, 0x38);
    Value* sp = b.build_read_sp();
    Value* below = b.build_ult(sp, limit);
    BasicBlock* yes = b.append_block("yes");
    BasicBlock* no = b.append_block("no");
    b.position_at_end(entry);
    b.build_br_if(below, yes, no);
    b.position_at_end(yes);
    b.build_ret(b.build_iconst_i64(1));
    b.position_at_end(no);
    b.build_ret(b.build_iconst_i64(0));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    brass::x64::X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    bool folded = false;
    size_t copies = 0;
    const PReg rsp = PReg::gpr(x64::GPR::RSP), r13 = PReg::gpr(x64::GPR::R13);
    for (const auto& blk : lir->blocks) {
        for (const auto& inst : blk->instructions) {
            if (inst->opcode == LirOpcode::Mov && inst->uses.size() == 1 && inst->uses[0].is_preg() &&
                (inst->uses[0].preg_val == rsp || inst->uses[0].preg_val == r13)) {
                ++copies;
            }
            if (inst->opcode == LirOpcode::Cmp && inst->uses.size() == 2 && inst->uses[0].is_preg() &&
                inst->uses[0].preg_val == rsp && inst->uses[1].is_mem() && inst->uses[1].mem_val.base_preg == r13 &&
                inst->uses[1].mem_val.disp == 0x38) {
                folded = true;
            }
        }
    }
    CHECK(folded);
    CHECK_EQ(copies, size_t(0));
}
