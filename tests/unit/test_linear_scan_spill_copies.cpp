#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/branch_threading.hpp>
#include <bit>
#include <string>

// What the allocator makes of a value in its spill slot: a copy to or from the
// slot is the load or the store itself, a value only on its way into a block
// parameter is the one to give its register up, an x64 ALU source left in its
// slot is read from it in place, and a loop-free function's joins are not
// padded as loop heads.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

struct Allocated {
    std::unique_ptr<Module> mod;
    std::unique_ptr<LirFunction> lir;
};

// `name` lowered for x64 SysV and allocated, its blocks in the order written.
Allocated allocate(const char* text, const char* name, bool rpo = true) {
    Allocated out;
    out.mod = parse_module(text);
    REQUIRE(out.mod != nullptr);
    Function* fn = out.mod->get_function(name);
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    out.lir = isel.lower(*fn);
    REQUIRE(out.lir != nullptr);
    if (rpo) out.lir->sort_blocks_rpo();
    LivenessAnalysis liveness(*out.lir);
    liveness.run();
    LinearScanAllocator regalloc(*out.lir, liveness, CallingConvention::sysv64());
    regalloc.allocate();
    return out;
}

bool reads_slot(const LirInst& inst) {
    for (const auto& u : inst.uses) {
        if (u.is_spill_slot()) return true;
    }
    return false;
}

// A tagged value cannot be split, and live across a call it lives in its
// slot for good: the bitcast that makes it is a copy into the slot, the one
// after the call a copy out of it.
const char* kAcrossCall = R"(module @m

func @twice(%a: i64) -> i64 {
entry:
  %r = add.i64 %a, %a
  ret %r
}

func @bits(%x: i64) -> i64 {
entry:
  %t = bitcast.tagged.i64 %x
  %c = call.i64 @twice(%x)
  %u = bitcast.i64.tagged %t
  %r = xor.i64 %u, %c
  ret %r
}
)";

} // namespace

TEST_CASE("Linear Scan - a copy out of a spill slot is the reload itself") {
    Allocated a = allocate(kAcrossCall, "bits");
    size_t stores = 0;
    size_t reloads = 0;
    auto gpr = [](const LirOperand& op) { return op.is_preg() && op.preg_val.reg_class == RegClass::GPR; };
    auto scratch = [&](const LirOperand& op) {
        return gpr(op) && (op.preg_val.as_gpr() == GPR::R10 || op.preg_val.as_gpr() == GPR::R11);
    };
    for (const auto& b : a.lir->blocks) {
        for (const auto& inst : b->instructions) {
            // The general path took the slot's value through a scratch
            // register: two moves, and the scratch's write on the chain.
            for (const auto& d : inst->defs) CHECK_FALSE(scratch(d));
            if (inst->opcode != LirOpcode::Mov || inst->defs.size() != 1 || inst->uses.size() != 1) continue;
            if (inst->defs[0].is_spill_slot() && gpr(inst->uses[0])) ++stores;
            if (gpr(inst->defs[0]) && inst->uses[0].is_spill_slot()) ++reloads;
        }
    }
    CHECK(stores >= 1u);
    CHECK(reloads >= 1u);

    JitExecutionEngine jit;
    auto mod = parse_module(kAcrossCall);
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(int64_t)>("bits");
    REQUIRE(fn != nullptr);
    // Small-integer bits, so the tagged value is no pointer a stack map
    // could be asked to trace.
    for (int64_t x : {int64_t{3}, int64_t{-8}, int64_t{1} << 40}) CHECK_EQ(fn(x), x ^ (x + x));
}

namespace {

// Nine gcrefs loaded off %p and handed to a join, filling every register
// but %p's; then twelve more loads off %p, each summed at once. A gcref
// cannot be split and is read only by the block-argument copy; %p is read
// by every load. Short of registers at the first i64 load, the value to
// give one up is a gcref, which then waits in its slot for the copy, not %p,
// which was reloaded at each of the twelve.
std::string gather_text() {
    std::string s = "module @m\n\nfunc @gather(%p: ptr) -> i64 {\nentry:\n";
    std::string args, params, stores;
    for (int i = 0; i < 9; ++i) {
        const std::string n = std::to_string(i);
        s += "  %a" + n + " = load.gcref %p, " + std::to_string(i * 8) + "\n";
        args += "%a" + n + ", ";
        params += "%b" + n + ": gcref, ";
        stores += "  store.gcref %p, " + std::to_string((8 - i) * 8) + ", %b" + n + "\n";
    }
    s += "  %s8 = iconst.i64 0\n";
    for (int k = 9; k < 21; ++k) {
        const std::string n = std::to_string(k);
        s += "  %v" + n + " = load.i64 %p, " + std::to_string(k * 8) + "\n";
        s += "  %s" + n + " = add.i64 %s" + std::to_string(k - 1) + ", %v" + n + "\n";
    }
    s += "  br join(" + args + "%s20)\njoin(" + params + "%sum: i64):\n" + stores + "  ret %sum\n}\n";
    return s;
}

} // namespace

TEST_CASE("Linear Scan - a value read only by block-argument copies gives its register up") {
    const std::string text = gather_text();
    Allocated a = allocate(text.c_str(), "gather");
    const LirBlock* entry = a.lir->blocks.front().get();
    size_t loads = 0;
    for (const auto& inst : entry->instructions) {
        // What follows the copy is the edge's own moves into the join.
        if (inst->opcode == LirOpcode::ParallelCopy) break;
        // Nothing before the copy reads a slot: %p stays in its register
        // through every load, where evicting it reloaded it at each.
        CHECK_FALSE(reads_slot(*inst));
        for (const auto& u : inst->uses) {
            if (u.is_mem()) ++loads;
        }
    }
    CHECK(loads >= 21u);
    CHECK(a.lir->frame.num_spill_slots > 0);

    JitExecutionEngine jit;
    auto mod = parse_module(text.c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(uint64_t*)>("gather");
    REQUIRE(fn != nullptr);
    uint64_t words[21];
    int64_t want = 0;
    for (int i = 0; i < 21; ++i) {
        words[i] = 0x1000u + 16u * static_cast<uint64_t>(i);
        if (i >= 9) want += static_cast<int64_t>(words[i]);
    }
    CHECK_EQ(fn(words), want);
    for (int i = 0; i < 9; ++i) CHECK_EQ(words[8 - i], 0x1000u + 16u * static_cast<uint64_t>(i));
}

namespace {

// Laid out as written: the cold arm last, jumping back to the join, which is
// branched to from after itself and so looks like a loop's head.
const char* kColdArm = R"(module @m

func @pick(%x: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %c = slt.i64 %x, %z
  br_if %c, cold, join(%x)
join(%v: i64):
  %one = iconst.i64 1
  %r = add.i64 %v, %one
  ret %r
cold:
  %n = sub.i64 %z, %x
  br join(%n)
}
)";

} // namespace

TEST_CASE("Branches - a loop-free function's join is not padded as a loop head") {
    Allocated a = allocate(kColdArm, "pick", /*rpo=*/false);
    LirFunction& fn = *a.lir;
    CHECK(fn.loop_depths_computed);
    size_t joins = 0;
    const BranchThreading threading(fn);
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        for (const LirBlock* s : fn.blocks[i]->successors) {
            for (size_t h = 0; h <= i; ++h) {
                if (fn.blocks[h].get() != s) continue;
                ++joins;
                CHECK_FALSE(threading.loop_head(h));
            }
        }
    }
    CHECK(joins >= 1u);  // the cold arm, and any edge block placed after the join

    // Without the depths, a block branched to from after itself is still
    // taken for the head the layout promises.
    fn.loop_depths_computed = false;
    const BranchThreading unanalysed(fn);
    bool head = false;
    for (size_t h = 0; h < fn.blocks.size(); ++h) head = head || unanalysed.loop_head(h);
    CHECK(head);

    JitExecutionEngine jit;
    auto mod = parse_module(kColdArm);
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto f = jit.get_function_ptr<int64_t (*)(int64_t)>("pick");
    REQUIRE(f != nullptr);
    CHECK_EQ(f(5), 6);
    CHECK_EQ(f(-5), 6);
}

namespace {

// Fourteen i64s loaded off %p and all read twice, xor-ed in order and then
// added in reverse: more live values than registers, so the first reads
// find some of them in their slots, each as the source of an xor or an add,
// the operand x64 encodes as r/m.
std::string fold_text() {
    std::string s = "module @m\n\nfunc @fold(%p: ptr) -> i64 {\nentry:\n";
    for (int i = 0; i < 14; ++i) {
        s += "  %v" + std::to_string(i) + " = load.i64 %p, " + std::to_string(i * 8) + "\n";
    }
    s += "  %s0 = iconst.i64 0\n";
    for (int i = 0; i < 14; ++i) {
        s += "  %s" + std::to_string(i + 1) + " = xor.i64 %s" + std::to_string(i) + ", %v" + std::to_string(i) + "\n";
    }
    for (int i = 0; i < 14; ++i) {
        s += "  %t" + std::to_string(i + 1) + " = add.i64 %" + (i ? "t" + std::to_string(i) : std::string("s14")) +
             ", %v" + std::to_string(13 - i) + "\n";
    }
    s += "  ret %t14\n}\n";
    return s;
}

} // namespace

TEST_CASE("Linear Scan - an x64 ALU source in its spill slot is read in place") {
    const std::string text = fold_text();
    Allocated a = allocate(text.c_str(), "fold");
    size_t in_place = 0;
    for (const auto& b : a.lir->blocks) {
        for (const auto& inst : b->instructions) {
            const bool alu = inst->opcode == LirOpcode::Add || inst->opcode == LirOpcode::Xor;
            if (!alu || inst->uses.empty()) continue;
            // The general path loaded the slot into R10 first; the slot
            // read by the instruction itself is that load.
            if (inst->uses.back().is_spill_slot()) ++in_place;
            // Never two memory operands: the def and other uses are registers.
            for (size_t i = 0; i + 1 < inst->uses.size(); ++i) CHECK_FALSE(inst->uses[i].is_spill_slot());
            for (const auto& d : inst->defs) CHECK_FALSE(d.is_spill_slot());
        }
    }
    CHECK(in_place >= 4u);
    // Nor did anything else take a slot through the scratch.
    for (const auto& b : a.lir->blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->opcode != LirOpcode::Mov || inst->defs.empty() || !inst->defs[0].is_preg()) continue;
            if (inst->uses.size() == 1 && inst->uses[0].is_spill_slot()) {
                CHECK(inst->defs[0].preg_val.as_gpr() != GPR::R10);
            }
        }
    }

    JitExecutionEngine jit;
    auto mod = parse_module(text.c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(uint64_t*)>("fold");
    REQUIRE(fn != nullptr);
    uint64_t words[14];
    uint64_t x = 0;
    for (int i = 0; i < 14; ++i) {
        words[i] = 0x9e3779b97f4a7c15ull * static_cast<uint64_t>(i + 1);
        x ^= words[i];
    }
    for (int i = 0; i < 14; ++i) x += words[13 - i];
    CHECK_EQ(static_cast<uint64_t>(fn(words)), x);
}
