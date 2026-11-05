#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/branch_threading.hpp>
#include <bit>
#include <string>

// What the allocator makes of a value in its spill slot: a copy to or from the
// slot is the load or the store itself, a value only on its way into a block
// parameter is the one to give its register up, an x64 ALU source left in its
// slot is read from it in place (and such a read of a value the loop does not
// define takes no register), and a loop-free function's joins are not padded
// as loop heads.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

struct Allocated {
    std::unique_ptr<Module> mod;
    std::unique_ptr<LirFunction> lir;
};

// `name` lowered for x64 SysV and allocated, its blocks in the order written;
// `prepare` sees the LIR before allocation.
Allocated allocate(const char* text, const char* name, bool rpo = true,
                   void (*prepare)(LirFunction&) = nullptr) {
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
    if (prepare) prepare(*out.lir);
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

// A tagged value cannot be split. Live across a call it takes a
// callee-saved register while one is free; with six of them across the
// call (SysV has five), at least one lives in its slot for good: the
// bitcast that makes it is a copy into the slot, the one after the call a
// copy out of it.
const char* kAcrossCall = R"(module @m

func @twice(%a: i64) -> i64 {
entry:
  %r = add.i64 %a, %a
  ret %r
}

func @bits(%x: i64) -> i64 {
entry:
  %one = iconst.i64 1
  %x1 = add.i64 %x, %one
  %x2 = add.i64 %x1, %one
  %x3 = add.i64 %x2, %one
  %x4 = add.i64 %x3, %one
  %x5 = add.i64 %x4, %one
  %t0 = bitcast.tagged.i64 %x
  %t1 = bitcast.tagged.i64 %x1
  %t2 = bitcast.tagged.i64 %x2
  %t3 = bitcast.tagged.i64 %x3
  %t4 = bitcast.tagged.i64 %x4
  %t5 = bitcast.tagged.i64 %x5
  %c = call.i64 @twice(%x)
  %u0 = bitcast.i64.tagged %t0
  %u1 = bitcast.i64.tagged %t1
  %u2 = bitcast.i64.tagged %t2
  %u3 = bitcast.i64.tagged %t3
  %u4 = bitcast.i64.tagged %t4
  %u5 = bitcast.i64.tagged %t5
  %r0 = xor.i64 %u0, %c
  %r1 = xor.i64 %r0, %u1
  %r2 = xor.i64 %r1, %u2
  %r3 = xor.i64 %r2, %u3
  %r4 = xor.i64 %r3, %u4
  %r = xor.i64 %r4, %u5
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
    for (int64_t x : {int64_t{3}, int64_t{-8}, int64_t{1} << 40}) {
        CHECK_EQ(fn(x), x ^ (x + x) ^ (x + 1) ^ (x + 2) ^ (x + 3) ^ (x + 4) ^ (x + 5));
    }
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

namespace {

// A loop reading %k (defined before it) and %acc (carried round it) as xor
// sources, and %i as a multiply's: only %k's read is one the allocator may
// leave to the slot for free. %acc's slot would be stored every iteration,
// and a multiply's operand is never read from memory (linear_scan.cpp
// mark_folded_uses). Many values live through the body, so some of them do
// wait in slots.
std::string folded_loop_text() {
    // %k is a constant too wide for an immediate, made once before the loop,
    // as a tag mask is (bronze's 0xfff1 << 48).
    std::string s = "module @m\n\nfunc @lx(%q: ptr, %n: i64) -> i64 {\nentry:\n";
    s += "  %k = iconst.i64 -4222124650659840\n  %z = iconst.i64 0\n  br head(%z, %z)\n";
    s += "head(%i: i64, %acc: i64):\n  %c = slt.i64 %i, %n\n  br_if %c, body, done\n";
    s += "body:\n";
    for (int j = 0; j < 14; ++j) {
        s += "  %w" + std::to_string(j) + " = load.i64 %q, " + std::to_string(j * 8) + "\n";
    }
    // %s is computed, not loaded, so no operand below is a folded load.
    // %s dies at the xor, so it is the operand the xor writes and %k the one
    // it reads (the isel copies an operand still live after into the result).
    s += "  %s = add.i64 %w0, %i\n  %x = xor.i64 %s, %k\n  %y = xor.i64 %x, %acc\n  %m0 = mul.i64 %y, %i\n";
    for (int j = 1; j < 13; ++j) {
        s += "  %m" + std::to_string(j) + " = mul.i64 %m" + std::to_string(j - 1) + ", %w" + std::to_string(j) + "\n";
    }
    s += "  %t = add.i64 %m12, %w13\n  %one = iconst.i64 1\n  %i2 = add.i64 %i, %one\n  br head(%i2, %t)\n";
    s += "done:\n  ret %acc\n}\n";
    return s;
}

int64_t folded_loop_ref(const uint64_t* q, int64_t n) {
    const uint64_t k = 0xfff1000000000000ull;
    uint64_t acc = 0;
    for (int64_t i = 0; i < n; ++i) {
        const uint64_t s = q[0] + static_cast<uint64_t>(i);
        uint64_t m = ((s ^ k) ^ acc) * static_cast<uint64_t>(i);
        for (int j = 1; j < 13; ++j) m *= q[j];
        acc = m + q[13];
    }
    return static_cast<int64_t>(acc);
}

// Per Xor / Imul of `fn` before allocation: its id and its last use's vreg.
struct LastUse {
    uint32_t id;
    VReg v;
    LirOpcode op;
};
std::vector<LastUse> g_last_uses;
void record_last_uses(LirFunction& fn) {
    g_last_uses.clear();
    for (const auto& b : fn.blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->opcode != LirOpcode::Xor && inst->opcode != LirOpcode::Imul) continue;
            if (inst->uses.empty() || !inst->uses.back().is_vreg()) continue;
            g_last_uses.push_back(LastUse{inst->id, inst->uses.back().vreg_val, inst->opcode});
        }
    }
}

} // namespace

TEST_CASE("Linear Scan - an x64 ALU source read from its slot in place takes no register") {
    auto mod = parse_module(folded_loop_text().c_str());
    REQUIRE(mod != nullptr);
    Function* f = mod->get_function("lx");
    REQUIRE(f != nullptr);
    f->rebuild_cfg_predecessors();
    for (bool x64 : {true, false}) {
        std::unique_ptr<LirFunction> lir;
        CallingConvention cc = x64 ? CallingConvention::sysv64() : CallingConvention::aapcs64();
        if (x64) {
            X64ISel isel(Target::x64_linux(), cc);
            lir = isel.lower(*f);
        } else {
            brass::aarch64::AArch64ISel isel(Target::aarch64_linux(), cc);
            lir = isel.lower(*f);
        }
        REQUIRE(lir != nullptr);
        lir->sort_blocks_rpo();
        LivenessAnalysis liveness(*lir);
        liveness.run();
        record_last_uses(*lir);
        LinearScanAllocator regalloc(*lir, liveness, cc);
        regalloc.allocate();

        // Per last use: whether it was left needing no register; and whether
        // its value is defined only outside the loop it is read in.
        size_t free_reads = 0, xor_reads = 0, mul_reads = 0;
        for (const LastUse& lu : g_last_uses) {
            const LiveInterval* iv = liveness.get_interval(lu.v);
            REQUIRE(iv != nullptr);
            const uint32_t depth = liveness.get_loop_depth_at(lu.id);
            bool invariant = true;
            bool needs_reg = true;
            for (const UsePosition& u : iv->use_positions) {
                if (u.is_def && liveness.get_loop_depth_at(u.inst_id) >= depth) invariant = false;
                if (u.inst_id == lu.id && !u.is_def) needs_reg = u.requires_reg;
            }
            if (lu.op == LirOpcode::Xor) ++xor_reads;
            if (lu.op == LirOpcode::Imul) ++mul_reads;
            if (!needs_reg) {
                ++free_reads;
                // Only an x64 xor's source, of a value the loop does not define.
                CHECK(x64);
                CHECK(lu.op == LirOpcode::Xor);
                CHECK(invariant);
            } else if (x64 && lu.op == LirOpcode::Xor && invariant && depth > 0) {
                CHECK(!"an x64 xor's loop-invariant source still wants a register");
            }
        }
        if (x64) {
            CHECK(xor_reads >= 2u);
            CHECK(mul_reads >= 1u);
        }
        CHECK_EQ(free_reads, x64 ? size_t(1) : size_t(0));  // %k's, in the body
    }

    JitExecutionEngine jit;
    auto run = parse_module(folded_loop_text().c_str());
    REQUIRE(run != nullptr);
    REQUIRE(jit.compile_and_load(*run));
    auto fn = jit.get_function_ptr<int64_t (*)(uint64_t*, int64_t)>("lx");
    REQUIRE(fn != nullptr);
    uint64_t q[14];
    for (int i = 0; i < 14; ++i) q[i] = 0x9e3779b97f4a7c15ull * static_cast<uint64_t>(i + 3) | 1;
    for (int64_t n : {int64_t{0}, int64_t{1}, int64_t{7}, int64_t{100}}) {
        CHECK_EQ(fn(q, n), folded_loop_ref(q, n));
    }
}

namespace {

// %t lives across the first call, so in a callee-saved register, and dies
// before the second: the second call finds a dead register holding a stale
// word, which a conservative scan from inside it would read as a root.
const char* kStaleAtCall = R"(module @m

func @g(%a: i64) -> i64 {
entry:
  %r = add.i64 %a, %a
  ret %r
}

func @stale(%a: i64) -> i64 {
entry:
  %t = mul.i64 %a, %a
  %c1 = call.i64 @g(%a)
  %u = add.i64 %t, %c1
  %c2 = call.i64 @g(%u)
  ret %c2
}
)";

// Zeroing moves into registers or slots ahead of the function's last call.
size_t clears_before_last_call(const LirFunction& fn) {
    size_t clears = 0, since_call = 0;
    for (const auto& b : fn.blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->is_call()) {
                clears = since_call;
                since_call = 0;
                continue;
            }
            const bool zero = inst->opcode == LirOpcode::Mov && inst->uses.size() == 1 && inst->uses[0].is_imm_int() &&
                              inst->uses[0].imm_int == 0;
            if (zero) ++since_call;
        }
    }
    return clears;
}

void mark_last_call_no_gc(LirFunction& fn) {
    LirInst* last = nullptr;
    for (auto& b : fn.blocks) {
        for (auto& inst : b->instructions) {
            if (inst->is_call()) last = inst.get();
        }
    }
    REQUIRE(last != nullptr);
    last->no_gc_point = true;
}

} // namespace

namespace {

// A loop that reads fourteen words a turn and compares three pairs of six
// values loaded before it: more live values than registers, so the
// allocator leaves some invariants in their slots, and some compare finds
// both of its sides there.
std::string cmp_loop_text() {
    std::string s = "module @m\n\nfunc @cmps(%q: ptr, %n: i64) -> i64 {\nentry:\n";
    for (int k = 0; k < 6; ++k) {
        s += "  %a" + std::to_string(k) + " = load.i64 %q, " + std::to_string((14 + k) * 8) + "\n";
    }
    s += "  %z = iconst.i64 0\n  br head(%z, %z)\n";
    s += "head(%i: i64, %acc: i64):\n  %c = slt.i64 %i, %n\n  br_if %c, body, done\n";
    s += "body:\n";
    for (int j = 0; j < 14; ++j) {
        s += "  %w" + std::to_string(j) + " = load.i64 %q, " + std::to_string(j * 8) + "\n";
    }
    for (int k = 0; k < 3; ++k) {
        const std::string n = std::to_string(k);
        s += "  %c" + n + " = slt.i64 %a" + std::to_string(2 * k) + ", %a" + std::to_string(2 * k + 1) + "\n";
        s += "  %m" + n + " = select.i64 %c" + n + ", %w" + n + ", %w" + std::to_string(13 - k) + "\n";
    }
    s += "  %x = add.i64 %w0, %i\n  %p0 = mul.i64 %x, %m0\n  %p1 = mul.i64 %p0, %m1\n  %p2 = mul.i64 %p1, %m2\n";
    for (int j = 1; j < 13; ++j) {
        s += "  %p" + std::to_string(j + 2) + " = mul.i64 %p" + std::to_string(j + 1) + ", %w" + std::to_string(j) + "\n";
    }
    s += "  %t = add.i64 %p14, %w13\n  %t2 = xor.i64 %t, %acc\n  %one = iconst.i64 1\n";
    s += "  %i2 = add.i64 %i, %one\n  br head(%i2, %t2)\n";
    s += "done:\n  ret %acc\n}\n";
    return s;
}

uint64_t cmp_loop_ref(const int64_t* q, int64_t n) {
    uint64_t acc = 0;
    for (int64_t i = 0; i < n; ++i) {
        uint64_t m[3];
        for (int k = 0; k < 3; ++k) {
            m[k] = static_cast<uint64_t>(q[14 + 2 * k] < q[15 + 2 * k] ? q[k] : q[13 - k]);
        }
        uint64_t p = (static_cast<uint64_t>(q[0]) + static_cast<uint64_t>(i)) * m[0] * m[1] * m[2];
        for (int j = 1; j < 13; ++j) p *= static_cast<uint64_t>(q[j]);
        acc = (p + static_cast<uint64_t>(q[13])) ^ acc;
    }
    return acc;
}

} // namespace

// A compare defines nothing, so a spilled left side takes a use scratch and
// the right side is still read from its slot in place: `mov r10, slot; cmp
// r10, slot`, not two reloads and a register compare.
TEST_CASE("Linear Scan - an x64 compare of two spilled values reads the right one in place") {
    const std::string text = cmp_loop_text();
    Allocated a = allocate(text.c_str(), "cmps");
    size_t both = 0;
    for (const auto& b : a.lir->blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->opcode != LirOpcode::Cmp || inst->uses.size() != 2) continue;
            const auto& l = inst->uses[0];
            const auto& r = inst->uses[1];
            CHECK_FALSE(l.is_spill_slot());
            if (!r.is_spill_slot() || !l.is_preg()) continue;
            const GPR g = l.preg_val.as_gpr();
            if (g == GPR::R10 || g == GPR::R11 || g == GPR::R15) ++both;
        }
    }
    CHECK(both >= 1u);

    JitExecutionEngine jit;
    auto mod = parse_module(text.c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(const int64_t*, int64_t)>("cmps");
    REQUIRE(fn != nullptr);
    int64_t q[20];
    uint64_t h = 0x243f6a8885a308d3ull;
    for (int i = 0; i < 20; ++i) {
        h = h * 6364136223846793005ull + 1442695040888963407ull;
        q[i] = static_cast<int64_t>(h);
    }
    for (int64_t n : {0, 1, 7}) CHECK_EQ(static_cast<uint64_t>(fn(q, n)), cmp_loop_ref(q, n));
}

TEST_CASE("Linear Scan - a call that reaches no GC point clears no dead register") {
    Allocated plain = allocate(kStaleAtCall, "stale");
    CHECK(clears_before_last_call(*plain.lir) >= 1u);
    Allocated no_gc = allocate(kStaleAtCall, "stale", true, mark_last_call_no_gc);
    CHECK_EQ(clears_before_last_call(*no_gc.lir), size_t(0));
}

namespace {

// kStaleAtCall in a module whose host reads frames only through stack maps
// (Module::precise_stack_roots): the dead register at the second call is no
// root to anything, so nothing zeroes it.
const char* kStaleAtCallPrecise = R"(module @m

attributes precise_stack_roots

func @g(%a: i64) -> i64 {
entry:
  %r = add.i64 %a, %a
  ret %r
}

func @stale(%a: i64) -> i64 {
entry:
  %t = mul.i64 %a, %a
  %c1 = call.i64 @g(%a)
  %u = add.i64 %t, %c1
  %c2 = call.i64 @g(%u)
  ret %c2
}
)";

size_t aarch64_clears_before_last_call(const char* text) {
    auto mod = parse_module(text);
    REQUIRE(mod != nullptr);
    Function* fn = mod->get_function("stale");
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    brass::aarch64::AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    lir->sort_blocks_rpo();
    LivenessAnalysis liveness(*lir);
    liveness.run();
    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::aapcs64());
    regalloc.allocate();
    return clears_before_last_call(*lir);
}

} // namespace

TEST_CASE("Linear Scan - a precise-roots module clears no dead register at a call") {
    Allocated precise = allocate(kStaleAtCallPrecise, "stale");
    CHECK(!precise.lir->clear_dead_roots_at_calls);
    CHECK_EQ(clears_before_last_call(*precise.lir), size_t(0));
    Allocated conservative = allocate(kStaleAtCall, "stale");
    CHECK(conservative.lir->clear_dead_roots_at_calls);
    CHECK(clears_before_last_call(*conservative.lir) >= 1u);
    CHECK(aarch64_clears_before_last_call(kStaleAtCall) >= 1u);
    CHECK_EQ(aarch64_clears_before_last_call(kStaleAtCallPrecise), size_t(0));
}

TEST_CASE("MIR - precise_stack_roots prints and parses back") {
    auto mod = parse_module(kStaleAtCallPrecise);
    REQUIRE(mod != nullptr);
    CHECK(mod->precise_stack_roots());
    const std::string text = brass::to_string(*mod);
    CHECK(text.find("precise_stack_roots") != std::string::npos);
    auto again = parse_module(text);
    REQUIRE(again != nullptr);
    CHECK(again->precise_stack_roots());
    auto plain = parse_module(kStaleAtCall);
    REQUIRE(plain != nullptr);
    CHECK(!plain->precise_stack_roots());
    CHECK(brass::to_string(*plain).find("precise_stack_roots") == std::string::npos);
}
