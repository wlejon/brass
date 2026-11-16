#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <string>

// The allocator's block frequencies (BlockSpan::level): a cold block runs
// 8^2 times less often than a hot one of its loop depth, so the moves a
// shortage of registers costs go where the shortage is, in the cold arm, and
// not on the loop's hot path. And the rewrite's third and fourth scratch
// registers (R15, R14; XMM13, XMM12) hold values: where an instruction takes
// one a value lives in, the value is saved around that one instruction.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

struct Allocated {
    std::unique_ptr<Module> mod;
    std::unique_ptr<LirFunction> lir;
};

Allocated allocate(const std::string& text, const char* name) {
    Allocated out;
    out.mod = parse_module(text.c_str());
    REQUIRE(out.mod != nullptr);
    Function* fn = out.mod->get_function(name);
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    out.lir = isel.lower(*fn);
    REQUIRE(out.lir != nullptr);
    out.lir->sort_blocks_rpo();
    LivenessAnalysis liveness(*out.lir);
    liveness.run();
    LinearScanAllocator regalloc(*out.lir, liveness, CallingConvention::sysv64());
    regalloc.allocate();
    return out;
}

// Stack accesses (a spill slot read or written, or a memory operand on the
// frame) in the function's hot blocks inside its loop.
size_t hot_loop_stack_accesses(const LirFunction& fn) {
    size_t n = 0;
    for (const auto& b : fn.blocks) {
        if (b->cold || b->loop_depth == 0) continue;
        for (const auto& inst : b->instructions) {
            for (const auto& d : inst->defs) n += d.is_spill_slot();
            for (const auto& u : inst->uses) n += u.is_spill_slot();
        }
    }
    return n;
}

// A loop whose hot path reads %k, %n, %i and %acc every iteration, and whose
// cold arm (taken when the loaded word equals %k) holds fourteen loads live
// at once: more values than registers, but only in the arm.
constexpr int kArmLoads = 14;
constexpr int kArmReads = 4;

std::string cold_arm_loop_text() {
    std::string s = R"(module @m

func @f(%p: ptr, %n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %k = load.i64 %p, 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %x = load.i64 %p, 8
  %c = eq.i64 %x, %k
  br_if %c, arm_cold, cont(%acc)
arm_cold:
)";
    for (int j = 0; j < kArmLoads; ++j) {
        s += "  %a" + std::to_string(j) + " = load.i64 %p, " + std::to_string(16 + 8 * j) + "\n";
    }
    // Each read kArmReads times, by alternate passes of xor in order and add
    // in reverse, so no load folds into its reader and every one is worth
    // more to the arm than a loop value is to one trip through it.
    std::string last = "%acc";
    int t = 0;
    for (int r = 0; r < kArmReads; ++r) {
        for (int i = 0; i < kArmLoads; ++i) {
            const int j = r % 2 ? kArmLoads - 1 - i : i;
            const std::string next = "%t" + std::to_string(t++);
            s += "  " + next + (r % 2 ? " = add.i64 " : " = xor.i64 ") + last + ", %a" + std::to_string(j) + "\n";
            last = next;
        }
    }
    s += "  br cont(" + last + ")\n";
    s += R"(cont(%v: i64):
  %acc2 = add.i64 %v, %k
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  %d = slt.i64 %i2, %n
  br_if %d, loop(%i2, %acc2), done
done:
  ret %acc2
}
)";
    return s;
}

} // namespace

TEST_CASE("Linear Scan levels - a cold arm's shortage of registers is paid in the arm") {
    const std::string text = cold_arm_loop_text();
    Allocated a = allocate(text, "f");
    bool arm_cold = false;
    for (const auto& b : a.lir->blocks) arm_cold = arm_cold || (b->cold && b->loop_depth > 0);
    REQUIRE(arm_cold);
    CHECK(a.lir->frame.num_spill_slots > 0);
    CHECK_EQ(hot_loop_stack_accesses(*a.lir), 0u);

    JitExecutionEngine jit;
    auto mod = parse_module(text.c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto f = jit.get_function_ptr<int64_t (*)(int64_t*, int64_t)>("f");
    REQUIRE(f != nullptr);
    int64_t words[2 + kArmLoads];
    for (int j = 0; j < 2 + kArmLoads; ++j) words[j] = 3 + 5 * j;
    auto expect = [&](int64_t n) {
        int64_t acc = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (words[1] == words[0]) {
                for (int r = 0; r < kArmReads; ++r) {
                    for (int i = 0; i < kArmLoads; ++i) {
                        const int j = r % 2 ? kArmLoads - 1 - i : i;
                        acc = r % 2 ? acc + words[2 + j] : acc ^ words[2 + j];
                    }
                }
            }
            acc += words[0];
        }
        return acc;
    };
    CHECK_EQ(f(words, 10), expect(10));  // the hot path only
    words[1] = words[0];
    CHECK_EQ(f(words, 10), expect(10));  // the cold arm every iteration
}

namespace {

// Twenty bases (each %p, read out of the table %t), twenty offsets and
// twenty values, all live to the end; each value stored at [base + offset +
// 512], an address of a base and an index, so the store reads three
// registers. With sixty values live, many of those stores find all three in
// their slots, and their rewrite takes a third scratch (R15) while R15 holds
// a value. Then everything is read again and summed, so a scratch use that
// lost a value shows in the sum.
constexpr int kPairs = 20;

std::string scatter_text() {
    std::string s = "module @m\n\nfunc @scatter(%p: ptr, %t: ptr) -> i64 {\nentry:\n";
    for (int j = 0; j < kPairs; ++j) {
        const std::string n = std::to_string(j);
        s += "  %b" + n + " = load.ptr %t, " + std::to_string(8 * j) + "\n";
        s += "  %o" + n + " = load.i64 %p, " + std::to_string(16 * j) + "\n";
        s += "  %v" + n + " = load.i64 %p, " + std::to_string(16 * j + 8) + "\n";
    }
    for (int j = 0; j < kPairs; ++j) {
        const std::string n = std::to_string(j);
        s += "  %q" + n + " = add.ptr %b" + n + ", %o" + n + "\n";
        s += "  store.i64 %q" + n + ", 512, %v" + n + "\n";
    }
    std::string last = "%o0";
    for (int j = 0; j < kPairs; ++j) {
        const std::string n = std::to_string(j);
        if (j) {
            s += "  %x" + n + " = xor.i64 " + last + ", %o" + n + "\n";
            last = "%x" + n;
        }
        s += "  %y" + n + " = add.i64 " + last + ", %v" + n + "\n";
        s += "  %w" + n + " = load.i64 %b" + n + ", 8\n";
        s += "  %z" + n + " = xor.i64 %y" + n + ", %w" + n + "\n";
        last = "%z" + n;
    }
    s += "  ret " + last + "\n}\n";
    return s;
}

} // namespace

TEST_CASE("Linear Scan - the third scratch register holds a value, saved around the instruction taking it") {
    const std::string text = scatter_text();
    Allocated a = allocate(text, "scatter");
    // R15 holds a value somewhere, and some instruction's rewrite saves it to
    // a slot (and restores it) to take it as a scratch.
    auto r15 = [](const LirOperand& op) { return op.is_preg() && op.preg_val == PReg::gpr(GPR::R15); };
    bool held = false;
    size_t saves = 0;
    size_t restores = 0;
    for (const auto& b : a.lir->blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->opcode != LirOpcode::Mov || inst->defs.size() != 1 || inst->uses.size() != 1) continue;
            if (inst->defs[0].is_spill_slot() && r15(inst->uses[0])) ++saves;
            if (r15(inst->defs[0]) && inst->uses[0].is_spill_slot()) ++restores;
            held = held || (r15(inst->defs[0]) && inst->uses[0].is_mem());
        }
    }
    CHECK(held);
    CHECK(saves >= 1u);
    CHECK(restores >= 1u);

    JitExecutionEngine jit;
    auto mod = parse_module(text.c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto f = jit.get_function_ptr<int64_t (*)(uint64_t*, uint64_t**)>("scatter");
    REQUIRE(f != nullptr);
    // Offsets point the stores at words 64 + 2j + 1.. past the pairs; the
    // values are distinct.
    uint64_t words[64 + 2 * kPairs + 2] = {};
    for (int j = 0; j < kPairs; ++j) {
        words[2 * j] = 16u * static_cast<uint64_t>(j);
        words[2 * j + 1] = 0x10001u * static_cast<uint64_t>(j + 1);
    }
    uint64_t* bases[kPairs];
    for (uint64_t*& b : bases) b = words;
    int64_t want = static_cast<int64_t>(words[0]);
    for (int j = 0; j < kPairs; ++j) {
        if (j) want ^= static_cast<int64_t>(words[2 * j]);
        want += static_cast<int64_t>(words[2 * j + 1]);
        want ^= static_cast<int64_t>(words[1]);
    }
    CHECK_EQ(f(words, bases), want);
    for (int j = 0; j < kPairs; ++j) CHECK_EQ(words[64 + 2 * j], 0x10001u * static_cast<uint64_t>(j + 1));
}
