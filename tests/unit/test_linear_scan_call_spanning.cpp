#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <string>

// Values live across calls: a value read all over a function keeps the
// callee-saved register it was given over values that need one briefly, R14
// and R15 hold values, a tagged value's bits are its own register, and a
// value live across a call made only on a cold path leaves its register on
// that path alone.

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
    out.mod = parse_module(text);
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

} // namespace

namespace {

// %t is read by sixty-four loads spread thinly over the function, across
// three calls; five values made just before the first call and read just
// after the second need callee-saved registers too, more of them than there
// are. Weighed per
// unit of its long life, %t looked cheaper to leave in its slot than any of
// them, lost its register to one, and was reloaded at each of its loads.
constexpr int kShort = 5;

std::string spread_text() {
    std::string s = "module @m\n\nfunc @mix(%a: i64) -> i64 {\nentry:\n  %k = iconst.i64 3\n"
                    "  %r = mul.i64 %a, %k\n  ret %r\n}\n\nfunc @spread(%base: ptr, %q: ptr) -> i64 {\nentry:\n"
                    "  %t = load.ptr %base, 0\n  %acc0 = iconst.i64 0\n";
    int acc = 0;
    auto add_loads = [&](int from, int to) {
        for (int k = from; k < to; ++k) {
            const std::string n = std::to_string(k);
            s += "  %x" + n + " = load.i64 %t, " + std::to_string(k * 8) + "\n";
            s += "  %y" + n + " = load.i64 %q, " + std::to_string(k * 8) + "\n";
            s += "  %z" + n + " = xor.i64 %x" + n + ", %y" + n + "\n";
            // Filler between %t's reads: its life is long and sparse.
            s += "  %e" + n + " = load.i64 %q, " + std::to_string(k * 8) + "\n";
            s += "  %g" + n + " = add.i64 %z" + n + ", %e" + n + "\n";
            s += "  %h" + n + " = sub.i64 %g" + n + ", %e" + n + "\n";
            s += "  %acc" + std::to_string(acc + 1) + " = add.i64 %acc" + std::to_string(acc) + ", %h" + n + "\n";
            ++acc;
        }
    };
    add_loads(0, 30);
    for (int j = 0; j < kShort; ++j) s += "  %s" + std::to_string(j) + " = load.i64 %q, " + std::to_string(1000 + j * 8) + "\n";
    s += "  %c0 = call.i64 @mix(%acc" + std::to_string(acc) + ")\n";
    s += "  %c1 = call.i64 @mix(%c0)\n";
    std::string sum = "%c1";
    for (int j = 0; j < kShort; ++j) {
        const std::string n = std::to_string(j);
        // %s first: a last use would be read from the slot in place.
        s += "  %u" + n + " = add.i64 %s" + n + ", " + sum + "\n";
        s += "  %w" + n + " = xor.i64 %s" + n + ", %u" + n + "\n";
        s += "  %v" + n + " = sub.i64 %s" + n + ", %w" + n + "\n";
        sum = "%v" + n;
    }
    s += "  %c2 = call.i64 @mix(" + sum + ")\n";
    add_loads(30, 64);
    s += "  %r = add.i64 %acc" + std::to_string(acc) + ", %c2\n  ret %r\n}\n";
    return s;
}

} // namespace

TEST_CASE("Linear Scan - a value read all over a function keeps its callee-saved register") {
    const std::string text = spread_text();
    Allocated a = allocate(text, "spread");
    // %t's loads: every load.i64 off a register other than %q's and %base's
    // reads it in place; none reloads it from a slot first.
    size_t reloads = 0;
    size_t loads_off_t = 0;
    for (const auto& b : a.lir->blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->opcode == LirOpcode::Mov && inst->defs.size() == 1 && inst->defs[0].is_preg() &&
                inst->uses.size() == 1 && inst->uses[0].is_spill_slot() &&
                (inst->defs[0].preg_val.as_gpr() == GPR::R10 || inst->defs[0].preg_val.as_gpr() == GPR::R11)) {
                ++reloads;
            }
            if (inst->opcode == LirOpcode::Mov && inst->uses.size() == 1 && inst->uses[0].is_mem() &&
                inst->uses[0].mem_val.base_preg.is_valid() && inst->uses[0].mem_val.disp < 1000) {
                ++loads_off_t;
            }
        }
    }
    // A base reloaded into a scratch at every load would be 64 reloads.
    CHECK(reloads < 16u);
    CHECK(loads_off_t >= 64u);

    JitExecutionEngine jit;
    auto mod = parse_module(text);
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(const int64_t* const*, const int64_t*)>("spread");
    REQUIRE(fn != nullptr);
    int64_t tv[64];
    int64_t qv[1000 / 8 + kShort];
    for (int k = 0; k < 64; ++k) tv[k] = k * 7 + 1;
    for (int k = 0; k < 1000 / 8 + kShort; ++k) qv[k] = k * 3 + 5;
    const int64_t* base[1] = {tv};
    auto mix = [](int64_t v) { return v * 3; };
    int64_t acc = 0;
    for (int k = 0; k < 30; ++k) acc += tv[k] ^ qv[k];
    int64_t c0 = 0;
    {
        int64_t a20 = 0;
        for (int k = 0; k < 30; ++k) a20 += tv[k] ^ qv[k];
        c0 = mix(a20);
    }
    int64_t sum = mix(c0);
    for (int j = 0; j < kShort; ++j) {
        const int64_t sj = qv[1000 / 8 + j];
        sum = sj - (sj ^ (sj + sum));
    }
    const int64_t c2 = mix(sum);
    for (int k = 30; k < 64; ++k) acc += tv[k] ^ qv[k];
    CHECK_EQ(fn(base, qv), acc + c2);
}

namespace {

// Six values loaded before a call and read after it: more values live
// across the call than RBX, R12 and R13 can hold, once the only callee-saved
// registers besides the rewrite's scratches R14 and R15.
const char* kAcross = R"(module @m

func @mix(%a: i64) -> i64 {
entry:
  %k = iconst.i64 3
  %r = mul.i64 %a, %k
  ret %r
}

func @keep(%p: ptr, %x: i64) -> i64 {
entry:
  %a = load.i64 %p, 0
  %b = load.i64 %p, 8
  %c = load.i64 %p, 16
  %d = load.i64 %p, 24
  %e = load.i64 %p, 32
  %f = load.i64 %p, 40
  %r = call.i64 @mix(%x)
  %s1 = add.i64 %a, %b
  %s2 = mul.i64 %s1, %c
  %s3 = sub.i64 %s2, %d
  %s4 = xor.i64 %s3, %e
  %s5 = mul.i64 %s4, %f
  %s6 = add.i64 %s5, %r
  %q = call.i64 @mix(%s6)
  %t1 = add.i64 %q, %a
  %t2 = add.i64 %t1, %f
  ret %t2
}
)";

} // namespace

TEST_CASE("Linear Scan - x64 R14 and R15 hold values live across calls") {
    Allocated a = allocate(kAcross, "keep");
    const uint32_t used = a.lir->frame.saved_callee_gprs;
    CHECK((used & reg_mask(GPR::R14)) != 0);
    CHECK((used & reg_mask(GPR::R15)) != 0);
    size_t named = 0;
    for (const auto& b : a.lir->blocks) {
        for (const auto& inst : b->instructions) {
            for (const auto& d : inst->defs) {
                if (d.is_preg() && (d.preg_val == PReg::gpr(GPR::R14) || d.preg_val == PReg::gpr(GPR::R15))) ++named;
            }
        }
    }
    CHECK(named >= 2u);

    JitExecutionEngine jit;
    auto mod = parse_module(kAcross);
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(const int64_t*, int64_t)>("keep");
    REQUIRE(fn != nullptr);
    const int64_t w[6] = {3, 5, 7, 11, 13, 17};
    const int64_t r = 9 * 3;
    const int64_t s6 = (((((3 + 5) * 7) - 11) ^ 13) * 17) + r;
    CHECK_EQ(fn(w, 9), s6 * 3 + 3 + 17);
}

namespace {

// A property check's shape: the receiver's bits are read for the hit path's
// address and passed to the miss path's call. A copy of them was a value of
// its own, which that call's first argument steered to RDI, so the hit path
// began with `mov rdi, <receiver>` for the miss's sake. The hit path reads
// the receiver again later, as a method body reads `this`.
const char* kBitsRead = R"(module @m

func @slow(%a: i64) -> i64 {
entry:
  %k = iconst.i64 3
  %r = mul.i64 %a, %k
  ret %r
}

func @check(%lim: i64, %t: tagged) -> i64 {
entry:
  %u = bitcast.i64.tagged %t
  %m = iconst.i64 281474976710655
  %p = and.i64 %u, %m
  %c = ult.i64 %p, %lim
  br_if %c, hit, miss
hit:
  %v = load.i64 %p, 8
  %u2 = bitcast.i64.tagged %t
  %s = sub.i64 %v, %u2
  ret %s
miss:
  %r = call.i64 @slow(%u)
  ret %r
}
)";

} // namespace

TEST_CASE("Linear Scan - a tagged value's bits are read from its own register, not a copy") {
    Allocated a = allocate(kBitsRead, "check");
    // Neither the entry nor the hit block copies a register into another:
    // the and reads the argument's register, and RDI is set up for the
    // call on the miss path only. A copy of the bits put them in a register
    // of their own in the entry block.
    for (const auto& b : a.lir->blocks) {
        if (b->name == "miss") continue;
        for (const auto& inst : b->instructions) {
            const bool reg_copy = inst->opcode == LirOpcode::Mov && inst->defs.size() == 1 && inst->uses.size() == 1 &&
                                  inst->defs[0].is_preg() && inst->uses[0].is_preg() &&
                                  inst->defs[0].preg_val != inst->uses[0].preg_val;
            CHECK(!reg_copy);
        }
    }

    JitExecutionEngine jit;
    auto mod = parse_module(kBitsRead);
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(int64_t, uint64_t)>("check");
    REQUIRE(fn != nullptr);
    // Not a heap object, and no collection runs (@slow does not allocate).
    const int64_t words[2] = {0, 77};
    const uint64_t addr = reinterpret_cast<uint64_t>(&words[0]);
    CHECK_EQ(fn(INT64_MAX, addr), static_cast<int64_t>(77 - addr));
    CHECK_EQ(fn(0, addr), static_cast<int64_t>(addr * 3));
}

namespace {

// Eight values loaded on entry and read after a join, live across a call
// made only on the cold arm: more of them than there are callee-saved
// registers, so some leave a caller-saved register for their slot around
// that call. Stored once at their definitions instead, they were stored on
// every pass of the hot path for the cold arm's sake.
const char* kColdCall = R"(module @m

func @mix(%a: i64) -> i64 {
entry:
  %k = iconst.i64 3
  %r = mul.i64 %a, %k
  ret %r
}

func @cold_call(%p: ptr, %c: i64) -> i64 {
entry:
  %a = load.i64 %p, 0
  %b = load.i64 %p, 8
  %d = load.i64 %p, 16
  %e = load.i64 %p, 24
  %f = load.i64 %p, 32
  %g = load.i64 %p, 40
  %h = load.i64 %p, 48
  %i = load.i64 %p, 56
  %z = iconst.i64 0
  %t = slt.i64 %c, %z
  br_if %t, slow, join(%z)
slow:
  %r = call.i64 @mix(%c)
  br join(%r)
join(%m: i64):
  %s1 = add.i64 %a, %b
  %s2 = xor.i64 %s1, %d
  %s3 = sub.i64 %s2, %e
  %s4 = add.i64 %s3, %f
  %s5 = xor.i64 %s4, %g
  %s6 = sub.i64 %s5, %h
  %s7 = add.i64 %s6, %i
  %s8 = add.i64 %s7, %m
  ret %s8
}
)";

} // namespace

TEST_CASE("Linear Scan - a value live across a call on a cold path only is stored on that path") {
    Allocated a;
    a.mod = parse_module(kColdCall);
    REQUIRE(a.mod != nullptr);
    Function* fn = a.mod->get_function("cold_call");
    REQUIRE(fn != nullptr);
    for (BasicBlock* bb : fn->blocks()) {
        if (bb->name() == "slow") bb->set_cold();
    }
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    a.lir = isel.lower(*fn);
    REQUIRE(a.lir != nullptr);
    a.lir->sort_blocks_rpo();
    LivenessAnalysis liveness(*a.lir);
    liveness.run();
    LinearScanAllocator regalloc(*a.lir, liveness, CallingConvention::sysv64());
    regalloc.allocate();

    size_t entry_stores = 0;
    size_t slots_written = 0;
    for (const auto& b : a.lir->blocks) {
        for (const auto& inst : b->instructions) {
            for (const auto& d : inst->defs) {
                if (!d.is_spill_slot()) continue;
                ++slots_written;
                if (b->name == "entry") ++entry_stores;
            }
        }
    }
    CHECK(slots_written > 0u);  // some value did leave its register at the call
    CHECK_EQ(entry_stores, 0u);

    JitExecutionEngine jit;
    auto mod = parse_module(kColdCall);
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto run = jit.get_function_ptr<int64_t (*)(const int64_t*, int64_t)>("cold_call");
    REQUIRE(run != nullptr);
    const int64_t w[8] = {3, 5, 7, 11, 13, 17, 19, 23};
    const int64_t base = (((((((3 + 5) ^ 7) - 11) + 13) ^ 17) - 19) + 23);
    CHECK_EQ(run(w, 4), base);
    CHECK_EQ(run(w, -4), base - 12);
}

TEST_CASE("Linear Scan - Hot-First Spill Slot Ordering Places Loop Spills in disp8 Slots and Preserves GC & Alignment") {
    LirFunction fn;
    fn.name = "hot_first_spills";
    fn.calling_conv = CallingConvention::sysv64();
    LirBlock* entry = fn.create_block_with_id(0, "entry");
    LirBlock* loop = fn.create_block_with_id(1, "loop");
    LirBlock* exit = fn.create_block_with_id(2, "exit");

    VReg ptr = fn.allocate_vreg(RegClass::GPR, 8);
    {
        auto init_ptr = std::make_unique<LirInst>(LirOpcode::Mov);
        init_ptr->add_def(LirOperand::vreg(ptr, 8));
        init_ptr->add_use(LirOperand::preg_gpr(GPR::RDI, 8), FixedConstraint::gpr(GPR::RDI));
        entry->append_inst(std::move(init_ptr));
    }

    // Load 20 non-constant entry-only GPR values (one gcref) and a 16B vector
    // value first in encounter order, all spanning calls and read once in exit.
    std::vector<VReg> outer(20);
    for (size_t i = 0; i < outer.size(); ++i) {
        outer[i] = fn.allocate_vreg(RegClass::GPR, 8, /*is_gcref=*/i == 3);
        auto ld = std::make_unique<LirInst>(LirOpcode::Mov);
        ld->add_def(LirOperand::vreg(outer[i], 8));
        ld->add_use(LirOperand::mem(ptr, static_cast<int32_t>(i * 8), 8));
        entry->append_inst(std::move(ld));
    }
    VReg vec16 = fn.allocate_vreg(RegClass::XMM, 16);
    {
        auto vld = std::make_unique<LirInst>(LirOpcode::Movups);
        vld->add_def(LirOperand::vreg(vec16, 16));
        vld->add_use(LirOperand::mem(ptr, 256, 16));
        entry->append_inst(std::move(vld));

        auto call = std::make_unique<LirInst>(LirOpcode::Call);
        call->add_use(LirOperand::symbol("ext_entry"));
        call->clobbered_gprs = fn.calling_conv.caller_saved_gpr_mask();
        call->clobbered_xmms = fn.calling_conv.caller_saved_xmm_mask();
        entry->append_inst(std::move(call));
        entry->append_inst(std::make_unique<LirInst>(LirOpcode::Safepoint));

        auto jmp = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp->add_use(LirOperand::label(1));
        entry->append_inst(std::move(jmp));
    }

    // Inside the loop (depth 1), 12 non-constant GPR values span a call so >= 7
    // must spill; hot-first ordering must place loop spills ahead of entry spills.
    std::vector<VReg> hot(12);
    for (size_t i = 0; i < hot.size(); ++i) {
        hot[i] = fn.allocate_vreg(RegClass::GPR, 8);
        auto ld = std::make_unique<LirInst>(LirOpcode::Mov);
        ld->add_def(LirOperand::vreg(hot[i], 8));
        ld->add_use(LirOperand::mem(ptr, static_cast<int32_t>(512 + i * 8), 8));
        loop->append_inst(std::move(ld));
    }
    {
        auto call = std::make_unique<LirInst>(LirOpcode::Call);
        call->add_use(LirOperand::symbol("ext_loop"));
        call->clobbered_gprs = fn.calling_conv.caller_saved_gpr_mask();
        call->clobbered_xmms = fn.calling_conv.caller_saved_xmm_mask();
        loop->append_inst(std::move(call));
        loop->append_inst(std::make_unique<LirInst>(LirOpcode::Safepoint));

        VReg sum = hot[0];
        for (size_t i = 1; i < hot.size(); ++i) {
            VReg next = fn.allocate_vreg(RegClass::GPR, 8);
            auto add = std::make_unique<LirInst>(LirOpcode::Add);
            add->add_def(LirOperand::vreg(next, 8));
            add->add_use(LirOperand::vreg(sum, 8));
            add->add_use(LirOperand::vreg(hot[i], 8));
            loop->append_inst(std::move(add));
            sum = next;
        }
        auto cmp = std::make_unique<LirInst>(LirOpcode::Cmp);
        cmp->add_use(LirOperand::vreg(sum, 8));
        cmp->add_use(LirOperand::imm(0, 8));
        loop->append_inst(std::move(cmp));

        auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc->condition = Condition::NE;
        jcc->add_use(LirOperand::label(1));
        loop->append_inst(std::move(jcc));

        auto jmp_exit = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_exit->add_use(LirOperand::label(2));
        loop->append_inst(std::move(jmp_exit));
    }

    {
        VReg acc = outer[0];
        for (size_t i = 1; i < outer.size(); ++i) {
            VReg next = fn.allocate_vreg(RegClass::GPR, 8);
            auto add = std::make_unique<LirInst>(LirOpcode::Add);
            add->add_def(LirOperand::vreg(next, 8));
            add->add_use(LirOperand::vreg(acc, 8));
            add->add_use(LirOperand::vreg(outer[i], 8));
            exit->append_inst(std::move(add));
            acc = next;
        }
        auto vuse = std::make_unique<LirInst>(LirOpcode::Movaps);
        vuse->add_def(LirOperand::preg_xmm(XMM::XMM0, 16), FixedConstraint::xmm(XMM::XMM0));
        vuse->add_use(LirOperand::vreg(vec16, 16));
        exit->append_inst(std::move(vuse));

        auto out = std::make_unique<LirInst>(LirOpcode::Mov);
        out->add_def(LirOperand::preg_gpr(GPR::RAX, 8), FixedConstraint::gpr(GPR::RAX));
        out->add_use(LirOperand::vreg(acc, 8));
        exit->append_inst(std::move(out));
        exit->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
    }

    link_blocks(*entry, *loop);
    link_blocks(*loop, *loop);
    link_blocks(*loop, *exit);

    LivenessAnalysis liveness(fn);
    liveness.run();
    LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
    regalloc.allocate();

    // Collect every stack slot referenced inside the hot loop vs only in entry/exit.
    std::vector<int32_t> loop_slots;
    for (const auto& inst : loop->instructions) {
        for (const auto& d : inst->defs) {
            if (d.is_spill_slot()) loop_slots.push_back(d.spill_slot);
        }
        for (const auto& u : inst->uses) {
            if (u.is_spill_slot()) loop_slots.push_back(u.spill_slot);
        }
    }
    REQUIRE(loop_slots.size() >= size_t(6));
    int32_t max_loop_slot = -1;
    for (int32_t s : loop_slots) {
        max_loop_slot = std::max(max_loop_slot, s);
        // Every hot loop spill access lands in x86-64 disp8 ([rbp - 8 .. -128]).
        CHECK(s < 16);
    }

    std::vector<int32_t> cold_scalar_slots;
    int32_t vslot = -1;
    for (const auto* blk : {entry, exit}) {
        for (const auto& inst : blk->instructions) {
            for (const auto& op : inst->uses) {
                if (!op.is_spill_slot()) continue;
                if (op.size == 16) vslot = op.spill_slot;
                else if (std::find(loop_slots.begin(), loop_slots.end(), op.spill_slot) == loop_slots.end()) {
                    cold_scalar_slots.push_back(op.spill_slot);
                }
            }
        }
    }
    REQUIRE(!cold_scalar_slots.empty());
    for (int32_t cs : cold_scalar_slots) {
        CHECK(max_loop_slot < cs);
    }

    // GC ref flag and safepoint live_gcrefs track the permuted slot of outer[3].
    const auto& gcref_info = fn.get_vreg_info(outer[3]);
    REQUIRE(gcref_info.is_spilled);
    const int32_t gc_slot = gcref_info.assigned_spill_slot;
    REQUIRE(static_cast<size_t>(gc_slot) < fn.frame.spill_slot_is_gcref.size());
    for (size_t s = 0; s < fn.frame.spill_slot_is_gcref.size(); ++s) {
        CHECK_EQ(fn.frame.spill_slot_is_gcref[s], s == static_cast<size_t>(gc_slot));
    }
    for (const auto& blk : {entry, loop}) {
        for (const auto& inst : blk->instructions) {
            if (inst->opcode != LirOpcode::Safepoint) continue;
            bool found = false;
            for (const VReg v : inst->live_gcrefs) {
                if (v == outer[3] && fn.get_vreg_info(v).assigned_spill_slot == gc_slot) found = true;
            }
            CHECK(found);
        }
    }

    // 16-byte XMM spill preserves 16-byte stack alignment ((slot + 1) * 8 % 16 == 0)
    // and never overlaps [vslot - 1, vslot] with any scalar spill slot.
    REQUIRE(vslot >= 1);
    CHECK_EQ(((vslot + 1) * 8) % 16, 0);
    for (int32_t s : loop_slots) {
        CHECK_NE(s, vslot);
        CHECK_NE(s, vslot - 1);
    }
    for (int32_t s : cold_scalar_slots) {
        CHECK_NE(s, vslot);
        CHECK_NE(s, vslot - 1);
    }
}

TEST_CASE("Linear Scan - Conditional XMM12 and XMM13 Allocation and 4-Scratch Blocking on x86-64") {
    // With <= 2 distinct XMM vregs per instruction, XMM0..XMM13 (14 registers) are all allocatable.
    {
        LirFunction fn;
        fn.name = "xmm_14_live";
        fn.calling_conv = CallingConvention::sysv64();
        LirBlock* entry = fn.create_block("entry");

        std::vector<VReg> xs(14);
        for (size_t i = 0; i < xs.size(); ++i) {
            xs[i] = fn.allocate_vreg(RegClass::XMM, 8);
            auto def = std::make_unique<LirInst>(LirOpcode::Movsd);
            def->add_def(LirOperand::vreg(xs[i], 8));
            LirMem mem;
            mem.base_preg = PReg::gpr(GPR::RDI);
            mem.disp = static_cast<int32_t>(i * 8);
            def->add_use(LirOperand::mem_custom(mem, 8));
            entry->append_inst(std::move(def));
        }
        for (size_t i = 1; i < xs.size(); ++i) {
            auto add = std::make_unique<LirInst>(LirOpcode::Addsd);
            add->add_def(LirOperand::vreg(xs[0], 8));
            add->add_use(LirOperand::vreg(xs[0], 8));
            add->add_use(LirOperand::vreg(xs[i], 8));
            entry->append_inst(std::move(add));
        }
        auto out = std::make_unique<LirInst>(LirOpcode::Movsd);
        out->add_def(LirOperand::preg_xmm(XMM::XMM0, 8), FixedConstraint::xmm(XMM::XMM0));
        out->add_use(LirOperand::vreg(xs[0], 8));
        entry->append_inst(std::move(out));
        entry->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        LivenessAnalysis liveness(fn);
        liveness.run();
        LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
        regalloc.allocate();

        CHECK_EQ(regalloc.num_spill_slots(), size_t(0));
        bool used_xmm12 = false;
        bool used_xmm13 = false;
        for (VReg v : xs) {
            const PReg p = fn.get_vreg_info(v).assigned_preg;
            CHECK(p.is_valid());
            if (p == PReg::xmm(XMM::XMM12)) used_xmm12 = true;
            if (p == PReg::xmm(XMM::XMM13)) used_xmm13 = true;
        }
        CHECK(used_xmm12);
        CHECK(used_xmm13);
    }

    // A 4-XMM-operand instruction (Vfmadd213sd: 1 def + 3 uses) with all 4 operands
    // spilled across a call gets 4 distinct XMM scratches (XMM14, XMM15, XMM13, XMM12).
    {
        LirFunction fn;
        fn.name = "xmm_4_scratch_fma";
        fn.calling_conv = CallingConvention::sysv64();
        LirBlock* entry = fn.create_block("entry");

        std::array<VReg, 3> u{};
        for (size_t i = 0; i < u.size(); ++i) {
            u[i] = fn.allocate_vreg(RegClass::XMM, 8);
            auto def = std::make_unique<LirInst>(LirOpcode::Xorpd);
            def->add_def(LirOperand::vreg(u[i], 8));
            def->add_use(LirOperand::vreg(u[i], 8));
            def->add_use(LirOperand::vreg(u[i], 8));
            entry->append_inst(std::move(def));
        }
        auto call1 = std::make_unique<LirInst>(LirOpcode::Call);
        call1->add_use(LirOperand::symbol("ext_xmm"));
        call1->clobbered_gprs = fn.calling_conv.caller_saved_gpr_mask();
        call1->clobbered_xmms = fn.calling_conv.caller_saved_xmm_mask();
        entry->append_inst(std::move(call1));

        VReg dst = fn.allocate_vreg(RegClass::XMM, 8);
        auto fma = std::make_unique<LirInst>(LirOpcode::Vfmadd213sd);
        fma->add_def(LirOperand::vreg(dst, 8));
        fma->add_use(LirOperand::vreg(u[0], 8));
        fma->add_use(LirOperand::vreg(u[1], 8));
        fma->add_use(LirOperand::vreg(u[2], 8));
        entry->append_inst(std::move(fma));

        auto call2 = std::make_unique<LirInst>(LirOpcode::Call);
        call2->add_use(LirOperand::symbol("ext_xmm"));
        call2->clobbered_gprs = fn.calling_conv.caller_saved_gpr_mask();
        call2->clobbered_xmms = fn.calling_conv.caller_saved_xmm_mask();
        entry->append_inst(std::move(call2));

        auto out = std::make_unique<LirInst>(LirOpcode::Movsd);
        out->add_def(LirOperand::preg_xmm(XMM::XMM0, 8), FixedConstraint::xmm(XMM::XMM0));
        out->add_use(LirOperand::vreg(dst, 8));
        entry->append_inst(std::move(out));
        entry->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        LivenessAnalysis liveness(fn);
        liveness.run();
        LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
        regalloc.allocate();

        const LirInst* rewritten_fma = nullptr;
        for (const auto& inst : entry->instructions) {
            if (inst->opcode == LirOpcode::Vfmadd213sd) rewritten_fma = inst.get();
        }
        REQUIRE(rewritten_fma != nullptr);
        CHECK_EQ(rewritten_fma->uses.size(), size_t(3));
        CHECK(rewritten_fma->defs[0].is_preg());
        CHECK(rewritten_fma->uses[0].is_preg());
        CHECK(rewritten_fma->uses[1].is_preg());
        CHECK(rewritten_fma->uses[2].is_preg());
        CHECK(rewritten_fma->uses[0].preg_val != rewritten_fma->uses[1].preg_val);
        CHECK(rewritten_fma->uses[0].preg_val != rewritten_fma->uses[2].preg_val);
        CHECK(rewritten_fma->uses[1].preg_val != rewritten_fma->uses[2].preg_val);
    }
}
