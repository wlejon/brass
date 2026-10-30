#include "test_framework.hpp"
#include <brass/codegen/peephole.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/target/x64/x64_registers.hpp>
#include <brass/target/aarch64/aarch64_registers.hpp>
#include <cstdlib>

// Repeated scratch reloads (peephole_reload.cpp): a reload of a spill slot
// into a scratch register that already holds that slot, with nothing between
// that writes either, is dropped. Every case that must keep the second reload
// is pinned here: a write of the register (explicit, a clobber, the emitters'
// implicit R11), of the slot, a call, a parallel copy, a different width, and
// a block that can be entered by a branch, a landing pad or a resume entry.
// x64 and AArch64 (X12) alike; the AArch64 cases run on any host.

using namespace brass;
using namespace brass::codegen;

namespace {

using x64::GPR;

std::unique_ptr<LirInst> reload(PReg r, int32_t slot, LirOpcode op = LirOpcode::Mov, uint8_t size = 8) {
    auto m = std::make_unique<LirInst>(op);
    m->add_def(LirOperand::preg(r, size));
    m->add_use(LirOperand::slot(slot, size));
    return m;
}

// A use of the scratch as a load's base: rax = [r + 16].
std::unique_ptr<LirInst> use_base(PReg r, PReg dst) {
    auto m = std::make_unique<LirInst>(LirOpcode::Mov);
    m->add_def(LirOperand::preg(dst, 8));
    LirMem mem;
    mem.base_preg = r;
    mem.disp = 16;
    m->add_use(LirOperand::mem_custom(mem, 8));
    return m;
}

size_t count_reloads(const LirBlock& bb, PReg r, int32_t slot) {
    size_t n = 0;
    for (const auto& inst : bb.instructions) {
        if (inst->defs.size() == 1 && inst->defs[0].is_preg() && inst->defs[0].preg_val == r &&
            inst->uses.size() == 1 && inst->uses[0].is_spill_slot() && inst->uses[0].spill_slot == slot) {
            ++n;
        }
    }
    return n;
}

const PReg kR10 = PReg::gpr(GPR::R10);
const PReg kR11 = PReg::gpr(GPR::R11);
const PReg kRax = PReg::gpr(GPR::RAX);
const PReg kRcx = PReg::gpr(GPR::RCX);

bool disabled() { return std::getenv("BRASS_NO_RELOAD_REUSE") != nullptr; }

// Two reloads of slot 3 into `r` around `between` (none: nothing between).
size_t reloads_left(std::unique_ptr<LirInst> between, PReg r = kR10, bool aarch64 = false) {
    LirFunction fn;
    fn.calling_conv = aarch64 ? CallingConvention::aapcs64() : CallingConvention::sysv64();
    LirBlock* bb = fn.create_block("entry");
    const PReg dst = aarch64 ? PReg::aarch64_gpr(aarch64::GPR::X0) : kRax;
    bb->append_inst(reload(r, 3));
    bb->append_inst(use_base(r, dst));
    if (between) bb->append_inst(std::move(between));
    bb->append_inst(reload(r, 3));
    bb->append_inst(use_base(r, aarch64 ? PReg::aarch64_gpr(aarch64::GPR::X1) : kRcx));
    run_lir_peephole_optimizations(fn);
    return count_reloads(*bb, r, 3);
}

}  // namespace

TEST_CASE("Reload reuse - a second reload of the same slot into the same scratch is dropped") {
    if (disabled()) return;
    CHECK_EQ(reloads_left(nullptr), size_t(1));
    CHECK_EQ(reloads_left(nullptr, kR11), size_t(1));
    CHECK_EQ(reloads_left(nullptr, PReg::gpr(GPR::R15)), size_t(1));
    // An unrelated instruction between changes nothing.
    auto add = std::make_unique<LirInst>(LirOpcode::Add);
    add->add_def(LirOperand::preg(kRcx, 8));
    add->add_use(LirOperand::preg(kRcx, 8));
    add->add_use(LirOperand::imm(1, 8));
    CHECK_EQ(reloads_left(std::move(add)), size_t(1));
}

TEST_CASE("Reload reuse - a write of the scratch keeps the reload") {
    // Explicit def (not a plain move, which the dead-move pass would drop
    // before the reload that overwrites it).
    auto add = std::make_unique<LirInst>(LirOpcode::Add);
    add->add_def(LirOperand::preg(kR10, 8));
    add->add_use(LirOperand::preg(kR10, 8));
    add->add_use(LirOperand::imm(5, 8));
    CHECK_EQ(reloads_left(std::move(add)), size_t(2));
    // A 32-bit write of it.
    auto x32 = std::make_unique<LirInst>(LirOpcode::Add32);
    x32->add_def(LirOperand::preg(kR10, 4));
    x32->add_use(LirOperand::preg(kR10, 4));
    x32->add_use(LirOperand::imm(1, 4));
    CHECK_EQ(reloads_left(std::move(x32)), size_t(2));
    // A recorded clobber.
    auto clob = std::make_unique<LirInst>(LirOpcode::Nop);
    clob->clobbered_gprs = x64::reg_mask(GPR::R10);
    CHECK_EQ(reloads_left(std::move(clob)), size_t(2));
}

TEST_CASE("Reload reuse - the emitters' implicit R11 writes keep an R11 reload") {
    // A memory-to-memory move goes through R11 (emit_alu.cpp).
    auto mm = std::make_unique<LirInst>(LirOpcode::Mov);
    mm->add_def(LirOperand::slot(7, 8));
    mm->add_use(LirOperand::slot(8, 8));
    CHECK_EQ(reloads_left(std::move(mm), kR11), size_t(2));
    // So does Fabs's mask (emit_sse.cpp).
    auto fabs = std::make_unique<LirInst>(LirOpcode::Fabs64);
    fabs->add_def(LirOperand::preg(PReg::xmm(x64::XMM::XMM0), 8));
    fabs->add_use(LirOperand::preg(PReg::xmm(x64::XMM::XMM0), 8));
    CHECK_EQ(reloads_left(std::move(fabs), kR11), size_t(2));
    // A memory-to-memory move does not touch R10: its reload is still reused.
    if (!disabled()) {
        auto mm2 = std::make_unique<LirInst>(LirOpcode::Mov);
        mm2->add_def(LirOperand::slot(7, 8));
        mm2->add_use(LirOperand::slot(8, 8));
        CHECK_EQ(reloads_left(std::move(mm2), kR10), size_t(1));
    }
}

TEST_CASE("Reload reuse - a write of the slot, a call, a parallel copy or a width change keeps the reload") {
    // Stored from the def scratch, which store-to-load forwarding leaves
    // alone (peephole.cpp is_reserved_scratch), so the reload stays a reload.
    auto st = std::make_unique<LirInst>(LirOpcode::Mov);
    st->add_def(LirOperand::slot(3, 8));
    st->add_use(LirOperand::preg(PReg::gpr(GPR::R14), 8));
    CHECK_EQ(reloads_left(std::move(st)), size_t(2));
    // A reload of another slot into the scratch, used.
    {
        LirFunction fn;
        fn.calling_conv = CallingConvention::sysv64();
        LirBlock* bb = fn.create_block("entry");
        bb->append_inst(reload(kR10, 3));
        bb->append_inst(use_base(kR10, kRax));
        bb->append_inst(reload(kR10, 4));
        bb->append_inst(use_base(kR10, kRcx));
        bb->append_inst(reload(kR10, 3));
        bb->append_inst(use_base(kR10, PReg::gpr(GPR::RDX)));
        run_lir_peephole_optimizations(fn);
        CHECK_EQ(count_reloads(*bb, kR10, 3), size_t(2));
    }
    // A wide (16-byte) store to the slot below covers slot 3 too.
    auto wide = std::make_unique<LirInst>(LirOpcode::Movaps);
    wide->add_def(LirOperand::slot(4, 16));
    wide->add_use(LirOperand::preg(PReg::xmm(x64::XMM::XMM1), 16));
    CHECK_EQ(reloads_left(std::move(wide)), size_t(2));
    // A store relative to the frame pointer, not through a slot operand.
    auto fp = std::make_unique<LirInst>(LirOpcode::Mov);
    LirMem mem;
    mem.base_preg = PReg::gpr(GPR::RBP);
    mem.disp = -32;
    fp->add_def(LirOperand::mem_custom(mem, 8));
    fp->add_use(LirOperand::preg(kRcx, 8));
    CHECK_EQ(reloads_left(std::move(fp)), size_t(2));
    auto call = std::make_unique<LirInst>(LirOpcode::Call);
    call->add_use(LirOperand::symbol("f"));
    CHECK_EQ(reloads_left(std::move(call)), size_t(2));
    auto pc = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
    pc->add_def(LirOperand::preg(PReg::gpr(GPR::RDX), 8));
    pc->add_use(LirOperand::preg(kRcx, 8));
    CHECK_EQ(reloads_left(std::move(pc)), size_t(2));
    auto sp = std::make_unique<LirInst>(LirOpcode::Safepoint);
    CHECK_EQ(reloads_left(std::move(sp)), size_t(2));

    // A 32-bit reload after a 64-bit one of the same slot is another value.
    LirFunction fn;
    fn.calling_conv = CallingConvention::sysv64();
    LirBlock* bb = fn.create_block("entry");
    bb->append_inst(reload(kR10, 3));
    bb->append_inst(use_base(kR10, kRax));
    bb->append_inst(reload(kR10, 3, LirOpcode::Mov32, 4));
    run_lir_peephole_optimizations(fn);
    CHECK_EQ(bb->instructions.size(), size_t(3));
}

TEST_CASE("Reload reuse - across blocks only into a block entered by falling through alone") {
    // entry: reload; use; jcc exit       (a guard)
    // next:  reload; use; jmp join       (fall-through only: reused)
    // other: reload; jmp join            (branched to: kept)
    // join:  reload; use; ret            (two predecessors: kept)
    // exit:  ret
    auto build = [](LirFunction& fn, bool resume_next, bool landing_next) {
        fn.calling_conv = CallingConvention::sysv64();
        LirBlock* entry = fn.create_block_with_id(0, "entry");
        LirBlock* next = fn.create_block_with_id(1, "next");
        LirBlock* other = fn.create_block_with_id(2, "other");
        LirBlock* join = fn.create_block_with_id(3, "join");
        LirBlock* exit = fn.create_block_with_id(4, "exit");
        entry->append_inst(reload(kR10, 3));
        entry->append_inst(use_base(kR10, kRax));
        auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc->condition = x64::Condition::E;
        jcc->add_use(LirOperand::label(4));
        entry->append_inst(std::move(jcc));
        next->append_inst(reload(kR10, 3));
        next->append_inst(use_base(kR10, kRcx));
        auto j1 = std::make_unique<LirInst>(LirOpcode::Jmp);
        j1->add_use(LirOperand::label(3));
        next->append_inst(std::move(j1));
        other->append_inst(reload(kR10, 3));
        auto j2 = std::make_unique<LirInst>(LirOpcode::Jmp);
        j2->add_use(LirOperand::label(3));
        other->append_inst(std::move(j2));
        join->append_inst(reload(kR10, 3));
        join->append_inst(use_base(kR10, kRcx));
        join->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
        exit->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
        link_blocks(*entry, *next);
        link_blocks(*entry, *exit);
        link_blocks(*next, *join);
        link_blocks(*other, *join);
        if (resume_next) fn.resume_entries.push_back({1, 1});
        if (landing_next) next->is_landing_pad = true;
        return std::array<LirBlock*, 5>{entry, next, other, join, exit};
    };
    {
        LirFunction fn;
        auto b = build(fn, false, false);
        run_lir_peephole_optimizations(fn);
        CHECK_EQ(count_reloads(*b[0], kR10, 3), size_t(1));
        CHECK_EQ(count_reloads(*b[1], kR10, 3), disabled() ? size_t(1) : size_t(0));
        CHECK_EQ(count_reloads(*b[2], kR10, 3), size_t(1));
        CHECK_EQ(count_reloads(*b[3], kR10, 3), size_t(1));
    }
    {
        LirFunction fn;
        auto b = build(fn, true, false);
        run_lir_peephole_optimizations(fn);
        CHECK_EQ(count_reloads(*b[1], kR10, 3), size_t(1));
    }
    {
        LirFunction fn;
        auto b = build(fn, false, true);
        run_lir_peephole_optimizations(fn);
        CHECK_EQ(count_reloads(*b[1], kR10, 3), size_t(1));
    }
}

TEST_CASE("Reload reuse - AArch64: X12 reloads are reused, the same writes keep them") {
    const PReg x12 = PReg::aarch64_gpr(aarch64::GPR::X12);
    if (!disabled()) CHECK_EQ(reloads_left(nullptr, x12, true), size_t(1));
    auto add = std::make_unique<LirInst>(LirOpcode::Add);
    add->add_def(LirOperand::preg(x12, 8));
    add->add_use(LirOperand::preg(x12, 8));
    add->add_use(LirOperand::imm(5, 8));
    CHECK_EQ(reloads_left(std::move(add), x12, true), size_t(2));
    auto st = std::make_unique<LirInst>(LirOpcode::Mov);
    st->add_def(LirOperand::slot(3, 8));
    st->add_use(LirOperand::preg(PReg::aarch64_gpr(aarch64::GPR::X14), 8));
    CHECK_EQ(reloads_left(std::move(st), x12, true), size_t(2));
    auto call = std::make_unique<LirInst>(LirOpcode::Call);
    call->add_use(LirOperand::symbol("f"));
    CHECK_EQ(reloads_left(std::move(call), x12, true), size_t(2));
    // X15 is the emitters' store scratch on AArch64: never tracked.
    const PReg x15 = PReg::aarch64_gpr(aarch64::GPR::X15);
    CHECK_EQ(reloads_left(nullptr, x15, true), size_t(2));
}
