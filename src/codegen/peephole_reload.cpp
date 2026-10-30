// Repeated spill reloads into one scratch register (PeepholeOptimizer::
// reuse_scratch_reloads).
//
// The rewriter (linear_scan_rewrite.cpp) reloads a spilled operand into a
// reserved scratch register at every instruction that reads it, so a value
// spilled for its whole life (a function-wide base pointer: a module's IC
// table, kept across calls when the three SysV callee-saved registers are
// taken) is reloaded at each use, often several times in a row with nothing
// in between that could change it: `mov -0x68(%rbp),%r10; mov 0x96868(%r10),
// %r9; mov -0x68(%rbp),%r10; mov 0x96848(%r10),%rsi`. A later reload of the
// same slot into the same register, with the same width, is the value the
// register already holds when neither was written since; it is dropped.
//
// The soundness of that rests on knowing every write to the scratch
// registers, including the ones no LIR operand names. Those were audited
// (emit_*.cpp, emit_context.cpp, emit_parallel_copy.cpp, x64_frame.cpp, the
// x64 encoders; aarch64_emit*.cpp, aarch64_frame.cpp):
//   x64     R10  only the cycle temporary of a ParallelCopy
//           R11  memory-to-memory moves (Mov, Mov32, narrow stores from
//                memory), the Fabs mask, a GuardExit's record, an exit
//                stub's stack arguments, a ParallelCopy's memory moves
//           R15  none (the rewriter's third use scratch, always a LIR def)
//           XMM12, XMM13  none (XMM14 / XMM15 are the emitters'; untracked)
//   AArch64 X12, X13  none (X14 / X15 / X16 / X17 are the emitters'; untracked)
//           V26, V28  none (V27 / V29 / V30 / V31 untracked)
// A call, safepoint, guard exit, parallel copy, write barrier or trap ends
// every tracked value, as does an instruction that may enter mid-block
// (resume_id) or that stores relative to the frame or stack pointer other
// than through a spill-slot operand. Blocks start empty: nothing is assumed
// about what a predecessor left in a scratch register.
//
// BRASS_NO_RELOAD_REUSE=1 turns it off (the A/B seam).

#include <brass/codegen/peephole.hpp>
#include <brass/target/x64/x64_operands.hpp>
#include <array>
#include <cstdlib>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass::codegen {

namespace {

using Held = PeepholeOptimizer::ScratchHeld;

bool reload_reuse_disabled() {
    static const bool off = std::getenv("BRASS_NO_RELOAD_REUSE") != nullptr;
    return off;
}

bool in_memory(const LirOperand& op) { return op.is_mem() || op.is_spill_slot(); }

// The instructions after which no scratch register is assumed to hold
// anything: they run code the LIR does not describe (a call, the runtime, an
// exit record built in R11) or are reached from elsewhere.
bool ends_all(const LirInst& inst) {
    if (inst.is_call() || inst.resume_id != 0) return true;
    // A safepoint of any kind may let a moving collection rewrite the root
    // a spill slot holds, leaving the register's copy stale; and a clobber
    // mask may name a tracked scratch without a LIR def for it.
    if (inst.safepoint_id != 0 || inst.clobbered_gprs != 0 || inst.clobbered_xmms != 0) return true;
    switch (inst.opcode) {
        case LirOpcode::Safepoint:
        case LirOpcode::GuardExit:
        case LirOpcode::ParallelCopy:
        case LirOpcode::WriteBarrier:
        case LirOpcode::Trap:
        case LirOpcode::Ret:
            return true;
        default:
            return false;
    }
}

}  // namespace

// Which blocks are entered only by falling through from the block laid out
// just before them: no branch anywhere names them, they are not the entry,
// a resume entry, a landing pad or a call's unwind target, their recorded
// predecessors (where kept) are that block alone, and the block before them ends
// in a conditional branch (a guard's `jcc` to its exit), a jump to them, or
// no terminator at all. What a scratch register holds at the end of the one
// block before is what it holds at their start.
static std::vector<char> fallthrough_only(const LirFunction& fn) {
    const size_t n = fn.blocks.size();
    std::vector<char> only(n, 0);
    std::unordered_map<uint32_t, uint32_t> branch_refs;   // block id -> branches naming it
    std::unordered_set<uint32_t> unwind_targets;
    for (const auto& b : fn.blocks) {
        for (const auto& inst : b->instructions) {
            if (!inst) continue;
            if (inst->is_branch()) {
                for (const auto& u : inst->uses) {
                    if (u.is_label()) ++branch_refs[u.label_id];
                }
            }
            if (inst->unwind_block_id != UINT32_MAX) unwind_targets.insert(inst->unwind_block_id);
        }
    }
    // A resume entry (a deopt or suspension resuming here) is entered from
    // outside, like a landing pad.
    for (const auto& rp : fn.resume_entries) unwind_targets.insert(rp.second);
    for (size_t k = 1; k < n; ++k) {
        const LirBlock& prev = *fn.blocks[k - 1];
        const LirBlock& cur = *fn.blocks[k];
        if (cur.is_landing_pad || unwind_targets.count(cur.id)) continue;
        // The CFG edges, where kept, must agree: one predecessor, the block before.
        if (!cur.predecessors.empty() && (cur.predecessors.size() != 1 || cur.predecessors[0] != &prev)) continue;
        const LirInst* last = nullptr;
        for (auto it = prev.instructions.rbegin(); it != prev.instructions.rend(); ++it) {
            if (*it) {
                last = it->get();
                break;
            }
        }
        uint32_t refs = 0;
        if (auto it = branch_refs.find(cur.id); it != branch_refs.end()) refs = it->second;
        bool falls_in = false;
        if (!last || !last->is_terminator()) {
            falls_in = refs == 0;
        } else if (last->opcode == LirOpcode::Jcc) {
            falls_in = refs == 0;
        } else if (last->opcode == LirOpcode::Jmp && !last->uses.empty() && last->uses[0].is_label() &&
                   last->uses[0].label_id == cur.id) {
            falls_in = refs == 1;
        }
        only[k] = falls_in ? 1 : 0;
    }
    return only;
}

bool PeepholeOptimizer::reuse_scratch_reloads() {
    if (reload_reuse_disabled()) return false;
    const bool a64 = fn_.calling_conv.target().is_aarch64();
    std::array<Held, 5> held;
    if (a64) {
        held = {Held{PReg::aarch64_gpr(aarch64::GPR::X12)}, Held{PReg::aarch64_gpr(aarch64::GPR::X13)},
                Held{PReg::aarch64_fpr(aarch64::FPR::V26)}, Held{PReg::aarch64_fpr(aarch64::FPR::V28)},
                Held{PReg{}}};
    } else {
        held = {Held{PReg::gpr(x64::GPR::R10)}, Held{PReg::gpr(x64::GPR::R11)}, Held{PReg::gpr(x64::GPR::R15)},
                Held{PReg::xmm(x64::XMM::XMM12)}, Held{PReg::xmm(x64::XMM::XMM13)}};
    }
    const PReg r11 = a64 ? PReg{} : PReg::gpr(x64::GPR::R11);
    const PReg fp = a64 ? PReg::aarch64_gpr(aarch64::GPR::X29) : PReg::gpr(x64::GPR::RBP);
    const PReg sp = a64 ? PReg::aarch64_gpr(aarch64::GPR::SP) : PReg::gpr(x64::GPR::RSP);
    auto forget_all = [&] {
        for (Held& h : held) h.slot = -1;
    };

    const std::vector<char> inherits = fallthrough_only(fn_);
    bool changed = false;
    for (size_t bi = 0; bi < fn_.blocks.size(); ++bi) {
        LirBlock& block = *fn_.blocks[bi];
        if (!inherits[bi]) forget_all();
        if (reuse_scratch_reloads_block(block, held, r11, fp, sp)) changed = true;
    }
    return changed;
}

bool PeepholeOptimizer::reuse_scratch_reloads_block(LirBlock& block, std::array<ScratchHeld, 5>& held, PReg r11,
                                                     PReg fp, PReg sp) {
    auto forget_all = [&] {
        for (Held& h : held) h.slot = -1;
    };
    bool changed = false;
    const size_t n = block.instructions.size();
    for (size_t i = 0; i < n; ++i) {
        if (!block.instructions[i]) continue;
        const LirInst& inst = *block.instructions[i];

        // A full-width plain reload of a slot into a tracked scratch.
        Held* reload = nullptr;
        const bool plain_load = (inst.opcode == LirOpcode::Mov || inst.opcode == LirOpcode::Movsd) &&
                                inst.defs.size() == 1 && inst.uses.size() == 1 && inst.defs[0].is_preg() &&
                                inst.uses[0].is_spill_slot() && inst.defs[0].size == 8 &&
                                inst.uses[0].size == 8 && inst.safepoint_id == 0 && inst.resume_id == 0 &&
                                !inst.is_patchable && inst.clobbered_gprs == 0 && inst.clobbered_xmms == 0;
        if (plain_load) {
            for (Held& h : held) {
                if (h.reg.is_valid() && h.reg == inst.defs[0].preg_val) reload = &h;
            }
        }
        if (reload && reload->slot == inst.uses[0].spill_slot && reload->opcode == inst.opcode &&
            reload->size == inst.defs[0].size) {
            block.instructions[i].reset();
            stats_.reloads_reused++;
            changed = true;
            continue;
        }

        if (ends_all(inst)) {
            forget_all();
            continue;
        }
        for (Held& h : held) {
            if (h.slot >= 0 && defines_register(inst, h.reg)) h.slot = -1;
        }
        // R11 behind the LIR's back (see the audit above); the rewriter
        // notes most of these as clobbers, but not the resolution moves.
        if (r11.is_valid() && !inst.defs.empty() && !inst.uses.empty() &&
            ((in_memory(inst.defs[0]) && in_memory(inst.uses[0])) || inst.opcode == LirOpcode::Fabs32 ||
             inst.opcode == LirOpcode::Fabs64)) {
            for (Held& h : held) {
                if (h.reg == r11) h.slot = -1;
            }
        }
        for (const LirOperand& d : inst.defs) {
            if (d.is_spill_slot()) {
                // A wide value covers the slots next to the one it is
                // known by (linear_scan.cpp allocate_spill_slot).
                if (d.size > 8) {
                    forget_all();
                } else {
                    for (Held& h : held) {
                        if (h.slot == d.spill_slot) h.slot = -1;
                    }
                }
            } else if (d.is_mem()) {
                const auto& m = d.mem_val;
                if (m.base_preg == fp || m.base_preg == sp || m.index_preg == fp || m.index_preg == sp ||
                    (!m.base_preg.is_valid() && !m.index_preg.is_valid())) {
                    forget_all();
                }
            }
        }
        if (reload) {
            reload->slot = inst.uses[0].spill_slot;
            reload->opcode = inst.opcode;
            reload->size = inst.defs[0].size;
        }
    }
    if (changed) {
        std::erase_if(block.instructions, [](const std::unique_ptr<LirInst>& p) { return !p; });
    }
    return changed;
}

}  // namespace brass::codegen
