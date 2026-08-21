#include <brass/codegen/peephole.hpp>
#include <brass/target/x64/x64_operands.hpp>
#include <algorithm>

namespace brass::codegen {

PeepholeOptimizer::PeepholeOptimizer(LirFunction& fn)
    : fn_(fn) {}

bool PeepholeOptimizer::is_protected(const LirInst& inst) noexcept {
    if (inst.safepoint_id != 0) return true;
    if (inst.resume_id != 0) return true;
    if (inst.is_patchable) return true;
    if (inst.opcode == LirOpcode::Safepoint) return true;
    if (inst.opcode == LirOpcode::GuardExit) return true;
    if (inst.opcode == LirOpcode::Call || inst.opcode == LirOpcode::CallIndirect) return true;
    if (!inst.live_gcrefs.empty()) return true;
    return false;
}

bool PeepholeOptimizer::operands_equal(const LirOperand& a, const LirOperand& b) noexcept {
    if (a.kind != b.kind || a.size != b.size) return false;
    switch (a.kind) {
        case LirOperandKind::PReg:
            return a.preg_val == b.preg_val;
        case LirOperandKind::VReg:
            return a.vreg_val == b.vreg_val;
        case LirOperandKind::SpillSlot:
            return a.spill_slot == b.spill_slot;
        case LirOperandKind::Mem:
            return a.mem_val == b.mem_val;
        case LirOperandKind::ImmInt:
            return a.imm_int == b.imm_int;
        case LirOperandKind::Label:
            return a.label_id == b.label_id;
        default:
            return false;
    }
}

bool PeepholeOptimizer::uses_register(const LirInst& inst, PReg reg) noexcept {
    for (const auto& u : inst.uses) {
        if (u.is_preg() && u.preg_val == reg) {
            return true;
        }
        if (u.is_mem()) {
            if (u.mem_val.base_preg == reg || u.mem_val.index_preg == reg) {
                return true;
            }
        }
    }
    for (const auto& d : inst.defs) {
        if (d.is_mem()) {
            if (d.mem_val.base_preg == reg || d.mem_val.index_preg == reg) {
                return true;
            }
        }
    }
    if (inst.is_call()) {
        return true;
    }
    return false;
}

bool PeepholeOptimizer::defines_register(const LirInst& inst, PReg reg) noexcept {
    for (const auto& d : inst.defs) {
        if (d.is_preg() && d.preg_val == reg) {
            return true;
        }
    }
    if (reg.is_gpr() && (inst.clobbered_gprs & (1u << reg.code))) {
        return true;
    }
    if (reg.is_xmm() && (inst.clobbered_xmms & (1u << reg.code))) {
        return true;
    }
    return false;
}

bool PeepholeOptimizer::touches_memory(const LirInst& inst) noexcept {
    if (inst.is_call() || inst.opcode == LirOpcode::Safepoint) return true;
    for (const auto& d : inst.defs) {
        if (d.is_mem() || d.is_spill_slot()) return true;
    }
    for (const auto& u : inst.uses) {
        if (u.is_mem() || u.is_spill_slot()) return true;
    }
    return false;
}

bool PeepholeOptimizer::eliminate_redundant_moves(LirBlock& block) {
    bool changed = false;
    auto it = block.instructions.begin();
    while (it != block.instructions.end()) {
        const auto& inst = **it;
        if (!is_protected(inst) &&
            (inst.opcode == LirOpcode::Mov || inst.opcode == LirOpcode::Mov32 ||
             inst.opcode == LirOpcode::Movsd || inst.opcode == LirOpcode::Movss) &&
            inst.defs.size() >= 1 && inst.uses.size() >= 1 &&
            inst.defs[0].is_preg() && inst.uses[0].is_preg() &&
            inst.defs[0].preg_val == inst.uses[0].preg_val) {
            it = block.instructions.erase(it);
            stats_.redundant_moves_eliminated++;
            changed = true;
        } else {
            ++it;
        }
    }
    return changed;
}

bool PeepholeOptimizer::eliminate_load_after_store(LirBlock& block) {
    bool changed = false;
    for (size_t i = 0; i < block.instructions.size(); ++i) {
        const auto& store_inst = *block.instructions[i];
        if (is_protected(store_inst)) continue;

        bool is_store = (store_inst.opcode == LirOpcode::Mov || store_inst.opcode == LirOpcode::Mov32 ||
                         store_inst.opcode == LirOpcode::Movsd || store_inst.opcode == LirOpcode::Movss) &&
                        !store_inst.defs.empty() && (store_inst.defs[0].is_spill_slot() || store_inst.defs[0].is_mem()) &&
                        !store_inst.uses.empty() && store_inst.uses[0].is_preg();
        if (!is_store) continue;

        const LirOperand& store_loc = store_inst.defs[0];
        PReg store_src_reg = store_inst.uses[0].preg_val;

        for (size_t j = i + 1; j < block.instructions.size(); ++j) {
            auto& candidate = *block.instructions[j];
            if (is_protected(candidate)) break;
            if (candidate.is_call() || candidate.is_branch() || candidate.is_terminator()) break;

            bool is_matching_load = (candidate.opcode == store_inst.opcode) &&
                                    !candidate.defs.empty() && candidate.defs[0].is_preg() &&
                                    !candidate.uses.empty() && operands_equal(candidate.uses[0], store_loc);

            if (is_matching_load) {
                PReg load_dst_reg = candidate.defs[0].preg_val;
                if (load_dst_reg == store_src_reg) {
                    // Exact redundant load: mov [loc], r; ...; mov r, [loc]
                    block.instructions.erase(block.instructions.begin() + static_cast<ptrdiff_t>(j));
                    stats_.load_after_store_eliminated++;
                    changed = true;
                    --j;
                    continue;
                } else {
                    // Forwarding: mov [loc], r1; ...; mov r2, [loc] -> mov r2, r1
                    candidate.uses[0] = LirOperand::preg(store_src_reg, store_loc.size);
                    stats_.load_after_store_eliminated++;
                    changed = true;
                    break;
                }
            }

            // Check for invalidation
            if (defines_register(candidate, store_src_reg)) {
                break;
            }
            if (candidate.defs.size() >= 1 && (candidate.defs[0].is_mem() || candidate.defs[0].is_spill_slot())) {
                if (operands_equal(candidate.defs[0], store_loc) || candidate.defs[0].is_mem()) {
                    break;
                }
            }
        }
    }
    return changed;
}

bool PeepholeOptimizer::eliminate_dead_moves(LirBlock& block) {
    bool changed = false;
    for (size_t i = 0; i < block.instructions.size(); ++i) {
        const auto& inst = *block.instructions[i];
        if (is_protected(inst)) continue;

        bool is_pure_move = (inst.opcode == LirOpcode::Mov || inst.opcode == LirOpcode::Mov32 ||
                             inst.opcode == LirOpcode::Movsd || inst.opcode == LirOpcode::Movss ||
                             inst.opcode == LirOpcode::Movabs || inst.opcode == LirOpcode::Movsx8 ||
                             inst.opcode == LirOpcode::Movsx16 || inst.opcode == LirOpcode::Movsxd ||
                             inst.opcode == LirOpcode::Movzx8 || inst.opcode == LirOpcode::Movzx16) &&
                            inst.defs.size() == 1 && inst.defs[0].is_preg();
        if (!is_pure_move) continue;

        PReg target_reg = inst.defs[0].preg_val;
        bool is_dead = false;

        for (size_t j = i + 1; j < block.instructions.size(); ++j) {
            const auto& candidate = *block.instructions[j];
            if (uses_register(candidate, target_reg)) {
                break;
            }
            if (candidate.is_call() || candidate.is_branch() || candidate.is_terminator() || is_protected(candidate)) {
                break;
            }
            if (candidate.defs.size() >= 1 && candidate.defs[0].is_preg() && candidate.defs[0].preg_val == target_reg) {
                is_dead = true;
                break;
            }
        }

        if (is_dead) {
            block.instructions.erase(block.instructions.begin() + static_cast<ptrdiff_t>(i));
            stats_.dead_moves_eliminated++;
            changed = true;
            --i;
        }
    }
    return changed;
}

bool PeepholeOptimizer::simplify_arithmetic(LirBlock& block) {
    bool changed = false;
    auto it = block.instructions.begin();
    while (it != block.instructions.end()) {
        auto& inst = **it;
        if (is_protected(inst)) {
            ++it;
            continue;
        }

        // 1. add reg, 0 / add32 reg, 0
        if ((inst.opcode == LirOpcode::Add || inst.opcode == LirOpcode::Add32) &&
            inst.defs.size() >= 1 && inst.defs[0].is_preg() &&
            !inst.uses.empty() && inst.uses.back().is_imm_int() && inst.uses.back().imm_int == 0) {
            it = block.instructions.erase(it);
            stats_.arithmetic_simplified++;
            changed = true;
            continue;
        }

        // 2. sub reg, 0 / sub32 reg, 0
        if ((inst.opcode == LirOpcode::Sub || inst.opcode == LirOpcode::Sub32) &&
            inst.defs.size() >= 1 && inst.defs[0].is_preg() &&
            !inst.uses.empty() && inst.uses.back().is_imm_int() && inst.uses.back().imm_int == 0) {
            it = block.instructions.erase(it);
            stats_.arithmetic_simplified++;
            changed = true;
            continue;
        }

        // 3. imul reg, 1 / imul32 reg, 1
        if ((inst.opcode == LirOpcode::Imul || inst.opcode == LirOpcode::Imul32) &&
            inst.defs.size() >= 1 && inst.defs[0].is_preg() &&
            !inst.uses.empty() && inst.uses.back().is_imm_int() && inst.uses.back().imm_int == 1) {
            it = block.instructions.erase(it);
            stats_.arithmetic_simplified++;
            changed = true;
            continue;
        }

        // 4. mov reg, 0 -> xor32 reg, reg (for GPR zeroing)
        if ((inst.opcode == LirOpcode::Mov || inst.opcode == LirOpcode::Mov32) &&
            inst.defs.size() == 1 && inst.defs[0].is_preg() && inst.defs[0].preg_val.is_gpr() &&
            inst.uses.size() == 1 && inst.uses[0].is_imm_int() && inst.uses[0].imm_int == 0) {
            PReg reg = inst.defs[0].preg_val;
            inst.opcode = LirOpcode::Xor32;
            inst.defs = {LirOperand::preg(reg, 4)};
            inst.uses = {LirOperand::preg(reg, 4), LirOperand::preg(reg, 4)};
            stats_.arithmetic_simplified++;
            changed = true;
            ++it;
            continue;
        }

        ++it;
    }
    return changed;
}

bool PeepholeOptimizer::simplify_branches(LirBlock& block, size_t block_index) {
    if (block_index + 1 >= fn_.blocks.size() || block.instructions.empty()) {
        return false;
    }

    uint32_t next_block_id = fn_.blocks[block_index + 1]->id;
    bool changed = false;

    // Case 1: Trailing unconditional jump to immediate fallthrough block
    if (block.instructions.back()->opcode == LirOpcode::Jmp &&
        !block.instructions.back()->uses.empty() &&
        block.instructions.back()->uses[0].is_label() &&
        block.instructions.back()->uses[0].label_id == next_block_id) {
        block.instructions.pop_back();
        stats_.branches_simplified++;
        changed = true;
    }

    // Case 2: Jcc Cond, T1 followed by Jmp T2
    if (block.instructions.size() >= 2) {
        size_t n = block.instructions.size();
        auto& jcc = *block.instructions[n - 2];
        auto& jmp = *block.instructions[n - 1];

        if (jcc.opcode == LirOpcode::Jcc && jmp.opcode == LirOpcode::Jmp &&
            !jcc.uses.empty() && jcc.uses[0].is_label() &&
            !jmp.uses.empty() && jmp.uses[0].is_label()) {
            uint32_t t_true = jcc.uses[0].label_id;
            uint32_t t_false = jmp.uses[0].label_id;

            if (t_false == next_block_id) {
                block.instructions.pop_back();
                stats_.branches_simplified++;
                changed = true;
            } else if (t_true == next_block_id) {
                jcc.condition = x64::invert(jcc.condition);
                jcc.uses[0] = LirOperand::label(t_false);
                block.instructions.pop_back();
                stats_.branches_simplified++;
                changed = true;
            }
        }
    }

    return changed;
}

bool PeepholeOptimizer::propagate_copies(LirBlock& block) {
    bool changed = false;
    for (size_t i = 0; i < block.instructions.size(); ++i) {
        const auto& mov_inst = *block.instructions[i];
        if (is_protected(mov_inst)) continue;

        bool is_reg_move = (mov_inst.opcode == LirOpcode::Mov || mov_inst.opcode == LirOpcode::Mov32 ||
                            mov_inst.opcode == LirOpcode::Movsd || mov_inst.opcode == LirOpcode::Movss) &&
                           mov_inst.defs.size() == 1 && mov_inst.defs[0].is_preg() &&
                           mov_inst.uses.size() == 1 && mov_inst.uses[0].is_preg() &&
                           mov_inst.defs[0].preg_val != mov_inst.uses[0].preg_val;
        if (!is_reg_move) continue;

        PReg dst_reg = mov_inst.defs[0].preg_val;
        PReg src_reg = mov_inst.uses[0].preg_val;
        uint8_t sz = mov_inst.defs[0].size;

        for (size_t j = i + 1; j < block.instructions.size(); ++j) {
            auto& candidate = *block.instructions[j];
            if (is_protected(candidate)) break;
            if (candidate.is_call() || candidate.is_branch() || candidate.is_terminator()) break;

            if (defines_register(candidate, src_reg)) {
                break;
            }

            for (size_t u_idx = 0; u_idx < candidate.uses.size(); ++u_idx) {
                if (!candidate.defs.empty() && candidate.defs[0].is_preg() && candidate.defs[0].preg_val == dst_reg) {
                    continue;
                }
                if (candidate.uses[u_idx].is_preg() && candidate.uses[u_idx].preg_val == dst_reg &&
                    candidate.uses[u_idx].size == sz) {
                    candidate.uses[u_idx].preg_val = src_reg;
                    stats_.redundant_moves_eliminated++;
                    changed = true;
                }
            }

            if (defines_register(candidate, dst_reg)) {
                break;
            }
        }
    }
    return changed;
}

bool PeepholeOptimizer::optimize_block(LirBlock& block, size_t block_index) {
    bool changed = false;
    changed |= eliminate_redundant_moves(block);
    changed |= propagate_copies(block);
    changed |= eliminate_load_after_store(block);
    changed |= eliminate_dead_moves(block);
    changed |= simplify_arithmetic(block);
    changed |= simplify_branches(block, block_index);
    return changed;
}

bool PeepholeOptimizer::run_pass() {
    bool changed = false;
    for (size_t i = 0; i < fn_.blocks.size(); ++i) {
        changed |= optimize_block(*fn_.blocks[i], i);
    }
    return changed;
}

PeepholeStats PeepholeOptimizer::run() {
    constexpr size_t kMaxPasses = 8;
    for (size_t pass = 0; pass < kMaxPasses; ++pass) {
        if (!run_pass()) {
            break;
        }
    }
    return stats_;
}

PeepholeStats run_lir_peephole_optimizations(LirFunction& fn) {
    PeepholeOptimizer opt(fn);
    return opt.run();
}

} // namespace brass::codegen
