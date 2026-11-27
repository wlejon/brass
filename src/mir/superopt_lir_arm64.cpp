#include <brass/mir/superopt.hpp>
#include <brass/verify/arm64_lir_equiv.hpp>
#include <cstdint>
#include <memory>
#include <vector>

namespace brass {

using codegen::LirBlock;
using codegen::LirInst;
using codegen::LirOpcode;
using codegen::LirOperand;

namespace {

LirBlock clone_arm64_block(const LirBlock& src) {
    LirBlock dst(src.id, src.name);
    dst.loop_depth = src.loop_depth;
    dst.x64_label = src.x64_label;
    dst.is_landing_pad = src.is_landing_pad;
    dst.instructions.reserve(src.instructions.size());
    for (const auto& inst : src.instructions) {
        if (inst) dst.instructions.push_back(std::make_unique<LirInst>(*inst));
    }
    return dst;
}

bool is_pure_arm64_alu_or_mov(LirOpcode op) noexcept {
    switch (op) {
        case LirOpcode::Mov:   case LirOpcode::Mov32:
        case LirOpcode::Add:   case LirOpcode::Add32:
        case LirOpcode::Sub:   case LirOpcode::Sub32:
        case LirOpcode::Imul:  case LirOpcode::Imul32:
        case LirOpcode::And:   case LirOpcode::And32:
        case LirOpcode::Or:    case LirOpcode::Or32:
        case LirOpcode::Xor:   case LirOpcode::Xor32:
        case LirOpcode::Shl:   case LirOpcode::Shl32:
        case LirOpcode::Shr:   case LirOpcode::Shr32:
        case LirOpcode::Sar:   case LirOpcode::Sar32:
            return true;
        default:
            return false;
    }
}

uint64_t eval_bin_imm64(LirOpcode op, uint64_t a, uint64_t b) noexcept {
    switch (op) {
        case LirOpcode::Add: return a + b;
        case LirOpcode::Sub: return a - b;
        case LirOpcode::And: return a & b;
        case LirOpcode::Or:  return a | b;
        case LirOpcode::Xor: return a ^ b;
        case LirOpcode::Shl: return a << (b & 63u);
        case LirOpcode::Shr: return a >> (b & 63u);
        case LirOpcode::Sar: return static_cast<uint64_t>(static_cast<int64_t>(a) >> (b & 63u));
        default: return a;
    }
}

bool is_protected_inst(const LirInst& inst) noexcept {
    return inst.is_patchable || inst.safepoint_id != 0 || inst.resume_id != 0 || !inst.live_gcrefs.empty();
}

bool has_symbolic_operand(const LirInst& inst) noexcept {
    for (const auto& d : inst.defs) {
        if (d.is_symbol() || d.is_label()) return true;
    }
    for (const auto& u : inst.uses) {
        if (u.is_symbol() || u.is_label()) return true;
    }
    return false;
}

bool operand_reads_gpr(const LirOperand& op, codegen::PReg r) noexcept {
    if (op.is_preg()) return op.preg_val == r;
    if (op.is_mem()) return op.mem_val.base_preg == r || op.mem_val.index_preg == r;
    return false;
}

bool is_add_or_sub_imm64(const LirInst& inst, codegen::PReg& reg, int64_t& signed_imm) noexcept {
    if (inst.opcode != LirOpcode::Add && inst.opcode != LirOpcode::Sub) return false;
    if (inst.defs.size() != 1 || inst.uses.size() != 2) return false;
    if (!inst.defs[0].is_preg() || !inst.uses[0].is_preg()) return false;
    if (!inst.defs[0].preg_val.is_gpr() || inst.defs[0].preg_val.code >= 31) return false;
    if (inst.defs[0].preg_val != inst.uses[0].preg_val || !inst.uses[1].is_imm_int()) return false;
    const int64_t v = inst.uses[1].imm_int;
    if (v < INT32_MIN || v > INT32_MAX) return false;
    reg = inst.defs[0].preg_val;
    signed_imm = (inst.opcode == LirOpcode::Sub) ? -v : v;
    return true;
}

} // namespace

bool run_arm64_lir_superopt(LirBlock& block, const SuperoptOptions& opts) {
    bool any_changed = false;
    bool progress = true;
    const uint32_t gpr_mask = (opts.lir_live_out_gpr_mask == 0xFFFFULL)
        ? 0xFFFFFFFFu
        : static_cast<uint32_t>(opts.lir_live_out_gpr_mask);

    while (progress) {
        progress = false;

        // 1. 2-instruction window collapse `[i, i + 1]` -> 1 instruction
        for (size_t i = 0; i + 1 < block.instructions.size(); ++i) {
            const LirInst* i0 = block.instructions[i].get();
            const LirInst* i1 = block.instructions[i + 1].get();
            if (!i0 || !i1 || is_protected_inst(*i0) || is_protected_inst(*i1) ||
                has_symbolic_operand(*i0) || has_symbolic_operand(*i1) ||
                !is_pure_arm64_alu_or_mov(i0->opcode) || !is_pure_arm64_alu_or_mov(i1->opcode)) continue;
            if (i0->defs.size() != 1 || i1->defs.size() != 1) continue;
            const LirOperand& d0 = i0->defs[0];
            const LirOperand& d1 = i1->defs[0];
            if (!d0.is_preg() || !d0.preg_val.is_gpr() || d0.preg_val.code >= 31 ||
                !d1.is_preg() || d0.preg_val != d1.preg_val) continue;

            std::vector<LirInst> candidates;
            // (a) Overwritten Mov/ALU where i1 is Mov/Mov32 not reading dst
            if ((i1->opcode == LirOpcode::Mov || i1->opcode == LirOpcode::Mov32) && i1->uses.size() == 1 &&
                !operand_reads_gpr(i1->uses[0], d0.preg_val)) {
                candidates.push_back(*i1);
            }
            // (b) Mov(r, imm1); ALU(r, r, imm2) -> Mov(r, imm1 op imm2)
            if (i0->opcode == LirOpcode::Mov && i0->uses.size() == 1 && i0->uses[0].is_imm_int() &&
                i1->uses.size() == 2 && i1->uses[0].is_preg() && i1->uses[0].preg_val == d0.preg_val &&
                i1->uses[1].is_imm_int() &&
                (i1->opcode == LirOpcode::Add || i1->opcode == LirOpcode::Sub ||
                 i1->opcode == LirOpcode::And || i1->opcode == LirOpcode::Or ||
                 i1->opcode == LirOpcode::Xor || i1->opcode == LirOpcode::Shl ||
                 i1->opcode == LirOpcode::Shr || i1->opcode == LirOpcode::Sar)) {
                LirInst folded = *i0;
                uint64_t r = eval_bin_imm64(i1->opcode,
                                            static_cast<uint64_t>(i0->uses[0].imm_int),
                                            static_cast<uint64_t>(i1->uses[1].imm_int));
                folded.uses[0] = LirOperand::imm(static_cast<int64_t>(r), 8);
                candidates.push_back(std::move(folded));
            }
            // (c) Add/Sub(r, r, imm1); Add/Sub(r, r, imm2) -> Add/Sub(r, r, imm1 +/- imm2)
            codegen::PReg r0, r1;
            int64_t s0 = 0, s1 = 0;
            if (is_add_or_sub_imm64(*i0, r0, s0) && is_add_or_sub_imm64(*i1, r1, s1) && r0 == r1) {
                int64_t sum = s0 + s1;
                if (sum >= -4095 && sum <= 4095 && sum != 0) {
                    LirInst folded(sum > 0 ? LirOpcode::Add : LirOpcode::Sub);
                    folded.add_def(LirOperand::preg(r0, 8));
                    folded.add_use(LirOperand::preg(r0, 8));
                    folded.add_use(LirOperand::imm(sum > 0 ? sum : -sum, 8));
                    candidates.push_back(std::move(folded));
                }
            }

            bool accepted = false;
            for (const LirInst& cand_inst : candidates) {
                LirBlock cand_block = clone_arm64_block(block);
                *cand_block.instructions[i] = cand_inst;
                cand_block.instructions.erase(cand_block.instructions.begin() + static_cast<std::ptrdiff_t>(i + 1));
                if (opts.stats) opts.stats->solver_queries++;
                auto eq = verify::verify_arm64_lir_block_equivalence(
                    block, cand_block, gpr_mask, verify::kArm64AllVecMask, opts.lir_check_flags);
                if (eq.status == verify::EquivStatus::ProvenEquivalent) {
                    block.instructions = std::move(cand_block.instructions);
                    if (opts.stats) opts.stats->lir_windows_optimized++;
                    any_changed = true;
                    progress = true;
                    accepted = true;
                    break;
                }
            }
            if (accepted) break;
        }
        if (progress) continue;

        // 2. Single-instruction identity elision (`[i]` -> empty)
        for (size_t i = 0; i < block.instructions.size(); ++i) {
            const LirInst* inst = block.instructions[i].get();
            if (!inst || is_protected_inst(*inst) || has_symbolic_operand(*inst) ||
                !is_pure_arm64_alu_or_mov(inst->opcode) || inst->defs.size() != 1) {
                continue;
            }
            const LirOperand& dst = inst->defs[0];
            if (!dst.is_preg() || !dst.preg_val.is_gpr() || dst.preg_val.code >= 31) continue;

            bool plausible = false;
            if (inst->opcode == LirOpcode::Mov && inst->uses.size() == 1 &&
                inst->uses[0].is_preg() && inst->uses[0].preg_val == dst.preg_val) {
                plausible = true;
            } else if ((inst->opcode == LirOpcode::Add || inst->opcode == LirOpcode::Sub ||
                        inst->opcode == LirOpcode::Or  || inst->opcode == LirOpcode::Xor ||
                        inst->opcode == LirOpcode::Shl || inst->opcode == LirOpcode::Shr ||
                        inst->opcode == LirOpcode::Sar) &&
                       inst->uses.size() == 2 && inst->uses[1].is_imm_int() && inst->uses[1].imm_int == 0) {
                plausible = true;
            } else if (inst->opcode == LirOpcode::Imul && inst->uses.size() == 2 &&
                       inst->uses[1].is_imm_int() && inst->uses[1].imm_int == 1) {
                plausible = true;
            } else if (inst->opcode == LirOpcode::And && inst->uses.size() == 2 &&
                       inst->uses[1].is_imm_int() && inst->uses[1].imm_int == -1) {
                plausible = true;
            }
            if (!plausible) continue;

            LirBlock cand_block = clone_arm64_block(block);
            cand_block.instructions.erase(cand_block.instructions.begin() + static_cast<std::ptrdiff_t>(i));
            if (opts.stats) opts.stats->solver_queries++;
            auto eq = verify::verify_arm64_lir_block_equivalence(
                block, cand_block, gpr_mask, verify::kArm64AllVecMask, opts.lir_check_flags);
            if (eq.status == verify::EquivStatus::ProvenEquivalent) {
                block.instructions = std::move(cand_block.instructions);
                if (opts.stats) opts.stats->lir_windows_optimized++;
                any_changed = true;
                progress = true;
                break;
            }
        }
    }
    return any_changed;
}

} // namespace brass
