#include <brass/codegen/linear_scan.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <vector>
#include <memory>
#include <algorithm>
#include <cassert>
#include <stdexcept>

namespace brass::codegen {

static bool is_xmm_opcode(LirOpcode op) {
    switch (op) {
        case LirOpcode::Movsd:
        case LirOpcode::Movss:
        case LirOpcode::Addsd:
        case LirOpcode::Subsd:
        case LirOpcode::Mulsd:
        case LirOpcode::Divsd:
        case LirOpcode::Sqrtsd:
        case LirOpcode::Addss:
        case LirOpcode::Subss:
        case LirOpcode::Mulss:
        case LirOpcode::Divss:
        case LirOpcode::Sqrtss:
        case LirOpcode::Ucomisd:
        case LirOpcode::Ucomiss:
        case LirOpcode::Xorpd:
        case LirOpcode::Xorps:
        case LirOpcode::Cvtsi2sd:
        case LirOpcode::Cvtsi2sd32:
        case LirOpcode::Cvtsi2ss:
        case LirOpcode::Cvtsi2ss32:
        case LirOpcode::Cvtsd2ss:
        case LirOpcode::Cvtss2sd:
        case LirOpcode::Floor32:
        case LirOpcode::Floor64:
        case LirOpcode::Ceil32:
        case LirOpcode::Ceil64:
        case LirOpcode::Round32:
        case LirOpcode::Round64:
        case LirOpcode::Fabs32:
        case LirOpcode::Fabs64:
        case LirOpcode::Minss:
        case LirOpcode::Minsd:
        case LirOpcode::Maxss:
        case LirOpcode::Maxsd:
        case LirOpcode::Movq_xg:
        case LirOpcode::Movaps:
        case LirOpcode::Movups:
        case LirOpcode::Movd_xg:
        case LirOpcode::Addps:
        case LirOpcode::Subps:
        case LirOpcode::Mulps:
        case LirOpcode::Divps:
        case LirOpcode::Minps:
        case LirOpcode::Maxps:
        case LirOpcode::Sqrtps:
        case LirOpcode::Addpd:
        case LirOpcode::Subpd:
        case LirOpcode::Mulpd:
        case LirOpcode::Divpd:
        case LirOpcode::Minpd:
        case LirOpcode::Maxpd:
        case LirOpcode::Sqrtpd:
        case LirOpcode::Paddd:
        case LirOpcode::Psubd:
        case LirOpcode::Pmulld:
        case LirOpcode::Pminsd:
        case LirOpcode::Pmaxsd:
        case LirOpcode::Paddq:
        case LirOpcode::Psubq:
        case LirOpcode::Pand:
        case LirOpcode::Por:
        case LirOpcode::Pxor:
        case LirOpcode::Pandn:
        case LirOpcode::Pcmpeqd:
        case LirOpcode::Pslld:
        case LirOpcode::Psllq:
        case LirOpcode::Shufps:
        case LirOpcode::Shufpd:
        case LirOpcode::Pshufd:
        case LirOpcode::Movddup:
        case LirOpcode::Pinsrd:
        case LirOpcode::Pinsrq:
        case LirOpcode::Insertps:
        case LirOpcode::Vfmadd213ss:
        case LirOpcode::Vfmadd231ss:
        case LirOpcode::Vfmadd213sd:
        case LirOpcode::Vfmadd231sd:
        case LirOpcode::Vfmadd213ps:
        case LirOpcode::Vfmadd231ps:
        case LirOpcode::Vfmadd213pd:
        case LirOpcode::Vfmadd231pd:
        case LirOpcode::Vbroadcastss:
        case LirOpcode::Vbroadcastsd:
            return true;
        default:
            return false;
    }
}

void LinearScanAllocator::rewrite_instructions() {
    using namespace brass::x64;

    auto resolve_operand = [this](const LirOperand& op) -> LirOperand {
        if (op.is_vreg()) {
            // A register operand the allocator gave no home would be
            // emitted as an arbitrary frame address; that is a bug in the
            // lowering or in liveness, never something to paper over.
            if (!op.vreg_val.is_valid() || op.vreg_val.id >= fn_.vreg_table.size()) {
                throw std::logic_error("register allocation: operand has no virtual register in " +
                                       std::string(fn_.name));
            }
            const VRegInfo& info = fn_.get_vreg_info(op.vreg_val);
            if (info.is_spilled) {
                return LirOperand::slot(info.assigned_spill_slot, op.size);
            } else if (info.assigned_preg.is_valid()) {
                return LirOperand::preg(info.assigned_preg, op.size);
            }
            throw std::logic_error("register allocation: v" + std::to_string(op.vreg_val.id) +
                                   " was given neither a register nor a spill slot in " + std::string(fn_.name));
        } else if (op.is_mem()) {
            LirMem mem = op.mem_val;
            if (mem.base_vreg.is_valid() && mem.base_vreg.id < fn_.vreg_table.size()) {
                const VRegInfo& b_info = fn_.get_vreg_info(mem.base_vreg);
                if (b_info.assigned_preg.is_valid()) {
                    mem.base_preg = b_info.assigned_preg;
                    mem.base_vreg = VReg{};
                }
            }
            if (mem.index_vreg.is_valid() && mem.index_vreg.id < fn_.vreg_table.size()) {
                const VRegInfo& i_info = fn_.get_vreg_info(mem.index_vreg);
                if (i_info.assigned_preg.is_valid()) {
                    mem.index_preg = i_info.assigned_preg;
                    mem.index_vreg = VReg{};
                }
            }
            if (cc_.target().is_x64()) {
                if (mem.index_preg.is_valid() && mem.index_preg == PReg::gpr(x64::GPR::R12) && mem.scale == Scale::One &&
                    mem.base_preg.is_valid() && mem.base_preg != PReg::gpr(x64::GPR::R12)) {
                    std::swap(mem.base_preg, mem.index_preg);
                }
            }
            return LirOperand::mem_custom(mem, op.size);
        }
        return op;
    };

    bool is_aarch64 = cc_.target().is_aarch64();
    PReg base_scratch_reg = is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X12) : PReg::gpr(brass::x64::GPR::R10);
    PReg idx_scratch_reg = is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X13) : PReg::gpr(brass::x64::GPR::R11);

    for (const auto& block : fn_.blocks) {
        std::vector<std::unique_ptr<LirInst>> rewritten;

        for (auto& inst : block->instructions) {
            bool base_reloaded = false;
            bool idx_reloaded = false;

            // First, check if any memory operands have spilled base or index registers
            for (auto* op_list : {&inst->defs, &inst->uses}) {
                for (auto& op : *op_list) {
                    if (op.is_mem()) {
                        if (op.mem_val.base_vreg.is_valid() && op.mem_val.base_vreg.id < fn_.vreg_table.size()) {
                            const VRegInfo& b_info = fn_.get_vreg_info(op.mem_val.base_vreg);
                            if (b_info.is_spilled) {
                                auto load_base = std::make_unique<LirInst>(LirOpcode::Mov);
                                load_base->add_def(LirOperand::preg(base_scratch_reg, 8));
                                load_base->add_use(LirOperand::slot(b_info.assigned_spill_slot, 8));
                                rewritten.push_back(std::move(load_base));

                                op.mem_val.base_preg = base_scratch_reg;
                                op.mem_val.base_vreg = VReg{};
                                base_reloaded = true;
                            }
                        }
                        if (op.mem_val.index_vreg.is_valid() && op.mem_val.index_vreg.id < fn_.vreg_table.size()) {
                            const VRegInfo& i_info = fn_.get_vreg_info(op.mem_val.index_vreg);
                            if (i_info.is_spilled) {
                                auto load_idx = std::make_unique<LirInst>(LirOpcode::Mov);
                                load_idx->add_def(LirOperand::preg(idx_scratch_reg, 8));
                                load_idx->add_use(LirOperand::slot(i_info.assigned_spill_slot, 8));
                                rewritten.push_back(std::move(load_idx));

                                op.mem_val.index_preg = idx_scratch_reg;
                                op.mem_val.index_vreg = VReg{};
                                idx_reloaded = true;
                            }
                        }
                    }
                }
            }

            // Track whether defs and uses were originally XMM operands
            bool orig_def_is_xmm = false;
            if (!inst->defs.empty()) {
                if (inst->defs[0].is_vreg() && inst->defs[0].vreg_val.is_valid() && inst->defs[0].vreg_val.id < fn_.vreg_table.size()) {
                    orig_def_is_xmm = fn_.get_vreg_info(inst->defs[0].vreg_val).vreg.is_xmm();
                } else if (inst->defs[0].is_preg()) {
                    orig_def_is_xmm = inst->defs[0].preg_val.is_xmm();
                }
            }

            std::vector<bool> orig_use_is_xmm(inst->uses.size(), false);
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                if (inst->uses[i].is_vreg() && inst->uses[i].vreg_val.is_valid() && inst->uses[i].vreg_val.id < fn_.vreg_table.size()) {
                    orig_use_is_xmm[i] = fn_.get_vreg_info(inst->uses[i].vreg_val).vreg.is_xmm();
                } else if (inst->uses[i].is_preg()) {
                    orig_use_is_xmm[i] = inst->uses[i].preg_val.is_xmm();
                }
            }

            // Rewrite defs and uses
            try {
                for (size_t i = 0; i < inst->defs.size(); ++i) {
                    inst->defs[i] = resolve_operand(inst->defs[i]);
                }
                for (size_t i = 0; i < inst->uses.size(); ++i) {
                    inst->uses[i] = resolve_operand(inst->uses[i]);
                }
            } catch (const std::logic_error& e) {
                throw std::logic_error(std::string(e.what()) + " (instruction `" + to_string(*inst) +
                                       "` in block " + block->name + ")");
            }

            if (inst->opcode == LirOpcode::Safepoint || inst->opcode == LirOpcode::ParallelCopy) {
                rewritten.push_back(std::move(inst));
                continue;
            }

            bool def_is_mem = !inst->defs.empty() && (inst->defs[0].is_mem() || inst->defs[0].is_spill_slot());
            bool use_is_mem = !inst->uses.empty() && (inst->uses[0].is_mem() || inst->uses[0].is_spill_slot());

            // Moves between memory / spill slots:
            if ((inst->opcode == LirOpcode::Mov || inst->opcode == LirOpcode::Mov32 ||
                 inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss ||
                 inst->opcode == LirOpcode::Movaps) &&
                def_is_mem && use_is_mem) {
                uint8_t sz = inst->uses[0].size;
                bool is_xmm = orig_def_is_xmm || orig_use_is_xmm[0] || (inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Movss ||
                               inst->opcode == LirOpcode::Movaps || inst->opcode == LirOpcode::Vmovaps ||
                               inst->opcode == LirOpcode::Vmovups || sz == 16 || sz == 32);
                PReg scratch = is_xmm ? (is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V29) : PReg::xmm(XMM::XMM14))
                                      : (is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X15) : PReg::gpr(GPR::R15));
                if (!is_xmm) {
                    PReg candidates[] = {
                        is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X15) : PReg::gpr(GPR::R15),
                        is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X13) : PReg::gpr(GPR::R11),
                        is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X12) : PReg::gpr(GPR::R10),
                    };
                    scratch = candidates[0];
                    for (auto c : candidates) {
                        bool conflict = false;
                        if (inst->defs[0].is_mem() && (inst->defs[0].mem_val.base_preg == c || inst->defs[0].mem_val.index_preg == c)) conflict = true;
                        if (inst->uses[0].is_mem() && (inst->uses[0].mem_val.base_preg == c || inst->uses[0].mem_val.index_preg == c)) conflict = true;
                        if (!conflict) {
                            scratch = c;
                            break;
                        }
                    }
                }

                LirOpcode op = is_xmm ? ((sz == 32) ? LirOpcode::Vmovups : ((sz == 16) ? LirOpcode::Movaps : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd)))
                                      : ((sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov);
                auto load_scratch = std::make_unique<LirInst>(op);
                load_scratch->add_def(LirOperand::preg(scratch, sz));
                load_scratch->add_use(inst->uses[0]);
                rewritten.push_back(std::move(load_scratch));

                // The 32-bit load zero-extended the value; a spill slot gets
                // all of it (see the spilled-def write-back below). A real
                // memory destination keeps the instruction's width.
                const bool widen = !is_xmm && sz == 4 && inst->defs[0].is_spill_slot();
                auto store_scratch = std::make_unique<LirInst>(widen ? LirOpcode::Mov : op);
                store_scratch->add_def(widen ? LirOperand::slot(inst->defs[0].spill_slot, 8) : inst->defs[0]);
                store_scratch->add_use(LirOperand::preg(scratch, widen ? uint8_t{8} : sz));
                rewritten.push_back(std::move(store_scratch));
                continue;
            }

            bool has_spill_def = !inst->defs.empty() && inst->defs[0].is_spill_slot();

            // Other instructions with spill def or spill uses
            LirOperand original_spill_def;
            PReg def_scratch;
            bool is_xmm_def = false;

            if (has_spill_def) {
                original_spill_def = inst->defs[0];
                uint8_t sz = original_spill_def.size;
                is_xmm_def = orig_def_is_xmm || is_xmm_opcode(inst->opcode) || sz == 16 || sz == 32;
                if (inst->is_call()) {
                    def_scratch = is_xmm_def ? (is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V0) : PReg::xmm(XMM::XMM0))
                                             : (is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X0) : PReg::gpr(GPR::RAX));
                } else {
                    def_scratch = is_xmm_def ? (is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V27) : PReg::xmm(XMM::XMM15))
                                             : (is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X14) : PReg::gpr(GPR::R14));
                }

                // If instruction reads from def (e.g. add dst, src), load initial value of def into scratch
                bool reads_def = false;
                for (size_t i = 0; i < inst->uses.size(); ++i) {
                    if (inst->uses[i].is_spill_slot() && inst->uses[i].spill_slot == original_spill_def.spill_slot) {
                        reads_def = true;
                        inst->uses[i] = LirOperand::preg(def_scratch, sz);
                    }
                }

                if (reads_def) {
                    LirOpcode load_op = is_xmm_def ? ((sz == 32) ? LirOpcode::Vmovups : ((sz == 16) ? LirOpcode::Movaps : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd)))
                                                   : ((sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov);
                    auto load_def = std::make_unique<LirInst>(load_op);
                    load_def->add_def(LirOperand::preg(def_scratch, sz));
                    load_def->add_use(original_spill_def);
                    rewritten.push_back(std::move(load_def));
                }

                inst->defs[0] = LirOperand::preg(def_scratch, sz);
            }

            // Handle any remaining spill uses with reserved scratch registers:
            // GPR: R10/R11/R15 (x64) or X12/X13/X15 (AArch64)
            // XMM: XMM12/XMM13/XMM14 (x64) or V26/V28/V29 (AArch64)
            std::vector<int32_t> orig_slot_indices(inst->uses.size(), -1);
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                if (inst->uses[i].is_spill_slot()) {
                    orig_slot_indices[i] = inst->uses[i].spill_slot;
                }
            }

            PReg gpr_scratches[3] = {
                is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X12) : PReg::gpr(GPR::R10),
                is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X13) : PReg::gpr(GPR::R11),
                is_aarch64 ? PReg::aarch64_gpr(brass::aarch64::GPR::X15) : PReg::gpr(GPR::R15)
            };
            PReg xmm_scratches[3] = {
                is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V26) : PReg::xmm(XMM::XMM12),
                is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V28) : PReg::xmm(XMM::XMM13),
                is_aarch64 ? PReg::aarch64_fpr(brass::aarch64::FPR::V29) : PReg::xmm(XMM::XMM14)
            };

            std::vector<PReg> busy_registers;
            if (base_reloaded) {
                busy_registers.push_back(base_scratch_reg);
            }
            if (idx_reloaded) {
                busy_registers.push_back(idx_scratch_reg);
            }
            if (has_spill_def) {
                busy_registers.push_back(def_scratch);
            }

            std::vector<PReg> assigned_use_scratches;

            for (size_t i = 0; i < inst->uses.size(); ++i) {
                if (inst->uses[i].is_spill_slot()) {
                    uint8_t sz = inst->uses[i].size;
                    bool is_xmm_use = (i < orig_use_is_xmm.size()) ? orig_use_is_xmm[i] : false;
                    if (!is_xmm_use) {
                        if (inst->opcode == LirOpcode::Pinsrd || inst->opcode == LirOpcode::Pinsrq) {
                            is_xmm_use = (i == 0);
                        } else if (inst->opcode == LirOpcode::Movd_xg || inst->opcode == LirOpcode::Movq_xg ||
                                   inst->opcode == LirOpcode::Cvtsi2sd || inst->opcode == LirOpcode::Cvtsi2sd32 ||
                                   inst->opcode == LirOpcode::Cvtsi2ss || inst->opcode == LirOpcode::Cvtsi2ss32) {
                            is_xmm_use = false;
                        } else if (inst->opcode == LirOpcode::Cvttsd2si || inst->opcode == LirOpcode::Cvttsd2si32 ||
                                   inst->opcode == LirOpcode::Cvttss2si || inst->opcode == LirOpcode::Cvttss2si32 ||
                                   inst->opcode == LirOpcode::Movq_gx || inst->opcode == LirOpcode::Movd_gx ||
                                   inst->opcode == LirOpcode::Pextrd || inst->opcode == LirOpcode::Pextrq ||
                                   inst->opcode == LirOpcode::Extractps) {
                            is_xmm_use = true;
                        } else {
                            is_xmm_use = is_xmm_opcode(inst->opcode) || sz == 16 || sz == 32;
                        }
                    }
                    int32_t slot = inst->uses[i].spill_slot;

                    // Check if this same slot was already loaded for a previous use
                    PReg use_scratch{};
                    bool already_loaded = false;
                    for (size_t j = 0; j < i; ++j) {
                        if (orig_slot_indices[j] == slot && inst->uses[j].is_preg()) {
                            use_scratch = inst->uses[j].preg_val;
                            already_loaded = true;
                            break;
                        }
                    }

                    if (!already_loaded) {
                        bool found = false;
                        if (is_xmm_use) {
                            for (PReg s : xmm_scratches) {
                                bool is_busy = false;
                                for (PReg b : busy_registers) {
                                    if (s == b) { is_busy = true; break; }
                                }
                                for (PReg a : assigned_use_scratches) {
                                    if (s == a) { is_busy = true; break; }
                                }
                                if (!is_busy) {
                                    use_scratch = s;
                                    found = true;
                                    break;
                                }
                            }
                        } else {
                            for (PReg s : gpr_scratches) {
                                bool is_busy = false;
                                for (PReg b : busy_registers) {
                                    if (s == b) { is_busy = true; break; }
                                }
                                for (PReg a : assigned_use_scratches) {
                                    if (s == a) { is_busy = true; break; }
                                }
                                if (!is_busy) {
                                    use_scratch = s;
                                    found = true;
                                    break;
                                }
                            }
                        }
                        if (!found) {
                            throw std::runtime_error("Insufficient scratch registers for spilled operands: scratch conflict or reuse");
                        }
                        for ([[maybe_unused]] PReg b : busy_registers) {
                            assert(use_scratch != b && "Scratch register clobbers a busy register (base, idx, or def)!");
                        }
                        for ([[maybe_unused]] PReg a : assigned_use_scratches) {
                            assert(use_scratch != a && "Scratch register reused within same instruction use list!");
                        }
                        assigned_use_scratches.push_back(use_scratch);

                        LirOpcode load_op = is_xmm_use ? ((sz == 32) ? LirOpcode::Vmovups : ((sz == 16) ? LirOpcode::Movaps : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd)))
                                                       : ((sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov);
                        auto load_use = std::make_unique<LirInst>(load_op);
                        load_use->add_def(LirOperand::preg(use_scratch, sz));
                        load_use->add_use(inst->uses[i]);
                        rewritten.push_back(std::move(load_use));
                    }

                    inst->uses[i] = LirOperand::preg(use_scratch, sz);
                }
            }

            rewritten.push_back(std::move(inst));

            // Write back to spill slot if def was spilled
            if (has_spill_def) {
                uint8_t sz = original_spill_def.size;
                // A 32-bit GPR operation zero-extends into the whole
                // register, and the value it defines may be 64 bits wide
                // (zext.i64 is a 32-bit move). Storing only the low half
                // would leave the slot's upper half stale; store all of it.
                if (!is_xmm_def && sz == 4) sz = 8;
                LirOpcode store_op = is_xmm_def ? ((sz == 32) ? LirOpcode::Vmovups : ((sz == 16) ? LirOpcode::Movaps : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd)))
                                                : ((sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov);
                auto store_back = std::make_unique<LirInst>(store_op);
                store_back->add_def(LirOperand::slot(original_spill_def.spill_slot, sz));
                store_back->add_use(LirOperand::preg(def_scratch, sz));
                rewritten.push_back(std::move(store_back));
            }
        }

        block->instructions = std::move(rewritten);
    }
}

} // namespace brass::codegen
