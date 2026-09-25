#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/mir/module.hpp>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <cassert>

namespace brass::aarch64 {

using namespace brass::codegen;

AArch64ISel::AArch64ISel()
    : target_(Target::aarch64_linux()), cc_(CallingConvention::for_target(Target::aarch64_linux())) {}

AArch64ISel::AArch64ISel(const Target& target)
    : target_(target), cc_(CallingConvention::for_target(target)) {}

AArch64ISel::AArch64ISel(const Target& target, const CallingConvention& cc)
    : target_(target), cc_(cc) {}

std::unique_ptr<LirFunction> AArch64ISel::lower(const Function& mir_fn) {
    auto lir = std::make_unique<LirFunction>();
    lir_fn_ = lir.get();
    mir_fn_ = &mir_fn;
    val_to_vreg_.clear();
    val_to_vreg_pair_.clear();
    elidable_insts_.clear();

    lir_fn_->name = std::string(mir_fn.name());
    lir_fn_->return_type = mir_fn.return_type();
    lir_fn_->calling_conv = cc_;
    if (mir_fn.parent() && mir_fn.parent()->pinned_tls_register()) {
        lir_fn_->reserved_gprs |= reg_mask(kPinnedTlsGpr);
    }

    const_cast<Function&>(mir_fn).rebuild_cfg_predecessors();

    analyze_function(mir_fn);

    for (const auto* bb : mir_fn.blocks()) {
        lir_fn_->create_block_with_id(bb->id(), std::string(bb->name()));
    }

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* param : bb->params()) {
            if (param->type().is_v256()) {
                get_or_alloc_vreg_pair(param);
            } else {
                get_or_alloc_vreg(param);
            }
        }
    }

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (!inst->produces_value()) continue;
            if (skipped_insts_.count(inst)) continue;
            if (inst->type().is_v256()) {
                get_or_alloc_vreg_pair(inst->result());
            } else {
                get_or_alloc_vreg(inst->result());
            }
        }
    }

    lower_entry_parameters(mir_fn);

    for (const auto* bb : mir_fn.blocks()) {
        lower_block(*bb);
    }

    // Connect the CFG (as the x64 selector does). The edge blocks created
    // while lowering (br_if / switch trampolines, guard deopt exits) were
    // linked then. A MIR edge whose block arguments went through such a
    // trampoline runs only through it: a direct edge too would make the
    // target's parameters look live, undefined, on every path into this
    // block, and a gcref parameter would be reported to the GC with whatever
    // its home holds.
    std::unordered_set<uint32_t> mir_block_ids;
    for (const auto* bb : mir_fn.blocks()) mir_block_ids.insert(bb->id());
    for (size_t i = 0; i < mir_fn.blocks().size(); ++i) {
        const auto* mir_bb = mir_fn.blocks()[i];
        auto* lir_bb = lir_fn_->blocks[i].get();

        auto jumps_directly_to = [&](uint32_t id) {
            for (const auto& li : lir_bb->instructions) {
                if (li->opcode != LirOpcode::Jmp && li->opcode != LirOpcode::Jcc) continue;
                for (const auto& u : li->uses) {
                    if (u.is_label() && u.label_id == id) return true;
                }
            }
            return false;
        };
        auto reached_by_trampoline = [&](const LirBlock* target) {
            for (const LirBlock* s : lir_bb->successors) {
                if (mir_block_ids.count(s->id)) continue;
                if (std::find(s->successors.begin(), s->successors.end(), target) != s->successors.end()) return true;
            }
            return false;
        };

        for (const auto* succ : mir_bb->successors()) {
            LirBlock* succ_lir = lir_fn_->get_block_by_id(succ->id());
            if (!succ_lir) continue;
            if (reached_by_trampoline(succ_lir) && !jumps_directly_to(succ_lir->id)) continue;
            link_blocks(*lir_bb, *succ_lir);
        }
    }

    auto* lir_entry = lir_fn_->entry_block();
    for (const auto& rp : mir_fn.resume_points()) {
        if (rp.second) {
            lir_fn_->resume_entries.push_back({rp.first, rp.second->id()});
            auto* target_lir = lir_fn_->get_block_by_id(rp.second->id());
            if (lir_entry && target_lir) {
                lir_entry->successors.push_back(target_lir);
                target_lir->predecessors.push_back(lir_entry);
            }
        }
    }

    eliminate_dead_materializations();
    return lir;
}

VReg AArch64ISel::get_or_alloc_vreg(const Value* val) {
    if (!val) return VReg{};
    auto it = val_to_vreg_.find(val);
    if (it != val_to_vreg_.end()) {
        return it->second;
    }

    Type t = val->type();
    if (t.is_v256()) {
        return get_or_alloc_vreg_pair(val).lo;
    }

    RegClass rc = (t.is_float() || t.is_vector()) ? RegClass::XMM : RegClass::GPR;
    uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
    if (sz == 0) sz = 8;
    bool is_gc = t.is_gcref();

    VReg v = lir_fn_->allocate_vreg(rc, sz, is_gc);
    val_to_vreg_[val] = v;
    return v;
}

VReg AArch64ISel::get_vreg(const Value* val) const {
    if (!val) return VReg{};
    auto it = val_to_vreg_.find(val);
    if (it != val_to_vreg_.end()) {
        return it->second;
    }
    return VReg{};
}

AArch64ISel::VRegPair AArch64ISel::get_or_alloc_vreg_pair(const Value* val) {
    if (!val) return VRegPair{};
    auto it = val_to_vreg_pair_.find(val);
    if (it != val_to_vreg_pair_.end()) {
        return it->second;
    }

    VReg lo = lir_fn_->allocate_vreg(RegClass::XMM, 16, false);
    VReg hi = lir_fn_->allocate_vreg(RegClass::XMM, 16, false);
    VRegPair pair{lo, hi};
    val_to_vreg_pair_[val] = pair;
    val_to_vreg_[val] = lo;
    return pair;
}

AArch64ISel::VRegPair AArch64ISel::get_vreg_pair(const Value* val) const {
    if (!val) return VRegPair{};
    auto it = val_to_vreg_pair_.find(val);
    if (it != val_to_vreg_pair_.end()) {
        return it->second;
    }
    return VRegPair{};
}

void AArch64ISel::lower_entry_parameters(const Function& mir_fn) {
    const auto* entry = mir_fn.entry_block();
    if (!entry || entry->params().empty()) return;

    auto* lir_entry = lir_fn_->entry_block();
    if (!lir_entry) return;

    auto pcopy = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
    size_t gpr_idx = 0;
    size_t fpr_idx = 0;
    size_t stack_bytes = 0;
    bool is_apple = (cc_.kind() == CallingConvKind::AppleAAPCS64);

    for (size_t i = 0; i < entry->param_count(); ++i) {
        const auto* param = entry->param(i);
        Type t = param->type();

        if (t.is_v256()) {
            VRegPair pair = get_vreg_pair(param);
            if (fpr_idx + 1 < 8) {
                FPR r_lo = static_cast<FPR>(fpr_idx++);
                FPR r_hi = static_cast<FPR>(fpr_idx++);
                pcopy->add_def(LirOperand::vreg(pair.lo, 16));
                pcopy->add_use(LirOperand::preg_aarch64_fpr(r_lo, 16), FixedConstraint::aarch64_fpr(r_lo));
                pcopy->add_def(LirOperand::vreg(pair.hi, 16));
                pcopy->add_use(LirOperand::preg_aarch64_fpr(r_hi, 16), FixedConstraint::aarch64_fpr(r_hi));
            } else if (fpr_idx < 8) {
                FPR r_lo = static_cast<FPR>(fpr_idx++);
                pcopy->add_def(LirOperand::vreg(pair.lo, 16));
                pcopy->add_use(LirOperand::preg_aarch64_fpr(r_lo, 16), FixedConstraint::aarch64_fpr(r_lo));

                size_t align = 16;
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t caller_offset = static_cast<int32_t>(stack_bytes);
                stack_bytes += 16;
                pcopy->add_def(LirOperand::vreg(pair.hi, 16));
                pcopy->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::FP), -1 - caller_offset, 16));
            } else {
                size_t align = 16;
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t caller_offset = static_cast<int32_t>(stack_bytes);
                stack_bytes += 32;

                pcopy->add_def(LirOperand::vreg(pair.lo, 16));
                pcopy->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::FP), -1 - caller_offset, 16));
                pcopy->add_def(LirOperand::vreg(pair.hi, 16));
                pcopy->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::FP), -1 - (caller_offset + 16), 16));
            }
            continue;
        }

        VReg param_vreg = get_vreg(param);
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;

        bool is_fpr = (t.is_float() || t.is_vector());
        if (is_fpr) {
            if (fpr_idx < 8) {
                FPR freg = static_cast<FPR>(fpr_idx++);
                pcopy->add_def(LirOperand::vreg(param_vreg, sz));
                pcopy->add_use(LirOperand::preg_aarch64_fpr(freg, sz), FixedConstraint::aarch64_fpr(freg));
            } else {
                size_t align = is_apple ? ((sz >= 16) ? 16 : (sz >= 8 ? 8 : (sz >= 4 ? 4 : (sz >= 2 ? 2 : 1))))
                                        : ((sz >= 16) ? 16 : 8);
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t caller_offset = static_cast<int32_t>(stack_bytes);
                stack_bytes += is_apple ? sz : ((sz >= 16) ? 16 : 8);

                pcopy->add_def(LirOperand::vreg(param_vreg, sz));
                pcopy->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::FP), -1 - caller_offset, sz));
            }
        } else {
            if (gpr_idx < 8) {
                GPR greg = static_cast<GPR>(gpr_idx++);
                pcopy->add_def(LirOperand::vreg(param_vreg, sz));
                pcopy->add_use(LirOperand::preg_aarch64_gpr(greg, sz), FixedConstraint::aarch64_gpr(greg));
            } else {
                size_t align = is_apple ? ((sz >= 16) ? 16 : (sz >= 8 ? 8 : (sz >= 4 ? 4 : (sz >= 2 ? 2 : 1))))
                                        : ((sz >= 16) ? 16 : 8);
                stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
                int32_t caller_offset = static_cast<int32_t>(stack_bytes);
                stack_bytes += is_apple ? sz : ((sz >= 16) ? 16 : 8);

                pcopy->add_def(LirOperand::vreg(param_vreg, sz));
                pcopy->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::FP), -1 - caller_offset, sz));
            }
        }
    }
    lir_entry->append_inst(std::move(pcopy));
    // No resume dispatch on entry: a resume block is reached only through
    // the resume table (docs/mir_reference.md, "Guard exits"), never by
    // matching an ordinary argument against a resume id.
}

void AArch64ISel::lower_block(const BasicBlock& bb) {
    auto* lir_bb = lir_fn_->get_block_by_id(bb.id());
    if (!lir_bb) return;

    for (const auto* inst : bb) {
        if (!skipped_insts_.count(inst)) {
            size_t before_count = lir_bb->instructions.size();
            if (!lower_narrow_width_op(*inst, *lir_bb)) {
                const auto widened = widen_narrow_operands(*inst, *lir_bb);
                lower_instruction(*inst, *lir_bb);
                restore_narrow_operands(widened);
            }
            const bool elidable = is_elidable_materialization(*inst);
            for (size_t i = before_count; i < lir_bb->instructions.size(); ++i) {
                if (lir_bb->instructions[i]) {
                    if (elidable) elidable_insts_.insert(lir_bb->instructions[i].get());
                    if (!lir_bb->instructions[i]->loc.is_valid() && inst->loc().is_valid()) {
                        lir_bb->instructions[i]->loc = inst->loc();
                    }
                    if (!lir_bb->instructions[i]->mir_origin) {
                        lir_bb->instructions[i]->mir_origin = inst;
                    }
                }
            }
        }
    }
}

void AArch64ISel::lower_instruction(const Instruction& inst, LirBlock& lir_bb) {
    if (skipped_insts_.count(&inst)) {
        return;
    }

    switch (inst.opcode()) {
        case Opcode::vadd:
        case Opcode::vsub:
        case Opcode::vmul:
        case Opcode::vdiv:
        case Opcode::vneg:
        case Opcode::vmin:
        case Opcode::vmax:
        case Opcode::vsqrt:
        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor:
        case Opcode::vnot:
        case Opcode::vload:
        case Opcode::vstore:
        case Opcode::vbroadcast:
        case Opcode::vextract_lane:
        case Opcode::vinsert_lane:
        case Opcode::vshuffle:
        case Opcode::vzero:
        case Opcode::vfma:
        case Opcode::fma_f32:
        case Opcode::fma_f64:
            lower_vector_instruction(inst, lir_bb);
            break;

        case Opcode::iconst_i32: {
            VReg dst = get_vreg(inst.result());
            int32_t val = inst.imm_i32();
            if (val == 0) {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Xor32);
                lir_inst->add_def(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::vreg(dst, 4));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov32);
                lir_inst->add_def(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::imm(val, 4));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            }
            break;
        }

        case Opcode::iconst_i64: {
            VReg dst = get_vreg(inst.result());
            int64_t val = inst.imm_i64();
            uint8_t sz = dst.size;
            if (val == 0) {
                auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
                lir_inst->add_def(LirOperand::vreg(dst, sz));
                lir_inst->add_use(LirOperand::vreg(dst, sz));
                lir_inst->add_use(LirOperand::vreg(dst, sz));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else if (sz == 4) {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov32);
                lir_inst->add_def(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::imm(val, 4));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else {
                if (val >= INT32_MIN && val <= INT32_MAX) {
                    auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov);
                    lir_inst->add_def(LirOperand::vreg(dst, 8));
                    lir_inst->add_use(LirOperand::imm(val, 8));
                    lir_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(lir_inst));
                } else {
                    auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movabs);
                    lir_inst->add_def(LirOperand::vreg(dst, 8));
                    lir_inst->add_use(LirOperand::imm(val, 8));
                    lir_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(lir_inst));
                }
            }
            break;
        }

        case Opcode::patchable_const_i32: {
            VReg dst = get_vreg(inst.result());
            int32_t val = inst.imm_i32();
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov32);
            lir_inst->is_patchable = true;
            lir_inst->patch_symbol = std::string(inst.symbol());
            lir_inst->add_def(LirOperand::vreg(dst, 4));
            lir_inst->add_use(LirOperand::imm(val, 4));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::patchable_const_i64: {
            VReg dst = get_vreg(inst.result());
            int64_t val = inst.imm_i64();
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movabs);
            lir_inst->is_patchable = true;
            lir_inst->patch_symbol = std::string(inst.symbol());
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::imm(val, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::func_addr: {
            VReg dst = get_vreg(inst.result());
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movabs);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::symbol(std::string(inst.symbol())));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::fconst_f64: {
            VReg dst = get_vreg(inst.result());
            uint8_t sz = (inst.type() == Type::f32()) ? 4 : 8;
            if (sz == 4) {
                float fval = static_cast<float>(inst.imm_f64());
                uint32_t raw_bits = 0;
                std::memcpy(&raw_bits, &fval, sizeof(float));
                if (raw_bits == 0) {
                    auto lir_inst = std::make_unique<LirInst>(LirOpcode::Xorps);
                    lir_inst->add_def(LirOperand::vreg(dst, 4));
                    lir_inst->add_use(LirOperand::vreg(dst, 4));
                    lir_inst->add_use(LirOperand::vreg(dst, 4));
                    lir_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(lir_inst));
                } else {
                    VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, 4);
                    auto m32 = std::make_unique<LirInst>(LirOpcode::Mov32);
                    m32->add_def(LirOperand::vreg(tmp, 4));
                    m32->add_use(LirOperand::imm(static_cast<int64_t>(raw_bits), 4));
                    lir_bb.append_inst(std::move(m32));

                    auto md = std::make_unique<LirInst>(LirOpcode::Movd_xg);
                    md->add_def(LirOperand::vreg(dst, 4));
                    md->add_use(LirOperand::vreg(tmp, 4));
                    md->mir_origin = &inst;
                    lir_bb.append_inst(std::move(md));
                }
            } else {
                double val = inst.imm_f64();
                uint64_t raw_bits = 0;
                std::memcpy(&raw_bits, &val, sizeof(double));
                if (raw_bits == 0) {
                    auto lir_inst = std::make_unique<LirInst>(LirOpcode::Xorpd);
                    lir_inst->add_def(LirOperand::vreg(dst, 8));
                    lir_inst->add_use(LirOperand::vreg(dst, 8));
                    lir_inst->add_use(LirOperand::vreg(dst, 8));
                    lir_inst->mir_origin = &inst;
                    lir_bb.append_inst(std::move(lir_inst));
                } else {
                    VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, 8);

                    auto mabs = std::make_unique<LirInst>(LirOpcode::Movabs);
                    mabs->add_def(LirOperand::vreg(tmp, 8));
                    mabs->add_use(LirOperand::imm(static_cast<int64_t>(raw_bits), 8));
                    lir_bb.append_inst(std::move(mabs));

                    auto mq = std::make_unique<LirInst>(LirOpcode::Movq_xg);
                    mq->add_def(LirOperand::vreg(dst, 8));
                    mq->add_use(LirOperand::vreg(tmp, 8));
                    mq->mir_origin = &inst;
                    lir_bb.append_inst(std::move(mq));
                }
            }
            break;
        }

        case Opcode::sext_i64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            if (inst.operand(0)->type() == Type::i8()) {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movsx8);
                lir_inst->add_def(LirOperand::vreg(dst, 8));
                lir_inst->add_use(LirOperand::vreg(src, 1));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else if (inst.operand(0)->type() == Type::i16()) {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movsx16);
                lir_inst->add_def(LirOperand::vreg(dst, 8));
                lir_inst->add_use(LirOperand::vreg(src, 2));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movsxd);
                lir_inst->add_def(LirOperand::vreg(dst, 8));
                lir_inst->add_use(LirOperand::vreg(src, 4));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            }
            break;
        }

        case Opcode::zext_i64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            if (inst.operand(0)->type() == Type::i8()) {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movzx8);
                lir_inst->add_def(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::vreg(src, 1));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else if (inst.operand(0)->type() == Type::i16()) {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movzx16);
                lir_inst->add_def(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::vreg(src, 2));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            } else {
                auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov32);
                lir_inst->add_def(LirOperand::vreg(dst, 4));
                lir_inst->add_use(LirOperand::vreg(src, 4));
                lir_inst->mir_origin = &inst;
                lir_bb.append_inst(std::move(lir_inst));
            }
            break;
        }

        case Opcode::trunc_i32: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov32);
            lir_inst->add_def(LirOperand::vreg(dst, 4));
            lir_inst->add_use(LirOperand::vreg(src, 4));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::trunc_i8: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Mov);
            lir_inst->add_def(LirOperand::vreg(dst, 1));
            lir_inst->add_use(LirOperand::vreg(src, 1));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::sqrt_f32:
        case Opcode::sqrt_f64:
        case Opcode::floor_f32:
        case Opcode::floor_f64:
        case Opcode::ceil_f32:
        case Opcode::ceil_f64:
        case Opcode::round_f32:
        case Opcode::round_f64:
        case Opcode::fabs_f32:
        case Opcode::fabs_f64:
        case Opcode::fmin_f32:
        case Opcode::fmin_f64:
        case Opcode::fmax_f32:
        case Opcode::fmax_f64:
        case Opcode::sitofp_f32_i32:
        case Opcode::sitofp_f32_i64:
        case Opcode::sitofp_f64_i32:
        case Opcode::sitofp_f64_i64:
        case Opcode::fptosi_i32_f32:
        case Opcode::fptosi_i64_f32:
        case Opcode::fptosi_i32:
        case Opcode::fptosi_i64:
        case Opcode::fptrunc_f32_f64:
        case Opcode::fpext_f64_f32:
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64:
            lower_fp_instruction(inst, lir_bb);
            break;

        case Opcode::add:
            lower_binary_alu(inst, lir_bb, LirOpcode::Add32, LirOpcode::Add, LirOpcode::Addsd, LirOpcode::Addss);
            break;
        case Opcode::sub:
            lower_binary_alu(inst, lir_bb, LirOpcode::Sub32, LirOpcode::Sub, LirOpcode::Subsd, LirOpcode::Subss);
            break;
        case Opcode::mul:
            lower_binary_alu(inst, lir_bb, LirOpcode::Imul32, LirOpcode::Imul, LirOpcode::Mulsd, LirOpcode::Mulss);
            break;
        case Opcode::sdiv:
            if (inst.type().is_float()) {
                lower_binary_alu(inst, lir_bb, LirOpcode::Nop, LirOpcode::Nop, LirOpcode::Divsd, LirOpcode::Divss);
            } else {
                lower_div_mod(inst, lir_bb, true, false);
            }
            break;
        case Opcode::udiv:
            lower_div_mod(inst, lir_bb, false, false);
            break;
        case Opcode::smod:
            lower_div_mod(inst, lir_bb, true, true);
            break;
        case Opcode::umod:
            lower_div_mod(inst, lir_bb, false, true);
            break;
        case Opcode::neg: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint8_t sz = dst.size;
            LirOpcode op = dst.is_xmm()
                ? (sz == 4 ? LirOpcode::Fneg32 : LirOpcode::Fneg)
                : (sz == 4 ? LirOpcode::Neg32 : LirOpcode::Neg);
            auto neg_inst = std::make_unique<LirInst>(op);
            neg_inst->add_def(LirOperand::vreg(dst, sz));
            neg_inst->add_use(LirOperand::vreg(src, sz));
            neg_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(neg_inst));
            break;
        }

        case Opcode::and_:
            lower_binary_alu(inst, lir_bb, LirOpcode::And32, LirOpcode::And, LirOpcode::Nop, LirOpcode::Nop);
            break;
        case Opcode::or_:
            lower_binary_alu(inst, lir_bb, LirOpcode::Or32, LirOpcode::Or, LirOpcode::Nop, LirOpcode::Nop);
            break;
        case Opcode::xor_:
            lower_binary_alu(inst, lir_bb, LirOpcode::Xor32, LirOpcode::Xor, LirOpcode::Xorpd, LirOpcode::Xorps);
            break;

        case Opcode::not_: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint8_t sz = dst.size;
            auto mov_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_inst->add_def(LirOperand::vreg(dst, sz));
            mov_inst->add_use(LirOperand::vreg(src, sz));
            lir_bb.append_inst(std::move(mov_inst));

            auto not_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Not32 : LirOpcode::Not);
            not_inst->add_def(LirOperand::vreg(dst, sz));
            not_inst->add_use(LirOperand::vreg(dst, sz));
            not_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(not_inst));
            break;
        }

        case Opcode::shl:
            lower_shift(inst, lir_bb, LirOpcode::Shl32, LirOpcode::Shl);
            break;
        case Opcode::lshr:
            lower_shift(inst, lir_bb, LirOpcode::Shr32, LirOpcode::Shr);
            break;
        case Opcode::ashr:
            lower_shift(inst, lir_bb, LirOpcode::Sar32, LirOpcode::Sar);
            break;

        case Opcode::clz: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint8_t sz = dst.size;
            auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Lzcnt32 : LirOpcode::Lzcnt);
            lir_inst->add_def(LirOperand::vreg(dst, sz));
            lir_inst->add_use(LirOperand::vreg(src, sz));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::ctz: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint8_t sz = dst.size;
            auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Tzcnt32 : LirOpcode::Tzcnt);
            lir_inst->add_def(LirOperand::vreg(dst, sz));
            lir_inst->add_use(LirOperand::vreg(src, sz));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::popcnt: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            uint8_t sz = dst.size;
            auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Popcnt32 : LirOpcode::Popcnt);
            lir_inst->add_def(LirOperand::vreg(dst, sz));
            lir_inst->add_use(LirOperand::vreg(src, sz));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::eq:
            lower_comparison(inst, lir_bb, x64::Condition::E, x64::Condition::E);
            break;
        case Opcode::ne:
            lower_comparison(inst, lir_bb, x64::Condition::NE, x64::Condition::NE);
            break;
        case Opcode::slt:
            lower_comparison(inst, lir_bb, x64::Condition::L, x64::Condition::B);
            break;
        case Opcode::ult:
            lower_comparison(inst, lir_bb, x64::Condition::B, x64::Condition::B);
            break;
        case Opcode::sle:
            lower_comparison(inst, lir_bb, x64::Condition::LE, x64::Condition::BE);
            break;
        case Opcode::ule:
            lower_comparison(inst, lir_bb, x64::Condition::BE, x64::Condition::BE);
            break;
        case Opcode::sgt:
            lower_comparison(inst, lir_bb, x64::Condition::G, x64::Condition::A);
            break;
        case Opcode::ugt:
            lower_comparison(inst, lir_bb, x64::Condition::A, x64::Condition::A);
            break;
        case Opcode::sge:
            lower_comparison(inst, lir_bb, x64::Condition::GE, x64::Condition::AE);
            break;
        case Opcode::uge:
            lower_comparison(inst, lir_bb, x64::Condition::AE, x64::Condition::AE);
            break;

        case Opcode::sadd_overflow:
        case Opcode::ssub_overflow:
        case Opcode::smul_overflow:
        case Opcode::uadd_overflow:
        case Opcode::usub_overflow:
        case Opcode::umul_overflow:
            lower_overflow_check(inst, lir_bb);
            break;

        case Opcode::select:
            lower_select(inst, lir_bb);
            break;

        case Opcode::alloca_:
            lower_alloca(inst, lir_bb);
            break;

        case Opcode::pinned_tls_read:
            lower_pinned_tls_read(inst, lir_bb);
            break;
        case Opcode::pinned_tls_write:
            lower_pinned_tls_write(inst, lir_bb);
            break;
        case Opcode::read_sp:
            lower_read_sp(inst, lir_bb);
            break;

        case Opcode::load:
            lower_load(inst, lir_bb);
            break;
        case Opcode::store:
            lower_store(inst, lir_bb);
            break;
        case Opcode::load_indexed:
            lower_load_indexed(inst, lir_bb);
            break;
        case Opcode::store_indexed:
            lower_store_indexed(inst, lir_bb);
            break;
        case Opcode::write_barrier:
            lower_write_barrier(inst, lir_bb);
            break;

        case Opcode::call:
        case Opcode::call_indirect:
        case Opcode::patchable_call:
            lower_call(inst, lir_bb);
            break;
        case Opcode::invoke:
            lower_invoke(inst, lir_bb);
            break;
        case Opcode::throw_:
            lower_throw(inst, lir_bb);
            break;
        case Opcode::resume:
            lower_resume(inst, lir_bb);
            break;
        case Opcode::landing_pad:
            lower_landing_pad(inst, lir_bb);
            break;

        case Opcode::coro_create:
        case Opcode::coro_suspend:
        case Opcode::coro_resume:
        case Opcode::coro_destroy:
            lower_coro(inst, lir_bb);
            break;

        case Opcode::safepoint:
            lower_safepoint(inst, lir_bb);
            break;
        case Opcode::guard:
            lower_guard(inst, lir_bb);
            break;
        case Opcode::resume_point:
            break;

        case Opcode::br:
            lower_branch(inst, lir_bb);
            break;
        case Opcode::br_if:
            lower_branch_if(inst, lir_bb);
            break;
        case Opcode::switch_:
            lower_switch(inst, lir_bb);
            break;
        case Opcode::ret:
            lower_return(inst, lir_bb);
            break;

        case Opcode::unreachable: {
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Trap);
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
        default:
            throw_unsupported("aarch64 isel", opcode_name(inst.opcode()));
    }
}



std::unique_ptr<LirFunction> lower_to_aarch64_lir(
    const Function& mir_fn,
    const Target& target,
    const CallingConvention& cc
) {
    AArch64ISel isel(target, cc);
    return isel.lower(mir_fn);
}

} // namespace brass::aarch64
