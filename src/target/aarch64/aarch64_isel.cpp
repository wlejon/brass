#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/osr.hpp>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace brass::aarch64 {

using namespace brass::codegen;

AArch64ISel::AArch64ISel()
    : target_(Target::aarch64_linux()), cc_(CallingConvention::for_target(Target::aarch64_linux())) {}

AArch64ISel::AArch64ISel(const Target& target)
    : target_(target), cc_(CallingConvention::for_target(target)) {}

AArch64ISel::AArch64ISel(const Target& target, const CallingConvention& cc)
    : target_(target), cc_(cc) {}

AArch64ISel::ImmIntInfo AArch64ISel::get_imm_int_info(const Value* val) const {
    if (!val) return {};
    if (val->is_instruction()) {
        const Instruction* def = val->defining_instruction();
        if (def) {
            if (def->opcode() == Opcode::iconst_i32) {
                return {true, static_cast<int64_t>(def->imm_i32()), true, def};
            }
            if (def->opcode() == Opcode::iconst_i64) {
                int64_t v = def->imm_i64();
                bool fits = (v >= INT32_MIN && v <= INT32_MAX);
                return {true, v, fits, def};
            }
        }
    }
    return {};
}

bool AArch64ISel::is_value_dead_after(
    const Function& mir_fn,
    const BasicBlock& bb,
    const Instruction* inst,
    const Value* val
) const {
    (void)mir_fn;
    if (!val) return false;
    auto it = use_count_.find(val);
    if (it == use_count_.end() || it->second == 0) return true;
    uint32_t total_uses = it->second;

    uint32_t uses_seen = 0;
    for (const auto* cur : bb) {
        for (const auto* op : cur->operands()) {
            if (op == val) uses_seen++;
        }
        if (cur == inst) break;
    }

    return (uses_seen == total_uses);
}

void AArch64ISel::analyze_function(const Function& mir_fn) {
    use_count_.clear();
    skipped_insts_.clear();

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            for (const auto* op : inst->operands()) {
                if (op) use_count_[op]++;
            }
            for (const auto* arg : inst->branch_target().args) {
                if (arg) use_count_[arg]++;
            }
            for (const auto* arg : inst->true_target().args) {
                if (arg) use_count_[arg]++;
            }
            for (const auto* arg : inst->false_target().args) {
                if (arg) use_count_[arg]++;
            }
            for (const auto* sv : inst->state_map()) {
                if (sv) use_count_[sv]++;
            }
        }
    }

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (inst->opcode() == Opcode::br_if || inst->opcode() == Opcode::guard || inst->opcode() == Opcode::select) {
                const Value* cond = inst->operand(0);
                if (cond && cond->is_instruction()) {
                    const Instruction* def_inst = cond->defining_instruction();
                    if (def_inst && def_inst->parent() == bb && is_comparison(def_inst->opcode())) {
                        if (use_count_[cond] == 1) {
                            skipped_insts_.insert(def_inst);
                            if (def_inst->opcode() == Opcode::eq || def_inst->opcode() == Opcode::ne) {
                                ImmIntInfo c0 = get_imm_int_info(def_inst->operand(0));
                                ImmIntInfo c1 = get_imm_int_info(def_inst->operand(1));
                                const Value* and_val = (c1.is_imm && c1.val == 0) ? def_inst->operand(0) : ((c0.is_imm && c0.val == 0) ? def_inst->operand(1) : nullptr);
                                if (and_val && and_val->is_instruction()) {
                                    const Instruction* and_inst = and_val->defining_instruction();
                                    if (and_inst && and_inst->parent() == bb && and_inst->opcode() == Opcode::and_ && use_count_[and_val] == 1) {
                                        skipped_insts_.insert(and_inst);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::unordered_map<const Value*, uint32_t> folded_uses;

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (skipped_insts_.count(inst)) continue;

            if (inst->opcode() == Opcode::load) {
                MemFold mf = match_address(inst->operand(0), inst->offset());
                for (const auto* fi : mf.folded_instructions) {
                    if (fi && fi->result()) folded_uses[fi->result()]++;
                }
                continue;
            }

            if (inst->opcode() == Opcode::store) {
                MemFold mf = match_address(inst->operand(0), inst->offset());
                for (const auto* fi : mf.folded_instructions) {
                    if (fi && fi->result()) folded_uses[fi->result()]++;
                }
                const Value* src = inst->operand(1);
                ImmIntInfo imm_src = get_imm_int_info(src);
                if (imm_src.is_imm && imm_src.fits_i32) {
                    folded_uses[src]++;
                }
                continue;
            }

            if (inst->opcode() == Opcode::load_indexed) {
                MemFold mf = match_indexed_address(inst->operand(0), inst->operand(1), x64::scale_from_int(inst->scale()), inst->offset());
                for (const auto* fi : mf.folded_instructions) {
                    if (fi && fi->result()) folded_uses[fi->result()]++;
                }
                continue;
            }

            if (inst->opcode() == Opcode::store_indexed) {
                MemFold mf = match_indexed_address(inst->operand(0), inst->operand(1), x64::scale_from_int(inst->scale()), inst->offset());
                for (const auto* fi : mf.folded_instructions) {
                    if (fi && fi->result()) folded_uses[fi->result()]++;
                }
                const Value* src = inst->operand(2);
                ImmIntInfo imm_src = get_imm_int_info(src);
                if (imm_src.is_imm && imm_src.fits_i32) {
                    folded_uses[src]++;
                }
                continue;
            }

            if (inst->opcode() == Opcode::br_if) {
                const Value* cond = inst->operand(0);
                if (cond && cond->is_instruction()) {
                    const Instruction* def_inst = cond->defining_instruction();
                    if (def_inst && def_inst->parent() == bb && is_comparison(def_inst->opcode()) && skipped_insts_.count(def_inst)) {
                        const Value* lhs = def_inst->operand(0);
                        const Value* rhs = def_inst->operand(1);
                        ImmIntInfo rhs_imm = get_imm_int_info(rhs);
                        ImmIntInfo lhs_imm = get_imm_int_info(lhs);

                        if (rhs_imm.is_imm && rhs_imm.fits_i32) {
                            folded_uses[rhs]++;
                        } else if (lhs_imm.is_imm && lhs_imm.fits_i32) {
                            folded_uses[lhs]++;
                        } else if (rhs && rhs->is_instruction() && can_fuse_load(rhs->defining_instruction(), inst)) {
                            skipped_insts_.insert(rhs->defining_instruction());
                        }
                        continue;
                    }
                }
            }

            bool is_alu = false;
            switch (inst->opcode()) {
                case Opcode::add: case Opcode::sub: case Opcode::mul:
                case Opcode::and_: case Opcode::or_: case Opcode::xor_:
                case Opcode::shl: case Opcode::lshr: case Opcode::ashr:
                case Opcode::udiv: case Opcode::umod:
                case Opcode::eq: case Opcode::ne:
                case Opcode::slt: case Opcode::ult: case Opcode::sle: case Opcode::ule:
                case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge:
                    is_alu = true;
                    break;
                default:
                    break;
            }

            if (is_alu && inst->operand_count() >= 2) {
                const Value* op0 = inst->operand(0);
                const Value* op1 = inst->operand(1);
                ImmIntInfo imm0 = get_imm_int_info(op0);
                ImmIntInfo imm1 = get_imm_int_info(op1);

                bool is_comm = (inst->opcode() == Opcode::add || inst->opcode() == Opcode::mul ||
                                inst->opcode() == Opcode::and_ || inst->opcode() == Opcode::or_ ||
                                inst->opcode() == Opcode::xor_);

                if (imm1.is_imm && imm1.fits_i32) {
                    folded_uses[op1]++;
                } else if (is_comm && imm0.is_imm && imm0.fits_i32) {
                    folded_uses[op0]++;
                } else if (op1 && op1->is_instruction() && can_fuse_load(op1->defining_instruction(), inst)) {
                    skipped_insts_.insert(op1->defining_instruction());
                } else if (is_comm && op0 && op0->is_instruction() && can_fuse_load(op0->defining_instruction(), inst)) {
                    skipped_insts_.insert(op0->defining_instruction());
                }
            }
        }
    }

    for (const auto& [val, fold_count] : folded_uses) {
        auto it = use_count_.find(val);
        if (it != use_count_.end() && it->second == fold_count) {
            if (val->is_instruction()) {
                skipped_insts_.insert(val->defining_instruction());
            }
        }
    }
}

std::unique_ptr<LirFunction> AArch64ISel::lower(const Function& mir_fn) {
    auto lir = std::make_unique<LirFunction>();
    lir_fn_ = lir.get();
    val_to_vreg_.clear();

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
            get_or_alloc_vreg(param);
        }
    }

    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (!inst->produces_value()) continue;
            if (skipped_insts_.count(inst)) continue;
            get_or_alloc_vreg(inst->result());
        }
    }

    lower_entry_parameters(mir_fn);

    for (const auto* bb : mir_fn.blocks()) {
        lower_block(*bb);
    }

    for (size_t i = 0; i < mir_fn.blocks().size(); ++i) {
        const auto* mir_bb = mir_fn.blocks()[i];
        auto* lir_bb = lir_fn_->blocks[i].get();

        for (const auto* pred : mir_bb->predecessors()) {
            lir_bb->predecessors.push_back(lir_fn_->get_block_by_id(pred->id()));
        }
        for (const auto* succ : mir_bb->successors()) {
            lir_bb->successors.push_back(lir_fn_->get_block_by_id(succ->id()));
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

    if (osr_target_ && osr_target_->is_valid()) {
        lir_fn_->osr_entry.enabled = true;
        lir_fn_->osr_entry.loop_header_id = osr_target_->loop_header_id;
        lir_fn_->osr_entry.live_in_vregs.clear();
        lir_fn_->osr_entry.slot_indices.clear();
        for (const Value* v : osr_target_->live_ins) {
            if (v) {
                VReg vr = get_or_alloc_vreg(v);
                lir_fn_->osr_entry.live_in_vregs.push_back(vr);
                lir_fn_->osr_entry.slot_indices.push_back(v->id());
            }
        }
    } else {
        for (const auto* bb : mir_fn.blocks()) {
            for (const auto* inst : *bb) {
                if (inst->opcode() == Opcode::osr_entry) {
                    lir_fn_->osr_entry.enabled = true;
                    lir_fn_->osr_entry.loop_header_id = static_cast<uint32_t>(inst->imm_i64());
                    for (const auto* op : inst->operands()) {
                        if (op) {
                            lir_fn_->osr_entry.live_in_vregs.push_back(get_or_alloc_vreg(op));
                            lir_fn_->osr_entry.slot_indices.push_back(op->id());
                        }
                    }
                    break;
                }
            }
            if (lir_fn_->osr_entry.enabled) break;
        }
    }

    return lir;
}

VReg AArch64ISel::get_or_alloc_vreg(const Value* val) {
    if (!val) return VReg{};
    auto it = val_to_vreg_.find(val);
    if (it != val_to_vreg_.end()) {
        return it->second;
    }

    Type t = val->type();
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
        VReg param_vreg = get_vreg(param);
        Type t = param->type();
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

    if (!mir_fn.resume_points().empty() && entry->param_count() > 0 && entry->param(0)->type() == Type::i32()) {
        const auto* param0 = entry->param(0);
        VReg param0_vreg = get_vreg(param0);
        for (const auto& rp : mir_fn.resume_points()) {
            if (rp.second) {
                auto cmp_inst = std::make_unique<LirInst>(LirOpcode::Cmp32);
                cmp_inst->add_use(LirOperand::vreg(param0_vreg, 4));
                cmp_inst->add_use(LirOperand::imm(static_cast<int32_t>(rp.first), 4));
                lir_entry->append_inst(std::move(cmp_inst));

                auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
                jcc_inst->condition = x64::Condition::E;
                jcc_inst->add_use(LirOperand::label(rp.second->id()));
                lir_entry->append_inst(std::move(jcc_inst));
            }
        }
    }
}

void AArch64ISel::lower_block(const BasicBlock& bb) {
    auto* lir_bb = lir_fn_->get_block_by_id(bb.id());
    if (!lir_bb) return;

    for (const auto* inst : bb) {
        if (!skipped_insts_.count(inst)) {
            size_t before_count = lir_bb->instructions.size();
            lower_instruction(*inst, *lir_bb);
            for (size_t i = before_count; i < lir_bb->instructions.size(); ++i) {
                if (lir_bb->instructions[i]) {
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

        case Opcode::osr_entry:
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
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movsxd);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 4));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
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

        case Opcode::fptosi_i32: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Cvttsd2si32);
            lir_inst->add_def(LirOperand::vreg(dst, 4));
            lir_inst->add_use(LirOperand::vreg(src, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::fptosi_i64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Cvttsd2si);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::sitofp_f64_i32: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Cvtsi2sd32);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 4));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::sitofp_f64_i64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Cvtsi2sd);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::bitcast_i64_f64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movq_gx);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

        case Opcode::bitcast_f64_i64: {
            VReg dst = get_vreg(inst.result());
            VReg src = get_vreg(inst.operand(0));
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Movq_xg);
            lir_inst->add_def(LirOperand::vreg(dst, 8));
            lir_inst->add_use(LirOperand::vreg(src, 8));
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }

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
            auto lir_inst = std::make_unique<LirInst>(LirOpcode::Nop);
            lir_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(lir_inst));
            break;
        }
    }
}

void AArch64ISel::lower_return(const Instruction& inst, LirBlock& lir_bb) {
    if (inst.operand_count() > 0) {
        const auto* ret_val = inst.operand(0);
        VReg ret_vreg = get_vreg(ret_val);
        Type t = ret_val->type();
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;

        if (t.is_float()) {
            LirOpcode ret_mov_op = (sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd;
            auto mov_ret = std::make_unique<LirInst>(ret_mov_op);
            mov_ret->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, sz), FixedConstraint::aarch64_fpr(FPR::V0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, sz));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, sz), FixedConstraint::aarch64_fpr(FPR::V0));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        } else if (t.is_vector()) {
            LirOpcode ret_mov_op = (sz == 32) ? LirOpcode::Vmovaps : LirOpcode::Movaps;
            auto mov_ret = std::make_unique<LirInst>(ret_mov_op);
            mov_ret->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, sz), FixedConstraint::aarch64_fpr(FPR::V0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, sz));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, sz), FixedConstraint::aarch64_fpr(FPR::V0));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        } else {
            auto mov_ret = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_ret->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, sz), FixedConstraint::aarch64_gpr(GPR::X0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, sz));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, sz), FixedConstraint::aarch64_gpr(GPR::X0));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        }
    }

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    ret_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(ret_inst));
}

void AArch64ISel::lower_branch(const Instruction& inst, LirBlock& lir_bb) {
    const auto& target = inst.branch_target();
    if (!target.block) return;

    if (!target.args.empty()) {
        if (target.args.size() == 1) {
            VReg arg_v = get_vreg(target.args[0]);
            VReg param_v = get_vreg(target.block->param(0));
            uint8_t sz = param_v.size;
            if (arg_v != param_v) {
                auto mov_inst = std::make_unique<LirInst>(
                    param_v.is_xmm() ? ((sz == 16) ? LirOpcode::Movaps : (sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd)) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                mov_inst->add_def(LirOperand::vreg(param_v, sz));
                mov_inst->add_use(LirOperand::vreg(arg_v, sz));
                lir_bb.append_inst(std::move(mov_inst));
            }
        } else {
            auto pcopy = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
            for (size_t i = 0; i < target.args.size(); ++i) {
                VReg arg_v = get_vreg(target.args[i]);
                VReg param_v = get_vreg(target.block->param(i));
                uint8_t sz = param_v.size;
                pcopy->add_def(LirOperand::vreg(param_v, sz));
                pcopy->add_use(LirOperand::vreg(arg_v, sz));
            }
            lir_bb.append_inst(std::move(pcopy));
        }
    }

    auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp_inst->add_use(LirOperand::label(target.block->id()));
    jmp_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(jmp_inst));
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
