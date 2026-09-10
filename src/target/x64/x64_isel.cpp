#include <brass/target/x64/x64_isel.hpp>
#include <brass/mir/osr.hpp>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace brass::x64 {

using namespace brass::codegen;

static std::pair<Condition, Condition> get_comparison_conditions(Opcode op) noexcept {
    switch (op) {
        case Opcode::eq:  return {Condition::E, Condition::E};
        case Opcode::ne:  return {Condition::NE, Condition::NE};
        case Opcode::slt: return {Condition::L, Condition::B};
        case Opcode::ult: return {Condition::B, Condition::B};
        case Opcode::sle: return {Condition::LE, Condition::BE};
        case Opcode::ule: return {Condition::BE, Condition::BE};
        case Opcode::sgt: return {Condition::G, Condition::A};
        case Opcode::ugt: return {Condition::A, Condition::A};
        case Opcode::sge: return {Condition::GE, Condition::AE};
        case Opcode::uge: return {Condition::AE, Condition::AE};
        default: return {Condition::None, Condition::None};
    }
}

static constexpr Condition swap_relational_condition(Condition cond) noexcept {
    switch (cond) {
        case Condition::E:   return Condition::E;
        case Condition::NE:  return Condition::NE;
        case Condition::L:   return Condition::G;
        case Condition::LE:  return Condition::GE;
        case Condition::G:   return Condition::L;
        case Condition::GE:  return Condition::LE;
        case Condition::B:   return Condition::A;
        case Condition::BE:  return Condition::AE;
        case Condition::A:   return Condition::B;
        case Condition::AE:  return Condition::BE;
        default: return cond;
    }
}

X64ISel::ImmIntInfo X64ISel::get_imm_int_info(const Value* val) const {
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



bool X64ISel::is_value_dead_after(
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

void X64ISel::analyze_function(const Function& mir_fn) {
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

    // 2. Identify fused comparisons in br_if and guard, and LEA fusions in add
    auto skip_operand_if_dead = [&](const Value* val) {
        if (val && val->is_instruction() && use_count_[val] == 1) {
            skipped_insts_.insert(val->defining_instruction());
        }
    };

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
                                        skip_operand_if_dead(and_inst->operand(0));
                                        skip_operand_if_dead(and_inst->operand(1));
                                    }
                                }
                            }
                        }
                    }
                }
            } else if (inst->opcode() == Opcode::add) {
                const Value* op0 = inst->operand(0);
                const Value* op1 = inst->operand(1);
                ImmIntInfo imm0 = get_imm_int_info(op0);
                ImmIntInfo imm1 = get_imm_int_info(op1);
                const Value* reg_val = (imm1.is_imm && imm1.fits_i32) ? op0 : ((imm0.is_imm && imm0.fits_i32) ? op1 : nullptr);
                if (reg_val && reg_val->is_instruction()) {
                    const Instruction* def = reg_val->defining_instruction();
                    if (def && def->parent() == bb && use_count_[reg_val] == 1) {
                        if (def->opcode() == Opcode::mul) {
                            ImmIntInfo m0 = get_imm_int_info(def->operand(0));
                            ImmIntInfo m1 = get_imm_int_info(def->operand(1));
                            int64_t mult = m1.is_imm ? m1.val : (m0.is_imm ? m0.val : 0);
                            if (mult == 2 || mult == 3 || mult == 4 || mult == 5 || mult == 8 || mult == 9) {
                                skipped_insts_.insert(def);
                                if (m0.is_imm) skip_operand_if_dead(def->operand(0));
                                if (m1.is_imm) skip_operand_if_dead(def->operand(1));
                            }
                        } else if (def->opcode() == Opcode::shl) {
                            ImmIntInfo s1 = get_imm_int_info(def->operand(1));
                            if (s1.is_imm && (s1.val == 1 || s1.val == 2 || s1.val == 3)) {
                                skipped_insts_.insert(def);
                                skip_operand_if_dead(def->operand(1));
                            }
                        } else if (def->opcode() == Opcode::add) {
                            ImmIntInfo a0 = get_imm_int_info(def->operand(0));
                            ImmIntInfo a1 = get_imm_int_info(def->operand(1));
                            if (!a0.is_imm && !a1.is_imm) {
                                skipped_insts_.insert(def);
                            }
                        }
                    }
                }
            }
        }
    }

    std::unordered_map<const Value*, uint32_t> folded_uses;

    // 3. Count folded uses in load/store/alu/branch/guard
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
                MemFold mf = match_indexed_address(inst->operand(0), inst->operand(1), scale_from_int(inst->scale()), inst->offset());
                for (const auto* fi : mf.folded_instructions) {
                    if (fi && fi->result()) folded_uses[fi->result()]++;
                }
                continue;
            }

            if (inst->opcode() == Opcode::store_indexed) {
                MemFold mf = match_indexed_address(inst->operand(0), inst->operand(1), scale_from_int(inst->scale()), inst->offset());
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

                if (imm1.is_imm && imm1.fits_i32) {
                    folded_uses[op1]++;
                } else if (imm0.is_imm && imm0.fits_i32) {
                    folded_uses[op0]++;
                } else if (op1 && op1->is_instruction() && can_fuse_load(op1->defining_instruction(), inst)) {
                    skipped_insts_.insert(op1->defining_instruction());
                } else if (op0 && op0->is_instruction() && can_fuse_load(op0->defining_instruction(), inst)) {
                    skipped_insts_.insert(op0->defining_instruction());
                }
            }
        }
    }

    // 4. Skip instructions whose uses have all been folded
    for (const auto& [val, fold_count] : folded_uses) {
        auto it = use_count_.find(val);
        if (it != use_count_.end() && it->second == fold_count) {
            if (val->is_instruction()) {
                skipped_insts_.insert(val->defining_instruction());
            }
        }
    }
}

X64ISel::X64ISel()
    : target_(Target::host()), cc_(CallingConvention::for_target(Target::host())) {}

X64ISel::X64ISel(const Target& target)
    : target_(target), cc_(CallingConvention::for_target(target)) {}

X64ISel::X64ISel(const Target& target, const CallingConvention& cc)
    : target_(target), cc_(cc) {}

std::unique_ptr<LirFunction> X64ISel::lower(const Function& mir_fn) {
    auto lir = std::make_unique<LirFunction>();
    lir_fn_ = lir.get();
    val_to_vreg_.clear();

    lir_fn_->name = std::string(mir_fn.name());
    lir_fn_->return_type = mir_fn.return_type();
    lir_fn_->calling_conv = cc_;

    const_cast<Function&>(mir_fn).rebuild_cfg_predecessors();

    // 0. Pre-analyze function to identify fusible comparisons, loads, and immediate folds
    analyze_function(mir_fn);

    // 1. Create all LIR blocks matching MIR blocks
    for (const auto* bb : mir_fn.blocks()) {
        auto* lir_bb = lir_fn_->create_block(std::string(bb->name()));
        lir_bb->id = bb->id();
    }

    // 2. Allocate VRegs for all block parameters and instructions
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

    // 3. Lower entry block parameters from calling convention registers / stack slots
    lower_entry_parameters(mir_fn);

    // 4. Lower all instructions block by block
    for (const auto* bb : mir_fn.blocks()) {
        lower_block(*bb);
    }

    // 5. Connect CFG predecessors and successors
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

    // 6. Record resume table entries and connect CFG edges for resume targets
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

    // 7. Setup OSR entry if requested or present in MIR
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

VReg X64ISel::get_or_alloc_vreg(const Value* val) {
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

VReg X64ISel::get_vreg(const Value* val) const {
    if (!val) return VReg{};
    auto it = val_to_vreg_.find(val);
    if (it != val_to_vreg_.end()) {
        return it->second;
    }
    return VReg{};
}

void X64ISel::lower_entry_parameters(const Function& mir_fn) {
    const auto* entry = mir_fn.entry_block();
    if (!entry || entry->params().empty()) return;

    auto* lir_entry = lir_fn_->entry_block();
    if (!lir_entry) return;

    size_t gpr_idx = 0, xmm_idx = 0;
    for (size_t i = 0; i < entry->param_count(); ++i) {
        const auto* param = entry->param(i);
        VReg param_vreg = get_vreg(param);
        Type t = param->type();
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;
        LirOpcode mov_op = t.is_vector() ? LirOpcode::Movaps : (t.is_float() ? ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov));

        if (cc_.kind() == CallingConvKind::Win64) {
            if (i < 4) {
                auto inst = std::make_unique<LirInst>(mov_op);
                inst->add_def(LirOperand::vreg(param_vreg, sz));
                if (t.is_float() || t.is_vector()) {
                    XMM xreg = static_cast<XMM>(i);
                    inst->add_use(LirOperand::preg_xmm(xreg, sz), FixedConstraint::xmm(xreg));
                } else {
                    GPR greg = cc_.arg_gpr(i);
                    inst->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                }
                lir_entry->append_inst(std::move(inst));
            } else {
                int32_t disp = static_cast<int32_t>(48 + (i - 4) * 8);
                auto inst = std::make_unique<LirInst>(mov_op);
                inst->add_def(LirOperand::vreg(param_vreg, sz));
                inst->add_use(LirOperand::mem(PReg::gpr(GPR::RBP), disp, sz));
                lir_entry->append_inst(std::move(inst));
            }
        } else {
            if (t.is_float() || t.is_vector()) {
                if (xmm_idx < cc_.num_arg_xmms()) {
                    XMM xreg = cc_.arg_xmm(xmm_idx++);
                    auto inst = std::make_unique<LirInst>(mov_op);
                    inst->add_def(LirOperand::vreg(param_vreg, sz));
                    inst->add_use(LirOperand::preg_xmm(xreg, sz), FixedConstraint::xmm(xreg));
                    lir_entry->append_inst(std::move(inst));
                } else {
                    size_t stack_idx = (xmm_idx - cc_.num_arg_xmms()) + (gpr_idx > cc_.num_arg_gprs() ? (gpr_idx - cc_.num_arg_gprs()) : 0);
                    int32_t disp = static_cast<int32_t>(16 + stack_idx * 8);
                    xmm_idx++;
                    auto inst = std::make_unique<LirInst>(mov_op);
                    inst->add_def(LirOperand::vreg(param_vreg, sz));
                    inst->add_use(LirOperand::mem(PReg::gpr(GPR::RBP), disp, sz));
                    lir_entry->append_inst(std::move(inst));
                }
            } else {
                if (gpr_idx < cc_.num_arg_gprs()) {
                    GPR greg = cc_.arg_gpr(gpr_idx++);
                    auto inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    inst->add_def(LirOperand::vreg(param_vreg, sz));
                    inst->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                    lir_entry->append_inst(std::move(inst));
                } else {
                    size_t stack_idx = (gpr_idx - cc_.num_arg_gprs()) + (xmm_idx > cc_.num_arg_xmms() ? (xmm_idx - cc_.num_arg_xmms()) : 0);
                    int32_t disp = static_cast<int32_t>(16 + stack_idx * 8);
                    gpr_idx++;
                    auto inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    inst->add_def(LirOperand::vreg(param_vreg, sz));
                    inst->add_use(LirOperand::mem(PReg::gpr(GPR::RBP), disp, sz));
                    lir_entry->append_inst(std::move(inst));
                }
            }
        }
    }

    // Emit resume point prologue dispatcher after parameters have been saved
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
                jcc_inst->condition = Condition::E;
                jcc_inst->add_use(LirOperand::label(rp.second->id()));
                lir_entry->append_inst(std::move(jcc_inst));
            }
        }
    }
}

void X64ISel::lower_block(const BasicBlock& bb) {
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

void X64ISel::lower_branch(const Instruction& inst, LirBlock& lir_bb) {
    const auto& target = inst.branch_target();
    if (!target.block) return;

    if (!target.args.empty()) {
        if (target.args.size() == 1) {
            VReg arg_v = get_vreg(target.args[0]);
            VReg param_v = get_vreg(target.block->param(0));
            uint8_t sz = param_v.size;
            if (arg_v != param_v) {
                auto mov_inst = std::make_unique<LirInst>(
                    param_v.is_xmm() ? ((sz == 16) ? LirOpcode::Movaps : LirOpcode::Movsd) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
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

void X64ISel::lower_branch_if(const Instruction& inst, LirBlock& lir_bb) {
    const Value* cond_val = inst.operand(0);
    const auto& t_target = inst.true_target();
    const auto& f_target = inst.false_target();

    const Instruction* cmp_inst = cond_val ? cond_val->defining_instruction() : nullptr;
    bool is_fused_cmp = cmp_inst && cmp_inst->parent() == inst.parent() && is_comparison(cmp_inst->opcode());

    Condition branch_cond = Condition::NE;

    if (is_fused_cmp) {
        Opcode cmp_op = cmp_inst->opcode();
        auto [gpr_c, float_c] = get_comparison_conditions(cmp_op);
        const Value* lhs = cmp_inst->operand(0);
        const Value* rhs = cmp_inst->operand(1);

        if (lhs->type().is_float()) {
            branch_cond = float_c;
            if (rhs && rhs->is_instruction() && can_fuse_load(rhs->defining_instruction(), &inst)) {
                auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
                ucomi->add_use(LirOperand::vreg(get_vreg(lhs), 8));
                ucomi->add_use(get_load_mem_operand(rhs->defining_instruction()));
                lir_bb.append_inst(std::move(ucomi));
            } else {
                auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
                ucomi->add_use(LirOperand::vreg(get_vreg(lhs), 8));
                ucomi->add_use(LirOperand::vreg(get_vreg(rhs), 8));
                lir_bb.append_inst(std::move(ucomi));
            }
        } else {
            uint8_t sz = static_cast<uint8_t>(lhs->type().size_in_bytes());
            if (sz == 0) sz = 8;
            LirOpcode cmp_lir_op = (sz == 4) ? LirOpcode::Cmp32 : LirOpcode::Cmp;
            LirOpcode test_lir_op = (sz == 4) ? LirOpcode::Test32 : LirOpcode::Test;

            ImmIntInfo rhs_imm = get_imm_int_info(rhs);
            ImmIntInfo lhs_imm = get_imm_int_info(lhs);

            if ((cmp_op == Opcode::eq || cmp_op == Opcode::ne) && ((rhs_imm.is_imm && rhs_imm.val == 0) || (lhs_imm.is_imm && lhs_imm.val == 0))) {
                const Value* non_zero = (rhs_imm.is_imm && rhs_imm.val == 0) ? lhs : rhs;
                if (non_zero->is_instruction() && skipped_insts_.count(non_zero->defining_instruction()) && non_zero->defining_instruction()->opcode() == Opcode::and_) {
                    const Instruction* and_inst = non_zero->defining_instruction();
                    const Value* a = and_inst->operand(0);
                    const Value* b = and_inst->operand(1);
                    ImmIntInfo imm_b = get_imm_int_info(b);
                    ImmIntInfo imm_a = get_imm_int_info(a);

                    auto test_lir = std::make_unique<LirInst>(test_lir_op);
                    if (imm_b.is_imm && imm_b.fits_i32) {
                        test_lir->add_use(LirOperand::vreg(get_vreg(a), sz));
                        test_lir->add_use(LirOperand::imm(imm_b.val, sz));
                    } else if (imm_a.is_imm && imm_a.fits_i32) {
                        test_lir->add_use(LirOperand::vreg(get_vreg(b), sz));
                        test_lir->add_use(LirOperand::imm(imm_a.val, sz));
                    } else {
                        test_lir->add_use(LirOperand::vreg(get_vreg(a), sz));
                        test_lir->add_use(LirOperand::vreg(get_vreg(b), sz));
                    }
                    lir_bb.append_inst(std::move(test_lir));
                    branch_cond = (cmp_op == Opcode::eq) ? Condition::E : Condition::NE;
                } else {
                    auto test_lir = std::make_unique<LirInst>(test_lir_op);
                    VReg reg = get_vreg(non_zero);
                    test_lir->add_use(LirOperand::vreg(reg, sz));
                    test_lir->add_use(LirOperand::vreg(reg, sz));
                    lir_bb.append_inst(std::move(test_lir));
                    branch_cond = (rhs_imm.is_imm && rhs_imm.val == 0) ? gpr_c : swap_relational_condition(gpr_c);
                }
            } else if (rhs_imm.is_imm && rhs_imm.val == 0) {
                auto test_lir = std::make_unique<LirInst>(test_lir_op);
                test_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                test_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                lir_bb.append_inst(std::move(test_lir));
                branch_cond = gpr_c;
            } else if (lhs_imm.is_imm && lhs_imm.val == 0) {
                auto test_lir = std::make_unique<LirInst>(test_lir_op);
                test_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                test_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                lir_bb.append_inst(std::move(test_lir));
                branch_cond = swap_relational_condition(gpr_c);
            } else if (rhs_imm.is_imm && rhs_imm.fits_i32) {
                branch_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                cmp_lir->add_use(LirOperand::imm(rhs_imm.val, sz));
                lir_bb.append_inst(std::move(cmp_lir));
            } else if (lhs_imm.is_imm && lhs_imm.fits_i32) {
                branch_cond = swap_relational_condition(gpr_c);
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                cmp_lir->add_use(LirOperand::imm(lhs_imm.val, sz));
                lir_bb.append_inst(std::move(cmp_lir));
            } else if (rhs && rhs->is_instruction() && can_fuse_load(rhs->defining_instruction(), &inst)) {
                branch_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                cmp_lir->add_use(get_load_mem_operand(rhs->defining_instruction()));
                lir_bb.append_inst(std::move(cmp_lir));
            } else {
                branch_cond = gpr_c;
                auto cmp_lir = std::make_unique<LirInst>(cmp_lir_op);
                cmp_lir->add_use(LirOperand::vreg(get_vreg(lhs), sz));
                cmp_lir->add_use(LirOperand::vreg(get_vreg(rhs), sz));
                lir_bb.append_inst(std::move(cmp_lir));
            }
        }
    } else {
        VReg cond = get_vreg(cond_val);
        uint8_t sz = cond.size;
        auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
        test_inst->add_use(LirOperand::vreg(cond, sz));
        test_inst->add_use(LirOperand::vreg(cond, sz));
        lir_bb.append_inst(std::move(test_inst));
        branch_cond = Condition::NE;
    }

    auto emit_target_args = [&](LirBlock& bb, const BranchTarget& target) {
        if (target.args.empty()) return;
        if (target.args.size() == 1) {
            VReg arg_v = get_vreg(target.args[0]);
            VReg param_v = get_vreg(target.block->param(0));
            uint8_t sz = param_v.size;
            if (arg_v != param_v) {
                auto mov_inst = std::make_unique<LirInst>(
                    param_v.is_xmm() ? ((sz == 16) ? LirOpcode::Movaps : LirOpcode::Movsd) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                mov_inst->add_def(LirOperand::vreg(param_v, sz));
                mov_inst->add_use(LirOperand::vreg(arg_v, sz));
                bb.append_inst(std::move(mov_inst));
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
            bb.append_inst(std::move(pcopy));
        }
    };

    if (t_target.args.empty() && f_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = branch_cond;
        jcc_inst->add_use(LirOperand::label(t_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_inst->add_use(LirOperand::label(f_target.block->id()));
        jmp_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_inst));
    } else if (t_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = branch_cond;
        jcc_inst->add_use(LirOperand::label(t_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        emit_target_args(lir_bb, f_target);

        auto jmp_f = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_f->add_use(LirOperand::label(f_target.block->id()));
        jmp_f->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_f));
    } else if (f_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = invert(branch_cond);
        jcc_inst->add_use(LirOperand::label(f_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        emit_target_args(lir_bb, t_target);

        auto jmp_t = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_t->add_use(LirOperand::label(t_target.block->id()));
        jmp_t->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_t));
    } else {
        auto* false_trampoline = lir_fn_->create_block("br_if_false");

        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = invert(branch_cond);
        jcc_inst->add_use(LirOperand::label(false_trampoline->id));
        lir_bb.append_inst(std::move(jcc_inst));

        emit_target_args(lir_bb, t_target);

        auto jmp_t = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_t->add_use(LirOperand::label(t_target.block->id()));
        jmp_t->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_t));

        emit_target_args(*false_trampoline, f_target);

        auto jmp_f = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_f->add_use(LirOperand::label(f_target.block->id()));
        false_trampoline->append_inst(std::move(jmp_f));
    }
}

void X64ISel::lower_return(const Instruction& inst, LirBlock& lir_bb) {
    if (inst.operand_count() > 0) {
        const auto* ret_val = inst.operand(0);
        VReg ret_vreg = get_vreg(ret_val);
        Type t = ret_val->type();
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;

        if (t.is_float()) {
            LirOpcode ret_mov_op = (sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd;
            auto mov_ret = std::make_unique<LirInst>(ret_mov_op);
            mov_ret->add_def(LirOperand::preg_xmm(XMM::XMM0, sz), FixedConstraint::xmm(XMM::XMM0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, sz));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_xmm(XMM::XMM0, sz), FixedConstraint::xmm(XMM::XMM0));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        } else if (t.is_vector()) {
            auto mov_ret = std::make_unique<LirInst>(LirOpcode::Movaps);
            mov_ret->add_def(LirOperand::preg_xmm(XMM::XMM0, 16), FixedConstraint::xmm(XMM::XMM0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, 16));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_xmm(XMM::XMM0, 16), FixedConstraint::xmm(XMM::XMM0));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        } else {
            auto mov_ret = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_ret->add_def(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, sz));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_gpr(GPR::RAX, sz), FixedConstraint::gpr(GPR::RAX));
            ret_inst->mir_origin = &inst;
            lir_bb.append_inst(std::move(ret_inst));
            return;
        }
    }

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    ret_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(ret_inst));
}

std::unique_ptr<LirFunction> lower_to_x64_lir(
    const Function& mir_fn,
    const Target& target,
    const CallingConvention& cc
) {
    X64ISel isel(target, cc);
    return isel.lower(mir_fn);
}

} // namespace brass::x64
