#include <brass/target/x64/x64_isel.hpp>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace brass::x64 {

using namespace brass::codegen;

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
        for (const auto* inst : *bb) {
            if (inst->produces_value()) {
                get_or_alloc_vreg(inst->result());
            }
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

    return lir;
}

VReg X64ISel::get_or_alloc_vreg(const Value* val) {
    if (!val) return VReg{};
    auto it = val_to_vreg_.find(val);
    if (it != val_to_vreg_.end()) {
        return it->second;
    }

    Type t = val->type();
    RegClass rc = t.is_float() ? RegClass::XMM : RegClass::GPR;
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

    size_t gpr_idx = 0;
    size_t xmm_idx = 0;

    for (size_t i = 0; i < entry->param_count(); ++i) {
        const auto* param = entry->param(i);
        VReg param_vreg = get_vreg(param);
        Type t = param->type();

        if (cc_.kind() == CallingConvKind::Win64) {
            if (i < 4) {
                if (t.is_float()) {
                    XMM xreg = static_cast<XMM>(i);
                    auto inst = std::make_unique<LirInst>(LirOpcode::Movsd);
                    inst->add_def(LirOperand::vreg(param_vreg, 8));
                    inst->add_use(LirOperand::preg_xmm(xreg, 8), FixedConstraint::xmm(xreg));
                    lir_entry->append_inst(std::move(inst));
                } else {
                    GPR greg = cc_.arg_gpr(i);
                    uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
                    auto inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    inst->add_def(LirOperand::vreg(param_vreg, sz));
                    inst->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                    lir_entry->append_inst(std::move(inst));
                }
            } else {
                int32_t disp = static_cast<int32_t>(48 + (i - 4) * 8);
                uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
                if (t.is_float()) {
                    auto inst = std::make_unique<LirInst>(LirOpcode::Movsd);
                    inst->add_def(LirOperand::vreg(param_vreg, 8));
                    inst->add_use(LirOperand::mem(PReg::gpr(GPR::RBP), disp, 8));
                    lir_entry->append_inst(std::move(inst));
                } else {
                    auto inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    inst->add_def(LirOperand::vreg(param_vreg, sz));
                    inst->add_use(LirOperand::mem(PReg::gpr(GPR::RBP), disp, sz));
                    lir_entry->append_inst(std::move(inst));
                }
            }
        } else {
            if (t.is_float()) {
                if (xmm_idx < cc_.num_arg_xmms()) {
                    XMM xreg = cc_.arg_xmm(xmm_idx++);
                    auto inst = std::make_unique<LirInst>(LirOpcode::Movsd);
                    inst->add_def(LirOperand::vreg(param_vreg, 8));
                    inst->add_use(LirOperand::preg_xmm(xreg, 8), FixedConstraint::xmm(xreg));
                    lir_entry->append_inst(std::move(inst));
                } else {
                    size_t stack_idx = (xmm_idx - cc_.num_arg_xmms()) + (gpr_idx > cc_.num_arg_gprs() ? (gpr_idx - cc_.num_arg_gprs()) : 0);
                    int32_t disp = static_cast<int32_t>(16 + stack_idx * 8);
                    xmm_idx++;
                    auto inst = std::make_unique<LirInst>(LirOpcode::Movsd);
                    inst->add_def(LirOperand::vreg(param_vreg, 8));
                    inst->add_use(LirOperand::mem(PReg::gpr(GPR::RBP), disp, 8));
                    lir_entry->append_inst(std::move(inst));
                }
            } else {
                uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
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
    if (!mir_fn.resume_points().empty() && entry->param_count() > 0) {
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
        lower_instruction(*inst, *lir_bb);
    }
}

void X64ISel::lower_call(const Instruction& inst, LirBlock& lir_bb) {
    size_t num_args = inst.operand_count();
    size_t start_arg = 0;
    VReg callee_vreg;

    if (inst.opcode() == Opcode::call_indirect) {
        callee_vreg = get_vreg(inst.operand(0));
        start_arg = 1;
        num_args = inst.operand_count() - 1;
    }

    size_t required_stack_space = 0;
    if (cc_.kind() == CallingConvKind::Win64) {
        required_stack_space = std::max(size_t(32), 32 + (num_args > 4 ? (num_args - 4) * 8 : 0));
    } else {
        size_t stack_args = (num_args > 6 ? (num_args - 6) : 0);
        required_stack_space = stack_args * 8;
    }
    lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, required_stack_space);

    auto call_lir = std::make_unique<LirInst>(
        inst.opcode() == Opcode::call_indirect ? LirOpcode::CallIndirect : LirOpcode::Call
    );

    size_t gpr_idx = 0;
    size_t xmm_idx = 0;

    for (size_t i = 0; i < num_args; ++i) {
        const auto* arg_val = inst.operand(start_arg + i);
        VReg arg_vreg = get_vreg(arg_val);
        Type t = arg_val->type();
        uint8_t sz = static_cast<uint8_t>(t.size_in_bytes());
        if (sz == 0) sz = 8;

        if (cc_.kind() == CallingConvKind::Win64) {
            if (i < 4) {
                if (t.is_float()) {
                    XMM xreg = static_cast<XMM>(i);
                    auto mov_arg = std::make_unique<LirInst>(LirOpcode::Movsd);
                    mov_arg->add_def(LirOperand::preg_xmm(xreg, 8), FixedConstraint::xmm(xreg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, 8));
                    lir_bb.append_inst(std::move(mov_arg));

                    call_lir->add_use(LirOperand::preg_xmm(xreg, 8), FixedConstraint::xmm(xreg));
                } else {
                    GPR greg = cc_.arg_gpr(i);
                    auto mov_arg = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_arg->add_def(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                    lir_bb.append_inst(std::move(mov_arg));

                    call_lir->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                }
            } else {
                int32_t disp = static_cast<int32_t>(32 + (i - 4) * 8);
                auto mov_stack = std::make_unique<LirInst>(
                    t.is_float() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                mov_stack->add_def(LirOperand::mem(PReg::gpr(GPR::RSP), disp, sz));
                mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                lir_bb.append_inst(std::move(mov_stack));
            }
        } else {
            if (t.is_float()) {
                if (xmm_idx < cc_.num_arg_xmms()) {
                    XMM xreg = cc_.arg_xmm(xmm_idx++);
                    auto mov_arg = std::make_unique<LirInst>(LirOpcode::Movsd);
                    mov_arg->add_def(LirOperand::preg_xmm(xreg, 8), FixedConstraint::xmm(xreg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, 8));
                    lir_bb.append_inst(std::move(mov_arg));

                    call_lir->add_use(LirOperand::preg_xmm(xreg, 8), FixedConstraint::xmm(xreg));
                } else {
                    size_t stack_idx = (xmm_idx - cc_.num_arg_xmms()) + (gpr_idx > cc_.num_arg_gprs() ? (gpr_idx - cc_.num_arg_gprs()) : 0);
                    int32_t disp = static_cast<int32_t>(stack_idx * 8);
                    xmm_idx++;
                    auto mov_stack = std::make_unique<LirInst>(LirOpcode::Movsd);
                    mov_stack->add_def(LirOperand::mem(PReg::gpr(GPR::RSP), disp, 8));
                    mov_stack->add_use(LirOperand::vreg(arg_vreg, 8));
                    lir_bb.append_inst(std::move(mov_stack));
                }
            } else {
                if (gpr_idx < cc_.num_arg_gprs()) {
                    GPR greg = cc_.arg_gpr(gpr_idx++);
                    auto mov_arg = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_arg->add_def(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                    mov_arg->add_use(LirOperand::vreg(arg_vreg, sz));
                    lir_bb.append_inst(std::move(mov_arg));

                    call_lir->add_use(LirOperand::preg_gpr(greg, sz), FixedConstraint::gpr(greg));
                } else {
                    size_t stack_idx = (gpr_idx - cc_.num_arg_gprs()) + (xmm_idx > cc_.num_arg_xmms() ? (xmm_idx - cc_.num_arg_xmms()) : 0);
                    int32_t disp = static_cast<int32_t>(stack_idx * 8);
                    gpr_idx++;
                    auto mov_stack = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
                    mov_stack->add_def(LirOperand::mem(PReg::gpr(GPR::RSP), disp, sz));
                    mov_stack->add_use(LirOperand::vreg(arg_vreg, sz));
                    lir_bb.append_inst(std::move(mov_stack));
                }
            }
        }
    }

    if (inst.opcode() == Opcode::call_indirect) {
        call_lir->add_use(LirOperand::vreg(callee_vreg, 8));
    } else if (inst.opcode() == Opcode::patchable_call) {
        call_lir->is_patchable = true;
        call_lir->patch_symbol = std::string(inst.symbol());
        std::string callee_name = inst.extra_symbol().empty() ? std::string(inst.symbol()) : std::string(inst.extra_symbol());
        call_lir->callee_symbol = callee_name;
        call_lir->add_use(LirOperand::symbol(callee_name));
    } else {
        call_lir->add_use(LirOperand::symbol(std::string(inst.symbol())));
    }

    call_lir->clobbered_gprs = cc_.caller_saved_gpr_mask();
    call_lir->clobbered_xmms = cc_.caller_saved_xmm_mask();

    Type ret_t = inst.type();
    if (!ret_t.is_void()) {
        uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
        if (ret_t.is_float()) {
            call_lir->add_def(LirOperand::preg_xmm(XMM::XMM0, 8), FixedConstraint::xmm(XMM::XMM0));
        } else {
            call_lir->add_def(LirOperand::preg_gpr(GPR::RAX, ret_sz), FixedConstraint::gpr(GPR::RAX));
        }
    }

    call_lir->mir_origin = &inst;
    lir_bb.append_inst(std::move(call_lir));

    if (inst.produces_value()) {
        VReg dst = get_vreg(inst.result());
        if (ret_t.is_float()) {
            auto mov_ret = std::make_unique<LirInst>(LirOpcode::Movsd);
            mov_ret->add_def(LirOperand::vreg(dst, 8));
            mov_ret->add_use(LirOperand::preg_xmm(XMM::XMM0, 8), FixedConstraint::xmm(XMM::XMM0));
            lir_bb.append_inst(std::move(mov_ret));
        } else {
            uint8_t ret_sz = static_cast<uint8_t>(ret_t.size_in_bytes());
            auto mov_ret = std::make_unique<LirInst>(ret_sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            mov_ret->add_def(LirOperand::vreg(dst, ret_sz));
            mov_ret->add_use(LirOperand::preg_gpr(GPR::RAX, ret_sz), FixedConstraint::gpr(GPR::RAX));
            lir_bb.append_inst(std::move(mov_ret));
        }
    }
}

void X64ISel::lower_branch(const Instruction& inst, LirBlock& lir_bb) {
    const auto& target = inst.branch_target();
    if (!target.block) return;

    if (!target.args.empty()) {
        for (size_t i = 0; i < target.args.size(); ++i) {
            VReg arg_v = get_vreg(target.args[i]);
            VReg param_v = get_vreg(target.block->param(i));
            uint8_t sz = param_v.size;

            auto mov_inst = std::make_unique<LirInst>(
                param_v.is_xmm() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
            );
            mov_inst->add_def(LirOperand::vreg(param_v, sz));
            mov_inst->add_use(LirOperand::vreg(arg_v, sz));
            lir_bb.append_inst(std::move(mov_inst));
        }
    }

    auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp_inst->add_use(LirOperand::label(target.block->id()));
    jmp_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(jmp_inst));
}

void X64ISel::lower_branch_if(const Instruction& inst, LirBlock& lir_bb) {
    VReg cond = get_vreg(inst.operand(0));
    const auto& t_target = inst.true_target();
    const auto& f_target = inst.false_target();

    auto cmp_inst = std::make_unique<LirInst>(LirOpcode::Cmp32);
    cmp_inst->add_use(LirOperand::vreg(cond, 4));
    cmp_inst->add_use(LirOperand::imm(0, 4));
    lir_bb.append_inst(std::move(cmp_inst));

    if (t_target.args.empty() && f_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = Condition::NE;
        jcc_inst->add_use(LirOperand::label(t_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_inst->add_use(LirOperand::label(f_target.block->id()));
        jmp_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_inst));
    } else {
        auto* true_trampoline = lir_fn_->create_block("br_if_true");

        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = Condition::NE;
        jcc_inst->add_use(LirOperand::label(true_trampoline->id));
        lir_bb.append_inst(std::move(jcc_inst));

        for (size_t i = 0; i < f_target.args.size(); ++i) {
            VReg arg_v = get_vreg(f_target.args[i]);
            VReg param_v = get_vreg(f_target.block->param(i));
            uint8_t sz = param_v.size;
            auto mov_inst = std::make_unique<LirInst>(
                param_v.is_xmm() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
            );
            mov_inst->add_def(LirOperand::vreg(param_v, sz));
            mov_inst->add_use(LirOperand::vreg(arg_v, sz));
            lir_bb.append_inst(std::move(mov_inst));
        }
        auto jmp_f = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_f->add_use(LirOperand::label(f_target.block->id()));
        lir_bb.append_inst(std::move(jmp_f));

        for (size_t i = 0; i < t_target.args.size(); ++i) {
            VReg arg_v = get_vreg(t_target.args[i]);
            VReg param_v = get_vreg(t_target.block->param(i));
            uint8_t sz = param_v.size;
            auto mov_inst = std::make_unique<LirInst>(
                param_v.is_xmm() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
            );
            mov_inst->add_def(LirOperand::vreg(param_v, sz));
            mov_inst->add_use(LirOperand::vreg(arg_v, sz));
            true_trampoline->append_inst(std::move(mov_inst));
        }
        auto jmp_t = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_t->add_use(LirOperand::label(t_target.block->id()));
        true_trampoline->append_inst(std::move(jmp_t));
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
            auto mov_ret = std::make_unique<LirInst>(LirOpcode::Movsd);
            mov_ret->add_def(LirOperand::preg_xmm(XMM::XMM0, 8), FixedConstraint::xmm(XMM::XMM0));
            mov_ret->add_use(LirOperand::vreg(ret_vreg, 8));
            lir_bb.append_inst(std::move(mov_ret));

            auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
            ret_inst->add_use(LirOperand::preg_xmm(XMM::XMM0, 8), FixedConstraint::xmm(XMM::XMM0));
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

void X64ISel::lower_load(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_vreg(inst.result());
    VReg base = get_vreg(inst.operand(0));
    int32_t disp = inst.offset();
    uint8_t sz = dst.size;

    LirOpcode op = dst.is_xmm() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    auto lir_inst = std::make_unique<LirInst>(op);
    lir_inst->add_def(LirOperand::vreg(dst, sz));
    lir_inst->add_use(LirOperand::mem(base, disp, sz));
    lir_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(lir_inst));
}

void X64ISel::lower_store(const Instruction& inst, LirBlock& lir_bb) {
    VReg base = get_vreg(inst.operand(0));
    VReg src = get_vreg(inst.operand(1));
    int32_t disp = inst.offset();
    uint8_t sz = src.size;

    LirOpcode op = src.is_xmm() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    auto lir_inst = std::make_unique<LirInst>(op);
    lir_inst->add_def(LirOperand::mem(base, disp, sz));
    lir_inst->add_use(LirOperand::vreg(src, sz));
    lir_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(lir_inst));
}

void X64ISel::lower_load_indexed(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_vreg(inst.result());
    VReg base = get_vreg(inst.operand(0));
    VReg index = get_vreg(inst.operand(1));
    Scale sc = scale_from_int(inst.scale());
    int32_t disp = inst.offset();
    uint8_t sz = dst.size;

    LirOpcode op = dst.is_xmm() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    auto lir_inst = std::make_unique<LirInst>(op);
    lir_inst->add_def(LirOperand::vreg(dst, sz));
    lir_inst->add_use(LirOperand::mem(base, index, sc, disp, sz));
    lir_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(lir_inst));
}

void X64ISel::lower_store_indexed(const Instruction& inst, LirBlock& lir_bb) {
    VReg base = get_vreg(inst.operand(0));
    VReg index = get_vreg(inst.operand(1));
    VReg src = get_vreg(inst.operand(2));
    Scale sc = scale_from_int(inst.scale());
    int32_t disp = inst.offset();
    uint8_t sz = src.size;

    LirOpcode op = src.is_xmm() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    auto lir_inst = std::make_unique<LirInst>(op);
    lir_inst->add_def(LirOperand::mem(base, index, sc, disp, sz));
    lir_inst->add_use(LirOperand::vreg(src, sz));
    lir_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(lir_inst));
}

void X64ISel::lower_safepoint(const Instruction& inst, LirBlock& lir_bb) {
    if (cc_.kind() == CallingConvKind::Win64) {
        lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, size_t(32));
    }

    auto sp_inst = std::make_unique<LirInst>(LirOpcode::Safepoint);
    sp_inst->safepoint_id = inst.resume_id();
    sp_inst->clobbered_gprs = cc_.caller_saved_gpr_mask();
    sp_inst->clobbered_xmms = cc_.caller_saved_xmm_mask();

    for (const auto& entry : val_to_vreg_) {
        if (entry.second.is_gcref) {
            sp_inst->live_gcrefs.push_back(entry.second);
        }
    }
    sp_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(sp_inst));
}

void X64ISel::lower_guard(const Instruction& inst, LirBlock& lir_bb) {
    VReg cond = get_vreg(inst.operand(0));

    auto cmp_inst = std::make_unique<LirInst>(LirOpcode::Cmp32);
    cmp_inst->add_use(LirOperand::vreg(cond, 4));
    cmp_inst->add_use(LirOperand::imm(0, 4));
    lir_bb.append_inst(std::move(cmp_inst));

    auto* deopt_block = lir_fn_->create_block("guard_deopt");

    auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc_inst->condition = Condition::E;
    jcc_inst->add_use(LirOperand::label(deopt_block->id));
    lir_bb.append_inst(std::move(jcc_inst));

    auto exit_inst = std::make_unique<LirInst>(LirOpcode::GuardExit);
    exit_inst->resume_id = inst.resume_id();
    exit_inst->exit_symbol = std::string(inst.symbol());
    for (const auto* state_val : inst.state_map()) {
        VReg sv = get_vreg(state_val);
        if (sv.is_valid()) {
            exit_inst->add_use(LirOperand::vreg(sv, sv.size));
        }
    }
    exit_inst->mir_origin = &inst;
    deopt_block->append_inst(std::move(exit_inst));
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
