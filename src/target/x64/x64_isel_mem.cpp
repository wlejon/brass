#include <brass/target/x64/x64_isel.hpp>
#include <cstring>
#include <algorithm>

namespace brass::x64 {

using namespace brass::codegen;

bool X64ISel::can_fuse_load(const Instruction* load_inst, const Instruction* user_inst) const {
    if (!load_inst || !user_inst) return false;
    if (load_inst->opcode() != Opcode::load && load_inst->opcode() != Opcode::load_indexed) {
        return false;
    }
    const Value* res = load_inst->result();
    if (!res) return false;
    auto it = use_count_.find(res);
    if (it == use_count_.end() || it->second != 1) {
        return false;
    }
    if (load_inst->parent() != user_inst->parent()) {
        return false;
    }
    for (const Instruction* cur = load_inst->next(); cur != nullptr && cur != user_inst; cur = cur->next()) {
        if (cur->opcode() == Opcode::store || cur->opcode() == Opcode::store_indexed ||
            cur->is_call() || cur->opcode() == Opcode::safepoint || cur->opcode() == Opcode::guard) {
            return false;
        }
    }
    return true;
}

X64ISel::MemFold X64ISel::match_address(const Value* ptr, int32_t offset) const {
    MemFold fold;
    fold.base_val = ptr;
    fold.disp = offset;
    if (!ptr || !ptr->is_instruction()) return fold;

    const Instruction* def = ptr->defining_instruction();
    if (!def || def->opcode() != Opcode::add) return fold;

    const Value* op0 = def->operand(0);
    const Value* op1 = def->operand(1);
    ImmIntInfo imm0 = get_imm_int_info(op0);
    ImmIntInfo imm1 = get_imm_int_info(op1);

    if (imm1.is_imm && imm1.fits_i32) {
        int64_t new_disp = static_cast<int64_t>(fold.disp) + imm1.val;
        if (new_disp >= INT32_MIN && new_disp <= INT32_MAX) {
            fold.disp = static_cast<int32_t>(new_disp);
            fold.base_val = op0;
            fold.folded_instructions.push_back(def);
            if (imm1.def_inst) fold.folded_instructions.push_back(imm1.def_inst);

            if (op0 && op0->is_instruction()) {
                const Instruction* def0 = op0->defining_instruction();
                if (def0 && def0->opcode() == Opcode::add && def0->parent() == def->parent()) {
                    const Value* a0 = def0->operand(0);
                    const Value* a1 = def0->operand(1);
                    ImmIntInfo imm_a0 = get_imm_int_info(a0);
                    ImmIntInfo imm_a1 = get_imm_int_info(a1);
                    if (!imm_a0.is_imm && !imm_a1.is_imm) {
                        if (a1 && a1->is_instruction()) {
                            const Instruction* s_def = a1->defining_instruction();
                            if (s_def && s_def->parent() == def->parent() && s_def->opcode() == Opcode::shl) {
                                ImmIntInfo s_imm = get_imm_int_info(s_def->operand(1));
                                if (s_imm.is_imm && s_imm.val >= 1 && s_imm.val <= 3) {
                                    fold.base_val = a0;
                                    fold.index_val = s_def->operand(0);
                                    fold.scale = scale_from_int(1 << s_imm.val);
                                    fold.folded_instructions.push_back(def0);
                                    fold.folded_instructions.push_back(s_def);
                                    if (s_imm.def_inst) fold.folded_instructions.push_back(s_imm.def_inst);
                                    return fold;
                                }
                            }
                        }
                        fold.base_val = a0;
                        fold.index_val = a1;
                        fold.scale = Scale::One;
                        fold.folded_instructions.push_back(def0);
                        return fold;
                    }
                }
            }
            return fold;
        }
    } else if (imm0.is_imm && imm0.fits_i32) {
        int64_t new_disp = static_cast<int64_t>(fold.disp) + imm0.val;
        if (new_disp >= INT32_MIN && new_disp <= INT32_MAX) {
            fold.disp = static_cast<int32_t>(new_disp);
            fold.base_val = op1;
            fold.folded_instructions.push_back(def);
            if (imm0.def_inst) fold.folded_instructions.push_back(imm0.def_inst);
            return fold;
        }
    } else {
        if (op1 && op1->is_instruction()) {
            const Instruction* s_def = op1->defining_instruction();
            if (s_def && s_def->parent() == def->parent() && s_def->opcode() == Opcode::shl) {
                ImmIntInfo s_imm = get_imm_int_info(s_def->operand(1));
                if (s_imm.is_imm && s_imm.val >= 1 && s_imm.val <= 3) {
                    fold.base_val = op0;
                    fold.index_val = s_def->operand(0);
                    fold.scale = scale_from_int(1 << s_imm.val);
                    fold.folded_instructions.push_back(def);
                    fold.folded_instructions.push_back(s_def);
                    if (s_imm.def_inst) fold.folded_instructions.push_back(s_imm.def_inst);
                    return fold;
                }
            }
        }
        fold.base_val = op0;
        fold.index_val = op1;
        fold.scale = Scale::One;
        fold.folded_instructions.push_back(def);
    }
    return fold;
}

X64ISel::MemFold X64ISel::match_indexed_address(const Value* base, const Value* index, Scale scale, int32_t offset) const {
    MemFold fold;
    fold.base_val = base;
    fold.index_val = index;
    fold.scale = scale;
    fold.disp = offset;

    if (index && index->is_instruction()) {
        const Instruction* def = index->defining_instruction();
        if (def && def->opcode() == Opcode::add) {
            ImmIntInfo imm0 = get_imm_int_info(def->operand(0));
            ImmIntInfo imm1 = get_imm_int_info(def->operand(1));
            int32_t sc = (scale == Scale::Eight ? 8 : (scale == Scale::Four ? 4 : (scale == Scale::Two ? 2 : 1)));
            if (imm1.is_imm && imm1.fits_i32) {
                int64_t new_disp = static_cast<int64_t>(fold.disp) + imm1.val * sc;
                if (new_disp >= INT32_MIN && new_disp <= INT32_MAX) {
                    fold.disp = static_cast<int32_t>(new_disp);
                    fold.index_val = def->operand(0);
                    fold.folded_instructions.push_back(def);
                    if (imm1.def_inst) fold.folded_instructions.push_back(imm1.def_inst);
                }
            } else if (imm0.is_imm && imm0.fits_i32) {
                int64_t new_disp = static_cast<int64_t>(fold.disp) + imm0.val * sc;
                if (new_disp >= INT32_MIN && new_disp <= INT32_MAX) {
                    fold.disp = static_cast<int32_t>(new_disp);
                    fold.index_val = def->operand(1);
                    fold.folded_instructions.push_back(def);
                    if (imm0.def_inst) fold.folded_instructions.push_back(imm0.def_inst);
                }
            }
        }
    }

    if (base && base->is_instruction()) {
        const Instruction* def = base->defining_instruction();
        if (def && def->opcode() == Opcode::add) {
            ImmIntInfo imm0 = get_imm_int_info(def->operand(0));
            ImmIntInfo imm1 = get_imm_int_info(def->operand(1));
            if (imm1.is_imm && imm1.fits_i32) {
                int64_t new_disp = static_cast<int64_t>(fold.disp) + imm1.val;
                if (new_disp >= INT32_MIN && new_disp <= INT32_MAX) {
                    fold.disp = static_cast<int32_t>(new_disp);
                    fold.base_val = def->operand(0);
                    fold.folded_instructions.push_back(def);
                    if (imm1.def_inst) fold.folded_instructions.push_back(imm1.def_inst);
                }
            } else if (imm0.is_imm && imm0.fits_i32) {
                int64_t new_disp = static_cast<int64_t>(fold.disp) + imm0.val;
                if (new_disp >= INT32_MIN && new_disp <= INT32_MAX) {
                    fold.disp = static_cast<int32_t>(new_disp);
                    fold.base_val = def->operand(1);
                    fold.folded_instructions.push_back(def);
                    if (imm0.def_inst) fold.folded_instructions.push_back(imm0.def_inst);
                }
            }
        }
    }

    return fold;
}

LirOperand X64ISel::get_load_mem_operand(const Instruction* load_inst) const {
    if (!load_inst) return LirOperand{};
    uint8_t sz = static_cast<uint8_t>(load_inst->type().size_in_bytes());
    if (sz == 0) sz = 8;
    if (load_inst->opcode() == Opcode::load) {
        MemFold mf = match_address(load_inst->operand(0), load_inst->offset());
        VReg base = get_vreg(mf.base_val);
        if (mf.index_val) {
            return LirOperand::mem(base, get_vreg(mf.index_val), mf.scale, mf.disp, sz);
        }
        return LirOperand::mem(base, mf.disp, sz);
    } else if (load_inst->opcode() == Opcode::load_indexed) {
        MemFold mf = match_indexed_address(load_inst->operand(0), load_inst->operand(1), scale_from_int(load_inst->scale()), load_inst->offset());
        VReg base = get_vreg(mf.base_val);
        if (mf.index_val) {
            return LirOperand::mem(base, get_vreg(mf.index_val), mf.scale, mf.disp, sz);
        }
        return LirOperand::mem(base, mf.disp, sz);
    }
    return LirOperand{};
}

void X64ISel::lower_load(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_vreg(inst.result());
    uint8_t sz = dst.size;
    MemFold mf = match_address(inst.operand(0), inst.offset());
    VReg base_vreg = get_vreg(mf.base_val);
    LirOperand mem_op;
    if (mf.index_val) {
        VReg idx_vreg = get_vreg(mf.index_val);
        mem_op = LirOperand::mem(base_vreg, idx_vreg, mf.scale, mf.disp, sz);
    } else {
        mem_op = LirOperand::mem(base_vreg, mf.disp, sz);
    }

    LirOpcode op = dst.is_xmm() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    auto lir_inst = std::make_unique<LirInst>(op);
    lir_inst->add_def(LirOperand::vreg(dst, sz));
    lir_inst->add_use(mem_op);
    lir_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(lir_inst));
}

void X64ISel::lower_store(const Instruction& inst, LirBlock& lir_bb) {
    const Value* src_val = inst.operand(1);
    ImmIntInfo imm_src = get_imm_int_info(src_val);
    uint8_t sz = static_cast<uint8_t>(src_val->type().size_in_bytes());
    if (sz == 0) sz = 8;
    MemFold mf = match_address(inst.operand(0), inst.offset());
    VReg base_vreg = get_vreg(mf.base_val);
    LirOperand mem_op;
    if (mf.index_val) {
        VReg idx_vreg = get_vreg(mf.index_val);
        mem_op = LirOperand::mem(base_vreg, idx_vreg, mf.scale, mf.disp, sz);
    } else {
        mem_op = LirOperand::mem(base_vreg, mf.disp, sz);
    }

    bool is_xmm = src_val->type().is_float();
    if (!is_xmm && imm_src.is_imm && imm_src.fits_i32) {
        auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        lir_inst->add_def(mem_op);
        lir_inst->add_use(LirOperand::imm(imm_src.val, sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    } else {
        VReg src = get_vreg(src_val);
        LirOpcode op = is_xmm ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        auto lir_inst = std::make_unique<LirInst>(op);
        lir_inst->add_def(mem_op);
        lir_inst->add_use(LirOperand::vreg(src, sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    }
}

void X64ISel::lower_load_indexed(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_vreg(inst.result());
    uint8_t sz = dst.size;
    MemFold mf = match_indexed_address(inst.operand(0), inst.operand(1), scale_from_int(inst.scale()), inst.offset());
    VReg base_vreg = get_vreg(mf.base_val);
    LirOperand mem_op;
    if (mf.index_val) {
        VReg idx_vreg = get_vreg(mf.index_val);
        mem_op = LirOperand::mem(base_vreg, idx_vreg, mf.scale, mf.disp, sz);
    } else {
        mem_op = LirOperand::mem(base_vreg, mf.disp, sz);
    }

    LirOpcode op = dst.is_xmm() ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
    auto lir_inst = std::make_unique<LirInst>(op);
    lir_inst->add_def(LirOperand::vreg(dst, sz));
    lir_inst->add_use(mem_op);
    lir_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(lir_inst));
}

void X64ISel::lower_store_indexed(const Instruction& inst, LirBlock& lir_bb) {
    const Value* src_val = inst.operand(2);
    ImmIntInfo imm_src = get_imm_int_info(src_val);
    uint8_t sz = static_cast<uint8_t>(src_val->type().size_in_bytes());
    if (sz == 0) sz = 8;
    MemFold mf = match_indexed_address(inst.operand(0), inst.operand(1), scale_from_int(inst.scale()), inst.offset());
    VReg base_vreg = get_vreg(mf.base_val);
    LirOperand mem_op;
    if (mf.index_val) {
        VReg idx_vreg = get_vreg(mf.index_val);
        mem_op = LirOperand::mem(base_vreg, idx_vreg, mf.scale, mf.disp, sz);
    } else {
        mem_op = LirOperand::mem(base_vreg, mf.disp, sz);
    }

    bool is_xmm = src_val->type().is_float();
    if (!is_xmm && imm_src.is_imm && imm_src.fits_i32) {
        auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        lir_inst->add_def(mem_op);
        lir_inst->add_use(LirOperand::imm(imm_src.val, sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    } else {
        VReg src = get_vreg(src_val);
        LirOpcode op = is_xmm ? LirOpcode::Movsd : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        auto lir_inst = std::make_unique<LirInst>(op);
        lir_inst->add_def(mem_op);
        lir_inst->add_use(LirOperand::vreg(src, sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    }
}

void X64ISel::lower_write_barrier(const Instruction& inst, LirBlock& lir_bb) {
    const Value* obj_val = inst.operand(0);
    const Value* val_val = inst.operand(1);
    if (!obj_val || !val_val) return;

    VReg obj_vreg = get_vreg(obj_val);
    VReg val_vreg = get_vreg(val_val);

    if (cc_.kind() == CallingConvKind::Win64) {
        lir_fn_->frame.outgoing_arg_space = std::max(lir_fn_->frame.outgoing_arg_space, size_t(32));

        auto mov_obj = std::make_unique<LirInst>(LirOpcode::Mov);
        mov_obj->add_def(LirOperand::preg_gpr(GPR::RCX, 8), FixedConstraint::gpr(GPR::RCX));
        mov_obj->add_use(LirOperand::vreg(obj_vreg, 8));
        lir_bb.append_inst(std::move(mov_obj));

        auto mov_val = std::make_unique<LirInst>(LirOpcode::Mov);
        mov_val->add_def(LirOperand::preg_gpr(GPR::RDX, 8), FixedConstraint::gpr(GPR::RDX));
        mov_val->add_use(LirOperand::vreg(val_vreg, 8));
        lir_bb.append_inst(std::move(mov_val));

        auto call_wb = std::make_unique<LirInst>(LirOpcode::Call);
        call_wb->callee_symbol = "brass_gc_write_barrier";
        call_wb->add_use(LirOperand::preg_gpr(GPR::RCX, 8), FixedConstraint::gpr(GPR::RCX));
        call_wb->add_use(LirOperand::preg_gpr(GPR::RDX, 8), FixedConstraint::gpr(GPR::RDX));
        call_wb->add_use(LirOperand::symbol("brass_gc_write_barrier"));
        call_wb->clobbered_gprs = cc_.caller_saved_gpr_mask();
        call_wb->clobbered_xmms = cc_.caller_saved_xmm_mask();
        call_wb->mir_origin = &inst;
        lir_bb.append_inst(std::move(call_wb));
    } else {
        auto mov_obj = std::make_unique<LirInst>(LirOpcode::Mov);
        mov_obj->add_def(LirOperand::preg_gpr(GPR::RDI, 8), FixedConstraint::gpr(GPR::RDI));
        mov_obj->add_use(LirOperand::vreg(obj_vreg, 8));
        lir_bb.append_inst(std::move(mov_obj));

        auto mov_val = std::make_unique<LirInst>(LirOpcode::Mov);
        mov_val->add_def(LirOperand::preg_gpr(GPR::RSI, 8), FixedConstraint::gpr(GPR::RSI));
        mov_val->add_use(LirOperand::vreg(val_vreg, 8));
        lir_bb.append_inst(std::move(mov_val));

        auto call_wb = std::make_unique<LirInst>(LirOpcode::Call);
        call_wb->callee_symbol = "brass_gc_write_barrier";
        call_wb->add_use(LirOperand::preg_gpr(GPR::RDI, 8), FixedConstraint::gpr(GPR::RDI));
        call_wb->add_use(LirOperand::preg_gpr(GPR::RSI, 8), FixedConstraint::gpr(GPR::RSI));
        call_wb->add_use(LirOperand::symbol("brass_gc_write_barrier"));
        call_wb->clobbered_gprs = cc_.caller_saved_gpr_mask();
        call_wb->clobbered_xmms = cc_.caller_saved_xmm_mask();
        call_wb->mir_origin = &inst;
        lir_bb.append_inst(std::move(call_wb));
    }
}

} // namespace brass::x64
