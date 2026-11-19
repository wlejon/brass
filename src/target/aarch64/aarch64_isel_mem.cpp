#include <brass/target/aarch64/aarch64_isel.hpp>
#include <cstdlib>
#include <cstring>
#include <algorithm>

namespace brass::aarch64 {

using namespace brass::codegen;
using x64::Scale;

namespace {

bool is_legal_aarch64_imm_offset(int64_t disp, uint8_t sz) noexcept {
    if (disp >= -256 && disp <= 255) return true;
    return sz > 0 && disp >= 0 && (disp % sz == 0) && (disp / sz <= 4095);
}

bool is_legal_aarch64_reg_scale(Scale scale, uint8_t sz) noexcept {
    switch (scale) {
        case Scale::One: return true;
        case Scale::Two: return sz == 2;
        case Scale::Four: return sz == 4;
        case Scale::Eight: return sz == 8;
    }
    return false;
}

uint8_t scale_to_shift(Scale scale) noexcept {
    switch (scale) {
        case Scale::Two: return 1;
        case Scale::Four: return 2;
        case Scale::Eight: return 3;
        default: return 0;
    }
}

} // namespace

// As on x64 (x64_isel_mem.cpp fold_two_register_sum): a register + register
// sum from another block that is a value of its own anyway stays one, rather
// than keeping both addends live to the use as well. AArch64 has no
// base + index + displacement form, so such a fold was also an add at the use.
bool AArch64ISel::fold_two_register_sum(const Instruction* sum, const Value* ptr, const BasicBlock* user_bb) const {
    return !user_bb || sum->parent() == user_bb || address_only_.count(ptr) != 0;
}

AArch64ISel::MemFold AArch64ISel::match_address(const Value* ptr, int32_t offset, uint8_t access_size,
                                                const BasicBlock* user_bb) const {
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
        if (new_disp >= INT32_MIN && new_disp <= INT32_MAX &&
            is_legal_aarch64_imm_offset(new_disp, access_size)) {
            fold.disp = static_cast<int32_t>(new_disp);
            fold.base_val = op0;
            fold.folded_instructions.push_back(def);
            if (imm1.def_inst) fold.folded_instructions.push_back(imm1.def_inst);

            if (fold.disp == 0 && op0 && op0->is_instruction()) {
                const Instruction* def0 = op0->defining_instruction();
                if (def0 && def0->opcode() == Opcode::add && def0->parent() == def->parent() &&
                    fold_two_register_sum(def0, ptr, user_bb)) {
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
                                    Scale sc = x64::scale_from_int(1 << s_imm.val);
                                    if (is_legal_aarch64_reg_scale(sc, access_size)) {
                                        fold.base_val = a0;
                                        fold.index_val = s_def->operand(0);
                                        fold.scale = sc;
                                        fold.folded_instructions.push_back(def0);
                                        fold.folded_instructions.push_back(s_def);
                                        if (s_imm.def_inst) fold.folded_instructions.push_back(s_imm.def_inst);
                                        return fold;
                                    }
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
        if (new_disp >= INT32_MIN && new_disp <= INT32_MAX &&
            is_legal_aarch64_imm_offset(new_disp, access_size)) {
            fold.disp = static_cast<int32_t>(new_disp);
            fold.base_val = op1;
            fold.folded_instructions.push_back(def);
            if (imm0.def_inst) fold.folded_instructions.push_back(imm0.def_inst);
            return fold;
        }
    } else if (offset == 0 && !imm0.is_imm && !imm1.is_imm && fold_two_register_sum(def, ptr, user_bb)) {
        if (op1 && op1->is_instruction()) {
            const Instruction* s_def = op1->defining_instruction();
            if (s_def && s_def->parent() == def->parent() && s_def->opcode() == Opcode::shl) {
                ImmIntInfo s_imm = get_imm_int_info(s_def->operand(1));
                if (s_imm.is_imm && s_imm.val >= 1 && s_imm.val <= 3) {
                    Scale sc = x64::scale_from_int(1 << s_imm.val);
                    if (is_legal_aarch64_reg_scale(sc, access_size)) {
                        fold.base_val = op0;
                        fold.index_val = s_def->operand(0);
                        fold.scale = sc;
                        fold.folded_instructions.push_back(def);
                        fold.folded_instructions.push_back(s_def);
                        if (s_imm.def_inst) fold.folded_instructions.push_back(s_imm.def_inst);
                        return fold;
                    }
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

AArch64ISel::MemFold AArch64ISel::match_indexed_address(const Value* base, const Value* index, Scale scale,
                                                        int32_t offset, uint8_t access_size) const {
    MemFold fold;
    fold.base_val = base;
    fold.index_val = index;
    fold.scale = scale;
    fold.disp = offset;

    if (!is_legal_aarch64_reg_scale(scale, access_size)) {
        return fold;
    }

    if (index && index->is_instruction()) {
        const Instruction* def = index->defining_instruction();
        if (def && def->opcode() == Opcode::add) {
            ImmIntInfo imm0 = get_imm_int_info(def->operand(0));
            ImmIntInfo imm1 = get_imm_int_info(def->operand(1));
            int32_t sc = (scale == Scale::Eight ? 8 : (scale == Scale::Four ? 4 : (scale == Scale::Two ? 2 : 1)));
            if (imm1.is_imm && imm1.fits_i32) {
                int64_t new_disp = static_cast<int64_t>(fold.disp) + imm1.val * sc;
                if (new_disp == 0) {
                    fold.disp = 0;
                    fold.index_val = def->operand(0);
                    fold.folded_instructions.push_back(def);
                    if (imm1.def_inst) fold.folded_instructions.push_back(imm1.def_inst);
                }
            } else if (imm0.is_imm && imm0.fits_i32) {
                int64_t new_disp = static_cast<int64_t>(fold.disp) + imm0.val * sc;
                if (new_disp == 0) {
                    fold.disp = 0;
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
                if (new_disp == 0) {
                    fold.disp = 0;
                    fold.base_val = def->operand(0);
                    fold.folded_instructions.push_back(def);
                    if (imm1.def_inst) fold.folded_instructions.push_back(imm1.def_inst);
                }
            } else if (imm0.is_imm && imm0.fits_i32) {
                int64_t new_disp = static_cast<int64_t>(fold.disp) + imm0.val;
                if (new_disp == 0) {
                    fold.disp = 0;
                    fold.base_val = def->operand(1);
                    fold.folded_instructions.push_back(def);
                    if (imm0.def_inst) fold.folded_instructions.push_back(imm0.def_inst);
                }
            }
        }
    }

    return fold;
}

LirOperand AArch64ISel::legalize_mem_operand(const MemFold& mf, uint8_t sz, LirBlock& lir_bb) {
    VReg base_vreg = get_vreg(mf.base_val);
    if (!mf.index_val) {
        if (is_legal_aarch64_imm_offset(mf.disp, sz)) {
            return LirOperand::mem(base_vreg, mf.disp, sz);
        }
        VReg ea = lir_fn_->allocate_vreg(RegClass::GPR, 8);
        auto add_inst = std::make_unique<LirInst>(LirOpcode::Add);
        add_inst->add_def(LirOperand::vreg(ea, 8));
        add_inst->add_use(LirOperand::vreg(base_vreg, 8));
        add_inst->add_use(LirOperand::imm(mf.disp, 8));
        lir_bb.append_inst(std::move(add_inst));
        return LirOperand::mem(ea, 0, sz);
    }

    VReg idx_vreg = get_vreg(mf.index_val);
    if (is_legal_aarch64_reg_scale(mf.scale, sz)) {
        if (mf.disp == 0) {
            return LirOperand::mem(base_vreg, idx_vreg, mf.scale, 0, sz);
        }
        VReg ea = lir_fn_->allocate_vreg(RegClass::GPR, 8);
        auto add_inst = std::make_unique<LirInst>(LirOpcode::Add);
        add_inst->add_def(LirOperand::vreg(ea, 8));
        add_inst->add_use(LirOperand::vreg(base_vreg, 8));
        add_inst->add_use(LirOperand::imm(mf.disp, 8));
        lir_bb.append_inst(std::move(add_inst));
        return LirOperand::mem(ea, idx_vreg, mf.scale, 0, sz);
    }

    uint8_t shift = scale_to_shift(mf.scale);
    VReg scaled = lir_fn_->allocate_vreg(RegClass::GPR, 8);
    auto shl_inst = std::make_unique<LirInst>(LirOpcode::Shl);
    shl_inst->add_def(LirOperand::vreg(scaled, 8));
    shl_inst->add_use(LirOperand::vreg(idx_vreg, 8));
    shl_inst->add_use(LirOperand::imm(shift, 1));
    lir_bb.append_inst(std::move(shl_inst));

    VReg ea = lir_fn_->allocate_vreg(RegClass::GPR, 8);
    auto add_idx = std::make_unique<LirInst>(LirOpcode::Add);
    add_idx->add_def(LirOperand::vreg(ea, 8));
    add_idx->add_use(LirOperand::vreg(base_vreg, 8));
    add_idx->add_use(LirOperand::vreg(scaled, 8));
    lir_bb.append_inst(std::move(add_idx));

    if (is_legal_aarch64_imm_offset(mf.disp, sz)) {
        return LirOperand::mem(ea, mf.disp, sz);
    }

    VReg ea2 = lir_fn_->allocate_vreg(RegClass::GPR, 8);
    auto add_disp = std::make_unique<LirInst>(LirOpcode::Add);
    add_disp->add_def(LirOperand::vreg(ea2, 8));
    add_disp->add_use(LirOperand::vreg(ea, 8));
    add_disp->add_use(LirOperand::imm(mf.disp, 8));
    lir_bb.append_inst(std::move(add_disp));
    return LirOperand::mem(ea2, 0, sz);
}

void AArch64ISel::lower_load(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_vreg(inst.result());
    uint8_t sz = dst.size;
    MemFold mf = match_address(inst.operand(0), inst.offset(), sz, inst.parent());
    LirOperand mem_op = legalize_mem_operand(mf, sz, lir_bb);

    LirOpcode op;
    if (dst.is_xmm()) {
        op = (sz == 16) ? LirOpcode::Movups : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd);
    } else {
        if (sz == 1) op = LirOpcode::Movzx8;
        else if (sz == 2) op = LirOpcode::Movzx16;
        else if (sz == 4) op = LirOpcode::Mov32;
        else op = LirOpcode::Mov;
    }

    auto lir_inst = std::make_unique<LirInst>(op);
    lir_inst->add_def(LirOperand::vreg(dst, sz));
    lir_inst->add_use(mem_op);
    lir_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(lir_inst));
}

void AArch64ISel::lower_store(const Instruction& inst, LirBlock& lir_bb) {
    const Value* src_val = inst.operand(1);
    ImmIntInfo imm_src = get_imm_int_info(src_val);
    uint8_t sz = static_cast<uint8_t>(src_val->type().size_in_bytes());
    if (sz == 0) sz = 8;
    MemFold mf = match_address(inst.operand(0), inst.offset(), sz, inst.parent());
    LirOperand mem_op = legalize_mem_operand(mf, sz, lir_bb);

    bool is_xmm = src_val->type().is_float() || src_val->type().is_vector();
    if (!is_xmm && imm_src.is_imm && imm_src.fits_i32) {
        LirOpcode op = (sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov;
        auto lir_inst = std::make_unique<LirInst>(op);
        lir_inst->add_def(mem_op);
        lir_inst->add_use(LirOperand::imm(imm_src.val, sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    } else {
        VReg src = get_vreg(src_val);
        LirOpcode op;
        if (is_xmm) {
            op = (sz == 16) ? LirOpcode::Movups : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd);
        } else {
            op = (sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov;
        }
        auto lir_inst = std::make_unique<LirInst>(op);
        lir_inst->add_def(mem_op);
        lir_inst->add_use(LirOperand::vreg(src, sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    }
}

void AArch64ISel::lower_load_indexed(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_vreg(inst.result());
    uint8_t sz = dst.size;
    MemFold mf = match_indexed_address(inst.operand(0), inst.operand(1), x64::scale_from_int(inst.scale()),
                                       inst.offset(), sz);
    LirOperand mem_op = legalize_mem_operand(mf, sz, lir_bb);

    LirOpcode op;
    if (dst.is_xmm()) {
        op = (sz == 16) ? LirOpcode::Movups : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd);
    } else {
        if (sz == 1) op = LirOpcode::Movzx8;
        else if (sz == 2) op = LirOpcode::Movzx16;
        else if (sz == 4) op = LirOpcode::Mov32;
        else op = LirOpcode::Mov;
    }

    auto lir_inst = std::make_unique<LirInst>(op);
    lir_inst->add_def(LirOperand::vreg(dst, sz));
    lir_inst->add_use(mem_op);
    lir_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(lir_inst));
}

void AArch64ISel::lower_store_indexed(const Instruction& inst, LirBlock& lir_bb) {
    const Value* src_val = inst.operand(2);
    ImmIntInfo imm_src = get_imm_int_info(src_val);
    uint8_t sz = static_cast<uint8_t>(src_val->type().size_in_bytes());
    if (sz == 0) sz = 8;
    MemFold mf = match_indexed_address(inst.operand(0), inst.operand(1), x64::scale_from_int(inst.scale()),
                                       inst.offset(), sz);
    LirOperand mem_op = legalize_mem_operand(mf, sz, lir_bb);

    bool is_xmm = src_val->type().is_float() || src_val->type().is_vector();
    if (!is_xmm && imm_src.is_imm && imm_src.fits_i32) {
        LirOpcode op = (sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov;
        auto lir_inst = std::make_unique<LirInst>(op);
        lir_inst->add_def(mem_op);
        lir_inst->add_use(LirOperand::imm(imm_src.val, sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    } else {
        VReg src = get_vreg(src_val);
        LirOpcode op;
        if (is_xmm) {
            op = (sz == 16) ? LirOpcode::Movups : ((sz == 4) ? LirOpcode::Movss : LirOpcode::Movsd);
        } else {
            op = (sz == 4) ? LirOpcode::Mov32 : LirOpcode::Mov;
        }
        auto lir_inst = std::make_unique<LirInst>(op);
        lir_inst->add_def(mem_op);
        lir_inst->add_use(LirOperand::vreg(src, sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    }
}

void AArch64ISel::lower_alloca(const Instruction& inst, LirBlock& lir_bb) {
    uint32_t size = static_cast<uint32_t>(inst.imm_i32());
    uint32_t align = static_cast<uint32_t>(inst.offset());
    if (align == 0) align = 8;

    size_t cur = lir_fn_->frame.local_frame_bytes;
    size_t aligned_cur = (cur + align - 1) & ~size_t(align - 1);
    lir_fn_->frame.local_frame_bytes = aligned_cur + size;
    if (inst.memory_type().is_tagged()) {
        lir_fn_->frame.tagged_locals.emplace_back(static_cast<uint32_t>(aligned_cur), size);
    }

    VReg dst = get_or_alloc_vreg(inst.result());
    auto lea_inst = std::make_unique<LirInst>(LirOpcode::Lea);
    lea_inst->add_def(LirOperand::vreg(dst, 8));
    lea_inst->add_use(LirOperand::local_slot(static_cast<int32_t>(aligned_cur), 8));
    lea_inst->mir_origin = &inst;
    lir_bb.append_inst(std::move(lea_inst));
}

} // namespace brass::aarch64

