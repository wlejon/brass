#include <brass/target/x64/x64_isel.hpp>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace brass::x64 {

using namespace brass::codegen;

LirOperand X64ISel::mem_operand(const MemFold& mf, uint8_t size) const {
    auto reg = [&](const Value* v, const char* role) {
        VReg r = get_vreg(v);
        if (v && !r.is_valid()) {
            throw std::logic_error(std::string("x64 isel: address ") + role + " %" + std::to_string(v->id()) +
                                   " has no register in " + std::string(mir_fn_ ? mir_fn_->name() : ""));
        }
        return r;
    };
    VReg base = reg(mf.base_val, "base");
    if (mf.index_val) {
        return LirOperand::mem(base, reg(mf.index_val, "index"), mf.scale, mf.disp, size);
    }
    return LirOperand::mem(base, mf.disp, size);
}

bool X64ISel::can_fuse_load(const Instruction* load_inst, const Instruction* user_inst) const {
    if (!load_inst || !user_inst) return false;
    if (load_inst->opcode() != Opcode::load && load_inst->opcode() != Opcode::load_indexed) {
        return false;
    }
    const Value* res = load_inst->result();
    if (!res) return false;
    // A load moves exactly its type's bits (docs/semantics.md, "Narrow
    // integers"). An i8 / i16 operation is encoded 32 or 64 bits wide, so a
    // fused narrow load would read past the value; it is never fused, and
    // gets a register through a movzx load that widen_narrow_operands uses.
    if (res->type() == Type::i8() || res->type() == Type::i16()) return false;
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

// A sum of two registers computed in another block than the memory operation,
// that is a value of its own anyway (it has a use other than as an address),
// stays one there: folding it as base + index would keep both addends live up
// to the use as well as the sum. A module's IC table base (the image address
// plus the thread's delta, computed once per function and also added to
// indices) was three values live across every call in place of one, the image
// address spilled and reloaded at each site. A sum only ever used as an
// address still folds everywhere: it is then never computed at all.
bool X64ISel::fold_two_register_sum(const Instruction* sum, const Value* ptr, const BasicBlock* user_bb) const {
    return !user_bb || sum->parent() == user_bb || address_only_.count(ptr) != 0;
}

// A value is address-only when its uses are all the address operand of a load
// or store (and it is not also the stored value, which counts as another use):
// folding it into every one leaves nothing to compute.
void X64ISel::find_address_only(const Function& mir_fn) {
    address_only_.clear();
    std::unordered_map<const Value*, uint32_t> address_uses;
    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if ((inst->opcode() == Opcode::load || inst->opcode() == Opcode::store) && inst->operand(0)) {
                address_uses[inst->operand(0)]++;
            }
        }
    }
    for (const auto& [v, n] : address_uses) {
        auto it = use_count_.find(v);
        if (it != use_count_.end() && it->second == n) address_only_.insert(v);
    }
}

bool X64ISel::materialized(const Value* v) const {
    if (!v || !v->is_instruction()) return true;
    return skipped_insts_.count(v->defining_instruction()) == 0;
}

X64ISel::MemFold X64ISel::match_address(const Value* ptr, int32_t offset, const BasicBlock* user_bb) const {
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

    if (imm1.is_imm && imm1.fits_i32 && materialized(op0)) {
        int64_t new_disp = static_cast<int64_t>(fold.disp) + imm1.val;
        if (new_disp >= INT32_MIN && new_disp <= INT32_MAX) {
            // `(a + b) + imm` from another block: the inner sum is one lea
            // with the constant (analyze_function's lea fusion skips it
            // as a value), so the address takes that lea's result whole.
            if (op0 && op0->is_instruction()) {
                const Instruction* def0 = op0->defining_instruction();
                if (def0 && def0->opcode() == Opcode::add && def0->parent() == def->parent() &&
                    !get_imm_int_info(def0->operand(0)).is_imm && !get_imm_int_info(def0->operand(1)).is_imm &&
                    !fold_two_register_sum(def0, ptr, user_bb)) {
                    return fold;
                }
            }
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
                    if (!imm_a0.is_imm && !imm_a1.is_imm && materialized(a0)) {
                        if (a1 && a1->is_instruction()) {
                            const Instruction* s_def = a1->defining_instruction();
                            if (s_def && s_def->parent() == def->parent() && s_def->opcode() == Opcode::shl &&
                                materialized(s_def->operand(0))) {
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
                        if (materialized(a1)) {
                            fold.base_val = a0;
                            fold.index_val = a1;
                            fold.scale = Scale::One;
                            fold.folded_instructions.push_back(def0);
                        }
                        return fold;
                    }
                }
            }
            return fold;
        }
    } else if (imm0.is_imm && imm0.fits_i32 && materialized(op1)) {
        int64_t new_disp = static_cast<int64_t>(fold.disp) + imm0.val;
        if (new_disp >= INT32_MIN && new_disp <= INT32_MAX) {
            fold.disp = static_cast<int32_t>(new_disp);
            fold.base_val = op1;
            fold.folded_instructions.push_back(def);
            if (imm0.def_inst) fold.folded_instructions.push_back(imm0.def_inst);
            return fold;
        }
    } else if (fold_two_register_sum(def, ptr, user_bb) && materialized(op0)) {
        if (op1 && op1->is_instruction()) {
            const Instruction* s_def = op1->defining_instruction();
            if (s_def && s_def->parent() == def->parent() && s_def->opcode() == Opcode::shl &&
                materialized(s_def->operand(0))) {
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
        if (materialized(op1)) {
            fold.base_val = op0;
            fold.index_val = op1;
            fold.scale = Scale::One;
            fold.folded_instructions.push_back(def);
        }
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
            if (imm1.is_imm && imm1.fits_i32 && materialized(def->operand(0))) {
                int64_t new_disp = static_cast<int64_t>(fold.disp) + imm1.val * sc;
                if (new_disp >= INT32_MIN && new_disp <= INT32_MAX) {
                    fold.disp = static_cast<int32_t>(new_disp);
                    fold.index_val = def->operand(0);
                    fold.folded_instructions.push_back(def);
                    if (imm1.def_inst) fold.folded_instructions.push_back(imm1.def_inst);
                }
            } else if (imm0.is_imm && imm0.fits_i32 && materialized(def->operand(1))) {
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
            if (imm1.is_imm && imm1.fits_i32 && materialized(def->operand(0))) {
                int64_t new_disp = static_cast<int64_t>(fold.disp) + imm1.val;
                if (new_disp >= INT32_MIN && new_disp <= INT32_MAX) {
                    fold.disp = static_cast<int32_t>(new_disp);
                    fold.base_val = def->operand(0);
                    fold.folded_instructions.push_back(def);
                    if (imm1.def_inst) fold.folded_instructions.push_back(imm1.def_inst);
                }
            } else if (imm0.is_imm && imm0.fits_i32 && materialized(def->operand(1))) {
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
        return mem_operand(match_address(load_inst->operand(0), load_inst->offset(), load_inst->parent()), sz);
    } else if (load_inst->opcode() == Opcode::load_indexed) {
        return mem_operand(match_indexed_address(load_inst->operand(0), load_inst->operand(1),
                                                 scale_from_int(load_inst->scale()), load_inst->offset()), sz);
    }
    return LirOperand{};
}

void X64ISel::lower_load(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_vreg(inst.result());
    uint8_t sz = dst.size;
    const LirOperand mem_op = mem_operand(match_address(inst.operand(0), inst.offset(), inst.parent()), sz);

    LirOpcode op = dst.is_xmm() ? (sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd)
                                : (sz == 1 ? LirOpcode::Movzx8 : sz == 2 ? LirOpcode::Movzx16
                                           : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov));
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
    const LirOperand mem_op = mem_operand(match_address(inst.operand(0), inst.offset(), inst.parent()), sz);

    bool is_xmm = src_val->type().is_float();
    if (!is_xmm && imm_src.is_imm && imm_src.fits_i32) {
        auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        lir_inst->add_def(mem_op);
        lir_inst->add_use(LirOperand::imm(imm_src.val, sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    } else {
        VReg src = get_vreg(src_val);
        LirOpcode op = is_xmm ? (sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
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
    const LirOperand mem_op = mem_operand(
        match_indexed_address(inst.operand(0), inst.operand(1), scale_from_int(inst.scale()), inst.offset()), sz);

    LirOpcode op = dst.is_xmm() ? (sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd)
                                : (sz == 1 ? LirOpcode::Movzx8 : sz == 2 ? LirOpcode::Movzx16
                                           : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov));
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
    const LirOperand mem_op = mem_operand(
        match_indexed_address(inst.operand(0), inst.operand(1), scale_from_int(inst.scale()), inst.offset()), sz);

    bool is_xmm = src_val->type().is_float();
    if (!is_xmm && imm_src.is_imm && imm_src.fits_i32) {
        auto lir_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
        lir_inst->add_def(mem_op);
        lir_inst->add_use(LirOperand::imm(imm_src.val, sz));
        lir_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(lir_inst));
    } else {
        VReg src = get_vreg(src_val);
        LirOpcode op = is_xmm ? (sz == 4 ? LirOpcode::Movss : LirOpcode::Movsd) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
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

// The pinned register is outside the allocator's pool (LirFunction::
// reserved_gprs), so a read is a plain copy out of it and a write a plain
// copy into it; the write also forces the prologue to save the register,
// since the module entry that writes it is an ordinary callee to whoever
// called it.
void X64ISel::lower_pinned_tls_read(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_or_alloc_vreg(inst.result());
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::vreg(dst, 8));
    mov->add_use(LirOperand::preg_gpr(kPinnedTlsGpr, 8), FixedConstraint::gpr(kPinnedTlsGpr));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
}

void X64ISel::lower_pinned_tls_write(const Instruction& inst, LirBlock& lir_bb) {
    VReg src = get_vreg(inst.operand(0));
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::preg_gpr(kPinnedTlsGpr, 8), FixedConstraint::gpr(kPinnedTlsGpr));
    mov->add_use(LirOperand::vreg(src, 8));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
    lir_fn_->forced_saved_gprs |= reg_mask(kPinnedTlsGpr);
}

void X64ISel::lower_read_sp(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_or_alloc_vreg(inst.result());
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::vreg(dst, 8));
    mov->add_use(LirOperand::preg_gpr(GPR::RSP, 8), FixedConstraint::gpr(GPR::RSP));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
}

void X64ISel::lower_alloca(const Instruction& inst, LirBlock& lir_bb) {
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

void X64ISel::lower_tagged_bitcast(const Instruction& inst, LirBlock& lir_bb) {
    VReg src = get_or_alloc_vreg(inst.operand(0));
    VReg dst = get_or_alloc_vreg(inst.result());
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::vreg(dst, 8));
    mov->add_use(LirOperand::vreg(src, 8));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
}

// The pinned TLS register and the stack pointer are read into a register by a
// plain copy (lower_pinned_tls_read, lower_read_sp). Where that copy's only
// readers are memory operands, and for the stack pointer a compare, they can
// name the register itself: the entry's stack check was `mov rax, r13; mov
// rcx, rsp; cmp rcx, [rax + limit]`, and is `cmp rsp, [r13 + limit]`. A copy
// whose readers all take the register is dropped. The stack pointer is only
// named in the block that read it, with no call or stack adjustment between,
// since it moves around a call's outgoing arguments; never as an index, which
// x64 cannot encode.
void fold_reserved_register_copies(LirFunction& fn, uint32_t pinned_mask) {
    struct Copy {
        PReg reg;
        const LirBlock* block = nullptr;
        size_t pos = 0;
        uint32_t defs = 0;
        bool kept = false;
    };
    std::unordered_map<uint32_t, Copy> copies;
    std::unordered_map<uint32_t, uint32_t> def_count;
    // A pinned register the function writes (the module entry setting up the
    // TLS block, a coroutine body re-fetching it) may differ between a copy
    // and its readers: those copies stay.
    for (const auto& block : fn.blocks) {
        for (const auto& in : block->instructions) {
            for (const auto& d : in->defs) {
                if (d.is_preg() && d.preg_val.is_gpr()) pinned_mask &= ~reg_mask(d.preg_val.as_gpr());
            }
        }
    }
    for (const auto& block : fn.blocks) {
        for (size_t i = 0; i < block->instructions.size(); ++i) {
            const LirInst& in = *block->instructions[i];
            for (const auto& d : in.defs) {
                if (d.is_vreg()) def_count[d.vreg_val.id]++;
            }
            if (in.opcode != LirOpcode::Mov || in.defs.size() != 1 || in.uses.size() != 1) continue;
            const LirOperand& d = in.defs[0];
            const LirOperand& u = in.uses[0];
            if (!d.is_vreg() || d.size != 8 || !u.is_preg() || !u.preg_val.is_gpr() || u.size != 8) continue;
            const GPR g = u.preg_val.as_gpr();
            const bool pinned = (pinned_mask & reg_mask(g)) != 0;
            if (!pinned && g != GPR::RSP) continue;
            copies[d.vreg_val.id] = Copy{u.preg_val, block.get(), i, 0, false};
        }
    }
    if (copies.empty()) return;
    for (auto& [id, c] : copies) c.defs = def_count[id];
    auto copy_of = [&](VReg v) -> Copy* {
        if (!v.is_valid()) return nullptr;
        auto it = copies.find(v.id);
        return (it == copies.end() || it->second.defs != 1) ? nullptr : &it->second;
    };
    // Whether the stack pointer still holds what the copy read at `pos`.
    auto sp_unmoved = [](const Copy& c, const LirBlock* block, size_t pos) {
        if (block != c.block || pos <= c.pos) return false;
        for (size_t k = c.pos + 1; k < pos; ++k) {
            const LirInst& m = *block->instructions[k];
            if (m.is_call() || m.opcode == LirOpcode::Push || m.opcode == LirOpcode::Pop) return false;
            for (const auto& d : m.defs) {
                if (d.is_preg() && d.preg_val.is_gpr() && d.preg_val.as_gpr() == GPR::RSP) return false;
            }
        }
        return true;
    };
    for (const auto& block : fn.blocks) {
        for (size_t i = 0; i < block->instructions.size(); ++i) {
            LirInst& in = *block->instructions[i];
            const bool compare = in.opcode == LirOpcode::Cmp || in.opcode == LirOpcode::Test;
            // A cmov's moved value (its last use; the first is the result,
            // tied to the def) is read in place, as a compare's operands are.
            const bool cmov = in.opcode == LirOpcode::Cmovcc;
            auto mem = [&](LirOperand& op) {
                LirMem& m = op.mem_val;
                if (Copy* c = copy_of(m.base_vreg)) {
                    if (c->reg.as_gpr() != GPR::RSP || sp_unmoved(*c, block.get(), i)) {
                        m.base_preg = c->reg;
                        m.base_vreg = VReg{};
                    } else {
                        c->kept = true;
                    }
                }
                if (Copy* c = copy_of(m.index_vreg)) {
                    if (c->reg.as_gpr() != GPR::RSP) {
                        m.index_preg = c->reg;
                        m.index_vreg = VReg{};
                    } else {
                        c->kept = true;
                    }
                }
            };
            for (auto& d : in.defs) {
                if (d.is_mem()) mem(d);
            }
            for (size_t k = 0; k < in.uses.size(); ++k) {
                LirOperand& u = in.uses[k];
                if (u.is_mem()) {
                    mem(u);
                    continue;
                }
                if (!u.is_vreg()) continue;
                Copy* c = copy_of(u.vreg_val);
                if (!c) continue;
                if (c->block == block.get() && c->pos == i) continue; // the copy itself
                const bool fixed = k < in.use_constraints.size() && in.use_constraints[k].has_fixed_preg;
                const bool sp_ok = c->reg.as_gpr() != GPR::RSP || sp_unmoved(*c, block.get(), i);
                const bool readable = compare || (cmov && k + 1 == in.uses.size() && k > 0);
                if (readable && !fixed && u.size == 8 && sp_ok) {
                    u = LirOperand::preg(c->reg, 8);
                } else {
                    c->kept = true;
                }
            }
        }
    }
    for (auto& [id, c] : copies) {
        if (c.defs != 1 || c.kept) continue;
        auto& ins = const_cast<LirBlock*>(c.block)->instructions;
        ins[c.pos].reset();
    }
    for (const auto& block : fn.blocks) {
        auto& ins = block->instructions;
        ins.erase(std::remove(ins.begin(), ins.end(), nullptr), ins.end());
    }
}

} // namespace brass::x64
