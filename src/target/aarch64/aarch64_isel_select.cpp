#include <brass/target/aarch64/aarch64_isel.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;
using LirCond = brass::x64::Condition;

// See x64_isel_mem.cpp: the pinned register sits outside the allocator's
// pool, so these are plain copies. `mov xd, sp` is `add xd, sp, #0`.
void AArch64ISel::lower_pinned_tls_read(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_or_alloc_vreg(inst.result());
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::vreg(dst, 8));
    mov->add_use(LirOperand::preg_aarch64_gpr(kPinnedTlsGpr, 8), FixedConstraint::aarch64_gpr(kPinnedTlsGpr));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
}

void AArch64ISel::lower_pinned_tls_write(const Instruction& inst, LirBlock& lir_bb) {
    VReg src = get_vreg(inst.operand(0));
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::preg_aarch64_gpr(kPinnedTlsGpr, 8), FixedConstraint::aarch64_gpr(kPinnedTlsGpr));
    mov->add_use(LirOperand::vreg(src, 8));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
    lir_fn_->forced_saved_gprs |= reg_mask(kPinnedTlsGpr);
}

void AArch64ISel::lower_read_sp(const Instruction& inst, LirBlock& lir_bb) {
    VReg dst = get_or_alloc_vreg(inst.result());
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::vreg(dst, 8));
    mov->add_use(LirOperand::preg_aarch64_gpr(GPR::SP, 8), FixedConstraint::aarch64_gpr(GPR::SP));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
}

void AArch64ISel::lower_tagged_bitcast(const Instruction& inst, LirBlock& lir_bb) {
    VReg src = get_or_alloc_vreg(inst.operand(0));
    VReg dst = get_or_alloc_vreg(inst.result());
    if (dst.id == src.id) return;
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::vreg(dst, 8));
    mov->add_use(LirOperand::vreg(src, 8));
    mov->mir_origin = &inst;
    lir_bb.append_inst(std::move(mov));
}

void AArch64ISel::lower_select(const Instruction& inst, LirBlock& lir_bb) {
    const Value* cond_val = inst.operand(0);
    const Value* true_val = inst.operand(1);
    const Value* false_val = inst.operand(2);

    VReg dst = get_vreg(inst.result());
    bool is_float = dst.is_xmm();
    uint8_t sz = dst.size;
    if (sz == 0) sz = 8;

    auto emit_cmp_for_select = [&]() -> LirCond {
        const Instruction* cmp_inst = cond_val ? cond_val->defining_instruction() : nullptr;
        if (fused_into(cmp_inst, inst)) {
            return emit_fused_compare(*cmp_inst, lir_bb);
        }
        VReg c_reg = get_vreg(cond_val);
        uint8_t c_sz = c_reg.size;
        if (c_sz == 0) c_sz = 4;
        auto test_inst = std::make_unique<LirInst>(c_sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
        test_inst->add_use(LirOperand::vreg(c_reg, c_sz));
        test_inst->add_use(LirOperand::vreg(c_reg, c_sz));
        lir_bb.append_inst(std::move(test_inst));
        return LirCond::NE;
    };

    if (is_float) {
        LirCond select_cond = emit_cmp_for_select();
        auto cmov = std::make_unique<LirInst>(LirOpcode::Cmovcc);
        cmov->condition = select_cond;
        cmov->add_def(LirOperand::vreg(dst, sz));
        cmov->add_use(LirOperand::vreg(get_vreg(false_val), sz));
        cmov->add_use(LirOperand::vreg(get_vreg(true_val), sz));
        cmov->mir_origin = &inst;
        lir_bb.append_inst(std::move(cmov));
        return;
    }

    auto materialize_gpr = [&](const Value* v, const ImmIntInfo& info) -> VReg {
        if (!info.is_imm) return get_vreg(v);
        VReg tmp = lir_fn_->allocate_vreg(RegClass::GPR, sz);
        if (info.val == 0) {
            auto z = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Xor32 : LirOpcode::Xor);
            z->add_def(LirOperand::vreg(tmp, sz));
            z->add_use(LirOperand::vreg(tmp, sz));
            z->add_use(LirOperand::vreg(tmp, sz));
            lir_bb.append_inst(std::move(z));
        } else if (sz == 4 || (info.val >= INT32_MIN && info.val <= INT32_MAX)) {
            auto m = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov);
            m->add_def(LirOperand::vreg(tmp, sz));
            m->add_use(LirOperand::imm(info.val, sz));
            lir_bb.append_inst(std::move(m));
        } else {
            auto m = std::make_unique<LirInst>(LirOpcode::Movabs);
            m->add_def(LirOperand::vreg(tmp, 8));
            m->add_use(LirOperand::imm(info.val, 8));
            lir_bb.append_inst(std::move(m));
        }
        return tmp;
    };

    if (sz == 4 || sz == 8) {
        const ImmIntInfo true_imm = get_imm_int_info(true_val);
        const ImmIntInfo false_imm = get_imm_int_info(false_val);
        auto eq_imm = [&](const ImmIntInfo& ii, int64_t target) -> bool {
            if (!ii.is_imm) return false;
            return (sz == 4) ? (static_cast<int32_t>(ii.val) == static_cast<int32_t>(target)) : (ii.val == target);
        };
        const bool t0 = eq_imm(true_imm, 0), t1 = eq_imm(true_imm, 1), tm1 = eq_imm(true_imm, -1);
        const bool f0 = eq_imm(false_imm, 0), f1 = eq_imm(false_imm, 1), fm1 = eq_imm(false_imm, -1);

        if ((t1 && f0) || (t0 && f1)) {
            LirCond cond = emit_cmp_for_select();
            auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
            setcc->condition = t1 ? cond : x64::invert(cond);
            setcc->add_def(LirOperand::vreg(dst, sz));
            setcc->mir_origin = &inst;
            lir_bb.append_inst(std::move(setcc));
            return;
        }
        if ((tm1 && f0) || (t0 && fm1)) {
            LirCond cond = emit_cmp_for_select();
            auto csinv = std::make_unique<LirInst>(LirOpcode::Csinv);
            csinv->condition = tm1 ? x64::invert(cond) : cond;
            csinv->add_def(LirOperand::vreg(dst, sz));
            csinv->add_use(LirOperand::imm(0, sz));
            csinv->add_use(LirOperand::imm(0, sz));
            csinv->mir_origin = &inst;
            lir_bb.append_inst(std::move(csinv));
            return;
        }

        auto append_csel_imm0 = [&](LirOpcode op, LirCond c, VReg reg_arm) {
            auto li = std::make_unique<LirInst>(op);
            li->condition = c;
            li->add_def(LirOperand::vreg(dst, sz));
            li->add_use(LirOperand::imm(0, sz));
            li->add_use(LirOperand::vreg(reg_arm, sz));
            li->mir_origin = &inst;
            lir_bb.append_inst(std::move(li));
        };

        if (f0 && !true_imm.is_imm) {
            append_csel_imm0(LirOpcode::Cmovcc, emit_cmp_for_select(), get_vreg(true_val));
            return;
        }
        if (t0 && !false_imm.is_imm) {
            append_csel_imm0(LirOpcode::Cmovcc, x64::invert(emit_cmp_for_select()), get_vreg(false_val));
            return;
        }
        if (f1 && !true_imm.is_imm) {
            append_csel_imm0(LirOpcode::Csinc, emit_cmp_for_select(), get_vreg(true_val));
            return;
        }
        if (t1 && !false_imm.is_imm) {
            append_csel_imm0(LirOpcode::Csinc, x64::invert(emit_cmp_for_select()), get_vreg(false_val));
            return;
        }
        if (fm1 && !true_imm.is_imm) {
            append_csel_imm0(LirOpcode::Csinv, emit_cmp_for_select(), get_vreg(true_val));
            return;
        }
        if (tm1 && !false_imm.is_imm) {
            append_csel_imm0(LirOpcode::Csinv, x64::invert(emit_cmp_for_select()), get_vreg(false_val));
            return;
        }

        auto is_skipped_val = [&](const Value* u) -> bool {
            return u && u->is_instruction() && skipped_insts_.count(u->defining_instruction()) != 0;
        };

        auto match_cs_op = [&](const Value* v, LirOpcode& out_op, VReg& out_x) -> bool {
            if (!v || !v->is_instruction()) return false;
            const Instruction* d = v->defining_instruction();
            if (!d || d->parent() != inst.parent() || skipped_insts_.count(d)) return false;
            if (d->result()->type().is_gc_root() || d->result()->type().size_in_bytes() != sz) return false;
            auto it = use_count_.find(v);
            if (it == use_count_.end() || it->second != 1) return false;
            if (d->opcode() == Opcode::add) {
                ImmIntInfo i0 = get_imm_int_info(d->operand(0));
                ImmIntInfo i1 = get_imm_int_info(d->operand(1));
                const Value* base_v = eq_imm(i1, 1) ? d->operand(0) : (eq_imm(i0, 1) ? d->operand(1) : nullptr);
                if (base_v && !is_skipped_val(base_v) && base_v->type().size_in_bytes() == sz) {
                    out_op = LirOpcode::Csinc;
                    out_x = get_vreg(base_v);
                    return out_x.is_valid();
                }
            } else if (d->opcode() == Opcode::neg) {
                const Value* x_v = d->operand(0);
                if (x_v && !is_skipped_val(x_v) && x_v->type().size_in_bytes() == sz) {
                    out_op = LirOpcode::Csneg;
                    out_x = get_vreg(x_v);
                    return out_x.is_valid();
                }
            } else if (d->opcode() == Opcode::sub && eq_imm(get_imm_int_info(d->operand(0)), 0)) {
                const Value* x_v = d->operand(1);
                if (x_v && !is_skipped_val(x_v) && x_v->type().size_in_bytes() == sz) {
                    out_op = LirOpcode::Csneg;
                    out_x = get_vreg(x_v);
                    return out_x.is_valid();
                }
            } else if (d->opcode() == Opcode::not_) {
                const Value* x_v = d->operand(0);
                if (x_v && !is_skipped_val(x_v) && x_v->type().size_in_bytes() == sz) {
                    out_op = LirOpcode::Csinv;
                    out_x = get_vreg(x_v);
                    return out_x.is_valid();
                }
            } else if (d->opcode() == Opcode::xor_) {
                ImmIntInfo i0 = get_imm_int_info(d->operand(0));
                ImmIntInfo i1 = get_imm_int_info(d->operand(1));
                const Value* x_v = eq_imm(i1, -1) ? d->operand(0) : (eq_imm(i0, -1) ? d->operand(1) : nullptr);
                if (x_v && !is_skipped_val(x_v) && x_v->type().size_in_bytes() == sz) {
                    out_op = LirOpcode::Csinv;
                    out_x = get_vreg(x_v);
                    return out_x.is_valid();
                }
            }
            return false;
        };

        LirOpcode cs_op = LirOpcode::Cmovcc;
        VReg cs_x{};
        if (match_cs_op(false_val, cs_op, cs_x)) {
            VReg t_reg = materialize_gpr(true_val, true_imm);
            LirCond cond = emit_cmp_for_select();
            auto li = std::make_unique<LirInst>(cs_op);
            li->condition = cond;
            li->add_def(LirOperand::vreg(dst, sz));
            li->add_use(LirOperand::vreg(cs_x, sz));
            li->add_use(LirOperand::vreg(t_reg, sz));
            li->mir_origin = &inst;
            lir_bb.append_inst(std::move(li));
            return;
        }
        if (match_cs_op(true_val, cs_op, cs_x)) {
            VReg f_reg = materialize_gpr(false_val, false_imm);
            LirCond cond = emit_cmp_for_select();
            auto li = std::make_unique<LirInst>(cs_op);
            li->condition = x64::invert(cond);
            li->add_def(LirOperand::vreg(dst, sz));
            li->add_use(LirOperand::vreg(cs_x, sz));
            li->add_use(LirOperand::vreg(f_reg, sz));
            li->mir_origin = &inst;
            lir_bb.append_inst(std::move(li));
            return;
        }
    }

    const VReg false_vreg = materialize_gpr(false_val, get_imm_int_info(false_val));
    const VReg src_vreg = materialize_gpr(true_val, get_imm_int_info(true_val));
    LirCond select_cond = emit_cmp_for_select();

    auto cmov = std::make_unique<LirInst>(LirOpcode::Cmovcc);
    cmov->condition = select_cond;
    cmov->add_def(LirOperand::vreg(dst, sz));
    cmov->add_use(LirOperand::vreg(false_vreg, sz));
    cmov->add_use(LirOperand::vreg(src_vreg, sz));
    cmov->mir_origin = &inst;
    lir_bb.append_inst(std::move(cmov));
}

} // namespace brass::aarch64
