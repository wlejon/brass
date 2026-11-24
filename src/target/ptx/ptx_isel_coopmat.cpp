// PtxISel: 16x16x16 Tensor Core WMMA cooperative matrix lowering.

#include "ptx_isel_intrinsics.hpp"

#include <stdexcept>
#include <string>

namespace brass::ptx {

template <uint32_t Use>
void PtxISel::Intrinsics::coop_alloc(PtxISel& isel, const brass::Instruction& inst) {
    CoopSlot slot;
    slot.use = Use;
    slot.layout = (Use == 1) ? WmmaLayout::col : WmmaLayout::row;
    slot.regs = isel.fn_->new_regs(Use == 2 ? RegClass::F32 : RegClass::B32, 8);
    isel.coop_[inst.result()] = std::move(slot);
}
template void PtxISel::Intrinsics::coop_alloc<0>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::coop_alloc<1>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::coop_alloc<2>(PtxISel&, const brass::Instruction&);

void PtxISel::Intrinsics::coop_fill_acc(PtxISel& isel, const brass::Instruction& inst) {
    auto it = isel.coop_.find(inst.operand(0));
    if (it == isel.coop_.end() || it->second.use != 2) {
        isel.malformed(inst, "ptx_coop_fill_acc requires an accumulator slot");
    }
    Operand v = isel.operand_of(inst.operand(1), Opcode::mov, 0, Type::f32, "fill value");
    for (Reg r : it->second.regs) {
        isel.emit(Inst::make(Opcode::mov, Type::f32).dst(r).src(v));
    }
}

template <uint32_t Use>
void PtxISel::Intrinsics::coop_load(PtxISel& isel, const brass::Instruction& inst) {
    auto it = isel.coop_.find(inst.operand(0));
    if (it == isel.coop_.end() || it->second.use != Use) {
        isel.malformed(inst, "coop_load slot mismatch");
    }
    int64_t col_major = 0;
    if (!isel.const_int(inst.operand(3), &col_major)) {
        isel.malformed(inst, "coop_load layout must be a compile-time constant");
    }
    int64_t src_is_f16 = 0;
    if (Use == 2 && inst.operand(4)) {
        if (!isel.const_int(inst.operand(4), &src_is_f16)) {
            isel.malformed(inst, "coop_load_acc src_is_f16 must be a compile-time constant");
        }
    }
    WmmaLayout lay = col_major ? WmmaLayout::col : WmmaLayout::row;
    it->second.layout = lay;
    WmmaMatrix mat = (Use == 0) ? WmmaMatrix::a : (Use == 1) ? WmmaMatrix::b : WmmaMatrix::c;
    StateSpace sp = isel.shared_ptrs_.count(inst.operand(1)) ? StateSpace::shared : StateSpace::global;
    Operand addr = Operand::addr(isel.reg_of(inst.operand(1), "coop ptr"), 0);
    Operand stride = isel.lane_operand(inst.operand(2), "coop stride");

    if (Use == 2 && src_is_f16) {
        std::vector<Reg> f16_regs = isel.fn_->new_regs(RegClass::B32, 4);
        isel.emit(Inst::make(Opcode::wmma_load, Type::f16).matrix(mat).layout(lay).space(sp)
                      .dst(Operand::vec(f16_regs)).src(addr).src(stride));
        for (size_t k = 0; k < 4; ++k) {
            isel.emit(Inst::make(Opcode::cvt, Type::f32).from(Type::f16)
                          .dst(it->second.regs[2 * k]).src(f16_regs[k]));
            Reg hi = isel.fn_->new_b32();
            isel.emit(Inst::make(Opcode::shr, Type::u32).dst(hi).src(f16_regs[k]).src(Operand::imm(16)));
            isel.emit(Inst::make(Opcode::cvt, Type::f32).from(Type::f16)
                          .dst(it->second.regs[2 * k + 1]).src(hi));
        }
        return;
    }

    Type ld_t = (Use == 2) ? Type::f32 : Type::f16;
    isel.emit(Inst::make(Opcode::wmma_load, ld_t).matrix(mat).layout(lay).space(sp)
                  .dst(Operand::vec(it->second.regs)).src(addr).src(stride));
}
template void PtxISel::Intrinsics::coop_load<0>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::coop_load<1>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::coop_load<2>(PtxISel&, const brass::Instruction&);

void PtxISel::Intrinsics::coop_mma(PtxISel& isel, const brass::Instruction& inst) {
    auto it_acc = isel.coop_.find(inst.operand(0));
    auto it_a   = isel.coop_.find(inst.operand(1));
    auto it_b   = isel.coop_.find(inst.operand(2));
    if (it_acc == isel.coop_.end() || it_a == isel.coop_.end() || it_b == isel.coop_.end() ||
        it_acc->second.use != 2 || it_a->second.use != 0 || it_b->second.use != 1) {
        isel.malformed(inst, "ptx_coop_mma requires (acc, a, b)");
    }
    isel.emit(Inst::make(Opcode::wmma_mma, Type::f32)
                  .ctype(Type::f32)
                  .layout(it_a->second.layout)
                  .layout_b(it_b->second.layout)
                  .dst(Operand::vec(it_acc->second.regs))
                  .src(Operand::vec(it_a->second.regs))
                  .src(Operand::vec(it_b->second.regs))
                  .src(Operand::vec(it_acc->second.regs)));
}

void PtxISel::Intrinsics::coop_store_acc(PtxISel& isel, const brass::Instruction& inst) {
    auto it = isel.coop_.find(inst.operand(0));
    if (it == isel.coop_.end() || it->second.use != 2) {
        isel.malformed(inst, "ptx_coop_store_acc requires an accumulator slot");
    }
    int64_t col_major = 0;
    int64_t dst_is_f16 = 0;
    if (!isel.const_int(inst.operand(3), &col_major) ||
        (inst.operand(4) && !isel.const_int(inst.operand(4), &dst_is_f16))) {
        isel.malformed(inst, "ptx_coop_store_acc layout and dst_is_f16 must be compile-time constants");
    }
    WmmaLayout lay = col_major ? WmmaLayout::col : WmmaLayout::row;
    StateSpace sp = isel.shared_ptrs_.count(inst.operand(1)) ? StateSpace::shared : StateSpace::global;
    Operand addr = Operand::addr(isel.reg_of(inst.operand(1), "coop ptr"), 0);
    Operand stride = isel.lane_operand(inst.operand(2), "coop stride");

    if (dst_is_f16) {
        std::vector<Reg> f16_regs = isel.fn_->new_regs(RegClass::B32, 4);
        for (size_t k = 0; k < 4; ++k) {
            Reg h0 = isel.fn_->new_b32();
            Reg h1 = isel.fn_->new_b32();
            Reg h0_lo = isel.fn_->new_b32();
            Reg h1_hi = isel.fn_->new_b32();
            isel.emit(Inst::make(Opcode::cvt, Type::f16).from(Type::f32).rnd(Rounding::rn)
                          .dst(h0).src(it->second.regs[2 * k]));
            isel.emit(Inst::make(Opcode::cvt, Type::f16).from(Type::f32).rnd(Rounding::rn)
                          .dst(h1).src(it->second.regs[2 * k + 1]));
            isel.emit(Inst::make(Opcode::and_, Type::b32).dst(h0_lo).src(h0).src(Operand::imm(0xFFFF)));
            isel.emit(Inst::make(Opcode::shl, Type::b32).dst(h1_hi).src(h1).src(Operand::imm(16)));
            isel.emit(Inst::make(Opcode::or_, Type::b32).dst(f16_regs[k]).src(h0_lo).src(h1_hi));
        }
        isel.emit(Inst::make(Opcode::wmma_store, Type::f16).matrix(WmmaMatrix::d).layout(lay).space(sp)
                      .src(addr).src(Operand::vec(f16_regs)).src(stride));
        return;
    }

    isel.emit(Inst::make(Opcode::wmma_store, Type::f32).matrix(WmmaMatrix::d).layout(lay).space(sp)
                  .src(addr).src(Operand::vec(it->second.regs)).src(stride));
}

void PtxISel::Intrinsics::coop_scale_acc(PtxISel& isel, const brass::Instruction& inst) {
    auto it = isel.coop_.find(inst.operand(0));
    if (it == isel.coop_.end() || it->second.use != 2) {
        isel.malformed(inst, "ptx_coop_scale_acc requires an accumulator slot");
    }
    Operand sc = isel.operand_of(inst.operand(1), Opcode::mul, 1, Type::f32, "scale");
    for (Reg r : it->second.regs) {
        isel.emit(Inst::make(Opcode::mul, Type::f32).dst(r).src(r).src(sc));
    }
}

void PtxISel::Intrinsics::coop_add_acc(PtxISel& isel, const brass::Instruction& inst) {
    auto it_d = isel.coop_.find(inst.operand(0));
    auto it_r = isel.coop_.find(inst.operand(1));
    if (it_d == isel.coop_.end() || it_r == isel.coop_.end() ||
        it_d->second.use != 2 || it_r->second.use != 2) {
        isel.malformed(inst, "ptx_coop_add_acc requires accumulator slots");
    }
    for (size_t k = 0; k < 8; ++k) {
        isel.emit(Inst::make(Opcode::add, Type::f32)
                      .dst(it_d->second.regs[k]).src(it_d->second.regs[k]).src(it_r->second.regs[k]));
    }
}

} // namespace brass::ptx
