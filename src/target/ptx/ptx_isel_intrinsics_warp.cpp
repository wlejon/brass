// PtxISel intrinsic rules: barriers, warp shuffles, atomics and the wide /
// fused integer multiplies. See ptx_isel_intrinsics.hpp for the contract.

#include "ptx_isel_intrinsics.hpp"

namespace brass::ptx {

// ---------------------------------------------------------------------------
// Barriers
// ---------------------------------------------------------------------------

// bar.sync 0
void PtxISel::Intrinsics::bar_sync(PtxISel& isel, const brass::Instruction&) {
    isel.emit(Inst::make(Opcode::bar).sync().src(Operand::imm(0)));
}

// bar.sync id          (id: immediate when constant, else a B32 register)
void PtxISel::Intrinsics::bar_sync_id(PtxISel& isel, const brass::Instruction& inst) {
    isel.emit(Inst::make(Opcode::bar).sync().src(isel.lane_operand(inst.operand(0), "barrier id")));
}

// bar.sync id, nthreads   (nthreads must be a multiple of the warp size)
void PtxISel::Intrinsics::bar_sync_count(PtxISel& isel, const brass::Instruction& inst) {
    isel.emit(Inst::make(Opcode::bar).sync()
                  .src(isel.lane_operand(inst.operand(0), "barrier id"))
                  .src(isel.lane_operand(inst.operand(1), "barrier thread count")));
}

// ---------------------------------------------------------------------------
// Shuffles. The clamp operand follows what nvcc emits for the CUDA
// __shfl_*_sync family: 0 for .up (no lower segment bound), 0x1f otherwise.
// ---------------------------------------------------------------------------

// shfl.sync.down.b32 d, value, delta, 0x1f, mask     for (mask, value, delta)
void PtxISel::Intrinsics::shfl_down_sync_f32(PtxISel& isel, const brass::Instruction& inst) {
    isel.emit(Inst::make(Opcode::shfl, Type::b32).sync().shfl(ShflMode::down)
                  .dst(isel.result_reg(inst))
                  .src(isel.reg_of(inst.operand(1), "value"))
                  .src(isel.lane_operand(inst.operand(2), "lane delta"))
                  .src(Operand::imm(0x1f))
                  .src(isel.reg_of(inst.operand(0), "member mask")));
}

// shfl.sync.<mode>.b32 d, value, delta_or_lane, clamp, 0xffffffff   for (value, delta)
template <ShflMode M>
void PtxISel::Intrinsics::shfl(PtxISel& isel, const brass::Instruction& inst) {
    const Value* value = inst.operand(0);
    if (!value) isel.malformed(inst, "missing value");
    if (value->type().size_in_bytes() != 4) isel.malformed(inst, "shfl value must be a 32-bit f32/i32");
    isel.emit(Inst::make(Opcode::shfl, Type::b32).sync().shfl(M)
                  .dst(isel.result_reg(inst))
                  .src(isel.reg_of(value, "value"))
                  .src(isel.lane_operand(inst.operand(1), "lane delta"))
                  .src(Operand::imm(M == ShflMode::up ? 0 : 0x1f))
                  .src(Operand::imm(0xffffffffLL)));
}
template void PtxISel::Intrinsics::shfl<ShflMode::down>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shfl<ShflMode::up>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shfl<ShflMode::bfly>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shfl<ShflMode::idx>(PtxISel&, const brass::Instruction&);

// ---------------------------------------------------------------------------
// Atomics: atom.<space>.<op>.<type> old, [ptr], value. The suffix follows
// the value type: f32/f64 as is, integers as .u32/.u64 for add, .s32/.s64
// for min/max and .b32/.b64 for exch.
// ---------------------------------------------------------------------------

template <AtomOp A, StateSpace S>
void PtxISel::Intrinsics::atom(PtxISel& isel, const brass::Instruction& inst) {
    const Value* value = inst.operand(1);
    if (!inst.operand(0) || !value) isel.malformed(inst, "missing argument");
    brass::Type vt = value->type();
    Type t = (A == AtomOp::min || A == AtomOp::max) ? signed_type_for(vt)
           : (A == AtomOp::exch)                     ? bit_type_for(vt)
                                                     : type_for(vt);
    if constexpr (A != AtomOp::add) {
        if (is_float(t)) isel.malformed(inst, "only atom.add supports float values");
    }
    isel.emit(Inst::make(Opcode::atom, t).space(S).atom(A)
                  .dst(isel.result_reg(inst))
                  .src(Operand::addr(isel.reg_of(inst.operand(0), "pointer"), 0))
                  .src(isel.operand_of(value, Opcode::atom, 1, t, "value")));
}
template void PtxISel::Intrinsics::atom<AtomOp::add, StateSpace::global>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::atom<AtomOp::min, StateSpace::global>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::atom<AtomOp::max, StateSpace::global>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::atom<AtomOp::exch, StateSpace::global>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::atom<AtomOp::add, StateSpace::shared>(PtxISel&, const brass::Instruction&);

// ---------------------------------------------------------------------------
// Wide and fused integer multiplies
// ---------------------------------------------------------------------------

// The multiply sources are checked against the narrow (.u32/.s32) type, so a
// .wide immediate always fits the operand width.

// mul.wide.u32 %rd, %r, %r    (i32, i32) -> i64
void PtxISel::Intrinsics::mul_wide_u32(PtxISel& isel, const brass::Instruction& inst) {
    isel.emit(Inst::make(Opcode::mul, Type::u32).wide()
                  .dst(isel.result_reg(inst))
                  .src(isel.operand_of(inst.operand(0), Opcode::mul, 0, Type::u32, "argument 0"))
                  .src(isel.operand_of(inst.operand(1), Opcode::mul, 1, Type::u32, "argument 1")));
}

// mul.wide.s32 %rd, %r, %r    (i32, i32) -> i64
void PtxISel::Intrinsics::mul_wide_s32(PtxISel& isel, const brass::Instruction& inst) {
    isel.emit(Inst::make(Opcode::mul, Type::s32).wide()
                  .dst(isel.result_reg(inst))
                  .src(isel.operand_of(inst.operand(0), Opcode::mul, 0, Type::s32, "argument 0"))
                  .src(isel.operand_of(inst.operand(1), Opcode::mul, 1, Type::s32, "argument 1")));
}

// mul.hi.u32 %r, %r, %r       (i32, i32) -> i32
void PtxISel::Intrinsics::mul_hi_u32(PtxISel& isel, const brass::Instruction& inst) {
    isel.emit(Inst::make(Opcode::mul, Type::u32).hi()
                  .dst(isel.result_reg(inst))
                  .src(isel.operand_of(inst.operand(0), Opcode::mul, 0, Type::u32, "argument 0"))
                  .src(isel.operand_of(inst.operand(1), Opcode::mul, 1, Type::u32, "argument 1")));
}

// mad.lo.u32 %r, a, b, c      (i32, i32, i32) -> i32
void PtxISel::Intrinsics::mad_lo_u32(PtxISel& isel, const brass::Instruction& inst) {
    isel.emit(Inst::make(Opcode::mad, Type::u32).lo()
                  .dst(isel.result_reg(inst))
                  .src(isel.operand_of(inst.operand(0), Opcode::mad, 0, Type::u32, "argument 0"))
                  .src(isel.operand_of(inst.operand(1), Opcode::mad, 1, Type::u32, "argument 1"))
                  .src(isel.operand_of(inst.operand(2), Opcode::mad, 2, Type::u32, "argument 2")));
}

} // namespace brass::ptx
