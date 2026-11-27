// PtxISel: Ampere/Ada+ tensor-pipeline primitives:
//   cvta.to.shared, cp.async.{cg,ca,commit_group,wait_group,wait_all},
//   ldmatrix.sync.aligned.m8n8.x{1,2,4}{.trans}.shared.b16,
//   mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32.

#include "ptx_isel_intrinsics.hpp"

#include <brass/mir/opcodes.hpp>

#include <vector>

namespace brass::ptx {

Operand PtxISel::Intrinsics::shared_addr_operand(
    PtxISel& isel,
    const brass::Instruction& inst,
    const Value* addr_val,
    const char* what) {
    Reg r = isel.reg_of(addr_val, what);
    if (r.cls != RegClass::B32 && r.cls != RegClass::B64) {
        isel.malformed(inst, "shared address must be i32 or i64/ptr");
    }
    return Operand::addr(r, 0);
}

template <Type T>
void PtxISel::Intrinsics::cvta_to_shared(PtxISel& isel, const brass::Instruction& inst) {
    Reg dst = isel.result_reg(inst);
    Reg src = isel.reg_of(inst.operand(0), "shared generic pointer");
    if (src.cls != RegClass::B64) isel.malformed(inst, "ptx_cvta_to_shared requires a 64-bit pointer operand");
    if constexpr (T == Type::u32) {
        // Under .address_size 64, cvta.to.shared requires .u64; narrow to .u32 for shared offsets.
        Reg wide = isel.fn_->new_b64();
        isel.emit(Inst::make(Opcode::cvta, Type::u64).space(StateSpace::shared).to_space().dst(wide).src(src));
        isel.emit(Inst::make(Opcode::cvt, Type::u32).from(Type::u64).dst(dst).src(wide));
    } else {
        isel.emit(Inst::make(Opcode::cvta, Type::u64).space(StateSpace::shared).to_space().dst(dst).src(src));
    }
}
template void PtxISel::Intrinsics::cvta_to_shared<Type::u32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::cvta_to_shared<Type::u64>(PtxISel&, const brass::Instruction&);

template <uint8_t Bytes, bool Cg>
void PtxISel::Intrinsics::cp_async(PtxISel& isel, const brass::Instruction& inst) {
    if (!inst.operand(0) || !inst.operand(1)) isel.malformed(inst, "cp_async requires shared and global pointers");
    Operand dst_addr = PtxISel::Intrinsics::shared_addr_operand(isel, inst, inst.operand(0), "cp.async shared dst");
    Operand src_addr = Operand::addr(isel.reg_of(inst.operand(1), "cp.async global src"), 0);
    Inst out = Inst::make(Opcode::cp_async).async_bytes(Bytes);
    if constexpr (Cg) out.cg();
    out.src(dst_addr).src(src_addr).src(Operand::imm(Bytes));
    if (inst.operand_count() >= 3 && inst.operand(2)) {
        out.src(isel.lane_operand(inst.operand(2), "cp.async src_size"));
    }
    isel.emit(std::move(out));
}
template void PtxISel::Intrinsics::cp_async<16, true>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::cp_async<4, false>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::cp_async<8, false>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::cp_async<16, false>(PtxISel&, const brass::Instruction&);

void PtxISel::Intrinsics::cp_async_commit(PtxISel& isel, const brass::Instruction&) {
    isel.emit(Inst::make(Opcode::cp_async_commit));
}

void PtxISel::Intrinsics::cp_async_wait_group(PtxISel& isel, const brass::Instruction& inst) {
    int64_t n = 0;
    if (!isel.const_int(inst.operand(0), &n) || n < 0 || n > 63) {
        isel.malformed(inst, "ptx_cp_async_wait_group requires compile-time integer constant in [0, 63]");
    }
    isel.emit(Inst::make(Opcode::cp_async_wait_group).src(Operand::imm(n)));
}

void PtxISel::Intrinsics::cp_async_wait_all(PtxISel& isel, const brass::Instruction&) {
    isel.emit(Inst::make(Opcode::cp_async_wait_all));
}

template <uint8_t Num, bool Trans>
void PtxISel::Intrinsics::ldmatrix(PtxISel& isel, const brass::Instruction& inst) {
    if (!inst.operand(0)) isel.malformed(inst, "ldmatrix requires shared address");
    Operand src_addr = PtxISel::Intrinsics::shared_addr_operand(isel, inst, inst.operand(0), "ldmatrix shared addr");

    if constexpr (Num == 4) {
        const std::vector<Reg>& dsts = isel.result_regs(inst);
        if (dsts.size() != 4) isel.malformed(inst, "ptx_ldmatrix_x4 requires i32x4 result");
        Inst out = Inst::make(Opcode::ldmatrix).matrix_num(4);
        if constexpr (Trans) out.trans();
        out.dst(Operand::vec(dsts)).src(src_addr);
        isel.emit(std::move(out));
    } else if constexpr (Num == 2) {
        Reg dst64 = isel.result_reg(inst);
        if (dst64.cls != RegClass::B64) isel.malformed(inst, "ptx_ldmatrix_x2 requires i64 result");
        std::vector<Reg> halves = isel.fn_->new_regs(RegClass::B32, 2);
        Inst out = Inst::make(Opcode::ldmatrix).matrix_num(2);
        if constexpr (Trans) out.trans();
        out.dst(Operand::vec(halves)).src(src_addr);
        isel.emit(std::move(out));
        isel.emit(Inst::make(Opcode::mov, Type::b64).dst(dst64).src(Operand::vec(halves)));
    } else {
        Reg dst32 = isel.result_reg(inst);
        if (dst32.cls != RegClass::B32) isel.malformed(inst, "ptx_ldmatrix_x1 requires i32 result");
        Inst out = Inst::make(Opcode::ldmatrix).matrix_num(1);
        if constexpr (Trans) out.trans();
        out.dst(Operand::vec({dst32})).src(src_addr);
        isel.emit(std::move(out));
    }
}
template void PtxISel::Intrinsics::ldmatrix<1, false>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::ldmatrix<1, true>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::ldmatrix<2, false>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::ldmatrix<2, true>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::ldmatrix<4, false>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::ldmatrix<4, true>(PtxISel&, const brass::Instruction&);

template <Type AB>
void PtxISel::Intrinsics::mma_m16n8k16(PtxISel& isel, const brass::Instruction& inst) {
    if (inst.operand_count() < 4) isel.malformed(inst, "mma_m16n8k16 requires (a:i32x4, b0:i32, b1:i32, c:f32x4)");
    const std::vector<Reg>& d_regs = isel.result_regs(inst);
    const std::vector<Reg>& a_regs = isel.regs_of(inst.operand(0), "a");
    Reg b0 = isel.reg_of(inst.operand(1), "b0");
    Reg b1 = isel.reg_of(inst.operand(2), "b1");
    const std::vector<Reg>& c_regs = isel.regs_of(inst.operand(3), "c");
    if (d_regs.size() != 4 || a_regs.size() != 4 || c_regs.size() != 4) {
        isel.malformed(inst, "mma_m16n8k16 expects f32x4 result/c and i32x4 a");
    }
    isel.emit(Inst::make(Opcode::mma_sync, Type::f32)
                  .ab_type(AB)
                  .ctype(Type::f32)
                  .dst(Operand::vec(d_regs))
                  .src(Operand::vec(a_regs))
                  .src(Operand::vec({b0, b1}))
                  .src(Operand::vec(c_regs)));
}
template void PtxISel::Intrinsics::mma_m16n8k16<Type::f16>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::mma_m16n8k16<Type::bf16>(PtxISel&, const brass::Instruction&);

} // namespace brass::ptx
