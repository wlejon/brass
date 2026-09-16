// PtxISel: loads and stores. Everything is .global for now; Stage 4 adds
// .shared arrays (ld.shared/st.shared through a state-space choice here).

#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/mir/instruction.hpp>

namespace brass::ptx {

// ---------------------------------------------------------------------------
// Scalar
// ---------------------------------------------------------------------------

void PtxISel::lower_load(const brass::Instruction& inst) {
    if (!inst.operand(0)) malformed(inst, "missing pointer");
    emit(Inst::make(Opcode::ld, type_for(inst.type())).space(StateSpace::global)
             .dst(result_reg(inst))
             .src(Operand::addr(reg_of(inst.operand(0), "pointer"), inst.offset())));
}

void PtxISel::lower_store(const brass::Instruction& inst) {
    if (!inst.operand(0) || !inst.operand(1)) malformed(inst, "missing operand");
    const Value* value = inst.operand(1);
    emit(Inst::make(Opcode::st, type_for(value->type())).space(StateSpace::global)
             .src(Operand::addr(reg_of(inst.operand(0), "pointer"), inst.offset()))
             .src(reg_of(value, "value")));
}

// ---------------------------------------------------------------------------
// Vector: a vector value is a contiguous register run, loaded/stored as a
// {..} tuple with the matching .vN modifier.
// ---------------------------------------------------------------------------

VecWidth PtxISel::vec_width_for(const brass::Instruction& inst, brass::Type vt) const {
    if (vt == brass::Type::f32x4()) return VecWidth::v4;
    if (vt == brass::Type::f64x2()) return VecWidth::v2;
    malformed(inst, "unsupported vector type (only f32x4/f64x2 are supported)");
}

void PtxISel::lower_vload(const brass::Instruction& inst) {
    if (!inst.operand(0)) malformed(inst, "missing pointer");
    VecWidth width = vec_width_for(inst, inst.type());
    emit(Inst::make(Opcode::ld, type_for(inst.type())).space(StateSpace::global).vec(width)
             .dst(Operand::vec(result_regs(inst)))
             .src(Operand::addr(reg_of(inst.operand(0), "pointer"), inst.offset())));
}

void PtxISel::lower_vstore(const brass::Instruction& inst) {
    if (!inst.operand(0) || !inst.operand(1)) malformed(inst, "missing operand");
    const Value* value = inst.operand(1);
    VecWidth width = vec_width_for(inst, value->type());
    emit(Inst::make(Opcode::st, type_for(value->type())).space(StateSpace::global).vec(width)
             .src(Operand::addr(reg_of(inst.operand(0), "pointer"), inst.offset()))
             .src(Operand::vec(regs_of(value, "value"))));
}

// ---------------------------------------------------------------------------
// Indexed: base + index * scale, computed into a fresh 64-bit register.
// A 32-bit index is sign-extended first.
// ---------------------------------------------------------------------------

Reg PtxISel::indexed_address(const brass::Instruction& inst, const Value* base, const Value* index) {
    if (!base || !index) malformed(inst, "missing operand");
    Reg base_reg = reg_of(base, "base");
    Reg idx = reg_of(index, "index");

    if (idx.cls == RegClass::B32) {
        Reg wide = fn_->new_b64();
        emit(Inst::make(Opcode::cvt, Type::s64).from(Type::s32).dst(wide).src(idx));
        idx = wide;
    }

    uint8_t scale = inst.scale();
    if (scale > 1) {
        Reg scaled = fn_->new_b64();
        switch (scale) {
            case 2: emit(Inst::make(Opcode::shl, Type::b64).dst(scaled).src(idx).src(Operand::imm(1))); break;
            case 4: emit(Inst::make(Opcode::shl, Type::b64).dst(scaled).src(idx).src(Operand::imm(2))); break;
            case 8: emit(Inst::make(Opcode::shl, Type::b64).dst(scaled).src(idx).src(Operand::imm(3))); break;
            default:
                emit(Inst::make(Opcode::mul, Type::s64).lo().dst(scaled).src(idx).src(Operand::imm(scale)));
                break;
        }
        idx = scaled;
    }

    Reg addr = fn_->new_b64();
    emit(Inst::make(Opcode::add, Type::s64).dst(addr).src(base_reg).src(idx));
    return addr;
}

void PtxISel::lower_load_indexed(const brass::Instruction& inst) {
    Reg addr = indexed_address(inst, inst.operand(0), inst.operand(1));
    emit(Inst::make(Opcode::ld, type_for(inst.type())).space(StateSpace::global)
             .dst(result_reg(inst))
             .src(Operand::addr(addr, inst.offset())));
}

void PtxISel::lower_store_indexed(const brass::Instruction& inst) {
    const Value* value = inst.operand(2);
    if (!value) malformed(inst, "missing value");
    Reg addr = indexed_address(inst, inst.operand(0), inst.operand(1));
    emit(Inst::make(Opcode::st, type_for(value->type())).space(StateSpace::global)
             .src(Operand::addr(addr, inst.offset()))
             .src(reg_of(value, "value")));
}

} // namespace brass::ptx
