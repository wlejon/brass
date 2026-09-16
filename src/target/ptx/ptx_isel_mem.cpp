// PtxISel: loads and stores. MIR load/store/vload/vstore are always .global;
// shared memory is reached only through the explicit ptx_shared_* intrinsics
// (ptx_isel_intrinsics_mem.cpp), never by pointer provenance.

#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/mir/instruction.hpp>

#include <cstddef>

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
// {..} tuple with the matching .vN modifier. 128-bit vectors are one tuple
// (v4 of 32-bit lanes or v2 of 64-bit lanes); 256-bit vectors are two
// tuples 16 bytes apart, which keeps every access a native 16-byte ld/st.
// ---------------------------------------------------------------------------

void PtxISel::emit_vector_access(Opcode op, StateSpace space, brass::Type vt, Reg base, int32_t disp,
                                 const std::vector<Reg>& lanes) {
    auto per_tuple = static_cast<size_t>(vt.element_type().size_in_bytes() == 8 ? 2 : 4);
    Type t = type_for(vt);
    for (size_t first = 0; first < lanes.size(); first += per_tuple) {
        std::vector<Reg> tuple(lanes.begin() + static_cast<std::ptrdiff_t>(first),
                               lanes.begin() + static_cast<std::ptrdiff_t>(first + per_tuple));
        auto tuple_disp = disp + static_cast<int32_t>(first / per_tuple) * 16;
        Inst access = Inst::make(op, t).space(space).vec(per_tuple == 2 ? VecWidth::v2 : VecWidth::v4);
        if (op == Opcode::ld) access.dst(Operand::vec(std::move(tuple))).src(Operand::addr(base, tuple_disp));
        else                  access.src(Operand::addr(base, tuple_disp)).src(Operand::vec(std::move(tuple)));
        emit(std::move(access));
    }
}

void PtxISel::lower_vload(const brass::Instruction& inst) {
    if (!inst.operand(0)) malformed(inst, "missing pointer");
    if (!inst.type().is_vector()) malformed(inst, "vector load of a non-vector type");
    emit_vector_access(Opcode::ld, StateSpace::global, inst.type(),
                       reg_of(inst.operand(0), "pointer"), inst.offset(), result_regs(inst));
}

void PtxISel::lower_vstore(const brass::Instruction& inst) {
    if (!inst.operand(0) || !inst.operand(1)) malformed(inst, "missing operand");
    const Value* value = inst.operand(1);
    if (!value->type().is_vector()) malformed(inst, "vector store of a non-vector value");
    emit_vector_access(Opcode::st, StateSpace::global, value->type(),
                       reg_of(inst.operand(0), "pointer"), inst.offset(), regs_of(value, "value"));
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
