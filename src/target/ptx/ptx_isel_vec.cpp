// PtxISel: vector MIR ops. PTX has no SIMD arithmetic on registers, so every
// vector value is a contiguous run of scalar registers (allocate_registers)
// and each vector op becomes one scalar instruction per lane. Type suffixes
// come from the element type through the ptx_ir tables, exactly as for the
// scalar ops (vadd on f32x4 is four add.f32, on i32x4 four add.s32).
//
// vload/vstore stay in ptx_isel_mem.cpp: those are the only vector ops with
// a native PTX form (ld/st .v2/.v4).

#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/mir/instruction.hpp>

namespace brass::ptx {

namespace {

// Zero immediate for a lane of the given element suffix.
Operand zero_for(Type t) {
    if (t == Type::f32) return Operand::imm_f32(0.0f);
    if (t == Type::f64) return Operand::imm_f64(0.0);
    return Operand::imm(0);
}

// vshuffle uses the x64 shufps/shufpd mask encoding (the interpreter and
// the x64 backend agree on it): per 128-bit half, a 4-lane result takes
// lanes 0..1 from `a` and 2..3 from `b`, each selected by a 2-bit field of
// the mask; a 2-lane result takes lane 0 from `a` and lane 1 from `b`, each
// selected by one bit. Returns (source is b?, source lane).
struct ShuffleSel { bool from_b; size_t lane; };

ShuffleSel shuffle_select(size_t lanes, uint32_t mask, size_t out_lane) {
    if (lanes == 4 || lanes == 8) {
        // 4 lanes per 128-bit half; the same 8-bit mask applies to each half.
        size_t half = out_lane / 4;
        size_t in_half = out_lane % 4;
        size_t sel = (mask >> (2 * in_half)) & 3;
        return ShuffleSel{in_half >= 2, half * 4 + sel};
    }
    // 2 lanes per 128-bit half (f64x2 / i64x2 / f64x4 / i64x4).
    size_t half = out_lane / 2;
    size_t in_half = out_lane % 2;
    size_t sel = (mask >> (half * 2 + in_half)) & 1;
    return ShuffleSel{in_half == 1, half * 2 + sel};
}

} // namespace

// Element suffix for the arithmetic form of the vector: f32/f64 or s32/s64.
Type PtxISel::vector_elem_type(const brass::Instruction& inst, brass::Type vt) const {
    if (!vt.is_vector()) malformed(inst, "operand is not a vector type");
    return signed_type_for(vt.element_type());
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

void PtxISel::lower_vector_binary(const brass::Instruction& inst, Opcode op) {
    Type t = vector_elem_type(inst, inst.type());
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(inst.operand(0), "operand 0");
    const auto& b = regs_of(inst.operand(1), "operand 1");
    if (a.size() != dst.size() || b.size() != dst.size()) malformed(inst, "operand lane count mismatch");

    for (size_t lane = 0; lane < dst.size(); ++lane) {
        Inst i = Inst::make(op, t).dst(dst[lane]).src(a[lane]).src(b[lane]);
        if (op == Opcode::mul && !is_float(t)) i.lo();
        if (op == Opcode::div && is_float(t)) i.rnd(Rounding::rn);
        emit(std::move(i));
    }
}

void PtxISel::lower_vector_unary(const brass::Instruction& inst, Opcode op) {
    Type t = vector_elem_type(inst, inst.type());
    if (op == Opcode::sqrt && !is_float(t)) malformed(inst, "vsqrt requires a float vector");
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(inst.operand(0), "operand 0");
    if (a.size() != dst.size()) malformed(inst, "operand lane count mismatch");

    for (size_t lane = 0; lane < dst.size(); ++lane) {
        Inst i = Inst::make(op, t).dst(dst[lane]).src(a[lane]);
        if (op == Opcode::sqrt) i.rnd(Rounding::rn);
        emit(std::move(i));
    }
}

// Float lanes: fma.rn; integer lanes: mad.lo.
void PtxISel::lower_vector_fma(const brass::Instruction& inst) {
    Type t = vector_elem_type(inst, inst.type());
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(inst.operand(0), "operand 0");
    const auto& b = regs_of(inst.operand(1), "operand 1");
    const auto& c = regs_of(inst.operand(2), "operand 2");
    if (a.size() != dst.size() || b.size() != dst.size() || c.size() != dst.size())
        malformed(inst, "operand lane count mismatch");

    for (size_t lane = 0; lane < dst.size(); ++lane) {
        if (is_float(t)) {
            emit(Inst::make(Opcode::fma, t).rnd(Rounding::rn)
                     .dst(dst[lane]).src(a[lane]).src(b[lane]).src(c[lane]));
        } else {
            emit(Inst::make(Opcode::mad, t).lo()
                     .dst(dst[lane]).src(a[lane]).src(b[lane]).src(c[lane]));
        }
    }
}

// ---------------------------------------------------------------------------
// Bitwise
// ---------------------------------------------------------------------------

void PtxISel::lower_vector_bitwise(const brass::Instruction& inst, Opcode op) {
    if (!inst.type().is_vector()) malformed(inst, "result is not a vector type");
    Type t = bit_type_for(inst.type().element_type());
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(inst.operand(0), "operand 0");
    const auto& b = regs_of(inst.operand(1), "operand 1");
    if (a.size() != dst.size() || b.size() != dst.size()) malformed(inst, "operand lane count mismatch");

    for (size_t lane = 0; lane < dst.size(); ++lane) {
        emit(Inst::make(op, t).dst(dst[lane]).src(a[lane]).src(b[lane]));
    }
}

void PtxISel::lower_vector_not(const brass::Instruction& inst) {
    if (!inst.type().is_vector()) malformed(inst, "result is not a vector type");
    Type t = bit_type_for(inst.type().element_type());
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(inst.operand(0), "operand 0");
    if (a.size() != dst.size()) malformed(inst, "operand lane count mismatch");

    for (size_t lane = 0; lane < dst.size(); ++lane) {
        emit(Inst::make(Opcode::not_, t).dst(dst[lane]).src(a[lane]));
    }
}

// ---------------------------------------------------------------------------
// Construction and lane access
// ---------------------------------------------------------------------------

void PtxISel::lower_vbroadcast(const brass::Instruction& inst) {
    if (!inst.type().is_vector()) malformed(inst, "result is not a vector type");
    Type t = bit_type_for(inst.type().element_type());
    Reg scalar = reg_of(inst.operand(0), "scalar operand");
    for (Reg lane : result_regs(inst)) {
        emit(Inst::make(Opcode::mov, t).dst(lane).src(scalar));
    }
}

void PtxISel::lower_vextract_lane(const brass::Instruction& inst) {
    const Value* vec = inst.operand(0);
    if (!vec) malformed(inst, "missing vector operand");
    const auto& src = regs_of(vec, "vector operand");
    size_t lane = inst.lane();
    if (lane >= src.size()) malformed(inst, "lane index out of range");
    Type t = bit_type_for(vec->type().element_type());
    emit(Inst::make(Opcode::mov, t).dst(result_reg(inst)).src(src[lane]));
}

void PtxISel::lower_vinsert_lane(const brass::Instruction& inst) {
    const Value* vec = inst.operand(0);
    if (!vec) malformed(inst, "missing vector operand");
    const auto& dst = result_regs(inst);
    const auto& src = regs_of(vec, "vector operand");
    if (src.size() != dst.size()) malformed(inst, "operand lane count mismatch");
    size_t lane = inst.lane();
    if (lane >= dst.size()) malformed(inst, "lane index out of range");
    Reg scalar = reg_of(inst.operand(1), "scalar operand");
    Type t = bit_type_for(vec->type().element_type());

    for (size_t i = 0; i < dst.size(); ++i) {
        emit(Inst::make(Opcode::mov, t).dst(dst[i]).src(i == lane ? scalar : src[i]));
    }
}

void PtxISel::lower_vshuffle(const brass::Instruction& inst) {
    const Value* va = inst.operand(0);
    const Value* vb = inst.operand(1);
    if (!va || !vb) malformed(inst, "missing vector operand");
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(va, "operand 0");
    const auto& b = regs_of(vb, "operand 1");
    if (a.size() != dst.size() || b.size() != dst.size()) malformed(inst, "operand lane count mismatch");
    Type t = bit_type_for(inst.type().element_type());
    uint32_t mask = inst.shuffle_mask();

    // The result run is fresh (SSA), so no source register is overwritten
    // before it is read and lanes can be moved in order.
    for (size_t lane = 0; lane < dst.size(); ++lane) {
        ShuffleSel sel = shuffle_select(dst.size(), mask, lane);
        const auto& from = sel.from_b ? b : a;
        emit(Inst::make(Opcode::mov, t).dst(dst[lane]).src(from[sel.lane]));
    }
}

void PtxISel::lower_vzero(const brass::Instruction& inst) {
    if (!inst.type().is_vector()) malformed(inst, "result is not a vector type");
    Type t = type_for(inst.type().element_type());
    Type mov_t = is_float(t) ? t : bit_type_for(inst.type().element_type());
    for (Reg lane : result_regs(inst)) {
        emit(Inst::make(Opcode::mov, mov_t).dst(lane).src(zero_for(t)));
    }
}

} // namespace brass::ptx
