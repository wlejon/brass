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

uint32_t narrow_lane_bits(brass::Type vt) noexcept {
    if (vt.kind() == TypeKind::I8x16) return 8;
    if (vt.kind() == TypeKind::I16x8) return 16;
    return 0;
}

Type vec_bit_type(brass::Type vt) noexcept {
    return narrow_lane_bits(vt) ? Type::b32 : bit_type_for(vt.element_type());
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
// Narrow i8/i16 vector lanes are held in B32 registers, so arithmetic runs
// on .s32 and explicitly masks/sign-extends back to the lane width.
Type PtxISel::vector_elem_type(const brass::Instruction& inst, brass::Type vt) const {
    if (!vt.is_vector()) malformed(inst, "operand is not a vector type");
    if (narrow_lane_bits(vt) != 0) return Type::s32;
    return signed_type_for(vt.element_type());
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

void PtxISel::lower_vector_binary(const brass::Instruction& inst, Opcode op) {
    Type t = vector_elem_type(inst, inst.type());
    const uint32_t bits = narrow_lane_bits(inst.type());
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(inst.operand(0), "operand 0");
    const auto& b = regs_of(inst.operand(1), "operand 1");
    if (a.size() != dst.size() || b.size() != dst.size()) malformed(inst, "operand lane count mismatch");

    auto sext_lane = [&](Reg r) {
        Reg tmp = fn_->new_b32();
        uint32_t sh = 32u - bits;
        emit(Inst::make(Opcode::shl, Type::b32).dst(tmp).src(r).src(Operand::imm(sh)));
        emit(Inst::make(Opcode::shr, Type::s32).dst(tmp).src(tmp).src(Operand::imm(sh)));
        return tmp;
    };
    const int64_t mask = (bits == 8) ? 0xFF : 0xFFFF;
    const bool needs_sext = (bits != 0) && (op == Opcode::min || op == Opcode::max || op == Opcode::div);

    for (size_t lane = 0; lane < dst.size(); ++lane) {
        Reg lhs = needs_sext ? sext_lane(a[lane]) : a[lane];
        Reg rhs = needs_sext ? sext_lane(b[lane]) : b[lane];
        Reg out = bits ? fn_->new_b32() : dst[lane];
        Inst i = Inst::make(op, t).dst(out).src(lhs).src(rhs);
        if (op == Opcode::mul && !is_float(t)) i.lo();
        if (op == Opcode::div && is_float(t)) i.rnd(Rounding::rn);
        emit(std::move(i));
        if (bits) emit(Inst::make(Opcode::and_, Type::b32).dst(dst[lane]).src(out).src(Operand::imm(mask)));
    }
}

void PtxISel::lower_vector_unary(const brass::Instruction& inst, Opcode op) {
    Type t = vector_elem_type(inst, inst.type());
    const uint32_t bits = narrow_lane_bits(inst.type());
    if (op == Opcode::sqrt && !is_float(t)) malformed(inst, "vsqrt requires a float vector");
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(inst.operand(0), "operand 0");
    if (a.size() != dst.size()) malformed(inst, "operand lane count mismatch");

    const int64_t mask = (bits == 8) ? 0xFF : 0xFFFF;
    for (size_t lane = 0; lane < dst.size(); ++lane) {
        Reg out = bits ? fn_->new_b32() : dst[lane];
        Inst i = Inst::make(op, t).dst(out).src(a[lane]);
        if (op == Opcode::sqrt) i.rnd(Rounding::rn);
        emit(std::move(i));
        if (bits) emit(Inst::make(Opcode::and_, Type::b32).dst(dst[lane]).src(out).src(Operand::imm(mask)));
    }
}

// Float lanes: fma.rn; integer lanes: mad.lo.
void PtxISel::lower_vector_fma(const brass::Instruction& inst) {
    Type t = vector_elem_type(inst, inst.type());
    const uint32_t bits = narrow_lane_bits(inst.type());
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(inst.operand(0), "operand 0");
    const auto& b = regs_of(inst.operand(1), "operand 1");
    const auto& c = regs_of(inst.operand(2), "operand 2");
    if (a.size() != dst.size() || b.size() != dst.size() || c.size() != dst.size())
        malformed(inst, "operand lane count mismatch");

    const int64_t mask = (bits == 8) ? 0xFF : 0xFFFF;
    for (size_t lane = 0; lane < dst.size(); ++lane) {
        Reg out = bits ? fn_->new_b32() : dst[lane];
        if (is_float(t)) {
            emit(Inst::make(Opcode::fma, t).rnd(Rounding::rn)
                     .dst(out).src(a[lane]).src(b[lane]).src(c[lane]));
        } else {
            emit(Inst::make(Opcode::mad, t).lo()
                     .dst(out).src(a[lane]).src(b[lane]).src(c[lane]));
        }
        if (bits) emit(Inst::make(Opcode::and_, Type::b32).dst(dst[lane]).src(out).src(Operand::imm(mask)));
    }
}

// ---------------------------------------------------------------------------
// Bitwise
// ---------------------------------------------------------------------------

void PtxISel::lower_vector_bitwise(const brass::Instruction& inst, Opcode op) {
    if (!inst.type().is_vector()) malformed(inst, "result is not a vector type");
    Type t = vec_bit_type(inst.type());
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
    const uint32_t bits = narrow_lane_bits(inst.type());
    Type t = vec_bit_type(inst.type());
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(inst.operand(0), "operand 0");
    if (a.size() != dst.size()) malformed(inst, "operand lane count mismatch");

    const int64_t mask = (bits == 8) ? 0xFF : 0xFFFF;
    for (size_t lane = 0; lane < dst.size(); ++lane) {
        if (bits) emit(Inst::make(Opcode::xor_, Type::b32).dst(dst[lane]).src(a[lane]).src(Operand::imm(mask)));
        else emit(Inst::make(Opcode::not_, t).dst(dst[lane]).src(a[lane]));
    }
}

// ---------------------------------------------------------------------------
// Construction and lane access
// ---------------------------------------------------------------------------

// A constant scalar is broadcast as one `mov` of the immediate per lane
// (bit_type_for keeps f32/f64, so float constants print as 0f/0d literals).
void PtxISel::lower_vbroadcast(const brass::Instruction& inst) {
    if (!inst.type().is_vector()) malformed(inst, "result is not a vector type");
    const uint32_t bits = narrow_lane_bits(inst.type());
    Type t = vec_bit_type(inst.type());
    Operand scalar = operand_of(inst.operand(0), Opcode::mov, 0, t, "scalar operand");
    if (bits) {
        const int64_t mask = (bits == 8) ? 0xFF : 0xFFFF;
        if (scalar.is_imm_int()) {
            scalar = Operand::imm(scalar.imm_int & mask);
        } else {
            Reg masked = fn_->new_b32();
            emit(Inst::make(Opcode::and_, Type::b32).dst(masked).src(scalar).src(Operand::imm(mask)));
            scalar = Operand::reg(masked);
        }
    }
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
    Type t = vec_bit_type(vec->type());
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
    const uint32_t bits = narrow_lane_bits(vec->type());
    Type t = vec_bit_type(vec->type());
    Operand scalar = operand_of(inst.operand(1), Opcode::mov, 0, t, "scalar operand");
    if (bits) {
        const int64_t mask = (bits == 8) ? 0xFF : 0xFFFF;
        if (scalar.is_imm_int()) {
            scalar = Operand::imm(scalar.imm_int & mask);
        } else {
            Reg masked = fn_->new_b32();
            emit(Inst::make(Opcode::and_, Type::b32).dst(masked).src(scalar).src(Operand::imm(mask)));
            scalar = Operand::reg(masked);
        }
    }

    for (size_t i = 0; i < dst.size(); ++i) {
        emit(Inst::make(Opcode::mov, t).dst(dst[i]).src(i == lane ? scalar : Operand::reg(src[i])));
    }
}

void PtxISel::lower_vshuffle(const brass::Instruction& inst) {
    const Value* va = inst.operand(0);
    const Value* vb = inst.operand(1);
    if (!va || !vb) malformed(inst, "missing vector operand");
    if (inst.type().element_type().size_in_bytes() < 4) {
        malformed(inst, "vshuffle is not supported for narrow integer vectors");
    }
    const auto& dst = result_regs(inst);
    const auto& a = regs_of(va, "operand 0");
    const auto& b = regs_of(vb, "operand 1");
    if (a.size() != dst.size() || b.size() != dst.size()) malformed(inst, "operand lane count mismatch");
    Type t = vec_bit_type(inst.type());
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
    Type mov_t = is_float(t) ? t : vec_bit_type(inst.type());
    for (Reg lane : result_regs(inst)) {
        emit(Inst::make(Opcode::mov, mov_t).dst(lane).src(zero_for(t)));
    }
}

} // namespace brass::ptx
