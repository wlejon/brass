// PtxISel: constants, arithmetic, bitwise, shifts, comparisons, select and
// conversions. Suffix selection goes through the ptx_ir tables only.

#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/mir/instruction.hpp>

namespace brass::ptx {

namespace {

// Comparison suffix: floats compare as .f32/.f64, unsigned MIR ops as
// .u32/.u64, everything else as .s32/.s64.
Type compare_type(brass::Type operand_type, bool is_unsigned) {
    Type t = type_for(operand_type);
    if (is_float(t)) return t;
    return is_unsigned ? t : signed_type_for(operand_type);
}

// Float `ne` is unordered (true when either side is NaN), matching the x64
// lowering; every other float comparison is ordered.
CmpOp float_compare_op(CmpOp cmp) {
    return cmp == CmpOp::ne ? CmpOp::neu : cmp;
}

} // namespace

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

void PtxISel::lower_iconst(const brass::Instruction& inst, Type t) {
    int64_t v = (t == Type::b32) ? static_cast<int64_t>(inst.imm_i32()) : inst.imm_i64();
    emit(Inst::make(Opcode::mov, t).dst(result_reg(inst)).src(Operand::imm(v)));
}

void PtxISel::lower_fconst(const brass::Instruction& inst) {
    Type t = type_for(inst.type());
    Operand imm = (t == Type::f32) ? Operand::imm_f32(static_cast<float>(inst.imm_f64()))
                                   : Operand::imm_f64(inst.imm_f64());
    emit(Inst::make(Opcode::mov, t).dst(result_reg(inst)).src(imm));
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

void PtxISel::lower_binary(const brass::Instruction& inst, Opcode op) {
    emit(Inst::make(op, signed_type_for(inst.type()))
             .dst(result_reg(inst))
             .src(reg_of(inst.operand(0), "operand 0"))
             .src(reg_of(inst.operand(1), "operand 1")));
}

void PtxISel::lower_mul(const brass::Instruction& inst) {
    Type t = signed_type_for(inst.type());
    Inst mul = Inst::make(Opcode::mul, t)
                   .dst(result_reg(inst))
                   .src(reg_of(inst.operand(0), "operand 0"))
                   .src(reg_of(inst.operand(1), "operand 1"));
    if (!is_float(t)) mul.lo();
    emit(std::move(mul));
}

void PtxISel::lower_div(const brass::Instruction& inst, bool is_unsigned) {
    Type t = is_unsigned ? type_for(inst.type()) : signed_type_for(inst.type());
    Inst div = Inst::make(Opcode::div, t)
                   .dst(result_reg(inst))
                   .src(reg_of(inst.operand(0), "operand 0"))
                   .src(reg_of(inst.operand(1), "operand 1"));
    if (is_float(t)) div.rnd(Rounding::rn);
    emit(std::move(div));
}

void PtxISel::lower_rem(const brass::Instruction& inst, bool is_unsigned) {
    Type t = is_unsigned ? type_for(inst.type()) : signed_type_for(inst.type());
    emit(Inst::make(Opcode::rem, t)
             .dst(result_reg(inst))
             .src(reg_of(inst.operand(0), "operand 0"))
             .src(reg_of(inst.operand(1), "operand 1")));
}

void PtxISel::lower_fma(const brass::Instruction& inst) {
    emit(Inst::make(Opcode::fma, type_for(inst.type())).rnd(Rounding::rn)
             .dst(result_reg(inst))
             .src(reg_of(inst.operand(0), "operand 0"))
             .src(reg_of(inst.operand(1), "operand 1"))
             .src(reg_of(inst.operand(2), "operand 2")));
}

void PtxISel::lower_neg(const brass::Instruction& inst) {
    emit(Inst::make(Opcode::neg, signed_type_for(inst.type()))
             .dst(result_reg(inst))
             .src(reg_of(inst.operand(0), "operand 0")));
}

// ---------------------------------------------------------------------------
// Bitwise and shifts
// ---------------------------------------------------------------------------

void PtxISel::lower_bitwise(const brass::Instruction& inst, Opcode op) {
    emit(Inst::make(op, bit_type_for(inst.type()))
             .dst(result_reg(inst))
             .src(reg_of(inst.operand(0), "operand 0"))
             .src(reg_of(inst.operand(1), "operand 1")));
}

void PtxISel::lower_not(const brass::Instruction& inst) {
    emit(Inst::make(Opcode::not_, bit_type_for(inst.type()))
             .dst(result_reg(inst))
             .src(reg_of(inst.operand(0), "operand 0")));
}

void PtxISel::lower_shift(const brass::Instruction& inst, Opcode op, Type t) {
    if (!inst.operand(1)) malformed(inst, "missing shift amount");
    Reg amount = shift_amount(inst.operand(1));
    emit(Inst::make(op, t)
             .dst(result_reg(inst))
             .src(reg_of(inst.operand(0), "operand 0"))
             .src(amount));
}

// ---------------------------------------------------------------------------
// Comparisons and select
// ---------------------------------------------------------------------------

// setp into the comparison's predicate; the i32 result is only materialized
// (selp.u32 r, 1, 0, p) when something consumes it as a value.
void PtxISel::lower_comparison(const brass::Instruction& inst, CmpOp cmp, bool is_unsigned) {
    const Value* lhs = inst.operand(0);
    const Value* rhs = inst.operand(1);
    if (!lhs || !rhs) malformed(inst, "missing operand");

    Type t = compare_type(lhs->type(), is_unsigned);
    if (is_float(t)) cmp = float_compare_op(cmp);

    Reg p = pred_of(inst.result());
    emit(Inst::make(Opcode::setp, t).cmp(cmp).dst(p)
             .src(reg_of(lhs, "operand 0"))
             .src(reg_of(rhs, "operand 1")));

    if (has_value_uses(inst.result())) {
        emit(Inst::make(Opcode::selp, Type::u32).dst(result_reg(inst))
                 .src(Operand::imm(1)).src(Operand::imm(0)).src(p));
    }
}

void PtxISel::lower_select(const brass::Instruction& inst) {
    Reg p = materialize_pred(inst.operand(0));
    emit(Inst::make(Opcode::selp, bit_type_for(inst.type()))
             .dst(result_reg(inst))
             .src(reg_of(inst.operand(1), "true value"))
             .src(reg_of(inst.operand(2), "false value"))
             .src(p));
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

void PtxISel::lower_cvt(const brass::Instruction& inst, Type dst, Type src, Rounding rnd) {
    emit(Inst::make(Opcode::cvt, dst).from(src).rnd(rnd)
             .dst(result_reg(inst))
             .src(reg_of(inst.operand(0), "operand 0")));
}

// sitofp_*: the destination float width comes from the instruction type
// (the MIR opcode name says f64 but f32 results are produced too).
void PtxISel::lower_sitofp(const brass::Instruction& inst, Type src) {
    Type dst = type_for(inst.type());
    if (!is_float(dst)) malformed(inst, "result is not a float");
    lower_cvt(inst, dst, src, Rounding::rn);
}

// fptosi_*: truncating conversion (cvt.rzi) from the operand's float width.
void PtxISel::lower_fptosi(const brass::Instruction& inst, Type dst) {
    if (!inst.operand(0)) malformed(inst, "missing operand");
    Type src = type_for(inst.operand(0)->type());
    if (!is_float(src)) malformed(inst, "operand is not a float");
    lower_cvt(inst, dst, src, Rounding::rzi);
}

void PtxISel::lower_bitcast(const brass::Instruction& inst) {
    emit(Inst::make(Opcode::mov, Type::b64)
             .dst(result_reg(inst))
             .src(reg_of(inst.operand(0), "operand 0")));
}

} // namespace brass::ptx
