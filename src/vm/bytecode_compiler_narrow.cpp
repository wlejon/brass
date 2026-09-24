// i8 / i16 integer operations for the bytecode compiler (docs/semantics.md,
// "Narrow integers"). Registers hold a narrow value zero-extended; each
// operation reads its operands sign- or zero-extended to i32 as its
// signedness requires, computes at i32 and truncates a narrow result back
// to its width.

#include "bytecode_compiler_impl.hpp"

namespace brass::detail {

namespace {

unsigned narrow_bits(const Value* v) {
    if (!v) return 0;
    const Type t = v->type();
    return t == Type::i8() ? 8u : t == Type::i16() ? 16u : 0u;
}

} // namespace

BcReg FunctionCompilerContext::widen_narrow(const Value* v, bool sign, BcReg scratch) {
    const unsigned bits = narrow_bits(v);
    BcReg r = get_reg(v);
    if (!bits) return r;
    const BytecodeOp op = sign ? (bits == 8 ? BytecodeOp::sext8 : BytecodeOp::sext16)
                               : (bits == 8 ? BytecodeOp::trunc8 : BytecodeOp::trunc16);
    emit(op, scratch, r);
    return scratch;
}

bool FunctionCompilerContext::lower_narrow_int(const Instruction& inst) {
    if (inst.operand_count() == 0 || inst.operand_count() > 2) return false;
    const unsigned bits = narrow_bits(inst.operand(0));
    if (!bits) return false;
    const BytecodeOp trunc = bits == 8 ? BytecodeOp::trunc8 : BytecodeOp::trunc16;
    const BytecodeOp sext = bits == 8 ? BytecodeOp::sext8 : BytecodeOp::sext16;
    const Opcode opc = inst.opcode();

    auto unary = [&](BytecodeOp op, bool sign) {
        const BcReg dst = get_result_reg(inst);
        emit(op, dst, widen_narrow(inst.operand(0), sign, scratch_reg));
        return dst;
    };
    auto binary = [&](BytecodeOp op, bool sign) {
        const BcReg dst = get_result_reg(inst);
        const BcReg a = widen_narrow(inst.operand(0), sign, scratch_reg);
        const BcReg b = widen_narrow(inst.operand(1), sign, scratch_reg2);
        emit(op, dst, a, b);
        return dst;
    };
    // A narrow result: the i32 result truncated to the width.
    auto value = [&](BcReg dst) { emit(trunc, dst, dst); };
    // An overflow flag: whether the exact i32 result of the widened
    // operands differs from its own narrow extension.
    auto overflow = [&](BytecodeOp op, bool sign) {
        const BcReg dst = binary(op, sign);
        emit(sign ? sext : trunc, scratch_reg, dst);
        emit(BytecodeOp::ne_i32, dst, dst, scratch_reg);
    };

    switch (opc) {
        case Opcode::add: value(binary(BytecodeOp::add_i32, false)); return true;
        case Opcode::sub: value(binary(BytecodeOp::sub_i32, false)); return true;
        case Opcode::mul: value(binary(BytecodeOp::mul_i32, false)); return true;
        case Opcode::and_: value(binary(BytecodeOp::and_i32, false)); return true;
        case Opcode::or_: value(binary(BytecodeOp::or_i32, false)); return true;
        case Opcode::xor_: value(binary(BytecodeOp::xor_i32, false)); return true;
        case Opcode::shl: value(binary(BytecodeOp::shl_i32, false)); return true;
        case Opcode::lshr: value(binary(BytecodeOp::lshr_i32, false)); return true;
        case Opcode::ashr: {
            const BcReg dst = get_result_reg(inst);
            const BcReg a = widen_narrow(inst.operand(0), true, scratch_reg);
            const BcReg b = widen_narrow(inst.operand(1), false, scratch_reg2);
            emit(BytecodeOp::ashr_i32, dst, a, b);
            value(dst);
            return true;
        }
        case Opcode::sdiv: value(binary(BytecodeOp::sdiv_i32, true)); return true;
        case Opcode::smod: value(binary(BytecodeOp::smod_i32, true)); return true;
        case Opcode::udiv: value(binary(BytecodeOp::udiv_i32, false)); return true;
        case Opcode::umod: value(binary(BytecodeOp::umod_i32, false)); return true;
        case Opcode::neg: value(unary(BytecodeOp::neg_i32, false)); return true;
        case Opcode::not_: value(unary(BytecodeOp::not_i32, false)); return true;
        case Opcode::popcnt: unary(BytecodeOp::popcnt_i32, false); return true;
        case Opcode::clz: {
            // clz32 of the zero-extended value counts 32 - bits extra zeros.
            const BcReg dst = unary(BytecodeOp::clz_i32, false);
            emit_const64(scratch_reg2, 32u - bits);
            emit(BytecodeOp::sub_i32, dst, dst, scratch_reg2);
            return true;
        }
        case Opcode::ctz: {
            // A bit just above the width makes ctz of zero the width.
            const BcReg dst = get_result_reg(inst);
            const BcReg a = widen_narrow(inst.operand(0), false, scratch_reg);
            emit_const64(scratch_reg2, uint64_t{1} << bits);
            emit(BytecodeOp::or_i32, scratch_reg, a, scratch_reg2);
            emit(BytecodeOp::ctz_i32, dst, scratch_reg);
            return true;
        }
        case Opcode::eq: binary(BytecodeOp::eq_i32, false); return true;
        case Opcode::ne: binary(BytecodeOp::ne_i32, false); return true;
        case Opcode::ult: binary(BytecodeOp::ult_i32, false); return true;
        case Opcode::ule: binary(BytecodeOp::ule_i32, false); return true;
        case Opcode::ugt: binary(BytecodeOp::ugt_i32, false); return true;
        case Opcode::uge: binary(BytecodeOp::uge_i32, false); return true;
        case Opcode::slt: binary(BytecodeOp::slt_i32, true); return true;
        case Opcode::sle: binary(BytecodeOp::sle_i32, true); return true;
        case Opcode::sgt: binary(BytecodeOp::sgt_i32, true); return true;
        case Opcode::sge: binary(BytecodeOp::sge_i32, true); return true;
        case Opcode::sadd_overflow: overflow(BytecodeOp::add_i32, true); return true;
        case Opcode::ssub_overflow: overflow(BytecodeOp::sub_i32, true); return true;
        case Opcode::smul_overflow: overflow(BytecodeOp::mul_i32, true); return true;
        case Opcode::uadd_overflow: overflow(BytecodeOp::add_i32, false); return true;
        case Opcode::usub_overflow: overflow(BytecodeOp::sub_i32, false); return true;
        case Opcode::umul_overflow: overflow(BytecodeOp::mul_i32, false); return true;
        default: return false;
    }
}

} // namespace brass::detail
