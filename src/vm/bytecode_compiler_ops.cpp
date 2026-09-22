#include "bytecode_compiler_impl.hpp"
#include <stdexcept>
#include <iostream>

namespace brass::detail {

void FunctionCompilerContext::lower_instruction(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::iconst_i32:
        case Opcode::iconst_i64:
        case Opcode::fconst_f64:
        case Opcode::patchable_const_i32:
        case Opcode::patchable_const_i64:
            lower_constant(inst);
            break;

        case Opcode::sext_i64:
        case Opcode::zext_i64:
        case Opcode::trunc_i32:
        case Opcode::trunc_i8:
        case Opcode::fptosi_i32:
        case Opcode::fptosi_i64:
        case Opcode::fptosi_i32_f32:
        case Opcode::fptosi_i64_f32:
        case Opcode::sitofp_f64_i32:
        case Opcode::sitofp_f64_i64:
        case Opcode::sitofp_f32_i32:
        case Opcode::sitofp_f32_i64:
        case Opcode::fptrunc_f32_f64:
        case Opcode::fpext_f64_f32:
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64:
            lower_conversion(inst);
            break;

        case Opcode::add:
        case Opcode::sub:
        case Opcode::mul:
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::smod:
        case Opcode::umod:
        case Opcode::neg:
        case Opcode::sadd_overflow:
        case Opcode::ssub_overflow:
        case Opcode::smul_overflow:
        case Opcode::uadd_overflow:
        case Opcode::usub_overflow:
        case Opcode::umul_overflow:
        case Opcode::fma_f32:
        case Opcode::fma_f64:
        case Opcode::sqrt_f32:
        case Opcode::sqrt_f64:
        case Opcode::floor_f32:
        case Opcode::floor_f64:
        case Opcode::ceil_f32:
        case Opcode::ceil_f64:
        case Opcode::round_f32:
        case Opcode::round_f64:
        case Opcode::fabs_f32:
        case Opcode::fabs_f64:
        case Opcode::fmin_f32:
        case Opcode::fmin_f64:
        case Opcode::fmax_f32:
        case Opcode::fmax_f64:
            lower_arithmetic(inst);
            break;

        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
        case Opcode::shl:
        case Opcode::lshr:
        case Opcode::ashr:
        case Opcode::not_:
        case Opcode::clz:
        case Opcode::ctz:
        case Opcode::popcnt:
            lower_bitwise(inst);
            break;

        case Opcode::eq:
        case Opcode::ne:
        case Opcode::slt:
        case Opcode::ult:
        case Opcode::sle:
        case Opcode::ule:
        case Opcode::sgt:
        case Opcode::ugt:
        case Opcode::sge:
        case Opcode::uge:
        case Opcode::select:
            lower_comparison(inst);
            break;

        case Opcode::alloca_:
        case Opcode::load:
        case Opcode::store:
        case Opcode::load_indexed:
        case Opcode::store_indexed:
        case Opcode::write_barrier:
            lower_memory(inst);
            break;

        case Opcode::call:
        case Opcode::call_indirect:
        case Opcode::patchable_call:
        case Opcode::func_addr:
            lower_call(inst);
            break;

        case Opcode::br:
        case Opcode::br_if:
        case Opcode::switch_:
        case Opcode::ret:
        case Opcode::unreachable:
            lower_terminator(inst);
            break;

        case Opcode::safepoint:
        case Opcode::guard:
        case Opcode::resume_point:
        case Opcode::osr_entry:
            lower_runtime_gc(inst);
            break;

        case Opcode::throw_:
        case Opcode::invoke:
        case Opcode::landing_pad:
        case Opcode::resume:
        case Opcode::coro_create:
        case Opcode::coro_suspend:
        case Opcode::coro_resume:
        case Opcode::coro_destroy:
            lower_coroutine(inst);
            break;

        case Opcode::vadd:
        case Opcode::vsub:
        case Opcode::vmul:
        case Opcode::vdiv:
        case Opcode::vfma:
        case Opcode::vneg:
        case Opcode::vmin:
        case Opcode::vmax:
        case Opcode::vsqrt:
        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor:
        case Opcode::vnot:
        case Opcode::vload:
        case Opcode::vstore:
        case Opcode::vbroadcast:
        case Opcode::vextract_lane:
        case Opcode::vinsert_lane:
        case Opcode::vshuffle:
        case Opcode::vzero:
            lower_vector(inst);
            break;

        default:
            emit(BytecodeOp::nop, 0, 0, 0);
            break;
    }
}

void FunctionCompilerContext::lower_constant(const Instruction& inst) {
    uint8_t dst = get_result_reg(inst);
    switch (inst.opcode()) {
        case Opcode::iconst_i32: {
            int32_t val = inst.imm_i32();
            if (val >= -32768 && val <= 32767) {
                emit_ad(BytecodeOp::mov_imm, dst, static_cast<int16_t>(val));
            } else {
                uint32_t c_idx = out.add_constant(static_cast<uint64_t>(static_cast<uint32_t>(val)));
                emit_ad(BytecodeOp::load_const, dst, static_cast<uint16_t>(c_idx));
            }
            break;
        }
        case Opcode::iconst_i64: {
            int64_t val = inst.imm_i64();
            if (val >= -32768 && val <= 32767) {
                emit_ad(BytecodeOp::mov_imm, dst, static_cast<int16_t>(val));
            } else {
                uint32_t c_idx = out.add_constant(static_cast<uint64_t>(val));
                emit_ad(BytecodeOp::load_const, dst, static_cast<uint16_t>(c_idx));
            }
            break;
        }
        case Opcode::fconst_f64: {
            uint32_t c_idx = 0;
            if (inst.type() == Type::f32()) {
                c_idx = out.add_constant_f32(static_cast<float>(inst.imm_f64()));
            } else {
                c_idx = out.add_constant_f64(inst.imm_f64());
            }
            emit_ad(BytecodeOp::load_const, dst, static_cast<uint16_t>(c_idx));
            break;
        }
        case Opcode::patchable_const_i32: {
            uint32_t s_idx = out.add_string_constant(inst.symbol());
            emit_ad(BytecodeOp::patchable_const32, dst, static_cast<uint16_t>(s_idx));
            break;
        }
        case Opcode::patchable_const_i64: {
            uint32_t s_idx = out.add_string_constant(inst.symbol());
            emit_ad(BytecodeOp::patchable_const64, dst, static_cast<uint16_t>(s_idx));
            break;
        }
        default:
            break;
    }
}

void FunctionCompilerContext::lower_conversion(const Instruction& inst) {
    uint8_t dst = get_result_reg(inst);
    uint8_t src = get_reg(inst.operand(0));

    switch (inst.opcode()) {
        case Opcode::sext_i64: emit(BytecodeOp::sext64, dst, src, 0); break;
        case Opcode::zext_i64: emit(BytecodeOp::zext64, dst, src, 0); break;
        case Opcode::trunc_i32: emit(BytecodeOp::trunc32, dst, src, 0); break;
        case Opcode::trunc_i8: emit(BytecodeOp::trunc8, dst, src, 0); break;
        case Opcode::fptosi_i32: emit(BytecodeOp::fptosi32, dst, src, 0); break;
        case Opcode::fptosi_i64: emit(BytecodeOp::fptosi64, dst, src, 0); break;
        case Opcode::fptosi_i32_f32: emit(BytecodeOp::fptosi32_f32, dst, src, 0); break;
        case Opcode::fptosi_i64_f32: emit(BytecodeOp::fptosi64_f32, dst, src, 0); break;
        case Opcode::sitofp_f64_i32: emit(BytecodeOp::sitofp_f64, dst, src, 0); break;
        case Opcode::sitofp_f64_i64: emit(BytecodeOp::sitofp_f64_i64, dst, src, 0); break;
        case Opcode::sitofp_f32_i32: emit(BytecodeOp::sitofp_f32, dst, src, 0); break;
        case Opcode::sitofp_f32_i64: emit(BytecodeOp::sitofp_f32_i64, dst, src, 0); break;
        case Opcode::fptrunc_f32_f64: emit(BytecodeOp::fptrunc_f32, dst, src, 0); break;
        case Opcode::fpext_f64_f32: emit(BytecodeOp::fpext_f64, dst, src, 0); break;
        case Opcode::bitcast_i64_f64: emit(BytecodeOp::bitcast_i64_f64, dst, src, 0); break;
        case Opcode::bitcast_f64_i64: emit(BytecodeOp::bitcast_f64_i64, dst, src, 0); break;
        default: break;
    }
}

void FunctionCompilerContext::lower_arithmetic(const Instruction& inst) {
    uint8_t dst = get_result_reg(inst);
    bool is_f32 = (inst.type() == Type::f32());
    bool is_f64 = (inst.type() == Type::f64());
    bool is_i32 = (inst.type() == Type::i32() || inst.type() == Type::i16() || inst.type() == Type::i8());

    if (inst.opcode() == Opcode::neg) {
        uint8_t src = get_reg(inst.operand(0));
        BytecodeOp op = is_f32 ? BytecodeOp::neg_f32 : (is_f64 ? BytecodeOp::neg_f64 : (is_i32 ? BytecodeOp::neg_i32 : BytecodeOp::neg_i64));
        emit(op, dst, src, 0);
        return;
    }

    if (inst.opcode() == Opcode::sqrt_f32 || inst.opcode() == Opcode::sqrt_f64 ||
        inst.opcode() == Opcode::fabs_f32 || inst.opcode() == Opcode::fabs_f64 ||
        inst.opcode() == Opcode::floor_f32 || inst.opcode() == Opcode::floor_f64 ||
        inst.opcode() == Opcode::ceil_f32 || inst.opcode() == Opcode::ceil_f64 ||
        inst.opcode() == Opcode::round_f32 || inst.opcode() == Opcode::round_f64) {
        uint8_t src = get_reg(inst.operand(0));
        BytecodeOp op = BytecodeOp::nop;
        switch (inst.opcode()) {
            case Opcode::sqrt_f32: op = BytecodeOp::sqrt_f32; break;
            case Opcode::sqrt_f64: op = BytecodeOp::sqrt_f64; break;
            case Opcode::fabs_f32: op = BytecodeOp::fabs_f32; break;
            case Opcode::fabs_f64: op = BytecodeOp::fabs_f64; break;
            case Opcode::floor_f32: op = BytecodeOp::floor_f32; break;
            case Opcode::floor_f64: op = BytecodeOp::floor_f64; break;
            case Opcode::ceil_f32: op = BytecodeOp::ceil_f32; break;
            case Opcode::ceil_f64: op = BytecodeOp::ceil_f64; break;
            case Opcode::round_f32: op = BytecodeOp::round_f32; break;
            case Opcode::round_f64: op = BytecodeOp::round_f64; break;
            default: break;
        }
        emit(op, dst, src, 0);
        return;
    }

    if (inst.opcode() == Opcode::fma_f32 || inst.opcode() == Opcode::fma_f64) {
        uint8_t a = get_reg(inst.operand(0));
        uint8_t b = get_reg(inst.operand(1));
        uint8_t c = get_reg(inst.operand(2));
        emit(BytecodeOp::mov, dst, c, 0);
        emit(inst.opcode() == Opcode::fma_f32 ? BytecodeOp::fma_f32 : BytecodeOp::fma_f64, dst, a, b);
        return;
    }

    uint8_t src1 = get_reg(inst.operand(0));
    uint8_t src2 = get_reg(inst.operand(1));
    BytecodeOp op = BytecodeOp::nop;

    switch (inst.opcode()) {
        case Opcode::add:
            op = is_f32 ? BytecodeOp::add_f32 : (is_f64 ? BytecodeOp::add_f64 : (is_i32 ? BytecodeOp::add_i32 : BytecodeOp::add_i64));
            break;
        case Opcode::sub:
            op = is_f32 ? BytecodeOp::sub_f32 : (is_f64 ? BytecodeOp::sub_f64 : (is_i32 ? BytecodeOp::sub_i32 : BytecodeOp::sub_i64));
            break;
        case Opcode::mul:
            op = is_f32 ? BytecodeOp::mul_f32 : (is_f64 ? BytecodeOp::mul_f64 : (is_i32 ? BytecodeOp::mul_i32 : BytecodeOp::mul_i64));
            break;
        case Opcode::sdiv:
            op = is_f32 ? BytecodeOp::fdiv_f32 : (is_f64 ? BytecodeOp::fdiv_f64 : (is_i32 ? BytecodeOp::sdiv_i32 : BytecodeOp::sdiv_i64));
            break;
        case Opcode::udiv:
            op = is_i32 ? BytecodeOp::udiv_i32 : BytecodeOp::udiv_i64;
            break;
        case Opcode::smod:
            op = is_i32 ? BytecodeOp::smod_i32 : BytecodeOp::smod_i64;
            break;
        case Opcode::umod:
            op = is_i32 ? BytecodeOp::umod_i32 : BytecodeOp::umod_i64;
            break;
        case Opcode::sadd_overflow:
            op = is_i32 ? BytecodeOp::sadd_overflow_i32 : BytecodeOp::sadd_overflow_i64;
            break;
        case Opcode::ssub_overflow:
            op = is_i32 ? BytecodeOp::ssub_overflow_i32 : BytecodeOp::ssub_overflow_i64;
            break;
        case Opcode::smul_overflow:
            op = is_i32 ? BytecodeOp::smul_overflow_i32 : BytecodeOp::smul_overflow_i64;
            break;
        case Opcode::uadd_overflow:
            op = is_i32 ? BytecodeOp::uadd_overflow_i32 : BytecodeOp::uadd_overflow_i64;
            break;
        case Opcode::usub_overflow:
            op = is_i32 ? BytecodeOp::usub_overflow_i32 : BytecodeOp::usub_overflow_i64;
            break;
        case Opcode::umul_overflow:
            op = is_i32 ? BytecodeOp::umul_overflow_i32 : BytecodeOp::umul_overflow_i64;
            break;
        case Opcode::fmin_f32: op = BytecodeOp::fmin_f32; break;
        case Opcode::fmin_f64: op = BytecodeOp::fmin_f64; break;
        case Opcode::fmax_f32: op = BytecodeOp::fmax_f32; break;
        case Opcode::fmax_f64: op = BytecodeOp::fmax_f64; break;
        default: break;
    }

    emit(op, dst, src1, src2);
}

void FunctionCompilerContext::lower_bitwise(const Instruction& inst) {
    uint8_t dst = get_result_reg(inst);
    bool is_i32 = (inst.type() == Type::i32() || inst.type() == Type::i16() || inst.type() == Type::i8());

    if (inst.opcode() == Opcode::not_ || inst.opcode() == Opcode::clz ||
        inst.opcode() == Opcode::ctz || inst.opcode() == Opcode::popcnt) {
        uint8_t src = get_reg(inst.operand(0));
        BytecodeOp op = BytecodeOp::nop;
        switch (inst.opcode()) {
            case Opcode::not_: op = is_i32 ? BytecodeOp::not_i32 : BytecodeOp::not_i64; break;
            case Opcode::clz: op = is_i32 ? BytecodeOp::clz_i32 : BytecodeOp::clz_i64; break;
            case Opcode::ctz: op = is_i32 ? BytecodeOp::ctz_i32 : BytecodeOp::ctz_i64; break;
            case Opcode::popcnt: op = is_i32 ? BytecodeOp::popcnt_i32 : BytecodeOp::popcnt_i64; break;
            default: break;
        }
        emit(op, dst, src, 0);
        return;
    }

    uint8_t src1 = get_reg(inst.operand(0));
    uint8_t src2 = get_reg(inst.operand(1));
    BytecodeOp op = BytecodeOp::nop;

    switch (inst.opcode()) {
        case Opcode::and_: op = is_i32 ? BytecodeOp::and_i32 : BytecodeOp::and_i64; break;
        case Opcode::or_: op = is_i32 ? BytecodeOp::or_i32 : BytecodeOp::or_i64; break;
        case Opcode::xor_: op = is_i32 ? BytecodeOp::xor_i32 : BytecodeOp::xor_i64; break;
        case Opcode::shl: op = is_i32 ? BytecodeOp::shl_i32 : BytecodeOp::shl_i64; break;
        case Opcode::lshr: op = is_i32 ? BytecodeOp::lshr_i32 : BytecodeOp::lshr_i64; break;
        case Opcode::ashr: op = is_i32 ? BytecodeOp::ashr_i32 : BytecodeOp::ashr_i64; break;
        default: break;
    }

    emit(op, dst, src1, src2);
}

void FunctionCompilerContext::lower_comparison(const Instruction& inst) {
    uint8_t dst = get_result_reg(inst);

    if (inst.opcode() == Opcode::select) {
        uint8_t cond = get_reg(inst.operand(0));
        uint8_t true_v = get_reg(inst.operand(1));
        uint8_t false_v = get_reg(inst.operand(2));
        emit(BytecodeOp::mov, dst, false_v, 0);
        emit(BytecodeOp::select, dst, cond, true_v);
        return;
    }

    uint8_t src1 = get_reg(inst.operand(0));
    uint8_t src2 = get_reg(inst.operand(1));
    Type op_ty = inst.operand(0)->type();
    bool is_f32 = (op_ty == Type::f32());
    bool is_f64 = (op_ty == Type::f64());
    bool is_i32 = (op_ty == Type::i32() || op_ty == Type::i16() || op_ty == Type::i8());

    BytecodeOp op = BytecodeOp::nop;
    switch (inst.opcode()) {
        case Opcode::eq: op = is_f32 ? BytecodeOp::eq_f32 : (is_f64 ? BytecodeOp::eq_f64 : (is_i32 ? BytecodeOp::eq_i32 : BytecodeOp::eq_i64)); break;
        case Opcode::ne: op = is_f32 ? BytecodeOp::ne_f32 : (is_f64 ? BytecodeOp::ne_f64 : (is_i32 ? BytecodeOp::ne_i32 : BytecodeOp::ne_i64)); break;
        case Opcode::slt: op = is_f32 ? BytecodeOp::lt_f32 : (is_f64 ? BytecodeOp::lt_f64 : (is_i32 ? BytecodeOp::slt_i32 : BytecodeOp::slt_i64)); break;
        case Opcode::ult: op = is_i32 ? BytecodeOp::ult_i32 : BytecodeOp::ult_i64; break;
        case Opcode::sle: op = is_f32 ? BytecodeOp::le_f32 : (is_f64 ? BytecodeOp::le_f64 : (is_i32 ? BytecodeOp::sle_i32 : BytecodeOp::sle_i64)); break;
        case Opcode::ule: op = is_i32 ? BytecodeOp::ule_i32 : BytecodeOp::ule_i64; break;
        case Opcode::sgt: op = is_f32 ? BytecodeOp::gt_f32 : (is_f64 ? BytecodeOp::gt_f64 : (is_i32 ? BytecodeOp::sgt_i32 : BytecodeOp::sgt_i64)); break;
        case Opcode::ugt: op = is_i32 ? BytecodeOp::ugt_i32 : BytecodeOp::ugt_i64; break;
        case Opcode::sge: op = is_f32 ? BytecodeOp::ge_f32 : (is_f64 ? BytecodeOp::ge_f64 : (is_i32 ? BytecodeOp::sge_i32 : BytecodeOp::sge_i64)); break;
        case Opcode::uge: op = is_i32 ? BytecodeOp::uge_i32 : BytecodeOp::uge_i64; break;
        default: break;
    }

    emit(op, dst, src1, src2);
}

void FunctionCompilerContext::lower_memory(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::alloca_: {
            uint8_t dst = get_result_reg(inst);
            int32_t size = inst.imm_i32();
            emit_ad(BytecodeOp::alloca_, dst, static_cast<uint16_t>(size > 0 ? size : 0));
            break;
        }
        case Opcode::load: {
            uint8_t dst = get_result_reg(inst);
            uint8_t base = get_reg(inst.operand(0));
            int32_t offset = inst.offset();

            BytecodeOp op = BytecodeOp::load64;
            if (inst.type() == Type::i8()) op = BytecodeOp::load8;
            else if (inst.type() == Type::i16()) op = BytecodeOp::load16;
            else if (inst.type() == Type::i32() || inst.type() == Type::f32()) op = BytecodeOp::load32;

            if (offset >= 0 && offset <= 255) {
                emit(op, dst, base, static_cast<uint8_t>(offset));
            } else {
                if (offset >= -32768 && offset <= 32767) {
                    emit_ad(BytecodeOp::mov_imm, scratch_reg, static_cast<int16_t>(offset));
                } else {
                    uint32_t c = out.add_constant(static_cast<uint64_t>(offset));
                    emit_ad(BytecodeOp::load_const, scratch_reg, static_cast<uint16_t>(c));
                }
                emit(BytecodeOp::add_i64, scratch_reg, base, scratch_reg);
                emit(op, dst, scratch_reg, 0);
            }
            break;
        }
        case Opcode::store: {
            uint8_t base = get_reg(inst.operand(0));
            uint8_t val = get_reg(inst.operand(1));
            int32_t offset = inst.offset();

            BytecodeOp op = BytecodeOp::store64;
            if (inst.memory_type() == Type::i8()) op = BytecodeOp::store8;
            else if (inst.memory_type() == Type::i16()) op = BytecodeOp::store16;
            else if (inst.memory_type() == Type::i32() || inst.memory_type() == Type::f32()) op = BytecodeOp::store32;

            if (offset >= 0 && offset <= 255) {
                emit(op, val, base, static_cast<uint8_t>(offset));
            } else {
                if (offset >= -32768 && offset <= 32767) {
                    emit_ad(BytecodeOp::mov_imm, scratch_reg, static_cast<int16_t>(offset));
                } else {
                    uint32_t c = out.add_constant(static_cast<uint64_t>(offset));
                    emit_ad(BytecodeOp::load_const, scratch_reg, static_cast<uint16_t>(c));
                }
                emit(BytecodeOp::add_i64, scratch_reg, base, scratch_reg);
                emit(op, val, scratch_reg, 0);
            }
            break;
        }
        case Opcode::load_indexed: {
            uint8_t dst = get_result_reg(inst);
            uint8_t base = get_reg(inst.operand(0));
            uint8_t idx = get_reg(inst.operand(1));
            // Calculate effective address in scratch_reg: base + idx * scale + offset
            uint8_t scale = inst.scale();
            if (scale > 1) {
                emit_ad(BytecodeOp::mov_imm, scratch_reg, static_cast<int16_t>(scale));
                emit(BytecodeOp::mul_i64, scratch_reg, idx, scratch_reg);
            } else {
                emit(BytecodeOp::mov, scratch_reg, idx, 0);
            }
            emit(BytecodeOp::add_i64, scratch_reg, base, scratch_reg);
            int32_t offset = inst.offset();
            if (offset != 0) {
                emit_ad(BytecodeOp::mov_imm, scratch_reg2, static_cast<int16_t>(offset));
                emit(BytecodeOp::add_i64, scratch_reg, scratch_reg, scratch_reg2);
            }
            BytecodeOp op = BytecodeOp::load64;
            if (inst.type() == Type::i8()) op = BytecodeOp::load8;
            else if (inst.type() == Type::i16()) op = BytecodeOp::load16;
            else if (inst.type() == Type::i32() || inst.type() == Type::f32()) op = BytecodeOp::load32;
            emit(op, dst, scratch_reg, 0);
            break;
        }
        case Opcode::store_indexed: {
            uint8_t base = get_reg(inst.operand(0));
            uint8_t idx = get_reg(inst.operand(1));
            uint8_t val = get_reg(inst.operand(2));
            uint8_t scale = inst.scale();
            if (scale > 1) {
                emit_ad(BytecodeOp::mov_imm, scratch_reg, static_cast<int16_t>(scale));
                emit(BytecodeOp::mul_i64, scratch_reg, idx, scratch_reg);
            } else {
                emit(BytecodeOp::mov, scratch_reg, idx, 0);
            }
            emit(BytecodeOp::add_i64, scratch_reg, base, scratch_reg);
            int32_t offset = inst.offset();
            if (offset != 0) {
                emit_ad(BytecodeOp::mov_imm, scratch_reg2, static_cast<int16_t>(offset));
                emit(BytecodeOp::add_i64, scratch_reg, scratch_reg, scratch_reg2);
            }
            BytecodeOp op = BytecodeOp::store64;
            if (inst.memory_type() == Type::i8()) op = BytecodeOp::store8;
            else if (inst.memory_type() == Type::i16()) op = BytecodeOp::store16;
            else if (inst.memory_type() == Type::i32() || inst.memory_type() == Type::f32()) op = BytecodeOp::store32;
            emit(op, val, scratch_reg, 0);
            break;
        }
        case Opcode::write_barrier: {
            uint8_t obj = get_reg(inst.operand(0));
            uint8_t val = get_reg(inst.operand(1));
            emit(BytecodeOp::write_barrier, 0, obj, val);
            break;
        }
        default: break;
    }
}

void FunctionCompilerContext::lower_call(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::func_addr: {
            uint8_t dst = get_result_reg(inst);
            uint32_t s_idx = out.add_string_constant(inst.symbol());
            emit_ad(BytecodeOp::func_addr, dst, static_cast<uint16_t>(s_idx));
            break;
        }
        case Opcode::call: {
            uint8_t dst = inst.result() ? get_result_reg(inst) : 255;
            CallSiteInfo cs;
            cs.callee = std::string(inst.symbol());
            cs.dst_reg = dst;
            for (size_t i = 0; i < inst.operand_count(); ++i) {
                cs.arg_regs.push_back(get_reg(inst.operand(i)));
            }
            uint32_t cs_idx = out.add_call_site(std::move(cs));
            emit_ad(BytecodeOp::call, dst, static_cast<uint16_t>(cs_idx));
            break;
        }
        case Opcode::call_indirect: {
            uint8_t callee = get_reg(inst.operand(0));
            uint8_t dst = inst.result() ? get_result_reg(inst) : 255;
            CallSiteInfo cs;
            cs.callee_reg = callee;
            cs.dst_reg = dst;
            cs.site_id = inst.site_id();
            for (size_t i = 1; i < inst.operand_count(); ++i) {
                cs.arg_regs.push_back(get_reg(inst.operand(i)));
            }
            uint32_t cs_idx = out.add_call_site(std::move(cs));
            emit_ad(BytecodeOp::call_indirect, dst, static_cast<uint16_t>(cs_idx));
            break;
        }
        case Opcode::patchable_call: {
            uint8_t dst = inst.result() ? get_result_reg(inst) : 255;
            CallSiteInfo cs;
            cs.callee = std::string(inst.symbol());
            cs.extra_symbol = std::string(inst.extra_symbol());
            cs.dst_reg = dst;
            for (size_t i = 0; i < inst.operand_count(); ++i) {
                cs.arg_regs.push_back(get_reg(inst.operand(i)));
            }
            uint32_t cs_idx = out.add_call_site(std::move(cs));
            emit_ad(BytecodeOp::patchable_call, dst, static_cast<uint16_t>(cs_idx));
            break;
        }
        default: break;
    }
}

void FunctionCompilerContext::lower_terminator(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::ret: {
            if (inst.operand_count() > 0 && inst.operand(0) != nullptr) {
                uint8_t val = get_reg(inst.operand(0));
                emit(BytecodeOp::ret, val, 0, 0);
            } else {
                emit(BytecodeOp::ret_void, 0, 0, 0);
            }
            break;
        }
        case Opcode::unreachable: {
            emit(BytecodeOp::unreachable, 0, 0, 0);
            break;
        }
        case Opcode::br: {
            const auto& target = inst.branch_target();
            emit_parallel_moves(target);
            size_t jump_idx = out.current_pc();
            emit_ad(BytecodeOp::jump, 0, static_cast<int16_t>(0));
            jump_fixups.push_back({jump_idx, target.block, 0, BytecodeOp::jump});
            break;
        }
        case Opcode::br_if: {
            uint8_t cond_reg = get_reg(inst.operand(0));
            const auto& true_t = inst.true_target();
            const auto& false_t = inst.false_target();

            if (true_t.args.empty() && false_t.args.empty()) {
                size_t jmp_if_idx = out.current_pc();
                emit_ad(BytecodeOp::jump_if, cond_reg, static_cast<int16_t>(0));
                jump_fixups.push_back({jmp_if_idx, true_t.block, cond_reg, BytecodeOp::jump_if});

                size_t jmp_false_idx = out.current_pc();
                emit_ad(BytecodeOp::jump, 0, static_cast<int16_t>(0));
                jump_fixups.push_back({jmp_false_idx, false_t.block, 0, BytecodeOp::jump});
            } else {
                size_t jmp_if_idx = out.current_pc();
                emit_ad(BytecodeOp::jump_if, cond_reg, static_cast<int16_t>(0));

                // False path
                emit_parallel_moves(false_t);
                size_t jmp_false_idx = out.current_pc();
                emit_ad(BytecodeOp::jump, 0, static_cast<int16_t>(0));
                jump_fixups.push_back({jmp_false_idx, false_t.block, 0, BytecodeOp::jump});

                // True path trampoline
                uint32_t true_trampoline_pc = static_cast<uint32_t>(out.current_pc());
                int32_t if_offset = static_cast<int32_t>(true_trampoline_pc) - static_cast<int32_t>(jmp_if_idx);
                out.code[jmp_if_idx] = encode_ad(BytecodeOp::jump_if, cond_reg, static_cast<int16_t>(if_offset));

                emit_parallel_moves(true_t);
                size_t jmp_true_idx = out.current_pc();
                emit_ad(BytecodeOp::jump, 0, static_cast<int16_t>(0));
                jump_fixups.push_back({jmp_true_idx, true_t.block, 0, BytecodeOp::jump});
            }
            break;
        }
        case Opcode::switch_: {
            uint8_t cond_reg = get_reg(inst.operand(0));
            SwitchTable st;
            size_t t_idx = out.switch_tables.size();
            for (const auto& sc : inst.switch_cases()) {
                st.cases.push_back({sc.value, 0});
                switch_fixups.push_back({t_idx, st.cases.size() - 1, sc.target.block, false});
            }
            switch_fixups.push_back({t_idx, 0, inst.default_target().block, true});
            uint32_t st_id = out.add_switch_table(std::move(st));
            emit_ad(BytecodeOp::switch_, cond_reg, static_cast<uint16_t>(st_id));
            break;
        }
        default: break;
    }
}

void FunctionCompilerContext::lower_runtime_gc(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::safepoint:
            emit(BytecodeOp::safepoint, 0, 0, 0);
            break;
        case Opcode::guard: {
            uint8_t cond_reg = get_reg(inst.operand(0));
            GuardInfo gi;
            gi.resume_id = inst.resume_id();
            gi.exit_stub = std::string(inst.symbol());
            for (const auto* v : inst.state_map()) {
                if (v) gi.state_regs.push_back(get_reg(v));
            }
            uint32_t g_idx = out.add_guard(std::move(gi));
            emit_ad(BytecodeOp::guard, cond_reg, static_cast<uint16_t>(g_idx));
            break;
        }
        case Opcode::resume_point:
            emit(BytecodeOp::resume_point, 0, 0, 0);
            break;
        case Opcode::osr_entry: {
            OsrEntry oe;
            oe.pc = static_cast<uint32_t>(out.current_pc());
            out.osr_entries.push_back(std::move(oe));
            emit(BytecodeOp::osr_entry, 0, 0, 0);
            break;
        }
        default: break;
    }
}

void FunctionCompilerContext::lower_coroutine(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::throw_:
            emit(BytecodeOp::throw_, get_reg(inst.operand(0)), 0, 0);
            break;
        case Opcode::landing_pad:
            emit(BytecodeOp::landing_pad, get_result_reg(inst), 0, 0);
            break;
        case Opcode::resume:
            emit(BytecodeOp::resume, inst.operand_count() > 0 && inst.operand(0) ? get_reg(inst.operand(0)) : 255, 0, 0);
            break;
        case Opcode::invoke: {
            uint32_t start_pc = static_cast<uint32_t>(out.current_pc());
            uint8_t dst = inst.result() ? get_result_reg(inst) : 255;
            CallSiteInfo cs;
            cs.callee = std::string(inst.symbol());
            cs.dst_reg = dst;
            for (size_t i = 0; i < inst.operand_count(); ++i) {
                cs.arg_regs.push_back(get_reg(inst.operand(i)));
            }
            uint32_t cs_idx = out.add_call_site(std::move(cs));
            emit_ad(BytecodeOp::invoke, dst, static_cast<uint16_t>(cs_idx));
            uint32_t end_pc = static_cast<uint32_t>(out.current_pc());

            ExceptionEntry ee;
            ee.start_pc = start_pc;
            ee.end_pc = end_pc;
            out.exception_table.push_back(ee);
            size_t ee_idx = out.exception_table.size() - 1;

            emit_parallel_moves(inst.normal_target());
            size_t jmp_normal = out.current_pc();
            emit_ad(BytecodeOp::jump, 0, static_cast<int16_t>(0));
            jump_fixups.push_back({jmp_normal, inst.normal_target().block, 0, BytecodeOp::jump});

            // Patch handler_pc when unwind block is resolved
            exception_fixups.push_back({ee_idx, inst.unwind_target().block});
            break;
        }
        case Opcode::coro_create: {
            uint8_t dst = get_result_reg(inst);
            CallSiteInfo cs;
            cs.callee = std::string(inst.symbol());
            cs.dst_reg = dst;
            for (size_t i = 0; i < inst.operand_count(); ++i) {
                cs.arg_regs.push_back(get_reg(inst.operand(i)));
            }
            uint32_t cs_idx = out.add_call_site(std::move(cs));
            emit_ad(BytecodeOp::coro_create, dst, static_cast<uint16_t>(cs_idx));
            break;
        }
        case Opcode::coro_suspend: {
            uint8_t yield_reg = inst.operand_count() > 0 && inst.operand(0) ? get_reg(inst.operand(0)) : 0;
            uint8_t dst_reg = inst.result() ? get_result_reg(inst) : 255;
            emit(BytecodeOp::coro_suspend, dst_reg, yield_reg, static_cast<uint8_t>(inst.resume_id()));
            break;
        }
        case Opcode::coro_resume: {
            uint8_t dst = get_result_reg(inst);
            uint8_t coro_reg = get_reg(inst.operand(0));
            uint8_t input_reg = inst.operand_count() > 1 && inst.operand(1) ? get_reg(inst.operand(1)) : 255;
            emit(BytecodeOp::coro_resume, dst, coro_reg, input_reg);
            break;
        }
        case Opcode::coro_destroy: {
            uint8_t coro_reg = get_reg(inst.operand(0));
            emit(BytecodeOp::coro_destroy, 0, coro_reg, 0);
            break;
        }
        default: break;
    }
}

void FunctionCompilerContext::lower_vector(const Instruction& inst) {
    uint8_t dst = get_result_reg(inst);
    BytecodeOp op = BytecodeOp::nop;

    switch (inst.opcode()) {
        case Opcode::vadd: op = BytecodeOp::vadd; break;
        case Opcode::vsub: op = BytecodeOp::vsub; break;
        case Opcode::vmul: op = BytecodeOp::vmul; break;
        case Opcode::vdiv: op = BytecodeOp::vdiv; break;
        case Opcode::vfma: op = BytecodeOp::vfma; break;
        case Opcode::vneg: op = BytecodeOp::vneg; break;
        case Opcode::vmin: op = BytecodeOp::vmin; break;
        case Opcode::vmax: op = BytecodeOp::vmax; break;
        case Opcode::vsqrt: op = BytecodeOp::vsqrt; break;
        case Opcode::vand: op = BytecodeOp::vand; break;
        case Opcode::vor: op = BytecodeOp::vor; break;
        case Opcode::vxor: op = BytecodeOp::vxor; break;
        case Opcode::vnot: op = BytecodeOp::vnot; break;
        case Opcode::vload: op = BytecodeOp::vload; break;
        case Opcode::vstore: op = BytecodeOp::vstore; break;
        case Opcode::vbroadcast: op = BytecodeOp::vbroadcast; break;
        case Opcode::vextract_lane: op = BytecodeOp::vextract_lane; break;
        case Opcode::vinsert_lane: op = BytecodeOp::vinsert_lane; break;
        case Opcode::vshuffle: op = BytecodeOp::vshuffle; break;
        case Opcode::vzero: op = BytecodeOp::vzero; break;
        default: break;
    }

    if (inst.opcode() == Opcode::vzero) {
        emit(op, dst, 0, 0);
    } else if (inst.operand_count() == 1) {
        emit(op, dst, get_reg(inst.operand(0)), 0);
    } else if (inst.operand_count() >= 2) {
        emit(op, dst, get_reg(inst.operand(0)), get_reg(inst.operand(1)));
    }
}

} // namespace brass::detail
