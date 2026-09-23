#include "bytecode_compiler_impl.hpp"
#include <cstring>

namespace brass::detail {

namespace {

bool is_narrow_int(Type t) {
    return t == Type::i32() || t == Type::i16() || t == Type::i8();
}

// Loads and stores move the memory type's width; narrow integers are
// zero-extended into registers (the i32 register form).
BytecodeOp load_op(Type t) {
    if (t == Type::i8()) return BytecodeOp::load8;
    if (t == Type::i16()) return BytecodeOp::load16;
    if (t == Type::i32() || t == Type::f32()) return BytecodeOp::load32;
    return BytecodeOp::load64;
}

BytecodeOp store_op(Type t) {
    if (t == Type::i8()) return BytecodeOp::store8;
    if (t == Type::i16()) return BytecodeOp::store16;
    if (t == Type::i32() || t == Type::f32()) return BytecodeOp::store32;
    return BytecodeOp::store64;
}

} // namespace

void FunctionCompilerContext::lower_instruction(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::iconst_i32:
        case Opcode::iconst_i64:
        case Opcode::fconst_f64:
        case Opcode::patchable_const_i32:
        case Opcode::patchable_const_i64:
            lower_constant(inst);
            break;

        case Opcode::sext_i64: case Opcode::zext_i64: case Opcode::trunc_i32: case Opcode::trunc_i8:
        case Opcode::fptosi_i32: case Opcode::fptosi_i64: case Opcode::fptosi_i32_f32: case Opcode::fptosi_i64_f32:
        case Opcode::sitofp_f64_i32: case Opcode::sitofp_f64_i64: case Opcode::sitofp_f32_i32:
        case Opcode::sitofp_f32_i64: case Opcode::fptrunc_f32_f64: case Opcode::fpext_f64_f32:
        case Opcode::bitcast_i64_f64: case Opcode::bitcast_f64_i64:
            lower_conversion(inst);
            break;

        case Opcode::add: case Opcode::sub: case Opcode::mul: case Opcode::sdiv: case Opcode::udiv:
        case Opcode::smod: case Opcode::umod: case Opcode::neg:
        case Opcode::sadd_overflow: case Opcode::ssub_overflow: case Opcode::smul_overflow:
        case Opcode::uadd_overflow: case Opcode::usub_overflow: case Opcode::umul_overflow:
        case Opcode::fma_f32: case Opcode::fma_f64: case Opcode::sqrt_f32: case Opcode::sqrt_f64:
        case Opcode::floor_f32: case Opcode::floor_f64: case Opcode::ceil_f32: case Opcode::ceil_f64:
        case Opcode::round_f32: case Opcode::round_f64: case Opcode::fabs_f32: case Opcode::fabs_f64:
        case Opcode::fmin_f32: case Opcode::fmin_f64: case Opcode::fmax_f32: case Opcode::fmax_f64:
            lower_arithmetic(inst);
            break;

        case Opcode::and_: case Opcode::or_: case Opcode::xor_: case Opcode::shl: case Opcode::lshr:
        case Opcode::ashr: case Opcode::not_: case Opcode::clz: case Opcode::ctz: case Opcode::popcnt:
            lower_bitwise(inst);
            break;

        case Opcode::eq: case Opcode::ne: case Opcode::slt: case Opcode::ult: case Opcode::sle:
        case Opcode::ule: case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge:
        case Opcode::select:
            lower_comparison(inst);
            break;

        case Opcode::alloca_: case Opcode::load: case Opcode::store: case Opcode::load_indexed:
        case Opcode::store_indexed: case Opcode::write_barrier:
            lower_memory(inst);
            break;

        case Opcode::call: case Opcode::call_indirect: case Opcode::patchable_call: case Opcode::func_addr:
            lower_call(inst);
            break;

        case Opcode::br: case Opcode::br_if: case Opcode::switch_: case Opcode::ret:
        case Opcode::unreachable: case Opcode::invoke:
            lower_terminator(inst);
            break;

        case Opcode::safepoint: case Opcode::guard: case Opcode::resume_point: case Opcode::osr_entry:
            lower_runtime_gc(inst);
            break;

        case Opcode::throw_: case Opcode::landing_pad: case Opcode::resume: case Opcode::coro_create:
        case Opcode::coro_suspend: case Opcode::coro_resume: case Opcode::coro_destroy:
            lower_coroutine(inst);
            break;

        case Opcode::vadd: case Opcode::vsub: case Opcode::vmul: case Opcode::vdiv: case Opcode::vfma:
        case Opcode::vneg: case Opcode::vmin: case Opcode::vmax: case Opcode::vsqrt: case Opcode::vand:
        case Opcode::vor: case Opcode::vxor: case Opcode::vnot: case Opcode::vload: case Opcode::vstore:
        case Opcode::vbroadcast: case Opcode::vextract_lane: case Opcode::vinsert_lane:
        case Opcode::vshuffle: case Opcode::vzero:
            lower_vector(inst);
            break;

        case Opcode::pinned_tls_read:
            emit(BytecodeOp::pinned_tls_read, get_result_reg(inst));
            break;
        case Opcode::pinned_tls_write:
            emit(BytecodeOp::pinned_tls_write, 0, get_reg(inst.operand(0)));
            break;
        case Opcode::read_sp:
            emit(BytecodeOp::read_sp, get_result_reg(inst));
            break;
    }
}

void FunctionCompilerContext::lower_constant(const Instruction& inst) {
    BcReg dst = get_result_reg(inst);
    switch (inst.opcode()) {
        case Opcode::iconst_i32:
            emit_ai(BytecodeOp::iconst32, dst, inst.imm_i32());
            break;
        case Opcode::iconst_i64:
            emit_const64(dst, static_cast<uint64_t>(inst.imm_i64()));
            break;
        case Opcode::fconst_f64: {
            uint64_t bits = 0;
            if (inst.type() == Type::f32()) {
                float f = static_cast<float>(inst.imm_f64());
                uint32_t b32 = 0;
                std::memcpy(&b32, &f, sizeof(f));
                bits = b32;
            } else {
                double d = inst.imm_f64();
                std::memcpy(&bits, &d, sizeof(d));
            }
            emit_ai(BytecodeOp::load_const, dst, static_cast<int32_t>(add_constant(bits)));
            break;
        }
        case Opcode::patchable_const_i32:
        case Opcode::patchable_const_i64: {
            const bool i32 = inst.opcode() == Opcode::patchable_const_i32;
            PatchConstSite site;
            site.symbol = std::string(inst.symbol());
            site.default_value = i32 ? static_cast<int64_t>(inst.imm_i32()) : inst.imm_i64();
            uint32_t idx = out.add_patch_const(std::move(site));
            emit_ai(i32 ? BytecodeOp::patchable_const32 : BytecodeOp::patchable_const64, dst,
                    static_cast<int32_t>(idx));
            break;
        }
        default:
            fail("not a constant");
    }
}

void FunctionCompilerContext::lower_conversion(const Instruction& inst) {
    BcReg dst = get_result_reg(inst);
    BcReg src = get_reg(inst.operand(0));
    BytecodeOp op = BytecodeOp::nop;
    switch (inst.opcode()) {
        case Opcode::sext_i64: op = BytecodeOp::sext64; break;
        case Opcode::zext_i64:
            // An i8 source is zero-extended from its low byte.
            op = inst.operand(0)->type() == Type::i8() ? BytecodeOp::trunc8 : BytecodeOp::zext64;
            break;
        case Opcode::trunc_i32: op = BytecodeOp::trunc32; break;
        case Opcode::trunc_i8: op = BytecodeOp::trunc8; break;
        case Opcode::fptosi_i32: op = BytecodeOp::fptosi32; break;
        case Opcode::fptosi_i64: op = BytecodeOp::fptosi64; break;
        case Opcode::fptosi_i32_f32: op = BytecodeOp::fptosi32_f32; break;
        case Opcode::fptosi_i64_f32: op = BytecodeOp::fptosi64_f32; break;
        case Opcode::sitofp_f64_i32: op = BytecodeOp::sitofp_f64; break;
        case Opcode::sitofp_f64_i64: op = BytecodeOp::sitofp_f64_i64; break;
        case Opcode::sitofp_f32_i32: op = BytecodeOp::sitofp_f32; break;
        case Opcode::sitofp_f32_i64: op = BytecodeOp::sitofp_f32_i64; break;
        case Opcode::fptrunc_f32_f64: op = BytecodeOp::fptrunc_f32; break;
        case Opcode::fpext_f64_f32: op = BytecodeOp::fpext_f64; break;
        case Opcode::bitcast_i64_f64: op = BytecodeOp::bitcast_i64_f64; break;
        case Opcode::bitcast_f64_i64: op = BytecodeOp::bitcast_f64_i64; break;
        default: fail("not a conversion");
    }
    emit(op, dst, src);
}

void FunctionCompilerContext::lower_arithmetic(const Instruction& inst) {
    BcReg dst = get_result_reg(inst);
    // Overflow checks are typed by their operands, the rest by the result.
    Type ty = inst.operand_count() > 0 && inst.operand(0) ? inst.operand(0)->type() : inst.type();
    const bool is_f32 = ty == Type::f32();
    const bool is_f64 = ty == Type::f64();
    const bool is_i32 = is_narrow_int(ty);
    auto by_type = [&](BytecodeOp f32, BytecodeOp f64, BytecodeOp i32, BytecodeOp i64) {
        return is_f32 ? f32 : is_f64 ? f64 : is_i32 ? i32 : i64;
    };
    auto int_only = [&](BytecodeOp i32, BytecodeOp i64) {
        if (is_f32 || is_f64) fail(std::string(opcode_name(inst.opcode())) + " on a float type");
        return is_i32 ? i32 : i64;
    };

    switch (inst.opcode()) {
        case Opcode::neg:
            emit(by_type(BytecodeOp::neg_f32, BytecodeOp::neg_f64, BytecodeOp::neg_i32, BytecodeOp::neg_i64),
                 dst, get_reg(inst.operand(0)));
            return;
        case Opcode::sqrt_f32: emit(BytecodeOp::sqrt_f32, dst, get_reg(inst.operand(0))); return;
        case Opcode::sqrt_f64: emit(BytecodeOp::sqrt_f64, dst, get_reg(inst.operand(0))); return;
        case Opcode::fabs_f32: emit(BytecodeOp::fabs_f32, dst, get_reg(inst.operand(0))); return;
        case Opcode::fabs_f64: emit(BytecodeOp::fabs_f64, dst, get_reg(inst.operand(0))); return;
        case Opcode::floor_f32: emit(BytecodeOp::floor_f32, dst, get_reg(inst.operand(0))); return;
        case Opcode::floor_f64: emit(BytecodeOp::floor_f64, dst, get_reg(inst.operand(0))); return;
        case Opcode::ceil_f32: emit(BytecodeOp::ceil_f32, dst, get_reg(inst.operand(0))); return;
        case Opcode::ceil_f64: emit(BytecodeOp::ceil_f64, dst, get_reg(inst.operand(0))); return;
        case Opcode::round_f32: emit(BytecodeOp::round_f32, dst, get_reg(inst.operand(0))); return;
        case Opcode::round_f64: emit(BytecodeOp::round_f64, dst, get_reg(inst.operand(0))); return;
        case Opcode::fma_f32:
        case Opcode::fma_f64:
            // dst never shares a register with an operand (closed live
            // ranges), so it can hold the addend first.
            emit(BytecodeOp::mov, dst, get_reg(inst.operand(2)));
            emit(inst.opcode() == Opcode::fma_f32 ? BytecodeOp::fma_f32 : BytecodeOp::fma_f64, dst,
                 get_reg(inst.operand(0)), get_reg(inst.operand(1)));
            return;
        default:
            break;
    }

    BcReg src1 = get_reg(inst.operand(0));
    BcReg src2 = get_reg(inst.operand(1));
    BytecodeOp op = BytecodeOp::nop;
    switch (inst.opcode()) {
        case Opcode::add: op = by_type(BytecodeOp::add_f32, BytecodeOp::add_f64, BytecodeOp::add_i32, BytecodeOp::add_i64); break;
        case Opcode::sub: op = by_type(BytecodeOp::sub_f32, BytecodeOp::sub_f64, BytecodeOp::sub_i32, BytecodeOp::sub_i64); break;
        case Opcode::mul: op = by_type(BytecodeOp::mul_f32, BytecodeOp::mul_f64, BytecodeOp::mul_i32, BytecodeOp::mul_i64); break;
        case Opcode::sdiv: op = by_type(BytecodeOp::fdiv_f32, BytecodeOp::fdiv_f64, BytecodeOp::sdiv_i32, BytecodeOp::sdiv_i64); break;
        case Opcode::udiv: op = int_only(BytecodeOp::udiv_i32, BytecodeOp::udiv_i64); break;
        case Opcode::smod: op = int_only(BytecodeOp::smod_i32, BytecodeOp::smod_i64); break;
        case Opcode::umod: op = int_only(BytecodeOp::umod_i32, BytecodeOp::umod_i64); break;
        case Opcode::sadd_overflow: op = int_only(BytecodeOp::sadd_overflow_i32, BytecodeOp::sadd_overflow_i64); break;
        case Opcode::ssub_overflow: op = int_only(BytecodeOp::ssub_overflow_i32, BytecodeOp::ssub_overflow_i64); break;
        case Opcode::smul_overflow: op = int_only(BytecodeOp::smul_overflow_i32, BytecodeOp::smul_overflow_i64); break;
        case Opcode::uadd_overflow: op = int_only(BytecodeOp::uadd_overflow_i32, BytecodeOp::uadd_overflow_i64); break;
        case Opcode::usub_overflow: op = int_only(BytecodeOp::usub_overflow_i32, BytecodeOp::usub_overflow_i64); break;
        case Opcode::umul_overflow: op = int_only(BytecodeOp::umul_overflow_i32, BytecodeOp::umul_overflow_i64); break;
        case Opcode::fmin_f32: op = BytecodeOp::fmin_f32; break;
        case Opcode::fmin_f64: op = BytecodeOp::fmin_f64; break;
        case Opcode::fmax_f32: op = BytecodeOp::fmax_f32; break;
        case Opcode::fmax_f64: op = BytecodeOp::fmax_f64; break;
        default: fail("not arithmetic");
    }
    emit(op, dst, src1, src2);
}

void FunctionCompilerContext::lower_bitwise(const Instruction& inst) {
    BcReg dst = get_result_reg(inst);
    const bool is_i32 = is_narrow_int(inst.operand(0)->type());
    auto pick = [&](BytecodeOp i32, BytecodeOp i64) { return is_i32 ? i32 : i64; };

    switch (inst.opcode()) {
        case Opcode::not_: emit(pick(BytecodeOp::not_i32, BytecodeOp::not_i64), dst, get_reg(inst.operand(0))); return;
        case Opcode::clz: emit(pick(BytecodeOp::clz_i32, BytecodeOp::clz_i64), dst, get_reg(inst.operand(0))); return;
        case Opcode::ctz: emit(pick(BytecodeOp::ctz_i32, BytecodeOp::ctz_i64), dst, get_reg(inst.operand(0))); return;
        case Opcode::popcnt: emit(pick(BytecodeOp::popcnt_i32, BytecodeOp::popcnt_i64), dst, get_reg(inst.operand(0))); return;
        default: break;
    }

    BytecodeOp op = BytecodeOp::nop;
    switch (inst.opcode()) {
        case Opcode::and_: op = pick(BytecodeOp::and_i32, BytecodeOp::and_i64); break;
        case Opcode::or_: op = pick(BytecodeOp::or_i32, BytecodeOp::or_i64); break;
        case Opcode::xor_: op = pick(BytecodeOp::xor_i32, BytecodeOp::xor_i64); break;
        case Opcode::shl: op = pick(BytecodeOp::shl_i32, BytecodeOp::shl_i64); break;
        case Opcode::lshr: op = pick(BytecodeOp::lshr_i32, BytecodeOp::lshr_i64); break;
        case Opcode::ashr: op = pick(BytecodeOp::ashr_i32, BytecodeOp::ashr_i64); break;
        default: fail("not bitwise");
    }
    emit(op, dst, get_reg(inst.operand(0)), get_reg(inst.operand(1)));
}

void FunctionCompilerContext::lower_comparison(const Instruction& inst) {
    BcReg dst = get_result_reg(inst);

    if (inst.opcode() == Opcode::select) {
        // dst never shares a register with an operand, so it can hold the
        // false value while the condition and true value are still live.
        BcReg cond = get_reg(inst.operand(0));
        BcReg true_v = get_reg(inst.operand(1));
        BcReg false_v = get_reg(inst.operand(2));
        const bool vec = reg_type(dst).is_vector();
        emit(vec ? BytecodeOp::vmov : BytecodeOp::mov, dst, false_v);
        emit(BytecodeOp::select, dst, cond, true_v, vec ? 1u : 0u);
        return;
    }

    BcReg src1 = get_reg(inst.operand(0));
    BcReg src2 = get_reg(inst.operand(1));
    Type op_ty = inst.operand(0)->type();
    const bool is_f32 = op_ty == Type::f32();
    const bool is_f64 = op_ty == Type::f64();
    const bool is_i32 = is_narrow_int(op_ty);
    auto by_type = [&](BytecodeOp f32, BytecodeOp f64, BytecodeOp i32, BytecodeOp i64) {
        return is_f32 ? f32 : is_f64 ? f64 : is_i32 ? i32 : i64;
    };
    auto int_only = [&](BytecodeOp i32, BytecodeOp i64) {
        if (is_f32 || is_f64) fail("unsigned comparison of floats");
        return is_i32 ? i32 : i64;
    };

    BytecodeOp op = BytecodeOp::nop;
    switch (inst.opcode()) {
        case Opcode::eq: op = by_type(BytecodeOp::eq_f32, BytecodeOp::eq_f64, BytecodeOp::eq_i32, BytecodeOp::eq_i64); break;
        case Opcode::ne: op = by_type(BytecodeOp::ne_f32, BytecodeOp::ne_f64, BytecodeOp::ne_i32, BytecodeOp::ne_i64); break;
        case Opcode::slt: op = by_type(BytecodeOp::lt_f32, BytecodeOp::lt_f64, BytecodeOp::slt_i32, BytecodeOp::slt_i64); break;
        case Opcode::sle: op = by_type(BytecodeOp::le_f32, BytecodeOp::le_f64, BytecodeOp::sle_i32, BytecodeOp::sle_i64); break;
        case Opcode::sgt: op = by_type(BytecodeOp::gt_f32, BytecodeOp::gt_f64, BytecodeOp::sgt_i32, BytecodeOp::sgt_i64); break;
        case Opcode::sge: op = by_type(BytecodeOp::ge_f32, BytecodeOp::ge_f64, BytecodeOp::sge_i32, BytecodeOp::sge_i64); break;
        case Opcode::ult: op = int_only(BytecodeOp::ult_i32, BytecodeOp::ult_i64); break;
        case Opcode::ule: op = int_only(BytecodeOp::ule_i32, BytecodeOp::ule_i64); break;
        case Opcode::ugt: op = int_only(BytecodeOp::ugt_i32, BytecodeOp::ugt_i64); break;
        case Opcode::uge: op = int_only(BytecodeOp::uge_i32, BytecodeOp::uge_i64); break;
        default: fail("not a comparison");
    }
    emit(op, dst, src1, src2);
}

void FunctionCompilerContext::lower_memory(const Instruction& inst) {
    // The address base + offset: the base register and an imm24 offset,
    // or `tmp` holding the sum when the offset does not fit (tmp may be
    // the base itself).
    auto address = [&](BcReg base, int64_t offset, BcReg& out_base, int32_t& out_off, BcReg tmp) {
        if (fits_imm24(offset)) {
            out_base = base;
            out_off = static_cast<int32_t>(offset);
            return;
        }
        BcReg k = tmp == base ? scratch_reg2 : tmp;
        emit_const64(k, static_cast<uint64_t>(offset));
        emit(BytecodeOp::add_i64, tmp, base, k);
        out_base = tmp;
        out_off = 0;
    };
    // scratch_reg <- base + sext(idx) * scale, for the indexed forms.
    auto indexed_base = [&](const Value* base_v, const Value* idx_v, uint32_t scale) {
        BcReg idx = get_reg(idx_v);
        if (is_narrow_int(idx_v->type())) {
            emit(BytecodeOp::sext64, scratch_reg2, idx);
            idx = scratch_reg2;
        }
        if (scale > 255) fail("index scale out of range");
        emit(BytecodeOp::index_addr, scratch_reg, get_reg(base_v), idx, scale);
    };

    switch (inst.opcode()) {
        case Opcode::alloca_: {
            BcReg dst = get_result_reg(inst);
            int32_t size = inst.imm_i32();
            int32_t align = inst.offset();
            if (align <= 0) align = 16;
            if (align > 255 || (align & (align - 1)) != 0) fail("alloca alignment " + std::to_string(align));
            emit(BytecodeOp::alloca_, dst, 0, 0, static_cast<uint32_t>(align));
            // The size is the next code word (read by the interpreter).
            out.emit(static_cast<BytecodeWord>(static_cast<uint32_t>(size > 0 ? size : 0)));
            break;
        }
        case Opcode::load: {
            BcReg dst = get_result_reg(inst);
            BcReg base = 0;
            int32_t off = 0;
            address(get_reg(inst.operand(0)), inst.offset(), base, off, scratch_reg);
            if (inst.type().is_vector()) {
                emit_abi(BytecodeOp::vload, dst, base, off);
            } else {
                emit_abi(load_op(inst.type()), dst, base, off);
            }
            break;
        }
        case Opcode::store: {
            BcReg val = get_reg(inst.operand(1));
            BcReg base = 0;
            int32_t off = 0;
            address(get_reg(inst.operand(0)), inst.offset(), base, off, scratch_reg);
            Type mt = inst.memory_type().is_void() ? inst.operand(1)->type() : inst.memory_type();
            if (mt.is_vector()) {
                emit_abi(BytecodeOp::vstore, val, base, off);
            } else {
                emit_abi(store_op(mt), val, base, off);
            }
            break;
        }
        case Opcode::load_indexed: {
            BcReg dst = get_result_reg(inst);
            indexed_base(inst.operand(0), inst.operand(1), inst.scale());
            BcReg base = 0;
            int32_t off = 0;
            address(scratch_reg, inst.offset(), base, off, scratch_reg);
            emit_abi(inst.type().is_vector() ? BytecodeOp::vload : load_op(inst.type()), dst, base, off);
            break;
        }
        case Opcode::store_indexed: {
            BcReg val = get_reg(inst.operand(2));
            indexed_base(inst.operand(0), inst.operand(1), inst.scale());
            BcReg base = 0;
            int32_t off = 0;
            address(scratch_reg, inst.offset(), base, off, scratch_reg);
            Type mt = inst.memory_type().is_void() ? inst.operand(2)->type() : inst.memory_type();
            emit_abi(mt.is_vector() ? BytecodeOp::vstore : store_op(mt), val, base, off);
            break;
        }
        case Opcode::write_barrier:
            emit(BytecodeOp::write_barrier, 0, get_reg(inst.operand(0)), get_reg(inst.operand(1)));
            break;
        default:
            fail("not a memory operation");
    }
}

void FunctionCompilerContext::lower_call(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::func_addr:
            emit_ai(BytecodeOp::func_addr, get_result_reg(inst), static_cast<int32_t>(add_string(inst.symbol())));
            break;
        case Opcode::call:
        case Opcode::patchable_call: {
            BcReg dst = result_reg_or_none(inst);
            CallSiteInfo cs;
            cs.callee = std::string(inst.symbol());
            if (inst.opcode() == Opcode::patchable_call) {
                cs.extra_symbol = std::string(inst.extra_symbol());
                cs.patchable = true;
            }
            cs.dst_reg = dst;
            for (size_t i = 0; i < inst.operand_count(); ++i) cs.arg_regs.push_back(get_reg(inst.operand(i)));
            uint32_t cs_idx = out.add_call_site(std::move(cs));
            emit_ai(inst.opcode() == Opcode::call ? BytecodeOp::call : BytecodeOp::patchable_call, dst,
                    static_cast<int32_t>(cs_idx));
            break;
        }
        case Opcode::call_indirect: {
            BcReg dst = result_reg_or_none(inst);
            CallSiteInfo cs;
            cs.callee_reg = get_reg(inst.operand(0));
            cs.dst_reg = dst;
            cs.site_id = inst.site_id();
            for (size_t i = 1; i < inst.operand_count(); ++i) cs.arg_regs.push_back(get_reg(inst.operand(i)));
            uint32_t cs_idx = out.add_call_site(std::move(cs));
            emit_ai(BytecodeOp::call_indirect, dst, static_cast<int32_t>(cs_idx));
            break;
        }
        default:
            fail("not a call");
    }
}

void FunctionCompilerContext::lower_runtime_gc(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::safepoint:
            emit(BytecodeOp::safepoint, 0);
            break;
        case Opcode::guard: {
            GuardInfo gi;
            gi.resume_id = inst.resume_id();
            gi.exit_stub = std::string(inst.symbol());
            for (const auto* v : inst.state_map()) {
                if (v) gi.state_regs.push_back(get_reg(v));
            }
            uint32_t g_idx = out.add_guard(std::move(gi));
            emit_ai(BytecodeOp::guard, get_reg(inst.operand(0)), static_cast<int32_t>(g_idx));
            break;
        }
        case Opcode::resume_point:
            emit(BytecodeOp::resume_point, 0);
            break;
        case Opcode::osr_entry: {
            OsrEntry oe;
            oe.pc = static_cast<uint32_t>(out.current_pc());
            out.osr_entries.push_back(std::move(oe));
            emit(BytecodeOp::osr_entry, 0);
            break;
        }
        default:
            fail("not a runtime operation");
    }
}

void FunctionCompilerContext::lower_coroutine(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::throw_:
            emit(BytecodeOp::throw_, get_reg(inst.operand(0)));
            break;
        case Opcode::landing_pad:
            emit(BytecodeOp::landing_pad, get_result_reg(inst));
            break;
        case Opcode::resume:
            emit(BytecodeOp::resume, inst.operand_count() > 0 && inst.operand(0) ? get_reg(inst.operand(0)) : kNoReg);
            break;
        case Opcode::coro_create: {
            BcReg dst = get_result_reg(inst);
            CallSiteInfo cs;
            cs.callee = std::string(inst.symbol());
            cs.dst_reg = dst;
            for (size_t i = 0; i < inst.operand_count(); ++i) cs.arg_regs.push_back(get_reg(inst.operand(i)));
            uint32_t cs_idx = out.add_call_site(std::move(cs));
            emit_ai(BytecodeOp::coro_create, dst, static_cast<int32_t>(cs_idx));
            break;
        }
        case Opcode::coro_suspend: {
            BcReg yield_reg = inst.operand_count() > 0 && inst.operand(0) ? get_reg(inst.operand(0)) : kNoReg;
            BcReg dst_reg = result_reg_or_none(inst);
            // The resume id is the next code word.
            emit(BytecodeOp::coro_suspend, dst_reg, yield_reg);
            out.emit(static_cast<BytecodeWord>(inst.resume_id()));
            break;
        }
        case Opcode::coro_resume: {
            BcReg input_reg = inst.operand_count() > 1 && inst.operand(1) ? get_reg(inst.operand(1)) : kNoReg;
            emit(BytecodeOp::coro_resume, get_result_reg(inst), get_reg(inst.operand(0)), input_reg);
            break;
        }
        case Opcode::coro_destroy:
            emit(BytecodeOp::coro_destroy, 0, get_reg(inst.operand(0)));
            break;
        default:
            fail("not an exception or coroutine operation");
    }
}

void FunctionCompilerContext::lower_vector(const Instruction& inst) {
    switch (inst.opcode()) {
        case Opcode::vstore:
            if (!fits_imm24(inst.offset())) fail("vstore offset out of range");
            emit_abi(BytecodeOp::vstore, get_reg(inst.operand(1)), get_reg(inst.operand(0)), inst.offset());
            return;
        case Opcode::vload:
            if (!fits_imm24(inst.offset())) fail("vload offset out of range");
            emit_abi(BytecodeOp::vload, get_result_reg(inst), get_reg(inst.operand(0)), inst.offset());
            return;
        default:
            break;
    }

    BcReg dst = get_result_reg(inst);
    switch (inst.opcode()) {
        case Opcode::vzero: emit(BytecodeOp::vzero, dst); return;
        case Opcode::vbroadcast: emit(BytecodeOp::vbroadcast, dst, get_reg(inst.operand(0))); return;
        case Opcode::vextract_lane:
            if (inst.lane() > 255) fail("lane out of range");
            emit(BytecodeOp::vextract_lane, dst, get_reg(inst.operand(0)), 0, inst.lane());
            return;
        case Opcode::vinsert_lane:
            if (inst.lane() > 255) fail("lane out of range");
            emit(BytecodeOp::vinsert_lane, dst, get_reg(inst.operand(0)), get_reg(inst.operand(1)), inst.lane());
            return;
        case Opcode::vshuffle:
            // The mask is the next code word.
            emit(BytecodeOp::vshuffle, dst, get_reg(inst.operand(0)), get_reg(inst.operand(1)));
            out.emit(static_cast<BytecodeWord>(inst.shuffle_mask()));
            return;
        case Opcode::vfma:
            // dst holds the addend (it shares no register with an operand).
            emit(BytecodeOp::vmov, dst, get_reg(inst.operand(2)));
            emit(BytecodeOp::vfma, dst, get_reg(inst.operand(0)), get_reg(inst.operand(1)));
            return;
        default:
            break;
    }

    BytecodeOp op = BytecodeOp::nop;
    switch (inst.opcode()) {
        case Opcode::vadd: op = BytecodeOp::vadd; break;
        case Opcode::vsub: op = BytecodeOp::vsub; break;
        case Opcode::vmul: op = BytecodeOp::vmul; break;
        case Opcode::vdiv: op = BytecodeOp::vdiv; break;
        case Opcode::vneg: op = BytecodeOp::vneg; break;
        case Opcode::vmin: op = BytecodeOp::vmin; break;
        case Opcode::vmax: op = BytecodeOp::vmax; break;
        case Opcode::vsqrt: op = BytecodeOp::vsqrt; break;
        case Opcode::vand: op = BytecodeOp::vand; break;
        case Opcode::vor: op = BytecodeOp::vor; break;
        case Opcode::vxor: op = BytecodeOp::vxor; break;
        case Opcode::vnot: op = BytecodeOp::vnot; break;
        default: fail("not a vector operation");
    }
    if (inst.operand_count() == 1) {
        emit(op, dst, get_reg(inst.operand(0)));
    } else {
        emit(op, dst, get_reg(inst.operand(0)), get_reg(inst.operand(1)));
    }
}

} // namespace brass::detail
