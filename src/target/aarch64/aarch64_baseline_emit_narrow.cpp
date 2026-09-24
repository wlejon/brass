// AArch64 baseline tier: integer operations on i8 / i16 operands
// (docs/semantics.md, "Narrow integers"). A narrow value lives zero-extended
// in the low 4 bytes of its slot; an operation reads only the value's width,
// sign- or zero-extended to 64 bits as its signedness requires, and writes a
// narrow result back zero-extended.
#include "aarch64_baseline_emit_internal.hpp"

namespace brass::aarch64 {

namespace {

Condition int_condition(Opcode op) {
    switch (op) {
        case Opcode::eq: return Condition::EQ;
        case Opcode::ne: return Condition::NE;
        case Opcode::slt: return Condition::LT;
        case Opcode::ult: return Condition::CC;
        case Opcode::sle: return Condition::LE;
        case Opcode::ule: return Condition::LS;
        case Opcode::sgt: return Condition::GT;
        case Opcode::ugt: return Condition::HI;
        case Opcode::sge: return Condition::GE;
        default: return Condition::CS; // uge
    }
}

} // namespace

bool emit_baseline_aarch64_narrow(AArch64BaselineEmitter& em, const Instruction& inst) {
    const size_t n = inst.operand_count();
    if (n == 0 || n > 2 || !inst.operand(0) || !inst.result()) return false;
    const Type t = inst.operand(0)->type();
    const unsigned bits = t == Type::i8() ? 8u : t == Type::i16() ? 16u : 0u;
    if (!bits) return false;
    auto& enc = em.enc;
    const Opcode op = inst.opcode();

    auto load = [&](GPR r, const Value* v, bool sign) {
        if (sign) {
            if (bits == 8) enc.ldrsb(r, em.slot_addr(v, 1));
            else enc.ldrsh(r, em.slot_addr(v, 2));
        } else {
            if (bits == 8) enc.ldrb(r, em.slot_addr(v, 1));
            else enc.ldrh(r, em.slot_addr(v, 2));
        }
    };
    auto load2 = [&](bool sign) {
        load(GPR::X0, inst.operand(0), sign);
        load(GPR::X1, inst.operand(1), sign);
    };
    // The low `bits` of `r`, zero-extended.
    auto narrow = [&](GPR dst, GPR r) {
        if (bits == 8) enc.uxtb(dst, r);
        else enc.uxth(dst, r);
    };
    auto store_value = [&]() {
        narrow(GPR::X0, GPR::X0);
        enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
    };
    auto store_flag = [&](Condition c) {
        enc.cset32(GPR::X0, c);
        enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
    };
    // Overflow: the exact result in X0 differs from its own narrow extension.
    auto store_overflow = [&](bool sign) {
        if (sign) {
            if (bits == 8) enc.sxtb(GPR::X1, GPR::X0);
            else enc.sxth(GPR::X1, GPR::X0);
        } else {
            narrow(GPR::X1, GPR::X0);
        }
        enc.cmp(GPR::X0, GPR::X1);
        store_flag(Condition::NE);
    };

    switch (op) {
        case Opcode::add: load2(false); enc.add(GPR::X0, GPR::X0, GPR::X1); store_value(); return true;
        case Opcode::sub: load2(false); enc.sub(GPR::X0, GPR::X0, GPR::X1); store_value(); return true;
        case Opcode::mul: load2(false); enc.mul(GPR::X0, GPR::X0, GPR::X1); store_value(); return true;
        case Opcode::and_: load2(false); enc.and_(GPR::X0, GPR::X0, GPR::X1); store_value(); return true;
        case Opcode::or_: load2(false); enc.orr(GPR::X0, GPR::X0, GPR::X1); store_value(); return true;
        case Opcode::xor_: load2(false); enc.eor(GPR::X0, GPR::X0, GPR::X1); store_value(); return true;
        case Opcode::neg: load(GPR::X0, inst.operand(0), false); enc.neg(GPR::X0, GPR::X0); store_value(); return true;
        case Opcode::not_: load(GPR::X0, inst.operand(0), false); enc.mvn(GPR::X0, GPR::X0); store_value(); return true;
        // Shifts of the widened value at 32 bits (count & 31), as the interpreters.
        case Opcode::shl: load2(false); enc.lsl32(GPR::X0, GPR::X0, GPR::X1); store_value(); return true;
        case Opcode::lshr: load2(false); enc.lsr32(GPR::X0, GPR::X0, GPR::X1); store_value(); return true;
        case Opcode::ashr:
            load(GPR::X0, inst.operand(0), true);
            load(GPR::X1, inst.operand(1), false);
            enc.asr32(GPR::X0, GPR::X0, GPR::X1);
            store_value();
            return true;
        case Opcode::sdiv:
        case Opcode::smod: {
            load2(true);
            enc.sdiv(GPR::X2, GPR::X0, GPR::X1);
            if (op == Opcode::smod) {
                enc.msub(GPR::X0, GPR::X2, GPR::X1, GPR::X0);
            } else {
                enc.mov(GPR::X0, GPR::X2);
            }
            store_value();
            return true;
        }
        case Opcode::udiv:
        case Opcode::umod: {
            load2(false);
            enc.udiv(GPR::X2, GPR::X0, GPR::X1);
            if (op == Opcode::umod) {
                enc.msub(GPR::X0, GPR::X2, GPR::X1, GPR::X0);
            } else {
                enc.mov(GPR::X0, GPR::X2);
            }
            store_value();
            return true;
        }
        case Opcode::clz:
            load(GPR::X1, inst.operand(0), false);
            enc.clz32(GPR::X0, GPR::X1);
            enc.sub32(GPR::X0, GPR::X0, static_cast<uint32_t>(32 - bits));
            enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
            return true;
        case Opcode::ctz:
            load(GPR::X0, inst.operand(0), false);
            enc.mov32(GPR::X2, static_cast<uint32_t>(1u << bits));
            enc.orr32(GPR::X0, GPR::X0, GPR::X2);
            enc.rbit32(GPR::X0, GPR::X0);
            enc.clz32(GPR::X0, GPR::X0);
            enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
            return true;
        case Opcode::popcnt:
            load(GPR::X0, inst.operand(0), false);
            enc.fmov_from_gpr(FPR::V0, GPR::X0);
            enc.cnt_8b(FPR::V0, FPR::V0);
            enc.uaddlv_h(FPR::V0, FPR::V0);
            enc.fmov_to_gpr32(GPR::X0, FPR::V0);
            enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
            return true;
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::ult:
        case Opcode::ule:
        case Opcode::ugt:
        case Opcode::uge:
            load2(false);
            enc.cmp(GPR::X0, GPR::X1);
            store_flag(int_condition(op));
            return true;
        case Opcode::slt:
        case Opcode::sle:
        case Opcode::sgt:
        case Opcode::sge:
            load2(true);
            enc.cmp(GPR::X0, GPR::X1);
            store_flag(int_condition(op));
            return true;
        case Opcode::sadd_overflow: load2(true); enc.add(GPR::X0, GPR::X0, GPR::X1); store_overflow(true); return true;
        case Opcode::ssub_overflow: load2(true); enc.sub(GPR::X0, GPR::X0, GPR::X1); store_overflow(true); return true;
        case Opcode::smul_overflow: load2(true); enc.mul(GPR::X0, GPR::X0, GPR::X1); store_overflow(true); return true;
        case Opcode::uadd_overflow: load2(false); enc.add(GPR::X0, GPR::X0, GPR::X1); store_overflow(false); return true;
        case Opcode::usub_overflow: load2(false); enc.sub(GPR::X0, GPR::X0, GPR::X1); store_overflow(false); return true;
        case Opcode::umul_overflow: load2(false); enc.mul(GPR::X0, GPR::X0, GPR::X1); store_overflow(false); return true;
        default:
            return false;
    }
}

} // namespace brass::aarch64
