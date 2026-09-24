// x64 baseline tier: integer operations on i8 / i16 operands
// (docs/semantics.md, "Narrow integers"). A narrow value lives zero-extended
// in the low 4 bytes of its slot; an operation reads only the value's width,
// sign- or zero-extended to 64 bits as its signedness requires, and writes a
// narrow result back zero-extended.
#include "baseline_emit_internal.hpp"

namespace brass::codegen {

using namespace brass::x64;

bool emit_baseline_x64_narrow(X64BaselineEmitter& em, const Instruction& inst) {
    const size_t n = inst.operand_count();
    if (n == 0 || n > 2 || !inst.operand(0) || !inst.result()) return false;
    const Type t = inst.operand(0)->type();
    const unsigned bits = t == Type::i8() ? 8u : t == Type::i16() ? 16u : 0u;
    if (!bits) return false;
    auto& enc = em.enc;
    const Opcode op = inst.opcode();

    auto load = [&](GPR r, const Value* v, bool sign) {
        const MemAddress m = em.slot_addr(v);
        if (bits == 8) { if (sign) enc.movsx8(r, m); else enc.movzx8(r, m); }
        else { if (sign) enc.movsx16(r, m); else enc.movzx16(r, m); }
    };
    auto load2 = [&](bool sign) {
        load(GPR::RAX, inst.operand(0), sign);
        load(GPR::RCX, inst.operand(1), sign);
    };
    // The low `bits` of `r`, zero-extended.
    auto narrow = [&](GPR dst, GPR r) {
        if (bits == 8) enc.movzx8(dst, r); else enc.movzx16(dst, r);
    };
    auto store_value = [&]() {
        narrow(GPR::RAX, GPR::RAX);
        enc.mov32(em.slot_addr(inst.result()), GPR::RAX);
    };
    auto store_flag = [&](Condition c) {
        enc.setcc(c, GPR::RAX);
        enc.movzx8(GPR::RAX, GPR::RAX);
        enc.mov32(em.slot_addr(inst.result()), GPR::RAX);
    };
    // Overflow: the exact result in RAX differs from its own narrow
    // extension.
    auto store_overflow = [&](bool sign) {
        if (sign) { if (bits == 8) enc.movsx8(GPR::RCX, GPR::RAX); else enc.movsx16(GPR::RCX, GPR::RAX); }
        else narrow(GPR::RCX, GPR::RAX);
        enc.cmp(GPR::RAX, GPR::RCX);
        store_flag(Condition::NE);
    };

    switch (op) {
        case Opcode::add: load2(false); enc.add(GPR::RAX, GPR::RCX); store_value(); return true;
        case Opcode::sub: load2(false); enc.sub(GPR::RAX, GPR::RCX); store_value(); return true;
        case Opcode::mul: load2(false); enc.imul(GPR::RAX, GPR::RCX); store_value(); return true;
        case Opcode::and_: load2(false); enc.and_(GPR::RAX, GPR::RCX); store_value(); return true;
        case Opcode::or_: load2(false); enc.or_(GPR::RAX, GPR::RCX); store_value(); return true;
        case Opcode::xor_: load2(false); enc.xor_(GPR::RAX, GPR::RCX); store_value(); return true;
        case Opcode::neg: load(GPR::RAX, inst.operand(0), false); enc.neg(GPR::RAX); store_value(); return true;
        case Opcode::not_: load(GPR::RAX, inst.operand(0), false); enc.not_(GPR::RAX); store_value(); return true;
        // Shifts of the widened value at 32 bits (count & 31), as the
        // interpreters.
        case Opcode::shl: load2(false); enc.shl32(GPR::RAX); store_value(); return true;
        case Opcode::lshr: load2(false); enc.shr32(GPR::RAX); store_value(); return true;
        case Opcode::ashr:
            load(GPR::RAX, inst.operand(0), true);
            load(GPR::RCX, inst.operand(1), false);
            enc.sar32(GPR::RAX);
            store_value();
            return true;
        case Opcode::sdiv:
        case Opcode::smod:
            load2(true);
            enc.cqo();
            enc.idiv_wrapping(GPR::RCX);
            if (op == Opcode::smod) enc.mov(GPR::RAX, GPR::RDX);
            store_value();
            return true;
        case Opcode::udiv:
        case Opcode::umod:
            load2(false);
            enc.xor32(GPR::RDX, GPR::RDX);
            enc.div(GPR::RCX);
            if (op == Opcode::umod) enc.mov(GPR::RAX, GPR::RDX);
            store_value();
            return true;
        case Opcode::clz:
            load(GPR::RCX, inst.operand(0), false);
            enc.lzcnt32(GPR::RAX, GPR::RCX);
            enc.sub32(GPR::RAX, static_cast<int32_t>(32 - bits));
            enc.mov32(em.slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::ctz:
            // A bit just above the width makes ctz of zero the width.
            load(GPR::RCX, inst.operand(0), false);
            enc.or32(GPR::RCX, static_cast<int32_t>(1u << bits));
            enc.tzcnt32(GPR::RAX, GPR::RCX);
            enc.mov32(em.slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::popcnt:
            load(GPR::RCX, inst.operand(0), false);
            enc.popcnt32(GPR::RAX, GPR::RCX);
            enc.mov32(em.slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::eq: load2(false); enc.cmp(GPR::RAX, GPR::RCX); store_flag(Condition::E); return true;
        case Opcode::ne: load2(false); enc.cmp(GPR::RAX, GPR::RCX); store_flag(Condition::NE); return true;
        case Opcode::ult: load2(false); enc.cmp(GPR::RAX, GPR::RCX); store_flag(Condition::B); return true;
        case Opcode::ule: load2(false); enc.cmp(GPR::RAX, GPR::RCX); store_flag(Condition::BE); return true;
        case Opcode::ugt: load2(false); enc.cmp(GPR::RAX, GPR::RCX); store_flag(Condition::A); return true;
        case Opcode::uge: load2(false); enc.cmp(GPR::RAX, GPR::RCX); store_flag(Condition::AE); return true;
        case Opcode::slt: load2(true); enc.cmp(GPR::RAX, GPR::RCX); store_flag(Condition::L); return true;
        case Opcode::sle: load2(true); enc.cmp(GPR::RAX, GPR::RCX); store_flag(Condition::LE); return true;
        case Opcode::sgt: load2(true); enc.cmp(GPR::RAX, GPR::RCX); store_flag(Condition::G); return true;
        case Opcode::sge: load2(true); enc.cmp(GPR::RAX, GPR::RCX); store_flag(Condition::GE); return true;
        case Opcode::sadd_overflow: load2(true); enc.add(GPR::RAX, GPR::RCX); store_overflow(true); return true;
        case Opcode::ssub_overflow: load2(true); enc.sub(GPR::RAX, GPR::RCX); store_overflow(true); return true;
        case Opcode::smul_overflow: load2(true); enc.imul(GPR::RAX, GPR::RCX); store_overflow(true); return true;
        case Opcode::uadd_overflow: load2(false); enc.add(GPR::RAX, GPR::RCX); store_overflow(false); return true;
        case Opcode::usub_overflow: load2(false); enc.sub(GPR::RAX, GPR::RCX); store_overflow(false); return true;
        case Opcode::umul_overflow: load2(false); enc.imul(GPR::RAX, GPR::RCX); store_overflow(false); return true;
        default:
            return false;
    }
}

} // namespace brass::codegen
