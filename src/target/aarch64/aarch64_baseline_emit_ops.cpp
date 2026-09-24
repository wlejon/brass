// AArch64 baseline tier: integer, pointer and memory opcodes. Floating-point
// arithmetic, comparisons of floats and float conversions are in
// aarch64_baseline_emit_fp.cpp. Slot conventions: codegen/baseline_frame.hpp.
#include "aarch64_baseline_emit_internal.hpp"
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <cstring>

namespace brass::aarch64 {

using namespace brass::codegen;

namespace {

bool is_float_op(const Instruction& inst) {
    if (inst.produces_value() && inst.type().is_float()) return true;
    return is_comparison(inst.opcode()) && inst.operand(0)->type().is_float();
}

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

// The effective type of a memory access: the instruction's memory type when
// set, else the loaded / stored value's.
Type access_type(const Instruction& inst, Type value_type) {
    Type mt = inst.memory_type();
    return mt.is_void() ? value_type : mt;
}

// A scalar load from [X0 + disp] into the result's slot (narrow integers
// zero-extended, as the x64 tier's movzx).
void emit_mem_load(AArch64BaselineEmitter& em, const Instruction& inst, int64_t disp) {
    auto& enc = em.enc;
    const Type t = access_type(inst, inst.type());
    if (t.is_float()) {
        if (bl_is_f32(t)) { enc.ldr_s(FPR::V0, em.based(GPR::X0, disp, 4)); em.store_fp(inst.result(), FPR::V0); }
        else { enc.ldr(FPR::V0, em.based(GPR::X0, disp, 8)); em.store_fp(inst.result(), FPR::V0); }
        return;
    }
    switch (t.size_in_bytes()) {
        case 1: enc.ldrb(GPR::X1, em.based(GPR::X0, disp, 1)); break;
        case 2: enc.ldrh(GPR::X1, em.based(GPR::X0, disp, 2)); break;
        case 4: enc.ldr32(GPR::X1, em.based(GPR::X0, disp, 4)); break;
        default: enc.ldr(GPR::X1, em.based(GPR::X0, disp, 8)); break;
    }
    em.store_gpr(inst.result(), GPR::X1);
}

// A scalar store of `val` to [X0 + disp], at the access width.
void emit_mem_store(AArch64BaselineEmitter& em, const Instruction& inst, int64_t disp, const Value* val) {
    auto& enc = em.enc;
    const Type t = access_type(inst, val->type());
    if (t.is_float()) {
        if (bl_is_f32(t)) { enc.ldr_s(FPR::V0, em.slot_addr(val, 4)); enc.str_s(FPR::V0, em.based(GPR::X0, disp, 4)); }
        else { enc.ldr(FPR::V0, em.slot_addr(val, 8)); enc.str(FPR::V0, em.based(GPR::X0, disp, 8)); }
        return;
    }
    enc.ldr(GPR::X1, em.slot_addr(val, 8));
    switch (t.size_in_bytes()) {
        case 1: enc.strb(GPR::X1, em.based(GPR::X0, disp, 1)); break;
        case 2: enc.strh(GPR::X1, em.based(GPR::X0, disp, 2)); break;
        case 4: enc.str32(GPR::X1, em.based(GPR::X0, disp, 4)); break;
        default: enc.str(GPR::X1, em.based(GPR::X0, disp, 8)); break;
    }
}

// X0 = base + index * scale (a narrow index sign-extended), for an access
// at [X0 + offset].
void indexed_base(AArch64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    enc.ldr(GPR::X0, em.slot_addr(inst.operand(0), 8));
    const Value* idx = inst.operand(1);
    if (bl_is_int32(idx->type())) enc.ldrsw(GPR::X1, em.slot_addr(idx, 4));
    else enc.ldr(GPR::X1, em.slot_addr(idx, 8));
    const int64_t scale = inst.scale();
    if (scale > 0 && (scale & (scale - 1)) == 0) {
        uint8_t sh = 0;
        while ((int64_t{1} << sh) < scale) ++sh;
        if (sh) enc.lsl(GPR::X1, GPR::X1, sh);
    } else {
        enc.mov(GPR::X2, static_cast<uint64_t>(scale));
        enc.mul(GPR::X1, GPR::X1, GPR::X2);
    }
    enc.add(GPR::X0, GPR::X0, GPR::X1);
}

// The result is the i32 overflow flag of the operation at operand 0's width.
void emit_overflow(AArch64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const bool w32 = bl_is_int32(inst.operand(0)->type());
    em.load_gpr(GPR::X0, inst.operand(0));
    em.load_gpr(GPR::X1, inst.operand(1));
    Condition cond = Condition::VS;
    switch (inst.opcode()) {
        case Opcode::sadd_overflow: if (w32) enc.adds32(GPR::X2, GPR::X0, GPR::X1); else enc.adds(GPR::X2, GPR::X0, GPR::X1); break;
        case Opcode::ssub_overflow: if (w32) enc.subs32(GPR::X2, GPR::X0, GPR::X1); else enc.subs(GPR::X2, GPR::X0, GPR::X1); break;
        case Opcode::uadd_overflow:
            if (w32) enc.adds32(GPR::X2, GPR::X0, GPR::X1); else enc.adds(GPR::X2, GPR::X0, GPR::X1);
            cond = Condition::CS; // carry out
            break;
        case Opcode::usub_overflow:
            if (w32) enc.subs32(GPR::X2, GPR::X0, GPR::X1); else enc.subs(GPR::X2, GPR::X0, GPR::X1);
            cond = Condition::CC; // borrow
            break;
        case Opcode::smul_overflow:
            cond = Condition::NE;
            if (w32) {
                // The 64-bit product of the sign-extended halves overflows
                // when it is not its own low word sign-extended.
                enc.sxtw(GPR::X0, GPR::X0);
                enc.sxtw(GPR::X1, GPR::X1);
                enc.mul(GPR::X2, GPR::X0, GPR::X1);
                enc.sxtw(GPR::X3, GPR::X2);
                enc.cmp(GPR::X2, GPR::X3);
            } else {
                // The high word is not the low word's sign.
                enc.mul(GPR::X2, GPR::X0, GPR::X1);
                enc.smulh(GPR::X3, GPR::X0, GPR::X1);
                enc.asr(GPR::X4, GPR::X2, static_cast<uint8_t>(63));
                enc.cmp(GPR::X3, GPR::X4);
            }
            break;
        default: // umul_overflow: the high half is not zero
            cond = Condition::NE;
            if (w32) {
                // ldr w zero-extended both operands.
                enc.mul(GPR::X2, GPR::X0, GPR::X1);
                enc.lsr(GPR::X3, GPR::X2, static_cast<uint8_t>(32));
            } else {
                enc.umulh(GPR::X3, GPR::X0, GPR::X1);
            }
            enc.cmp(GPR::X3, 0u);
            break;
    }
    enc.cset32(GPR::X0, cond);
    enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
}

} // namespace

bool emit_baseline_aarch64_op(AArch64BaselineEmitter& emitter, const Instruction& inst) {
    if (emit_baseline_aarch64_narrow(emitter, inst)) return true;

    auto& enc = emitter.enc;
    const Opcode op = inst.opcode();

    if (is_float_op(inst) && op != Opcode::select && op != Opcode::load && op != Opcode::load_indexed &&
        op != Opcode::call && op != Opcode::call_indirect && op != Opcode::patchable_call) {
        return false; // aarch64_baseline_emit_fp.cpp
    }

    // Integer binary op at the result's width: X0 = op0 <op> op1.
    auto binop = [&](auto op32, auto op64) {
        const bool w32 = bl_is_int32(inst.type());
        emitter.load_gpr(GPR::X0, inst.operand(0));
        emitter.load_gpr(GPR::X1, inst.operand(1));
        if (w32) (enc.*op32)(GPR::X0, GPR::X0, GPR::X1);
        else (enc.*op64)(GPR::X0, GPR::X0, GPR::X1);
        emitter.store_gpr(inst.result(), GPR::X0);
    };
    using RRR = void (AArch64Encoder::*)(GPR, GPR, GPR);

    switch (op) {
        // Constants
        case Opcode::iconst_i32:
        case Opcode::patchable_const_i32:
            enc.mov32(GPR::X0, static_cast<uint32_t>(inst.imm_i32()));
            enc.str32(GPR::X0, emitter.slot_addr(inst.result(), 4));
            return true;
        case Opcode::iconst_i64:
        case Opcode::patchable_const_i64:
            enc.mov(GPR::X0, static_cast<uint64_t>(inst.imm_i64()));
            enc.str(GPR::X0, emitter.slot_addr(inst.result(), 8));
            return true;
        case Opcode::func_addr: {
            // Unresolved now: the lazy-link stub, callable before the target
            // is registered and the same address after. A stub is code, so
            // a data symbol (the module's own string, or one declared
            // `data`) must resolve here.
            const Module* mod = emitter.fn.parent();
            const bool data = mod && (mod->string_symbol(inst.symbol()) ||
                                      mod->has_symbol_role(inst.symbol(), SymbolRole::Data));
            void* addr = nullptr;
            if (data) {
                addr = emitter.resolve_sym(inst.symbol());
                if (!addr) {
                    throw_unsupported(kA64BaselineStage, "func_addr of data symbol " + std::string(inst.symbol()) +
                                                             ", which does not resolve, in " +
                                                             std::string(emitter.fn.name()));
                }
            } else {
                addr = emitter.resolve_or_stub(inst.symbol());
            }
            enc.mov(GPR::X0, reinterpret_cast<uint64_t>(addr));
            enc.str(GPR::X0, emitter.slot_addr(inst.result(), 8));
            return true;
        }

        // Integer conversions (as the interpreter: narrow values are i32s)
        case Opcode::sext_i64:
            if (inst.operand(0)->type() == Type::i8()) enc.ldrsb(GPR::X0, emitter.slot_addr(inst.operand(0), 1));
            else if (inst.operand(0)->type() == Type::i16()) enc.ldrsh(GPR::X0, emitter.slot_addr(inst.operand(0), 2));
            else enc.ldrsw(GPR::X0, emitter.slot_addr(inst.operand(0), 4));
            enc.str(GPR::X0, emitter.slot_addr(inst.result(), 8));
            return true;
        case Opcode::zext_i64:
            if (inst.operand(0)->type() == Type::i8()) enc.ldrb(GPR::X0, emitter.slot_addr(inst.operand(0), 1));
            else if (inst.operand(0)->type() == Type::i16()) enc.ldrh(GPR::X0, emitter.slot_addr(inst.operand(0), 2));
            else enc.ldr32(GPR::X0, emitter.slot_addr(inst.operand(0), 4));
            enc.str(GPR::X0, emitter.slot_addr(inst.result(), 8));
            return true;
        case Opcode::trunc_i32:
            enc.ldr32(GPR::X0, emitter.slot_addr(inst.operand(0), 4));
            emitter.store_gpr(inst.result(), GPR::X0);
            return true;
        case Opcode::trunc_i8:
            // The whole 32-bit slot, so a later 32-bit read sees 0..255.
            enc.ldrb(GPR::X0, emitter.slot_addr(inst.operand(0), 1));
            enc.str32(GPR::X0, emitter.slot_addr(inst.result(), 4));
            return true;

        // Integer arithmetic & logic
        case Opcode::add: binop(static_cast<RRR>(&AArch64Encoder::add32), static_cast<RRR>(&AArch64Encoder::add)); return true;
        case Opcode::sub: binop(static_cast<RRR>(&AArch64Encoder::sub32), static_cast<RRR>(&AArch64Encoder::sub)); return true;
        case Opcode::mul: binop(static_cast<RRR>(&AArch64Encoder::mul32), static_cast<RRR>(&AArch64Encoder::mul)); return true;
        case Opcode::and_: binop(static_cast<RRR>(&AArch64Encoder::and32), static_cast<RRR>(&AArch64Encoder::and_)); return true;
        case Opcode::or_: binop(static_cast<RRR>(&AArch64Encoder::orr32), static_cast<RRR>(&AArch64Encoder::orr)); return true;
        case Opcode::xor_: binop(static_cast<RRR>(&AArch64Encoder::eor32), static_cast<RRR>(&AArch64Encoder::eor)); return true;
        // Shifts: the count is masked to the width by the instruction, as
        // the interpreter masks it.
        case Opcode::shl: binop(static_cast<RRR>(&AArch64Encoder::lsl32), static_cast<RRR>(&AArch64Encoder::lsl)); return true;
        case Opcode::lshr: binop(static_cast<RRR>(&AArch64Encoder::lsr32), static_cast<RRR>(&AArch64Encoder::lsr)); return true;
        case Opcode::ashr: binop(static_cast<RRR>(&AArch64Encoder::asr32), static_cast<RRR>(&AArch64Encoder::asr)); return true;
        case Opcode::sdiv:
        case Opcode::smod:
        case Opcode::udiv:
        case Opcode::umod: {
            // A zero divisor traps (the x64 #DE); INT_MIN / -1 wraps, as
            // sdiv does.
            const bool w32 = bl_is_int32(inst.type());
            const bool sgn = (op == Opcode::sdiv || op == Opcode::smod);
            emitter.load_gpr(GPR::X0, inst.operand(0));
            emitter.load_gpr(GPR::X1, inst.operand(1));
            enc.brk_if_zero(GPR::X1, !w32, kBrkIntegerDivideByZero);
            if (w32) { if (sgn) enc.sdiv32(GPR::X2, GPR::X0, GPR::X1); else enc.udiv32(GPR::X2, GPR::X0, GPR::X1); }
            else { if (sgn) enc.sdiv(GPR::X2, GPR::X0, GPR::X1); else enc.udiv(GPR::X2, GPR::X0, GPR::X1); }
            if (op == Opcode::smod || op == Opcode::umod) {
                if (w32) enc.msub32(GPR::X2, GPR::X2, GPR::X1, GPR::X0);
                else enc.msub(GPR::X2, GPR::X2, GPR::X1, GPR::X0);
            }
            emitter.store_gpr(inst.result(), GPR::X2);
            return true;
        }
        case Opcode::neg:
        case Opcode::not_:
        case Opcode::clz:
        case Opcode::ctz:
        case Opcode::popcnt: {
            const bool w32 = bl_is_int32(inst.type());
            emitter.load_gpr(GPR::X0, inst.operand(0));
            switch (op) {
                case Opcode::neg: if (w32) enc.neg32(GPR::X0, GPR::X0); else enc.neg(GPR::X0, GPR::X0); break;
                case Opcode::not_: if (w32) enc.mvn32(GPR::X0, GPR::X0); else enc.mvn(GPR::X0, GPR::X0); break;
                case Opcode::clz: if (w32) enc.clz32(GPR::X0, GPR::X0); else enc.clz(GPR::X0, GPR::X0); break;
                case Opcode::ctz:
                    if (w32) { enc.rbit32(GPR::X0, GPR::X0); enc.clz32(GPR::X0, GPR::X0); }
                    else { enc.rbit(GPR::X0, GPR::X0); enc.clz(GPR::X0, GPR::X0); }
                    break;
                default: // popcnt: ldr w cleared the high half, so one form counts both widths
                    enc.fmov_from_gpr(FPR::V0, GPR::X0);
                    enc.cnt_8b(FPR::V0, FPR::V0);
                    enc.uaddlv_h(FPR::V0, FPR::V0);
                    enc.fmov_to_gpr32(GPR::X0, FPR::V0);
                    break;
            }
            emitter.store_gpr(inst.result(), GPR::X0);
            return true;
        }

        // Overflow-checked arithmetic: the result is the i32 overflow flag.
        case Opcode::sadd_overflow:
        case Opcode::ssub_overflow:
        case Opcode::smul_overflow:
        case Opcode::uadd_overflow:
        case Opcode::usub_overflow:
        case Opcode::umul_overflow:
            emit_overflow(emitter, inst);
            return true;

        // Integer comparisons
        case Opcode::eq: case Opcode::ne: case Opcode::slt: case Opcode::ult: case Opcode::sle:
        case Opcode::ule: case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge: {
            const bool w32 = bl_is_int32(inst.operand(0)->type());
            emitter.load_gpr(GPR::X0, inst.operand(0));
            emitter.load_gpr(GPR::X1, inst.operand(1));
            if (w32) enc.cmp32(GPR::X0, GPR::X1); else enc.cmp(GPR::X0, GPR::X1);
            enc.cset32(GPR::X0, int_condition(op));
            enc.str32(GPR::X0, emitter.slot_addr(inst.result(), 4));
            return true;
        }

        // Selection: whole slots, so any scalar type.
        case Opcode::select:
            if (inst.type().is_vector()) return false;
            enc.ldr(GPR::X1, emitter.slot_addr(inst.operand(1), 8));
            enc.ldr(GPR::X2, emitter.slot_addr(inst.operand(2), 8));
            emitter.load_gpr(GPR::X0, inst.operand(0));
            if (bl_is_int32(inst.operand(0)->type())) enc.cmp32(GPR::X0, 0u); else enc.cmp(GPR::X0, 0u);
            enc.csel(GPR::X0, GPR::X1, GPR::X2, Condition::NE);
            enc.str(GPR::X0, emitter.slot_addr(inst.result(), 8));
            return true;

        // Memory
        case Opcode::alloca_: {
            auto it = emitter.layout.alloca_offsets.find(&inst);
            if (it == emitter.layout.alloca_offsets.end()) {
                throw_unsupported(kA64BaselineStage, "alloca with no frame buffer");
            }
            enc.sub(GPR::X0, GPR::FP, static_cast<uint32_t>(it->second));
            enc.str(GPR::X0, emitter.slot_addr(inst.result(), 8));
            return true;
        }
        case Opcode::load:
            if (inst.type().is_vector()) return false;
            enc.ldr(GPR::X0, emitter.slot_addr(inst.operand(0), 8));
            emit_mem_load(emitter, inst, inst.offset());
            return true;
        case Opcode::store:
            if (inst.operand(1)->type().is_vector()) return false;
            enc.ldr(GPR::X0, emitter.slot_addr(inst.operand(0), 8));
            emit_mem_store(emitter, inst, inst.offset(), inst.operand(1));
            return true;
        case Opcode::load_indexed:
            if (inst.type().is_vector()) return false;
            indexed_base(emitter, inst);
            emit_mem_load(emitter, inst, inst.offset());
            return true;
        case Opcode::store_indexed:
            if (inst.operand(2)->type().is_vector()) return false;
            indexed_base(emitter, inst);
            emit_mem_store(emitter, inst, inst.offset(), inst.operand(2));
            return true;
        case Opcode::write_barrier: {
            // brass_gc_write_barrier(obj, val), through the symbol table so
            // a host's barrier, when it has one, is the one called, as it is from
            // optimized code.
            void* wb = emitter.resolve_sym("brass_gc_write_barrier");
            if (!wb) throw_unsupported(kA64BaselineStage, "write_barrier: brass_gc_write_barrier is not registered");
            enc.ldr(GPR::X0, emitter.slot_addr(inst.operand(0), 8));
            enc.ldr(GPR::X1, emitter.slot_addr(inst.operand(1), 8));
            emitter.call_abs(wb);
            return true;
        }

        case Opcode::pinned_tls_read:
            enc.str(kPinnedTlsGpr, emitter.slot_addr(inst.result(), 8));
            return true;
        case Opcode::pinned_tls_write:
            enc.ldr(kPinnedTlsGpr, emitter.slot_addr(inst.operand(0), 8));
            return true;
        case Opcode::read_sp:
            enc.mov(GPR::X0, GPR::SP);
            enc.str(GPR::X0, emitter.slot_addr(inst.result(), 8));
            return true;

        default:
            return false;
    }
}

} // namespace brass::aarch64
