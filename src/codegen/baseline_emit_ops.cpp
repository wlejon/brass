// x64 baseline tier: integer, pointer and memory opcodes. Floating-point
// arithmetic, comparisons of floats and float conversions are in
// baseline_emit_fp.cpp. Slot conventions: baseline_emit_internal.hpp.
#include "baseline_emit_internal.hpp"
#include <algorithm>
#include <cstring>

namespace brass::codegen {

using namespace brass::x64;

namespace {

bool is_float_op(const Instruction& inst) {
    if (inst.produces_value() && inst.type().is_float()) return true;
    return is_comparison(inst.opcode()) && inst.operand(0)->type().is_float();
}

Condition int_condition(Opcode op) {
    switch (op) {
        case Opcode::eq: return Condition::E;
        case Opcode::ne: return Condition::NE;
        case Opcode::slt: return Condition::L;
        case Opcode::ult: return Condition::B;
        case Opcode::sle: return Condition::LE;
        case Opcode::ule: return Condition::BE;
        case Opcode::sgt: return Condition::G;
        case Opcode::ugt: return Condition::A;
        case Opcode::sge: return Condition::GE;
        default: return Condition::AE; // uge
    }
}

// The effective type of a memory access: the instruction's memory type when
// set, else the loaded / stored value's.
Type access_type(const Instruction& inst, Type value_type) {
    Type mt = inst.memory_type();
    return mt.is_void() ? value_type : mt;
}

void emit_mem_load(X64BaselineEmitter& em, const Instruction& inst, const MemAddress& mem) {
    auto& enc = em.enc;
    Type t = access_type(inst, inst.type());
    const MemAddress dst = em.slot_addr(inst.result());
    if (t.is_float()) {
        if (bl_is_f32(t)) { enc.movss(XMM::XMM0, mem); enc.movss(dst, XMM::XMM0); }
        else { enc.movsd(XMM::XMM0, mem); enc.movsd(dst, XMM::XMM0); }
        return;
    }
    switch (t.size_in_bytes()) {
        case 1: enc.movzx8(GPR::RDX, mem); break;
        case 2: enc.movzx16(GPR::RDX, mem); break;
        case 4: enc.mov32(GPR::RDX, mem); break;
        default: enc.mov(GPR::RDX, mem); break;
    }
    em.store_gpr(inst.result(), GPR::RDX);
}

void emit_mem_store(X64BaselineEmitter& em, const Instruction& inst, const MemAddress& mem, const Value* val) {
    auto& enc = em.enc;
    Type t = access_type(inst, val->type());
    const MemAddress src = em.slot_addr(val);
    if (t.is_float()) {
        if (bl_is_f32(t)) { enc.movss(XMM::XMM0, src); enc.movss(mem, XMM::XMM0); }
        else { enc.movsd(XMM::XMM0, src); enc.movsd(mem, XMM::XMM0); }
        return;
    }
    enc.mov(GPR::RDX, src);
    switch (t.size_in_bytes()) {
        case 1: enc.mov8(mem, GPR::RDX); break;
        case 2: enc.mov16(mem, GPR::RDX); break;
        case 4: enc.mov32(mem, GPR::RDX); break;
        default: enc.mov(mem, GPR::RDX); break;
    }
}

// Base in RAX, sign-extended index in RCX.
MemAddress indexed_address(X64BaselineEmitter& em, const Instruction& inst) {
    em.enc.mov(GPR::RAX, em.slot_addr(inst.operand(0)));
    const Value* idx = inst.operand(1);
    if (bl_is_int32(idx->type())) em.enc.movsxd(GPR::RCX, em.slot_addr(idx));
    else em.enc.mov(GPR::RCX, em.slot_addr(idx));
    return MemAddress::base_index(GPR::RAX, GPR::RCX, scale_from_int(inst.scale()), inst.offset());
}

void emit_overflow(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const bool w32 = bl_is_int32(inst.operand(0)->type());
    const MemAddress rhs = em.slot_addr(inst.operand(1));
    em.load_gpr(GPR::RAX, inst.operand(0));
    Condition cond = Condition::O;
    switch (inst.opcode()) {
        case Opcode::sadd_overflow: if (w32) enc.add32(GPR::RAX, rhs); else enc.add(GPR::RAX, rhs); break;
        case Opcode::ssub_overflow: if (w32) enc.sub32(GPR::RAX, rhs); else enc.sub(GPR::RAX, rhs); break;
        case Opcode::smul_overflow: if (w32) enc.imul32(GPR::RAX, rhs); else enc.imul(GPR::RAX, rhs); break;
        case Opcode::uadd_overflow:
            if (w32) enc.add32(GPR::RAX, rhs); else enc.add(GPR::RAX, rhs);
            cond = Condition::B;
            break;
        case Opcode::usub_overflow:
            if (w32) enc.sub32(GPR::RAX, rhs); else enc.sub(GPR::RAX, rhs);
            cond = Condition::B;
            break;
        default: // umul_overflow: CF = OF = (high half != 0); clobbers RDX
            if (w32) enc.mul32(rhs); else enc.mul(rhs);
            cond = Condition::B;
            break;
    }
    enc.setcc(cond, GPR::RAX);
    enc.movzx8(GPR::RAX, GPR::RAX);
    enc.mov32(em.slot_addr(inst.result()), GPR::RAX);
}

} // namespace

bool emit_baseline_x64_op(X64BaselineEmitter& emitter, const Instruction& inst) {
    auto& enc = emitter.enc;
    auto slot_addr = [&](const Value* val) { return emitter.slot_addr(val); };
    Opcode op = inst.opcode();

    if (is_float_op(inst) && op != Opcode::select && op != Opcode::load && op != Opcode::load_indexed &&
        op != Opcode::call && op != Opcode::call_indirect && op != Opcode::patchable_call) {
        return false; // baseline_emit_fp.cpp
    }

    // Integer binary op at the result's width: RAX = op0 <op> op1.
    auto binop = [&](auto op32, auto op64) {
        const bool w32 = bl_is_int32(inst.type());
        emitter.load_gpr(GPR::RAX, inst.operand(0));
        if (w32) (enc.*op32)(GPR::RAX, slot_addr(inst.operand(1)));
        else (enc.*op64)(GPR::RAX, slot_addr(inst.operand(1)));
        emitter.store_gpr(inst.result(), GPR::RAX);
    };
    using MemOp = void (X64Encoder::*)(GPR, const MemAddress&);
    using RegOp = void (X64Encoder::*)(GPR);

    switch (op) {
        // Constants
        case Opcode::iconst_i32:
        case Opcode::patchable_const_i32:
            enc.mov32(GPR::RAX, static_cast<uint32_t>(inst.imm_i32()));
            enc.mov32(slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::iconst_i64:
        case Opcode::patchable_const_i64:
            enc.movabs(GPR::RAX, static_cast<uint64_t>(inst.imm_i64()));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
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
            if (!data && emitter.function_address && (addr = emitter.function_address(inst.symbol()))) {
                // A program function: its canonical stub, the pointer every
                // tier yields. It compiles the function on first call, or
                // bridges into Tier 0 when the baseline tier rejects it.
                emitter.uses_lazy_stubs = true;
                auto& syms = emitter.lazy_addr_symbols;
                if (std::find(syms.begin(), syms.end(), inst.symbol()) == syms.end()) {
                    syms.emplace_back(inst.symbol());
                }
            } else if (data) {
                addr = emitter.resolve_sym(inst.symbol());
                if (!addr) {
                    throw_unsupported(kX64BaselineStage, "func_addr of data symbol " + std::string(inst.symbol()) +
                                                             ", which does not resolve, in " +
                                                             std::string(emitter.fn.name()));
                }
            } else if (!(addr = emitter.resolve_sym(inst.symbol()))) {
                // A module function's stub compiles it on first call
                // (MultiTierPipeline::compile_tier1_on_demand); a host
                // symbol must be registered by then.
                addr = emitter.resolve_or_stub(inst.symbol());
                auto& syms = emitter.lazy_addr_symbols;
                if (std::find(syms.begin(), syms.end(), inst.symbol()) == syms.end()) {
                    syms.emplace_back(inst.symbol());
                }
            }
            enc.movabs(GPR::RAX, reinterpret_cast<uint64_t>(addr));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        }

        // Integer conversions (as the interpreter: narrow values are i32s)
        case Opcode::sext_i64:
            enc.movsxd(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::zext_i64:
            if (inst.operand(0)->type() == Type::i8()) enc.movzx8(GPR::RAX, slot_addr(inst.operand(0)));
            else enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::trunc_i32:
            enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
            emitter.store_gpr(inst.result(), GPR::RAX);
            return true;
        case Opcode::trunc_i8:
            // The whole 32-bit slot, so a later 32-bit read sees 0..255.
            enc.movzx8(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov32(slot_addr(inst.result()), GPR::RAX);
            return true;

        // Integer arithmetic & logic
        case Opcode::add: binop(static_cast<MemOp>(&X64Encoder::add32), static_cast<MemOp>(&X64Encoder::add)); return true;
        case Opcode::sub: binop(static_cast<MemOp>(&X64Encoder::sub32), static_cast<MemOp>(&X64Encoder::sub)); return true;
        case Opcode::mul: binop(static_cast<MemOp>(&X64Encoder::imul32), static_cast<MemOp>(&X64Encoder::imul)); return true;
        case Opcode::and_: binop(static_cast<MemOp>(&X64Encoder::and32), static_cast<MemOp>(&X64Encoder::and_)); return true;
        case Opcode::or_: binop(static_cast<MemOp>(&X64Encoder::or32), static_cast<MemOp>(&X64Encoder::or_)); return true;
        case Opcode::xor_: binop(static_cast<MemOp>(&X64Encoder::xor32), static_cast<MemOp>(&X64Encoder::xor_)); return true;
        case Opcode::sdiv:
        case Opcode::smod: {
            const bool w32 = bl_is_int32(inst.type());
            emitter.load_gpr(GPR::RAX, inst.operand(0));
            emitter.load_gpr(GPR::RCX, inst.operand(1));
            if (w32) { enc.cdq(); enc.idiv32_wrapping(GPR::RCX); }
            else { enc.cqo(); enc.idiv_wrapping(GPR::RCX); }
            emitter.store_gpr(inst.result(), op == Opcode::sdiv ? GPR::RAX : GPR::RDX);
            return true;
        }
        case Opcode::udiv:
        case Opcode::umod: {
            const bool w32 = bl_is_int32(inst.type());
            emitter.load_gpr(GPR::RAX, inst.operand(0));
            emitter.load_gpr(GPR::RCX, inst.operand(1));
            enc.xor32(GPR::RDX, GPR::RDX);
            if (w32) enc.div32(GPR::RCX); else enc.div(GPR::RCX);
            emitter.store_gpr(inst.result(), op == Opcode::udiv ? GPR::RAX : GPR::RDX);
            return true;
        }
        case Opcode::neg:
        case Opcode::not_: {
            const bool w32 = bl_is_int32(inst.type());
            emitter.load_gpr(GPR::RAX, inst.operand(0));
            if (op == Opcode::neg) { if (w32) enc.neg32(GPR::RAX); else enc.neg(GPR::RAX); }
            else { if (w32) enc.not32(GPR::RAX); else enc.not_(GPR::RAX); }
            emitter.store_gpr(inst.result(), GPR::RAX);
            return true;
        }
        case Opcode::shl:
        case Opcode::lshr:
        case Opcode::ashr: {
            // The count is masked to the width, as the interpreter does.
            const bool w32 = bl_is_int32(inst.type());
            emitter.load_gpr(GPR::RAX, inst.operand(0));
            emitter.load_gpr(GPR::RCX, inst.operand(1));
            RegOp f = nullptr;
            if (op == Opcode::shl) f = w32 ? static_cast<RegOp>(&X64Encoder::shl32) : static_cast<RegOp>(&X64Encoder::shl);
            else if (op == Opcode::lshr) f = w32 ? static_cast<RegOp>(&X64Encoder::shr32) : static_cast<RegOp>(&X64Encoder::shr);
            else f = w32 ? static_cast<RegOp>(&X64Encoder::sar32) : static_cast<RegOp>(&X64Encoder::sar);
            (enc.*f)(GPR::RAX);
            emitter.store_gpr(inst.result(), GPR::RAX);
            return true;
        }
        case Opcode::clz:
        case Opcode::ctz:
        case Opcode::popcnt: {
            const bool w32 = bl_is_int32(inst.type());
            const MemAddress src = slot_addr(inst.operand(0));
            if (op == Opcode::clz) { if (w32) enc.lzcnt32(GPR::RAX, src); else enc.lzcnt(GPR::RAX, src); }
            else if (op == Opcode::ctz) { if (w32) enc.tzcnt32(GPR::RAX, src); else enc.tzcnt(GPR::RAX, src); }
            else { if (w32) enc.popcnt32(GPR::RAX, src); else enc.popcnt(GPR::RAX, src); }
            emitter.store_gpr(inst.result(), GPR::RAX);
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
            emitter.load_gpr(GPR::RAX, inst.operand(0));
            if (w32) enc.cmp32(GPR::RAX, slot_addr(inst.operand(1)));
            else enc.cmp(GPR::RAX, slot_addr(inst.operand(1)));
            enc.setcc(int_condition(op), GPR::RAX);
            enc.movzx8(GPR::RAX, GPR::RAX);
            enc.mov32(slot_addr(inst.result()), GPR::RAX);
            return true;
        }

        // Selection: whole slots, so any scalar type.
        case Opcode::select:
            enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
            enc.mov(GPR::RDX, slot_addr(inst.operand(2)));
            emitter.test_cond(inst.operand(0));
            enc.cmove(GPR::RCX, GPR::RDX);
            enc.mov(slot_addr(inst.result()), GPR::RCX);
            return true;

        // Memory
        case Opcode::alloca_: {
            auto it = emitter.alloca_offsets.find(&inst);
            if (it == emitter.alloca_offsets.end()) {
                throw_unsupported(kX64BaselineStage, "alloca with no frame buffer");
            }
            enc.lea(GPR::RAX, MemAddress::base_disp(GPR::RBP, -it->second));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        }
        case Opcode::load:
            enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
            emit_mem_load(emitter, inst, MemAddress::base_disp(GPR::RAX, inst.offset()));
            return true;
        case Opcode::store:
            enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
            emit_mem_store(emitter, inst, MemAddress::base_disp(GPR::RAX, inst.offset()), inst.operand(1));
            return true;
        case Opcode::load_indexed:
            emit_mem_load(emitter, inst, indexed_address(emitter, inst));
            return true;
        case Opcode::store_indexed:
            emit_mem_store(emitter, inst, indexed_address(emitter, inst), inst.operand(2));
            return true;
        case Opcode::write_barrier: {
            // brass_gc_write_barrier(obj, val) in the C convention, through
            // the symbol table so a host's barrier (bronze's) is the one
            // called, as it is from optimized code.
            void* wb = emitter.resolve_sym("brass_gc_write_barrier");
            if (!wb) throw_unsupported(kX64BaselineStage, "write_barrier: brass_gc_write_barrier is not registered");
            const bool win = emitter.target.is_windows();
            enc.mov(win ? GPR::RCX : GPR::RDI, slot_addr(inst.operand(0)));
            enc.mov(win ? GPR::RDX : GPR::RSI, slot_addr(inst.operand(1)));
            emitter.call_abs(wb);
            return true;
        }

        case Opcode::pinned_tls_read:
            enc.mov(slot_addr(inst.result()), GPR::R13);
            return true;
        case Opcode::pinned_tls_write:
            enc.mov(GPR::R13, slot_addr(inst.operand(0)));
            return true;
        case Opcode::read_sp:
            enc.mov(slot_addr(inst.result()), GPR::RSP);
            return true;

        default:
            return false;
    }
}

} // namespace brass::codegen
