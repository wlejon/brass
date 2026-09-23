#include "aarch64_baseline_emit_internal.hpp"
#include <brass/gc/runtime_gc.hpp>
#include <cstring>
#include <cmath>
#include <stdexcept>

namespace brass::aarch64 {

using namespace brass::codegen;

bool emit_baseline_aarch64_op(AArch64BaselineEmitter& emitter, const Instruction& inst) {
    auto& enc = emitter.enc;
    auto& buffer = emitter.buffer;
    auto slot_addr = [&](const Value* val) { return emitter.slot_addr(val); };
    auto ensure_accessible_mem = [&](const MemAddress& mem, GPR scratch = GPR::X16, int size_bytes = 8) {
        return emitter.ensure_accessible_mem(mem, scratch, size_bytes);
    };
    Opcode op = inst.opcode();

    switch (op) {
        // Constants
        case Opcode::iconst_i32:
        case Opcode::patchable_const_i32: {
            enc.mov32(GPR::X0, static_cast<uint32_t>(inst.imm_i32()));
            enc.str32(GPR::X0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::iconst_i64:
        case Opcode::patchable_const_i64: {
            enc.mov(GPR::X0, static_cast<uint64_t>(inst.imm_i64()));
            enc.str(GPR::X0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::fconst_f64: {
            double fv = inst.imm_f64();
            uint64_t bits;
            std::memcpy(&bits, &fv, 8);
            enc.mov(GPR::X0, bits);
            enc.str(GPR::X0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::func_addr: {
            void* addr = emitter.resolve_sym(inst.symbol());
            enc.mov(GPR::X0, reinterpret_cast<uint64_t>(addr));
            enc.str(GPR::X0, slot_addr(inst.result()));
            return true;
        }

        // Conversions
        case Opcode::sext_i64: {
            enc.ldrsw(GPR::X0, slot_addr(inst.operand(0)));
            enc.str(GPR::X0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::zext_i64: {
            if (inst.operand(0)->type() == Type::i8()) {
                enc.ldrb(GPR::X0, slot_addr(inst.operand(0)));
            } else {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
            }
            enc.str(GPR::X0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::trunc_i32: {
            enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
            enc.str32(GPR::X0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::trunc_i8: {
            enc.ldrb(GPR::X0, slot_addr(inst.operand(0)));
            enc.strb(GPR::X0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::fptosi_i32: {
            enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
            enc.fcvtzs_d32(GPR::X0, FPR::V0);
            enc.str32(GPR::X0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::fptosi_i64: {
            enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
            enc.fcvtzs_d(GPR::X0, FPR::V0);
            enc.str(GPR::X0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::sitofp_f64_i32: {
            enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
            enc.scvtf_d32(FPR::V0, GPR::X0);
            enc.str(FPR::V0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::sitofp_f64_i64: {
            enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
            enc.scvtf_d(FPR::V0, GPR::X0);
            enc.str(FPR::V0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64: {
            enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
            enc.str(GPR::X0, slot_addr(inst.result()));
            return true;
        }

        // Arithmetic & Logic
        case Opcode::add: {
            if (inst.type().is_float()) {
                enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                enc.fadd(FPR::V0, FPR::V0, FPR::V1);
                enc.str(FPR::V0, slot_addr(inst.result()));
            } else if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.add32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.add(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::sub: {
            if (inst.type().is_float()) {
                enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                enc.fsub(FPR::V0, FPR::V0, FPR::V1);
                enc.str(FPR::V0, slot_addr(inst.result()));
            } else if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.sub32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.sub(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::mul: {
            if (inst.type().is_float()) {
                enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                enc.fmul(FPR::V0, FPR::V0, FPR::V1);
                enc.str(FPR::V0, slot_addr(inst.result()));
            } else if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.mul32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.mul(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::sdiv: {
            if (inst.type().is_float()) {
                enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                enc.fdiv(FPR::V0, FPR::V0, FPR::V1);
                enc.str(FPR::V0, slot_addr(inst.result()));
            } else if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.brk_if_zero(GPR::X1, false, kBrkIntegerDivideByZero);
                enc.sdiv32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.brk_if_zero(GPR::X1, true, kBrkIntegerDivideByZero);
                enc.sdiv(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::smod: {
            if (inst.type().is_float()) {
                enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                void* fmod_ptr = reinterpret_cast<void*>(static_cast<double(*)(double, double)>(&std::fmod));
                enc.mov(GPR::X16, reinterpret_cast<uint64_t>(fmod_ptr));
                enc.blr(GPR::X16);
                enc.str(FPR::V0, slot_addr(inst.result()));
            } else if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.brk_if_zero(GPR::X1, false, kBrkIntegerDivideByZero);
                enc.sdiv32(GPR::X2, GPR::X0, GPR::X1);
                enc.msub32(GPR::X0, GPR::X2, GPR::X1, GPR::X0);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.brk_if_zero(GPR::X1, true, kBrkIntegerDivideByZero);
                enc.sdiv(GPR::X2, GPR::X0, GPR::X1);
                enc.msub(GPR::X0, GPR::X2, GPR::X1, GPR::X0);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::udiv: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.brk_if_zero(GPR::X1, false, kBrkIntegerDivideByZero);
                enc.udiv32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.brk_if_zero(GPR::X1, true, kBrkIntegerDivideByZero);
                enc.udiv(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::umod: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.brk_if_zero(GPR::X1, false, kBrkIntegerDivideByZero);
                enc.udiv32(GPR::X2, GPR::X0, GPR::X1);
                enc.msub32(GPR::X0, GPR::X2, GPR::X1, GPR::X0);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.brk_if_zero(GPR::X1, true, kBrkIntegerDivideByZero);
                enc.udiv(GPR::X2, GPR::X0, GPR::X1);
                enc.msub(GPR::X0, GPR::X2, GPR::X1, GPR::X0);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::neg: {
            if (inst.type().is_float()) {
                enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                enc.fneg(FPR::V0, FPR::V0);
                enc.str(FPR::V0, slot_addr(inst.result()));
            } else if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.neg32(GPR::X0, GPR::X0);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.neg(GPR::X0, GPR::X0);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::and_: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.and32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.and_(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::or_: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.orr32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.orr(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::xor_: {
            if (inst.type().is_float()) {
                enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                enc.vec_eor(FPR::V0, FPR::V0, FPR::V1);
                enc.str(FPR::V0, slot_addr(inst.result()));
            } else if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.eor32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.eor(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::not_: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.mvn32(GPR::X0, GPR::X0);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.mvn(GPR::X0, GPR::X0);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::shl: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.lsl32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.lsl(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::lshr: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.lsr32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.lsr(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::ashr: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.asr32(GPR::X0, GPR::X0, GPR::X1);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.asr(GPR::X0, GPR::X0, GPR::X1);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::clz: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.clz32(GPR::X0, GPR::X0);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.clz(GPR::X0, GPR::X0);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::ctz: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.rbit32(GPR::X0, GPR::X0);
                enc.clz32(GPR::X0, GPR::X0);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.rbit(GPR::X0, GPR::X0);
                enc.clz(GPR::X0, GPR::X0);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::popcnt: {
            if (inst.type().is_i32()) {
                enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                enc.fmov_from_gpr32(FPR::V0, GPR::X0);
                enc.cnt_8b(FPR::V0, FPR::V0);
                enc.uaddlv_h(FPR::V0, FPR::V0);
                enc.fmov_to_gpr32(GPR::X0, FPR::V0);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                enc.fmov_from_gpr(FPR::V0, GPR::X0);
                enc.cnt_8b(FPR::V0, FPR::V0);
                enc.uaddlv_h(FPR::V0, FPR::V0);
                enc.fmov_to_gpr(GPR::X0, FPR::V0);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }

        // Comparisons
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::slt:
        case Opcode::ult:
        case Opcode::sle:
        case Opcode::ule:
        case Opcode::sgt:
        case Opcode::ugt:
        case Opcode::sge:
        case Opcode::uge: {
            bool is_flt = inst.operand(0)->type().is_float();
            if (is_flt) {
                enc.ldr(FPR::V0, slot_addr(inst.operand(0)));
                enc.ldr(FPR::V1, slot_addr(inst.operand(1)));
                enc.fcmp(FPR::V0, FPR::V1);
                Condition cond = Condition::EQ;
                switch (op) {
                    case Opcode::eq:  cond = Condition::EQ; break;
                    case Opcode::ne:  cond = Condition::NE; break;
                    case Opcode::slt:
                    case Opcode::ult: cond = Condition::MI; break;
                    case Opcode::sle:
                    case Opcode::ule: cond = Condition::LS; break;
                    case Opcode::sgt:
                    case Opcode::ugt: cond = Condition::GT; break;
                    case Opcode::sge:
                    case Opcode::uge: cond = Condition::GE; break;
                    default: break;
                }
                enc.cset(GPR::X0, cond);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                bool is_32 = inst.operand(0)->type().is_i32();
                if (is_32) {
                    enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
                    enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                    enc.cmp32(GPR::X0, GPR::X1);
                } else {
                    enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
                    enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                    enc.cmp(GPR::X0, GPR::X1);
                }
                Condition cond = Condition::EQ;
                switch (op) {
                    case Opcode::eq:  cond = Condition::EQ; break;
                    case Opcode::ne:  cond = Condition::NE; break;
                    case Opcode::slt: cond = Condition::LT; break;
                    case Opcode::ult: cond = Condition::CC; break;
                    case Opcode::sle: cond = Condition::LE; break;
                    case Opcode::ule: cond = Condition::LS; break;
                    case Opcode::sgt: cond = Condition::GT; break;
                    case Opcode::ugt: cond = Condition::HI; break;
                    case Opcode::sge: cond = Condition::GE; break;
                    case Opcode::uge: cond = Condition::CS; break;
                    default: break;
                }
                enc.cset(GPR::X0, cond);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }

        // Selection
        case Opcode::select: {
            enc.ldr32(GPR::X0, slot_addr(inst.operand(0)));
            enc.cmp32(GPR::X0, 0);
            if (inst.type().is_float()) {
                Label done_lbl = buffer.create_label();
                enc.ldr(FPR::V0, slot_addr(inst.operand(1)));
                enc.b(Condition::NE, done_lbl);
                enc.ldr(FPR::V0, slot_addr(inst.operand(2)));
                buffer.bind(done_lbl);
                enc.str(FPR::V0, slot_addr(inst.result()));
            } else if (inst.type().is_i32()) {
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.ldr32(GPR::X2, slot_addr(inst.operand(2)));
                enc.csel32(GPR::X0, GPR::X1, GPR::X2, Condition::NE);
                enc.str32(GPR::X0, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.ldr(GPR::X2, slot_addr(inst.operand(2)));
                enc.csel(GPR::X0, GPR::X1, GPR::X2, Condition::NE);
                enc.str(GPR::X0, slot_addr(inst.result()));
            }
            return true;
        }

        // Memory
        case Opcode::alloca_: {
            auto it = emitter.alloca_offsets.find(&inst);
            int32_t buf_off = (it != emitter.alloca_offsets.end()) ? (16 + it->second) : 16;
            if (buf_off >= 0) {
                if (buf_off <= 4095) {
                    enc.add(GPR::X0, GPR::FP, static_cast<uint32_t>(buf_off));
                } else {
                    enc.mov(GPR::X16, static_cast<uint64_t>(buf_off));
                    enc.add(GPR::X0, GPR::FP, GPR::X16);
                }
            } else {
                if (-buf_off <= 4095) {
                    enc.sub(GPR::X0, GPR::FP, static_cast<uint32_t>(-buf_off));
                } else {
                    enc.mov(GPR::X16, static_cast<uint64_t>(-buf_off));
                    enc.sub(GPR::X0, GPR::FP, GPR::X16);
                }
            }
            enc.str(GPR::X0, slot_addr(inst.result()));
            return true;
        }
        case Opcode::load: {
            enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
            MemAddress mem = ensure_accessible_mem(ptr(GPR::X0, inst.offset()), GPR::X16);
            if (inst.type().is_float()) {
                if (inst.type().kind() == TypeKind::F32) {
                    enc.ldr_s(FPR::V0, mem);
                    enc.str_s(FPR::V0, slot_addr(inst.result()));
                } else {
                    enc.ldr(FPR::V0, mem);
                    enc.str(FPR::V0, slot_addr(inst.result()));
                }
            } else if (inst.type().is_i32()) {
                enc.ldr32(GPR::X1, mem);
                enc.str32(GPR::X1, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X1, mem);
                enc.str(GPR::X1, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::store: {
            enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
            MemAddress mem = ensure_accessible_mem(ptr(GPR::X0, inst.offset()), GPR::X16);
            if (inst.operand(1)->type().is_float()) {
                if (inst.operand(1)->type().kind() == TypeKind::F32) {
                    enc.ldr_s(FPR::V0, slot_addr(inst.operand(1)));
                    enc.str_s(FPR::V0, mem);
                } else {
                    enc.ldr(FPR::V0, slot_addr(inst.operand(1)));
                    enc.str(FPR::V0, mem);
                }
            } else if (inst.operand(1)->type().is_i32()) {
                enc.ldr32(GPR::X1, slot_addr(inst.operand(1)));
                enc.str32(GPR::X1, mem);
            } else {
                enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
                enc.str(GPR::X1, mem);
            }
            return true;
        }
        case Opcode::load_indexed: {
            enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
            enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
            uint8_t shift = 0;
            if (inst.scale() == 8) shift = 3;
            else if (inst.scale() == 4) shift = 2;
            else if (inst.scale() == 2) shift = 1;

            if (inst.offset() != 0) {
                enc.mov(GPR::X16, static_cast<uint64_t>(inst.offset()));
                enc.add(GPR::X0, GPR::X0, GPR::X16);
            }
            MemAddress mem = MemAddress::base_index(GPR::X0, GPR::X1, ExtendType::UXTX, shift);
            if (inst.type().is_float()) {
                enc.ldr(FPR::V0, mem);
                enc.str(FPR::V0, slot_addr(inst.result()));
            } else if (inst.type().is_i32()) {
                enc.ldr32(GPR::X2, mem);
                enc.str32(GPR::X2, slot_addr(inst.result()));
            } else {
                enc.ldr(GPR::X2, mem);
                enc.str(GPR::X2, slot_addr(inst.result()));
            }
            return true;
        }
        case Opcode::store_indexed: {
            enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
            enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
            uint8_t shift = 0;
            if (inst.scale() == 8) shift = 3;
            else if (inst.scale() == 4) shift = 2;
            else if (inst.scale() == 2) shift = 1;

            if (inst.offset() != 0) {
                enc.mov(GPR::X16, static_cast<uint64_t>(inst.offset()));
                enc.add(GPR::X0, GPR::X0, GPR::X16);
            }
            MemAddress mem = MemAddress::base_index(GPR::X0, GPR::X1, ExtendType::UXTX, shift);
            if (inst.operand(2)->type().is_float()) {
                enc.ldr(FPR::V0, slot_addr(inst.operand(2)));
                enc.str(FPR::V0, mem);
            } else if (inst.operand(2)->type().is_i32()) {
                enc.ldr32(GPR::X2, slot_addr(inst.operand(2)));
                enc.str32(GPR::X2, mem);
            } else {
                enc.ldr(GPR::X2, slot_addr(inst.operand(2)));
                enc.str(GPR::X2, mem);
            }
            return true;
        }
        case Opcode::write_barrier: {
            enc.ldr(GPR::X0, slot_addr(inst.operand(0)));
            enc.ldr(GPR::X1, slot_addr(inst.operand(1)));
            // Through the symbol table, so a host's barrier (bronze's) is the
            // one called, as it is from optimized code.
            void* wb = emitter.resolve_sym("brass_gc_write_barrier");
            if (!wb) throw std::runtime_error("baseline JIT: brass_gc_write_barrier is not registered");
            enc.mov(GPR::X16, reinterpret_cast<uint64_t>(wb));
            enc.blr(GPR::X16);
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass::aarch64
