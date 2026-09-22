#include "baseline_emit_internal.hpp"
#include <brass/gc/runtime_gc.hpp>
#include <cstring>
#include <cmath>

namespace brass::codegen {

using namespace brass::x64;

bool emit_baseline_x64_op(X64BaselineEmitter& emitter, const Instruction& inst) {
    auto& enc = emitter.enc;
    auto& buffer = emitter.buffer;
    auto slot_addr = [&](const Value* val) { return emitter.slot_addr(val); };
    Opcode op = inst.opcode();

    switch (op) {
        // Constants
        case Opcode::iconst_i32:
        case Opcode::patchable_const_i32: {
            enc.mov32(GPR::RAX, static_cast<uint32_t>(inst.imm_i32()));
            enc.mov32(slot_addr(inst.result()), GPR::RAX);
            return true;
        }
        case Opcode::iconst_i64:
        case Opcode::patchable_const_i64: {
            enc.movabs(GPR::RAX, static_cast<uint64_t>(inst.imm_i64()));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        }
        case Opcode::fconst_f64: {
            double fv = inst.imm_f64();
            uint64_t bits;
            std::memcpy(&bits, &fv, 8);
            enc.movabs(GPR::RAX, bits);
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        }
        case Opcode::func_addr: {
            void* addr = emitter.resolve_sym(inst.symbol());
            enc.movabs(GPR::RAX, reinterpret_cast<uint64_t>(addr));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        }

        // Conversions
        case Opcode::sext_i64: {
            enc.movsxd(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        }
        case Opcode::zext_i64:
            if (inst.operand(0)->type() == Type::i8()) enc.movzx8(GPR::RAX, slot_addr(inst.operand(0)));
            else enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::trunc_i32:
            enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov32(slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::trunc_i8:
            enc.movzx8(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov8(slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::fptosi_i32: {
            enc.movsd(XMM::XMM0, slot_addr(inst.operand(0)));
            enc.cvttsd2si32(GPR::RAX, XMM::XMM0);
            enc.mov32(slot_addr(inst.result()), GPR::RAX);
            return true;
        }
        case Opcode::fptosi_i64: {
            enc.movsd(XMM::XMM0, slot_addr(inst.operand(0)));
            enc.cvttsd2si(GPR::RAX, XMM::XMM0);
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        }
        case Opcode::sitofp_f64_i32: {
            enc.cvtsi2sd32(XMM::XMM0, slot_addr(inst.operand(0)));
            enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            return true;
        }
        case Opcode::sitofp_f64_i64: {
            enc.cvtsi2sd(XMM::XMM0, slot_addr(inst.operand(0)));
            enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            return true;
        }
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64: {
            enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        }

        // Arithmetic & Logic
        case Opcode::add: {
            if (inst.type().is_float()) {
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(0)));
                enc.addsd(XMM::XMM0, slot_addr(inst.operand(1)));
                enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            } else if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.add32(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.add(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::sub: {
            if (inst.type().is_float()) {
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(0)));
                enc.subsd(XMM::XMM0, slot_addr(inst.operand(1)));
                enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            } else if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.sub32(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.sub(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::mul: {
            if (inst.type().is_float()) {
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(0)));
                enc.mulsd(XMM::XMM0, slot_addr(inst.operand(1)));
                enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            } else if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.imul32(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.imul(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::sdiv: {
            if (inst.type().is_float()) {
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(0)));
                enc.divsd(XMM::XMM0, slot_addr(inst.operand(1)));
                enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            } else if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.cdq();
                enc.mov32(GPR::RCX, slot_addr(inst.operand(1)));
                enc.idiv32(GPR::RCX);
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.cqo();
                enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
                enc.idiv(GPR::RCX);
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::smod: {
            if (inst.type().is_float()) {
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(0)));
                enc.movsd(XMM::XMM1, slot_addr(inst.operand(1)));
                void* fmod_ptr = reinterpret_cast<void*>(static_cast<double(*)(double, double)>(&std::fmod));
                enc.movabs(GPR::R11, reinterpret_cast<uint64_t>(fmod_ptr));
                enc.call(GPR::R11);
                enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            } else if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.cdq();
                enc.mov32(GPR::RCX, slot_addr(inst.operand(1)));
                enc.idiv32(GPR::RCX);
                enc.mov32(slot_addr(inst.result()), GPR::RDX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.cqo();
                enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
                enc.idiv(GPR::RCX);
                enc.mov(slot_addr(inst.result()), GPR::RDX);
            }
            return true;
        }
        case Opcode::udiv: {
            if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.xor32(GPR::RDX, GPR::RDX);
                enc.mov32(GPR::RCX, slot_addr(inst.operand(1)));
                enc.div32(GPR::RCX);
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.xor_(GPR::RDX, GPR::RDX);
                enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
                enc.div(GPR::RCX);
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::umod: {
            if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.xor32(GPR::RDX, GPR::RDX);
                enc.mov32(GPR::RCX, slot_addr(inst.operand(1)));
                enc.div32(GPR::RCX);
                enc.mov32(slot_addr(inst.result()), GPR::RDX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.xor_(GPR::RDX, GPR::RDX);
                enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
                enc.div(GPR::RCX);
                enc.mov(slot_addr(inst.result()), GPR::RDX);
            }
            return true;
        }
        case Opcode::neg: {
            if (inst.type().is_float()) {
                enc.movabs(GPR::RAX, 0x8000000000000000ULL);
                enc.movq(XMM::XMM1, GPR::RAX);
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(0)));
                enc.xorpd(XMM::XMM0, XMM::XMM1);
                enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            } else if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.neg32(GPR::RAX);
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.neg(GPR::RAX);
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::and_: {
            if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.and32(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.and_(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::or_: {
            if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.or32(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.or_(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::xor_: {
            if (inst.type().is_float()) {
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(0)));
                enc.xorpd(XMM::XMM0, slot_addr(inst.operand(1)));
                enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            } else if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.xor32(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.xor_(GPR::RAX, slot_addr(inst.operand(1)));
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::not_: {
            if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.not32(GPR::RAX);
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.not_(GPR::RAX);
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::shl: {
            if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov32(GPR::RCX, slot_addr(inst.operand(1)));
                enc.shl32(GPR::RAX);
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
                enc.shl(GPR::RAX);
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::lshr: {
            if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov32(GPR::RCX, slot_addr(inst.operand(1)));
                enc.shr32(GPR::RAX);
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
                enc.shr(GPR::RAX);
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::ashr: {
            if (inst.type().is_i32()) {
                enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov32(GPR::RCX, slot_addr(inst.operand(1)));
                enc.sar32(GPR::RAX);
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
                enc.sar(GPR::RAX);
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::clz: {
            if (inst.type().is_i32()) {
                enc.lzcnt32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.lzcnt(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::ctz: {
            if (inst.type().is_i32()) {
                enc.tzcnt32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.tzcnt(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::popcnt: {
            if (inst.type().is_i32()) {
                enc.popcnt32(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                enc.popcnt(GPR::RAX, slot_addr(inst.operand(0)));
                enc.mov(slot_addr(inst.result()), GPR::RAX);
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
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(0)));
                enc.ucomisd(XMM::XMM0, slot_addr(inst.operand(1)));
                Condition cond = Condition::E;
                switch (op) {
                    case Opcode::eq: cond = Condition::E; break;
                    case Opcode::ne: cond = Condition::NE; break;
                    case Opcode::slt:
                    case Opcode::ult: cond = Condition::B; break;
                    case Opcode::sle:
                    case Opcode::ule: cond = Condition::BE; break;
                    case Opcode::sgt:
                    case Opcode::ugt: cond = Condition::A; break;
                    case Opcode::sge:
                    case Opcode::uge: cond = Condition::AE; break;
                    default: break;
                }
                enc.setcc(cond, GPR::RAX);
                enc.movzx8(GPR::RAX, GPR::RAX);
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            } else {
                bool is_32 = inst.operand(0)->type().is_i32();
                if (is_32) {
                    enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
                    enc.cmp32(GPR::RAX, slot_addr(inst.operand(1)));
                } else {
                    enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
                    enc.cmp(GPR::RAX, slot_addr(inst.operand(1)));
                }
                Condition cond = Condition::E;
                switch (op) {
                    case Opcode::eq: cond = Condition::E; break;
                    case Opcode::ne: cond = Condition::NE; break;
                    case Opcode::slt: cond = Condition::L; break;
                    case Opcode::ult: cond = Condition::B; break;
                    case Opcode::sle: cond = Condition::LE; break;
                    case Opcode::ule: cond = Condition::BE; break;
                    case Opcode::sgt: cond = Condition::G; break;
                    case Opcode::ugt: cond = Condition::A; break;
                    case Opcode::sge: cond = Condition::GE; break;
                    case Opcode::uge: cond = Condition::AE; break;
                    default: break;
                }
                enc.setcc(cond, GPR::RAX);
                enc.movzx8(GPR::RAX, GPR::RAX);
                enc.mov32(slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }

        // Selection
        case Opcode::select: {
            enc.mov32(GPR::RAX, slot_addr(inst.operand(0)));
            enc.test32(GPR::RAX, GPR::RAX);
            if (inst.type().is_float()) {
                Label done_lbl = buffer.create_label();
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(1)));
                enc.jne(done_lbl);
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(2)));
                buffer.bind(done_lbl);
                enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            } else {
                enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
                enc.mov(GPR::RDX, slot_addr(inst.operand(2)));
                enc.cmove(GPR::RCX, GPR::RDX);
                enc.mov(slot_addr(inst.result()), GPR::RCX);
            }
            return true;
        }

        // Memory
        case Opcode::alloca_: {
            auto it = emitter.alloca_offsets.find(&inst);
            int32_t buf_off = (it != emitter.alloca_offsets.end()) ? it->second : 0;
            enc.lea(GPR::RAX, MemAddress::base_disp(GPR::RBP, -buf_off));
            enc.mov(slot_addr(inst.result()), GPR::RAX);
            return true;
        }
        case Opcode::load: {
            enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
            MemAddress mem = MemAddress::base_disp(GPR::RAX, inst.offset());
            if (inst.type().is_float()) {
                if (inst.type().kind() == TypeKind::F32) {
                    enc.movss(XMM::XMM0, mem);
                    enc.movss(slot_addr(inst.result()), XMM::XMM0);
                } else {
                    enc.movsd(XMM::XMM0, mem);
                    enc.movsd(slot_addr(inst.result()), XMM::XMM0);
                }
            } else if (inst.type().is_i32()) {
                enc.mov32(GPR::RDX, mem);
                enc.mov32(slot_addr(inst.result()), GPR::RDX);
            } else {
                enc.mov(GPR::RDX, mem);
                enc.mov(slot_addr(inst.result()), GPR::RDX);
            }
            return true;
        }
        case Opcode::store: {
            enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
            MemAddress mem = MemAddress::base_disp(GPR::RAX, inst.offset());
            if (inst.operand(1)->type().is_float()) {
                if (inst.operand(1)->type().kind() == TypeKind::F32) {
                    enc.movss(XMM::XMM0, slot_addr(inst.operand(1)));
                    enc.movss(mem, XMM::XMM0);
                } else {
                    enc.movsd(XMM::XMM0, slot_addr(inst.operand(1)));
                    enc.movsd(mem, XMM::XMM0);
                }
            } else if (inst.operand(1)->type().is_i32()) {
                enc.mov32(GPR::RDX, slot_addr(inst.operand(1)));
                enc.mov32(mem, GPR::RDX);
            } else {
                enc.mov(GPR::RDX, slot_addr(inst.operand(1)));
                enc.mov(mem, GPR::RDX);
            }
            return true;
        }
        case Opcode::load_indexed: {
            enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
            Scale sc = scale_from_int(inst.scale());
            MemAddress mem = MemAddress::base_index(GPR::RAX, GPR::RCX, sc, inst.offset());
            if (inst.type().is_float()) {
                enc.movsd(XMM::XMM0, mem);
                enc.movsd(slot_addr(inst.result()), XMM::XMM0);
            } else if (inst.type().is_i32()) {
                enc.mov32(GPR::RDX, mem);
                enc.mov32(slot_addr(inst.result()), GPR::RDX);
            } else {
                enc.mov(GPR::RDX, mem);
                enc.mov(slot_addr(inst.result()), GPR::RDX);
            }
            return true;
        }
        case Opcode::store_indexed: {
            enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
            Scale sc = scale_from_int(inst.scale());
            MemAddress mem = MemAddress::base_index(GPR::RAX, GPR::RCX, sc, inst.offset());
            if (inst.operand(2)->type().is_float()) {
                enc.movsd(XMM::XMM0, slot_addr(inst.operand(2)));
                enc.movsd(mem, XMM::XMM0);
            } else if (inst.operand(2)->type().is_i32()) {
                enc.mov32(GPR::RDX, slot_addr(inst.operand(2)));
                enc.mov32(mem, GPR::RDX);
            } else {
                enc.mov(GPR::RDX, slot_addr(inst.operand(2)));
                enc.mov(mem, GPR::RDX);
            }
            return true;
        }
        case Opcode::write_barrier: {
            enc.mov(GPR::RAX, slot_addr(inst.operand(0)));
            enc.mov(GPR::RCX, slot_addr(inst.operand(1)));
            void* wb = reinterpret_cast<void*>(&brass_gc_write_barrier);
            enc.movabs(GPR::R11, reinterpret_cast<uint64_t>(wb));
            enc.call(GPR::R11);
            return true;
        }

        case Opcode::pinned_tls_read: {
            enc.mov(slot_addr(inst.result()), GPR::R13);
            return true;
        }
        case Opcode::pinned_tls_write: {
            enc.mov(GPR::R13, slot_addr(inst.operand(0)));
            return true;
        }
        case Opcode::read_sp: {
            enc.mov(slot_addr(inst.result()), GPR::RSP);
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass::codegen
