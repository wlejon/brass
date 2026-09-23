// x64 baseline tier: floating-point arithmetic, comparisons and conversions.
// Rounding, min/max, fma and fmod call the C library, so the results are the
// interpreter's bit for bit (it evaluates them with the same functions).
#include "baseline_emit_internal.hpp"
#include <cmath>
#include <cstring>

namespace brass::codegen {

using namespace brass::x64;

namespace {

double bl_floor(double x) { return std::floor(x); }
float bl_floorf(float x) { return std::floor(x); }
double bl_ceil(double x) { return std::ceil(x); }
float bl_ceilf(float x) { return std::ceil(x); }
double bl_round(double x) { return std::round(x); }
float bl_roundf(float x) { return std::round(x); }
double bl_fmin(double a, double b) { return std::fmin(a, b); }
float bl_fminf(float a, float b) { return std::fmin(a, b); }
double bl_fmax(double a, double b) { return std::fmax(a, b); }
float bl_fmaxf(float a, float b) { return std::fmax(a, b); }
double bl_fma(double a, double b, double c) { return std::fma(a, b, c); }
float bl_fmaf(float a, float b, float c) { return std::fma(a, b, c); }
double bl_fmod(double a, double b) { return std::fmod(a, b); }
float bl_fmodf(float a, float b) { return std::fmod(a, b); }

void load_fp(X64BaselineEmitter& em, XMM x, const Value* v) {
    if (bl_is_f32(v->type())) em.enc.movss(x, em.slot_addr(v));
    else em.enc.movsd(x, em.slot_addr(v));
}

void store_fp(X64BaselineEmitter& em, const Value* v, XMM x) {
    if (bl_is_f32(v->type())) em.enc.movss(em.slot_addr(v), x);
    else em.enc.movsd(em.slot_addr(v), x);
}

// result = fn(operands...) for a C helper whose arguments and result are all
// floats: both conventions pass them in XMM0.. and return in XMM0.
void call_fp_helper(X64BaselineEmitter& em, const Instruction& inst, const void* f64_fn, const void* f32_fn) {
    static constexpr XMM kArgs[3] = {XMM::XMM0, XMM::XMM1, XMM::XMM2};
    for (size_t i = 0; i < inst.operand_count() && i < 3; ++i) load_fp(em, kArgs[i], inst.operand(i));
    em.call_abs(bl_is_f32(inst.type()) ? f32_fn : f64_fn);
    store_fp(em, inst.result(), XMM::XMM0);
}

// C comparison semantics: an unordered pair compares false except under ne.
void emit_fp_compare(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const Value* a = inst.operand(0);
    const Value* b = inst.operand(1);
    const bool f32 = bl_is_f32(a->type());
    auto ucomi = [&](const Value* lhs, const Value* rhs) {
        load_fp(em, XMM::XMM0, lhs);
        if (f32) enc.ucomiss(XMM::XMM0, em.slot_addr(rhs));
        else enc.ucomisd(XMM::XMM0, em.slot_addr(rhs));
    };
    switch (inst.opcode()) {
        case Opcode::eq:
            ucomi(a, b);
            enc.setcc(Condition::E, GPR::RAX);
            enc.setcc(Condition::NP, GPR::RCX);
            enc.and32(GPR::RAX, GPR::RCX);
            break;
        case Opcode::ne:
            ucomi(a, b);
            enc.setcc(Condition::NE, GPR::RAX);
            enc.setcc(Condition::P, GPR::RCX);
            enc.or32(GPR::RAX, GPR::RCX);
            break;
        case Opcode::slt: case Opcode::ult: ucomi(b, a); enc.setcc(Condition::A, GPR::RAX); break;
        case Opcode::sle: case Opcode::ule: ucomi(b, a); enc.setcc(Condition::AE, GPR::RAX); break;
        case Opcode::sgt: case Opcode::ugt: ucomi(a, b); enc.setcc(Condition::A, GPR::RAX); break;
        default: /* sge, uge */ ucomi(a, b); enc.setcc(Condition::AE, GPR::RAX); break;
    }
    enc.movzx8(GPR::RAX, GPR::RAX);
    enc.mov32(em.slot_addr(inst.result()), GPR::RAX);
}

void emit_fp_arith(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const bool f32 = bl_is_f32(inst.type());
    const MemAddress rhs = em.slot_addr(inst.operand(1));
    load_fp(em, XMM::XMM0, inst.operand(0));
    switch (inst.opcode()) {
        case Opcode::add: if (f32) enc.addss(XMM::XMM0, rhs); else enc.addsd(XMM::XMM0, rhs); break;
        case Opcode::sub: if (f32) enc.subss(XMM::XMM0, rhs); else enc.subsd(XMM::XMM0, rhs); break;
        case Opcode::mul: if (f32) enc.mulss(XMM::XMM0, rhs); else enc.mulsd(XMM::XMM0, rhs); break;
        default: /* sdiv */ if (f32) enc.divss(XMM::XMM0, rhs); else enc.divsd(XMM::XMM0, rhs); break;
    }
    store_fp(em, inst.result(), XMM::XMM0);
}

// Sign-bit and bitwise operations on a float, in a GPR at its width.
void emit_fp_bits(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const bool f32 = bl_is_f32(inst.type());
    em.load_gpr(GPR::RAX, inst.operand(0));
    switch (inst.opcode()) {
        case Opcode::neg:
            if (f32) enc.xor32(GPR::RAX, static_cast<int32_t>(0x80000000u));
            else { enc.movabs(GPR::RCX, 0x8000000000000000ULL); enc.xor_(GPR::RAX, GPR::RCX); }
            break;
        case Opcode::fabs_f32:
        case Opcode::fabs_f64:
            if (f32) enc.and32(GPR::RAX, 0x7fffffff);
            else { enc.shl(GPR::RAX, 1); enc.shr(GPR::RAX, 1); }
            break;
        case Opcode::and_:
            if (f32) enc.and32(GPR::RAX, em.slot_addr(inst.operand(1)));
            else enc.and_(GPR::RAX, em.slot_addr(inst.operand(1)));
            break;
        case Opcode::or_:
            if (f32) enc.or32(GPR::RAX, em.slot_addr(inst.operand(1)));
            else enc.or_(GPR::RAX, em.slot_addr(inst.operand(1)));
            break;
        default: /* xor_ */
            if (f32) enc.xor32(GPR::RAX, em.slot_addr(inst.operand(1)));
            else enc.xor_(GPR::RAX, em.slot_addr(inst.operand(1)));
            break;
    }
    em.store_gpr(inst.result(), GPR::RAX);
}

} // namespace

bool emit_baseline_x64_fp_op(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const Opcode op = inst.opcode();

    if (is_comparison(op)) {
        if (!inst.operand(0)->type().is_float()) return false;
        emit_fp_compare(em, inst);
        return true;
    }

    switch (op) {
        case Opcode::fconst_f64: {
            double fv = inst.imm_f64();
            if (bl_is_f32(inst.type())) {
                float f = static_cast<float>(fv);
                uint32_t bits;
                std::memcpy(&bits, &f, 4);
                enc.mov32(GPR::RAX, bits);
                enc.mov32(em.slot_addr(inst.result()), GPR::RAX);
            } else {
                uint64_t bits;
                std::memcpy(&bits, &fv, 8);
                enc.movabs(GPR::RAX, bits);
                enc.mov(em.slot_addr(inst.result()), GPR::RAX);
            }
            return true;
        }
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64:
            enc.mov(GPR::RAX, em.slot_addr(inst.operand(0)));
            enc.mov(em.slot_addr(inst.result()), GPR::RAX);
            return true;

        // Float -> integer: truncating, as a C cast.
        case Opcode::fptosi_i32:
            enc.cvttsd2si32(GPR::RAX, em.slot_addr(inst.operand(0)));
            enc.mov32(em.slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::fptosi_i64:
            enc.cvttsd2si(GPR::RAX, em.slot_addr(inst.operand(0)));
            enc.mov(em.slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::fptosi_i32_f32:
            enc.cvttss2si32(GPR::RAX, em.slot_addr(inst.operand(0)));
            enc.mov32(em.slot_addr(inst.result()), GPR::RAX);
            return true;
        case Opcode::fptosi_i64_f32:
            enc.cvttss2si(GPR::RAX, em.slot_addr(inst.operand(0)));
            enc.mov(em.slot_addr(inst.result()), GPR::RAX);
            return true;

        // Integer -> float
        case Opcode::sitofp_f64_i32:
            enc.xorpd(XMM::XMM0, XMM::XMM0);
            enc.cvtsi2sd32(XMM::XMM0, em.slot_addr(inst.operand(0)));
            enc.movsd(em.slot_addr(inst.result()), XMM::XMM0);
            return true;
        case Opcode::sitofp_f64_i64:
            enc.xorpd(XMM::XMM0, XMM::XMM0);
            enc.cvtsi2sd(XMM::XMM0, em.slot_addr(inst.operand(0)));
            enc.movsd(em.slot_addr(inst.result()), XMM::XMM0);
            return true;
        case Opcode::sitofp_f32_i32:
            enc.xorps(XMM::XMM0, XMM::XMM0);
            enc.cvtsi2ss32(XMM::XMM0, em.slot_addr(inst.operand(0)));
            enc.movss(em.slot_addr(inst.result()), XMM::XMM0);
            return true;
        case Opcode::sitofp_f32_i64:
            enc.xorps(XMM::XMM0, XMM::XMM0);
            enc.cvtsi2ss(XMM::XMM0, em.slot_addr(inst.operand(0)));
            enc.movss(em.slot_addr(inst.result()), XMM::XMM0);
            return true;

        // Float width
        case Opcode::fptrunc_f32_f64:
            enc.cvtsd2ss(XMM::XMM0, em.slot_addr(inst.operand(0)));
            enc.movss(em.slot_addr(inst.result()), XMM::XMM0);
            return true;
        case Opcode::fpext_f64_f32:
            enc.cvtss2sd(XMM::XMM0, em.slot_addr(inst.operand(0)));
            enc.movsd(em.slot_addr(inst.result()), XMM::XMM0);
            return true;

        // Arithmetic
        case Opcode::add:
        case Opcode::sub:
        case Opcode::mul:
        case Opcode::sdiv:
            if (!inst.type().is_float()) return false;
            emit_fp_arith(em, inst);
            return true;
        case Opcode::smod:
            if (!inst.type().is_float()) return false;
            call_fp_helper(em, inst, reinterpret_cast<const void*>(&bl_fmod), reinterpret_cast<const void*>(&bl_fmodf));
            return true;
        case Opcode::neg:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
            if (!inst.type().is_float()) return false;
            emit_fp_bits(em, inst);
            return true;
        case Opcode::fabs_f32:
        case Opcode::fabs_f64:
            emit_fp_bits(em, inst);
            return true;
        case Opcode::sqrt_f32:
            enc.sqrtss(XMM::XMM0, em.slot_addr(inst.operand(0)));
            enc.movss(em.slot_addr(inst.result()), XMM::XMM0);
            return true;
        case Opcode::sqrt_f64:
            enc.sqrtsd(XMM::XMM0, em.slot_addr(inst.operand(0)));
            enc.movsd(em.slot_addr(inst.result()), XMM::XMM0);
            return true;
        case Opcode::floor_f32:
        case Opcode::floor_f64:
            call_fp_helper(em, inst, reinterpret_cast<const void*>(&bl_floor), reinterpret_cast<const void*>(&bl_floorf));
            return true;
        case Opcode::ceil_f32:
        case Opcode::ceil_f64:
            call_fp_helper(em, inst, reinterpret_cast<const void*>(&bl_ceil), reinterpret_cast<const void*>(&bl_ceilf));
            return true;
        case Opcode::round_f32:
        case Opcode::round_f64:
            call_fp_helper(em, inst, reinterpret_cast<const void*>(&bl_round), reinterpret_cast<const void*>(&bl_roundf));
            return true;
        case Opcode::fmin_f32:
        case Opcode::fmin_f64:
            call_fp_helper(em, inst, reinterpret_cast<const void*>(&bl_fmin), reinterpret_cast<const void*>(&bl_fminf));
            return true;
        case Opcode::fmax_f32:
        case Opcode::fmax_f64:
            call_fp_helper(em, inst, reinterpret_cast<const void*>(&bl_fmax), reinterpret_cast<const void*>(&bl_fmaxf));
            return true;
        case Opcode::fma_f32:
        case Opcode::fma_f64:
            call_fp_helper(em, inst, reinterpret_cast<const void*>(&bl_fma), reinterpret_cast<const void*>(&bl_fmaf));
            return true;
        default:
            return false;
    }
}

} // namespace brass::codegen
