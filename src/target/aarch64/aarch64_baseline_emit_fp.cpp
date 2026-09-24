// AArch64 baseline tier: floating-point arithmetic, comparisons and
// conversions. Rounding, min/max, fma and fmod call the C library, so the
// results are the interpreter's bit for bit (it evaluates them with the same
// functions), as the x64 tier does.
#include "aarch64_baseline_emit_internal.hpp"
#include <cmath>
#include <cstring>

namespace brass::aarch64 {

using namespace brass::codegen;

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

// result = fn(operands...) for a C helper whose arguments and result are all
// floats: V0.. in, V0 out.
void call_fp_helper(AArch64BaselineEmitter& em, const Instruction& inst, const void* f64_fn, const void* f32_fn) {
    static constexpr FPR kArgs[3] = {FPR::V0, FPR::V1, FPR::V2};
    for (size_t i = 0; i < inst.operand_count() && i < 3; ++i) em.load_fp(kArgs[i], inst.operand(i));
    em.call_abs(bl_is_f32(inst.type()) ? f32_fn : f64_fn);
    em.store_fp(inst.result(), FPR::V0);
}

// C comparison semantics: an unordered pair compares false except under ne.
// After fcmp an unordered pair is NZCV = 0011, so each condition below is
// false for it but NE.
void emit_fp_compare(AArch64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const bool f32 = bl_is_f32(inst.operand(0)->type());
    em.load_fp(FPR::V0, inst.operand(0));
    em.load_fp(FPR::V1, inst.operand(1));
    if (f32) enc.fcmp_s(FPR::V0, FPR::V1); else enc.fcmp(FPR::V0, FPR::V1);
    Condition cond = Condition::EQ;
    switch (inst.opcode()) {
        case Opcode::eq: cond = Condition::EQ; break;
        case Opcode::ne: cond = Condition::NE; break;
        case Opcode::slt: case Opcode::ult: cond = Condition::MI; break;
        case Opcode::sle: case Opcode::ule: cond = Condition::LS; break;
        case Opcode::sgt: case Opcode::ugt: cond = Condition::GT; break;
        default: /* sge, uge */ cond = Condition::GE; break;
    }
    enc.cset32(GPR::X0, cond);
    enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
}

void emit_fp_arith(AArch64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const bool f32 = bl_is_f32(inst.type());
    em.load_fp(FPR::V0, inst.operand(0));
    em.load_fp(FPR::V1, inst.operand(1));
    switch (inst.opcode()) {
        case Opcode::add: if (f32) enc.fadd_s(FPR::V0, FPR::V0, FPR::V1); else enc.fadd(FPR::V0, FPR::V0, FPR::V1); break;
        case Opcode::sub: if (f32) enc.fsub_s(FPR::V0, FPR::V0, FPR::V1); else enc.fsub(FPR::V0, FPR::V0, FPR::V1); break;
        case Opcode::mul: if (f32) enc.fmul_s(FPR::V0, FPR::V0, FPR::V1); else enc.fmul(FPR::V0, FPR::V0, FPR::V1); break;
        default: /* sdiv */ if (f32) enc.fdiv_s(FPR::V0, FPR::V0, FPR::V1); else enc.fdiv(FPR::V0, FPR::V0, FPR::V1); break;
    }
    em.store_fp(inst.result(), FPR::V0);
}

// Sign-bit and bitwise operations on a float, in a GPR at its width.
void emit_fp_bits(AArch64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const bool f32 = bl_is_f32(inst.type());
    em.load_gpr(GPR::X0, inst.operand(0));
    switch (inst.opcode()) {
        case Opcode::neg:
            if (f32) enc.eor32_imm(GPR::X0, GPR::X0, 0x80000000u);
            else enc.eor_imm(GPR::X0, GPR::X0, 0x8000000000000000ULL);
            break;
        case Opcode::fabs_f32:
        case Opcode::fabs_f64:
            if (f32) enc.and32_imm(GPR::X0, GPR::X0, 0x7fffffffu);
            else enc.and_imm(GPR::X0, GPR::X0, 0x7fffffffffffffffULL);
            break;
        default: {
            em.load_gpr(GPR::X1, inst.operand(1));
            switch (inst.opcode()) {
                case Opcode::and_: if (f32) enc.and32(GPR::X0, GPR::X0, GPR::X1); else enc.and_(GPR::X0, GPR::X0, GPR::X1); break;
                case Opcode::or_: if (f32) enc.orr32(GPR::X0, GPR::X0, GPR::X1); else enc.orr(GPR::X0, GPR::X0, GPR::X1); break;
                default: /* xor_ */ if (f32) enc.eor32(GPR::X0, GPR::X0, GPR::X1); else enc.eor(GPR::X0, GPR::X0, GPR::X1); break;
            }
            break;
        }
    }
    em.store_gpr(inst.result(), GPR::X0);
}

} // namespace

bool emit_baseline_aarch64_fp_op(AArch64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const Opcode op = inst.opcode();

    if (is_comparison(op)) {
        if (!inst.operand(0)->type().is_float()) return false;
        emit_fp_compare(em, inst);
        return true;
    }

    switch (op) {
        case Opcode::fconst_f64: {
            const double fv = inst.imm_f64();
            if (bl_is_f32(inst.type())) {
                const float f = static_cast<float>(fv);
                uint32_t bits;
                std::memcpy(&bits, &f, 4);
                enc.mov32(GPR::X0, bits);
                enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
            } else {
                uint64_t bits;
                std::memcpy(&bits, &fv, 8);
                enc.mov(GPR::X0, bits);
                enc.str(GPR::X0, em.slot_addr(inst.result(), 8));
            }
            return true;
        }
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64:
            enc.ldr(GPR::X0, em.slot_addr(inst.operand(0), 8));
            enc.str(GPR::X0, em.slot_addr(inst.result(), 8));
            return true;

        // Float -> integer: truncating, as a C cast (which is fcvtzs here).
        case Opcode::fptosi_i32:
            enc.ldr(FPR::V0, em.slot_addr(inst.operand(0), 8));
            enc.fcvtzs_d32(GPR::X0, FPR::V0);
            enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
            return true;
        case Opcode::fptosi_i64:
            enc.ldr(FPR::V0, em.slot_addr(inst.operand(0), 8));
            enc.fcvtzs_d(GPR::X0, FPR::V0);
            enc.str(GPR::X0, em.slot_addr(inst.result(), 8));
            return true;
        case Opcode::fptosi_i32_f32:
            enc.ldr_s(FPR::V0, em.slot_addr(inst.operand(0), 4));
            enc.fcvtzs_s32(GPR::X0, FPR::V0);
            enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
            return true;
        case Opcode::fptosi_i64_f32:
            enc.ldr_s(FPR::V0, em.slot_addr(inst.operand(0), 4));
            enc.fcvtzs_s(GPR::X0, FPR::V0);
            enc.str(GPR::X0, em.slot_addr(inst.result(), 8));
            return true;

        // Integer -> float
        case Opcode::sitofp_f64_i32:
            enc.ldr32(GPR::X0, em.slot_addr(inst.operand(0), 4));
            enc.scvtf_d32(FPR::V0, GPR::X0);
            enc.str(FPR::V0, em.slot_addr(inst.result(), 8));
            return true;
        case Opcode::sitofp_f64_i64:
            enc.ldr(GPR::X0, em.slot_addr(inst.operand(0), 8));
            enc.scvtf_d(FPR::V0, GPR::X0);
            enc.str(FPR::V0, em.slot_addr(inst.result(), 8));
            return true;
        case Opcode::sitofp_f32_i32:
            enc.ldr32(GPR::X0, em.slot_addr(inst.operand(0), 4));
            enc.scvtf_s32(FPR::V0, GPR::X0);
            enc.str_s(FPR::V0, em.slot_addr(inst.result(), 4));
            return true;
        case Opcode::sitofp_f32_i64:
            enc.ldr(GPR::X0, em.slot_addr(inst.operand(0), 8));
            enc.scvtf_s(FPR::V0, GPR::X0);
            enc.str_s(FPR::V0, em.slot_addr(inst.result(), 4));
            return true;

        // Float width
        case Opcode::fptrunc_f32_f64:
            enc.ldr(FPR::V0, em.slot_addr(inst.operand(0), 8));
            enc.fcvt_s_d(FPR::V0, FPR::V0);
            enc.str_s(FPR::V0, em.slot_addr(inst.result(), 4));
            return true;
        case Opcode::fpext_f64_f32:
            enc.ldr_s(FPR::V0, em.slot_addr(inst.operand(0), 4));
            enc.fcvt_d_s(FPR::V0, FPR::V0);
            enc.str(FPR::V0, em.slot_addr(inst.result(), 8));
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
        case Opcode::sqrt_f64:
            em.load_fp(FPR::V0, inst.operand(0));
            if (op == Opcode::sqrt_f32) enc.fsqrt_s(FPR::V0, FPR::V0); else enc.fsqrt(FPR::V0, FPR::V0);
            em.store_fp(inst.result(), FPR::V0);
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

} // namespace brass::aarch64
