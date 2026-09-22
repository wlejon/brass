#pragma once

#include <brass/target/aarch64/aarch64_registers.hpp>
#include <brass/target/aarch64/aarch64_operands.hpp>
#include <brass/target/aarch64/code_buffer.hpp>
#include <brass/target/aarch64/aarch64_logical_imm.hpp>
#include <cstdint>
#include <cstddef>
#include <string>

namespace brass::aarch64 {

// brk immediates for the program errors the backend detects itself.
inline constexpr uint16_t kBrkIntegerDivideByZero = 0x0D0;

class AArch64Encoder {
public:
    explicit AArch64Encoder(CodeBuffer& buffer) noexcept : buffer_(buffer) {}

    CodeBuffer& buffer() noexcept { return buffer_; }
    const CodeBuffer& buffer() const noexcept { return buffer_; }

    // =========================================================================
    // ALU: ADD, ADDS, SUB, SUBS, CMP, CMN, NEG, NEGS
    // =========================================================================

    // ADD (64-bit & 32-bit)
    void add(GPR dst, GPR src1, GPR src2);
    void add32(GPR dst, GPR src1, GPR src2);
    void add(GPR dst, GPR src, uint32_t imm);
    void add32(GPR dst, GPR src, uint32_t imm);

    // ADDS (with flags)
    void adds(GPR dst, GPR src1, GPR src2);
    void adds32(GPR dst, GPR src1, GPR src2);
    void adds(GPR dst, GPR src, uint32_t imm);
    void adds32(GPR dst, GPR src, uint32_t imm);

    // SUB (64-bit & 32-bit)
    void sub(GPR dst, GPR src1, GPR src2);
    void sub32(GPR dst, GPR src1, GPR src2);
    void sub(GPR dst, GPR src, uint32_t imm);
    void sub32(GPR dst, GPR src, uint32_t imm);

    // SUBS (with flags)
    void subs(GPR dst, GPR src1, GPR src2);
    void subs32(GPR dst, GPR src1, GPR src2);
    void subs(GPR dst, GPR src, uint32_t imm);
    void subs32(GPR dst, GPR src, uint32_t imm);

    // CMP (alias for SUBS XZR/WZR, src1, src2/imm)
    void cmp(GPR src1, GPR src2);
    void cmp32(GPR src1, GPR src2);
    void cmp(GPR src, uint32_t imm);
    void cmp32(GPR src, uint32_t imm);

    // CMN (alias for ADDS XZR/WZR, src1, src2/imm)
    void cmn(GPR src1, GPR src2);
    void cmn32(GPR src1, GPR src2);
    void cmn(GPR src, uint32_t imm);
    void cmn32(GPR src, uint32_t imm);

    // NEG (alias for SUB dst, XZR, src)
    void neg(GPR dst, GPR src);
    void neg32(GPR dst, GPR src);
    void negs(GPR dst, GPR src);
    void negs32(GPR dst, GPR src);

    // =========================================================================
    // Logical: AND, BIC, ORR, ORN, EOR, EON, TST, MVN
    // =========================================================================

    void and_(GPR dst, GPR src1, GPR src2);
    void and32(GPR dst, GPR src1, GPR src2);
    void bic(GPR dst, GPR src1, GPR src2);
    void bic32(GPR dst, GPR src1, GPR src2);
    void orr(GPR dst, GPR src1, GPR src2);
    void orr32(GPR dst, GPR src1, GPR src2);
    void orn(GPR dst, GPR src1, GPR src2);
    void orn32(GPR dst, GPR src1, GPR src2);
    void eor(GPR dst, GPR src1, GPR src2);
    void eor32(GPR dst, GPR src1, GPR src2);
    void eon(GPR dst, GPR src1, GPR src2);
    void eon32(GPR dst, GPR src1, GPR src2);

    // TST (alias for ANDS XZR/WZR, src1, src2)
    void tst(GPR src1, GPR src2);
    void tst32(GPR src1, GPR src2);

    // MVN (alias for ORN dst, XZR, src)
    void mvn(GPR dst, GPR src);
    void mvn32(GPR dst, GPR src);

    // Logical Immediates (ARMv8 Bitmask)
    static bool encode_logical_immediate(uint64_t val, bool is_64bit, uint32_t& n, uint32_t& immr, uint32_t& imms) noexcept {
        return ::brass::aarch64::encode_logical_immediate(val, is_64bit, n, immr, imms);
    }

    void and_imm(GPR dst, GPR src, uint64_t imm);
    void and32_imm(GPR dst, GPR src, uint32_t imm);
    void orr_imm(GPR dst, GPR src, uint64_t imm);
    void orr32_imm(GPR dst, GPR src, uint32_t imm);
    void eor_imm(GPR dst, GPR src, uint64_t imm);
    void eor32_imm(GPR dst, GPR src, uint32_t imm);
    void ands_imm(GPR dst, GPR src, uint64_t imm);
    void ands32_imm(GPR dst, GPR src, uint32_t imm);
    void tst_imm(GPR src, uint64_t imm);
    void tst32_imm(GPR src, uint32_t imm);

    // =========================================================================
    // Multiply & Divide
    // =========================================================================

    void mul(GPR dst, GPR src1, GPR src2);
    void mul32(GPR dst, GPR src1, GPR src2);
    void madd(GPR dst, GPR src1, GPR src2, GPR acc);
    void madd32(GPR dst, GPR src1, GPR src2, GPR acc);
    void msub(GPR dst, GPR src1, GPR src2, GPR acc);
    void msub32(GPR dst, GPR src1, GPR src2, GPR acc);
    void smulh(GPR dst, GPR src1, GPR src2);
    void umulh(GPR dst, GPR src1, GPR src2);
    void sdiv(GPR dst, GPR src1, GPR src2);
    void sdiv32(GPR dst, GPR src1, GPR src2);
    void udiv(GPR dst, GPR src1, GPR src2);
    void udiv32(GPR dst, GPR src1, GPR src2);

    // =========================================================================
    // Shifts and Bit Manipulation
    // =========================================================================

    void lsl(GPR dst, GPR src1, GPR src2);
    void lsl32(GPR dst, GPR src1, GPR src2);
    void lsl(GPR dst, GPR src, uint8_t shift);
    void lsl32(GPR dst, GPR src, uint8_t shift);

    void lsr(GPR dst, GPR src1, GPR src2);
    void lsr32(GPR dst, GPR src1, GPR src2);
    void lsr(GPR dst, GPR src, uint8_t shift);
    void lsr32(GPR dst, GPR src, uint8_t shift);

    void asr(GPR dst, GPR src1, GPR src2);
    void asr32(GPR dst, GPR src1, GPR src2);
    void asr(GPR dst, GPR src, uint8_t shift);
    void asr32(GPR dst, GPR src, uint8_t shift);

    void ror(GPR dst, GPR src1, GPR src2);
    void ror32(GPR dst, GPR src1, GPR src2);

    void clz(GPR dst, GPR src);
    void clz32(GPR dst, GPR src);
    void rbit(GPR dst, GPR src);
    void rbit32(GPR dst, GPR src);
    void rev(GPR dst, GPR src);
    void rev32(GPR dst, GPR src);
    void rev16(GPR dst, GPR src);

    // Sign/zero extend
    void sxtb(GPR dst, GPR src);
    void sxth(GPR dst, GPR src);
    void sxtw(GPR dst, GPR src);
    void uxtb(GPR dst, GPR src);
    void uxth(GPR dst, GPR src);

    // =========================================================================
    // Move & Constant Loading
    // =========================================================================

    void mov(GPR dst, GPR src);
    void mov32(GPR dst, GPR src);

    void movz(GPR dst, uint16_t imm, uint8_t hw_shift = 0);
    void movz32(GPR dst, uint16_t imm, uint8_t hw_shift = 0);
    void movk(GPR dst, uint16_t imm, uint8_t hw_shift = 0);
    void movk32(GPR dst, uint16_t imm, uint8_t hw_shift = 0);
    void movn(GPR dst, uint16_t imm, uint8_t hw_shift = 0);
    void movn32(GPR dst, uint16_t imm, uint8_t hw_shift = 0);

    // Full 64-bit/32-bit constant materialization
    void mov(GPR dst, uint64_t imm);
    void mov32(GPR dst, uint32_t imm);

    // =========================================================================
    // Conditional Selection & Set
    // =========================================================================

    void csel(GPR dst, GPR src1, GPR src2, Condition cond);
    void csel32(GPR dst, GPR src1, GPR src2, Condition cond);
    void csinc(GPR dst, GPR src1, GPR src2, Condition cond);
    void csinc32(GPR dst, GPR src1, GPR src2, Condition cond);
    void csinv(GPR dst, GPR src1, GPR src2, Condition cond);
    void csinv32(GPR dst, GPR src1, GPR src2, Condition cond);
    void csneg(GPR dst, GPR src1, GPR src2, Condition cond);
    void csneg32(GPR dst, GPR src1, GPR src2, Condition cond);

    void cset(GPR dst, Condition cond);
    void cset32(GPR dst, Condition cond);
    void csetm(GPR dst, Condition cond);
    void csetm32(GPR dst, Condition cond);

    // =========================================================================
    // Control Flow: Branches, Calls, Returns
    // =========================================================================

    void b(Label target);
    void b(Condition cond, Label target);
    void b(const std::string& symbol);
    void bl(Label target);
    void bl(const std::string& symbol);
    void blr(GPR target);
    void br(GPR target);
    void ret(GPR target = GPR::LR);

    void cbz(GPR reg, Label target);
    void cbz32(GPR reg, Label target);
    void cbnz(GPR reg, Label target);
    void cbnz32(GPR reg, Label target);

    // =========================================================================
    // Memory: Load & Store (GPR)
    // =========================================================================

    void ldr(GPR dst, const MemAddress& mem);
    void ldr32(GPR dst, const MemAddress& mem);
    void ldrb(GPR dst, const MemAddress& mem);
    void ldrh(GPR dst, const MemAddress& mem);
    void ldrsb(GPR dst, const MemAddress& mem);
    void ldrsh(GPR dst, const MemAddress& mem);
    void ldrsw(GPR dst, const MemAddress& mem);

    void str(GPR src, const MemAddress& mem);
    void str32(GPR src, const MemAddress& mem);
    void strb(GPR src, const MemAddress& mem);
    void strh(GPR src, const MemAddress& mem);

    void ldp(GPR dst1, GPR dst2, const MemAddress& mem);
    void ldp32(GPR dst1, GPR dst2, const MemAddress& mem);
    void stp(GPR src1, GPR src2, const MemAddress& mem);
    void stp32(GPR src1, GPR src2, const MemAddress& mem);

    // =========================================================================
    // Floating-Point & SIMD: Arithmetic, Conversions, Loads/Stores
    // =========================================================================

    // Double precision
    void fadd(FPR dst, FPR src1, FPR src2);
    void fsub(FPR dst, FPR src1, FPR src2);
    void fmul(FPR dst, FPR src1, FPR src2);
    void fdiv(FPR dst, FPR src1, FPR src2);
    void fmadd_d(FPR dst, FPR src1, FPR src2, FPR addend);
    void fsqrt(FPR dst, FPR src);
    void fabs(FPR dst, FPR src);
    void fneg(FPR dst, FPR src);
    void fmin(FPR dst, FPR src1, FPR src2);
    void fmax(FPR dst, FPR src1, FPR src2);
    void frintm(FPR dst, FPR src);
    void frintp(FPR dst, FPR src);
    void frinta(FPR dst, FPR src);
    void fcmp(FPR src1, FPR src2);
    void fcmp_zero(FPR src);

    // Single precision
    void fadd_s(FPR dst, FPR src1, FPR src2);
    void fsub_s(FPR dst, FPR src1, FPR src2);
    void fmul_s(FPR dst, FPR src1, FPR src2);
    void fdiv_s(FPR dst, FPR src1, FPR src2);
    void fmadd_s(FPR dst, FPR src1, FPR src2, FPR addend);
    void fsqrt_s(FPR dst, FPR src);
    void fabs_s(FPR dst, FPR src);
    void fneg_s(FPR dst, FPR src);
    void fmin_s(FPR dst, FPR src1, FPR src2);
    void fmax_s(FPR dst, FPR src1, FPR src2);
    void frintm_s(FPR dst, FPR src);
    void frintp_s(FPR dst, FPR src);
    void frinta_s(FPR dst, FPR src);
    void fcmp_s(FPR src1, FPR src2);
    void fcmp_zero_s(FPR src);

    // Moves & Conversions
    void fmov(FPR dst, FPR src);
    void fmov_s(FPR dst, FPR src);
    void fmov_to_gpr(GPR dst, FPR src);
    void fmov_from_gpr(FPR dst, GPR src);
    void fmov_to_gpr32(GPR dst, FPR src);
    void fmov_from_gpr32(FPR dst, GPR src);

    void fcvt_d_s(FPR dst, FPR src); // f32 -> f64
    void fcvt_s_d(FPR dst, FPR src); // f64 -> f32

    void scvtf_d(FPR dst, GPR src);  // i64 -> f64
    void scvtf_s(FPR dst, GPR src);  // i64 -> f32
    void scvtf_d32(FPR dst, GPR src);// i32 -> f64
    void scvtf_s32(FPR dst, GPR src);// i32 -> f32

    void fcvtzs_d(GPR dst, FPR src); // f64 -> i64
    void fcvtzs_d32(GPR dst, FPR src);// f64 -> i32
    void fcvtzs_s(GPR dst, FPR src); // f32 -> i64
    void fcvtzs_s32(GPR dst, FPR src);// f32 -> i32

    // FP Memory
    void ldr(FPR dst, const MemAddress& mem);    // 64-bit Dd
    void ldr_s(FPR dst, const MemAddress& mem);  // 32-bit Sd
    void ldr_q(FPR dst, const MemAddress& mem);  // 128-bit Qd
    void str(FPR src, const MemAddress& mem);    // 64-bit Dd
    void str_s(FPR src, const MemAddress& mem);  // 32-bit Sd
    void str_q(FPR src, const MemAddress& mem);  // 128-bit Qd

    void ldp(FPR dst1, FPR dst2, const MemAddress& mem); // Dd pair
    void stp(FPR src1, FPR src2, const MemAddress& mem); // Dd pair

    // 128-bit SIMD Vector (Float)
    void vec_fadd_4s(FPR dst, FPR src1, FPR src2);
    void vec_fsub_4s(FPR dst, FPR src1, FPR src2);
    void vec_fmul_4s(FPR dst, FPR src1, FPR src2);
    void vec_fdiv_4s(FPR dst, FPR src1, FPR src2);
    void vec_fmla_4s(FPR dst, FPR src1, FPR src2);
    void vec_fmin_4s(FPR dst, FPR src1, FPR src2);
    void vec_fmax_4s(FPR dst, FPR src1, FPR src2);
    void vec_fneg_4s(FPR dst, FPR src);
    void vec_fsqrt_4s(FPR dst, FPR src);

    void vec_fadd_2d(FPR dst, FPR src1, FPR src2);
    void vec_fsub_2d(FPR dst, FPR src1, FPR src2);
    void vec_fmul_2d(FPR dst, FPR src1, FPR src2);
    void vec_fdiv_2d(FPR dst, FPR src1, FPR src2);
    void vec_fmla_2d(FPR dst, FPR src1, FPR src2);
    void vec_fmin_2d(FPR dst, FPR src1, FPR src2);
    void vec_fmax_2d(FPR dst, FPR src1, FPR src2);
    void vec_fneg_2d(FPR dst, FPR src);
    void vec_fsqrt_2d(FPR dst, FPR src);

    // 128-bit SIMD Vector (Integer & Bitwise)
    void vec_add_4s(FPR dst, FPR src1, FPR src2);
    void vec_sub_4s(FPR dst, FPR src1, FPR src2);
    void vec_mul_4s(FPR dst, FPR src1, FPR src2);
    void vec_smin_4s(FPR dst, FPR src1, FPR src2);
    void vec_smax_4s(FPR dst, FPR src1, FPR src2);
    void vec_add_2d(FPR dst, FPR src1, FPR src2);
    void vec_sub_2d(FPR dst, FPR src1, FPR src2);
    void vec_and(FPR dst, FPR src1, FPR src2);
    void vec_orr(FPR dst, FPR src1, FPR src2);
    void vec_eor(FPR dst, FPR src1, FPR src2);
    void cnt_8b(FPR dst, FPR src);
    void uaddlv_h(FPR dst, FPR src);

    // =========================================================================
    // System & Miscellaneous
    // =========================================================================

    void nop();
    void brk(uint16_t imm = 0);
    // cbnz reg, ok; brk #imm; ok: -- the guard in front of every integer
    // division (sdiv/udiv return 0 for a zero divisor, where MIR defines
    // division by zero as a program error, like the x64 #DE fault).
    void brk_if_zero(GPR reg, bool is_64bit, uint16_t imm);
    void adr(GPR dst, Label target);
    void adrp(GPR dst, Label target);

private:
    CodeBuffer& buffer_;

    void emit_add_sub_imm(bool is_64, bool is_sub, bool set_flags, GPR dst, GPR src, uint32_t imm);
    void emit_logical_imm(uint32_t opc, bool is_64, GPR dst, GPR src, uint64_t imm);
    void emit_load_store_imm(bool is_load, uint8_t size_bytes, bool is_signed, GPR reg, const MemAddress& mem);
    void emit_load_store_pair(bool is_load, bool is_64, GPR reg1, GPR reg2, const MemAddress& mem);
};

} // namespace brass::aarch64
