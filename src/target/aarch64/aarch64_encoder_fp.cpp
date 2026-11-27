#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <stdexcept>

namespace brass::aarch64 {

// Double precision
void AArch64Encoder::fadd(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E602800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fsub(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E603800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fmul(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E600800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fdiv(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E601800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fmadd_d(FPR dst, FPR src1, FPR src2, FPR addend) {
    buffer_.emit_inst(0x1F400000u | (reg_code(src2) << 16) | (reg_code(addend) << 10) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fsqrt(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E61C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fabs(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E60C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fneg(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E614000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmin(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E605800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fmax(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E604800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::frintm(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E654000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::frintp(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E64C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::frinta(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E664000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcmp(FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E602000u | (reg_code(src2) << 16) | (reg_code(src1) << 5));
}

void AArch64Encoder::fcmp_zero(FPR src) {
    buffer_.emit_inst(0x1E602008u | (reg_code(src) << 5));
}

// Single precision
void AArch64Encoder::fadd_s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E202800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fsub_s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E203800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fmul_s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E200800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fdiv_s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E201800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fmadd_s(FPR dst, FPR src1, FPR src2, FPR addend) {
    buffer_.emit_inst(0x1F000000u | (reg_code(src2) << 16) | (reg_code(addend) << 10) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fsqrt_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E21C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fabs_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E20C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fneg_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E214000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmin_s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E205800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fmax_s(FPR dst, FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E204800u | (reg_code(src2) << 16) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::frintm_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E254000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::frintp_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E24C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::frinta_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E264000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcmp_s(FPR src1, FPR src2) {
    buffer_.emit_inst(0x1E202000u | (reg_code(src2) << 16) | (reg_code(src1) << 5));
}

void AArch64Encoder::fcmp_zero_s(FPR src) {
    buffer_.emit_inst(0x1E202008u | (reg_code(src) << 5));
}

void AArch64Encoder::fcsel(FPR dst, FPR src1, FPR src2, Condition cond) {
    buffer_.emit_inst(0x1E600C00u | (reg_code(src2) << 16) |
                      (static_cast<uint32_t>(cond) << 12) | (reg_code(src1) << 5) | reg_code(dst));
}

void AArch64Encoder::fcsel_s(FPR dst, FPR src1, FPR src2, Condition cond) {
    buffer_.emit_inst(0x1E200C00u | (reg_code(src2) << 16) |
                      (static_cast<uint32_t>(cond) << 12) | (reg_code(src1) << 5) | reg_code(dst));
}

// Moves & Conversions
bool AArch64Encoder::encode_fp64_imm8(uint64_t bits, uint8_t& out_imm8) noexcept {
    const uint64_t sign = bits >> 63;
    const int32_t exp = static_cast<int32_t>((bits >> 52) & 0x7FFu) - 1023;
    const uint64_t frac = bits & ((1ULL << 52) - 1);
    if (exp < -3 || exp > 4 || (frac & ((1ULL << 48) - 1)) != 0) return false;
    const uint32_t enc_exp = static_cast<uint32_t>(exp + 3) ^ 4u;
    out_imm8 = static_cast<uint8_t>((sign << 7) | ((enc_exp & 7u) << 4) | (frac >> 48));
    return true;
}

bool AArch64Encoder::encode_fp32_imm8(uint32_t bits, uint8_t& out_imm8) noexcept {
    const uint32_t sign = bits >> 31;
    const int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFFu) - 127;
    const uint32_t frac = bits & ((1u << 23) - 1);
    if (exp < -3 || exp > 4 || (frac & ((1u << 19) - 1)) != 0) return false;
    const uint32_t enc_exp = static_cast<uint32_t>(exp + 3) ^ 4u;
    out_imm8 = static_cast<uint8_t>((sign << 7) | ((enc_exp & 7u) << 4) | (frac >> 19));
    return true;
}

void AArch64Encoder::fmov(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E604000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmov_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E204000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmov_imm(FPR dst, uint8_t imm8) {
    buffer_.emit_inst(0x1E601000u | (static_cast<uint32_t>(imm8) << 13) | reg_code(dst));
}

void AArch64Encoder::fmov_imm_s(FPR dst, uint8_t imm8) {
    buffer_.emit_inst(0x1E201000u | (static_cast<uint32_t>(imm8) << 13) | reg_code(dst));
}

void AArch64Encoder::movi_zero(FPR dst) {
    buffer_.emit_inst(0x2F00E400u | reg_code(dst));
}

void AArch64Encoder::fmov_to_gpr(GPR dst, FPR src) {
    buffer_.emit_inst(0x9E660000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmov_from_gpr(FPR dst, GPR src) {
    buffer_.emit_inst(0x9E670000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmov_to_gpr32(GPR dst, FPR src) {
    buffer_.emit_inst(0x1E260000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fmov_from_gpr32(FPR dst, GPR src) {
    buffer_.emit_inst(0x1E270000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvt_d_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E22C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvt_s_d(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E624000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvt_s_h(FPR dst, FPR src) {
    buffer_.emit_inst(0x1EE24000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvt_h_s(FPR dst, FPR src) {
    buffer_.emit_inst(0x1E23C000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::scvtf_d(FPR dst, GPR src) {
    buffer_.emit_inst(0x9E620000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::scvtf_s(FPR dst, GPR src) {
    buffer_.emit_inst(0x9E220000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::scvtf_d32(FPR dst, GPR src) {
    buffer_.emit_inst(0x1E620000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::scvtf_s32(FPR dst, GPR src) {
    buffer_.emit_inst(0x1E220000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvtzs_d(GPR dst, FPR src) {
    buffer_.emit_inst(0x9E780000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvtzs_d32(GPR dst, FPR src) {
    buffer_.emit_inst(0x1E780000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvtzs_s(GPR dst, FPR src) {
    buffer_.emit_inst(0x9E380000u | (reg_code(src) << 5) | reg_code(dst));
}

void AArch64Encoder::fcvtzs_s32(GPR dst, FPR src) {
    buffer_.emit_inst(0x1E380000u | (reg_code(src) << 5) | reg_code(dst));
}

// FP Loads & Stores
void AArch64Encoder::ldr(FPR dst, const MemAddress& mem) {
    // 64-bit Dd load
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(dst);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (3u << 30) | (7u << 27) | (1u << 26) | (1u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 8 == 0) && (off / 8 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 8) & 0xFFFu;
        buffer_.emit_inst(0xFD400000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0xFC400000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: FP load offset out of range");
    }
}

void AArch64Encoder::ldr_s(FPR dst, const MemAddress& mem) {
    // 32-bit Sd load
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(dst);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (2u << 30) | (7u << 27) | (1u << 26) | (1u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 4 == 0) && (off / 4 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 4) & 0xFFFu;
        buffer_.emit_inst(0xBD400000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0xBC400000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: FP load offset out of range");
    }
}

void AArch64Encoder::ldr_q(FPR dst, const MemAddress& mem) {
    // 128-bit Qd load
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(dst);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (0u << 30) | (7u << 27) | (1u << 26) | (3u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 16 == 0) && (off / 16 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 16) & 0xFFFu;
        buffer_.emit_inst(0x3DC00000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0x3CC00000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: 128-bit FP load offset out of range");
    }
}

void AArch64Encoder::str(FPR src, const MemAddress& mem) {
    // 64-bit Dd store
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(src);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (3u << 30) | (7u << 27) | (1u << 26) | (0u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 8 == 0) && (off / 8 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 8) & 0xFFFu;
        buffer_.emit_inst(0xFD000000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0xFC000000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: FP store offset out of range");
    }
}

void AArch64Encoder::str_s(FPR src, const MemAddress& mem) {
    // 32-bit Sd store
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(src);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (2u << 30) | (7u << 27) | (1u << 26) | (0u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 4 == 0) && (off / 4 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 4) & 0xFFFu;
        buffer_.emit_inst(0xBD000000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0xBC000000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: FP store offset out of range");
    }
}

void AArch64Encoder::str_q(FPR src, const MemAddress& mem) {
    // 128-bit Qd store
    uint32_t rn = reg_code(mem.base);
    uint32_t rt = reg_code(src);
    if (mem.mode == AddrMode::RegOffset && mem.has_index()) {
        uint32_t rm = reg_code(mem.index);
        uint32_t ext = static_cast<uint32_t>(mem.extend);
        uint32_t s = (mem.shift > 0) ? 1u : 0u;
        uint32_t inst = (0u << 30) | (7u << 27) | (1u << 26) | (2u << 22) | (1u << 21) |
                        (rm << 16) | (ext << 13) | (s << 12) | (2u << 10) | (rn << 5) | rt;
        buffer_.emit_inst(inst);
        return;
    }
    int64_t off = mem.offset;

    if (off >= 0 && (off % 16 == 0) && (off / 16 <= 4095)) {
        uint32_t imm12 = static_cast<uint32_t>(off / 16) & 0xFFFu;
        buffer_.emit_inst(0x3D800000u | (imm12 << 10) | (rn << 5) | rt);
    } else if (off >= -256 && off <= 255) {
        uint32_t simm9 = static_cast<uint32_t>(off) & 0x1FFu;
        buffer_.emit_inst(0x3C800000u | (simm9 << 12) | (rn << 5) | rt);
    } else {
        throw std::runtime_error("AArch64Encoder: 128-bit FP store offset out of range");
    }
}

void AArch64Encoder::ldp(FPR dst1, FPR dst2, const MemAddress& mem) {
    int64_t off = mem.offset;
    if (off % 8 != 0) throw std::runtime_error("AArch64Encoder: FP LDP offset must be 8-byte aligned");
    int64_t simm7 = off / 8;
    if (simm7 < -64 || simm7 > 63) throw std::runtime_error("AArch64Encoder: FP LDP offset out of range");
    uint32_t imm7 = static_cast<uint32_t>(simm7) & 0x7Fu;
    uint32_t mode = mem.is_post_indexed() ? 1u : (mem.is_pre_indexed() ? 3u : 2u);
    buffer_.emit_inst((1u << 30) | (5u << 27) | (1u << 26) | (mode << 23) | (1u << 22) |
                      (imm7 << 15) | (reg_code(dst2) << 10) | (reg_code(mem.base) << 5) | reg_code(dst1));
}

void AArch64Encoder::stp(FPR src1, FPR src2, const MemAddress& mem) {
    int64_t off = mem.offset;
    if (off % 8 != 0) throw std::runtime_error("AArch64Encoder: FP STP offset must be 8-byte aligned");
    int64_t simm7 = off / 8;
    if (simm7 < -64 || simm7 > 63) throw std::runtime_error("AArch64Encoder: FP STP offset out of range");
    uint32_t imm7 = static_cast<uint32_t>(simm7) & 0x7Fu;
    uint32_t mode = mem.is_post_indexed() ? 1u : (mem.is_pre_indexed() ? 3u : 2u);
    buffer_.emit_inst((1u << 30) | (5u << 27) | (1u << 26) | (mode << 23) | (0u << 22) |
                      (imm7 << 15) | (reg_code(src2) << 10) | (reg_code(mem.base) << 5) | reg_code(src1));
}

static void emit_fp_pair(CodeBuffer& buf, bool is_load, uint32_t opc, int64_t scale,
                         FPR r1, FPR r2, const MemAddress& mem, const char* err_tag) {
    int64_t off = mem.offset;
    if (off % scale != 0) throw std::runtime_error(std::string("AArch64Encoder: ") + err_tag + " offset unaligned");
    int64_t simm7 = off / scale;
    if (simm7 < -64 || simm7 > 63) {
        throw std::runtime_error(std::string("AArch64Encoder: ") + err_tag + " offset out of range");
    }
    uint32_t imm7 = static_cast<uint32_t>(simm7) & 0x7Fu;
    uint32_t mode = mem.is_post_indexed() ? 1u : (mem.is_pre_indexed() ? 3u : 2u);
    uint32_t l = is_load ? 1u : 0u;
    buf.emit_inst((opc << 30) | (5u << 27) | (1u << 26) | (mode << 23) | (l << 22) |
                  (imm7 << 15) | (reg_code(r2) << 10) | (reg_code(mem.base) << 5) | reg_code(r1));
}

void AArch64Encoder::ldp_s(FPR dst1, FPR dst2, const MemAddress& mem) {
    emit_fp_pair(buffer_, true, 0u, 4, dst1, dst2, mem, "FP LDP.S");
}

void AArch64Encoder::stp_s(FPR src1, FPR src2, const MemAddress& mem) {
    emit_fp_pair(buffer_, false, 0u, 4, src1, src2, mem, "FP STP.S");
}

void AArch64Encoder::ldp_q(FPR dst1, FPR dst2, const MemAddress& mem) {
    emit_fp_pair(buffer_, true, 2u, 16, dst1, dst2, mem, "FP LDP.Q");
}

void AArch64Encoder::stp_q(FPR src1, FPR src2, const MemAddress& mem) {
    emit_fp_pair(buffer_, false, 2u, 16, src1, src2, mem, "FP STP.Q");
}

} // namespace brass::aarch64
