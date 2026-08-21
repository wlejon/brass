#include <brass/target/x64/x64_encoder.hpp>

namespace brass::x64 {

// Helper for standard ALU operations
void X64Encoder::emit_alu_op(uint8_t op_reg_rm, uint8_t op_ext, GPR dst, GPR src, bool w) {
    (void)op_ext;
    emit_rex(w, is_extended(src), false, is_extended(dst));
    buffer_.emit8(op_reg_rm);
    emit_modrm(3, reg_code(src), reg_code(dst));
}

void X64Encoder::emit_alu_imm(uint8_t op_ext, GPR dst, int32_t imm, bool w) {
    if (imm >= -128 && imm <= 127) {
        emit_rex(w, false, false, is_extended(dst));
        buffer_.emit8(0x83);
        emit_modrm(3, op_ext, reg_code(dst));
        buffer_.emit8(static_cast<uint8_t>(static_cast<int8_t>(imm)));
    } else {
        emit_rex(w, false, false, is_extended(dst));
        buffer_.emit8(0x81);
        emit_modrm(3, op_ext, reg_code(dst));
        buffer_.emit32(static_cast<uint32_t>(imm));
    }
}

void X64Encoder::emit_alu_mem(uint8_t op_rm_reg, GPR dst, const MemAddress& src, bool w) {
    emit_rex(w, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(op_rm_reg);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::emit_alu_mem_reg(uint8_t op_reg_rm, const MemAddress& dst, GPR src, bool w) {
    emit_rex(w, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(op_reg_rm);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::emit_alu_mem_imm(uint8_t op_ext, const MemAddress& dst, int32_t imm, bool w) {
    emit_rex(w, false, is_extended(dst.index), is_extended(dst.base));
    if (imm >= -128 && imm <= 127) {
        buffer_.emit8(0x83);
        emit_mem_operand(op_ext, dst);
        buffer_.emit8(static_cast<uint8_t>(static_cast<int8_t>(imm)));
    } else {
        buffer_.emit8(0x81);
        emit_mem_operand(op_ext, dst);
        buffer_.emit32(static_cast<uint32_t>(imm));
    }
}

// =========================================================================
// ALU Implementations
// =========================================================================

void X64Encoder::add(GPR dst, GPR src) { emit_alu_op(0x01, 0, dst, src, true); }
void X64Encoder::add32(GPR dst, GPR src) { emit_alu_op(0x01, 0, dst, src, false); }
void X64Encoder::add(GPR dst, int32_t imm) { emit_alu_imm(0, dst, imm, true); }
void X64Encoder::add32(GPR dst, int32_t imm) { emit_alu_imm(0, dst, imm, false); }
void X64Encoder::add(GPR dst, const MemAddress& src) { emit_alu_mem(0x03, dst, src, true); }
void X64Encoder::add32(GPR dst, const MemAddress& src) { emit_alu_mem(0x03, dst, src, false); }
void X64Encoder::add(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x01, dst, src, true); }
void X64Encoder::add32(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x01, dst, src, false); }
void X64Encoder::add(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(0, dst, imm, true); }
void X64Encoder::add32(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(0, dst, imm, false); }

void X64Encoder::sub(GPR dst, GPR src) { emit_alu_op(0x29, 5, dst, src, true); }
void X64Encoder::sub32(GPR dst, GPR src) { emit_alu_op(0x29, 5, dst, src, false); }
void X64Encoder::sub(GPR dst, int32_t imm) { emit_alu_imm(5, dst, imm, true); }
void X64Encoder::sub32(GPR dst, int32_t imm) { emit_alu_imm(5, dst, imm, false); }
void X64Encoder::sub(GPR dst, const MemAddress& src) { emit_alu_mem(0x2B, dst, src, true); }
void X64Encoder::sub32(GPR dst, const MemAddress& src) { emit_alu_mem(0x2B, dst, src, false); }
void X64Encoder::sub(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x29, dst, src, true); }
void X64Encoder::sub32(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x29, dst, src, false); }
void X64Encoder::sub(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(5, dst, imm, true); }
void X64Encoder::sub32(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(5, dst, imm, false); }

void X64Encoder::and_(GPR dst, GPR src) { emit_alu_op(0x21, 4, dst, src, true); }
void X64Encoder::and32(GPR dst, GPR src) { emit_alu_op(0x21, 4, dst, src, false); }
void X64Encoder::and_(GPR dst, int32_t imm) { emit_alu_imm(4, dst, imm, true); }
void X64Encoder::and32(GPR dst, int32_t imm) { emit_alu_imm(4, dst, imm, false); }
void X64Encoder::and_(GPR dst, const MemAddress& src) { emit_alu_mem(0x23, dst, src, true); }
void X64Encoder::and32(GPR dst, const MemAddress& src) { emit_alu_mem(0x23, dst, src, false); }
void X64Encoder::and_(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x21, dst, src, true); }
void X64Encoder::and32(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x21, dst, src, false); }
void X64Encoder::and_(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(4, dst, imm, true); }
void X64Encoder::and32(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(4, dst, imm, false); }

void X64Encoder::or_(GPR dst, GPR src) { emit_alu_op(0x09, 1, dst, src, true); }
void X64Encoder::or32(GPR dst, GPR src) { emit_alu_op(0x09, 1, dst, src, false); }
void X64Encoder::or_(GPR dst, int32_t imm) { emit_alu_imm(1, dst, imm, true); }
void X64Encoder::or32(GPR dst, int32_t imm) { emit_alu_imm(1, dst, imm, false); }
void X64Encoder::or_(GPR dst, const MemAddress& src) { emit_alu_mem(0x0B, dst, src, true); }
void X64Encoder::or32(GPR dst, const MemAddress& src) { emit_alu_mem(0x0B, dst, src, false); }
void X64Encoder::or_(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x09, dst, src, true); }
void X64Encoder::or32(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x09, dst, src, false); }
void X64Encoder::or_(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(1, dst, imm, true); }
void X64Encoder::or32(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(1, dst, imm, false); }

void X64Encoder::xor_(GPR dst, GPR src) { emit_alu_op(0x31, 6, dst, src, true); }
void X64Encoder::xor32(GPR dst, GPR src) { emit_alu_op(0x31, 6, dst, src, false); }
void X64Encoder::xor_(GPR dst, int32_t imm) { emit_alu_imm(6, dst, imm, true); }
void X64Encoder::xor32(GPR dst, int32_t imm) { emit_alu_imm(6, dst, imm, false); }
void X64Encoder::xor_(GPR dst, const MemAddress& src) { emit_alu_mem(0x33, dst, src, true); }
void X64Encoder::xor32(GPR dst, const MemAddress& src) { emit_alu_mem(0x33, dst, src, false); }
void X64Encoder::xor_(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x31, dst, src, true); }
void X64Encoder::xor32(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x31, dst, src, false); }
void X64Encoder::xor_(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(6, dst, imm, true); }
void X64Encoder::xor32(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(6, dst, imm, false); }

void X64Encoder::cmp(GPR dst, GPR src) { emit_alu_op(0x39, 7, dst, src, true); }
void X64Encoder::cmp32(GPR dst, GPR src) { emit_alu_op(0x39, 7, dst, src, false); }
void X64Encoder::cmp(GPR dst, int32_t imm) { emit_alu_imm(7, dst, imm, true); }
void X64Encoder::cmp32(GPR dst, int32_t imm) { emit_alu_imm(7, dst, imm, false); }
void X64Encoder::cmp(GPR dst, const MemAddress& src) { emit_alu_mem(0x3B, dst, src, true); }
void X64Encoder::cmp32(GPR dst, const MemAddress& src) { emit_alu_mem(0x3B, dst, src, false); }
void X64Encoder::cmp(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x39, dst, src, true); }
void X64Encoder::cmp32(const MemAddress& dst, GPR src) { emit_alu_mem_reg(0x39, dst, src, false); }
void X64Encoder::cmp(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(7, dst, imm, true); }
void X64Encoder::cmp32(const MemAddress& dst, int32_t imm) { emit_alu_mem_imm(7, dst, imm, false); }

void X64Encoder::test(GPR reg1, GPR reg2) {
    emit_rex(true, is_extended(reg2), false, is_extended(reg1));
    buffer_.emit8(0x85);
    emit_modrm(3, reg_code(reg2), reg_code(reg1));
}

void X64Encoder::test32(GPR reg1, GPR reg2) {
    emit_rex(false, is_extended(reg2), false, is_extended(reg1));
    buffer_.emit8(0x85);
    emit_modrm(3, reg_code(reg2), reg_code(reg1));
}

void X64Encoder::test(GPR reg, int32_t imm) {
    emit_rex(true, false, false, is_extended(reg));
    buffer_.emit8(0xF7);
    emit_modrm(3, 0, reg_code(reg));
    buffer_.emit32(static_cast<uint32_t>(imm));
}

void X64Encoder::test32(GPR reg, int32_t imm) {
    emit_rex(false, false, false, is_extended(reg));
    buffer_.emit8(0xF7);
    emit_modrm(3, 0, reg_code(reg));
    buffer_.emit32(static_cast<uint32_t>(imm));
}

void X64Encoder::test(const MemAddress& mem, GPR reg) {
    emit_rex(true, is_extended(reg), is_extended(mem.index), is_extended(mem.base));
    buffer_.emit8(0x85);
    emit_mem_operand(reg_code(reg), mem);
}

void X64Encoder::test32(const MemAddress& mem, GPR reg) {
    emit_rex(false, is_extended(reg), is_extended(mem.index), is_extended(mem.base));
    buffer_.emit8(0x85);
    emit_mem_operand(reg_code(reg), mem);
}

void X64Encoder::test(const MemAddress& mem, int32_t imm) {
    emit_rex(true, false, is_extended(mem.index), is_extended(mem.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(0, mem);
    buffer_.emit32(static_cast<uint32_t>(imm));
}

void X64Encoder::test32(const MemAddress& mem, int32_t imm) {
    emit_rex(false, false, is_extended(mem.index), is_extended(mem.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(0, mem);
    buffer_.emit32(static_cast<uint32_t>(imm));
}

void X64Encoder::test8(GPR reg1, GPR reg2) {
    bool need_rex = is_extended(reg1) || is_extended(reg2) || reg_id(reg1) >= 4 || reg_id(reg2) >= 4;
    if (need_rex) {
        emit_rex_force(false, is_extended(reg2), false, is_extended(reg1));
    }
    buffer_.emit8(0x84);
    emit_modrm(3, reg_code(reg2), reg_code(reg1));
}

void X64Encoder::test8(GPR reg, uint8_t imm) {
    bool need_rex = is_extended(reg) || reg_id(reg) >= 4;
    if (need_rex) {
        emit_rex_force(false, false, false, is_extended(reg));
    }
    buffer_.emit8(0xF6);
    emit_modrm(3, 0, reg_code(reg));
    buffer_.emit8(imm);
}

void X64Encoder::not_(GPR dst) {
    emit_rex(true, false, false, is_extended(dst));
    buffer_.emit8(0xF7);
    emit_modrm(3, 2, reg_code(dst));
}

void X64Encoder::not32(GPR dst) {
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(0xF7);
    emit_modrm(3, 2, reg_code(dst));
}

void X64Encoder::not_(const MemAddress& dst) {
    emit_rex(true, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(2, dst);
}

void X64Encoder::not32(const MemAddress& dst) {
    emit_rex(false, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(2, dst);
}

void X64Encoder::neg(GPR dst) {
    emit_rex(true, false, false, is_extended(dst));
    buffer_.emit8(0xF7);
    emit_modrm(3, 3, reg_code(dst));
}

void X64Encoder::neg32(GPR dst) {
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(0xF7);
    emit_modrm(3, 3, reg_code(dst));
}

void X64Encoder::neg(const MemAddress& dst) {
    emit_rex(true, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(3, dst);
}

void X64Encoder::neg32(const MemAddress& dst) {
    emit_rex(false, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(3, dst);
}

// Multiplication & Division
void X64Encoder::mul(GPR src) {
    emit_rex(true, false, false, is_extended(src));
    buffer_.emit8(0xF7);
    emit_modrm(3, 4, reg_code(src));
}

void X64Encoder::mul32(GPR src) {
    emit_rex(false, false, false, is_extended(src));
    buffer_.emit8(0xF7);
    emit_modrm(3, 4, reg_code(src));
}

void X64Encoder::mul(const MemAddress& src) {
    emit_rex(true, false, is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(4, src);
}

void X64Encoder::mul32(const MemAddress& src) {
    emit_rex(false, false, is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(4, src);
}

void X64Encoder::div(GPR src) {
    emit_rex(true, false, false, is_extended(src));
    buffer_.emit8(0xF7);
    emit_modrm(3, 6, reg_code(src));
}

void X64Encoder::div32(GPR src) {
    emit_rex(false, false, false, is_extended(src));
    buffer_.emit8(0xF7);
    emit_modrm(3, 6, reg_code(src));
}

void X64Encoder::div(const MemAddress& src) {
    emit_rex(true, false, is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(6, src);
}

void X64Encoder::div32(const MemAddress& src) {
    emit_rex(false, false, is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(6, src);
}

void X64Encoder::idiv(GPR src) {
    emit_rex(true, false, false, is_extended(src));
    buffer_.emit8(0xF7);
    emit_modrm(3, 7, reg_code(src));
}

void X64Encoder::idiv32(GPR src) {
    emit_rex(false, false, false, is_extended(src));
    buffer_.emit8(0xF7);
    emit_modrm(3, 7, reg_code(src));
}

void X64Encoder::idiv(const MemAddress& src) {
    emit_rex(true, false, is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(7, src);
}

void X64Encoder::idiv32(const MemAddress& src) {
    emit_rex(false, false, is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0xF7);
    emit_mem_operand(7, src);
}

void X64Encoder::imul(GPR src) {
    emit_rex(true, false, false, is_extended(src));
    buffer_.emit8(0xF7);
    emit_modrm(3, 5, reg_code(src));
}

void X64Encoder::imul32(GPR src) {
    emit_rex(false, false, false, is_extended(src));
    buffer_.emit8(0xF7);
    emit_modrm(3, 5, reg_code(src));
}

void X64Encoder::imul(GPR dst, GPR src) {
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xAF);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::imul32(GPR dst, GPR src) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xAF);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::imul(GPR dst, const MemAddress& src) {
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xAF);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::imul32(GPR dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xAF);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::imul(GPR dst, GPR src, int32_t imm) {
    emit_rex(true, is_extended(dst), false, is_extended(src));
    if (imm >= -128 && imm <= 127) {
        buffer_.emit8(0x6B);
        emit_modrm(3, reg_code(dst), reg_code(src));
        buffer_.emit8(static_cast<uint8_t>(static_cast<int8_t>(imm)));
    } else {
        buffer_.emit8(0x69);
        emit_modrm(3, reg_code(dst), reg_code(src));
        buffer_.emit32(static_cast<uint32_t>(imm));
    }
}

void X64Encoder::imul32(GPR dst, GPR src, int32_t imm) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    if (imm >= -128 && imm <= 127) {
        buffer_.emit8(0x6B);
        emit_modrm(3, reg_code(dst), reg_code(src));
        buffer_.emit8(static_cast<uint8_t>(static_cast<int8_t>(imm)));
    } else {
        buffer_.emit8(0x69);
        emit_modrm(3, reg_code(dst), reg_code(src));
        buffer_.emit32(static_cast<uint32_t>(imm));
    }
}

void X64Encoder::imul(GPR dst, int32_t imm) {
    imul(dst, dst, imm);
}

void X64Encoder::imul32(GPR dst, int32_t imm) {
    imul32(dst, dst, imm);
}

// Shifts & Rotates
void X64Encoder::shl(GPR dst, uint8_t imm) {
    emit_rex(true, false, false, is_extended(dst));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_modrm(3, 4, reg_code(dst));
    } else {
        buffer_.emit8(0xC1);
        emit_modrm(3, 4, reg_code(dst));
        buffer_.emit8(imm);
    }
}

void X64Encoder::shl(GPR dst) {
    emit_rex(true, false, false, is_extended(dst));
    buffer_.emit8(0xD3);
    emit_modrm(3, 4, reg_code(dst));
}

void X64Encoder::shl32(GPR dst, uint8_t imm) {
    emit_rex(false, false, false, is_extended(dst));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_modrm(3, 4, reg_code(dst));
    } else {
        buffer_.emit8(0xC1);
        emit_modrm(3, 4, reg_code(dst));
        buffer_.emit8(imm);
    }
}

void X64Encoder::shl32(GPR dst) {
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(0xD3);
    emit_modrm(3, 4, reg_code(dst));
}

void X64Encoder::shl(const MemAddress& dst, uint8_t imm) {
    emit_rex(true, false, is_extended(dst.index), is_extended(dst.base));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_mem_operand(4, dst);
    } else {
        buffer_.emit8(0xC1);
        emit_mem_operand(4, dst);
        buffer_.emit8(imm);
    }
}

void X64Encoder::shl_cl(const MemAddress& dst) {
    emit_rex(true, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xD3);
    emit_mem_operand(4, dst);
}

void X64Encoder::shl32(const MemAddress& dst, uint8_t imm) {
    emit_rex(false, false, is_extended(dst.index), is_extended(dst.base));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_mem_operand(4, dst);
    } else {
        buffer_.emit8(0xC1);
        emit_mem_operand(4, dst);
        buffer_.emit8(imm);
    }
}

void X64Encoder::shl32_cl(const MemAddress& dst) {
    emit_rex(false, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xD3);
    emit_mem_operand(4, dst);
}

void X64Encoder::shr(GPR dst, uint8_t imm) {
    emit_rex(true, false, false, is_extended(dst));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_modrm(3, 5, reg_code(dst));
    } else {
        buffer_.emit8(0xC1);
        emit_modrm(3, 5, reg_code(dst));
        buffer_.emit8(imm);
    }
}

void X64Encoder::shr(GPR dst) {
    emit_rex(true, false, false, is_extended(dst));
    buffer_.emit8(0xD3);
    emit_modrm(3, 5, reg_code(dst));
}

void X64Encoder::shr32(GPR dst, uint8_t imm) {
    emit_rex(false, false, false, is_extended(dst));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_modrm(3, 5, reg_code(dst));
    } else {
        buffer_.emit8(0xC1);
        emit_modrm(3, 5, reg_code(dst));
        buffer_.emit8(imm);
    }
}

void X64Encoder::shr32(GPR dst) {
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(0xD3);
    emit_modrm(3, 5, reg_code(dst));
}

void X64Encoder::shr(const MemAddress& dst, uint8_t imm) {
    emit_rex(true, false, is_extended(dst.index), is_extended(dst.base));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_mem_operand(5, dst);
    } else {
        buffer_.emit8(0xC1);
        emit_mem_operand(5, dst);
        buffer_.emit8(imm);
    }
}

void X64Encoder::shr_cl(const MemAddress& dst) {
    emit_rex(true, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xD3);
    emit_mem_operand(5, dst);
}

void X64Encoder::shr32(const MemAddress& dst, uint8_t imm) {
    emit_rex(false, false, is_extended(dst.index), is_extended(dst.base));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_mem_operand(5, dst);
    } else {
        buffer_.emit8(0xC1);
        emit_mem_operand(5, dst);
        buffer_.emit8(imm);
    }
}

void X64Encoder::shr32_cl(const MemAddress& dst) {
    emit_rex(false, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xD3);
    emit_mem_operand(5, dst);
}

void X64Encoder::sar(GPR dst, uint8_t imm) {
    emit_rex(true, false, false, is_extended(dst));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_modrm(3, 7, reg_code(dst));
    } else {
        buffer_.emit8(0xC1);
        emit_modrm(3, 7, reg_code(dst));
        buffer_.emit8(imm);
    }
}

void X64Encoder::sar(GPR dst) {
    emit_rex(true, false, false, is_extended(dst));
    buffer_.emit8(0xD3);
    emit_modrm(3, 7, reg_code(dst));
}

void X64Encoder::sar32(GPR dst, uint8_t imm) {
    emit_rex(false, false, false, is_extended(dst));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_modrm(3, 7, reg_code(dst));
    } else {
        buffer_.emit8(0xC1);
        emit_modrm(3, 7, reg_code(dst));
        buffer_.emit8(imm);
    }
}

void X64Encoder::sar32(GPR dst) {
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(0xD3);
    emit_modrm(3, 7, reg_code(dst));
}

void X64Encoder::sar(const MemAddress& dst, uint8_t imm) {
    emit_rex(true, false, is_extended(dst.index), is_extended(dst.base));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_mem_operand(7, dst);
    } else {
        buffer_.emit8(0xC1);
        emit_mem_operand(7, dst);
        buffer_.emit8(imm);
    }
}

void X64Encoder::sar_cl(const MemAddress& dst) {
    emit_rex(true, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xD3);
    emit_mem_operand(7, dst);
}

void X64Encoder::sar32(const MemAddress& dst, uint8_t imm) {
    emit_rex(false, false, is_extended(dst.index), is_extended(dst.base));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_mem_operand(7, dst);
    } else {
        buffer_.emit8(0xC1);
        emit_mem_operand(7, dst);
        buffer_.emit8(imm);
    }
}

void X64Encoder::sar32_cl(const MemAddress& dst) {
    emit_rex(false, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xD3);
    emit_mem_operand(7, dst);
}

void X64Encoder::rol(GPR dst, uint8_t imm) {
    emit_rex(true, false, false, is_extended(dst));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_modrm(3, 0, reg_code(dst));
    } else {
        buffer_.emit8(0xC1);
        emit_modrm(3, 0, reg_code(dst));
        buffer_.emit8(imm);
    }
}

void X64Encoder::rol(GPR dst) {
    emit_rex(true, false, false, is_extended(dst));
    buffer_.emit8(0xD3);
    emit_modrm(3, 0, reg_code(dst));
}

void X64Encoder::rol32(GPR dst, uint8_t imm) {
    emit_rex(false, false, false, is_extended(dst));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_modrm(3, 0, reg_code(dst));
    } else {
        buffer_.emit8(0xC1);
        emit_modrm(3, 0, reg_code(dst));
        buffer_.emit8(imm);
    }
}

void X64Encoder::rol32(GPR dst) {
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(0xD3);
    emit_modrm(3, 0, reg_code(dst));
}

void X64Encoder::ror(GPR dst, uint8_t imm) {
    emit_rex(true, false, false, is_extended(dst));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_modrm(3, 1, reg_code(dst));
    } else {
        buffer_.emit8(0xC1);
        emit_modrm(3, 1, reg_code(dst));
        buffer_.emit8(imm);
    }
}

void X64Encoder::ror(GPR dst) {
    emit_rex(true, false, false, is_extended(dst));
    buffer_.emit8(0xD3);
    emit_modrm(3, 1, reg_code(dst));
}

void X64Encoder::ror32(GPR dst, uint8_t imm) {
    emit_rex(false, false, false, is_extended(dst));
    if (imm == 1) {
        buffer_.emit8(0xD1);
        emit_modrm(3, 1, reg_code(dst));
    } else {
        buffer_.emit8(0xC1);
        emit_modrm(3, 1, reg_code(dst));
        buffer_.emit8(imm);
    }
}

void X64Encoder::ror32(GPR dst) {
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(0xD3);
    emit_modrm(3, 1, reg_code(dst));
}

} // namespace brass::x64
