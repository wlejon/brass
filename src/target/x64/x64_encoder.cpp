#include <brass/target/x64/x64_encoder.hpp>

namespace brass::x64 {

void X64Encoder::emit_rex(bool w, bool r, bool x, bool b) {
    if (w || r || x || b) {
        uint8_t byte = 0x40;
        if (w) byte |= 0x08;
        if (r) byte |= 0x04;
        if (x) byte |= 0x02;
        if (b) byte |= 0x01;
        buffer_.emit8(byte);
    }
}

void X64Encoder::emit_rex_force(bool w, bool r, bool x, bool b) {
    uint8_t byte = 0x40;
    if (w) byte |= 0x08;
    if (r) byte |= 0x04;
    if (x) byte |= 0x02;
    if (b) byte |= 0x01;
    buffer_.emit8(byte);
}

void X64Encoder::emit_modrm(uint8_t mod, uint8_t reg, uint8_t rm) {
    buffer_.emit8(static_cast<uint8_t>(((mod & 0x03) << 6) | ((reg & 0x07) << 3) | (rm & 0x07)));
}

void X64Encoder::emit_sib(Scale scale, uint8_t index, uint8_t base) {
    uint8_t ss = scale_bits(scale);
    buffer_.emit8(static_cast<uint8_t>(((ss & 0x03) << 6) | ((index & 0x07) << 3) | (base & 0x07)));
}

void X64Encoder::emit_mem_operand(uint8_t reg_code_val, const MemAddress& mem) {
    if (mem.is_rip_rel) {
        emit_modrm(0, reg_code_val, 5);
        buffer_.emit32(static_cast<uint32_t>(mem.disp));
        return;
    }

    if (mem.has_base() && !mem.has_index()) {
        uint8_t b_code = reg_code(mem.base);
        if (b_code == 4) { // RSP or R12
            if (mem.disp == 0) {
                emit_modrm(0, reg_code_val, 4);
                emit_sib(Scale::One, 4, 4);
            } else if (mem.disp >= -128 && mem.disp <= 127) {
                emit_modrm(1, reg_code_val, 4);
                emit_sib(Scale::One, 4, 4);
                buffer_.emit8(static_cast<uint8_t>(static_cast<int8_t>(mem.disp)));
            } else {
                emit_modrm(2, reg_code_val, 4);
                emit_sib(Scale::One, 4, 4);
                buffer_.emit32(static_cast<uint32_t>(mem.disp));
            }
            return;
        }

        if (b_code == 5) { // RBP or R13
            if (mem.disp >= -128 && mem.disp <= 127) {
                emit_modrm(1, reg_code_val, 5);
                buffer_.emit8(static_cast<uint8_t>(static_cast<int8_t>(mem.disp)));
            } else {
                emit_modrm(2, reg_code_val, 5);
                buffer_.emit32(static_cast<uint32_t>(mem.disp));
            }
            return;
        }

        // Other base registers
        if (mem.disp == 0) {
            emit_modrm(0, reg_code_val, b_code);
        } else if (mem.disp >= -128 && mem.disp <= 127) {
            emit_modrm(1, reg_code_val, b_code);
            buffer_.emit8(static_cast<uint8_t>(static_cast<int8_t>(mem.disp)));
        } else {
            emit_modrm(2, reg_code_val, b_code);
            buffer_.emit32(static_cast<uint32_t>(mem.disp));
        }
        return;
    }

    if (mem.has_base() && mem.has_index()) {
        uint8_t b_code = reg_code(mem.base);
        uint8_t i_code = reg_code(mem.index);

        if (b_code == 5) { // RBP or R13
            if (mem.disp >= -128 && mem.disp <= 127) {
                emit_modrm(1, reg_code_val, 4);
                emit_sib(mem.scale, i_code, b_code);
                buffer_.emit8(static_cast<uint8_t>(static_cast<int8_t>(mem.disp)));
            } else {
                emit_modrm(2, reg_code_val, 4);
                emit_sib(mem.scale, i_code, b_code);
                buffer_.emit32(static_cast<uint32_t>(mem.disp));
            }
            return;
        }

        if (mem.disp == 0) {
            emit_modrm(0, reg_code_val, 4);
            emit_sib(mem.scale, i_code, b_code);
        } else if (mem.disp >= -128 && mem.disp <= 127) {
            emit_modrm(1, reg_code_val, 4);
            emit_sib(mem.scale, i_code, b_code);
            buffer_.emit8(static_cast<uint8_t>(static_cast<int8_t>(mem.disp)));
        } else {
            emit_modrm(2, reg_code_val, 4);
            emit_sib(mem.scale, i_code, b_code);
            buffer_.emit32(static_cast<uint32_t>(mem.disp));
        }
        return;
    }

    if (!mem.has_base() && mem.has_index()) {
        emit_modrm(0, reg_code_val, 4);
        emit_sib(mem.scale, reg_code(mem.index), 5);
        buffer_.emit32(static_cast<uint32_t>(mem.disp));
        return;
    }

    // No base, no index: [disp32]
    emit_modrm(0, reg_code_val, 4);
    emit_sib(Scale::One, 4, 5); // 0x25
    buffer_.emit32(static_cast<uint32_t>(mem.disp));
}

// =========================================================================
// MOVES & EXTENSIONS
// =========================================================================

void X64Encoder::mov(GPR dst, GPR src) {
    emit_rex(true, is_extended(src), false, is_extended(dst));
    buffer_.emit8(0x89);
    emit_modrm(3, reg_code(src), reg_code(dst));
}

void X64Encoder::mov32(GPR dst, GPR src) {
    emit_rex(false, is_extended(src), false, is_extended(dst));
    buffer_.emit8(0x89);
    emit_modrm(3, reg_code(src), reg_code(dst));
}

void X64Encoder::mov(GPR dst, int64_t imm) {
    if (imm >= INT32_MIN && imm <= INT32_MAX) {
        emit_rex(true, false, false, is_extended(dst));
        buffer_.emit8(0xC7);
        emit_modrm(3, 0, reg_code(dst));
        buffer_.emit32(static_cast<uint32_t>(static_cast<int32_t>(imm)));
    } else {
        movabs(dst, static_cast<uint64_t>(imm));
    }
}

void X64Encoder::mov32(GPR dst, uint32_t imm) {
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(static_cast<uint8_t>(0xB8 + reg_code(dst)));
    buffer_.emit32(imm);
}

void X64Encoder::movabs(GPR dst, uint64_t imm) {
    emit_rex(true, false, false, is_extended(dst));
    buffer_.emit8(static_cast<uint8_t>(0xB8 + reg_code(dst)));
    buffer_.emit64(imm);
}

void X64Encoder::movabs(GPR dst, const std::string& symbol) {
    emit_rex(true, false, false, is_extended(dst));
    buffer_.emit8(static_cast<uint8_t>(0xB8 + reg_code(dst)));
    size_t patch_off = buffer_.size();
    buffer_.emit64(0);
    buffer_.add_relocation(patch_off, RelocationKind::Abs64, symbol, 0);
}

void X64Encoder::mov64(GPR dst, uint64_t imm) {
    movabs(dst, imm);
}

void X64Encoder::mov(GPR dst, const MemAddress& src) {
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x8B);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::mov32(GPR dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x8B);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::mov(const MemAddress& dst, GPR src) {
    emit_rex(true, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0x89);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::mov32(const MemAddress& dst, GPR src) {
    emit_rex(false, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0x89);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::mov(const MemAddress& dst, int32_t imm) {
    emit_rex(true, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xC7);
    emit_mem_operand(0, dst);
    buffer_.emit32(static_cast<uint32_t>(imm));
}

void X64Encoder::mov32(const MemAddress& dst, int32_t imm) {
    emit_rex(false, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xC7);
    emit_mem_operand(0, dst);
    buffer_.emit32(static_cast<uint32_t>(imm));
}

void X64Encoder::xchg(GPR dst, GPR src) {
    if (dst == src) return;
    emit_rex(true, is_extended(src), false, is_extended(dst));
    buffer_.emit8(0x87);
    emit_modrm(3, reg_code(src), reg_code(dst));
}

void X64Encoder::xchg32(GPR dst, GPR src) {
    if (dst == src) return;
    emit_rex(false, is_extended(src), false, is_extended(dst));
    buffer_.emit8(0x87);
    emit_modrm(3, reg_code(src), reg_code(dst));
}

// 8-bit & 16-bit Moves
void X64Encoder::mov8(GPR dst, GPR src) {
    bool need_rex = is_extended(src) || is_extended(dst) || reg_id(src) >= 4 || reg_id(dst) >= 4;
    if (need_rex) {
        emit_rex_force(false, is_extended(src), false, is_extended(dst));
    }
    buffer_.emit8(0x88);
    emit_modrm(3, reg_code(src), reg_code(dst));
}

void X64Encoder::mov8(GPR dst, uint8_t imm) {
    bool need_rex = is_extended(dst) || reg_id(dst) >= 4;
    if (need_rex) {
        emit_rex_force(false, false, false, is_extended(dst));
    }
    buffer_.emit8(static_cast<uint8_t>(0xB0 + reg_code(dst)));
    buffer_.emit8(imm);
}

void X64Encoder::mov8(GPR dst, const MemAddress& src) {
    bool need_rex = is_extended(dst) || is_extended(src.index) || is_extended(src.base) || reg_id(dst) >= 4;
    if (need_rex) {
        emit_rex_force(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    }
    buffer_.emit8(0x8A);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::mov8(const MemAddress& dst, GPR src) {
    bool need_rex = is_extended(src) || is_extended(dst.index) || is_extended(dst.base) || reg_id(src) >= 4;
    if (need_rex) {
        emit_rex_force(false, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    }
    buffer_.emit8(0x88);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::mov8(const MemAddress& dst, uint8_t imm) {
    bool need_rex = is_extended(dst.index) || is_extended(dst.base);
    if (need_rex) {
        emit_rex_force(false, false, is_extended(dst.index), is_extended(dst.base));
    }
    buffer_.emit8(0xC6);
    emit_mem_operand(0, dst);
    buffer_.emit8(imm);
}

void X64Encoder::mov16(GPR dst, GPR src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(src), false, is_extended(dst));
    buffer_.emit8(0x89);
    emit_modrm(3, reg_code(src), reg_code(dst));
}

void X64Encoder::mov16(GPR dst, uint16_t imm) {
    buffer_.emit8(0x66);
    emit_rex(false, false, false, is_extended(dst));
    buffer_.emit8(static_cast<uint8_t>(0xB8 + reg_code(dst)));
    buffer_.emit16(imm);
}

void X64Encoder::mov16(GPR dst, const MemAddress& src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x8B);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::mov16(const MemAddress& dst, GPR src) {
    buffer_.emit8(0x66);
    emit_rex(false, is_extended(src), is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0x89);
    emit_mem_operand(reg_code(src), dst);
}

void X64Encoder::mov16(const MemAddress& dst, uint16_t imm) {
    buffer_.emit8(0x66);
    emit_rex(false, false, is_extended(dst.index), is_extended(dst.base));
    buffer_.emit8(0xC7);
    emit_mem_operand(0, dst);
    buffer_.emit16(imm);
}

// Sign & Zero Extensions
void X64Encoder::movsxd(GPR dst, GPR src) {
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x63);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movsxd(GPR dst, const MemAddress& src) {
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x63);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movzx8(GPR dst, GPR src) {
    bool need_rex = is_extended(dst) || is_extended(src) || reg_id(src) >= 4;
    if (need_rex) {
        emit_rex_force(false, is_extended(dst), false, is_extended(src));
    }
    buffer_.emit8(0x0F);
    buffer_.emit8(0xB6);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movzx8(GPR dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xB6);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movzx16(GPR dst, GPR src) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xB7);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movzx16(GPR dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xB7);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movsx8(GPR dst, GPR src) {
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBE);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movsx8(GPR dst, const MemAddress& src) {
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBE);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movsx8_32(GPR dst, GPR src) {
    bool need_rex = is_extended(dst) || is_extended(src) || reg_id(src) >= 4;
    if (need_rex) {
        emit_rex_force(false, is_extended(dst), false, is_extended(src));
    }
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBE);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movsx8_32(GPR dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBE);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movsx16(GPR dst, GPR src) {
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBF);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movsx16(GPR dst, const MemAddress& src) {
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBF);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::movsx16_32(GPR dst, GPR src) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBF);
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::movsx16_32(GPR dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(0xBF);
    emit_mem_operand(reg_code(dst), src);
}

// =========================================================================
// CONTROL FLOW & BRANCHES
// =========================================================================

void X64Encoder::jmp(int32_t rel32_disp) {
    buffer_.emit8(0xE9);
    buffer_.emit32(static_cast<uint32_t>(rel32_disp));
}

void X64Encoder::jmp_rel8(int8_t disp) {
    buffer_.emit8(0xEB);
    buffer_.emit8(static_cast<uint8_t>(disp));
}

void X64Encoder::jmp(Label label) {
    jmp_near(label);
}

void X64Encoder::jmp_short(Label label) {
    buffer_.emit8(0xEB);
    size_t patch_off = buffer_.size();
    buffer_.emit8(0x00);
    buffer_.record_label_fixup(label, patch_off, FixupKind::Rel8);
}

void X64Encoder::jmp_near(Label label) {
    buffer_.emit8(0xE9);
    size_t patch_off = buffer_.size();
    buffer_.emit32(0x00000000);
    buffer_.record_label_fixup(label, patch_off, FixupKind::Rel32);
}

void X64Encoder::jmp(GPR target) {
    emit_rex(false, false, false, is_extended(target));
    buffer_.emit8(0xFF);
    emit_modrm(3, 4, reg_code(target));
}

void X64Encoder::jmp(const MemAddress& target) {
    emit_rex(false, false, is_extended(target.index), is_extended(target.base));
    buffer_.emit8(0xFF);
    emit_mem_operand(4, target);
}

void X64Encoder::j(Condition cond, int32_t rel32_disp) {
    buffer_.emit8(0x0F);
    buffer_.emit8(static_cast<uint8_t>(0x80 + static_cast<uint8_t>(cond)));
    buffer_.emit32(static_cast<uint32_t>(rel32_disp));
}

void X64Encoder::j_rel8(Condition cond, int8_t disp) {
    buffer_.emit8(static_cast<uint8_t>(0x70 + static_cast<uint8_t>(cond)));
    buffer_.emit8(static_cast<uint8_t>(disp));
}

void X64Encoder::j(Condition cond, Label label) {
    j_near(cond, label);
}

void X64Encoder::j_short(Condition cond, Label label) {
    buffer_.emit8(static_cast<uint8_t>(0x70 + static_cast<uint8_t>(cond)));
    size_t patch_off = buffer_.size();
    buffer_.emit8(0x00);
    buffer_.record_label_fixup(label, patch_off, FixupKind::Rel8);
}

void X64Encoder::j_near(Condition cond, Label label) {
    buffer_.emit8(0x0F);
    buffer_.emit8(static_cast<uint8_t>(0x80 + static_cast<uint8_t>(cond)));
    size_t patch_off = buffer_.size();
    buffer_.emit32(0x00000000);
    buffer_.record_label_fixup(label, patch_off, FixupKind::Rel32);
}

// SETcc
void X64Encoder::setcc(Condition cond, GPR dst8) {
    bool need_rex = is_extended(dst8) || reg_id(dst8) >= 4;
    if (need_rex) {
        emit_rex_force(false, false, false, is_extended(dst8));
    }
    buffer_.emit8(0x0F);
    buffer_.emit8(static_cast<uint8_t>(0x90 + static_cast<uint8_t>(cond)));
    emit_modrm(3, 0, reg_code(dst8));
}

void X64Encoder::setcc(Condition cond, const MemAddress& dst8) {
    bool need_rex = is_extended(dst8.index) || is_extended(dst8.base);
    if (need_rex) {
        emit_rex(false, false, is_extended(dst8.index), is_extended(dst8.base));
    }
    buffer_.emit8(0x0F);
    buffer_.emit8(static_cast<uint8_t>(0x90 + static_cast<uint8_t>(cond)));
    emit_mem_operand(0, dst8);
}

// CMOVcc
void X64Encoder::cmovcc(Condition cond, GPR dst, GPR src) {
    emit_rex(true, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(static_cast<uint8_t>(0x40 + static_cast<uint8_t>(cond)));
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cmovcc32(Condition cond, GPR dst, GPR src) {
    emit_rex(false, is_extended(dst), false, is_extended(src));
    buffer_.emit8(0x0F);
    buffer_.emit8(static_cast<uint8_t>(0x40 + static_cast<uint8_t>(cond)));
    emit_modrm(3, reg_code(dst), reg_code(src));
}

void X64Encoder::cmovcc(Condition cond, GPR dst, const MemAddress& src) {
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(static_cast<uint8_t>(0x40 + static_cast<uint8_t>(cond)));
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::cmovcc32(Condition cond, GPR dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x0F);
    buffer_.emit8(static_cast<uint8_t>(0x40 + static_cast<uint8_t>(cond)));
    emit_mem_operand(reg_code(dst), src);
}

// CALL & RET
void X64Encoder::call(int32_t rel32_disp) {
    buffer_.emit8(0xE8);
    buffer_.emit32(static_cast<uint32_t>(rel32_disp));
}

void X64Encoder::call(Label label) {
    buffer_.emit8(0xE8);
    size_t patch_off = buffer_.size();
    buffer_.emit32(0x00000000);
    buffer_.record_label_fixup(label, patch_off, FixupKind::Rel32);
}

void X64Encoder::call(const std::string& symbol) {
    buffer_.emit8(0xE8);
    size_t patch_off = buffer_.size();
    buffer_.emit32(0x00000000);
    buffer_.add_relocation(patch_off, RelocationKind::PCRel32, symbol, -4);
}

void X64Encoder::call(GPR target) {
    emit_rex(false, false, false, is_extended(target));
    buffer_.emit8(0xFF);
    emit_modrm(3, 2, reg_code(target));
}

void X64Encoder::call(const MemAddress& target) {
    emit_rex(false, false, is_extended(target.index), is_extended(target.base));
    buffer_.emit8(0xFF);
    emit_mem_operand(2, target);
}

void X64Encoder::ret() {
    buffer_.emit8(0xC3);
}

void X64Encoder::ret(uint16_t imm) {
    buffer_.emit8(0xC2);
    buffer_.emit16(imm);
}

// =========================================================================
// STACK & FRAME OPERATIONS
// =========================================================================

void X64Encoder::push(GPR reg) {
    if (is_extended(reg)) {
        emit_rex(false, false, false, true);
    }
    buffer_.emit8(static_cast<uint8_t>(0x50 + reg_code(reg)));
}

void X64Encoder::push(int32_t imm) {
    if (imm >= -128 && imm <= 127) {
        buffer_.emit8(0x6A);
        buffer_.emit8(static_cast<uint8_t>(static_cast<int8_t>(imm)));
    } else {
        buffer_.emit8(0x68);
        buffer_.emit32(static_cast<uint32_t>(imm));
    }
}

void X64Encoder::push(const MemAddress& mem) {
    emit_rex(false, false, is_extended(mem.index), is_extended(mem.base));
    buffer_.emit8(0xFF);
    emit_mem_operand(6, mem);
}

void X64Encoder::pop(GPR reg) {
    if (is_extended(reg)) {
        emit_rex(false, false, false, true);
    }
    buffer_.emit8(static_cast<uint8_t>(0x58 + reg_code(reg)));
}

void X64Encoder::pop(const MemAddress& mem) {
    emit_rex(false, false, is_extended(mem.index), is_extended(mem.base));
    buffer_.emit8(0x8F);
    emit_mem_operand(0, mem);
}

void X64Encoder::lea(GPR dst, const MemAddress& src) {
    emit_rex(true, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x8D);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::lea32(GPR dst, const MemAddress& src) {
    emit_rex(false, is_extended(dst), is_extended(src.index), is_extended(src.base));
    buffer_.emit8(0x8D);
    emit_mem_operand(reg_code(dst), src);
}

void X64Encoder::int3() {
    buffer_.emit8(0xCC);
}

void X64Encoder::ud2() {
    buffer_.emit8(0x0F);
    buffer_.emit8(0x0B);
}

void X64Encoder::cqo() {
    buffer_.emit8(0x48);
    buffer_.emit8(0x99);
}

void X64Encoder::cdq() {
    buffer_.emit8(0x99);
}

void X64Encoder::nop() {
    buffer_.emit8(0x90);
}

void X64Encoder::nop(size_t count) {
    buffer_.emit_nops(count);
}

} // namespace brass::x64
