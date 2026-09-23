#include <brass/object/aarch64_reloc.hpp>

namespace brass::object::a64 {

namespace {

constexpr uint32_t kImm12Mask = 0xFFFu << 10;

std::string hex(uint64_t v) {
    static const char digits[] = "0123456789abcdef";
    std::string s;
    do {
        s.insert(s.begin(), digits[v & 0xF]);
        v >>= 4;
    } while (v != 0);
    return "0x" + s;
}

uint32_t with_adrp_imm(uint32_t inst, int64_t imm21) {
    const uint32_t bits = static_cast<uint32_t>(imm21) & 0x1FFFFFu;
    const uint32_t immlo = (bits & 0x3u) << 29;
    const uint32_t immhi = ((bits >> 2) & 0x7FFFFu) << 5;
    return (inst & 0x9F00001Fu) | immlo | immhi;
}

} // namespace

bool is_adrp(uint32_t inst) noexcept {
    return (inst & 0x9F000000u) == 0x90000000u;
}

bool is_add_imm12(uint32_t inst) noexcept {
    // sf | op=0 | S=0 | 100010 | sh=0
    return (inst & 0x7FC00000u) == 0x11000000u;
}

bool is_branch26(uint32_t inst) noexcept {
    return (inst & 0x7C000000u) == 0x14000000u;
}

bool is_ldr_x_uimm(uint32_t inst) noexcept {
    return (inst & 0xFFC00000u) == 0xF9400000u;
}

int ldst_uimm_scale(uint32_t inst) noexcept {
    // size | 111 | V | 01 | opc | imm12 | Rn | Rt
    if ((inst & 0x3B000000u) != 0x39000000u) return -1;
    const int size = static_cast<int>(inst >> 30);
    const bool simd = (inst & (1u << 26)) != 0;
    const bool opc_hi = (inst & (1u << 23)) != 0;
    if (simd && size == 0 && opc_hi) return 4;   // LDR/STR Qt
    return size;
}

int lo12_scale(RelocKind kind) noexcept {
    switch (kind) {
        case RelocKind::LdSt8Lo12:   return 0;
        case RelocKind::LdSt16Lo12:  return 1;
        case RelocKind::LdSt32Lo12:  return 2;
        case RelocKind::LdSt64Lo12:  return 3;
        case RelocKind::GotLo12:     return 3;
        case RelocKind::LdSt128Lo12: return 4;
        default:                     return -1;
    }
}

bool is_instruction_kind(RelocKind kind) noexcept {
    switch (kind) {
        case RelocKind::AdrPage21:
        case RelocKind::AddLo12:
        case RelocKind::LdSt8Lo12:
        case RelocKind::LdSt16Lo12:
        case RelocKind::LdSt32Lo12:
        case RelocKind::LdSt64Lo12:
        case RelocKind::LdSt128Lo12:
        case RelocKind::GotPage21:
        case RelocKind::GotLo12:
            return true;
        default:
            return false;
    }
}

bool is_got_kind(RelocKind kind) noexcept {
    return kind == RelocKind::GotPage21 || kind == RelocKind::GotLo12;
}

std::string patch(RelocKind kind, uint32_t& inst, uint64_t place, uint64_t value) {
    switch (kind) {
        case RelocKind::Plt32: {
            if (!is_branch26(inst)) return "branch relocation on an instruction that is not B/BL";
            const int64_t disp = static_cast<int64_t>(value - place);
            if ((disp & 3) != 0) return "branch target " + hex(value) + " is not 4-byte aligned";
            if (disp < -(int64_t(1) << 27) || disp >= (int64_t(1) << 27)) {
                return "branch target " + hex(value) + " is out of the +-128 MB range of " + hex(place);
            }
            inst = (inst & 0xFC000000u) | (static_cast<uint32_t>(disp >> 2) & 0x03FFFFFFu);
            return {};
        }
        case RelocKind::AdrPage21:
        case RelocKind::GotPage21: {
            if (!is_adrp(inst)) return "page relocation on an instruction that is not ADRP";
            // Page arithmetic, not byte arithmetic: the pages of the target
            // and of the ADRP itself, each rounded down.
            const int64_t pages = static_cast<int64_t>((value >> 12) - (place >> 12));
            if (pages < -(int64_t(1) << 20) || pages >= (int64_t(1) << 20)) {
                return "ADRP target " + hex(value) + " is out of the +-4 GB range of " + hex(place);
            }
            inst = with_adrp_imm(inst, pages);
            return {};
        }
        case RelocKind::AddLo12: {
            if (!is_add_imm12(inst)) return "ADD page-offset relocation on an instruction that is not ADD (immediate)";
            inst = (inst & ~kImm12Mask) | (static_cast<uint32_t>(value & 0xFFFu) << 10);
            return {};
        }
        case RelocKind::LdSt8Lo12:
        case RelocKind::LdSt16Lo12:
        case RelocKind::LdSt32Lo12:
        case RelocKind::LdSt64Lo12:
        case RelocKind::LdSt128Lo12:
        case RelocKind::GotLo12: {
            const int want = lo12_scale(kind);
            if (kind == RelocKind::GotLo12 && !is_ldr_x_uimm(inst)) {
                return "GOT page-offset relocation on an instruction that is not LDR Xt, [Xn, #imm]";
            }
            const int have = ldst_uimm_scale(inst);
            if (have < 0) return "load/store page-offset relocation on an instruction that is not LDR/STR (unsigned immediate)";
            if (have != want) {
                return "load/store page-offset relocation scaled for " + std::to_string(1 << want) +
                       "-byte access on a " + std::to_string(1 << have) + "-byte load/store";
            }
            const uint64_t lo12 = value & 0xFFFu;
            if ((lo12 & ((uint64_t(1) << want) - 1)) != 0) {
                return "target " + hex(value) + " is not aligned to the " + std::to_string(1 << want) +
                       "-byte access of its load/store";
            }
            inst = (inst & ~kImm12Mask) | (static_cast<uint32_t>(lo12 >> want) << 10);
            return {};
        }
        default:
            return "not an AArch64 instruction relocation";
    }
}

bool relax_got_ldr_to_add(uint32_t& inst) noexcept {
    if (!is_ldr_x_uimm(inst)) return false;
    const uint32_t rn = (inst >> 5) & 0x1Fu;
    const uint32_t rt = inst & 0x1Fu;
    const uint32_t imm12 = (inst >> 10) & 0xFFFu;
    // The scaled LDR offset becomes the byte offset ADD takes.
    const uint32_t bytes = imm12 << 3;
    if (bytes > 0xFFFu) return false;
    inst = 0x91000000u | (bytes << 10) | (rn << 5) | rt;
    return true;
}

bool encode_implicit_addend(RelocKind kind, uint32_t& inst, int64_t addend) noexcept {
    if (addend == 0) return true;
    switch (kind) {
        case RelocKind::AdrPage21:
            if (!is_adrp(inst) || addend < -(int64_t(1) << 20) || addend >= (int64_t(1) << 20)) return false;
            inst = with_adrp_imm(inst, addend);
            return true;
        case RelocKind::AddLo12:
            if (!is_add_imm12(inst) || addend < 0 || addend > 0xFFF) return false;
            inst = (inst & ~kImm12Mask) | (static_cast<uint32_t>(addend) << 10);
            return true;
        case RelocKind::LdSt8Lo12:
        case RelocKind::LdSt16Lo12:
        case RelocKind::LdSt32Lo12:
        case RelocKind::LdSt64Lo12:
        case RelocKind::LdSt128Lo12: {
            const int scale = lo12_scale(kind);
            if (ldst_uimm_scale(inst) != scale || addend < 0) return false;
            if ((addend & ((int64_t(1) << scale) - 1)) != 0 || (addend >> scale) > 0xFFF) return false;
            inst = (inst & ~kImm12Mask) | (static_cast<uint32_t>(addend >> scale) << 10);
            return true;
        }
        default:
            return false;
    }
}

} // namespace brass::object::a64
