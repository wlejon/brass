#pragma once

#include <cstdint>

namespace brass::object {

// What a relocation patches and how. The x64 kinds each describe a data
// field; the AArch64 instruction kinds each describe one instruction form,
// because the immediate an AArch64 instruction carries (and its scaling)
// depends on the instruction, and every object format and linker has to
// know which form it is looking at.
enum class RelocKind : uint8_t {
    PCRel32,   // 32-bit PC-relative displacement (data, or an x64 disp32)
    Abs64,     // 64-bit absolute VA
    SecRel32,  // 32-bit section-relative offset (debug info)
    Addr32NB,  // 32-bit RVA without base (COFF .pdata / .xdata)
    Plt32,     // x64: 32-bit call displacement; AArch64: B/BL imm26
    Abs32,     // 32-bit absolute VA (ELF)
    SecIdx,    // 16-bit section index (COFF debug relocation IMAGE_REL_AMD64_SECTION)
    // 32-bit PC-relative displacement to the slot holding the symbol's
    // address (x64 `mov r64, [rip + disp32]`, addend -4): a GOT entry on
    // ELF and Mach-O, an IAT entry on PE, a slot the JIT owns in-process.
    // Every address a function materialises is emitted this way; the one
    // against a symbol the same object defines is relaxed to a `lea` of the
    // symbol before the image is laid out (relax_got_loads), so only
    // imports go through a slot.
    GotPCRel32,

    // ---- AArch64 instruction relocations --------------------------------
    // ADRP: Page(S + A) - Page(P), +-4 GB.
    AdrPage21,
    // ADD (immediate, 12-bit, unshifted): (S + A) & 0xFFF.
    AddLo12,
    // LDR/STR (unsigned immediate): ((S + A) & 0xFFF) >> log2(access size).
    // The access size is part of the kind: the object formats that encode
    // it (ELF) need it, and a linker checks it against the instruction.
    LdSt8Lo12,
    LdSt16Lo12,
    LdSt32Lo12,
    LdSt64Lo12,
    LdSt128Lo12,
    // ADRP of the page holding the symbol's GOT slot (ELF ADR_GOT_PAGE,
    // Mach-O GOT_LOAD_PAGE21), and the `ldr x, [x, #lo12]` of the slot
    // itself (ELF LD64_GOT_LO12_NC, Mach-O GOT_LOAD_PAGEOFF12). The AArch64
    // counterpart of GotPCRel32: every address a function materialises is
    // emitted as this pair, and the pair against a defined symbol is relaxed
    // to ADRP + ADD (relax_got_loads).
    GotPage21,
    GotLo12,
};

} // namespace brass::object
