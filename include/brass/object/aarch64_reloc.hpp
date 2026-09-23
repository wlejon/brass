#pragma once

#include <brass/object/reloc_kind.hpp>
#include <cstdint>
#include <string>

// The one place an AArch64 instruction relocation is applied: the JIT loader
// and every image writer (PE, ELF .so, Mach-O dylib) resolve through it, so
// page-relative ADRP arithmetic, access-size scaling of LDR/STR offsets and
// the range and alignment checks are written once.
namespace brass::object::a64 {

// Instruction-form predicates.
bool is_adrp(uint32_t inst) noexcept;
bool is_add_imm12(uint32_t inst) noexcept;       // ADD (immediate), unshifted, 32- or 64-bit
bool is_branch26(uint32_t inst) noexcept;        // B or BL
bool is_ldr_x_uimm(uint32_t inst) noexcept;      // LDR Xt, [Xn, #uimm12]
// log2 of the access size of a load/store (unsigned immediate), or -1 when
// `inst` is not one.
int ldst_uimm_scale(uint32_t inst) noexcept;

// log2 of the access size a LdSt*Lo12 / GotLo12 kind scales by, or -1.
int lo12_scale(RelocKind kind) noexcept;
bool is_instruction_kind(RelocKind kind) noexcept;   // AdrPage21 .. GotLo12
bool is_got_kind(RelocKind kind) noexcept;           // GotPage21, GotLo12

// Patches `inst` (the instruction at address `place`) for `kind`, where
// `value` is the resolved S + A (for the GOT kinds: the slot's address).
// Plt32 is the B/BL imm26 form. Returns an empty string on success, and on
// failure a description (out of range, misaligned, wrong instruction form):
// nothing is patched then, and the caller refuses the image.
std::string patch(RelocKind kind, uint32_t& inst, uint64_t place, uint64_t value);

// `ldr Xt, [Xn, #imm]` -> `add Xt, Xn, #imm`: the GOT-load relaxation for a
// symbol the image defines. False (and `inst` untouched) if `inst` is not an
// LDR X (unsigned immediate).
bool relax_got_ldr_to_add(uint32_t& inst) noexcept;

// Writes addend `addend` into the immediate field of `inst` for the object
// formats whose AArch64 relocations carry an implicit addend (COFF): the
// byte offset in ADRP's imm21, the byte offset in ADD's imm12, the scaled
// offset in LDR/STR's imm12. Returns false if the addend does not fit.
bool encode_implicit_addend(RelocKind kind, uint32_t& inst, int64_t addend) noexcept;

} // namespace brass::object::a64
