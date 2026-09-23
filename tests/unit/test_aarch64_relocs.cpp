#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/aarch64_reloc.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include <cstring>
#include <string>
#include <vector>

// AArch64 relocations end to end: the one patch routine (a64::patch), the
// page-offset kinds split by instruction form and access size, the GOT pair,
// and how each object format writes them.

using namespace brass;
using namespace brass::object;

namespace {

uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t rd64(const uint8_t* p) { return rd32(p) | (static_cast<uint64_t>(rd32(p + 4)) << 32); }

constexpr uint32_t kAdrpX0 = 0x90000000u;      // adrp x0, #0
constexpr uint32_t kAddX0 = 0x91000000u;       // add x0, x0, #0
constexpr uint32_t kLdrX1 = 0xF9400021u;       // ldr x1, [x1]
constexpr uint32_t kLdrW1 = 0xB9400021u;       // ldr w1, [x1]
constexpr uint32_t kLdrbW1 = 0x39400021u;      // ldrb w1, [x1]
constexpr uint32_t kLdrhW1 = 0x79400021u;      // ldrh w1, [x1]
constexpr uint32_t kLdrQ1 = 0x3DC00021u;       // ldr q1, [x1]
constexpr uint32_t kStrX1 = 0xF9000021u;       // str x1, [x1]
constexpr uint32_t kBl = 0x94000000u;          // bl .

int64_t adrp_pages(uint32_t inst) {
    const uint32_t immlo = (inst >> 29) & 3u;
    const uint32_t immhi = (inst >> 5) & 0x7FFFFu;
    int64_t imm = static_cast<int64_t>((immhi << 2) | immlo);
    if (imm & (int64_t(1) << 20)) imm -= int64_t(1) << 21;
    return imm;
}
uint32_t imm12(uint32_t inst) { return (inst >> 10) & 0xFFFu; }

// A one-function aarch64 object plus the given instruction words appended to
// .text; returns the text offset of the first appended word.
ObjectFile object_with_code(const Target& target, const std::vector<uint32_t>& words, size_t& base) {
    Module mod("a64_reloc_mod");
    Function* fn = mod.create_function("anchor_fn", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    b.build_ret(x);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    ObjectFile obj = compile_module_to_object(mod, target);
    Section* text = obj.get_section(".text");
    REQUIRE(text != nullptr);
    text->align_to(4);
    base = text->data.size();
    for (uint32_t w : words) text->emit32(w);
    return obj;
}

void add_reloc(ObjectFile& obj, std::string_view sec, size_t off, RelocKind kind, std::string sym,
               int64_t addend = 0) {
    Section* s = obj.get_section(sec);
    REQUIRE(s != nullptr);
    s->relocations.push_back({off, kind, std::move(sym), addend, 0});
}

void add_data_symbol(ObjectFile& obj, const std::string& name, size_t bytes,
                     SymbolBinding binding = SymbolBinding::Global) {
    Section& data = obj.get_or_create_section(".data", SectionKind::Data,
                                              SectionFlags::Read | SectionFlags::Write | SectionFlags::Alloc, 16);
    data.align_to(16);
    ObjectSymbol sym;
    sym.name = name;
    sym.section_index = obj.get_section_index(".data");
    sym.value = data.data.size();
    sym.size = bytes;
    sym.type = SymbolType::Object;
    sym.binding = binding;
    obj.add_symbol(sym);
    for (size_t i = 0; i < bytes; ++i) data.emit8(0);
}

struct ElfSec {
    std::string name;
    uint32_t type = 0;
    uint64_t flags = 0;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint32_t info = 0;
};

std::vector<ElfSec> elf_sections(const std::vector<uint8_t>& f) {
    const uint64_t shoff = rd64(f.data() + 0x28);
    const uint16_t shnum = rd16(f.data() + 0x3C);
    const uint16_t shstrndx = rd16(f.data() + 0x3E);
    const uint8_t* strhdr = f.data() + shoff + uint64_t(shstrndx) * 64;
    const uint8_t* strtab = f.data() + rd64(strhdr + 24);
    std::vector<ElfSec> out;
    for (uint16_t i = 0; i < shnum; ++i) {
        const uint8_t* h = f.data() + shoff + uint64_t(i) * 64;
        ElfSec s;
        s.name = reinterpret_cast<const char*>(strtab + rd32(h));
        s.type = rd32(h + 4);
        s.flags = rd64(h + 8);
        s.offset = rd64(h + 24);
        s.size = rd64(h + 32);
        s.info = rd32(h + 44);
        out.push_back(s);
    }
    return out;
}

// Relocation types at each offset of .rela.text.
std::vector<std::pair<uint64_t, uint32_t>> elf_text_relocs(const std::vector<uint8_t>& f) {
    std::vector<std::pair<uint64_t, uint32_t>> out;
    for (const auto& s : elf_sections(f)) {
        if (s.name != ".rela.text") continue;
        for (uint64_t off = 0; off + 24 <= s.size; off += 24) {
            const uint8_t* r = f.data() + s.offset + off;
            out.emplace_back(rd64(r), static_cast<uint32_t>(rd64(r + 8) & 0xFFFFFFFFu));
        }
    }
    return out;
}

uint32_t type_at(const std::vector<std::pair<uint64_t, uint32_t>>& relocs, uint64_t off) {
    for (const auto& [o, t] : relocs) {
        if (o == off) return t;
    }
    return 0xFFFFFFFFu;
}

} // namespace

// =============================================================================
// a64::patch
// =============================================================================

TEST_CASE("AArch64 reloc - ADRP is page arithmetic, not byte arithmetic") {
    uint32_t inst = kAdrpX0;
    // place is at the last word of page 1, target on page 3: two pages, even
    // though the byte distance is barely over one page.
    CHECK_EQ(a64::patch(RelocKind::AdrPage21, inst, 0x1FFC, 0x3000), std::string());
    CHECK_EQ(adrp_pages(inst), int64_t(2));
    CHECK_EQ(inst & 0x9F00001Fu, kAdrpX0);  // opcode and Rd kept

    inst = kAdrpX0;
    CHECK_EQ(a64::patch(RelocKind::AdrPage21, inst, 0x5000, 0x1FFF), std::string());
    CHECK_EQ(adrp_pages(inst), int64_t(-4));

    // Out of the +-4 GB reach.
    inst = kAdrpX0;
    CHECK_NE(a64::patch(RelocKind::AdrPage21, inst, 0, uint64_t(1) << 33), std::string());
    CHECK_EQ(inst, kAdrpX0);  // untouched on failure

    // Not an ADRP.
    inst = kAddX0;
    CHECK_NE(a64::patch(RelocKind::AdrPage21, inst, 0, 0x1000), std::string());
}

TEST_CASE("AArch64 reloc - page offsets: ADD is unscaled, loads/stores scale by access size") {
    uint32_t add = kAddX0;
    CHECK_EQ(a64::patch(RelocKind::AddLo12, add, 0, 0x12345), std::string());
    CHECK_EQ(imm12(add), 0x345u);

    uint32_t ldr = kLdrX1;
    CHECK_EQ(a64::patch(RelocKind::LdSt64Lo12, ldr, 0, 0x12348), std::string());
    CHECK_EQ(imm12(ldr), 0x348u >> 3);

    uint32_t str = kStrX1;
    CHECK_EQ(a64::patch(RelocKind::LdSt64Lo12, str, 0, 0x7FF8), std::string());
    CHECK_EQ(imm12(str), 0xFF8u >> 3);

    uint32_t ldrw = kLdrW1;
    CHECK_EQ(a64::patch(RelocKind::LdSt32Lo12, ldrw, 0, 0x1ABC), std::string());
    CHECK_EQ(imm12(ldrw), 0xABCu >> 2);

    uint32_t ldrh = kLdrhW1;
    CHECK_EQ(a64::patch(RelocKind::LdSt16Lo12, ldrh, 0, 0x1ABE), std::string());
    CHECK_EQ(imm12(ldrh), 0xABEu >> 1);

    uint32_t ldrb = kLdrbW1;
    CHECK_EQ(a64::patch(RelocKind::LdSt8Lo12, ldrb, 0, 0x1ABF), std::string());
    CHECK_EQ(imm12(ldrb), 0xABFu);

    uint32_t ldrq = kLdrQ1;
    CHECK_EQ(a64::patch(RelocKind::LdSt128Lo12, ldrq, 0, 0x1AB0), std::string());
    CHECK_EQ(imm12(ldrq), 0xAB0u >> 4);

    // Misaligned for the access size.
    uint32_t bad = kLdrX1;
    CHECK_NE(a64::patch(RelocKind::LdSt64Lo12, bad, 0, 0x1234), std::string());
    CHECK_EQ(bad, kLdrX1);
    // Kind says 4-byte access, instruction is an 8-byte load.
    bad = kLdrX1;
    CHECK_NE(a64::patch(RelocKind::LdSt32Lo12, bad, 0, 0x1000), std::string());
    // A load/store kind on an ADD, and ADD's kind on a load.
    bad = kAddX0;
    CHECK_NE(a64::patch(RelocKind::LdSt64Lo12, bad, 0, 0x1000), std::string());
    bad = kLdrX1;
    CHECK_NE(a64::patch(RelocKind::AddLo12, bad, 0, 0x1000), std::string());
}

TEST_CASE("AArch64 reloc - branch26 range, alignment and GOT load form") {
    uint32_t bl = kBl;
    CHECK_EQ(a64::patch(RelocKind::Plt32, bl, 0x10000, 0x10010), std::string());
    CHECK_EQ(bl, 0x94000004u);
    bl = kBl;
    CHECK_NE(a64::patch(RelocKind::Plt32, bl, 0, uint64_t(1) << 28), std::string());
    bl = kBl;
    CHECK_NE(a64::patch(RelocKind::Plt32, bl, 0, 0x102), std::string());

    uint32_t got = kLdrX1;
    CHECK_EQ(a64::patch(RelocKind::GotLo12, got, 0, 0x4010), std::string());
    CHECK_EQ(imm12(got), 0x010u >> 3);
    got = kLdrW1;  // a GOT slot is 8 bytes: only LDR X
    CHECK_NE(a64::patch(RelocKind::GotLo12, got, 0, 0x4010), std::string());

    uint32_t relaxed = 0xF9400821u;  // ldr x1, [x1, #16]
    CHECK(a64::relax_got_ldr_to_add(relaxed));
    CHECK_EQ(relaxed, 0x91004021u);  // add x1, x1, #16
    uint32_t not_ldr = kAddX0;
    CHECK_FALSE(a64::relax_got_ldr_to_add(not_ldr));
}

// =============================================================================
// Object formats
// =============================================================================

TEST_CASE("AArch64 reloc - ELF object: one relocation type per instruction form, GOT pair, GNU-stack note") {
    size_t base = 0;
    ObjectFile obj = object_with_code(Target::aarch64_linux(),
        {kAdrpX0, kAddX0, kLdrbW1, kLdrhW1, kLdrW1, kLdrX1, kLdrQ1, kAdrpX0, kLdrX1}, base);
    add_data_symbol(obj, "elf_data", 64);
    add_reloc(obj, ".text", base + 0, RelocKind::AdrPage21, "elf_data");
    add_reloc(obj, ".text", base + 4, RelocKind::AddLo12, "elf_data");
    add_reloc(obj, ".text", base + 8, RelocKind::LdSt8Lo12, "elf_data");
    add_reloc(obj, ".text", base + 12, RelocKind::LdSt16Lo12, "elf_data");
    add_reloc(obj, ".text", base + 16, RelocKind::LdSt32Lo12, "elf_data");
    add_reloc(obj, ".text", base + 20, RelocKind::LdSt64Lo12, "elf_data");
    add_reloc(obj, ".text", base + 24, RelocKind::LdSt128Lo12, "elf_data");
    add_reloc(obj, ".text", base + 28, RelocKind::GotPage21, "external_thing");
    add_reloc(obj, ".text", base + 32, RelocKind::GotLo12, "external_thing");

    std::vector<uint8_t> f = emit_elf_object(obj);
    REQUIRE(f.size() > 64);
    auto relocs = elf_text_relocs(f);
    CHECK_EQ(type_at(relocs, base + 0), elf::R_AARCH64_ADR_PREL_PG_HI21);
    CHECK_EQ(type_at(relocs, base + 4), elf::R_AARCH64_ADD_ABS_LO12_NC);
    CHECK_EQ(type_at(relocs, base + 8), elf::R_AARCH64_LDST8_ABS_LO12_NC);
    CHECK_EQ(type_at(relocs, base + 12), elf::R_AARCH64_LDST16_ABS_LO12_NC);
    CHECK_EQ(type_at(relocs, base + 16), elf::R_AARCH64_LDST32_ABS_LO12_NC);
    CHECK_EQ(type_at(relocs, base + 20), elf::R_AARCH64_LDST64_ABS_LO12_NC);
    CHECK_EQ(type_at(relocs, base + 24), elf::R_AARCH64_LDST128_ABS_LO12_NC);
    CHECK_EQ(type_at(relocs, base + 28), elf::R_AARCH64_ADR_GOT_PAGE);
    CHECK_EQ(type_at(relocs, base + 32), elf::R_AARCH64_LD64_GOT_LO12_NC);

    bool gnu_stack = false;
    for (const auto& s : elf_sections(f)) {
        if (s.name == ".note.GNU-stack") {
            gnu_stack = true;
            CHECK_EQ(s.size, uint64_t(0));
            CHECK_EQ(s.flags, uint64_t(0));  // no SHF_EXECINSTR: non-executable stack
        }
    }
    CHECK(gnu_stack);
}

TEST_CASE("AArch64 reloc - ELF object: GOT pair against a defined global symbol stays a GOT pair") {
    // A global symbol is preemptible in a shared library, where ADRP of it
    // does not link; the system linker relaxes the GOT pair when it can.
    size_t base = 0;
    ObjectFile obj = object_with_code(Target::aarch64_linux(), {kAdrpX0, 0xF9400800u /* ldr x0, [x0, #16] */}, base);
    add_data_symbol(obj, "global_data", 16);
    add_reloc(obj, ".text", base + 0, RelocKind::GotPage21, "global_data");
    add_reloc(obj, ".text", base + 4, RelocKind::GotLo12, "global_data");

    std::vector<uint8_t> f = emit_elf_object(obj);
    auto relocs = elf_text_relocs(f);
    CHECK_EQ(type_at(relocs, base + 0), elf::R_AARCH64_ADR_GOT_PAGE);
    CHECK_EQ(type_at(relocs, base + 4), elf::R_AARCH64_LD64_GOT_LO12_NC);
}

TEST_CASE("AArch64 reloc - ELF object: GOT pair against a defined local symbol relaxes to ADRP + ADD") {
    size_t base = 0;
    ObjectFile obj = object_with_code(Target::aarch64_linux(), {kAdrpX0, 0xF9400800u /* ldr x0, [x0, #16] */}, base);
    add_data_symbol(obj, "local_data", 16, SymbolBinding::Local);
    add_reloc(obj, ".text", base + 0, RelocKind::GotPage21, "local_data");
    add_reloc(obj, ".text", base + 4, RelocKind::GotLo12, "local_data");

    std::vector<uint8_t> f = emit_elf_object(obj);
    auto relocs = elf_text_relocs(f);
    CHECK_EQ(type_at(relocs, base + 0), elf::R_AARCH64_ADR_PREL_PG_HI21);
    CHECK_EQ(type_at(relocs, base + 4), elf::R_AARCH64_ADD_ABS_LO12_NC);
    // The instruction itself became ADD x0, x0, #16.
    bool found = false;
    for (const auto& s : elf_sections(f)) {
        if (s.name == ".text") {
            CHECK_EQ(rd32(f.data() + s.offset + base + 4), 0x91004000u);
            found = true;
        }
    }
    CHECK(found);

    // A GOT page-offset relocation on anything but an LDR X is malformed.
    size_t base2 = 0;
    ObjectFile bad = object_with_code(Target::aarch64_linux(), {kAdrpX0, kAddX0}, base2);
    add_data_symbol(bad, "local_data", 16, SymbolBinding::Local);
    add_reloc(bad, ".text", base2 + 4, RelocKind::GotLo12, "local_data");
    bool threw = false;
    try {
        (void)emit_elf_object(bad);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("AArch64 reloc - COFF: 12A for ADD, 12L for loads, implicit addends, GOT through slots") {
    size_t base = 0;
    ObjectFile obj = object_with_code(Target::aarch64_windows(),
        {kAdrpX0, kAddX0, kLdrX1, kAdrpX0, kLdrX1}, base);
    add_data_symbol(obj, "coff_data", 64);
    add_reloc(obj, ".text", base + 0, RelocKind::AdrPage21, "coff_data", 24);
    add_reloc(obj, ".text", base + 4, RelocKind::AddLo12, "coff_data", 24);
    add_reloc(obj, ".text", base + 8, RelocKind::LdSt64Lo12, "coff_data", 24);
    add_reloc(obj, ".text", base + 12, RelocKind::GotPage21, "imported_thing");
    add_reloc(obj, ".text", base + 16, RelocKind::GotLo12, "imported_thing");

    std::vector<uint8_t> f = emit_coff_object(obj);
    REQUIRE(f.size() > 20);
    const uint16_t nsec = rd16(f.data() + 2);
    const uint32_t symoff = rd32(f.data() + 8);
    const uint32_t nsyms = rd32(f.data() + 12);
    const uint8_t* strtab = f.data() + symoff + nsyms * 18;

    auto sym_name = [&](uint32_t index) {
        const uint8_t* s = f.data() + symoff + index * 18;
        if (rd32(s) == 0) return std::string(reinterpret_cast<const char*>(strtab + rd32(s + 4)));
        char n[9] = {};
        std::memcpy(n, s, 8);
        return std::string(n);
    };

    bool text_found = false;
    for (uint16_t i = 0; i < nsec; ++i) {
        const uint8_t* sh = f.data() + 20 + i * 40;
        char name[9] = {};
        std::memcpy(name, sh, 8);
        if (std::string(name) != ".text") continue;
        text_found = true;
        const uint8_t* raw = f.data() + rd32(sh + 20);
        const uint8_t* rel = f.data() + rd32(sh + 24);
        const uint16_t nrel = rd16(sh + 32);
        auto type_of = [&](uint32_t off, std::string* sym = nullptr) -> int {
            for (uint16_t r = 0; r < nrel; ++r) {
                if (rd32(rel + r * 10) == off) {
                    if (sym) *sym = sym_name(rd32(rel + r * 10 + 4));
                    return rd16(rel + r * 10 + 8);
                }
            }
            return -1;
        };
        const uint32_t b = static_cast<uint32_t>(base);
        CHECK_EQ(type_of(b + 0), int(coff::IMAGE_REL_ARM64_PAGE21));
        CHECK_EQ(type_of(b + 4), int(coff::IMAGE_REL_ARM64_PAGEOFFSET_12A));
        CHECK_EQ(type_of(b + 8), int(coff::IMAGE_REL_ARM64_PAGEOFFSET_12L));
        // COFF ARM64 addends live in the instruction: ADRP carries the byte
        // addend in imm21, ADD in imm12, the 8-byte LDR scaled by 8.
        CHECK_EQ(adrp_pages(rd32(raw + b + 0)), int64_t(24));
        CHECK_EQ(imm12(rd32(raw + b + 4)), 24u);
        CHECK_EQ(imm12(rd32(raw + b + 8)), 3u);
        // The import goes through a slot: ADRP + LDR X against sym$got.
        std::string s1, s2;
        CHECK_EQ(type_of(b + 12, &s1), int(coff::IMAGE_REL_ARM64_PAGE21));
        CHECK_EQ(type_of(b + 16, &s2), int(coff::IMAGE_REL_ARM64_PAGEOFFSET_12L));
        CHECK_EQ(s1, std::string("imported_thing$got"));
        CHECK_EQ(s2, std::string("imported_thing$got"));
    }
    CHECK(text_found);

    // An addend that does not fit the instruction is refused, not truncated.
    size_t base2 = 0;
    ObjectFile bad = object_with_code(Target::aarch64_windows(), {kLdrX1}, base2);
    add_data_symbol(bad, "coff_data", 64);
    add_reloc(bad, ".text", base2, RelocKind::LdSt64Lo12, "coff_data", 4);  // not a multiple of 8
    bool threw = false;
    try {
        (void)emit_coff_object(bad);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("AArch64 reloc - Mach-O: PAGEOFF12 for every page offset, ADDEND entries, GOT_LOAD pair") {
    size_t base = 0;
    ObjectFile obj = object_with_code(Target::aarch64_macos(), {kAdrpX0, kLdrX1, kAdrpX0, kLdrX1}, base);
    add_data_symbol(obj, "macho_data", 64);
    add_reloc(obj, ".text", base + 0, RelocKind::AdrPage21, "macho_data", 16);
    add_reloc(obj, ".text", base + 4, RelocKind::LdSt64Lo12, "macho_data", 16);
    add_reloc(obj, ".text", base + 8, RelocKind::GotPage21, "dylib_thing");
    add_reloc(obj, ".text", base + 12, RelocKind::GotLo12, "dylib_thing");

    std::vector<uint8_t> f = emit_macho_object(obj);
    REQUIRE(f.size() > 32);
    const uint8_t* seg = f.data() + 32;
    const uint32_t nsects = rd32(seg + 64);

    bool text_found = false;
    for (uint32_t i = 0; i < nsects; ++i) {
        const uint8_t* sec = seg + 72 + i * 80;
        char name[17] = {};
        std::memcpy(name, sec, 16);
        if (std::string(name) != "__text") continue;
        text_found = true;
        const uint8_t* rel = f.data() + rd32(sec + 56);
        const uint32_t nrel = rd32(sec + 60);
        // (address, type, symbolnum/value) in file order.
        struct R { uint32_t addr; uint32_t type; uint32_t sym; };
        std::vector<R> rs;
        for (uint32_t r = 0; r < nrel; ++r) {
            const uint32_t info = rd32(rel + r * 8 + 4);
            rs.push_back({rd32(rel + r * 8), info >> 28, info & 0xFFFFFFu});
        }
        auto find = [&](uint32_t addr, uint32_t type) -> int {
            for (size_t k = 0; k < rs.size(); ++k) {
                if (rs[k].addr == addr && rs[k].type == type) return static_cast<int>(k);
            }
            return -1;
        };
        const uint32_t b = static_cast<uint32_t>(base);
        const int page = find(b + 0, macho::ARM64_RELOC_PAGE21);
        const int off = find(b + 4, macho::ARM64_RELOC_PAGEOFF12);
        REQUIRE(page > 0);
        REQUIRE(off > 0);
        // Each addend rides in an ARM64_RELOC_ADDEND entry just before the
        // relocation it modifies, its value in the symbolnum field.
        CHECK_EQ(rs[size_t(page) - 1].type, macho::ARM64_RELOC_ADDEND);
        CHECK_EQ(rs[size_t(page) - 1].sym, 16u);
        CHECK_EQ(rs[size_t(off) - 1].type, macho::ARM64_RELOC_ADDEND);
        CHECK_EQ(rs[size_t(off) - 1].sym, 16u);
        CHECK(find(b + 8, macho::ARM64_RELOC_GOT_LOAD_PAGE21) >= 0);
        CHECK(find(b + 12, macho::ARM64_RELOC_GOT_LOAD_PAGEOFF12) >= 0);
    }
    CHECK(text_found);

    // No PC-relative 32-bit data relocation exists on arm64 Mach-O.
    size_t base2 = 0;
    ObjectFile bad = object_with_code(Target::aarch64_macos(), {0}, base2);
    add_data_symbol(bad, "macho_data", 8);
    add_reloc(bad, ".text", base2, RelocKind::PCRel32, "macho_data");
    bool threw = false;
    try {
        (void)emit_macho_object(bad);
    } catch (const std::exception&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("AArch64 reloc - materialize_got_slots: import slots and defined-symbol relaxation") {
    size_t base = 0;
    ObjectFile obj = object_with_code(Target::aarch64_windows(),
        {kAdrpX0, kLdrX1, kAdrpX0, kLdrX1, kAdrpX0, 0xF9400000u /* ldr x0, [x0] */}, base);
    add_data_symbol(obj, "mine", 8);
    add_reloc(obj, ".text", base + 0, RelocKind::GotPage21, "theirs");
    add_reloc(obj, ".text", base + 4, RelocKind::GotLo12, "theirs");
    add_reloc(obj, ".text", base + 8, RelocKind::GotPage21, "theirs");
    add_reloc(obj, ".text", base + 12, RelocKind::GotLo12, "theirs");
    add_reloc(obj, ".text", base + 16, RelocKind::GotPage21, "mine");
    add_reloc(obj, ".text", base + 20, RelocKind::GotLo12, "mine");

    materialize_got_slots(obj, ".rdata", SectionKind::RoData, SectionFlags::Read | SectionFlags::Alloc);

    const Section* text = obj.get_section(".text");
    REQUIRE(text != nullptr);
    auto reloc_at = [&](size_t off) -> const ObjectRelocation* {
        for (const auto& r : text->relocations) {
            if (r.offset == off) return &r;
        }
        return nullptr;
    };
    REQUIRE(reloc_at(base + 0) != nullptr);
    CHECK(reloc_at(base + 0)->kind == RelocKind::AdrPage21);
    CHECK(reloc_at(base + 4)->kind == RelocKind::LdSt64Lo12);
    CHECK_EQ(reloc_at(base + 4)->symbol_name, std::string("theirs$got"));
    CHECK_EQ(reloc_at(base + 12)->symbol_name, std::string("theirs$got"));  // one slot per symbol
    CHECK(reloc_at(base + 16)->kind == RelocKind::AdrPage21);
    CHECK(reloc_at(base + 20)->kind == RelocKind::AddLo12);
    CHECK_EQ(reloc_at(base + 20)->symbol_name, std::string("mine"));
    CHECK_EQ(rd32(text->data.data() + base + 20), 0x91000000u);  // ldr became add

    const Section* slots = obj.get_section(".rdata");
    REQUIRE(slots != nullptr);
    size_t abs64 = 0;
    for (const auto& r : slots->relocations) {
        if (r.kind == RelocKind::Abs64 && r.symbol_name == "theirs") ++abs64;
    }
    CHECK_EQ(abs64, size_t(1));
}
