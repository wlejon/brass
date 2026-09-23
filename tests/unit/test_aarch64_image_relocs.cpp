#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/pe_dll_writer.hpp>
#include <brass/target/elf_so_writer.hpp>
#include <brass/target/macho_dylib_writer.hpp>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

// The image linkers (PE DLL, ELF .so, Mach-O dylib) resolving AArch64
// ADRP + ADD / LDR pairs: the linked instructions are decoded and must land
// on the target's page and offset, and a GOT load of an import must land on
// the slot the loader fills.

using namespace brass;
using namespace brass::object;
using namespace brass::target;

namespace {

uint16_t rd16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t rd64(const uint8_t* p) { return rd32(p) | (static_cast<uint64_t>(rd32(p + 4)) << 32); }

constexpr uint32_t kMarker = 0xD297DDE9u;          // movz x9, #0xBEEF
constexpr uint64_t kDataMarker = 0x0123456789ABCDEFull;

int64_t adrp_pages(uint32_t inst) {
    const uint32_t immlo = (inst >> 29) & 3u;
    const uint32_t immhi = (inst >> 5) & 0x7FFFFu;
    int64_t imm = static_cast<int64_t>((immhi << 2) | immlo);
    if (imm & (int64_t(1) << 20)) imm -= int64_t(1) << 21;
    return imm;
}
uint64_t imm12(uint32_t inst) { return (inst >> 10) & 0xFFFu; }
uint64_t adrp_target_page(uint64_t pc, uint32_t inst) {
    return (pc & ~uint64_t(0xFFF)) + static_cast<uint64_t>(adrp_pages(inst) * 4096);
}

// The unique file offset of `bytes`, or nullopt.
std::optional<size_t> find_unique(const std::vector<uint8_t>& f, const void* bytes, size_t n) {
    std::optional<size_t> at;
    for (size_t i = 0; i + n <= f.size(); ++i) {
        if (std::memcmp(f.data() + i, bytes, n) == 0) {
            if (at) return std::nullopt;
            at = i;
        }
    }
    return at;
}

// Text: marker; adrp x0 / add x0 (data); adrp x1 / ldr x1 (data + 8);
// adrp x2 / ldr x2 (GOT load of `import`, when given); ret.
ObjectFile build_object(const Target& target, const std::string& import) {
    Module mod("a64_image_mod");
    Function* fn = mod.create_function("a64_image_fn", Type::i64(), {Type::i64()});
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
    const size_t base = text->data.size();
    for (uint32_t w : {kMarker, 0x90000000u, 0x91000000u, 0x90000001u, 0xF9400021u,
                       0x90000002u, 0xF9400042u, 0xD65F03C0u}) {
        text->emit32(w);
    }

    Section& data = obj.get_or_create_section(".data", SectionKind::Data,
                                              SectionFlags::Read | SectionFlags::Write | SectionFlags::Alloc, 16);
    data.align_to(16);
    ObjectSymbol sym;
    sym.name = "a64_image_data";
    sym.section_index = obj.get_section_index(".data");
    sym.value = data.data.size();
    sym.size = 16;
    sym.type = SymbolType::Object;
    obj.add_symbol(sym);
    data.emit64(kDataMarker);
    data.emit64(0);

    auto add = [&](size_t off, RelocKind kind, const std::string& s, int64_t addend) {
        obj.get_section(".text")->relocations.push_back({base + off, kind, s, addend, 0});
    };
    add(4, RelocKind::AdrPage21, "a64_image_data", 0);
    add(8, RelocKind::AddLo12, "a64_image_data", 0);
    add(12, RelocKind::AdrPage21, "a64_image_data", 8);
    add(16, RelocKind::LdSt64Lo12, "a64_image_data", 8);
    if (!import.empty()) {
        add(20, RelocKind::GotPage21, import, 0);
        add(24, RelocKind::GotLo12, import, 0);
    }
    return obj;
}

struct Linked {
    uint64_t code_addr = 0;   // address of the marker
    const uint8_t* code = nullptr;
    uint64_t data_addr = 0;   // address of a64_image_data
};

// Checks the ADRP + ADD and ADRP + LDR pairs against the data address.
void check_data_pairs(const Linked& l) {
    const uint32_t adrp0 = rd32(l.code + 4), add0 = rd32(l.code + 8);
    const uint32_t adrp1 = rd32(l.code + 12), ldr1 = rd32(l.code + 16);
    CHECK_EQ(adrp_target_page(l.code_addr + 4, adrp0) + imm12(add0), l.data_addr);
    CHECK_EQ(adrp_target_page(l.code_addr + 12, adrp1) + imm12(ldr1) * 8, l.data_addr + 8);
    // Registers and opcodes kept.
    CHECK_EQ(adrp0 & 0x9F00001Fu, 0x90000000u);
    CHECK_EQ(add0 & 0xFFC003FFu, 0x91000000u);
    CHECK_EQ(ldr1 & 0xFFC003FFu, 0xF9400021u);
}

uint64_t got_slot(const Linked& l) {
    const uint32_t adrp2 = rd32(l.code + 20), ldr2 = rd32(l.code + 24);
    CHECK_EQ(ldr2 & 0xFFC003FFu, 0xF9400042u);  // still a load from the slot
    return adrp_target_page(l.code_addr + 20, adrp2) + imm12(ldr2) * 8;
}

// ---- PE ---------------------------------------------------------------------

struct PeSection { uint32_t rva, vsize, raw_size, raw_off; };

std::vector<PeSection> pe_sections(const std::vector<uint8_t>& f, const uint8_t** opt_hdr) {
    const uint8_t* file_hdr = f.data() + rd32(f.data() + 0x3C) + 4;
    const uint16_t nsec = rd16(file_hdr + 2);
    *opt_hdr = file_hdr + 20;
    const uint8_t* sh = file_hdr + 20 + rd16(file_hdr + 16);
    std::vector<PeSection> out;
    for (uint16_t i = 0; i < nsec; ++i, sh += 40) {
        out.push_back({rd32(sh + 12), rd32(sh + 8), rd32(sh + 16), rd32(sh + 20)});
    }
    return out;
}

std::optional<uint64_t> pe_rva_of(const std::vector<PeSection>& secs, size_t file_off) {
    for (const auto& s : secs) {
        if (file_off >= s.raw_off && file_off < uint64_t(s.raw_off) + s.raw_size) return s.rva + (file_off - s.raw_off);
    }
    return std::nullopt;
}

// ---- ELF --------------------------------------------------------------------

struct Load { uint32_t type; uint64_t off, vaddr, filesz; };

std::vector<Load> elf_phdrs(const std::vector<uint8_t>& f) {
    const uint64_t phoff = rd64(f.data() + 0x20);
    const uint16_t phnum = rd16(f.data() + 0x38);
    std::vector<Load> out;
    for (uint16_t i = 0; i < phnum; ++i) {
        const uint8_t* p = f.data() + phoff + uint64_t(i) * 56;
        out.push_back({rd32(p), rd64(p + 8), rd64(p + 16), rd64(p + 32)});
    }
    return out;
}

std::optional<uint64_t> elf_vaddr_of(const std::vector<Load>& ph, size_t file_off) {
    for (const auto& p : ph) {
        if (p.type == 1 && file_off >= p.off && file_off < p.off + p.filesz) return p.vaddr + (file_off - p.off);
    }
    return std::nullopt;
}

std::optional<size_t> elf_file_off_of(const std::vector<Load>& ph, uint64_t vaddr) {
    for (const auto& p : ph) {
        if (p.type == 1 && vaddr >= p.vaddr && vaddr < p.vaddr + p.filesz) return p.off + (vaddr - p.vaddr);
    }
    return std::nullopt;
}

} // namespace

TEST_CASE("AArch64 image relocs - PE DLL resolves ADRP pairs by page and GOT loads through the IAT") {
    ObjectFile obj = build_object(Target::aarch64_windows(), "ext_fn");
    PeDllOptions opts;
    opts.module_name = "a64_image.dll";
    opts.imports = {{"ext.dll", {"ext_fn"}}};
    std::string err;
    std::vector<uint8_t> dll = PeDllWriter::emit(obj, opts, &err);
    REQUIRE_EQ(err, std::string());
    REQUIRE(!dll.empty());

    const uint8_t* opt_hdr = nullptr;
    auto secs = pe_sections(dll, &opt_hdr);
    auto code_off = find_unique(dll, &kMarker, 4);
    auto data_off = find_unique(dll, &kDataMarker, 8);
    REQUIRE(code_off.has_value());
    REQUIRE(data_off.has_value());
    Linked l;
    l.code = dll.data() + *code_off;
    auto code_rva = pe_rva_of(secs, *code_off);
    auto data_rva = pe_rva_of(secs, *data_off);
    REQUIRE(code_rva.has_value());
    REQUIRE(data_rva.has_value());
    // Page arithmetic is the same on RVAs as on addresses: the image base is
    // page aligned.
    l.code_addr = *code_rva;
    l.data_addr = *data_rva;
    check_data_pairs(l);

    const uint64_t slot = got_slot(l);
    const uint32_t iat_rva = rd32(opt_hdr + 112 + 12 * 8);
    const uint32_t iat_size = rd32(opt_hdr + 112 + 12 * 8 + 4);
    REQUIRE(iat_size > 0);
    CHECK(slot >= iat_rva);
    CHECK(slot + 8 <= uint64_t(iat_rva) + iat_size);
    CHECK_EQ(slot % 8, uint64_t(0));
}

TEST_CASE("AArch64 image relocs - ELF .so resolves ADRP pairs by page and GOT loads through GLOB_DAT slots") {
    ObjectFile obj = build_object(Target::aarch64_linux(), "ext_fn");
    ElfSoOptions opts;
    opts.soname = "liba64image.so";
    opts.imports = {{"libext.so", {"ext_fn"}}};
    std::string err;
    std::vector<uint8_t> so = ElfSoWriter::emit(obj, opts, &err);
    REQUIRE_EQ(err, std::string());
    REQUIRE(!so.empty());

    auto ph = elf_phdrs(so);
    auto code_off = find_unique(so, &kMarker, 4);
    auto data_off = find_unique(so, &kDataMarker, 8);
    REQUIRE(code_off.has_value());
    REQUIRE(data_off.has_value());
    Linked l;
    l.code = so.data() + *code_off;
    auto code_va = elf_vaddr_of(ph, *code_off);
    auto data_va = elf_vaddr_of(ph, *data_off);
    REQUIRE(code_va.has_value());
    REQUIRE(data_va.has_value());
    l.code_addr = *code_va;
    l.data_addr = *data_va;
    check_data_pairs(l);

    // The slot must be one the dynamic loader fills: a GLOB_DAT for it in
    // DT_RELA.
    const uint64_t slot = got_slot(l);
    uint64_t rela = 0, relasz = 0;
    for (const auto& p : ph) {
        if (p.type != 2) continue;  // PT_DYNAMIC
        for (uint64_t o = p.off; o + 16 <= p.off + p.filesz; o += 16) {
            const uint64_t tag = rd64(so.data() + o), val = rd64(so.data() + o + 8);
            if (tag == 0) break;
            if (tag == 7) rela = val;
            if (tag == 8) relasz = val;
        }
    }
    REQUIRE(rela != 0);
    auto rela_off = elf_file_off_of(ph, rela);
    REQUIRE(rela_off.has_value());
    bool glob_dat = false;
    for (uint64_t o = 0; o + 24 <= relasz; o += 24) {
        const uint8_t* r = so.data() + *rela_off + o;
        if (rd64(r) == slot && (rd64(r + 8) & 0xFFFFFFFFu) == elf64::R_AARCH64_GLOB_DAT) glob_dat = true;
    }
    CHECK(glob_dat);
}

TEST_CASE("AArch64 image relocs - Mach-O dylib resolves ADRP pairs by page") {
    ObjectFile obj = build_object(Target::aarch64_macos(), "");
    MachODylibOptions opts;
    opts.install_name = "liba64image.dylib";
    std::string err;
    std::vector<uint8_t> dylib = MachODylibWriter::emit(obj, opts, &err);
    REQUIRE_EQ(err, std::string());
    REQUIRE(!dylib.empty());

    // file offset -> vmaddr through the LC_SEGMENT_64 commands
    std::vector<Load> segs;
    const uint32_t ncmds = rd32(dylib.data() + 16);
    const uint8_t* cmd = dylib.data() + 32;
    for (uint32_t i = 0; i < ncmds; ++i) {
        if (rd32(cmd) == 0x19) segs.push_back({1, rd64(cmd + 40), rd64(cmd + 24), rd64(cmd + 48)});
        cmd += rd32(cmd + 4);
    }
    auto code_off = find_unique(dylib, &kMarker, 4);
    auto data_off = find_unique(dylib, &kDataMarker, 8);
    REQUIRE(code_off.has_value());
    REQUIRE(data_off.has_value());
    Linked l;
    l.code = dylib.data() + *code_off;
    auto code_va = elf_vaddr_of(segs, *code_off);
    auto data_va = elf_vaddr_of(segs, *data_off);
    REQUIRE(code_va.has_value());
    REQUIRE(data_va.has_value());
    l.code_addr = *code_va;
    l.data_addr = *data_va;
    check_data_pairs(l);
}

TEST_CASE("AArch64 image relocs - a page-offset relocation on the wrong instruction fails the link") {
    ObjectFile obj = build_object(Target::aarch64_windows(), "");
    // Aim a load/store page offset at the ADD: refused, not bit-mangled.
    Section* text = obj.get_section(".text");
    for (auto& r : text->relocations) {
        if (r.kind == RelocKind::AddLo12) r.kind = RelocKind::LdSt64Lo12;
    }
    std::string err;
    std::vector<uint8_t> dll = PeDllWriter::emit(obj, PeDllOptions{}, &err);
    CHECK(dll.empty());
    CHECK(!err.empty());

    ObjectFile obj2 = build_object(Target::aarch64_linux(), "");
    for (auto& r : obj2.get_section(".text")->relocations) {
        if (r.kind == RelocKind::LdSt64Lo12) r.kind = RelocKind::LdSt32Lo12;  // wrong access size
    }
    err.clear();
    std::vector<uint8_t> so = ElfSoWriter::emit(obj2, ElfSoOptions{}, &err);
    CHECK(so.empty());
    CHECK(!err.empty());
}
