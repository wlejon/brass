#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/debug/dwarf_emitter.hpp>
#include <brass/debug/codeview_emitter.hpp>
#include <brass/debug/debug_section.hpp>
#include <cstring>

using namespace brass;
using namespace brass::debug;
using namespace brass::object;

namespace {

uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0] |
                                 (static_cast<uint32_t>(p[1]) << 8) |
                                 (static_cast<uint32_t>(p[2]) << 16) |
                                 (static_cast<uint32_t>(p[3]) << 24));
}

uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (size_t i = 0; i < 8; ++i) {
        v |= (static_cast<uint64_t>(p[i]) << (i * 8));
    }
    return v;
}

} // namespace

TEST_CASE("Debug Relocations - ELF DWARF Relocation Formats") {
    Module mod("test_elf_relocs");
    mod.debug_context().get_or_add_file("app.brass");

    Function* fn = mod.create_function("foo", Type::i32(), {Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.set_current_loc(DebugLoc(1, 10, 2));
    Value* p = b.add_block_param(entry, Type::i32());
    b.set_current_loc(DebugLoc(1, 11, 4));
    Value* c = b.build_iconst_i32(10);
    b.set_current_loc(DebugLoc(1, 12, 2));
    Value* r = b.build_add(p, c);
    b.build_ret(r);
    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::x64_linux());
    std::vector<uint8_t> elf_bytes = emit_elf_object(obj);

    REQUIRE(!elf_bytes.empty());
    REQUIRE(elf_bytes.size() >= 64);

    // Verify ELF header
    CHECK_EQ(elf_bytes[0], 0x7F);
    CHECK_EQ(elf_bytes[1], 'E');
    CHECK_EQ(elf_bytes[2], 'L');
    CHECK_EQ(elf_bytes[3], 'F');

    // Parse section headers to find .rela.debug_info and .rela.debug_line
    uint64_t shoff = read_u64_le(elf_bytes.data() + 40);
    uint16_t shentsize = read_u16_le(elf_bytes.data() + 58);
    uint16_t shnum = read_u16_le(elf_bytes.data() + 60);
    uint16_t shstrndx = read_u16_le(elf_bytes.data() + 62);

    const uint8_t* shstrtab_sh = elf_bytes.data() + shoff + shstrndx * shentsize;
    uint64_t shstrtab_off = read_u64_le(shstrtab_sh + 24);
    const char* strtab = reinterpret_cast<const char*>(elf_bytes.data() + shstrtab_off);

    bool found_rela_debug_info = false;
    bool found_rela_debug_line = false;

    for (uint16_t i = 0; i < shnum; ++i) {
        const uint8_t* sh = elf_bytes.data() + shoff + i * shentsize;
        uint32_t name_idx = read_u32_le(sh);
        const char* sec_name = strtab + name_idx;
        uint32_t sh_type = read_u32_le(sh + 4);
        uint64_t sh_size = read_u64_le(sh + 32);

        if (std::strcmp(sec_name, ".rela.debug_info") == 0) {
            found_rela_debug_info = true;
            CHECK_EQ(sh_type, 4u); // SHT_RELA
            CHECK(sh_size > 0);
        }
        if (std::strcmp(sec_name, ".rela.debug_line") == 0) {
            found_rela_debug_line = true;
            CHECK_EQ(sh_type, 4u); // SHT_RELA
            CHECK(sh_size > 0);
        }
    }

    CHECK(found_rela_debug_info);
    CHECK(found_rela_debug_line);
}

TEST_CASE("Debug Relocations - COFF CodeView Relocation Formats") {
    Module mod("test_coff_relocs");
    mod.debug_context().get_or_add_file("calc.brass");

    Function* fn = mod.create_function("bar", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    b.set_current_loc(DebugLoc(1, 20, 1));
    Value* x = b.add_block_param(entry, Type::i64());
    b.set_current_loc(DebugLoc(1, 21, 5));
    Value* c = b.build_iconst_i64(5);
    b.set_current_loc(DebugLoc(1, 22, 1));
    Value* prod = b.build_mul(x, c);
    b.build_ret(prod);
    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::x64_windows());
    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);

    REQUIRE(!coff_bytes.empty());
    REQUIRE(coff_bytes.size() >= 20);

    const uint8_t* hdr = coff_bytes.data();
    uint16_t num_sec = read_u16_le(hdr + 2);
    uint32_t num_syms = read_u32_le(hdr + 12);
    const uint8_t* sec_hdrs = hdr + 20;

    bool found_debug_s = false;
    uint32_t debug_s_reloc_ptr = 0;
    uint16_t debug_s_num_relocs = 0;

    for (uint16_t i = 0; i < num_sec; ++i) {
        const uint8_t* sh = sec_hdrs + i * 40;
        char name[9] = {0};
        std::memcpy(name, sh, 8);
        if (std::strcmp(name, ".debug$S") == 0) {
            found_debug_s = true;
            debug_s_reloc_ptr = read_u32_le(sh + 24);
            debug_s_num_relocs = read_u16_le(sh + 32);
            break;
        }
    }

    CHECK(found_debug_s);
    CHECK(debug_s_num_relocs > 0);
    CHECK(debug_s_reloc_ptr > 0);

    // Validate COFF relocation records in .debug$S
    bool has_secrel = false;
    bool has_secidx = false;

    for (uint16_t r = 0; r < debug_s_num_relocs; ++r) {
        const uint8_t* r_entry = coff_bytes.data() + debug_s_reloc_ptr + r * 10;
        uint32_t r_vaddr = read_u32_le(r_entry + 0);
        uint32_t r_sym_idx = read_u32_le(r_entry + 4);
        uint16_t r_type = read_u16_le(r_entry + 8);

        CHECK(r_vaddr < coff_bytes.size());
        CHECK(r_sym_idx < num_syms);

        if (r_type == coff::IMAGE_REL_AMD64_SECREL) {
            has_secrel = true;
        }
        if (r_type == coff::IMAGE_REL_AMD64_SECTION) {
            has_secidx = true;
        }
    }

    CHECK(has_secrel);
    CHECK(has_secidx);
}