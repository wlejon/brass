#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/target/elf_so_writer.hpp>
#include <brass/target/macho_dylib_writer.hpp>
#include <cstring>
#include <vector>
#include <string>

using namespace brass;
using namespace brass::object;
using namespace brass::target;

namespace {

uint16_t read_u16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_u32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0] | (static_cast<uint32_t>(p[1]) << 8) |
                                 (static_cast<uint32_t>(p[2]) << 16) |
                                 (static_cast<uint32_t>(p[3]) << 24));
}

uint64_t read_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (static_cast<uint64_t>(p[i]) << (i * 8));
    }
    return v;
}

} // namespace

// =============================================================================
// Test 1: ELF Object Writer & DWARF CFI on AArch64 Linux
// =============================================================================
TEST_CASE("AArch64 ObjectWriter - ELF64 Header, Relocations, and DWARF CFI") {
    Module mod("test_aarch64_elf_mod");
    Function* callee = mod.create_function("elf_callee", Type::i64(), {Type::i64()});
    Function* caller = mod.create_function("elf_caller", Type::i64(), {Type::i64()});

    {
        Builder b(mod);
        b.set_function(callee);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* five = b.build_iconst_i64(5);
        Value* sum = b.build_add(x, five);
        b.build_ret(sum);
        callee->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*callee));
    }

    {
        Builder b(mod);
        b.set_function(caller);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* call_res = b.build_call("elf_callee", Type::i64(), {x});
        b.build_ret(call_res);
        caller->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*caller));
    }

    Target target = Target::aarch64_linux();
    ObjectFile obj = compile_module_to_object(mod, target);

    // Add a data section with Abs64 relocation to test R_AARCH64_ABS64
    obj.get_or_create_section(".data", SectionKind::Data, SectionFlags::Read | SectionFlags::Write | SectionFlags::Alloc, 8);
    Section* data_sec = obj.get_section(".data");
    REQUIRE(data_sec != nullptr);
    size_t data_off = data_sec->data.size();
    data_sec->emit64(0);
    ObjectRelocation r_abs;
    r_abs.offset = data_off;
    r_abs.kind = RelocKind::Abs64;
    r_abs.symbol_name = "elf_callee";
    r_abs.addend = 0;
    data_sec->relocations.push_back(r_abs);

    std::vector<uint8_t> elf_bytes = emit_elf_object(obj);
    REQUIRE(elf_bytes.size() >= 64);

    // 1. ELF Header
    const uint8_t* ehdr = elf_bytes.data();
    CHECK_EQ(ehdr[0], 0x7F);
    CHECK_EQ(ehdr[1], 'E');
    CHECK_EQ(ehdr[2], 'L');
    CHECK_EQ(ehdr[3], 'F');
    CHECK_EQ(ehdr[4], elf::ELFCLASS64);
    CHECK_EQ(ehdr[5], elf::ELFDATA2LSB);

    uint16_t e_type = read_u16(ehdr + 16);
    uint16_t e_machine = read_u16(ehdr + 18);
    CHECK_EQ(e_type, elf::ET_REL);
    CHECK_EQ(e_machine, elf::EM_AARCH64);

    uint64_t e_shoff = read_u64(ehdr + 40);
    uint16_t e_shnum = read_u16(ehdr + 60);
    uint16_t e_shstrndx = read_u16(ehdr + 62);
    REQUIRE(e_shoff > 0);
    REQUIRE(e_shstrndx < e_shnum);

    const uint8_t* shstr_shdr = ehdr + e_shoff + e_shstrndx * 64;
    uint64_t shstr_offset = read_u64(shstr_shdr + 24);
    const char* shstrtab = reinterpret_cast<const char*>(ehdr + shstr_offset);

    bool found_rela_text = false;
    bool found_rela_data = false;
    const uint8_t* eh_frame_ptr = nullptr;
    uint64_t eh_frame_size = 0;

    for (uint16_t i = 0; i < e_shnum; ++i) {
        const uint8_t* shdr = ehdr + e_shoff + i * 64;
        uint32_t sh_name = read_u32(shdr + 0);
        uint32_t sh_type = read_u32(shdr + 4);
        const char* name = shstrtab + sh_name;

        if (std::strcmp(name, ".rela.text") == 0) {
            found_rela_text = true;
            CHECK_EQ(sh_type, elf::SHT_RELA);
            uint64_t rela_off = read_u64(shdr + 24);
            uint64_t rela_size = read_u64(shdr + 32);
            REQUIRE(rela_size >= 24);

            const uint8_t* rela = ehdr + rela_off;
            uint64_t r_info = read_u64(rela + 8);
            uint32_t r_type = static_cast<uint32_t>(r_info & 0xFFFFFFFF);
            CHECK_EQ(r_type, elf::R_AARCH64_CALL26);
            int64_t addend = static_cast<int64_t>(read_u64(rela + 16));
            CHECK_EQ(addend, 0); // No -4 subtraction on AArch64
        } else if (std::strcmp(name, ".rela.data") == 0) {
            found_rela_data = true;
            CHECK_EQ(sh_type, elf::SHT_RELA);
            uint64_t rela_off = read_u64(shdr + 24);
            const uint8_t* rela = ehdr + rela_off;
            uint64_t r_info = read_u64(rela + 8);
            uint32_t r_type = static_cast<uint32_t>(r_info & 0xFFFFFFFF);
            CHECK_EQ(r_type, elf::R_AARCH64_ABS64);
        } else if (std::strcmp(name, ".eh_frame") == 0) {
            eh_frame_ptr = ehdr + read_u64(shdr + 24);
            eh_frame_size = read_u64(shdr + 32);
        }
    }

    CHECK(found_rela_text);
    CHECK(found_rela_data);
    REQUIRE(eh_frame_ptr != nullptr);
    REQUIRE(eh_frame_size >= 32);

    // Verify AArch64 CIE in .eh_frame
    uint32_t cie_len = read_u32(eh_frame_ptr + 0);
    uint32_t cie_id = read_u32(eh_frame_ptr + 4);
    uint8_t version = eh_frame_ptr[8];
    const char* aug = reinterpret_cast<const char*>(eh_frame_ptr + 9);

    CHECK(cie_len > 0);
    CHECK_EQ(cie_id, uint32_t(0));
    CHECK_EQ(version, uint8_t(1));
    CHECK_EQ(std::string(aug), "zR");

    // Augmentation is "zR\0" => 3 bytes (indices 9, 10, 11)
    size_t cie_idx = 12;
    uint8_t code_align = eh_frame_ptr[cie_idx++];
    CHECK_EQ(code_align, uint8_t(4)); // 4-byte ARM64 instructions

    uint8_t data_align = eh_frame_ptr[cie_idx++];
    CHECK_EQ(data_align, uint8_t(0x78)); // -8 (SLEB128)

    uint8_t return_reg = eh_frame_ptr[cie_idx++];
    CHECK_EQ(return_reg, uint8_t(30)); // LR (X30)

    uint8_t aug_data_len = eh_frame_ptr[cie_idx++];
    CHECK_EQ(aug_data_len, uint8_t(1));
    cie_idx += aug_data_len; // Skip pointer encoding

    // Initial instruction: DW_CFA_def_cfa (0x0C) reg 31 (SP), offset 0
    uint8_t cfa_op = eh_frame_ptr[cie_idx++];
    CHECK_EQ(cfa_op, elf::DW_CFA_def_cfa);
    uint8_t cfa_reg = eh_frame_ptr[cie_idx++];
    CHECK_EQ(cfa_reg, uint8_t(31)); // SP = 31 in DWARF AArch64
    uint8_t cfa_off = eh_frame_ptr[cie_idx++];
    CHECK_EQ(cfa_off, uint8_t(0));
}

// =============================================================================
// Test 2: Mach-O Object Writer on AArch64 macOS
// =============================================================================
TEST_CASE("AArch64 ObjectWriter - Mach-O 64-bit Header and ARM64 Relocations") {
    Module mod("test_aarch64_macho_mod");
    Function* f1 = mod.create_function("macho_callee", Type::i64(), {Type::i64()});
    Function* f2 = mod.create_function("macho_caller", Type::i64(), {Type::i64()});

    {
        Builder b(mod);
        b.set_function(f1);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* two = b.build_iconst_i64(2);
        Value* res = b.build_mul(x, two);
        b.build_ret(res);
        f1->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f1));
    }

    {
        Builder b(mod);
        b.set_function(f2);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* call_res = b.build_call("macho_callee", Type::i64(), {x});
        b.build_ret(call_res);
        f2->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f2));
    }

    Target target = Target::aarch64_macos();
    ObjectFile obj = compile_module_to_object(mod, target);

    // Explicitly add synthetic relocations to check AdrPage21 (PAGE21), AddLo12 (PAGEOFF12), Abs64 (UNSIGNED)
    Section* text_sec = obj.get_section(".text");
    REQUIRE(text_sec != nullptr);

    ObjectRelocation r_page21;
    r_page21.offset = 4;
    r_page21.kind = RelocKind::AdrPage21;
    r_page21.symbol_name = "macho_callee";
    r_page21.addend = 0;
    text_sec->relocations.push_back(r_page21);

    ObjectRelocation r_pageoff12;
    r_pageoff12.offset = 8;
    r_pageoff12.kind = RelocKind::AddLo12;
    r_pageoff12.symbol_name = "macho_callee";
    r_pageoff12.addend = 0;
    text_sec->relocations.push_back(r_pageoff12);

    ObjectRelocation r_abs64;
    r_abs64.offset = 16;
    r_abs64.kind = RelocKind::Abs64;
    r_abs64.symbol_name = "macho_callee";
    r_abs64.addend = 0;
    text_sec->relocations.push_back(r_abs64);

    std::vector<uint8_t> macho_bytes = emit_macho_object(obj);
    REQUIRE(macho_bytes.size() >= 32);

    // 1. Check mach_header_64
    const uint8_t* hdr = macho_bytes.data();
    uint32_t magic = read_u32(hdr + 0);
    uint32_t cputype = read_u32(hdr + 4);
    uint32_t cpusubtype = read_u32(hdr + 8);
    uint32_t filetype = read_u32(hdr + 12);

    CHECK_EQ(magic, macho::MH_MAGIC_64);
    CHECK_EQ(cputype, static_cast<uint32_t>(macho::CPU_TYPE_ARM64));
    CHECK_EQ(cpusubtype, static_cast<uint32_t>(macho::CPU_SUBTYPE_ARM64_ALL));
    CHECK_EQ(filetype, macho::MH_OBJECT);

    // 2. Check Relocations in __text section
    const uint8_t* cmd_ptr = hdr + 32;
    uint32_t nsects = read_u32(cmd_ptr + 64);
    const uint8_t* sec_ptr = cmd_ptr + 72;

    bool found_branch26 = false;
    bool found_page21 = false;
    bool found_pageoff12 = false;
    bool found_unsigned64 = false;

    for (uint32_t i = 0; i < nsects; ++i) {
        char sectname[17] = {0};
        std::memcpy(sectname, sec_ptr + i * 80, 16);

        if (std::strcmp(sectname, "__text") == 0) {
            uint32_t reloff = read_u32(sec_ptr + i * 80 + 56);
            uint32_t nreloc = read_u32(sec_ptr + i * 80 + 60);

            if (nreloc > 0 && reloff > 0) {
                const uint8_t* reloc_ptr = macho_bytes.data() + reloff;
                for (uint32_t r = 0; r < nreloc; ++r) {
                    uint32_t r_info = read_u32(reloc_ptr + r * 8 + 4);
                    uint32_t r_pcrel = (r_info >> 24) & 1;
                    uint32_t r_length = (r_info >> 25) & 3;
                    uint32_t r_type = (r_info >> 28) & 0xF;

                    if (r_type == macho::ARM64_RELOC_BRANCH26) {
                        found_branch26 = true;
                        CHECK_EQ(r_pcrel, 1u);
                        CHECK_EQ(r_length, 2u);
                    } else if (r_type == macho::ARM64_RELOC_PAGE21) {
                        found_page21 = true;
                        CHECK_EQ(r_pcrel, 1u);
                        CHECK_EQ(r_length, 2u);
                    } else if (r_type == macho::ARM64_RELOC_PAGEOFF12) {
                        found_pageoff12 = true;
                        CHECK_EQ(r_pcrel, 0u);
                        CHECK_EQ(r_length, 2u);
                    } else if (r_type == macho::ARM64_RELOC_UNSIGNED && r_length == 3) {
                        found_unsigned64 = true;
                        CHECK_EQ(r_pcrel, 0u);
                    }
                }
            }
        }
    }

    CHECK(found_branch26);
    CHECK(found_page21);
    CHECK(found_pageoff12);
    CHECK(found_unsigned64);
}

// =============================================================================
// Test 3: COFF Object Writer on AArch64 Windows
// =============================================================================
TEST_CASE("AArch64 ObjectWriter - COFF Header and ARM64 Relocations") {
    Module mod("test_aarch64_coff_mod");
    Function* f1 = mod.create_function("coff_callee", Type::i64(), {Type::i64()});
    Function* f2 = mod.create_function("coff_caller", Type::i64(), {Type::i64()});

    {
        Builder b(mod);
        b.set_function(f1);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* seven = b.build_iconst_i64(7);
        Value* res = b.build_add(x, seven);
        b.build_ret(res);
        f1->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f1));
    }

    {
        Builder b(mod);
        b.set_function(f2);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* call_res = b.build_call("coff_callee", Type::i64(), {x});
        b.build_ret(call_res);
        f2->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*f2));
    }

    Target target = Target::aarch64_windows();
    ObjectFile obj = compile_module_to_object(mod, target);

    // Add relocations for testing PAGE21, PAGEOFFSET_12A, SECREL, ADDR64
    Section* text_sec = obj.get_section(".text");
    REQUIRE(text_sec != nullptr);

    ObjectRelocation r_p21;
    r_p21.offset = 4;
    r_p21.kind = RelocKind::AdrPage21;
    r_p21.symbol_name = "coff_callee";
    text_sec->relocations.push_back(r_p21);

    ObjectRelocation r_lo12;
    r_lo12.offset = 8;
    r_lo12.kind = RelocKind::AddLo12;
    r_lo12.symbol_name = "coff_callee";
    text_sec->relocations.push_back(r_lo12);

    ObjectRelocation r_sec;
    r_sec.offset = 16;
    r_sec.kind = RelocKind::SecRel32;
    r_sec.symbol_name = "coff_callee";
    text_sec->relocations.push_back(r_sec);

    ObjectRelocation r_a64;
    r_a64.offset = 24;
    r_a64.kind = RelocKind::Abs64;
    r_a64.symbol_name = "coff_callee";
    text_sec->relocations.push_back(r_a64);

    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);
    REQUIRE(coff_bytes.size() >= 20);

    // 1. IMAGE_FILE_HEADER
    const uint8_t* fhdr = coff_bytes.data();
    uint16_t machine = read_u16(fhdr + 0);
    uint16_t num_sec = read_u16(fhdr + 2);
    CHECK_EQ(machine, coff::IMAGE_FILE_MACHINE_ARM64);
    CHECK(num_sec >= 1);

    // 2. Inspect relocations
    bool found_branch26 = false;
    bool found_page21 = false;
    bool found_pageoffset = false;
    bool found_secrel = false;
    bool found_addr64 = false;

    for (uint16_t i = 0; i < num_sec; ++i) {
        const uint8_t* shdr = fhdr + 20 + i * 40;
        char name[9] = {0};
        std::memcpy(name, shdr, 8);

        if (std::strcmp(name, ".text") == 0) {
            uint32_t reloff = read_u32(shdr + 24);
            uint16_t nreloc = read_u16(shdr + 32);

            if (nreloc > 0 && reloff > 0) {
                const uint8_t* rptr = fhdr + reloff;
                for (uint16_t r = 0; r < nreloc; ++r) {
                    uint16_t r_type = read_u16(rptr + r * 10 + 8);
                    if (r_type == coff::IMAGE_REL_ARM64_BRANCH26) {
                        found_branch26 = true;
                    } else if (r_type == coff::IMAGE_REL_ARM64_PAGE21) {
                        found_page21 = true;
                    } else if (r_type == coff::IMAGE_REL_ARM64_PAGEOFFSET_12A) {
                        found_pageoffset = true;
                    } else if (r_type == coff::IMAGE_REL_ARM64_SECREL) {
                        found_secrel = true;
                    } else if (r_type == coff::IMAGE_REL_ARM64_ADDR64) {
                        found_addr64 = true;
                    }
                }
            }
        }
    }

    CHECK(found_branch26);
    CHECK(found_page21);
    CHECK(found_pageoffset);
    CHECK(found_secrel);
    CHECK(found_addr64);
}

// =============================================================================
// Test 4: Standalone In-Memory Linkers (ElfSoWriter & MachODylibWriter) AArch64
// =============================================================================
TEST_CASE("AArch64 Standalone Linkers - ElfSoWriter and MachODylibWriter") {
    // 1. ElfSoWriter
    {
        Module mod("test_elf_so_aarch64");
        Function* f1 = mod.create_function("foo", Type::i64(), {Type::i64()});
        Function* f2 = mod.create_function("bar", Type::i64(), {Type::i64()});

        {
            Builder b(mod);
            b.set_function(f1);
            BasicBlock* entry = b.append_block("entry");
            Value* x = b.add_block_param(entry, Type::i64());
            Value* c = b.build_iconst_i64(10);
            Value* sum = b.build_add(x, c);
            b.build_ret(sum);
            f1->rebuild_cfg_predecessors();
            REQUIRE(verify_function(*f1));
        }

        {
            Builder b(mod);
            b.set_function(f2);
            BasicBlock* entry = b.append_block("entry");
            Value* x = b.add_block_param(entry, Type::i64());
            Value* res = b.build_call("foo", Type::i64(), {x});
            b.build_ret(res);
            f2->rebuild_cfg_predecessors();
            REQUIRE(verify_function(*f2));
        }

        Target target = Target::aarch64_linux();
        ObjectFile obj = compile_module_to_object(mod, target);

        ElfSoOptions opts;
        opts.soname = "libtest_aarch64.so";
        std::vector<uint8_t> so = ElfSoWriter::emit(obj, opts);
        REQUIRE(so.size() >= 64);

        // Verify ELF header e_machine
        uint16_t e_machine = read_u16(so.data() + 18);
        CHECK_EQ(e_machine, elf64::EM_AARCH64);

        uint16_t e_type = read_u16(so.data() + 16);
        CHECK_EQ(e_type, elf64::ET_DYN);
    }

    // 2. MachODylibWriter
    {
        Module mod("test_dylib_aarch64");
        Function* f1 = mod.create_function("func_a", Type::i64(), {Type::i64()});
        Function* f2 = mod.create_function("func_b", Type::i64(), {Type::i64()});

        {
            Builder b(mod);
            b.set_function(f1);
            BasicBlock* entry = b.append_block("entry");
            Value* x = b.add_block_param(entry, Type::i64());
            Value* c = b.build_iconst_i64(42);
            Value* sum = b.build_add(x, c);
            b.build_ret(sum);
            f1->rebuild_cfg_predecessors();
            REQUIRE(verify_function(*f1));
        }

        {
            Builder b(mod);
            b.set_function(f2);
            BasicBlock* entry = b.append_block("entry");
            Value* x = b.add_block_param(entry, Type::i64());
            Value* res = b.build_call("func_a", Type::i64(), {x});
            b.build_ret(res);
            f2->rebuild_cfg_predecessors();
            REQUIRE(verify_function(*f2));
        }

        Target target = Target::aarch64_macos();
        ObjectFile obj = compile_module_to_object(mod, target);

        MachODylibOptions opts;
        opts.install_name = "libtest_aarch64.dylib";
        std::vector<uint8_t> dylib = MachODylibWriter::emit(obj, opts);
        REQUIRE(dylib.size() >= 32);

        // Verify Mach-O header
        const uint8_t* hdr = dylib.data();
        uint32_t magic = read_u32(hdr + 0);
        uint32_t cputype = read_u32(hdr + 4);
        uint32_t cpusubtype = read_u32(hdr + 8);
        uint32_t filetype = read_u32(hdr + 12);

        CHECK_EQ(magic, macho::MH_MAGIC_64);
        CHECK_EQ(cputype, static_cast<uint32_t>(macho::CPU_TYPE_ARM64));
        CHECK_EQ(cpusubtype, static_cast<uint32_t>(macho::CPU_SUBTYPE_ARM64_ALL));
        CHECK_EQ(filetype, macho::MH_DYLIB);
    }
}
