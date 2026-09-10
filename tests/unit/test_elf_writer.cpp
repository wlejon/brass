#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <cstring>

using namespace brass;
using namespace brass::object;

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

TEST_CASE("ELF64 Writer - Header and Section Table Layout") {
    Module mod("test_elf");
    Function* fn = mod.create_function("simple_elf_add", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* sum = b.build_add(a, c);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::x64_linux());
    std::vector<uint8_t> elf_bytes = emit_elf_object(obj);

    CHECK(!elf_bytes.empty());
    CHECK(elf_bytes.size() >= 64);

    // 1. Elf64_Ehdr
    const uint8_t* ehdr = elf_bytes.data();
    CHECK_EQ(ehdr[0], 0x7F);
    CHECK_EQ(ehdr[1], 'E');
    CHECK_EQ(ehdr[2], 'L');
    CHECK_EQ(ehdr[3], 'F');
    CHECK_EQ(ehdr[4], elf::ELFCLASS64);
    CHECK_EQ(ehdr[5], elf::ELFDATA2LSB);
    CHECK_EQ(ehdr[6], elf::EV_CURRENT);

    uint16_t e_type = read_u16(ehdr + 16);
    uint16_t e_machine = read_u16(ehdr + 18);
    uint64_t e_shoff = read_u64(ehdr + 40);
    uint16_t e_ehsize = read_u16(ehdr + 52);
    uint16_t e_shentsize = read_u16(ehdr + 58);
    uint16_t e_shnum = read_u16(ehdr + 60);
    uint16_t e_shstrndx = read_u16(ehdr + 62);

    CHECK_EQ(e_type, elf::ET_REL);
    CHECK_EQ(e_machine, elf::EM_X86_64);
    CHECK_EQ(e_ehsize, uint16_t(64));
    CHECK_EQ(e_shentsize, uint16_t(64));
    CHECK(e_shnum >= 5);
    CHECK(e_shoff > 0);
    CHECK(e_shstrndx < e_shnum);

    // 2. Read Section Headers & .shstrtab
    const uint8_t* shstr_shdr = ehdr + e_shoff + e_shstrndx * 64;
    uint64_t shstr_offset = read_u64(shstr_shdr + 24);
    const char* shstrtab = reinterpret_cast<const char*>(ehdr + shstr_offset);

    bool found_text = false;
    bool found_symtab = false;
    bool found_strtab = false;
    bool found_eh_frame = false;

    for (uint16_t i = 0; i < e_shnum; ++i) {
        const uint8_t* shdr = ehdr + e_shoff + i * 64;
        uint32_t sh_name = read_u32(shdr + 0);
        uint32_t sh_type = read_u32(shdr + 4);
        uint64_t sh_flags = read_u64(shdr + 8);
        const char* name = shstrtab + sh_name;

        if (std::strcmp(name, ".text") == 0) {
            found_text = true;
            CHECK_EQ(sh_type, elf::SHT_PROGBITS);
            CHECK((sh_flags & elf::SHF_EXECINSTR) != 0);
            CHECK((sh_flags & elf::SHF_ALLOC) != 0);
        } else if (std::strcmp(name, ".symtab") == 0) {
            found_symtab = true;
            CHECK_EQ(sh_type, elf::SHT_SYMTAB);
        } else if (std::strcmp(name, ".strtab") == 0) {
            found_strtab = true;
            CHECK_EQ(sh_type, elf::SHT_STRTAB);
        } else if (std::strcmp(name, ".eh_frame") == 0) {
            found_eh_frame = true;
            CHECK_EQ(sh_type, elf::SHT_PROGBITS);
            CHECK((sh_flags & elf::SHF_ALLOC) != 0);
        }
    }

    CHECK(found_text);
    CHECK(found_symtab);
    CHECK(found_strtab);
    CHECK(found_eh_frame);
}

TEST_CASE("ELF64 Writer - Symbol Table and Relocations") {
    Module mod("test_elf_relocs");
    Function* f1 = mod.create_function("elf_callee", Type::i64(), {Type::i64()});
    Function* f2 = mod.create_function("elf_caller", Type::i64(), {Type::i64()});

    {
        Builder b(mod);
        b.set_function(f1);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* two = b.build_iconst_i64(2);
        Value* res = b.build_mul(x, two);
        b.build_ret(res);
        f1->rebuild_cfg_predecessors();
        CHECK(verify_function(*f1));
    }

    {
        Builder b(mod);
        b.set_function(f2);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* call_res = b.build_call("elf_callee", Type::i64(), {x});
        b.build_ret(call_res);
        f2->rebuild_cfg_predecessors();
        CHECK(verify_function(*f2));
    }

    ObjectFile obj = compile_module_to_object(mod, Target::x64_linux());
    std::vector<uint8_t> elf_bytes = emit_elf_object(obj);

    const uint8_t* ehdr = elf_bytes.data();
    uint64_t e_shoff = read_u64(ehdr + 40);
    uint16_t e_shnum = read_u16(ehdr + 60);
    uint16_t e_shstrndx = read_u16(ehdr + 62);

    const uint8_t* shstr_shdr = ehdr + e_shoff + e_shstrndx * 64;
    uint64_t shstr_offset = read_u64(shstr_shdr + 24);
    const char* shstrtab = reinterpret_cast<const char*>(ehdr + shstr_offset);

    bool found_rela_text = false;
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
            CHECK(rela_size >= 24); // At least 1 Elf64_Rela entry

            const uint8_t* rela = ehdr + rela_off;
            uint64_t r_info = read_u64(rela + 8);
            uint32_t r_type = static_cast<uint32_t>(r_info & 0xFFFFFFFF);
            CHECK(r_type == elf::R_X86_64_PLT32 || r_type == elf::R_X86_64_PC32);
        }
    }
    CHECK(found_rela_text);
}

TEST_CASE("ELF64 Writer - SysV DWARF CFI in .eh_frame") {
    Module mod("test_elf_cfi");
    Function* fn = mod.create_function("cfi_fn", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* one = b.build_iconst_i64(1);
    Value* res = b.build_add(x, one);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::x64_linux());
    std::vector<uint8_t> elf_bytes = emit_elf_object(obj);

    const uint8_t* ehdr = elf_bytes.data();
    uint64_t e_shoff = read_u64(ehdr + 40);
    uint16_t e_shnum = read_u16(ehdr + 60);
    uint16_t e_shstrndx = read_u16(ehdr + 62);

    const uint8_t* shstr_shdr = ehdr + e_shoff + e_shstrndx * 64;
    uint64_t shstr_offset = read_u64(shstr_shdr + 24);
    const char* shstrtab = reinterpret_cast<const char*>(ehdr + shstr_offset);

    const uint8_t* eh_frame_ptr = nullptr;
    uint64_t eh_frame_size = 0;

    for (uint16_t i = 0; i < e_shnum; ++i) {
        const uint8_t* shdr = ehdr + e_shoff + i * 64;
        uint32_t sh_name = read_u32(shdr + 0);
        const char* name = shstrtab + sh_name;

        if (std::strcmp(name, ".eh_frame") == 0) {
            eh_frame_ptr = ehdr + read_u64(shdr + 24);
            eh_frame_size = read_u64(shdr + 32);
        }
    }

    CHECK(eh_frame_ptr != nullptr);
    CHECK(eh_frame_size >= 32);

    // Inspect CIE
    uint32_t cie_len = read_u32(eh_frame_ptr + 0);
    uint32_t cie_id = read_u32(eh_frame_ptr + 4);
    uint8_t version = eh_frame_ptr[8];
    const char* aug = reinterpret_cast<const char*>(eh_frame_ptr + 9);

    CHECK(cie_len > 0);
    CHECK_EQ(cie_id, uint32_t(0)); // 0 in .eh_frame
    CHECK_EQ(version, uint8_t(1));
    CHECK_EQ(std::string(aug), "zR");
}
