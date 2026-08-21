#pragma once

#include <brass/object/object_writer.hpp>
#include <vector>
#include <string>
#include <string_view>
#include <cstdint>

namespace brass::object {

namespace elf {
    constexpr uint8_t ELFCLASS64 = 2;
    constexpr uint8_t ELFDATA2LSB = 1;
    constexpr uint8_t EV_CURRENT = 1;
    constexpr uint8_t ELFOSABI_SYSV = 0;
    constexpr uint8_t ELFOSABI_LINUX = 3;

    constexpr uint16_t ET_REL = 1;
    constexpr uint16_t EM_X86_64 = 62;

    constexpr uint32_t SHT_NULL     = 0;
    constexpr uint32_t SHT_PROGBITS = 1;
    constexpr uint32_t SHT_SYMTAB   = 2;
    constexpr uint32_t SHT_STRTAB   = 3;
    constexpr uint32_t SHT_RELA     = 4;
    constexpr uint32_t SHT_NOBITS   = 8;

    constexpr uint64_t SHF_WRITE     = 0x1;
    constexpr uint64_t SHF_ALLOC     = 0x2;
    constexpr uint64_t SHF_EXECINSTR = 0x4;
    constexpr uint64_t SHF_INFO_LINK = 0x40;

    constexpr uint8_t STB_LOCAL  = 0;
    constexpr uint8_t STB_GLOBAL = 1;
    constexpr uint8_t STB_WEAK   = 2;

    constexpr uint8_t STT_NOTYPE  = 0;
    constexpr uint8_t STT_OBJECT  = 1;
    constexpr uint8_t STT_FUNC    = 2;
    constexpr uint8_t STT_SECTION = 3;

    constexpr uint32_t R_X86_64_NONE     = 0;
    constexpr uint32_t R_X86_64_64       = 1;
    constexpr uint32_t R_X86_64_PC32     = 2;
    constexpr uint32_t R_X86_64_PLT32    = 4;
    constexpr uint32_t R_X86_64_32       = 10;
    constexpr uint32_t R_X86_64_32S      = 11;
    constexpr uint32_t R_X86_64_PC64     = 24;

    // DWARF CFI Call Frame Instructions
    constexpr uint8_t DW_CFA_advance_loc        = 0x40;
    constexpr uint8_t DW_CFA_offset             = 0x80;
    constexpr uint8_t DW_CFA_nop                = 0x00;
    constexpr uint8_t DW_CFA_set_loc            = 0x01;
    constexpr uint8_t DW_CFA_advance_loc1       = 0x02;
    constexpr uint8_t DW_CFA_advance_loc2       = 0x03;
    constexpr uint8_t DW_CFA_advance_loc4       = 0x04;
    constexpr uint8_t DW_CFA_offset_extended    = 0x05;
    constexpr uint8_t DW_CFA_def_cfa            = 0x0C;
    constexpr uint8_t DW_CFA_def_cfa_register   = 0x0D;
    constexpr uint8_t DW_CFA_def_cfa_offset     = 0x0E;

    // DWARF Pointer Encodings for Augmentation "zR"
    constexpr uint8_t DW_EH_PE_absptr = 0x00;
    constexpr uint8_t DW_EH_PE_pcrel  = 0x10;
    constexpr uint8_t DW_EH_PE_sdata4 = 0x0B;
}

class ElfCfiBuilder {
public:
    static void build_eh_frame(
        ObjectFile& obj,
        Section& eh_frame_sec
    );
};

class ElfWriter {
public:
    explicit ElfWriter(const ObjectFile& obj);

    std::vector<uint8_t> write();
    bool write_to_file(const std::string& path);

private:
    ObjectFile obj_;
};

std::vector<uint8_t> emit_elf_object(const ObjectFile& obj);

} // namespace brass::object
