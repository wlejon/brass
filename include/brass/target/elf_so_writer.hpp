#pragma once

#include <brass/object/object_writer.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace brass::target {

namespace elf64 {
    constexpr uint8_t ELFCLASS64 = 2;
    constexpr uint8_t ELFDATA2LSB = 1;
    constexpr uint8_t EV_CURRENT = 1;
    constexpr uint8_t ELFOSABI_SYSV = 0;

    constexpr uint16_t ET_DYN = 3;
    constexpr uint16_t EM_X86_64 = 62;

    constexpr uint32_t PT_NULL    = 0;
    constexpr uint32_t PT_LOAD    = 1;
    constexpr uint32_t PT_DYNAMIC = 2;
    constexpr uint32_t PT_PHDR    = 6;

    constexpr uint32_t PF_X = 0x1;
    constexpr uint32_t PF_W = 0x2;
    constexpr uint32_t PF_R = 0x4;

    constexpr uint32_t SHT_NULL     = 0;
    constexpr uint32_t SHT_PROGBITS = 1;
    constexpr uint32_t SHT_SYMTAB   = 2;
    constexpr uint32_t SHT_STRTAB   = 3;
    constexpr uint32_t SHT_RELA     = 4;
    constexpr uint32_t SHT_HASH     = 5;
    constexpr uint32_t SHT_DYNAMIC  = 6;
    constexpr uint32_t SHT_NOBITS   = 8;
    constexpr uint32_t SHT_DYNSYM   = 11;

    constexpr uint64_t SHF_WRITE     = 0x1;
    constexpr uint64_t SHF_ALLOC     = 0x2;
    constexpr uint64_t SHF_EXECINSTR = 0x4;

    constexpr int64_t DT_NULL    = 0;
    constexpr int64_t DT_NEEDED  = 1;
    constexpr int64_t DT_PLTRELSZ= 2;
    constexpr int64_t DT_PLTGOT  = 3;
    constexpr int64_t DT_HASH    = 4;
    constexpr int64_t DT_STRTAB  = 5;
    constexpr int64_t DT_SYMTAB  = 6;
    constexpr int64_t DT_RELA    = 7;
    constexpr int64_t DT_RELASZ  = 8;
    constexpr int64_t DT_RELAENT = 9;
    constexpr int64_t DT_STRSZ   = 10;
    constexpr int64_t DT_SYMENT  = 11;
    constexpr int64_t DT_SONAME  = 14;

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
    constexpr uint32_t R_X86_64_RELATIVE = 8;
    constexpr uint32_t R_X86_64_32       = 10;
}

struct ElfSoOptions {
    std::string soname;
    bool export_all_functions = true;
    std::vector<std::string> explicit_exports;
};

class ElfSoWriter {
public:
    ElfSoWriter(const object::ObjectFile& obj, const ElfSoOptions& options);
    explicit ElfSoWriter(const object::ObjectFile& obj);

    std::vector<uint8_t> write();
    bool write_to_file(const std::string& path);

    static std::vector<uint8_t> emit(const object::ObjectFile& obj, const ElfSoOptions& options);
    static std::vector<uint8_t> emit(const object::ObjectFile& obj);

private:
    object::ObjectFile obj_;
    ElfSoOptions options_;
};

} // namespace brass::target
