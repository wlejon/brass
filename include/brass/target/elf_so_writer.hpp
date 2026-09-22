#pragma once

#include <brass/object/object_writer.hpp>
#include <brass/target/image_imports.hpp>
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
    constexpr uint16_t EM_AARCH64 = 183;

    constexpr uint32_t PT_NULL    = 0;
    constexpr uint32_t PT_LOAD    = 1;
    constexpr uint32_t PT_DYNAMIC = 2;
    constexpr uint32_t PT_PHDR    = 6;
    constexpr uint32_t PT_GNU_STACK = 0x6474e551;
    constexpr uint32_t PT_GNU_RELRO = 0x6474e552;

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

    constexpr uint16_t SHN_UNDEF = 0;

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
    constexpr int64_t DT_TEXTREL = 22;
    constexpr int64_t DT_RUNPATH = 29;
    constexpr int64_t DT_FLAGS   = 30;
    constexpr int64_t DT_FLAGS_1 = 0x6ffffffb;

    constexpr uint64_t DF_TEXTREL  = 0x4;
    constexpr uint64_t DF_BIND_NOW = 0x8;
    constexpr uint64_t DF_1_NOW    = 0x1;

    constexpr uint8_t STB_LOCAL  = 0;
    constexpr uint8_t STB_GLOBAL = 1;
    constexpr uint8_t STB_WEAK   = 2;

    constexpr uint8_t STT_NOTYPE  = 0;
    constexpr uint8_t STT_OBJECT  = 1;
    constexpr uint8_t STT_FUNC    = 2;
    constexpr uint8_t STT_SECTION = 3;

    constexpr uint32_t R_X86_64_NONE      = 0;
    constexpr uint32_t R_X86_64_64        = 1;
    constexpr uint32_t R_X86_64_PC32      = 2;
    constexpr uint32_t R_X86_64_PLT32     = 4;
    constexpr uint32_t R_X86_64_GLOB_DAT  = 6;
    constexpr uint32_t R_X86_64_JUMP_SLOT = 7;
    constexpr uint32_t R_X86_64_RELATIVE  = 8;
    constexpr uint32_t R_X86_64_32        = 10;
    constexpr uint32_t R_AARCH64_ABS64     = 257;
    constexpr uint32_t R_AARCH64_GLOB_DAT  = 1025;
    constexpr uint32_t R_AARCH64_JUMP_SLOT = 1026;
    constexpr uint32_t R_AARCH64_RELATIVE  = 1027;
}

struct ElfSoOptions {
    std::string soname;
    bool export_all_functions = true;
    std::vector<std::string> explicit_exports;
    // Where every undefined symbol a relocation names comes from: one
    // DT_NEEDED per library used, and an error for a symbol in none of them.
    std::vector<ImportLibrary> imports;
    // DT_RUNPATH entries, in order.
    std::vector<std::string> rpaths;
    // Page alignment for loadable segments (0 = default: 0x10000 for AArch64, 0x1000 for x86_64).
    uint64_t page_size = 0;
};

class ElfSoWriter {
public:
    ElfSoWriter(const object::ObjectFile& obj, const ElfSoOptions& options);
    explicit ElfSoWriter(const object::ObjectFile& obj);

    // Empty on failure, with `error()` saying why.
    std::vector<uint8_t> write();
    bool write_to_file(const std::string& path);
    const std::string& error() const noexcept { return error_; }

    static std::vector<uint8_t> emit(const object::ObjectFile& obj, const ElfSoOptions& options);
    static std::vector<uint8_t> emit(const object::ObjectFile& obj, const ElfSoOptions& options,
                                     std::string* error_out);
    static std::vector<uint8_t> emit(const object::ObjectFile& obj);

private:
    object::ObjectFile obj_;
    ElfSoOptions options_;
    std::string error_;
};

} // namespace brass::target
