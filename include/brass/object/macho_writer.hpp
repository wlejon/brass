#pragma once

#include <brass/object/object_writer.hpp>
#include <vector>
#include <string>
#include <string_view>
#include <cstdint>

namespace brass::object {

namespace macho {
    constexpr uint32_t MH_MAGIC_64 = 0xFEEDFACF;
    constexpr uint32_t MH_CIGAM_64 = 0xCFFAEDFE;

    constexpr int32_t CPU_TYPE_X86_64 = 0x01000007; // CPU_ARCH_ABI64 | 0x7
    constexpr int32_t CPU_SUBTYPE_X86_64_ALL = 3;

    constexpr uint32_t MH_OBJECT  = 0x1;
    constexpr uint32_t MH_EXECUTE = 0x2;
    constexpr uint32_t MH_DYLIB   = 0x6;

    constexpr uint32_t MH_NOUNDEFS = 0x1;
    constexpr uint32_t MH_DYLDLINK = 0x4;
    constexpr uint32_t MH_TWOLEVEL = 0x80;
    constexpr uint32_t MH_SUBSECTIONS_VIA_SYMBOLS = 0x2000;

    constexpr uint32_t LC_SEGMENT_64     = 0x19;
    constexpr uint32_t LC_SYMTAB         = 0x2;
    constexpr uint32_t LC_DYSYMTAB       = 0xb;
    constexpr uint32_t LC_LOAD_DYLIB     = 0xc;
    constexpr uint32_t LC_ID_DYLIB       = 0xd;
    constexpr uint32_t LC_LOAD_DYLINKER  = 0xe;
    constexpr uint32_t LC_DYLD_INFO_ONLY = 0x80000022;
    constexpr uint32_t LC_BUILD_VERSION  = 0x32;

    constexpr int32_t VM_PROT_NONE    = 0;
    constexpr int32_t VM_PROT_READ    = 1;
    constexpr int32_t VM_PROT_WRITE   = 2;
    constexpr int32_t VM_PROT_EXECUTE = 4;

    constexpr uint32_t S_REGULAR                 = 0x0;
    constexpr uint32_t S_ZEROFILL                = 0x1;
    constexpr uint32_t S_CSTRING_LITERALS        = 0x2;
    constexpr uint32_t S_4BYTE_LITERALS          = 0x3;
    constexpr uint32_t S_8BYTE_LITERALS          = 0x4;
    constexpr uint32_t S_LITERAL_POINTERS        = 0x5;
    constexpr uint32_t S_ATTR_PURE_INSTRUCTIONS  = 0x80000000;
    constexpr uint32_t S_ATTR_SOME_INSTRUCTIONS  = 0x00000400;

    constexpr uint8_t N_EXT  = 0x01;
    constexpr uint8_t N_TYPE = 0x0e;
    constexpr uint8_t N_STAB = 0xe0;
    constexpr uint8_t N_PEXT = 0x10;

    constexpr uint8_t N_UNDF = 0x0;
    constexpr uint8_t N_ABS  = 0x2;
    constexpr uint8_t N_SECT = 0xe;
    constexpr uint8_t N_PBUD = 0xc;
    constexpr uint8_t N_INDR = 0xa;

    constexpr uint8_t NO_SECT = 0;

    // x86_64 relocation types
    constexpr uint8_t X86_64_RELOC_UNSIGNED   = 0;
    constexpr uint8_t X86_64_RELOC_SIGNED     = 1;
    constexpr uint8_t X86_64_RELOC_BRANCH     = 2;
    constexpr uint8_t X86_64_RELOC_GOT_LOAD   = 3;
    constexpr uint8_t X86_64_RELOC_GOT        = 4;
    constexpr uint8_t X86_64_RELOC_SUBTRACTOR = 5;
    constexpr uint8_t X86_64_RELOC_SIGNED_1   = 6;
    constexpr uint8_t X86_64_RELOC_SIGNED_2   = 7;
    constexpr uint8_t X86_64_RELOC_SIGNED_4   = 8;
    constexpr uint8_t X86_64_RELOC_TLV        = 9;
}

class MachOWriter {
public:
    explicit MachOWriter(const ObjectFile& obj);

    std::vector<uint8_t> write();
    bool write_to_file(const std::string& path);

private:
    ObjectFile obj_;
};

std::vector<uint8_t> emit_macho_object(const ObjectFile& obj);

} // namespace brass::object
