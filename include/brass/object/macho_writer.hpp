#pragma once

#include <brass/object/object_writer.hpp>
#include <optional>
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
    constexpr int32_t CPU_TYPE_ARM64 = 0x0100000C;  // CPU_ARCH_ABI64 | 0xC
    constexpr int32_t CPU_SUBTYPE_ARM64_ALL = 0;

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

    // LC_BUILD_VERSION platforms.
    constexpr uint32_t PLATFORM_MACOS        = 1;
    constexpr uint32_t PLATFORM_IOS          = 2;
    constexpr uint32_t PLATFORM_IOSSIMULATOR = 7;

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

    // ARM64 relocation types
    constexpr uint32_t ARM64_RELOC_UNSIGNED            = 0;
    constexpr uint32_t ARM64_RELOC_SUBTRACTOR          = 1;
    constexpr uint32_t ARM64_RELOC_BRANCH26            = 2;
    constexpr uint32_t ARM64_RELOC_PAGE21              = 3;
    constexpr uint32_t ARM64_RELOC_PAGEOFF12           = 4;
    constexpr uint32_t ARM64_RELOC_GOT_LOAD_PAGE21     = 5;
    constexpr uint32_t ARM64_RELOC_GOT_LOAD_PAGEOFF12  = 6;
    constexpr uint32_t ARM64_RELOC_POINTER_TO_GOT      = 7;
    constexpr uint32_t ARM64_RELOC_TLVP_LOAD_PAGE21    = 8;
    constexpr uint32_t ARM64_RELOC_TLVP_LOAD_PAGEOFF12 = 9;
    constexpr uint32_t ARM64_RELOC_ADDEND              = 10;
}

// What LC_BUILD_VERSION says an image is built for: the platform and the
// minimum OS and SDK versions, each packed as major << 16 | minor << 8 |
// patch (X.Y.Z). ld warns about an object without one ("no platform load
// command found") and checks its minimum against the link's.
//
// brass's Target has no iOS: an arm64 iOS or simulator object is an
// aarch64-macos object (the same Darwin calling convention) with platform
// PLATFORM_IOS or PLATFORM_IOSSIMULATOR here.
struct MachOBuildVersion {
    uint32_t platform = macho::PLATFORM_MACOS;
    uint32_t minos = 0;  // 0: the default below
    uint32_t sdk = 0;    // 0: the default below

    // Parses "X", "X.Y" or "X.Y.Z" (a deployment target); nullopt if it is not one.
    static std::optional<uint32_t> parse_version(std::string_view text);

    // `requested` with its zero fields filled in. The minimum OS is, in
    // order: the deployment-target environment variable a compiler and ld
    // read (MACOSX_DEPLOYMENT_TARGET, or IPHONEOS_DEPLOYMENT_TARGET for iOS
    // and its simulator); on macOS, the deployment target brass itself was
    // built for, which is the project's when brass builds in the same tree
    // (CMAKE_OSX_DEPLOYMENT_TARGET), so its objects link with the others
    // without warnings; else 11.0 for arm64 (its first macOS), 10.15 for
    // x86_64, 14.0 for iOS. An arm64 macOS minimum is never below 11.0. The
    // SDK is the one brass was built against on macOS, and never below the
    // minimum.
    static MachOBuildVersion resolve(const Target& target, const MachOBuildVersion& requested);
};

class MachOWriter {
public:
    explicit MachOWriter(const ObjectFile& obj);
    // `build_version`'s zero fields are resolved (MachOBuildVersion::resolve).
    MachOWriter(const ObjectFile& obj, const MachOBuildVersion& build_version);

    std::vector<uint8_t> write();
    bool write_to_file(const std::string& path);

private:
    ObjectFile obj_;
    MachOBuildVersion build_version_;
};

std::vector<uint8_t> emit_macho_object(const ObjectFile& obj);

} // namespace brass::object
