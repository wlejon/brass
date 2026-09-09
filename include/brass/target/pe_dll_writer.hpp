#pragma once

#include <brass/object/object_writer.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace brass::target {

namespace pe {
    constexpr uint16_t IMAGE_DOS_SIGNATURE = 0x5A4D;     // "MZ"
    constexpr uint32_t IMAGE_NT_SIGNATURE  = 0x00004550; // "PE\0\0"
    constexpr uint16_t IMAGE_FILE_MACHINE_AMD64 = 0x8664;

    constexpr uint16_t IMAGE_FILE_RELOCS_STRIPPED         = 0x0001;
    constexpr uint16_t IMAGE_FILE_EXECUTABLE_IMAGE        = 0x0002;
    constexpr uint16_t IMAGE_FILE_LARGE_ADDRESS_AWARE     = 0x0020;
    constexpr uint16_t IMAGE_FILE_DLL                     = 0x2000;

    constexpr uint16_t IMAGE_NT_OPTIONAL_HDR64_MAGIC      = 0x020B;

    constexpr uint16_t IMAGE_SUBSYSTEM_WINDOWS_GUI        = 2;
    constexpr uint16_t IMAGE_SUBSYSTEM_WINDOWS_CUI        = 3;

    constexpr uint16_t IMAGE_DLLCHARACTERISTICS_HIGH_ENTROPY_VA = 0x0020;
    constexpr uint16_t IMAGE_DLLCHARACTERISTICS_DYNAMIC_BASE    = 0x0040;
    constexpr uint16_t IMAGE_DLLCHARACTERISTICS_NX_COMPAT       = 0x0100;

    constexpr uint32_t IMAGE_DIRECTORY_ENTRY_EXPORT    = 0;
    constexpr uint32_t IMAGE_DIRECTORY_ENTRY_IMPORT    = 1;
    constexpr uint32_t IMAGE_DIRECTORY_ENTRY_RESOURCE  = 2;
    constexpr uint32_t IMAGE_DIRECTORY_ENTRY_EXCEPTION = 3;
    constexpr uint32_t IMAGE_DIRECTORY_ENTRY_SECURITY  = 4;
    constexpr uint32_t IMAGE_DIRECTORY_ENTRY_BASERELOC = 5;

    constexpr uint16_t IMAGE_REL_BASED_ABSOLUTE = 0;
    constexpr uint16_t IMAGE_REL_BASED_HIGHLOW  = 3;
    constexpr uint16_t IMAGE_REL_BASED_DIR64    = 10;
}

struct PeDllOptions {
    uint64_t image_base = 0x180000000ULL;
    std::string module_name = "brass_module.dll";
    bool export_all_functions = true;
    std::vector<std::string> explicit_exports;
};

class PeDllWriter {
public:
    PeDllWriter(const object::ObjectFile& obj, const PeDllOptions& options);
    explicit PeDllWriter(const object::ObjectFile& obj);

    std::vector<uint8_t> write();
    bool write_to_file(const std::string& path);

    static std::vector<uint8_t> emit(const object::ObjectFile& obj, const PeDllOptions& options);
    static std::vector<uint8_t> emit(const object::ObjectFile& obj);

private:
    object::ObjectFile obj_;
    PeDllOptions options_;
};

} // namespace brass::target
