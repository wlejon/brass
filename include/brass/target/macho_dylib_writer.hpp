#pragma once

#include <brass/object/object_writer.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace brass::target {

struct MachODylibOptions {
    uint64_t image_base = 0x0ULL;
    std::string install_name = "brass_module.dylib";
    bool export_all_functions = true;
    std::vector<std::string> explicit_exports;
};

class MachODylibWriter {
public:
    MachODylibWriter(const object::ObjectFile& obj, const MachODylibOptions& options);
    explicit MachODylibWriter(const object::ObjectFile& obj);

    std::vector<uint8_t> write();
    bool write_to_file(const std::string& path);

    static std::vector<uint8_t> emit(const object::ObjectFile& obj, const MachODylibOptions& options);
    static std::vector<uint8_t> emit(const object::ObjectFile& obj);

private:
    object::ObjectFile obj_;
    MachODylibOptions options_;
};

} // namespace brass::target
