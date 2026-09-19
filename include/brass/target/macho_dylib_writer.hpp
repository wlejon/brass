#pragma once

#include <brass/object/object_writer.hpp>
#include <brass/target/image_imports.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace brass::target {

struct MachODylibOptions {
    uint64_t image_base = 0x0ULL;
    std::string install_name = "brass_module.dylib";
    bool export_all_functions = true;
    std::vector<std::string> explicit_exports;
    // Where every undefined symbol a relocation names comes from: one
    // LC_LOAD_DYLIB per library used (its install name), and an error for a
    // symbol in none of them.
    std::vector<ImportLibrary> imports;
    // LC_RPATH entries, in order.
    std::vector<std::string> rpaths;
};

class MachODylibWriter {
public:
    MachODylibWriter(const object::ObjectFile& obj, const MachODylibOptions& options);
    explicit MachODylibWriter(const object::ObjectFile& obj);

    // Empty on failure, with `error()` saying why.
    std::vector<uint8_t> write();
    bool write_to_file(const std::string& path);
    const std::string& error() const noexcept { return error_; }

    static std::vector<uint8_t> emit(const object::ObjectFile& obj, const MachODylibOptions& options);
    static std::vector<uint8_t> emit(const object::ObjectFile& obj, const MachODylibOptions& options,
                                     std::string* error_out);
    static std::vector<uint8_t> emit(const object::ObjectFile& obj);

private:
    object::ObjectFile obj_;
    MachODylibOptions options_;
    std::string error_;
};

} // namespace brass::target
