#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/module.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/image_imports.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace brass::target {

enum class OutputFormat : uint8_t {
    Auto,
    WindowsPeDll,
    LinuxElfSo,
    MacOSMachODylib
};

struct LinkerOptions {
    OutputFormat format = OutputFormat::Auto;
    std::string module_name = "brass_module.dll";
    std::string soname;
    uint64_t image_base = 0x180000000ULL;
    bool export_all_functions = true;
    std::vector<std::string> explicit_exports;
    // Where undefined symbols come from. Every undefined symbol a relocation
    // names must be listed under one of these libraries (a DLL name, an ELF
    // soname, or a Mach-O install name such as "@rpath/libfoo.dylib");
    // one that is not is a link error naming the symbol.
    std::vector<ImportLibrary> imports;
    // Run-time search paths for those libraries (DT_RUNPATH / LC_RPATH);
    // PE has no equivalent and ignores them.
    std::vector<std::string> rpaths;
};

class AotLinker {
public:
    // The image bytes, or empty on failure with `*error_out` (when given)
    // saying why.
    static std::vector<uint8_t> link(const object::ObjectFile& obj, const LinkerOptions& options,
                                     std::string* error_out);
    static std::vector<uint8_t> link(const object::ObjectFile& obj, const LinkerOptions& options);
    static std::vector<uint8_t> link(const object::ObjectFile& obj);

    static std::vector<uint8_t> link(const Module& mod, const Target& target, const LinkerOptions& options);
    static std::vector<uint8_t> link(const Module& mod, const Target& target);
    static std::vector<uint8_t> link(const Module& mod, const LinkerOptions& options);
    static std::vector<uint8_t> link(const Module& mod);

    static bool link_to_file(const object::ObjectFile& obj, const std::string& path, const LinkerOptions& options,
                             std::string* error_out);
    static bool link_to_file(const object::ObjectFile& obj, const std::string& path, const LinkerOptions& options);
    static bool link_to_file(const object::ObjectFile& obj, const std::string& path);

    static bool link_to_file(const Module& mod, const std::string& path, const Target& target, const LinkerOptions& options);
    static bool link_to_file(const Module& mod, const std::string& path, const Target& target);
    static bool link_to_file(const Module& mod, const std::string& path, const LinkerOptions& options);
    static bool link_to_file(const Module& mod, const std::string& path);
};

} // namespace brass::target
