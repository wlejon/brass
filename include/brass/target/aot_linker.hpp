#pragma once

#include <brass/target/target.hpp>
#include <brass/mir/module.hpp>
#include <brass/object/object_writer.hpp>
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
};

class AotLinker {
public:
    static std::vector<uint8_t> link(const object::ObjectFile& obj, const LinkerOptions& options);
    static std::vector<uint8_t> link(const object::ObjectFile& obj);

    static std::vector<uint8_t> link(const Module& mod, const Target& target, const LinkerOptions& options);
    static std::vector<uint8_t> link(const Module& mod, const Target& target);
    static std::vector<uint8_t> link(const Module& mod, const LinkerOptions& options);
    static std::vector<uint8_t> link(const Module& mod);

    static bool link_to_file(const object::ObjectFile& obj, const std::string& path, const LinkerOptions& options);
    static bool link_to_file(const object::ObjectFile& obj, const std::string& path);

    static bool link_to_file(const Module& mod, const std::string& path, const Target& target, const LinkerOptions& options);
    static bool link_to_file(const Module& mod, const std::string& path, const Target& target);
    static bool link_to_file(const Module& mod, const std::string& path, const LinkerOptions& options);
    static bool link_to_file(const Module& mod, const std::string& path);
};

} // namespace brass::target
