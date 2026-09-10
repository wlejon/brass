#include <brass/target/aot_linker.hpp>
#include <brass/target/pe_dll_writer.hpp>
#include <brass/target/elf_so_writer.hpp>
#include <brass/target/macho_dylib_writer.hpp>
#include <fstream>
#include <filesystem>
#include <cstdlib>

namespace brass::target {

namespace {

std::string extract_filename(const std::string& path) {
    std::filesystem::path p(path);
    return p.filename().string();
}

bool write_bytes_to_file(const std::string& path, const std::vector<uint8_t>& bytes) {
    std::ofstream file(path, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        return false;
    }
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return true;
}

} // namespace

std::vector<uint8_t> AotLinker::link(const object::ObjectFile& obj, const LinkerOptions& options) {
    OutputFormat fmt = options.format;
    if (fmt == OutputFormat::Auto) {
        if (obj.target.is_windows()) {
            fmt = OutputFormat::WindowsPeDll;
        } else if (obj.target.is_macos()) {
            fmt = OutputFormat::MacOSMachODylib;
        } else {
            fmt = OutputFormat::LinuxElfSo;
        }
    }

    if (fmt == OutputFormat::WindowsPeDll) {
        PeDllOptions pe_opts;
        pe_opts.image_base = options.image_base;
        pe_opts.module_name = options.module_name.empty() ? "brass_module.dll" : options.module_name;
        pe_opts.export_all_functions = options.export_all_functions;
        pe_opts.explicit_exports = options.explicit_exports;
        return PeDllWriter::emit(obj, pe_opts);
    } else if (fmt == OutputFormat::MacOSMachODylib) {
        MachODylibOptions macho_opts;
        macho_opts.image_base = 0;
        macho_opts.install_name = !options.soname.empty() ? options.soname : (!options.module_name.empty() ? options.module_name : "brass_module.dylib");
        macho_opts.export_all_functions = options.export_all_functions;
        macho_opts.explicit_exports = options.explicit_exports;
        return MachODylibWriter::emit(obj, macho_opts);
    } else {
        ElfSoOptions elf_opts;
        elf_opts.soname = options.soname.empty() ? options.module_name : options.soname;
        elf_opts.export_all_functions = options.export_all_functions;
        elf_opts.explicit_exports = options.explicit_exports;
        return ElfSoWriter::emit(obj, elf_opts);
    }
}

std::vector<uint8_t> AotLinker::link(const object::ObjectFile& obj) {
    return link(obj, LinkerOptions());
}

std::vector<uint8_t> AotLinker::link(const Module& mod, const Target& target, const LinkerOptions& options) {
    object::ObjectFile obj = object::compile_module_to_object(mod, target);
    return link(obj, options);
}

std::vector<uint8_t> AotLinker::link(const Module& mod, const Target& target) {
    return link(mod, target, LinkerOptions());
}

std::vector<uint8_t> AotLinker::link(const Module& mod, const LinkerOptions& options) {
    return link(mod, Target::host(), options);
}

std::vector<uint8_t> AotLinker::link(const Module& mod) {
    return link(mod, Target::host(), LinkerOptions());
}

bool AotLinker::link_to_file(const object::ObjectFile& obj, const std::string& path, const LinkerOptions& options) {
    LinkerOptions effective_opts = options;
    if (effective_opts.module_name == "brass_module.dll") {
        if (obj.target.is_macos()) {
            effective_opts.module_name = "brass_module.dylib";
        }
    }
    if (!path.empty()) {
        effective_opts.module_name = extract_filename(path);
    }
    std::vector<uint8_t> data = link(obj, effective_opts);
    if (data.empty()) return false;
    bool ok = write_bytes_to_file(path, data);
    if (!ok) return false;

#if defined(__APPLE__)
    if (obj.target.is_macos() || effective_opts.format == OutputFormat::MacOSMachODylib) {
        std::string cmd = "codesign -s - -f \"" + path + "\" > /dev/null 2>&1";
        int res = std::system(cmd.c_str());
        (void)res;
    }
#endif

    return true;
}

bool AotLinker::link_to_file(const object::ObjectFile& obj, const std::string& path) {
    return link_to_file(obj, path, LinkerOptions());
}

bool AotLinker::link_to_file(const Module& mod, const std::string& path, const Target& target, const LinkerOptions& options) {
    object::ObjectFile obj = object::compile_module_to_object(mod, target);
    return link_to_file(obj, path, options);
}

bool AotLinker::link_to_file(const Module& mod, const std::string& path, const Target& target) {
    return link_to_file(mod, path, target, LinkerOptions());
}

bool AotLinker::link_to_file(const Module& mod, const std::string& path, const LinkerOptions& options) {
    return link_to_file(mod, path, Target::host(), options);
}

bool AotLinker::link_to_file(const Module& mod, const std::string& path) {
    return link_to_file(mod, path, Target::host(), LinkerOptions());
}

} // namespace brass::target
