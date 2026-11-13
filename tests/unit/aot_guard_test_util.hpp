// Helpers of the AOT guard tests: a module parsed, compiled to an object,
// linked into this host's shared image and loaded into the test process.
#pragma once

#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/aot_linker.hpp>
#include <brass/target/dynamic_library.hpp>

#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace brass::test_aot_guard {

inline std::unique_ptr<Module> parse_checked(std::string_view text) {
    DiagnosticReporter diag;
    auto mod = parse_module(text, &diag);
    if (!mod) {
        std::cerr << diag.format_all() << "\n";
        return nullptr;
    }
    DiagnosticReporter vdiag;
    if (!verify_module(*mod, &vdiag)) {
        std::cerr << vdiag.format_all() << "\n";
        return nullptr;
    }
    return mod;
}

struct LoadedImage {
    object::ObjectFile obj;
    // Symbols the object leaves to the loader.
    std::vector<std::string> undefined;
    std::string error;
    std::filesystem::path path;
    std::unique_ptr<target::DynamicLibrary> lib;

    ~LoadedImage() {
        lib.reset();
        std::error_code ec;
        if (!path.empty()) std::filesystem::remove(path, ec);
    }
    LoadedImage() = default;
    LoadedImage(LoadedImage&&) = default;
    LoadedImage& operator=(LoadedImage&&) = default;

    const object::CompiledFunctionInfo* function(std::string_view name) const {
        for (const auto& f : obj.functions) {
            if (f.name == name) return &f;
        }
        return nullptr;
    }
};

inline LoadedImage link_and_load(const Module& mod, const std::string& stem,
                                 const DeoptStressPlan* stress = nullptr,
                                 const std::vector<target::ImportLibrary>& imports = {}) {
    LoadedImage img;
    const Target host = Target::host();
    object::ModuleCompiler compiler(host);
    if (stress) compiler.set_deopt_stress(*stress);
    try {
        img.obj = compiler.compile(mod);
    } catch (const std::exception& e) {
        img.error = e.what();
        std::cerr << "link_and_load(" << stem << "): " << img.error << "\n";
        return img;
    }
    for (const auto& sym : img.obj.symbols) {
        if (sym.section_index == object::SECTION_UNDEF && !sym.name.empty()) img.undefined.push_back(sym.name);
    }
    const std::string ext = host.is_windows() ? ".dll" : (host.is_macos() ? ".dylib" : ".so");
    img.path = brass::test::scratch_dir() / (stem + ext);
    std::error_code ec;
    std::filesystem::remove(img.path, ec);
    target::LinkerOptions opts;
    opts.module_name = stem + ext;
    opts.export_all_functions = true;
    opts.imports = imports;
    if (target::AotLinker::link_to_file(img.obj, img.path.string(), opts, &img.error)) {
        img.lib = target::DynamicLibrary::open(img.path.string(), &img.error);
    }
    if (!img.lib) {
        std::cerr << "link_and_load(" << stem << "): " << img.error;
        for (const auto& u : img.undefined) std::cerr << " [undefined " << u << "]";
        std::cerr << "\n";
    }
    return img;
}

} // namespace brass::test_aot_guard
