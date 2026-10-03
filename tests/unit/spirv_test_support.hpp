#pragma once

// Shared helpers for the SPIR-V target tests: spirv-val / spirv-dis on the
// emitted binary (visible [SKIP] when the tools are missing), compile +
// validate in one call, and an expected-diagnostic check.

#include "test_framework.hpp"

#include <brass/codegen/kernel_jit.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/target/spirv/spirv_isel.hpp>
#include <brass/target/spirv_target.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef BRASS_SPIRV_VAL
#define BRASS_SPIRV_VAL ""
#endif
#ifndef BRASS_SPIRV_DIS
#define BRASS_SPIRV_DIS ""
#endif

namespace spvtest {

inline void report_skip(const char* what) {
    std::cout << "  [SKIP] " << what << "\n" << std::flush;
}

inline std::string tool(const char* configured, const char* name) {
    std::string t = configured;
    return t.empty() ? std::string(name) : t;
}

inline std::string redirect(const std::filesystem::path& out) {
    return " >\"" + out.string() + "\" 2>&1";
}

inline bool tool_available(const char* configured, const char* name) {
    std::filesystem::path log = brass::test::scratch_dir() / "spirv_tool_probe.txt";
    return std::system(("\"" + tool(configured, name) + "\" --version" + redirect(log)).c_str()) == 0;
}

inline bool spirv_val_available() {
    static const bool ok = tool_available(BRASS_SPIRV_VAL, "spirv-val");
    if (!ok) report_skip("spirv-val not found: SPIR-V modules not validated against the Vulkan rules");
    return ok;
}

inline bool spirv_dis_available() {
    static const bool ok = tool_available(BRASS_SPIRV_DIS, "spirv-dis");
    if (!ok) report_skip("spirv-dis not found: disassembly not checked");
    return ok;
}

inline std::string read_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

inline std::filesystem::path write_spv(const std::vector<uint32_t>& words, const char* name) {
    std::filesystem::path p = brass::test::scratch_dir() / name;
    std::ofstream f(p, std::ios::binary);
    f.write(reinterpret_cast<const char*>(words.data()), static_cast<std::streamsize>(words.size() * 4));
    return p;
}

// spirv-val --target-env vulkan1.2 (vulkan1.3 for SPIR-V 1.6). True when the
// module is valid or the tool is missing; `log` gets the tool's output.
inline bool spirv_val(const std::vector<uint32_t>& words, std::string* log = nullptr) {
    if (!spirv_val_available()) return true;
    const char* env = (words.size() > 1 && words[1] >= 0x00010600) ? "vulkan1.3" : "vulkan1.2";
    std::filesystem::path in = write_spv(words, "brass_spirv_check.spv");
    std::filesystem::path out = brass::test::scratch_dir() / "brass_spirv_check.txt";
    std::string cmd = "\"" + tool(BRASS_SPIRV_VAL, "spirv-val") + "\" --target-env " + env + " \"" + in.string() + "\"" +
                      redirect(out);
    bool ok = std::system(cmd.c_str()) == 0;
    if (log) *log = read_file(out);
    return ok;
}

// spirv-dis text, or "" when the tool is missing.
inline std::string disassemble(const std::vector<uint32_t>& words) {
    if (!spirv_dis_available()) return "";
    std::filesystem::path in = write_spv(words, "brass_spirv_dis.spv");
    std::filesystem::path out = brass::test::scratch_dir() / "brass_spirv_dis.txt";
    std::string cmd = "\"" + tool(BRASS_SPIRV_DIS, "spirv-dis") + "\" --raw-id \"" + in.string() + "\"" + redirect(out);
    if (std::system(cmd.c_str()) != 0) return "";
    return read_file(out);
}

// Requires spirv-val to accept `words`; prints the tool output and the
// module dump otherwise.
inline void require_valid(const std::vector<uint32_t>& words, const std::string& dump = "") {
    std::string log;
    bool ok = spirv_val(words, &log);
    if (!ok) std::cerr << "spirv-val:\n" << log << "\n" << dump << "\n";
    REQUIRE(ok);
}

// Compile through the facade (ISel + spirv::verify) and spirv-val the result.
inline brass::target::SpirvKernel compile_checked(const brass::Function& fn, const brass::target::SpirvOptions& opts = {}) {
    brass::target::SpirvKernel k;
    try {
        k = brass::target::SpirvTarget::compile(fn, opts);
    } catch (const std::exception& e) {
        std::cerr << "SpirvTarget::compile threw: " << e.what() << "\n";
        REQUIRE(false);
    }
    std::string log;
    if (!spirv_val(k.words, &log)) {
        std::cerr << "spirv-val:\n" << log << "\n" << brass::target::SpirvTarget::dump_function(fn, opts) << "\n";
        REQUIRE(false);
    }
    return k;
}

// The text listing of the lowered module (spirv::dump).
inline std::string dump_of(const brass::Function& fn, const brass::target::SpirvOptions& opts = {}) {
    return brass::target::SpirvTarget::dump_function(fn, opts);
}

inline bool has(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

inline size_t count_of(const std::string& text, const std::string& needle) {
    size_t n = 0;
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size())) ++n;
    return n;
}

// Requires compile() to throw a std::runtime_error whose message contains
// every one of `needles`.
inline void require_diagnostic(const brass::Function& fn, std::initializer_list<const char*> needles) {
    std::string msg;
    try {
        brass::target::SpirvTarget::compile(fn);
    } catch (const std::runtime_error& e) {
        msg = e.what();
    }
    if (msg.empty()) std::cerr << "expected a diagnostic, compile succeeded\n";
    REQUIRE(!msg.empty());
    for (const char* n : needles) {
        if (!has(msg, n)) std::cerr << "diagnostic '" << msg << "' lacks '" << n << "'\n";
        CHECK(has(msg, n));
    }
}

// A void kernel with the given parameters; `body` builds from the entry
// block (its parameters are the kernel's) and must end in a terminator.
struct Kernel {
    brass::Module mod{"m"};
    brass::Function* fn = nullptr;
    brass::Builder b{mod};
    brass::BasicBlock* entry = nullptr;
    std::vector<brass::Value*> params;

    explicit Kernel(std::vector<brass::Type> types, const char* name = "k") {
        fn = mod.create_function(name, brass::Type::void_type(), types);
        b.set_function(fn);
        entry = b.append_block("entry");
        b.position_at_end(entry);
        for (brass::Type t : types) params.push_back(b.add_block_param(entry, t));
    }
    brass::Value* p(size_t i) const { return params.at(i); }
};

} // namespace spvtest
