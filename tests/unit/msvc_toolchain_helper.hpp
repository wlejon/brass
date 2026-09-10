#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <iostream>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace brass::test {

class MsvcToolchain {
public:
    static std::string find_vcvars64() {
#if defined(_WIN32)
        static const std::vector<std::string> candidates = {
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\Auxiliary\\Build\\vcvars64.bat",
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Auxiliary\\Build\\vcvars64.bat",
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat",
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat",
            "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Professional\\VC\\Auxiliary\\Build\\vcvars64.bat"
        };
        for (const auto& path : candidates) {
            if (std::filesystem::exists(path)) {
                return path;
            }
        }
#endif
        return "";
    }

    static bool is_available() {
#if defined(_WIN32)
        if (!find_vcvars64().empty()) {
            return true;
        }
        return (std::system("where cl.exe >nul 2>nul") == 0) && (std::system("where link.exe >nul 2>nul") == 0);
#else
        return false;
#endif
    }

    static std::filesystem::path temp_dir() {
        auto p = std::filesystem::temp_directory_path() / "brass_msvc_tests";
        std::filesystem::create_directories(p);
        return p;
    }

    static int run_msvc_cmd(const std::string& cmd_line) {
        auto runner_bat = temp_dir() / "msvc_runner.bat";
        runner_bat.make_preferred();
        std::string vcvars = find_vcvars64();
        {
            std::ofstream ofs(runner_bat);
            if (!vcvars.empty()) {
                ofs << "@call \"" << vcvars << "\" >nul 2>nul\n";
            }
            ofs << "@" << cmd_line << "\n";
        }
        std::string invoke_cmd = "\"" + runner_bat.string() + "\"";
        int code = std::system(invoke_cmd.c_str());
        if (code != 0) {
            std::cerr << "[MSVC Toolchain CMD FAILED (exit " << code << ")]: " << cmd_line << "\n";
        }
        return code;
    }
};

} // namespace brass::test
