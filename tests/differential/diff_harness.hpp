#pragma once

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/loop_opt.hpp>
#include <vector>
#include <string_view>
#include <iostream>
#include <cmath>
#include <iomanip>

namespace brass::test {

inline void assert_diff(
    const Module& mod,
    std::string_view fn_name,
    const std::vector<RuntimeValue>& args
) {
    DiagnosticReporter ver_diag;
    bool ok_ver = verify_module(mod, &ver_diag);
    if (!ok_ver) {
        std::cerr << "Module verification error in " << fn_name << ":\n" << ver_diag.format_all() << "\n";
        print_module(mod, std::cerr);
    }
    REQUIRE(ok_ver);

    Interpreter interp;
    RuntimeValue interp_res = interp.run(mod, fn_name, args);

    codegen::JitExecutionEngine jit(Target::host());
    bool ok_jit = jit.compile_and_load(mod);
    if (!ok_jit) {
        std::cerr << "JIT Compilation error in " << fn_name << "\n";
    }
    REQUIRE(ok_jit);

    RuntimeValue jit_res = jit.invoke(fn_name, args);

    if (interp_res.is_f64()) {
        double d1 = interp_res.as_f64();
        double d2 = jit_res.as_f64();
        if (std::isnan(d1)) {
            CHECK(std::isnan(d2));
        } else {
            CHECK(std::abs(d1 - d2) < 1e-6);
        }
    } else if (interp_res.is_f32()) {
        float f1 = interp_res.as_f32();
        float f2 = jit_res.as_f32();
        if (std::isnan(f1)) {
            CHECK(std::isnan(f2));
        } else {
            if (std::abs(f1 - f2) >= 1e-6f) {
                std::cout << "MISMATCH f32 in " << fn_name << ": interp=" << f1 << ", jit=" << f2 << "\n";
            }
            CHECK(std::abs(f1 - f2) < 1e-6f);
        }
    } else if (interp_res.is_vector()) {
        if (interp_res != jit_res) {
            std::cout << "MISMATCH in " << fn_name << "\n";
            std::cout << "Interp: " << to_string(interp_res) << " JIT: " << to_string(jit_res) << "\n";
            print_module(mod, std::cout);
        }
        CHECK(interp_res == jit_res);
    } else {
        if (interp_res.raw_bits() != jit_res.raw_bits()) {
            std::cout << "MISMATCH in " << fn_name << "\n";
            std::cout << "Interp: " << interp_res.raw_bits() << " JIT: " << jit_res.raw_bits() << "\n";
            print_module(mod, std::cout);
        }
        CHECK_EQ(interp_res.raw_bits(), jit_res.raw_bits());
    }
}

inline void assert_diff_3way(
    const Module& mod,
    std::string_view fn_name,
    const std::vector<RuntimeValue>& args,
    const LoopOptOptions& opt_opts = {}
) {
    DiagnosticReporter ver_diag;
    bool ok_ver = verify_module(mod, &ver_diag);
    if (!ok_ver) {
        std::cerr << "Module verification error in " << fn_name << ":\n" << ver_diag.format_all() << "\n";
        print_module(mod, std::cerr);
    }
    REQUIRE(ok_ver);

    // 1. Interpreter execution
    Interpreter interp;
    RuntimeValue interp_res = interp.run(mod, fn_name, args);

    // 2. JIT unoptimized execution
    codegen::JitExecutionEngine jit_unopt(Target::host());
    bool ok_jit = jit_unopt.compile_and_load(mod);
    if (!ok_jit) {
        std::cerr << "JIT Compilation error (unoptimized) in " << fn_name << "\n";
    }
    REQUIRE(ok_jit);
    RuntimeValue jit_unopt_res = jit_unopt.invoke(fn_name, args);

    // 3. JIT optimized execution with verifier validation
    auto opt_mod = clone_module(mod);
    REQUIRE(opt_mod != nullptr);

    optimize_module_loops(*opt_mod, opt_opts);

    DiagnosticReporter opt_ver_diag;
    bool ok_opt_ver = verify_module(*opt_mod, &opt_ver_diag);
    if (!ok_opt_ver) {
        std::cerr << "Optimized module verification error in " << fn_name << ":\n" << opt_ver_diag.format_all() << "\n";
        print_module(*opt_mod, std::cerr);
    }
    REQUIRE(ok_opt_ver);

    codegen::JitExecutionEngine jit_opt(Target::host());
    bool ok_jit_opt = jit_opt.compile_and_load(*opt_mod);
    if (!ok_jit_opt) {
        std::cerr << "JIT Compilation error (optimized) in " << fn_name << "\n";
    }
    REQUIRE(ok_jit_opt);
    RuntimeValue jit_opt_res = jit_opt.invoke(fn_name, args);

    // 4. Bit-exact equivalence verification across all three
    if (interp_res.is_f64()) {
        double d_interp = interp_res.as_f64();
        double d_unopt = jit_unopt_res.as_f64();
        double d_opt = jit_opt_res.as_f64();

        if (std::isnan(d_interp)) {
            CHECK(std::isnan(d_unopt));
            CHECK(std::isnan(d_opt));
        } else {
            if (interp_res.raw_bits() != jit_unopt_res.raw_bits() ||
                jit_unopt_res.raw_bits() != jit_opt_res.raw_bits()) {
                std::cerr << "FLOAT BIT MISMATCH in " << fn_name << ":\n"
                          << "  Interp:    " << d_interp << " (0x" << std::hex << interp_res.raw_bits() << ")\n"
                          << "  JIT Unopt: " << d_unopt << " (0x" << jit_unopt_res.raw_bits() << ")\n"
                          << "  JIT Opt:   " << d_opt << " (0x" << jit_opt_res.raw_bits() << std::dec << ")\n";
                print_module(mod, std::cerr);
                print_module(*opt_mod, std::cerr);
            }
            CHECK_EQ(interp_res.raw_bits(), jit_unopt_res.raw_bits());
            CHECK_EQ(jit_unopt_res.raw_bits(), jit_opt_res.raw_bits());
        }
    } else {
        if (interp_res.raw_bits() != jit_unopt_res.raw_bits() ||
            jit_unopt_res.raw_bits() != jit_opt_res.raw_bits()) {
            std::cerr << "INTEGER/POINTER BIT MISMATCH in " << fn_name << ":\n"
                      << "  Interp:    0x" << std::hex << interp_res.raw_bits() << "\n"
                      << "  JIT Unopt: 0x" << jit_unopt_res.raw_bits() << "\n"
                      << "  JIT Opt:   0x" << jit_opt_res.raw_bits() << std::dec << "\n";
            print_module(mod, std::cerr);
            print_module(*opt_mod, std::cerr);
        }
        CHECK_EQ(interp_res.raw_bits(), jit_unopt_res.raw_bits());
        CHECK_EQ(jit_unopt_res.raw_bits(), jit_opt_res.raw_bits());
    }
}

} // namespace brass::test
