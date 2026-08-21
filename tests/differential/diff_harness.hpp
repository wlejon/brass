#pragma once

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <vector>
#include <string_view>
#include <iostream>
#include <cmath>

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
    } else {
        if (interp_res.raw_bits() != jit_res.raw_bits()) {
            std::cout << "MISMATCH in " << fn_name << "\n";
            std::cout << "Interp: " << interp_res.raw_bits() << " JIT: " << jit_res.raw_bits() << "\n";
            print_module(mod, std::cout);
        }
        CHECK_EQ(interp_res.raw_bits(), jit_res.raw_bits());
    }
}

} // namespace brass::test
