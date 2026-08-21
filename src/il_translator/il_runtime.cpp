#include "il_runtime.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>

namespace brass::il {

extern "C" {

void bronze_print_f64(double v) {
    std::cout << v << " ";
}

void bronze_print_i32(int32_t v) {
    std::cout << v << " ";
}

void bronze_print_dynamic(int64_t v) {
    uint64_t u = static_cast<uint64_t>(v);
    if (u < 0xFFF8000000000000ULL) {
        double d;
        std::memcpy(&d, &v, sizeof(double));
        std::cout << d << " ";
    } else if ((u >> 48) == 0xFFF9) {
        int32_t iv = static_cast<int32_t>(u & 0xFFFFFFFFULL);
        std::cout << iv << " ";
    } else if ((u >> 48) == 0xFFFA) {
        std::cout << "null ";
    } else if ((u >> 48) == 0xFFFB) {
        bool b = (u & 1) != 0;
        std::cout << (b ? "true " : "false ");
    } else {
        std::cout << "undefined ";
    }
}

void bronze_print_newline() {
    std::cout << "\n";
}

double bronze_f64_mod(double a, double b) {
    return std::fmod(a, b);
}

} // extern "C"

void register_all_runtime_symbols(codegen::JitExecutionEngine& jit) {
    jit.register_external_symbol("bronze_print_f64", reinterpret_cast<void*>(&bronze_print_f64));
    jit.register_external_symbol("bronze_print_i32", reinterpret_cast<void*>(&bronze_print_i32));
    jit.register_external_symbol("bronze_print_dynamic", reinterpret_cast<void*>(&bronze_print_dynamic));
    jit.register_external_symbol("bronze_print_newline", reinterpret_cast<void*>(&bronze_print_newline));
    jit.register_external_symbol("bronze_f64_mod", reinterpret_cast<void*>(&bronze_f64_mod));
}

void register_bronze_runtime_symbols(void* jit_engine_ptr) {
    if (jit_engine_ptr) {
        auto* jit = reinterpret_cast<codegen::JitExecutionEngine*>(jit_engine_ptr);
        register_all_runtime_symbols(*jit);
    }
}

} // namespace brass::il
