#pragma once

#include <brass/mir/module.hpp>
#include <brass/core/diagnostics.hpp>
#include <memory>
#include <string_view>
#include <string>

namespace brass::il {

struct TranslatorOptions {
    bool enable_optimizations = true;
    bool allow_fp_reassociation = false;
    bool trace_lowering = false;
};

struct TranslationResult {
    bool success = false;
    std::unique_ptr<Module> module;
    std::string error_message;
};

// Translate Bronze textual IL directly to a Brass MIR Module
TranslationResult translate_bronze_il(
    std::string_view il_text,
    const TranslatorOptions& options = {},
    DiagnosticReporter* diag = nullptr
);

// Register Bronze runtime helper symbols into a JitExecutionEngine or runtime symbol table
void register_bronze_runtime_symbols(void* jit_engine_ptr);

// Runtime helper for printing numbers / dynamic values from Bronze print instructions
extern "C" void bronze_print_f64(double v);
extern "C" void bronze_print_i32(int32_t v);
extern "C" void bronze_print_dynamic(int64_t v);
extern "C" void bronze_print_newline();
extern "C" double bronze_f64_mod(double a, double b);

} // namespace brass::il
