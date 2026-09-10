#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <string>

namespace brass {

struct FmaOptStats {
    size_t scalar_fma_f32_count = 0;
    size_t scalar_fma_f64_count = 0;
    size_t vector_vfma_count = 0;

    size_t total_fused() const {
        return scalar_fma_f32_count + scalar_fma_f64_count + vector_vfma_count;
    }

    std::string format_report() const;
};

struct FmaOptOptions {
    bool enable_scalar = true;
    bool enable_vector = true;
    FmaOptStats* stats = nullptr;
};

// Optimizes scalar and vector floating-point multiplications followed by additions into FMA instructions:
//   scalar: a * b + c  -> fma_f32 / fma_f64
//   vector: a * b + c  -> vfma
bool fma_opt_pass(Function& fn, const FmaOptOptions& options = {});
bool fma_opt_module_pass(Module& mod, const FmaOptOptions& options = {});

// Helper function
bool run_fma_opt(Function& fn, FmaOptStats* stats = nullptr);

} // namespace brass
