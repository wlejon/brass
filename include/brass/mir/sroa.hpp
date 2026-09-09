#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <cstdint>

namespace brass {

struct SroaStats {
    size_t allocations_eliminated = 0;
    size_t loads_eliminated = 0;
    size_t stores_eliminated = 0;
    size_t block_params_created = 0;
};

struct SroaOptions {
    size_t max_fields_per_object = 32;
    bool allow_partial_sroa = false;
    bool enable_dce_after = true;
    SroaStats* stats = nullptr;
};

// Scalar replacement of aggregates on a single function
bool sroa_function(Function& fn);
bool sroa_function(Function& fn, const SroaOptions& options);

// Scalar replacement of aggregates across an entire module
bool sroa_module(Module& mod);
bool sroa_module(Module& mod, const SroaOptions& options);

} // namespace brass
