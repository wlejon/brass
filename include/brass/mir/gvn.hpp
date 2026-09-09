#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <cstdint>

namespace brass {

struct GvnStats {
    size_t expressions_eliminated = 0;
    size_t loads_forwarded = 0;
    size_t loads_eliminated = 0;
    size_t dead_stores_eliminated = 0;
};

struct GvnOptions {
    bool enable_cse = true;
    bool enable_rle = true;
    bool enable_dse = true;
    GvnStats* stats = nullptr;
};

// Global Value Numbering on a single function
bool gvn_function(Function& fn);
bool gvn_function(Function& fn, const GvnOptions& options);

// Global Value Numbering across an entire module
bool gvn_module(Module& mod);
bool gvn_module(Module& mod, const GvnOptions& options);

} // namespace brass
