#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <iosfwd>

namespace brass {

struct GvnPreStats {
    size_t expressions_hoisted = 0;
    size_t expressions_eliminated = 0;
    size_t critical_edges_split = 0;
    size_t block_params_inserted = 0;

    std::string format_report() const;
    void dump(std::ostream& os) const;
};

struct GvnPreOptions {
    bool enable_pre = true;
    bool enable_load_pre = true;
    bool enable_critical_edge_splitting = true;
    size_t max_iterations = 4;
    GvnPreStats* stats = nullptr;
};

// Global Value Numbering with Partial Redundancy Elimination on a single function
bool gvn_pre_function(Function& fn);
bool gvn_pre_function(Function& fn, const GvnPreOptions& options);

// Global Value Numbering with Partial Redundancy Elimination across an entire module
bool gvn_pre_module(Module& mod);
bool gvn_pre_module(Module& mod, const GvnPreOptions& options);

} // namespace brass
