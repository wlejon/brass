#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <cstdint>

namespace brass {

struct CfgSimplifyStats {
    size_t branches_simplified = 0;
    size_t blocks_removed = 0;
    size_t blocks_merged = 0;
    size_t params_removed = 0;
    size_t trampolines_eliminated = 0;
};

struct CfgSimplifyOptions {
    bool enable_branch_simplify = true;
    bool enable_dead_block_removal = true;
    bool enable_block_merge = true;
    bool enable_param_elimination = true;
    bool enable_trampoline_elimination = true;
    size_t max_iterations = 16;
    CfgSimplifyStats* stats = nullptr;
};

// CFG Simplification & Dead Block Compaction on a single function
bool cfg_simplify_function(Function& fn);
bool cfg_simplify_function(Function& fn, const CfgSimplifyOptions& options);

// CFG Simplification across an entire module
bool cfg_simplify_module(Module& mod);
bool cfg_simplify_module(Module& mod, const CfgSimplifyOptions& options);

} // namespace brass
