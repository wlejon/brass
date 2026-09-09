#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>

namespace brass {

struct JumpThreadingOptions {
    size_t max_block_size = 20;
    size_t max_iterations = 4;
};

struct JumpThreadingStats {
    size_t edges_threaded = 0;
};

// Run SSA Jump Threading on a single function.
bool run_jump_threading(Function& fn, const JumpThreadingOptions& opts, JumpThreadingStats* stats = nullptr);
bool run_jump_threading(Function& fn);

// Run SSA Jump Threading across all functions in a module.
bool jump_thread_module(Module& mod, const JumpThreadingOptions& opts, JumpThreadingStats* stats = nullptr);
bool jump_thread_module(Module& mod);

} // namespace brass
