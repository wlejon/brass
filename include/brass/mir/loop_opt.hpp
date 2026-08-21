#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <memory>

namespace brass {

struct LoopOptOptions {
    bool enable_licm = true;
    bool enable_ivsr = true;
    bool enable_dce = true;
    size_t max_iterations = 8;
};

// Optimize loops in a single function (LICM, IVSR, Constant Folding, DCE)
bool optimize_function_loops(Function& fn, const LoopOptOptions& options = {});

// Optimize loops in an entire module
bool optimize_module_loops(Module& mod, const LoopOptOptions& options = {});

// Helper to clone a function into a target module
Function* clone_function(const Function& src, Module& dst_mod);

// Helper to clone an entire module
std::unique_ptr<Module> clone_module(const Module& src);

} // namespace brass
