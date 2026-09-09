#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>

namespace brass {

// Devirtualizes a single patchable_call instruction to a direct call if the target callee
// is defined as a function within the module.
bool devirtualize_call(Instruction* inst, Module& mod);

// Devirtualizes all monomorphic patchable_call sites in a function to direct calls.
bool devirtualize_function(Function& fn, Module& mod);

// Devirtualizes all monomorphic patchable_call sites across an entire module.
bool devirtualize_module(Module& mod);

} // namespace brass
