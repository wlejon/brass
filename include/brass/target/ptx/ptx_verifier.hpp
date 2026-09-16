#pragma once

// PtxVerifier: structural checks on a ptx::Function that run on any platform
// without ptxas. See docs/ptx_backend_design.md for the required check list.

#include <brass/target/ptx/ptx_ir.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace brass::ptx {

struct Diagnostic {
    static constexpr size_t kNoInst = static_cast<size_t>(-1);

    std::string function;
    std::string block;              // empty for function-level diagnostics
    size_t inst_index = kNoInst;    // index within the block, or kNoInst
    std::string inst_text;          // printed instruction, empty for function-level
    std::string message;
};

// "fn:$L_entry[3] 'add.f32 %f0, %r0, %f1;': source 0 is %r0 (b32) but .f32 requires an f32 register"
std::string to_string(const Diagnostic& d);
std::string format_diagnostics(const std::vector<Diagnostic>& diags);

// Returns every problem found; an empty vector means the function is well-formed.
std::vector<Diagnostic> verify(const Function& fn);

} // namespace brass::ptx
