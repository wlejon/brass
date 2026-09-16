#pragma once

// PtxPrinter: ptx::Function -> PTX text. A single dumb walk that makes no
// decisions beyond formatting. Register counts, params, shared declarations
// and instruction suffixes all come from the IR.

#include <brass/target/ptx/ptx_ir.hpp>
#include <brass/target/ptx_target.hpp>

#include <string>
#include <vector>

namespace brass::ptx {

// Exact-hex float literals as PTX requires them: 0f3F800000 / 0d3FF0000000000000.
std::string format_f32_hex(float f);
std::string format_f64_hex(double d);

// Mnemonic with every modifier/type suffix in the order PTX requires, e.g.
// "ld.global.v4.f32", "cvt.rn.f32.u32", "shfl.sync.down.b32", "mad.lo.u32".
std::string mnemonic(const Inst& inst);

// Formatting of individual pieces (used by the verifier for diagnostics).
std::string to_string(const Operand& op);
std::string to_string(const Inst& inst);   // "@%p1 add.f32 %f0, %f1, %f2;" (no indent, no newline)

// ".version / .target / .address_size" preamble.
std::string print_header(const target::PtxOptions& opts);

// Function body only: signature, declarations, blocks.
std::string print_body(const Function& fn);

// Header + one function.
std::string print(const Function& fn, const target::PtxOptions& opts = {});

// Header + several functions.
std::string print_module(const std::vector<const Function*>& fns, const target::PtxOptions& opts = {});

} // namespace brass::ptx
