#pragma once

#include <brass/core/arena.hpp>
#include <brass/core/bitset.hpp>
#include <brass/core/diagnostics.hpp>
#include <brass/core/span.hpp>
#include <brass/core/string_pool.hpp>

#include <brass/mir/types.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/call_graph.hpp>
#include <brass/mir/devirtualize.hpp>
#include <brass/mir/inline_transform.hpp>
#include <brass/mir/inliner.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/memory_ssa.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/lexer.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>

#include <brass/interpreter/value.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/interpreter/frame.hpp>
#include <brass/interpreter/interpreter.hpp>

#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/target/x64/x64_registers.hpp>
#include <brass/target/x64/x64_operands.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/jit_exec.hpp>

#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/elf_writer.hpp>

#include <brass/runtime/deopt.hpp>
#include <brass/runtime/resume_table.hpp>
#include <brass/runtime/patcher.hpp>

#include <brass/embedding/nanbox.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/embedding/embedding.hpp>
#include <brass/embedding/brass_c_api.h>

#include <string_view>

namespace brass {

constexpr int version_major() noexcept { return 0; }
constexpr int version_minor() noexcept { return 1; }
constexpr int version_patch() noexcept { return 0; }
constexpr std::string_view version_string() noexcept { return "0.1.0"; }

} // namespace brass
