#pragma once

// One function's code generation for ModuleCompiler::compile (object_writer.cpp):
// the optional per-function MIR loop pipeline, instruction selection,
// scheduling, register allocation, layout, peephole and emission, into a
// result that names no position in the object. compile() runs this for many
// functions at once and then places the results in module order
// (place_function), which is where every module-wide offset is decided.
//
// compile_function reads the module and the function and writes only its
// result: what it builds (LIR, a private module for a function it rewrites)
// lives and dies on the calling thread, as the ScratchPool requires.
#include <brass/object/object_writer.hpp>
#include <brass/mir/pass_manager.hpp>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace brass::object::detail {

struct FunctionCodegenConfig {
    Target target = Target::host();
    CallingConvention cc;
    codegen::SchedOptions sched;
    DeoptStressPlan deopt_stress;
    bool enable_trace_layout = false;
    bool x64_avx = false;
    // The loop pipeline per FP-reassociation setting, or null for none (no
    // MIR optimization here: off, or the module's loops are optimized).
    const Pipeline* loop_pipes[2] = {nullptr, nullptr};
};

struct EmittedFunction {
    bool emitted = false;  // false: instruction selection produced nothing
    uint32_t alignment = 16;
    std::vector<uint8_t> code;
    // Offsets from the function's first byte.
    std::vector<ObjectRelocation> relocations;
    // Everything but the placement: text_offset, the stack map's and the
    // exception table's code offsets, and the patch sites' offsets are
    // relative to the function until place_function moves them.
    CompiledFunctionInfo info;
    FunctionDebugTable debug_table;
    // BRASS_BLOCK_MAP's lines for the function, when that diagnostic is on.
    std::string block_map;
};

EmittedFunction compile_function(const Module& mod, const Function& fn, const FunctionCodegenConfig& cfg);

// The section ".text" and the per-function tables of `obj`; `mapped` keeps
// the names whose stack maps were appended without a search.
struct ObjectPlacement {
    ObjectFile& obj;
    Section& text;
    std::unordered_set<std::string>& mapped;
};

// Appends `fn` at the end of .text (aligned as it asks) and registers its
// symbol, tables and relocations at that offset.
void place_function(ObjectPlacement& at, EmittedFunction&& fn);

// Appends BRASS_BLOCK_MAP's lines (the diagnostic's file, in function order).
void write_block_map(const std::string& lines);

} // namespace brass::object::detail
