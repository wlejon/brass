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
#include <unordered_map>
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

// The order compile() places `fns` in (indices into it), as `placement` asks
// (object_writer.hpp, FunctionPlacement).
std::vector<size_t> placement_order(const std::vector<const Function*>& fns, const FunctionPlacement& placement);

// Appends BRASS_BLOCK_MAP's lines (the diagnostic's file, in function order).
void write_block_map(const std::string& lines);

// BRANCH ISLANDS. An AArch64 B or BL reaches +-128 MB (the reach given). A
// module whose code is larger gets an island every half reach of .text: for
// every symbol the module's branches name, a veneer
//   adrp x16, :got:target ; ldr x16, [x16, :got_lo12:target] ; br x16
// which the image writers relax to adrp / add for a symbol the object
// defines (relax_got_loads). Once every function is placed, each branch
// whose target lies out of reach is pointed at the nearest island's veneer
// for that target, its displacement encoded in place and its relocation
// dropped. x16 is IP0, which AAPCS64 lets a veneer clobber between a call
// and its callee. Islands sit between functions and are placed before any
// table records an offset (the stack maps are encoded after placement), so
// nothing else moves. Inactive on x64 (rel32 reaches 2 GB) and for a module
// whose code fits in reach.
class BranchIslands {
public:
    BranchIslands(const Target& target, const std::vector<EmittedFunction>& emitted, const std::vector<size_t>& order,
                  uint64_t reach);
    // Before the next function is placed: an island, when .text has passed
    // the next mark.
    void before_function(ObjectPlacement& at);
    // After the last function: the branches out of reach, through islands.
    void retarget(ObjectPlacement& at);
    bool active() const noexcept { return active_; }
    size_t islands() const noexcept { return islands_.size(); }

private:
    void emit_island(ObjectPlacement& at);

    bool active_ = false;
    uint64_t reach_ = 0;
    uint64_t spacing_ = 0;
    uint64_t next_mark_ = 0;
    std::vector<std::string> targets_;  // sorted: the islands' bytes do not depend on thread order
    std::unordered_map<std::string, uint32_t> target_index_;
    std::vector<uint64_t> islands_;     // the .text offset of each island's first veneer
};

} // namespace brass::object::detail
