#pragma once

#include <brass/codegen/lir.hpp>
#include <brass/target/calling_conv.hpp>
#include <cstdint>
#include <vector>

// Shrink-wrapping (x64, System V and Mach-O): a function whose paths from
// entry to a return need no frame runs them with none. The frame (push rbp,
// mov rbp rsp, sub rsp, the callee-saved saves) is built at the top of each
// save point instead of at the entry, and torn down on every edge that
// leaves the region the save point dominates; a return outside every region
// is a bare `ret`.
//
// A region is the dominator subtree of its save point. Every block that
// needs the frame (a call, a safepoint, a guard exit, a spill slot, a
// callee-saved register, rbp or an rsp-based address) lies in one; a save
// point is in no loop, so each region is set up at most once per call, and
// is always entered without a frame (an edge from another region tears that
// one down first). Where the frame exists its layout is the prologue's, so
// stack maps, exception tables, the frame-pointer chain and every
// rbp-relative slot read the same as in an unwrapped function: they are
// only ever consulted at a call, and every call is inside a region. The
// unwind information (elf_cfi.cpp, from CompilationResult::frame_rows)
// states the frame at every instruction, so a signal anywhere unwinds.
// Win64 (whose unwind codes describe one prologue) and AArch64 keep the
// prologue at entry.
//
// Two halves share choose_frame_regions:
// - split_for_shrink_wrap, before register allocation, picks regions from
//   where the calls are and splits every value live into or out of a region
//   at its boundary (a copy at the save point's top, copies back on the exit
//   edges), so a value used on the frameless path and across a call in the
//   region gets a caller-saved register on the one and a callee-saved
//   register or slot on the other.
// - The x64 emitter (emit_context.cpp) decides afresh from the allocated
//   code, with exactly what each block names; it wraps only what that proves
//   sound, whatever the first half did.
namespace brass::codegen {

// A control-flow graph by node index; node 0 is the entry.
struct FrameGraph {
    std::vector<std::vector<uint32_t>> succs;
    std::vector<uint8_t> cold;
    std::vector<uint8_t> needs_frame;
    std::vector<uint8_t> returns;  // the node returns to the caller
};

struct FrameRegions {
    static constexpr uint32_t kNone = UINT32_MAX;
    bool wrapped = false;
    // Per node: the save point node of the region holding it, or kNone (no
    // frame). Nodes the entry does not reach get the region kNone too.
    std::vector<uint32_t> region;
    std::vector<uint32_t> saves;
};

// The regions of `g`, or wrapped = false when the entry needs the frame, a
// region would have to start in a loop, no return is reachable without one,
// or there would be more than kMaxRegions.
FrameRegions choose_frame_regions(const FrameGraph& g);
constexpr size_t kMaxFrameRegions = 16;

// Whether the function's shape allows shrink-wrapping at all: x64 outside
// Win64, no landing pads, invokes or resume entries, no locals, stack
// arguments or forced saves, no push/pop, no label used but by a branch.
bool shrink_wrap_shape_ok(const LirFunction& fn, const CallingConvention& cc);

// The general registers code without a frame may not name: rbp and every
// callee-saved register but those the function reserves (the pinned TLS
// register is the caller's, and only read).
uint32_t frame_gpr_mask(const LirFunction& fn, const CallingConvention& cc);

// Whether `inst` needs the frame: a call, safepoint, guard exit, push or pop;
// a register of `mask` named or clobbered; rsp written; a spill slot (but
// those in `free_slots`, sorted: the constant pool's), a local, or an
// address on rbp, rsp or nothing. A read of rsp itself does not: without a
// frame the emitter reads it as the prologue would have left it.
bool inst_needs_frame(const LirInst& inst, uint32_t mask, const std::vector<int32_t>& free_slots);
bool reads_rsp(const LirInst& inst);

// Before register allocation (on virtual registers). Returns whether it
// changed the function.
bool split_for_shrink_wrap(LirFunction& fn, const CallingConvention& cc);

} // namespace brass::codegen
