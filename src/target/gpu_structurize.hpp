#pragma once

// Shared GPU CFG structurizer: recovers structured control flow from a MIR
// CFG for SPIR-V and MSL (docs/spirv_backend_design.md, "Structurization").
//
// The result is a graph of nodes -- one per reachable MIR block plus
// synthetic ones -- in which every loop has a dedicated header node (with
// OpLoopMerge), a dedicated continue node (the only back-edge source), one
// merge node, and every two-way branch has a selection merge node. Synthetic
// nodes are:
//   prologue       the function's first block (kernel parameter loads); it
//                  branches to the MIR entry
//   forward        an empty block that takes the phis of the block it
//                  branches to and passes them on; used as a loop header in
//                  front of the MIR loop head, and as a fresh merge block
//                  when the natural one is taken or not dominated
//   loop_continue  a forward node that is a loop's continue target; every
//                  back edge is redirected to it
//   unreachable    an OpUnreachable merge for a construct that never
//                  reconverges (every arm returns, breaks or continues)
//
// Edges keep the MIR branch target they came from (its arguments feed the
// phis of whatever node the edge ends at after redirection); edges leaving
// a synthetic node pass that node's own phis.

#include <brass/mir/block.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace brass::gpu {

constexpr size_t kNoNode = static_cast<size_t>(-1);

struct CfgEdge {
    size_t from = kNoNode;
    size_t to = kNoNode;
    const BranchTarget* target = nullptr; // MIR branch target when `from` is a MIR block
};

struct CfgNode {
    enum class Kind { prologue, block, forward, loop_continue, unreachable };
    Kind kind = Kind::block;
    const BasicBlock* block = nullptr;  // Kind::block
    const BasicBlock* params = nullptr; // whose block parameters this node's phis carry (nullptr: none)
    std::vector<size_t> out;            // edge indices, in terminator order (true target first)
    std::vector<size_t> in;             // edge indices
    size_t merge = kNoNode;             // selection or loop merge node
    size_t cont = kNoNode;              // loop continue node (loop headers only)
    std::string name;                   // for OpName and diagnostics
};

struct StructuredCfg {
    std::vector<CfgNode> nodes;
    std::vector<CfgEdge> edges;
    std::vector<size_t> order; // emission order: reverse postorder, unreachable merges last
};

// Throws std::runtime_error naming the backend, function, and block(s) for an
// irreducible CFG, a loop with more than one exit target, an unsupported
// terminator (switch, invoke, throw, ...), or a block without a terminator.
StructuredCfg structurize(const brass::Function& fn, std::string_view backend_name = "SpirvISel");

std::string block_name(const BasicBlock* bb);

} // namespace brass::gpu

namespace brass::spirv {
using brass::gpu::kNoNode;
using brass::gpu::CfgEdge;
using brass::gpu::CfgNode;
using brass::gpu::StructuredCfg;
using brass::gpu::structurize;
using brass::gpu::block_name;
} // namespace brass::spirv
