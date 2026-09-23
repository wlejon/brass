#pragma once

// Register allocation for the bytecode compiler: SSA values are mapped to
// VM registers so that values whose live ranges do not overlap share one.
//
// Each value gets one conservative interval, the hull of every position it
// is live at (its definition, its uses, the start of every block it is live
// into and the end of every block it is live out of). Intervals are closed,
// so a value never shares a register with an operand of the instruction
// that defines it, and the lowering may write a result before it has read
// all of its operands. Values share registers only with values of the same
// type, which keeps BytecodeFunction::register_types exact (the GC and the
// deopt/throw paths read it).

#include <brass/vm/bytecode.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace brass::detail {

// Blocks in emission order (the entry block first) and the CFG between
// them, including the edges the bytecode adds: invoke -> unwind block and
// guard -> its resume block.
struct BlockLayout {
    std::vector<const BasicBlock*> order;
    std::unordered_map<const BasicBlock*, uint32_t> index;
    std::vector<std::vector<uint32_t>> succs;
    std::vector<std::vector<uint32_t>> preds;
};

BlockLayout build_block_layout(const Function& fn);

struct RegisterAssignment {
    std::unordered_map<const Value*, BcReg> reg;
    std::vector<Type> register_types; // per register
    uint32_t num_registers = 0;       // allocated registers (no scratch)
    uint32_t num_values = 0;          // SSA values allocated
};

// Throws std::runtime_error when the function needs more registers than
// the encoding holds, or uses a value it never defines.
RegisterAssignment allocate_bytecode_registers(const Function& fn, const BlockLayout& layout);

// The branch targets of a terminator (br / br_if / switch / invoke), in a
// fixed order; empty for other instructions.
std::vector<const BranchTarget*> branch_targets_of(const Instruction& inst);

} // namespace brass::detail
