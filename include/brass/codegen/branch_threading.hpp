#pragma once

#include <brass/codegen/lir.hpp>
#include <cstdint>
#include <vector>

namespace brass::codegen {

// A block's closing branches, the ones the emitters lay out themselves:
// `jcc taken; jmp other` (count 2), `jcc taken` falling into the next block
// (count 1, `other` that block), or `jmp other` (count 1, not conditional).
struct ClosingBranches {
    size_t count = 0;
    bool conditional = false;
    uint32_t taken = 0;
    uint32_t other = 0;
};
ClosingBranches closing_branches(const LirFunction& fn, size_t index) noexcept;

// Where a block's closing branches go, once blocks that only jump on are
// seen through. Instruction selection gives an edge that carries block
// arguments a block of its own: the copies, then a jump to the target. When
// register allocation leaves no copies there, a loop's latch branched out of
// the loop to fall into that block and jump back, two branches an
// iteration; branching straight to where it goes, the block is entered by
// nothing and is not emitted.
//
// Only closing branches are redirected. Any other reference to a block (a
// branch inside a block, an unwind target, a resume entry, the entry) keeps
// the block.
class BranchThreading {
public:
    explicit BranchThreading(const LirFunction& fn);

    // Where a closing branch to block `id` goes.
    uint32_t target(uint32_t id) const noexcept {
        return id < forward_.size() ? forward_[id] : id;
    }
    // The block at position `index` of fn.blocks is emitted as nothing.
    bool skipped(size_t index) const noexcept { return index < skip_.size() && skip_[index] != 0; }
    // The block at position `index` is branched to from itself or a block
    // after it: a loop's head, in a layout that keeps loops contiguous.
    bool loop_head(size_t index) const noexcept { return index < head_.size() && head_[index] != 0; }

private:
    std::vector<uint32_t> forward_;  // by block id
    std::vector<uint8_t> skip_;      // by block position
    std::vector<uint8_t> head_;      // by block position
};

} // namespace brass::codegen
