#pragma once

#include <brass/mir/dense_id_map.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/function.hpp>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace brass {

// Whether a stretch of a function can run the GC, for GVN's merge of two
// `bitcast_i64_tagged` of one tagged value (gc_refs.hpp). The bits a bitcast
// reads name where the object is now; two reads of the same tagged value
// agree exactly when no GC point (may_trigger_gc) can run between them, on
// any path. A frontend re-derives the bits after every call it emits (the
// contract), so within a stretch with no GC point the first read stands for
// the rest: what follows it, `and` / loads / compares, merges as usual.
//
// In a function with interrupt polls (interrupt_poll.hpp) the back edges
// count as GC points: a poll goes on each after the optimizer, and its
// handler may collect. So no
// stretch that enters a loop header, other than the one `from` is in,
// is GC-free: a path through a header can go round its loop first.
class GcFreePaths {
public:
    // Built on the first question: most functions GVN visits never ask one.
    // The answers are those of the function as it stood when GVN's walk
    // began, since the walk only inserts constants (no GC point, nothing
    // that moves an instruction it asks about) before it removes anything.
    // `dom` is the function's dominator tree, which the walk reads.
    GcFreePaths(const Function& fn, const DominatorTree& dom) : fn_(&fn), dom_(&dom) {}

    // True when no GC point lies on any path from just after `from` to just
    // before `to`, where `from` dominates `to` (and `to` is reachable). A
    // path that runs `from` again starts afresh there, so paths are taken
    // from the last run of `from`.
    bool between(const Instruction* from, const Instruction* to);

private:
    struct BlockInfo {
        uint32_t first_gc = UINT32_MAX;  // index of the first GC point, or none
        uint32_t last_gc = 0;            // index of the last one (valid when has_gc)
        bool has_gc = false;
        bool header = false;             // a back edge's target
    };

    void build();
    bool blocks_between_gc_free(const BasicBlock* a, const BasicBlock* b);
    // `inst`'s place in its block (its block numbered on first ask), or
    // false when it is in no block of the function.
    bool position(const Instruction* inst, uint32_t& at);

    const Function* fn_;
    const DominatorTree* dom_;
    bool built_ = false;

    // Block b's number (its place in the function's block order), the key
    // of every per-block vector here.
    uint32_t number(const BasicBlock* bb) const { return block_number_.value_or(bb, 0); }

    std::vector<BlockInfo> info_;
    std::unordered_map<const Instruction*, uint32_t> index_;
    std::vector<uint8_t> numbered_;  // per block: its instructions are in index_
    std::unordered_map<uint64_t, bool> cache_;
    DenseIdMap<BasicBlock, uint32_t> block_number_;
    uint32_t block_count_ = 0;
};

} // namespace brass
