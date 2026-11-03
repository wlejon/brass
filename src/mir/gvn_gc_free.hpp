#pragma once

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
class GcFreePaths {
public:
    explicit GcFreePaths(const Function& fn);

    // True when no GC point lies on any path from just after `from` to just
    // before `to`, where `from` dominates `to`. A path that runs `from` again
    // starts afresh there, so paths are taken from the last run of `from`.
    bool between(const Instruction* from, const Instruction* to);

private:
    struct BlockInfo {
        uint32_t first_gc = UINT32_MAX;  // index of the first GC point, or none
        uint32_t last_gc = 0;            // index of the last one (valid when has_gc)
        bool has_gc = false;
    };

    bool blocks_between_gc_free(const BasicBlock* a, const BasicBlock* b);

    std::unordered_map<const BasicBlock*, BlockInfo> info_;
    std::unordered_map<const Instruction*, uint32_t> index_;
    std::unordered_map<uint64_t, bool> cache_;
    std::unordered_map<const BasicBlock*, uint32_t> block_number_;
};

} // namespace brass
