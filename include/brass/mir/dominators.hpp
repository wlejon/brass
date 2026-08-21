#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace brass {

class DominatorTree {
public:
    explicit DominatorTree(const Function& fn);

    bool is_reachable(const BasicBlock* bb) const noexcept;
    bool dominates(const BasicBlock* a, const BasicBlock* b) const noexcept;
    const BasicBlock* immediate_dominator(const BasicBlock* bb) const noexcept;
    const std::vector<const BasicBlock*>& children(const BasicBlock* bb) const noexcept;
    const std::vector<const BasicBlock*>& reachable_blocks() const noexcept { return reachable_blocks_; }

private:
    void build(const Function& fn);

    const BasicBlock* entry_block_ = nullptr;
    std::vector<const BasicBlock*> reachable_blocks_;
    std::unordered_map<const BasicBlock*, int> block_idx_;
    std::vector<int> idom_;
    std::vector<std::vector<const BasicBlock*>> dom_children_;
    std::vector<int> dfs_in_;
    std::vector<int> dfs_out_;
    std::vector<const BasicBlock*> empty_children_;
};

} // namespace brass
