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

// Walks the dominator subtree rooted at `root` depth-first with an explicit
// stack (the tree is as deep as the longest dominance chain, which can be
// every block of the function). enter(bb) runs before bb's children, in the
// order children() lists them; leave(bb) runs after all of them.
template <typename Enter, typename Leave>
void walk_dominator_tree(const DominatorTree& dom, const BasicBlock* root, Enter&& enter, Leave&& leave) {
    if (!root) return;
    std::vector<std::pair<const BasicBlock*, size_t>> stack;
    enter(root);
    stack.push_back({root, 0});
    while (!stack.empty()) {
        const BasicBlock* bb = stack.back().first;
        const std::vector<const BasicBlock*>& kids = dom.children(bb);
        size_t& next = stack.back().second;
        if (next == kids.size()) {
            stack.pop_back();
            leave(bb);
            continue;
        }
        const BasicBlock* child = kids[next++];
        if (!child) continue;
        enter(child);
        stack.push_back({child, 0});
    }
}

} // namespace brass
