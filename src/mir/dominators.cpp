#include <brass/mir/dominators.hpp>
#include <queue>
#include <algorithm>

namespace brass {

DominatorTree::DominatorTree(const Function& fn) {
    build(fn);
}

bool DominatorTree::is_reachable(const BasicBlock* bb) const noexcept {
    if (!bb) return false;
    return block_idx_.find(bb) != block_idx_.end();
}

bool DominatorTree::dominates(const BasicBlock* a, const BasicBlock* b) const noexcept {
    if (!a || !b) return false;
    if (a == b) return true;
    if (!is_reachable(a) || !is_reachable(b)) return false;

    int idx_a = block_idx_.at(a);
    int idx_b = block_idx_.at(b);
    if (idx_a >= static_cast<int>(dfs_in_.size()) || idx_b >= static_cast<int>(dfs_in_.size())) {
        return false;
    }

    return dfs_in_[idx_a] <= dfs_in_[idx_b] && dfs_out_[idx_a] >= dfs_out_[idx_b];
}

const BasicBlock* DominatorTree::immediate_dominator(const BasicBlock* bb) const noexcept {
    if (!bb) return nullptr;
    auto it = block_idx_.find(bb);
    if (it == block_idx_.end()) return nullptr;
    int idx = it->second;
    int id = idom_[idx];
    if (id < 0 || id == idx) return nullptr;
    return reachable_blocks_[id];
}

const std::vector<const BasicBlock*>& DominatorTree::children(const BasicBlock* bb) const noexcept {
    if (!bb) return empty_children_;
    auto it = block_idx_.find(bb);
    if (it == block_idx_.end()) return empty_children_;
    int idx = it->second;
    if (idx >= 0 && idx < static_cast<int>(dom_children_.size())) {
        return dom_children_[idx];
    }
    return empty_children_;
}

void DominatorTree::build(const Function& fn) {
    entry_block_ = fn.entry_block();
    if (!entry_block_) return;

    // 1. BFS to collect reachable basic blocks
    std::unordered_set<const BasicBlock*> visited;
    std::queue<const BasicBlock*> q;
    q.push(entry_block_);
    visited.insert(entry_block_);

    while (!q.empty()) {
        const BasicBlock* curr = q.front();
        q.pop();
        reachable_blocks_.push_back(curr);

        for (const BasicBlock* succ : curr->successors()) {
            if (succ && visited.insert(succ).second) {
                q.push(succ);
            }
        }
    }

    int n = static_cast<int>(reachable_blocks_.size());
    for (int i = 0; i < n; ++i) {
        block_idx_[reachable_blocks_[i]] = i;
    }

    // 2. DFS post-order numbering
    std::vector<bool> dfs_visited(n, false);
    std::vector<int> post_order;
    std::vector<int> rpo_num(n, 0);

    auto dfs = [&](auto& self, int u) -> void {
        dfs_visited[u] = true;
        for (const BasicBlock* succ : reachable_blocks_[u]->successors()) {
            if (!succ) continue;
            auto it = block_idx_.find(succ);
            if (it != block_idx_.end() && !dfs_visited[it->second]) {
                self(self, it->second);
            }
        }
        post_order.push_back(u);
    };

    dfs(dfs, block_idx_[entry_block_]);

    int rpo_count = n;
    for (int u : post_order) {
        rpo_num[u] = --rpo_count;
    }

    // 3. Iterative Dominators (Cooper-Harvey-Kennedy)
    idom_.assign(n, -1);
    int entry_idx = block_idx_[entry_block_];
    idom_[entry_idx] = entry_idx;

    auto intersect = [&](int b1, int b2) -> int {
        int finger1 = b1;
        int finger2 = b2;
        while (finger1 != finger2) {
            while (rpo_num[finger1] > rpo_num[finger2]) {
                finger1 = idom_[finger1];
            }
            while (rpo_num[finger2] > rpo_num[finger1]) {
                finger2 = idom_[finger2];
            }
        }
        return finger1;
    };

    bool changed = true;
    while (changed) {
        changed = false;
        for (int i = n - 1; i >= 0; --i) {
            int b = post_order[i];
            if (b == entry_idx) continue;

            int new_idom = -1;
            for (const BasicBlock* pred : reachable_blocks_[b]->predecessors()) {
                if (!pred) continue;
                auto it = block_idx_.find(pred);
                if (it != block_idx_.end()) {
                    int p = it->second;
                    if (idom_[p] != -1) {
                        new_idom = p;
                        break;
                    }
                }
            }

            if (new_idom == -1) continue;

            for (const BasicBlock* pred : reachable_blocks_[b]->predecessors()) {
                if (!pred) continue;
                auto it = block_idx_.find(pred);
                if (it != block_idx_.end()) {
                    int p = it->second;
                    if (p != new_idom && idom_[p] != -1) {
                        new_idom = intersect(p, new_idom);
                    }
                }
            }

            if (idom_[b] != new_idom) {
                idom_[b] = new_idom;
                changed = true;
            }
        }
    }

    // 4. Construct Dominator Tree children
    dom_children_.assign(n, {});
    for (int i = 0; i < n; ++i) {
        if (i != entry_idx && idom_[i] >= 0) {
            dom_children_[idom_[i]].push_back(reachable_blocks_[i]);
        }
    }

    // 5. Compute DFS in/out timestamps for O(1) dominance queries
    dfs_in_.assign(n, 0);
    dfs_out_.assign(n, 0);
    int timer = 0;

    auto dfs_dom = [&](auto& self, int u) -> void {
        dfs_in_[u] = ++timer;
        for (const BasicBlock* child : dom_children_[u]) {
            int child_idx = block_idx_.at(child);
            self(self, child_idx);
        }
        dfs_out_[u] = ++timer;
    };

    dfs_dom(dfs_dom, entry_idx);
}

} // namespace brass
