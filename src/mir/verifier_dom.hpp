#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <vector>

namespace brass {

class DominanceCalculator {
public:
    explicit DominanceCalculator(const Function& fn) {
        build(fn);
    }

    bool is_reachable(const BasicBlock* bb) const {
        return reachable_.find(bb) != reachable_.end();
    }

    bool dominates(const BasicBlock* a, const BasicBlock* b) const {
        if (!a || !b) return false;
        if (a == b) return true;
        if (!is_reachable(a) || !is_reachable(b)) {
            return false;
        }

        auto it_b = block_idx_.find(b);
        auto it_a = block_idx_.find(a);
        if (it_b == block_idx_.end() || it_a == block_idx_.end()) {
            return false;
        }

        int curr = it_b->second;
        int target = it_a->second;
        int entry = block_idx_.at(entry_block_);

        while (curr != -1 && curr != entry) {
            curr = idom_[curr];
            if (curr == target) {
                return true;
            }
        }
        return false;
    }

private:
    void build(const Function& fn) {
        entry_block_ = fn.entry_block();
        if (!entry_block_) return;

        // 1. Find reachable blocks via BFS
        std::queue<const BasicBlock*> q;
        q.push(entry_block_);
        reachable_.insert(entry_block_);

        std::vector<const BasicBlock*> reachable_blocks;
        while (!q.empty()) {
            const BasicBlock* curr = q.front();
            q.pop();
            reachable_blocks.push_back(curr);

            for (const BasicBlock* succ : curr->successors()) {
                if (succ && reachable_.insert(succ).second) {
                    q.push(succ);
                }
            }
        }

        int n = static_cast<int>(reachable_blocks.size());
        for (int i = 0; i < n; ++i) {
            block_idx_[reachable_blocks[i]] = i;
        }

        // 2. Compute DFS post-order
        std::vector<bool> visited(n, false);
        std::vector<int> post_order;
        std::vector<int> rpo_num(n, 0);

        auto dfs = [&](auto& self, int u) -> void {
            visited[u] = true;
            for (const BasicBlock* succ : reachable_blocks[u]->successors()) {
                if (!succ) continue;
                auto it = block_idx_.find(succ);
                if (it != block_idx_.end() && !visited[it->second]) {
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
            // Iterate in Reverse Post-Order (skip entry)
            for (int i = n - 1; i >= 0; --i) {
                int b = post_order[i];
                if (b == entry_idx) continue;

                int new_idom = -1;
                // Find first processed predecessor
                for (const BasicBlock* pred : reachable_blocks[b]->predecessors()) {
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

                for (const BasicBlock* pred : reachable_blocks[b]->predecessors()) {
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
    }

    const BasicBlock* entry_block_ = nullptr;
    std::unordered_set<const BasicBlock*> reachable_;
    std::unordered_map<const BasicBlock*, int> block_idx_;
    std::vector<int> idom_;
};

} // namespace brass
