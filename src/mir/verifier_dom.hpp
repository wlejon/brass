#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/dense_id_map.hpp>
#include <vector>

namespace brass {

class DominanceCalculator {
public:
    explicit DominanceCalculator(const Function& fn) {
        build(fn);
    }

    bool is_reachable(const BasicBlock* bb) const {
        return bb && block_idx_.contains(bb);
    }

    bool dominates(const BasicBlock* a, const BasicBlock* b) const {
        if (!a || !b) return false;
        if (a == b) return true;
        const int* found_b = block_idx_.find(b);
        const int* found_a = block_idx_.find(a);
        if (!found_b || !found_a) return false;

        // a dominates b exactly when b's dominator-tree interval nests in
        // a's. Walking b's idom chain instead was O(depth) per query and
        // quadratic on deeply nested control flow.
        const int ia = *found_a;
        const int ib = *found_b;
        return tin_[ia] <= tin_[ib] && tout_[ib] <= tout_[ia];
    }

private:
    void build(const Function& fn) {
        entry_block_ = fn.entry_block();
        if (!entry_block_) return;

        // 1. Find reachable blocks via BFS, numbered in visit order
        //    (reachable_blocks is the queue).
        block_idx_.reserve(fn.current_next_block_id());
        std::vector<const BasicBlock*> reachable_blocks;
        reachable_blocks.reserve(fn.block_count());
        block_idx_.insert(entry_block_, 0);
        reachable_blocks.push_back(entry_block_);
        for (size_t head = 0; head < reachable_blocks.size(); ++head) {
            for (const BasicBlock* succ : reachable_blocks[head]->successors()) {
                if (succ && block_idx_.insert(succ, static_cast<int>(reachable_blocks.size()))) {
                    reachable_blocks.push_back(succ);
                }
            }
        }

        int n = static_cast<int>(reachable_blocks.size());

        // 2. Compute DFS post-order
        std::vector<bool> visited(n, false);
        std::vector<int> post_order;
        std::vector<int> rpo_num(n, 0);

        // An explicit stack, not recursion: a chain of many thousands of
        // blocks overflowed the native stack.
        struct Frame {
            int u;
            SuccessorList succs;
            size_t next;
        };
        std::vector<Frame> stack;
        const int entry = 0;
        visited[entry] = true;
        stack.push_back({entry, reachable_blocks[entry]->successors(), 0});
        while (!stack.empty()) {
            Frame& f = stack.back();
            if (f.next == f.succs.size()) {
                post_order.push_back(f.u);
                stack.pop_back();
                continue;
            }
            const BasicBlock* succ = f.succs[f.next++];
            if (!succ) continue;
            const int* found = block_idx_.find(succ);
            if (found && !visited[*found]) {
                const int v = *found;
                visited[v] = true;
                stack.push_back({v, reachable_blocks[v]->successors(), 0});
            }
        }

        int rpo_count = n;
        for (int u : post_order) {
            rpo_num[u] = --rpo_count;
        }

        // 3. Iterative Dominators (Cooper-Harvey-Kennedy)
        idom_.assign(n, -1);
        const int entry_idx = 0;
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
                    if (const int* found = block_idx_.find(pred)) {
                        int p = *found;
                        if (idom_[p] != -1) {
                            new_idom = p;
                            break;
                        }
                    }
                }

                if (new_idom == -1) continue;

                for (const BasicBlock* pred : reachable_blocks[b]->predecessors()) {
                    if (!pred) continue;
                    if (const int* found = block_idx_.find(pred)) {
                        int p = *found;
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

        // 4. Entry and exit times of an iterative walk of the dominator
        // tree, so dominates() is a constant-time interval test.
        std::vector<std::vector<int>> children(n);
        for (int b = 0; b < n; ++b) {
            if (b != entry_idx && idom_[b] != -1) children[idom_[b]].push_back(b);
        }
        tin_.assign(n, -1);
        tout_.assign(n, -1);
        int clock = 0;
        std::vector<std::pair<int, size_t>> walk;
        walk.push_back({entry_idx, 0});
        tin_[entry_idx] = clock++;
        while (!walk.empty()) {
            auto& [u, next] = walk.back();
            if (next == children[u].size()) {
                tout_[u] = clock++;
                walk.pop_back();
                continue;
            }
            const int v = children[u][next++];
            tin_[v] = clock++;
            walk.push_back({v, 0});
        }
    }

    const BasicBlock* entry_block_ = nullptr;
    DenseIdMap<BasicBlock, int> block_idx_;
    std::vector<int> idom_;
    std::vector<int> tin_;
    std::vector<int> tout_;
};

} // namespace brass
