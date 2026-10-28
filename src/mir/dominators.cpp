#include <brass/mir/dominators.hpp>
#include <algorithm>

namespace brass {

DominatorTree::DominatorTree(const Function& fn) {
    build(fn);
}

bool DominatorTree::is_reachable(const BasicBlock* bb) const noexcept {
    if (!bb) return false;
    return block_idx_.contains(bb);
}

bool DominatorTree::dominates(const BasicBlock* a, const BasicBlock* b) const noexcept {
    if (!a || !b) return false;
    if (a == b) return true;
    const int* found_a = block_idx_.find(a);
    const int* found_b = block_idx_.find(b);
    if (!found_a || !found_b) return false;
    const int idx_a = *found_a;
    const int idx_b = *found_b;
    if (idx_a >= static_cast<int>(dfs_in_.size()) || idx_b >= static_cast<int>(dfs_in_.size())) {
        return false;
    }

    return dfs_in_[idx_a] <= dfs_in_[idx_b] && dfs_out_[idx_a] >= dfs_out_[idx_b];
}

const BasicBlock* DominatorTree::immediate_dominator(const BasicBlock* bb) const noexcept {
    if (!bb) return nullptr;
    const int* found = block_idx_.find(bb);
    if (!found) return nullptr;
    int idx = *found;
    int id = idom_[idx];
    if (id < 0 || id == idx) return nullptr;
    return reachable_blocks_[id];
}

std::span<const BasicBlock* const> DominatorTree::children(const BasicBlock* bb) const noexcept {
    if (!bb) return {};
    const int* found = block_idx_.find(bb);
    if (!found) return {};
    const size_t idx = static_cast<size_t>(*found);
    if (idx + 1 >= child_start_.size()) return {};
    return {child_list_.data() + child_start_[idx], child_list_.data() + child_start_[idx + 1]};
}

void DominatorTree::build(const Function& fn) {
    entry_block_ = fn.entry_block();
    if (!entry_block_) return;

    // 1. BFS to collect reachable basic blocks, numbered in visit order
    //    (reachable_blocks_ is the queue).
    block_idx_.reserve(fn.current_next_block_id());
    reachable_blocks_.reserve(fn.block_count());
    block_idx_.insert(entry_block_, 0);
    reachable_blocks_.push_back(entry_block_);
    for (size_t head = 0; head < reachable_blocks_.size(); ++head) {
        for (const BasicBlock* succ : reachable_blocks_[head]->successors()) {
            if (succ && block_idx_.insert(succ, static_cast<int>(reachable_blocks_.size()))) {
                reachable_blocks_.push_back(succ);
            }
        }
    }

    int n = static_cast<int>(reachable_blocks_.size());

    // 2. DFS post-order numbering
    std::vector<bool> dfs_visited(n, false);
    std::vector<int> post_order;
    std::vector<int> rpo_num(n, 0);

    // An explicit stack, not recursion: a chain of many thousands of blocks
    // overflowed the native stack.
    struct Frame {
        int u;
        SuccessorList succs;
        size_t next;
    };
    std::vector<Frame> stack;
    const int entry = 0;
    dfs_visited[entry] = true;
    stack.push_back({entry, reachable_blocks_[entry]->successors(), 0});
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
        if (found && !dfs_visited[*found]) {
            const int v = *found;
            dfs_visited[v] = true;
            stack.push_back({v, reachable_blocks_[v]->successors(), 0});
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
        for (int i = n - 1; i >= 0; --i) {
            int b = post_order[i];
            if (b == entry_idx) continue;

            int new_idom = -1;
            for (const BasicBlock* pred : reachable_blocks_[b]->predecessors()) {
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

            for (const BasicBlock* pred : reachable_blocks_[b]->predecessors()) {
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

    // 4. Construct Dominator Tree children, each block's in index order
    child_start_.assign(static_cast<size_t>(n) + 1, 0);
    for (int i = 0; i < n; ++i) {
        if (i != entry_idx && idom_[i] >= 0) ++child_start_[static_cast<size_t>(idom_[i]) + 1];
    }
    for (int i = 0; i < n; ++i) child_start_[i + 1] += child_start_[i];
    child_list_.assign(child_start_[n], nullptr);
    {
        std::vector<uint32_t> fill(child_start_.begin(), child_start_.end() - 1);
        for (int i = 0; i < n; ++i) {
            if (i != entry_idx && idom_[i] >= 0) child_list_[fill[idom_[i]]++] = reachable_blocks_[i];
        }
    }

    // 5. Compute DFS in/out timestamps for O(1) dominance queries
    dfs_in_.assign(n, 0);
    dfs_out_.assign(n, 0);
    int timer = 0;

    // An explicit stack: the tree is as deep as the longest dominance chain.
    std::vector<std::pair<int, size_t>> dom_stack;
    dfs_in_[entry_idx] = ++timer;
    dom_stack.push_back({entry_idx, 0});
    while (!dom_stack.empty()) {
        auto& [u, next] = dom_stack.back();
        if (next == child_start_[u + 1] - child_start_[u]) {
            dfs_out_[u] = ++timer;
            dom_stack.pop_back();
            continue;
        }
        const int child_idx = *block_idx_.find(child_list_[child_start_[u] + next++]);
        dfs_in_[child_idx] = ++timer;
        dom_stack.push_back({child_idx, 0});
    }
}

} // namespace brass
