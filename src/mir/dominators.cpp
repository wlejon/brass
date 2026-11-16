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
    //    (reachable_blocks_ is the queue), each block's successors recorded
    //    once by number (succ_start / succ_list), in successors() order.
    block_idx_.reserve(fn.current_next_block_id());
    reachable_blocks_.reserve(fn.block_count());
    block_idx_.insert(entry_block_, 0);
    reachable_blocks_.push_back(entry_block_);
    std::vector<uint32_t> succ_start;
    std::vector<int> succ_list;
    succ_start.reserve(fn.block_count() + 1);
    succ_list.reserve(fn.block_count() * 2);
    for (size_t head = 0; head < reachable_blocks_.size(); ++head) {
        succ_start.push_back(static_cast<uint32_t>(succ_list.size()));
        for (const BasicBlock* succ : reachable_blocks_[head]->successors()) {
            if (!succ) continue;
            const int next = static_cast<int>(reachable_blocks_.size());
            if (block_idx_.insert(succ, next)) {
                reachable_blocks_.push_back(succ);
                succ_list.push_back(next);
            } else {
                succ_list.push_back(*block_idx_.find(succ));
            }
        }
    }
    succ_start.push_back(static_cast<uint32_t>(succ_list.size()));

    int n = static_cast<int>(reachable_blocks_.size());

    // 2. DFS post-order numbering
    std::vector<uint8_t> dfs_visited(n, 0);
    std::vector<int> post_order;
    post_order.reserve(n);
    std::vector<int> rpo_num(n, 0);

    // An explicit stack, not recursion: a chain of many thousands of blocks
    // overflowed the native stack.
    std::vector<std::pair<int, uint32_t>> stack;
    const int entry = 0;
    dfs_visited[entry] = 1;
    stack.push_back({entry, succ_start[entry]});
    while (!stack.empty()) {
        auto& [u, next] = stack.back();
        if (next == succ_start[u + 1]) {
            post_order.push_back(u);
            stack.pop_back();
            continue;
        }
        const int v = succ_list[next++];
        if (!dfs_visited[v]) {
            dfs_visited[v] = 1;
            stack.push_back({v, succ_start[v]});
        }
    }

    int rpo_count = n;
    for (int u : post_order) {
        rpo_num[u] = --rpo_count;
    }

    // Each reachable block's reachable predecessors by number, in
    // predecessors() order.
    std::vector<uint32_t> pred_start(static_cast<size_t>(n) + 1, 0);
    std::vector<int> pred_list;
    pred_list.reserve(succ_list.size());
    for (int b = 0; b < n; ++b) {
        pred_start[b] = static_cast<uint32_t>(pred_list.size());
        for (const BasicBlock* pred : reachable_blocks_[b]->predecessors()) {
            if (!pred) continue;
            if (const int* found = block_idx_.find(pred)) pred_list.push_back(*found);
        }
    }
    pred_start[n] = static_cast<uint32_t>(pred_list.size());

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
            for (uint32_t k = pred_start[b]; k < pred_start[b + 1]; ++k) {
                const int p = pred_list[k];
                if (idom_[p] != -1) {
                    new_idom = p;
                    break;
                }
            }

            if (new_idom == -1) continue;

            for (uint32_t k = pred_start[b]; k < pred_start[b + 1]; ++k) {
                const int p = pred_list[k];
                if (p != new_idom && idom_[p] != -1) {
                    new_idom = intersect(p, new_idom);
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
    std::vector<int> child_index(child_start_[n]);
    {
        std::vector<uint32_t> fill(child_start_.begin(), child_start_.end() - 1);
        for (int i = 0; i < n; ++i) {
            if (i != entry_idx && idom_[i] >= 0) {
                child_index[fill[idom_[i]]] = i;
                child_list_[fill[idom_[i]]++] = reachable_blocks_[i];
            }
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
        const int child_idx = child_index[child_start_[u] + next++];
        dfs_in_[child_idx] = ++timer;
        dom_stack.push_back({child_idx, 0});
    }
}

} // namespace brass
