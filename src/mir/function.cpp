#include <brass/mir/function.hpp>
#include <algorithm>
#include <unordered_map>

namespace brass {

void Function::append_block(BasicBlock* bb) {
    if (!bb) return;
    bb->set_parent(this);
    blocks_.push_back(bb);
}

void Function::prepend_block(BasicBlock* bb) {
    if (!bb) return;
    bb->set_parent(this);
    blocks_.insert(blocks_.begin(), bb);
}

void Function::remove_block(BasicBlock* bb) {
    if (!bb) return;
    auto it = std::find(blocks_.begin(), blocks_.end(), bb);
    if (it != blocks_.end()) {
        blocks_.erase(it);
    }
}

BasicBlock* Function::get_block_by_name(std::string_view name) const noexcept {
    for (BasicBlock* bb : blocks_) {
        if (bb && bb->name() == name) {
            return bb;
        }
    }
    return nullptr;
}

BasicBlock* Function::get_block_by_id(uint32_t id) const noexcept {
    for (BasicBlock* bb : blocks_) {
        if (bb && bb->id() == id) {
            return bb;
        }
    }
    return nullptr;
}

void Function::add_resume_point(uint32_t resume_id, BasicBlock* target) {
    for (auto& entry : resume_points_) {
        if (entry.first == resume_id) {
            entry.second = target;
            return;
        }
    }
    resume_points_.push_back({resume_id, target});
}

BasicBlock* Function::get_resume_target(uint32_t resume_id) const noexcept {
    for (const auto& entry : resume_points_) {
        if (entry.first == resume_id) {
            return entry.second;
        }
    }
    return nullptr;
}

void Function::rebuild_cfg_predecessors() {
    for (BasicBlock* bb : blocks_) {
        if (bb) {
            bb->clear_predecessors();
        }
    }

    for (BasicBlock* bb : blocks_) {
        if (!bb) continue;
        for (BasicBlock* succ : bb->successors()) {
            if (succ) {
                succ->add_predecessor(bb);
            }
        }
    }
}

void Function::sort_blocks_rpo() {
    if (blocks_.size() <= 1) return;
    BasicBlock* entry = entry_block();
    if (!entry) return;

    std::unordered_map<const BasicBlock*, size_t> block_idx;
    block_idx.reserve(blocks_.size());
    for (size_t i = 0; i < blocks_.size(); ++i) {
        if (blocks_[i]) block_idx[blocks_[i]] = i;
    }

    std::vector<bool> visited(blocks_.size(), false);
    std::vector<BasicBlock*> entry_po;
    entry_po.reserve(blocks_.size());

    // An explicit stack, not recursion: a chain of many thousands of blocks
    // overflowed the native stack.
    struct Frame {
        BasicBlock* bb;
        std::vector<BasicBlock*> succs;
        size_t next;
    };
    std::vector<Frame> stack;
    auto dfs = [&](BasicBlock* root, std::vector<BasicBlock*>& po) {
        auto enter = [&](BasicBlock* bb) {
            auto it = block_idx.find(bb);
            if (it == block_idx.end() || visited[it->second]) return;
            visited[it->second] = true;
            stack.push_back({bb, bb->successors(), 0});
        };
        enter(root);
        while (!stack.empty()) {
            Frame& f = stack.back();
            if (f.next == f.succs.size()) {
                po.push_back(f.bb);
                stack.pop_back();
                continue;
            }
            BasicBlock* succ = f.succs[f.next++];
            if (succ) enter(succ);
        }
    };

    dfs(entry, entry_po);
    std::reverse(entry_po.begin(), entry_po.end());

    std::vector<BasicBlock*> other_po;
    for (const auto& rp : resume_points_) {
        if (rp.second) dfs(rp.second, other_po);
    }

    for (BasicBlock* bb : blocks_) {
        if (bb) dfs(bb, other_po);
    }

    std::reverse(other_po.begin(), other_po.end());

    entry_po.insert(entry_po.end(), other_po.begin(), other_po.end());
    blocks_ = std::move(entry_po);
}

} // namespace brass

