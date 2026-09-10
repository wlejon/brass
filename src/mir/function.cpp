#include <brass/mir/function.hpp>
#include <algorithm>

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

} // namespace brass
