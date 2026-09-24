#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/instruction.hpp>
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

void Function::remove_resume_point(uint32_t resume_id) {
    std::erase_if(resume_points_, [resume_id](const auto& entry) { return entry.first == resume_id; });
}

BasicBlock* Function::get_resume_target(uint32_t resume_id) const noexcept {
    for (const auto& entry : resume_points_) {
        if (entry.first == resume_id) {
            return entry.second;
        }
    }
    return nullptr;
}

const Instruction* Function::find_guard(uint32_t resume_id) const noexcept {
    for (const auto* bb : blocks_) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (inst && inst->opcode() == Opcode::guard && inst->resume_id() == resume_id) return inst;
        }
    }
    return nullptr;
}

uint32_t Function::next_guard_resume_id() const noexcept {
    bool any = false;
    uint32_t max_id = 0;
    for (const auto* bb : blocks_) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (inst && inst->opcode() == Opcode::guard) {
                any = true;
                if (inst->resume_id() > max_id) max_id = inst->resume_id();
            }
        }
    }
    return any ? max_id + 1 : 0;
}

const Function* Function::guard_exit_stub(const Instruction& guard) const noexcept {
    if (!parent_ || guard.symbol().empty()) return nullptr;
    return parent_->get_function(guard.symbol());
}

bool Function::guard_exit_stub_matches(const Instruction& guard, const Function& stub, std::string& why) const {
    const auto& state = guard.state_map();
    const auto& params = stub.param_types();
    const std::string who = "exit stub '" + std::string(stub.name()) + "'";
    if (params.size() != state.size()) {
        why = who + " takes " + std::to_string(params.size()) + " parameters but the guard has " +
              std::to_string(state.size()) + " state values (it is called as stub(state values...))";
        return false;
    }
    for (size_t i = 0; i < state.size(); ++i) {
        if (state[i] && state[i]->type() != params[i]) {
            why = who + " parameter " + std::to_string(i) + " is " + std::string(params[i].name()) +
                  " but state value " + std::to_string(i) + " is " + std::string(state[i]->type().name());
            return false;
        }
    }
    if (stub.return_type() != return_type_) {
        why = who + " returns " + std::string(stub.return_type().name()) + " but '" + std::string(name_) +
              "' returns " + std::string(return_type_.name()) + " (the stub's result is the function's)";
        return false;
    }
    return true;
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

