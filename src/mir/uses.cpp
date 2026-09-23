#include <brass/mir/uses.hpp>
#include <stdexcept>
#include <string>

namespace brass {

bool uses_value(const Instruction& inst, const Value* val) noexcept {
    if (!val) return false;
    bool hit = false;
    for_each_use(inst, [&](const Value* v) { hit |= (v == val); });
    return hit;
}

size_t replace_uses_in(Instruction& inst, const Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return 0;
    size_t n = 0;
    for_each_use_slot(inst, [&](Value*& slot) {
        if (slot == old_val) {
            slot = new_val;
            ++n;
        }
    });
    return n;
}

size_t replace_uses_in(BasicBlock& bb, const Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return 0;
    size_t n = 0;
    for (Instruction* inst : bb) {
        if (inst) n += replace_uses_in(*inst, old_val, new_val);
    }
    return n;
}

size_t replace_all_uses(Function& fn, const Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return 0;
    size_t n = 0;
    for (BasicBlock* bb : fn.blocks()) {
        if (bb) n += replace_uses_in(*bb, old_val, new_val);
    }
    return n;
}

size_t replace_uses_if(Function& fn, const Value* old_val, Value* new_val,
                       const std::function<bool(const Instruction&)>& where) {
    if (!old_val || !new_val || old_val == new_val) return 0;
    size_t n = 0;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (inst && where(*inst)) n += replace_uses_in(*inst, old_val, new_val);
        }
    }
    return n;
}

size_t count_uses(const Function& fn, const Value* val) {
    if (!val) return 0;
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            for_each_use(*inst, [&](const Value* v) { if (v == val) ++n; });
        }
    }
    return n;
}

bool has_uses(const Function& fn, const Value* val) {
    if (!val) return false;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && uses_value(*inst, val)) return true;
        }
    }
    return false;
}

std::unordered_map<const Value*, uint32_t> compute_use_counts(const Function& fn) {
    std::unordered_map<const Value*, uint32_t> counts;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            for_each_use(*inst, [&](const Value* v) { ++counts[v]; });
        }
    }
    return counts;
}

void remove_block_param(BasicBlock& bb, size_t index) {
    auto& params = bb.params();
    if (index >= params.size()) {
        throw std::out_of_range("remove_block_param: block has no parameter " + std::to_string(index));
    }
    params.erase(params.begin() + static_cast<std::ptrdiff_t>(index));
    for (size_t k = index; k < params.size(); ++k) {
        params[k]->set_block_param(&bb, static_cast<uint32_t>(k));
    }
    Function* fn = bb.parent();
    if (!fn) return;
    for (BasicBlock* pred : fn->blocks()) {
        if (!pred) continue;
        Instruction* term = pred->terminator();
        if (!term) continue;
        for_each_edge(*term, [&](BranchTarget& bt) {
            if (bt.block == &bb && index < bt.args.size()) {
                bt.args.erase(bt.args.begin() + static_cast<std::ptrdiff_t>(index));
            }
        });
    }
}

} // namespace brass
