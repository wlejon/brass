#include <brass/mir/block.hpp>
#include <algorithm>

namespace brass {

void BasicBlock::append_instruction(Instruction* inst) {
    if (!inst) return;
    inst->set_parent(this);
    inst->set_next(nullptr);
    inst->set_prev(tail_);

    if (tail_) {
        tail_->set_next(inst);
        tail_ = inst;
    } else {
        head_ = inst;
        tail_ = inst;
    }
    instruction_count_++;
}

void BasicBlock::prepend_instruction(Instruction* inst) {
    if (!inst) return;
    inst->set_parent(this);
    inst->set_prev(nullptr);
    inst->set_next(head_);

    if (head_) {
        head_->set_prev(inst);
        head_ = inst;
    } else {
        head_ = inst;
        tail_ = inst;
    }
    instruction_count_++;
}

void BasicBlock::insert_before(Instruction* inst, Instruction* before_inst) {
    if (!inst) return;
    if (!before_inst) {
        append_instruction(inst);
        return;
    }
    inst->set_parent(this);
    Instruction* prev = before_inst->prev();
    inst->set_prev(prev);
    inst->set_next(before_inst);
    before_inst->set_prev(inst);

    if (prev) {
        prev->set_next(inst);
    } else {
        head_ = inst;
    }
    instruction_count_++;
}

void BasicBlock::insert_after(Instruction* inst, Instruction* after_inst) {
    if (!inst) return;
    if (!after_inst) {
        prepend_instruction(inst);
        return;
    }
    inst->set_parent(this);
    Instruction* next = after_inst->next();
    inst->set_prev(after_inst);
    inst->set_next(next);
    after_inst->set_next(inst);

    if (next) {
        next->set_prev(inst);
    } else {
        tail_ = inst;
    }
    instruction_count_++;
}

void BasicBlock::remove_instruction(Instruction* inst) {
    if (!inst || inst->parent() != this) return;

    Instruction* prev = inst->prev();
    Instruction* next = inst->next();

    if (prev) {
        prev->set_next(next);
    } else {
        head_ = next;
    }

    if (next) {
        next->set_prev(prev);
    } else {
        tail_ = prev;
    }

    inst->set_prev(nullptr);
    inst->set_next(nullptr);
    inst->set_parent(nullptr);
    if (instruction_count_ > 0) {
        instruction_count_--;
    }
}

void BasicBlock::add_predecessor(BasicBlock* pred) {
    if (pred && std::find(predecessors_.begin(), predecessors_.end(), pred) == predecessors_.end()) {
        predecessors_.push_back(pred);
    }
}

void BasicBlock::remove_predecessor(BasicBlock* pred) {
    auto it = std::find(predecessors_.begin(), predecessors_.end(), pred);
    if (it != predecessors_.end()) {
        predecessors_.erase(it);
    }
}

std::vector<BasicBlock*> BasicBlock::successors() const {
    std::vector<BasicBlock*> succs;
    Instruction* term = terminator();
    if (!term) return succs;

    if (term->opcode() == Opcode::br) {
        if (term->branch_target().block) {
            succs.push_back(term->branch_target().block);
        }
    } else if (term->opcode() == Opcode::br_if) {
        if (term->true_target().block) {
            succs.push_back(term->true_target().block);
        }
        if (term->false_target().block) {
            succs.push_back(term->false_target().block);
        }
    } else if (term->opcode() == Opcode::switch_) {
        if (term->default_target().block) {
            succs.push_back(term->default_target().block);
        }
        for (const auto& sc : term->switch_cases()) {
            if (sc.target.block && std::find(succs.begin(), succs.end(), sc.target.block) == succs.end()) {
                succs.push_back(sc.target.block);
            }
        }
    } else if (term->opcode() == Opcode::invoke) {
        if (term->normal_target().block) {
            succs.push_back(term->normal_target().block);
        }
        if (term->unwind_target().block && std::find(succs.begin(), succs.end(), term->unwind_target().block) == succs.end()) {
            succs.push_back(term->unwind_target().block);
        }
    }
    return succs;
}

} // namespace brass
