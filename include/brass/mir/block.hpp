#pragma once

#include <brass/mir/instruction.hpp>
#include <string_view>
#include <vector>
#include <cstdint>
#include <iterator>

namespace brass {

class Function;

class BasicBlock {
public:
    class InstructionIterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Instruction*;
        using difference_type = std::ptrdiff_t;
        using pointer = Instruction**;
        using reference = Instruction*&;

        InstructionIterator() noexcept : current_(nullptr) {}
        explicit InstructionIterator(Instruction* inst) noexcept : current_(inst) {}

        Instruction* operator*() const noexcept { return current_; }
        Instruction* operator->() const noexcept { return current_; }

        InstructionIterator& operator++() noexcept {
            if (current_) {
                current_ = current_->next();
            }
            return *this;
        }

        InstructionIterator operator++(int) noexcept {
            InstructionIterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const InstructionIterator& other) const noexcept {
            return current_ == other.current_;
        }

        bool operator!=(const InstructionIterator& other) const noexcept {
            return current_ != other.current_;
        }

    private:
        Instruction* current_ = nullptr;
    };

    BasicBlock() noexcept = default;
    explicit BasicBlock(uint32_t id, std::string_view name = "") noexcept
        : id_(id), name_(name) {}

    uint32_t id() const noexcept { return id_; }
    void set_id(uint32_t id) noexcept { id_ = id; }

    std::string_view name() const noexcept { return name_; }
    void set_name(std::string_view name) noexcept { name_ = name; }

    Function* parent() const noexcept { return parent_; }
    void set_parent(Function* fn) noexcept { parent_ = fn; }

    const std::vector<Value*>& params() const noexcept { return params_; }
    std::vector<Value*>& params() noexcept { return params_; }
    Value* param(size_t index) const noexcept {
        return (index < params_.size()) ? params_[index] : nullptr;
    }
    size_t param_count() const noexcept { return params_.size(); }
    void add_param(Value* val) {
        if (val) {
            val->set_block_param(this, static_cast<uint32_t>(params_.size()));
            params_.push_back(val);
        }
    }

    Instruction* head() const noexcept { return head_; }
    Instruction* tail() const noexcept { return tail_; }
    Instruction* terminator() const noexcept {
        if (tail_ && tail_->is_terminator()) {
            return tail_;
        }
        return nullptr;
    }

    bool is_empty() const noexcept { return head_ == nullptr; }
    size_t instruction_count() const noexcept { return instruction_count_; }

    void append_instruction(Instruction* inst);
    void prepend_instruction(Instruction* inst);
    void insert_before(Instruction* inst, Instruction* before_inst);
    void insert_after(Instruction* inst, Instruction* after_inst);
    void remove_instruction(Instruction* inst);

    const std::vector<BasicBlock*>& predecessors() const noexcept { return predecessors_; }
    std::vector<BasicBlock*>& predecessors() noexcept { return predecessors_; }
    void add_predecessor(BasicBlock* pred);
    void remove_predecessor(BasicBlock* pred);
    void clear_predecessors() noexcept { predecessors_.clear(); }

    std::vector<BasicBlock*> successors() const;

    InstructionIterator begin() noexcept { return InstructionIterator(head_); }
    InstructionIterator end() noexcept { return InstructionIterator(nullptr); }
    InstructionIterator begin() const noexcept { return InstructionIterator(head_); }
    InstructionIterator end() const noexcept { return InstructionIterator(nullptr); }

private:
    uint32_t id_ = 0;
    std::string_view name_;
    Function* parent_ = nullptr;

    std::vector<Value*> params_;

    Instruction* head_ = nullptr;
    Instruction* tail_ = nullptr;
    size_t instruction_count_ = 0;

    std::vector<BasicBlock*> predecessors_;
};

} // namespace brass
