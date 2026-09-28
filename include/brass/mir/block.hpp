#pragma once

#include <brass/mir/instruction.hpp>
#include <string_view>
#include <vector>
#include <cstdint>
#include <iterator>

namespace brass {

class Function;
class BasicBlock;

// A block's successors, by value: up to two (every br, br_if and invoke)
// held inline, a switch's larger set on the heap. Iterates, indexes and
// converts to a vector like the vector it replaced, without allocating in
// the common case (CFG walks ask for it once per block visit).
class SuccessorList {
public:
    using value_type = BasicBlock*;
    using const_iterator = BasicBlock* const*;
    using iterator = const_iterator;

    SuccessorList() noexcept = default;

    void push_back(BasicBlock* bb) {
        if (!spill_.empty()) {
            spill_.push_back(bb);
        } else if (size_ < kInline) {
            inline_[size_] = bb;
        } else {
            spill_.assign(inline_, inline_ + size_);
            spill_.push_back(bb);
        }
        ++size_;
    }

    const_iterator begin() const noexcept { return spill_.empty() ? inline_ : spill_.data(); }
    const_iterator end() const noexcept { return begin() + size_; }
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    BasicBlock* operator[](size_t i) const noexcept { return begin()[i]; }
    BasicBlock* front() const noexcept { return begin()[0]; }
    BasicBlock* back() const noexcept { return begin()[size_ - 1]; }

    operator std::vector<BasicBlock*>() const { return std::vector<BasicBlock*>(begin(), end()); }

private:
    static constexpr size_t kInline = 2;
    BasicBlock* inline_[kInline] = {};
    size_t size_ = 0;
    std::vector<BasicBlock*> spill_;
};

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

    SuccessorList successors() const;

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
