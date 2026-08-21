#pragma once

#include <brass/core/span.hpp>
#include <brass/mir/types.hpp>
#include <brass/mir/opcodes.hpp>
#include <cstdint>
#include <string_view>
#include <vector>

namespace brass {

class BasicBlock;
class Function;
class Instruction;

enum class ValueKind : uint8_t {
    InstructionResult,
    BlockParam
};

class Value {
public:
    Value() noexcept = default;
    Value(uint32_t id, Type type, ValueKind kind) noexcept
        : id_(id), type_(type), kind_(kind) {}

    uint32_t id() const noexcept { return id_; }
    void set_id(uint32_t id) noexcept { id_ = id; }

    Type type() const noexcept { return type_; }
    void set_type(Type type) noexcept { type_ = type; }

    ValueKind kind() const noexcept { return kind_; }
    bool is_block_param() const noexcept { return kind_ == ValueKind::BlockParam; }
    bool is_instruction() const noexcept { return kind_ == ValueKind::InstructionResult; }

    Instruction* defining_instruction() const noexcept { return def_inst_; }
    void set_defining_instruction(Instruction* inst) noexcept {
        def_inst_ = inst;
        kind_ = ValueKind::InstructionResult;
    }

    BasicBlock* defining_block() const noexcept { return def_block_; }
    uint32_t param_index() const noexcept { return param_index_; }
    void set_block_param(BasicBlock* block, uint32_t idx) noexcept {
        def_block_ = block;
        param_index_ = idx;
        kind_ = ValueKind::BlockParam;
    }

private:
    uint32_t id_ = 0;
    Type type_ = Type::void_type();
    ValueKind kind_ = ValueKind::InstructionResult;
    Instruction* def_inst_ = nullptr;
    BasicBlock* def_block_ = nullptr;
    uint32_t param_index_ = 0;
};

struct BranchTarget {
    BasicBlock* block = nullptr;
    std::vector<Value*> args;

    BranchTarget() = default;
    explicit BranchTarget(BasicBlock* b) : block(b) {}
    BranchTarget(BasicBlock* b, std::vector<Value*> a) : block(b), args(std::move(a)) {}
};

class Instruction {
public:
    explicit Instruction(Opcode op = Opcode::unreachable, Type type = Type::void_type()) noexcept
        : opcode_(op), type_(type) {}

    Opcode opcode() const noexcept { return opcode_; }
    void set_opcode(Opcode op) noexcept { opcode_ = op; }

    Type type() const noexcept { return type_; }
    void set_type(Type type) noexcept { type_ = type; }

    Value* result() const noexcept { return result_; }
    void set_result(Value* res) noexcept { result_ = res; }

    BasicBlock* parent() const noexcept { return parent_; }
    void set_parent(BasicBlock* bb) noexcept { parent_ = bb; }

    Instruction* prev() const noexcept { return prev_; }
    void set_prev(Instruction* p) noexcept { prev_ = p; }

    Instruction* next() const noexcept { return next_; }
    void set_next(Instruction* n) noexcept { next_ = n; }

    const std::vector<Value*>& operands() const noexcept { return operands_; }
    std::vector<Value*>& operands() noexcept { return operands_; }
    Value* operand(size_t idx) const noexcept {
        return (idx < operands_.size()) ? operands_[idx] : nullptr;
    }
    size_t operand_count() const noexcept { return operands_.size(); }

    void add_operand(Value* val) { operands_.push_back(val); }
    void set_operand(size_t idx, Value* val) {
        if (idx < operands_.size()) {
            operands_[idx] = val;
        }
    }

    int64_t imm_i64() const noexcept { return imm_i64_; }
    void set_imm_i64(int64_t val) noexcept { imm_i64_ = val; }

    int32_t imm_i32() const noexcept { return static_cast<int32_t>(imm_i64_); }
    void set_imm_i32(int32_t val) noexcept { imm_i64_ = static_cast<int64_t>(val); }

    double imm_f64() const noexcept { return imm_f64_; }
    void set_imm_f64(double val) noexcept { imm_f64_ = val; }

    uint8_t scale() const noexcept { return scale_; }
    void set_scale(uint8_t scale) noexcept { scale_ = scale; }

    int32_t offset() const noexcept { return offset_; }
    void set_offset(int32_t offset) noexcept { offset_ = offset; }

    Type memory_type() const noexcept { return memory_type_; }
    void set_memory_type(Type t) noexcept { memory_type_ = t; }

    std::string_view symbol() const noexcept { return symbol_; }
    void set_symbol(std::string_view sym) noexcept { symbol_ = sym; }

    std::string_view extra_symbol() const noexcept { return extra_symbol_; }
    void set_extra_symbol(std::string_view sym) noexcept { extra_symbol_ = sym; }

    uint32_t resume_id() const noexcept { return static_cast<uint32_t>(imm_i64_); }
    void set_resume_id(uint32_t id) noexcept { imm_i64_ = static_cast<int64_t>(id); }

    const BranchTarget& branch_target() const noexcept { return branch_target_; }
    BranchTarget& branch_target() noexcept { return branch_target_; }
    void set_branch_target(BranchTarget target) { branch_target_ = std::move(target); }

    const BranchTarget& true_target() const noexcept { return true_target_; }
    BranchTarget& true_target() noexcept { return true_target_; }
    void set_true_target(BranchTarget target) { true_target_ = std::move(target); }

    const BranchTarget& false_target() const noexcept { return false_target_; }
    BranchTarget& false_target() noexcept { return false_target_; }
    void set_false_target(BranchTarget target) { false_target_ = std::move(target); }

    const std::vector<Value*>& state_map() const noexcept { return state_map_; }
    std::vector<Value*>& state_map() noexcept { return state_map_; }
    void add_state_value(Value* val) { state_map_.push_back(val); }

    bool is_terminator() const noexcept { return brass::is_terminator(opcode_); }
    bool is_branch() const noexcept { return brass::is_branch(opcode_); }
    bool is_call() const noexcept { return brass::is_call(opcode_); }
    bool has_side_effects() const noexcept { return brass::has_side_effects(opcode_); }
    bool produces_value() const noexcept { return result_ != nullptr && !type_.is_void(); }

private:
    Opcode opcode_ = Opcode::unreachable;
    Type type_ = Type::void_type();
    Value* result_ = nullptr;
    BasicBlock* parent_ = nullptr;
    Instruction* prev_ = nullptr;
    Instruction* next_ = nullptr;

    std::vector<Value*> operands_;

    int64_t imm_i64_ = 0;
    double imm_f64_ = 0.0;
    int32_t offset_ = 0;
    uint8_t scale_ = 1;
    Type memory_type_ = Type::void_type();

    std::string_view symbol_;
    std::string_view extra_symbol_;

    BranchTarget branch_target_;
    BranchTarget true_target_;
    BranchTarget false_target_;

    std::vector<Value*> state_map_;
};

} // namespace brass
