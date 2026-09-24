#pragma once

#include <brass/mir/types.hpp>
#include <brass/mir/block.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include <utility>

namespace brass {

class Module;

class Function {
public:
    Function() noexcept = default;
    Function(std::string_view name, Type return_type, std::vector<Type> param_types)
        : name_(name), return_type_(return_type), param_types_(std::move(param_types)) {}

    std::string_view name() const noexcept { return name_; }
    void set_name(std::string_view name) noexcept { name_ = name; }

    Type return_type() const noexcept { return return_type_; }
    void set_return_type(Type t) noexcept { return_type_ = t; }

    const std::vector<Type>& param_types() const noexcept { return param_types_; }
    void set_param_types(std::vector<Type> types) { param_types_ = std::move(types); }
    size_t param_count() const noexcept { return param_types_.size(); }
    Type param_type(size_t idx) const noexcept {
        return (idx < param_types_.size()) ? param_types_[idx] : Type::void_type();
    }

    bool is_param_noalias(size_t idx) const noexcept {
        return idx < param_noalias_.size() && param_noalias_[idx];
    }
    void set_param_noalias(size_t idx, bool noalias = true) {
        if (idx >= param_noalias_.size()) {
            param_noalias_.resize(idx + 1, false);
        }
        param_noalias_[idx] = noalias;
    }

    Module* parent() const noexcept { return parent_; }
    void set_parent(Module* m) noexcept { parent_ = m; }

    const std::vector<BasicBlock*>& blocks() const noexcept { return blocks_; }
    std::vector<BasicBlock*>& blocks() noexcept { return blocks_; }
    size_t block_count() const noexcept { return blocks_.size(); }

    BasicBlock* entry_block() const noexcept {
        return blocks_.empty() ? nullptr : blocks_.front();
    }

    void append_block(BasicBlock* bb);
    void prepend_block(BasicBlock* bb);
    void remove_block(BasicBlock* bb);
    void sort_blocks_rpo();
    BasicBlock* get_block_by_name(std::string_view name) const noexcept;
    BasicBlock* get_block_by_id(uint32_t id) const noexcept;

    void add_resume_point(uint32_t resume_id, BasicBlock* target);
    BasicBlock* get_resume_target(uint32_t resume_id) const noexcept;
    const std::vector<std::pair<uint32_t, BasicBlock*>>& resume_points() const noexcept {
        return resume_points_;
    }

    // The first guard of this function with `resume_id`, or null.
    const Instruction* find_guard(uint32_t resume_id) const noexcept;
    // The resume id a new guard of this function gets: one past the largest
    // id of its guards, 0 when it has none. A resume id names one Tier-0
    // guard; tier-2 deopt finds the guard to resume at by it.
    uint32_t next_guard_resume_id() const noexcept;
    // A guard's exit stub: the function of this function's module named by
    // the guard's exit label, or null when the label names none (it is then
    // only a label). Called as stub(state values...); its result is this
    // function's result. See "Guard exits" in docs/mir_reference.md.
    const Function* guard_exit_stub(const Instruction& guard) const noexcept;
    // Whether `stub` has the exit-stub signature for `guard` of this
    // function: one parameter per state value, of its type, and this
    // function's return type. When not, `why` says what differs.
    bool guard_exit_stub_matches(const Instruction& guard, const Function& stub, std::string& why) const;

    bool allow_fp_reassociation() const noexcept { return allow_fp_reassociation_; }
    void set_allow_fp_reassociation(bool allow) noexcept { allow_fp_reassociation_ = allow; }

    uint32_t next_value_id() noexcept { return next_value_id_++; }
    uint32_t next_block_id() noexcept { return next_block_id_++; }
    uint32_t current_next_value_id() const noexcept { return next_value_id_; }
    uint32_t current_next_block_id() const noexcept { return next_block_id_; }
    void set_next_value_id(uint32_t id) noexcept { next_value_id_ = id; }
    void set_next_block_id(uint32_t id) noexcept { next_block_id_ = id; }

    void rebuild_cfg_predecessors();

private:
    std::string_view name_;
    Type return_type_ = Type::void_type();
    std::vector<Type> param_types_;
    std::vector<bool> param_noalias_;

    std::vector<BasicBlock*> blocks_;
    Module* parent_ = nullptr;

    std::vector<std::pair<uint32_t, BasicBlock*>> resume_points_;

    uint32_t next_value_id_ = 0;
    uint32_t next_block_id_ = 0;
    bool allow_fp_reassociation_ = false;
};

} // namespace brass
