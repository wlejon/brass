#pragma once

#include <brass/mir/types.hpp>
#include <brass/mir/block.hpp>
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
    BasicBlock* get_block_by_name(std::string_view name) const noexcept;
    BasicBlock* get_block_by_id(uint32_t id) const noexcept;

    void add_resume_point(uint32_t resume_id, BasicBlock* target);
    BasicBlock* get_resume_target(uint32_t resume_id) const noexcept;
    const std::vector<std::pair<uint32_t, BasicBlock*>>& resume_points() const noexcept {
        return resume_points_;
    }

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

    std::vector<BasicBlock*> blocks_;
    Module* parent_ = nullptr;

    std::vector<std::pair<uint32_t, BasicBlock*>> resume_points_;

    uint32_t next_value_id_ = 0;
    uint32_t next_block_id_ = 0;
    bool allow_fp_reassociation_ = false;
};

} // namespace brass
