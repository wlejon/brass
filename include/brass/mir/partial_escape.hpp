#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <cstdint>
#include <iosfwd>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

enum class ObjectState : uint8_t {
    Virtual = 0,
    Materialized = 1,
    Unknown = 2
};

std::string_view object_state_name(ObjectState state) noexcept;
std::ostream& operator<<(std::ostream& os, ObjectState state);

struct CFGEdge {
    const BasicBlock* from = nullptr;
    const BasicBlock* to = nullptr;

    bool operator==(const CFGEdge& o) const noexcept {
        return from == o.from && to == o.to;
    }
    bool operator!=(const CFGEdge& o) const noexcept {
        return !(*this == o);
    }
    bool operator<(const CFGEdge& o) const noexcept {
        if (from != o.from) return from < o.from;
        return to < o.to;
    }
};

struct VirtualField {
    int32_t offset = 0;
    Type type = Type::void_type();
    Value* value = nullptr;

    bool operator==(const VirtualField& o) const noexcept {
        return offset == o.offset && type == o.type && value == o.value;
    }
};

class VirtualObject {
public:
    VirtualObject() = default;
    VirtualObject(uint32_t id, Instruction* alloc_inst)
        : id_(id), alloc_inst_(alloc_inst) {}

    uint32_t id() const noexcept { return id_; }
    void set_id(uint32_t id) noexcept { id_ = id; }

    Instruction* alloc_inst() const noexcept { return alloc_inst_; }
    void set_alloc_inst(Instruction* inst) noexcept { alloc_inst_ = inst; }

    Value* alloc_value() const noexcept {
        return alloc_inst_ ? alloc_inst_->result() : nullptr;
    }

    ObjectState state() const noexcept { return state_; }
    void set_state(ObjectState s) noexcept { state_ = s; }

    void set_field(int32_t offset, Type type, Value* val);
    Value* get_field(int32_t offset) const;
    Type get_field_type(int32_t offset) const;
    bool has_field(int32_t offset) const;
    const std::map<int32_t, VirtualField>& fields() const noexcept { return fields_; }
    std::map<int32_t, VirtualField>& fields() noexcept { return fields_; }
    void clear_fields() noexcept { fields_.clear(); }

    bool equals_fields(const VirtualObject& other) const noexcept;

private:
    uint32_t id_ = 0;
    Instruction* alloc_inst_ = nullptr;
    ObjectState state_ = ObjectState::Virtual;
    std::map<int32_t, VirtualField> fields_;
};

struct PartialEscapeStats {
    size_t virtual_allocations = 0;
    size_t materialized_allocations = 0;
    size_t materialization_edges = 0;
    size_t scalarized_loads = 0;
    size_t scalarized_stores = 0;
    size_t sunk_allocations = 0;

    std::string format_report() const;
};

struct PartialEscapeOptions {
    size_t max_fields = 32;
    bool allow_loop_exit_escape = true;
    bool allow_cold_branch_escape = true;
};

class PartialEscapeAnalysis {
public:
    explicit PartialEscapeAnalysis(const Function& fn, const PartialEscapeOptions& options = {});
    ~PartialEscapeAnalysis();

    PartialEscapeAnalysis(const PartialEscapeAnalysis&) = delete;
    PartialEscapeAnalysis& operator=(const PartialEscapeAnalysis&) = delete;
    PartialEscapeAnalysis(PartialEscapeAnalysis&&) noexcept;
    PartialEscapeAnalysis& operator=(PartialEscapeAnalysis&&) noexcept;

    const Function& function() const noexcept { return *fn_; }

    // Analysis queries
    bool is_candidate(const Value* alloc_val) const;
    const std::vector<const Value*>& candidate_allocations() const noexcept { return candidates_; }

    ObjectState get_block_state(const BasicBlock* bb, const Value* alloc_val) const;
    const VirtualObject* get_virtual_object(const BasicBlock* bb, const Value* alloc_val) const;

    bool is_virtual_in_loop(const Value* alloc_val, const LoopInfo& loop) const;
    bool escapes_in_loop(const Value* alloc_val, const LoopInfo& loop) const;

    std::vector<CFGEdge> get_materialization_frontier(const Value* alloc_val) const;

    const std::unordered_set<const Value*>& get_aliases(const Value* alloc_val) const;

    void dump(std::ostream& os) const;

private:
    void analyze();
    void analyze_allocation(const Value* alloc_val);

    const Function* fn_ = nullptr;
    PartialEscapeOptions options_;
    std::vector<const Value*> candidates_;
    std::unordered_map<const Value*, std::unordered_set<const Value*>> aliases_map_;
    // Map: alloc_val -> (BasicBlock* -> VirtualObject)
    std::unordered_map<const Value*, std::unordered_map<const BasicBlock*, VirtualObject>> block_state_;
    // Map: alloc_val -> minimal materialization frontier
    std::unordered_map<const Value*, std::vector<CFGEdge>> frontiers_;
    static const std::unordered_set<const Value*> empty_aliases_;
};

} // namespace brass
