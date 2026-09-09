#pragma once

#include <brass/mir/escape_analysis.hpp>
#include <brass/mir/instruction.hpp>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

enum class CGNodeType : uint8_t {
    Object,
    Field,
    Reference
};

struct CGNode {
    uint32_t id = 0;
    CGNodeType kind = CGNodeType::Reference;
    EscapeState state = EscapeState::NoEscape;
    const Value* value = nullptr;
    int32_t offset = 0;
    CGNode* parent_object = nullptr;

    std::unordered_map<int32_t, CGNode*> fields;
    std::unordered_set<CGNode*> points_to;
    std::unordered_set<CGNode*> deferred;

    bool is_object() const noexcept { return kind == CGNodeType::Object; }
    bool is_field() const noexcept { return kind == CGNodeType::Field; }
    bool is_reference() const noexcept { return kind == CGNodeType::Reference; }
};

class ConnectionGraph {
public:
    ConnectionGraph();
    ~ConnectionGraph();

    ConnectionGraph(const ConnectionGraph&) = delete;
    ConnectionGraph& operator=(const ConnectionGraph&) = delete;
    ConnectionGraph(ConnectionGraph&&) noexcept;
    ConnectionGraph& operator=(ConnectionGraph&&) noexcept;

    CGNode* get_or_create_ref_node(const Value* val);
    CGNode* create_object_node(const Value* val, EscapeState initial_state);
    CGNode* create_phantom_object(EscapeState initial_state);
    CGNode* get_or_create_field_node(CGNode* obj_node, int32_t offset);

    void add_points_to(CGNode* from, CGNode* to_obj);
    void add_deferred(CGNode* from, CGNode* to);
    void mark_escape_state(CGNode* node, EscapeState state);

    void propagate();

    EscapeState get_escape_state(const Value* val) const;
    bool does_escape(const Value* val) const;

    CGNode* find_ref_node(const Value* val) const;
    CGNode* find_obj_node(const Value* val) const;

    const std::vector<std::unique_ptr<CGNode>>& all_nodes() const noexcept { return nodes_; }

private:
    uint32_t next_node_id_ = 1;
    std::vector<std::unique_ptr<CGNode>> nodes_;
    std::unordered_map<const Value*, CGNode*> val_to_ref_;
    std::unordered_map<const Value*, CGNode*> val_to_obj_;
};

} // namespace brass
