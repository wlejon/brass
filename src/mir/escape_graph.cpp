#include "escape_graph.hpp"
#include <queue>
#include <algorithm>

namespace brass {

ConnectionGraph::ConnectionGraph() = default;
ConnectionGraph::~ConnectionGraph() = default;
ConnectionGraph::ConnectionGraph(ConnectionGraph&&) noexcept = default;
ConnectionGraph& ConnectionGraph::operator=(ConnectionGraph&&) noexcept = default;

CGNode* ConnectionGraph::get_or_create_ref_node(const Value* val) {
    if (!val) return nullptr;
    auto it = val_to_ref_.find(val);
    if (it != val_to_ref_.end()) {
        return it->second;
    }

    auto node = std::make_unique<CGNode>();
    node->id = next_node_id_++;
    node->kind = CGNodeType::Reference;
    node->value = val;
    node->state = EscapeState::NoEscape;

    CGNode* raw = node.get();
    val_to_ref_[val] = raw;
    nodes_.push_back(std::move(node));
    return raw;
}

CGNode* ConnectionGraph::create_object_node(const Value* val, EscapeState initial_state) {
    auto node = std::make_unique<CGNode>();
    node->id = next_node_id_++;
    node->kind = CGNodeType::Object;
    node->value = val;
    node->state = initial_state;

    CGNode* raw = node.get();
    if (val) {
        val_to_obj_[val] = raw;
    }
    nodes_.push_back(std::move(node));
    return raw;
}

CGNode* ConnectionGraph::create_phantom_object(EscapeState initial_state) {
    return create_object_node(nullptr, initial_state);
}

CGNode* ConnectionGraph::get_or_create_field_node(CGNode* obj_node, int32_t offset) {
    if (!obj_node || !obj_node->is_object()) return nullptr;

    auto it = obj_node->fields.find(offset);
    if (it != obj_node->fields.end()) {
        return it->second;
    }

    auto node = std::make_unique<CGNode>();
    node->id = next_node_id_++;
    node->kind = CGNodeType::Field;
    node->offset = offset;
    node->parent_object = obj_node;
    node->state = obj_node->state;

    CGNode* raw = node.get();
    obj_node->fields[offset] = raw;
    nodes_.push_back(std::move(node));
    return raw;
}

void ConnectionGraph::add_points_to(CGNode* from, CGNode* to_obj) {
    if (!from || !to_obj) return;
    from->points_to.insert(to_obj);
}

void ConnectionGraph::add_deferred(CGNode* from, CGNode* to) {
    if (!from || !to || from == to) return;
    from->deferred.insert(to);
}

void ConnectionGraph::mark_escape_state(CGNode* node, EscapeState state) {
    if (!node) return;
    if (static_cast<uint8_t>(state) > static_cast<uint8_t>(node->state)) {
        node->state = state;
    }
}

void ConnectionGraph::propagate() {
    // 1. Fixed-point propagation of points-to sets along deferred edges
    bool points_to_changed = true;
    while (points_to_changed) {
        points_to_changed = false;
        for (const auto& u : nodes_) {
            if (!u) continue;
            for (CGNode* v : u->deferred) {
                if (!v) continue;
                size_t old_size = v->points_to.size();
                for (CGNode* obj : u->points_to) {
                    if (obj) {
                        v->points_to.insert(obj);
                    }
                }
                if (v->points_to.size() > old_size) {
                    points_to_changed = true;
                }
            }
        }
    }

    // 2. Escape State Propagation Worklist
    std::queue<CGNode*> worklist;
    for (const auto& node : nodes_) {
        if (node && node->state > EscapeState::NoEscape) {
            worklist.push(node.get());
        }
    }

    while (!worklist.empty()) {
        CGNode* u = worklist.front();
        worklist.pop();
        EscapeState u_st = u->state;

        auto update_state = [&](CGNode* target, EscapeState new_st) {
            if (!target) return;
            if (static_cast<uint8_t>(new_st) > static_cast<uint8_t>(target->state)) {
                target->state = new_st;
                worklist.push(target);
            }
        };

        if (u->is_object()) {
            // Fields of an escaping object inherit its escape state
            for (auto& entry : u->fields) {
                update_state(entry.second, u_st);
            }
            // If an object escapes, references pointing to it should reflect escape state
            for (const auto& other : nodes_) {
                if (other && other->points_to.count(u) > 0) {
                    update_state(other.get(), u_st);
                }
            }
        } else if (u->is_field()) {
            // Target objects pointed to by escaping field inherit escape state
            for (CGNode* target_obj : u->points_to) {
                update_state(target_obj, u_st);
            }
            // Field escape can also propagate to its parent object if parent state is lower
            if (u->parent_object) {
                update_state(u->parent_object, u_st);
            }
        } else if (u->is_reference()) {
            // Target objects pointed to by escaping reference inherit escape state
            for (CGNode* target_obj : u->points_to) {
                update_state(target_obj, u_st);
            }
            // Propagate along deferred edges
            for (CGNode* succ : u->deferred) {
                update_state(succ, u_st);
            }
        }
    }
}

EscapeState ConnectionGraph::get_escape_state(const Value* val) const {
    if (!val) return EscapeState::GlobalEscape;

    // Check if directly an allocation object
    auto obj_it = val_to_obj_.find(val);
    if (obj_it != val_to_obj_.end()) {
        return obj_it->second->state;
    }

    // Check reference node
    auto ref_it = val_to_ref_.find(val);
    if (ref_it != val_to_ref_.end()) {
        const CGNode* ref = ref_it->second;
        if (ref->points_to.empty()) {
            return ref->state;
        }
        EscapeState max_st = ref->state;
        for (const CGNode* obj : ref->points_to) {
            if (obj && static_cast<uint8_t>(obj->state) > static_cast<uint8_t>(max_st)) {
                max_st = obj->state;
            }
        }
        return max_st;
    }

    // Untracked value escapes globally by default
    return EscapeState::GlobalEscape;
}

bool ConnectionGraph::does_escape(const Value* val) const {
    return get_escape_state(val) != EscapeState::NoEscape;
}

CGNode* ConnectionGraph::find_ref_node(const Value* val) const {
    auto it = val_to_ref_.find(val);
    return it != val_to_ref_.end() ? it->second : nullptr;
}

CGNode* ConnectionGraph::find_obj_node(const Value* val) const {
    auto it = val_to_obj_.find(val);
    return it != val_to_obj_.end() ? it->second : nullptr;
}

} // namespace brass
