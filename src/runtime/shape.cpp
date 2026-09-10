#include <brass/runtime/shape.hpp>
#include <stdexcept>
#include <string>
#include <cstring>

namespace brass::runtime {

Shape::Shape(uint32_t id, Shape* parent, PropertyDescriptor transition_prop, uint32_t slot_count)
    : id_(id),
      parent_(parent),
      transition_property_(std::move(transition_prop)),
      slot_count_(slot_count) {
    std::memset(fast_symbol_transitions_, 0, sizeof(fast_symbol_transitions_));
    if (parent_ != nullptr) {
        properties_ = parent_->properties_;
        name_to_prop_idx_ = parent_->name_to_prop_idx_;
        symbol_to_prop_idx_ = parent_->symbol_to_prop_idx_;
        std::memcpy(fast_symbol_to_slot_, parent_->fast_symbol_to_slot_, sizeof(fast_symbol_to_slot_));
    } else {
        std::memset(fast_symbol_to_slot_, 0xFF, sizeof(fast_symbol_to_slot_));
    }

    if (!transition_property_.name.empty() || transition_property_.symbol_id != 0) {
        uint32_t idx = static_cast<uint32_t>(properties_.size());
        properties_.push_back(transition_property_);
        if (!transition_property_.name.empty()) {
            name_to_prop_idx_[transition_property_.name] = idx;
        }
        if (transition_property_.symbol_id != 0) {
            symbol_to_prop_idx_[transition_property_.symbol_id] = idx;
            if (transition_property_.symbol_id < FAST_SYMBOL_CAP) {
                fast_symbol_to_slot_[transition_property_.symbol_id] = static_cast<int16_t>(transition_property_.slot_index);
            }
        }
    }
}

const PropertyDescriptor* Shape::find_property(std::string_view name) const noexcept {
    auto it = name_to_prop_idx_.find(std::string(name));
    if (it != name_to_prop_idx_.end()) {
        return &properties_[it->second];
    }
    return nullptr;
}

const PropertyDescriptor* Shape::find_property(uint32_t symbol_id) const noexcept {
    auto it = symbol_to_prop_idx_.find(symbol_id);
    if (it != symbol_to_prop_idx_.end()) {
        return &properties_[it->second];
    }
    return nullptr;
}

std::optional<uint32_t> Shape::find_slot(std::string_view name) const noexcept {
    const auto* prop = find_property(name);
    if (prop != nullptr) {
        return prop->slot_index;
    }
    return std::nullopt;
}

std::optional<uint32_t> Shape::find_slot(uint32_t symbol_id) const noexcept {
    if (symbol_id < FAST_SYMBOL_CAP) {
        int16_t s = fast_symbol_to_slot_[symbol_id];
        if (s >= 0) return static_cast<uint32_t>(s);
    }
    return find_slot_slow(symbol_id);
}

std::optional<uint32_t> Shape::find_slot_slow(uint32_t symbol_id) const noexcept {
    const auto* prop = find_property(symbol_id);
    if (prop != nullptr) {
        return prop->slot_index;
    }
    return std::nullopt;
}

Shape* Shape::find_transition(std::string_view name) const noexcept {
    auto it = string_transitions_.find(std::string(name));
    if (it != string_transitions_.end()) {
        return it->second;
    }
    return nullptr;
}

Shape* Shape::find_transition(uint32_t symbol_id) const noexcept {
    if (symbol_id < FAST_SYMBOL_CAP) {
        Shape* trans = fast_symbol_transitions_[symbol_id];
        if (trans != nullptr) return trans;
    }
    return find_transition_slow(symbol_id);
}

Shape* Shape::find_transition_slow(uint32_t symbol_id) const noexcept {
    auto it = symbol_transitions_.find(symbol_id);
    if (it != symbol_transitions_.end()) {
        return it->second;
    }
    return nullptr;
}

void Shape::add_transition(std::string_view name, Shape* child) {
    if (!name.empty() && child != nullptr) {
        string_transitions_[std::string(name)] = child;
    }
}

void Shape::add_transition(uint32_t symbol_id, Shape* child) {
    if (symbol_id != 0 && child != nullptr) {
        if (symbol_id < FAST_SYMBOL_CAP) {
            fast_symbol_transitions_[symbol_id] = child;
        }
        symbol_transitions_[symbol_id] = child;
    }
}

ShapeRegistry::ShapeRegistry() {
    auto root = std::make_unique<Shape>(next_id_++, nullptr, PropertyDescriptor{}, 0);
    root_shape_ = root.get();
    shapes_.push_back(std::move(root));
}

ShapeRegistry::ShapeRegistry(ShapeRegistry&& other) noexcept {
    std::lock_guard<std::mutex> lock(other.mutex_);
    shapes_ = std::move(other.shapes_);
    root_shape_ = other.root_shape_;
    next_id_ = other.next_id_;
    other.root_shape_ = nullptr;
}

ShapeRegistry& ShapeRegistry::operator=(ShapeRegistry&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(mutex_, other.mutex_);
        shapes_ = std::move(other.shapes_);
        root_shape_ = other.root_shape_;
        next_id_ = other.next_id_;
        other.root_shape_ = nullptr;
    }
    return *this;
}

Shape* ShapeRegistry::get_root_shape() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!root_shape_) {
        auto root = std::make_unique<Shape>(next_id_++, nullptr, PropertyDescriptor{}, 0);
        root_shape_ = root.get();
        shapes_.push_back(std::move(root));
    }
    return root_shape_;
}

Shape* ShapeRegistry::transition_to(
    Shape* current,
    std::string_view name,
    uint32_t symbol_id,
    PropertyAttributes attrs
) {
    if (!current) {
        current = get_root_shape();
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // 1. Check for existing transition (duplicate property reuse)
    if (!name.empty()) {
        Shape* existing = current->find_transition(name);
        if (existing) {
            return existing;
        }
    }
    if (symbol_id != 0) {
        Shape* existing = current->find_transition(symbol_id);
        if (existing) {
            return existing;
        }
    }

    // 2. Check if the property already exists on current shape with same attributes
    const PropertyDescriptor* existing_desc = !name.empty() ? current->find_property(name) : nullptr;
    if (!existing_desc && symbol_id != 0) {
        existing_desc = current->find_property(symbol_id);
    }
    if (existing_desc != nullptr && existing_desc->attributes == attrs) {
        return current;
    }

    // 3. Allocate new slot
    uint32_t new_slot = current->slot_count();
    PropertyDescriptor new_prop(std::string(name), symbol_id, new_slot, attrs);

    auto child = std::make_unique<Shape>(next_id_++, current, std::move(new_prop), new_slot + 1);
    Shape* child_ptr = child.get();

    if (!name.empty()) {
        current->add_transition(name, child_ptr);
    }
    if (symbol_id != 0) {
        current->add_transition(symbol_id, child_ptr);
    }

    shapes_.push_back(std::move(child));
    return child_ptr;
}

Shape* ShapeRegistry::transition_to(
    Shape* current,
    uint32_t symbol_id,
    PropertyAttributes attrs
) {
    return transition_to(current, "", symbol_id, attrs);
}

const Shape* ShapeRegistry::find_shape_by_id(uint32_t id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& s : shapes_) {
        if (s->id() == id) {
            return s.get();
        }
    }
    return nullptr;
}

size_t ShapeRegistry::shape_count() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return shapes_.size();
}

void ShapeRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    shapes_.clear();
    next_id_ = 1;
    auto root = std::make_unique<Shape>(next_id_++, nullptr, PropertyDescriptor{}, 0);
    root_shape_ = root.get();
    shapes_.push_back(std::move(root));
}

ShapeRegistry& ShapeRegistry::global() noexcept {
    static ShapeRegistry s_global;
    return s_global;
}

} // namespace brass::runtime
