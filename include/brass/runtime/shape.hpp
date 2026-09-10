#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <optional>

namespace brass::runtime {

enum class PropertyAttributes : uint8_t {
    None        = 0,
    ReadOnly    = 1 << 0,
    DontEnum    = 1 << 1,
    DontDelete  = 1 << 2,
    Method      = 1 << 3
};

inline constexpr PropertyAttributes operator|(PropertyAttributes a, PropertyAttributes b) noexcept {
    return static_cast<PropertyAttributes>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b));
}

inline constexpr PropertyAttributes operator&(PropertyAttributes a, PropertyAttributes b) noexcept {
    return static_cast<PropertyAttributes>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b));
}

inline constexpr PropertyAttributes operator~(PropertyAttributes a) noexcept {
    return static_cast<PropertyAttributes>(~static_cast<uint8_t>(a));
}

inline constexpr bool has_attribute(PropertyAttributes attrs, PropertyAttributes flag) noexcept {
    return (static_cast<uint8_t>(attrs) & static_cast<uint8_t>(flag)) != 0;
}

struct PropertyDescriptor {
    std::string name;
    uint32_t symbol_id = 0;
    uint32_t slot_index = 0;
    PropertyAttributes attributes = PropertyAttributes::None;

    PropertyDescriptor() = default;
    PropertyDescriptor(std::string prop_name, uint32_t sym_id, uint32_t slot, PropertyAttributes attrs = PropertyAttributes::None)
        : name(std::move(prop_name)), symbol_id(sym_id), slot_index(slot), attributes(attrs) {}

    [[nodiscard]] constexpr bool is_readonly() const noexcept {
        return has_attribute(attributes, PropertyAttributes::ReadOnly);
    }
    [[nodiscard]] constexpr bool is_enumerable() const noexcept {
        return !has_attribute(attributes, PropertyAttributes::DontEnum);
    }
    [[nodiscard]] constexpr bool is_configurable() const noexcept {
        return !has_attribute(attributes, PropertyAttributes::DontDelete);
    }
    [[nodiscard]] constexpr bool is_method() const noexcept {
        return has_attribute(attributes, PropertyAttributes::Method);
    }

    bool operator==(const PropertyDescriptor& other) const noexcept {
        return name == other.name &&
               symbol_id == other.symbol_id &&
               slot_index == other.slot_index &&
               attributes == other.attributes;
    }
};

class ShapeRegistry;

class Shape {
public:
    Shape(uint32_t id, Shape* parent, PropertyDescriptor transition_prop, uint32_t slot_count);

    Shape(const Shape&) = delete;
    Shape& operator=(const Shape&) = delete;

    [[nodiscard]] uint32_t id() const noexcept { return id_; }
    [[nodiscard]] const Shape* parent_shape() const noexcept { return parent_; }
    [[nodiscard]] Shape* parent_shape() noexcept { return parent_; }
    [[nodiscard]] const PropertyDescriptor& transition_property() const noexcept { return transition_property_; }
    [[nodiscard]] uint32_t slot_count() const noexcept { return slot_count_; }
    [[nodiscard]] size_t property_count() const noexcept { return properties_.size(); }
    [[nodiscard]] const std::vector<PropertyDescriptor>& properties() const noexcept { return properties_; }

    static constexpr size_t FAST_SYMBOL_CAP = 64;

    [[nodiscard]] inline int16_t fast_symbol_to_slot(uint32_t symbol_id) const noexcept {
        if (symbol_id < FAST_SYMBOL_CAP) {
            return fast_symbol_to_slot_[symbol_id];
        }
        return -1;
    }

    [[nodiscard]] const PropertyDescriptor* find_property(std::string_view name) const noexcept;
    [[nodiscard]] const PropertyDescriptor* find_property(uint32_t symbol_id) const noexcept;
    [[nodiscard]] std::optional<uint32_t> find_slot(std::string_view name) const noexcept;
    [[nodiscard]] std::optional<uint32_t> find_slot(uint32_t symbol_id) const noexcept;
    [[nodiscard]] std::optional<uint32_t> find_slot_slow(uint32_t symbol_id) const noexcept;

    // Transition lookup
    [[nodiscard]] Shape* find_transition(std::string_view name) const noexcept;
    [[nodiscard]] Shape* find_transition(uint32_t symbol_id) const noexcept;
    [[nodiscard]] Shape* find_transition_slow(uint32_t symbol_id) const noexcept;

    // Add transition to child shape
    void add_transition(std::string_view name, Shape* child);
    void add_transition(uint32_t symbol_id, Shape* child);

    [[nodiscard]] const std::unordered_map<std::string, Shape*>& string_transitions() const noexcept {
        return string_transitions_;
    }
    [[nodiscard]] const std::unordered_map<uint32_t, Shape*>& symbol_transitions() const noexcept {
        return symbol_transitions_;
    }

private:
    uint32_t id_ = 0;
    Shape* parent_ = nullptr;
    PropertyDescriptor transition_property_;
    uint32_t slot_count_ = 0;
    std::vector<PropertyDescriptor> properties_;

    // Fast lookup tables for small symbol IDs (0..63)
    int16_t fast_symbol_to_slot_[FAST_SYMBOL_CAP];
    Shape* fast_symbol_transitions_[FAST_SYMBOL_CAP];

    // Fast lookup maps for properties
    std::unordered_map<std::string, uint32_t> name_to_prop_idx_;
    std::unordered_map<uint32_t, uint32_t> symbol_to_prop_idx_;

    // Outgoing transitions to children
    std::unordered_map<std::string, Shape*> string_transitions_;
    std::unordered_map<uint32_t, Shape*> symbol_transitions_;
};

class ShapeRegistry {
public:
    ShapeRegistry();
    ~ShapeRegistry() = default;

    ShapeRegistry(const ShapeRegistry&) = delete;
    ShapeRegistry& operator=(const ShapeRegistry&) = delete;
    ShapeRegistry(ShapeRegistry&&) noexcept;
    ShapeRegistry& operator=(ShapeRegistry&&) noexcept;

    // Access the canonical root shape (empty shape with 0 properties)
    Shape* get_root_shape();

    // Canonical transitions: reuses child if transition already exists from current shape
    Shape* transition_to(
        Shape* current,
        std::string_view name,
        uint32_t symbol_id = 0,
        PropertyAttributes attrs = PropertyAttributes::None
    );

    Shape* transition_to(
        Shape* current,
        uint32_t symbol_id,
        PropertyAttributes attrs = PropertyAttributes::None
    );

    [[nodiscard]] const Shape* find_shape_by_id(uint32_t id) const;
    [[nodiscard]] size_t shape_count() const noexcept;

    void clear();

    // Global canonical shape registry
    static ShapeRegistry& global() noexcept;

private:
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<Shape>> shapes_;
    Shape* root_shape_ = nullptr;
    uint32_t next_id_ = 1;
};

} // namespace brass::runtime
