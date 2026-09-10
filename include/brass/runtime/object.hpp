#pragma once

#include <brass/embedding/nanbox.hpp>
#include <brass/runtime/shape.hpp>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <string_view>
#include <optional>

namespace brass {
class HostGC;
}

namespace brass::runtime {

struct DynamicObjectBuffer {
    uint32_t capacity = 0;
    uint32_t length = 0;
    // Followed immediately by HostValue data[capacity]
};

class DynamicObject {
public:
    static constexpr size_t DEFAULT_INLINE_SLOTS = 8;
    static constexpr uint32_t TYPE_TAG_DYNAMIC_OBJECT = 100;
    static constexpr uint32_t TYPE_TAG_OOL_BUFFER     = 101;
    static constexpr uint32_t TYPE_TAG_ELEMENT_BUFFER = 102;

    // Pointer mask for Cheney GC / HostGC:
    // Word 2: out_of_line_slots (bit 2)
    // Words 3..10: inline_slots[0..7] (bits 3..10)
    // Word 12: elements (bit 12)
    static constexpr uint64_t POINTER_MASK = (1ULL << 2) | (0xFFULL << 3) | (1ULL << 12);

    // Object Layout
    Shape* shape = nullptr;                                    // Word 0 (offset 0)
    uint32_t inline_capacity = DEFAULT_INLINE_SLOTS;          // Word 1 (offset 8) lower 32
    uint32_t out_of_line_capacity = 0;                         // Word 1 (offset 8) upper 32
    uintptr_t out_of_line_slots = 0;                           // Word 2 (offset 16)
    HostValue inline_slots[DEFAULT_INLINE_SLOTS];              // Words 3..10 (offsets 24..80)

    uint32_t element_count = 0;                                // Word 11 (offset 88) lower 32
    uint32_t element_capacity = 0;                             // Word 11 (offset 88) upper 32
    uintptr_t elements = 0;                                    // Word 12 (offset 96)

    // Construction / Allocation
    static DynamicObject* create(HostGC* gc = nullptr, Shape* initial_shape = nullptr);
    static DynamicObject* create_array(HostGC* gc = nullptr, size_t initial_cap = 8);
    static void destroy_non_gc(DynamicObject* obj) noexcept;

    // Property Access
    [[nodiscard]] HostValue get_slot(uint32_t slot_index) const noexcept;
    void set_slot(uint32_t slot_index, HostValue value, HostGC* gc = nullptr);

    [[nodiscard]] HostValue get_property(std::string_view name) const;
    [[nodiscard]] HostValue get_property(uint32_t symbol_id) const;
    [[nodiscard]] bool has_property(std::string_view name) const;
    [[nodiscard]] bool has_property(uint32_t symbol_id) const;

    void set_property(std::string_view name, HostValue value, ShapeRegistry& registry, HostGC* gc = nullptr);
    void set_property(uint32_t symbol_id, HostValue value, ShapeRegistry& registry, HostGC* gc = nullptr);

    // Indexed Element Access
    [[nodiscard]] HostValue get_element(int64_t index) const noexcept;
    void set_element(int64_t index, HostValue value, HostGC* gc = nullptr);
    [[nodiscard]] size_t length() const noexcept { return element_count; }
    void set_length(size_t len) noexcept { element_count = static_cast<uint32_t>(len); }

    // Moving GC root visiting
    void visit_roots(
        const std::function<void(HostValue*)>& val_visitor,
        const std::function<void(uintptr_t*)>& ptr_visitor
    );

private:
    void ensure_out_of_line_capacity(size_t needed_cap, HostGC* gc);
    void ensure_element_capacity(size_t needed_cap, HostGC* gc);
};

// C Bridge Runtime Helpers
extern "C" {
    uint64_t brass_dynamic_object_create();
    uint64_t brass_dynamic_object_create_array(int32_t size);
    uint64_t brass_dynamic_object_get_prop_str(uint64_t obj_raw, const char* name);
    uint64_t brass_dynamic_object_get_prop_sym(uint64_t obj_raw, uint32_t symbol_id);
    void brass_dynamic_object_set_prop_str(uint64_t obj_raw, const char* name, uint64_t val_raw);
    void brass_dynamic_object_set_prop_sym(uint64_t obj_raw, uint32_t symbol_id, uint64_t val_raw);
    uint64_t brass_dynamic_object_get_elem(uint64_t obj_raw, int64_t index);
    void brass_dynamic_object_set_elem(uint64_t obj_raw, int64_t index, uint64_t val_raw);
}

} // namespace brass::runtime
