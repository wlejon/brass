#include <brass/runtime/object.hpp>
#include <brass/embedding/host_gc.hpp>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace brass::runtime {

namespace {

static uint64_t compute_buffer_pointer_mask(size_t cap) noexcept {
    if (cap == 0) return 0;
    size_t clamped = std::min(cap, size_t(62));
    uint64_t mask = (clamped >= 62) ? ~0ULL : ((1ULL << clamped) - 1ULL);
    return mask << 1; // Word 0 is capacity+length header
}

static uintptr_t allocate_buffer(size_t cap, uint32_t type_tag, HostGC* gc) {
    size_t byte_size = sizeof(DynamicObjectBuffer) + cap * sizeof(HostValue);
    uint64_t mask = compute_buffer_pointer_mask(cap);

    if (gc != nullptr) {
        uintptr_t addr = gc->allocate(byte_size, mask, type_tag);
        auto* buf = reinterpret_cast<DynamicObjectBuffer*>(addr);
        buf->capacity = static_cast<uint32_t>(cap);
        buf->length = 0;
        auto* slots = reinterpret_cast<HostValue*>(addr + sizeof(DynamicObjectBuffer));
        for (size_t i = 0; i < cap; ++i) {
            slots[i] = HostValue::undefined_val();
        }
        return addr;
    }

    void* mem = std::calloc(1, byte_size);
    if (!mem) return 0;
    auto* buf = static_cast<DynamicObjectBuffer*>(mem);
    buf->capacity = static_cast<uint32_t>(cap);
    buf->length = 0;
    auto* slots = reinterpret_cast<HostValue*>(static_cast<uint8_t*>(mem) + sizeof(DynamicObjectBuffer));
    for (size_t i = 0; i < cap; ++i) {
        slots[i] = HostValue::undefined_val();
    }
    return reinterpret_cast<uintptr_t>(mem);
}

static void free_buffer_non_gc(uintptr_t buf_addr) noexcept {
    if (buf_addr != 0) {
        std::free(reinterpret_cast<void*>(buf_addr));
    }
}

} // anonymous namespace

DynamicObject* DynamicObject::create(HostGC* gc, Shape* initial_shape) {
    if (!initial_shape) {
        initial_shape = ShapeRegistry::global().get_root_shape();
    }

    if (gc != nullptr) {
        uintptr_t addr = gc->allocate(sizeof(DynamicObject), POINTER_MASK, TYPE_TAG_DYNAMIC_OBJECT);
        auto* obj = reinterpret_cast<DynamicObject*>(addr);
        obj->shape = initial_shape;
        obj->inline_capacity = static_cast<uint32_t>(DEFAULT_INLINE_SLOTS);
        obj->out_of_line_capacity = 0;
        obj->out_of_line_slots = 0;
        for (size_t i = 0; i < DEFAULT_INLINE_SLOTS; ++i) {
            obj->inline_slots[i] = HostValue::undefined_val();
        }
        obj->element_count = 0;
        obj->element_capacity = 0;
        obj->elements = 0;
        return obj;
    }

    void* mem = std::calloc(1, sizeof(DynamicObject));
    if (!mem) return nullptr;
    auto* obj = static_cast<DynamicObject*>(mem);
    obj->shape = initial_shape;
    obj->inline_capacity = static_cast<uint32_t>(DEFAULT_INLINE_SLOTS);
    obj->out_of_line_capacity = 0;
    obj->out_of_line_slots = 0;
    for (size_t i = 0; i < DEFAULT_INLINE_SLOTS; ++i) {
        obj->inline_slots[i] = HostValue::undefined_val();
    }
    obj->element_count = 0;
    obj->element_capacity = 0;
    obj->elements = 0;
    return obj;
}

DynamicObject* DynamicObject::create_array(HostGC* gc, size_t initial_cap) {
    DynamicObject* obj = create(gc, nullptr);
    if (!obj) return nullptr;
    if (initial_cap > 0) {
        obj->ensure_element_capacity(initial_cap, gc);
    }
    return obj;
}

void DynamicObject::destroy_non_gc(DynamicObject* obj) noexcept {
    if (obj) {
        free_buffer_non_gc(obj->out_of_line_slots);
        free_buffer_non_gc(obj->elements);
        std::free(obj);
    }
}

void DynamicObject::ensure_out_of_line_capacity(size_t needed_cap, HostGC* gc) {
    if (needed_cap <= out_of_line_capacity) return;

    size_t new_cap = std::max(needed_cap, static_cast<size_t>(out_of_line_capacity * 2));
    if (new_cap < 8) new_cap = 8;

    uintptr_t new_buf_addr = allocate_buffer(new_cap, TYPE_TAG_OOL_BUFFER, gc);
    auto* new_slots = reinterpret_cast<HostValue*>(new_buf_addr + sizeof(DynamicObjectBuffer));

    if (out_of_line_slots != 0) {
        const auto* old_slots = reinterpret_cast<const HostValue*>(out_of_line_slots + sizeof(DynamicObjectBuffer));
        for (uint32_t i = 0; i < out_of_line_capacity; ++i) {
            new_slots[i] = old_slots[i];
        }
        if (gc == nullptr) {
            free_buffer_non_gc(out_of_line_slots);
        }
    }

    out_of_line_slots = new_buf_addr;
    out_of_line_capacity = static_cast<uint32_t>(new_cap);
}

void DynamicObject::ensure_element_capacity(size_t needed_cap, HostGC* gc) {
    if (needed_cap <= element_capacity) return;

    size_t new_cap = std::max(needed_cap, static_cast<size_t>(element_capacity * 2));
    if (new_cap < 8) new_cap = 8;

    uintptr_t new_buf_addr = allocate_buffer(new_cap, TYPE_TAG_ELEMENT_BUFFER, gc);
    auto* new_slots = reinterpret_cast<HostValue*>(new_buf_addr + sizeof(DynamicObjectBuffer));

    if (elements != 0) {
        const auto* old_slots = reinterpret_cast<const HostValue*>(elements + sizeof(DynamicObjectBuffer));
        for (uint32_t i = 0; i < element_capacity; ++i) {
            new_slots[i] = old_slots[i];
        }
        if (gc == nullptr) {
            free_buffer_non_gc(elements);
        }
    }

    elements = new_buf_addr;
    element_capacity = static_cast<uint32_t>(new_cap);
}

HostValue DynamicObject::get_slot(uint32_t slot_index) const noexcept {
    if (slot_index < inline_capacity) {
        return inline_slots[slot_index];
    }
    uint32_t ool_idx = slot_index - inline_capacity;
    if (out_of_line_slots != 0 && ool_idx < out_of_line_capacity) {
        const auto* slots = reinterpret_cast<const HostValue*>(out_of_line_slots + sizeof(DynamicObjectBuffer));
        return slots[ool_idx];
    }
    return HostValue::undefined_val();
}

void DynamicObject::set_slot(uint32_t slot_index, HostValue value, HostGC* gc) {
    if (slot_index < inline_capacity) {
        inline_slots[slot_index] = value;
        return;
    }
    uint32_t ool_idx = slot_index - inline_capacity;
    ensure_out_of_line_capacity(ool_idx + 1, gc);
    auto* slots = reinterpret_cast<HostValue*>(out_of_line_slots + sizeof(DynamicObjectBuffer));
    slots[ool_idx] = value;
}

HostValue DynamicObject::get_property(std::string_view name) const {
    if (!shape) return HostValue::undefined_val();
    auto slot = shape->find_slot(name);
    if (slot) {
        return get_slot(*slot);
    }
    return HostValue::undefined_val();
}

HostValue DynamicObject::get_property(uint32_t symbol_id) const {
    if (!shape) return HostValue::undefined_val();
    auto slot = shape->find_slot(symbol_id);
    if (slot) {
        return get_slot(*slot);
    }
    return HostValue::undefined_val();
}

bool DynamicObject::has_property(std::string_view name) const {
    return shape && shape->find_property(name) != nullptr;
}

bool DynamicObject::has_property(uint32_t symbol_id) const {
    return shape && shape->find_property(symbol_id) != nullptr;
}

void DynamicObject::set_property(
    std::string_view name,
    HostValue value,
    ShapeRegistry& registry,
    HostGC* gc
) {
    if (!shape) {
        shape = registry.get_root_shape();
    }
    auto slot = shape->find_slot(name);
    if (slot) {
        set_slot(*slot, value, gc);
    } else {
        shape = registry.transition_to(shape, name);
        auto new_slot = shape->find_slot(name);
        if (new_slot) {
            set_slot(*new_slot, value, gc);
        }
    }
}

void DynamicObject::set_property(
    uint32_t symbol_id,
    HostValue value,
    ShapeRegistry& registry,
    HostGC* gc
) {
    if (!shape) {
        shape = registry.get_root_shape();
    }
    auto slot = shape->find_slot(symbol_id);
    if (slot) {
        set_slot(*slot, value, gc);
    } else {
        shape = registry.transition_to(shape, symbol_id);
        auto new_slot = shape->find_slot(symbol_id);
        if (new_slot) {
            set_slot(*new_slot, value, gc);
        }
    }
}

HostValue DynamicObject::get_element(int64_t index) const noexcept {
    if (index < 0 || static_cast<uint64_t>(index) >= element_count || elements == 0) {
        return HostValue::undefined_val();
    }
    const auto* slots = reinterpret_cast<const HostValue*>(elements + sizeof(DynamicObjectBuffer));
    return slots[index];
}

void DynamicObject::set_element(int64_t index, HostValue value, HostGC* gc) {
    if (index < 0) return;
    uint32_t uidx = static_cast<uint32_t>(index);
    ensure_element_capacity(uidx + 1, gc);
    auto* slots = reinterpret_cast<HostValue*>(elements + sizeof(DynamicObjectBuffer));
    slots[uidx] = value;
    if (uidx + 1 > element_count) {
        element_count = uidx + 1;
    }
}

void DynamicObject::visit_roots(
    const std::function<void(HostValue*)>& val_visitor,
    const std::function<void(uintptr_t*)>& ptr_visitor
) {
    // 1. Inline slots
    for (size_t i = 0; i < inline_capacity; ++i) {
        val_visitor(&inline_slots[i]);
    }

    // 2. Out-of-line storage
    if (out_of_line_slots != 0) {
        ptr_visitor(&out_of_line_slots);
        auto* ool_slots = reinterpret_cast<HostValue*>(out_of_line_slots + sizeof(DynamicObjectBuffer));
        for (uint32_t i = 0; i < out_of_line_capacity; ++i) {
            val_visitor(&ool_slots[i]);
        }
    }

    // 3. Array elements
    if (elements != 0) {
        ptr_visitor(&elements);
        auto* elem_slots = reinterpret_cast<HostValue*>(elements + sizeof(DynamicObjectBuffer));
        for (uint32_t i = 0; i < element_capacity; ++i) {
            val_visitor(&elem_slots[i]);
        }
    }
}

} // namespace brass::runtime

extern "C" {

using namespace brass;
using namespace brass::runtime;

uint64_t brass_dynamic_object_create() {
    HostGC* gc = get_active_host_gc();
    DynamicObject* obj = DynamicObject::create(gc, ShapeRegistry::global().get_root_shape());
    return HostValue::from_gcref(obj).raw();
}

uint64_t brass_dynamic_object_create_array(int32_t size) {
    HostGC* gc = get_active_host_gc();
    size_t initial_cap = size > 0 ? static_cast<size_t>(size) : 8;
    DynamicObject* obj = DynamicObject::create_array(gc, initial_cap);
    if (obj && size > 0) {
        obj->set_length(static_cast<size_t>(size));
    }
    return HostValue::from_gcref(obj).raw();
}

static inline DynamicObject* unpack_dynamic_object_bridge(uint64_t obj_raw) {
    if (obj_raw == 0) return nullptr;
    HostValue hv(obj_raw);
    if (hv.is_gcref()) {
        return hv.as_gcref_ptr<DynamicObject>();
    }
    if (obj_raw < 0x0000800000000000ULL && obj_raw >= 0x1000ULL) {
        return reinterpret_cast<DynamicObject*>(obj_raw);
    }
    return nullptr;
}

uint64_t brass_dynamic_object_get_prop_str(uint64_t obj_raw, const char* name) {
    if (!name) return HostValue::undefined_val().raw();
    auto* obj = unpack_dynamic_object_bridge(obj_raw);
    if (!obj) return HostValue::undefined_val().raw();
    return obj->get_property(name).raw();
}

uint64_t brass_dynamic_object_get_prop_sym(uint64_t obj_raw, uint32_t symbol_id) {
    auto* obj = unpack_dynamic_object_bridge(obj_raw);
    if (!obj) return HostValue::undefined_val().raw();
    return obj->get_property(symbol_id).raw();
}

void brass_dynamic_object_set_prop_str(uint64_t obj_raw, const char* name, uint64_t val_raw) {
    if (!name) return;
    auto* obj = unpack_dynamic_object_bridge(obj_raw);
    if (!obj) return;
    HostGC* gc = get_active_host_gc();
    obj->set_property(name, HostValue(val_raw), ShapeRegistry::global(), gc);
}

void brass_dynamic_object_set_prop_sym(uint64_t obj_raw, uint32_t symbol_id, uint64_t val_raw) {
    auto* obj = unpack_dynamic_object_bridge(obj_raw);
    if (!obj) return;
    HostGC* gc = get_active_host_gc();
    obj->set_property(symbol_id, HostValue(val_raw), ShapeRegistry::global(), gc);
}

uint64_t brass_dynamic_object_get_elem(uint64_t obj_raw, int64_t index) {
    auto* obj = unpack_dynamic_object_bridge(obj_raw);
    if (!obj) return HostValue::undefined_val().raw();
    return obj->get_element(index).raw();
}

void brass_dynamic_object_set_elem(uint64_t obj_raw, int64_t index, uint64_t val_raw) {
    auto* obj = unpack_dynamic_object_bridge(obj_raw);
    if (!obj) return;
    HostGC* gc = get_active_host_gc();
    obj->set_element(index, HostValue(val_raw), gc);
}

} // extern "C"
