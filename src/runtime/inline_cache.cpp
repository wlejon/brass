#include <brass/runtime/inline_cache.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <iostream>
#include <iomanip>

namespace brass::runtime {

std::string_view to_string(ICState state) noexcept {
    switch (state) {
        case ICState::Uninitialized: return "Uninitialized";
        case ICState::Monomorphic:   return "Monomorphic";
        case ICState::Polymorphic:   return "Polymorphic";
        case ICState::Megamorphic:   return "Megamorphic";
    }
    return "Unknown";
}

std::ostream& operator<<(std::ostream& os, ICState state) {
    return os << to_string(state);
}

InlineCache::InlineCache(uint32_t site_id, std::string prop_name, uint32_t symbol_id, bool is_load)
    : site_id_(site_id),
      prop_name_(std::move(prop_name)),
      symbol_id_(symbol_id),
      is_load_(is_load) {}

const Shape* InlineCache::cached_shape(size_t idx) const noexcept {
    if (idx < entry_count_) {
        return cached_shapes_[idx];
    }
    return nullptr;
}

uint32_t InlineCache::cached_slot(size_t idx) const noexcept {
    if (idx < entry_count_) {
        return cached_slots_[idx];
    }
    return 0;
}

int32_t InlineCache::probe_get(const Shape* shape) noexcept {
    if (!shape) return -1;
    if (state_ == ICState::Monomorphic) {
        if (shape == cached_shapes_[0]) {
            return static_cast<int32_t>(cached_slots_[0]);
        }
        return -1;
    }
    if (state_ == ICState::Polymorphic) {
        for (size_t i = 0; i < entry_count_; ++i) {
            if (shape == cached_shapes_[i]) {
                return static_cast<int32_t>(cached_slots_[i]);
            }
        }
        return -1;
    }
    return -1;
}

int32_t InlineCache::probe_set(const Shape* shape) noexcept {
    return probe_get(shape);
}

HostValue InlineCache::execute_get(DynamicObject* obj) {
    if (!obj) {
        return HostValue::undefined_val();
    }

    const Shape* s = obj->shape;
    if (s != nullptr) {
        int32_t slot = probe_get(s);
        if (slot >= 0) {
            record_hit();
            return obj->get_slot(static_cast<uint32_t>(slot));
        }
    }

    record_miss();
    return miss_handler_get(obj);
}

HostValue InlineCache::miss_handler_get(DynamicObject* obj) {
    if (!obj || !obj->shape) {
        return HostValue::undefined_val();
    }

    const Shape* s = obj->shape;
    const PropertyDescriptor* prop = nullptr;
    if (!prop_name_.empty()) {
        prop = s->find_property(prop_name_);
    }
    if (!prop && symbol_id_ != 0) {
        prop = s->find_property(symbol_id_);
    }

    if (!prop) {
        return HostValue::undefined_val();
    }

    uint32_t slot = prop->slot_index;

    // Transition state
    if (state_ == ICState::Uninitialized) {
        state_ = ICState::Monomorphic;
        cached_shapes_[0] = s;
        cached_slots_[0] = slot;
        entry_count_ = 1;
        patch_monomorphic_ic(*this, s, slot);
    } else if (state_ == ICState::Monomorphic) {
        if (s != cached_shapes_[0]) {
            state_ = ICState::Polymorphic;
            cached_shapes_[1] = s;
            cached_slots_[1] = slot;
            entry_count_ = 2;
        }
    } else if (state_ == ICState::Polymorphic) {
        bool found = false;
        for (size_t i = 0; i < entry_count_; ++i) {
            if (cached_shapes_[i] == s) {
                found = true;
                break;
            }
        }
        if (!found) {
            if (entry_count_ < POLYMORPHIC_LIMIT) {
                cached_shapes_[entry_count_] = s;
                cached_slots_[entry_count_] = slot;
                entry_count_++;
            } else {
                state_ = ICState::Megamorphic;
            }
        }
    }

    return obj->get_slot(slot);
}

void InlineCache::execute_set(
    DynamicObject* obj,
    HostValue val,
    ShapeRegistry& registry,
    HostGC* gc
) {
    if (!obj) return;

    const Shape* s = obj->shape;
    if (s != nullptr) {
        int32_t slot = probe_set(s);
        if (slot >= 0) {
            record_hit();
            obj->set_slot(static_cast<uint32_t>(slot), val, gc);
            return;
        }
    }

    record_miss();
    miss_handler_set(obj, val, registry, gc);
}

void InlineCache::miss_handler_set(
    DynamicObject* obj,
    HostValue val,
    ShapeRegistry& registry,
    HostGC* gc
) {
    if (!obj) return;

    if (!prop_name_.empty()) {
        obj->set_property(prop_name_, val, registry, gc);
    } else if (symbol_id_ != 0) {
        obj->set_property(symbol_id_, val, registry, gc);
    } else {
        return;
    }

    const Shape* new_shape = obj->shape;
    if (!new_shape) return;

    const PropertyDescriptor* prop = !prop_name_.empty() ? new_shape->find_property(prop_name_) : new_shape->find_property(symbol_id_);
    if (!prop) return;

    uint32_t slot = prop->slot_index;

    // Transition state
    if (state_ == ICState::Uninitialized) {
        state_ = ICState::Monomorphic;
        cached_shapes_[0] = new_shape;
        cached_slots_[0] = slot;
        entry_count_ = 1;
        patch_monomorphic_ic(*this, new_shape, slot);
    } else if (state_ == ICState::Monomorphic) {
        if (new_shape != cached_shapes_[0]) {
            state_ = ICState::Polymorphic;
            cached_shapes_[1] = new_shape;
            cached_slots_[1] = slot;
            entry_count_ = 2;
        }
    } else if (state_ == ICState::Polymorphic) {
        bool found = false;
        for (size_t i = 0; i < entry_count_; ++i) {
            if (cached_shapes_[i] == new_shape) {
                found = true;
                break;
            }
        }
        if (!found) {
            if (entry_count_ < POLYMORPHIC_LIMIT) {
                cached_shapes_[entry_count_] = new_shape;
                cached_slots_[entry_count_] = slot;
                entry_count_++;
            } else {
                state_ = ICState::Megamorphic;
            }
        }
    }
}

void InlineCache::reset() noexcept {
    state_ = ICState::Uninitialized;
    for (size_t i = 0; i < POLYMORPHIC_LIMIT; ++i) {
        cached_shapes_[i] = nullptr;
        cached_slots_[i] = 0;
    }
    entry_count_ = 0;
}

void InlineCache::invalidate() noexcept {
    reset();
}

InlineCache* ICRegistry::get_or_create_ic(
    uint32_t site_id,
    std::string_view prop_name,
    uint32_t symbol_id,
    bool is_load
) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = id_to_ic_.find(site_id);
    if (it != id_to_ic_.end()) {
        if (it->second->prop_name() == prop_name && it->second->is_load() == is_load) {
            return it->second;
        }
    }

    auto ic = std::make_unique<InlineCache>(site_id, std::string(prop_name), symbol_id, is_load);
    InlineCache* ptr = ic.get();
    id_to_ic_[site_id] = ptr;
    caches_.push_back(std::move(ic));
    return ptr;
}

InlineCache* ICRegistry::find_ic(uint32_t site_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = id_to_ic_.find(site_id);
    if (it != id_to_ic_.end()) {
        return it->second;
    }
    return nullptr;
}

size_t ICRegistry::size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return caches_.size();
}

void ICRegistry::reset_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& ic : caches_) {
        ic->reset();
    }
}

void ICRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    caches_.clear();
    id_to_ic_.clear();
}

void ICRegistry::dump_stats(std::ostream& os) const {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t uninit_count = 0;
    size_t mono_count = 0;
    size_t poly_count = 0;
    size_t mega_count = 0;
    uint64_t total_hits = 0;
    uint64_t total_misses = 0;

    for (const auto& ic : caches_) {
        switch (ic->state()) {
            case ICState::Uninitialized: uninit_count++; break;
            case ICState::Monomorphic:   mono_count++; break;
            case ICState::Polymorphic:   poly_count++; break;
            case ICState::Megamorphic:   mega_count++; break;
        }
        total_hits += ic->hit_count();
        total_misses += ic->miss_count();
    }

    uint64_t total_accesses = total_hits + total_misses;
    double hit_ratio = total_accesses > 0 ? (100.0 * static_cast<double>(total_hits) / static_cast<double>(total_accesses)) : 0.0;

    os << "\n=== Polymorphic Inline Cache (PIC) Statistics ===\n";
    os << "Total IC Sites:    " << caches_.size() << "\n";
    os << "  Uninitialized:   " << uninit_count << "\n";
    os << "  Monomorphic:     " << mono_count << "\n";
    os << "  Polymorphic:     " << poly_count << "\n";
    os << "  Megamorphic:     " << mega_count << "\n";
    os << "Total Invocations: " << total_accesses << "\n";
    os << "  Cache Hits:      " << total_hits << " (" << std::fixed << std::setprecision(1) << hit_ratio << "%)\n";
    os << "  Cache Misses:    " << total_misses << "\n";
    os << "\n--- Per-Site Breakdown ---\n";
    os << std::left
       << std::setw(8)  << "Site ID"
       << std::setw(6)  << "Kind"
       << std::setw(16) << "Property"
       << std::setw(16) << "State"
       << std::setw(8)  << "Entries"
       << std::setw(12) << "Hits"
       << std::setw(12) << "Misses"
       << "\n";

    for (const auto& ic : caches_) {
        std::string prop = std::string(ic->prop_name());
        if (prop.empty() && ic->symbol_id() != 0) {
            prop = "#" + std::to_string(ic->symbol_id());
        }
        os << std::left
           << std::setw(8)  << ic->site_id()
           << std::setw(6)  << (ic->is_load() ? "GET" : "SET")
           << std::setw(16) << prop
           << std::setw(16) << to_string(ic->state())
           << std::setw(8)  << ic->entry_count()
           << std::setw(12) << ic->hit_count()
           << std::setw(12) << ic->miss_count()
           << "\n";
    }
    os << "=================================================\n\n";
}

ICRegistry& ICRegistry::global() noexcept {
    static ICRegistry s_global;
    return s_global;
}

} // namespace brass::runtime
