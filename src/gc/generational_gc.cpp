#include <brass/gc/generational_gc.hpp>
#include <algorithm>
#include <cstring>
#include <cassert>

namespace brass {

GenerationalGC::GenerationalGC(
    size_t nursery_size,
    size_t survivor_size,
    size_t tenured_size,
    uint8_t tenuring_threshold
)   : nursery_size_((nursery_size + 7) & ~static_cast<size_t>(7)),
      survivor_size_((survivor_size + 7) & ~static_cast<size_t>(7)),
      tenured_size_((tenured_size + 7) & ~static_cast<size_t>(7)),
      tenuring_threshold_(tenuring_threshold) {
    init_heap();
}

void GenerationalGC::init_heap() {
    size_t total_size = nursery_size_ + (survivor_size_ * 2) + tenured_size_;
    heap_data_.assign(total_size, 0);
    heap_base_ = reinterpret_cast<uintptr_t>(heap_data_.data());

    nursery_offset_ = 0;
    survivor_a_offset_ = nursery_offset_ + nursery_size_;
    survivor_b_offset_ = survivor_a_offset_ + survivor_size_;
    tenured_offset_ = survivor_b_offset_ + survivor_size_;

    survivor_from_offset_ = survivor_a_offset_;
    survivor_to_offset_ = survivor_b_offset_;

    nursery_free_ = 0;
    survivor_from_free_ = 0;
    survivor_to_free_ = 0;
    tenured_free_ = 0;

    tenured_objects_.clear();
    card_table_.init(heap_base_, total_size);
    card_table_.clean_all();
}

bool GenerationalGC::can_allocate_fast(size_t size) const noexcept {
    size_t aligned_size = (size + 7) & ~static_cast<size_t>(7);
    size_t total_size = sizeof(GenGcHeader) + aligned_size;
    return !stress_mode_ && (total_size <= (nursery_size_ / 2)) &&
           (nursery_free_ + total_size <= nursery_size_);
}

bool GenerationalGC::is_in_nursery(uintptr_t addr) const noexcept {
    uintptr_t start = heap_base_ + nursery_offset_;
    return (addr >= start + sizeof(GenGcHeader)) && (addr < start + nursery_free_);
}

bool GenerationalGC::is_in_survivor(uintptr_t addr) const noexcept {
    uintptr_t sf_start = heap_base_ + survivor_from_offset_;
    uintptr_t st_start = heap_base_ + survivor_to_offset_;
    bool in_from = (addr >= sf_start + sizeof(GenGcHeader)) && (addr < sf_start + survivor_from_free_);
    bool in_to = (addr >= st_start + sizeof(GenGcHeader)) && (addr < st_start + survivor_to_free_);
    return in_from || in_to;
}

bool GenerationalGC::is_young(uintptr_t addr) const noexcept {
    return is_in_nursery(addr) || is_in_survivor(addr);
}

bool GenerationalGC::is_in_tenured(uintptr_t addr) const noexcept {
    uintptr_t start = heap_base_ + tenured_offset_;
    return (addr >= start + sizeof(GenGcHeader)) && (addr < start + tenured_free_);
}

bool GenerationalGC::is_old(uintptr_t addr) const noexcept {
    return is_in_tenured(addr);
}

bool GenerationalGC::is_valid_object(uintptr_t addr) const noexcept {
    return is_young(addr) || is_old(addr);
}

GenGcHeader* GenerationalGC::get_header(uintptr_t obj_addr) noexcept {
    if (!is_valid_object(obj_addr)) return nullptr;
    return reinterpret_cast<GenGcHeader*>(obj_addr - sizeof(GenGcHeader));
}

const GenGcHeader* GenerationalGC::get_header(uintptr_t obj_addr) const noexcept {
    if (!is_valid_object(obj_addr)) return nullptr;
    return reinterpret_cast<const GenGcHeader*>(obj_addr - sizeof(GenGcHeader));
}

uintptr_t GenerationalGC::allocate_nursery(size_t aligned_size, uint64_t pointer_mask, uint32_t type_tag) {
    size_t total_size = sizeof(GenGcHeader) + aligned_size;
    assert(nursery_free_ + total_size <= nursery_size_);

    uint8_t* pos = heap_data_.data() + nursery_offset_ + nursery_free_;
    nursery_free_ += total_size;

    auto* hdr = reinterpret_cast<GenGcHeader*>(pos);
    hdr->size = static_cast<uint32_t>(aligned_size);
    hdr->type_tag = type_tag;
    hdr->pointer_mask = pointer_mask;
    hdr->forwarding_address = 0;
    hdr->age = 0;
    hdr->generation = GEN_YOUNG;
    hdr->reserved16 = 0;
    hdr->reserved32 = 0;

    uintptr_t payload = reinterpret_cast<uintptr_t>(pos + sizeof(GenGcHeader));
    std::memset(reinterpret_cast<void*>(payload), 0, aligned_size);

    total_allocations_++;
    total_allocated_bytes_ += total_size;
    return payload;
}

uintptr_t GenerationalGC::allocate_tenured(size_t aligned_size, uint64_t pointer_mask, uint32_t type_tag, uint8_t age) {
    size_t total_size = sizeof(GenGcHeader) + aligned_size;
    if (tenured_free_ + total_size > tenured_size_) {
        return 0; // Out of tenured space
    }

    uint8_t* pos = heap_data_.data() + tenured_offset_ + tenured_free_;
    tenured_free_ += total_size;

    auto* hdr = reinterpret_cast<GenGcHeader*>(pos);
    hdr->size = static_cast<uint32_t>(aligned_size);
    hdr->type_tag = type_tag;
    hdr->pointer_mask = pointer_mask;
    hdr->forwarding_address = 0;
    hdr->age = age;
    hdr->generation = GEN_OLD;
    hdr->reserved16 = 0;
    hdr->reserved32 = 0;

    uintptr_t payload = reinterpret_cast<uintptr_t>(pos + sizeof(GenGcHeader));
    std::memset(reinterpret_cast<void*>(payload), 0, aligned_size);

    tenured_objects_.push_back(payload);
    total_allocations_++;
    total_allocated_bytes_ += total_size;
    return payload;
}

uintptr_t GenerationalGC::allocate_survivor(size_t aligned_size, uint64_t pointer_mask, uint32_t type_tag, uint8_t age) {
    size_t total_size = sizeof(GenGcHeader) + aligned_size;
    if (survivor_to_free_ + total_size > survivor_size_) {
        return 0; // Survivor overflow, will promote to tenured
    }

    uint8_t* pos = heap_data_.data() + survivor_to_offset_ + survivor_to_free_;
    survivor_to_free_ += total_size;

    auto* hdr = reinterpret_cast<GenGcHeader*>(pos);
    hdr->size = static_cast<uint32_t>(aligned_size);
    hdr->type_tag = type_tag;
    hdr->pointer_mask = pointer_mask;
    hdr->forwarding_address = 0;
    hdr->age = age;
    hdr->generation = GEN_YOUNG;
    hdr->reserved16 = 0;
    hdr->reserved32 = 0;

    uintptr_t payload = reinterpret_cast<uintptr_t>(pos + sizeof(GenGcHeader));
    std::memset(reinterpret_cast<void*>(payload), 0, aligned_size);

    return payload;
}

uintptr_t GenerationalGC::allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    std::vector<uintptr_t*> dummy_roots;
    return allocate(size, pointer_mask, type_tag, dummy_roots);
}

uintptr_t GenerationalGC::allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag, std::vector<uintptr_t*>& extra_roots) {
    size_t aligned_size = (size + 7) & ~static_cast<size_t>(7);
    size_t total_size = sizeof(GenGcHeader) + aligned_size;

    // Large objects directly in tenured
    if (total_size > (nursery_size_ / 2)) {
        if (tenured_free_ + total_size > tenured_size_) {
            major_collect(extra_roots);
        }
        uintptr_t p = allocate_tenured(aligned_size, pointer_mask, type_tag, tenuring_threshold_);
        if (p == 0) {
            major_collect(extra_roots);
            p = allocate_tenured(aligned_size, pointer_mask, type_tag, tenuring_threshold_);
            if (p == 0) {
                throw std::runtime_error("GenerationalGC: Out of tenured memory for large allocation");
            }
        }
        return p;
    }

    if (stress_mode_) {
        minor_collect(extra_roots);
    }

    if (nursery_free_ + total_size <= nursery_size_) {
        return allocate_nursery(aligned_size, pointer_mask, type_tag);
    }

    // Nursery full -> Minor GC
    minor_collect(extra_roots);

    if (nursery_free_ + total_size <= nursery_size_) {
        return allocate_nursery(aligned_size, pointer_mask, type_tag);
    }

    // Still full -> Major GC
    major_collect(extra_roots);

    if (nursery_free_ + total_size <= nursery_size_) {
        return allocate_nursery(aligned_size, pointer_mask, type_tag);
    }

    throw std::runtime_error("GenerationalGC: Out of memory after Major GC");
}

void GenerationalGC::write_barrier(uintptr_t obj_addr, uintptr_t val) noexcept {
    if (is_old(obj_addr) && is_young(val)) {
        card_table_.mark_card(obj_addr);
    }
}

void GenerationalGC::poison_range(uint8_t* start, size_t size) noexcept {
    if (!start || size == 0) return;
    uint64_t* p = reinterpret_cast<uint64_t*>(start);
    size_t count = size / sizeof(uint64_t);
    for (size_t i = 0; i < count; ++i) {
        p[i] = POISON_PATTERN;
    }
}

uintptr_t GenerationalGC::evacuate_young_object(uintptr_t obj_addr) {
    if (!is_young(obj_addr)) return obj_addr;

    auto* old_hdr = reinterpret_cast<GenGcHeader*>(obj_addr - sizeof(GenGcHeader));
    if (old_hdr->forwarding_address != 0) {
        return old_hdr->forwarding_address;
    }

    uint8_t next_age = static_cast<uint8_t>(old_hdr->age + 1);
    uintptr_t new_payload = 0;

    if (next_age >= tenuring_threshold_) {
        // Promote to tenured
        new_payload = allocate_tenured(old_hdr->size, old_hdr->pointer_mask, old_hdr->type_tag, next_age);
        if (new_payload == 0) {
            // Tenured full, fallback to survivor
            new_payload = allocate_survivor(old_hdr->size, old_hdr->pointer_mask, old_hdr->type_tag, next_age);
        } else {
            promoted_bytes_ += sizeof(GenGcHeader) + old_hdr->size;
        }
    } else {
        // Evacuate to survivor
        new_payload = allocate_survivor(old_hdr->size, old_hdr->pointer_mask, old_hdr->type_tag, next_age);
        if (new_payload == 0) {
            // Survivor full, promote to tenured
            new_payload = allocate_tenured(old_hdr->size, old_hdr->pointer_mask, old_hdr->type_tag, next_age);
            if (new_payload != 0) {
                promoted_bytes_ += sizeof(GenGcHeader) + old_hdr->size;
            }
        }
    }

    if (new_payload == 0) {
        throw std::runtime_error("GenerationalGC: Failed to evacuate object during scavenge (both survivor and tenured full)");
    }

    // Copy payload
    std::memcpy(reinterpret_cast<void*>(new_payload), reinterpret_cast<const void*>(obj_addr), old_hdr->size);

    // Forwarding
    old_hdr->forwarding_address = new_payload;
    return new_payload;
}

void GenerationalGC::gather_all_roots(std::vector<uintptr_t*>& roots, std::vector<uintptr_t*>& extra_roots) {
    roots.insert(roots.end(), registered_roots_.begin(), registered_roots_.end());
    roots.insert(roots.end(), extra_roots.begin(), extra_roots.end());
    if (root_provider_) {
        root_provider_(roots);
    }
}

void GenerationalGC::minor_collect() {
    std::vector<uintptr_t*> dummy_roots;
    minor_collect(dummy_roots);
}

void GenerationalGC::minor_collect(std::vector<uintptr_t*>& extra_roots) {
    std::vector<uintptr_t*> all_roots;
    gather_all_roots(all_roots, extra_roots);

    // Track starting tenured offset for promoted objects scan
    size_t tenured_scan_offset = tenured_free_;

    // 1. Scan dirty cards in Tenured space to identify roots into Young generation
    size_t tenured_start_card = card_table_.card_index(heap_base_ + tenured_offset_);
    size_t tenured_end_card = card_table_.card_index(heap_base_ + tenured_offset_ + tenured_size_);

    std::vector<uintptr_t*> card_roots;
    std::vector<size_t> dirty_cards;

    card_table_.for_each_dirty_card_in_range(tenured_start_card, tenured_end_card, [&](size_t card_idx) {
        dirty_cards.push_back(card_idx);
        uintptr_t card_start = card_table_.card_address(card_idx);
        uintptr_t card_end = card_start + CardTable::CARD_SIZE;

        // Binary search for tenured objects overlapping this card
        auto it = std::lower_bound(tenured_objects_.begin(), tenured_objects_.end(), card_start);
        if (it != tenured_objects_.begin()) {
            auto prev_it = it - 1;
            const auto* prev_hdr = get_header(*prev_it);
            if (prev_hdr && (*prev_it + prev_hdr->size) > card_start) {
                // Preceding object overlaps card
                size_t num_fields = prev_hdr->size / 8;
                for (size_t f = 0; f < num_fields; ++f) {
                    if (prev_hdr->pointer_mask & (1ULL << f)) {
                        auto* slot = reinterpret_cast<uintptr_t*>(*prev_it + f * 8);
                        if (*slot != 0 && is_young(*slot)) {
                            card_roots.push_back(slot);
                        }
                    }
                }
            }
        }

        for (; it != tenured_objects_.end() && *it < card_end; ++it) {
            const auto* hdr = get_header(*it);
            if (!hdr) continue;
            size_t num_fields = hdr->size / 8;
            for (size_t f = 0; f < num_fields; ++f) {
                if (hdr->pointer_mask & (1ULL << f)) {
                    auto* slot = reinterpret_cast<uintptr_t*>(*it + f * 8);
                    if (*slot != 0 && is_young(*slot)) {
                        card_roots.push_back(slot);
                    }
                }
            }
        }
    });

    // 2. Evacuate objects referenced by stack/thread roots
    for (uintptr_t* root_slot : all_roots) {
        if (root_slot && *root_slot != 0 && is_young(*root_slot)) {
            *root_slot = evacuate_young_object(*root_slot);
        }
    }

    // 3. Evacuate objects referenced by dirty card roots
    for (uintptr_t* card_root : card_roots) {
        if (card_root && *card_root != 0 && is_young(*card_root)) {
            *card_root = evacuate_young_object(*card_root);
        }
    }

    // 4. Cheney breadth-first scan of survivor_to and promoted tenured objects
    size_t survivor_scan = 0;
    size_t tenured_scan = tenured_scan_offset;

    while (survivor_scan < survivor_to_free_ || tenured_scan < tenured_free_) {
        if (survivor_scan < survivor_to_free_) {
            uint8_t* obj_pos = heap_data_.data() + survivor_to_offset_ + survivor_scan;
            auto* hdr = reinterpret_cast<GenGcHeader*>(obj_pos);
            uintptr_t payload = reinterpret_cast<uintptr_t>(obj_pos + sizeof(GenGcHeader));
            survivor_scan += sizeof(GenGcHeader) + hdr->size;

            size_t num_fields = hdr->size / 8;
            for (size_t f = 0; f < num_fields; ++f) {
                if (hdr->pointer_mask & (1ULL << f)) {
                    auto* slot = reinterpret_cast<uintptr_t*>(payload + f * 8);
                    if (*slot != 0 && is_young(*slot)) {
                        *slot = evacuate_young_object(*slot);
                    }
                }
            }
        } else if (tenured_scan < tenured_free_) {
            uint8_t* obj_pos = heap_data_.data() + tenured_offset_ + tenured_scan;
            auto* hdr = reinterpret_cast<GenGcHeader*>(obj_pos);
            uintptr_t payload = reinterpret_cast<uintptr_t>(obj_pos + sizeof(GenGcHeader));
            tenured_scan += sizeof(GenGcHeader) + hdr->size;

            size_t num_fields = hdr->size / 8;
            bool points_to_young = false;
            for (size_t f = 0; f < num_fields; ++f) {
                if (hdr->pointer_mask & (1ULL << f)) {
                    auto* slot = reinterpret_cast<uintptr_t*>(payload + f * 8);
                    if (*slot != 0 && is_young(*slot)) {
                        *slot = evacuate_young_object(*slot);
                        if (is_young(*slot)) {
                            points_to_young = true;
                        }
                    }
                }
            }
            if (points_to_young) {
                card_table_.mark_card(payload);
            }
        }
    }

    // 5. Clean dirty cards that no longer contain young pointers
    for (size_t card_idx : dirty_cards) {
        uintptr_t card_start = card_table_.card_address(card_idx);
        uintptr_t card_end = card_start + CardTable::CARD_SIZE;
        bool still_dirty = false;

        auto it = std::lower_bound(tenured_objects_.begin(), tenured_objects_.end(), card_start);
        if (it != tenured_objects_.begin()) {
            auto prev_it = it - 1;
            const auto* prev_hdr = get_header(*prev_it);
            if (prev_hdr && (*prev_it + prev_hdr->size) > card_start) {
                size_t num_fields = prev_hdr->size / 8;
                for (size_t f = 0; f < num_fields; ++f) {
                    if (prev_hdr->pointer_mask & (1ULL << f)) {
                        uintptr_t child = *reinterpret_cast<const uintptr_t*>(*prev_it + f * 8);
                        if (child != 0 && is_young(child)) {
                            still_dirty = true;
                            break;
                        }
                    }
                }
            }
        }

        if (!still_dirty) {
            for (; it != tenured_objects_.end() && *it < card_end; ++it) {
                const auto* hdr = get_header(*it);
                if (!hdr) continue;
                size_t num_fields = hdr->size / 8;
                for (size_t f = 0; f < num_fields; ++f) {
                    if (hdr->pointer_mask & (1ULL << f)) {
                        uintptr_t child = *reinterpret_cast<const uintptr_t*>(*it + f * 8);
                        if (child != 0 && is_young(child)) {
                            still_dirty = true;
                            break;
                        }
                    }
                }
                if (still_dirty) break;
            }
        }

        if (!still_dirty) {
            card_table_.clean_card(card_idx);
        }
    }

    // 6. Reset Nursery
    poison_range(heap_data_.data() + nursery_offset_, nursery_free_);
    nursery_free_ = 0;

    // 7. Swap Survivors
    poison_range(heap_data_.data() + survivor_from_offset_, survivor_from_free_);
    std::swap(survivor_from_offset_, survivor_to_offset_);
    survivor_from_free_ = survivor_to_free_;
    survivor_to_free_ = 0;

    minor_collection_count_++;
}

void GenerationalGC::major_collect() {
    std::vector<uintptr_t*> dummy_roots;
    major_collect(dummy_roots);
}

void GenerationalGC::major_collect(std::vector<uintptr_t*>& extra_roots) {
    std::vector<uintptr_t*> all_roots;
    gather_all_roots(all_roots, extra_roots);

    // Major GC evacuates all reachable objects across both generations into a fresh tenured buffer
    std::vector<uint8_t> new_tenured(tenured_size_, 0);
    size_t new_tenured_free = 0;
    std::vector<uintptr_t> new_tenured_objects;

    auto evacuate_to_tenured = [&](uintptr_t obj_addr) -> uintptr_t {
        if (obj_addr == 0 || !is_valid_object(obj_addr)) return obj_addr;

        auto* old_hdr = reinterpret_cast<GenGcHeader*>(obj_addr - sizeof(GenGcHeader));
        if (old_hdr->forwarding_address != 0) {
            return old_hdr->forwarding_address;
        }

        size_t total_sz = sizeof(GenGcHeader) + old_hdr->size;
        if (new_tenured_free + total_sz > tenured_size_) {
            throw std::runtime_error("GenerationalGC: Tenured space exhausted during Major GC");
        }

        uint8_t* dest = new_tenured.data() + new_tenured_free;
        new_tenured_free += total_sz;

        auto* new_hdr = reinterpret_cast<GenGcHeader*>(dest);
        new_hdr->size = old_hdr->size;
        new_hdr->type_tag = old_hdr->type_tag;
        new_hdr->pointer_mask = old_hdr->pointer_mask;
        new_hdr->forwarding_address = 0;
        new_hdr->age = tenuring_threshold_;
        new_hdr->generation = GEN_OLD;
        new_hdr->reserved16 = 0;
        new_hdr->reserved32 = 0;

        uintptr_t final_payload = heap_base_ + tenured_offset_ + (dest - new_tenured.data()) + sizeof(GenGcHeader);
        std::memcpy(dest + sizeof(GenGcHeader), reinterpret_cast<const void*>(obj_addr), old_hdr->size);

        old_hdr->forwarding_address = final_payload;
        new_tenured_objects.push_back(final_payload);
        return final_payload;
    };

    // 1. Evacuate from all roots
    for (uintptr_t* root_slot : all_roots) {
        if (root_slot && *root_slot != 0) {
            *root_slot = evacuate_to_tenured(*root_slot);
        }
    }

    // 2. Cheney scan within new_tenured
    size_t scan_offset = 0;
    while (scan_offset < new_tenured_free) {
        uint8_t* obj_pos = new_tenured.data() + scan_offset;
        auto* hdr = reinterpret_cast<GenGcHeader*>(obj_pos);
        scan_offset += sizeof(GenGcHeader) + hdr->size;

        size_t num_fields = hdr->size / 8;
        for (size_t f = 0; f < num_fields; ++f) {
            if (hdr->pointer_mask & (1ULL << f)) {
                auto* slot = reinterpret_cast<uintptr_t*>(obj_pos + sizeof(GenGcHeader) + f * 8);
                if (*slot != 0 && is_valid_object(*slot)) {
                    *slot = evacuate_to_tenured(*slot);
                }
            }
        }
    }

    // 3. Copy new_tenured into tenured space
    std::memcpy(heap_data_.data() + tenured_offset_, new_tenured.data(), new_tenured_free);
    tenured_free_ = new_tenured_free;
    tenured_objects_ = std::move(new_tenured_objects);

    // 4. Reset Nursery and Survivors completely
    if (stress_mode_) {
        poison_range(heap_data_.data() + nursery_offset_, nursery_size_);
        poison_range(heap_data_.data() + survivor_from_offset_, survivor_size_);
        poison_range(heap_data_.data() + survivor_to_offset_, survivor_size_);
    }
    nursery_free_ = 0;
    survivor_from_free_ = 0;
    survivor_to_free_ = 0;

    // 5. Clean all cards
    card_table_.clean_all();

    major_collection_count_++;
}

void GenerationalGC::collect() {
    std::vector<uintptr_t*> dummy;
    collect(dummy);
}

void GenerationalGC::collect(std::vector<uintptr_t*>& extra_roots) {
    if (static_cast<double>(tenured_free_) / static_cast<double>(tenured_size_) >= DEFAULT_TENURED_OCCUPANCY_THRESHOLD) {
        major_collect(extra_roots);
    } else {
        minor_collect(extra_roots);
    }
}

void GenerationalGC::register_root(uintptr_t* root_slot) {
    if (root_slot) {
        registered_roots_.push_back(root_slot);
    }
}

void GenerationalGC::unregister_root(uintptr_t* root_slot) {
    auto it = std::remove(registered_roots_.begin(), registered_roots_.end(), root_slot);
    registered_roots_.erase(it, registered_roots_.end());
}

void GenerationalGC::set_root_provider(RootProvider provider) {
    root_provider_ = std::move(provider);
}

uint64_t GenerationalGC::read_field(uintptr_t obj_addr, size_t field_idx) const {
    const auto* hdr = get_header(obj_addr);
    if (!hdr) throw std::runtime_error("GenerationalGC::read_field on invalid object");
    if ((field_idx + 1) * 8 > hdr->size) throw std::out_of_range("Field index out of range");
    return *reinterpret_cast<const uint64_t*>(obj_addr + field_idx * 8);
}

void GenerationalGC::write_field(uintptr_t obj_addr, size_t field_idx, uint64_t val) {
    auto* hdr = get_header(obj_addr);
    if (!hdr) throw std::runtime_error("GenerationalGC::write_field on invalid object");
    if ((field_idx + 1) * 8 > hdr->size) throw std::out_of_range("Field index out of range");
    *reinterpret_cast<uint64_t*>(obj_addr + field_idx * 8) = val;
    write_barrier(obj_addr, static_cast<uintptr_t>(val));
}

void GenerationalGC::write_field(uintptr_t obj_addr, size_t field_idx, RuntimeValue val) {
    write_field(obj_addr, field_idx, val.raw_bits());
}

RuntimeValue GenerationalGC::read_memory(uintptr_t base, int32_t offset, Type t) const {
    uintptr_t target = base + static_cast<uintptr_t>(offset);
    if (t.is_v128()) {
        uint8_t bytes[16];
        std::memcpy(bytes, reinterpret_cast<const void*>(target), 16);
        return RuntimeValue::from_v128(t, bytes);
    }
    if (t.is_v256()) {
        uint8_t bytes[32];
        std::memcpy(bytes, reinterpret_cast<const void*>(target), 32);
        return RuntimeValue::from_v256(t, bytes);
    }
    if (t.is_float()) {
        double d = *reinterpret_cast<const double*>(target);
        return RuntimeValue::from_f64(d);
    }
    if (t.size_in_bytes() == 4) {
        int32_t v = *reinterpret_cast<const int32_t*>(target);
        return RuntimeValue::from_i32(v);
    }
    uint64_t v = *reinterpret_cast<const uint64_t*>(target);
    if (t.is_gcref()) return RuntimeValue::from_gcref(static_cast<uintptr_t>(v));
    if (t.is_pointer()) return RuntimeValue::from_ptr(reinterpret_cast<void*>(v));
    return RuntimeValue::from_i64(static_cast<int64_t>(v));
}

void GenerationalGC::write_memory(uintptr_t base, int32_t offset, Type t, RuntimeValue val) {
    uintptr_t target = base + static_cast<uintptr_t>(offset);
    if (t.is_v128()) {
        std::memcpy(reinterpret_cast<void*>(target), val.v128_bytes(), 16);
    } else if (t.is_v256()) {
        std::memcpy(reinterpret_cast<void*>(target), val.vec_bytes(), 32);
    } else if (t.is_float()) {
        *reinterpret_cast<double*>(target) = val.as_f64();
    } else if (t.size_in_bytes() == 4) {
        *reinterpret_cast<int32_t*>(target) = val.as_i32();
    } else {
        *reinterpret_cast<uint64_t*>(target) = val.raw_bits();
    }

    if (t.is_pointer_or_gcref() || t.kind() == TypeKind::I64) {
        write_barrier(base, val.raw_bits());
    }
}

void GenerationalGC::reset() {
    init_heap();
    minor_collection_count_ = 0;
    major_collection_count_ = 0;
    total_allocations_ = 0;
    total_allocated_bytes_ = 0;
    promoted_bytes_ = 0;
}

} // namespace brass
