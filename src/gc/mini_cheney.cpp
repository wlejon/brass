#include <brass/gc/mini_cheney.hpp>
#include <algorithm>
#include <cstring>
#include <iostream>

namespace brass {

MiniCheneyGC::MiniCheneyGC(size_t semispace_size)
    : semispace_size_((semispace_size + 7) & ~static_cast<size_t>(7)) {
    if (semispace_size_ < 64 * 1024) {
        semispace_size_ = 64 * 1024;
    }
    from_space_.resize(semispace_size_, 0);
    to_space_.resize(semispace_size_);
    poison_space(to_space_.data(), semispace_size_);
}

void MiniCheneyGC::poison_space(uint8_t* space, size_t size) noexcept {
    uint64_t* p64 = reinterpret_cast<uint64_t*>(space);
    size_t count = size / sizeof(uint64_t);
    for (size_t i = 0; i < count; ++i) {
        p64[i] = POISON_PATTERN;
    }
}

void MiniCheneyGC::reset() {
    free_ptr_ = 0;
    collection_count_ = 0;
    total_allocations_ = 0;
    total_allocated_bytes_ = 0;
    std::fill(from_space_.begin(), from_space_.end(), uint8_t(0));
    poison_space(to_space_.data(), semispace_size_);
    registered_roots_.clear();
}

void MiniCheneyGC::register_root(uintptr_t* root_slot) {
    if (root_slot) {
        registered_roots_.push_back(root_slot);
    }
}

void MiniCheneyGC::unregister_root(uintptr_t* root_slot) {
    auto it = std::find(registered_roots_.begin(), registered_roots_.end(), root_slot);
    if (it != registered_roots_.end()) {
        registered_roots_.erase(it);
    }
}

void MiniCheneyGC::set_root_provider(RootProvider provider) {
    root_provider_ = std::move(provider);
}

void MiniCheneyGC::gather_all_roots(std::vector<uintptr_t*>& roots) {
    for (uintptr_t* r : registered_roots_) {
        roots.push_back(r);
    }
    if (root_provider_) {
        root_provider_(roots);
    }
}

uintptr_t MiniCheneyGC::allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    std::vector<uintptr_t*> roots;
    return allocate(size, pointer_mask, type_tag, roots);
}

uintptr_t MiniCheneyGC::allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag, std::vector<uintptr_t*>& extra_roots) {
    size_t aligned_size = (size + 7) & ~static_cast<size_t>(7);
    size_t total_size = sizeof(GcHeader) + aligned_size;

    if (total_size > semispace_size_) {
        throw std::bad_alloc();
    }

    if (stress_mode_ || free_ptr_ + total_size > semispace_size_) {
        std::vector<uintptr_t*> all_roots;
        gather_all_roots(all_roots);
        for (uintptr_t* r : extra_roots) {
            all_roots.push_back(r);
        }
        collect(all_roots);
    }

    if (free_ptr_ + total_size > semispace_size_) {
        throw std::bad_alloc();
    }

    uint8_t* alloc_pos = from_space_.data() + free_ptr_;
    GcHeader* hdr = reinterpret_cast<GcHeader*>(alloc_pos);
    hdr->size = static_cast<uint32_t>(aligned_size);
    hdr->type_tag = type_tag;
    hdr->pointer_mask = pointer_mask;
    hdr->forwarding_address = 0;

    uint8_t* payload_pos = alloc_pos + sizeof(GcHeader);
    std::memset(payload_pos, 0, aligned_size);

    free_ptr_ += total_size;
    total_allocations_++;
    total_allocated_bytes_ += total_size;

    return reinterpret_cast<uintptr_t>(payload_pos);
}

void MiniCheneyGC::collect() {
    std::vector<uintptr_t*> roots;
    gather_all_roots(roots);
    collect(roots);
}

void MiniCheneyGC::collect(std::vector<uintptr_t*>& roots) {
    size_t scan_ptr = 0;
    size_t to_free_ptr = 0;

    // 1. Evacuate roots
    for (uintptr_t* root_slot : roots) {
        if (root_slot != nullptr && *root_slot != 0) {
            *root_slot = evacuate_object(*root_slot, to_free_ptr);
        }
    }

    // 2. Scan evacuated objects in To-Space (Cheney scan loop)
    while (scan_ptr < to_free_ptr) {
        GcHeader* hdr = reinterpret_cast<GcHeader*>(to_space_.data() + scan_ptr);
        uintptr_t payload_addr = reinterpret_cast<uintptr_t>(to_space_.data() + scan_ptr + sizeof(GcHeader));

        size_t num_fields = hdr->size / 8;
        for (size_t i = 0; i < num_fields && i < 64; ++i) {
            if ((hdr->pointer_mask & (1ULL << i)) != 0) {
                uintptr_t* field_ptr = reinterpret_cast<uintptr_t*>(payload_addr + i * 8);
                if (*field_ptr != 0) {
                    *field_ptr = evacuate_object(*field_ptr, to_free_ptr);
                }
            }
        }

        scan_ptr += sizeof(GcHeader) + hdr->size;
    }

    // 3. Swap spaces: To-Space becomes the new From-Space
    std::swap(from_space_, to_space_);
    free_ptr_ = to_free_ptr;

    // 4. Poison the old space
    poison_space(to_space_.data(), semispace_size_);

    // 5. Update collection stats
    collection_count_++;
}

uintptr_t MiniCheneyGC::evacuate_object(uintptr_t obj_addr, size_t& to_free_ptr) {
    if (obj_addr == 0) {
        return 0;
    }

    uintptr_t from_start = reinterpret_cast<uintptr_t>(from_space_.data());
    uintptr_t from_end = from_start + semispace_size_;

    if (obj_addr < from_start + sizeof(GcHeader) || obj_addr >= from_end) {
        return obj_addr;
    }

    GcHeader* old_hdr = reinterpret_cast<GcHeader*>(obj_addr - sizeof(GcHeader));

    // Check if already forwarded
    if (old_hdr->forwarding_address != 0) {
        return old_hdr->forwarding_address;
    }

    size_t total_size = sizeof(GcHeader) + old_hdr->size;
    if (to_free_ptr + total_size > semispace_size_) {
        throw std::bad_alloc();
    }

    uint8_t* dest = to_space_.data() + to_free_ptr;
    std::memcpy(dest, old_hdr, total_size);

    GcHeader* new_hdr = reinterpret_cast<GcHeader*>(dest);
    new_hdr->forwarding_address = 0;

    uintptr_t new_payload_addr = reinterpret_cast<uintptr_t>(dest + sizeof(GcHeader));
    old_hdr->forwarding_address = new_payload_addr;

    to_free_ptr += total_size;
    return new_payload_addr;
}

bool MiniCheneyGC::is_address_in_active_space(uintptr_t addr) const noexcept {
    uintptr_t start = reinterpret_cast<uintptr_t>(from_space_.data());
    return addr >= start && addr < start + free_ptr_;
}

bool MiniCheneyGC::is_valid_object(uintptr_t obj_addr) const noexcept {
    if (obj_addr == 0 || (obj_addr % 8) != 0) {
        return false;
    }

    uintptr_t active_start = reinterpret_cast<uintptr_t>(from_space_.data());
    uintptr_t active_end = active_start + free_ptr_;

    if (obj_addr < active_start + sizeof(GcHeader) || obj_addr >= active_end) {
        return false;
    }

    const GcHeader* hdr = reinterpret_cast<const GcHeader*>(obj_addr - sizeof(GcHeader));
    if (hdr->forwarding_address != 0) {
        return false;
    }

    if (hdr->size > semispace_size_ || (hdr->size % 8) != 0) {
        return false;
    }

    const uint64_t* raw = reinterpret_cast<const uint64_t*>(hdr);
    if (raw[0] == POISON_PATTERN || raw[1] == POISON_PATTERN || raw[2] == POISON_PATTERN) {
        return false;
    }

    return true;
}

GcHeader* MiniCheneyGC::get_header(uintptr_t obj_addr) noexcept {
    if (!is_valid_object(obj_addr)) {
        return nullptr;
    }
    return reinterpret_cast<GcHeader*>(obj_addr - sizeof(GcHeader));
}

const GcHeader* MiniCheneyGC::get_header(uintptr_t obj_addr) const noexcept {
    if (!is_valid_object(obj_addr)) {
        return nullptr;
    }
    return reinterpret_cast<const GcHeader*>(obj_addr - sizeof(GcHeader));
}

uint64_t MiniCheneyGC::read_field(uintptr_t obj_addr, size_t field_idx) const {
    if (!is_valid_object(obj_addr)) {
        throw std::runtime_error("GC Error: Invalid object pointer passed to read_field");
    }
    const GcHeader* hdr = get_header(obj_addr);
    if ((field_idx + 1) * 8 > hdr->size) {
        throw std::runtime_error("GC Error: Field index out of bounds in read_field");
    }
    return *reinterpret_cast<const uint64_t*>(obj_addr + field_idx * 8);
}

void MiniCheneyGC::write_field(uintptr_t obj_addr, size_t field_idx, uint64_t val) {
    if (!is_valid_object(obj_addr)) {
        throw std::runtime_error("GC Error: Invalid object pointer passed to write_field");
    }
    GcHeader* hdr = get_header(obj_addr);
    if ((field_idx + 1) * 8 > hdr->size) {
        throw std::runtime_error("GC Error: Field index out of bounds in write_field");
    }
    *reinterpret_cast<uint64_t*>(obj_addr + field_idx * 8) = val;
}

void MiniCheneyGC::write_field(uintptr_t obj_addr, size_t field_idx, RuntimeValue val) {
    write_field(obj_addr, field_idx, val.raw_bits());
}

RuntimeValue MiniCheneyGC::read_memory(uintptr_t base, int32_t offset, Type t) const {
    if (base == 0) {
        throw std::runtime_error("Memory Error: Null pointer dereference in read_memory");
    }

    uintptr_t effective_addr = static_cast<uintptr_t>(static_cast<int64_t>(base) + offset);
    size_t access_size = t.size_in_bytes();

    if (is_address_in_active_space(base)) {
        if (!is_valid_object(base)) {
            throw std::runtime_error("Memory Error: Invalid GC object access in read_memory");
        }
        const GcHeader* hdr = get_header(base);
        if (offset < 0 || static_cast<size_t>(offset) + access_size > hdr->size) {
            throw std::runtime_error("Memory Error: GC object access out of bounds");
        }
    }

    if (access_size == 4) {
        uint32_t val32 = 0;
        std::memcpy(&val32, reinterpret_cast<const void*>(effective_addr), 4);
        return RuntimeValue::from_bits(t, static_cast<uint64_t>(val32));
    } else if (access_size == 8) {
        uint64_t raw_val = 0;
        std::memcpy(&raw_val, reinterpret_cast<const void*>(effective_addr), 8);
        return RuntimeValue::from_bits(t, raw_val);
    } else if (access_size == 16) {
        uint8_t bytes[16];
        std::memcpy(bytes, reinterpret_cast<const void*>(effective_addr), 16);
        return RuntimeValue::from_v128(t, bytes);
    } else {
        throw std::runtime_error("Memory Error: Unsupported access size in read_memory");
    }
}

void MiniCheneyGC::write_memory(uintptr_t base, int32_t offset, Type t, RuntimeValue val) {
    if (base == 0) {
        throw std::runtime_error("Memory Error: Null pointer dereference in write_memory");
    }

    uintptr_t effective_addr = static_cast<uintptr_t>(static_cast<int64_t>(base) + offset);
    size_t access_size = t.size_in_bytes();

    if (is_address_in_active_space(base)) {
        if (!is_valid_object(base)) {
            throw std::runtime_error("Memory Error: Invalid GC object access in write_memory");
        }
        const GcHeader* hdr = get_header(base);
        if (offset < 0 || static_cast<size_t>(offset) + access_size > hdr->size) {
            throw std::runtime_error("Memory Error: GC object access out of bounds");
        }
    }

    if (access_size == 4) {
        uint32_t val32 = static_cast<uint32_t>(val.raw_bits() & 0xFFFFFFFFULL);
        std::memcpy(reinterpret_cast<void*>(effective_addr), &val32, 4);
    } else if (access_size == 8) {
        uint64_t raw_val = val.raw_bits();
        std::memcpy(reinterpret_cast<void*>(effective_addr), &raw_val, 8);
    } else if (access_size == 16) {
        std::memcpy(reinterpret_cast<void*>(effective_addr), val.v128_bytes(), 16);
    } else {
        throw std::runtime_error("Memory Error: Unsupported access size in write_memory");
    }
}

} // namespace brass
