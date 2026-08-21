#include <brass/embedding/host_gc.hpp>
#include <algorithm>
#include <cstring>
#include <stdexcept>

#if defined(_MSC_VER)
#include <intrin.h>
extern "C" uintptr_t brass_get_rbp();
#endif

namespace {

inline void get_caller_frame(uintptr_t& caller_rbp, uintptr_t& caller_ip) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
    void** ret_addr_slot = reinterpret_cast<void**>(_AddressOfReturnAddress());
    caller_ip = reinterpret_cast<uintptr_t>(*ret_addr_slot);
    caller_rbp = brass_get_rbp();
#elif defined(__GNUC__) || defined(__clang__)
    void* cur_frame = __builtin_frame_address(0);
    if (cur_frame) {
        caller_rbp = *reinterpret_cast<uintptr_t*>(cur_frame);
        caller_ip = *reinterpret_cast<uintptr_t*>(reinterpret_cast<uintptr_t>(cur_frame) + 8);
    }
#else
    caller_rbp = 0;
    caller_ip = 0;
#endif
}

} // namespace

namespace brass {

namespace {

static HostGC* g_active_host_gc = nullptr;

} // namespace

void set_active_host_gc(HostGC* gc) noexcept {
    g_active_host_gc = gc;
}

HostGC* get_active_host_gc() noexcept {
    return g_active_host_gc;
}

HostGC::HostGC(size_t semispace_size)
    : semispace_size_(semispace_size),
      from_space_(semispace_size, 0),
      to_space_(semispace_size, 0),
      free_ptr_(0) {
    poison_space(from_space_.data(), semispace_size_);
    poison_space(to_space_.data(), semispace_size_);
}

HostGC::~HostGC() {
    if (g_active_host_gc == this) {
        g_active_host_gc = nullptr;
    }
}

void HostGC::poison_space(uint8_t* space, size_t size) noexcept {
    if (!space || size == 0) return;
    auto* words = reinterpret_cast<uint64_t*>(space);
    size_t num_words = size / sizeof(uint64_t);
    for (size_t i = 0; i < num_words; ++i) {
        words[i] = POISON_PATTERN;
    }
}

bool HostGC::is_address_in_active_space(uintptr_t addr) const noexcept {
    uintptr_t base = reinterpret_cast<uintptr_t>(from_space_.data());
    return (addr >= base) && (addr < base + semispace_size_);
}

bool HostGC::is_valid_object(uintptr_t obj_addr) const noexcept {
    if (!is_address_in_active_space(obj_addr)) {
        return false;
    }
    uintptr_t base = reinterpret_cast<uintptr_t>(from_space_.data());
    if (obj_addr < base + sizeof(HostGcHeader)) {
        return false;
    }
    uintptr_t hdr_addr = obj_addr - sizeof(HostGcHeader);
    if ((hdr_addr % 8) != 0) {
        return false;
    }
    return true;
}

HostGcHeader* HostGC::get_header(uintptr_t obj_addr) noexcept {
    if (!is_valid_object(obj_addr)) return nullptr;
    return reinterpret_cast<HostGcHeader*>(obj_addr - sizeof(HostGcHeader));
}

const HostGcHeader* HostGC::get_header(uintptr_t obj_addr) const noexcept {
    if (!is_valid_object(obj_addr)) return nullptr;
    return reinterpret_cast<const HostGcHeader*>(obj_addr - sizeof(HostGcHeader));
}

uint64_t HostGC::read_field(uintptr_t obj_addr, size_t field_idx) const {
    const auto* hdr = get_header(obj_addr);
    if (!hdr) {
        throw std::runtime_error("HostGC: Invalid object address in read_field");
    }
    if ((field_idx + 1) * 8 > hdr->size) {
        throw std::runtime_error("HostGC: Field index out of bounds in read_field");
    }
    const auto* field_ptr = reinterpret_cast<const uint64_t*>(obj_addr + field_idx * 8);
    return *field_ptr;
}

void HostGC::write_field(uintptr_t obj_addr, size_t field_idx, uint64_t val) {
    auto* hdr = get_header(obj_addr);
    if (!hdr) {
        throw std::runtime_error("HostGC: Invalid object address in write_field");
    }
    if ((field_idx + 1) * 8 > hdr->size) {
        throw std::runtime_error("HostGC: Field index out of bounds in write_field");
    }
    auto* field_ptr = reinterpret_cast<uint64_t*>(obj_addr + field_idx * 8);
    *field_ptr = val;
}

HostValue HostGC::read_value_field(uintptr_t obj_addr, size_t field_idx) const {
    return HostValue::from_raw(read_field(obj_addr, field_idx));
}

void HostGC::write_value_field(uintptr_t obj_addr, size_t field_idx, HostValue val) {
    write_field(obj_addr, field_idx, val.raw());
}

void HostGC::register_root(HostValue* root_slot) {
    if (root_slot) {
        registered_val_roots_.push_back(root_slot);
    }
}

void HostGC::unregister_root(HostValue* root_slot) {
    auto it = std::remove(registered_val_roots_.begin(), registered_val_roots_.end(), root_slot);
    registered_val_roots_.erase(it, registered_val_roots_.end());
}

void HostGC::register_root(uintptr_t* root_slot) {
    if (root_slot) {
        registered_ptr_roots_.push_back(root_slot);
    }
}

void HostGC::unregister_root(uintptr_t* root_slot) {
    auto it = std::remove(registered_ptr_roots_.begin(), registered_ptr_roots_.end(), root_slot);
    registered_ptr_roots_.erase(it, registered_ptr_roots_.end());
}

void HostGC::set_root_provider(RootProvider provider) {
    root_provider_ = std::move(provider);
}

uintptr_t HostGC::evacuate_object(uintptr_t obj_addr, size_t& to_free_ptr) {
    if (!is_address_in_active_space(obj_addr)) {
        return obj_addr;
    }

    auto* hdr = reinterpret_cast<HostGcHeader*>(obj_addr - sizeof(HostGcHeader));
    if (hdr->forwarding_address != 0) {
        return hdr->forwarding_address;
    }

    size_t aligned_size = (hdr->size + 7) & ~static_cast<size_t>(7);
    size_t total_size = sizeof(HostGcHeader) + aligned_size;

    if (to_free_ptr + total_size > semispace_size_) {
        throw std::runtime_error("HostGC: To-Space overflow during object evacuation");
    }

    uint8_t* dest = to_space_.data() + to_free_ptr;
    std::memcpy(dest, reinterpret_cast<const void*>(obj_addr - sizeof(HostGcHeader)), total_size);

    auto* new_hdr = reinterpret_cast<HostGcHeader*>(dest);
    new_hdr->forwarding_address = 0;
    uintptr_t new_payload_addr = reinterpret_cast<uintptr_t>(new_hdr + 1);

    hdr->forwarding_address = new_payload_addr;
    to_free_ptr += total_size;

    return new_payload_addr;
}

void HostGC::collect(uintptr_t top_rbp, uintptr_t top_return_ip) {
    std::vector<uintptr_t*> extra_ptr_roots;
    std::vector<HostValue*> extra_val_roots;
    collect(extra_ptr_roots, extra_val_roots, top_rbp, top_return_ip);
}

void HostGC::collect(
    std::vector<uintptr_t*>& extra_ptr_roots,
    std::vector<HostValue*>& extra_val_roots,
    uintptr_t top_rbp,
    uintptr_t top_return_ip
) {
    size_t to_free_ptr = 0;

    // 1. Relocate roots from native stack frames via Brass stack maps
    if (top_rbp != 0 && top_return_ip != 0 && stack_maps_ != nullptr) {
        brass_stack_walk(top_rbp, top_return_ip, *stack_maps_, [&](void** slot) {
            if (!slot || !*slot) return;
            auto raw_val = reinterpret_cast<uintptr_t>(*slot);
            HostValue hv(raw_val);
            if (hv.is_gcref()) {
                uintptr_t old_addr = hv.as_gcref();
                if (is_address_in_active_space(old_addr)) {
                    uintptr_t new_addr = evacuate_object(old_addr, to_free_ptr);
                    hv.update_gcref(new_addr);
                    *reinterpret_cast<uint64_t*>(slot) = hv.raw();
                }
            } else if (is_address_in_active_space(raw_val)) {
                uintptr_t new_addr = evacuate_object(raw_val, to_free_ptr);
                *slot = reinterpret_cast<void*>(new_addr);
            }
        });
    }

    // 2. Relocate registered and passed-in host HostValue roots
    for (auto* val_root : registered_val_roots_) {
        if (val_root && val_root->is_gcref()) {
            uintptr_t old_addr = val_root->as_gcref();
            if (is_address_in_active_space(old_addr)) {
                uintptr_t new_addr = evacuate_object(old_addr, to_free_ptr);
                val_root->update_gcref(new_addr);
            }
        }
    }
    for (auto* val_root : extra_val_roots) {
        if (val_root && val_root->is_gcref()) {
            uintptr_t old_addr = val_root->as_gcref();
            if (is_address_in_active_space(old_addr)) {
                uintptr_t new_addr = evacuate_object(old_addr, to_free_ptr);
                val_root->update_gcref(new_addr);
            }
        }
    }

    // 3. Relocate registered and passed-in host raw pointer roots
    for (auto* ptr_root : registered_ptr_roots_) {
        if (ptr_root && *ptr_root && is_address_in_active_space(*ptr_root)) {
            *ptr_root = evacuate_object(*ptr_root, to_free_ptr);
        }
    }
    for (auto* ptr_root : extra_ptr_roots) {
        if (ptr_root && *ptr_root && is_address_in_active_space(*ptr_root)) {
            *ptr_root = evacuate_object(*ptr_root, to_free_ptr);
        }
    }

    // 4. Relocate provider roots
    if (root_provider_) {
        std::vector<uintptr_t*> prov_ptrs;
        std::vector<HostValue*> prov_vals;
        root_provider_(prov_ptrs, prov_vals);
        for (auto* val_root : prov_vals) {
            if (val_root && val_root->is_gcref()) {
                uintptr_t old_addr = val_root->as_gcref();
                if (is_address_in_active_space(old_addr)) {
                    uintptr_t new_addr = evacuate_object(old_addr, to_free_ptr);
                    val_root->update_gcref(new_addr);
                }
            }
        }
        for (auto* ptr_root : prov_ptrs) {
            if (ptr_root && *ptr_root && is_address_in_active_space(*ptr_root)) {
                *ptr_root = evacuate_object(*ptr_root, to_free_ptr);
            }
        }
    }

    // 5. Cheney scan queue: scan evacuated objects in To-Space
    size_t scan_ptr = 0;
    while (scan_ptr < to_free_ptr) {
        auto* hdr = reinterpret_cast<HostGcHeader*>(to_space_.data() + scan_ptr);
        uintptr_t obj_payload = reinterpret_cast<uintptr_t>(hdr + 1);
        size_t num_fields = hdr->size / 8;

        for (size_t i = 0; i < num_fields && i < 64; ++i) {
            if ((hdr->pointer_mask & (1ULL << i)) != 0) {
                auto* field_ptr = reinterpret_cast<uint64_t*>(obj_payload + i * 8);
                HostValue field_val(*field_ptr);
                if (field_val.is_gcref()) {
                    uintptr_t child = field_val.as_gcref();
                    if (is_address_in_active_space(child)) {
                        uintptr_t new_child = evacuate_object(child, to_free_ptr);
                        field_val.update_gcref(new_child);
                        *field_ptr = field_val.raw();
                    }
                } else if (is_address_in_active_space(static_cast<uintptr_t>(*field_ptr))) {
                    uintptr_t child = static_cast<uintptr_t>(*field_ptr);
                    uintptr_t new_child = evacuate_object(child, to_free_ptr);
                    *field_ptr = static_cast<uint64_t>(new_child);
                }
            }
        }

        size_t aligned_size = (hdr->size + 7) & ~static_cast<size_t>(7);
        scan_ptr += sizeof(HostGcHeader) + aligned_size;
    }

    // 6. Swap semispaces and poison old From-Space
    std::swap(from_space_, to_space_);
    free_ptr_ = to_free_ptr;
    collection_count_++;
    poison_space(to_space_.data(), semispace_size_);
}

void HostGC::safepoint(uintptr_t top_rbp, uintptr_t top_return_ip) {
    uintptr_t rbp = top_rbp;
    uintptr_t ip = top_return_ip;
    if (rbp == 0 || ip == 0) {
        get_caller_frame(rbp, ip);
    }
    collect(rbp, ip);
}

uintptr_t HostGC::allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    std::vector<uintptr_t*> extra_ptr_roots;
    std::vector<HostValue*> extra_val_roots;
    return allocate(size, pointer_mask, type_tag, extra_ptr_roots, extra_val_roots);
}

uintptr_t HostGC::allocate(
    size_t size,
    uint64_t pointer_mask,
    uint32_t type_tag,
    std::vector<uintptr_t*>& extra_ptr_roots,
    std::vector<HostValue*>& extra_val_roots
) {
    size_t aligned_size = (size + 7) & ~static_cast<size_t>(7);
    size_t total_size = sizeof(HostGcHeader) + aligned_size;

    if (total_size > semispace_size_) {
        throw std::runtime_error("HostGC: Object size exceeds semispace capacity");
    }

    if (free_ptr_ + total_size > semispace_size_) {
        collect(extra_ptr_roots, extra_val_roots, 0, 0);

        if (free_ptr_ + total_size > semispace_size_) {
            throw std::runtime_error("HostGC: Out of memory after collection");
        }
    }

    uint8_t* obj_mem = from_space_.data() + free_ptr_;
    auto* hdr = reinterpret_cast<HostGcHeader*>(obj_mem);
    hdr->size = static_cast<uint32_t>(aligned_size);
    hdr->type_tag = type_tag;
    hdr->pointer_mask = pointer_mask;
    hdr->forwarding_address = 0;

    uintptr_t payload_addr = reinterpret_cast<uintptr_t>(hdr + 1);
    std::memset(reinterpret_cast<void*>(payload_addr), 0, aligned_size);

    free_ptr_ += total_size;
    total_allocations_++;
    total_allocated_bytes_ += aligned_size;

    return payload_addr;
}

HostValue HostGC::allocate_value(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    uintptr_t addr = allocate(size, pointer_mask, type_tag);
    return HostValue::from_gcref(addr);
}

void HostGC::reset() {
    free_ptr_ = 0;
    collection_count_ = 0;
    total_allocations_ = 0;
    total_allocated_bytes_ = 0;
    registered_val_roots_.clear();
    registered_ptr_roots_.clear();
    poison_space(from_space_.data(), semispace_size_);
    poison_space(to_space_.data(), semispace_size_);
}

} // namespace brass

#if defined(_MSC_VER)

extern "C" {

void host_gc_safepoint_bridge(uintptr_t caller_rbp, uintptr_t caller_ip) {
    auto* gc = brass::get_active_host_gc();
    if (!gc) return;
    gc->safepoint(caller_rbp, caller_ip);
}

uintptr_t host_gc_alloc_bridge(size_t size, uint64_t pointer_mask, uint32_t type_tag, uintptr_t caller_rbp, uintptr_t caller_ip) {
    auto* gc = brass::get_active_host_gc();
    if (!gc) return 0;

    if (!gc->can_allocate_fast(size)) {
        std::vector<uintptr_t*> ptr_roots;
        std::vector<brass::HostValue*> val_roots;
        gc->collect(ptr_roots, val_roots, caller_rbp, caller_ip);
    }
    return gc->allocate(size, pointer_mask, type_tag);
}

uint64_t host_gc_alloc_nanbox_bridge(size_t size, uint64_t pointer_mask, uint32_t type_tag, uintptr_t caller_rbp, uintptr_t caller_ip) {
    auto* gc = brass::get_active_host_gc();
    if (!gc) return brass::HostValue::null_val().raw();

    if (!gc->can_allocate_fast(size)) {
        std::vector<uintptr_t*> ptr_roots;
        std::vector<brass::HostValue*> val_roots;
        gc->collect(ptr_roots, val_roots, caller_rbp, caller_ip);
    }
    return gc->allocate_value(size, pointer_mask, type_tag).raw();
}

} // extern "C"

#else

extern "C" {

void host_gc_safepoint() {
    auto* gc = brass::get_active_host_gc();
    if (!gc) return;

    void* frame = __builtin_frame_address(0);
    uintptr_t caller_rbp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
    uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
    gc->safepoint(caller_rbp, caller_ip);
}

uintptr_t host_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    auto* gc = brass::get_active_host_gc();
    if (!gc) return 0;

    if (!gc->can_allocate_fast(size)) {
        void* frame = __builtin_frame_address(0);
        uintptr_t caller_rbp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
        uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
        std::vector<uintptr_t*> ptr_roots;
        std::vector<brass::HostValue*> val_roots;
        gc->collect(ptr_roots, val_roots, caller_rbp, caller_ip);
    }
    return gc->allocate(size, pointer_mask, type_tag);
}

uint64_t host_gc_alloc_nanbox(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    auto* gc = brass::get_active_host_gc();
    if (!gc) return brass::HostValue::null_val().raw();

    if (!gc->can_allocate_fast(size)) {
        void* frame = __builtin_frame_address(0);
        uintptr_t caller_rbp = frame ? *reinterpret_cast<uintptr_t*>(frame) : 0;
        uintptr_t caller_ip = reinterpret_cast<uintptr_t>(__builtin_return_address(0));
        std::vector<uintptr_t*> ptr_roots;
        std::vector<brass::HostValue*> val_roots;
        gc->collect(ptr_roots, val_roots, caller_rbp, caller_ip);
    }
    return gc->allocate_value(size, pointer_mask, type_tag).raw();
}

void host_gc_collect() {
    auto* gc = brass::get_active_host_gc();
    if (!gc) return;
    gc->collect();
}

} // extern "C"

#endif
