#include <brass/embedding/host_gc.hpp>
#include <brass/gc/tlab.hpp>
#include <brass/runtime/coroutine.hpp>
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
    reset_active_tlabs(true);
    if (g_active_host_gc == this) {
        g_active_host_gc = nullptr;
    }
}

HostGC::HostGC(HostGC&& other) noexcept {
    std::lock_guard<std::recursive_mutex> lock(other.gc_mutex_);
    semispace_size_ = other.semispace_size_;
    from_space_ = std::move(other.from_space_);
    to_space_ = std::move(other.to_space_);
    free_ptr_.store(other.free_ptr_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    stress_mode_ = other.stress_mode_;
    collection_count_.store(other.collection_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    total_allocations_.store(other.total_allocations_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    total_allocated_bytes_.store(other.total_allocated_bytes_.load(std::memory_order_relaxed), std::memory_order_relaxed);
    stack_maps_ = other.stack_maps_;
    registered_val_roots_ = std::move(other.registered_val_roots_);
    registered_ptr_roots_ = std::move(other.registered_ptr_roots_);
    registered_tlabs_ = std::move(other.registered_tlabs_);
    other.registered_tlabs_.clear();
    adopt_registered_tlabs(&other);
    root_provider_ = std::move(other.root_provider_);
    coro_frames_ = std::move(other.coro_frames_);
}

HostGC& HostGC::operator=(HostGC&& other) noexcept {
    if (this != &other) {
        // Our own TLABs would otherwise be dropped from the list while still
        // naming us as owner.
        reset_active_tlabs(true);
        std::scoped_lock lock(gc_mutex_, other.gc_mutex_);
        semispace_size_ = other.semispace_size_;
        from_space_ = std::move(other.from_space_);
        to_space_ = std::move(other.to_space_);
        free_ptr_.store(other.free_ptr_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        stress_mode_ = other.stress_mode_;
        collection_count_.store(other.collection_count_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_allocations_.store(other.total_allocations_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        total_allocated_bytes_.store(other.total_allocated_bytes_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        stack_maps_ = other.stack_maps_;
        registered_val_roots_ = std::move(other.registered_val_roots_);
        registered_ptr_roots_ = std::move(other.registered_ptr_roots_);
        registered_tlabs_ = std::move(other.registered_tlabs_);
        other.registered_tlabs_.clear();
        adopt_registered_tlabs(&other);
        root_provider_ = std::move(other.root_provider_);
        coro_frames_ = std::move(other.coro_frames_);
    }
    return *this;
}

// The TLABs moved over from `from` still name it as owner; `from` is about to
// be destroyed or reused, so point them at the collector that now owns the
// space they allocate from.
void HostGC::adopt_registered_tlabs(HostGC* from) noexcept {
    for (auto* tlab : registered_tlabs_) {
        if (tlab && tlab->owner_gc == from) {
            tlab->owner_gc = this;
        }
    }
    if (g_active_host_gc == from) {
        g_active_host_gc = this;
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
        std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
        registered_val_roots_.push_back(root_slot);
    }
}

void HostGC::unregister_root(HostValue* root_slot) {
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
    auto it = std::remove(registered_val_roots_.begin(), registered_val_roots_.end(), root_slot);
    registered_val_roots_.erase(it, registered_val_roots_.end());
}

void HostGC::register_root(uintptr_t* root_slot) {
    if (root_slot) {
        std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
        registered_ptr_roots_.push_back(root_slot);
    }
}

void HostGC::unregister_root(uintptr_t* root_slot) {
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
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
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
    reset_active_tlabs();
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

    // 3.5. Relocate active coroutine roots (their slots are read and
    // written under the registry's lock)
    {
        runtime::CoroRootsLock coro_lock(coro_frames());
        std::vector<uintptr_t*> coro_roots;
        runtime::append_active_coro_roots(coro_frames(), coro_roots);
        for (auto* ptr_root : coro_roots) {
            if (ptr_root && *ptr_root && is_address_in_active_space(*ptr_root)) {
                *ptr_root = evacuate_object(*ptr_root, to_free_ptr);
            }
        }
    }

    // 4. Custom root provider hook
    if (root_provider_) {
        std::vector<uintptr_t*> provider_ptr_roots;
        std::vector<HostValue*> provider_val_roots;
        root_provider_(provider_ptr_roots, provider_val_roots);
        for (auto* val_root : provider_val_roots) {
            if (val_root && val_root->is_gcref()) {
                uintptr_t old_addr = val_root->as_gcref();
                if (is_address_in_active_space(old_addr)) {
                    uintptr_t new_addr = evacuate_object(old_addr, to_free_ptr);
                    val_root->update_gcref(new_addr);
                }
            }
        }
        for (auto* ptr_root : provider_ptr_roots) {
            if (ptr_root && *ptr_root && is_address_in_active_space(*ptr_root)) {
                *ptr_root = evacuate_object(*ptr_root, to_free_ptr);
            }
        }
    }

    // 5. Cheney scanning of newly evacuated objects in To-Space
    size_t scan_ptr = 0;
    while (scan_ptr < to_free_ptr) {
        auto* hdr = reinterpret_cast<HostGcHeader*>(to_space_.data() + scan_ptr);
        size_t field_count = hdr->size / sizeof(uint64_t);
        auto* fields = reinterpret_cast<uint64_t*>(hdr + 1);

        for (size_t i = 0; i < field_count; ++i) {
            if (hdr->is_field_pointer(i)) {
                auto* field_ptr = &fields[i];
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
    free_ptr_.store(to_free_ptr, std::memory_order_release);
    collection_count_.fetch_add(1, std::memory_order_relaxed);
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
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
    reset_active_tlabs();

    size_t aligned_size = (size + 7) & ~static_cast<size_t>(7);
    size_t total_size = sizeof(HostGcHeader) + aligned_size;

    if (total_size > semispace_size_) {
        throw std::runtime_error("HostGC: Object size exceeds semispace capacity");
    }

    size_t cur_free = free_ptr_.load(std::memory_order_relaxed);
    if (cur_free + total_size > semispace_size_) {
        uintptr_t caller_rbp = 0;
        uintptr_t caller_ip = 0;
        get_caller_frame(caller_rbp, caller_ip);
        collect(extra_ptr_roots, extra_val_roots, caller_rbp, caller_ip);

        cur_free = free_ptr_.load(std::memory_order_relaxed);
        if (cur_free + total_size > semispace_size_) {
            throw std::runtime_error("HostGC: Out of memory after collection");
        }
    }

    uint8_t* obj_mem = from_space_.data() + cur_free;
    auto* hdr = reinterpret_cast<HostGcHeader*>(obj_mem);
    hdr->size = static_cast<uint32_t>(aligned_size);
    hdr->type_tag = type_tag;
    hdr->pointer_mask = pointer_mask;
    hdr->forwarding_address = 0;

    uintptr_t payload_addr = reinterpret_cast<uintptr_t>(hdr + 1);
    std::memset(reinterpret_cast<void*>(payload_addr), 0, aligned_size);

    free_ptr_.store(cur_free + total_size, std::memory_order_release);
    total_allocations_.fetch_add(1, std::memory_order_relaxed);
    total_allocated_bytes_.fetch_add(aligned_size, std::memory_order_relaxed);

    return payload_addr;
}

HostValue HostGC::allocate_value(size_t size, uint64_t pointer_mask, uint32_t type_tag) {
    uintptr_t addr = allocate(size, pointer_mask, type_tag);
    return HostValue::from_gcref(addr);
}

void HostGC::reset() {
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
    reset_active_tlabs(true);
    free_ptr_.store(0, std::memory_order_relaxed);
    collection_count_.store(0, std::memory_order_relaxed);
    total_allocations_.store(0, std::memory_order_relaxed);
    total_allocated_bytes_.store(0, std::memory_order_relaxed);
    registered_val_roots_.clear();
    registered_ptr_roots_.clear();
    poison_space(from_space_.data(), semispace_size_);
    poison_space(to_space_.data(), semispace_size_);
    coro_frames_.reset(); // its frames are gone with the heap's contents
}

runtime::CoroFrameRegistry& HostGC::coro_frames() {
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
    if (!coro_frames_) coro_frames_ = runtime::make_coro_frame_registry();
    return *coro_frames_;
}

bool HostGC::allocate_tlab(size_t min_bytes, size_t preferred_size, uintptr_t& out_top, uintptr_t& out_end) {
    if (stress_mode_) {
        out_top = 0;
        out_end = 0;
        return false;
    }
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
    size_t aligned_min = (min_bytes + 7) & ~static_cast<size_t>(7);
    size_t chunk_size = std::max(aligned_min, preferred_size);
    chunk_size = (chunk_size + 7) & ~static_cast<size_t>(7);

    if (chunk_size > semispace_size_) {
        chunk_size = aligned_min;
        if (chunk_size > semispace_size_) {
            out_top = 0;
            out_end = 0;
            return false;
        }
    }

    size_t cur_free = free_ptr_.load(std::memory_order_relaxed);
    if (cur_free + chunk_size > semispace_size_) {
        if (cur_free + aligned_min <= semispace_size_) {
            chunk_size = (semispace_size_ - cur_free) & ~static_cast<size_t>(7);
        } else {
            std::vector<uintptr_t*> ptr_roots;
            std::vector<HostValue*> val_roots;
            uintptr_t caller_rbp = 0;
            uintptr_t caller_ip = 0;
            get_caller_frame(caller_rbp, caller_ip);
            collect(ptr_roots, val_roots, caller_rbp, caller_ip);

            cur_free = free_ptr_.load(std::memory_order_relaxed);
            if (cur_free + aligned_min > semispace_size_) {
                out_top = 0;
                out_end = 0;
                return false;
            }
            if (cur_free + chunk_size > semispace_size_) {
                chunk_size = (semispace_size_ - cur_free) & ~static_cast<size_t>(7);
            }
        }
    }

    uint8_t* mem = from_space_.data() + cur_free;
    out_top = reinterpret_cast<uintptr_t>(mem);
    out_end = out_top + chunk_size;
    free_ptr_.store(cur_free + chunk_size, std::memory_order_release);
    return true;
}

void HostGC::retire_tlab(uintptr_t top, uintptr_t end) {
    if (end == 0 || top > end) return;
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
    uintptr_t space_start = reinterpret_cast<uintptr_t>(from_space_.data());
    uintptr_t space_cur = space_start + free_ptr_.load(std::memory_order_relaxed);
    if (end == space_cur) {
        size_t unused = end - top;
        free_ptr_.fetch_sub(unused, std::memory_order_relaxed);
    }
}

void HostGC::register_tlab(ThreadLocalAllocBuffer* tlab) {
    if (!tlab) return;
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
    for (auto* t : registered_tlabs_) {
        if (t == tlab) return;
    }
    registered_tlabs_.push_back(tlab);
}

void HostGC::unregister_tlab(ThreadLocalAllocBuffer* tlab) {
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
    auto it = std::remove(registered_tlabs_.begin(), registered_tlabs_.end(), tlab);
    registered_tlabs_.erase(it, registered_tlabs_.end());
}

void HostGC::reset_active_tlabs(bool clear_owner) {
    std::lock_guard<std::recursive_mutex> lock(gc_mutex_);
    auto* active = get_active_tlab();
    if (active && active->owner_gc == this) {
        active->reset();
        if (clear_owner) {
            active->owner_gc = nullptr;
        }
    }
    for (auto* tlab : registered_tlabs_) {
        if (tlab && tlab != active && tlab->owner_gc == this) {
            tlab->reset();
            if (clear_owner) {
                tlab->owner_gc = nullptr;
            }
        }
    }
    if (clear_owner) {
        registered_tlabs_.clear();
    }
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
