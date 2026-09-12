#pragma once

#include <brass/embedding/nanbox.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/gc/stack_walker.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>

namespace brass {

struct ThreadLocalAllocBuffer;

struct HostGcHeader {
    uint32_t size;                // Object payload size in bytes (8-byte aligned)
    uint32_t type_tag;            // User/runtime type tag
    uint64_t pointer_mask;        // Bit i is 1 if 8-byte field i is a GCRef pointer or HostValue containing a GCRef
    uintptr_t forwarding_address; // Forwarding address during collection when evacuated to To-Space
};

class HostGC {
public:
    static constexpr size_t DEFAULT_SEMISPACE_SIZE = 1024 * 1024; // 1 MB
    static constexpr uint64_t POISON_PATTERN = 0xDEADBEEFDEADBEEFULL;

    using RootProvider = std::function<void(std::vector<uintptr_t*>&, std::vector<HostValue*>&)>;

    explicit HostGC(size_t semispace_size = DEFAULT_SEMISPACE_SIZE);
    ~HostGC();

    HostGC(const HostGC&) = delete;
    HostGC& operator=(const HostGC&) = delete;
    HostGC(HostGC&&) noexcept = default;
    HostGC& operator=(HostGC&&) noexcept = default;

    // Active stack maps registration
    void set_stack_maps(const ModuleStackMap* maps) noexcept { stack_maps_ = maps; }
    const ModuleStackMap* stack_maps() const noexcept { return stack_maps_; }

    // Allocation
    uintptr_t allocate(size_t size, uint64_t pointer_mask = 0, uint32_t type_tag = 0);
    uintptr_t allocate(
        size_t size,
        uint64_t pointer_mask,
        uint32_t type_tag,
        std::vector<uintptr_t*>& extra_ptr_roots,
        std::vector<HostValue*>& extra_val_roots
    );
    HostValue allocate_value(size_t size, uint64_t pointer_mask = 0, uint32_t type_tag = 0);

    // Garbage Collection & Safepoints
    void collect(uintptr_t top_rbp = 0, uintptr_t top_return_ip = 0);
    void collect(
        std::vector<uintptr_t*>& extra_ptr_roots,
        std::vector<HostValue*>& extra_val_roots,
        uintptr_t top_rbp = 0,
        uintptr_t top_return_ip = 0
    );
    void safepoint(uintptr_t top_rbp = 0, uintptr_t top_return_ip = 0);

    // Root management
    void register_root(HostValue* root_slot);
    void unregister_root(HostValue* root_slot);
    void register_root(uintptr_t* root_slot);
    void unregister_root(uintptr_t* root_slot);
    void set_root_provider(RootProvider provider);

    // Stress mode: Forces a full collection on every single allocation and safepoint
    void set_stress_mode(bool enable) noexcept { stress_mode_ = enable; }
    bool stress_mode() const noexcept { return stress_mode_; }

    // Fast allocation check
    bool can_allocate_fast(size_t size) const noexcept {
        size_t aligned_size = (size + 7) & ~static_cast<size_t>(7);
        return !stress_mode_ && (free_ptr_ + sizeof(HostGcHeader) + aligned_size <= semispace_size_);
    }

    // Object Validation
    bool is_valid_object(uintptr_t obj_addr) const noexcept;
    bool is_address_in_active_space(uintptr_t addr) const noexcept;

    // Header inspection
    HostGcHeader* get_header(uintptr_t obj_addr) noexcept;
    const HostGcHeader* get_header(uintptr_t obj_addr) const noexcept;

    // Field accessors (8-byte fields)
    uint64_t read_field(uintptr_t obj_addr, size_t field_idx) const;
    void write_field(uintptr_t obj_addr, size_t field_idx, uint64_t val);
    HostValue read_value_field(uintptr_t obj_addr, size_t field_idx) const;
    void write_value_field(uintptr_t obj_addr, size_t field_idx, HostValue val);

    // Statistics
    size_t semispace_size() const noexcept { return semispace_size_; }
    size_t bytes_allocated_active() const noexcept { return free_ptr_; }
    size_t total_allocated_bytes() const noexcept { return total_allocated_bytes_; }
    size_t total_allocations() const noexcept { return total_allocations_; }
    size_t collection_count() const noexcept { return collection_count_; }

    // TLAB Integration
    bool allocate_tlab(size_t min_bytes, size_t preferred_size, uintptr_t& out_top, uintptr_t& out_end);
    void retire_tlab(uintptr_t top, uintptr_t end);
    void register_tlab(ThreadLocalAllocBuffer* tlab);
    void unregister_tlab(ThreadLocalAllocBuffer* tlab);
    void reset_active_tlabs(bool clear_owner = false);

    // Reset heap
    void reset();

private:
    uintptr_t evacuate_object(uintptr_t obj_addr, size_t& to_free_ptr);
    void poison_space(uint8_t* space, size_t size) noexcept;

    size_t semispace_size_;
    std::vector<uint8_t> from_space_;
    std::vector<uint8_t> to_space_;
    size_t free_ptr_ = 0;

    bool stress_mode_ = false;
    size_t collection_count_ = 0;
    size_t total_allocations_ = 0;
    size_t total_allocated_bytes_ = 0;

    const ModuleStackMap* stack_maps_ = nullptr;
    std::vector<HostValue*> registered_val_roots_;
    std::vector<uintptr_t*> registered_ptr_roots_;
    std::vector<ThreadLocalAllocBuffer*> registered_tlabs_;
    RootProvider root_provider_;
};

// Global active HostGC for native JIT bridge callbacks
void set_active_host_gc(HostGC* gc) noexcept;
HostGC* get_active_host_gc() noexcept;

} // namespace brass

// Extern C runtime bridge symbols callable from JIT code
extern "C" {

void host_gc_safepoint();
uintptr_t host_gc_alloc(size_t size, uint64_t pointer_mask, uint32_t type_tag);
uint64_t host_gc_alloc_nanbox(size_t size, uint64_t pointer_mask, uint32_t type_tag);
void host_gc_collect();

}
