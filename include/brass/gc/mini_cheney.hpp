#pragma once

#include <brass/interpreter/value.hpp>
#include <brass/mir/types.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>

namespace brass {

struct GcHeader {
    uint32_t size;                // Object payload size in bytes (8-byte aligned)
    uint32_t type_tag;            // User/runtime type tag
    uint64_t pointer_mask;        // Bit i is 1 if 8-byte field i is a gcref pointer
    uintptr_t forwarding_address; // Used during collection when object is evacuated to To-Space
};

class MiniCheneyGC {
public:
    static constexpr size_t DEFAULT_SEMISPACE_SIZE = 1024 * 1024; // 1 MB
    static constexpr uint64_t POISON_PATTERN = 0xDEADBEEFDEADBEEFULL;

    using RootProvider = std::function<void(std::vector<uintptr_t*>&)>;

    explicit MiniCheneyGC(size_t semispace_size = DEFAULT_SEMISPACE_SIZE);
    ~MiniCheneyGC() = default;

    MiniCheneyGC(const MiniCheneyGC&) = delete;
    MiniCheneyGC& operator=(const MiniCheneyGC&) = delete;
    MiniCheneyGC(MiniCheneyGC&&) noexcept = default;
    MiniCheneyGC& operator=(MiniCheneyGC&&) noexcept = default;

    // Allocation
    uintptr_t allocate(size_t size, uint64_t pointer_mask = 0, uint32_t type_tag = 0);
    uintptr_t allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag, std::vector<uintptr_t*>& extra_roots);

    // Garbage Collection
    void collect();
    void collect(std::vector<uintptr_t*>& roots);

    // Root management
    void register_root(uintptr_t* root_slot);
    void unregister_root(uintptr_t* root_slot);
    void set_root_provider(RootProvider provider);

    // Stress Mode: Forces a full collection on every single allocation and safepoint
    void set_stress_mode(bool enable) noexcept { stress_mode_ = enable; }
    bool stress_mode() const noexcept { return stress_mode_; }

    // Fast allocation check
    bool can_allocate_fast(size_t size) const noexcept {
        size_t aligned_size = (size + 7) & ~static_cast<size_t>(7);
        return !stress_mode_ && (free_ptr_ + sizeof(GcHeader) + aligned_size <= semispace_size_);
    }

    // Object Validation
    bool is_valid_object(uintptr_t obj_addr) const noexcept;
    bool is_address_in_active_space(uintptr_t addr) const noexcept;

    // Header inspection
    GcHeader* get_header(uintptr_t obj_addr) noexcept;
    const GcHeader* get_header(uintptr_t obj_addr) const noexcept;

    // Field accessors (8-byte fields)
    uint64_t read_field(uintptr_t obj_addr, size_t field_idx) const;
    void write_field(uintptr_t obj_addr, size_t field_idx, uint64_t val);
    void write_field(uintptr_t obj_addr, size_t field_idx, RuntimeValue val);

    // Memory access helpers (handles ptr and gcref)
    RuntimeValue read_memory(uintptr_t base, int32_t offset, Type t) const;
    void write_memory(uintptr_t base, int32_t offset, Type t, RuntimeValue val);

    // Statistics
    size_t semispace_size() const noexcept { return semispace_size_; }
    size_t bytes_allocated_active() const noexcept { return free_ptr_; }
    size_t total_allocated_bytes() const noexcept { return total_allocated_bytes_; }
    size_t total_allocations() const noexcept { return total_allocations_; }
    size_t collection_count() const noexcept { return collection_count_; }

    // Reset heap
    void reset();

private:
    uintptr_t evacuate_object(uintptr_t obj_addr, size_t& to_free_ptr);
    void poison_space(uint8_t* space, size_t size) noexcept;
    void gather_all_roots(std::vector<uintptr_t*>& roots);

    size_t semispace_size_;
    std::vector<uint8_t> from_space_;
    std::vector<uint8_t> to_space_;
    size_t free_ptr_ = 0;

    bool stress_mode_ = false;
    size_t collection_count_ = 0;
    size_t total_allocations_ = 0;
    size_t total_allocated_bytes_ = 0;

    std::vector<uintptr_t*> registered_roots_;
    RootProvider root_provider_;
};

} // namespace brass
