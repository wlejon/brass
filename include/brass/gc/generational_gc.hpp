#pragma once

#include <brass/interpreter/value.hpp>
#include <brass/mir/types.hpp>
#include <brass/gc/card_table.hpp>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <functional>
#include <memory>
#include <stdexcept>

namespace brass {

static constexpr uint8_t GEN_YOUNG = 0;
static constexpr uint8_t GEN_OLD = 1;

struct GenGcHeader {
    uint32_t size;                // Object payload size in bytes (8-byte aligned)
    uint32_t type_tag;            // User/runtime type tag
    uint64_t pointer_mask;        // Bit i is 1 if 8-byte field i is a gcref pointer
    uintptr_t forwarding_address; // Used during collection when object is evacuated
    uint8_t age;                  // Age in minor collection cycles survived
    uint8_t generation;           // GEN_YOUNG = 0, GEN_OLD = 1
    uint16_t reserved16;          // Alignment / flags
    uint32_t reserved32;          // 8-byte header alignment padding (total 32 bytes)
};

class GenerationalGC {
public:
    static constexpr size_t DEFAULT_NURSERY_SIZE = 512 * 1024;       // 512 KB
    static constexpr size_t DEFAULT_SURVIVOR_SIZE = 256 * 1024;      // 256 KB
    static constexpr size_t DEFAULT_TENURED_SIZE = 4 * 1024 * 1024;  // 4 MB
    static constexpr uint8_t DEFAULT_TENURING_THRESHOLD = 2;
    static constexpr double DEFAULT_TENURED_OCCUPANCY_THRESHOLD = 0.80;
    static constexpr uint64_t POISON_PATTERN = 0xDEADBEEFDEADBEEFULL;

    using RootProvider = std::function<void(std::vector<uintptr_t*>&)>;

    explicit GenerationalGC(
        size_t nursery_size = DEFAULT_NURSERY_SIZE,
        size_t survivor_size = DEFAULT_SURVIVOR_SIZE,
        size_t tenured_size = DEFAULT_TENURED_SIZE,
        uint8_t tenuring_threshold = DEFAULT_TENURING_THRESHOLD
    );
    ~GenerationalGC() = default;

    GenerationalGC(const GenerationalGC&) = delete;
    GenerationalGC& operator=(const GenerationalGC&) = delete;
    GenerationalGC(GenerationalGC&&) noexcept = default;
    GenerationalGC& operator=(GenerationalGC&&) noexcept = default;

    // Allocation
    uintptr_t allocate(size_t size, uint64_t pointer_mask = 0, uint32_t type_tag = 0);
    uintptr_t allocate(size_t size, uint64_t pointer_mask, uint32_t type_tag, std::vector<uintptr_t*>& extra_roots);

    // Fast allocation check in nursery
    [[nodiscard]] bool can_allocate_fast(size_t size) const noexcept;

    // Garbage Collection
    void minor_collect();
    void minor_collect(std::vector<uintptr_t*>& extra_roots);
    void major_collect();
    void major_collect(std::vector<uintptr_t*>& extra_roots);
    void collect();
    void collect(std::vector<uintptr_t*>& extra_roots);
    void collect_major() { major_collect(); }
    void collect_major(std::vector<uintptr_t*>& extra_roots) { major_collect(extra_roots); }

    // Card Table
    [[nodiscard]] CardTable& card_table() noexcept { return card_table_; }
    [[nodiscard]] const CardTable& card_table() const noexcept { return card_table_; }

    // Generation queries
    [[nodiscard]] bool is_young(uintptr_t addr) const noexcept;
    [[nodiscard]] bool is_old(uintptr_t addr) const noexcept;
    [[nodiscard]] bool is_in_nursery(uintptr_t addr) const noexcept;
    [[nodiscard]] bool is_in_survivor(uintptr_t addr) const noexcept;
    [[nodiscard]] bool is_in_tenured(uintptr_t addr) const noexcept;
    [[nodiscard]] bool is_valid_object(uintptr_t addr) const noexcept;

    // Header inspection
    [[nodiscard]] GenGcHeader* get_header(uintptr_t obj_addr) noexcept;
    [[nodiscard]] const GenGcHeader* get_header(uintptr_t obj_addr) const noexcept;

    // Field access & write barrier helper
    [[nodiscard]] uint64_t read_field(uintptr_t obj_addr, size_t field_idx) const;
    void write_field(uintptr_t obj_addr, size_t field_idx, uint64_t val);
    void write_field(uintptr_t obj_addr, size_t field_idx, RuntimeValue val);
    void write_barrier(uintptr_t obj_addr, uintptr_t val) noexcept;

    // Memory access helpers (handles ptr and gcref)
    [[nodiscard]] RuntimeValue read_memory(uintptr_t base, int32_t offset, Type t) const;
    void write_memory(uintptr_t base, int32_t offset, Type t, RuntimeValue val);

    // Root management
    void register_root(uintptr_t* root_slot);
    void unregister_root(uintptr_t* root_slot);
    void set_root_provider(RootProvider provider);

    // Stress mode: Forces collection on every single allocation
    void set_stress_mode(bool enable) noexcept { stress_mode_ = enable; }
    [[nodiscard]] bool stress_mode() const noexcept { return stress_mode_; }

    void set_tenuring_threshold(uint8_t threshold) noexcept { tenuring_threshold_ = threshold; }
    [[nodiscard]] uint8_t tenuring_threshold() const noexcept { return tenuring_threshold_; }

    // Statistics
    [[nodiscard]] size_t nursery_size() const noexcept { return nursery_size_; }
    [[nodiscard]] size_t nursery_used() const noexcept { return nursery_free_; }
    [[nodiscard]] size_t tenured_size() const noexcept { return tenured_size_; }
    [[nodiscard]] size_t tenured_used() const noexcept { return tenured_free_; }
    [[nodiscard]] size_t minor_collection_count() const noexcept { return minor_collection_count_; }
    [[nodiscard]] size_t major_collection_count() const noexcept { return major_collection_count_; }
    [[nodiscard]] size_t minor_collections() const noexcept { return minor_collection_count_; }
    [[nodiscard]] size_t major_collections() const noexcept { return major_collection_count_; }
    [[nodiscard]] size_t promoted_objects() const noexcept { return tenured_objects_.size(); }
    [[nodiscard]] size_t total_allocations() const noexcept { return total_allocations_; }
    [[nodiscard]] size_t total_allocated_bytes() const noexcept { return total_allocated_bytes_; }
    [[nodiscard]] size_t promoted_bytes() const noexcept { return promoted_bytes_; }

    // Reset heap to empty state
    void reset();

private:
    void init_heap();
    uintptr_t allocate_nursery(size_t aligned_size, uint64_t pointer_mask, uint32_t type_tag);
    uintptr_t allocate_tenured(size_t aligned_size, uint64_t pointer_mask, uint32_t type_tag, uint8_t age);
    uintptr_t allocate_survivor(size_t aligned_size, uint64_t pointer_mask, uint32_t type_tag, uint8_t age);
    uintptr_t evacuate_young_object(uintptr_t obj_addr);
    void scan_dirty_cards();
    void scan_object_fields(uintptr_t obj_addr, GenGcHeader* hdr);
    void gather_all_roots(std::vector<uintptr_t*>& roots, std::vector<uintptr_t*>& extra_roots);
    void poison_range(uint8_t* start, size_t size) noexcept;

    size_t nursery_size_;
    size_t survivor_size_;
    size_t tenured_size_;
    uint8_t tenuring_threshold_;

    std::vector<uint8_t> heap_data_;
    uintptr_t heap_base_ = 0;

    // Space offsets within heap_data_
    size_t nursery_offset_ = 0;
    size_t survivor_a_offset_ = 0;
    size_t survivor_b_offset_ = 0;
    size_t tenured_offset_ = 0;

    size_t nursery_free_ = 0;
    size_t survivor_from_offset_ = 0;
    size_t survivor_to_offset_ = 0;
    size_t survivor_from_free_ = 0;
    size_t survivor_to_free_ = 0;
    size_t tenured_free_ = 0;

    std::vector<uintptr_t> tenured_objects_;
    CardTable card_table_;

    bool stress_mode_ = false;
    size_t minor_collection_count_ = 0;
    size_t major_collection_count_ = 0;
    size_t total_allocations_ = 0;
    size_t total_allocated_bytes_ = 0;
    size_t promoted_bytes_ = 0;

    std::vector<uintptr_t*> registered_roots_;
    RootProvider root_provider_;
};

} // namespace brass
