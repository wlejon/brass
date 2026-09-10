#pragma once

#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/patcher.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <iosfwd>

namespace brass::runtime {

enum class ICState : uint8_t {
    Uninitialized,
    Monomorphic,
    Polymorphic,
    Megamorphic
};

std::string_view to_string(ICState state) noexcept;
std::ostream& operator<<(std::ostream& os, ICState state);

class InlineCache {
public:
    static constexpr size_t POLYMORPHIC_LIMIT = 4;

    InlineCache(uint32_t site_id, std::string prop_name, uint32_t symbol_id = 0, bool is_load = true);

    InlineCache(const InlineCache&) = delete;
    InlineCache& operator=(const InlineCache&) = delete;

    [[nodiscard]] uint32_t site_id() const noexcept { return site_id_; }
    [[nodiscard]] std::string_view prop_name() const noexcept { return prop_name_; }
    [[nodiscard]] uint32_t symbol_id() const noexcept { return symbol_id_; }
    [[nodiscard]] bool is_load() const noexcept { return is_load_; }

    [[nodiscard]] ICState state() const noexcept { return state_; }
    [[nodiscard]] size_t entry_count() const noexcept { return entry_count_; }
    [[nodiscard]] const Shape* cached_shape(size_t idx = 0) const noexcept;
    [[nodiscard]] uint32_t cached_slot(size_t idx = 0) const noexcept;

    [[nodiscard]] uint64_t hit_count() const noexcept { return hit_count_; }
    [[nodiscard]] uint64_t miss_count() const noexcept { return miss_count_; }

    [[nodiscard]] void* call_site_address() const noexcept { return call_site_address_; }
    void set_call_site_address(void* addr) noexcept { call_site_address_ = addr; }

    [[nodiscard]] void* patch_point() const noexcept { return patch_point_; }
    void set_patch_point(void* addr) noexcept { patch_point_ = addr; }

    [[nodiscard]] std::string_view patch_site_name() const noexcept { return patch_site_name_; }
    void set_patch_site_name(std::string name) noexcept { patch_site_name_ = std::move(name); }

    // Fast-path Probe: returns slot index on hit, -1 on miss
    [[nodiscard]] int32_t probe_get(const Shape* shape) noexcept;
    [[nodiscard]] int32_t probe_set(const Shape* shape) noexcept;

    // Full property access execution (fast path + miss fallback)
    HostValue execute_get(DynamicObject* obj);
    void execute_set(DynamicObject* obj, HostValue val, ShapeRegistry& registry, HostGC* gc = nullptr);

    // Miss handlers
    HostValue miss_handler_get(DynamicObject* obj);
    void miss_handler_set(DynamicObject* obj, HostValue val, ShapeRegistry& registry, HostGC* gc = nullptr);

    // Cache mutation & invalidation
    void record_hit() noexcept { hit_count_++; }
    void record_miss() noexcept { miss_count_++; }
    void reset() noexcept;
    void invalidate() noexcept;

private:
    uint32_t site_id_ = 0;
    std::string prop_name_;
    uint32_t symbol_id_ = 0;
    bool is_load_ = true;
    ICState state_ = ICState::Uninitialized;

    const Shape* cached_shapes_[POLYMORPHIC_LIMIT] = {nullptr, nullptr, nullptr, nullptr};
    uint32_t cached_slots_[POLYMORPHIC_LIMIT] = {0, 0, 0, 0};
    size_t entry_count_ = 0;

    uint64_t hit_count_ = 0;
    uint64_t miss_count_ = 0;

    void* call_site_address_ = nullptr;
    void* patch_point_ = nullptr;
    std::string patch_site_name_;
};

class ICRegistry {
public:
    ICRegistry() = default;
    ~ICRegistry() = default;

    ICRegistry(const ICRegistry&) = delete;
    ICRegistry& operator=(const ICRegistry&) = delete;

    InlineCache* get_or_create_ic(
        uint32_t site_id,
        std::string_view prop_name,
        uint32_t symbol_id = 0,
        bool is_load = true
    );

    [[nodiscard]] InlineCache* find_ic(uint32_t site_id) const;
    [[nodiscard]] size_t size() const noexcept;
    void reset_all();
    void clear();

    void dump_stats(std::ostream& os) const;

    static ICRegistry& global() noexcept;

private:
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<InlineCache>> caches_;
    std::unordered_map<uint32_t, InlineCache*> id_to_ic_;
};

// Codegen & Patching Integration
bool patch_monomorphic_ic(
    InlineCache& ic,
    const Shape* shape,
    uint32_t slot,
    PatchRegistry* patch_registry = nullptr
);

bool patch_ic_call_target(InlineCache& ic, const void* new_stub_target);

} // namespace brass::runtime

// C Bridge runtime stubs for JIT execution
extern "C" {
    uint64_t brass_ic_get_prop(uint32_t site_id, uint64_t obj_raw, const char* name, uint32_t symbol_id);
    void brass_ic_set_prop(uint32_t site_id, uint64_t obj_raw, const char* name, uint32_t symbol_id, uint64_t val_raw);
    uint64_t brass_ic_miss_handler_get(uint32_t site_id, uint64_t obj_raw);
    void brass_ic_miss_handler_set(uint32_t site_id, uint64_t obj_raw, uint64_t val_raw);
}
