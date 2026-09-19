#pragma once

#include <brass/mir/builder.hpp>
#include <brass/mir/instruction.hpp>
#include <functional>
#include <string_view>
#include <string>
#include <cstdint>

namespace brass::il {

class PropertyLoweringHelper {
public:
    explicit PropertyLoweringHelper(bool enable_pic = true, bool enable_inlined_fastpaths = true)
        : enable_pic_(enable_pic), enable_inlined_fastpaths_(enable_inlined_fastpaths) {}

    [[nodiscard]] bool enable_pic() const noexcept { return enable_pic_; }
    void set_enable_pic(bool enable) noexcept { enable_pic_ = enable; }

    [[nodiscard]] bool enable_inlined_fastpaths() const noexcept { return enable_inlined_fastpaths_; }
    void set_enable_inlined_fastpaths(bool enable) noexcept { enable_inlined_fastpaths_ = enable; }

    [[nodiscard]] const std::string& key_map_sym() const noexcept { return key_map_sym_; }
    void set_key_map_sym(std::string sym) { key_map_sym_ = std::move(sym); }

    // The module's inline-cache table: `site_count` sites of
    // kBronzeIcSiteSize bytes at the data symbol `sym`. Zero sites (the
    // default) means no table, and every site pointer stays null.
    void set_ic_table(std::string sym, uint32_t site_count) {
        ic_table_sym_ = std::move(sym);
        ic_site_count_ = site_count;
    }
    [[nodiscard]] uint32_t ic_site_count() const noexcept { return ic_site_count_; }

    // The address of site `ic_index`'s way 0 — what the bronze helpers take
    // as their BRONZE_ABI_MU64 operand — or null when the index names no site
    // in this module's table (kNoIcIndex, or a module compiled without one).
    Value* ic_site(Builder& b, uint32_t ic_index);

    Value* lower_prop_get(
        Builder& b,
        Value* obj,
        std::string_view prop_name,
        uint32_t symbol_id,
        uint32_t site_id,
        Value* ic_entry = nullptr
    );

    // A bronze property read with the monomorphic inline hit in front of the
    // helper: the receiver's shape word against the site's way 0 and its
    // slot word below the inline-slot count (bronze_abi.h,
    // BRONZE_ABI_IC_SLOTWORD_OFFSET), one slot load on a hit and
    // `bronze_prop_get` — with `emit_exception_check` run on that arm — on a
    // miss. `key_index` is the module's key constant. Requires a real site.
    Value* lower_prop_get_mono(
        Builder& b,
        Value* obj,
        uint32_t key_index,
        Value* ic_entry,
        const std::function<void()>& emit_exception_check
    );

    Value* lower_prop_get(
        Builder& b,
        Value* obj,
        uint32_t slot_idx
    );

    Value* lower_prop_get_slot(
        Builder& b,
        Value* obj,
        uint32_t slot_idx
    );

    Value* lower_prop_get_guarded(
        Builder& b,
        Value* obj,
        const void* expected_shape,
        uint32_t slot_idx,
        std::string_view prop_name = "",
        uint32_t symbol_id = 0,
        uint32_t site_id = 0
    );

    void lower_prop_set(
        Builder& b,
        Value* obj,
        std::string_view prop_name,
        uint32_t symbol_id,
        Value* val,
        uint32_t slot_idx,
        uint32_t imm,
        uint32_t site_id,
        Value* ic_entry = nullptr
    );

    void lower_prop_set(
        Builder& b,
        Value* obj,
        uint32_t slot_idx,
        Value* val
    );

    void lower_prop_set_slot(
        Builder& b,
        Value* obj,
        uint32_t slot_idx,
        Value* val
    );

    void lower_prop_set_guarded(
        Builder& b,
        Value* obj,
        const void* expected_shape,
        uint32_t slot_idx,
        Value* val,
        std::string_view prop_name = "",
        uint32_t symbol_id = 0,
        uint32_t site_id = 0
    );

    Value* lower_elem_get(
        Builder& b,
        Value* obj,
        Value* index
    );

    void lower_elem_set(
        Builder& b,
        Value* obj,
        Value* index,
        Value* val,
        uint32_t ic_slot
    );

    void lower_method_def(
        Builder& b,
        Value* obj,
        std::string_view prop_name,
        uint32_t symbol_id,
        Value* closure
    );

private:
    bool enable_pic_ = true;
    bool enable_inlined_fastpaths_ = true;
    uint32_t next_auto_site_id_ = 1000;
    std::string key_map_sym_ = "__bronze_key_map";
    std::string ic_table_sym_ = "__bronze_ic_table";
    uint32_t ic_site_count_ = 0;
};

} // namespace brass::il
