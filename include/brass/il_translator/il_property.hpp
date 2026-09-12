#pragma once

#include <brass/mir/builder.hpp>
#include <brass/mir/instruction.hpp>
#include <string_view>
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

    Value* lower_prop_get(
        Builder& b,
        Value* obj,
        std::string_view prop_name,
        uint32_t symbol_id,
        uint32_t site_id
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
        uint32_t site_id
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
};

} // namespace brass::il
