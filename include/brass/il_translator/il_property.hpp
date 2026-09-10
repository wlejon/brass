#pragma once

#include <brass/mir/builder.hpp>
#include <brass/mir/instruction.hpp>
#include <string_view>
#include <cstdint>

namespace brass::il {

class PropertyLoweringHelper {
public:
    explicit PropertyLoweringHelper(bool enable_pic = true)
        : enable_pic_(enable_pic) {}

    [[nodiscard]] bool enable_pic() const noexcept { return enable_pic_; }
    void set_enable_pic(bool enable) noexcept { enable_pic_ = enable; }

    Value* lower_prop_get(
        Builder& b,
        Value* obj,
        std::string_view prop_name,
        uint32_t symbol_id,
        uint32_t site_id
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
    uint32_t next_auto_site_id_ = 1000;
};

} // namespace brass::il
