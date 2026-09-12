#pragma once

#include <brass/mir/builder.hpp>
#include <brass/mir/instruction.hpp>
#include <cstdint>

namespace brass::il {

class AllocLoweringHelper {
public:
    explicit AllocLoweringHelper(bool enable_tlab = true)
        : enable_tlab_(enable_tlab) {}

    [[nodiscard]] bool enable_tlab() const noexcept { return enable_tlab_; }
    void set_enable_tlab(bool enable) noexcept { enable_tlab_ = enable; }

    Value* lower_create_object(Builder& b);
    Value* lower_create_array(Builder& b, Value* size_val, uint32_t param_count);
    Value* lower_env_create(Builder& b, Value* parent_val, Value* size_val, uint32_t param_count);

private:
    bool enable_tlab_ = true;
};

} // namespace brass::il
