#pragma once

#include <brass/mir/builder.hpp>
#include <brass/mir/instruction.hpp>
#include <cstdint>

namespace brass::il {

class AllocLoweringHelper {
public:
    enum class Model {
        BrassHostGC,
        BronzeTLS
    };

    explicit AllocLoweringHelper(bool enable_tlab = true, Model model = Model::BrassHostGC)
        : enable_tlab_(enable_tlab), model_(model) {}

    [[nodiscard]] bool enable_tlab() const noexcept { return enable_tlab_; }
    void set_enable_tlab(bool enable) noexcept { enable_tlab_ = enable; }

    [[nodiscard]] Model model() const noexcept { return model_; }
    void set_model(Model m) noexcept { model_ = m; }

    Value* lower_create_object(Builder& b);
    Value* lower_create_array(Builder& b, Value* size_val, uint32_t param_count);
    Value* lower_env_create(Builder& b, Value* parent_val, Value* size_val, uint32_t param_count);

private:
    Value* lower_create_object_brass(Builder& b);
    Value* lower_create_array_brass(Builder& b, Value* size_val, uint32_t param_count);
    Value* lower_env_create_brass(Builder& b, Value* parent_val, Value* size_val, uint32_t param_count);

    Value* lower_create_object_bronze(Builder& b);
    Value* lower_create_array_bronze(Builder& b, Value* size_val, uint32_t param_count);
    Value* lower_env_create_bronze(Builder& b, Value* parent_val, Value* size_val, uint32_t param_count);

    bool enable_tlab_ = true;
    Model model_ = Model::BrassHostGC;
};

} // namespace brass::il
