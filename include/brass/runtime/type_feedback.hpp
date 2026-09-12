#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <iosfwd>

namespace brass::runtime {

class Shape;

enum class FeedbackSlotKind : uint8_t {
    Call,
    Property,
    BinaryOp
};

std::string_view to_string(FeedbackSlotKind kind) noexcept;
std::ostream& operator<<(std::ostream& os, FeedbackSlotKind kind);

struct CallFeedback {
    uintptr_t target_addr = 0;
    std::string target_name;
    uint64_t count = 0;
};

struct FeedbackSlot {
    FeedbackSlotKind kind = FeedbackSlotKind::Call;
    uint32_t site_id = 0;
    uint64_t total_invocations = 0;

    // For calls
    std::vector<CallFeedback> targets;

    [[nodiscard]] bool is_monomorphic() const noexcept {
        return targets.size() == 1;
    }

    [[nodiscard]] bool is_polymorphic() const noexcept {
        return targets.size() > 1 && targets.size() <= 4;
    }

    [[nodiscard]] bool is_megamorphic() const noexcept {
        return targets.size() > 4;
    }

    [[nodiscard]] const CallFeedback* get_monomorphic_target() const noexcept {
        return is_monomorphic() ? &targets[0] : nullptr;
    }

    // For properties
    std::vector<Shape*> observed_shapes;
    uint32_t slot_index = 0;

    [[nodiscard]] bool is_property_monomorphic() const noexcept {
        return observed_shapes.size() == 1;
    }

    [[nodiscard]] bool is_property_polymorphic() const noexcept {
        return observed_shapes.size() > 1 && observed_shapes.size() <= 4;
    }

    [[nodiscard]] bool is_property_megamorphic() const noexcept {
        return observed_shapes.size() > 4;
    }

    [[nodiscard]] const Shape* get_monomorphic_shape() const noexcept {
        return observed_shapes.size() == 1 ? observed_shapes[0] : nullptr;
    }
};

class TypeFeedbackVector {
public:
    TypeFeedbackVector() = default;
    explicit TypeFeedbackVector(std::string_view function_name)
        : function_name_(function_name) {}

    [[nodiscard]] std::string_view function_name() const noexcept { return function_name_; }
    void set_function_name(std::string_view name) { function_name_ = std::string(name); }

    [[nodiscard]] const std::vector<FeedbackSlot>& slots() const noexcept { return slots_; }
    [[nodiscard]] std::vector<FeedbackSlot>& slots() noexcept { return slots_; }

    [[nodiscard]] FeedbackSlot* find_slot(uint32_t site_id) noexcept;
    [[nodiscard]] const FeedbackSlot* find_slot(uint32_t site_id) const noexcept;

    FeedbackSlot& get_or_create_slot(uint32_t site_id, FeedbackSlotKind kind);

    void record_call_target(uint32_t site_id, uintptr_t target_addr, std::string_view target_name);
    void record_property_shape(uint32_t site_id, Shape* shape, uint32_t slot_idx);

    [[nodiscard]] size_t slot_count() const noexcept { return slots_.size(); }
    void clear() noexcept { slots_.clear(); }

    void dump_stats(std::ostream& os) const;

private:
    std::string function_name_;
    std::vector<FeedbackSlot> slots_;
};

class FeedbackRegistry {
public:
    static FeedbackRegistry& instance();

    TypeFeedbackVector& get_or_create(std::string_view fn_name);
    [[nodiscard]] const TypeFeedbackVector* find(std::string_view fn_name) const;
    [[nodiscard]] TypeFeedbackVector* find(std::string_view fn_name);
    [[nodiscard]] bool has(std::string_view fn_name) const;
    void clear();

    [[nodiscard]] const std::unordered_map<std::string, TypeFeedbackVector>& all_tfvs() const noexcept {
        return tfv_map_;
    }

    void dump_stats(std::ostream& os) const;

private:
    FeedbackRegistry() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, TypeFeedbackVector> tfv_map_;
};

} // namespace brass::runtime

extern "C" {
void brass_record_call_feedback(const char* fn_name, uint32_t site_id, uintptr_t target_addr, const char* target_name);
void brass_record_property_feedback(const char* fn_name, uint32_t site_id, brass::runtime::Shape* shape, uint32_t slot_idx);
}
