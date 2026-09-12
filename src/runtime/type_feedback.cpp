#include <brass/runtime/type_feedback.hpp>
#include <brass/runtime/shape.hpp>
#include <iostream>
#include <iomanip>

namespace brass::runtime {

std::string_view to_string(FeedbackSlotKind kind) noexcept {
    switch (kind) {
        case FeedbackSlotKind::Call:     return "Call";
        case FeedbackSlotKind::Property: return "Property";
        case FeedbackSlotKind::BinaryOp: return "BinaryOp";
    }
    return "Unknown";
}

std::ostream& operator<<(std::ostream& os, FeedbackSlotKind kind) {
    return os << to_string(kind);
}

FeedbackSlot* TypeFeedbackVector::find_slot(uint32_t site_id) noexcept {
    for (auto& slot : slots_) {
        if (slot.site_id == site_id) {
            return &slot;
        }
    }
    return nullptr;
}

const FeedbackSlot* TypeFeedbackVector::find_slot(uint32_t site_id) const noexcept {
    for (const auto& slot : slots_) {
        if (slot.site_id == site_id) {
            return &slot;
        }
    }
    return nullptr;
}

FeedbackSlot& TypeFeedbackVector::get_or_create_slot(uint32_t site_id, FeedbackSlotKind kind) {
    FeedbackSlot* existing = find_slot(site_id);
    if (existing) {
        return *existing;
    }
    slots_.push_back(FeedbackSlot{
        kind,
        site_id,
        0,
        {},
        {},
        0
    });
    return slots_.back();
}

void TypeFeedbackVector::record_call_target(
    uint32_t site_id,
    uintptr_t target_addr,
    std::string_view target_name
) {
    FeedbackSlot& slot = get_or_create_slot(site_id, FeedbackSlotKind::Call);
    slot.total_invocations++;

    for (auto& t : slot.targets) {
        bool match = false;
        if (target_addr != 0 && t.target_addr != 0) {
            match = (t.target_addr == target_addr);
        } else if (!target_name.empty() && !t.target_name.empty()) {
            match = (t.target_name == target_name);
        }

        if (match) {
            t.count++;
            if (t.target_addr == 0 && target_addr != 0) {
                t.target_addr = target_addr;
            }
            if (t.target_name.empty() && !target_name.empty()) {
                t.target_name = std::string(target_name);
            }
            return;
        }
    }

    slot.targets.push_back(CallFeedback{
        target_addr,
        std::string(target_name),
        1
    });
}

void TypeFeedbackVector::record_property_shape(
    uint32_t site_id,
    Shape* shape,
    uint32_t slot_idx
) {
    FeedbackSlot& slot = get_or_create_slot(site_id, FeedbackSlotKind::Property);
    slot.total_invocations++;
    slot.slot_index = slot_idx;

    if (shape) {
        for (const auto* s : slot.observed_shapes) {
            if (s == shape) {
                return;
            }
        }
        slot.observed_shapes.push_back(shape);
    }
}

void TypeFeedbackVector::dump_stats(std::ostream& os) const {
    os << "Function: " << function_name_ << " (" << slots_.size() << " slots)\n";
    for (const auto& slot : slots_) {
        os << "  [Site " << slot.site_id << " (" << to_string(slot.kind) << ")]: total="
           << slot.total_invocations;

        if (slot.kind == FeedbackSlotKind::Call) {
            std::string state_str = slot.is_monomorphic() ? "Monomorphic" :
                                    slot.is_polymorphic() ? "Polymorphic" :
                                    slot.is_megamorphic() ? "Megamorphic" : "Uninitialized";
            os << ", state=" << state_str << ", targets=" << slot.targets.size() << "\n";
            for (const auto& t : slot.targets) {
                double pct = slot.total_invocations > 0
                    ? (static_cast<double>(t.count) * 100.0 / static_cast<double>(slot.total_invocations))
                    : 0.0;
                os << "    -> target @" << (t.target_name.empty() ? "<anon>" : t.target_name)
                   << " (0x" << std::hex << t.target_addr << std::dec << "): "
                   << t.count << " calls (" << std::fixed << std::setprecision(1) << pct << "%)\n";
            }
        } else if (slot.kind == FeedbackSlotKind::Property) {
            std::string state_str = slot.is_property_monomorphic() ? "Monomorphic" :
                                    slot.is_property_polymorphic() ? "Polymorphic" :
                                    slot.is_property_megamorphic() ? "Megamorphic" : "Uninitialized";
            os << ", state=" << state_str << ", slot=" << slot.slot_index
               << ", shapes=" << slot.observed_shapes.size() << "\n";
        } else {
            os << "\n";
        }
    }
}

FeedbackRegistry& FeedbackRegistry::instance() {
    static FeedbackRegistry s_instance;
    return s_instance;
}

TypeFeedbackVector& FeedbackRegistry::get_or_create(std::string_view fn_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string key(fn_name);
    auto it = tfv_map_.find(key);
    if (it != tfv_map_.end()) {
        return it->second;
    }
    auto [inserted_it, _] = tfv_map_.emplace(key, TypeFeedbackVector(fn_name));
    return inserted_it->second;
}

const TypeFeedbackVector* FeedbackRegistry::find(std::string_view fn_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tfv_map_.find(std::string(fn_name));
    if (it != tfv_map_.end()) {
        return &it->second;
    }
    return nullptr;
}

TypeFeedbackVector* FeedbackRegistry::find(std::string_view fn_name) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = tfv_map_.find(std::string(fn_name));
    if (it != tfv_map_.end()) {
        return &it->second;
    }
    return nullptr;
}

bool FeedbackRegistry::has(std::string_view fn_name) const {
    return find(fn_name) != nullptr;
}

void FeedbackRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    tfv_map_.clear();
}

void FeedbackRegistry::dump_stats(std::ostream& os) const {
    std::lock_guard<std::mutex> lock(mutex_);
    os << "=== Type Feedback Vector (TFV) Statistics ===\n";
    if (tfv_map_.empty()) {
        os << "  (No type feedback recorded)\n";
        os << "=============================================\n";
        return;
    }
    for (const auto& [name, tfv] : tfv_map_) {
        tfv.dump_stats(os);
    }
    os << "=============================================\n";
}

} // namespace brass::runtime

extern "C" {

void brass_record_call_feedback(
    const char* fn_name,
    uint32_t site_id,
    uintptr_t target_addr,
    const char* target_name
) {
    if (!fn_name) return;
    brass::runtime::FeedbackRegistry::instance()
        .get_or_create(fn_name)
        .record_call_target(site_id, target_addr, target_name ? target_name : "");
}

void brass_record_property_feedback(
    const char* fn_name,
    uint32_t site_id,
    brass::runtime::Shape* shape,
    uint32_t slot_idx
) {
    if (!fn_name) return;
    brass::runtime::FeedbackRegistry::instance()
        .get_or_create(fn_name)
        .record_property_shape(site_id, shape, slot_idx);
}

}
