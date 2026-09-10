#include <brass/runtime/tiering.hpp>
#include <ostream>
#include <iomanip>

namespace brass::runtime {

std::string_view to_string(TierLevel tier) noexcept {
    switch (tier) {
        case TierLevel::Tier0_Interpreter: return "Tier0_Interpreter";
        case TierLevel::Tier1_Baseline:    return "Tier1_Baseline";
        case TierLevel::Tier2_Optimized:   return "Tier2_Optimized";
    }
    return "Unknown";
}

std::ostream& operator<<(std::ostream& os, TierLevel tier) {
    return os << to_string(tier);
}

TieringFeedback::TieringFeedback(const TieringConfig& config)
    : config_(config) {}

TieringFeedback::TieringFeedback(std::string_view fn_name, const TieringConfig& config)
    : fn_name_(fn_name), config_(config) {}

uint64_t TieringFeedback::loop_backedges(uint32_t loop_header_id) const noexcept {
    auto it = loop_backedges_.find(loop_header_id);
    if (it != loop_backedges_.end()) {
        return it->second;
    }
    return 0;
}

uint64_t TieringFeedback::record_backedge(uint32_t loop_header_id) noexcept {
    total_backedges_++;
    return ++loop_backedges_[loop_header_id];
}

uint32_t TieringFeedback::guard_deopt_count(uint32_t guard_site_id) const noexcept {
    auto it = guard_failures_.find(guard_site_id);
    if (it != guard_failures_.end()) {
        return it->second;
    }
    return 0;
}

void TieringFeedback::record_deopt(uint32_t guard_site_id) noexcept {
    deopt_count_++;
    guard_failures_[guard_site_id]++;
    if (deopt_count_ >= config_.deopt_threshold * 2) {
        trigger_bailout("Excessive deoptimizations (" + std::to_string(deopt_count_) + ")");
    } else if (deopt_count_ >= config_.deopt_threshold) {
        if (tier_ == TierLevel::Tier2_Optimized) {
            tier_ = TierLevel::Tier1_Baseline;
        } else if (tier_ == TierLevel::Tier1_Baseline) {
            tier_ = TierLevel::Tier0_Interpreter;
        }
    }
}

bool TieringFeedback::is_speculation_invalid(uint32_t guard_site_id) const noexcept {
    if (guard_deopt_count(guard_site_id) >= config_.deopt_threshold) {
        return true;
    }
    if (deopt_count_ >= config_.deopt_threshold * 4) {
        return true;
    }
    return false;
}

void TieringFeedback::trigger_bailout(std::string_view reason) {
    bailout_triggered_ = true;
    last_bailout_reason_ = std::string(reason);
}

void TieringFeedback::reset() noexcept {
    invocations_ = 0;
    total_backedges_ = 0;
    deopt_count_ = 0;
    loop_backedges_.clear();
    guard_failures_.clear();
    bailout_triggered_ = false;
    last_bailout_reason_.clear();
    tier_ = TierLevel::Tier0_Interpreter;
}

bool TieringFeedback::should_tier_up() const noexcept {
    if (bailout_triggered_) return false;
    if (tier_ == TierLevel::Tier0_Interpreter && invocations_ >= config_.invocation_tier1_threshold) {
        return true;
    }
    if (tier_ == TierLevel::Tier1_Baseline && invocations_ >= config_.invocation_tier2_threshold) {
        return true;
    }
    return false;
}

bool TieringFeedback::should_osr(uint32_t loop_header_id) const noexcept {
    if (!config_.enable_osr || bailout_triggered_) return false;
    return loop_backedges(loop_header_id) >= config_.backedge_osr_threshold;
}

TieringRegistry& TieringRegistry::instance() {
    static TieringRegistry registry;
    return registry;
}

TieringFeedback& TieringRegistry::get_feedback(std::string_view fn_name) {
    std::string key(fn_name);
    auto it = feedback_map_.find(key);
    if (it == feedback_map_.end()) {
        auto [inserted, _] = feedback_map_.emplace(key, TieringFeedback(fn_name, config_));
        return inserted->second;
    }
    return it->second;
}

const TieringFeedback* TieringRegistry::find_feedback(std::string_view fn_name) const {
    auto it = feedback_map_.find(std::string(fn_name));
    if (it != feedback_map_.end()) {
        return &it->second;
    }
    return nullptr;
}

void TieringRegistry::clear() {
    feedback_map_.clear();
}

void TieringRegistry::dump_stats(std::ostream& os) const {
    os << "=== Tiering Feedback Statistics ===\n";
    if (feedback_map_.empty()) {
        os << "  (No function profiles recorded)\n";
        return;
    }
    for (const auto& [fn_name, fb] : feedback_map_) {
        os << "Function: " << fn_name << "\n";
        os << "  Current Tier:      " << fb.current_tier() << "\n";
        os << "  Invocations:       " << fb.invocation_count() << " (Threshold: "
           << fb.invocation_tier1_threshold() << ")\n";
        os << "  Total Backedges:   " << fb.backedge_count() << " (OSR Threshold: "
           << fb.backedge_osr_threshold() << ")\n";
        os << "  Deoptimizations:   " << fb.deopt_count() << " (Threshold: "
           << fb.deopt_threshold() << ")\n";
        if (fb.is_bailout_set()) {
            os << "  Bail-out triggered: " << fb.last_bailout_reason() << "\n";
        }
        if (!fb.loop_backedges_map().empty()) {
            os << "  Loop Headers:\n";
            for (const auto& [loop_id, cnt] : fb.loop_backedges_map()) {
                os << "    [Loop Block " << loop_id << "]: " << cnt << " backedges\n";
            }
        }
        if (!fb.guard_failures_map().empty()) {
            os << "  Failing Guards:\n";
            for (const auto& [gid, cnt] : fb.guard_failures_map()) {
                os << "    [Guard Site " << gid << "]: " << cnt << " deopts"
                   << (fb.is_speculation_invalid(gid) ? " (SPECULATION INVALID)" : "") << "\n";
            }
        }
    }
    os << "====================================\n";
}

} // namespace brass::runtime
