// Functions the baseline tier rejects: why it rejected each, and their way
// to Tier 2 without tier-1 code. A rejected function stays interpreted only
// until it is as hot as tier-1 code is when it asks for Tier 2; it is then
// compiled to Tier 2 straight from Tier 0, from the feedback Tier 0
// collected, and a failed guard of that code finishes the call back in
// Tier 0 (tier_deopt.cpp), from where it may tier up again.
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/code_installer.hpp>
#include <algorithm>

namespace brass::runtime {

bool MultiTierPipeline::record_rejection_locked(std::string_view name, std::string reason) {
    if (reason.empty()) reason = "rejected by the baseline tier";
    return baseline_rejected_.emplace(std::string(name), std::move(reason)).second;
}

void MultiTierPipeline::note_rejected(std::string_view name) {
    stats_.tier1_rejections.fetch_add(1, std::memory_order_relaxed);
    tiering().get_feedback(name).mark_tier1_rejected();
}

std::string MultiTierPipeline::baseline_rejection(std::string_view fn_name) const {
    std::lock_guard<std::mutex> lock(compiling_mutex_);
    auto it = baseline_rejected_.find(std::string(fn_name));
    return it != baseline_rejected_.end() ? it->second : std::string();
}

std::vector<BaselineRejection> MultiTierPipeline::baseline_rejections() const {
    std::vector<BaselineRejection> out;
    {
        std::lock_guard<std::mutex> lock(compiling_mutex_);
        out.reserve(baseline_rejected_.size());
        for (const auto& [name, reason] : baseline_rejected_) out.push_back({name, reason});
    }
    std::sort(out.begin(), out.end(),
              [](const BaselineRejection& a, const BaselineRejection& b) { return a.name < b.name; });
    return out;
}

bool MultiTierPipeline::promote_tier0(TieringFeedback& fb) {
    const std::string_view fn_name = fb.function_name();
    if (!initialized_ || config_.max_tier < TierLevel::Tier2_Optimized || fb.is_bailout_set()) return true;
    if (!is_baseline_rejected(fn_name)) {
        // The rejections were forgotten (clear_baseline_cache): Tier 1 is
        // tried again first.
        fb.clear_tier1_rejected();
        return false;
    }
    FunctionHandle* handle = table_->find(fn_name);
    // Native code installed some other way, or tier 2 rejected the function
    // already: it stays where it is.
    if (!handle || handle->has_native_entry() || handle->tier2_rejected()) return true;
    const bool background = config_.enable_background_compile || tiering().is_background_compile_enabled();
    const bool requested = background ? enqueue_tier2(fn_name, nullptr, handle) : compile_tier2_now(fn_name, handle);
    if (requested) stats_.tier0_promotions.fetch_add(1, std::memory_order_relaxed);
    // Checked again until the code is installed (which ends the checks) or
    // tier 2 rejects the function: a compile that finished with neither (a
    // function its code calls still compiling) is asked for again, and a
    // request already queued is not queued twice.
    return handle->tier2_rejected() || handle->has_native_entry();
}

} // namespace brass::runtime
