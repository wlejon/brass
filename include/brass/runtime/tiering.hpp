#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <iosfwd>
#include <memory>

namespace brass {
class Module;
}

namespace brass::runtime {

class TypeFeedbackVector;

enum class TierLevel : uint8_t {
    Tier0_Interpreter = 0,
    Tier1_Baseline = 1,
    Tier2_Optimized = 2
};

std::string_view to_string(TierLevel tier) noexcept;
std::ostream& operator<<(std::ostream& os, TierLevel tier);

// Standard default threshold constants
inline constexpr uint64_t INVOCATION_TIER1_THRESHOLD = 50;
inline constexpr uint64_t INVOCATION_TIER2_THRESHOLD = 200;
inline constexpr uint64_t BACKEDGE_OSR_THRESHOLD = 100;
inline constexpr uint64_t DEOPT_THRESHOLD = 5;

struct TieringConfig {
    uint64_t invocation_tier1_threshold = INVOCATION_TIER1_THRESHOLD;
    uint64_t invocation_tier2_threshold = INVOCATION_TIER2_THRESHOLD;
    uint64_t backedge_osr_threshold = BACKEDGE_OSR_THRESHOLD;
    uint64_t deopt_threshold = DEOPT_THRESHOLD;
    bool enable_osr = true;
    bool enable_background_compile = false;
    size_t jit_threads = 2;
};

class FunctionHandle;

class TieringFeedback {
public:
    explicit TieringFeedback(const TieringConfig& config = TieringConfig{});
    explicit TieringFeedback(std::string_view fn_name, const TieringConfig& config = TieringConfig{});

    std::string_view function_name() const noexcept { return fn_name_; }

    // Invocations
    uint64_t invocation_count() const noexcept { return invocations_; }
    uint64_t record_invocation() noexcept;

    // Type feedback vector
    TypeFeedbackVector* type_feedback_vector();
    const TypeFeedbackVector* type_feedback_vector() const;

    // Loop Backedges
    uint64_t backedge_count() const noexcept { return total_backedges_; }
    uint64_t loop_backedges(uint32_t loop_header_id) const noexcept;
    uint64_t record_backedge(uint32_t loop_header_id = 0) noexcept;

    // Deoptimizations & Guard failure tracker (ratchet)
    uint64_t deopt_count() const noexcept { return deopt_count_; }
    uint32_t guard_deopt_count(uint32_t guard_site_id) const noexcept;
    void record_deopt(uint32_t guard_site_id = 0) noexcept;
    void record_deoptimization(uint32_t guard_site_id = 0) noexcept { record_deopt(guard_site_id); }
    bool is_speculation_invalid(uint32_t guard_site_id) const noexcept;

    // Bail-out tracker
    bool is_bailout_set() const noexcept { return bailout_triggered_; }
    bool is_bailed_out() const noexcept { return bailout_triggered_; }
    void trigger_bailout(std::string_view reason);
    void record_bailout(std::string_view reason) { trigger_bailout(reason); }
    std::string_view last_bailout_reason() const noexcept { return last_bailout_reason_; }
    std::string_view bailout_reason() const noexcept { return last_bailout_reason_; }
    void clear_bailout() noexcept {
        bailout_triggered_ = false;
        last_bailout_reason_.clear();
    }

    // Tier state machine
    TierLevel current_tier() const noexcept { return tier_; }
    TierLevel tier_level() const noexcept { return tier_; }
    void set_tier(TierLevel t) noexcept { tier_ = t; }
    void set_tier_level(TierLevel t) noexcept { tier_ = t; }

    bool should_tier_up() const noexcept;
    bool should_tier_up(uint64_t custom_threshold) const noexcept {
        if (bailout_triggered_) return false;
        return invocations_ >= custom_threshold;
    }

    bool should_osr(uint32_t loop_header_id) const noexcept;
    bool should_trigger_osr(uint64_t custom_threshold, uint32_t loop_header_id = 0) const noexcept {
        if (!config_.enable_osr || bailout_triggered_) return false;
        return loop_backedges(loop_header_id) >= custom_threshold || total_backedges_ >= custom_threshold;
    }

    void reset() noexcept;

    // Configurable thresholds
    uint64_t invocation_tier1_threshold() const noexcept { return config_.invocation_tier1_threshold; }
    uint64_t backedge_osr_threshold() const noexcept { return config_.backedge_osr_threshold; }
    uint64_t deopt_threshold() const noexcept { return config_.deopt_threshold; }

    void set_invocation_tier1_threshold(uint64_t val) noexcept { config_.invocation_tier1_threshold = val; }
    void set_backedge_osr_threshold(uint64_t val) noexcept { config_.backedge_osr_threshold = val; }
    void set_deopt_threshold(uint64_t val) noexcept { config_.deopt_threshold = val; }

    const TieringConfig& config() const noexcept { return config_; }
    void set_config(const TieringConfig& c) noexcept { config_ = c; }

    // Direct access to maps for introspection
    const std::unordered_map<uint32_t, uint64_t>& loop_backedges_map() const noexcept { return loop_backedges_; }
    const std::unordered_map<uint32_t, uint32_t>& guard_failures_map() const noexcept { return guard_failures_; }

private:
    std::string fn_name_;
    TieringConfig config_;
    TierLevel tier_ = TierLevel::Tier0_Interpreter;
    uint64_t invocations_ = 0;
    uint64_t total_backedges_ = 0;
    uint64_t deopt_count_ = 0;
    std::unordered_map<uint32_t, uint64_t> loop_backedges_;
    std::unordered_map<uint32_t, uint32_t> guard_failures_;
    bool bailout_triggered_ = false;
    std::string last_bailout_reason_;
};

class TieringRegistry {
public:
    static TieringRegistry& instance();

    TieringFeedback& get_feedback(std::string_view fn_name);
    TieringFeedback& get_or_create(std::string_view fn_name) { return get_feedback(fn_name); }
    const TieringFeedback* find_feedback(std::string_view fn_name) const;
    bool has(std::string_view fn_name) const noexcept { return find_feedback(fn_name) != nullptr; }
    void clear();

    const TieringConfig& default_config() const noexcept { return config_; }
    TieringConfig& default_config() noexcept { return config_; }
    void set_default_config(const TieringConfig& config) noexcept { config_ = config; }

    bool is_background_compile_enabled() const noexcept { return config_.enable_background_compile; }
    void set_background_compile_enabled(bool enabled) noexcept { config_.enable_background_compile = enabled; }

    size_t jit_threads() const noexcept { return config_.jit_threads; }
    void set_jit_threads(size_t threads) noexcept { config_.jit_threads = threads; }

    void set_active_module(const Module* mod) noexcept { active_module_ = mod; }
    const Module* active_module() const noexcept { return active_module_; }

    bool on_invocation_threshold_reached(std::string_view fn_name);
    bool enqueue_compilation(
        std::string_view fn_name,
        const Module* mod = nullptr,
        FunctionHandle* handle = nullptr
    );

    void dump_stats(std::ostream& os) const;

private:
    TieringRegistry() = default;
    TieringConfig config_;
    const Module* active_module_ = nullptr;
    std::unordered_map<std::string, TieringFeedback> feedback_map_;
};

} // namespace brass::runtime
