#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <iosfwd>
#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace brass {
class Module;
}

namespace brass::runtime {

class TypeFeedbackVector;
class FeedbackRegistry;

enum class TierLevel : uint8_t {
    Tier0_Interpreter = 0,
    Tier1_Baseline = 1,
    Tier2_Optimized = 2
};

std::string_view to_string(TierLevel tier) noexcept;
std::ostream& operator<<(std::ostream& os, TierLevel tier);

enum class Tier0Interpreter : uint8_t {
    Oracle = 0,
    Fast = 1,
    FastInterpreter = 1
};
using Tier0Engine = Tier0Interpreter;

std::string_view to_string(Tier0Interpreter kind) noexcept;
std::ostream& operator<<(std::ostream& os, Tier0Interpreter kind);

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
    Tier0Interpreter tier0_interpreter = Tier0Interpreter::Oracle;
    // The highest tier the pipeline compiles a program function to. At
    // Tier0_Interpreter nothing of the program is compiled: every function
    // stays interpreted, and native code calling one through its address
    // reaches it through a native-to-Tier-0 bridge. At Tier1_Baseline no
    // function is optimized.
    TierLevel max_tier = TierLevel::Tier2_Optimized;

    bool use_fast_interpreter() const noexcept {
        return tier0_interpreter == Tier0Interpreter::Fast;
    }
    void set_tier0_interpreter(Tier0Interpreter kind) noexcept {
        tier0_interpreter = kind;
    }
    void set_use_fast_interpreter(bool enable) noexcept {
        tier0_interpreter = enable ? Tier0Interpreter::Fast : Tier0Interpreter::Oracle;
    }
};

class FunctionHandle;
class FunctionDispatchTable;
class MultiTierPipeline;
class TieringRegistry;

// The tiering registry of `table`'s program; null is the default program
// (TieringRegistry::instance()).
TieringRegistry& tiering_of(FunctionDispatchTable* table) noexcept;

// Bumped whenever a runtime registry drops or replaces entries that callers
// may have cached pointers to (FunctionDispatchTable handles, TieringRegistry
// feedback). A cache that recorded the generation it resolved under is
// stale once this differs. Entries are retired, never freed, so a stale
// pointer stays safe to read until it is re-resolved.
uint64_t registry_generation() noexcept;
void bump_registry_generation() noexcept;

// Per-function tiering counters. Invocation and backedge counting is
// lock-free: callers resolve the TieringFeedback once (by name, under the
// registry lock) and then count through the pointer.
//
// A feedback knows the registry (the program) it belongs to, so code that
// holds only the pointer (tier-1 code bakes it in) reaches that program's
// pipeline without a by-name lookup. A free-standing feedback (no registry)
// belongs to the default program.
class TieringFeedback {
public:
    explicit TieringFeedback(const TieringConfig& config = TieringConfig{});
    explicit TieringFeedback(std::string_view fn_name, const TieringConfig& config = TieringConfig{},
                             TieringRegistry* registry = nullptr);

    TieringFeedback(const TieringFeedback&) = delete;
    TieringFeedback& operator=(const TieringFeedback&) = delete;

    std::string_view function_name() const noexcept { return fn_name_; }
    TieringRegistry& registry() const noexcept;

    // Invocations (exact under concurrency). Fires the tier-up hook when the
    // count reaches the Tier 1 threshold; an attempt that cannot finish yet
    // (a callee compiling on another thread) is retried with exponential
    // backoff, at no cost to the calls in between beyond one compare.
    uint64_t invocation_count() const noexcept { return invocations_.load(std::memory_order_relaxed); }
    uint64_t record_invocation() noexcept;
    // Tier 1 attempts made after the first (introspection).
    uint32_t tier1_retries() const noexcept { return tier1_retries_.load(std::memory_order_relaxed); }

    // Type feedback vector
    TypeFeedbackVector* type_feedback_vector();
    const TypeFeedbackVector* type_feedback_vector() const;

    // Loop Backedges
    uint64_t backedge_count() const noexcept { return total_backedges_.load(std::memory_order_relaxed); }
    uint64_t loop_backedges(uint32_t loop_header_id) const noexcept;
    uint64_t record_backedge(uint32_t loop_header_id = 0) noexcept;
    // The interpreter's per-iteration count of backedge_count(): a relaxed
    // load and store rather than a locked read-modify-write, so concurrent
    // runs of one function may lose a few counts (it is a hotness
    // heuristic, not an exact total).
    void count_backedge_fast() noexcept {
        total_backedges_.store(total_backedges_.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
    }

    // Deoptimizations & Guard failure tracker (ratchet)
    uint64_t deopt_count() const noexcept { return deopt_count_.load(std::memory_order_relaxed); }
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
    TierLevel current_tier() const noexcept { return tier_.load(std::memory_order_acquire); }
    TierLevel tier_level() const noexcept { return current_tier(); }
    void set_tier(TierLevel t) noexcept { tier_.store(t, std::memory_order_release); }
    void set_tier_level(TierLevel t) noexcept { set_tier(t); }

    bool should_tier_up() const noexcept;
    bool should_tier_up(uint64_t custom_threshold) const noexcept {
        if (bailout_triggered_) return false;
        return invocation_count() >= custom_threshold;
    }

    bool should_osr(uint32_t loop_header_id) const noexcept;
    bool should_trigger_osr(uint64_t custom_threshold, uint32_t loop_header_id = 0) const noexcept {
        if (!config_.enable_osr || bailout_triggered_) return false;
        return loop_backedges(loop_header_id) >= custom_threshold || backedge_count() >= custom_threshold;
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

    // Snapshots for introspection (backedges per loop header id; id 0 is
    // the unkeyed count).
    std::unordered_map<uint32_t, uint64_t> loop_backedges_map() const;
    std::unordered_map<uint32_t, uint32_t> guard_failures_map() const;

private:
    void attempt_tier1(uint64_t n) noexcept;

    std::string fn_name_;
    TieringRegistry* registry_ = nullptr;
    TieringConfig config_;
    std::atomic<TierLevel> tier_{TierLevel::Tier0_Interpreter};
    std::atomic<uint64_t> invocations_{0};
    // The invocation count at which a Tier 1 attempt that could not finish
    // is retried (UINT64_MAX: none wanted), and how many retries were made.
    std::atomic<uint64_t> tier1_retry_at_{UINT64_MAX};
    std::atomic<uint32_t> tier1_retries_{0};
    std::atomic<uint64_t> total_backedges_{0};
    std::atomic<uint64_t> unkeyed_backedges_{0};
    std::atomic<uint64_t> deopt_count_{0};
    // Keyed backedges and guard failures are rare (OSR and deopt paths).
    mutable std::mutex maps_mutex_;
    std::unordered_map<uint32_t, uint64_t> loop_backedges_;
    std::unordered_map<uint32_t, uint32_t> guard_failures_;
    bool bailout_triggered_ = false;
    std::string last_bailout_reason_;
};

// The tiering feedback of one program, keyed by function name within it.
//
// instance() is the default program's; an owned FunctionDispatchTable owns
// its own (FunctionDispatchTable::tiering()), so same-named functions of two
// programs count invocations, deopts and bailouts independently. Threshold
// crossings go to the program's pipeline (dispatch_table().pipeline()).
class TieringRegistry {
public:
    static TieringRegistry& instance();
    // The registry of an owned program; `table` must outlive it (the table
    // owns it).
    explicit TieringRegistry(FunctionDispatchTable& table);

    TieringRegistry(const TieringRegistry&) = delete;
    TieringRegistry& operator=(const TieringRegistry&) = delete;

    bool is_default() const noexcept { return table_ == nullptr; }
    FunctionDispatchTable& dispatch_table() const noexcept;
    MultiTierPipeline& pipeline() const noexcept;
    // The program's type feedback (call targets, property shapes):
    // FeedbackRegistry::instance() for the default program, else owned here.
    FeedbackRegistry& type_feedback() const noexcept;

    TieringFeedback& get_feedback(std::string_view fn_name);
    TieringFeedback& get_or_create(std::string_view fn_name) { return get_feedback(fn_name); }
    const TieringFeedback* find_feedback(std::string_view fn_name) const;
    bool has(std::string_view fn_name) const noexcept { return find_feedback(fn_name) != nullptr; }
    void clear();

    const TieringConfig& default_config() const noexcept { return config_; }
    TieringConfig& default_config() noexcept { return config_; }
    void set_default_config(const TieringConfig& config) noexcept { config_ = config; }

    Tier0Interpreter tier0_interpreter() const noexcept { return config_.tier0_interpreter; }
    void set_tier0_interpreter(Tier0Interpreter kind) noexcept { config_.tier0_interpreter = kind; }
    bool use_fast_interpreter() const noexcept { return config_.use_fast_interpreter(); }
    void set_use_fast_interpreter(bool enable) noexcept { config_.set_use_fast_interpreter(enable); }

    bool is_background_compile_enabled() const noexcept { return config_.enable_background_compile; }
    void set_background_compile_enabled(bool enabled) noexcept { config_.enable_background_compile = enabled; }

    size_t jit_threads() const noexcept { return config_.jit_threads; }
    void set_jit_threads(size_t threads) noexcept { config_.jit_threads = threads; }

    void set_active_module(const Module* mod) noexcept { active_module_ = mod; }
    const Module* active_module() const noexcept { return active_module_; }
    // The module tier 2 compiles `handle`'s function from: the one owning
    // the Function it is bound to (a host may bind handles to a module other
    // than the active one), else the active module.
    const Module* tier2_source_module(const FunctionHandle* handle) const noexcept;

    // Clears the active module if it is `mod` (which is being destroyed).
    // Static and a no-op once the registry is gone, so a Module destroyed
    // during static teardown never touches a dead registry.
    static void forget_module(const Module* mod) noexcept;
    // This registry's part of forget_module.
    void forget(const Module* mod) noexcept;
    ~TieringRegistry();

    bool on_invocation_threshold_reached(std::string_view fn_name);
    // After on_invocation_threshold_reached returned false: whether asking
    // again later can succeed (the pipeline compiles synchronously and has
    // not rejected the function).
    bool tier1_retry_possible(std::string_view fn_name) const;
    bool enqueue_compilation(
        std::string_view fn_name,
        const Module* mod = nullptr,
        FunctionHandle* handle = nullptr
    );

    void dump_stats(std::ostream& os) const;

private:
    TieringRegistry();
    FunctionDispatchTable* const table_ = nullptr; // null: the default program
    std::unique_ptr<FeedbackRegistry> type_feedback_; // owned programs only
    TieringConfig config_;
    std::atomic<const Module*> active_module_{nullptr};
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<TieringFeedback>> feedback_map_;
    // Entries dropped by clear(): kept alive for pointers cached before it.
    std::vector<std::unique_ptr<TieringFeedback>> retired_;
};

} // namespace brass::runtime
