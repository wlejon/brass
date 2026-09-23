#include <brass/runtime/tiering.hpp>
#include <brass/runtime/background_compiler.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <atomic>
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

std::string_view to_string(Tier0Interpreter kind) noexcept {
    switch (kind) {
        case Tier0Interpreter::Oracle: return "Oracle";
        case Tier0Interpreter::Fast:   return "FastInterpreter";
    }
    return "Unknown";
}

std::ostream& operator<<(std::ostream& os, Tier0Interpreter kind) {
    return os << to_string(kind);
}

namespace {
std::atomic<uint64_t> g_registry_generation{1};
} // namespace

uint64_t registry_generation() noexcept {
    return g_registry_generation.load(std::memory_order_acquire);
}

void bump_registry_generation() noexcept {
    g_registry_generation.fetch_add(1, std::memory_order_acq_rel);
}

TieringFeedback::TieringFeedback(const TieringConfig& config)
    : config_(config) {}

TieringFeedback::TieringFeedback(std::string_view fn_name, const TieringConfig& config,
                                 TieringRegistry* registry)
    : fn_name_(fn_name), registry_(registry), config_(config) {}

TieringRegistry& TieringFeedback::registry() const noexcept {
    return registry_ ? *registry_ : TieringRegistry::instance();
}

uint64_t TieringFeedback::record_invocation() noexcept {
    const uint64_t n = invocations_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (n == config_.invocation_tier1_threshold) {
        registry().on_invocation_threshold_reached(fn_name_);
    }
    return n;
}

TypeFeedbackVector* TieringFeedback::type_feedback_vector() {
    return &FeedbackRegistry::instance().get_or_create(fn_name_);
}

const TypeFeedbackVector* TieringFeedback::type_feedback_vector() const {
    return FeedbackRegistry::instance().find(fn_name_);
}

uint64_t TieringFeedback::loop_backedges(uint32_t loop_header_id) const noexcept {
    if (loop_header_id == 0) return unkeyed_backedges_.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(maps_mutex_);
    auto it = loop_backedges_.find(loop_header_id);
    return it != loop_backedges_.end() ? it->second : 0;
}

uint64_t TieringFeedback::record_backedge(uint32_t loop_header_id) noexcept {
    total_backedges_.fetch_add(1, std::memory_order_relaxed);
    if (loop_header_id == 0) return unkeyed_backedges_.fetch_add(1, std::memory_order_relaxed) + 1;
    std::lock_guard<std::mutex> lock(maps_mutex_);
    return ++loop_backedges_[loop_header_id];
}

uint32_t TieringFeedback::guard_deopt_count(uint32_t guard_site_id) const noexcept {
    std::lock_guard<std::mutex> lock(maps_mutex_);
    auto it = guard_failures_.find(guard_site_id);
    return it != guard_failures_.end() ? it->second : 0;
}

void TieringFeedback::record_deopt(uint32_t guard_site_id) noexcept {
    const uint64_t n = deopt_count_.fetch_add(1, std::memory_order_relaxed) + 1;
    {
        std::lock_guard<std::mutex> lock(maps_mutex_);
        guard_failures_[guard_site_id]++;
    }
    if (n >= config_.deopt_threshold * 2) {
        trigger_bailout("Excessive deoptimizations (" + std::to_string(n) + ")");
    } else if (n >= config_.deopt_threshold) {
        TierLevel t = current_tier();
        if (t == TierLevel::Tier2_Optimized) {
            set_tier(TierLevel::Tier1_Baseline);
        } else if (t == TierLevel::Tier1_Baseline) {
            set_tier(TierLevel::Tier0_Interpreter);
        }
    }
}

bool TieringFeedback::is_speculation_invalid(uint32_t guard_site_id) const noexcept {
    if (guard_deopt_count(guard_site_id) >= config_.deopt_threshold) {
        return true;
    }
    return deopt_count() >= config_.deopt_threshold * 4;
}

std::unordered_map<uint32_t, uint64_t> TieringFeedback::loop_backedges_map() const {
    std::unordered_map<uint32_t, uint64_t> out;
    {
        std::lock_guard<std::mutex> lock(maps_mutex_);
        out = loop_backedges_;
    }
    if (uint64_t n = unkeyed_backedges_.load(std::memory_order_relaxed)) out[0] = n;
    return out;
}

std::unordered_map<uint32_t, uint32_t> TieringFeedback::guard_failures_map() const {
    std::lock_guard<std::mutex> lock(maps_mutex_);
    return guard_failures_;
}

void TieringFeedback::trigger_bailout(std::string_view reason) {
    bailout_triggered_ = true;
    last_bailout_reason_ = std::string(reason);
}

void TieringFeedback::reset() noexcept {
    invocations_.store(0, std::memory_order_relaxed);
    total_backedges_.store(0, std::memory_order_relaxed);
    unkeyed_backedges_.store(0, std::memory_order_relaxed);
    deopt_count_.store(0, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(maps_mutex_);
        loop_backedges_.clear();
        guard_failures_.clear();
    }
    bailout_triggered_ = false;
    last_bailout_reason_.clear();
    set_tier(TierLevel::Tier0_Interpreter);
}

bool TieringFeedback::should_tier_up() const noexcept {
    if (bailout_triggered_) return false;
    const TierLevel t = current_tier();
    const uint64_t n = invocation_count();
    if (t == TierLevel::Tier0_Interpreter && n >= config_.invocation_tier1_threshold) {
        return true;
    }
    if (t == TierLevel::Tier1_Baseline && n >= config_.invocation_tier2_threshold) {
        return true;
    }
    return false;
}

bool TieringFeedback::should_osr(uint32_t loop_header_id) const noexcept {
    if (!config_.enable_osr || bailout_triggered_) return false;
    return loop_backedges(loop_header_id) >= config_.backedge_osr_threshold;
}

namespace {
// Guards forget_module against a registry never built or already destroyed.
std::atomic<bool> g_tiering_registry_alive{false};
} // namespace

TieringRegistry& TieringRegistry::instance() {
    static TieringRegistry registry;
    g_tiering_registry_alive.store(true, std::memory_order_release);
    return registry;
}

TieringRegistry::TieringRegistry(FunctionDispatchTable& table) : table_(&table) {}

TieringRegistry::~TieringRegistry() {
    if (is_default()) g_tiering_registry_alive.store(false, std::memory_order_release);
}

FunctionDispatchTable& TieringRegistry::dispatch_table() const noexcept {
    return table_ ? *table_ : FunctionDispatchTable::instance();
}

MultiTierPipeline& TieringRegistry::pipeline() const noexcept {
    return dispatch_table().pipeline();
}

void TieringRegistry::forget_module(const Module* mod) noexcept {
    if (!g_tiering_registry_alive.load(std::memory_order_acquire)) return;
    instance().forget(mod);
}

void TieringRegistry::forget(const Module* mod) noexcept {
    const Module* expected = mod;
    active_module_.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
}

TieringFeedback& TieringRegistry::get_feedback(std::string_view fn_name) {
    std::string key(fn_name);
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = feedback_map_.find(key);
    if (it == feedback_map_.end()) {
        auto fb = std::make_unique<TieringFeedback>(fn_name, config_, this);
        auto* ptr = fb.get();
        feedback_map_.emplace(std::move(key), std::move(fb));
        return *ptr;
    }
    return *it->second;
}

const TieringFeedback* TieringRegistry::find_feedback(std::string_view fn_name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = feedback_map_.find(std::string(fn_name));
    if (it != feedback_map_.end()) {
        return it->second.get();
    }
    return nullptr;
}

void TieringRegistry::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Interpreters cache TieringFeedback pointers; retire rather than free
    // them and bump the generation so the caches re-resolve.
    for (auto& [name, fb] : feedback_map_) retired_.push_back(std::move(fb));
    feedback_map_.clear();
    bump_registry_generation();
}

bool TieringRegistry::on_invocation_threshold_reached(std::string_view fn_name) {
    MultiTierPipeline& p = pipeline();
    if (p.is_initialized()) {
        return p.compile_and_install_tier1(fn_name);
    }
    if (!config_.enable_background_compile) {
        return false;
    }
    return enqueue_compilation(fn_name, active_module());
}

bool TieringRegistry::enqueue_compilation(
    std::string_view fn_name,
    const Module* mod,
    FunctionHandle* handle
) {
    const Module* target_mod = mod ? mod : active_module();
    if (!target_mod) {
        return false;
    }
    if (!handle) {
        handle = dispatch_table().get_or_create(fn_name);
    }
    return pipeline().background_compiler().enqueue(
        fn_name,
        *target_mod,
        handle,
        CompilePriority::Normal,
        TierLevel::Tier2_Optimized
    );
}

void TieringRegistry::dump_stats(std::ostream& os) const {
    std::lock_guard<std::mutex> lock(mutex_);
    os << "=== Tiering Feedback Statistics ===\n";
    if (feedback_map_.empty()) {
        os << "  (No function profiles recorded)\n";
        return;
    }
    for (const auto& [fn_name, fb_ptr] : feedback_map_) {
        const auto& fb = *fb_ptr;
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
        const auto loops = fb.loop_backedges_map();
        if (!loops.empty()) {
            os << "  Loop Headers:\n";
            for (const auto& [loop_id, cnt] : loops) {
                os << "    [Loop Block " << loop_id << "]: " << cnt << " backedges\n";
            }
        }
        const auto guards = fb.guard_failures_map();
        if (!guards.empty()) {
            os << "  Failing Guards:\n";
            for (const auto& [gid, cnt] : guards) {
                os << "    [Guard Site " << gid << "]: " << cnt << " deopts"
                   << (fb.is_speculation_invalid(gid) ? " (SPECULATION INVALID)" : "") << "\n";
            }
        }
    }
    os << "====================================\n";
    if (config_.enable_background_compile) {
        pipeline().background_compiler().dump_stats(os);
    }
}

} // namespace brass::runtime
