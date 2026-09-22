#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/osr.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/interpreter/frame.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/embedding/embedding.hpp>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>

namespace brass {
class Interpreter;
}

namespace brass::runtime {

class OsrCoordinator {
public:
    static OsrCoordinator& instance();

    // Configuration
    bool is_enabled() const noexcept { return enabled_; }
    void set_enabled(bool enabled) noexcept { enabled_ = enabled; }

    uint64_t threshold() const noexcept { return threshold_; }
    void set_threshold(uint64_t threshold) noexcept {
        threshold_ = threshold;
        TieringRegistry::instance().default_config().backedge_osr_threshold = threshold;
    }

    // Check whether an edge (from_bb -> to_bb) is a loop backedge for loop header `to_bb`
    bool is_loop_backedge(const Function& fn, const BasicBlock* from_bb, const BasicBlock* to_bb);

    // Called on a loop backedge.
    // If threshold crossed, compiles OSR stub, packages frame, runs native OSR,
    // and returns true with out_result populated.
    bool try_osr_migration(
        Interpreter& interp,
        const Function& fn,
        BasicBlock* loop_header,
        InterpreterFrame& frame,
        RuntimeValue& out_result
    );

    // Bi-directional deoptimization handler registered with brass::runtime::register_deopt_handler
    void* handle_native_deopt(const DeoptFrame& deopt_frame);

    // Active interpreter registration for current thread / migration session
    void set_active_interpreter(Interpreter* interp) noexcept;
    Interpreter* active_interpreter() const noexcept;

    void set_active_frame(InterpreterFrame* frame) noexcept;
    InterpreterFrame* active_frame() const noexcept;

    void clear_cache();

    // Statistics
    uint64_t total_osr_migrations() const noexcept { return total_osr_migrations_.load(std::memory_order_relaxed); }
    uint64_t total_native_deopts() const noexcept { return total_native_deopts_.load(std::memory_order_relaxed); }
    void reset_stats() noexcept {
        total_osr_migrations_.store(0, std::memory_order_relaxed);
        total_native_deopts_.store(0, std::memory_order_relaxed);
    }

private:
    OsrCoordinator();
    ~OsrCoordinator() = default;

    bool enabled_ = false;
    uint64_t threshold_ = BACKEDGE_OSR_THRESHOLD;
    std::atomic<uint64_t> total_osr_migrations_{0};
    std::atomic<uint64_t> total_native_deopts_{0};

    mutable std::mutex osr_mutex_;
    std::unordered_map<std::string, std::unique_ptr<CompiledModule>> osr_modules_;
    std::unordered_map<std::string, OsrTarget> osr_targets_;
    std::unordered_map<std::string, std::unordered_set<const BasicBlock*>> loop_latches_;
    std::unordered_map<std::string, std::unordered_set<uint64_t>> backedge_pairs_;
};

} // namespace brass::runtime
