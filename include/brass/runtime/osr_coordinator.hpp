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
    void set_active_interpreter(Interpreter* interp) noexcept { active_interpreter_ = interp; }
    Interpreter* active_interpreter() const noexcept { return active_interpreter_; }

    void clear_cache();

    // Statistics
    uint64_t total_osr_migrations() const noexcept { return total_osr_migrations_; }
    uint64_t total_native_deopts() const noexcept { return total_native_deopts_; }
    void reset_stats() noexcept {
        total_osr_migrations_ = 0;
        total_native_deopts_ = 0;
    }

private:
    OsrCoordinator();
    ~OsrCoordinator() = default;

    bool enabled_ = false;
    uint64_t threshold_ = BACKEDGE_OSR_THRESHOLD;
    uint64_t total_osr_migrations_ = 0;
    uint64_t total_native_deopts_ = 0;
    Interpreter* active_interpreter_ = nullptr;
    const Function* active_fn_ = nullptr;
    InterpreterFrame* active_frame_ = nullptr;

    std::unordered_map<std::string, std::unique_ptr<CompiledModule>> osr_modules_;
    std::unordered_map<std::string, OsrTarget> osr_targets_;
    std::unordered_map<std::string, std::unordered_set<const BasicBlock*>> loop_latches_;
};

} // namespace brass::runtime
