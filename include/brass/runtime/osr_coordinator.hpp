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
class FastInterpreter;
struct FastFrame;
}

namespace brass::runtime {

// On-stack replacement for one program: its enablement and threshold, its
// compiled OSR stubs and loop analysis (keyed by function name within the
// program), and the TieringRegistry its backedge and deopt counts go to.
//
// instance() is the default program's (counting into
// TieringRegistry::instance()); an owned FunctionDispatchTable owns its own
// (FunctionDispatchTable::osr()), and interpreters use their table's.
class OsrCoordinator {
public:
    static OsrCoordinator& instance();
    // The coordinator of an owned program; `registry` (not the default
    // program's) must outlive it.
    explicit OsrCoordinator(TieringRegistry& registry);
    ~OsrCoordinator();

    OsrCoordinator(const OsrCoordinator&) = delete;
    OsrCoordinator& operator=(const OsrCoordinator&) = delete;

    TieringRegistry& registry() const noexcept;

    // Configuration
    bool is_enabled() const noexcept { return enabled_; }
    void set_enabled(bool enabled) noexcept { enabled_ = enabled; }

    uint64_t threshold() const noexcept { return threshold_; }
    // Also the program's default backedge OSR threshold.
    void set_threshold(uint64_t threshold) noexcept;

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

    bool try_osr_migration(
        FastInterpreter& interp,
        const Function& fn,
        BasicBlock* loop_header,
        FastFrame& frame,
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

    // Stops OSR for a program being destroyed (FunctionDispatchTable, before
    // its handles retire): drops its queued OSR compiles and waits out the
    // running ones, and later backedges migrate nowhere.
    void release_program();
    // Drops this program's queued OSR compiles and waits out the running
    // ones; a loop whose compile was dropped asks again on a later backedge.
    void stop_compiles();
    // Whether OSR here is into the program's own code (a program whose
    // MultiTierPipeline runs it), compiled in the background: a backedge
    // then only polls for the code now and then (FastInterpreter).
    bool runs_program() const noexcept;

    // Statistics
    uint64_t total_osr_migrations() const noexcept { return total_osr_migrations_.load(std::memory_order_relaxed); }
    uint64_t total_native_deopts() const noexcept { return total_native_deopts_.load(std::memory_order_relaxed); }
    void reset_stats() noexcept {
        total_osr_migrations_.store(0, std::memory_order_relaxed);
        total_native_deopts_.store(0, std::memory_order_relaxed);
    }

private:
    OsrCoordinator();

    // OSR in a program whose MultiTierPipeline runs it (osr_program.cpp): a
    // loop header of a function gets an OSR entry function of its own
    // (osr_entry.hpp), compiled with the program's tier-2 passes on the
    // shared CompilePool while the interpreter keeps running the loop; a
    // later backedge to it, once it is ready, carries the frame's live
    // values into it.
    struct ProgramEntry;
    bool try_program_osr(FastInterpreter& interp, const Function& fn, BasicBlock* loop_header, FastFrame& frame,
                         RuntimeValue& out_result);
    void request_program_osr(const Function& fn, const BasicBlock& header, const std::shared_ptr<ProgramEntry>& e);
    void compile_program_osr(const Function& fn, Module& copy, ProgramEntry& e);
    bool enter_program_osr(const Function& fn, FastFrame& frame, const ProgramEntry& e, RuntimeValue& out_result);

    std::mutex program_mutex_;
    bool program_released_ = false;
    std::unordered_map<const BasicBlock*, std::shared_ptr<ProgramEntry>> program_entries_;

    TieringRegistry* const registry_ = nullptr; // null: the default program
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
