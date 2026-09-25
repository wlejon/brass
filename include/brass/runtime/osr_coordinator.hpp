#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/runtime/tiering.hpp>
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>

namespace brass {
class FastInterpreter;
struct FastFrame;
class Module;
}

namespace brass::runtime {

// On-stack replacement for one program: its enablement and threshold, and
// the OSR entries of its hot loops.
//
// OSR happens in a program whose MultiTierPipeline runs it, on the
// FastInterpreter the pipeline runs it with. A backedge past the threshold
// to a loop header asks for the header's OSR entry function
// (mir/osr_entry.hpp): planned and copied, with the bodies of what it calls,
// on the interpreter's thread, then optimized with the program's tier-2
// passes and compiled on the shared CompilePool while the interpreter keeps
// running the loop. A later backedge to the header, once the code is ready,
// hands it the frame's live values and returns what it returns. The code
// links to the program as its tier-2 code does: calls go through the
// program's stubs, a failed guard finishes the call in Tier 0, and the
// program's host is told of it under the function's own name, so a stack
// trace shows the loop's frame as the function's. A loop the plan refuses (in
// a function with resume points, or live across it a gcref or a vector)
// stays interpreted. An interpreter running
// outside such a program (the reference Interpreter, or a FastInterpreter
// whose program's pipeline is not initialized) only counts its backedges.
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

    // Whether this program's loops can OSR: its MultiTierPipeline runs it.
    bool runs_program() const noexcept;

    // A backedge of `frame` (running `fn`) to `loop_header`, past the
    // threshold. Asks for the header's OSR entry the first time; once the
    // entry is compiled, runs the rest of the frame's activation in it and
    // returns true with the activation's result in `out_result`. False: the
    // interpreter carries on with the loop.
    bool try_osr_migration(FastInterpreter& interp, const Function& fn, BasicBlock* loop_header, FastFrame& frame,
                           RuntimeValue& out_result);

    // Stops OSR for a program being destroyed (FunctionDispatchTable, before
    // its handles retire): drops its queued OSR compiles and waits out the
    // running ones, and later backedges migrate nowhere.
    void release_program();
    // Drops this program's queued OSR compiles and waits out the running
    // ones; a loop whose compile was dropped asks again on a later backedge.
    void stop_compiles();

    // Statistics
    uint64_t total_osr_migrations() const noexcept { return total_osr_migrations_.load(std::memory_order_relaxed); }
    void reset_stats() noexcept { total_osr_migrations_.store(0, std::memory_order_relaxed); }

private:
    OsrCoordinator();

    struct Entry;
    void request_entry(const Function& fn, const BasicBlock& header, const std::shared_ptr<Entry>& e);
    void compile_entry(const Function& fn, Module& copy, Entry& e);
    bool enter(const Function& fn, FastFrame& frame, const Entry& e, RuntimeValue& out_result);

    std::mutex mutex_;
    bool released_ = false;
    std::unordered_map<const BasicBlock*, std::shared_ptr<Entry>> entries_;

    TieringRegistry* const registry_ = nullptr; // null: the default program
    bool enabled_ = false;
    uint64_t threshold_ = BACKEDGE_OSR_THRESHOLD;
    std::atomic<uint64_t> total_osr_migrations_{0};
};

} // namespace brass::runtime
