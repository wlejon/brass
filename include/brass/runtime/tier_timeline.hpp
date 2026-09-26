#pragma once

// When a program's code reached its tiers, and what it cost to get there.
//
// While the timeline is on, the tiering pipeline records each compile and
// each speculation failure as a timed event: which function, which kind,
// when it started (from the timeline's epoch), how long it took and whether
// it ran on a CompilePool worker or on the thread running the program. From
// those events and the program's tiering counters, a report answers how long
// the program's hottest functions ran before their optimized code did, how
// much compile time tiering spent, and how much of it the program's own
// thread waited out (its stalls).
//
//   BRASS_TIER_LOG=1       turn it on for the process, and write each
//                          program's report to stderr when it is released
//   BRASS_TIER_LOG=<path>  the same, appending the reports to <path>
//   BRASS_TIER_LOG_EVENTS=1  the reports list every event too
//
// Off (the default), recording is one relaxed load per compile.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <string_view>
#include <vector>

namespace brass::runtime {

class TieringRegistry;

enum class TierEventKind : uint8_t {
    Tier1Compile,  // baseline code for one function (a call cycle's members each)
    Tier2Enqueue,  // queueing a background tier-2 compile; the copy it takes runs on the worker
    Tier2Compile,  // optimizing, compiling and installing one function's tier-2 code
    OsrRequest,    // the program's thread planning and copying a hot loop's OSR entry
    OsrCompile,    // optimizing and compiling that entry
    OsrEnter,      // a frame moved into an OSR entry (the first time per entry)
    Deopt,         // a guard of tier-2 or OSR code failed and the call finished in Tier 0
    Invalidate,    // tier-2 or OSR code dropped after its speculation kept failing
    FrontPass,     // the front end's tier-2 speculation pass over one copy
};

std::string_view to_string(TierEventKind kind) noexcept;

struct TierEvent {
    TierEventKind kind = TierEventKind::Tier1Compile;
    bool on_worker = false;  // on a CompilePool worker; else the program's own thread
    bool ok = true;          // the compile produced code
    uint64_t start_ns = 0;   // from the timeline's epoch
    uint64_t dur_ns = 0;     // zero for instant events
    std::string name;
};

namespace detail {
extern std::atomic<bool> g_tier_timeline_on;
extern thread_local bool t_on_compile_worker;  // set by CompilePool's workers
}

// Whether events are being recorded.
inline bool tier_timeline_enabled() noexcept {
    return detail::g_tier_timeline_on.load(std::memory_order_relaxed);
}
// Turns recording on or off. Turning it on drops the events recorded so far
// and starts the epoch now.
void enable_tier_timeline(bool on);
// Reads BRASS_TIER_LOG once and turns recording on when it is set. Called as
// a pipeline initializes; a no-op after the first call.
void enable_tier_timeline_from_env();
// Drops the events and restarts the epoch.
void clear_tier_timeline();
// Nanoseconds since the epoch.
uint64_t tier_timeline_now_ns() noexcept;

// Whether the calling thread is a CompilePool worker.
bool on_compile_worker() noexcept;

void record_tier_event(TierEventKind kind, std::string_view name, uint64_t start_ns, uint64_t dur_ns, bool ok = true);
inline void record_tier_instant(TierEventKind kind, std::string_view name) {
    if (tier_timeline_enabled()) record_tier_event(kind, name, tier_timeline_now_ns(), 0);
}

// Times a compile from construction to destruction; `ok` may be cleared on
// the way out.
class TierEventScope {
public:
    TierEventScope(TierEventKind kind, std::string_view name) noexcept
        : kind_(kind), name_(name), on_(tier_timeline_enabled()), start_(on_ ? tier_timeline_now_ns() : 0) {}
    ~TierEventScope();
    TierEventScope(const TierEventScope&) = delete;
    TierEventScope& operator=(const TierEventScope&) = delete;
    void set_ok(bool ok) noexcept { ok_ = ok; }

private:
    TierEventKind kind_;
    std::string_view name_;
    bool on_;
    bool ok_ = true;
    uint64_t start_;
};

std::vector<TierEvent> tier_timeline_snapshot();

// One program's report.
struct TierHotFunction {
    std::string name;
    uint64_t invocations = 0;
    uint64_t backedges = 0;
    // When its first optimized code (tier-2 or OSR) was ready; negative: never.
    double tier2_ms = -1.0;
    double tier1_ms = -1.0;
    // When tiering first asked for its optimized code (a tier-2 or OSR
    // request); negative: never.
    double requested_ms = -1.0;
};

struct TierTimelineSummary {
    double wall_ms = 0.0;             // epoch to the report
    double first_tier2_ms = -1.0;     // the first optimized code of any function
    double hot_tier2_ms = 0.0;        // the last of the hot functions' optimized code
    double hot_latency_ms = 0.0;      // the longest of theirs from request to code
    size_t hot_untiered = 0;          // hot functions never optimized
    double compile_ms = 0.0;          // every compile's time, on every thread
    double worker_compile_ms = 0.0;   // of which on CompilePool workers
    double stall_ms = 0.0;            // tiering work on the program's own thread
    double max_stall_ms = 0.0;        // the longest single piece of it
    size_t tier1_compiles = 0;
    size_t tier2_compiles = 0;
    size_t osr_compiles = 0;
    size_t failed_compiles = 0;
    size_t deopts = 0;
    size_t invalidations = 0;
    std::vector<TierHotFunction> hot;  // hottest first
};

// Summarizes the events recorded so far for the functions of `registry`'s
// program (all events when null), taking as hot the `hot_count` functions
// with the most invocations and backedges that crossed a tier-up threshold.
TierTimelineSummary summarize_tier_timeline(const TieringRegistry* registry, size_t hot_count = 8);
// "tier-timeline ..." lines: the summary as key=value, the hot functions and,
// with `events`, every event.
void write_tier_report(std::ostream& os, const TierTimelineSummary& summary, bool events = false);
// The report BRASS_TIER_LOG asks for, of `registry`'s program; nothing when
// the variable is unset.
void emit_tier_report_from_env(const TieringRegistry& registry);

}  // namespace brass::runtime
