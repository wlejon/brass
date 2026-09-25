// BRASS_GC_LOG=1 (or HeapConfig::log): one line on stderr per collection,
// naming its kind, why it ran, its pause and where the pause went, and how
// much it copied, promoted and marked; and a summary line per heap when the
// heap is destroyed.

#include "heap_internal.hpp"

#include <chrono>
#include <cstdio>

namespace brass::gc::detail {

namespace {

const char* trigger_name(GcTrigger t) noexcept {
    switch (t) {
    case GcTrigger::Explicit: return "explicit";
    case GcTrigger::EdenFull: return "eden-full";
    case GcTrigger::OldGrowth: return "old-growth";
    case GcTrigger::Requested: return "requested";
    case GcTrigger::Stress: return "stress";
    case GcTrigger::Exhausted: return "exhausted";
    }
    return "?";
}

double mb(uint64_t bytes) noexcept { return static_cast<double>(bytes) / (1024.0 * 1024.0); }
double ms(uint64_t ns) noexcept { return static_cast<double>(ns) / 1e6; }

} // namespace

uint64_t gc_now_ns() noexcept {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

void log_collection(const GcState& g, uint64_t pause_ns) {
    const HeapState& s = g.s;
    const HeapStats& st = s.stats;
    const uint64_t index = st.minor_collections + st.full_collections;
    const bool minor = g.kind == CollectionKind::Minor;
    std::fprintf(stderr,
                 "[brass gc %p] #%llu %s (%s) %.3f ms | roots %.3f cards %.3f trace %.3f weak %.3f hooks %.3f "
                 "sweep %.3f | eden %.1fM copied %.2fM/%llu promoted %.2fM/%llu marked %.1fM/%llu cards %llu | "
                 "old %.1fM threshold %.1fM\n",
                 static_cast<const void*>(&s.heap), static_cast<unsigned long long>(index),
                 minor ? "minor" : "full", trigger_name(s.trigger), ms(pause_ns), ms(g.phase_ns[kPhaseRoots]),
                 ms(g.phase_ns[kPhaseCards]), ms(g.phase_ns[kPhaseTrace]), ms(g.phase_ns[kPhaseWeak]),
                 ms(g.phase_ns[kPhaseHooks]), ms(g.phase_ns[kPhaseSweep]), mb(g.eden_used), mb(g.copied_bytes),
                 static_cast<unsigned long long>(g.copied_objects), mb(g.promoted_bytes),
                 static_cast<unsigned long long>(g.promoted_objects), mb(g.marked_bytes),
                 static_cast<unsigned long long>(g.marked_objects), static_cast<unsigned long long>(g.dirty_cards),
                 mb(s.heap.old_used_bytes()), mb(s.full_threshold));
}

void log_summary(const HeapState& s) {
    const HeapStats& st = s.stats;
    std::fprintf(stderr,
                 "[brass gc %p] summary: minor %llu (%.3f ms total, %.3f ms max), full %llu (%.3f ms total, "
                 "%.3f ms max), allocated %.1fM, promoted %.1fM\n",
                 static_cast<const void*>(&s.heap), static_cast<unsigned long long>(st.minor_collections),
                 ms(st.minor_pause_ns_total), ms(st.minor_pause_ns_max),
                 static_cast<unsigned long long>(st.full_collections), ms(st.full_pause_ns_total),
                 ms(st.full_pause_ns_max), mb(s.heap.allocated_bytes()), mb(st.promoted_bytes));
}

} // namespace brass::gc::detail
