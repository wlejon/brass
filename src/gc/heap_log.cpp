// BRASS_GC_LOG=1 (or HeapConfig::log): one line on stderr per collection,
// naming its kind, why it ran, its pause and where the pause went, and how
// much it copied, promoted and marked, and the eden limit it ran at and the
// one it left (HeapConfig::adaptive_eden); a line per lazy sweep once it is
// done (its time by who did it: host slices, the allocator, the completion at
// the next cycle's start); and a summary line per heap when the heap is
// destroyed.

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
    case GcTrigger::Incremental: return "incremental";
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
    // A full collection's cycle: its start (and of that, the last cycle's
    // lazy sweep it finished), mark slices and finish pauses.
    char cycle[128] = "";
    if (!minor) {
        std::snprintf(cycle, sizeof cycle, " | start %.3f (sweep %.3f) mark %.3f/%llu finish %.3f", ms(g.start_ns),
                      ms(g.start_sweep_ns), ms(g.mark_ns), static_cast<unsigned long long>(g.slices),
                      ms(g.finish_ns));
    }
    std::fprintf(stderr,
                 "[brass gc %p] #%llu %s (%s) %.3f ms | roots %.3f cards %.3f trace %.3f weak %.3f hooks %.3f "
                 "sweep %.3f | eden %.1fM copied %.2fM/%llu promoted %.2fM/%llu marked %.1fM/%llu cards %llu | "
                 "old %.1fM threshold %.1fM | eden limit %.1fM -> %.1fM%s | threads %u\n",
                 static_cast<const void*>(&s.heap), static_cast<unsigned long long>(index),
                 minor ? "minor" : "full", trigger_name(s.trigger), ms(pause_ns), ms(g.phase_ns[kPhaseRoots]),
                 ms(g.phase_ns[kPhaseCards]), ms(g.phase_ns[kPhaseTrace]), ms(g.phase_ns[kPhaseWeak]),
                 ms(g.phase_ns[kPhaseHooks]), ms(g.phase_ns[kPhaseSweep]), mb(g.eden_used), mb(g.copied_bytes),
                 static_cast<unsigned long long>(g.copied_objects), mb(g.promoted_bytes),
                 static_cast<unsigned long long>(g.promoted_objects), mb(g.marked_bytes),
                 static_cast<unsigned long long>(g.marked_objects), static_cast<unsigned long long>(g.dirty_cards),
                 mb(s.heap.old_used_bytes()), mb(s.full_threshold), mb(g.eden_limit), mb(s.eden_limit - s.eden_lo), cycle,
                 g.minor_threads);
}

void log_sweep(const HeapState& s) {
    const LazySweep& w = s.sweep;
    const uint64_t total = w.slice_ns + w.allocator_ns + w.complete_ns;
    std::fprintf(stderr,
                 "[brass gc %p] #%llu sweep lazy %.3f ms in %llu slices (%.3f ms, max %.3f) | allocator %.3f ms/%llu "
                 "blocks | completed %.3f ms | blocks %llu swept %llu freed | large %llu freed %.1fM | "
                 "released %.1fM\n",
                 static_cast<const void*>(&s.heap), static_cast<unsigned long long>(w.collection_index), ms(total),
                 static_cast<unsigned long long>(w.slices), ms(w.slice_ns), ms(w.slice_max_ns), ms(w.allocator_ns),
                 static_cast<unsigned long long>(w.allocator_blocks), ms(w.complete_ns),
                 static_cast<unsigned long long>(w.blocks_swept), static_cast<unsigned long long>(w.blocks_freed),
                 static_cast<unsigned long long>(w.large_freed), mb(w.large_freed_bytes), mb(w.released_bytes));
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
