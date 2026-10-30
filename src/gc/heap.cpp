// gc::Heap: construction, the thread's current heap, allocation, roots,
// hooks, finalizer registration, modes and statistics. The collections are
// heap_collect.cpp and heap_weak.cpp; queries over objects heap_verify.cpp.

#include "heap_internal.hpp"

#include <brass/core/env.hpp>
#include <brass/gc/fiber.hpp>
#include <brass/gc/gc_limits.hpp>
#include <brass/runtime/coroutine.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>

namespace brass::gc {

using detail::Collector;
using detail::HeapState;
using detail::MutatorFrame;

// The thread's heap (Heap::current); runtime_gc.cpp reads it directly on
// generated code's allocation path.
namespace detail {
#if defined(_MSC_VER)
thread_local Heap* t_current_heap = nullptr;
#else
__attribute__((visibility("default"))) thread_local Heap* t_current_heap = nullptr;
#endif
}
using detail::t_current_heap;

namespace {

// Per stack (StackLocal): set and restored around each interpreter frame
// and runtime entry on one stack. A fiber starts on the heap current where
// it was first resumed.
[[maybe_unused]] const int k_current_heap_local = register_stack_local(StackLocal{
    [](void** slot) { *slot = t_current_heap; },
    [](void** slot, bool fresh) {
        if (!fresh) t_current_heap = static_cast<Heap*>(*slot);
        *slot = nullptr;
    },
    nullptr});

constexpr size_t kYoungGranularity = 64 * 1024;

size_t round_up(size_t value, size_t unit) { return (value + unit - 1) / unit * unit; }

void apply_environment(HeapConfig& config) {
    if (!config.read_environment) return;
    const std::string stress = env_value("BRASS_GC_STRESS");
    if (stress == "minor" || stress == "1") config.stress = StressMode::Minor;
    else if (stress == "full") config.stress = StressMode::Full;
    else if (stress == "alternate") config.stress = StressMode::Alternate;
    else if (stress == "incremental") config.stress = StressMode::Incremental;
    else if (stress == "0" || stress == "none") config.stress = StressMode::None;
    const std::string verify = env_value("BRASS_GC_VERIFY");
    if (!verify.empty() && verify != "0") config.verify = true;
    const std::string poison = env_value("BRASS_GC_POISON");
    if (!poison.empty() && poison != "0") config.poison = true;
    const std::string log = env_value("BRASS_GC_LOG");
    if (!log.empty() && log != "0") config.log = true;
    // The A/B switch for the adaptive eden: every minor collection at a full eden.
    const std::string fixed_eden = env_value("BRASS_GC_FIXED_EDEN");
    if (!fixed_eden.empty() && fixed_eden != "0") config.adaptive_eden = false;
    // The A/B switch for the lazy sweep: the old generation swept inside the
    // full collection's pause.
    const std::string eager_sweep = env_value("BRASS_GC_EAGER_SWEEP");
    if (!eager_sweep.empty() && eager_sweep != "0") config.lazy_sweep = false;
    const std::string threads = env_value("BRASS_GC_MARK_THREADS");
    if (!threads.empty()) config.mark_threads = static_cast<unsigned>(std::strtoul(threads.c_str(), nullptr, 10));
    // The A/B switch for the parked marker threads: created and joined by
    // every drain.
    if (env_value("BRASS_GC_MARK_POOL") == "0") config.mark_thread_pool = false;
    // Both old spaces' reservations, in MB: a small one makes the
    // out-of-memory path reachable from a test.
    const std::string old_mb = env_value("BRASS_GC_OLD_RESERVE_MB");
    if (const unsigned long long mb = std::strtoull(old_mb.c_str(), nullptr, 10); mb > 0) {
        config.mature_reserve_bytes = static_cast<size_t>(mb) << 20;
        config.large_reserve_bytes = static_cast<size_t>(mb) << 20;
    }
}

} // namespace

Heap::Heap(const HeapConfig& config) : s_(std::make_unique<HeapState>(*this)) {
    HeapState& s = *s_;
    s.config = config;
    apply_environment(s.config);
    s.config.tenure_age = static_cast<uint8_t>(std::clamp<unsigned>(s.config.tenure_age, 1u, 7u));

    const size_t eden = round_up(std::max(s.config.eden_bytes, kMinEdenBytes), kYoungGranularity);
    const size_t survivor = round_up(std::max<size_t>(s.config.survivor_bytes, kYoungGranularity), kYoungGranularity);
    const size_t young = eden + 2 * survivor;
    const size_t mature = round_up(std::max<size_t>(s.config.mature_reserve_bytes, 64 * kBlockBytes), kBlockBytes);
    const size_t large = round_up(std::max<size_t>(s.config.large_reserve_bytes, 64 * kPageBytes), kBlockBytes);
    s.reserve_bytes = young + mature + large;
    s.base = reinterpret_cast<uintptr_t>(detail::vm_reserve(s.reserve_bytes, kYoungGranularity));

    s.eden_lo = s.base;
    s.eden_hi = s.base + eden;
    s.survivor_bytes = survivor;
    s.survivor_lo[0] = s.eden_hi;
    s.survivor_lo[1] = s.eden_hi + survivor;
    s.survivor_top[0] = s.survivor_lo[0];
    s.survivor_top[1] = s.survivor_lo[1];
    detail::vm_commit(reinterpret_cast<void*>(s.base), young);
    s.young_starts.assign(young / kGranuleBytes / 64, 0);

    s.mature_lo = s.base + young;
    s.mature_reserve = mature;
    s.large_lo = s.mature_lo + mature;
    s.large_reserve = large;

    s.cards_bytes = (mature + large) >> kCardShift;
    const size_t card_page = detail::os_page_bytes();
    s.cards = static_cast<uint8_t*>(detail::vm_reserve(s.cards_bytes, card_page));
    s.card_pages_committed.assign((s.cards_bytes + card_page - 1) / card_page, 0);
    s.region_bytes.assign(((mature + large) >> kCardRegionShift) + 1, 0);
    s.regions = s.region_bytes.data();
    regions_ = s.regions;

    young_lo_ = s.base;
    young_span_ = young;
    old_lo_ = s.mature_lo;
    old_span_ = mature + large;
    heap_lo_ = s.base;
    heap_span_ = s.reserve_bytes;
    cards_ = s.cards;
    set_barrier_marking(false);

    // What one collection could promote, doubled for fragmentation, held back
    // from the mutator (HeapState::reserve_mature_blocks); never more than a
    // quarter of a space.
    const size_t promotable = 2 * (eden + survivor);
    s.reserve_mature_blocks = std::min((promotable + kBlockBytes - 1) / kBlockBytes + 4, mature / kBlockBytes / 4);
    s.reserve_large_bytes = std::min(round_up(promotable, kPageBytes), large / 4);

    const size_t large_threshold = std::max(s.config.large_object_bytes, kMinLargeObjectBytes);
    max_young_total_ = std::max(std::min(large_threshold + kHeaderBytes, eden / 4), kMinLargeObjectBytes + kHeaderBytes);

    if (!s.config.reference_tags.empty()) {
        s.reference_tag_bits.assign(65536 / 64, 0);
        for (uint16_t tag : s.config.reference_tags) {
            s.reference_tag_bits[tag >> 6] |= uint64_t{1} << (tag & 63);
        }
        ref_tags_ = s.reference_tag_bits.data();
        if (!is_reference_tag(0)) {
            raw_address_tag_ = static_cast<uint64_t>(s.config.reference_tags.front()) << kTagShift;
        }
    }

    s.full_threshold = s.config.min_full_threshold_bytes;
    s.stress = s.config.stress;
    s.verify = s.config.verify;
    s.poison = s.config.poison;
    s.log = s.config.log;
    s.sweep.lazy = s.config.lazy_sweep;
    s.reset_eden_limit();
    s.reset_eden();
}

Heap::~Heap() {
    if (t_current_heap == this) t_current_heap = nullptr;
    HeapState& s = *s_;
    if (s.log && s.stats.minor_collections + s.stats.full_collections > 0) detail::log_summary(s);
    s.coro_frames.reset();  // its frames are gone with the heap
    detail::vm_release(s.cards, s.cards_bytes);
    detail::vm_release(reinterpret_cast<void*>(s.base), s.reserve_bytes);
}

namespace {
std::mutex& default_config_mutex() {
    static std::mutex m;
    return m;
}
HeapConfig& default_config_storage() {
    static HeapConfig config;
    return config;
}
} // namespace

HeapConfig Heap::default_config() {
    std::lock_guard<std::mutex> lock(default_config_mutex());
    return default_config_storage();
}

void Heap::set_default_config(const HeapConfig& config) {
    std::lock_guard<std::mutex> lock(default_config_mutex());
    default_config_storage() = config;
}

#if defined(_MSC_VER)
Heap* Heap::current() noexcept { return t_current_heap; }
void Heap::set_current(Heap* heap) noexcept { t_current_heap = heap; }
#else
__attribute__((visibility("default"))) Heap* Heap::current() noexcept { return t_current_heap; }
__attribute__((visibility("default"))) void Heap::set_current(Heap* heap) noexcept { t_current_heap = heap; }
#endif

// ---- young generation bookkeeping --------------------------------------------

namespace detail {

void HeapState::reset_eden() noexcept {
    heap.alloc_->top = eden_lo;
    eden_synced = eden_lo;
    update_fast_limit();
}

void HeapState::update_fast_limit() noexcept {
    // Generated code bumps against `end` and nothing else, so the adaptive
    // limit takes effect there with no change to the code. `end` is never
    // below `top`: an object allowed past eden_limit (Heap::allocate_at) puts
    // `top` above it, and the bump path computes end - top unsigned. Stress
    // mode shuts it, so every allocation reaches the runtime. An open marking
    // cycle does not: young allocation is the same during one (young objects
    // are never marked; minor collections and the finish trace them).
    const uintptr_t top = heap.alloc_->top;
    const bool shut = stress != StressMode::None;
    heap.alloc_->end = shut ? top : std::max(top, eden_limit);
}

namespace {
// The least eden the adaptive limit leaves: below this a minor collection's
// fixed costs (roots, cards) outweigh the copying it bounds.
constexpr size_t kMinEdenLimitBytes = size_t{1} << 20;
// A minor collection that collected less eden than this (one a host ran just
// after another) says nothing of an eden's survival: its survivor-space
// copies would read as a rate many times 1.
constexpr uint64_t kMinSurvivalSampleBytes = 64 * 1024;
// How much of the survival estimate carries to the next minor collection
// when that one sees less: 0.5 doubles the limit back per collection, so a
// spike's end costs about one eden's worth of extra, smaller collections.
constexpr double kSurvivalDecay = 0.5;
// The weight of the newest shrunk minor collection in the running averages
// of what died in the survivor space (HeapState::aged_average and the rest).
constexpr double kDyingWeight = 1.0 / 8.0;
// The shrunk minor collections averaged before the first verdict, and the
// most a verdict can come to need (HeapState::dying_needed doubles).
constexpr uint32_t kFirstDyingNeeded = 4;
constexpr uint32_t kMaxDyingNeeded = 64;
// A lifetime under this many whole edens sends eden back whole. Not one: the
// survivors' deaths read low, since a container promoted at a shrunk eden
// (a growing array's store) keeps the young objects it names alive through
// its dirty card after it has died, until a full collection; and the whole
// eden's first collection judges again with no such distortion.
constexpr double kDyingLifetimeEdens = 4.0;
// The minor collections at the whole eden, after the lifetime test restored
// it, that cannot shrink it again (HeapState::restore_hold).
constexpr uint32_t kRestoreHold = 4;
} // namespace

void HeapState::forget_dying() noexcept {
    aged_average = 0.0;
    dead_average = 0.0;
    collected_average = 0.0;
    dying_measured = 0;
}

void HeapState::reset_eden_limit() noexcept {
    survival_estimate = 0.0;
    forget_dying();
    restored = false;
    restore_hold = 0;
    eden_limit = eden_hi;
}

void HeapState::adapt_eden_limit(const MinorSample& m) noexcept {
    if (!config.adaptive_eden) {
        eden_limit = eden_hi;
        return;
    }
    const size_t eden = eden_hi - eden_lo;
    if (m.eden_collected >= kMinSurvivalSampleBytes) {
        // A smaller eden spreads copying out only when the survivors are data
        // the program keeps (a build-up), whose copying is proportional to
        // eden. Young data that lives a while and then dies (a working set)
        // reads as survival too while eden is shorter than its life, the
        // more so the smaller eden is, so after a build-up nothing in the
        // survival rate would ever bring the whole eden back, and shrinking
        // only promotes what the whole eden would let die young. What tells
        // them apart is how many of last collection's survivors, aged in the
        // survivor space (minor() keeps promote_all off while eden is
        // shrunk), die before this one: a build-up keeps nearly all of them,
        // and a working set loses a share that is the eden allocated between
        // the two over its lifetime. That lifetime, averaged over the last
        // several collections and short against the whole eden, means the
        // whole eden may let the data die young: eden goes back whole for the
        // collection there to judge (kDyingLifetimeEdens).
        bool dying = false;
        if (!m.at_whole_eden && m.tenure_age >= 2 && m.aged >= kMinSurvivalSampleBytes) {
            const double dead = static_cast<double>(m.aged - std::min(m.tenured, m.aged));
            aged_average += (static_cast<double>(m.aged) - aged_average) * kDyingWeight;
            dead_average += (dead - dead_average) * kDyingWeight;
            collected_average += (static_cast<double>(m.eden_collected) - collected_average) * kDyingWeight;
            ++dying_measured;
            // lifetime = collected * aged / dead, compared free of a division.
            dying = dying_measured >= dying_needed && collected_average * aged_average <
                                                          kDyingLifetimeEdens * static_cast<double>(eden) * dead_average;
        }
        if (dying) {
            survival_estimate = 0.0;
            forget_dying();
            restored = true;
            restore_hold = kRestoreHold;
        } else {
            double seen = static_cast<double>(m.copied) / static_cast<double>(m.eden_collected);
            // At the whole eden the survivors must also be shown kept before
            // eden shrinks at all: overflowing the survivor space, or a
            // quarter of what was copied promoted out of it (kept through two
            // collections, not promoted wholesale by promote_all).
            if (m.at_whole_eden && !m.overflow && m.tenured * 4 < m.copied) seen = 0.0;
            if (m.at_whole_eden && restore_hold > 0) {
                // Just after the lifetime test restored the whole eden, what
                // the shrunk eden promoted still keeps young objects alive
                // through its dirty cards and the survivor space holds what
                // its short span aged: a few collections must pass before the
                // whole eden's own survival says anything.
                --restore_hold;
                seen = 0.0;
            } else if (m.at_whole_eden && restored) {
                // The first collection that judges: shrinking again says the
                // test misread a build-up, so its next verdict averages twice
                // as many collections; staying whole says it read a working
                // set right.
                dying_needed = seen > 0.0 ? std::min(dying_needed * 2, kMaxDyingNeeded) : kFirstDyingNeeded;
                restored = false;
            }
            // A shrink starts: the averages are about what comes now.
            if (m.at_whole_eden && seen > 0.0) forget_dying();
            survival_estimate = std::max(seen, survival_estimate * kSurvivalDecay);
        }
    }
    const size_t floor = std::min(kMinEdenLimitBytes, eden);
    double want = static_cast<double>(eden);
    if (survival_estimate > 0.0) {
        want = std::min(want, static_cast<double>(config.minor_copy_budget_bytes) / survival_estimate);
    }
    // Whole pages, so the limit does not wander by a few bytes per collection.
    const size_t bytes = std::clamp(static_cast<size_t>(want) / kPageBytes * kPageBytes, floor, eden);
    eden_limit = eden_lo + bytes;
}

void HeapState::sync_eden_starts() noexcept {
    uintptr_t p = eden_synced;
    const uintptr_t top = heap.alloc_->top;
    while (p < top) {
        const auto* header = reinterpret_cast<const ObjectHeader*>(p);
        set_young_start(p + kHeaderBytes);
        p += kHeaderBytes + header->size;
    }
    eden_synced = p;
}

void HeapState::clear_young_starts(uintptr_t lo, uintptr_t hi) noexcept {
    if (hi <= lo) return;
    const size_t first = young_bit(lo) >> 6;
    const size_t last = (young_bit(hi) + 63) >> 6;
    std::memset(young_starts.data() + first, 0, (std::min(last, young_starts.size()) - first) * sizeof(uint64_t));
}

} // namespace detail

// ---- allocation ---------------------------------------------------------------

uintptr_t Heap::allocate_at(size_t bytes, LayoutId layout, uint32_t flags, uint8_t host_bits,
                            uintptr_t caller_fp, uintptr_t caller_ip) {
    HeapState& s = *s_;
    if (s.collecting) {
        detail::gc_fatal("allocation during a collection (a trace function, root source or "
                         "post-collection hook allocated on the heap it runs for)");
    }
    if (bytes > (size_t{1} << 31)) throw std::bad_alloc();
    const size_t payload = payload_bytes_for(bytes);
    const size_t total = payload + kHeaderBytes;
    const bool zero = (flags & kAllocUninitialized) == 0;
    if (!zero && layout_descriptor(layout).kind != LayoutKind::Leaf) {
        detail::gc_fatal("kAllocUninitialized for a layout the collector traces");
    }
    MutatorFrame frame(s, caller_fp, caller_ip);

    if (s.stress != StressMode::None) {
        stress_step(0, 0);
    } else if (s.full_requested) {
        s.trigger = detail::GcTrigger::Requested;
        collect_at(CollectionKind::Full, 0, 0);
    } else if (s.minor_requested) {
        s.trigger = detail::GcTrigger::Requested;
        collect_at(CollectionKind::Minor, 0, 0);
    }
    // A collection has had to promote into the reserve (HeapState::
    // old_critical): a full collection may give the room back; if it does not,
    // this allocation fails rather than let the heap reach a collection with
    // nowhere to promote to. The failure leaves the heap consistent, eden
    // empty, so the program can drop references and go on.
    if (s.old_critical) {
        s.trigger = detail::GcTrigger::Exhausted;
        collect_at(CollectionKind::Full, 0, 0);
        if (s.old_critical) throw std::bad_alloc();
    }

    const bool old =(flags & (kAllocOld | kAllocPinned)) != 0 || total > max_young_total_;
    if (old) {
        if (s.old_allocated_since_full + total > s.full_threshold) {
            s.trigger = detail::GcTrigger::OldGrowth;
            collect_at(CollectionKind::Full, 0, 0);
        }
        uintptr_t header = s.old_allocate(total);
        if (header == 0) {
            s.trigger = detail::GcTrigger::Exhausted;
            collect_at(CollectionKind::Full, 0, 0);
            header = s.old_allocate(total);
            if (header == 0) throw std::bad_alloc();
        }
        auto* h = reinterpret_cast<ObjectHeader*>(header);
        h->size = static_cast<uint32_t>(payload);
        h->layout = layout;
        h->gc_bits = (flags & kAllocPinned) ? kGcPinned : 0;
        h->host_bits = host_bits;
        // A large object made where the pending sweep has yet to walk is
        // marked, so the walk keeps it (HeapState::note_large_allocated).
        s.note_large_allocated(header);
        if (zero) std::memset(reinterpret_cast<void*>(header + kHeaderBytes), 0, payload);
        s.old_allocated_since_full += total;
        s.old_direct_bytes += total;
        // While a cycle is open a new old object is live to it (grey: the
        // marker scans it once its caller has filled it, at a later GC
        // point) and its card dirty, so a caller that fills it with stores
        // the barrier never sees (a frame, B13 in the design) has them
        // rescanned by the next minor collection too.
        if (s.mark_phase == detail::MarkPhase::Marking) Collector::allocated_old(*this, header + kHeaderBytes);
        return header + kHeaderBytes;
    }

    for (int attempt = 0; attempt < 3; ++attempt) {
        // Allocation stops at the adaptive limit, but an empty eden takes any
        // young object whole: max_young_total_ can be above the least limit.
        const uintptr_t limit = alloc_->top == s.eden_lo ? s.eden_hi : std::max(s.eden_limit, alloc_->top);
        if (limit - alloc_->top >= total) {
            const uintptr_t at = alloc_->top;
            alloc_->top = at + total;
            s.update_fast_limit();
            auto* h = reinterpret_cast<ObjectHeader*>(at);
            h->size = static_cast<uint32_t>(payload);
            h->layout = layout;
            h->gc_bits = 0;
            h->host_bits = host_bits;
            if (zero) std::memset(reinterpret_cast<void*>(at + kHeaderBytes), 0, payload);
            return at + kHeaderBytes;
        }
        s.trigger = detail::GcTrigger::EdenFull;
        collect_at(CollectionKind::Minor, 0, 0);
    }
    throw std::bad_alloc();
}

// ---- collection entry points ---------------------------------------------------

void Heap::collect_at(CollectionKind kind, uintptr_t caller_fp, uintptr_t caller_ip) {
    HeapState& s = *s_;
    if (s.collecting) {
        detail::gc_fatal("a collection was requested during a collection (a trace function, root "
                         "source or post-collection hook allocated or collected)");
    }
    MutatorFrame frame(s, caller_fp, caller_ip);
    Collector::collect(*this, kind);
    if (kind == CollectionKind::Minor && s.full_requested) {
        s.trigger = detail::GcTrigger::OldGrowth;
        Collector::collect(*this, CollectionKind::Full);
    }
    run_pending_finalizers();
}

void Heap::run_pending_finalizers() {
    HeapState& s = *s_;
    // Finalizers run once the heap is consistent again; they may allocate, and
    // so collect. A collection a finalizer starts leaves what it finds dead
    // queued, and the loop below runs it after the batch under way.
    if (s.finalizers_running) return;
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t ran = 0;
    // A finalizer that throws (an allocation out of memory) leaves the rest of
    // its batch unrun, as it always has, but must not leave the loop marked
    // running, or no finalizer would run again.
    struct RunningScope {
        detail::HeapState& s;
        explicit RunningScope(detail::HeapState& state) : s(state) { s.finalizers_running = true; }
        ~RunningScope() {
            s.running_finalizers.clear();
            s.finalizers_running = false;
        }
    } running(s);
    while (!s.pending_finalizers.empty()) {
        // Swapped, not copied: the two vectors trade buffers, so neither
        // allocates once both have grown to the program's batch size.
        s.running_finalizers.swap(s.pending_finalizers);
        // Indexed: a finalizer's own collection appends to pending_finalizers,
        // never to the batch being run, so the batch does not move under it.
        for (size_t i = 0; i < s.running_finalizers.size(); ++i) {
            const detail::FinalizerEntry& f = s.running_finalizers[i];
            f.fn(f.a, f.b);
        }
        ran += s.running_finalizers.size();
        s.running_finalizers.clear();
    }
    s.stats.last_finalizers = ran;
    s.stats.last_finalizer_ns = ran == 0 ? 0 : static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    if (s.log && ran > 0)
        std::fprintf(stderr, "[brass gc %p] finalizers: %llu in %.3f ms\n", static_cast<void*>(this),
                     static_cast<unsigned long long>(ran), s.stats.last_finalizer_ns / 1e6);
}

void Heap::safepoint_at(uintptr_t caller_fp, uintptr_t caller_ip) {
    HeapState& s = *s_;
    if (s.collecting) return;
    if (s.stress != StressMode::None) {
        stress_step(caller_fp, caller_ip);
    } else if (s.full_requested) {
        s.trigger = detail::GcTrigger::Requested;
        collect_at(CollectionKind::Full, caller_fp, caller_ip);
    } else if (s.minor_requested) {
        s.trigger = detail::GcTrigger::Requested;
        collect_at(CollectionKind::Minor, caller_fp, caller_ip);
    }
}

void Heap::stress_step(uintptr_t caller_fp, uintptr_t caller_ip) {
    HeapState& s = *s_;
    ++s.stress_counter;
    s.trigger = detail::GcTrigger::Stress;
    if (s.stress == StressMode::Incremental) {
        MutatorFrame frame(s, caller_fp, caller_ip);
        Collector::stress_incremental(*this);
        return;
    }
    const bool full = s.stress == StressMode::Full ||
                      (s.stress == StressMode::Alternate && (s.stress_counter & 7) == 0);
    collect_at(full ? CollectionKind::Full : CollectionKind::Minor, caller_fp, caller_ip);
}

void Heap::request_collection(CollectionKind kind) noexcept {
    if (kind == CollectionKind::Full) s_->full_requested = true;
    else s_->minor_requested = true;
}

bool Heap::in_collection() const noexcept { return s_->collecting; }

// ---- roots, hooks, finalizers ---------------------------------------------------

void Heap::add_root(uint64_t* slot) {
    if (slot) s_->roots.push_back(slot);
}

void Heap::remove_root(uint64_t* slot) {
    auto& roots = s_->roots;
    for (size_t i = roots.size(); i-- > 0;) {
        if (roots[i] == slot) {
            roots.erase(roots.begin() + static_cast<std::ptrdiff_t>(i));
            return;
        }
    }
}

Heap::RootSourceId Heap::add_root_source(RootSourceFn fn, void* context) {
    const uint64_t id = s_->next_id++;
    s_->sources.push_back(detail::RootSourceEntry{id, fn, context, {}});
    return id;
}

Heap::RootSourceId Heap::add_root_source(std::function<void(Tracer&)> fn) {
    const uint64_t id = s_->next_id++;
    s_->sources.push_back(detail::RootSourceEntry{id, nullptr, nullptr, std::move(fn)});
    return id;
}

void Heap::remove_root_source(RootSourceId id) noexcept {
    auto& sources = s_->sources;
    sources.erase(std::remove_if(sources.begin(), sources.end(),
                                 [id](const detail::RootSourceEntry& e) { return e.id == id; }),
                  sources.end());
}

Heap::RootSourceId Heap::add_post_collection_hook(PostCollectionHook hook) {
    const uint64_t id = s_->next_id++;
    s_->hooks.push_back(detail::HookEntry{id, std::move(hook)});
    return id;
}

void Heap::remove_post_collection_hook(RootSourceId id) noexcept {
    auto& hooks = s_->hooks;
    hooks.erase(std::remove_if(hooks.begin(), hooks.end(),
                               [id](const detail::HookEntry& e) { return e.id == id; }),
                hooks.end());
}

namespace {
// The one-context form, stored as a two-context entry: `b` carries the
// callback (a function pointer held as a data pointer, which every platform
// brass targets round-trips exactly).
void run_one_context_finalizer(void* context, void* fn) {
    reinterpret_cast<void (*)(void*)>(fn)(context);
}
} // namespace

void Heap::add_finalizer(uintptr_t object, void (*fn)(void*), void* context) {
    if (!fn) return;
    add_finalizer(object, &run_one_context_finalizer, context, reinterpret_cast<void*>(fn));
}

void Heap::add_finalizer(uintptr_t object, void (*fn)(void* a, void* b), void* a, void* b) {
    if (!fn) return;
    if (!is_valid_object(object)) {
        throw std::invalid_argument("brass::gc::Heap::add_finalizer: not an object of this heap");
    }
    auto& list = is_young(object) ? s_->young_finalizers : s_->old_finalizers;
    list.push_back(detail::FinalizerEntry{object, fn, a, b});
}

// ---- modes and statistics ----------------------------------------------------------

void Heap::set_stress(StressMode mode) noexcept {
    s_->stress = mode;
    s_->update_fast_limit();
}

StressMode Heap::stress() const noexcept { return s_->stress; }
void Heap::set_verify(bool enable) noexcept { s_->verify = enable; }
bool Heap::verifying() const noexcept { return s_->verify; }
void Heap::set_poison(bool enable) noexcept { s_->poison = enable; }
const HeapStats& Heap::stats() const noexcept { return s_->stats; }
uint64_t Heap::collection_count() const noexcept {
    return s_->stats.minor_collections + s_->stats.full_collections;
}
uint64_t Heap::relocation_epoch() const noexcept { return s_->relocation_epoch; }

size_t Heap::release_unused_young() noexcept {
    HeapState& s = *s_;
    if (in_collection()) return 0;
    const size_t page = detail::os_page_bytes();
    size_t released = 0;
    auto release = [&](uintptr_t lo, uintptr_t hi) {
        lo = (lo + page - 1) & ~(uintptr_t{page} - 1);
        hi &= ~(uintptr_t{page} - 1);
        if (hi <= lo) return;
        // Decommitted and committed again: fresh zero pages, as at the start.
        try {
            detail::vm_decommit(reinterpret_cast<void*>(lo), hi - lo);
            detail::vm_commit(reinterpret_cast<void*>(lo), hi - lo);
        } catch (...) {
            // vm_commit throws only when the system refuses the range, which
            // the heap cannot run without: as at a failed allocation.
            std::abort();
        }
        released += hi - lo;
    };
    release(alloc_->top, s.eden_hi);
    // Between collections only one survivor space holds anything; each is
    // released above its top either way.
    for (int i = 0; i < 2; ++i) release(s.survivor_top[i], s.survivor_lo[i] + s.survivor_bytes);
    return released;
}

size_t Heap::young_used_bytes() const noexcept {
    const HeapState& s = *s_;
    const int from = s.from_survivor;
    return (alloc_->top - s.eden_lo) + (s.survivor_top[from] - s.survivor_lo[from]);
}

HeapPressure Heap::pressure() const noexcept {
    const HeapState& s = *s_;
    return {alloc_->top - s.eden_lo, s.eden_limit - s.eden_lo, s.old_allocated_since_full, s.full_threshold,
            s.sweep.unswept_blocks, s.sweep.active};
}

size_t Heap::old_used_bytes() const noexcept {
    const HeapState& s = *s_;
    // While blocks or large objects are unswept, the line counts and the
    // large bytes still hold the dead: the last cycle's marked bytes stand in
    // for them until the sweep has counted again.
    if (s.sweep.unswept_blocks > 0 || s.sweep.large_active) return s.sweep.live_bytes + s.old_allocated_since_full;
    return s.mature_used_lines * kLineBytes + s.large_used_bytes + s.old_allocated_since_full;
}

size_t Heap::committed_bytes() const noexcept {
    const HeapState& s = *s_;
    size_t blocks = 0;
    for (const auto& meta : s.blocks) blocks += meta->committed ? 1 : 0;
    return (s.eden_hi - s.eden_lo) + 2 * s.survivor_bytes + blocks * kBlockBytes + s.large_used_bytes +
           s.large_retained_bytes;
}

uint64_t Heap::allocated_bytes() const noexcept {
    return s_->eden_bytes_retired + (alloc_->top - s_->eden_lo) + s_->old_direct_bytes;
}

void Heap::bind_allocation_buffer(AllocationBuffer* buffer) noexcept {
    AllocationBuffer* next = buffer ? buffer : &own_alloc_;
    if (next == alloc_) return;
    *next = *alloc_;
    alloc_ = next;
}

runtime::CoroFrameRegistry& Heap::coro_frames() {
    if (!s_->coro_frames) {
        s_->coro_frames = runtime::make_coro_frame_registry([this](uintptr_t address) {
            return is_valid_object(address) && type_tag_of(address) == runtime::TYPE_TAG_CORO_FRAME;
        });
    }
    return *s_->coro_frames;
}

} // namespace brass::gc
