// gc::Heap: construction, the thread's current heap, allocation, roots,
// hooks, finalizer registration, modes and statistics. The collections are
// heap_collect.cpp and heap_weak.cpp; queries over objects heap_verify.cpp.

#include "heap_internal.hpp"

#include <brass/gc/gc_limits.hpp>
#include <brass/runtime/coroutine.hpp>

#include <algorithm>
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

namespace {

thread_local Heap* t_current_heap = nullptr;

constexpr size_t kYoungGranularity = 64 * 1024;

size_t round_up(size_t value, size_t unit) { return (value + unit - 1) / unit * unit; }

std::string env_value(const char* name) {
#if defined(_MSC_VER)
    // MSVC deprecates getenv (C4996, an error under /WX); _dupenv_s is its
    // owned-copy equivalent.
    char* owned = nullptr;
    size_t len = 0;
    if (_dupenv_s(&owned, &len, name) != 0 || !owned) return {};
    std::string v(owned);
    std::free(owned);
    return v;
#else
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
#endif
}

void apply_environment(HeapConfig& config) {
    if (!config.read_environment) return;
    const std::string stress = env_value("BRASS_GC_STRESS");
    if (stress == "minor" || stress == "1") config.stress = StressMode::Minor;
    else if (stress == "full") config.stress = StressMode::Full;
    else if (stress == "alternate") config.stress = StressMode::Alternate;
    else if (stress == "0" || stress == "none") config.stress = StressMode::None;
    const std::string verify = env_value("BRASS_GC_VERIFY");
    if (!verify.empty() && verify != "0") config.verify = true;
    const std::string poison = env_value("BRASS_GC_POISON");
    if (!poison.empty() && poison != "0") config.poison = true;
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
    s.cards = static_cast<uint8_t*>(detail::vm_reserve(s.cards_bytes, 4096));
    s.card_pages_committed.assign((s.cards_bytes + 4095) / 4096, 0);

    young_lo_ = s.base;
    young_span_ = young;
    old_lo_ = s.mature_lo;
    old_span_ = mature + large;
    heap_lo_ = s.base;
    heap_span_ = s.reserve_bytes;
    cards_ = s.cards;

    const size_t large_threshold = std::max(s.config.large_object_bytes, kMinLargeObjectBytes);
    max_young_total_ = std::min(large_threshold + kHeaderBytes, eden / 2);

    if (!s.config.reference_tags.empty()) {
        s.reference_tag_bits.assign(65536 / 64, 0);
        for (uint16_t tag : s.config.reference_tags) {
            s.reference_tag_bits[tag >> 6] |= uint64_t{1} << (tag & 63);
        }
        ref_tags_ = s.reference_tag_bits.data();
    }

    s.full_threshold = s.config.min_full_threshold_bytes;
    s.stress = s.config.stress;
    s.verify = s.config.verify;
    s.poison = s.config.poison;
    s.reset_eden();
}

Heap::~Heap() {
    if (t_current_heap == this) t_current_heap = nullptr;
    HeapState& s = *s_;
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

Heap* Heap::current() noexcept { return t_current_heap; }
void Heap::set_current(Heap* heap) noexcept { t_current_heap = heap; }

// ---- young generation bookkeeping --------------------------------------------

namespace detail {

void HeapState::reset_eden() noexcept {
    heap.alloc_->top = eden_lo;
    eden_synced = eden_lo;
    update_fast_limit();
}

void HeapState::update_fast_limit() noexcept {
    heap.alloc_->end = stress != StressMode::None ? heap.alloc_->top : eden_hi;
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
    MutatorFrame frame(s, caller_fp, caller_ip);

    if (s.stress != StressMode::None) {
        ++s.stress_counter;
        const bool full = s.stress == StressMode::Full ||
                          (s.stress == StressMode::Alternate && (s.stress_counter & 7) == 0);
        collect_at(full ? CollectionKind::Full : CollectionKind::Minor, 0, 0);
    } else if (s.full_requested) {
        collect_at(CollectionKind::Full, 0, 0);
    } else if (s.minor_requested) {
        collect_at(CollectionKind::Minor, 0, 0);
    }

    const bool old = (flags & (kAllocOld | kAllocPinned)) != 0 || total > max_young_total_;
    if (old) {
        if (s.old_allocated_since_full + total > s.full_threshold) collect_at(CollectionKind::Full, 0, 0);
        uintptr_t header = s.old_allocate(total);
        if (header == 0) {
            collect_at(CollectionKind::Full, 0, 0);
            header = s.old_allocate(total);
            if (header == 0) throw std::bad_alloc();
        }
        auto* h = reinterpret_cast<ObjectHeader*>(header);
        h->size = static_cast<uint32_t>(payload);
        h->layout = layout;
        h->gc_bits = (flags & kAllocPinned) ? kGcPinned : 0;
        h->host_bits = host_bits;
        if (total <= kMaxMediumObjectBytes) {
            std::memset(reinterpret_cast<void*>(header + kHeaderBytes), 0, payload);
        }  // a large object's pages were committed for it and read as zero
        s.old_allocated_since_full += total;
        s.old_direct_bytes += total;
        return header + kHeaderBytes;
    }

    for (int attempt = 0; attempt < 3; ++attempt) {
        if (s.eden_hi - alloc_->top >= total) {
            const uintptr_t at = alloc_->top;
            alloc_->top = at + total;
            s.update_fast_limit();
            auto* h = reinterpret_cast<ObjectHeader*>(at);
            h->size = static_cast<uint32_t>(payload);
            h->layout = layout;
            h->gc_bits = 0;
            h->host_bits = host_bits;
            std::memset(reinterpret_cast<void*>(at + kHeaderBytes), 0, payload);
            return at + kHeaderBytes;
        }
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
    if (kind == CollectionKind::Minor && s.full_requested) Collector::collect(*this, CollectionKind::Full);

    // Finalizers run once the heap is consistent again; they may allocate.
    while (!s.pending_finalizers.empty()) {
        std::vector<detail::FinalizerEntry> pending;
        pending.swap(s.pending_finalizers);
        for (const auto& f : pending) f.fn(f.context);
    }
}

void Heap::safepoint_at(uintptr_t caller_fp, uintptr_t caller_ip) {
    HeapState& s = *s_;
    if (s.collecting) return;
    if (s.stress != StressMode::None) {
        ++s.stress_counter;
        const bool full = s.stress == StressMode::Full ||
                          (s.stress == StressMode::Alternate && (s.stress_counter & 7) == 0);
        collect_at(full ? CollectionKind::Full : CollectionKind::Minor, caller_fp, caller_ip);
    } else if (s.full_requested) {
        collect_at(CollectionKind::Full, caller_fp, caller_ip);
    } else if (s.minor_requested) {
        collect_at(CollectionKind::Minor, caller_fp, caller_ip);
    }
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

void Heap::add_finalizer(uintptr_t object, void (*fn)(void*), void* context) {
    if (!fn) return;
    if (!is_valid_object(object)) {
        throw std::invalid_argument("brass::gc::Heap::add_finalizer: not an object of this heap");
    }
    auto& list = is_young(object) ? s_->young_finalizers : s_->old_finalizers;
    list.push_back(detail::FinalizerEntry{object, fn, context});
}

// ---- modes and statistics ----------------------------------------------------------

void Heap::set_stress(StressMode mode) noexcept {
    s_->stress = mode;
    s_->update_fast_limit();
}

StressMode Heap::stress() const noexcept { return s_->stress; }
void Heap::set_verify(bool enable) noexcept { s_->verify = enable; }
void Heap::set_poison(bool enable) noexcept { s_->poison = enable; }
const HeapStats& Heap::stats() const noexcept { return s_->stats; }
uint64_t Heap::collection_count() const noexcept {
    return s_->stats.minor_collections + s_->stats.full_collections;
}
uint64_t Heap::relocation_epoch() const noexcept { return s_->relocation_epoch; }

size_t Heap::young_used_bytes() const noexcept {
    const HeapState& s = *s_;
    const int from = s.from_survivor;
    return (alloc_->top - s.eden_lo) + (s.survivor_top[from] - s.survivor_lo[from]);
}

size_t Heap::old_used_bytes() const noexcept {
    return s_->mature_used_lines * kLineBytes + s_->large_used_bytes + s_->old_allocated_since_full;
}

size_t Heap::committed_bytes() const noexcept {
    const HeapState& s = *s_;
    size_t blocks = 0;
    for (const auto& meta : s.blocks) blocks += meta->committed ? 1 : 0;
    return (s.eden_hi - s.eden_lo) + 2 * s.survivor_bytes + blocks * kBlockBytes + s.large_used_bytes;
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
