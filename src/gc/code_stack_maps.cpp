#include <brass/gc/code_stack_maps.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace brass {

namespace {

// The maps of one registration. It lives as long as a segment indexes it, so
// an entry's map stays readable after the code is unregistered; lookups skip
// the entries of a registration that is no longer alive.
struct RegistrationData {
    std::vector<FunctionStackMap> maps;
    std::atomic<bool> alive{true};
};

struct Entry {
    uintptr_t begin = 0;
    uintptr_t end = 0; // exclusive
    const FunctionStackMap* map = nullptr;
    const RegistrationData* owner = nullptr;
};

// An immutable run of entries, sorted by begin and disjoint, with the
// registrations they point into.
struct Segment {
    std::vector<Entry> entries;
    std::vector<std::shared_ptr<const RegistrationData>> owners;
};

} // namespace

// The published index: a few segments (a log-structured merge, newest last).
// A registration adds a small segment and merges it with its neighbours of
// similar size, so each entry is copied O(log n) times over its life rather
// than once per registration. An unregistration only clears its alive flag;
// dead entries are dropped when their segment is next merged, and all of them
// once they outnumber the live ones.
class CodeStackMapSnapshot {
public:
    std::vector<std::shared_ptr<const Segment>> segments;
};

namespace {

// std::atomic<std::shared_ptr> is C++20, but libc++ (every Xcode through at
// least 16.4) does not ship it. Its stand-in there is the shared_ptr overloads
// of std::atomic_load / std::atomic_store: deprecated in C++20 for exactly this
// class template, and the same guarantee.
template <typename T>
class AtomicSharedPtr {
public:
    explicit AtomicSharedPtr(std::shared_ptr<T> p) : p_(std::move(p)) {}
#if defined(__cpp_lib_atomic_shared_ptr)
    std::shared_ptr<T> load(std::memory_order o) const noexcept { return p_.load(o); }
    void store(std::shared_ptr<T> p, std::memory_order o) noexcept { p_.store(std::move(p), o); }

private:
    std::atomic<std::shared_ptr<T>> p_;
#else
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
    std::shared_ptr<T> load(std::memory_order o) const noexcept { return std::atomic_load_explicit(&p_, o); }
    void store(std::shared_ptr<T> p, std::memory_order o) noexcept { std::atomic_store_explicit(&p_, std::move(p), o); }
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

private:
    std::shared_ptr<T> p_;
#endif
};

struct Registry {
    std::mutex write_mutex; // serializes writers; readers only load `current`
    size_t live_entries = 0;
    size_t dead_entries = 0; // still indexed by a segment
    AtomicSharedPtr<const CodeStackMapSnapshot> current{
        std::make_shared<const CodeStackMapSnapshot>()};
};

// Never destroyed: code may be unregistered from static destructors.
Registry& registry() {
    static auto* r = new Registry();
    return *r;
}

bool code_range(const FunctionStackMap& fn, uintptr_t& begin, uintptr_t& end) {
    if (fn.function_address == 0 || fn.records.empty()) return false;
    begin = fn.function_address;
    if (fn.code_size > 0) {
        end = begin + fn.code_size;
    } else {
        // Without a size, the code is known to extend past its last record.
        uint32_t last = 0;
        for (const auto& r : fn.records) last = std::max(last, r.instruction_offset);
        end = begin + static_cast<uintptr_t>(last) + 1;
    }
    return true;
}

[[noreturn]] void fatal_overlap(const FunctionStackMap& fn, const FunctionStackMap& other) {
    std::fprintf(stderr,
                 "brass: fatal: stack maps registered for '%s' at %p overlap the registered code of '%s' at %p; "
                 "the owner of freed code did not drop its stack map registration\n",
                 fn.function_name.c_str(), reinterpret_cast<void*>(fn.function_address),
                 other.function_name.c_str(), reinterpret_cast<void*>(other.function_address));
    std::fflush(stderr);
    std::abort();
}

bool is_alive(const Entry& e) noexcept { return e.owner->alive.load(std::memory_order_acquire); }

// The live entry of `seg` that overlaps [begin, end), or null.
const Entry* live_overlap(const Segment& seg, uintptr_t begin, uintptr_t end) noexcept {
    const auto& v = seg.entries;
    auto it = std::lower_bound(v.begin(), v.end(), end, [](const Entry& e, uintptr_t x) { return e.begin < x; });
    // Entries are disjoint and sorted, so their ends are sorted too: only
    // those starting before `end` and ending after `begin` overlap, and they
    // are the last ones before `it`.
    while (it != v.begin()) {
        --it;
        if (it->end <= begin) return nullptr;
        if (is_alive(*it)) return &*it;
    }
    return nullptr;
}

// Merges segments into one, dropping dead entries (and their owners).
std::shared_ptr<const Segment> merge(const std::vector<std::shared_ptr<const Segment>>& parts, size_t& dropped) {
    auto out = std::make_shared<Segment>();
    size_t total = 0;
    for (const auto& p : parts) total += p->entries.size();
    out->entries.reserve(total);
    for (const auto& p : parts) {
        for (const auto& e : p->entries) {
            if (is_alive(e)) {
                out->entries.push_back(e);
            } else {
                ++dropped;
            }
        }
        for (const auto& o : p->owners) {
            if (o->alive.load(std::memory_order_acquire)) out->owners.push_back(o);
        }
    }
    std::sort(out->entries.begin(), out->entries.end(), [](const Entry& a, const Entry& b) { return a.begin < b.begin; });
    // A registration's entries may span several segments: one owner each.
    std::sort(out->owners.begin(), out->owners.end());
    out->owners.erase(std::unique(out->owners.begin(), out->owners.end()), out->owners.end());
    return out;
}

// Publishes `segs`, first merging the newest segments while the one below is
// no more than twice as large (a binary counter of segment sizes), or all of
// them once dead entries outnumber live ones. Holds write_mutex.
void publish_segments(Registry& r, std::vector<std::shared_ptr<const Segment>> segs) {
    if (r.dead_entries > 64 && r.dead_entries > r.live_entries && !segs.empty()) {
        size_t dropped = 0;
        auto all = merge(segs, dropped);
        r.dead_entries -= dropped;
        segs.clear();
        if (!all->entries.empty()) segs.push_back(std::move(all));
    }
    while (segs.size() >= 2 && segs[segs.size() - 2]->entries.size() <= 2 * segs.back()->entries.size()) {
        size_t dropped = 0;
        auto merged = merge({segs[segs.size() - 2], segs.back()}, dropped);
        r.dead_entries -= dropped;
        segs.pop_back();
        segs.pop_back();
        if (!merged->entries.empty()) segs.push_back(std::move(merged));
    }
    auto next = std::make_shared<CodeStackMapSnapshot>();
    next->segments = std::move(segs);
    r.current.store(std::move(next), std::memory_order_release);
}

// Returned to the registrant: dropping it unregisters the maps.
struct Token {
    std::shared_ptr<RegistrationData> data;
    ~Token();
};

std::shared_ptr<const void> publish(std::shared_ptr<RegistrationData> data) {
    auto seg = std::make_shared<Segment>();
    for (const auto& fn : data->maps) {
        Entry e;
        if (!code_range(fn, e.begin, e.end)) continue;
        e.map = &fn;
        e.owner = data.get();
        seg->entries.push_back(e);
    }
    std::sort(seg->entries.begin(), seg->entries.end(), [](const Entry& a, const Entry& b) { return a.begin < b.begin; });
    for (size_t i = 1; i < seg->entries.size(); ++i) {
        if (seg->entries[i].begin < seg->entries[i - 1].end) {
            fatal_overlap(*seg->entries[i].map, *seg->entries[i - 1].map);
        }
    }
    seg->owners.push_back(data);

    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.write_mutex);
    auto old = r.current.load(std::memory_order_acquire);
    for (const auto& e : seg->entries) {
        for (const auto& s : old->segments) {
            if (const Entry* other = live_overlap(*s, e.begin, e.end)) fatal_overlap(*e.map, *other->map);
        }
    }
    r.live_entries += seg->entries.size();
    auto segs = old->segments;
    segs.push_back(std::move(seg));
    publish_segments(r, std::move(segs));
    auto token = std::make_shared<Token>();
    token->data = std::move(data);
    return token;
}

Token::~Token() {
    if (!data) return;
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.write_mutex);
    // Every map of a registration has a code range: one entry each.
    const size_t n = data->maps.size();
    data->alive.store(false, std::memory_order_release);
    r.live_entries -= n;
    r.dead_entries += n;
    if (r.dead_entries > 64 && r.dead_entries > r.live_entries) {
        publish_segments(r, r.current.load(std::memory_order_acquire)->segments);
    }
}

} // namespace

std::shared_ptr<const void> register_code_stack_maps(const ModuleStackMap& maps) {
    auto data = std::make_shared<RegistrationData>();
    data->maps.reserve(maps.size());
    for (const auto& fn : maps.functions()) {
        uintptr_t b = 0, e = 0;
        if (code_range(fn, b, e)) data->maps.push_back(fn);
    }
    if (data->maps.empty()) return nullptr;
    return publish(std::move(data));
}

std::shared_ptr<const void> register_code_stack_map(const FunctionStackMap& map) {
    uintptr_t b = 0, e = 0;
    if (!code_range(map, b, e)) return nullptr;
    auto data = std::make_shared<RegistrationData>();
    data->maps.push_back(map);
    return publish(std::move(data));
}

std::shared_ptr<const CodeStackMapSnapshot> code_stack_map_snapshot() noexcept {
    return registry().current.load(std::memory_order_acquire);
}

const FunctionStackMap* find_code_stack_map(const CodeStackMapSnapshot* snapshot, uintptr_t ip) noexcept {
    if (!snapshot) return nullptr;
    // Live entries are disjoint across segments: at most one contains ip.
    for (auto s = snapshot->segments.rbegin(); s != snapshot->segments.rend(); ++s) {
        const auto& v = (*s)->entries;
        auto it = std::upper_bound(v.begin(), v.end(), ip, [](uintptr_t x, const Entry& e) { return x < e.begin; });
        if (it == v.begin()) continue;
        --it;
        if (ip < it->end && is_alive(*it)) return it->map;
    }
    return nullptr;
}

bool code_stack_maps_cover(uintptr_t ip) noexcept {
    auto snap = code_stack_map_snapshot();
    return find_code_stack_map(snap.get(), ip) != nullptr;
}

size_t code_stack_map_segment_count() noexcept {
    return code_stack_map_snapshot()->segments.size();
}

} // namespace brass
