#include <brass/gc/code_stack_maps.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace brass {

class CodeStackMapSnapshot {
public:
    struct Entry {
        uintptr_t begin = 0;
        uintptr_t end = 0; // exclusive
        const FunctionStackMap* map = nullptr;
        uint64_t owner = 0;
    };
    std::vector<Entry> entries; // sorted by begin, disjoint
};

namespace {

// The maps of one registration. Snapshot entries point into `maps`; an entry
// is only dereferenced for an address on the walking thread's stack, which is
// code that is still loaded, so still registered.
struct Registration {
    uint64_t id = 0;
    std::vector<FunctionStackMap> maps;
    ~Registration();
};

struct Registry {
    std::mutex write_mutex; // serializes writers; readers only load `current`
    uint64_t next_id = 1;
    std::atomic<std::shared_ptr<const CodeStackMapSnapshot>> current{
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

std::shared_ptr<const void> publish(std::shared_ptr<Registration> reg) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.write_mutex);
    reg->id = r.next_id++;
    auto old = r.current.load(std::memory_order_acquire);
    auto next = std::make_shared<CodeStackMapSnapshot>(*old);
    for (const auto& fn : reg->maps) {
        CodeStackMapSnapshot::Entry e;
        if (!code_range(fn, e.begin, e.end)) continue;
        e.map = &fn;
        e.owner = reg->id;
        next->entries.push_back(e);
    }
    std::sort(next->entries.begin(), next->entries.end(),
              [](const auto& a, const auto& b) { return a.begin < b.begin; });
    for (size_t i = 1; i < next->entries.size(); ++i) {
        if (next->entries[i].begin < next->entries[i - 1].end) {
            fatal_overlap(*next->entries[i].map, *next->entries[i - 1].map);
        }
    }
    r.current.store(std::move(next), std::memory_order_release);
    return reg;
}

Registration::~Registration() {
    if (id == 0) return;
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.write_mutex);
    auto old = r.current.load(std::memory_order_acquire);
    auto next = std::make_shared<CodeStackMapSnapshot>();
    next->entries.reserve(old->entries.size());
    for (const auto& e : old->entries) {
        if (e.owner != id) next->entries.push_back(e);
    }
    r.current.store(std::move(next), std::memory_order_release);
}

} // namespace

std::shared_ptr<const void> register_code_stack_maps(const ModuleStackMap& maps) {
    auto reg = std::make_shared<Registration>();
    reg->maps.reserve(maps.size());
    for (const auto& fn : maps.functions()) {
        uintptr_t b = 0, e = 0;
        if (code_range(fn, b, e)) reg->maps.push_back(fn);
    }
    if (reg->maps.empty()) return nullptr;
    return publish(std::move(reg));
}

std::shared_ptr<const void> register_code_stack_map(const FunctionStackMap& map) {
    uintptr_t b = 0, e = 0;
    if (!code_range(map, b, e)) return nullptr;
    auto reg = std::make_shared<Registration>();
    reg->maps.push_back(map);
    return publish(std::move(reg));
}

std::shared_ptr<const CodeStackMapSnapshot> code_stack_map_snapshot() noexcept {
    return registry().current.load(std::memory_order_acquire);
}

const FunctionStackMap* find_code_stack_map(const CodeStackMapSnapshot* snapshot, uintptr_t ip) noexcept {
    if (!snapshot || snapshot->entries.empty()) return nullptr;
    const auto& entries = snapshot->entries;
    auto it = std::upper_bound(entries.begin(), entries.end(), ip,
                               [](uintptr_t v, const auto& e) { return v < e.begin; });
    if (it == entries.begin()) return nullptr;
    --it;
    return ip < it->end ? it->map : nullptr;
}

bool code_stack_maps_cover(uintptr_t ip) noexcept {
    auto snap = code_stack_map_snapshot();
    return find_code_stack_map(snap.get(), ip) != nullptr;
}

} // namespace brass
