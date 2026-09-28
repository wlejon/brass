// brass's own copy of the unwind tables it registers with the OS, for a
// sampler that must not block on ntdll's lock (jit_unwind_registry.hpp).

#include <brass/debug/jit_unwind_registry.hpp>

#include <algorithm>
#include <mutex>
#include <vector>

#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__) || defined(_M_ARM64) || defined(__aarch64__))
#define BRASS_UNWIND_REGISTRY 1
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace brass::debug {

#if defined(BRASS_UNWIND_REGISTRY)

namespace {

struct Table {
    uintptr_t lo = 0;  // lowest covered address
    uintptr_t hi = 0;  // one past the highest
    const RUNTIME_FUNCTION* entries = nullptr;
    uint32_t count = 0;
    uintptr_t base = 0;
};

struct Registry {
    std::mutex mutex;
    std::vector<Table> tables;  // sorted by lo
};

Registry& registry() {
    static auto* r = new Registry();
    return *r;
}

uint32_t entryEnd(const RUNTIME_FUNCTION& f) {
#if defined(_M_ARM64) || defined(__aarch64__)
    // ARM64 packs the length into UnwindData when Flag != 0; a full xdata
    // record holds it otherwise. Callers only need a bound, so a packed
    // length is used when present and the next entry bounds the rest.
    const uint32_t flag = f.UnwindData & 3u;
    if (flag != 0) return f.BeginAddress + ((f.UnwindData >> 2) & 0x7FFu) * 4u;
    return f.BeginAddress + 4u;
#else
    return f.EndAddress;
#endif
}

}  // namespace

void register_jit_unwind_table(const void* table, uint32_t count, uintptr_t image_base) noexcept {
    if (!table || count == 0) return;
    const auto* rf = static_cast<const RUNTIME_FUNCTION*>(table);
    Table t;
    t.entries = rf;
    t.count = count;
    t.base = image_base;
    t.lo = UINTPTR_MAX;
    for (uint32_t i = 0; i < count; ++i) {
        t.lo = std::min<uintptr_t>(t.lo, image_base + rf[i].BeginAddress);
        t.hi = std::max<uintptr_t>(t.hi, image_base + entryEnd(rf[i]));
    }
    try {
        Registry& r = registry();
        std::lock_guard<std::mutex> lock(r.mutex);
        auto at = std::upper_bound(r.tables.begin(), r.tables.end(), t.lo,
                                   [](uintptr_t v, const Table& e) { return v < e.lo; });
        r.tables.insert(at, t);
    } catch (...) {
        // Out of memory: the table is only unknown to the sampler, whose walk
        // then stops at this code's frames.
    }
}

void unregister_jit_unwind_table(const void* table) noexcept {
    if (!table) return;
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.tables.erase(std::remove_if(r.tables.begin(), r.tables.end(),
                                  [table](const Table& t) { return t.entries == table; }),
                   r.tables.end());
}

bool find_jit_unwind_entry_nonblocking(uintptr_t ip, const void** function_entry, uintptr_t* image_base) noexcept {
    Registry& r = registry();
    if (!r.mutex.try_lock()) return false;
    bool found = false;
    // Tables are sorted by their low bound; ranges of different blocks do not
    // overlap, so the candidate is the last table starting at or before ip.
    auto it = std::upper_bound(r.tables.begin(), r.tables.end(), ip,
                               [](uintptr_t v, const Table& e) { return v < e.lo; });
    if (it != r.tables.begin()) {
        const Table& t = *(it - 1);
        if (ip < t.hi) {
            const uint32_t rva = static_cast<uint32_t>(ip - t.base);
            for (uint32_t i = 0; i < t.count; ++i) {
                const RUNTIME_FUNCTION& f = t.entries[i];
                if (rva >= f.BeginAddress && rva < entryEnd(f)) {
                    if (function_entry) *function_entry = &f;
                    if (image_base) *image_base = t.base;
                    found = true;
                    break;
                }
            }
        }
    }
    r.mutex.unlock();
    return found;
}

#else

void register_jit_unwind_table(const void*, uint32_t, uintptr_t) noexcept {}
void unregister_jit_unwind_table(const void*) noexcept {}
bool find_jit_unwind_entry_nonblocking(uintptr_t, const void**, uintptr_t*) noexcept { return false; }

#endif

}  // namespace brass::debug
