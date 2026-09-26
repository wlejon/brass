// The JIT code registry (include/brass/debug/jit_code_registry.hpp): which
// named function of which tier every JIT address belongs to, kept for exactly
// as long as the code lives, and fanned out to the native tools asked for.

#include <brass/debug/jit_code_registry.hpp>
#include "jit_sinks.hpp"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>

namespace brass::debug {

namespace {

std::string env_value(const char* name) {
#if defined(_MSC_VER)
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

bool env_on(const char* name, bool fallback) {
    const std::string v = env_value(name);
    if (v.empty()) return fallback;
    return !(v == "0" || v == "false" || v == "off" || v == "no");
}

JitProfilerConfig read_config() {
    JitProfilerConfig c;
    c.perf_map = env_on("BRASS_PERF_MAP", false);
    c.jitdump = env_on("BRASS_JITDUMP", false);
    c.gdb_jit = env_on("BRASS_GDB_JIT", false);
    // VTune sets this when it collects from a process it launched or attached.
#if defined(_WIN64) || defined(__x86_64__) || defined(__aarch64__) || defined(_M_ARM64)
    c.vtune = !env_value("INTEL_JIT_PROFILER64").empty();
#else
    c.vtune = !env_value("INTEL_JIT_PROFILER32").empty();
#endif
#if defined(_WIN32)
    c.crash_report = env_on("BRASS_JIT_CRASH_REPORT", true);
#else
    c.crash_report = env_on("BRASS_JIT_CRASH_REPORT", false);
#endif
    c.perf_dir = env_value("BRASS_PERF_DIR");
    if (c.perf_dir.empty()) {
#if defined(_WIN32)
        c.perf_dir = env_value("TEMP");
        if (c.perf_dir.empty()) c.perf_dir = ".";
#else
        c.perf_dir = "/tmp";
#endif
    }
    return c;
}

struct Entry {
    std::string name;
    size_t size = 0;
    JitTier tier = JitTier::Optimized;
    uint64_t serial = 0;
};

struct Registry {
    std::mutex mutex;          // the map
    std::mutex sink_mutex;     // the tools' files and lists
    std::map<uintptr_t, Entry> by_start;
    uint64_t next_serial = 1;
    JitProfilerConfig config;
    bool config_read = false;
    bool any_sink = false;
};

Registry& registry() {
    // Never destroyed: code registered by static objects may be released
    // after this translation unit's statics are gone.
    static Registry* r = new Registry();
    return *r;
}

void ensure_config(Registry& r) {
    if (r.config_read) return;
    r.config = read_config();
    r.config_read = true;
    r.any_sink = r.config.perf_map || r.config.jitdump || r.config.gdb_jit || r.config.vtune;
}

// What a registration handed out; its destruction undoes it.
struct Registration {
    std::vector<std::pair<uintptr_t, uint64_t>> keys;
    void* gdb_entry = nullptr;
    std::vector<uint32_t> vtune_ids;

    ~Registration() {
        Registry& r = registry();
        if (gdb_entry || !vtune_ids.empty()) {
            std::lock_guard<std::mutex> lock(r.sink_mutex);
            if (gdb_entry) detail::gdb_jit_unregister(gdb_entry);
            if (!vtune_ids.empty()) detail::vtune_unregister(vtune_ids);
        }
        std::lock_guard<std::mutex> lock(r.mutex);
        for (const auto& [start, serial] : keys) {
            auto it = r.by_start.find(start);
            // A later registration at the same address (the code was freed
            // and the address reused before this handle went) keeps its entry.
            if (it != r.by_start.end() && it->second.serial == serial) r.by_start.erase(it);
        }
    }
};

const Entry* find_locked(const Registry& r, uintptr_t ip, uintptr_t* start) {
    auto it = r.by_start.upper_bound(ip);
    if (it == r.by_start.begin()) return nullptr;
    --it;
    if (ip - it->first >= it->second.size) return nullptr;
    *start = it->first;
    return &it->second;
}

} // namespace

const char* jit_tier_label(JitTier tier) noexcept {
    switch (tier) {
        case JitTier::Stub: return "stub";
        case JitTier::Baseline: return "tier 1";
        case JitTier::Optimized: return "tier 2";
        case JitTier::Osr: return "tier 2 osr";
    }
    return "jit";
}

namespace detail {
std::string tool_name(std::string_view name, JitTier tier) {
    std::string s(name.empty() ? std::string_view("<anonymous>") : name);
    s += " [";
    s += jit_tier_label(tier);
    s += "]";
    return s;
}
} // namespace detail

std::shared_ptr<const void> register_jit_code(JitTier tier, const JitCodeRange* ranges, size_t count) {
    Registry& r = registry();
    auto reg = std::make_shared<Registration>();
    bool crash_report = false;
    bool any_sink = false;
    JitProfilerConfig cfg;
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        ensure_config(r);
        for (size_t i = 0; i < count; ++i) {
            const JitCodeRange& range = ranges[i];
            if (!range.code || range.size == 0) continue;
            const auto start = reinterpret_cast<uintptr_t>(range.code);
            Entry& e = r.by_start[start];
            e.name.assign(range.name.data(), range.name.size());
            e.size = range.size;
            e.tier = tier;
            e.serial = r.next_serial++;
            reg->keys.emplace_back(start, e.serial);
        }
        crash_report = r.config.crash_report;
        any_sink = r.any_sink;
        if (any_sink) cfg = r.config;
    }
    if (reg->keys.empty()) return nullptr;
    if (crash_report && !jit_crash_report_installed()) install_jit_crash_report();
    if (any_sink) {
        std::lock_guard<std::mutex> lock(r.sink_mutex);
        if (cfg.perf_map) detail::perf_map_append(cfg, tier, ranges, count);
        if (cfg.jitdump) detail::jitdump_append(cfg, tier, ranges, count);
        if (cfg.gdb_jit) reg->gdb_entry = detail::gdb_jit_register(detail::build_gdb_symfile(tier, ranges, count));
        if (cfg.vtune) reg->vtune_ids = detail::vtune_register(tier, ranges, count);
    }
    return reg;
}

std::shared_ptr<const void> register_jit_code(JitTier tier, std::string_view name, const void* code, size_t size) {
    const JitCodeRange range{name, code, size};
    return register_jit_code(tier, &range, 1);
}

bool find_jit_code(uintptr_t ip, JitCodeInfo* out) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    uintptr_t start = 0;
    const Entry* e = find_locked(r, ip, &start);
    if (!e) return false;
    if (out) {
        out->name = e->name;
        out->start = start;
        out->size = e->size;
        out->tier = e->tier;
    }
    return true;
}

bool find_jit_code_nonblocking(uintptr_t ip, char* name, size_t name_cap, JitTier* tier, uintptr_t* start) noexcept {
    Registry& r = registry();
    if (!r.mutex.try_lock()) return false;
    uintptr_t s = 0;
    const Entry* e = find_locked(r, ip, &s);
    if (e) {
        if (name && name_cap) {
            const size_t n = e->name.size() < name_cap - 1 ? e->name.size() : name_cap - 1;
            std::memcpy(name, e->name.data(), n);
            name[n] = '\0';
        }
        if (tier) *tier = e->tier;
        if (start) *start = s;
    }
    r.mutex.unlock();
    return e != nullptr;
}

std::string describe_jit_address(uintptr_t ip) {
    JitCodeInfo info;
    if (!find_jit_code(ip, &info)) return {};
    char off[32];
    std::snprintf(off, sizeof off, "+0x%llx", static_cast<unsigned long long>(ip - info.start));
    return detail::tool_name(info.name, info.tier) + off;
}

size_t jit_code_count() {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    return r.by_start.size();
}

const JitProfilerConfig& jit_profiler_config() {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    ensure_config(r);
    return r.config;
}

void reload_jit_profiler_config() {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    r.config_read = false;
    ensure_config(r);
}

std::string perf_map_path() {
    return jit_profiler_config().perf_dir + "/perf-" + std::to_string(detail::current_pid()) + ".map";
}

std::string jitdump_path() {
    return jit_profiler_config().perf_dir + "/jit-" + std::to_string(detail::current_pid()) + ".dump";
}

} // namespace brass::debug
