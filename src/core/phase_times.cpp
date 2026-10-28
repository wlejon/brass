#include <brass/core/phase_times.hpp>
#include <brass/core/env.hpp>

#include <algorithm>
#include <cstdio>
#if defined(_MSC_VER)
#include <share.h>
#endif
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace brass {

namespace {

struct PhaseTotal {
    uint64_t ns = 0;
    uint64_t count = 0;
};

struct PhaseTable {
    std::mutex mutex;
    std::map<std::string, PhaseTotal, std::less<>> totals;

    ~PhaseTable() {
        if (totals.empty()) return;
        const std::string where = env_value("BRASS_PASS_TIMES");
        FILE* out = stderr;
        if (!where.empty() && where != "1") {
#if defined(_MSC_VER)
            FILE* f = _fsopen(where.c_str(), "a", _SH_DENYNO);
#else
            FILE* f = std::fopen(where.c_str(), "a");
#endif
            if (f) out = f;
        }
        std::vector<std::pair<std::string, PhaseTotal>> rows(totals.begin(), totals.end());
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
        std::fprintf(out, "brass phase times (wall, summed over threads):\n");
        for (const auto& [name, t] : rows) {
            std::fprintf(out, "  %10.3f ms  %8llu x  %s\n", static_cast<double>(t.ns) / 1e6,
                         static_cast<unsigned long long>(t.count), name.c_str());
        }
        if (out != stderr) std::fclose(out);
    }
};

PhaseTable& table() {
    static PhaseTable t;
    return t;
}

thread_local std::string_view current_subject;

const std::string& subject_filter() {
    static const std::string f = env_value("BRASS_PASS_TIMES_FUNCTION");
    return f;
}

} // namespace

bool phase_times_enabled() noexcept {
    static const bool on = [] {
        const std::string v = env_value("BRASS_PASS_TIMES");
        if (v.empty() || v == "0") return false;
        table(); // constructed before any user, so it is destroyed after them
        return true;
    }();
    return on;
}

PhaseTimeSubject::PhaseTimeSubject(std::string_view subject) noexcept : on_(phase_times_enabled()) {
    if (!on_) return;
    previous_ = current_subject;
    current_subject = subject;
}

PhaseTimeSubject::~PhaseTimeSubject() {
    if (on_) current_subject = previous_;
}

void record_phase_time(std::string_view name, uint64_t ns) noexcept {
    if (const std::string& only = subject_filter(); !only.empty() && current_subject != only) return;
    PhaseTable& t = table();
    std::lock_guard<std::mutex> lock(t.mutex);
    auto it = t.totals.find(name);
    if (it == t.totals.end()) it = t.totals.emplace(std::string(name), PhaseTotal{}).first;
    it->second.ns += ns;
    ++it->second.count;
}

} // namespace brass
