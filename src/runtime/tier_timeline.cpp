// The tiering timeline (tier_timeline.hpp): timed compile and deopt events,
// and the report that reads time-to-optimized-code and stalls off them.

#include <brass/runtime/tier_timeline.hpp>
#include <brass/runtime/tiering.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace brass::runtime {

namespace detail {
std::atomic<bool> g_tier_timeline_on{false};
thread_local bool t_on_compile_worker = false;
}  // namespace detail

namespace {

using Clock = std::chrono::steady_clock;

struct Timeline {
    std::mutex mutex;
    Clock::time_point epoch = Clock::now();
    std::vector<TierEvent> events;
};

Timeline& timeline() {
    // Leaked: compile workers may record while statics are torn down.
    static Timeline* t = new Timeline();
    return *t;
}

// The variable's value; empty when it is unset or "0".
std::string env_value(const char* name) {
#if defined(_MSC_VER)
    char* owned = nullptr;
    size_t len = 0;
    if (_dupenv_s(&owned, &len, name) != 0 || !owned) return {};
    std::string v(owned);
    std::free(owned);
#else
    const char* raw = std::getenv(name);
    std::string v = raw ? raw : "";
#endif
    return v == "0" ? std::string() : v;
}

double ms(uint64_t ns) { return static_cast<double>(ns) / 1e6; }

}  // namespace

std::string_view to_string(TierEventKind kind) noexcept {
    switch (kind) {
        case TierEventKind::Tier1Compile: return "tier1";
        case TierEventKind::Tier2Enqueue: return "tier2-enqueue";
        case TierEventKind::Tier2Compile: return "tier2";
        case TierEventKind::OsrRequest: return "osr-request";
        case TierEventKind::OsrCompile: return "osr";
        case TierEventKind::OsrEnter: return "osr-enter";
        case TierEventKind::Deopt: return "deopt";
        case TierEventKind::Invalidate: return "invalidate";
        case TierEventKind::FrontPass: return "front-pass";
    }
    return "?";
}

void enable_tier_timeline(bool on) {
    if (on) clear_tier_timeline();
    detail::g_tier_timeline_on.store(on, std::memory_order_relaxed);
}

void enable_tier_timeline_from_env() {
    static std::once_flag once;
    std::call_once(once, [] {
        if (!env_value("BRASS_TIER_LOG").empty()) enable_tier_timeline(true);
    });
}

void clear_tier_timeline() {
    Timeline& t = timeline();
    std::lock_guard<std::mutex> lock(t.mutex);
    t.events.clear();
    t.epoch = Clock::now();
}

uint64_t tier_timeline_now_ns() noexcept {
    const auto d = Clock::now() - timeline().epoch;
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(d).count();
    return ns > 0 ? static_cast<uint64_t>(ns) : 0;
}

bool on_compile_worker() noexcept { return detail::t_on_compile_worker; }

void record_tier_event(TierEventKind kind, std::string_view name, uint64_t start_ns, uint64_t dur_ns, bool ok) {
    if (!tier_timeline_enabled()) return;
    TierEvent e;
    e.kind = kind;
    e.on_worker = on_compile_worker();
    e.ok = ok;
    e.start_ns = start_ns;
    e.dur_ns = dur_ns;
    e.name.assign(name);
    Timeline& t = timeline();
    std::lock_guard<std::mutex> lock(t.mutex);
    t.events.push_back(std::move(e));
}

TierEventScope::~TierEventScope() {
    if (!on_) return;
    const uint64_t end = tier_timeline_now_ns();
    record_tier_event(kind_, name_, start_, end > start_ ? end - start_ : 0, ok_);
}

std::vector<TierEvent> tier_timeline_snapshot() {
    Timeline& t = timeline();
    std::lock_guard<std::mutex> lock(t.mutex);
    return t.events;
}

TierTimelineSummary summarize_tier_timeline(const TieringRegistry* registry, size_t hot_count) {
    TierTimelineSummary s;
    const std::vector<TierEvent> events = tier_timeline_snapshot();
    s.wall_ms = ms(tier_timeline_now_ns());

    // Per function: when its first baseline and optimized code was ready.
    std::unordered_map<std::string, double> tier1_at, tier2_at, requested_at;
    auto first = [](std::unordered_map<std::string, double>& m, const std::string& name, double at) {
        auto [it, fresh] = m.emplace(name, at);
        if (!fresh && at < it->second) it->second = at;
    };
    for (const TierEvent& e : events) {
        const double end = ms(e.start_ns + e.dur_ns);
        const bool compile = e.kind == TierEventKind::Tier1Compile || e.kind == TierEventKind::Tier2Compile ||
                             e.kind == TierEventKind::OsrCompile;
        // Copying a function out for a compile is compile work too.
        const bool copy = e.kind == TierEventKind::Tier2Enqueue || e.kind == TierEventKind::OsrRequest;
        if (compile || copy) {
            s.compile_ms += ms(e.dur_ns);
            if (e.on_worker) s.worker_compile_ms += ms(e.dur_ns);
        }
        if (compile && !e.ok) ++s.failed_compiles;
        // The front pass runs inside a tier-2 or OSR compile: its time is
        // already that compile's.
        if (!e.on_worker && e.dur_ns && e.kind != TierEventKind::FrontPass) {
            s.stall_ms += ms(e.dur_ns);
            s.max_stall_ms = std::max(s.max_stall_ms, ms(e.dur_ns));
        }
        // The request: the program's thread asking (a synchronous tier-2
        // compile asks by starting it).
        if (e.kind == TierEventKind::Tier2Enqueue || e.kind == TierEventKind::Tier2Compile ||
            e.kind == TierEventKind::OsrRequest) {
            if (!e.on_worker) first(requested_at, e.name, ms(e.start_ns));
        }
        switch (e.kind) {
            case TierEventKind::Tier1Compile:
                ++s.tier1_compiles;
                if (e.ok) first(tier1_at, e.name, end);
                break;
            case TierEventKind::Tier2Compile:
            case TierEventKind::OsrCompile:
                ++(e.kind == TierEventKind::Tier2Compile ? s.tier2_compiles : s.osr_compiles);
                if (e.ok) {
                    first(tier2_at, e.name, end);
                    if (s.first_tier2_ms < 0 || end < s.first_tier2_ms) s.first_tier2_ms = end;
                }
                break;
            case TierEventKind::Deopt: ++s.deopts; break;
            case TierEventKind::Invalidate: ++s.invalidations; break;
            default: break;
        }
    }

    if (registry) {
        // Hot: crossed the tier-2 invocation threshold or the OSR backedge
        // threshold, the functions tiering means to optimize.
        std::vector<TierHotFunction> hot;
        const TieringConfig& cfg = registry->default_config();
        registry->for_each_feedback([&](const TieringFeedback& fb) {
            const uint64_t inv = fb.invocation_count();
            const uint64_t be = fb.backedge_count();
            if (inv < cfg.invocation_tier2_threshold && be < cfg.backedge_osr_threshold) return;
            TierHotFunction h;
            h.name.assign(fb.function_name());
            h.invocations = inv;
            h.backedges = be;
            hot.push_back(std::move(h));
        });
        std::sort(hot.begin(), hot.end(), [](const TierHotFunction& a, const TierHotFunction& b) {
            const uint64_t wa = a.invocations + a.backedges, wb = b.invocations + b.backedges;
            return wa != wb ? wa > wb : a.name < b.name;
        });
        if (hot.size() > hot_count) hot.resize(hot_count);
        for (TierHotFunction& h : hot) {
            if (auto it = tier1_at.find(h.name); it != tier1_at.end()) h.tier1_ms = it->second;
            if (auto it = requested_at.find(h.name); it != requested_at.end()) h.requested_ms = it->second;
            if (auto it = tier2_at.find(h.name); it != tier2_at.end()) {
                h.tier2_ms = it->second;
                s.hot_tier2_ms = std::max(s.hot_tier2_ms, h.tier2_ms);
                if (h.requested_ms >= 0) s.hot_latency_ms = std::max(s.hot_latency_ms, h.tier2_ms - h.requested_ms);
            } else {
                ++s.hot_untiered;
            }
        }
        s.hot = std::move(hot);
    }
    return s;
}

void write_tier_report(std::ostream& os, const TierTimelineSummary& s, bool events) {
    const std::ios::fmtflags flags = os.flags();
    const std::streamsize precision = os.precision();
    os << std::fixed << std::setprecision(3);
    os << "tier-timeline wall_ms=" << s.wall_ms << " first_tier2_ms=" << s.first_tier2_ms
       << " hot_tier2_ms=" << s.hot_tier2_ms << " hot_latency_ms=" << s.hot_latency_ms
       << " hot_untiered=" << s.hot_untiered
       << " compile_ms=" << s.compile_ms << " worker_compile_ms=" << s.worker_compile_ms
       << " stall_ms=" << s.stall_ms << " max_stall_ms=" << s.max_stall_ms << " tier1=" << s.tier1_compiles
       << " tier2=" << s.tier2_compiles << " osr=" << s.osr_compiles << " failed=" << s.failed_compiles
       << " deopts=" << s.deopts << " invalidations=" << s.invalidations << "\n";
    for (const TierHotFunction& h : s.hot) {
        os << "tier-timeline hot " << h.name << " invocations=" << h.invocations << " backedges=" << h.backedges
           << " tier1_ms=" << h.tier1_ms << " requested_ms=" << h.requested_ms << " tier2_ms=" << h.tier2_ms
           << "\n";
    }
    if (events) {
        for (const TierEvent& e : tier_timeline_snapshot()) {
            os << "tier-timeline event " << to_string(e.kind) << " " << e.name << " at_ms=" << ms(e.start_ns)
               << " dur_ms=" << ms(e.dur_ns) << (e.on_worker ? " worker" : " mutator") << (e.ok ? "" : " failed")
               << "\n";
        }
    }
    os.flags(flags);
    os.precision(precision);
}

void emit_tier_report_from_env(const TieringRegistry& registry) {
    const std::string where = env_value("BRASS_TIER_LOG");
    if (where.empty() || !tier_timeline_enabled()) return;
    std::ostringstream out;
    write_tier_report(out, summarize_tier_timeline(&registry), !env_value("BRASS_TIER_LOG_EVENTS").empty());
    if (where == "1") {
        std::fputs(out.str().c_str(), stderr);
        std::fflush(stderr);
        return;
    }
    std::ofstream file(where, std::ios::app);
    file << out.str();
}

}  // namespace brass::runtime
