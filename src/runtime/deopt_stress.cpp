// Deopt stress (include/brass/runtime/deopt_stress.hpp): the configuration,
// read from BRASS_DEOPT_STRESS once, and the counters every tier shares.
#include <brass/runtime/deopt_stress.hpp>

#include <cstdlib>
#include <string>

namespace brass::runtime {

namespace detail {
std::atomic<uint64_t> g_deopt_stress_period{0};
}

namespace {

std::atomic<int64_t> g_resume_id{-1};
// Plain words that native code increments in place (not atomically: stress
// counting across racing threads may drift, which only moves which
// evaluation fails); C++ updates them through atomic_ref.
alignas(8) uint64_t g_evaluations = 0;
alignas(8) uint64_t g_forced = 0;

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

uint64_t parse_u64(const std::string& s, uint64_t fallback) {
    if (s.empty()) return fallback;
    char* end = nullptr;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    return (end && *end == '\0') ? static_cast<uint64_t>(v) : fallback;
}

DeoptStressConfig parse_environment() {
    DeoptStressConfig cfg;
    const std::string v = env_value("BRASS_DEOPT_STRESS");
    if (v.empty() || v == "0" || v == "none") return cfg;
    if (v == "all") {
        cfg.period = 1;
        return cfg;
    }
    if (v.rfind("site:", 0) == 0) {
        const std::string rest = v.substr(5);
        const size_t colon = rest.find(':');
        const std::string id = colon == std::string::npos ? rest : rest.substr(0, colon);
        const std::string period = colon == std::string::npos ? std::string("1") : rest.substr(colon + 1);
        const uint64_t rid = parse_u64(id, UINT64_MAX);
        const uint64_t p = parse_u64(period, 0);
        if (rid <= UINT32_MAX && p != 0) {
            cfg.resume_id = static_cast<int64_t>(rid);
            cfg.period = p;
        }
        return cfg;
    }
    cfg.period = parse_u64(v, 0);
    return cfg;
}

struct EnvironmentInit {
    EnvironmentInit() { set_deopt_stress(parse_environment()); }
};
const EnvironmentInit g_environment_init;

} // namespace

DeoptStressConfig deopt_stress_config() noexcept {
    DeoptStressConfig cfg;
    cfg.period = detail::g_deopt_stress_period.load(std::memory_order_relaxed);
    cfg.resume_id = g_resume_id.load(std::memory_order_relaxed);
    return cfg;
}

void set_deopt_stress(const DeoptStressConfig& config) noexcept {
    g_resume_id.store(config.resume_id, std::memory_order_relaxed);
    detail::g_deopt_stress_period.store(config.period, std::memory_order_relaxed);
}

namespace {

// Counts one evaluation of a selected guard; true when it must fail.
bool tick(uint64_t period) noexcept {
    const uint64_t n = std::atomic_ref<uint64_t>(g_evaluations).fetch_add(1, std::memory_order_relaxed) + 1;
    if (n % period != 0) return false;
    std::atomic_ref<uint64_t>(g_forced).fetch_add(1, std::memory_order_relaxed);
    return true;
}

} // namespace

bool deopt_stress_should_fail(uint32_t resume_id) noexcept {
    const uint64_t period = detail::g_deopt_stress_period.load(std::memory_order_relaxed);
    if (period == 0) return false;
    const int64_t selected = g_resume_id.load(std::memory_order_relaxed);
    if (selected >= 0 && static_cast<uint64_t>(selected) != resume_id) return false;
    return tick(period);
}

} // namespace brass::runtime

extern "C" uint32_t brass_deopt_stress_poll() {
    const uint64_t period = brass::runtime::detail::g_deopt_stress_period.load(std::memory_order_relaxed);
    return (period != 0 && brass::runtime::tick(period)) ? 0u : 1u;
}

namespace brass::runtime {

uint64_t deopt_stress_evaluations() noexcept {
    return std::atomic_ref<uint64_t>(g_evaluations).load(std::memory_order_relaxed);
}

uint64_t deopt_stress_forced() noexcept {
    return std::atomic_ref<uint64_t>(g_forced).load(std::memory_order_relaxed);
}

void reset_deopt_stress_counters() noexcept {
    std::atomic_ref<uint64_t>(g_evaluations).store(0, std::memory_order_relaxed);
    std::atomic_ref<uint64_t>(g_forced).store(0, std::memory_order_relaxed);
}

uint64_t* deopt_stress_evaluation_counter_address() noexcept { return &g_evaluations; }
uint64_t* deopt_stress_forced_counter_address() noexcept { return &g_forced; }

DeoptStressScope::DeoptStressScope(const DeoptStressConfig& config) noexcept : previous_(deopt_stress_config()) {
    set_deopt_stress(config);
    reset_deopt_stress_counters();
}

DeoptStressScope::~DeoptStressScope() {
    set_deopt_stress(previous_);
    reset_deopt_stress_counters();
}

} // namespace brass::runtime
