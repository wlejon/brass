// Last-level cache domains and compile-worker placement (cache_domains.hpp).

#include "cache_domains.hpp"

#if defined(__linux__)

#include <sched.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace brass::runtime::detail {

namespace {

// The CPUs the process may run on, grouped by the last-level cache they
// share; empty when there is one group or the topology cannot be read.
struct Topology {
    cpu_set_t allowed;
    std::vector<cpu_set_t> domains;
    std::vector<int> domain_of; // by CPU number; -1: not allowed or unknown
};

std::string read_line(const std::string& path) {
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    return line;
}

// "0-7,16-23" -> the CPUs it names.
std::vector<int> parse_cpu_list(const std::string& list) {
    std::vector<int> cpus;
    size_t at = 0;
    while (at < list.size()) {
        size_t comma = list.find(',', at);
        if (comma == std::string::npos) comma = list.size();
        const std::string part = list.substr(at, comma - at);
        at = comma + 1;
        if (part.empty()) continue;
        int lo = -1, hi = -1;
        if (std::sscanf(part.c_str(), "%d-%d", &lo, &hi) == 2) {
            for (int c = lo; c <= hi && c >= 0; ++c) cpus.push_back(c);
        } else if (std::sscanf(part.c_str(), "%d", &lo) == 1 && lo >= 0) {
            cpus.push_back(lo);
        }
    }
    return cpus;
}

// The CPU list of the highest-level cache `cpu` has (the last-level one).
std::string last_level_cache_of(int cpu) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/cache/index";
    int best_level = 0;
    std::string best;
    for (int i = 0; i < 16; ++i) {
        const std::string level = read_line(base + std::to_string(i) + "/level");
        if (level.empty()) break;
        const int l = std::atoi(level.c_str());
        if (l > best_level) {
            const std::string shared = read_line(base + std::to_string(i) + "/shared_cpu_list");
            if (!shared.empty()) {
                best_level = l;
                best = shared;
            }
        }
    }
    return best;
}

Topology read_topology() {
    Topology t;
    CPU_ZERO(&t.allowed);
    // The main thread's mask (its tid is the pid), not the caller's: the
    // caller is a worker, and what it may run on is what it was started with.
    if (sched_getaffinity(getpid(), sizeof t.allowed, &t.allowed) != 0) return t;
    std::map<std::string, int> index;
    std::vector<cpu_set_t> domains;
    std::vector<int> domain_of(CPU_SETSIZE, -1);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &t.allowed)) continue;
        const std::string key = last_level_cache_of(cpu);
        if (key.empty()) return t;
        auto [it, fresh] = index.emplace(key, static_cast<int>(domains.size()));
        if (fresh) {
            cpu_set_t set;
            CPU_ZERO(&set);
            for (int c : parse_cpu_list(key)) {
                if (c < CPU_SETSIZE && CPU_ISSET(c, &t.allowed)) CPU_SET(c, &set);
            }
            domains.push_back(set);
        }
        domain_of[cpu] = it->second;
    }
    if (domains.size() < 2) return t;
    t.domains = std::move(domains);
    t.domain_of = std::move(domain_of);
    return t;
}

const Topology& topology() {
    // Read once and kept: CPUs coming and going later are not followed.
    // Never destroyed: a compile worker still placing itself while the main
    // thread runs exit()'s static destructors would read a freed vector.
    static const Topology* const t = new Topology(read_topology());
    return *t;
}

// The domain this thread was last moved away from; -1: it may run anywhere.
thread_local int t_away_from = -1;

} // namespace

int current_cpu() noexcept {
    return sched_getcpu();
}

void place_away_from(int cpu) noexcept {
    const Topology& t = topology();
    if (t.domains.empty()) return;
    const int d = (cpu >= 0 && cpu < static_cast<int>(t.domain_of.size())) ? t.domain_of[cpu] : -1;
    if (d == t_away_from) return;
    cpu_set_t mask = t.allowed;
    if (d >= 0) {
        for (int c = 0; c < CPU_SETSIZE; ++c) {
            if (CPU_ISSET(c, &t.domains[static_cast<size_t>(d)])) CPU_CLR(c, &mask);
        }
    }
    if (sched_setaffinity(0, sizeof mask, &mask) == 0) t_away_from = d;
}

} // namespace brass::runtime::detail

#else

namespace brass::runtime::detail {

// Windows (GetLogicalProcessorInformationEx's cache relations and
// SetThreadGroupAffinity) and macOS (no thread affinity) place nothing yet.
int current_cpu() noexcept { return -1; }
void place_away_from(int) noexcept {}

} // namespace brass::runtime::detail

#endif
