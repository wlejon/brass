#include "bench_load.hpp"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <unistd.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

namespace brass::bench {

namespace {

double wall_now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

#if defined(_WIN32)
unsigned long long ft_ticks(const FILETIME& ft) {
    return (static_cast<unsigned long long>(ft.dwHighDateTime) << 32) | ft.dwLowDateTime;
}
#else
double self_cpu_seconds() {
    struct rusage r {};
    if (getrusage(RUSAGE_SELF, &r) != 0) return 0.0;
    return static_cast<double>(r.ru_utime.tv_sec + r.ru_stime.tv_sec) +
           static_cast<double>(r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e6;
}
#endif

void print_verdict(const char* when, double pct) {
    std::cout << std::fixed << std::setprecision(1);
    if (pct < 0.0) {
        std::cout << "[LOAD] " << when << ": CPU load unavailable on this platform; cannot vouch for a quiet machine.\n";
    } else if (pct > kBusyMachinePercent) {
        std::cout << "[LOAD] " << when << ": " << pct << "% of " << logical_cpu_count()
                  << " logical CPUs busy (> " << kBusyMachinePercent << "%).\n"
                  << "[LOAD] WARNING: MACHINE WAS BUSY. These timings are untrustworthy; "
                     "do not read a failure as a regression or --update-ratchet from this run.\n";
    } else {
        std::cout << "[LOAD] " << when << ": " << pct << "% of " << logical_cpu_count()
                  << " logical CPUs busy (quiet; threshold " << kBusyMachinePercent << "%).\n";
    }
}

} // namespace

unsigned logical_cpu_count() {
    unsigned n = std::thread::hardware_concurrency();
    return n ? n : 1u;
}

CpuLoadSample sample_cpu_load() {
    CpuLoadSample s;
    s.wall_seconds = wall_now_seconds();
#if defined(_WIN32)
    FILETIME idle, kernel, user;
    if (GetSystemTimes(&idle, &kernel, &user)) {
        // Kernel time includes idle time.
        const unsigned long long total = ft_ticks(kernel) + ft_ticks(user);
        s.total = total;
        s.busy = total - ft_ticks(idle);
        s.valid = true;
    }
    FILETIME c, e, pk, pu;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &pk, &pu)) {
        s.self_seconds = static_cast<double>(ft_ticks(pk) + ft_ticks(pu)) / 1e7;
    }
#elif defined(__linux__)
    std::ifstream stat("/proc/stat");
    std::string line;
    if (stat && std::getline(stat, line) && line.rfind("cpu ", 0) == 0) {
        std::istringstream in(line.substr(4));
        unsigned long long v[8] = {};
        for (auto& x : v) in >> x;
        // user nice system idle iowait irq softirq steal
        const unsigned long long idle = v[3] + v[4];
        unsigned long long total = 0;
        for (auto x : v) total += x;
        s.total = total;
        s.busy = total - idle;
        s.valid = true;
    }
    s.self_seconds = self_cpu_seconds();
#else
    s.self_seconds = self_cpu_seconds();
#endif
    return s;
}

double cpu_busy_percent(const CpuLoadSample& a, const CpuLoadSample& b) {
    if (!a.valid || !b.valid || b.total <= a.total) return -1.0;
    return 100.0 * static_cast<double>(b.busy - a.busy) / static_cast<double>(b.total - a.total);
}

double cpu_other_percent(const CpuLoadSample& a, const CpuLoadSample& b) {
    const double busy = cpu_busy_percent(a, b);
    const double wall = b.wall_seconds - a.wall_seconds;
    if (busy < 0.0 || wall <= 0.0) return -1.0;
    const double self = 100.0 * (b.self_seconds - a.self_seconds) / (wall * logical_cpu_count());
    const double other = busy - self;
    return other < 0.0 ? 0.0 : other;
}

double report_load_before_run(double seconds) {
    const CpuLoadSample a = sample_cpu_load();
    std::this_thread::sleep_for(std::chrono::duration<double>(seconds));
    const CpuLoadSample b = sample_cpu_load();
    const double pct = cpu_busy_percent(a, b);
    print_verdict("before run", pct);
    return pct;
}

double report_load_during_run(const CpuLoadSample& start) {
    const double pct = cpu_other_percent(start, sample_cpu_load());
    print_verdict("during run (other processes)", pct);
    return pct;
}

bool pin_current_thread(int cpu) {
    if (cpu < 0 || static_cast<unsigned>(cpu) >= logical_cpu_count() || cpu >= 64) return false;
#if defined(_WIN32)
    return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu) != 0;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return sched_setaffinity(0, sizeof(set), &set) == 0;
#else
    return false;
#endif
}

} // namespace brass::bench
