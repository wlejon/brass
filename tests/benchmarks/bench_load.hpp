#pragma once

// Machine-load guard for the benchmark suite. A ratchet run on a busy machine
// measures the machine, not brass: other processes steal cores, drop turbo
// clocks and share memory bandwidth. The suite samples system-wide CPU use
// before it starts and across the whole run (less its own CPU time) and says
// plainly when either was high enough that the numbers are not trustworthy.

namespace brass::bench {

struct CpuLoadSample {
    bool valid = false;
    unsigned long long busy = 0;   // all CPUs, arbitrary ticks
    unsigned long long total = 0;  // busy + idle, same ticks
    double self_seconds = 0.0;     // this process's user + kernel CPU time
    double wall_seconds = 0.0;     // monotonic wall clock
};

// Percent of all logical CPUs that were busy above which a run is flagged.
constexpr double kBusyMachinePercent = 20.0;

CpuLoadSample sample_cpu_load();

// System-wide busy percent between two samples; -1 if unavailable.
double cpu_busy_percent(const CpuLoadSample& a, const CpuLoadSample& b);

// Busy percent of the machine not accounted for by this process; -1 if
// unavailable.
double cpu_other_percent(const CpuLoadSample& a, const CpuLoadSample& b);

unsigned logical_cpu_count();

// Samples the machine for `seconds` while idle and prints the result.
// Returns the busy percent (-1 if unavailable).
double report_load_before_run(double seconds);

// Prints the load other processes put on the machine during the run.
// Returns that percent (-1 if unavailable).
double report_load_during_run(const CpuLoadSample& start);

// Pins the calling thread to one logical CPU. On a part whose cores differ
// (the 7950X3D's V-cache and frequency CCDs, hybrid P/E cores) an unpinned
// run lands on whichever the scheduler picks, and ratios then move between
// runs by more than any regression margin. Returns false if it could not.
bool pin_current_thread(int cpu);

} // namespace brass::bench
