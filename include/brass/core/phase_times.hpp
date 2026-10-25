#pragma once

#include <chrono>
#include <cstdint>
#include <string_view>

// Compile-time accounting: with BRASS_PASS_TIMES set, every optimizer step
// (by its pipeline name) and every backend phase records the wall time it
// took, summed over all functions and compile threads, and the process prints
// the table on exit (to stderr, or to the file BRASS_PASS_TIMES names when it
// is not "1"). Unset, a scope costs one predictable branch.
namespace brass {

bool phase_times_enabled() noexcept;
void record_phase_time(std::string_view name, uint64_t ns) noexcept;

class PhaseTimeScope {
public:
    explicit PhaseTimeScope(std::string_view name) noexcept
        : name_(name), on_(phase_times_enabled()) {
        if (on_) start_ = std::chrono::steady_clock::now();
    }
    ~PhaseTimeScope() {
        if (on_) {
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start_).count();
            record_phase_time(name_, static_cast<uint64_t>(ns));
        }
    }
    PhaseTimeScope(const PhaseTimeScope&) = delete;
    PhaseTimeScope& operator=(const PhaseTimeScope&) = delete;

private:
    std::string_view name_;
    bool on_;
    std::chrono::steady_clock::time_point start_{};
};

} // namespace brass
