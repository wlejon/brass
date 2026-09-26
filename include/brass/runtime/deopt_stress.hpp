#pragma once

// Deopt stress: guards forced to fail, so that every tier's deopt path, and
// every state map behind it, runs whether or not the speculation it guards
// ever goes wrong. The deopt counterpart of BRASS_GC_STRESS.
//
// BRASS_DEOPT_STRESS, read once at startup:
//   unset, 0, none  off
//   1, all          every eligible guard evaluation fails
//   <N>             every Nth eligible guard evaluation fails (one counter
//                   for the process, shared by every tier)
//   site:<R>        every evaluation of an eligible guard whose resume id
//                   is R fails, and no other guard is touched
//   site:<R>:<N>    every Nth evaluation of those
//
// An eligible guard is one that has somewhere to go: an exit stub (a
// function of its module with the stub signature) or a resume target in a
// tier that can continue one (the interpreters, the baseline JIT, and
// tier-2 code installed by a pipeline, which finishes in Tier 0). Guards
// that exit only through a deopt handler, and resume-only guards in
// standalone tier-2 code, are never forced: they have no exit that the
// stress could take in place of the fast path.
//
// Interpreters check at each evaluation. Native tiers decide at compile
// time: code compiled while stress is on counts and fails its guards,
// code compiled while it is off never does. Tier 2 does it with a guard of
// its own placed just before each eligible guard, with the same exits and
// state values, over code the optimizer has already finished with, so the
// real guard's fused compare and the values live at it are unchanged.
//
// A forced failure is correct speculation only when the guard's exits
// compute what its fast path computes. For any program, every tier under
// period 1 (or one site) exits where the interpreter under it does and
// must give the interpreter's answer; for a program whose exits honour
// that contract, every period gives the answer with stress off.

#include <atomic>
#include <cstdint>

namespace brass::runtime {

struct DeoptStressConfig {
    // 0: off. 1: every eligible evaluation fails. N: every Nth fails.
    uint64_t period = 0;
    // Only guards with this resume id are counted and forced (-1: all).
    int64_t resume_id = -1;

    bool active() const noexcept { return period != 0; }
    bool selects(uint32_t guard_resume_id) const noexcept {
        return resume_id < 0 || static_cast<uint64_t>(resume_id) == guard_resume_id;
    }
};

namespace detail {
// The current period (0 when off): the interpreters' one-load fast check.
extern std::atomic<uint64_t> g_deopt_stress_period;
}

// Whether deopt stress is on (a relaxed load; interpreters test it on every
// guard that passes).
inline bool deopt_stress_active() noexcept {
    return detail::g_deopt_stress_period.load(std::memory_order_relaxed) != 0;
}

// The current configuration: BRASS_DEOPT_STRESS until set_deopt_stress.
DeoptStressConfig deopt_stress_config() noexcept;
// Replaces the configuration (tests; the environment is read once, before).
// Resets nothing: see reset_deopt_stress_counters.
void set_deopt_stress(const DeoptStressConfig& config) noexcept;

// Counts one eligible evaluation of a guard with `resume_id` and says
// whether it must fail. False, counting nothing, when stress is off or the
// guard is not selected.
bool deopt_stress_should_fail(uint32_t resume_id) noexcept;

// Eligible evaluations counted and failures forced, by every tier. Native
// tier-2 code updates them in place (deopt_stress_counter_address).
uint64_t deopt_stress_evaluations() noexcept;
uint64_t deopt_stress_forced() noexcept;
void reset_deopt_stress_counters() noexcept;
// The two 64-bit counters, for code that updates them inline.
uint64_t* deopt_stress_evaluation_counter_address() noexcept;
uint64_t* deopt_stress_forced_counter_address() noexcept;

// Sets a configuration for a scope and restores the previous one after,
// resetting the counters at both ends.
class DeoptStressScope {
public:
    explicit DeoptStressScope(const DeoptStressConfig& config) noexcept;
    explicit DeoptStressScope(uint64_t period, int64_t resume_id = -1) noexcept
        : DeoptStressScope(DeoptStressConfig{period, resume_id}) {}
    ~DeoptStressScope();
    DeoptStressScope(const DeoptStressScope&) = delete;
    DeoptStressScope& operator=(const DeoptStressScope&) = delete;

private:
    DeoptStressConfig previous_;
};

} // namespace brass::runtime

extern "C" {
// The baseline JIT's poll before an eligible guard (compiled in only while
// stress is on, for the guards the configuration selects): counts the
// evaluation, returning 0 when the guard must fail and 1 otherwise.
uint32_t brass_deopt_stress_poll();
}
