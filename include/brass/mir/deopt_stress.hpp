#pragma once

// Tier 2's half of deopt stress (runtime/deopt_stress.hpp): before each
// eligible guard of an optimized function, a guard of its own with the same
// exits and state values, whose condition counts the evaluation and fails
// every `period`th one. It runs after the optimizer, on the MIR about to be
// lowered, so nothing folds it away and the real guard is lowered as it
// would be without it.

#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <cstdint>

namespace brass {

struct DeoptStressPlan {
    uint64_t period = 0;
    // Only guards with this resume id (-1: all).
    int64_t resume_id = -1;
    // The counters the stress guards update in place.
    uint64_t* evaluations = nullptr;
    uint64_t* forced = nullptr;
    // The code has a lower tier to continue a resume block in (a pipeline's
    // tier-2 and OSR code). Without one only guards with an exit stub are
    // forced.
    bool resume_targets = false;
};

// Whether forcing `guard` of `fn` to fail takes one of its exits: it has an
// exit stub (a function of `stub_module`, else of fn's module, with the stub
// signature), or, when `resume_targets`, a resume target.
bool deopt_stress_guard_eligible(const Function& fn, const Instruction& guard, const Module* stub_module,
                                 bool resume_targets);

// Whether `fn` has a guard the plan would instrument.
bool has_deopt_stress_guards(const Function& fn, const Module* stub_module, const DeoptStressPlan& plan);

// Instruments `fn` in place; returns how many guards it placed.
size_t insert_deopt_stress_guards(Function& fn, const Module* stub_module, const DeoptStressPlan& plan);

} // namespace brass
