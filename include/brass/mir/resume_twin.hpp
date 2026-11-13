#pragma once

// Resume twins: where a speculated function's failed guards go when there is
// no lower tier to finish the call in (an AOT image).
//
// A front end lowers a function once in site form, as for the JIT tiers
// (speculation_sites.hpp): each speculation site is a guard whose resume
// target, a block of the same function, is the generic code for the site,
// and whose state is every value live into that block. From that one
// function it gets two:
//
//   - the twin, `(i32 resume id, ptr state) -> R` (build_resume_twin): the
//     function's code with every guarded resume point as an entry. Its entry
//     block switches on the resume id; each case reads that guard's state
//     values from `state`, 8 bytes each in state order, and continues at the
//     resume block. A guard of the site form is a branch in the twin (to its
//     resume block when its condition fails), so the twin never exits: it is
//     ordinary code and the whole optimizer runs on it.
//
//   - the speculated function (speculate_against_twin): a copy of the site
//     form, with whatever guards the front end armed, whose guards exit into
//     the twin: `guard %c, @twin, [state]`. The x64 backend lowers such a
//     guard to a compare and a branch to out-of-line code that stores the
//     state under the frame and calls twin(resume id, &state), returning its
//     result. See docs/speculation_and_deopt.md, "Guards in AOT images", for
//     the contract.

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace brass {

struct ResumeTwinOptions {
    // Also makes the function's entry an entry of the twin, under this
    // resume id (one no guard of the function has): its state is the
    // function's arguments. build_twin_entry then gives the unspeculated
    // function without a second copy of its code.
    std::optional<uint32_t> entry_id;
};

// Adds the resume twin of `fn` to `dst` as `name`. `fn` is in site form: the
// twin gets an entry for each resume point of `fn` that a guard without an
// exit stub names. Null (with `why`) when `fn` has none, is a coroutine
// body, has a guard state the twin cannot carry (a vector), or the result
// does not verify.
Function* build_resume_twin(const Function& fn, Module& dst, std::string_view name,
                            const ResumeTwinOptions& options = {}, std::string* why = nullptr);

// Adds `name` to the twin's module: a function with `fn`'s signature that
// runs the twin from its `entry_id` entry (ResumeTwinOptions::entry_id), so
// it computes what `fn` computes with no speculation. This is what a caller
// uses once it has stopped trusting the speculated function.
Function* build_twin_entry(const Function& fn, const Function& twin, uint32_t entry_id, std::string_view name,
                           std::string* why = nullptr);

// Makes `fn`, a function of the twin's module that is a copy of the function
// the twin was built from (with its guards armed or dropped as the front end
// chose), exit into `twin`: every guard that resumes in fn's resume table
// names the twin instead, the resume table is dropped, and the blocks only
// it kept reachable are removed. False (with `why`) when a guard's state
// does not match what the twin reads at its resume id.
bool speculate_against_twin(Function& fn, const Function& twin, std::string* why = nullptr);

// The two steps for a whole module, in place: every function with a
// site-form guard gets its twin, named resume_twin_name(function), and then
// exits into it. Functions without one are left alone. False (with `why`)
// when a twin cannot be built; the module is then partly rewritten.
bool speculate_module_against_twins(Module& mod, std::string* why = nullptr);
std::string resume_twin_name(std::string_view function_name);

} // namespace brass
