#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/types.hpp>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace brass {

// On-stack replacement into a separate entry function. A frame of `fn`
// stopped at the start of `block` (a loop header, as the interpreter sees
// it on a backedge) continues in the OSR entry function: a function of its
// own, taking a pointer to the frame's live values, whose entry block loads
// them and branches to `block`, followed by the blocks of the loop `block`
// heads (those reachable from it that reach it again) and the landing pads
// the loop's invokes unwind to. Its control flow is
// honest (nothing jumps into the middle of code the entry does not
// dominate), so the whole optimizer runs on it like on any other function.
//
// Only the loop: a hot loop early in a long function run once (a bundle's
// top level) would otherwise take the rest of the function with it, and the
// optimizer would build all of it to finish one loop. Each edge out of the
// loop goes to an EXIT instead: the values live into the block it went to
// are written to the buffer, `left_slot()` is set to 2 + the exit's index,
// and the function returns; the frame then goes on at that block in the
// tier that entered the code (OsrCoordinator::enter). A `ret` inside the
// loop still returns from the frame.
struct OsrEntryPlan {
    struct LiveIn {
        const Value* value = nullptr;
        // A constant: the entry recomputes it and reads nothing.
        bool rematerialize = false;
    };

    const Function* function = nullptr;
    const BasicBlock* block = nullptr;
    // The values live into `block`: its parameters first, in order, then
    // every other live value, by id. Live-in i that is not rematerialized
    // is read from buffer[i], 8 bytes each (a narrower value from the low
    // bytes).
    std::vector<LiveIn> live_ins;
    // `block`, the rest of its loop and the pads its invokes unwind to, in
    // fn's order.
    std::vector<const BasicBlock*> region;
    // A block outside the region a region block branches to, and the values
    // live into it: its parameters first, in order, then every other live
    // value by id. On leaving by it, value j is at buffer[exit_slot(j)].
    struct Exit {
        const BasicBlock* block = nullptr;
        std::vector<const Value*> values;
    };
    std::vector<Exit> exits;
    // A way out for a frame whose code has been invalidated. Its guards
    // failing finish the whole call in Tier 0, but a guard of a callee body
    // the code carries finishes only the callee and returns into the loop,
    // which would call the same failing copy on every iteration. With
    // leave_check, buffer[leave_flag_slot()] holds the address of a u64
    // that is nonzero once the code must not run on; every backedge to
    // `block` reads it, and when it is set the code writes each live value
    // that is not rematerialized back to its buffer slot, stores 1 to
    // buffer[left_slot()] and returns a meaningless result (read from
    // buffer[left_slot() + 1]). The frame then goes on at `block` in the
    // tier that entered the code. Only a region with a call can be
    // invalidated from inside, so only such a region gets the check.
    bool leave_check = false;

    // Whether the code can come back with the frame unfinished: a leave or
    // an exit. buffer[left_slot()] then says which (0: finished, 1: left at
    // `block`, 2 + i: left by exit i).
    bool can_leave() const noexcept { return leave_check || !exits.empty(); }
    size_t leave_flag_slot() const noexcept { return live_ins.size(); }
    size_t left_slot() const noexcept { return live_ins.size() + 1; }
    size_t exit_slot(size_t j) const noexcept { return live_ins.size() + 3 + j; }
    size_t buffer_slots() const noexcept {
        size_t widest = 0;
        for (const Exit& x : exits) widest = std::max(widest, x.values.size());
        return live_ins.size() + (can_leave() ? 3 + widest : 0);
    }
};

// Plans the OSR entry of `fn` at `block`; nothing (with `why`) when it has
// none: a coroutine (resume points), the entry block itself, or a live
// value the buffer cannot carry (a vector, a gcref). leave_check is off;
// the caller that can set the flag turns it on (region_has_call says when
// it would ever be read set).
std::optional<OsrEntryPlan> plan_osr_entry(const Function& fn, const BasicBlock& block, std::string* why = nullptr);

// Whether the plan's region calls anything, so that a callee's failed guard
// can invalidate the code while a frame is in it.
bool region_has_call(const OsrEntryPlan& plan);

// Adds the OSR entry function of `plan` to `dst` as `name`: (ptr buffer) ->
// fn's return type. A value defined inside the region that is also live
// into the block (one of an enclosing loop's, say) gets its SSA form back
// through a stack slot the entry seeds, which SROA then promotes. With the
// plan's leave_check (and a scalar or void result) each backedge to the
// block polls the leave flag first. Null (with `why`) if the result does
// not verify.
Function* build_osr_entry_function(const OsrEntryPlan& plan, Module& dst, std::string_view name,
                                   std::string* why = nullptr);

} // namespace brass
