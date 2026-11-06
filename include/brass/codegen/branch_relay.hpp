#pragma once

#include <brass/target/x64/x64_registers.hpp>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace brass::codegen {

// Branch relays (x64). A block-to-block branch too far from its target for
// the 2-byte form can still take it when another branch to the same target
// lies within reach and is sure to be taken from where the first one stands:
// one whose condition the first one's implies (a `ja` reaches through a
// `jae` or a `jne`, an `je` through a `jbe`, anything through a `jmp`).
// Jumping onto that branch's own first byte, with the flags as they are,
// takes it, so the target is reached with nothing else run on the way.
//
// What it is for: an inline cache's or a static slot's guard misses to its
// site's cold block, laid out after the function's hot code and so a 6-byte
// `jcc rel32`, and the guards of one site (the object tag test, the header
// test, the shape test) all miss to the same block within a few dozen bytes
// of each other. Every one relayed is four bytes less of hot code, at the cost
// of one more taken branch on a path that seldom runs.

// Whether every flags state that satisfies `taken` also satisfies `then`
// (Condition::None is the unconditional jump, which any state satisfies).
bool condition_implies(x64::Condition taken, x64::Condition then);

// One block-to-block branch of an emission, by ordinal.
struct BranchSite {
    x64::Condition cond = x64::Condition::None;
    uint32_t target = 0;   // the block it goes to
    size_t start = 0;      // its offset
    bool is_long = false;  // emitted in its rel32 form
};

// For each branch, the ordinal of the branch it relays to, or UINT32_MAX.
// Only a long branch relays, never onto one that itself relays (so a relay
// is one hop), and only to a branch whose start lies within `reach` bytes of
// where its 2-byte form would end, measured on `sites` as emitted; the
// caller emits the plan and drops any relay the new offsets put out of reach.
std::vector<uint32_t> plan_branch_relays(const std::vector<BranchSite>& sites, int64_t reach);

} // namespace brass::codegen
