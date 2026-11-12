#include <brass/codegen/branch_relay.hpp>
#include <algorithm>
#include <unordered_map>

namespace brass::codegen {

using x64::Condition;

bool condition_implies(Condition taken, Condition then) {
    if (then == Condition::None) return true;
    if (taken == Condition::None) return false;
    if (taken == then) return true;
    // Over every flags state, not only what a `cmp` leaves: an `add` that
    // carries out to zero sets ZF and CF at once, so ZF=1 says nothing of CF,
    // nor of SF and OF.
    switch (taken) {
        // ZF=1
        case Condition::E: return then == Condition::BE || then == Condition::LE;
        // CF=1
        case Condition::B: return then == Condition::BE;
        // CF=0 and ZF=0
        case Condition::A: return then == Condition::AE || then == Condition::NE;
        // SF!=OF
        case Condition::L: return then == Condition::LE;
        // ZF=0 and SF==OF
        case Condition::G: return then == Condition::GE || then == Condition::NE;
        default: return false;
    }
}

std::vector<uint32_t> plan_branch_relays(const std::vector<BranchSite>& sites, int64_t reach) {
    const uint32_t none = UINT32_MAX;
    std::vector<uint32_t> relay(sites.size(), none);
    std::vector<uint8_t> hub(sites.size(), 0);
    std::unordered_map<uint32_t, std::vector<uint32_t>> by_target;
    for (uint32_t i = 0; i < sites.size(); ++i) by_target[sites[i].target].push_back(i);
    for (auto& [target, group] : by_target) {
        if (group.size() < 2) continue;
        // In emission order, so `group` is sorted by offset. A branch takes an
        // existing hub when one reaches; else it makes the nearest branch it
        // can reach a hub (one not relaying already).
        for (uint32_t b : group) {
            const BranchSite& s = sites[b];
            if (!s.is_long || !s.seldom || hub[b]) continue;
            const int64_t from = static_cast<int64_t>(s.start) + 2;
            uint32_t best = none;
            bool best_hub = false;
            int64_t best_dist = 0;
            for (uint32_t h : group) {
                if (h == b || relay[h] != none) continue;
                const int64_t d = static_cast<int64_t>(sites[h].start) - from;
                if (d < -reach || d > reach) continue;
                if (!condition_implies(s.cond, sites[h].cond)) continue;
                const int64_t dist = d < 0 ? -d : d;
                const bool is_hub = hub[h] != 0;
                if (best == none || (is_hub && !best_hub) || (is_hub == best_hub && dist < best_dist)) {
                    best = h;
                    best_hub = is_hub;
                    best_dist = dist;
                }
            }
            if (best == none) continue;
            relay[b] = best;
            hub[best] = 1;
        }
    }
    return relay;
}

} // namespace brass::codegen
