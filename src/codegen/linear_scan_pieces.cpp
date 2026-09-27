#include <brass/codegen/linear_scan.hpp>
#include <algorithm>
#include <bit>

// The scan: each piece, in order of its start, gets a register free for all
// of it, else one free for a prefix that holds a use (the rest is split off
// and scanned later), else the register whose holders need it furthest away
// (they are split where they last used it before and wait in their slots
// until their next use), else its spill slot until just before its own next
// use. Split positions move to the shallowest loop depth the range allows,
// so a value evicted for a loop is stored before the loop and a value
// reloaded for one is reloaded before it.

namespace brass::codegen {

namespace {

constexpr uint32_t kNever = UINT32_MAX;

int class_index(RegClass rc) { return rc == RegClass::GPR ? 0 : 1; }

// The last position a piece holding a register may reach, from the first
// position at which the register is blocked (an instruction there, so the
// piece must end one before it).
uint32_t limit_before(uint32_t blocked_at) {
    if (blocked_at == kNever) return kNever;
    return blocked_at == 0 ? 0 : blocked_at - 1;
}

// A piece's segments, clamped to its window.
struct SegCursor {
    const AllocPiece& p;
    std::vector<LiveRangeSegment>::const_iterator it;
    std::vector<LiveRangeSegment>::const_iterator end;

    explicit SegCursor(const AllocPiece& piece) : p(piece), end(piece.base->segments.end()) {
        const auto& segs = piece.base->segments;
        it = std::lower_bound(segs.begin(), segs.end(), piece.from,
            [](const LiveRangeSegment& s, uint32_t v) { return s.end < v; });
    }
    bool valid() const { return it != end && it->start <= p.to; }
    uint32_t lo() const { return std::max(it->start, p.from); }
    uint32_t hi() const { return std::min(it->end, p.to); }
    void next() { ++it; }
};

} // namespace

uint32_t LinearScanAllocator::new_piece(const LiveInterval* base, uint32_t from, uint32_t to) {
    AllocPiece p;
    p.base = base;
    p.from = from;
    p.to = to;
    p.seq = next_seq_++;
    const auto& segs = base->segments;
    auto first = std::lower_bound(segs.begin(), segs.end(), from,
        [](const LiveRangeSegment& s, uint32_t v) { return s.end < v; });
    p.start = first == segs.end() ? from : std::max(first->start, from);
    auto last = std::upper_bound(segs.begin(), segs.end(), to,
        [](uint32_t v, const LiveRangeSegment& s) { return v < s.start; });
    p.end = last == segs.begin() ? to : std::min((last - 1)->end, to);
    pieces_.push_back(p);
    return static_cast<uint32_t>(pieces_.size() - 1);
}

bool LinearScanAllocator::piece_covers(const AllocPiece& p, uint32_t id) const {
    return id >= p.from && id <= p.to && p.base->covers(id);
}

uint32_t LinearScanAllocator::next_use(const AllocPiece& p, uint32_t pos, bool reg_only) const {
    const auto& uses = p.base->use_positions;
    auto it = std::lower_bound(uses.begin(), uses.end(), std::max(pos, p.from),
        [](const UsePosition& u, uint32_t v) { return u.inst_id < v; });
    for (; it != uses.end() && it->inst_id <= p.to; ++it) {
        if (!reg_only || it->requires_reg) return it->inst_id;
    }
    return kNever;
}

// In layout order a loop's next iteration lies behind: a value read early in
// the body and carried round the back edge has no use after a position late
// in the body, and by next use alone looked the best one to evict, for a
// reload at every back edge, over a value idle until after the loop.
uint32_t LinearScanAllocator::needed_again(const AllocPiece& h, uint32_t pos) const {
    const uint32_t u = next_use(h, pos, true);
    auto it = std::upper_bound(spans_.begin(), spans_.end(), pos,
        [](uint32_t v, const BlockSpan& s) { return v < s.start; });
    if (it == spans_.begin()) return u;
    const BlockSpan& s = *(it - 1);
    if (s.depth == 0 || pos > s.loop_to) return u;
    if (u != kNever && u <= s.loop_to) return u;
    if (!piece_covers(h, s.loop_to)) return u;
    const auto& uses = h.base->use_positions;
    auto a = std::lower_bound(uses.begin(), uses.end(), s.loop_from,
        [](const UsePosition& x, uint32_t v) { return x.inst_id < v; });
    for (; a != uses.end() && a->inst_id < pos; ++a) {
        if (a->requires_reg) return s.loop_to;
    }
    return u;
}

uint32_t LinearScanAllocator::last_use_at_or_before(const AllocPiece& p, uint32_t pos) const {
    const auto& uses = p.base->use_positions;
    auto it = std::upper_bound(uses.begin(), uses.end(), std::min(pos, p.to),
        [](uint32_t v, const UsePosition& u) { return v < u.inst_id; });
    if (it == uses.begin()) return kNever;
    --it;
    return it->inst_id >= p.from ? it->inst_id : kNever;
}

// The first position at which both pieces are live, by the segments' strict
// overlap: one that ends where the other starts does not meet it (an
// instruction may read one value and write another to the same register,
// and a move may empty a register another move fills).
uint32_t LinearScanAllocator::first_intersection(const AllocPiece& a, const AllocPiece& b) const {
    if (a.end <= b.start || b.end <= a.start) return kNever;
    SegCursor x(a), y(b);
    while (x.valid() && y.valid()) {
        const uint32_t x1 = x.lo(), x2 = x.hi(), y1 = y.lo(), y2 = y.hi();
        if (x1 < y2 && y1 < x2) return std::max(x1, y1);
        if (x2 < y2) x.next(); else y.next();
    }
    return kNever;
}

uint32_t LinearScanAllocator::own_fixed_regs(const AllocPiece& p, uint32_t pos) const {
    const VReg v = p.vreg();
    uint32_t own = 0;
    const auto& uses = p.base->use_positions;
    auto it = std::lower_bound(uses.begin(), uses.end(), pos,
        [](const UsePosition& u, uint32_t x) { return u.inst_id < x; });
    for (; it != uses.end() && it->inst_id == pos; ++it) {
        if (it->fixed_reg.is_valid() && it->fixed_reg.reg_class == v.reg_class && it->fixed_reg.code < 32) {
            own |= (1u << it->fixed_reg.code);
        }
    }
    if (!vreg_at_preg_hints_.empty()) {
        auto [h_beg, h_end] = vreg_at_preg_hints_.equal_range((static_cast<uint64_t>(v.id) << 32) | pos);
        for (auto h = h_beg; h != h_end; ++h) {
            if (h->second.is_valid() && h->second.reg_class == v.reg_class && h->second.code < 32) {
                own |= (1u << h->second.code);
            }
        }
    }
    return own;
}

uint32_t LinearScanAllocator::always_blocked(const AllocPiece& p) const {
    const VReg v = p.vreg();
    if (!v.is_gpr()) return 0;
    uint32_t blocked = fn_.reserved_gprs;
    if (cc_.target().is_x64() && v.id < is_mem_index_.size() && is_mem_index_[v.id]) {
        blocked |= (1u << static_cast<uint8_t>(x64::GPR::R12)) | (1u << static_cast<uint8_t>(x64::GPR::RSP));
    }
    return blocked;
}

// The first constrained instruction index in [i, end) whose word meets
// `want`, or `end`.
size_t LinearScanAllocator::first_constrained(int cls, bool wide, size_t i, size_t end, uint32_t want) const {
    const auto& t = constraint_or_[cls][wide ? 1 : 0];
    while (i < end) {
        if (t[0][i] & want) return i;
        size_t k = 0;
        while (k + 1 < t.size() && i + (size_t{2} << k) <= end && (t[k + 1][i] & want) == 0) ++k;
        i += size_t{1} << k;
    }
    return end;
}

// For each register of the piece's class, the first position inside the
// piece at which an instruction blocks it (kNever: none; 0: always).
void LinearScanAllocator::reg_limits(const AllocPiece& p, uint32_t out[32]) const {
    const VReg v = p.vreg();
    const int cls = class_index(v.reg_class);
    const bool wide = cls == 1 && fpr_wider_than_callee_saved(v);
    uint32_t pool_mask = 0;
    for (PReg r : (cls == 0 ? available_gprs_ : available_xmms_)) {
        if (r.code < 32) pool_mask |= (1u << r.code);
    }
    for (int r = 0; r < 32; ++r) out[r] = kNever;
    const uint32_t blocked = always_blocked(p) & pool_mask;
    for (uint32_t m = blocked; m; m &= m - 1) out[std::countr_zero(m)] = 0;
    uint32_t remaining = pool_mask & ~blocked;

    const auto& ids = constrained_ids_[cls];
    const auto& words = constraint_words_[cls];
    if (ids.empty()) return;
    for (SegCursor s(p); s.valid() && remaining; s.next()) {
        size_t i = static_cast<size_t>(std::lower_bound(ids.begin(), ids.end(), s.lo()) - ids.begin());
        const size_t iend = static_cast<size_t>(std::upper_bound(ids.begin() + i, ids.end(), s.hi()) - ids.begin());
        while (i < iend && remaining) {
            i = first_constrained(cls, wide, i, iend, remaining);
            if (i == iend) break;
            const ConstraintWord& w = words[i];
            uint32_t hit = w.clob | (wide ? w.call : 0u);
            if (w.pin & remaining) hit |= w.pin & ~own_fixed_regs(p, ids[i]);
            hit &= remaining;
            for (uint32_t m = hit; m; m &= m - 1) out[std::countr_zero(m)] = ids[i];
            remaining &= ~hit;
            ++i;
        }
    }
}

// An odd position in [lo, hi] to split at: the start of the block entered
// at the least loop depth there, or `hi` itself when that is no deeper;
// the later of equals.
uint32_t LinearScanAllocator::split_position(uint32_t lo, uint32_t hi) const {
    if ((lo & 1u) == 0) ++lo;
    if ((hi & 1u) == 0) {
        if (hi == 0) return kNever;
        --hi;
    }
    if (lo > hi) return kNever;
    uint32_t best = hi;
    uint32_t best_depth = liveness_.get_loop_depth_at(hi - 1);
    // Blocks starting in [lo + 1, hi + 1]: splitting at start - 1.
    auto i0 = std::lower_bound(spans_.begin(), spans_.end(), lo + 1,
        [](const BlockSpan& s, uint32_t v) { return s.start < v; });
    auto i1 = std::upper_bound(spans_.begin(), spans_.end(), hi + 1,
        [](uint32_t v, const BlockSpan& s) { return v < s.start; });
    if (i0 < i1) {
        const size_t a = static_cast<size_t>(i0 - spans_.begin());
        const size_t b = static_cast<size_t>(i1 - spans_.begin());
        const size_t k = static_cast<size_t>(std::bit_width(b - a) - 1);
        const uint32_t x = depth_rmq_[k][a];
        const uint32_t y = depth_rmq_[k][b - (size_t{1} << k)];
        uint32_t m;
        if (spans_[x].entry_depth != spans_[y].entry_depth) {
            m = spans_[x].entry_depth < spans_[y].entry_depth ? x : y;
        } else {
            m = std::max(x, y);
        }
        const uint32_t q = spans_[m].start - 1;
        if (spans_[m].entry_depth < best_depth || q == hi) best = q;
    }
    return best;
}

// Cuts piece `idx` at odd position q (start < q < end): it keeps [from, q],
// and the new piece returned takes [q, to].
uint32_t LinearScanAllocator::split_piece(uint32_t idx, uint32_t q) {
    const LiveInterval* base = pieces_[idx].base;
    const uint32_t to = pieces_[idx].to;
    const uint32_t right = new_piece(base, q, to);
    AllocPiece& left = pieces_[idx];
    left.to = q;
    const auto& segs = base->segments;
    auto last = std::upper_bound(segs.begin(), segs.end(), q,
        [](uint32_t v, const LiveRangeSegment& s) { return v < s.start; });
    left.end = last == segs.begin() ? q : std::min((last - 1)->end, q);
    auto& list = vreg_pieces_[base->vreg.id];
    auto at = std::upper_bound(list.begin(), list.end(), q,
        [this](uint32_t v, uint32_t i) { return v < pieces_[i].from; });
    list.insert(at, right);
    return right;
}

void LinearScanAllocator::push_unhandled(std::vector<uint32_t>& unhandled, uint32_t idx) const {
    unhandled.push_back(idx);
    std::push_heap(unhandled.begin(), unhandled.end(), [this](uint32_t a, uint32_t b) {
        const AllocPiece& x = pieces_[a];
        const AllocPiece& y = pieces_[b];
        if (x.start != y.start) return x.start > y.start;
        if (x.end != y.end) return x.end > y.end;
        return x.seq > y.seq;
    });
}

void LinearScanAllocator::assign(uint32_t idx, PReg reg) {
    AllocPiece& p = pieces_[idx];
    p.reg = reg;
    p.spilled = false;
    mark_callee_saved(reg);
    active_.push_back(idx);
}

// Piece `idx` lives in its slot; the part of it from just before its next
// use after `after` (moved out of loops) goes back to the scan.
void LinearScanAllocator::spill_and_requeue(uint32_t idx, uint32_t after, std::vector<uint32_t>& unhandled) {
    AllocPiece& p = pieces_[idx];
    const VReg v = p.vreg();
    p.spilled = true;
    p.reg = PReg{};
    slot_of(v);
    if (unsplittable_[v.id]) return;
    const uint32_t lo = std::max(p.start, after) + 1;
    const uint32_t u = next_use(p, lo + 1, true);
    if (u == kNever) return;
    const uint32_t q = split_position(lo, u - 1);
    if (q == kNever || q <= p.start || q >= p.end) return;
    push_unhandled(unhandled, split_piece(idx, q));
}

// Takes the register of active piece `idx` away from position `pos` on: it
// keeps the register up to a split at or before pos (after its last use
// before pos) and waits in its slot from there.
void LinearScanAllocator::evict_from(uint32_t idx, uint32_t pos, std::vector<uint32_t>& unhandled) {
    active_.erase(std::remove(active_.begin(), active_.end(), idx), active_.end());
    const AllocPiece& a = pieces_[idx];
    if (unsplittable_[a.vreg().id]) {
        AllocPiece& w = pieces_[idx];
        w.spilled = true;
        w.reg = PReg{};
        slot_of(w.vreg());
        return;
    }
    const uint32_t last = pos == 0 ? kNever : last_use_at_or_before(a, pos - 1);
    const uint32_t lo = std::max(a.start + 1, last == kNever ? 0u : last + 1);
    const uint32_t q = lo <= pos ? split_position(lo, pos) : kNever;
    if (q == kNever || q <= a.start || q >= a.end) {
        spill_and_requeue(idx, pos, unhandled);
        return;
    }
    const uint32_t right = split_piece(idx, q);
    spill_and_requeue(right, pos, unhandled);
}

std::vector<PReg> LinearScanAllocator::hint_regs(const AllocPiece& p) const {
    const VReg v = p.vreg();
    const auto& pool = v.is_gpr() ? available_gprs_ : available_xmms_;
    std::vector<PReg> hints;
    // Only a register the allocator could have picked anyway: a copy from
    // the stack pointer (`read_sp`) hints its destination at SP, and a value
    // that lives in SP is not a value. AArch64's register 31 is SP only to
    // the few encodings that say so and XZR to every other one, so `cmp`
    // against a vreg allocated there compared against zero.
    auto add = [&](PReg r) {
        if (!r.is_valid() || r.reg_class != v.reg_class || r.code >= 32) return;
        if (std::none_of(pool.begin(), pool.end(), [&](const PReg& x) { return x.code == r.code; })) return;
        if (std::find(hints.begin(), hints.end(), r) != hints.end()) return;
        hints.push_back(r);
    };
    // The value's own previous piece: the same register needs no move.
    const auto& list = vreg_pieces_[v.id];
    for (uint32_t i : list) {
        const AllocPiece& q = pieces_[i];
        if (q.to == p.from && &q != &p && !q.spilled) add(q.reg);
    }
    const auto& uses = p.base->use_positions;
    auto it = std::lower_bound(uses.begin(), uses.end(), p.from,
        [](const UsePosition& u, uint32_t x) { return u.inst_id < x; });
    for (; it != uses.end() && it->inst_id <= p.to; ++it) add(it->fixed_reg);
    if (auto f = fixed_preg_hints_.find(v.id); f != fixed_preg_hints_.end()) {
        for (PReg r : f->second) add(r);
    }
    if (auto c = coalesce_hints_.find(v.id); c != coalesce_hints_.end()) {
        for (VReg partner : c->second) {
            if (partner.id >= vreg_pieces_.size()) continue;
            for (uint32_t i : vreg_pieces_[partner.id]) {
                const AllocPiece& q = pieces_[i];
                if (q.spilled || !q.reg.is_valid()) continue;
                if (q.start <= p.end + 1 && p.start <= q.end + 1) add(q.reg);
            }
        }
    }
    return hints;
}

bool LinearScanAllocator::try_allocate_free(uint32_t idx, std::vector<uint32_t>& unhandled) {
    const AllocPiece p = pieces_[idx];
    const VReg v = p.vreg();
    const auto& pool = v.is_gpr() ? available_gprs_ : available_xmms_;

    uint32_t max_q[32];
    reg_limits(p, max_q);
    for (uint32_t& m : max_q) m = limit_before(m);
    for (uint32_t a : active_) {
        const AllocPiece& other = pieces_[a];
        if (other.vreg().reg_class != v.reg_class || other.reg.code >= 32) continue;
        const uint32_t c = first_intersection(other, p);
        if (c != kNever) max_q[other.reg.code] = std::min(max_q[other.reg.code], c);
    }

    const std::vector<PReg> hints = hint_regs(p);
    auto whole = [&](PReg r) { return max_q[r.code] >= p.end; };
    for (PReg r : hints) {
        if (whole(r)) { assign(idx, r); return true; }
    }
    for (int callee = 0; callee < 2; ++callee) {
        for (PReg r : pool) {
            if (is_callee_saved(r) == (callee == 1) && whole(r)) { assign(idx, r); return true; }
        }
    }
    if (unsplittable_[v.id]) return false;

    // A register free for a prefix that holds a use: keep it there.
    PReg best{};
    uint32_t best_q = 0;
    for (PReg r : hints) {
        if (max_q[r.code] > best_q) { best = r; best_q = max_q[r.code]; }
    }
    for (PReg r : pool) {
        if (max_q[r.code] > best_q) { best = r; best_q = max_q[r.code]; }
    }
    if (!best.is_valid() || best_q <= p.start) return false;
    const uint32_t u = next_use(p, p.start, true);
    if (u == kNever || u >= best_q) return false;
    const uint32_t q = split_position(u + 1, best_q);
    if (q == kNever || q <= p.start || q >= p.end) return false;
    const uint32_t right = split_piece(idx, q);
    assign(idx, best);
    push_unhandled(unhandled, right);
    return true;
}

void LinearScanAllocator::allocate_blocked(uint32_t idx, std::vector<uint32_t>& unhandled) {
    const AllocPiece p = pieces_[idx];
    const VReg v = p.vreg();
    const auto& pool = v.is_gpr() ? available_gprs_ : available_xmms_;
    const bool whole_only = unsplittable_[v.id] != 0;

    uint32_t hard_q[32];
    reg_limits(p, hard_q);
    for (uint32_t& m : hard_q) m = limit_before(m);
    uint32_t use_q[32];
    std::copy(hard_q, hard_q + 32, use_q);
    float whole_cost[32] = {};
    std::vector<uint32_t> holders[32];
    for (uint32_t a : active_) {
        const AllocPiece& other = pieces_[a];
        if (other.vreg().reg_class != v.reg_class || other.reg.code >= 32) continue;
        const uint32_t c = first_intersection(other, p);
        if (c == kNever) continue;
        const uint8_t r = other.reg.code;
        if (unsplittable_[other.vreg().id]) {
            if (whole_only) {
                whole_cost[r] += other.base->spill_weight;
                holders[r].push_back(a);
            } else {
                // A value that cannot be split keeps its register.
                hard_q[r] = std::min(hard_q[r], c);
                use_q[r] = std::min(use_q[r], c);
            }
        } else {
            holders[r].push_back(a);
            use_q[r] = std::min(use_q[r], needed_again(other, p.start));
        }
    }

    const std::vector<PReg> hints = hint_regs(p);
    if (whole_only) {
        // The value takes a register for its whole life, over holders that
        // cost less to spill than it does, or its slot.
        PReg best{};
        float best_cost = p.base->spill_weight;
        auto consider = [&](PReg r) {
            if (hard_q[r.code] < p.end) return;
            if (whole_cost[r.code] < best_cost) { best = r; best_cost = whole_cost[r.code]; }
        };
        for (PReg r : hints) consider(r);
        for (PReg r : pool) consider(r);
        if (!best.is_valid()) {
            AllocPiece& w = pieces_[idx];
            w.spilled = true;
            w.reg = PReg{};
            slot_of(v);
            return;
        }
        for (uint32_t a : holders[best.code]) evict_from(a, p.start, unhandled);
        assign(idx, best);
        return;
    }

    // The register whose holders cost least to evict, each cost weighed by
    // the depth of the loop it is paid in: a reload where the holder is
    // needed again, and a store where it is split off (evict_from), unless
    // its slot already holds its value (a value defined once, before the
    // loop this is in: store elimination wrote it there). Among equals, the
    // one needed furthest away. Only a register whose holders are not needed
    // before this value's first use is a candidate.
    auto weight = [this](uint32_t pos) {
        const uint32_t d = std::min<uint32_t>(liveness_.get_loop_depth_at(pos), 6);
        return static_cast<double>(uint64_t{1} << (3 * d));
    };
    // A move at a split opening a block runs on the edges into it: the
    // depth of entering the block.
    auto weight_split = [&](uint32_t q) {
        auto it = std::lower_bound(spans_.begin(), spans_.end(), q + 1,
            [](const BlockSpan& s, uint32_t v) { return s.start < v; });
        if (it != spans_.end() && it->start == q + 1) {
            return static_cast<double>(uint64_t{1} << (3 * std::min<uint32_t>(it->entry_depth, 6)));
        }
        return weight(q);
    };
    const uint32_t depth_here = liveness_.get_loop_depth_at(p.start);
    double cost[32] = {};
    for (int r = 0; r < 32; ++r) {
        for (uint32_t a : holders[r]) {
            const AllocPiece& h = pieces_[a];
            const uint32_t again = needed_again(h, p.start);
            if (again != kNever) cost[r] += weight(again);
            const uint32_t def = def_pos_[h.vreg().id];
            if (def != kNever && liveness_.get_loop_depth_at(def) < depth_here) continue;
            const uint32_t last = p.start == 0 ? kNever : last_use_at_or_before(h, p.start - 1);
            const uint32_t lo = std::max(h.start + 1, last == kNever ? 0u : last + 1);
            const uint32_t q = lo <= p.start ? split_position(lo, p.start) : kNever;
            cost[r] += q == kNever ? weight(p.start) : weight_split(q);
        }
    }
    const uint32_t first = next_use(p, p.start, true);
    PReg best{};
    uint32_t best_use = 0;
    double best_cost = 0;
    auto consider = [&](PReg r) {
        if (use_q[r.code] < first || hard_q[r.code] <= p.start) return;
        if (!best.is_valid() || cost[r.code] < best_cost ||
            (cost[r.code] == best_cost && use_q[r.code] > best_use)) {
            best = r;
            best_use = use_q[r.code];
            best_cost = cost[r.code];
        }
    };
    for (PReg r : hints) consider(r);
    for (PReg r : pool) consider(r);
    if (first == kNever || !best.is_valid()) {
        spill_and_requeue(idx, p.start, unhandled);
        return;
    }
    // A value already in its slot (its piece before this one waits there,
    // and this one does not start by defining it) is read from the slot at
    // its next use when taking a register would pay back less than it
    // costs. Taking it, the holder is reloaded before its own next use (and
    // stored here, if dirty); not taking it, each read of this value before
    // then is a load. Each is weighed by the depth of the loop it runs in:
    // a counter of an outer loop, idle through an inner one, gives up its
    // register to a value the inner loop reads, and a value read once
    // stays in its slot over one the same loop needs. Each use gets its own
    // try, and a free register is always taken (try_allocate_free).
    if (!whole_only && first > p.start) {
        const auto& list = vreg_pieces_[v.id];
        const bool waits_in_slot = std::any_of(list.begin(), list.end(), [&](uint32_t i) {
            return i != idx && pieces_[i].to == p.from && pieces_[i].spilled;
        });
        const auto& uses = p.base->use_positions;
        auto it = std::lower_bound(uses.begin(), uses.end(), p.from,
            [](const UsePosition& u, uint32_t x) { return u.inst_id < x; });
        const bool starts_with_def = it != uses.end() && it->inst_id <= p.to && it->is_def;
        if (waits_in_slot && !starts_with_def) {
            double stay = 0;
            bool fixed = false;
            for (; it != uses.end() && it->inst_id < best_use && it->inst_id <= p.to; ++it) {
                if (it->requires_reg && !it->is_def) stay += weight(it->inst_id);
                if (it->inst_id == first && it->fixed_reg.is_valid()) fixed = true;
            }
            if (!fixed && stay <= best_cost) {
                spill_and_requeue(idx, first, unhandled);
                return;
            }
        }
    }
    if (hard_q[best.code] < p.end) {
        // Blocked later on: hold the register up to there.
        const uint32_t q = hard_q[best.code] > first ? split_position(first + 1, hard_q[best.code]) : kNever;
        if (q == kNever || q <= p.start || q >= p.end) {
            spill_and_requeue(idx, p.start, unhandled);
            return;
        }
        push_unhandled(unhandled, split_piece(idx, q));
    }
    assign(idx, best);
    for (uint32_t a : holders[best.code]) {
        if (first_intersection(pieces_[a], pieces_[idx]) != kNever) evict_from(a, p.start, unhandled);
    }
}

void LinearScanAllocator::run_scan() {
    std::vector<uint32_t> unhandled;
    for (auto& interval : liveness_.intervals()) {
        if (!interval.vreg.is_valid() || interval.start_id > interval.end_id || interval.segments.empty()) continue;
        if (interval.vreg.id >= vreg_pieces_.size()) continue;
        const uint32_t idx = new_piece(&interval, 0, kNever);
        vreg_pieces_[interval.vreg.id].push_back(idx);
        push_unhandled(unhandled, idx);
    }
    auto later = [this](uint32_t a, uint32_t b) {
        const AllocPiece& x = pieces_[a];
        const AllocPiece& y = pieces_[b];
        if (x.start != y.start) return x.start > y.start;
        if (x.end != y.end) return x.end > y.end;
        return x.seq > y.seq;
    };
    while (!unhandled.empty()) {
        std::pop_heap(unhandled.begin(), unhandled.end(), later);
        const uint32_t idx = unhandled.back();
        unhandled.pop_back();
        const uint32_t pos = pieces_[idx].start;
        active_.erase(std::remove_if(active_.begin(), active_.end(),
                                     [&](uint32_t a) { return pieces_[a].end <= pos; }),
                      active_.end());
        const VReg v = pieces_[idx].vreg();
        // A GC reference live across a call stays in its slot, where the
        // stack map finds it.
        if (v.is_gcref && pieces_[idx].base->spans_call) {
            pieces_[idx].spilled = true;
            slot_of(v);
            continue;
        }
        if (!try_allocate_free(idx, unhandled)) allocate_blocked(idx, unhandled);
    }
}

} // namespace brass::codegen
