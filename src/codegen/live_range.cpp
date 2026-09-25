#include <brass/codegen/live_range.hpp>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace brass::codegen {

[[maybe_unused]] static bool contains_vreg(const std::vector<VReg>& vec, VReg v) {
    return std::find(vec.begin(), vec.end(), v) != vec.end();
}

[[maybe_unused]] static void add_vreg_unique(std::vector<VReg>& vec, VReg v) {
    if (v.is_valid() && !contains_vreg(vec, v)) {
        vec.push_back(v);
    }
}

void LiveInterval::add_range(uint32_t s, uint32_t e) {
    if (s > e) return;
    start_id = std::min(start_id, s);
    end_id = std::max(end_id, e);

    if (segments.empty()) {
        segments.push_back({s, e});
        return;
    }

    // Merge with the existing segments: every one that touches or overlaps
    // [s, e] — they are sorted, disjoint and never adjacent, so those form
    // one contiguous run — collapses with it into a single segment in place.
    auto lo = std::lower_bound(segments.begin(), segments.end(), s,
        [](const LiveRangeSegment& seg, uint32_t v) { return seg.end + 1 < v; });
    auto hi = std::upper_bound(lo, segments.end(), e,
        [](uint32_t v, const LiveRangeSegment& seg) { return v + 1 < seg.start; });
    LiveRangeSegment merged{s, e};
    if (lo != hi) {
        merged.start = std::min(merged.start, lo->start);
        merged.end = std::max(merged.end, (hi - 1)->end);
        *lo = merged;
        segments.erase(lo + 1, hi);
    } else {
        segments.insert(lo, merged);
    }
}

void LiveInterval::shorten_start(uint32_t from_id) {
    if (segments.empty()) {
        segments.push_back({from_id, from_id + 1});
        start_id = from_id;
        end_id = std::max(end_id, from_id + 1);
        return;
    }

    for (auto& seg : segments) {
        if (seg.start < from_id && seg.end >= from_id) {
            seg.start = from_id;
            break;
        }
    }

    start_id = UINT32_MAX;
    for (const auto& seg : segments) {
        start_id = std::min(start_id, seg.start);
    }
}

void LiveInterval::add_use_pos(uint32_t id, bool is_def, bool requires_reg, PReg fixed) {
    use_positions.push_back(UsePosition{id, is_def, requires_reg, fixed});
    start_id = std::min(start_id, id);
    end_id = std::max(end_id, id);
}

// Segments are kept sorted by start and disjoint (add_range merges, shorten_start
// only moves a start later), so both queries below are searches, not scans.
bool LiveInterval::covers(uint32_t id) const noexcept {
    if (id < start_id || id > end_id) return false;
    auto it = std::upper_bound(segments.begin(), segments.end(), id,
        [](uint32_t v, const LiveRangeSegment& seg) { return v < seg.start; });
    return it != segments.begin() && (it - 1)->contains(id);
}

bool LiveInterval::overlaps(const LiveInterval& other) const noexcept {
    if (end_id <= other.start_id || other.end_id <= start_id) return false;
    auto a = segments.begin();
    auto b = other.segments.begin();
    while (a != segments.end() && b != other.segments.end()) {
        if (a->overlaps(*b)) return true;
        if (a->end < b->end) ++a; else ++b;
    }
    return false;
}

uint32_t LiveInterval::next_use_after(uint32_t id) const noexcept {
    for (const auto& pos : use_positions) {
        if (pos.inst_id > id) {
            return pos.inst_id;
        }
    }
    return UINT32_MAX;
}

uint32_t LiveInterval::first_use() const noexcept {
    if (use_positions.empty()) return UINT32_MAX;
    uint32_t min_id = UINT32_MAX;
    for (const auto& pos : use_positions) {
        min_id = std::min(min_id, pos.inst_id);
    }
    return min_id;
}

std::string to_string(const LiveInterval& interval) {
    std::ostringstream ss;
    ss << "%v" << interval.vreg.id << " [" << interval.start_id << ", " << interval.end_id << "] segments: ";
    for (size_t i = 0; i < interval.segments.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << "[" << interval.segments[i].start << ", " << interval.segments[i].end << "]";
    }
    ss << " uses: {";
    for (size_t i = 0; i < interval.use_positions.size(); ++i) {
        if (i > 0) ss << ", ";
        ss << interval.use_positions[i].inst_id << (interval.use_positions[i].is_def ? "d" : "u");
    }
    ss << "} weight=" << interval.spill_weight;
    return ss.str();
}

LivenessAnalysis::LivenessAnalysis(LirFunction& fn)
    : fn_(fn) {}

void LivenessAnalysis::run() {
    intervals_.clear();
    block_liveness_.clear();
    call_inst_ids_.clear();

    // 1. Initialize intervals for all virtual registers
    intervals_.resize(fn_.vreg_table.size());
    for (size_t i = 0; i < fn_.vreg_table.size(); ++i) {
        intervals_[i] = LiveInterval(fn_.vreg_table[i].vreg);
    }

    // 2. Assign instruction IDs
    assign_instruction_ids();

    // 3. Compute local defs/uses for each block
    compute_local_liveness();

    // 4. Compute global LiveIn and LiveOut
    compute_global_liveness();

    // 5. Build live intervals
    build_intervals();

    // 5.5 Compute loop depths
    compute_loop_depths();

    // 6. Compute spill weights
    compute_spill_weights();
}

void LivenessAnalysis::assign_instruction_ids() {
    uint32_t current_id = 0;
    for (auto& block : fn_.blocks) {
        BlockLiveness& bl = block_liveness_[block.get()];
        bl.start_id = current_id;
        for (auto& inst : block->instructions) {
            inst->id = current_id;
            if (inst->is_call() || inst->opcode == LirOpcode::Safepoint) {
                call_inst_ids_.push_back(current_id);
            }
            current_id += 2; // Step by 2 for split points
        }
        bl.end_id = (current_id > 0) ? current_id - 2 : 0;
    }
}

void LivenessAnalysis::compute_local_liveness() {
    const size_t num_vregs = fn_.vreg_table.size();
    std::vector<uint8_t> def_set(num_vregs, 0);
    std::vector<uint8_t> use_set(num_vregs, 0);

    for (const auto& block : fn_.blocks) {
        BlockLiveness& bl = block_liveness_[block.get()];
        bl.defs.clear();
        bl.uses.clear();

        auto add_use = [&](VReg v) {
            if (v.is_valid() && v.id < num_vregs && !def_set[v.id]) {
                if (!use_set[v.id]) {
                    use_set[v.id] = 1;
                    bl.uses.push_back(v);
                }
            }
        };

        auto add_def = [&](VReg v) {
            if (v.is_valid() && v.id < num_vregs) {
                if (!def_set[v.id]) {
                    def_set[v.id] = 1;
                    bl.defs.push_back(v);
                }
            }
        };

        for (const auto& inst : block->instructions) {
            for (const auto& u : inst->uses) {
                if (u.is_vreg()) {
                    add_use(u.vreg_val);
                } else if (u.is_mem()) {
                    add_use(u.mem_val.base_vreg);
                    add_use(u.mem_val.index_vreg);
                }
            }
            for (const auto& d : inst->defs) {
                if (d.is_vreg()) {
                    add_def(d.vreg_val);
                } else if (d.is_mem()) {
                    add_use(d.mem_val.base_vreg);
                    add_use(d.mem_val.index_vreg);
                }
            }
        }

        for (VReg v : bl.defs) def_set[v.id] = 0;
        for (VReg v : bl.uses) use_set[v.id] = 0;
    }
}

// The least solution of
//     live_in(B)  = uses(B) ∪ (live_out(B) − defs(B))
//     live_out(B) = ∪ live_in(S) over B's successors S
// found one register at a time: from each block with an upward-exposed use,
// walk predecessors backwards, marking the register live out of each and live
// into each that does not define it, and stop where it is defined or already
// marked. The work is the size of the answer. The textbook solver — a bit
// vector per block over every register — is blocks x registers of memory and
// of work, and a bundled library's top level (sixty thousand blocks, three
// hundred thousand registers) needed gigabytes and minutes.
void LivenessAnalysis::compute_global_liveness() {
    const size_t num_vregs = fn_.vreg_table.size();
    const size_t n = fn_.blocks.size();
    std::vector<BlockLiveness*> info(n);
    for (size_t i = 0; i < n; ++i) {
        info[i] = &block_liveness_[fn_.blocks[i].get()];
        info[i]->live_in.clear();
        info[i]->live_out.clear();
    }
    if (num_vregs == 0 || n == 0) return;

    std::unordered_map<const LirBlock*, uint32_t> index;
    index.reserve(n);
    for (size_t i = 0; i < n; ++i) index.emplace(fn_.blocks[i].get(), static_cast<uint32_t>(i));

    // Predecessors come from the successor lists, the edges the equations
    // read, so the walk cannot miss one.
    std::vector<std::vector<uint32_t>> preds(n);
    for (size_t i = 0; i < n; ++i) {
        for (const auto* succ : fn_.blocks[i]->successors) {
            if (auto it = index.find(succ); it != index.end()) {
                preds[it->second].push_back(static_cast<uint32_t>(i));
            }
        }
    }

    // Per register, the blocks using it upward-exposed and the blocks
    // defining it: flat arrays indexed by per-register offsets.
    auto bucket = [&](std::vector<VReg> BlockLiveness::*member, std::vector<uint32_t>& start,
                      std::vector<uint32_t>& blocks) {
        start.assign(num_vregs + 1, 0);
        for (size_t i = 0; i < n; ++i) {
            for (VReg v : info[i]->*member) {
                if (v.is_valid() && v.id < num_vregs) ++start[v.id + 1];
            }
        }
        for (size_t v = 0; v < num_vregs; ++v) start[v + 1] += start[v];
        blocks.resize(start[num_vregs]);
        std::vector<uint32_t> fill(start.begin(), start.end() - 1);
        for (size_t i = 0; i < n; ++i) {
            for (VReg v : info[i]->*member) {
                if (v.is_valid() && v.id < num_vregs) blocks[fill[v.id]++] = static_cast<uint32_t>(i);
            }
        }
    };
    std::vector<uint32_t> use_start, use_blocks, def_start, def_blocks;
    bucket(&BlockLiveness::uses, use_start, use_blocks);
    bucket(&BlockLiveness::defs, def_start, def_blocks);

    // Stamped with the register's id + 1, so nothing is reset per register.
    std::vector<uint32_t> def_mark(n, 0), in_mark(n, 0), out_mark(n, 0);
    std::vector<uint32_t> worklist;
    // Registers in id order, so every block's lists come out sorted by id.
    for (uint32_t v = 0; v < num_vregs; ++v) {
        if (use_start[v] == use_start[v + 1]) continue;
        const uint32_t stamp = v + 1;
        const VReg reg = fn_.vreg_table[v].vreg;
        for (uint32_t k = def_start[v]; k < def_start[v + 1]; ++k) def_mark[def_blocks[k]] = stamp;
        worklist.clear();
        for (uint32_t k = use_start[v]; k < use_start[v + 1]; ++k) {
            const uint32_t b = use_blocks[k];
            if (in_mark[b] == stamp) continue;
            in_mark[b] = stamp;
            info[b]->live_in.push_back(reg);
            worklist.push_back(b);
        }
        while (!worklist.empty()) {
            const uint32_t b = worklist.back();
            worklist.pop_back();
            for (uint32_t p : preds[b]) {
                if (out_mark[p] != stamp) {
                    out_mark[p] = stamp;
                    info[p]->live_out.push_back(reg);
                }
                if (def_mark[p] == stamp || in_mark[p] == stamp) continue;
                in_mark[p] = stamp;
                info[p]->live_in.push_back(reg);
                worklist.push_back(p);
            }
        }
    }
}

void LivenessAnalysis::build_intervals() {
    for (auto it = fn_.blocks.rbegin(); it != fn_.blocks.rend(); ++it) {
        auto* block = it->get();
        const BlockLiveness& bl = block_liveness_[block];

        // Add whole-block live range for each variable live out of the block
        for (VReg v : bl.live_out) {
            if (v.is_valid() && v.id < intervals_.size()) {
                intervals_[v.id].add_range(bl.start_id, bl.end_id + 1);
            }
        }

        // Walk instructions in reverse
        for (auto inst_it = block->instructions.rbegin(); inst_it != block->instructions.rend(); ++inst_it) {
            const auto& inst = *inst_it;
            uint32_t inst_id = inst->id;

            // Defs shorten the live range start to inst_id
            for (size_t i = 0; i < inst->defs.size(); ++i) {
                const auto& d = inst->defs[i];
                if (d.is_vreg() && d.vreg_val.is_valid() && d.vreg_val.id < intervals_.size()) {
                    FixedConstraint fc = (i < inst->def_constraints.size()) ? inst->def_constraints[i] : FixedConstraint::none();
                    intervals_[d.vreg_val.id].add_use_pos(inst_id, true, true, fc.fixed_preg);
                    intervals_[d.vreg_val.id].shorten_start(inst_id);
                } else if (d.is_mem()) {
                    if (d.mem_val.base_vreg.is_valid() && d.mem_val.base_vreg.id < intervals_.size()) {
                        intervals_[d.mem_val.base_vreg.id].add_use_pos(inst_id, false, true);
                        intervals_[d.mem_val.base_vreg.id].add_range(bl.start_id, inst_id);
                    }
                    if (d.mem_val.index_vreg.is_valid() && d.mem_val.index_vreg.id < intervals_.size()) {
                        intervals_[d.mem_val.index_vreg.id].add_use_pos(inst_id, false, true);
                        intervals_[d.mem_val.index_vreg.id].add_range(bl.start_id, inst_id);
                    }
                }
            }

            // Uses extend the live range from bl.start_id to inst_id
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                const auto& u = inst->uses[i];
                FixedConstraint fc = (i < inst->use_constraints.size()) ? inst->use_constraints[i] : FixedConstraint::none();

                if (u.is_vreg() && u.vreg_val.is_valid() && u.vreg_val.id < intervals_.size()) {
                    intervals_[u.vreg_val.id].add_use_pos(inst_id, false, true, fc.fixed_preg);
                    intervals_[u.vreg_val.id].add_range(bl.start_id, inst_id);
                } else if (u.is_mem()) {
                    if (u.mem_val.base_vreg.is_valid() && u.mem_val.base_vreg.id < intervals_.size()) {
                        intervals_[u.mem_val.base_vreg.id].add_use_pos(inst_id, false, true);
                        intervals_[u.mem_val.base_vreg.id].add_range(bl.start_id, inst_id);
                    }
                    if (u.mem_val.index_vreg.is_valid() && u.mem_val.index_vreg.id < intervals_.size()) {
                        intervals_[u.mem_val.index_vreg.id].add_use_pos(inst_id, false, true);
                        intervals_[u.mem_val.index_vreg.id].add_range(bl.start_id, inst_id);
                    }
                }
            }
        }
    }

    // Sort use positions for all intervals and check if interval spans a call.
    // call_inst_ids_ is in id order, so a segment spans a call exactly when the
    // first call id at or after its start is still inside it.
    for (auto& interval : intervals_) {
        std::sort(interval.use_positions.begin(), interval.use_positions.end(),
            [](const UsePosition& a, const UsePosition& b) {
                return a.inst_id < b.inst_id;
            });

        for (const auto& seg : interval.segments) {
            auto it = std::lower_bound(call_inst_ids_.begin(), call_inst_ids_.end(), seg.start);
            if (it != call_inst_ids_.end() && *it <= seg.end) {
                interval.spans_call = true;
                break;
            }
        }
    }
}

LiveInterval* LivenessAnalysis::get_interval(VReg v) {
    if (v.id < intervals_.size()) {
        return &intervals_[v.id];
    }
    return nullptr;
}

const LiveInterval* LivenessAnalysis::get_interval(VReg v) const {
    if (v.id < intervals_.size()) {
        return &intervals_[v.id];
    }
    return nullptr;
}

const BlockLiveness& LivenessAnalysis::block_liveness(const LirBlock* b) const {
    auto it = block_liveness_.find(b);
    if (it != block_liveness_.end()) {
        return it->second;
    }
    static BlockLiveness empty;
    return empty;
}

uint32_t LivenessAnalysis::get_loop_depth_at(uint32_t inst_id) const {
    // block_ranges_ holds the non-empty blocks by start id; the one that
    // starts last at or before inst_id is the only one that can hold it.
    auto it = std::upper_bound(block_ranges_.begin(), block_ranges_.end(), inst_id,
        [](uint32_t id, const BlockRange& r) { return id < r.start_id; });
    if (it == block_ranges_.begin()) return 0;
    --it;
    return inst_id <= it->end_id ? it->loop_depth : 0;
}

void LivenessAnalysis::compute_loop_depths() {
    block_ranges_.clear();
    size_t n = fn_.blocks.size();
    if (n == 0) return;

    std::unordered_map<const LirBlock*, uint32_t> block_to_idx;
    block_to_idx.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        block_to_idx[fn_.blocks[i].get()] = static_cast<uint32_t>(i);
        fn_.blocks[i]->loop_depth = 0;
    }
    std::vector<std::vector<uint32_t>> preds(n), succs(n);
    for (size_t i = 0; i < n; ++i) {
        for (const auto* p : fn_.blocks[i]->predecessors) {
            if (auto it = block_to_idx.find(p); it != block_to_idx.end()) preds[i].push_back(it->second);
        }
        for (const auto* s : fn_.blocks[i]->successors) {
            if (auto it = block_to_idx.find(s); it != block_to_idx.end()) succs[i].push_back(it->second);
        }
    }

    // Dominators. The roots are the entry and every block without
    // predecessors (resume entries), each dominated by itself alone: a
    // virtual root `n` above them all. Immediate dominators by the
    // Cooper-Harvey-Kennedy iteration in reverse post-order, and a dominance
    // question answered by the dominator tree's DFS intervals — a bit set of
    // dominators per block is blocks² of memory and of work, which a
    // sixty-thousand-block function cannot afford. A block no root reaches
    // keeps what the maximal solution of the set equations gives it: every
    // block dominates it.
    const uint32_t root = static_cast<uint32_t>(n);
    const uint32_t kNone = UINT32_MAX;
    std::vector<uint32_t> rpo_num(n + 1, kNone);
    std::vector<uint32_t> rpo;  // blocks, reverse post-order, root excluded
    {
        std::vector<uint32_t> post;
        std::vector<uint8_t> seen(n, 0);
        std::vector<std::pair<uint32_t, uint32_t>> stack;
        auto dfs_from = [&](uint32_t start) {
            if (seen[start]) return;
            seen[start] = 1;
            stack.push_back({start, 0});
            while (!stack.empty()) {
                auto& [b, k] = stack.back();
                if (k < succs[b].size()) {
                    const uint32_t s = succs[b][k++];
                    if (!seen[s]) {
                        seen[s] = 1;
                        stack.push_back({s, 0});
                    }
                } else {
                    post.push_back(b);
                    stack.pop_back();
                }
            }
        };
        // The root's successors, last first, so the entry is first in RPO.
        for (size_t i = n; i-- > 1;) {
            if (fn_.blocks[i]->predecessors.empty()) dfs_from(static_cast<uint32_t>(i));
        }
        dfs_from(0);
        rpo.assign(post.rbegin(), post.rend());
        rpo_num[root] = 0;
        for (size_t k = 0; k < rpo.size(); ++k) rpo_num[rpo[k]] = static_cast<uint32_t>(k + 1);
    }
    auto is_root_child = [&](uint32_t b) { return b == 0 || fn_.blocks[b]->predecessors.empty(); };
    std::vector<uint32_t> idom(n + 1, kNone);
    idom[root] = root;
    auto intersect = [&](uint32_t a, uint32_t b) {
        while (a != b) {
            while (rpo_num[a] > rpo_num[b]) a = idom[a];
            while (rpo_num[b] > rpo_num[a]) b = idom[b];
        }
        return a;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (uint32_t b : rpo) {
            uint32_t d = is_root_child(b) ? root : kNone;
            if (b != 0) {
                for (uint32_t p : preds[b]) {
                    if (idom[p] == kNone) continue;  // unprocessed or unreachable
                    d = d == kNone ? p : intersect(p, d);
                }
            }
            if (d != kNone && idom[b] != d) {
                idom[b] = d;
                changed = true;
            }
        }
    }
    std::vector<uint32_t> pre(n + 1, 0), post_end(n + 1, 0);
    {
        std::vector<std::vector<uint32_t>> children(n + 1);
        for (uint32_t b : rpo) {
            if (idom[b] != kNone) children[idom[b]].push_back(b);
        }
        uint32_t clock = 0;
        std::vector<std::pair<uint32_t, uint32_t>> stack{{root, 0}};
        pre[root] = clock++;
        while (!stack.empty()) {
            auto& [b, k] = stack.back();
            if (k < children[b].size()) {
                const uint32_t c = children[b][k++];
                pre[c] = clock++;
                stack.push_back({c, 0});
            } else {
                post_end[b] = clock;
                stack.pop_back();
            }
        }
    }
    auto reached = [&](uint32_t b) { return idom[b] != kNone; };
    auto dominates = [&](uint32_t a, uint32_t b) {
        if (!reached(b)) return true;
        if (!reached(a)) return false;
        return pre[a] <= pre[b] && pre[b] < post_end[a];
    };

    // Identify backedges and natural loops. A loop's blocks are found by
    // walking predecessors back from the latch to the header, marked with
    // the loop's own stamp so nothing is cleared between loops.
    std::vector<uint32_t> in_loop(n, 0);
    uint32_t loop_stamp = 0;
    std::vector<uint32_t> worklist;
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t s_idx : succs[i]) {
            if (!dominates(s_idx, i)) continue;
            // Backedge i -> s_idx
            ++loop_stamp;
            in_loop[s_idx] = loop_stamp;
            in_loop[i] = loop_stamp;
            fn_.blocks[s_idx]->loop_depth++;
            if (i != s_idx) fn_.blocks[i]->loop_depth++;
            worklist.clear();
            if (i != s_idx) worklist.push_back(i);
            while (!worklist.empty()) {
                const uint32_t curr = worklist.back();
                worklist.pop_back();
                for (uint32_t p_idx : preds[curr]) {
                    if (in_loop[p_idx] == loop_stamp) continue;
                    in_loop[p_idx] = loop_stamp;
                    fn_.blocks[p_idx]->loop_depth++;
                    worklist.push_back(p_idx);
                }
            }
        }
    }

    for (const auto& blk : fn_.blocks) {
        BlockLiveness& bl = block_liveness_[blk.get()];
        bl.loop_depth = blk->loop_depth;
        // Ids are assigned in block order, so this is already sorted by start;
        // an empty block owns no id and is left out.
        if (!blk->instructions.empty()) {
            block_ranges_.push_back(BlockRange{bl.start_id, bl.end_id, bl.loop_depth});
        }
    }
}

void LivenessAnalysis::compute_spill_weights() {
    for (auto& interval : intervals_) {
        if (interval.start_id > interval.end_id) continue;
        float len = static_cast<float>(interval.end_id - interval.start_id + 1);
        float weighted_uses = 0.0f;
        uint32_t max_depth = 0;

        for (const auto& pos : interval.use_positions) {
            uint32_t depth = get_loop_depth_at(pos.inst_id);
            max_depth = std::max(max_depth, depth);
            weighted_uses += std::pow(8.0f, static_cast<float>(depth));
        }

        interval.max_loop_depth = max_depth;
        interval.spill_weight = weighted_uses / len;
    }
}

} // namespace brass::codegen
