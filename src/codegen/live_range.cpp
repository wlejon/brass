#include <brass/codegen/live_range.hpp>
#include <sstream>
#include <algorithm>
#include <cmath>

namespace brass::codegen {

static bool contains_vreg(const std::vector<VReg>& vec, VReg v) {
    return std::find(vec.begin(), vec.end(), v) != vec.end();
}

static void add_vreg_unique(std::vector<VReg>& vec, VReg v) {
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

    // Merge with existing segments
    std::vector<LiveRangeSegment> new_segments;
    LiveRangeSegment current{s, e};

    for (const auto& seg : segments) {
        if (current.end + 1 < seg.start) {
            new_segments.push_back(current);
            current = seg;
        } else if (seg.end + 1 < current.start) {
            new_segments.push_back(seg);
        } else {
            current.start = std::min(current.start, seg.start);
            current.end = std::max(current.end, seg.end);
        }
    }
    new_segments.push_back(current);
    segments = std::move(new_segments);
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

bool LiveInterval::covers(uint32_t id) const noexcept {
    if (id < start_id || id > end_id) return false;
    for (const auto& seg : segments) {
        if (seg.contains(id)) return true;
    }
    return false;
}

bool LiveInterval::overlaps(const LiveInterval& other) const noexcept {
    if (end_id <= other.start_id || other.end_id <= start_id) return false;
    for (const auto& seg1 : segments) {
        for (const auto& seg2 : other.segments) {
            if (seg1.overlaps(seg2)) return true;
        }
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

void LivenessAnalysis::compute_global_liveness() {
    const size_t num_vregs = fn_.vreg_table.size();
    const size_t num_words = (num_vregs + 63) / 64;
    if (num_words == 0) return;

    std::unordered_map<const LirBlock*, std::vector<uint64_t>> live_in_bv;
    std::unordered_map<const LirBlock*, std::vector<uint64_t>> live_out_bv;
    std::unordered_map<const LirBlock*, std::vector<uint64_t>> defs_bv;
    std::unordered_map<const LirBlock*, std::vector<uint64_t>> uses_bv;

    for (const auto& block : fn_.blocks) {
        const auto* b = block.get();
        const auto& bl = block_liveness_[b];
        auto& d = defs_bv[b]; d.assign(num_words, 0);
        auto& u = uses_bv[b]; u.assign(num_words, 0);
        live_in_bv[b].assign(num_words, 0);
        live_out_bv[b].assign(num_words, 0);

        for (VReg v : bl.defs) {
            if (v.is_valid() && v.id < num_vregs) {
                d[v.id / 64] |= (1ULL << (v.id % 64));
            }
        }
        for (VReg v : bl.uses) {
            if (v.is_valid() && v.id < num_vregs) {
                u[v.id / 64] |= (1ULL << (v.id % 64));
            }
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;

        for (auto it = fn_.blocks.rbegin(); it != fn_.blocks.rend(); ++it) {
            const auto* block = it->get();
            auto& out_vec = live_out_bv[block];
            auto& in_vec = live_in_bv[block];
            const auto& def_vec = defs_bv[block];
            const auto& use_vec = uses_bv[block];

            for (const auto* succ : block->successors) {
                const auto& succ_in = live_in_bv[succ];
                for (size_t w = 0; w < num_words; ++w) {
                    out_vec[w] |= succ_in[w];
                }
            }

            for (size_t w = 0; w < num_words; ++w) {
                uint64_t new_in = use_vec[w] | (out_vec[w] & ~def_vec[w]);
                if (new_in != in_vec[w]) {
                    in_vec[w] = new_in;
                    changed = true;
                }
            }
        }
    }

    for (const auto& block : fn_.blocks) {
        const auto* b = block.get();
        BlockLiveness& bl = block_liveness_[b];
        bl.live_in.clear();
        bl.live_out.clear();

        const auto& in_vec = live_in_bv[b];
        const auto& out_vec = live_out_bv[b];

        for (size_t w = 0; w < num_words; ++w) {
            uint64_t in_word = in_vec[w];
            while (in_word != 0) {
                int bit = __builtin_ctzll(in_word);
                uint32_t vid = static_cast<uint32_t>(w * 64 + bit);
                if (vid < num_vregs) {
                    bl.live_in.push_back(fn_.vreg_table[vid].vreg);
                }
                in_word &= in_word - 1;
            }

            uint64_t out_word = out_vec[w];
            while (out_word != 0) {
                int bit = __builtin_ctzll(out_word);
                uint32_t vid = static_cast<uint32_t>(w * 64 + bit);
                if (vid < num_vregs) {
                    bl.live_out.push_back(fn_.vreg_table[vid].vreg);
                }
                out_word &= out_word - 1;
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

    // Sort use positions for all intervals and check if interval spans a call
    for (auto& interval : intervals_) {
        std::sort(interval.use_positions.begin(), interval.use_positions.end(),
            [](const UsePosition& a, const UsePosition& b) {
                return a.inst_id < b.inst_id;
            });

        for (uint32_t call_id : call_inst_ids_) {
            if (interval.covers(call_id)) {
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
    for (const auto& pair : block_liveness_) {
        if (inst_id >= pair.second.start_id && inst_id <= pair.second.end_id) {
            return pair.second.loop_depth;
        }
    }
    return 0;
}

void LivenessAnalysis::compute_loop_depths() {
    size_t n = fn_.blocks.size();
    if (n == 0) return;

    std::unordered_map<const LirBlock*, size_t> block_to_idx;
    for (size_t i = 0; i < n; ++i) {
        block_to_idx[fn_.blocks[i].get()] = i;
        fn_.blocks[i]->loop_depth = 0;
    }

    // Dominance analysis using iterative dataflow
    std::vector<std::vector<bool>> dom(n, std::vector<bool>(n, true));
    dom[0].assign(n, false);
    dom[0][0] = true;

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 1; i < n; ++i) {
            const auto* blk = fn_.blocks[i].get();
            std::vector<bool> new_dom(n, true);

            if (blk->predecessors.empty()) {
                new_dom.assign(n, false);
            } else {
                for (const auto* pred : blk->predecessors) {
                    auto it = block_to_idx.find(pred);
                    if (it != block_to_idx.end()) {
                        size_t p_idx = it->second;
                        for (size_t k = 0; k < n; ++k) {
                            if (!dom[p_idx][k]) {
                                new_dom[k] = false;
                            }
                        }
                    }
                }
            }
            new_dom[i] = true;

            if (new_dom != dom[i]) {
                dom[i] = std::move(new_dom);
                changed = true;
            }
        }
    }

    // Identify backedges and natural loops
    for (size_t i = 0; i < n; ++i) {
        const auto* blk = fn_.blocks[i].get();
        for (const auto* succ : blk->successors) {
            auto it = block_to_idx.find(succ);
            if (it == block_to_idx.end()) continue;
            size_t s_idx = it->second;

            if (dom[i][s_idx]) {
                // Backedge i -> s_idx
                std::vector<bool> in_loop(n, false);
                in_loop[s_idx] = true;
                in_loop[i] = true;

                std::vector<size_t> worklist;
                if (i != s_idx) {
                    worklist.push_back(i);
                }

                while (!worklist.empty()) {
                    size_t curr = worklist.back();
                    worklist.pop_back();

                    const auto* curr_blk = fn_.blocks[curr].get();
                    for (const auto* pred : curr_blk->predecessors) {
                        auto p_it = block_to_idx.find(pred);
                        if (p_it != block_to_idx.end()) {
                            size_t p_idx = p_it->second;
                            if (!in_loop[p_idx]) {
                                in_loop[p_idx] = true;
                                worklist.push_back(p_idx);
                            }
                        }
                    }
                }

                for (size_t k = 0; k < n; ++k) {
                    if (in_loop[k]) {
                        fn_.blocks[k]->loop_depth++;
                    }
                }
            }
        }
    }

    for (const auto& blk : fn_.blocks) {
        block_liveness_[blk.get()].loop_depth = blk->loop_depth;
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
