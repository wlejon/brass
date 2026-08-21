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
    if (end_id < other.start_id || other.end_id < start_id) return false;
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

    // 6. Compute spill weights
    compute_spill_weights();
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

void LivenessAnalysis::assign_instruction_ids() {
    uint32_t current_id = 0;
    for (const auto& block : fn_.blocks) {
        BlockLiveness& bl = block_liveness_[block.get()];
        bl.start_id = current_id;

        for (const auto& inst : block->instructions) {
            inst->id = current_id;
            if (inst->is_call()) {
                call_inst_ids_.push_back(current_id);
            }
            current_id += 2;
        }

        bl.end_id = current_id > 0 ? (current_id - 2) : 0;
    }
}

void LivenessAnalysis::compute_local_liveness() {
    for (const auto& block : fn_.blocks) {
        BlockLiveness& bl = block_liveness_[block.get()];

        for (const auto& inst : block->instructions) {
            // Uses
            for (const auto& u : inst->uses) {
                if (u.is_vreg()) {
                    if (!contains_vreg(bl.defs, u.vreg_val)) {
                        add_vreg_unique(bl.uses, u.vreg_val);
                    }
                } else if (u.is_mem()) {
                    if (u.mem_val.base_vreg.is_valid() && !contains_vreg(bl.defs, u.mem_val.base_vreg)) {
                        add_vreg_unique(bl.uses, u.mem_val.base_vreg);
                    }
                    if (u.mem_val.index_vreg.is_valid() && !contains_vreg(bl.defs, u.mem_val.index_vreg)) {
                        add_vreg_unique(bl.uses, u.mem_val.index_vreg);
                    }
                }
            }

            // Defs
            for (const auto& d : inst->defs) {
                if (d.is_vreg()) {
                    add_vreg_unique(bl.defs, d.vreg_val);
                }
            }
        }
    }
}

void LivenessAnalysis::compute_global_liveness() {
    bool changed = true;
    while (changed) {
        changed = false;

        // Process blocks in reverse post-order
        for (auto it = fn_.blocks.rbegin(); it != fn_.blocks.rend(); ++it) {
            auto* block = it->get();
            BlockLiveness& bl = block_liveness_[block];

            // LiveOut = union of LiveIn of all successors
            std::vector<VReg> new_live_out;
            for (const auto* succ : block->successors) {
                const auto& succ_bl = block_liveness_[succ];
                for (VReg v : succ_bl.live_in) {
                    add_vreg_unique(new_live_out, v);
                }
            }

            // LiveIn = Uses union (LiveOut \ Defs)
            std::vector<VReg> new_live_in = bl.uses;
            for (VReg v : new_live_out) {
                if (!contains_vreg(bl.defs, v)) {
                    add_vreg_unique(new_live_in, v);
                }
            }

            if (new_live_in != bl.live_in || new_live_out != bl.live_out) {
                bl.live_in = std::move(new_live_in);
                bl.live_out = std::move(new_live_out);
                changed = true;
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
            intervals_[v.id].add_range(bl.start_id, bl.end_id + 1);
        }

        // Walk instructions in reverse
        for (auto inst_it = block->instructions.rbegin(); inst_it != block->instructions.rend(); ++inst_it) {
            const auto& inst = *inst_it;
            uint32_t inst_id = inst->id;

            // Defs shorten the live range start to inst_id
            for (size_t i = 0; i < inst->defs.size(); ++i) {
                const auto& d = inst->defs[i];
                if (d.is_vreg()) {
                    FixedConstraint fc = (i < inst->def_constraints.size()) ? inst->def_constraints[i] : FixedConstraint::none();
                    intervals_[d.vreg_val.id].add_use_pos(inst_id, true, true, fc.fixed_preg);
                    intervals_[d.vreg_val.id].add_range(inst_id, inst_id + 1);
                }
            }

            // Uses extend the live range from bl.start_id to inst_id
            for (size_t i = 0; i < inst->uses.size(); ++i) {
                const auto& u = inst->uses[i];
                FixedConstraint fc = (i < inst->use_constraints.size()) ? inst->use_constraints[i] : FixedConstraint::none();

                if (u.is_vreg()) {
                    intervals_[u.vreg_val.id].add_use_pos(inst_id, false, true, fc.fixed_preg);
                    intervals_[u.vreg_val.id].add_range(bl.start_id, inst_id);
                } else if (u.is_mem()) {
                    if (u.mem_val.base_vreg.is_valid()) {
                        intervals_[u.mem_val.base_vreg.id].add_use_pos(inst_id, false, true);
                        intervals_[u.mem_val.base_vreg.id].add_range(bl.start_id, inst_id);
                    }
                    if (u.mem_val.index_vreg.is_valid()) {
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

void LivenessAnalysis::compute_spill_weights() {
    for (auto& interval : intervals_) {
        if (interval.start_id > interval.end_id) continue;
        float len = static_cast<float>(interval.end_id - interval.start_id + 1);
        float uses_count = static_cast<float>(interval.use_positions.size());
        interval.spill_weight = uses_count / len;
    }
}

} // namespace brass::codegen
