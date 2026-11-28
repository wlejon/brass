#include <brass/target/ptx/ptx_schedule.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace brass::ptx {

namespace {

uint64_t pack_reg(Reg r) noexcept {
    return (static_cast<uint64_t>(static_cast<uint8_t>(r.cls)) << 32) | static_cast<uint64_t>(r.index);
}

template <typename F>
void visit_operand_regs(const Operand& o, F&& f) {
    switch (o.kind) {
        case OperandKind::Reg:    if (o.reg_val.valid()) f(o.reg_val); break;
        case OperandKind::Addr:   if (o.addr_base.valid()) f(o.addr_base); break;
        case OperandKind::Vector: for (const Reg& r : o.elems) { if (r.valid()) f(r); } break;
        default: break;
    }
}

void collect_defs_uses(const Inst& inst, std::vector<Reg>& defs, std::vector<Reg>& uses) {
    defs.clear();
    uses.clear();
    for (const auto& o : inst.dsts) {
        visit_operand_regs(o, [&](Reg r) { defs.push_back(r); });
    }
    if (inst.has_guard && inst.guard_reg.valid()) {
        uses.push_back(inst.guard_reg);
    }
    for (const auto& o : inst.srcs) {
        visit_operand_regs(o, [&](Reg r) { uses.push_back(r); });
    }
}

bool inst_is_hard_fence(const Inst& inst) noexcept {
    switch (inst.op) {
        case Opcode::bar:
        case Opcode::cp_async:
        case Opcode::cp_async_commit:
        case Opcode::cp_async_wait_group:
        case Opcode::cp_async_wait_all:
        case Opcode::shfl:
        case Opcode::atom:
        case Opcode::call:
        case Opcode::bra:
        case Opcode::ret:
        case Opcode::exit:
        case Opcode::trap:
        case Opcode::vote:
        case Opcode::activemask:
        case Opcode::wmma_load:
        case Opcode::wmma_mma:
        case Opcode::wmma_store:
        case Opcode::createpolicy_evict_first:
            return true;
        default:
            break;
    }
    if (inst.state_space == StateSpace::param) return true;
    for (const auto& o : inst.srcs) {
        if (o.kind == OperandKind::Param) return true;
        if (o.kind == OperandKind::Special && !is_invariant(o.special_reg)) return true;
    }
    return false;
}

uint32_t inst_latency(const Inst& inst) noexcept {
    switch (inst.op) {
        case Opcode::ld:
            if (inst.state_space == StateSpace::global || inst.state_space == StateSpace::none) return 24u;
            if (inst.state_space == StateSpace::shared) return 8u;
            return 6u;
        case Opcode::ldmatrix: return 8u;
        case Opcode::mma_sync: return 12u;
        case Opcode::div:
        case Opcode::rem:      return 20u;
        case Opcode::ex2:
        case Opcode::lg2:
        case Opcode::rcp:
        case Opcode::rsqrt:
        case Opcode::sqrt:
        case Opcode::sin:
        case Opcode::cos:
        case Opcode::tanh:     return 8u;
        case Opcode::mul:
        case Opcode::mad:
        case Opcode::fma:
        case Opcode::dp4a:     return 4u;
        case Opcode::st:       return 4u;
        default:               return 2u;
    }
}

bool is_pure_remat_candidate(const Inst& inst) noexcept {
    if (inst.has_guard || inst.dsts.size() != 1 || !inst.dsts[0].is_reg()) return false;
    switch (inst.op) {
        case Opcode::mov:
            return inst.srcs.size() == 1 && (inst.srcs[0].is_imm_int() || inst.srcs[0].is_imm_float());
        case Opcode::add:
        case Opcode::sub:
        case Opcode::shl:
        case Opcode::shr:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
            for (const auto& s : inst.srcs) {
                if (!s.is_reg() && !s.is_imm_int()) return false;
            }
            return inst.srcs.size() <= 2;
        default:
            return false;
    }
}

// Counts definitions/uses globally to verify safe single-def/single-use rematerialization.
struct GlobalRegInfo {
    std::array<std::vector<uint32_t>, kRegClassCount> defs;
    std::array<std::vector<uint32_t>, kRegClassCount> uses;

    explicit GlobalRegInfo(const Function& fn) {
        std::array<uint32_t, kRegClassCount> sz = fn.reg_counts;
        auto grow = [&](Reg r) {
            const size_t c = static_cast<size_t>(r.cls);
            if (sz[c] <= r.index) sz[c] = r.index + 1;
        };
        std::vector<Reg> d, u;
        for (const auto& blk : fn.blocks) {
            for (const auto& inst : blk->insts) {
                collect_defs_uses(inst, d, u);
                for (Reg r : d) grow(r);
                for (Reg r : u) grow(r);
            }
        }
        for (size_t c = 0; c < kRegClassCount; ++c) {
            defs[c].assign(sz[c], 0);
            uses[c].assign(sz[c], 0);
        }
        for (const auto& blk : fn.blocks) {
            for (const auto& inst : blk->insts) {
                collect_defs_uses(inst, d, u);
                for (Reg r : d) defs[static_cast<size_t>(r.cls)][r.index]++;
                for (Reg r : u) uses[static_cast<size_t>(r.cls)][r.index]++;
            }
        }
    }
};

uint32_t remat_sink_block(Block& blk, const GlobalRegInfo& ginfo) {
    uint32_t remats = 0;
    const size_t n = blk.insts.size();
    if (n < 14) return 0;

    std::vector<Reg> defs, uses;
    bool progress = true;
    uint32_t pass_cap = 32;
    while (progress && pass_cap-- > 0) {
        progress = false;
        const size_t cur_n = blk.insts.size();
        for (size_t i = 0; i < cur_n; ++i) {
            const Inst& def_inst = blk.insts[i];
            if (!is_pure_remat_candidate(def_inst)) continue;
            const Reg dreg = def_inst.dsts[0].reg_val;
            if (!dreg.valid()) continue;
            const size_t dc = static_cast<size_t>(dreg.cls);
            if (ginfo.defs[dc][dreg.index] != 1u || ginfo.uses[dc][dreg.index] != 1u) continue;

            // Collect source registers of `def_inst`
            collect_defs_uses(def_inst, defs, uses);
            std::vector<Reg> src_regs = uses;
            if (src_regs.size() > 1) continue; // move imm or single-src offset/mask

            // Locate unique use in the same block
            size_t use_idx = cur_n;
            bool aborted = false;
            for (size_t j = i + 1; j < cur_n; ++j) {
                const Inst& J = blk.insts[j];
                if (J.op == Opcode::bra || J.op == Opcode::ret || J.op == Opcode::exit || J.op == Opcode::trap) {
                    aborted = true;
                    break;
                }
                collect_defs_uses(J, defs, uses);
                bool reads_d = false;
                for (Reg ur : uses) { if (ur == dreg) { reads_d = true; break; } }
                if (reads_d) {
                    use_idx = j;
                    break;
                }
                for (Reg dr : defs) {
                    for (Reg sr : src_regs) {
                        if (dr == sr) { aborted = true; break; }
                    }
                    if (dr == dreg) { aborted = true; break; }
                }
                if (aborted) break;
            }
            if (aborted || use_idx >= cur_n || use_idx <= i + 10) continue;

            // If `def_inst` uses a register source `sr`, only sink it if `sr` already stays live at or past `use_idx`
            bool src_already_live_after = src_regs.empty();
            if (!src_already_live_after) {
                const Reg sr = src_regs[0];
                for (size_t k = use_idx; k < cur_n; ++k) {
                    collect_defs_uses(blk.insts[k], defs, uses);
                    for (Reg ur : uses) { if (ur == sr) { src_already_live_after = true; break; } }
                    if (src_already_live_after) break;
                    bool redefined = false;
                    for (Reg dr : defs) { if (dr == sr) { redefined = true; break; } }
                    if (redefined) break;
                }
            }
            if (!src_already_live_after) continue;

            Inst moved = std::move(blk.insts[i]);
            blk.insts.erase(blk.insts.begin() + static_cast<std::ptrdiff_t>(i));
            blk.insts.insert(blk.insts.begin() + static_cast<std::ptrdiff_t>(use_idx - 1), std::move(moved));
            remats++;
            progress = true;
            break;
        }
    }
    return remats;
}

uint32_t schedule_region(std::vector<Inst>& region,
                         const PtxScheduleOptions& opts,
                         uint32_t fn_peak_live,
                         bool has_mma) {
    const size_t n = region.size();
    if (n <= 1 || opts.mode == PtxSchedMode::None) return 0;

    std::vector<std::vector<Reg>> inst_defs(n), inst_uses(n);
    for (size_t i = 0; i < n; ++i) {
        collect_defs_uses(region[i], inst_defs[i], inst_uses[i]);
    }

    // Build exact RAW, WAR, WAW, and memory-space edges (directed acyclic graph: u -> v means u precedes v)
    std::vector<std::vector<uint32_t>> succ(n);
    std::vector<uint32_t> indeg(n, 0);
    std::vector<uint8_t> edge_mat(n * n, 0);
    auto add_edge = [&](size_t u, size_t v) {
        if (u >= v) return;
        if (!edge_mat[u * n + v]) {
            edge_mat[u * n + v] = 1;
            succ[u].push_back(static_cast<uint32_t>(v));
            indeg[v]++;
        }
    };

    for (size_t i = 0; i < n; ++i) {
        const bool i_is_st = (region[i].op == Opcode::st);
        const bool i_is_ld = (region[i].op == Opcode::ld);
        const bool i_is_mem = i_is_st || i_is_ld;
        for (size_t j = i + 1; j < n; ++j) {
            bool dep = false;
            // Register RAW, WAW, WAR
            for (Reg d : inst_defs[i]) {
                for (Reg u : inst_uses[j]) { if (d == u) { dep = true; break; } }
                if (dep) break;
                for (Reg d2 : inst_defs[j]) { if (d == d2) { dep = true; break; } }
                if (dep) break;
            }
            if (!dep) {
                for (Reg u : inst_uses[i]) {
                    for (Reg d2 : inst_defs[j]) { if (u == d2) { dep = true; break; } }
                    if (dep) break;
                }
            }
            // Same-space (or unknown space) memory hazard involving any store
            if (!dep && i_is_mem) {
                const bool j_is_st = (region[j].op == Opcode::st);
                const bool j_is_ld = (region[j].op == Opcode::ld);
                if ((i_is_st && (j_is_st || j_is_ld)) || (i_is_ld && j_is_st)) {
                    const StateSpace si = region[i].state_space;
                    const StateSpace sj = region[j].state_space;
                    if (si == sj || si == StateSpace::none || sj == StateSpace::none ||
                        si == StateSpace::global || sj == StateSpace::global) {
                        dep = true;
                    }
                }
            }
            if (dep) add_edge(i, j);
        }
    }

    // Critical path height to exit of region
    std::vector<uint32_t> height(n, 0);
    for (size_t ii = n; ii-- > 0;) {
        uint32_t mx = 0;
        for (uint32_t v : succ[ii]) mx = (std::max)(mx, height[v]);
        height[ii] = mx + inst_latency(region[ii]);
    }

    // Track last use of registers inside this region for live-delta scoring
    std::vector<uint64_t> active_regs;
    std::vector<std::vector<uint64_t>> live_uses_at(n);
    for (size_t i = 0; i < n; ++i) {
        for (Reg r : inst_uses[i]) live_uses_at[i].push_back(pack_reg(r));
    }

    std::vector<uint8_t> scheduled(n, 0);
    std::vector<uint32_t> remaining_uses_count;
    std::vector<uint64_t> all_regs_seen;
    for (size_t i = 0; i < n; ++i) {
        for (Reg r : inst_uses[i]) all_regs_seen.push_back(pack_reg(r));
        for (Reg r : inst_defs[i]) all_regs_seen.push_back(pack_reg(r));
    }
    std::sort(all_regs_seen.begin(), all_regs_seen.end());
    all_regs_seen.erase(std::unique(all_regs_seen.begin(), all_regs_seen.end()), all_regs_seen.end());
    remaining_uses_count.assign(all_regs_seen.size(), 0);
    auto reg_slot = [&](Reg r) -> int {
        const uint64_t pk = pack_reg(r);
        auto it = std::lower_bound(all_regs_seen.begin(), all_regs_seen.end(), pk);
        return (it != all_regs_seen.end() && *it == pk) ? static_cast<int>(it - all_regs_seen.begin()) : -1;
    };
    for (size_t i = 0; i < n; ++i) {
        for (Reg r : inst_uses[i]) {
            const int s = reg_slot(r);
            if (s >= 0) remaining_uses_count[s]++;
        }
    }
    std::vector<uint8_t> reg_live(all_regs_seen.size(), 0);

    std::vector<uint32_t> ready;
    ready.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        if (indeg[i] == 0) ready.push_back(static_cast<uint32_t>(i));
    }

    PtxSchedMode eff_mode = opts.mode;
    if (eff_mode == PtxSchedMode::Auto) {
        if (fn_peak_live > opts.reg_pressure_limit) eff_mode = PtxSchedMode::MinRegPressure;
        else if (has_mma) eff_mode = PtxSchedMode::MmaInterleave;
        else eff_mode = PtxSchedMode::LatencyHide;
    }

    std::vector<uint32_t> order;
    order.reserve(n);
    bool last_was_mma = false;
    uint32_t cur_live = 0;

    while (!ready.empty()) {
        size_t best_pos = 0;
        int64_t best_score = INT64_MIN;

        for (size_t rpos = 0; rpos < ready.size(); ++rpos) {
            const uint32_t idx = ready[rpos];
            // Net register pressure change if issued next
            int32_t net_delta = 0;
            for (Reg ur : inst_uses[idx]) {
                const int s = reg_slot(ur);
                if (s >= 0 && reg_live[s] && remaining_uses_count[s] == 1) net_delta--;
            }
            for (Reg dr : inst_defs[idx]) {
                const int s = reg_slot(dr);
                if (s >= 0 && !reg_live[s]) net_delta++;
            }

            int64_t score = 0;
            const bool is_mma = (region[idx].op == Opcode::mma_sync);
            const bool is_ld = (region[idx].op == Opcode::ld || region[idx].op == Opcode::ldmatrix);
            if (eff_mode == PtxSchedMode::MinRegPressure) {
                score = static_cast<int64_t>(-net_delta) * 10000LL + static_cast<int64_t>(height[idx]) * 10LL;
            } else if (eff_mode == PtxSchedMode::MmaInterleave) {
                const int64_t alt_bonus = (last_was_mma != is_mma) ? 5000LL : 0LL;
                const int64_t ld_bonus = is_ld ? 1200LL : 0LL;
                const int64_t press_pen = (cur_live > opts.reg_pressure_limit) ? (static_cast<int64_t>(net_delta) * 3000LL) : 0LL;
                score = alt_bonus + ld_bonus + static_cast<int64_t>(height[idx]) * 20LL - press_pen;
            } else {
                // LatencyHide
                const int64_t ld_bonus = is_ld ? 1000LL : 0LL;
                score = static_cast<int64_t>(height[idx]) * 100LL + ld_bonus - static_cast<int64_t>(net_delta) * 5LL;
            }
            // Deterministic tie-break preserving original order
            score = score * 1024LL - static_cast<int64_t>(idx);
            if (score > best_score) {
                best_score = score;
                best_pos = rpos;
            }
        }

        const uint32_t chosen = ready[best_pos];
        ready.erase(ready.begin() + static_cast<std::ptrdiff_t>(best_pos));
        scheduled[chosen] = 1;
        order.push_back(chosen);
        last_was_mma = (region[chosen].op == Opcode::mma_sync);

        for (Reg ur : inst_uses[chosen]) {
            const int s = reg_slot(ur);
            if (s >= 0 && remaining_uses_count[s] > 0) {
                remaining_uses_count[s]--;
                if (reg_live[s] && remaining_uses_count[s] == 0) {
                    reg_live[s] = 0;
                    if (cur_live > 0) cur_live--;
                }
            }
        }
        for (Reg dr : inst_defs[chosen]) {
            const int s = reg_slot(dr);
            if (s >= 0 && !reg_live[s]) {
                reg_live[s] = 1;
                cur_live++;
            }
        }
        for (uint32_t v : succ[chosen]) {
            if (!scheduled[v] && --indeg[v] == 0) {
                ready.push_back(v);
            }
        }
    }

    if (order.size() != n) return 0; // Defensive safety bailout
    uint32_t moved = 0;
    for (size_t i = 0; i < n; ++i) {
        if (order[i] != i) moved++;
    }
    if (moved == 0) return 0;

    std::vector<Inst> reordered;
    reordered.reserve(n);
    for (uint32_t idx : order) {
        reordered.push_back(std::move(region[idx]));
    }
    region = std::move(reordered);
    return moved;
}

} // namespace

uint32_t estimate_peak_live_regs(
    const Function& fn,
    std::array<uint32_t, kRegClassCount>* per_class_peak) noexcept {
    std::array<uint32_t, kRegClassCount> sz = fn.reg_counts;
    std::vector<Reg> defs, uses;
    for (const auto& blk : fn.blocks) {
        for (const auto& inst : blk->insts) {
            collect_defs_uses(inst, defs, uses);
            for (Reg r : defs) {
                const size_t c = static_cast<size_t>(r.cls);
                if (sz[c] <= r.index) sz[c] = r.index + 1;
            }
            for (Reg r : uses) {
                const size_t c = static_cast<size_t>(r.cls);
                if (sz[c] <= r.index) sz[c] = r.index + 1;
            }
        }
    }

    std::array<uint32_t, kRegClassCount> max_class{};
    uint32_t max_total = 0;
    std::array<std::vector<uint8_t>, kRegClassCount> live;
    for (size_t c = 0; c < kRegClassCount; ++c) live[c].assign(sz[c], 0);

    for (const auto& blk : fn.blocks) {
        for (size_t c = 0; c < kRegClassCount; ++c) std::fill(live[c].begin(), live[c].end(), uint8_t{0});
        std::array<uint32_t, kRegClassCount> cur_class{};
        uint32_t cur_total = 0;

        for (size_t ii = blk->insts.size(); ii-- > 0;) {
            collect_defs_uses(blk->insts[ii], defs, uses);
            for (Reg dr : defs) {
                const size_t c = static_cast<size_t>(dr.cls);
                if (live[c][dr.index]) {
                    live[c][dr.index] = 0;
                    cur_class[c]--;
                    cur_total--;
                }
            }
            for (Reg ur : uses) {
                const size_t c = static_cast<size_t>(ur.cls);
                if (!live[c][ur.index]) {
                    live[c][ur.index] = 1;
                    cur_class[c]++;
                    cur_total++;
                }
            }
            max_total = (std::max)(max_total, cur_total);
            for (size_t c = 0; c < kRegClassCount; ++c) {
                max_class[c] = (std::max)(max_class[c], cur_class[c]);
            }
        }
    }

    if (per_class_peak) *per_class_peak = max_class;
    return max_total;
}

PtxScheduleStats schedule_and_remat(Function& fn, const PtxScheduleOptions& opts) {
    PtxScheduleStats stats;
    stats.peak_live_before = estimate_peak_live_regs(fn);

    bool has_mma = false;
    for (const auto& blk : fn.blocks) {
        for (const auto& inst : blk->insts) {
            if (inst.op == Opcode::mma_sync) { has_mma = true; break; }
        }
        if (has_mma) break;
    }

    if (opts.enable_remat) {
        const GlobalRegInfo ginfo(fn);
        for (auto& blk : fn.blocks) {
            stats.remats_applied += remat_sink_block(*blk, ginfo);
        }
    }

    if (opts.mode != PtxSchedMode::None) {
        for (auto& blk : fn.blocks) {
            std::vector<Inst> new_insts;
            new_insts.reserve(blk->insts.size());
            std::vector<Inst> region;

            auto flush_region = [&]() {
                if (region.empty()) return;
                if (region.size() >= 2 && region.size() <= 512) {
                    stats.regions_scheduled++;
                    stats.insts_reordered += schedule_region(region, opts, stats.peak_live_before, has_mma);
                }
                for (auto& ri : region) new_insts.push_back(std::move(ri));
                region.clear();
            };

            for (auto& inst : blk->insts) {
                if (inst_is_hard_fence(inst)) {
                    flush_region();
                    new_insts.push_back(std::move(inst));
                } else {
                    region.push_back(std::move(inst));
                    if (region.size() >= 512) flush_region();
                }
            }
            flush_region();
            blk->insts = std::move(new_insts);
        }
    }

    stats.peak_live_after = estimate_peak_live_regs(fn);
    return stats;
}

} // namespace brass::ptx
