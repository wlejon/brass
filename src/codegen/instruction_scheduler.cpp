#include <brass/codegen/instruction_scheduler.hpp>
#include <brass/codegen/lir_flags.hpp>
#include <algorithm>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace brass::codegen {

namespace {

struct RegUsage {
    uint32_t defs_gpr = 0;
    uint32_t defs_xmm = 0;
    std::vector<uint32_t> uses_gpr;
    std::vector<uint32_t> uses_xmm;
};

static bool is_block_pre_ra(const LirBlock& block) {
    for (const auto& inst : block.instructions) {
        for (const auto& d : inst->defs) {
            if (d.is_vreg()) return true;
            if (d.is_mem() && (d.mem_val.base_vreg.is_valid() || d.mem_val.index_vreg.is_valid())) {
                return true;
            }
        }
        for (const auto& u : inst->uses) {
            if (u.is_vreg()) return true;
            if (u.is_mem() && (u.mem_val.base_vreg.is_valid() || u.mem_val.index_vreg.is_valid())) {
                return true;
            }
        }
    }
    return false;
}

// Fills `usage` in place, so a reused one keeps its lists' storage.
static void get_instruction_reg_usage(const LirInst& inst, bool is_pre_ra, RegUsage& usage) {
    usage.defs_gpr = 0;
    usage.defs_xmm = 0;
    usage.uses_gpr.clear();
    usage.uses_xmm.clear();

    if (is_pre_ra) {
        for (const auto& d : inst.defs) {
            if (d.is_vreg()) {
                if (d.vreg_val.is_xmm()) usage.defs_xmm++;
                else usage.defs_gpr++;
            }
            if (d.is_mem()) {
                if (d.mem_val.base_vreg.is_valid()) {
                    if (d.mem_val.base_vreg.is_xmm()) usage.uses_xmm.push_back(d.mem_val.base_vreg.id);
                    else usage.uses_gpr.push_back(d.mem_val.base_vreg.id);
                }
                if (d.mem_val.index_vreg.is_valid()) {
                    if (d.mem_val.index_vreg.is_xmm()) usage.uses_xmm.push_back(d.mem_val.index_vreg.id);
                    else usage.uses_gpr.push_back(d.mem_val.index_vreg.id);
                }
            }
        }
        for (const auto& u : inst.uses) {
            if (u.is_vreg()) {
                if (u.vreg_val.is_xmm()) usage.uses_xmm.push_back(u.vreg_val.id);
                else usage.uses_gpr.push_back(u.vreg_val.id);
            }
            if (u.is_mem()) {
                if (u.mem_val.base_vreg.is_valid()) {
                    if (u.mem_val.base_vreg.is_xmm()) usage.uses_xmm.push_back(u.mem_val.base_vreg.id);
                    else usage.uses_gpr.push_back(u.mem_val.base_vreg.id);
                }
                if (u.mem_val.index_vreg.is_valid()) {
                    if (u.mem_val.index_vreg.is_xmm()) usage.uses_xmm.push_back(u.mem_val.index_vreg.id);
                    else usage.uses_gpr.push_back(u.mem_val.index_vreg.id);
                }
            }
        }
    } else {
        for (const auto& d : inst.defs) {
            if (d.is_preg() && d.preg_val.is_valid()) {
                if (d.preg_val.is_xmm()) usage.defs_xmm++;
                else usage.defs_gpr++;
            }
            if (d.is_mem()) {
                if (d.mem_val.base_preg.is_valid()) {
                    if (d.mem_val.base_preg.is_xmm()) usage.uses_xmm.push_back(d.mem_val.base_preg.code);
                    else usage.uses_gpr.push_back(d.mem_val.base_preg.code);
                }
                if (d.mem_val.index_preg.is_valid()) {
                    if (d.mem_val.index_preg.is_xmm()) usage.uses_xmm.push_back(d.mem_val.index_preg.code);
                    else usage.uses_gpr.push_back(d.mem_val.index_preg.code);
                }
            }
        }
        for (uint8_t c = 0; c < 32; ++c) {
            if (inst.clobbered_gprs & (1u << c)) usage.defs_gpr++;
            if (inst.clobbered_xmms & (1u << c)) usage.defs_xmm++;
        }
        for (const auto& u : inst.uses) {
            if (u.is_preg() && u.preg_val.is_valid()) {
                if (u.preg_val.is_xmm()) usage.uses_xmm.push_back(u.preg_val.code);
                else usage.uses_gpr.push_back(u.preg_val.code);
            }
            if (u.is_mem()) {
                if (u.mem_val.base_preg.is_valid()) {
                    if (u.mem_val.base_preg.is_xmm()) usage.uses_xmm.push_back(u.mem_val.base_preg.code);
                    else usage.uses_gpr.push_back(u.mem_val.base_preg.code);
                }
                if (u.mem_val.index_preg.is_valid()) {
                    if (u.mem_val.index_preg.is_xmm()) usage.uses_xmm.push_back(u.mem_val.index_preg.code);
                    else usage.uses_gpr.push_back(u.mem_val.index_preg.code);
                }
            }
        }
    }
}

struct Score {
    bool is_term;
    int32_t delta;
    uint32_t height;
    bool is_load;
    uint32_t id;
};

// One function's blocks are scheduled with the same DAG and working arrays,
// so a block reuses the storage of the blocks before it.
struct BlockScheduler {
    Arch arch;
    std::optional<SchedDAG> dag_slot;
    std::unordered_map<uint64_t, uint32_t> dense;
    std::vector<uint32_t> remaining_uses;
    std::vector<RegUsage> node_usage;
    std::vector<uint32_t> ready_queue;
    std::vector<uint32_t> ready_cycle;
    std::vector<uint8_t> in_ready_queue;
    std::vector<uint32_t> scheduled_order;
    std::vector<uint32_t> ready_now;
    std::vector<Score> scores;
    std::vector<std::unique_ptr<LirInst>> reordered;

    SchedStats run(LirBlock& block, const SchedOptions& opts);
};

SchedStats BlockScheduler::run(LirBlock& block, const SchedOptions& opts) {
    SchedStats stats;
    size_t inst_count = block.instructions.size();
    if (inst_count <= 1 || inst_count > opts.max_block_instructions) {
        return stats;
    }

    bool is_pre_ra = is_block_pre_ra(block);
    if (is_pre_ra && !opts.enable_pre_ra) return stats;
    if (!is_pre_ra && !opts.enable_post_ra) return stats;

    if (dag_slot) dag_slot->rebind(block, is_pre_ra);
    else dag_slot.emplace(block, arch, is_pre_ra);
    SchedDAG& dag = *dag_slot;
    dag.build();

    size_t n = dag.size();
    if (n != inst_count) return stats;

    // Each register the block reads becomes a dense index (the usage lists
    // hold those), with how many reads of it are still to be scheduled.
    dense.clear();
    remaining_uses.clear();
    if (node_usage.size() < n) node_usage.resize(n);
    auto to_dense = [&](std::vector<uint32_t>& regs, uint64_t reg_class) {
        for (uint32_t& reg : regs) {
            auto [it, fresh] = dense.try_emplace((reg_class << 32) | reg, static_cast<uint32_t>(remaining_uses.size()));
            if (fresh) remaining_uses.push_back(0);
            reg = it->second;
            remaining_uses[reg]++;
        }
    };

    for (size_t i = 0; i < n; ++i) {
        get_instruction_reg_usage(*dag.node(i).inst, is_pre_ra, node_usage[i]);
        to_dense(node_usage[i].uses_gpr, 0);
        to_dense(node_usage[i].uses_xmm, 1);
    }

    ready_queue.clear();
    ready_cycle.assign(n, 0);
    in_ready_queue.assign(n, false);

    for (size_t i = 0; i < n; ++i) {
        if (dag.node(i).unscheduled_preds == 0) {
            ready_queue.push_back(static_cast<uint32_t>(i));
            in_ready_queue[i] = true;
        }
    }

    scheduled_order.clear();

    uint32_t current_cycle = 0;
    bool last_scheduled_was_load = false;
    size_t live_gprs = 0;
    size_t live_xmms = 0;

    while (!ready_queue.empty()) {
        // 1. Check which ready nodes are ready at current_cycle, noting the
        // earliest cycle any becomes ready in the same pass.
        ready_now.clear();
        uint32_t min_r = UINT32_MAX;
        for (uint32_t u : ready_queue) {
            if (ready_cycle[u] <= current_cycle) {
                ready_now.push_back(u);
            } else {
                min_r = std::min(min_r, ready_cycle[u]);
            }
        }

        // 2. If nothing ready at current_cycle, advance current_cycle to min ready_cycle
        if (ready_now.empty()) {
            if (min_r > current_cycle && min_r != UINT32_MAX) {
                stats.stalls_hidden += (min_r - current_cycle);
                current_cycle = min_r;
            }
            for (uint32_t u : ready_queue) {
                if (ready_cycle[u] <= current_cycle) {
                    ready_now.push_back(u);
                }
            }
        }

        // 3. Priority heuristics selection
        bool pressure_high = (live_gprs >= opts.gpr_pressure_threshold) ||
                             (live_xmms >= opts.xmm_pressure_threshold);

        // The pressure delta is read only under high pressure; counting
        // the kills of every ready node at every step otherwise was most of
        // scheduling a big block.
        scores.clear();
        for (uint32_t node_idx : ready_now) {
            const auto& sn = dag.node(node_idx);
            int32_t delta = 0;
            if (pressure_high) {
                const auto& ru = node_usage[node_idx];
                int32_t kills = 0;
                for (uint32_t reg : ru.uses_gpr) kills += remaining_uses[reg] == 1;
                for (uint32_t reg : ru.uses_xmm) kills += remaining_uses[reg] == 1;
                delta = static_cast<int32_t>(ru.defs_gpr + ru.defs_xmm) - kills;
            }
            scores.push_back({sn.inst->is_terminator(), delta, sn.height, sn.is_load, sn.id});
        }

        auto compare_nodes = [&](const Score& a, const Score& b) -> bool {
            auto [term_a, delta_a, height_a, is_load_a, id_a] = a;
            auto [term_b, delta_b, height_b, is_load_b, id_b] = b;

            // Heuristic 0: Terminators must be scheduled last
            if (term_a != term_b) {
                return !term_a; // non-terminator before terminator
            }

            // Heuristic 1: Register pressure awareness
            if (pressure_high && delta_a != delta_b) {
                return delta_a < delta_b; // lower delta frees more registers
            }

            // Heuristic 2: Critical path height (latency hiding)
            // If significant height difference (>= 4), choose higher critical path distance
            int32_t h_diff = static_cast<int32_t>(height_a) - static_cast<int32_t>(height_b);
            if (std::abs(h_diff) >= 4) {
                return height_a > height_b;
            }

            // Heuristic 3: Execution port multi-issue balancing
            if (opts.balance_issue_ports && is_load_a != is_load_b) {
                if (last_scheduled_was_load) {
                    return !is_load_a; // prefer ALU/compute after load
                } else {
                    return is_load_a;  // prefer load after ALU/compute
                }
            }

            // Heuristic 4: Critical path height tie-breaker
            if (height_a != height_b) {
                return height_a > height_b;
            }

            // Heuristic 5: Stable deterministic tie-breaker by original program order
            return id_a < id_b;
        };

        const uint32_t chosen = std::min_element(scores.begin(), scores.end(), compare_nodes)->id;

        // Check if load was hoisted
        if (dag.node(chosen).is_load && chosen > scheduled_order.size()) {
            stats.loads_hoisted++;
        }
        if (pressure_high) {
            stats.pressure_throttles++;
        }

        // Remove from ready queue
        // A node is queued once (in_ready_queue), so erasing its one entry
        // keeps the order the rest were queued in.
        ready_queue.erase(std::find(ready_queue.begin(), ready_queue.end(), chosen));
        scheduled_order.push_back(chosen);

        // Update register live counts and remaining uses
        const auto& ru = node_usage[chosen];
        for (uint32_t reg : ru.uses_gpr) {
            if (remaining_uses[reg] == 1 && live_gprs > 0) live_gprs--;
            remaining_uses[reg]--;
        }
        for (uint32_t reg : ru.uses_xmm) {
            if (remaining_uses[reg] == 1 && live_xmms > 0) live_xmms--;
            remaining_uses[reg]--;
        }
        live_gprs += ru.defs_gpr;
        live_xmms += ru.defs_xmm;

        last_scheduled_was_load = dag.node(chosen).is_load;

        // Advance cycle
        current_cycle++;

        // Update successors
        for (const auto& s : dag.node(chosen).succs) {
            ready_cycle[s.target_node] = std::max(ready_cycle[s.target_node], current_cycle + s.latency - 1);
            if (dag.node(s.target_node).unscheduled_preds > 0) {
                dag.node(s.target_node).unscheduled_preds--;
                if (dag.node(s.target_node).unscheduled_preds == 0 && !in_ready_queue[s.target_node]) {
                    ready_queue.push_back(s.target_node);
                    in_ready_queue[s.target_node] = true;
                }
            }
        }
    }

    if (scheduled_order.size() == n) {
        reordered.clear();
        for (uint32_t idx : scheduled_order) {
            reordered.push_back(std::move(block.instructions[idx]));
        }
        block.instructions.swap(reordered);
        stats.blocks_scheduled = 1;
        stats.instructions_scheduled = n;
    }

    return stats;
}

} // namespace

SchedStats schedule_block(LirBlock& block, Arch arch, const SchedOptions& opts) {
    BlockScheduler scheduler{arch};
    return scheduler.run(block, opts);
}

SchedStats schedule_block(LirBlock& block, Arch arch) {
    SchedOptions default_opts;
    return schedule_block(block, arch, default_opts);
}

SchedStats schedule_function(LirFunction& fn, const SchedOptions& opts) {
    SchedStats total_stats;
    BlockScheduler scheduler{lir_arch(fn)};
    for (auto& block : fn.blocks) {
        if (block) {
            SchedStats bs = scheduler.run(*block, opts);
            total_stats.blocks_scheduled += bs.blocks_scheduled;
            total_stats.instructions_scheduled += bs.instructions_scheduled;
            total_stats.stalls_hidden += bs.stalls_hidden;
            total_stats.loads_hoisted += bs.loads_hoisted;
            total_stats.pressure_throttles += bs.pressure_throttles;
        }
    }
    return total_stats;
}

SchedStats schedule_function(LirFunction& fn) {
    SchedOptions default_opts;
    return schedule_function(fn, default_opts);
}

} // namespace brass::codegen
