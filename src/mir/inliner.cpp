#include <brass/mir/inliner.hpp>
#include <brass/mir/inline_transform.hpp>
#include <brass/mir/devirtualize.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <algorithm>
#include <brass/pgo/profile_data.hpp>
#include <brass/mir/branch_probability.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/speculative_inliner.hpp>

namespace brass {

namespace {

size_t get_function_instruction_count(const Function& fn) {
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb) {
            count += bb->instruction_count();
        }
    }
    return count;
}

struct InlineCandidate {
    Instruction* call_inst = nullptr;
    Function* callee = nullptr;
    size_t loop_depth = 0;
    size_t callee_size = 0;
    bool is_leaf = false;
    double priority = 0.0;
};

} // namespace

bool should_inline_call(
    const Function& caller,
    const CallSite& site,
    const Function& callee,
    const CallGraph& cg,
    size_t current_depth,
    size_t loop_depth,
    size_t baseline_caller_size,
    size_t current_caller_size,
    const InlinerOptions& options
) {
    (void)site;
    if (&caller == &callee) return false;
    if (callee.blocks().empty() || !callee.entry_block()) return false;

    // Reject recursive cycles to prevent code explosion
    if (cg.is_recursive(&callee)) return false;

    size_t effective_max_depth = options.max_inline_depth;
    double threshold = static_cast<double>(options.leaf_instruction_threshold);

    // PGO Profitability Model
    if (options.enable_pgo && options.profile_data) {
        const auto* caller_prof = options.profile_data->find_function(std::string(caller.name()));
        if (caller_prof && site.instruction && site.instruction->parent()) {
            mir::BranchProbabilityAnalysis bpa(caller, *caller_prof);
            const auto& bfi = bpa.block_frequency_info();
            const BasicBlock* call_bb = site.instruction->parent();
            uint64_t call_count = bfi.get_block_count(call_bb);
            double call_freq = bfi.get_block_frequency(call_bb);

            // Cold call site: suppress inlining
            if (call_count == 0 || call_freq < 0.001) {
                return false;
            }

            // Hot call site: grant significant bonus to inline depth and instruction thresholds
            if (call_count >= 100 || call_freq >= 0.10) {
                effective_max_depth += 2;
                threshold *= 3.0;
            }
        }
    }

    // Depth limit
    if (current_depth >= effective_max_depth) return false;

    // Callee size check
    size_t callee_size = get_function_instruction_count(callee);
    if (callee_size > options.max_callee_instruction_count) return false;

    // Projected caller size
    size_t projected_caller_size = current_caller_size + (callee_size > 0 ? callee_size - 1 : 0);
    if (projected_caller_size > options.max_total_caller_instructions) return false;

    // Caller growth limit
    if (current_caller_size > 40) {
        double growth = static_cast<double>(projected_caller_size) / static_cast<double>(std::max<size_t>(1, baseline_caller_size));
        if (growth > options.max_caller_growth_factor) return false;
    }

    // Profitability model
    // threshold initialized above and modified by PGO if active

    // Loop call priority bonus
    if (options.enable_loop_priority && loop_depth >= 1) {
        double bonus = options.loop_call_priority_bonus;
        if (loop_depth > 1) {
            bonus += static_cast<double>(loop_depth - 1) * 2.0;
        }
        threshold *= bonus;
    }

    // Leaf function bonus (no outgoing calls)
    if (cg.is_leaf(&callee)) {
        threshold *= 1.25;
    }

    // Single-block leaf bonus
    if (callee.blocks().size() == 1) {
        threshold = std::max(threshold, 40.0);
    }

    return static_cast<double>(callee_size) <= threshold;
}

bool inline_function(Function& fn, Module& mod) {
    InlinerOptions opts;
    return inline_function(fn, mod, opts);
}

bool inline_function(Function& fn, Module& mod, const InlinerOptions& options, const CallGraph* external_cg) {
    if (fn.blocks().empty() || !fn.entry_block()) return false;

    if (options.enable_devirtualization) {
        devirtualize_function(fn, mod);
    }

    bool any_changed = false;
    size_t baseline_size = get_function_instruction_count(fn);
    size_t current_size = baseline_size;

    std::unique_ptr<CallGraph> local_cg;
    const CallGraph& cg = external_cg ? *external_cg : *(local_cg = std::make_unique<CallGraph>(mod));

    for (size_t depth = 0; depth < options.max_inline_depth; ++depth) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopAnalysis loops(fn, dom);

        std::vector<LoopInfo*> post_loops = loops.post_order_loops();

        // Discover candidate call sites
        std::vector<InlineCandidate> candidates;
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                if (!inst || inst->opcode() != Opcode::call) continue;

                Function* callee = mod.get_function(inst->symbol());
                if (!callee || callee == &fn) continue;
                if (callee->blocks().empty() || !callee->entry_block()) continue;

                // Find loop depth
                size_t l_depth = 0;
                for (LoopInfo* loop : post_loops) {
                    if (loop && loop->contains(inst)) {
                        l_depth = std::max(l_depth, loop->depth());
                    }
                }

                size_t c_size = get_function_instruction_count(*callee);
                bool is_leaf = cg.is_leaf(callee);

                double priority = static_cast<double>(l_depth) * 1000.0 +
                                  (1000.0 / (static_cast<double>(c_size) + 1.0)) +
                                  (is_leaf ? 500.0 : 0.0);

                InlineCandidate cand;
                cand.call_inst = inst;
                cand.callee = callee;
                cand.loop_depth = l_depth;
                cand.callee_size = c_size;
                cand.is_leaf = is_leaf;
                cand.priority = priority;
                candidates.push_back(cand);
            }
        }

        if (candidates.empty()) break;

        // Sort by priority descending
        std::sort(candidates.begin(), candidates.end(), [](const InlineCandidate& a, const InlineCandidate& b) {
            return a.priority > b.priority;
        });

        bool inlined_any_in_round = false;
        for (const auto& cand : candidates) {
            // Verify call instruction is still in a valid block of fn
            BasicBlock* p = cand.call_inst->parent();
            if (!p || p->parent() != &fn) continue;

            CallSite dummy_site;
            dummy_site.instruction = cand.call_inst;
            dummy_site.caller = &fn;
            dummy_site.callee = cand.callee;

            if (!should_inline_call(fn, dummy_site, *cand.callee, cg, depth, cand.loop_depth, baseline_size, current_size, options)) {
                continue;
            }

            InlineResult res = inline_call_site(fn, cand.call_inst, *cand.callee);
            if (res.success) {
                inlined_any_in_round = true;
                any_changed = true;
                current_size = get_function_instruction_count(fn);
                // Re-evaluate CFG after each inline to avoid stale blocks
                break;
            }
        }

        if (!inlined_any_in_round) break;
    }

    if (any_changed) {
        fn.rebuild_cfg_predecessors();
    }
    return any_changed;
}

bool inline_module(Module& mod) {
    InlinerOptions opts;
    return inline_module(mod, opts);
}

bool inline_module(Module& mod, const InlinerOptions& options) {
    if (options.enable_devirtualization) {
        devirtualize_module(mod);
    }

    CallGraph cg(mod);
    const std::vector<Function*>& order = cg.bottom_up_order();

    bool changed = false;
    for (Function* fn : order) {
        if (!fn) continue;
        // Don't inline into a function that is strictly recursive with itself if forbidden
        changed |= inline_function(*fn, mod, options, &cg);
    }

    return changed;
}

bool optimize_module_ipo(Module& mod) {
    InlinerOptions inline_opts;
    LoopOptOptions loop_opts;
    return optimize_module_ipo(mod, inline_opts, loop_opts);
}

bool optimize_module_ipo(Module& mod, const InlinerOptions& inline_opts, const LoopOptOptions& loop_opts) {
    bool changed = false;

    auto check_ipo = [&](const char* step) {
        if (!verify_module(mod)) {
            std::fprintf(stderr, "[FATAL] Broken in IPO during: %s\n", step);
            return false;
        }
        return true;
    };

    // 1. Devirtualize monomorphic patchable_calls
    if (inline_opts.enable_devirtualization) {
        changed |= devirtualize_module(mod);
        if (!check_ipo("devirtualize_module")) return false;
    }

    // 1b. Speculative Devirtualization & Inlining via Type Feedback Vector (TFV)
    if (inline_opts.enable_speculative_devirtualization) {
        SpeculativeInlinerOptions spec_opts;
        spec_opts.enable_inlining = true;
        spec_opts.enable_polymorphic = true;
        spec_opts.max_callee_instruction_count = inline_opts.max_callee_instruction_count;
        changed |= run_speculative_devirtualization(mod, spec_opts);
        if (!check_ipo("run_speculative_devirtualization")) return false;
    }

    // 2. Inlining in bottom-up leaf-first order
    changed |= inline_module(mod, inline_opts);
    if (!check_ipo("inline_module")) return false;

    // 2b. Escape Analysis & SROA pass
    if (inline_opts.enable_sroa) {
        SroaOptions sroa_opts;
        changed |= sroa_module(mod, sroa_opts);
        if (!check_ipo("sroa_module")) return false;
    }

    // 2c. GVN (CSE + RLE + DSE) pass
    if (inline_opts.enable_gvn) {
        GvnOptions gvn_opts;
        changed |= gvn_module(mod, gvn_opts);
        if (!check_ipo("gvn_module")) return false;
    }

    // 2d. SCCP & Guard Elim & CFG Simplify pass
    if (loop_opts.enable_sccp) {
        SccpOptions sccp_opts;
        sccp_opts.enable_guard_elim = loop_opts.enable_guard_elim;
        changed |= sccp_module(mod, sccp_opts);
        if (!check_ipo("sccp_module")) return false;
    }
    if (loop_opts.enable_cfg_simplify) {
        CfgSimplifyOptions cfg_opts;
        changed |= cfg_simplify_module(mod, cfg_opts);
        if (!check_ipo("cfg_simplify_module")) return false;
    }

    // 2e. Range Analysis & BCE pass
    if (loop_opts.enable_bce) {
        RangeAnalysisOptions bce_opts;
        bce_opts.enable_bce = true;
        bce_opts.enable_hoisting = true;
        bce_opts.dump_stats = loop_opts.dump_range_stats;
        if (loop_opts.range_stats) {
            bce_opts.stats = loop_opts.range_stats;
        } else if (loop_opts.stats) {
            bce_opts.stats = &loop_opts.stats->bce_stats;
        }
        changed |= run_bounds_check_elimination(mod, bce_opts);
        if (!check_ipo("run_bounds_check_elimination")) return false;
    }

    // 3. Re-run loop optimizations, constant folding, CSE, DCE & f64 demotion
    changed |= optimize_module_loops(mod, loop_opts);
    if (!check_ipo("optimize_module_loops")) return false;

    return changed;
}

} // namespace brass
