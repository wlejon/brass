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
#include <brass/mir/pass_catalog.hpp>
#include <brass/core/parallel_for.hpp>
#include <cstdio>
#include <unordered_map>

namespace brass {

namespace {

// A function's size as inlining weighs it: its blocks' instructions, cold
// ones not counted (a slow path inlined costs code, not time).
size_t get_function_instruction_count(const Function& fn) {
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb && !bb->is_cold()) {
            count += bb->instruction_count();
        }
    }
    return count;
}

// Function sizes as inlining weighs them, kept while inline_module runs: a
// function's size changes only when calls are inlined into it (forget()),
// and a callee called from thousands of sites is otherwise walked for each.
class InlineSizes {
public:
    size_t of(const Function& fn) {
        auto [it, fresh] = sizes_.try_emplace(&fn, 0);
        if (fresh) it->second = get_function_instruction_count(fn);
        return it->second;
    }
    void forget(const Function& fn) { sizes_.erase(&fn); }

private:
    std::unordered_map<const Function*, size_t> sizes_;
};

struct InlineCandidate {
    Instruction* call_inst = nullptr;
    Function* callee = nullptr;
    size_t loop_depth = 0;
    size_t callee_size = 0;
    bool is_leaf = false;
    double priority = 0.0;
};

// The calls in `fn`'s blocks that are not cold to a function of `mod` with
// a body, other than `fn`, in block order: the sites inlining considers.
std::vector<std::pair<Instruction*, Function*>> inline_sites(Function& fn, const Module& mod) {
    std::vector<std::pair<Instruction*, Function*>> sites;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || bb->is_cold()) continue;
        for (Instruction* inst : *bb) {
            if (!inst || (inst->opcode() != Opcode::call && inst->opcode() != Opcode::invoke)) continue;
            Function* callee = mod.get_function(inst->symbol());
            if (!callee || callee == &fn) continue;
            if (callee->blocks().empty() || !callee->entry_block()) continue;
            sites.emplace_back(inst, callee);
        }
    }
    return sites;
}

// The depth of the innermost loop around each of `sites` (0: none).
std::vector<size_t> site_loop_depths(Function& fn, const std::vector<std::pair<Instruction*, Function*>>& sites) {
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);
    std::vector<LoopInfo*> post_loops = loops.post_order_loops();
    std::vector<size_t> depths;
    depths.reserve(sites.size());
    for (const auto& site : sites) {
        size_t l_depth = 0;
        for (LoopInfo* loop : post_loops) {
            if (loop && loop->contains(site.first)) {
                l_depth = std::max(l_depth, loop->depth());
            }
        }
        depths.push_back(l_depth);
    }
    return depths;
}

// A function's first round of inlining as it can be weighed before any
// function is inlined into: its sites and their loop depths.
struct FirstRound {
    bool ready = false;
    std::vector<std::pair<Instruction*, Function*>> sites;
    std::vector<size_t> loop_depths;
};

} // namespace

namespace {

// What decides a call before the loop it is in: refused, taken, or taken
// when the callee's size is within `threshold` once the loop bonus is
// applied.
struct InlineGate {
    enum Verdict { No, Yes, BySize } verdict = No;
    double threshold = 0.0;
    size_t callee_size = 0;
};

InlineGate inline_gate(
    const Function& caller,
    const CallSite& site,
    const Function& callee,
    const CallGraph& cg,
    size_t current_depth,
    size_t baseline_caller_size,
    size_t current_caller_size,
    const InlinerOptions& options,
    InlineSizes& sizes
) {
    InlineGate no;
    if (&caller == &callee) return no;
    if (caller.name().starts_with("__wrapper_") || callee.name().starts_with("__wrapper_")) return no;
    if (callee.blocks().empty() || !callee.entry_block()) return no;

    // Reject recursive cycles to prevent code explosion
    if (cg.is_recursive(&callee)) return no;

    // A call its front end asks to inline: the hinted budget alone.
    if (site.instruction && site.instruction->inline_hint()) {
        const size_t size = sizes.of(callee);
        const bool yes = current_depth < options.max_inline_depth && size <= options.hinted_callee_instruction_count &&
                         current_caller_size + size <= options.max_total_caller_instructions;
        return {yes ? InlineGate::Yes : InlineGate::No, 0.0, 0};
    }

    // If restricted to leaf functions for stack trace / unwind safety, reject non-leaf callees
    if (options.only_inline_leaf_functions && !cg.is_leaf(&callee)) return no;

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
                return no;
            }

            // Hot call site: grant significant bonus to inline depth and instruction thresholds
            if (call_count >= 100 || call_freq >= 0.10) {
                effective_max_depth += 2;
                threshold *= 3.0;
            }
        }
    }

    // Depth limit
    if (current_depth >= effective_max_depth) return no;

    // Callee size check
    size_t callee_size = sizes.of(callee);
    if (callee_size > options.max_callee_instruction_count) return no;

    // Projected caller size
    size_t projected_caller_size = current_caller_size + (callee_size > 0 ? callee_size - 1 : 0);
    if (projected_caller_size > options.max_total_caller_instructions) return no;

    // Caller growth limit
    if (current_caller_size > 40) {
        double growth = static_cast<double>(projected_caller_size) / static_cast<double>(std::max<size_t>(1, baseline_caller_size));
        if (growth > options.max_caller_growth_factor) return no;
    }
    return {InlineGate::BySize, threshold, callee_size};
}

bool decide_inline(
    const Function& caller,
    const CallSite& site,
    const Function& callee,
    const CallGraph& cg,
    size_t current_depth,
    size_t loop_depth,
    size_t baseline_caller_size,
    size_t current_caller_size,
    const InlinerOptions& options,
    InlineSizes& sizes
) {
    const InlineGate gate = inline_gate(caller, site, callee, cg, current_depth, baseline_caller_size,
                                        current_caller_size, options, sizes);
    if (gate.verdict != InlineGate::BySize) return gate.verdict == InlineGate::Yes;

    // Profitability model
    double threshold = gate.threshold;

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

    return static_cast<double>(gate.callee_size) <= threshold;
}

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
    InlineSizes sizes;
    return decide_inline(caller, site, callee, cg, current_depth, loop_depth, baseline_caller_size,
                         current_caller_size, options, sizes);
}

bool inline_function(Function& fn, Module& mod) {
    InlinerOptions opts;
    return inline_function(fn, mod, opts);
}

namespace {

bool inline_into(Function& fn, Module& mod, const InlinerOptions& options, const CallGraph& cg,
                 InlineSizes& sizes, FirstRound* first = nullptr);

} // namespace

bool inline_function(Function& fn, Module& mod, const InlinerOptions& options, const CallGraph* external_cg) {
    std::unique_ptr<CallGraph> local_cg;
    const CallGraph& cg = external_cg ? *external_cg : *(local_cg = std::make_unique<CallGraph>(mod));
    InlineSizes sizes;
    return inline_into(fn, mod, options, cg, sizes);
}

namespace {

bool inline_into(Function& fn, Module& mod, const InlinerOptions& options, const CallGraph& cg,
                 InlineSizes& sizes, FirstRound* first) {
    if (fn.blocks().empty() || !fn.entry_block()) return false;

    if (options.enable_devirtualization) {
        devirtualize_function(fn, mod);
        sizes.forget(fn);
    }

    bool any_changed = false;
    size_t baseline_size = get_function_instruction_count(fn);
    size_t current_size = baseline_size;

    for (size_t depth = 0; depth < options.max_inline_depth; ++depth) {
        fn.rebuild_cfg_predecessors();

        // The call sites a callee body could be inlined at, found before
        // the dominator tree and loops that only weigh them are built: most
        // functions have none. The first round's were found by
        // inline_module, from the function as it is now.
        const bool given = depth == 0 && first && first->ready;
        const std::vector<std::pair<Instruction*, Function*>> sites =
            given ? std::move(first->sites) : inline_sites(fn, mod);
        // Nor when every site is refused whatever loop it is in: then no
        // candidate below would be taken.
        bool any_possible = false;
        for (const auto& [inst, callee] : sites) {
            CallSite site;
            site.instruction = inst;
            site.caller = &fn;
            site.callee = callee;
            if (inline_gate(fn, site, *callee, cg, depth, baseline_size, current_size, options, sizes).verdict !=
                InlineGate::No) {
                any_possible = true;
                break;
            }
        }
        if (!any_possible) break;

        const std::vector<size_t> loop_depths = given ? std::move(first->loop_depths) : site_loop_depths(fn, sites);

        // Weigh them
        std::vector<InlineCandidate> candidates;
        candidates.reserve(sites.size());
        for (size_t k = 0; k < sites.size(); ++k) {
            const auto& [inst, callee] = sites[k];
            const size_t l_depth = loop_depths[k];

            size_t c_size = sizes.of(*callee);
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

            if (!decide_inline(fn, dummy_site, *cand.callee, cg, depth, cand.loop_depth, baseline_size, current_size,
                               options, sizes)) {
                continue;
            }

            InlineResult res = inline_call_site(fn, cand.call_inst, *cand.callee);
            if (res.success) {
                inlined_any_in_round = true;
                any_changed = true;
                sizes.forget(fn);
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

} // namespace

bool inline_module(Module& mod) {
    InlinerOptions opts;
    return inline_module(mod, opts);
}

bool inline_module(Module& mod, const InlinerOptions& options) {
    if (options.enable_devirtualization) {
        devirtualize_module(mod);
    }

    CallGraph cg(mod, options.threads);
    const std::vector<Function*>& order = cg.bottom_up_order();

    // What inline_function does first to a function, for all of them at
    // once: rebuild its predecessor lists, find the calls it could inline,
    // and the loops they are in. Inlining into one function reads its
    // callees and changes only it, so a function is the same here as when
    // its turn comes, and one without calls is done: inline_function would
    // stop there. (Devirtualization rewrites calls first, so it takes the
    // serial way.)
    std::vector<uint8_t> has_sites(order.size(), 1);
    std::vector<FirstRound> firsts(order.size());
    if (!options.enable_devirtualization) {
        std::vector<size_t> weights(order.size(), 0);
        size_t total = 0;
        for (size_t i = 0; i < order.size(); ++i) {
            if (order[i]) weights[i] = order[i]->block_count();
            total += weights[i];
        }
        const size_t n = threads_for_work(options.threads, order.size(), total, 2000);
        parallel_for_each(heaviest_first(weights), n, [&](size_t i) {
            Function* fn = order[i];
            if (!fn || fn->name().starts_with("__wrapper_") || fn->blocks().empty() || !fn->entry_block()) return;
            fn->rebuild_cfg_predecessors();
            FirstRound& f = firsts[i];
            f.sites = inline_sites(*fn, mod);
            has_sites[i] = f.sites.empty() ? 0 : 1;
            if (f.sites.empty()) return;
            f.loop_depths = site_loop_depths(*fn, f.sites);
            f.ready = true;
        });
    }

    InlineSizes sizes;
    bool changed = false;
    for (size_t i = 0; i < order.size(); ++i) {
        Function* fn = order[i];
        if (!fn || !has_sites[i]) continue;
        if (fn->name().starts_with("__wrapper_")) continue;
        changed |= inline_into(*fn, mod, options, cg, sizes, &firsts[i]);
        firsts[i] = FirstRound{};
    }

    return changed;
}

bool optimize_module_ipo(Module& mod) {
    InlinerOptions inline_opts;
    LoopOptOptions loop_opts;
    return optimize_module_ipo(mod, inline_opts, loop_opts);
}

Pipeline ipo_pipeline(const InlinerOptions& options) {
    Pipeline p;
    if (options.enable_devirtualization) p.add(passes::devirtualize());
    if (options.enable_speculative_devirtualization) {
        p.add(passes::speculative_devirtualization(options.max_callee_instruction_count));
    }
    p.add(passes::inline_calls(options));
    if (options.enable_sroa) p.add(passes::sroa());
    return p;
}

bool optimize_module_ipo(Module& mod, const InlinerOptions& inline_opts, const LoopOptOptions& loop_opts) {
    (void)loop_opts;
    PassPipelineHooks hooks;
    hooks.after_pass = [&mod](std::string_view step) {
        if (verify_module(mod)) return true;
        std::fprintf(stderr, "[FATAL] Broken in IPO during: %.*s\n", static_cast<int>(step.size()), step.data());
        return false;
    };
    const PipelineResult result = run_pipeline(mod, ipo_pipeline(inline_opts), hooks);
    return result.completed && result.changed;
}

} // namespace brass
