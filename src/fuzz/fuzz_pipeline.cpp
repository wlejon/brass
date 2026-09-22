#include <brass/fuzz/fuzz_pipeline.hpp>
#include <brass/il_translator/il_pipeline.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/inliner.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/gvn_pre.hpp>
#include <brass/mir/sccp.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/critical_edge.hpp>
#include <brass/mir/f64_demote.hpp>
#include <brass/mir/select_opt.hpp>
#include <brass/mir/allocation_sinking.hpp>
#include <brass/mir/loop_tile.hpp>
#include <brass/mir/loop_distribution.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/array_contraction.hpp>
#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/slp_vectorize.hpp>
#include <brass/mir/loop_vectorize.hpp>
#include <brass/mir/fma_opt.hpp>
#include <brass/mir/loop_unroll.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <brass/mir/dominators.hpp>
#include <functional>

namespace brass::fuzz {

std::string_view pipeline_name(FuzzPipeline pipeline) noexcept {
    switch (pipeline) {
        case FuzzPipeline::AllPasses: return "all";
        case FuzzPipeline::Bronze: return "bronze";
        case FuzzPipeline::Legacy: return "legacy";
    }
    return "unknown";
}

bool parse_pipeline(std::string_view text, FuzzPipeline& out) noexcept {
    if (text == "all") { out = FuzzPipeline::AllPasses; return true; }
    if (text == "bronze") { out = FuzzPipeline::Bronze; return true; }
    if (text == "legacy") { out = FuzzPipeline::Legacy; return true; }
    return false;
}

namespace {

bool host_has_avx2() noexcept {
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

// optimize_function_loops' own eligibility rule: loop transforms must not
// touch coroutine state machines, OSR entries or ABI wrappers.
bool loop_eligible(const Function& fn) {
    if (!fn.resume_points().empty() || fn.name().starts_with("__wrapper_")) return false;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == Opcode::osr_entry) return false;
        }
    }
    return true;
}

void for_each_loop_fn(Module& mod, const std::function<void(Function&)>& body) {
    // A snapshot: auto-parallelization adds kernel functions to the module.
    const std::vector<Function*> fns = mod.functions();
    for (Function* fn : fns) {
        if (fn && loop_eligible(*fn)) {
            fn->rebuild_cfg_predecessors();
            body(*fn);
            fn->rebuild_cfg_predecessors();
        }
    }
}

// A loop-pass step that needs a fresh dominator tree per function.
std::function<void(Module&)> with_dom(const std::function<void(Function&, const DominatorTree&)>& pass) {
    return [pass](Module& mod) {
        for_each_loop_fn(mod, [&](Function& fn) {
            DominatorTree dom(fn);
            pass(fn, dom);
        });
    };
}

bool fp_reassoc(const Function& fn) {
    return fn.allow_fp_reassociation() || (fn.parent() && fn.parent()->allow_fp_reassociation());
}

// LICM, IVSR, constant folding, CSE and DCE are internal to loop_opt.cpp;
// optimize_function_loops with every other transform off runs just them.
LoopOptOptions scalar_loop_cleanup_options() {
    LoopOptOptions o;
    o.enable_licm = true;
    o.enable_ivsr = true;
    o.enable_dce = true;
    o.enable_diamond_select = false;
    o.enable_unroll = false;
    o.enable_f64_demote = false;
    o.enable_slp = false;
    o.enable_vectorize = false;
    o.enable_sroa = false;
    o.enable_loop_tile = false;
    o.enable_loop_fusion = false;
    o.enable_loop_distribution = false;
    o.enable_array_contraction = false;
    o.enable_partial_escape = false;
    o.enable_allocation_sinking = false;
    o.enable_fma = false;
    o.enable_parallel_loops = false;
    o.enable_bce = false;
    return o;
}

struct Step {
    const char* name;
    std::function<void(Module&)> run;
};

std::vector<Step> all_passes_steps() {
    const bool avx2 = host_has_avx2();
    std::vector<Step> s;
    s.push_back({"ipo", [](Module& m) {
        InlinerOptions io;
        io.enable_sroa = true;
        io.enable_gvn = false;
        io.enable_devirtualization = true;
        io.enable_speculative_devirtualization = true;
        optimize_module_ipo(m, io, LoopOptOptions{});
    }});
    s.push_back({"sroa", [](Module& m) { sroa_module(m); }});
    s.push_back({"gvn", [](Module& m) { gvn_module(m); }});
    s.push_back({"gvn_pre", [](Module& m) { gvn_pre_module(m); }});
    s.push_back({"sccp", [](Module& m) {
        SccpOptions o;
        o.enable_guard_elim = true;
        sccp_module(m, o);
    }});
    s.push_back({"cfg_simplify", [](Module& m) { cfg_simplify_module(m); }});
    s.push_back({"loop_unswitch", [](Module& m) { unswitch_loops_in_module(m); }});
    s.push_back({"jump_threading", [](Module& m) { jump_thread_module(m); }});
    s.push_back({"cfg_simplify 2", [](Module& m) { cfg_simplify_module(m); }});
    s.push_back({"bce", [](Module& m) {
        RangeAnalysisOptions o;
        o.enable_bce = true;
        o.enable_implied_checks = true;
        run_bounds_check_elimination(m, o);
    }});
    s.push_back({"f64_demote", [](Module& m) {
        for_each_loop_fn(m, [](Function& fn) { f64_demote_pass(fn); });
    }});
    s.push_back({"select_opt", [](Module& m) {
        for_each_loop_fn(m, [](Function& fn) { simplify_cfg_diamonds(fn); });
    }});
    s.push_back({"allocation_sinking", [](Module& m) {
        for_each_loop_fn(m, [](Function& fn) { sink_allocations(fn); });
    }});
    s.push_back({"licm_ivsr_dce", [](Module& m) {
        const LoopOptOptions o = scalar_loop_cleanup_options();
        for_each_loop_fn(m, [&](Function& fn) { optimize_function_loops(fn, o); });
    }});
    s.push_back({"loop_tile", with_dom([](Function& fn, const DominatorTree& dom) {
        loop_tile_pass(fn, dom, LoopTileOptions{});
    })});
    s.push_back({"loop_distribution", with_dom([](Function& fn, const DominatorTree& dom) {
        loop_distribution_pass(fn, dom);
    })});
    s.push_back({"loop_fusion", with_dom([](Function& fn, const DominatorTree& dom) {
        loop_fusion_pass(fn, dom);
    })});
    s.push_back({"array_contraction", with_dom([](Function& fn, const DominatorTree& dom) {
        array_contraction_pass(fn, dom);
    })});
    s.push_back({"parallel_loops", with_dom([](Function& fn, const DominatorTree& dom) {
        ParallelLoopOptions o;
        o.allow_fp_reassociation = fp_reassoc(fn);
        auto_parallelize_function(fn, dom, o);
    })});
    s.push_back({"slp_vectorize", [](Module& m) {
        for_each_loop_fn(m, [](Function& fn) {
            SlpOptions o;
            o.allow_fp_reassociation = fp_reassoc(fn);
            slp_vectorize_function(fn, o);
        });
    }});
    s.push_back({"loop_vectorize", with_dom([avx2](Function& fn, const DominatorTree& dom) {
        LoopVectorizeOptions o;
        o.enable_avx2 = avx2;
        o.vector_width = avx2 ? 256u : 0u;
        o.allow_fp_reassociation = fp_reassoc(fn);
        loop_vectorize_pass(fn, dom, o);
    })});
    s.push_back({"fma", [](Module& m) {
        for_each_loop_fn(m, [](Function& fn) { fma_opt_pass(fn); });
    }});
    s.push_back({"loop_unroll", with_dom([](Function& fn, const DominatorTree& dom) {
        LoopUnrollOptions o;
        o.enable_fp_reduction_jam = fp_reassoc(fn);
        loop_unroll_pass(fn, dom, o);
    })});
    s.push_back({"licm_ivsr_dce 2", [](Module& m) {
        const LoopOptOptions o = scalar_loop_cleanup_options();
        for_each_loop_fn(m, [&](Function& fn) { optimize_function_loops(fn, o); });
    }});
    s.push_back({"select_opt 2", [](Module& m) {
        for_each_loop_fn(m, [](Function& fn) { simplify_cfg_diamonds(fn); });
    }});
    s.push_back({"split_critical_edges", [](Module& m) {
        for (Function* fn : m.functions()) {
            if (!fn) continue;
            fn->rebuild_cfg_predecessors();
            split_critical_edges(*fn);
            fn->rebuild_cfg_predecessors();
        }
    }});
    // Production runs WBE last, after everything that could move stores.
    s.push_back({"wbe", [](Module& m) {
        WriteBarrierElimination wbe;
        wbe.run_on_module(m);
    }});
    return s;
}

std::vector<Step> legacy_steps() {
    std::vector<Step> s;
    s.push_back({"gvn_pre", [](Module& m) { gvn_pre_module(m); }});
    s.push_back({"wbe", [](Module& m) {
        WriteBarrierElimination wbe;
        wbe.run_on_module(m);
    }});
    s.push_back({"optimize_module", [](Module& m) {
        LoopOptOptions o;
        o.enable_gvn = true;
        o.enable_sccp = true;
        o.enable_cfg_simplify = true;
        o.enable_licm = true;
        o.enable_ivsr = true;
        o.enable_dce = true;
        o.enable_diamond_select = true;
        o.enable_unroll = true;
        o.enable_slp = true;
        o.enable_vectorize = true;
        o.enable_avx2 = host_has_avx2();
        o.enable_fma = true;
        o.enable_partial_escape = true;
        o.enable_allocation_sinking = true;
        o.enable_loop_fusion = true;
        o.enable_loop_distribution = true;
        o.enable_array_contraction = true;
        optimize_module(m, o);
    }});
    return s;
}

// "cfg_simplify" also names its repeats ("cfg_simplify 2").
bool skipped(std::string_view step, const std::vector<std::string>& skip) {
    const std::string_view base = step.substr(0, step.find(' '));
    for (const std::string& s : skip) {
        if (s == step || s == base) return true;
    }
    return false;
}

bool run_steps(Module& mod, const std::vector<Step>& steps, const PassPipelineHooks& hooks,
               const std::vector<std::string>& skip) {
    for (const Step& step : steps) {
        if (skipped(step.name, skip)) continue;
        if (hooks.before_pass) hooks.before_pass(step.name);
        step.run(mod);
        if (hooks.after_pass && !hooks.after_pass(step.name)) return false;
    }
    return true;
}

} // namespace

bool run_fuzz_pipeline(Module& mod, FuzzPipeline pipeline, const PassPipelineHooks& hooks,
                       const std::vector<std::string>& skip) {
    switch (pipeline) {
        case FuzzPipeline::AllPasses:
            return run_steps(mod, all_passes_steps(), hooks, skip);
        case FuzzPipeline::Bronze: {
            PassPipelineOptions o = il::pass_pipeline_options(il::bronze_translator_options());
            const struct { const char* name; bool* flag; } switches[] = {
                {"sroa", &o.enable_sroa}, {"ipo", &o.enable_inlining}, {"gvn", &o.enable_gvn},
                {"gvn_pre", &o.enable_gvn_pre}, {"sccp", &o.enable_sccp},
                {"cfg_simplify", &o.enable_cfg_simplify}, {"loop_unswitch", &o.enable_loop_unswitch},
                {"jump_threading", &o.enable_jump_threading}, {"bce", &o.enable_bce}, {"wbe", &o.enable_wbe},
                // Transforms inside the "loops" stage, named as AllPasses names them.
                {"bce", &o.loop.enable_bce}, {"f64_demote", &o.loop.enable_f64_demote},
                {"select_opt", &o.loop.enable_diamond_select}, {"sroa", &o.loop.enable_sroa},
                {"allocation_sinking", &o.loop.enable_allocation_sinking},
                {"allocation_sinking", &o.loop.enable_partial_escape},
                {"loop_tile", &o.loop.enable_loop_tile}, {"loop_distribution", &o.loop.enable_loop_distribution},
                {"loop_fusion", &o.loop.enable_loop_fusion}, {"array_contraction", &o.loop.enable_array_contraction},
                {"parallel_loops", &o.loop.enable_parallel_loops}, {"slp_vectorize", &o.loop.enable_slp},
                {"loop_vectorize", &o.loop.enable_vectorize}, {"fma", &o.loop.enable_fma},
                {"loop_unroll", &o.loop.enable_unroll},
            };
            for (const auto& s : switches) {
                if (skipped(s.name, skip)) *s.flag = false;
            }
            return run_pass_pipeline(mod, o, hooks);
        }
        case FuzzPipeline::Legacy:
            return run_steps(mod, legacy_steps(), hooks, skip);
    }
    return true;
}

} // namespace brass::fuzz
