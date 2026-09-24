#include <brass/fuzz/fuzz_pipeline.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/pass_catalog.hpp>

namespace brass::fuzz {

std::string_view pipeline_name(FuzzPipeline pipeline) noexcept {
    switch (pipeline) {
        case FuzzPipeline::AllPasses: return "all";
        case FuzzPipeline::Production: return "production";
        case FuzzPipeline::Legacy: return "legacy";
    }
    return "unknown";
}

bool parse_pipeline(std::string_view text, FuzzPipeline& out) noexcept {
    if (text == "all") { out = FuzzPipeline::AllPasses; return true; }
    if (text == "production") { out = FuzzPipeline::Production; return true; }
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

// Every loop transform on, vectorizing for what the host runs.
LoopOptOptions all_loop_options() {
    LoopOptOptions o;
    o.enable_licm = true;
    o.enable_ivsr = true;
    o.enable_dce = true;
    o.enable_diamond_select = true;
    o.enable_unroll = true;
    o.enable_f64_demote = true;
    o.enable_slp = true;
    o.enable_vectorize = true;
    o.enable_avx2 = host_has_avx2();
    o.vector_width = o.enable_avx2 ? 256u : 0u;
    // FMA contraction is off in every differential pipeline: it changes FP
    // results by design (one rounding instead of two), and the reference
    // interpreter evaluates the uncontracted program.
    o.enable_fma = false;
    o.enable_sroa = false;  // the module stage runs it
    o.enable_partial_escape = true;
    o.enable_allocation_sinking = true;
    o.enable_loop_tile = true;
    o.enable_loop_fusion = true;
    o.enable_loop_distribution = true;
    o.enable_array_contraction = true;
    o.enable_parallel_loops = true;
    o.enable_bce = true;
    return o;
}

Pipeline all_passes_pipeline() {
    const LoopOptOptions loops = all_loop_options();
    InlinerOptions inliner;
    Pipeline p;
    p.add(passes::devirtualize());
    p.add(passes::speculative_devirtualization());
    p.add(passes::inline_calls(inliner));
    p.add(passes::sroa());
    p.add(passes::gvn());
    p.add(passes::gvn_pre());
    p.add(passes::sccp(true));
    p.add(passes::cfg_simplify());
    p.add(passes::loop_unswitch(loops, false));
    p.add(passes::jump_threading(loops, false));
    p.add(passes::cfg_simplify("cfg_simplify 2"));
    p.add(passes::bce("bce", false, nullptr, false));
    p.append(loop_pipeline(loops));
    p.add(passes::split_critical_edges());
    // Production runs WBE last, after everything that could move stores.
    p.add(passes::write_barrier_elim());
    return p;
}

Pipeline production_pipeline() {
    PassPipelineOptions options = production_pass_pipeline_options();
    options.loop.enable_fma = false;  // see all_loop_options
    return pass_pipeline(options);
}

// The fuzzer's original sequence: GVN-PRE, WBE, then optimize_module's
// function pipeline with every transform it offers.
Pipeline legacy_pipeline() {
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
    o.enable_fma = false;  // see all_loop_options
    o.enable_partial_escape = true;
    o.enable_allocation_sinking = true;
    o.enable_loop_fusion = true;
    o.enable_loop_distribution = true;
    o.enable_array_contraction = true;
    Pipeline p;
    p.add(passes::gvn_pre());
    p.add(passes::write_barrier_elim());
    p.append(function_pipeline(o));
    return p;
}

} // namespace

Pipeline fuzz_pipeline(FuzzPipeline pipeline) {
    switch (pipeline) {
        case FuzzPipeline::AllPasses: return all_passes_pipeline();
        case FuzzPipeline::Production: return production_pipeline();
        case FuzzPipeline::Legacy: return legacy_pipeline();
    }
    return {};
}

std::vector<std::string> unknown_skip_names(FuzzPipeline pipeline, const std::vector<std::string>& skip) {
    const Pipeline p = fuzz_pipeline(pipeline);
    std::vector<std::string> unknown;
    for (const std::string& name : skip) {
        if (p.without({name}).steps().size() == p.steps().size()) unknown.push_back(name);
    }
    return unknown;
}

bool run_fuzz_pipeline(Module& mod, FuzzPipeline pipeline, const PassPipelineHooks& hooks,
                       const std::vector<std::string>& skip) {
    return run_pipeline(mod, fuzz_pipeline(pipeline).without(skip), hooks).completed;
}

} // namespace brass::fuzz
