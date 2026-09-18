#pragma once

#include <brass/mir/module.hpp>
#include <brass/core/diagnostics.hpp>
#include <brass/il_translator/il_ast.hpp>
#include <memory>
#include <string_view>
#include <string>
#include <unordered_map>

namespace brass {

class HostGC;

struct DemoteStats;
struct PartialEscapeStats;
struct GvnPreStats;
struct LoopOptStats;
struct FmaOptStats;
struct ParallelLoopStats;
struct RangeAnalysisStats;

namespace il {

struct FunctionMeta {
    bool needs_env = false;
    bool needs_this = false;
    bool needs_arguments = false;
    bool has_rest_param = false;
    bool is_strict = false;
    uint32_t first_source_param = 0;
    uint32_t fn_flags = 0x03; // BRONZE_ABI_FN_FLAGS_ORDINARY (CONSTRUCT | PROTOTYPE)
    uint32_t name_key = 0xFFFFFFFFu; // BRONZE_ABI_FN_NAME_NONE
    uint32_t required_args = 0;
    uint32_t adapt_arity = 0;
    std::vector<bool> params_pinned;
    std::vector<uint32_t> param_pin_keys;
};

struct TranslatorOptions {
    bool enable_optimizations = true;
    // Re-verify the whole module after EVERY optimization pass, naming the
    // pass that broke it. A pass-author's tool: it multiplies the verifier's
    // cost by the number of passes on every compile, which is a large share of
    // an in-memory JIT's turnaround. The module is always verified once
    // before the passes and once after them; this is only the per-pass
    // bisection in between. Defaults on in debug builds, off in release.
#if defined(NDEBUG)
    bool verify_after_each_pass = false;
#else
    bool verify_after_each_pass = true;
#endif
    bool allow_fp_reassociation = false;
    bool trace_lowering = false;
    bool enable_f64_demote = true;
    bool demote_stats = false;
    bool enable_inlining = false;
    bool enable_sroa = false;
    bool enable_gvn = true;
    bool enable_sccp = true;
    bool enable_guard_elim = true;
    bool enable_cfg_simplify = true;
    bool enable_loop_unswitch = true;
    bool enable_jump_threading = true;
    bool enable_trace_layout = true;
    bool run_escape_analysis = false;
    bool enable_partial_escape = false;
    bool enable_allocation_sinking = false;
    bool dump_pea_stats = false;
    bool run_alias_analysis = false;
    bool enable_vectorize = true;
    bool enable_slp = true;
    bool enable_loop_tile = true;
    size_t tile_size = 16;
    bool enable_pic = true;
    bool enable_inlined_fastpaths = true;
    bool enable_tlab = false;
    bool use_bronze_tlab = false;
    bool dump_ic_stats = false;
    bool enable_wbe = true;
    bool dump_wbe_stats = false;
    bool enable_gvn_pre = true;
    bool dump_pre_stats = false;
    bool enable_osr = false;
    uint64_t osr_threshold = 100;
    bool dump_tiering_stats = false;
    bool enable_loop_fusion = false;
    bool enable_loop_distribution = false;
    bool enable_array_contraction = false;
    bool dump_loop_transform_stats = false;
    DemoteStats* demote_stats_collector = nullptr;
    PartialEscapeStats* pea_stats_collector = nullptr;
    GvnPreStats* pre_stats_collector = nullptr;
    LoopOptStats* loop_transform_stats_collector = nullptr;
    bool enable_avx2 = false;
    bool enable_fma = false;
    uint32_t vector_width = 0;
    bool dump_fma_stats = false;
    FmaOptStats* fma_stats_collector = nullptr;
    bool enable_parallel_loops = false;
    uint64_t parallel_threshold = 1000;
    uint32_t parallel_workers = 0;
    bool dump_parallel_stats = false;
    ParallelLoopStats* parallel_stats_collector = nullptr;
    bool enable_bce = true;
    bool dump_range_stats = false;
    RangeAnalysisStats* range_stats_collector = nullptr;
    bool enable_speculative_inlining = false;
    bool dump_tfv_stats = false;
    std::vector<std::string> key_constants;
    std::unordered_map<std::string, FunctionMeta> function_meta;
    std::string entry_symbol;
    bool propagate_exceptions_in_entry = false;
    // Keep the runtime's thread-local block in a pinned callee-saved register
    // (Module::set_pinned_tls_register): the module entry fetches it once
    // through `bronze_tls_enter`, and every exception check, allocation fast
    // path and stack-limit check reads through the register instead of
    // calling `bronze_tls_block_addr`. Requires the runtime to enter compiled
    // code only through its trampoline (which sets the register).
    bool pin_tls_register = false;
    bool enable_census = false;
    uint32_t census_site_count = 0;
    struct SourceFileMeta {
        uint32_t text_len = 0;
        uint32_t entry_count = 0;
    };
    std::vector<SourceFileMeta> source_files;
};

struct TranslationResult {
    bool success = false;
    std::unique_ptr<Module> module;
    std::string error_message;
};

// Translate Bronze in-memory AST directly to a Brass MIR Module
TranslationResult translate_bronze_ast(
    const BronzeModuleAST& ast,
    const TranslatorOptions& options = {},
    DiagnosticReporter* diag = nullptr
);

// Translate Bronze textual IL directly to a Brass MIR Module
TranslationResult translate_bronze_il(
    std::string_view il_text,
    const TranslatorOptions& options = {},
    DiagnosticReporter* diag = nullptr
);

// Register Bronze runtime helper symbols into a JitExecutionEngine or runtime symbol table
void register_bronze_runtime_symbols(void* jit_engine_ptr);
void register_bronze_baseline_symbols(void* baseline_jit_ptr);
void register_bronze_interpreter_symbols(void* interp_ptr);

// Optional custom function resolver for AOT or dynamic function lookup
void set_bronze_function_resolver(void* (*resolver)(const char*));

// Control whether Bronze print statements output to stdout
void bronze_set_print_enabled(bool enabled);


} // namespace il
} // namespace brass
