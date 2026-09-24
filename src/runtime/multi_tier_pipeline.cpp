#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/il_translator/il_translator.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <vector>

// The baseline tiers' (x64 and aarch64) invocation hook: `feedback` is the function's
// TieringFeedback, resolved from its program's registry when the code was
// compiled, so counting takes no lock, hashes no name, and reaches the
// program's own pipeline through the feedback's registry.
extern "C" void brass_tier1_record_invocation_fb(void* feedback) {
    if (!feedback) return;
    auto& fb = *static_cast<brass::runtime::TieringFeedback*>(feedback);
    brass::runtime::MultiTierPipeline& pipeline = fb.registry().pipeline();
    if (!pipeline.is_initialized()) return;
    pipeline.on_invocation(fb);
}

namespace brass::runtime {

namespace detail {
// Code compiled for Tier 1 whose install waits for a function further up
// the compile stack (it calls that function through a lazy stub).
struct Tier1Deferred {
    std::string name;
    FunctionHandle* handle;
    std::shared_ptr<codegen::BaselineCompiledFunction> code;
    uint64_t elapsed_us;
};
} // namespace detail

namespace {
using detail::Tier1Deferred;

// Lets ~Module skip the pipeline when it was never built or is gone.
std::atomic<bool> g_pipeline_alive{false};

// The functions this thread is compiling for Tier 1, innermost last. A
// callee that is one of them (a call cycle) has no native entry yet: code
// calling it depends on that frame, and is installed with it (in its
// `group`) or not at all, so no installed code calls a function that may
// still be rejected. `depends_on` is the outermost frame this frame's code
// depends on, its own index when none.
struct Tier1CompileFrame {
    const MultiTierPipeline* pipeline;
    std::string_view name;
    size_t depends_on;
    std::vector<Tier1Deferred> group;
};
thread_local std::vector<Tier1CompileFrame> t_tier1_compiling;
} // namespace

MultiTierPipeline& MultiTierPipeline::instance() {
    static MultiTierPipeline pipeline{DefaultTag{}};
    g_pipeline_alive.store(true, std::memory_order_release);
    return pipeline;
}

MultiTierPipeline::MultiTierPipeline(DefaultTag)
    : table_(&FunctionDispatchTable::instance()), is_default_(true) {
    // Every map added here is of code that registered its own (baseline
    // functions, tier-2 modules), and threads walk while it grows.
    active_stack_maps_.set_indexed_by_code_registry(true);
    // Registered symbols are the host's; program functions are installed in
    // the dispatch table one at a time, and shadow a symbol of their name.
    baseline_compiler_.set_module_functions_shadow(true);
}

MultiTierPipeline::MultiTierPipeline(FunctionDispatchTable& table)
    : table_(&table), is_default_(false) {
    if (table.is_default()) {
        throw std::logic_error("MultiTierPipeline: the default program's pipeline is MultiTierPipeline::instance()");
    }
    active_stack_maps_.set_indexed_by_code_registry(true);
    baseline_compiler_.set_dispatch_table(&table);
    baseline_compiler_.set_module_functions_shadow(true);
}

MultiTierPipeline::~MultiTierPipeline() {
    if (is_default_) {
        g_pipeline_alive.store(false, std::memory_order_release);
        return;
    }
    release_program();
}

TieringRegistry& MultiTierPipeline::tiering() const noexcept {
    return table_->tiering();
}

BackgroundCompiler& MultiTierPipeline::background_compiler() {
    if (is_default_) return BackgroundCompiler::instance();
    std::lock_guard<std::mutex> lock(bg_mutex_);
    if (!bg_) {
        BackgroundCompilerConfig cfg;
        cfg.num_threads = std::max<size_t>(config_.jit_threads, 1);
        cfg.table = table_;
        bg_ = std::make_unique<BackgroundCompiler>(cfg);
    }
    return *bg_;
}

void MultiTierPipeline::release_program() {
    if (is_default_) {
        throw std::logic_error("MultiTierPipeline::release_program: the default program is never released");
    }
    {
        // Drops the queued compiles, then joins the workers: an in-flight
        // compile finishes installing into the program's still-live handles.
        std::lock_guard<std::mutex> lock(bg_mutex_);
        if (bg_) {
            bg_->cancel_pending();
            bg_->stop();
        }
    }
    clear_baseline_cache();
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_ = false;
    if (brass_get_active_stack_maps() == &active_stack_maps_) {
        // Hand the GC back to the default program's maps, not a dead pointer.
        const bool default_live = g_pipeline_alive.load(std::memory_order_acquire) && instance().is_initialized();
        brass_set_active_stack_maps(default_live ? &instance().active_stack_maps_ : nullptr);
    }
    if (!fast_interp_busy_.load(std::memory_order_acquire)) {
        fast_interp_.reset();
        fast_interp_module_ = nullptr;
    }
}

void MultiTierPipeline::add_stack_maps(const ModuleStackMap& maps) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& fn_map : maps.functions()) {
        active_stack_maps_.add_function(fn_map);
    }
    brass_set_active_stack_maps(&active_stack_maps_);
}

void MultiTierPipeline::forget_module(const Module* mod) noexcept {
    if (!g_pipeline_alive.load(std::memory_order_acquire)) return;
    instance().forget(mod);
}

void MultiTierPipeline::forget(const Module* mod) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (fast_interp_module_ != mod) return;
    fast_interp_module_ = nullptr;
    if (!fast_interp_busy_.load(std::memory_order_acquire)) fast_interp_.reset();
}

void MultiTierPipeline::initialize(const TieringConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    initialized_ = true;

    tiering().set_default_config(config_);

    // Setup symbol resolution on baseline compiler
    FunctionDispatchTable* table = table_;
    baseline_compiler_.set_symbol_resolver([table](std::string_view name) -> void* {
        auto* handle = table->find(name);
        if (handle && handle->native_entry()) {
            return handle->native_entry();
        }
        return nullptr;
    });

    brass_set_active_stack_maps(&active_stack_maps_);
}

void MultiTierPipeline::shutdown() {
    clear_baseline_cache();
    if (is_default_) {
        if (BackgroundCompiler::instance().is_running()) BackgroundCompiler::instance().stop();
    } else {
        std::lock_guard<std::mutex> lock(bg_mutex_);
        if (bg_) bg_->stop();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    initialized_ = false;
    if (!fast_interp_busy_.load(std::memory_order_acquire)) {
        fast_interp_.reset();
        fast_interp_module_ = nullptr;
    }
}

void MultiTierPipeline::set_config(const TieringConfig& config) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    tiering().set_default_config(config_);
}

void MultiTierPipeline::set_tier0_interpreter(Tier0Interpreter kind) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    config_.tier0_interpreter = kind;
    tiering().set_default_config(config_);
}

void MultiTierPipeline::set_use_fast_interpreter(bool enable) noexcept {
    set_tier0_interpreter(enable ? Tier0Interpreter::Fast : Tier0Interpreter::Oracle);
}

void MultiTierPipeline::register_external_symbol(std::string_view name, void* addr) {
    std::lock_guard<std::mutex> lock(mutex_);
    external_symbols_[std::string(name)] = addr;
    baseline_compiler_.register_external_symbol(name, addr);
    ++symbols_gen_;
}

void MultiTierPipeline::register_external_function(std::string_view name, FastHostFn fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    external_functions_[std::string(name)] = std::move(fn);
    ++symbols_gen_;
}

void MultiTierPipeline::register_baseline_compiled(std::shared_ptr<codegen::BaselineCompiledFunction> fn) {
    if (!fn) return;
    std::lock_guard<std::mutex> lock(mutex_);
    baseline_functions_[std::string(fn->name())] = fn;
    active_stack_maps_.add_function(fn->stack_map());
    brass_set_active_stack_maps(&active_stack_maps_);
}

std::shared_ptr<codegen::BaselineCompiledFunction> MultiTierPipeline::find_baseline_compiled(std::string_view name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = baseline_functions_.find(std::string(name));
    if (it != baseline_functions_.end()) {
        return it->second;
    }
    return nullptr;
}

void MultiTierPipeline::clear_baseline_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    baseline_functions_.clear();
    active_stack_maps_.clear();
    {
        std::lock_guard<std::mutex> c_lock(compiling_mutex_);
        in_progress_compilations_.clear();
        baseline_rejected_.clear();
    }
}

bool MultiTierPipeline::is_baseline_rejected(std::string_view fn_name) const {
    std::lock_guard<std::mutex> lock(compiling_mutex_);
    return baseline_rejected_.count(std::string(fn_name)) != 0;
}

bool MultiTierPipeline::compile_and_install_tier1(std::string_view fn_name, const Function* fn) {
    if (find_baseline_compiled(fn_name)) {
        return true;
    }

    {
        std::lock_guard<std::mutex> lock(compiling_mutex_);
        if (in_progress_compilations_.find(std::string(fn_name)) != in_progress_compilations_.end() ||
            baseline_rejected_.count(std::string(fn_name))) {
            return false;
        }
        in_progress_compilations_.emplace(std::string(fn_name));
    }

    // A function whose install is deferred to an outer frame's group stays
    // in progress until the group is installed or dropped.
    struct InProgressGuard {
        std::mutex& mtx;
        std::unordered_set<std::string>& set;
        std::string name;
        bool handed_off = false;
        ~InProgressGuard() {
            if (handed_off) return;
            std::lock_guard<std::mutex> lock(mtx);
            set.erase(name);
        }
    } guard{compiling_mutex_, in_progress_compilations_, std::string(fn_name)};

    if (tier1_compile_hook_) tier1_compile_hook_(fn_name);

    FunctionHandle* handle = table_->get_or_create(fn_name, fn);
    if (!fn) {
        fn = handle->mir_function();
    }
    if (!fn) {
        const Module* active_mod = tiering().active_module();
        if (active_mod) {
            fn = active_mod->get_function(fn_name);
        }
    }
    if (!fn) {
        return false;
    }

    handle->set_signature(fn->return_type(), fn->param_types());

    auto start = std::chrono::high_resolution_clock::now();
    codegen::BaselineCompiledFunction compiled;
    try {
        compiled = baseline_compiler_.compile(*fn, Target::host());
    } catch (const codegen::UnsupportedOperation&) {
        // The baseline tier does not compile this function (an opcode it
        // rejects, e.g. exceptions or coroutines): it stays in Tier 0.
        std::lock_guard<std::mutex> lock(compiling_mutex_);
        baseline_rejected_.emplace(fn_name);
        return false;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed_us = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    // A callee that has never run has no native entry, and the lazy stub
    // the code calls it through would trap: it is compiled now. The code is
    // not installed while a callee cannot be compiled.
    const size_t depth = t_tier1_compiling.size();
    t_tier1_compiling.push_back({this, guard.name, depth, {}});
    CalleeLink link;
    try {
        link = link_tier1_callees(*fn, compiled);
    } catch (...) {
        Tier1CompileFrame frame = std::move(t_tier1_compiling.back());
        t_tier1_compiling.pop_back();
        drop_tier1_group(frame.group);
        throw;
    }
    Tier1CompileFrame frame = std::move(t_tier1_compiling.back());
    t_tier1_compiling.pop_back();
    if (link != CalleeLink::Ready) {
        // Nothing that calls this function through a stub is installed.
        drop_tier1_group(frame.group);
        if (link == CalleeLink::Rejected) {
            std::lock_guard<std::mutex> lock(compiling_mutex_);
            baseline_rejected_.emplace(fn_name);
        }
        return false;
    }

    auto compiled_ptr = std::make_shared<codegen::BaselineCompiledFunction>(std::move(compiled));
    if (frame.depends_on < depth) {
        // Calls a function an outer frame is still compiling, which may yet
        // be rejected: installed with that frame's group, or dropped with it.
        Tier1CompileFrame& outer = t_tier1_compiling[frame.depends_on];
        for (auto& d : frame.group) outer.group.push_back(std::move(d));
        outer.group.push_back({guard.name, handle, std::move(compiled_ptr), elapsed_us});
        guard.handed_off = true;
        Tier1CompileFrame& parent = t_tier1_compiling.back();
        parent.depends_on = std::min(parent.depends_on, frame.depends_on);
        return true;
    }
    frame.group.push_back({guard.name, handle, std::move(compiled_ptr), elapsed_us});
    install_tier1_group(frame.group);
    return true;
}

void MultiTierPipeline::install_tier1_group(std::vector<detail::Tier1Deferred>& group) {
    // The group's code calls within the group through lazy stubs. Every
    // member's maps are registered and every stub filled before any member
    // becomes reachable through its handle, so no call can reach a stub of
    // a member not yet installed.
    for (auto& d : group) register_baseline_compiled(d.code);
    for (auto& d : group) baseline_compiler_.lazy_symbols()->define(d.name, d.code->entry_point());
    // The outermost (last) member is the one tiering asked for: last.
    for (auto& d : group) {
        d.handle->set_baseline_function(d.code);
        d.handle->set_native_entry(d.code->entry_point());
        d.handle->set_tier(TierLevel::Tier1_Baseline);
        tiering().get_feedback(d.name).set_tier(TierLevel::Tier1_Baseline);
        stats_.tier1_compilations.fetch_add(1, std::memory_order_relaxed);
        stats_.total_tier1_compile_time_us.fetch_add(d.elapsed_us, std::memory_order_relaxed);
    }
    std::lock_guard<std::mutex> lock(compiling_mutex_);
    // The last member's own guard releases it.
    for (size_t i = 0; i + 1 < group.size(); ++i) in_progress_compilations_.erase(group[i].name);
}

void MultiTierPipeline::drop_tier1_group(std::vector<detail::Tier1Deferred>& group) {
    // Left in Tier 0, to be compiled again when next asked for.
    std::lock_guard<std::mutex> lock(compiling_mutex_);
    for (const auto& d : group) in_progress_compilations_.erase(d.name);
    group.clear();
}

MultiTierPipeline::CalleeLink MultiTierPipeline::link_tier1_callees(
    const Function& fn, const codegen::BaselineCompiledFunction& compiled) {
    const Module* mod = fn.parent() ? fn.parent() : tiering().active_module();
    CalleeLink result = CalleeLink::Ready;
    for (const std::string& sym : compiled.lazy_call_symbols()) {
        // Anything else is a host symbol, bound when it is registered.
        const Function* def = mod ? mod->get_function(sym) : nullptr;
        if (!def || def->block_count() == 0) continue;
        if (FunctionHandle* h = table_->find(sym); h && h->native_entry()) continue;
        // A function this thread is compiling, or has compiled into a
        // frame's group: this code depends on that frame.
        bool on_stack = false;
        for (size_t i = 0; i < t_tier1_compiling.size() && !on_stack; ++i) {
            const Tier1CompileFrame& f = t_tier1_compiling[i];
            if (f.pipeline != this) continue;
            on_stack = f.name == sym || std::any_of(f.group.begin(), f.group.end(),
                                                    [&](const Tier1Deferred& d) { return d.name == sym; });
            if (on_stack) {
                size_t& dep = t_tier1_compiling.back().depends_on;
                dep = std::min(dep, i);
            }
        }
        if (on_stack) continue;
        if (compile_and_install_tier1(sym, def)) continue;
        if (is_baseline_rejected(sym)) return CalleeLink::Rejected;
        // Compiling on another thread: this function tiers up later.
        result = CalleeLink::Pending;
    }
    return result;
}

bool MultiTierPipeline::enqueue_tier2(
    std::string_view fn_name,
    const Module* mod,
    FunctionHandle* handle
) {
    const Module* target_mod = mod ? mod : tiering().active_module();
    if (!target_mod) return false;
    if (!handle) {
        handle = table_->get_or_create(fn_name);
    }
    if (!handle) return false;

    bool enqueued = background_compiler().enqueue(
        fn_name,
        *target_mod,
        handle,
        CompilePriority::Normal,
        TierLevel::Tier2_Optimized
    );

    if (enqueued) {
        stats_.tier2_compilations.fetch_add(1, std::memory_order_relaxed);
    }
    return enqueued;
}

void MultiTierPipeline::on_invocation(std::string_view fn_name) {
    auto* handle = table_->find(fn_name);
    if (handle) {
        handle->record_call();
    }
    tier_invocation(tiering().get_feedback(fn_name), fn_name, handle);
}

void MultiTierPipeline::on_invocation(TieringFeedback& fb) {
    tier_invocation(fb, fb.function_name(), nullptr);
}

void MultiTierPipeline::tier_invocation(TieringFeedback& fb, std::string_view fn_name, FunctionHandle* handle) {
    TierLevel tier = fb.current_tier();
    uint64_t count = fb.record_invocation();

    if (tier == TierLevel::Tier0_Interpreter) {
        stats_.tier0_invocations.fetch_add(1, std::memory_order_relaxed);
        if (count >= config_.invocation_tier1_threshold && !fb.is_bailout_set()) {
            compile_and_install_tier1(fn_name);
        }
    } else if (tier == TierLevel::Tier1_Baseline) {
        stats_.tier1_invocations.fetch_add(1, std::memory_order_relaxed);
        if (count >= config_.invocation_tier2_threshold && !fb.is_bailout_set()) {
            if (config_.enable_background_compile || tiering().is_background_compile_enabled()) {
                enqueue_tier2(fn_name, tiering().active_module(), handle);
            }
        }
    } else if (tier == TierLevel::Tier2_Optimized) {
        stats_.tier2_invocations.fetch_add(1, std::memory_order_relaxed);
    }
}

RuntimeValue MultiTierPipeline::execute(
    Module& mod,
    std::string_view entry_fn,
    const std::vector<RuntimeValue>& args
) {
    ProgramScope program_scope(*table_);
    tiering().set_active_module(&mod);
    for (const auto* fn : mod.functions()) {
        if (fn) {
            table_->get_or_create(fn->name(), fn);
        }
    }

    if (config_.enable_background_compile) {
        background_compiler().start(config_.jit_threads);
    }

    auto* fn = mod.get_function(entry_fn);
    if (!fn) {
        throw std::runtime_error("MultiTierPipeline::execute: function '" + std::string(entry_fn) + "' not found");
    }

    auto* handle = table_->get_or_create(entry_fn, fn);
    RuntimeValue result;

    if (config_.use_fast_interpreter()) {
        bool expected = false;
        if (fast_interp_busy_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            struct Release {
                std::atomic<bool>& busy;
                ~Release() { busy.store(false, std::memory_order_release); }
            } release{fast_interp_busy_};
            FastInterpreter& fast_interp = persistent_fast_interpreter(mod);
            // Each execute starts with a fresh TLS block, as a new
            // interpreter would.
            fast_interp.set_tls_block(0);
            result = handle->call(fast_interp, args);
        } else {
            // Re-entrant or concurrent execute: the kept interpreter is in
            // use, so this one gets its own.
            FastInterpreter fast_interp;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                setup_fast_interpreter(fast_interp, mod);
            }
            result = handle->call(fast_interp, args);
        }
    } else {
        Interpreter interp;
        interp.set_dispatch_table(table_);
        il::register_bronze_interpreter_symbols(&interp);
        result = handle->call(interp, args);
    }

    if (config_.enable_background_compile) {
        background_compiler().wait_idle();
    }

    return result;
}

void MultiTierPipeline::setup_fast_interpreter(FastInterpreter& interp, Module& mod) {
    interp.set_dispatch_table(table_);
    interp.set_module(&mod);
    il::register_bronze_fast_interpreter_symbols(&interp);
    for (const auto& [sym, addr] : external_symbols_) {
        interp.register_external_symbol(sym, addr);
    }
    for (const auto& [name, fn_ptr] : external_functions_) {
        interp.register_external_function(name, fn_ptr);
    }
}

FastInterpreter& MultiTierPipeline::persistent_fast_interpreter(Module& mod) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!fast_interp_ || fast_interp_module_ != &mod || fast_interp_symbols_gen_ != symbols_gen_) {
        fast_interp_.reset();
        auto interp = std::make_unique<FastInterpreter>();
        setup_fast_interpreter(*interp, mod);
        fast_interp_ = std::move(interp);
        fast_interp_module_ = &mod;
        fast_interp_symbols_gen_ = symbols_gen_;
    }
    return *fast_interp_;
}

MultiTierStats MultiTierPipeline::stats() const {
    return stats_;
}

void MultiTierPipeline::reset_stats() {
    stats_.tier0_invocations.store(0, std::memory_order_relaxed);
    stats_.tier1_compilations.store(0, std::memory_order_relaxed);
    stats_.tier1_invocations.store(0, std::memory_order_relaxed);
    stats_.tier2_compilations.store(0, std::memory_order_relaxed);
    stats_.tier2_invocations.store(0, std::memory_order_relaxed);
    stats_.total_tier1_compile_time_us.store(0, std::memory_order_relaxed);
}

void MultiTierPipeline::dump_stats(std::ostream& os) const {
    MultiTierStats s = stats();
    uint64_t t0_invs = s.tier0_invocations.load(std::memory_order_relaxed);
    uint64_t t1_comps = s.tier1_compilations.load(std::memory_order_relaxed);
    uint64_t t1_invs = s.tier1_invocations.load(std::memory_order_relaxed);
    uint64_t t1_time = s.total_tier1_compile_time_us.load(std::memory_order_relaxed);
    uint64_t t2_comps = s.tier2_compilations.load(std::memory_order_relaxed);
    uint64_t t2_invs = s.tier2_invocations.load(std::memory_order_relaxed);

    os << "=== Multi-Tier Execution Pipeline Statistics ===\n";
    os << "  Tier 0 (" << to_string(config_.tier0_interpreter) << ") Invocations: " << t0_invs << "\n";
    os << "  Tier 1 (Baseline JIT) Compilations: " << t1_comps << "\n";
    os << "  Tier 1 (Baseline JIT) Invocations:   " << t1_invs << "\n";
    if (t1_comps > 0) {
        double avg_us = static_cast<double>(t1_time) / static_cast<double>(t1_comps);
        double fns_per_sec = (avg_us > 0.0) ? (1000000.0 / avg_us) : 0.0;
        os << "  Tier 1 Total Compile Time:          " << t1_time << " us ("
           << std::fixed << std::setprecision(2) << avg_us << " us/fn, ~"
           << static_cast<uint64_t>(fns_per_sec) << " fns/sec)\n";
    }
    os << "  Tier 2 (Optimized JIT) Compilations: " << t2_comps << "\n";
    os << "  Tier 2 (Optimized JIT) Invocations:  " << t2_invs << "\n";
    os << "=================================================\n";
    tiering().dump_stats(os);
}

} // namespace brass::runtime
