#include "tier2_link.hpp"

#include <brass/gc/runtime_gc.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/pass_catalog.hpp>
#include <brass/mir/pass_pipeline.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/runtime/host_symbols.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

extern "C" void brass_pgo_inc(uint32_t);

namespace brass::runtime::detail {

namespace {

// The tier-2 JIT pipeline: feedback-driven speculative devirtualization
// (from the type feedback of the program being compiled for), then the
// program's own passes when it set them, else the scalar and CFG passes,
// the default loop stage and write-barrier elimination.
Pipeline tier2_pipeline(const FeedbackRegistry& feedback, const std::optional<PassPipelineOptions>& passes) {
    Pipeline p;
    p.add(passes::speculative_devirtualization(feedback));
    if (passes) {
        p.append(pass_pipeline(*passes));
        return p;
    }
    LoopOptOptions loop_opts;
    p.add(passes::gvn());
    p.add(passes::gvn_pre());
    p.add(passes::sccp(true));
    p.add(passes::cfg_simplify());
    p.add(passes::loop_unswitch(loop_opts, false));
    p.add(passes::jump_threading(loop_opts, false));
    p.add(passes::cfg_simplify("cfg_simplify 2"));
    p.append(loop_pipeline(loop_opts));
    p.add(passes::write_barrier_elim());
    return p;
}

// The prefix of the symbol a canonicalized func_addr links against.
constexpr std::string_view kCanonicalFnPtrPrefix = "brass.fn_ptr:";

} // namespace

std::vector<std::string> speculated_call_targets(FunctionDispatchTable& table, std::string_view fn_name) {
    std::vector<std::string> targets;
    if (const TypeFeedbackVector* tfv = table.tiering().type_feedback().find(fn_name)) {
        for (const FeedbackSlot& slot : tfv->slots()) {
            if (slot.kind != FeedbackSlotKind::Call || slot.is_megamorphic()) continue;
            for (const CallFeedback& t : slot.targets) targets.push_back(t.target_name);
        }
    }
    return targets;
}

void bind_declared_handles(FunctionDispatchTable& table, const Module& copy, const Module& src) {
    for (std::string_view name : copy.external_symbols()) {
        if (copy.get_function(name)) continue;
        const Function* def = src.get_function(name);
        if (def && def->block_count() != 0 && !table.find(name)) table.get_or_create(name, def);
    }
}

bool run_tier2_optimization_pipeline(Module& mod, FunctionDispatchTable& table, std::string& errors) {
    run_pipeline(mod, tier2_pipeline(table.tiering().type_feedback(), table.pipeline().tier2_passes()));
    DiagnosticReporter diag;
    if (verify_module(mod, &diag)) return true;
    errors = diag.format_all();
    return false;
}

std::shared_ptr<codegen::JitExecutionEngine> make_tier2_engine(FunctionDispatchTable& table, const Target& target) {
    auto jit = std::make_shared<codegen::JitExecutionEngine>(target);
    jit->register_external_symbol("brass_gc_safepoint", reinterpret_cast<void*>(&brass_gc_safepoint));
    jit->register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
    jit->register_external_symbol("brass_gc_collect", reinterpret_cast<void*>(&brass_gc_collect));
    jit->register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));
    jit->register_external_symbol("brass_record_call_feedback", reinterpret_cast<void*>(&brass_record_call_feedback));
    jit->register_external_symbol("brass_record_property_feedback",
                                  reinterpret_cast<void*>(&brass_record_property_feedback));
    install_host_symbols(*jit);
    // The program's own symbols over the host's.
    table.pipeline().install_external_symbols(*jit);
    return jit;
}

void canonicalize_function_addresses(Module& mod, const std::function<bool(std::string_view)>& known,
                                     MultiTierPipeline& pipeline, codegen::JitExecutionEngine& jit) {
    std::unordered_map<std::string, std::string> canonical;  // name -> symbol
    auto symbol_for = [&](std::string_view name) -> const std::string* {
        const std::string key(name);
        if (auto it = canonical.find(key); it != canonical.end()) return &it->second;
        const Function* def = mod.get_function(name);
        if (!def || def->block_count() == 0) return nullptr;
        if (!known(name)) return nullptr;
        // No Function: the handle exists and keeps what it is bound to.
        void* stub = pipeline.function_address(name, nullptr);
        if (!stub) return nullptr;
        std::string sym = std::string(kCanonicalFnPtrPrefix) + key;
        jit.register_external_symbol(sym, stub);
        return &canonical.emplace(key, std::move(sym)).first->second;
    };
    for (Function* fn : mod.functions()) {
        if (!fn) continue;
        for (BasicBlock* bb : fn->blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                if (!inst || inst->opcode() != Opcode::func_addr) continue;
                if (const std::string* sym = symbol_for(inst->symbol())) {
                    inst->set_symbol(mod.string_pool().intern(*sym));
                }
            }
        }
    }
}

void link_declared_functions(const Module& mod, FunctionDispatchTable& table, codegen::JitExecutionEngine& jit) {
    for (std::string_view name : mod.external_symbols()) {
        if (mod.get_function(name)) continue;
        const FunctionHandle* h = table.find(name);
        const Function* def = h ? h->mir_function() : nullptr;
        if (!def || def->block_count() == 0) continue;
        if (void* stub = table.pipeline().function_address(name, nullptr)) {
            jit.register_external_symbol(name, stub);
        }
    }
}

void register_tier2_resumer(FunctionDispatchTable& table, FunctionHandle& handle, void* entry,
                            const Function* compiled_from) {
    // The resumer is unregistered when the handle retires, which the table
    // does before it goes away, so capturing both raw is safe.
    FunctionHandle* hp = &handle;
    FunctionDispatchTable* tp = &table;
    register_deopt_resumer(entry, [hp, tp, entry](const DeoptFrame& frame) -> uint64_t {
        const Function* from = hp->deopt_function(entry);
        if (!from) {
            std::fprintf(stderr, "brass: fatal deoptimization error: tier-2 code of '%s' deoptimized but the "
                                 "Function it was compiled from is gone\n",
                         std::string(hp->name()).c_str());
            std::fflush(stderr);
            std::abort();
        }
        // A MIR exception the Tier-0 continuation throws must reach the
        // native callers' landing pads (an `invoke` in tier-2 code) as a
        // native throw does; a C++ exception passes them by.
        try {
            return tp->pipeline().resume_after_deopt(*hp, frame, *tp, *from);
        } catch (const InterpreterThrownException& ex) {
            deopt_handler_throw_native(ex.value().raw_bits(), UINTPTR_MAX, std::current_exception());
        } catch (const BrassException& ex) {
            deopt_handler_throw_native(ex.value().raw(), UINTPTR_MAX, std::current_exception());
        }
        return 0;
    });
    handle.add_deopt_entry(entry, compiled_from);
}

} // namespace brass::runtime::detail
