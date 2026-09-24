// Deoptimization of tier-2 code into Tier 0: install-time validation of
// every guard's deopt target, and the continuation that finishes a call in
// the interpreter when an optimized guard fails.
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/runtime/host_symbols.hpp>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace brass::runtime {

namespace {

const Instruction* find_guard(const Function& fn, uint32_t resume_id) {
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (inst && inst->opcode() == Opcode::guard && inst->resume_id() == resume_id) return inst;
        }
    }
    return nullptr;
}

const Function* exit_stub_of(const Function& fn, const Instruction& guard) {
    const Module* mod = fn.parent();
    if (!mod || guard.symbol().empty()) return nullptr;
    return mod->get_function(guard.symbol());
}

[[noreturn]] void deopt_fatal(const std::string& msg) {
    std::fprintf(stderr, "brass: fatal deoptimization error: %s\n", msg.c_str());
    std::fflush(stderr);
    std::abort();
}

// Raw slot bits as the value the interpreter holds for a value of type `t`.
RuntimeValue materialize(Type t, uint64_t bits) {
    switch (t.kind()) {
        case TypeKind::I8:
        case TypeKind::I16:
        case TypeKind::I32:
        case TypeKind::F32:
            bits &= 0xFFFFFFFFull;
            break;
        default:
            break;
    }
    return RuntimeValue::from_bits(t, bits);
}

} // namespace

bool deopt_targets_valid(const Function& optimized, const Function* tier0, std::string& why) {
    for (const auto* bb : optimized.blocks()) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (!inst || inst->opcode() != Opcode::guard) continue;
            const std::string site = "guard (resume id " + std::to_string(inst->resume_id()) + ")";
            if (!tier0) {
                why = site + " but no Tier-0 function to resume in";
                return false;
            }
            const Instruction* g = find_guard(*tier0, inst->resume_id());
            if (!g) {
                why = site + " has no matching guard in the Tier-0 function";
                return false;
            }
            if (g->state_map().size() != inst->state_map().size()) {
                why = site + " state map has " + std::to_string(inst->state_map().size()) +
                      " values, Tier-0 guard has " + std::to_string(g->state_map().size());
                return false;
            }
            if (!exit_stub_of(*tier0, *g) && !tier0->get_resume_target(g->resume_id())) {
                why = site + " has neither an exit stub function nor a resume target";
                return false;
            }
        }
    }
    return true;
}

uint64_t MultiTierPipeline::resume_after_deopt(FunctionHandle& handle, const DeoptFrame& frame) {
    return resume_after_deopt(handle, frame, *table_);
}

uint64_t MultiTierPipeline::resume_after_deopt(FunctionHandle& handle, const DeoptFrame& frame,
                                               FunctionDispatchTable& table) {
    if (&table != table_) {
        // The deopt counts and the Tier-0 config are this pipeline's
        // program's; resuming another program's code here would mix them.
        deopt_fatal("tier-2 code of '" + std::string(handle.name()) +
                    "' deoptimized into the pipeline of a different program");
    }
    const Function* fn = handle.mir_function();
    if (!fn) {
        deopt_fatal("optimized code of '" + std::string(handle.name()) + "' deoptimized but its MIR is gone");
    }
    const std::string fname(fn->name());
    const Instruction* guard = find_guard(*fn, frame.resume_id);
    if (!guard) {
        deopt_fatal("no guard with resume id " + std::to_string(frame.resume_id) + " in '" + fname + "'");
    }
    const auto& state_map = guard->state_map();
    if (state_map.size() != frame.count) {
        deopt_fatal("deopt frame of '" + fname + "' carries " + std::to_string(frame.count) +
                    " values, guard expects " + std::to_string(state_map.size()));
    }
    std::vector<RuntimeValue> state;
    state.reserve(frame.count);
    for (size_t i = 0; i < frame.count; ++i) {
        const Value* sv = state_map[i];
        state.push_back(materialize(sv ? sv->type() : Type::i64(), frame.slots[i]));
    }

    tier2_deopts_.fetch_add(1, std::memory_order_relaxed);
    TieringFeedback& fb = table.tiering().get_feedback(handle.name());
    fb.record_deopt(frame.resume_id);
    if (handle.tier() == TierLevel::Tier2_Optimized && fb.is_speculation_invalid(frame.resume_id)) {
        // The speculation is wrong for this program: stop entering the
        // optimized code and never recompile it (tier 2 has no
        // non-speculating variant to fall back to).
        fb.trigger_bailout("guard " + std::to_string(frame.resume_id) + " failed repeatedly in tier-2 code");
        handle.invalidate_optimized();
        fb.set_tier(handle.tier());
        tier2_invalidations_.fetch_add(1, std::memory_order_relaxed);
    }

    // The same exits the interpreter's guard takes, in its order.
    const Function* stub = exit_stub_of(*fn, *guard);
    const bool has_resume = fn->get_resume_target(frame.resume_id) != nullptr;
    if (!stub && !has_resume) {
        deopt_fatal("guard " + std::to_string(frame.resume_id) + " of '" + fname +
                    "' has neither an exit stub nor a resume target");
    }
    RuntimeValue result = stub ? run_fresh_tier0(table, stub, state)
                               : run_fresh_tier0(table, nullptr, state, fn, frame.resume_id);
    return result.is_void() ? 0 : result.raw_bits();
}

RuntimeValue MultiTierPipeline::run_fresh_tier0(FunctionDispatchTable& table, const Function* fn,
                                                const std::vector<RuntimeValue>& args, const Function* resume_fn,
                                                uint32_t resume_id) {
    const Function* owner = fn ? fn : resume_fn;
    Module* mod = owner ? owner->parent() : nullptr;
    if (config_.use_fast_interpreter()) {
        FastInterpreter interp;
        interp.set_dispatch_table(&table);
        if (mod) {
            std::lock_guard<std::mutex> lock(mutex_);
            setup_fast_interpreter(interp, *mod);
        }
        return fn ? interp.run(*fn, args) : interp.resume(*resume_fn, resume_id, args);
    }
    // As execute() sets up the oracle interpreter.
    Interpreter interp;
    interp.set_dispatch_table(&table);
    install_host_symbols(interp);
    return fn ? interp.run(*fn, args) : interp.resume(*resume_fn, resume_id, args);
}

} // namespace brass::runtime
