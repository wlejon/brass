// Tier 6 of the differential fuzzer: the x64 baseline JIT (the tiering
// layer's Tier 1) on the module as given.
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/stack_map.hpp>

#include <chrono>
#include <memory>
#include <unordered_set>

namespace brass::fuzz {

namespace {

double since_ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// Module functions `fn` calls directly (other than itself).
std::vector<std::string> direct_callees(const Module& mod, const Function& fn) {
    std::vector<std::string> out;
    for (const auto* bb : fn.blocks()) {
        if (!bb) continue;
        for (const auto* inst : *bb) {
            if (!inst) continue;
            std::string_view callee;
            if (inst->opcode() == Opcode::call || inst->opcode() == Opcode::func_addr) callee = inst->symbol();
            else if (inst->opcode() == Opcode::patchable_call) callee = inst->extra_symbol();
            if (!callee.empty() && callee != fn.name() && mod.get_function(callee)) out.emplace_back(callee);
        }
    }
    return out;
}

} // namespace

TierResult DiffFuzzer::run_baseline(const Module& mod, std::string_view fn_name,
                                    const std::vector<RuntimeValue>& args, bool& rejected) {
    TierResult res;
    rejected = false;
    const auto t0 = std::chrono::steady_clock::now();

    struct Compiled {
        codegen::BaselineJitCompiler compiler{Target::host()};
        std::vector<std::shared_ptr<codegen::BaselineCompiledFunction>> fns;
        ModuleStackMap stack_maps;
    };
    auto state = std::make_shared<Compiled>();
    state->compiler.register_external_symbol("brass_parallel_for", reinterpret_cast<void*>(&brass_parallel_for));

    // Callees first, so every direct call resolves to compiled code; a call
    // cycle between functions leaves the tier out for this program.
    std::string compile_fault;
    bool cycle = false;
    const bool compile_ok = run_protected([&]() {
        std::unordered_set<std::string> done;
        size_t remaining = 0;
        for (const auto* f : mod.functions()) if (f) ++remaining;
        while (remaining > 0) {
            bool progress = false;
            for (const auto* f : mod.functions()) {
                if (!f || done.count(std::string(f->name()))) continue;
                bool ready = true;
                for (const auto& c : direct_callees(mod, *f)) ready = ready && done.count(c);
                if (!ready) continue;
                std::shared_ptr<codegen::BaselineCompiledFunction> compiled;
                try {
                    compiled = std::make_shared<codegen::BaselineCompiledFunction>(state->compiler.compile(*f));
                } catch (const codegen::UnsupportedOperation& e) {
                    rejected = true;
                    res.fault_message = e.operation();
                    return;
                }
                state->compiler.register_external_symbol(f->name(), compiled->entry_point());
                state->stack_maps.add_function(compiled->stack_map());
                state->fns.push_back(std::move(compiled));
                done.emplace(f->name());
                --remaining;
                progress = true;
            }
            if (!progress) {
                cycle = true;
                res.fault_message = "call cycle between functions";
                return;
            }
        }
    }, compile_fault);
    if (compile_ok && (rejected || cycle)) {
        rejected = true;
        return res;
    }
    if (!compile_ok) {
        res.status = ExecutionStatus::CompilationFailure;
        res.fault_message = "baseline JIT compilation failed: " + compile_fault;
        res.duration_ms = since_ms(t0);
        return res;
    }

    std::shared_ptr<codegen::BaselineCompiledFunction> entry;
    for (const auto& f : state->fns) if (f->name() == fn_name) entry = f;
    if (!entry) {
        res.status = ExecutionStatus::CompilationFailure;
        res.fault_message = "baseline JIT: no function " + std::string(fn_name);
        return res;
    }

    auto res_box = std::make_shared<TierResult>();
    auto worker_task = [state, entry, res_box, args]() {
        auto gc = std::make_unique<MiniCheneyGC>(256 * 1024);
        MiniCheneyGC* old_gc = brass_get_active_gc();
        const ModuleStackMap* old_maps = brass_get_active_stack_maps();
        brass_set_active_gc(gc.get());
        brass_set_active_stack_maps(&state->stack_maps);
        struct Guard {
            MiniCheneyGC* gc;
            const ModuleStackMap* maps;
            ~Guard() {
                brass_set_active_gc(gc);
                brass_set_active_stack_maps(maps);
            }
        } guard{old_gc, old_maps};

        std::string fault;
        bool prot_ok = run_protected([&]() {
            res_box->value = entry->invoke(args);
            res_box->status = ExecutionStatus::Success;
        }, fault);
        if (!prot_ok) {
            res_box->status = ExecutionStatus::CrashOrFault;
            res_box->fault_message = std::move(fault);
        }
    };

    if (!run_with_watchdog(worker_task, options_.timeout_ms, WatchdogPolicy::ExitOnTimeout, "baseline JIT")) {
        res.status = ExecutionStatus::Timeout;
        res.fault_message = "Watchdog timeout exceeded in the baseline JIT";
    } else {
        res = std::move(*res_box);
    }
    res.duration_ms = since_ms(t0);
    return res;
}

} // namespace brass::fuzz
