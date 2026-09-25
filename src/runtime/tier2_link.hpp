#pragma once

// What a tier-2 compile (CodeInstaller::install_tier2) and an OSR entry
// compile (OsrCoordinator, osr_coordinator.cpp) of one program's code share: the optimization
// passes, the engine with the program's symbols, the program's function
// pointers and stubs linked in, and the deopt continuation.

#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/module.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/target/target.hpp>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brass::runtime {
class MultiTierPipeline;
}

namespace brass::runtime::detail {

// The call targets the type feedback of `fn_name` names (at most a
// polymorphic site's), for their bodies to be copied in for speculative
// inlining.
std::vector<std::string> speculated_call_targets(FunctionDispatchTable& table, std::string_view fn_name);

// Gives each program function `copy` declares but does not define a handle
// bound to its definition in `src` (as the program's first call to it
// would), so the copy links to its stub.
void bind_declared_handles(FunctionDispatchTable& table, const Module& copy, const Module& src);

// Runs the program's tier-2 passes (MultiTierPipeline::tier2_passes) over
// `mod` and verifies it; false with the verifier's report in `errors`.
bool run_tier2_optimization_pipeline(Module& mod, FunctionDispatchTable& table, std::string& errors);

// An engine for `target` that resolves the runtime's GC and feedback
// symbols, the host's, and the program's own (its data tables).
std::shared_ptr<codegen::JitExecutionEngine> make_tier2_engine(FunctionDispatchTable& table, const Target& target);

// Makes every func_addr of `mod` naming a program function `known` accepts
// yield the program's one address of it, its lazy stub (the address every
// other tier's func_addr yields), so a pointer compares equal whichever
// tier made it.
void canonicalize_function_addresses(Module& mod, const std::function<bool(std::string_view)>& known,
                                     MultiTierPipeline& pipeline, codegen::JitExecutionEngine& jit);

// Links the program functions `mod` names but does not define against
// their stubs: a call reaches whatever code each has.
void link_declared_functions(const Module& mod, FunctionDispatchTable& table, codegen::JitExecutionEngine& jit);

// Registers the deopt continuation of the tier-2 code at `entry`, compiled
// from `compiled_from` for `handle`'s function: a failed guard finishes
// the call in Tier 0 (MultiTierPipeline::resume_after_deopt).
void register_tier2_resumer(FunctionDispatchTable& table, FunctionHandle& handle, void* entry,
                            const Function* compiled_from);

} // namespace brass::runtime::detail
