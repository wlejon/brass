// PtxTarget: the public facade over the PTX backend. Each function goes
// MIR -> PtxISel -> ptx::cleanup -> ptx::verify -> PtxPrinter. A verifier
// failure is a compiler bug (the ISel or the cleanup produced ill-typed PTX),
// so it is reported loudly with the full diagnostic list rather than handed
// to ptxas.

#include <brass/target/ptx_target.hpp>
#include <brass/target/ptx/ptx_cleanup.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/ptx/ptx_printer.hpp>
#include <brass/target/ptx/ptx_verifier.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace brass::target {

namespace {

ptx::Function lower_and_verify(const Function& fn, const PtxOptions& opts) {
    ptx::PtxISel isel;
    ptx::Function lowered = isel.lower(fn);
    if (opts.cleanup) ptx::cleanup(lowered);
    std::vector<ptx::Diagnostic> diags = ptx::verify(lowered);
    if (!diags.empty()) {
        throw std::runtime_error("PtxTarget: PtxISel produced ill-formed PTX for kernel '" +
                                 std::string(fn.name()) + "' (compiler bug):\n" +
                                 ptx::format_diagnostics(diags));
    }
    return lowered;
}

} // namespace

std::string PtxTarget::emit_function(const Function& fn, const PtxOptions& opts) {
    return ptx::print(lower_and_verify(fn, opts), opts);
}

std::string PtxTarget::emit_module(const Module& mod, const PtxOptions& opts) {
    std::vector<ptx::Function> lowered;
    lowered.reserve(mod.functions().size());
    for (const auto& fn_ptr : mod.functions()) {
        lowered.push_back(lower_and_verify(*fn_ptr, opts));
    }
    std::vector<const ptx::Function*> fns;
    fns.reserve(lowered.size());
    for (const auto& fn : lowered) fns.push_back(&fn);
    return ptx::print_module(fns, opts);
}

} // namespace brass::target
