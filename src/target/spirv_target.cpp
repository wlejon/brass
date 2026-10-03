// SpirvTarget: the public facade over the SPIR-V backend. Each function goes
// MIR -> SpirvISel (structurizer + selection) -> spirv::verify -> write. A
// verifier failure is a compiler bug and throws with every diagnostic
// attached; nothing that fails verification is returned.

#define SPV_ENABLE_UTILITY_CODE 1
#include <brass/target/spirv_target.hpp>
#include <brass/target/spirv/spirv_isel.hpp>
#include <brass/target/spirv/spirv_ir.hpp>

#include <stdexcept>
#include <string>

namespace brass::target {

namespace {

void check(const spirv::Module& m, const std::string& what) {
    std::vector<spirv::Diagnostic> diags = spirv::verify(m);
    if (!diags.empty()) {
        throw std::runtime_error("SpirvTarget: SpirvISel produced an ill-formed module for " + what +
                                 " (compiler bug):\n" + spirv::format_diagnostics(diags) + spirv::dump(m));
    }
}

spirv::KernelLayout lower_one(spirv::Module& m, const Function& fn, const SpirvOptions& opts) {
    spirv::SpirvISel isel(m, opts);
    return isel.lower(fn);
}

} // namespace

std::vector<uint32_t> SpirvTarget::emit_function(const Function& fn, const SpirvOptions& opts) {
    return compile(fn, opts).words;
}

std::vector<uint32_t> SpirvTarget::emit_module(const Module& mod, const SpirvOptions& opts) {
    spirv::Module m(opts.spirv_version);
    for (const auto& fn : mod.functions()) lower_one(m, *fn, opts);
    check(m, "module '" + std::string(mod.name()) + "'");
    return spirv::write(m);
}

SpirvKernel SpirvTarget::compile(const Function& fn, const SpirvOptions& opts) {
    spirv::Module m(opts.spirv_version);
    spirv::KernelLayout layout = lower_one(m, fn, opts);
    check(m, "kernel '" + std::string(fn.name()) + "'");

    SpirvKernel k;
    k.entry = std::string(fn.name());
    k.words = spirv::write(m);
    k.params = std::move(layout.params);
    k.push_constant_bytes = layout.push_constant_bytes;
    k.shared_bytes = layout.shared_bytes;
    k.local_size[0] = opts.local_size_x;
    k.local_size[1] = opts.local_size_y;
    k.local_size[2] = opts.local_size_z;
    k.local_size_spec_constants = opts.local_size_spec_constants;
    for (spv::Capability c : m.capabilities) k.capabilities.emplace_back(spv::CapabilityToString(c));
    k.extensions = m.extensions;
    return k;
}

std::string SpirvTarget::dump_function(const Function& fn, const SpirvOptions& opts) {
    spirv::Module m(opts.spirv_version);
    lower_one(m, fn, opts);
    check(m, "kernel '" + std::string(fn.name()) + "'");
    return spirv::dump(m);
}

} // namespace brass::target
