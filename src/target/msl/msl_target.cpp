#include "msl_isel.hpp"
#include <brass/mir/module.hpp>
#include <sstream>

namespace brass::target {

namespace {

std::string msl_preamble(bool needs_bf16) {
    std::ostringstream ss;
    ss << "#include <metal_stdlib>\n"
       << "using namespace metal;\n\n";
    if (needs_bf16) {
        ss << "inline int32_t _brass_f32_to_bf16(float x) {\n"
           << "    uint32_t u = as_type<uint32_t>(x);\n"
           << "    uint32_t hi = u >> 16;\n"
           << "    uint32_t rnd = (u + 0x7FFFu + (hi & 1u)) >> 16;\n"
           << "    bool is_nan = ((u & 0x7F800000u) == 0x7F800000u) && ((u & 0x007FFFFFu) != 0u);\n"
           << "    return as_type<int32_t>(is_nan ? (hi | 0x40u) : rnd);\n"
           << "}\n\n";
    }
    return ss.str();
}

} // namespace

MslKernel MslTarget::compile(const Function& fn, const MslOptions& opts) {
    msl::MslEmitter::KernelResult lowered = msl::MslEmitter::lower(fn, opts);
    MslKernel k;
    k.entry_name = std::string(fn.name());
    k.num_buffers = lowered.num_buffers;
    k.scalar_param_offsets = std::move(lowered.scalar_param_offsets);
    k.scalar_params_bytes = lowered.scalar_params_bytes;
    k.workgroup_x = opts.workgroup_x;
    k.workgroup_y = opts.workgroup_y;
    k.workgroup_z = opts.workgroup_z;
    k.uses_shared_memory = lowered.uses_shared_memory;
    k.shared_memory_bytes = lowered.shared_memory_bytes;
    k.source = msl_preamble(lowered.needs_bf16_helper) +
               lowered.params_struct_code +
               (lowered.params_struct_code.empty() ? "" : "\n") +
               lowered.function_code;
    return k;
}

std::string MslTarget::compile_module(const Module& mod, const MslOptions& opts) {
    std::vector<msl::MslEmitter::KernelResult> kernels;
    bool needs_bf16 = false;
    for (const auto* fn : mod.functions()) {
        if (!fn || fn->blocks().empty()) continue;
        auto kr = msl::MslEmitter::lower(*fn, opts);
        needs_bf16 |= kr.needs_bf16_helper;
        kernels.push_back(std::move(kr));
    }
    std::ostringstream ss;
    ss << msl_preamble(needs_bf16);
    for (size_t i = 0; i < kernels.size(); ++i) {
        if (i) ss << "\n";
        if (!kernels[i].params_struct_code.empty()) {
            ss << kernels[i].params_struct_code << "\n";
        }
        ss << kernels[i].function_code;
    }
    return ss.str();
}

} // namespace brass::target
