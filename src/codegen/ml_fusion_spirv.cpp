// Backend-generic access to the fused ML GPU kernels: build any of them by
// id, and lower it through SpirvTarget for Vulkan. The MIR is exactly what
// the PTX emitters lower (ml_fusion_ptx_kernels*.cpp); nothing in it is
// PTX-specific beyond the ptx_* intrinsic names, which SpirvISel maps (warp
// code on 32-lane segments of the subgroup, see the design doc). Validated
// on device by tests/unit/test_spirv_kernels*.cpp.

#include <brass/codegen/ml_fusion.hpp>

#include <stdexcept>

namespace brass::codegen {

const char* gpu_kernel_entry(GpuKernel k) {
    switch (k) {
        case GpuKernel::swiglu: return "fused_swiglu_kernel";
        case GpuKernel::swiglu_packed: return "fused_swiglu_packed_kernel";
        case GpuKernel::adaln_modulate: return "fused_adaln_modulate_kernel";
        case GpuKernel::adaln_modulate_gated: return "fused_adaln_modulate_gated_kernel";
        case GpuKernel::residual_rms_norm: return "fused_residual_rms_norm_kernel";
        case GpuKernel::layernorm_modulate: return "fused_layernorm_modulate_kernel";
        case GpuKernel::residual_layernorm: return "fused_residual_layernorm_kernel";
        case GpuKernel::gemv_swiglu: return "fused_gemv_swiglu_kernel";
        case GpuKernel::gemv_residual: return "fused_gemv_residual_kernel";
        case GpuKernel::gemv_q8_0: return "fused_gemv_q8_0_kernel";
        case GpuKernel::gemv_q4_k: return "fused_gemv_q4_k_kernel";
    }
    return "";
}

Function* MlFusionCompiler::build_gpu_kernel(Module& mod, GpuKernel k) {
    switch (k) {
        case GpuKernel::swiglu: return build_ptx_swiglu(mod);
        case GpuKernel::swiglu_packed: return build_ptx_swiglu_packed(mod);
        case GpuKernel::adaln_modulate: return build_ptx_adaln_modulate(mod, false);
        case GpuKernel::adaln_modulate_gated: return build_ptx_adaln_modulate(mod, true);
        case GpuKernel::residual_rms_norm: return build_ptx_residual_rms_norm(mod);
        case GpuKernel::layernorm_modulate: return build_ptx_layernorm_modulate(mod);
        case GpuKernel::residual_layernorm: return build_ptx_residual_layernorm(mod);
        case GpuKernel::gemv_swiglu: return build_ptx_gemv_swiglu(mod);
        case GpuKernel::gemv_residual: return build_ptx_gemv_residual(mod);
        case GpuKernel::gemv_q8_0: return build_ptx_gemv_q8_0(mod);
        case GpuKernel::gemv_q4_k: return build_ptx_gemv_q4_k(mod);
    }
    throw std::invalid_argument("MlFusionCompiler::build_gpu_kernel: unknown kernel id");
}

target::SpirvKernel MlFusionCompiler::compile_spirv(GpuKernel k, const target::SpirvOptions& opts) {
#if BRASS_WITH_SPIRV
    Module mod(std::string("mod_spirv_") + gpu_kernel_entry(k));
    Function* fn = build_gpu_kernel(mod, k);
    return target::SpirvTarget::compile(*fn, opts);
#else
    (void)k;
    (void)opts;
    throw std::runtime_error("MlFusionCompiler::compile_spirv: brass was built without the SPIR-V target");
#endif
}

} // namespace brass::codegen
