// PTX emitters for the fused ML GPU kernels: each builds the kernel as MIR
// (ml_fusion_ptx_kernels*.cpp) and lowers it through PtxTarget::emit_function
// (PtxISel -> verify -> print). The former hand-written string templates live
// in ml_fusion_ptx_legacy*.cpp until Stage 6, only as the reference side of
// tests/unit/test_gpu_kernel_migration*.cpp. The quantized GEMV emitters are
// still strings in ml_fusion_quant_ptx.cpp.

#include <brass/codegen/ml_fusion.hpp>

namespace brass::codegen {

// =========================================================================
// Stage 5a: residual RMSNorm, SwiGLU, AdaLN modulate (ml_fusion_ptx_kernels.cpp)
// =========================================================================

std::string MlFusionCompiler::emit_ptx_fused_residual_rms_norm(const target::PtxOptions& opts) {
    Module mod("mod_ptx_residual_rms_norm");
    Function* fn = build_ptx_residual_rms_norm(mod);
    return target::PtxTarget::emit_function(*fn, opts);
}

std::string MlFusionCompiler::emit_ptx_residual_rms_norm(const target::PtxOptions& opts) {
    return emit_ptx_fused_residual_rms_norm(opts);
}

std::string MlFusionCompiler::emit_ptx_swiglu(const target::PtxOptions& opts) {
    Module mod("mod_ptx_swiglu");
    Function* fn = build_ptx_swiglu(mod);
    return target::PtxTarget::emit_function(*fn, opts);
}

std::string MlFusionCompiler::emit_ptx_adaln_modulate(bool gated, const target::PtxOptions& opts) {
    Module mod(gated ? "mod_ptx_adaln_gated" : "mod_ptx_adaln");
    Function* fn = build_ptx_adaln_modulate(mod, gated);
    return target::PtxTarget::emit_function(*fn, opts);
}

// =========================================================================
// Stage 5b: LayerNorm-modulate, residual LayerNorm (ml_fusion_ptx_kernels_norm.cpp)
// =========================================================================

std::string MlFusionCompiler::emit_ptx_fused_layernorm_modulate(const target::PtxOptions& opts) {
    Module mod("mod_ptx_layernorm_modulate");
    Function* fn = build_ptx_layernorm_modulate(mod);
    return target::PtxTarget::emit_function(*fn, opts);
}

std::string MlFusionCompiler::emit_ptx_fused_residual_layernorm(const target::PtxOptions& opts) {
    Module mod("mod_ptx_residual_layernorm");
    Function* fn = build_ptx_residual_layernorm(mod);
    return target::PtxTarget::emit_function(*fn, opts);
}

// =========================================================================
// Stage 5b: GEMV SwiGLU, GEMV residual (ml_fusion_ptx_kernels_gemv.cpp)
// =========================================================================

std::string MlFusionCompiler::emit_ptx_fused_gemv_swiglu(const target::PtxOptions& opts) {
    Module mod("mod_ptx_gemv_swiglu");
    Function* fn = build_ptx_gemv_swiglu(mod);
    return target::PtxTarget::emit_function(*fn, opts);
}

std::string MlFusionCompiler::emit_ptx_fused_gemv_residual(const target::PtxOptions& opts) {
    Module mod("mod_ptx_gemv_residual");
    Function* fn = build_ptx_gemv_residual(mod);
    return target::PtxTarget::emit_function(*fn, opts);
}

} // namespace brass::codegen
