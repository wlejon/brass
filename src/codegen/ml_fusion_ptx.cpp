// PTX emitters for the fused ML GPU kernels: each builds the kernel as MIR
// (ml_fusion_ptx_kernels*.cpp) and lowers it through PtxTarget::emit_function
// (PtxISel -> cleanup -> verify -> print). The kernels are validated on
// device by tests/unit/test_gpu_kernels*.cpp and test_gpu_execution.cpp.

#include <brass/codegen/ml_fusion.hpp>

namespace brass::codegen {

// =========================================================================
// residual RMSNorm, SwiGLU, AdaLN modulate (ml_fusion_ptx_kernels.cpp)
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
// LayerNorm-modulate, residual LayerNorm (ml_fusion_ptx_kernels_norm.cpp)
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
// GEMV SwiGLU, GEMV residual (ml_fusion_ptx_kernels_gemv.cpp)
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

// =========================================================================
// GEMV Q8_0, GEMV Q4_K (ml_fusion_ptx_kernels_quant.cpp)
// =========================================================================

std::string MlFusionCompiler::emit_ptx_fused_gemv_q8_0(const target::PtxOptions& opts) {
    Module mod("mod_ptx_gemv_q8_0");
    Function* fn = build_ptx_gemv_q8_0(mod);
    return target::PtxTarget::emit_function(*fn, opts);
}

std::string MlFusionCompiler::emit_ptx_fused_gemv_q4_k(const target::PtxOptions& opts) {
    Module mod("mod_ptx_gemv_q4_k");
    Function* fn = build_ptx_gemv_q4_k(mod);
    return target::PtxTarget::emit_function(*fn, opts);
}

} // namespace brass::codegen
