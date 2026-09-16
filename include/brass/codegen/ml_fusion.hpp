#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/target/ptx_target.hpp>

#include <string>
#include <string_view>
#include <cstdint>
#include <cstddef>
#include <memory>

namespace brass::codegen {

// 1. FusedResidualRmsNorm:
// Updates residual[i] = x[i] + residual[i] in-place,
// and computes out[i] = residual[i] * rsqrt(mean(residual^2) + eps) * weight[i].
using FusedResidualRmsNormFn = void (*)(
    const float* x,
    float* residual,
    const float* weight,
    float* out,
    uint64_t n,
    float eps,
    float inv_n
);

inline void run_residual_rms_norm(
    FusedResidualRmsNormFn fn,
    const float* x,
    float* residual,
    const float* weight,
    float* out,
    uint64_t n,
    float eps = 1e-5f
) {
    float inv_n = 1.0f / static_cast<float>(n);
    fn(x, residual, weight, out, n, eps, inv_n);
}

// 2. FusedSwiGLU:
// Computes out[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i].
using FusedSwiGLUFn = void (*)(
    const float* gate,
    const float* up,
    float* out,
    uint64_t n
);

// 3. FusedAdaLNModulate:
// Ungated: out[i] = x[i] * (1.0f + scale[i]) + shift[i].
// Gated:   out[i] = (x[i] * (1.0f + scale[i]) + shift[i]) * gate[i].
using FusedAdaLNModulateFn = void (*)(
    const float* x,
    const float* scale,
    const float* shift,
    float* out,
    uint64_t n
);

using FusedAdaLNModulateGatedFn = void (*)(
    const float* x,
    const float* scale,
    const float* shift,
    const float* gate,
    float* out,
    uint64_t n
);

// 4. QuantizedQ8DotProduct:
// Computes dot product of two quantized int8 vectors with fp32 scaling:
// out[0] = sum(a[i] * b[i]) * scale_a * scale_b.
using QuantizedQ8DotProductFn = void (*)(
    const int8_t* a,
    const int8_t* b,
    float scale_a,
    float scale_b,
    float* out,
    uint64_t n
);

// Block Q8_0 structure: 32 int8 elements per block with fp32 scale d
struct BlockQ8_0 {
    float d;
    int8_t qs[32];
};

using BlockQ8DotProductFn = void (*)(
    const BlockQ8_0* a,
    const BlockQ8_0* b,
    float* out,
    uint64_t num_blocks
);

class MlFusionCompiler {
public:
    MlFusionCompiler();
    explicit MlFusionCompiler(KernelJit jit);
    ~MlFusionCompiler() = default;

    KernelJit& jit() noexcept { return jit_; }
    const KernelJit& jit() const noexcept { return jit_; }

    using GemvQ8_0Fn = void (*)(const void* W_q8_0, const float* X, float* Y, uint64_t N, uint64_t K);
    using GemvQ4_KFn = void (*)(const void* W_q4_k, const float* X, float* Y, uint64_t N, uint64_t K);

    // --- MIR Function Builders ---
    Function* build_residual_rms_norm(Module& mod, std::string_view name = "fused_residual_rms_norm");
    Function* build_swiglu(Module& mod, std::string_view name = "fused_swiglu");
    Function* build_adaln_modulate(Module& mod, bool gated = false, std::string_view name = "fused_adaln_modulate");
    Function* build_q8_dot(Module& mod, std::string_view name = "fused_q8_dot");
    Function* build_block_q8_dot(Module& mod, std::string_view name = "fused_block_q8_dot");
    Function* build_gemv_q8_0(Module& mod, std::string_view name = "gemv_q8_0");
    Function* build_gemv_q4_k(Module& mod, std::string_view name = "gemv_q4_k");

    // --- GPU kernel builders (PTX-only MIR: ptx_* intrinsics, shared memory) ---
    // These mirror the launch contract of the former hand-written PTX kernels
    // (entry name, parameter order and types, one block per row / grid-stride)
    // and are what emit_ptx_swiglu / emit_ptx_adaln_modulate /
    // emit_ptx_fused_residual_rms_norm lower. See src/codegen/ml_fusion_ptx_kernels.cpp.
    //
    //   fused_swiglu_kernel(gate, up, out, u32 n)                       grid-stride, v4 + scalar tail
    //   fused_adaln_modulate_kernel(x, scale, shift, y, u32 l, u32 d)   one block per row
    //   fused_adaln_modulate_gated_kernel(x, scale, shift, gate, y, u32 l, u32 d)
    //   fused_residual_rms_norm_kernel(x, res, gamma, y, u32 b, u32 d, f32 eps)
    //       x[row] += res[row] in place; y = x * gamma * rsqrt(mean(x^2) + eps); block size % 32 == 0
    //   fused_layernorm_modulate_kernel(x, gamma, beta, scale, shift, y, u32 r, u32 d, f32 eps)
    //       y = ((x - mean) * rstd * gamma + beta) * (1 + scale) + shift; two-pass; block size % 32 == 0
    //   fused_residual_layernorm_kernel(x, res, gamma, beta, y, u32 b, u32 d, f32 eps)
    //       x[row] += res[row] in place; y = (x - mean) * rstd * gamma + beta; block size % 32 == 0
    //   fused_gemv_swiglu_kernel(w_gate, w_up, x, y, u32 n, u32 k)    one block per output row
    //       y[row] = silu(w_gate[row] . x) * (w_up[row] . x); block size % 32 == 0
    //   fused_gemv_residual_kernel(w_down, x, res, y, u32 n, u32 k)
    //       y[row] = w_down[row] . x + res[row]; block size % 32 == 0
    //   fused_gemv_q8_0_kernel(w, x, y, u32 n, u32 k)                  one block per output row
    //       y[row] = dequant(w[row]) . x over Q8_0 blocks (34 bytes: f16 d + 32 int8);
    //       k % 32 == 0; block size % 32 == 0 (8 threads per Q8_0 block)
    //   fused_gemv_q4_k_kernel(w, x, y, u32 n, u32 k)
    //       same over Q4_K super-blocks (144 bytes: f16 d, f16 dmin, 12 bytes of 6-bit
    //       scales/mins, 128 bytes of nibbles); k % 256 == 0; block size % 64 == 0
    //       (64 threads per super-block)
    // (ml_fusion_ptx_kernels_norm.cpp / ml_fusion_ptx_kernels_gemv.cpp /
    //  ml_fusion_ptx_kernels_quant.cpp for the last six)
    Function* build_ptx_swiglu(Module& mod);
    Function* build_ptx_adaln_modulate(Module& mod, bool gated);
    Function* build_ptx_residual_rms_norm(Module& mod);
    Function* build_ptx_layernorm_modulate(Module& mod);
    Function* build_ptx_residual_layernorm(Module& mod);
    Function* build_ptx_gemv_swiglu(Module& mod);
    Function* build_ptx_gemv_residual(Module& mod);
    Function* build_ptx_gemv_q8_0(Module& mod);
    Function* build_ptx_gemv_q4_k(Module& mod);

    // --- CPU Compilation (Zero GC overhead) ---
    KernelFunction compile_residual_rms_norm();
    KernelFunction compile_swiglu();
    KernelFunction compile_adaln_modulate(bool gated = false);
    KernelFunction compile_q8_dot();
    KernelFunction compile_block_q8_dot();
    KernelFunction compile_gemv_q8_0();
    KernelFunction compile_gemv_q4_k();

    // --- PTX CUDA Code Generation ---
    std::string emit_ptx_residual_rms_norm(const target::PtxOptions& opts = {});
    std::string emit_ptx_fused_residual_rms_norm(const target::PtxOptions& opts = {});
    std::string emit_ptx_fused_residual_layernorm(const target::PtxOptions& opts = {});
    std::string emit_ptx_fused_layernorm_modulate(const target::PtxOptions& opts = {});
    std::string emit_ptx_swiglu(const target::PtxOptions& opts = {});
    std::string emit_ptx_adaln_modulate(bool gated = false, const target::PtxOptions& opts = {});
    std::string emit_ptx_q8_dot(const target::PtxOptions& opts = {});
    std::string emit_ptx_block_q8_dot(const target::PtxOptions& opts = {});
    std::string emit_ptx_fused_gemv_swiglu(const target::PtxOptions& opts = {});
    std::string emit_ptx_fused_gemv_residual(const target::PtxOptions& opts = {});
    std::string emit_ptx_fused_gemv_q8_0(const target::PtxOptions& opts = {});
    std::string emit_ptx_fused_gemv_q4_k(const target::PtxOptions& opts = {});

private:
    KernelJit jit_;
};

} // namespace brass::codegen
