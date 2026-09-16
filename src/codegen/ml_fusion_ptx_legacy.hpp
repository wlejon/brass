#pragma once

// LEGACY: the hand-written PTX string templates that Stage 5a/5b replaced
// with MIR builders (build_ptx_* in ml_fusion_ptx_kernels*.cpp). They are
// kept only so tests/unit/test_gpu_kernel_migration*.cpp can run the old and
// new kernels on identical inputs; nothing in the library calls them. Deleted
// in Stage 6 together with this header. Internal: not installed, not part of
// the public brass API.
//
//   ml_fusion_ptx_legacy.cpp       SwiGLU, AdaLN modulate x2, residual RMSNorm (5a)
//   ml_fusion_ptx_legacy_norm.cpp  LayerNorm-modulate, residual LayerNorm (5b)
//   ml_fusion_ptx_legacy_gemv.cpp  GEMV SwiGLU, GEMV residual (5b)

#include <brass/target/ptx_target.hpp>

#include <string>

namespace brass::codegen::legacy {

// The module header the string emitters prepended (.version/.target/.address_size).
std::string legacy_ptx_header(const target::PtxOptions& opts);

std::string legacy_ptx_swiglu(const target::PtxOptions& opts = {});
std::string legacy_ptx_adaln_modulate(bool gated, const target::PtxOptions& opts = {});
std::string legacy_ptx_residual_rms_norm(const target::PtxOptions& opts = {});

std::string legacy_ptx_layernorm_modulate(const target::PtxOptions& opts = {});
std::string legacy_ptx_residual_layernorm(const target::PtxOptions& opts = {});
std::string legacy_ptx_gemv_swiglu(const target::PtxOptions& opts = {});
std::string legacy_ptx_gemv_residual(const target::PtxOptions& opts = {});

} // namespace brass::codegen::legacy
