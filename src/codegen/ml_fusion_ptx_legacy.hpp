#pragma once

// LEGACY: the hand-written PTX string templates that Stage 5a replaced with
// MIR builders (build_ptx_swiglu / build_ptx_adaln_modulate /
// build_ptx_residual_rms_norm in ml_fusion_ptx_kernels.cpp). They are kept
// only so tests/unit/test_gpu_kernel_migration.cpp can run the old and new
// kernels on identical inputs; nothing in the library calls them. Deleted in
// Stage 6 together with this header. Internal: not installed, not part of
// the public brass API.

#include <brass/target/ptx_target.hpp>

#include <string>

namespace brass::codegen::legacy {

std::string legacy_ptx_swiglu(const target::PtxOptions& opts = {});
std::string legacy_ptx_adaln_modulate(bool gated, const target::PtxOptions& opts = {});
std::string legacy_ptx_residual_rms_norm(const target::PtxOptions& opts = {});

} // namespace brass::codegen::legacy
