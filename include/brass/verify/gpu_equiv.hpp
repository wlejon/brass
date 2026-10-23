#pragma once

#include <brass/mir/function.hpp>
#include <brass/target/ptx/ptx_ir.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/target/spirv/spirv_ir.hpp>
#include <brass/target/spirv_target.hpp>
#include <brass/verify/smt_equiv.hpp>
#include <string>

namespace brass::verify {

/// Verify symbolic bitvector equivalence between a MIR GPU kernel `mir_fn` and a lowered
/// `spirv::Module` `spv_mod` across all 32-bit/64-bit/IEEE-754 kernel parameters,
/// GPU thread coordinates (`tid`, `ctaid`, `ntid`, `nctaid`, `laneid`, `global_tid_x`),
/// workgroup shared memory (`StorageClassWorkgroup`), and global buffer memory
/// (`PhysicalStorageBuffer`).
EquivResult verify_spirv_equivalence(
    const Function& mir_fn,
    const spirv::Module& spv_mod,
    std::string* err_out = nullptr,
    const EquivOptions& opts = {});

inline EquivResult verify_spirv_equivalence(
    const Function& mir_fn,
    const spirv::Module& spv_mod,
    const EquivOptions& opts) {
    return verify_spirv_equivalence(mir_fn, spv_mod, nullptr, opts);
}

/// Convenience overload: lower `mir_fn` via `SpirvISel` with `spv_opts` and verify
/// symbolic equivalence between `mir_fn` and the lowered `spirv::Module`.
EquivResult verify_spirv_equivalence(
    const Function& mir_fn,
    const target::SpirvOptions& spv_opts = {},
    std::string* err_out = nullptr,
    const EquivOptions& opts = {});

/// Verify symbolic bitvector equivalence between a MIR GPU kernel `mir_fn` and a lowered
/// `ptx::Function` (`ptx::Module` / `ptx::Kernel`) across all kernel parameters,
/// special registers (`%tid`, `%ctaid`, `%ntid`, `%nctaid`, `%laneid`),
/// `.shared` arrays, and `.global` memory accesses.
EquivResult verify_ptx_equivalence(
    const Function& mir_fn,
    const ptx::Function& ptx_mod,
    std::string* err_out = nullptr,
    const EquivOptions& opts = {});

inline EquivResult verify_ptx_equivalence(
    const Function& mir_fn,
    const ptx::Function& ptx_mod,
    const EquivOptions& opts) {
    return verify_ptx_equivalence(mir_fn, ptx_mod, nullptr, opts);
}

/// Convenience overload: lower `mir_fn` via `PtxISel` (and `ptx::cleanup` if enabled in `ptx_opts`)
/// and verify symbolic equivalence between `mir_fn` and the lowered `ptx::Function`.
EquivResult verify_ptx_equivalence(
    const Function& mir_fn,
    const target::PtxOptions& ptx_opts = {},
    std::string* err_out = nullptr,
    const EquivOptions& opts = {});

/// Cross-backend symbolic equivalence check: verify `mir_fn == spv_mod == ptx_mod`
/// in a single shared `BVContext` (comparing MIR vs SPIR-V, MIR vs PTX, and SPIR-V vs PTX).
EquivResult verify_spirv_ptx_equivalence(
    const Function& mir_fn,
    const spirv::Module& spv_mod,
    const ptx::Function& ptx_mod,
    std::string* err_out = nullptr,
    const EquivOptions& opts = {});

/// Convenience overload: lower `mir_fn` to both SPIR-V and PTX and prove `MIR == SPIR-V == PTX`.
EquivResult verify_spirv_ptx_equivalence(
    const Function& mir_fn,
    std::string* err_out = nullptr,
    const EquivOptions& opts = {});

} // namespace brass::verify

namespace brass {
using verify::verify_ptx_equivalence;
using verify::verify_spirv_equivalence;
using verify::verify_spirv_ptx_equivalence;
} // namespace brass
