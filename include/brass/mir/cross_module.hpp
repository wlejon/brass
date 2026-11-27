#pragma once

#include <brass/debug/source_loc.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/gpu_portability.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/module.hpp>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace brass {

// ---------------------------------------------------------------------------
// Source-mapped provenance helpers across modules
// ---------------------------------------------------------------------------

/// Formats `loc` (plus any nested `InlinedScope` chain) using `ctx`.
/// Returns an empty string if `!loc.is_valid()`.
std::string format_debug_loc(const DebugContext& ctx, DebugLoc loc);

/// Merges the file table entries and `InlinedScope` chains referenced by
/// `cloned_fn` from `src.debug_context()` into `dst.debug_context()`, and
/// rewrites every instruction's `DebugLoc` in `cloned_fn` to use `dst`'s IDs.
void remap_function_debug_locs(Module& dst, const Module& src, Function& cloned_fn);

// ---------------------------------------------------------------------------
// Cross-module MIR linking / importing
// ---------------------------------------------------------------------------

struct CrossModuleLinkOptions {
    std::string symbol_prefix;
    bool import_callees = true;
    size_t max_callee_depth = 8;
    bool allow_overwrite = false;
};

struct CrossModuleLinkResult {
    bool ok = false;
    Function* imported = nullptr;
    std::vector<Function*> imported_callees;
    std::vector<std::string> imported_symbols;
    std::string diagnostic;
};

/// Clones `fn_name` (and optionally its transitive internal callees up to
/// `opts.max_callee_depth`) from `src` into `dst`, rewriting internal symbols,
/// merging referenced external declarations, `SymbolRole`s, string symbols,
/// and remapping `DebugContext` file/inline tables.
CrossModuleLinkResult import_function_into_module(
    Module& dst,
    const Module& src,
    std::string_view fn_name,
    const CrossModuleLinkOptions& opts = {});

// ---------------------------------------------------------------------------
// Domain compatibility auditing (unboxed numeric / no-GC / GPU)
// ---------------------------------------------------------------------------

enum class DomainViolationKind : uint8_t {
    GcRefType,
    TaggedType,
    WriteBarrier,
    AllocationCall,
    ImpureExternalCall,
    PinnedTls,
    DeoptOrGuard,
    ExceptionHandling,
    GpuPortability
};

struct DomainViolation {
    DomainViolationKind kind = DomainViolationKind::GcRefType;
    std::string function_name;
    DebugLoc loc;
    std::string formatted_loc;
    std::string detail;
};

struct DomainAuditOptions {
    bool require_no_gc = true;
    bool require_no_pinned_tls = true;
    bool require_no_alloc = true;
    bool require_no_deopt = false;
    bool require_no_eh = false;
    bool require_gpu_portable = false;
    GpuPortabilityOptions gpu_opts{};
};

struct DomainAuditReport {
    bool ok = true;
    std::vector<DomainViolation> violations;
    std::string diagnostic;
};

/// Audits `fn` inside `mod` against `opts`, producing structured violations
/// with formatted `DebugLoc` provenance from `mod.debug_context()`.
DomainAuditReport audit_function_domain(
    const Function& fn,
    const Module& mod,
    const DomainAuditOptions& opts = {});

// ---------------------------------------------------------------------------
// Boundary unboxing specialization & cross-domain call fusion
// ---------------------------------------------------------------------------

struct NaNBoxSchema {
    uint64_t int32_tag = 0x7FF9000000000000ULL;
    uint64_t bool_tag = 0x7FFA000000000000ULL;
    uint64_t tag_mask = 0xFFFF000000000000ULL;
    uint64_t payload_mask = 0x0000FFFFFFFFFFFFULL;
    bool doubles_unboxed = true;
};

enum class BoundaryConv : uint8_t {
    Identity,
    UnboxedF64ToTagged,
    UnboxedF32ToTaggedF64,
    UnboxedI32ToTagged,
    UnboxedI64ToTagged,
    TaggedToUnboxedF64,
    TaggedToUnboxedF32,
    TaggedToUnboxedI32,
    TaggedToUnboxedI64
};

struct DomainSpecializationSpec {
    std::string specialized_name;
    std::vector<Type> param_types;
    Type return_type = Type::void_type();
    std::vector<BoundaryConv> param_convs;
    BoundaryConv return_conv = BoundaryConv::Identity;
    NaNBoxSchema nanbox{};
    DomainAuditOptions audit_opts{};
};

struct DomainSpecializationResult {
    bool ok = false;
    Function* specialized = nullptr;
    DomainAuditReport audit;
    std::string diagnostic;
};

/// Creates a specialized unboxed wrapper around `callee` inside `dst`,
/// inlines `callee`, runs scalar/GVN/CFG/untag cleanups to collapse box/unbox
/// pairs, and validates the specialized body against `spec.audit_opts`.
/// On failure, any temporary specialized function is removed from `dst`.
DomainSpecializationResult specialize_callee_for_domain(
    Module& dst,
    const Function& callee,
    const DomainSpecializationSpec& spec);

struct CrossDomainFusionOptions {
    CrossModuleLinkOptions link_opts{};
    NaNBoxSchema nanbox{};
    DomainAuditOptions audit_opts{};
    size_t max_inlined_insts = 512;
    bool allow_nanbox_unboxing = true;
};

struct CrossDomainFusionResult {
    bool fused = false;
    Function* specialized_callee = nullptr;
    DomainAuditReport audit;
    std::string diagnostic;
};

/// Imports `callee_name` (and transitive callees) from `src` into `dst`,
/// specializes it for the caller's unboxed signature if needed, audits it
/// against `opts.audit_opts`, and inlines it into `call_inst` in `caller`.
/// Leaves `caller` untouched if any check fails.
CrossDomainFusionResult fuse_cross_module_call(
    Function& caller,
    Instruction* call_inst,
    Module& dst,
    const Module& src,
    std::string_view callee_name,
    const CrossDomainFusionOptions& opts = {});

} // namespace brass
