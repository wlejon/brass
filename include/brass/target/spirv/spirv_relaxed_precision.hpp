#pragma once

// SPIR-V Range-Proven RelaxedPrecision / Float16 Optimizer Pass
// (`brass::spirv::promote_relaxed_precision`).
//
// Analyzes 32-bit float SSA definitions in a SPIR-V module using monotone
// interval arithmetic ([lo, hi]) seeded from constants and bounded intrinsics
// (sin, cos, tanh, sign, step, smoothstep, fract, clamp) and propagated through
// arithmetic, vector composites, and GLSL.std.450 math ops. Any float SSA
// instruction whose result and float operands are provably bounded within
// [-max_abs_fp16, +max_abs_fp16] (default 1024.0f, well inside IEEE FP16
// [-65504, 65504]) receives `OpDecorate %id RelaxedPrecision`, while unbounded
// world-space coordinates and large accumulators remain in full FP32.

#include <brass/target/spirv/spirv_ir.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace brass::spirv {

struct FloatInterval {
    double lo = -std::numeric_limits<double>::infinity();
    double hi = +std::numeric_limits<double>::infinity();
    bool has_nan_or_inf = true;

    static FloatInterval unbounded() noexcept {
        return FloatInterval{-std::numeric_limits<double>::infinity(),
                             +std::numeric_limits<double>::infinity(), true};
    }

    static FloatInterval exact(double v) noexcept {
        if (!std::isfinite(v)) return unbounded();
        return FloatInterval{v, v, false};
    }

    static FloatInterval closed(double a, double b) noexcept {
        if (!std::isfinite(a) || !std::isfinite(b) || a > b) return unbounded();
        return FloatInterval{a, b, false};
    }

    bool is_bounded() const noexcept {
        return !has_nan_or_inf && std::isfinite(lo) && std::isfinite(hi) && lo <= hi;
    }

    bool fits_within(double max_abs) const noexcept {
        return is_bounded() && lo >= -max_abs && hi <= max_abs;
    }
};

struct SpirvRelaxedPrecisionOptions {
    // Maximum absolute value allowed for RelaxedPrecision promotion.
    // Default 1024.0f is safely within IEEE 754 binary16 [-65504, 65504] with
    // at least 1 ULP = 1.0f integer precision headroom.
    float max_abs_fp16 = 1024.0f;
    // When true, stage Input variables with Location decorations are assumed to
    // lie in [-1.0, 1.0] (e.g. normalized vertex attributes / UVs / colors).
    bool assume_stage_inputs_unit_range = false;
    // Optional caller-provided interval bounds for specific SPIR-V IDs.
    std::unordered_map<Id, FloatInterval> id_bounds;
};

struct SpirvRelaxedPrecisionEntry {
    Id id = 0;
    spv::Op op = spv::OpNop;
    uint32_t ext_op = 0;
    FloatInterval interval;
    bool promoted = false;
    bool already_decorated = false;
};

struct SpirvRelaxedPrecisionReport {
    bool parse_ok = false;
    uint32_t total_float_defs = 0;
    uint32_t proven_bounded_defs = 0;
    uint32_t promoted_count = 0;
    uint32_t already_relaxed_count = 0;
    uint32_t kept_fp32_count = 0;
    std::vector<Id> promoted_ids;
    std::vector<SpirvRelaxedPrecisionEntry> entries;
    std::string error;

    bool ok() const noexcept { return parse_ok && error.empty(); }
};

// Runs range analysis on a parsed `spirv::Module` and injects
// `OpDecorate %id RelaxedPrecision` for provably bounded f32 instructions.
SpirvRelaxedPrecisionReport promote_relaxed_precision(
    Module& mod,
    const SpirvRelaxedPrecisionOptions& opts = {});

// Parses `words`, injects `OpDecorate %id RelaxedPrecision` for provably bounded
// f32 instructions, and re-encodes `words` in-place.
SpirvRelaxedPrecisionReport promote_relaxed_precision(
    std::vector<uint32_t>& words,
    const SpirvRelaxedPrecisionOptions& opts = {});

} // namespace brass::spirv
