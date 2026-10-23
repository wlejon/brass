// SPIR-V Range-Proven RelaxedPrecision / Float16 Optimizer Pass
// (`brass::spirv::promote_relaxed_precision`).

#define SPV_ENABLE_UTILITY_CODE 1
#include <brass/target/spirv/spirv_relaxed_precision.hpp>
#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <spirv/unified1/GLSL.std.450.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace brass::spirv {

namespace {

FloatInterval hull(const FloatInterval& a, const FloatInterval& b) noexcept {
    if (!a.is_bounded() || !b.is_bounded()) return FloatInterval::unbounded();
    return FloatInterval::closed(std::min(a.lo, b.lo), std::max(a.hi, b.hi));
}

FloatInterval iv_neg(const FloatInterval& a) noexcept {
    if (!a.is_bounded()) return FloatInterval::unbounded();
    return FloatInterval::closed(-a.hi, -a.lo);
}

FloatInterval iv_add(const FloatInterval& a, const FloatInterval& b) noexcept {
    if (!a.is_bounded() || !b.is_bounded()) return FloatInterval::unbounded();
    return FloatInterval::closed(a.lo + b.lo, a.hi + b.hi);
}

FloatInterval iv_sub(const FloatInterval& a, const FloatInterval& b) noexcept {
    if (!a.is_bounded() || !b.is_bounded()) return FloatInterval::unbounded();
    return FloatInterval::closed(a.lo - b.hi, a.hi - b.lo);
}

FloatInterval iv_mul(const FloatInterval& a, const FloatInterval& b) noexcept {
    if (!a.is_bounded() || !b.is_bounded()) return FloatInterval::unbounded();
    double p1 = a.lo * b.lo;
    double p2 = a.lo * b.hi;
    double p3 = a.hi * b.lo;
    double p4 = a.hi * b.hi;
    return FloatInterval::closed(std::min({p1, p2, p3, p4}), std::max({p1, p2, p3, p4}));
}

FloatInterval iv_div(const FloatInterval& a, const FloatInterval& b) noexcept {
    if (!a.is_bounded() || !b.is_bounded()) return FloatInterval::unbounded();
    // Avoid zero or near-subnormal FP16 divisors (FP16 normal min is ~6.1e-5).
    if (b.lo <= 1e-3 && b.hi >= -1e-3) return FloatInterval::unbounded();
    double q1 = a.lo / b.lo;
    double q2 = a.lo / b.hi;
    double q3 = a.hi / b.lo;
    double q4 = a.hi / b.hi;
    return FloatInterval::closed(std::min({q1, q2, q3, q4}), std::max({q1, q2, q3, q4}));
}

FloatInterval iv_abs(const FloatInterval& a) noexcept {
    if (!a.is_bounded()) return FloatInterval::unbounded();
    if (a.lo <= 0.0 && a.hi >= 0.0) {
        return FloatInterval::closed(0.0, std::max(-a.lo, a.hi));
    }
    double x = std::abs(a.lo);
    double y = std::abs(a.hi);
    return FloatInterval::closed(std::min(x, y), std::max(x, y));
}

FloatInterval iv_min(const FloatInterval& a, const FloatInterval& b) noexcept {
    if (!a.is_bounded() || !b.is_bounded()) return FloatInterval::unbounded();
    return FloatInterval::closed(std::min(a.lo, b.lo), std::min(a.hi, b.hi));
}

FloatInterval iv_max(const FloatInterval& a, const FloatInterval& b) noexcept {
    if (!a.is_bounded() || !b.is_bounded()) return FloatInterval::unbounded();
    return FloatInterval::closed(std::max(a.lo, b.lo), std::max(a.hi, b.hi));
}

FloatInterval iv_clamp(const FloatInterval& x, const FloatInterval& lo, const FloatInterval& hi) noexcept {
    if (!lo.is_bounded() || !hi.is_bounded()) return FloatInterval::unbounded();
    double c_lo = std::min(lo.lo, hi.lo);
    double c_hi = std::max(lo.hi, hi.hi);
    if (c_lo > c_hi) return FloatInterval::unbounded();
    if (x.is_bounded()) {
        double r_lo = std::clamp(x.lo, c_lo, c_hi);
        double r_hi = std::clamp(x.hi, c_lo, c_hi);
        if (r_lo <= r_hi) return FloatInterval::closed(r_lo, r_hi);
    }
    return FloatInterval::closed(c_lo, c_hi);
}

FloatInterval iv_mix(const FloatInterval& x, const FloatInterval& y, const FloatInterval& a) noexcept {
    if (!x.is_bounded() || !y.is_bounded() || !a.is_bounded()) return FloatInterval::unbounded();
    if (a.lo >= 0.0 && a.hi <= 1.0) {
        // Convex combination of x and y.
        return FloatInterval::closed(std::min(x.lo, y.lo), std::max(x.hi, y.hi));
    }
    FloatInterval one_minus_a = iv_sub(FloatInterval::exact(1.0), a);
    return iv_add(iv_mul(x, one_minus_a), iv_mul(y, a));
}

bool is_f32_scalar_or_composite(const Module& mod, Id type_id, uint32_t* out_lanes = nullptr) {
    const TypeInfo* ti = mod.type_info(type_id);
    if (!ti) return false;
    if (ti->op == spv::OpTypeFloat && ti->width == 32) {
        if (out_lanes) *out_lanes = 1;
        return true;
    }
    if (ti->op == spv::OpTypeVector) {
        const TypeInfo* elem = mod.type_info(ti->elem);
        if (elem && elem->op == spv::OpTypeFloat && elem->width == 32) {
            if (out_lanes) *out_lanes = ti->width;
            return true;
        }
    }
    if (ti->op == spv::OpTypeMatrix) {
        uint32_t col_lanes = 0;
        if (is_f32_scalar_or_composite(mod, ti->elem, &col_lanes)) {
            if (out_lanes) *out_lanes = col_lanes * ti->width;
            return true;
        }
    }
    return false;
}

bool is_range_establishing_ext(uint32_t ext_op) noexcept {
    switch (ext_op) {
        case GLSLstd450Sin:
        case GLSLstd450Cos:
        case GLSLstd450Tanh:
        case GLSLstd450FSign:
        case GLSLstd450Step:
        case GLSLstd450SmoothStep:
        case GLSLstd450Fract:
        case GLSLstd450FClamp:
        case GLSLstd450NClamp:
        case GLSLstd450Normalize:
            return true;
        default:
            return false;
    }
}

FloatInterval eval_glsl_ext(const Module& mod, const Inst& inst,
                            const std::unordered_map<Id, FloatInterval>& intervals) {
    if (inst.operands.size() < 2) return FloatInterval::unbounded();
    uint32_t ext_op = inst.operands[1].value;
    auto get = [&](size_t arg_idx) -> FloatInterval {
        size_t op_idx = 2 + arg_idx;
        if (op_idx >= inst.operands.size()) return FloatInterval::unbounded();
        auto it = intervals.find(inst.operands[op_idx].value);
        return it == intervals.end() ? FloatInterval::unbounded() : it->second;
    };

    switch (ext_op) {
        case GLSLstd450Sin:
        case GLSLstd450Cos:
        case GLSLstd450Tanh:
        case GLSLstd450FSign:
        case GLSLstd450Normalize:
            return FloatInterval::closed(-1.0, 1.0);

        case GLSLstd450Step:
        case GLSLstd450SmoothStep:
        case GLSLstd450Fract:
            return FloatInterval::closed(0.0, 1.0);

        case GLSLstd450FClamp:
        case GLSLstd450NClamp:
            return iv_clamp(get(0), get(1), get(2));

        case GLSLstd450FAbs:
            return iv_abs(get(0));

        case GLSLstd450FMin:
        case GLSLstd450NMin:
            return iv_min(get(0), get(1));

        case GLSLstd450FMax:
        case GLSLstd450NMax:
            return iv_max(get(0), get(1));

        case GLSLstd450FMix:
            return iv_mix(get(0), get(1), get(2));

        case GLSLstd450Fma:
            return iv_add(iv_mul(get(0), get(1)), get(2));

        case GLSLstd450Sqrt: {
            FloatInterval a = get(0);
            if (!a.is_bounded() || a.lo < 0.0) return FloatInterval::unbounded();
            return FloatInterval::closed(std::sqrt(a.lo), std::sqrt(a.hi));
        }

        case GLSLstd450InverseSqrt: {
            FloatInterval a = get(0);
            if (!a.is_bounded() || a.lo < 1e-4) return FloatInterval::unbounded();
            return FloatInterval::closed(1.0 / std::sqrt(a.hi), 1.0 / std::sqrt(a.lo));
        }

        case GLSLstd450Floor:
        case GLSLstd450Ceil:
        case GLSLstd450Trunc:
        case GLSLstd450Round:
        case GLSLstd450RoundEven: {
            FloatInterval a = get(0);
            if (!a.is_bounded()) return FloatInterval::unbounded();
            return FloatInterval::closed(std::floor(a.lo), std::ceil(a.hi));
        }

        case GLSLstd450Exp: {
            FloatInterval a = get(0);
            if (!a.is_bounded() || a.hi > 10.0) return FloatInterval::unbounded();
            return FloatInterval::closed(std::exp(a.lo), std::exp(a.hi));
        }

        case GLSLstd450Exp2: {
            FloatInterval a = get(0);
            if (!a.is_bounded() || a.hi > 14.0) return FloatInterval::unbounded();
            return FloatInterval::closed(std::exp2(a.lo), std::exp2(a.hi));
        }

        case GLSLstd450Length: {
            if (inst.operands.size() < 3) return FloatInterval::unbounded();
            FloatInterval a = get(0);
            if (!a.is_bounded()) return FloatInterval::unbounded();
            uint32_t lanes = 1;
            is_f32_scalar_or_composite(mod, mod.type_of_value(inst.operands[2].value), &lanes);
            double m = std::max(std::abs(a.lo), std::abs(a.hi));
            return FloatInterval::closed(0.0, std::sqrt(static_cast<double>(lanes)) * m);
        }

        default:
            return FloatInterval::unbounded();
    }
}

FloatInterval eval_instruction(
    const Module& mod,
    const Inst& inst,
    const std::unordered_map<Id, FloatInterval>& intervals,
    std::unordered_map<Id, std::vector<FloatInterval>>& comp_intervals) {
    auto get = [&](size_t idx) -> FloatInterval {
        if (idx >= inst.operands.size()) return FloatInterval::unbounded();
        auto it = intervals.find(inst.operands[idx].value);
        return it == intervals.end() ? FloatInterval::unbounded() : it->second;
    };

    switch (inst.op) {
        case spv::OpFNegate:
            return iv_neg(get(0));
        case spv::OpFAdd:
            return iv_add(get(0), get(1));
        case spv::OpFSub:
            return iv_sub(get(0), get(1));
        case spv::OpFMul:
        case spv::OpVectorTimesScalar:
            return iv_mul(get(0), get(1));
        case spv::OpDot: {
            if (inst.operands.size() < 2) return FloatInterval::unbounded();
            FloatInterval prod = iv_mul(get(0), get(1));
            if (!prod.is_bounded()) return FloatInterval::unbounded();
            uint32_t lanes = 4;
            is_f32_scalar_or_composite(mod, mod.type_of_value(inst.operands[0].value), &lanes);
            auto n = static_cast<double>(std::max(1u, lanes));
            return FloatInterval::closed(prod.lo * n, prod.hi * n);
        }
        case spv::OpFDiv:
            return iv_div(get(0), get(1));
        case spv::OpCompositeConstruct: {
            if (inst.operands.empty()) return FloatInterval::unbounded();
            std::vector<FloatInterval> comps;
            comps.reserve(inst.operands.size());
            FloatInterval acc = get(0);
            comps.push_back(acc);
            for (size_t i = 1; i < inst.operands.size(); ++i) {
                FloatInterval ci = get(i);
                comps.push_back(ci);
                acc = hull(acc, ci);
            }
            comp_intervals[inst.result] = std::move(comps);
            return acc;
        }
        case spv::OpCompositeExtract: {
            if (inst.operands.empty()) return FloatInterval::unbounded();
            Id src = inst.operands[0].value;
            if (inst.operands.size() == 2) {
                auto cit = comp_intervals.find(src);
                uint32_t idx = inst.operands[1].value;
                if (cit != comp_intervals.end() && idx < cit->second.size()) {
                    return cit->second[idx];
                }
            }
            return get(0);
        }
        case spv::OpCompositeInsert:
        case spv::OpVectorShuffle:
            return hull(get(0), get(1));
        case spv::OpSelect:
            return hull(get(1), get(2));
        case spv::OpPhi: {
            if (inst.operands.size() < 2) return FloatInterval::unbounded();
            FloatInterval acc = get(0);
            for (size_t k = 2; k < inst.operands.size(); k += 2) {
                acc = hull(acc, get(k));
            }
            return acc;
        }
        case spv::OpLoad:
            return get(0);
        case spv::OpExtInst:
            return eval_glsl_ext(mod, inst, intervals);
        default:
            return FloatInterval::unbounded();
    }
}

} // namespace

SpirvRelaxedPrecisionReport promote_relaxed_precision(
    Module& mod,
    const SpirvRelaxedPrecisionOptions& opts) {
    SpirvRelaxedPrecisionReport report;
    report.parse_ok = true;

    const double max_abs = static_cast<double>(opts.max_abs_fp16);
    std::unordered_set<Id> already_relaxed;
    std::unordered_set<Id> location_inputs;

    for (const Inst& ann : mod.annotations) {
        if (ann.op == spv::OpDecorate && ann.operands.size() >= 2) {
            Id target = ann.operands[0].value;
            auto dec = static_cast<spv::Decoration>(ann.operands[1].value);
            if (dec == spv::DecorationRelaxedPrecision) already_relaxed.insert(target);
            if (dec == spv::DecorationLocation) location_inputs.insert(target);
        }
    }

    std::unordered_map<Id, FloatInterval> intervals = opts.id_bounds;
    std::unordered_map<Id, std::vector<FloatInterval>> comp_intervals;

    // Seed intervals from 32-bit float constants, composite constants, and optional stage inputs.
    for (const Inst& g : mod.globals) {
        if (g.result == 0 || !g.type) continue;
        const TypeInfo* ti = mod.type_info(g.type);
        if ((g.op == spv::OpConstant || g.op == spv::OpSpecConstant) &&
            ti && ti->op == spv::OpTypeFloat && ti->width == 32 && !g.operands.empty()) {
            float f = 0.0f;
            uint32_t bits = g.operands[0].value;
            std::memcpy(&f, &bits, 4);
            intervals[g.result] = FloatInterval::exact(static_cast<double>(f));
        } else if (g.op == spv::OpConstantNull && is_f32_scalar_or_composite(mod, g.type)) {
            intervals[g.result] = FloatInterval::exact(0.0);
        } else if ((g.op == spv::OpConstantComposite || g.op == spv::OpSpecConstantComposite) &&
                   is_f32_scalar_or_composite(mod, g.type) && !g.operands.empty()) {
            std::vector<FloatInterval> comps;
            FloatInterval acc = intervals[g.operands[0].value];
            comps.push_back(acc);
            for (size_t i = 1; i < g.operands.size(); ++i) {
                FloatInterval ci = intervals[g.operands[i].value];
                comps.push_back(ci);
                acc = hull(acc, ci);
            }
            comp_intervals[g.result] = std::move(comps);
            intervals[g.result] = acc;
        } else if (g.op == spv::OpVariable && !g.operands.empty()) {
            auto sc = static_cast<spv::StorageClass>(g.operands[0].value);
            if (opts.assume_stage_inputs_unit_range && sc == spv::StorageClassInput &&
                location_inputs.count(g.result)) {
                intervals.emplace(g.result, FloatInterval::closed(-1.0, 1.0));
            }
        }
    }

    // Propagate intervals across functions (up to 4 passes for multi-block DAGs/loops).
    for (const Function& fn : mod.functions) {
        std::unordered_set<Id> local_vars;
        for (const Block& b : fn.blocks) {
            for (const Inst& inst : b.insts) {
                if (inst.op == spv::OpVariable && !inst.operands.empty()) {
                    auto sc = static_cast<spv::StorageClass>(inst.operands[0].value);
                    if (sc == spv::StorageClassFunction || sc == spv::StorageClassPrivate) {
                        local_vars.insert(inst.result);
                    }
                }
            }
        }

        for (int pass = 0; pass < 4; ++pass) {
            std::unordered_map<Id, FloatInterval> store_hull;
            std::unordered_set<Id> store_seen;
            for (const Block& b : fn.blocks) {
                for (const Inst& inst : b.insts) {
                    if (inst.result != 0 && inst.type != 0 && is_f32_scalar_or_composite(mod, inst.type)) {
                        if (!opts.id_bounds.count(inst.result)) {
                            intervals[inst.result] = eval_instruction(mod, inst, intervals, comp_intervals);
                        }
                    } else if (inst.op == spv::OpStore && inst.operands.size() >= 2) {
                        Id ptr = inst.operands[0].value;
                        Id val = inst.operands[1].value;
                        if (local_vars.count(ptr)) {
                            auto vit = intervals.find(val);
                            FloatInterval vi = (vit != intervals.end()) ? vit->second : FloatInterval::unbounded();
                            if (!store_seen.count(ptr)) {
                                store_seen.insert(ptr);
                                store_hull[ptr] = vi;
                            } else {
                                store_hull[ptr] = hull(store_hull[ptr], vi);
                            }
                        }
                    }
                }
            }
            for (const auto& [ptr, iv] : store_hull) {
                if (!opts.id_bounds.count(ptr)) intervals[ptr] = iv;
            }
        }

        // Audit and decorate eligible f32 SSA instructions.
        for (const Block& b : fn.blocks) {
            for (const Inst& inst : b.insts) {
                if (inst.result == 0 || inst.type == 0 || !is_f32_scalar_or_composite(mod, inst.type)) continue;
                if (inst.op == spv::OpVariable || inst.op == spv::OpFunctionParameter || inst.op == spv::OpUndef) {
                    continue;
                }

                ++report.total_float_defs;
                FloatInterval res_iv = intervals[inst.result];
                if (res_iv.is_bounded()) ++report.proven_bounded_defs;

                uint32_t ext_op = (inst.op == spv::OpExtInst && inst.operands.size() >= 2)
                                      ? inst.operands[1].value
                                      : 0u;
                bool range_source = (inst.op == spv::OpExtInst) && is_range_establishing_ext(ext_op);

                bool operands_safe = true;
                for (size_t i = 0; i < inst.operands.size(); ++i) {
                    const Operand& opd = inst.operands[i];
                    if (!opd.is_id()) continue;
                    if (inst.op == spv::OpExtInst && i == 0) continue; // ExtInstImport id
                    if (inst.op == spv::OpPhi && (i % 2 == 1)) continue; // predecessor block label
                    if (inst.op == spv::OpLoad) continue; // pointer operand

                    Id op_type = mod.type_of_value(opd.value);
                    if (op_type != 0 && !is_f32_scalar_or_composite(mod, op_type)) continue;

                    auto oit = intervals.find(opd.value);
                    FloatInterval oiv = (oit != intervals.end()) ? oit->second : FloatInterval::unbounded();
                    if (range_source) {
                        // Range-establishing ops (clamp, sin, cos, tanh, smoothstep, step)
                        // bound an unbounded input, but reject if an operand is known to
                        // exceed [-max_abs_fp16, +max_abs_fp16].
                        if (oiv.is_bounded() && !oiv.fits_within(max_abs)) {
                            operands_safe = false;
                            break;
                        }
                    } else {
                        if (!oiv.fits_within(max_abs)) {
                            operands_safe = false;
                            break;
                        }
                    }
                }

                bool can_promote = res_iv.fits_within(max_abs) && operands_safe;
                bool was_decorated = already_relaxed.count(inst.result) != 0;
                bool promoted_now = false;

                if (was_decorated) {
                    ++report.already_relaxed_count;
                } else if (can_promote) {
                    mod.decorate(inst.result, spv::DecorationRelaxedPrecision);
                    already_relaxed.insert(inst.result);
                    report.promoted_ids.push_back(inst.result);
                    ++report.promoted_count;
                    promoted_now = true;
                } else {
                    ++report.kept_fp32_count;
                }

                report.entries.push_back(SpirvRelaxedPrecisionEntry{
                    inst.result, inst.op, ext_op, res_iv, promoted_now, was_decorated});
            }
        }
    }

    return report;
}

SpirvRelaxedPrecisionReport promote_relaxed_precision(
    std::vector<uint32_t>& words,
    const SpirvRelaxedPrecisionOptions& opts) {
    Module mod;
    std::string err;
    if (!parse_binary(words, mod, &err)) {
        SpirvRelaxedPrecisionReport rep;
        rep.parse_ok = false;
        rep.error = std::move(err);
        return rep;
    }
    SpirvRelaxedPrecisionReport rep = promote_relaxed_precision(mod, opts);
    if (rep.ok() && rep.promoted_count > 0) {
        words = write(mod);
    }
    return rep;
}

} // namespace brass::spirv
