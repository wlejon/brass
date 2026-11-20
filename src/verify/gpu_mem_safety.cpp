#include "gpu_equiv_internal.hpp"
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <vector>

namespace brass::verify {

namespace {

bool is_disjoint_add_like_bv(const BVContext& ctx, const BVNode& n) noexcept {
    if (n.op == BVOp::Add) return true;
    if (n.op == BVOp::Or) {
        const uint64_t m = width_mask(n.width);
        return ((ctx.interval(n.a).known_zeros | ctx.interval(n.b).known_zeros) & m) == m;
    }
    return false;
}

bool match_bv_factor(const BVContext& ctx, BVNodeId a, BVNodeId b) noexcept {
    if (a == b) return true;
    const BVNode& na = bv_node(ctx, a);
    const BVNode& nb = bv_node(ctx, b);
    return na.op == BVOp::Const && nb.op == BVOp::Const && na.imm == nb.imm;
}

bool match_two_bv_factors(const BVContext& ctx, BVNodeId node, BVNodeId fp, BVNodeId fq) noexcept {
    const BVNode& n = bv_node(ctx, node);
    const BVNode& nfp = bv_node(ctx, fp);
    const BVNode& nfq = bv_node(ctx, fq);
    if (n.op == BVOp::Const && nfp.op == BVOp::Const && nfq.op == BVOp::Const) {
        return n.imm == nfp.imm * nfq.imm;
    }
    if (n.op == BVOp::Mul) {
        return (match_bv_factor(ctx, n.a, fp) && match_bv_factor(ctx, n.b, fq)) ||
               (match_bv_factor(ctx, n.a, fq) && match_bv_factor(ctx, n.b, fp));
    }
    if (n.op == BVOp::Shl && bv_node(ctx, n.b).op == BVOp::Const && bv_node(ctx, n.b).imm < 30) {
        uint64_t pow2 = 1ULL << bv_node(ctx, n.b).imm;
        return (nfp.op == BVOp::Const && nfp.imm == pow2 && match_bv_factor(ctx, n.a, fq)) ||
               (nfq.op == BVOp::Const && nfq.imm == pow2 && match_bv_factor(ctx, n.a, fp));
    }
    return false;
}

void flatten_add32_into(const BVContext& ctx, BVNodeId expr32, std::vector<BVNodeId>& out) {
    const BVNode& n = bv_node(ctx, expr32);
    if (is_disjoint_add_like_bv(ctx, n)) {
        flatten_add32_into(ctx, n.a, out);
        flatten_add32_into(ctx, n.b, out);
        return;
    }
    if (n.op == BVOp::Const && n.imm == 0) {
        return;
    }
    out.push_back(expr32);
}

BVNodeId sum_terms_32(BVContext& ctx, const std::vector<BVNodeId>& terms) {
    if (terms.empty()) return bv_const(ctx, 32, 0);
    BVNodeId acc = terms[0];
    for (size_t i = 1; i < terms.size(); ++i) {
        acc = bv_bin(ctx, BVOp::Add, 32, acc, terms[i]);
    }
    return acc;
}

BVNodeId sum_terms_64(BVContext& ctx, const std::vector<BVNodeId>& terms) {
    if (terms.empty()) return bv_const(ctx, 64, 0);
    BVNodeId acc = bv_ext(ctx, false, 64, terms[0]);
    for (size_t i = 1; i < terms.size(); ++i) {
        acc = bv_bin(ctx, BVOp::Add, 64, acc, bv_ext(ctx, false, 64, terms[i]));
    }
    return acc;
}

bool peel_single_64_to_32(
    BVContext& ctx,
    BVNodeId node64,
    uint32_t elem_bytes,
    BVNodeId& out_e32,
    bool& out_saw_sext) {
    const BVNode& n = bv_node(ctx, node64);
    if (n.op == BVOp::Const) {
        if (n.imm == 0) {
            out_e32 = bv_const(ctx, 32, 0);
            return true;
        }
        if (elem_bytes > 0 && n.imm % elem_bytes == 0 && (n.imm / elem_bytes) < 0x40000000ULL) {
            out_e32 = bv_const(ctx, 32, n.imm / elem_bytes);
            return true;
        }
        return false;
    }

    auto unwrap_ext32 = [&](BVNodeId id, BVNodeId& out_id) -> bool {
        const BVNode& z = bv_node(ctx, id);
        if ((z.op == BVOp::ZExt || z.op == BVOp::SExt) && ctx.width(z.a) == 32) {
            out_id = z.a;
            if (z.op == BVOp::SExt) out_saw_sext = true;
            return true;
        }
        return false;
    };

    if (elem_bytes == 1 && unwrap_ext32(node64, out_e32)) {
        return true;
    }

    if (n.op == BVOp::Shl && bv_node(ctx, n.b).op == BVOp::Const) {
        uint64_t sh = bv_node(ctx, n.b).imm;
        if (sh < 31 && (1ULL << sh) == elem_bytes && unwrap_ext32(n.a, out_e32)) {
            return true;
        }
    }

    if (n.op == BVOp::Mul) {
        const BVNode& na = bv_node(ctx, n.a);
        const BVNode& nb = bv_node(ctx, n.b);
        if (nb.op == BVOp::Const && nb.imm == elem_bytes && unwrap_ext32(n.a, out_e32)) {
            return true;
        }
        if (na.op == BVOp::Const && na.imm == elem_bytes && unwrap_ext32(n.b, out_e32)) {
            return true;
        }
    }

    return false;
}

bool peel_byte_offset_64(
    BVContext& ctx,
    BVNodeId node64,
    uint32_t elem_bytes,
    std::vector<BVNodeId>& out_terms32,
    BVNodeId& out_single32,
    bool& out_from_add64,
    bool& out_saw_sext,
    int depth = 0) {
    if (depth > 12) return false;
    BVNodeId e32 = 0;
    if (peel_single_64_to_32(ctx, node64, elem_bytes, e32, out_saw_sext)) {
        flatten_add32_into(ctx, e32, out_terms32);
        if (out_single32 == 0) {
            out_single32 = e32;
        } else {
            out_from_add64 = true;
            out_single32 = 0;
        }
        return true;
    }
    const BVNode& n = bv_node(ctx, node64);
    if (is_disjoint_add_like_bv(ctx, n)) {
        std::vector<BVNodeId> lhs_terms, rhs_terms;
        BVNodeId lhs_single = 0, rhs_single = 0;
        bool lhs_add64 = false, rhs_add64 = false;
        if (!peel_byte_offset_64(ctx, n.a, elem_bytes, lhs_terms, lhs_single, lhs_add64, out_saw_sext, depth + 1) ||
            !peel_byte_offset_64(ctx, n.b, elem_bytes, rhs_terms, rhs_single, rhs_add64, out_saw_sext, depth + 1)) {
            return false;
        }
        out_terms32.insert(out_terms32.end(), lhs_terms.begin(), lhs_terms.end());
        out_terms32.insert(out_terms32.end(), rhs_terms.begin(), rhs_terms.end());
        if (lhs_terms.empty() && !rhs_add64 && rhs_single != 0) {
            out_single32 = rhs_single;
            return true;
        }
        if (rhs_terms.empty() && !lhs_add64 && lhs_single != 0) {
            out_single32 = lhs_single;
            return true;
        }
        out_from_add64 = true;
        out_single32 = 0;
        return true;
    }
    return false;
}

bool match_term_div_single(const BVContext& ctx, BVNodeId term, BVNodeId factor, BVNodeId& out_q) noexcept {
    if (match_bv_factor(ctx, term, factor)) {
        return true; // sentinel handled by caller if out_q == 0
    }
    const BVNode& n = bv_node(ctx, term);
    if (n.op == BVOp::Mul) {
        if (match_bv_factor(ctx, n.b, factor)) {
            out_q = n.a;
            return true;
        }
        if (match_bv_factor(ctx, n.a, factor)) {
            out_q = n.b;
            return true;
        }
    }
    if (n.op == BVOp::Shl && bv_node(ctx, n.b).op == BVOp::Const && bv_node(ctx, n.b).imm < 30) {
        const BVNode& f = bv_node(ctx, factor);
        if (f.op == BVOp::Const && f.imm == (1ULL << bv_node(ctx, n.b).imm)) {
            out_q = n.a;
            return true;
        }
    }
    return false;
}

bool match_term_div_two(const BVContext& ctx, BVNodeId term, BVNodeId fp, BVNodeId fq, BVNodeId& out_q) noexcept {
    if (match_two_bv_factors(ctx, term, fp, fq)) {
        return true; // caller handles out_q == 0 as const(1)
    }
    const BVNode& n = bv_node(ctx, term);
    if (n.op == BVOp::Mul) {
        if (match_two_bv_factors(ctx, n.b, fp, fq)) {
            out_q = n.a;
            return true;
        }
        if (match_two_bv_factors(ctx, n.a, fp, fq)) {
            out_q = n.b;
            return true;
        }
    }
    return false;
}

bool prove_all_nonzero(BVContext& ctx, BVNodeId active, const std::vector<BVNodeId>& factors, uint32_t max_conflicts) {
    for (BVNodeId f : factors) {
        BVNodeId eq0 = bv_bin(ctx, BVOp::Eq, 1, f, bv_const(ctx, 32, 0));
        if (solve_nonzero(ctx, bv_bin(ctx, BVOp::And, 1, active, eq0), max_conflicts).status != SolveOutcome::Unsat) {
            return false;
        }
    }
    return true;
}

bool try_prove_terms_lt_product(
    BVContext& ctx,
    BVNodeId active,
    const std::vector<BVNodeId>& terms,
    BVNodeId single_e32,
    bool from_add64,
    bool saw_sext,
    const std::vector<BVNodeId>& factors,
    uint32_t max_conflicts,
    int depth) {
    if (depth > 4 || factors.empty()) return false;

    if (terms.empty()) {
        return prove_all_nonzero(ctx, active, factors, max_conflicts);
    }

    if (factors.size() == 1) {
        BVNodeId D = factors[0];
        if (from_add64 && terms.size() > 1) {
            if (saw_sext) return false;
            BVNodeId sum64 = sum_terms_64(ctx, terms);
            BVNodeId d64 = bv_ext(ctx, false, 64, D);
            BVNodeId bad = bv_bin(ctx, BVOp::And, 1, active, bv_bin(ctx, BVOp::Uge, 1, sum64, d64));
            return solve_nonzero(ctx, bad, max_conflicts).status == SolveOutcome::Unsat;
        }
        BVNodeId expr = (terms.size() == 1) ? terms[0] : (single_e32 ? single_e32 : sum_terms_32(ctx, terms));
        const BVNode& en = bv_node(ctx, expr);
        if (en.op == BVOp::URem) {
            BVNodeId denom = en.b;
            BVNodeId bad_mod = bv_bin(ctx, BVOp::Or, 1,
                bv_bin(ctx, BVOp::Eq, 1, denom, bv_const(ctx, 32, 0)),
                bv_bin(ctx, BVOp::Ugt, 1, denom, D));
            if (solve_nonzero(ctx, bv_bin(ctx, BVOp::And, 1, active, bad_mod), max_conflicts).status == SolveOutcome::Unsat) {
                return true;
            }
        }
        BVNodeId bad = bv_bin(ctx, BVOp::And, 1, active, bv_bin(ctx, BVOp::Uge, 1, expr, D));
        return solve_nonzero(ctx, bad, max_conflicts).status == SolveOutcome::Unsat;
    }

    // Inductive case: match a 1-factor or 2-factor stride subset F_lo
    auto try_partition = [&](const std::vector<BVNodeId>& F_lo, const std::vector<BVNodeId>& F_hi) -> bool {
        if (F_lo.empty() || F_hi.empty()) return false;
        std::vector<BVNodeId> q_parts;
        std::vector<BVNodeId> rem_terms;
        for (BVNodeId t : terms) {
            BVNodeId q = 0;
            bool matched = (F_lo.size() == 1)
                ? match_term_div_single(ctx, t, F_lo[0], q)
                : match_term_div_two(ctx, t, F_lo[0], F_lo[1], q);
            if (matched) {
                if (q == 0) q = bv_const(ctx, 32, 1);
                flatten_add32_into(ctx, q, q_parts);
            } else {
                rem_terms.push_back(t);
            }
        }
        if (q_parts.empty()) return false;
        if (from_add64 && q_parts.size() > 1) return false;
        if (!try_prove_terms_lt_product(ctx, active, q_parts, 0, false, false, F_hi, max_conflicts, depth + 1)) {
            return false;
        }
        return try_prove_terms_lt_product(ctx, active, rem_terms, 0, from_add64, saw_sext, F_lo, max_conflicts, depth + 1);
    };

    // Prefer 2-factor strides first when factors.size() >= 3 (e.g. b * (rows * cols))
    if (factors.size() >= 3) {
        for (size_t i = 0; i < factors.size(); ++i) {
            for (size_t j = i + 1; j < factors.size(); ++j) {
                std::vector<BVNodeId> F_lo = {factors[i], factors[j]};
                std::vector<BVNodeId> F_hi;
                for (size_t k = 0; k < factors.size(); ++k) {
                    if (k != i && k != j) F_hi.push_back(factors[k]);
                }
                if (try_partition(F_lo, F_hi)) return true;
            }
        }
    }

    for (size_t i = 0; i < factors.size(); ++i) {
        std::vector<BVNodeId> F_lo = {factors[i]};
        std::vector<BVNodeId> F_hi;
        for (size_t k = 0; k < factors.size(); ++k) {
            if (k != i) F_hi.push_back(factors[k]);
        }
        if (try_partition(F_lo, F_hi)) return true;
    }

    // Fallback when leading index is 0 (e.g. inner loop r = 0 simplifies r * cols + c -> c < rows * cols)
    for (size_t i = 0; i < factors.size(); ++i) {
        BVNodeId hi_gt_0 = bv_bin(ctx, BVOp::Ugt, 1, factors[i], bv_const(ctx, 32, 0));
        if (solve_nonzero(ctx, bv_bin(ctx, BVOp::And, 1, active, bv_un(ctx, BVOp::Not, 1, hi_gt_0)), max_conflicts).status != SolveOutcome::Unsat) {
            continue;
        }
        std::vector<BVNodeId> F_lo;
        for (size_t k = 0; k < factors.size(); ++k) {
            if (k != i) F_lo.push_back(factors[k]);
        }
        if (try_prove_terms_lt_product(ctx, active, terms, single_e32, from_add64, saw_sext, F_lo, max_conflicts, depth + 1)) {
            return true;
        }
    }

    return false;
}

std::vector<uint32_t> collect_buffer_dim_params(const GpuBufferSpec& bspec, const GpuSymEnv& env) {
    std::vector<uint32_t> dims;
    const int32_t raw[3] = {bspec.length_param_index, bspec.length_param_index2, bspec.length_param_index3};
    for (int32_t idx : raw) {
        if (idx >= 0 && static_cast<size_t>(idx) < env.param_vars.size()) {
            dims.push_back(static_cast<uint32_t>(idx));
        }
    }
    return dims;
}

bool try_prove_global_access_algebraic(
    BVContext& ctx,
    BVNodeId active,
    BVNodeId byte_off,
    uint32_t access_bytes,
    const GpuBufferSpec* bspec,
    const GpuSymEnv& env,
    uint32_t max_conflicts) {
    if (!bspec || bspec->static_byte_size > 0 || bspec->elem_bytes == 0 || access_bytes != bspec->elem_bytes) {
        return false;
    }
    std::vector<uint32_t> dims = collect_buffer_dim_params(*bspec, env);
    if (dims.empty()) return false;

    std::vector<BVNodeId> cap_factors;
    cap_factors.reserve(dims.size() + 1);
    for (uint32_t idx : dims) {
        cap_factors.push_back(gpu_coerce_width(ctx, env.param_vars[idx], 32, false));
    }
    if (bspec->elem_scale > 1) {
        cap_factors.push_back(bv_const(ctx, 32, bspec->elem_scale));
    }

    std::vector<BVNodeId> terms32;
    BVNodeId single32 = 0;
    bool from_add64 = false;
    bool saw_sext = false;
    if (!peel_byte_offset_64(ctx, byte_off, bspec->elem_bytes, terms32, single32, from_add64, saw_sext)) {
        return false;
    }
    return try_prove_terms_lt_product(ctx, active, terms32, single32, from_add64, saw_sext, cap_factors, max_conflicts, 0);
}

BVNodeId build_buffer_capacity_64(BVContext& ctx, const GpuBufferSpec& bspec, const GpuSymEnv& env) {
    if (bspec.static_byte_size > 0) {
        return bv_const(ctx, 64, bspec.static_byte_size);
    }
    std::vector<uint32_t> dims = collect_buffer_dim_params(bspec, env);
    if (dims.empty()) return 0;

    BVNodeId cap = bv_const(ctx, 64, std::max(1u, bspec.elem_scale));
    for (uint32_t idx : dims) {
        BVNodeId l64 = gpu_coerce_width(ctx, env.param_vars[idx], 64, false);
        cap = (bv_node(ctx, cap).op == BVOp::Const && bv_node(ctx, cap).imm == 1)
            ? l64
            : bv_bin(ctx, BVOp::Mul, 64, cap, l64);
    }
    return gpu_mul_or_shl(ctx, 64, cap, bv_const(ctx, 64, std::max(1u, bspec.elem_bytes)));
}

} // namespace

GpuMemorySafetyResult check_trace_memory_safety(
    BVContext& ctx,
    const GpuSymEnv& env,
    const GpuExecTrace& trace,
    const GpuProverOptions& opts,
    bool has_static_shared) {
    GpuMemorySafetyResult res;
    if (!trace.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = trace.error_msg;
        return res;
    }

    BVNodeId domain = bv_bin(ctx, BVOp::And, 1, env.coord_bounds,
        bv_bin(ctx, BVOp::And, 1, trace.completed, trace.domain_assumption));

    // Apply caller parameter constraints (e.g. d >= 1, width <= pitch)
    for (const auto& c : opts.param_constraints) {
        if (c.param_index >= env.param_vars.size()) continue;
        BVNodeId lhs = env.param_vars[c.param_index];
        uint32_t lw = ctx.width(lhs);
        if (c.min_val > 0) {
            domain = bv_bin(ctx, BVOp::And, 1, domain,
                bv_bin(ctx, BVOp::Uge, 1, lhs, bv_const(ctx, lw, c.min_val)));
        }
        if (c.max_val > 0) {
            domain = bv_bin(ctx, BVOp::And, 1, domain,
                bv_bin(ctx, BVOp::Ule, 1, lhs, bv_const(ctx, lw, c.max_val)));
        }
        if (c.rhs_param_index != UINT32_MAX && c.rhs_param_index < env.param_vars.size()) {
            BVNodeId rhs = gpu_coerce_width(ctx, env.param_vars[c.rhs_param_index], lw, false);
            domain = bv_bin(ctx, BVOp::And, 1, domain, bv_bin(ctx, BVOp::Ule, 1, lhs, rhs));
        }
    }

    // Constrain dynamic length parameters so their product * elem_scale fits in 30 bits (no multipliers in domain)
    for (const auto& bspec : opts.buffers) {
        std::vector<uint32_t> dims = collect_buffer_dim_params(bspec, env);
        uint32_t scale = std::max(1u, bspec.elem_scale);
        auto bound_dim = [&](uint32_t idx, uint64_t hi) {
            BVNodeId lvar = env.param_vars[idx];
            uint32_t lw = ctx.width(lvar);
            domain = bv_bin(ctx, BVOp::And, 1, domain,
                bv_bin(ctx, BVOp::Ule, 1, lvar, bv_const(ctx, lw, hi)));
        };
        if (dims.size() == 1) {
            bound_dim(dims[0], 0x3FFFFFFFULL / scale);
        } else if (dims.size() == 2) {
            bound_dim(dims[0], 0x7FFFULL);
            bound_dim(dims[1], 0x7FFFULL / scale);
        } else if (dims.size() >= 3) {
            bound_dim(dims[0], 0x3FFULL);
            bound_dim(dims[1], 0x3FFULL);
            bound_dim(dims[2], 0x3FFULL / scale);
        }
    }

    auto find_buffer_spec = [&](uint32_t param_idx) -> const GpuBufferSpec* {
        for (const auto& b : opts.buffers) {
            if (b.param_index == param_idx) return &b;
        }
        if (opts.buffers.size() == 1 && param_idx == UINT32_MAX) {
            return &opts.buffers[0];
        }
        return nullptr;
    };

    for (const auto& acc : trace.accesses) {
        if (!acc.is_shared && !opts.check_global_bounds) {
            continue;
        }
        BVNodeId active = bv_bin(ctx, BVOp::And, 1, domain, acc.guard);
        if (bv_node(ctx, active).op == BVOp::Const && bv_node(ctx, active).imm == 0) {
            continue;
        }

        // 1. Check natural alignment if required_align > 1
        if (acc.required_align > 1) {
            if (!acc.is_shared && !acc.addr_resolved) {
                if (opts.check_global_bounds && res.status == EquivStatus::ProvenEquivalent) {
                    res.status = EquivStatus::Unknown;
                    if (res.diagnostic.empty()) res.diagnostic = "Unresolved global pointer alignment";
                }
            } else {
                const GpuBufferSpec* align_spec = acc.is_shared ? nullptr : find_buffer_spec(acc.region_index);
                if (!acc.is_shared && opts.check_global_bounds && align_spec &&
                    align_spec->base_alignment < acc.required_align) {
                    if (res.status == EquivStatus::ProvenEquivalent) {
                        res.status = EquivStatus::Unknown;
                        if (res.diagnostic.empty()) res.diagnostic = "Buffer base alignment below access alignment";
                    }
                } else {
                    BVNodeId off64 = acc.byte_offset_64 ? acc.byte_offset_64 : bv_const(ctx, 64, 0);
                    uint64_t mask = static_cast<uint64_t>(acc.required_align - 1);
                    BVNodeId misalign = bv_bin(ctx, BVOp::Ne, 1,
                        bv_bin(ctx, BVOp::And, 64, off64, bv_const(ctx, 64, mask)),
                        bv_const(ctx, 64, 0));
                    BVNodeId bad_align = bv_bin(ctx, BVOp::And, 1, active, misalign);
                    SolveOutcome out = solve_nonzero(ctx, bad_align, opts.max_sat_conflicts);
                    if (out.status == SolveOutcome::Unsat) {
                        ++res.alignments_proven;
                    } else if (out.status == SolveOutcome::Sat) {
                        std::vector<uint64_t> nvals;
                        ctx.evaluate_all(out.model, nvals);
                        auto mv = [&](BVNodeId id) { return id < nvals.size() ? nvals[id] : 0ULL; };
                        GpuAccessViolation v;
                        v.is_shared = acc.is_shared;
                        v.is_alignment = true;
                        v.region_index = acc.region_index == UINT32_MAX ? 0 : acc.region_index;
                        v.tid_x = static_cast<uint32_t>(mv(env.tid[0]));
                        v.tid_y = static_cast<uint32_t>(mv(env.tid[1]));
                        v.tid_z = static_cast<uint32_t>(mv(env.tid[2]));
                        v.ctaid_x = static_cast<uint32_t>(mv(env.ctaid[0]));
                        v.ctaid_y = static_cast<uint32_t>(mv(env.ctaid[1]));
                        v.ctaid_z = static_cast<uint32_t>(mv(env.ctaid[2]));
                        v.offending_index_or_offset = static_cast<int64_t>(mv(off64));
                        v.bound_limit = acc.required_align;
                        std::ostringstream oss;
                        oss << "Misaligned " << (acc.is_shared ? "shared" : "global")
                            << " access: byte_offset=" << v.offending_index_or_offset
                            << " not aligned to " << acc.required_align << " bytes at tid=("
                            << v.tid_x << "," << v.tid_y << "," << v.tid_z << ")";
                        v.description = oss.str();
                        res.violations.push_back(std::move(v));
                        res.status = EquivStatus::Counterexample;
                    } else if (res.status == EquivStatus::ProvenEquivalent) {
                        res.status = EquivStatus::Unknown;
                    }
                }
            }
        } else {
            ++res.alignments_proven;
        }

        // 2. Check bounds
        if (acc.is_shared) {
            uint32_t count = acc.shared_elem_count ? acc.shared_elem_count : 256u;
            BVNodeId oob = bv_bin(ctx, BVOp::Uge, 1, acc.elem_index_32, bv_const(ctx, 32, count));
            BVNodeId bad = bv_bin(ctx, BVOp::And, 1, active, oob);
            SolveOutcome out = solve_nonzero(ctx, bad, opts.max_sat_conflicts);
            if (out.status == SolveOutcome::Unsat) {
                ++res.shared_accesses_proven;
            } else if (out.status == SolveOutcome::Sat) {
                std::vector<uint64_t> nvals;
                ctx.evaluate_all(out.model, nvals);
                auto mv = [&](BVNodeId id) { return id < nvals.size() ? nvals[id] : 0ULL; };
                GpuAccessViolation v;
                v.is_shared = true;
                v.is_alignment = false;
                v.region_index = acc.region_index;
                v.tid_x = static_cast<uint32_t>(mv(env.tid[0]));
                v.tid_y = static_cast<uint32_t>(mv(env.tid[1]));
                v.tid_z = static_cast<uint32_t>(mv(env.tid[2]));
                v.ctaid_x = static_cast<uint32_t>(mv(env.ctaid[0]));
                v.ctaid_y = static_cast<uint32_t>(mv(env.ctaid[1]));
                v.ctaid_z = static_cast<uint32_t>(mv(env.ctaid[2]));
                v.offending_index_or_offset = static_cast<int32_t>(static_cast<uint32_t>(mv(acc.elem_index_32)));
                v.bound_limit = count;
                std::ostringstream oss;
                oss << "Shared memory OOB on array " << v.region_index
                    << ": index=" << v.offending_index_or_offset << " (limit=" << count
                    << ") at tid=(" << v.tid_x << "," << v.tid_y << "," << v.tid_z << ")";
                v.description = oss.str();
                res.violations.push_back(std::move(v));
                res.status = EquivStatus::Counterexample;
            } else if (res.status == EquivStatus::ProvenEquivalent) {
                res.status = EquivStatus::Unknown;
            }
        } else {
            if (!acc.addr_resolved) {
                if (opts.check_global_bounds && res.status == EquivStatus::ProvenEquivalent) {
                    res.status = EquivStatus::Unknown;
                    if (res.diagnostic.empty()) res.diagnostic = "Unresolved global pointer base in bounds check";
                }
                continue;
            }
            const GpuBufferSpec* bspec = find_buffer_spec(acc.region_index);
            BVNodeId byte_off = acc.byte_offset_64 ? acc.byte_offset_64 : bv_const(ctx, 64, 0);
            if (try_prove_global_access_algebraic(
                    ctx, active, byte_off, acc.access_bytes, bspec, env, opts.max_sat_conflicts)) {
                ++res.global_accesses_proven;
                continue;
            }

            BVNodeId neg_off = bv_bin(ctx, BVOp::Slt, 1, byte_off, bv_const(ctx, 64, 0));
            BVNodeId oob = neg_off;
            BVNodeId cap_node = 0;

            if (bspec) {
                cap_node = build_buffer_capacity_64(ctx, *bspec, env);
                if (cap_node != 0) {
                    BVNodeId end_off = bv_bin(ctx, BVOp::Add, 64, byte_off, bv_const(ctx, 64, acc.access_bytes));
                    BVNodeId past_end = bv_bin(ctx, BVOp::Ugt, 1, end_off, cap_node);
                    BVNodeId start_past = bv_bin(ctx, BVOp::Uge, 1, byte_off, cap_node);
                    oob = bv_bin(ctx, BVOp::Or, 1, oob, bv_bin(ctx, BVOp::Or, 1, past_end, start_past));
                }
            }

            BVNodeId bad = bv_bin(ctx, BVOp::And, 1, active, oob);
            SolveOutcome out = solve_nonzero(ctx, bad, opts.max_sat_conflicts);
            if (out.status == SolveOutcome::Unsat) {
                if (cap_node == 0 && opts.check_global_bounds) {
                    if (res.status == EquivStatus::ProvenEquivalent) {
                        res.status = EquivStatus::Unknown;
                        if (res.diagnostic.empty()) {
                            res.diagnostic = "Missing GpuBufferSpec capacity for parameter " +
                                             std::to_string(acc.region_index);
                        }
                    }
                } else {
                    ++res.global_accesses_proven;
                }
            } else if (out.status == SolveOutcome::Sat) {
                std::vector<uint64_t> nvals;
                ctx.evaluate_all(out.model, nvals);
                auto mv = [&](BVNodeId id) { return id < nvals.size() ? nvals[id] : 0ULL; };
                GpuAccessViolation v;
                v.is_shared = false;
                v.is_alignment = false;
                v.region_index = acc.region_index == UINT32_MAX ? 0 : acc.region_index;
                v.tid_x = static_cast<uint32_t>(mv(env.tid[0]));
                v.tid_y = static_cast<uint32_t>(mv(env.tid[1]));
                v.tid_z = static_cast<uint32_t>(mv(env.tid[2]));
                v.ctaid_x = static_cast<uint32_t>(mv(env.ctaid[0]));
                v.ctaid_y = static_cast<uint32_t>(mv(env.ctaid[1]));
                v.ctaid_z = static_cast<uint32_t>(mv(env.ctaid[2]));
                v.offending_index_or_offset = static_cast<int64_t>(mv(byte_off));
                v.bound_limit = cap_node ? mv(cap_node) : 0;
                std::ostringstream oss;
                oss << "Global buffer OOB on param " << v.region_index
                    << ": byte_offset=" << v.offending_index_or_offset
                    << " + " << acc.access_bytes << " > capacity=" << v.bound_limit
                    << " at tid=(" << v.tid_x << "," << v.tid_y << "," << v.tid_z
                    << ") ctaid=(" << v.ctaid_x << "," << v.ctaid_y << "," << v.ctaid_z << ")";
                v.description = oss.str();
                res.violations.push_back(std::move(v));
                res.status = EquivStatus::Counterexample;
            } else if (res.status == EquivStatus::ProvenEquivalent) {
                res.status = EquivStatus::Unknown;
            }
        }
    }

    if ((opts.check_global_bounds || has_static_shared) &&
        res.status == EquivStatus::ProvenEquivalent &&
        !(bv_node(ctx, trace.completed).op == BVOp::Const && bv_node(ctx, trace.completed).imm == 1)) {
        BVNodeId trunc = bv_bin(ctx, BVOp::And, 1, env.coord_bounds,
            bv_bin(ctx, BVOp::And, 1, trace.domain_assumption,
                   bv_un(ctx, BVOp::Not, 1, trace.completed)));
        if (solve_nonzero(ctx, trunc, opts.max_sat_conflicts).status != SolveOutcome::Unsat) {
            res.status = EquivStatus::Unknown;
            res.diagnostic = "GPU memory safety proof truncated at loop_unroll_bound";
        }
    }

    if (!res.violations.empty()) {
        res.diagnostic = res.violations.front().description;
    }
    return res;
}

} // namespace brass::verify
