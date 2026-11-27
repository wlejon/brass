#include "gpu_mem_safety_internal.hpp"
#include <algorithm>
#include <bit>
#include <cstdint>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass::verify {


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

    BVNodeId light_domain = env.coord_bounds;

    // Apply caller parameter constraints (e.g. d >= 1, width <= pitch, n % d == 0)
    std::vector<DivConstraint> div_pairs;
    std::vector<BVNodeId> sym_div_guards;
    for (const auto& c : opts.param_constraints) {
        if (c.param_index >= env.param_vars.size()) continue;
        BVNodeId lhs = env.param_vars[c.param_index];
        uint32_t lw = ctx.width(lhs);
        if (c.min_val > 0) {
            light_domain = bv_bin(ctx, BVOp::And, 1, light_domain,
                bv_bin(ctx, BVOp::Uge, 1, lhs, bv_const(ctx, lw, c.min_val)));
        }
        if (c.max_val > 0) {
            light_domain = bv_bin(ctx, BVOp::And, 1, light_domain,
                bv_bin(ctx, BVOp::Ule, 1, lhs, bv_const(ctx, lw, c.max_val)));
        }
        if (c.rhs_param_index != UINT32_MAX && c.rhs_param_index < env.param_vars.size()) {
            BVNodeId rhs = gpu_coerce_width(ctx, env.param_vars[c.rhs_param_index], lw, false);
            light_domain = bv_bin(ctx, BVOp::And, 1, light_domain, bv_bin(ctx, BVOp::Ule, 1, lhs, rhs));
        }
        if (c.divisor_param != UINT32_MAX && c.divisor_param < env.param_vars.size()) {
            BVNodeId rhs32 = gpu_coerce_width(ctx, env.param_vars[c.divisor_param], 32, false);
            BVNodeId lhs32 = gpu_coerce_width(ctx, lhs, 32, false);
            light_domain = bv_bin(ctx, BVOp::And, 1, light_domain, bv_bin(ctx, BVOp::Ne, 1, rhs32, bv_const(ctx, 32, 0)));
            div_pairs.push_back({lhs32, rhs32});
            BVNodeId rhs_lw = gpu_coerce_width(ctx, env.param_vars[c.divisor_param], lw, false);
            sym_div_guards.push_back(
                bv_bin(ctx, BVOp::Eq, 1, bv_bin(ctx, BVOp::URem, lw, lhs, rhs_lw), bv_const(ctx, lw, 0)));
        }
        if (c.divisor_const > 1) {
            BVNodeId lhs32 = gpu_coerce_width(ctx, lhs, 32, false);
            div_pairs.push_back({lhs32, bv_const(ctx, 32, c.divisor_const)});
            if (std::has_single_bit(c.divisor_const)) {
                light_domain = bv_bin(ctx, BVOp::And, 1, light_domain,
                    bv_bin(ctx, BVOp::Eq, 1,
                           bv_bin(ctx, BVOp::And, lw, lhs, bv_const(ctx, lw, c.divisor_const - 1)),
                           bv_const(ctx, lw, 0)));
            } else {
                sym_div_guards.push_back(
                    bv_bin(ctx, BVOp::Eq, 1,
                           bv_bin(ctx, BVOp::URem, lw, lhs, bv_const(ctx, lw, c.divisor_const)),
                           bv_const(ctx, lw, 0)));
            }
        }
    }

    auto with_div_guards = [&](BVNodeId base_cond) -> BVNodeId {
        BVNodeId full = base_cond;
        for (BVNodeId dg : sym_div_guards) full = bv_bin(ctx, BVOp::And, 1, full, dg);
        return full;
    };

    // Constrain dynamic length parameters so their product * elem_scale fits in 30 bits (no multipliers in domain)
    for (const auto& bspec : opts.buffers) {
        std::vector<uint32_t> dims = collect_buffer_dim_params(bspec, env);
        uint32_t scale = std::max(1u, bspec.elem_scale);
        auto bound_dim = [&](uint32_t idx, uint64_t hi) {
            BVNodeId lvar = env.param_vars[idx];
            uint32_t lw = ctx.width(lvar);
            light_domain = bv_bin(ctx, BVOp::And, 1, light_domain,
                bv_bin(ctx, BVOp::Ule, 1, lvar, bv_const(ctx, lw, hi)));
        };
        if (dims.size() == 1) {
            bound_dim(dims[0], kMaxProverDim1D / scale);
        } else if (dims.size() == 2) {
            bound_dim(dims[0], kMaxProverDim2D);
            bound_dim(dims[1], kMaxProverDim2D / scale);
        } else if (dims.size() >= 3) {
            bound_dim(dims[0], kMaxProverDim3D);
            bound_dim(dims[1], kMaxProverDim3D);
            bound_dim(dims[2], kMaxProverDim3D / scale);
        }
    }

    BVNodeId full_domain = bv_bin(ctx, BVOp::And, 1, light_domain,
        bv_bin(ctx, BVOp::And, 1, trace.completed, trace.domain_assumption));

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
        BVNodeId guard_domain = bv_bin(ctx, BVOp::And, 1, light_domain, acc.guard);
        if (bv_node(ctx, guard_domain).op == BVOp::Const && bv_node(ctx, guard_domain).imm == 0) {
            continue;
        }
        BVNodeId active = bv_bin(ctx, BVOp::And, 1, full_domain, acc.guard);
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
                    SolveOutcome out = solve_nonzero(
                        ctx, bv_bin(ctx, BVOp::And, 1, light_domain, misalign), opts.max_sat_conflicts);
                    if (out.status != SolveOutcome::Unsat && guard_domain != light_domain) {
                        out = solve_nonzero(
                            ctx, bv_bin(ctx, BVOp::And, 1, guard_domain, misalign), opts.max_sat_conflicts);
                    }
                    if (out.status != SolveOutcome::Unsat && active != guard_domain) {
                        out = solve_nonzero(
                            ctx, bv_bin(ctx, BVOp::And, 1, active, misalign), opts.max_sat_conflicts);
                    }
                    if (out.status == SolveOutcome::Sat && !sym_div_guards.empty()) {
                        BVNodeId guarded_bad = bv_bin(ctx, BVOp::And, 1, with_div_guards(active), misalign);
                        out = solve_nonzero(ctx, guarded_bad, opts.max_sat_conflicts);
                    }
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
            SolveOutcome out = solve_nonzero(
                ctx, bv_bin(ctx, BVOp::And, 1, light_domain, oob), opts.max_sat_conflicts);
            if (out.status != SolveOutcome::Unsat && guard_domain != light_domain) {
                out = solve_nonzero(
                    ctx, bv_bin(ctx, BVOp::And, 1, guard_domain, oob), opts.max_sat_conflicts);
            }
            if (out.status != SolveOutcome::Unsat && active != guard_domain) {
                out = solve_nonzero(
                    ctx, bv_bin(ctx, BVOp::And, 1, active, oob), opts.max_sat_conflicts);
            }
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
                    ctx, light_domain, guard_domain, active, byte_off, acc.access_bytes, bspec, env, div_pairs, opts.max_sat_conflicts)) {
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

            SolveOutcome out = solve_nonzero(
                ctx, bv_bin(ctx, BVOp::And, 1, with_div_guards(guard_domain), oob), opts.max_sat_conflicts);
            if (out.status != SolveOutcome::Unsat && active != guard_domain) {
                out = solve_nonzero(
                    ctx, bv_bin(ctx, BVOp::And, 1, with_div_guards(active), oob), opts.max_sat_conflicts);
            }
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
