#include "gpu_equiv_internal.hpp"
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/verify/gpu_kernel_prover.h>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <unordered_map>

namespace brass::verify {

namespace {

GpuMemorySafetyResult check_trace_memory_safety(
    BVContext& ctx,
    const GpuSymEnv& env,
    const GpuExecTrace& trace,
    const GpuProverOptions& opts) {
    GpuMemorySafetyResult res;
    if (!trace.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = trace.error_msg;
        return res;
    }

    BVNodeId domain = bv_bin(ctx, BVOp::And, 1, env.coord_bounds,
        bv_bin(ctx, BVOp::And, 1, trace.completed, trace.domain_assumption));

    // Constrain any dynamic length_param_index parameters to non-negative 30-bit counts
    for (const auto& bspec : opts.buffers) {
        if (bspec.length_param_index >= 0 &&
            static_cast<size_t>(bspec.length_param_index) < env.param_vars.size()) {
            BVNodeId lvar = env.param_vars[static_cast<size_t>(bspec.length_param_index)];
            uint32_t lw = ctx.width(lvar);
            domain = bv_bin(ctx, BVOp::And, 1, domain,
                bv_bin(ctx, BVOp::Ule, 1, lvar, bv_const(ctx, lw, 0x3FFFFFFFULL)));
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
        BVNodeId active = bv_bin(ctx, BVOp::And, 1, domain, acc.guard);
        if (bv_node(ctx, active).op == BVOp::Const && bv_node(ctx, active).imm == 0) {
            continue;
        }

        // 1. Check natural alignment if required_align > 1
        if (acc.required_align > 1 && acc.byte_offset_64 != 0) {
            uint64_t mask = static_cast<uint64_t>(acc.required_align - 1);
            BVNodeId misalign = bv_bin(ctx, BVOp::Ne, 1,
                bv_bin(ctx, BVOp::And, 64, acc.byte_offset_64, bv_const(ctx, 64, mask)),
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
                v.offending_index_or_offset = static_cast<int64_t>(mv(acc.byte_offset_64));
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
            const GpuBufferSpec* bspec = find_buffer_spec(acc.region_index);
            BVNodeId byte_off = acc.byte_offset_64 ? acc.byte_offset_64 : bv_const(ctx, 64, 0);
            BVNodeId neg_off = bv_bin(ctx, BVOp::Slt, 1, byte_off, bv_const(ctx, 64, 0));
            BVNodeId oob = neg_off;
            BVNodeId cap_node = 0;

            if (bspec) {
                if (bspec->static_byte_size > 0) {
                    cap_node = bv_const(ctx, 64, bspec->static_byte_size);
                } else if (bspec->length_param_index >= 0 &&
                           static_cast<size_t>(bspec->length_param_index) < env.param_vars.size()) {
                    BVNodeId lvar = env.param_vars[static_cast<size_t>(bspec->length_param_index)];
                    BVNodeId l64 = gpu_coerce_width(ctx, lvar, 64, false);
                    cap_node = gpu_mul_or_shl(ctx, 64, l64, bv_const(ctx, 64, bspec->elem_bytes));
                }
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
                ++res.global_accesses_proven;
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

    if (!res.violations.empty()) {
        res.diagnostic = res.violations.front().description;
    }
    return res;
}

void setup_two_thread_envs(
    BVContext& ctx,
    GpuSymEnv& env_a,
    GpuSymEnv& env_b,
    const std::vector<Type>& param_types,
    const GpuProverOptions& opts,
    BVNodeId& out_same_subgroup) {
    uint32_t fixed_ntid[3] = {
        std::max(1u, opts.local_size_x),
        std::max(1u, opts.local_size_y),
        std::max(1u, opts.local_size_z)};
    uint32_t max_cta[3] = {
        std::max(1u, opts.grid_size_x),
        std::max(1u, opts.grid_size_y),
        std::max(1u, opts.grid_size_z)};

    init_gpu_sym_env(ctx, env_a, param_types, "_a", fixed_ntid, fixed_ntid, max_cta);
    init_gpu_sym_env(ctx, env_b, param_types, "_b", fixed_ntid, fixed_ntid, max_cta);

    // Share kernel parameters and workgroup coordinates (ctaid, ntid, nctaid)
    env_b.param_vars = env_a.param_vars;
    env_b.param_types = env_a.param_types;
    for (int d = 0; d < 3; ++d) {
        env_b.ctaid[d] = env_a.ctaid[d];
        env_b.ntid[d] = env_a.ntid[d];
        env_b.nctaid[d] = env_a.nctaid[d];
        env_b.global_tid[d] = bv_bin(
            ctx, BVOp::Add, 32,
            bv_bin(ctx, BVOp::Mul, 32, env_b.ctaid[d], env_b.ntid[d]),
            env_b.tid[d]);
    }

    // Bind raw_subgroup_lane and subgroup ID to linear local thread ID
    auto compute_linear_tid = [&](const GpuSymEnv& e) -> BVNodeId {
        BVNodeId lin = e.tid[0];
        if (fixed_ntid[1] > 1) {
            lin = bv_bin(ctx, BVOp::Add, 32, lin,
                gpu_mul_or_shl(ctx, 32, e.tid[1], bv_const(ctx, 32, fixed_ntid[0])));
        }
        if (fixed_ntid[2] > 1) {
            lin = bv_bin(ctx, BVOp::Add, 32, lin,
                gpu_mul_or_shl(ctx, 32, e.tid[2], bv_const(ctx, 32, fixed_ntid[0] * fixed_ntid[1])));
        }
        return lin;
    };

    uint32_t sg_size = std::max(1u, opts.subgroup_size);
    BVNodeId lin_a = compute_linear_tid(env_a);
    BVNodeId lin_b = compute_linear_tid(env_b);

    BVNodeId sg_id_a = bv_bin(ctx, BVOp::UDiv, 32, lin_a, bv_const(ctx, 32, sg_size));
    BVNodeId sg_id_b = bv_bin(ctx, BVOp::UDiv, 32, lin_b, bv_const(ctx, 32, sg_size));
    out_same_subgroup = bv_bin(ctx, BVOp::Eq, 1, sg_id_a, sg_id_b);

    auto bind_lane = [&](GpuSymEnv& e, BVNodeId lin) {
        e.raw_subgroup_lane = bv_bin(ctx, BVOp::URem, 32, lin, bv_const(ctx, 32, sg_size));
        e.laneid = bv_bin(ctx, BVOp::And, 32, e.raw_subgroup_lane, bv_const(ctx, 32, 31));
        e.lanebase = bv_bin(ctx, BVOp::And, 32, e.raw_subgroup_lane, bv_const(ctx, 32, ~31u));
    };
    bind_lane(env_a, lin_a);
    bind_lane(env_b, lin_b);
}

BarrierUniformityResult check_traces_barrier_uniformity(
    BVContext& ctx,
    const GpuSymEnv& env_a,
    const GpuSymEnv& env_b,
    const GpuExecTrace& trace_a,
    const GpuExecTrace& trace_b,
    BVNodeId same_subgroup,
    bool compare_barrier_id,
    const GpuProverOptions& opts) {
    BarrierUniformityResult res;
    if (!trace_a.supported || !trace_b.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = !trace_a.supported ? trace_a.error_msg : trace_b.error_msg;
        return res;
    }

    BVNodeId domain = bv_bin(ctx, BVOp::And, 1,
        bv_bin(ctx, BVOp::And, 1, env_a.coord_bounds, env_b.coord_bounds),
        bv_bin(ctx, BVOp::And, 1,
            bv_bin(ctx, BVOp::And, 1, trace_a.completed, trace_b.completed),
            bv_bin(ctx, BVOp::And, 1, trace_a.domain_assumption, trace_b.domain_assumption)));

    size_t n = std::min(trace_a.barriers.size(), trace_b.barriers.size());
    for (size_t i = 0; i < n; ++i) {
        const auto& ba = trace_a.barriers[i];
        const auto& bb = trace_b.barriers[i];
        BVNodeId cond_diff = bv_bin(ctx, BVOp::Xor, 1, ba.reach_cond, bb.reach_cond);
        if (compare_barrier_id && !ba.is_subgroup_collective) {
            BVNodeId id_diff = bv_bin(ctx, BVOp::And, 1, ba.reach_cond,
                bv_bin(ctx, BVOp::Ne, 1, ba.barrier_id, bb.barrier_id));
            cond_diff = bv_bin(ctx, BVOp::Or, 1, cond_diff, id_diff);
        }
        BVNodeId scope_guard = ba.is_subgroup_collective
            ? bv_bin(ctx, BVOp::And, 1, domain, same_subgroup)
            : domain;
        BVNodeId bad = bv_bin(ctx, BVOp::And, 1, scope_guard, cond_diff);

        SolveOutcome out = solve_nonzero(ctx, bad, opts.max_sat_conflicts);
        if (out.status == SolveOutcome::Unsat) {
            if (ba.is_subgroup_collective) ++res.collectives_proven;
            else ++res.barriers_proven;
        } else if (out.status == SolveOutcome::Sat) {
            std::vector<uint64_t> nvals;
            ctx.evaluate_all(out.model, nvals);
            auto mv = [&](BVNodeId id) { return id < nvals.size() ? static_cast<uint32_t>(nvals[id]) : 0u; };
            BarrierDivergence div;
            div.is_subgroup_collective = ba.is_subgroup_collective;
            div.inst_index = ba.inst_index;
            for (int d = 0; d < 3; ++d) {
                div.local_id_a[d] = mv(env_a.tid[d]);
                div.local_id_b[d] = mv(env_b.tid[d]);
            }
            std::ostringstream oss;
            oss << (div.is_subgroup_collective ? "Subgroup collective" : "Workgroup barrier")
                << " divergence at barrier #" << div.inst_index
                << ": local_id_a=(" << div.local_id_a[0] << "," << div.local_id_a[1] << "," << div.local_id_a[2]
                << ") vs local_id_b=(" << div.local_id_b[0] << "," << div.local_id_b[1] << "," << div.local_id_b[2] << ")";
            div.description = oss.str();
            res.divergences.push_back(std::move(div));
            res.status = EquivStatus::Counterexample;
        } else if (res.status == EquivStatus::ProvenEquivalent) {
            res.status = EquivStatus::Unknown;
        }
    }

    if (!res.divergences.empty()) {
        res.diagnostic = res.divergences.front().description;
    }
    return res;
}

} // namespace

GpuMemorySafetyResult verify_gpu_memory_safety(
    const Function& mir_fn,
    const GpuProverOptions& opts) {
    BVContext ctx;
    GpuSymEnv env;
    uint32_t fixed_ntid[3] = {
        std::max(1u, opts.local_size_x),
        std::max(1u, opts.local_size_y),
        std::max(1u, opts.local_size_z)};
    uint32_t max_cta[3] = {
        std::max(1u, opts.grid_size_x),
        std::max(1u, opts.grid_size_y),
        std::max(1u, opts.grid_size_z)};
    init_gpu_sym_env(ctx, env, mir_fn.param_types(), "", fixed_ntid, fixed_ntid, max_cta);

    EquivOptions eq_opts;
    eq_opts.loop_unroll_bound = opts.loop_unroll_bound;
    eq_opts.max_sat_conflicts = opts.max_sat_conflicts;
    GpuExecTrace trace = execute_gpu_mir(ctx, mir_fn, env, eq_opts);
    return check_trace_memory_safety(ctx, env, trace, opts);
}

GpuMemorySafetyResult verify_spirv_memory_safety(
    const spirv::Module& spv_mod,
    const GpuProverOptions& opts) {
    BVContext ctx;
    GpuSymEnv env;
    uint32_t fixed_ntid[3] = {
        std::max(1u, opts.local_size_x),
        std::max(1u, opts.local_size_y),
        std::max(1u, opts.local_size_z)};
    uint32_t max_cta[3] = {
        std::max(1u, opts.grid_size_x),
        std::max(1u, opts.grid_size_y),
        std::max(1u, opts.grid_size_z)};
    init_gpu_sym_env(ctx, env, {}, "", fixed_ntid, fixed_ntid, max_cta);

    EquivOptions eq_opts;
    eq_opts.loop_unroll_bound = opts.loop_unroll_bound;
    eq_opts.max_sat_conflicts = opts.max_sat_conflicts;
    GpuExecTrace trace = execute_gpu_spirv(ctx, spv_mod, env, eq_opts);
    if (!trace.supported) {
        spirv::ComputeShaderAuditReport rep = spirv::audit_compute_shader(spv_mod);
        GpuMemorySafetyResult res;
        if (!rep.parse_ok || !rep.structural_ok || !rep.dominance_ok) {
            res.status = EquivStatus::Unsupported;
            res.diagnostic = trace.error_msg;
            return res;
        }
        for (const auto& sa : rep.shared_accesses) {
            if (sa.proven_in_bounds) {
                ++res.shared_accesses_proven;
            } else {
                GpuAccessViolation v;
                v.is_shared = true;
                v.is_alignment = false;
                v.region_index = sa.variable;
                v.bound_limit = sa.array_length;
                v.description = sa.detail;
                res.violations.push_back(std::move(v));
                res.status = EquivStatus::Counterexample;
            }
        }
        if (!res.violations.empty()) {
            res.diagnostic = res.violations.front().description;
        }
        return res;
    }
    return check_trace_memory_safety(ctx, env, trace, opts);
}

GpuMemorySafetyResult verify_ptx_memory_safety(
    const ptx::Function& ptx_fn,
    const GpuProverOptions& opts) {
    BVContext ctx;
    GpuSymEnv env;
    uint32_t fixed_ntid[3] = {
        std::max(1u, opts.local_size_x),
        std::max(1u, opts.local_size_y),
        std::max(1u, opts.local_size_z)};
    uint32_t max_cta[3] = {
        std::max(1u, opts.grid_size_x),
        std::max(1u, opts.grid_size_y),
        std::max(1u, opts.grid_size_z)};
    init_gpu_sym_env(ctx, env, {}, "", fixed_ntid, fixed_ntid, max_cta);

    EquivOptions eq_opts;
    eq_opts.loop_unroll_bound = opts.loop_unroll_bound;
    eq_opts.max_sat_conflicts = opts.max_sat_conflicts;
    GpuExecTrace trace = execute_gpu_ptx(ctx, ptx_fn, env, eq_opts);
    return check_trace_memory_safety(ctx, env, trace, opts);
}

BarrierUniformityResult verify_gpu_barrier_uniformity(
    const Function& mir_fn,
    const GpuProverOptions& opts) {
    BVContext ctx;
    GpuSymEnv env_a, env_b;
    BVNodeId same_sg = 0;
    setup_two_thread_envs(ctx, env_a, env_b, mir_fn.param_types(), opts, same_sg);

    EquivOptions eq_opts;
    eq_opts.loop_unroll_bound = opts.loop_unroll_bound;
    eq_opts.max_sat_conflicts = opts.max_sat_conflicts;
    GpuExecTrace trace_a = execute_gpu_mir(ctx, mir_fn, env_a, eq_opts);
    env_b.uf_calls = env_a.uf_calls;
    env_b.global_init_bytes = env_a.global_init_bytes;
    env_b.shared_init_bytes = env_a.shared_init_bytes;
    GpuExecTrace trace_b = execute_gpu_mir(ctx, mir_fn, env_b, eq_opts);
    return check_traces_barrier_uniformity(ctx, env_a, env_b, trace_a, trace_b, same_sg, true, opts);
}

BarrierUniformityResult verify_spirv_barrier_uniformity(
    const spirv::Module& spv_mod,
    const GpuProverOptions& opts) {
    BVContext ctx;
    GpuSymEnv env_a, env_b;
    BVNodeId same_sg = 0;
    setup_two_thread_envs(ctx, env_a, env_b, {}, opts, same_sg);

    EquivOptions eq_opts;
    eq_opts.loop_unroll_bound = opts.loop_unroll_bound;
    eq_opts.max_sat_conflicts = opts.max_sat_conflicts;
    GpuExecTrace trace_a = execute_gpu_spirv(ctx, spv_mod, env_a, eq_opts);
    if (!trace_a.supported) {
        spirv::ComputeShaderAuditReport rep = spirv::audit_compute_shader(spv_mod);
        BarrierUniformityResult res;
        if (!rep.parse_ok || !rep.structural_ok || !rep.dominance_ok) {
            res.status = EquivStatus::Unsupported;
            res.diagnostic = trace_a.error_msg;
            return res;
        }
        for (size_t i = 0; i < rep.barriers.size(); ++i) {
            const auto& b = rep.barriers[i];
            const bool is_coll = (b.op != spv::OpControlBarrier);
            if (b.uniform_control_flow) {
                if (is_coll) ++res.collectives_proven;
                else ++res.barriers_proven;
            } else {
                BarrierDivergence div;
                div.is_subgroup_collective = is_coll;
                div.inst_index = static_cast<uint32_t>(i);
                div.description = b.detail;
                res.divergences.push_back(std::move(div));
                res.status = EquivStatus::Counterexample;
            }
        }
        if (!res.divergences.empty()) {
            res.diagnostic = res.divergences.front().description;
        }
        return res;
    }
    env_b.param_vars = env_a.param_vars;
    env_b.param_types = env_a.param_types;
    env_b.uf_calls = env_a.uf_calls;
    env_b.global_init_bytes = env_a.global_init_bytes;
    env_b.shared_init_bytes = env_a.shared_init_bytes;
    GpuExecTrace trace_b = execute_gpu_spirv(ctx, spv_mod, env_b, eq_opts);
    return check_traces_barrier_uniformity(ctx, env_a, env_b, trace_a, trace_b, same_sg, false, opts);
}

BarrierUniformityResult verify_ptx_barrier_uniformity(
    const ptx::Function& ptx_fn,
    const GpuProverOptions& opts) {
    BVContext ctx;
    GpuSymEnv env_a, env_b;
    BVNodeId same_sg = 0;
    setup_two_thread_envs(ctx, env_a, env_b, {}, opts, same_sg);

    EquivOptions eq_opts;
    eq_opts.loop_unroll_bound = opts.loop_unroll_bound;
    eq_opts.max_sat_conflicts = opts.max_sat_conflicts;
    GpuExecTrace trace_a = execute_gpu_ptx(ctx, ptx_fn, env_a, eq_opts);
    env_b.param_vars = env_a.param_vars;
    env_b.param_types = env_a.param_types;
    env_b.uf_calls = env_a.uf_calls;
    env_b.global_init_bytes = env_a.global_init_bytes;
    env_b.shared_init_bytes = env_a.shared_init_bytes;
    GpuExecTrace trace_b = execute_gpu_ptx(ctx, ptx_fn, env_b, eq_opts);
    return check_traces_barrier_uniformity(ctx, env_a, env_b, trace_a, trace_b, same_sg, true, opts);
}

TileBoundsProofResult prove_tile_interior_bounds(
    uint32_t tile_base_min,
    uint32_t tile_base_max,
    uint32_t tile_extent,
    uint32_t dim_size,
    std::string* diagnostic_out) {
    TileBoundsProofResult res;
    if (tile_extent == 0 || tile_base_min > tile_base_max) {
        return res;
    }

    BVContext ctx;
    BVNodeId tb = ctx.make_var("tile_base", 32);
    BVNodeId lo = ctx.make_var("local_offset", 32);

    BVNodeId precond = bv_bin(ctx, BVOp::And, 1,
        bv_bin(ctx, BVOp::And, 1,
            bv_bin(ctx, BVOp::Uge, 1, tb, bv_const(ctx, 32, tile_base_min)),
            bv_bin(ctx, BVOp::Ule, 1, tb, bv_const(ctx, 32, tile_base_max))),
        bv_bin(ctx, BVOp::Ult, 1, lo, bv_const(ctx, 32, tile_extent)));

    BVNodeId sum = bv_bin(ctx, BVOp::Add, 32, tb, lo);
    BVNodeId uovf = bv_bin(ctx, BVOp::UAddOvf, 1, tb, lo);
    BVNodeId sovf = bv_bin(ctx, BVOp::Slt, 1, sum, bv_const(ctx, 32, 0));
    BVNodeId oob = bv_bin(ctx, BVOp::Uge, 1, sum, bv_const(ctx, 32, dim_size));

    BVNodeId bad = bv_bin(ctx, BVOp::And, 1, precond,
        bv_bin(ctx, BVOp::Or, 1, uovf, bv_bin(ctx, BVOp::Or, 1, sovf, oob)));

    SolveOutcome out = solve_nonzero(ctx, bad);
    if (out.status == SolveOutcome::Unsat) {
        res.status = EquivStatus::ProvenEquivalent;
        return res;
    }
    if (out.status == SolveOutcome::Unknown) {
        res.status = EquivStatus::Unknown;
        res.diagnostic = "SAT solver budget exhausted in prove_tile_interior_bounds";
        if (diagnostic_out) *diagnostic_out = res.diagnostic;
        return res;
    }

    std::vector<uint64_t> nvals;
    ctx.evaluate_all(out.model, nvals);
    res.status = EquivStatus::Counterexample;
    res.counterexample_tile_base = tb < nvals.size() ? static_cast<uint32_t>(nvals[tb]) : 0;
    res.counterexample_local_offset = lo < nvals.size() ? static_cast<uint32_t>(nvals[lo]) : 0;
    std::ostringstream oss;
    oss << "Tile interior bounds violation: tile_base=" << res.counterexample_tile_base
        << " + local_offset=" << res.counterexample_local_offset
        << " >= dim_size=" << dim_size;
    res.diagnostic = oss.str();
    if (diagnostic_out) *diagnostic_out = res.diagnostic;
    return res;
}

uint32_t eliminate_interior_tile_bounds(
    Function& fn,
    Module& mod,
    uint32_t tile_extent,
    uint32_t dim_size) {
    if (fn.blocks().empty() || tile_extent == 0 || dim_size == 0) return 0;

    BVContext ctx;
    std::unordered_map<uint32_t, BVNodeId> val_nodes;
    BVNodeId coord_assumptions = bv_true(ctx);

    const BasicBlock* entry = fn.entry_block();
    for (size_t i = 0; i < entry->param_count(); ++i) {
        const Value* p = entry->param(i);
        if (!p) continue;
        uint32_t w = gpu_type_bv_width(p->type());
        if (w == 0) w = 32;
        val_nodes[p->id()] = ctx.make_var("p_" + std::to_string(i), w);
    }

    auto get_node = [&](const Value* v) -> BVNodeId {
        if (!v) return bv_const(ctx, 32, 0);
        auto it = val_nodes.find(v->id());
        if (it != val_nodes.end()) return it->second;
        uint32_t w = gpu_type_bv_width(v->type());
        if (w == 0) w = 32;
        BVNodeId var = ctx.make_var("v_" + std::to_string(v->id()), w);
        val_nodes[v->id()] = var;
        return var;
    };

    // Build symbolic expressions for all instructions
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (!inst->result()) continue;
            Opcode op = inst->opcode();
            uint32_t rw = gpu_type_bv_width(inst->type());
            if (rw == 0) rw = 32;
            BVNodeId res = 0;
            switch (op) {
            case Opcode::iconst_i32:
                res = bv_const(ctx, 32, static_cast<uint32_t>(inst->imm_i32()));
                break;
            case Opcode::iconst_i64:
                res = bv_const(ctx, 64, static_cast<uint64_t>(inst->imm_i64()));
                break;
            case Opcode::add:
            case Opcode::sub:
            case Opcode::mul:
            case Opcode::shl:
            case Opcode::lshr:
            case Opcode::ashr:
            case Opcode::and_:
            case Opcode::or_:
            case Opcode::xor_: {
                BVNodeId a = gpu_coerce_width(ctx, get_node(inst->operand(0)), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_node(inst->operand(1)), rw);
                BVOp bop = (op == Opcode::add) ? BVOp::Add : (op == Opcode::sub) ? BVOp::Sub
                         : (op == Opcode::mul) ? BVOp::Mul : (op == Opcode::shl) ? BVOp::Shl
                         : (op == Opcode::lshr) ? BVOp::LShr : (op == Opcode::ashr) ? BVOp::AShr
                         : (op == Opcode::and_) ? BVOp::And : (op == Opcode::or_) ? BVOp::Or : BVOp::Xor;
                res = bv_bin(ctx, bop, rw, a, b);
                break;
            }
            case Opcode::eq:
            case Opcode::ne:
            case Opcode::slt:
            case Opcode::sle:
            case Opcode::sgt:
            case Opcode::sge:
            case Opcode::ult:
            case Opcode::ule:
            case Opcode::ugt:
            case Opcode::uge: {
                BVNodeId a = get_node(inst->operand(0));
                BVNodeId b = gpu_coerce_width(ctx, get_node(inst->operand(1)), ctx.width(a));
                BVOp cop = (op == Opcode::eq) ? BVOp::Eq : (op == Opcode::ne) ? BVOp::Ne
                         : (op == Opcode::slt) ? BVOp::Slt : (op == Opcode::sle) ? BVOp::Sle
                         : (op == Opcode::sgt) ? BVOp::Sgt : (op == Opcode::sge) ? BVOp::Sge
                         : (op == Opcode::ult) ? BVOp::Ult : (op == Opcode::ule) ? BVOp::Ule
                         : (op == Opcode::ugt) ? BVOp::Ugt : BVOp::Uge;
                res = bv_ext(ctx, false, 32, bv_bin(ctx, cop, 1, a, b));
                break;
            }
            case Opcode::call: {
                std::string_view sym = inst->symbol();
                res = ctx.make_var(std::string(sym) + "_" + std::to_string(inst->result()->id()), rw);
                if (sym == "ptx_tid_x" || sym == "ptx_tid_y" || sym == "ptx_tid_z") {
                    coord_assumptions = bv_bin(ctx, BVOp::And, 1, coord_assumptions,
                        bv_bin(ctx, BVOp::Ult, 1, res, bv_const(ctx, 32, tile_extent)));
                } else if (sym == "ptx_ntid_x") {
                    res = bv_const(ctx, 32, tile_extent);
                } else if (sym == "ptx_ctaid_x" || sym == "ptx_ctaid_y" || sym == "ptx_ctaid_z") {
                    coord_assumptions = bv_bin(ctx, BVOp::And, 1, coord_assumptions,
                        bv_bin(ctx, BVOp::Ult, 1, res, bv_const(ctx, 32, 65535)));
                }
                break;
            }
            default:
                break;
            }
            if (res != 0) val_nodes[inst->result()->id()] = res;
        }
    }

    // Collect single-predecessor conditional branch guards for each block
    std::unordered_map<const BasicBlock*, std::vector<std::pair<const BasicBlock*, BVNodeId>>> preds;
    for (const BasicBlock* bb : fn.blocks()) {
        const Instruction* term = bb->terminator();
        if (!term) continue;
        if (term->opcode() == Opcode::br && term->branch_target().block) {
            preds[term->branch_target().block].push_back({bb, bv_true(ctx)});
        } else if (term->opcode() == Opcode::br_if) {
            BVNodeId c = get_node(term->operand(0));
            BVNodeId c1 = ctx.width(c) == 1 ? c : bv_bin(ctx, BVOp::Ne, 1, c, bv_const(ctx, ctx.width(c), 0));
            if (term->true_target().block) {
                preds[term->true_target().block].push_back({bb, c1});
            }
            if (term->false_target().block) {
                preds[term->false_target().block].push_back({bb, bv_un(ctx, BVOp::Not, 1, c1)});
            }
        }
    }

    uint32_t eliminated = 0;
    for (BasicBlock* bb : fn.blocks()) {
        BVNodeId path_assump = coord_assumptions;
        const BasicBlock* cur = bb;
        for (int depth = 0; depth < 8 && cur && preds[cur].size() == 1; ++depth) {
            path_assump = bv_bin(ctx, BVOp::And, 1, path_assump, preds[cur][0].second);
            cur = preds[cur][0].first;
        }
        if (path_assump == coord_assumptions && bb != entry) continue;

        for (Instruction* inst = bb->head(); inst; inst = inst->next()) {
            Opcode op = inst->opcode();
            if (op != Opcode::ult && op != Opcode::slt && op != Opcode::ule && op != Opcode::sle) {
                continue;
            }
            if (!inst->result()) continue;
            BVNodeId cmp32 = get_node(inst->result());
            BVNodeId cmp1 = bv_bin(ctx, BVOp::Ne, 1, cmp32, bv_const(ctx, 32, 0));

            // Check if path_assump proves cmp1 is ALWAYS true (i.e. path_assump & !cmp1 is UNSAT)
            // while path_assump itself is satisfiable.
            if (solve_nonzero(ctx, path_assump, 2000).status != SolveOutcome::Sat) continue;
            BVNodeId not_cmp = bv_bin(ctx, BVOp::And, 1, path_assump, bv_un(ctx, BVOp::Not, 1, cmp1));
            if (solve_nonzero(ctx, not_cmp, 4000).status == SolveOutcome::Unsat) {
                inst->set_opcode(Opcode::iconst_i32);
                inst->operands().clear();
                inst->set_imm_i32(1);
                ++eliminated;

                // If the block terminator branches on this proven comparison, fold br_if -> br
                Instruction* term = bb->terminator();
                if (term && term->opcode() == Opcode::br_if && term->operand(0) == inst->result()) {
                    BranchTarget taken = term->true_target();
                    term->set_opcode(Opcode::br);
                    term->operands().clear();
                    term->true_target() = BranchTarget();
                    term->false_target() = BranchTarget();
                    term->set_branch_target(std::move(taken));
                }
            }
        }
    }
    (void)mod;
    return eliminated;
}

} // namespace brass::verify

extern "C" {

static void copy_prover_err(const std::string& s, char* buf, size_t cap) {
    if (!buf || cap == 0) return;
    size_t n = std::min(s.size(), cap - 1);
    std::memcpy(buf, s.data(), n);
    buf[n] = '\0';
}

int32_t brass_verify_gpu_memory_safety(
    const struct BrassFunction_T* mir_fn,
    uint32_t local_size_x,
    uint32_t grid_size_x,
    uint64_t buffer0_static_bytes,
    char* err_buf,
    size_t err_buf_cap) {
    if (!mir_fn) return -3;
    brass::verify::GpuProverOptions opts;
    opts.local_size_x = local_size_x ? local_size_x : 256;
    opts.grid_size_x = grid_size_x ? grid_size_x : 256;
    if (buffer0_static_bytes > 0) {
        opts.buffers.push_back({0, buffer0_static_bytes, -1, 4, 16});
    }
    auto res = brass::verify::verify_gpu_memory_safety(
        *reinterpret_cast<const brass::Function*>(mir_fn), opts);
    copy_prover_err(res.diagnostic, err_buf, err_buf_cap);
    return static_cast<int32_t>(res.status);
}

int32_t brass_verify_gpu_barrier_uniformity(
    const struct BrassFunction_T* mir_fn,
    uint32_t local_size_x,
    uint32_t subgroup_size,
    char* err_buf,
    size_t err_buf_cap) {
    if (!mir_fn) return -3;
    brass::verify::GpuProverOptions opts;
    opts.local_size_x = local_size_x ? local_size_x : 256;
    opts.subgroup_size = subgroup_size ? subgroup_size : 32;
    auto res = brass::verify::verify_gpu_barrier_uniformity(
        *reinterpret_cast<const brass::Function*>(mir_fn), opts);
    copy_prover_err(res.diagnostic, err_buf, err_buf_cap);
    return static_cast<int32_t>(res.status);
}

int32_t brass_prove_tile_interior_bounds(
    uint32_t tile_base_min,
    uint32_t tile_base_max,
    uint32_t tile_extent,
    uint32_t dim_size,
    char* err_buf,
    size_t err_buf_cap) {
    std::string diag;
    auto res = brass::verify::prove_tile_interior_bounds(
        tile_base_min, tile_base_max, tile_extent, dim_size, &diag);
    copy_prover_err(diag, err_buf, err_buf_cap);
    return static_cast<int32_t>(res.status);
}

} // extern "C"
