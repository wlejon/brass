#include <brass/verify/smt_equiv.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/mir/function.hpp>
#include "lir_sym_exec_arm64.hpp"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace brass::verify {

using aarch64::FPR;
using aarch64::GPR;
using codegen::LirBlock;
using codegen::LirFunction;
using codegen::LirInst;
using codegen::LirOpcode;
using codegen::LirOperand;

namespace {

constexpr uint64_t kArm64LocalSlotBase = 0x7FFF000000000000ULL;
constexpr uint64_t kArm64CanonicalFp   = 0x7FFD000000000000ULL;

BVNodeId fit_width(BVContext& ctx, BVNodeId n, uint32_t w, bool sext = false) {
    uint32_t cur = ctx.width(n);
    if (cur == w) return n;
    if (cur > w) return ctx.make_extract(n, w - 1, 0);
    return sext ? ctx.make_sext(n, w - cur) : ctx.make_zext(n, w - cur);
}

std::pair<BVNodeId, BVNodeId> pack_v128_lanes(BVContext& ctx, Type t, const std::vector<BVNodeId>& lanes) {
    uint32_t ew = type_bv_width(t.element_type());
    size_t half = 64u / ew;
    BVNodeId lo = fit_width(ctx, lanes[half - 1], ew);
    BVNodeId hi = fit_width(ctx, lanes[2 * half - 1], ew);
    for (size_t k = half - 1; k-- > 0;) {
        lo = ctx.make_concat(lo, fit_width(ctx, lanes[k], ew));
        hi = ctx.make_concat(hi, fit_width(ctx, lanes[half + k], ew));
    }
    return {lo, hi};
}

std::vector<BVNodeId> unpack_v128_lanes(BVContext& ctx, Type t, BVNodeId lo, BVNodeId hi) {
    uint32_t ew = type_bv_width(t.element_type());
    if (ew == 64) {
        return {fit_width(ctx, lo, 64), fit_width(ctx, hi, 64)};
    }
    size_t half = 64u / ew;
    std::vector<BVNodeId> out(2 * half);
    for (size_t k = 0; k < half; ++k) {
        uint32_t bit = static_cast<uint32_t>(k) * ew;
        out[k] = ctx.make_extract(lo, bit + ew - 1, bit);
        out[half + k] = ctx.make_extract(hi, bit + ew - 1, bit);
    }
    return out;
}

void merge_u32_map(BVContext& ctx, std::unordered_map<uint32_t, BVNodeId>& dst,
                   BVNodeId ec, const std::unordered_map<uint32_t, BVNodeId>& src,
                   BVNodeId default_fallback = UINT32_MAX) {
    auto merge_node = [&](BVNodeId& d, BVNodeId s) {
        if (d == s) return;
        uint32_t w = std::max(ctx.width(d), ctx.width(s));
        d = ctx.make_ite(ec, fit_width(ctx, s, w), fit_width(ctx, d, w));
    };
    if (default_fallback != UINT32_MAX) {
        for (auto& [k, dv] : dst) {
            auto sit = src.find(k);
            if (sit != src.end()) merge_node(dv, sit->second);
        }
    }
    for (const auto& [k, sv] : src) {
        auto it = dst.find(k);
        if (it == dst.end()) dst[k] = sv;
        else merge_node(it->second, sv);
    }
}

template <typename KeyT>
void merge_slot_map(BVContext& ctx, std::unordered_map<KeyT, BVNodeId>& dst,
                    BVNodeId ec, const std::unordered_map<KeyT, BVNodeId>& src) {
    for (const auto& [k, sv] : src) {
        auto it = dst.find(k);
        if (it == dst.end()) {
            dst[k] = sv;
        } else if (it->second != sv) {
            uint32_t w = std::max(ctx.width(it->second), ctx.width(sv));
            it->second = ctx.make_ite(ec, fit_width(ctx, sv, w), fit_width(ctx, it->second, w));
        }
    }
}

void merge_arm64_state(BVContext& ctx, Arm64ExecState& dst, BVNodeId& dst_cond,
                       BVNodeId ec, const Arm64ExecState& src) {
    dst_cond = ctx.make_binary(BVOp::Or, dst_cond, ec, 1);
    for (size_t i = 0; i < 32; ++i) {
        if (dst.gpr[i] != src.gpr[i]) dst.gpr[i] = ctx.make_ite(ec, src.gpr[i], dst.gpr[i]);
        if (dst.vec_lo[i] != src.vec_lo[i]) dst.vec_lo[i] = ctx.make_ite(ec, src.vec_lo[i], dst.vec_lo[i]);
        if (dst.vec_hi[i] != src.vec_hi[i]) dst.vec_hi[i] = ctx.make_ite(ec, src.vec_hi[i], dst.vec_hi[i]);
    }
    merge_u32_map(ctx, dst.vregs, ec, src.vregs);
    merge_u32_map(ctx, dst.vreg_hi, ec, src.vreg_hi);
    merge_slot_map(ctx, dst.spill_slots, ec, src.spill_slots);
    merge_slot_map(ctx, dst.spill_slots_hi, ec, src.spill_slots_hi);
    merge_slot_map(ctx, dst.local_slots, ec, src.local_slots);
    merge_slot_map(ctx, dst.local_slots_hi, ec, src.local_slots_hi);

    auto mb = [&](BVNodeId& d, BVNodeId s) {
        if (d != s) {
            d = ctx.make_ite(ec, s, d);
            dst.nzcv.from_subs = false;
        }
    };
    mb(dst.nzcv.n, src.nzcv.n);
    mb(dst.nzcv.z, src.nzcv.z);
    mb(dst.nzcv.c, src.nzcv.c);
    mb(dst.nzcv.v, src.nzcv.v);
    dst.uses_vec128 = dst.uses_vec128 || src.uses_vec128;
}

void compute_arm64_lir_rpo(const LirFunction& fn, const std::unordered_map<uint32_t, size_t>& id_to_idx,
                           size_t bi, std::vector<uint8_t>& visited, std::vector<size_t>& postorder) {
    if (bi >= fn.blocks.size() || visited[bi]) return;
    visited[bi] = 1;
    const LirBlock* bb = fn.blocks[bi].get();
    bool has_uncond_end = false;
    for (const auto& inst : bb->instructions) {
        if (!inst) continue;
        const LirOpcode op = inst->opcode;
        if (op == LirOpcode::Jmp) {
            for (const auto& u : inst->uses) {
                if (u.is_label()) {
                    auto it = id_to_idx.find(u.label_id);
                    if (it != id_to_idx.end()) compute_arm64_lir_rpo(fn, id_to_idx, it->second, visited, postorder);
                }
            }
            has_uncond_end = true;
        } else if (op == LirOpcode::Jcc) {
            for (const auto& u : inst->uses) {
                if (u.is_label()) {
                    auto it = id_to_idx.find(u.label_id);
                    if (it != id_to_idx.end()) compute_arm64_lir_rpo(fn, id_to_idx, it->second, visited, postorder);
                }
            }
        } else if (op == LirOpcode::Ret || op == LirOpcode::Trap) {
            has_uncond_end = true;
        }
    }
    if (!has_uncond_end && bi + 1 < fn.blocks.size()) compute_arm64_lir_rpo(fn, id_to_idx, bi + 1, visited, postorder);
    postorder.push_back(bi);
}

struct Arm64ExecResult {
    bool supported = true;
    bool hit_truncation = false;
    std::string error_msg;
    BVNodeId completed = 0;
    BVNodeId trapped = 0;
    BVNodeId returned = 0;
    BVNodeId ret_val = 0;
    std::vector<BVNodeId> ret_vec;
    std::vector<SymStore> stores;
};

struct PendingIncoming {
    BVNodeId edge_cond = 0;
    Arm64ExecState state;
};

Arm64ExecResult symbolically_execute_arm64_lir(
    BVContext& ctx, const Function& mir_fn, const LirFunction& lir_fn,
    const std::vector<BVNodeId>& param_vars, const std::vector<std::vector<BVNodeId>>& param_vec_vars,
    InitMemEnv& init_mem, const EquivOptions& opts) {

    Arm64ExecResult out;
    out.completed = ctx.make_const(0, 1);
    out.trapped = ctx.make_const(0, 1);
    out.returned = ctx.make_const(0, 1);
    uint32_t ret_w = type_bv_width(mir_fn.return_type());
    out.ret_val = ret_w > 0 ? ctx.make_const(0, ret_w) : 0;
    if (mir_fn.return_type().is_v128()) {
        uint32_t ew = type_bv_width(mir_fn.return_type().element_type());
        out.ret_vec.assign(mir_fn.return_type().vector_lanes(), ctx.make_const(0, ew ? ew : 32));
    }

    if (lir_fn.blocks.empty()) {
        out.supported = false;
        out.error_msg = "LirFunction has no blocks";
        return out;
    }

    std::unordered_map<uint32_t, size_t> id_to_idx;
    for (size_t i = 0; i < lir_fn.blocks.size(); ++i) id_to_idx[lir_fn.blocks[i]->id] = i;
    std::vector<uint8_t> visited(lir_fn.blocks.size(), 0);
    std::vector<size_t> postorder;
    compute_arm64_lir_rpo(lir_fn, id_to_idx, 0, visited, postorder);
    std::reverse(postorder.begin(), postorder.end());
    std::vector<size_t> rpo_pos(lir_fn.blocks.size(), SIZE_MAX);
    for (size_t i = 0; i < postorder.size(); ++i) rpo_pos[postorder[i]] = i;

    Arm64InitEnv env;
    env.init_defaults(ctx);
    env.shared_mem = &init_mem;

    Arm64ExecState seed;
    for (int i = 0; i < 32; ++i) {
        seed.gpr[i] = env.gpr_init[i];
        seed.vec_lo[i] = env.vec_lo_init[i];
        seed.vec_hi[i] = env.vec_hi_init[i];
    }
    seed.gpr[static_cast<size_t>(GPR::FP)] = ctx.make_const(kArm64CanonicalFp, 64);
    seed.gpr[static_cast<size_t>(GPR::SP)] = ctx.make_const(kArm64CanonicalFp, 64);
    seed.nzcv.n = seed.nzcv.z = seed.nzcv.c = seed.nzcv.v = ctx.make_const(0, 1);
    seed.shared_stores = &out.stores;
    seed.local_slot_base = kArm64LocalSlotBase;
    seed.canonical_fp = kArm64CanonicalFp;

    // Seed ABI parameters identically to AArch64ISel::lower_entry_parameters.
    size_t gpr_idx = 0, fpr_idx = 0, stack_bytes = 0;
    const bool is_apple = (lir_fn.calling_conv.kind() == CallingConvKind::AppleAAPCS64);
    const BVNodeId always = ctx.make_const(1, 1);
    for (size_t i = 0; i < mir_fn.param_count(); ++i) {
        Type pt = mir_fn.param_type(i);
        if (pt.is_v256()) {
            out.supported = false;
            out.error_msg = "v256 parameters not yet supported in AArch64 ISel verifier";
            return out;
        }
        uint8_t sz = static_cast<uint8_t>(pt.size_in_bytes());
        if (sz == 0) sz = 8;

        auto alloc_caller_offset = [&]() -> int32_t {
            size_t align = is_apple ? ((sz >= 16) ? 16 : (sz >= 8 ? 8 : (sz >= 4 ? 4 : (sz >= 2 ? 2 : 1))))
                                    : ((sz >= 16) ? 16 : 8);
            stack_bytes = (stack_bytes + align - 1) & ~(align - 1);
            int32_t off = static_cast<int32_t>(stack_bytes);
            stack_bytes += is_apple ? sz : ((sz >= 16) ? 16 : 8);
            return off;
        };

        if (pt.is_v128()) {
            auto [vlo, vhi] = pack_v128_lanes(ctx, pt, param_vec_vars[i]);
            if (fpr_idx < 8) {
                seed.vec_lo[fpr_idx] = vlo;
                seed.vec_hi[fpr_idx] = vhi;
                ++fpr_idx;
            } else {
                int32_t off = alloc_caller_offset();
                BVNodeId a_lo = ctx.make_const(kArm64CanonicalFp + static_cast<uint64_t>(static_cast<int64_t>(-1 - off)), 64);
                BVNodeId a_hi = ctx.make_const(kArm64CanonicalFp + static_cast<uint64_t>(static_cast<int64_t>(-1 - off)) + 8ULL, 64);
                out.stores.push_back({always, a_lo, 8, vlo, true});
                out.stores.push_back({always, a_hi, 8, vhi, true});
            }
            seed.uses_vec128 = true;
        } else if (pt.is_float()) {
            BVNodeId pv = fit_width(ctx, param_vars[i], 64, false);
            if (fpr_idx < 8) {
                seed.vec_lo[fpr_idx] = pv;
                seed.vec_hi[fpr_idx] = ctx.make_const(0, 64);
                ++fpr_idx;
            } else {
                int32_t off = alloc_caller_offset();
                BVNodeId saddr = ctx.make_const(kArm64CanonicalFp + static_cast<uint64_t>(static_cast<int64_t>(-1 - off)), 64);
                out.stores.push_back({always, saddr, sz, fit_width(ctx, param_vars[i], sz * 8u, false), true});
            }
        } else {
            BVNodeId pv = fit_width(ctx, param_vars[i], 64, false);
            if (gpr_idx < 8) {
                seed.gpr[gpr_idx++] = pv;
            } else {
                int32_t off = alloc_caller_offset();
                BVNodeId saddr = ctx.make_const(kArm64CanonicalFp + static_cast<uint64_t>(static_cast<int64_t>(-1 - off)), 64);
                out.stores.push_back({always, saddr, sz, fit_width(ctx, param_vars[i], sz * 8u, false), true});
            }
        }
    }

    const uint32_t max_level = opts.loop_unroll_bound;
    std::vector<std::vector<std::vector<PendingIncoming>>> incoming(
        max_level + 1, std::vector<std::vector<PendingIncoming>>(postorder.size()));
    incoming[0][0].push_back({ctx.make_const(1, 1), std::move(seed)});

    auto route_label = [&](size_t src_rpo, uint32_t level, BVNodeId edge_cond, uint32_t label_id,
                           const Arm64ExecState& st) {
        if (ctx.nodes()[edge_cond].op == BVOp::Const && ctx.nodes()[edge_cond].imm == 0) return;
        auto it = id_to_idx.find(label_id);
        if (it == id_to_idx.end()) return;
        size_t dst_rpo = rpo_pos[it->second];
        if (dst_rpo == SIZE_MAX) return;
        uint32_t dst_level = (dst_rpo <= src_rpo) ? (level + 1) : level;
        if (dst_level > max_level) {
            out.hit_truncation = true;
            return;
        }
        incoming[dst_level][dst_rpo].push_back({edge_cond, st});
    };

    for (uint32_t level = 0; level <= max_level; ++level) {
        for (size_t ri = 0; ri < postorder.size(); ++ri) {
            auto& edges = incoming[level][ri];
            if (edges.empty()) continue;

            BVNodeId cur_cond = edges[0].edge_cond;
            Arm64ExecState st = std::move(edges[0].state);
            for (size_t ei = 1; ei < edges.size(); ++ei) {
                merge_arm64_state(ctx, st, cur_cond, edges[ei].edge_cond, edges[ei].state);
            }
            edges.clear();

            const LirBlock* bb = lir_fn.blocks[postorder[ri]].get();
            bool terminated = false;

            auto emit_trap = [&](BVNodeId trap_c) {
                BVNodeId trap_now = ctx.make_binary(BVOp::And, cur_cond, trap_c, 1);
                out.trapped = ctx.make_binary(BVOp::Or, out.trapped, trap_now, 1);
                out.completed = ctx.make_binary(BVOp::Or, out.completed, trap_now, 1);
                cur_cond = ctx.make_binary(BVOp::And, cur_cond, ctx.make_unary(BVOp::Not, trap_c, 1), 1);
            };

            for (const auto& ip : bb->instructions) {
                if (!ip) continue;
                if (ctx.nodes()[cur_cond].op == BVOp::Const && ctx.nodes()[cur_cond].imm == 0) {
                    terminated = true;
                    break;
                }
                const LirInst& inst = *ip;
                const LirOpcode op = inst.opcode;
                st.path_cond = cur_cond;

                if (op == LirOpcode::Trap) {
                    out.trapped = ctx.make_binary(BVOp::Or, out.trapped, cur_cond, 1);
                    out.completed = ctx.make_binary(BVOp::Or, out.completed, cur_cond, 1);
                    cur_cond = ctx.make_const(0, 1);
                    terminated = true;
                    break;
                }
                if (op == LirOpcode::Ret) {
                    const bool first_ret = (ctx.nodes()[out.returned].op == BVOp::Const &&
                                            ctx.nodes()[out.returned].imm == 0);
                    out.returned = ctx.make_binary(BVOp::Or, out.returned, cur_cond, 1);
                    out.completed = ctx.make_binary(BVOp::Or, out.completed, cur_cond, 1);
                    if (!mir_fn.return_type().is_void()) {
                        if (mir_fn.return_type().is_v128()) {
                            std::vector<BVNodeId> rlanes =
                                unpack_v128_lanes(ctx, mir_fn.return_type(), st.vec_lo[0], st.vec_hi[0]);
                            for (size_t li = 0; li < out.ret_vec.size() && li < rlanes.size(); ++li) {
                                out.ret_vec[li] = first_ret ? rlanes[li]
                                                            : ctx.make_ite(cur_cond, rlanes[li], out.ret_vec[li]);
                            }
                        } else {
                            BVNodeId rv = mir_fn.return_type().is_float()
                                ? fit_width(ctx, st.vec_lo[0], ret_w, false)
                                : fit_width(ctx, st.gpr[0], ret_w, false);
                            out.ret_val = first_ret ? rv : ctx.make_ite(cur_cond, rv, out.ret_val);
                        }
                    }
                    terminated = true;
                    break;
                }
                if (op == LirOpcode::Idiv || op == LirOpcode::Idiv32 ||
                    op == LirOpcode::Div || op == LirOpcode::Div32) {
                    const uint32_t w = (op == LirOpcode::Idiv32 || op == LirOpcode::Div32) ? 32u : 64u;
                    const bool is_s = (op == LirOpcode::Idiv || op == LirOpcode::Idiv32);
                    BVNodeId a = arm64_read_op(ctx, env, st, inst.uses.size() >= 2 ? inst.uses[0] : inst.defs[0], w);
                    BVNodeId b = arm64_read_op(ctx, env, st, inst.uses.back(), w);
                    BVNodeId trap_c = ctx.make_binary(BVOp::Eq, b, ctx.make_const(0, w), 1);
                    if (is_s) {
                        uint64_t smin = (w == 32) ? 0x80000000ULL : 0x8000000000000000ULL;
                        uint64_t neg1 = (w == 32) ? 0xFFFFFFFFULL : ~0ULL;
                        BVNodeId ovf = ctx.make_binary(
                            BVOp::And,
                            ctx.make_binary(BVOp::Eq, a, ctx.make_const(smin, w), 1),
                            ctx.make_binary(BVOp::Eq, b, ctx.make_const(neg1, w), 1), 1);
                        trap_c = ctx.make_binary(BVOp::Or, trap_c, ovf, 1);
                    }
                    emit_trap(trap_c);
                }

                st.has_cond_branch = false;
                st.has_uncond_branch = false;
                if (!execute_arm64_inst(ctx, env, st, inst)) {
                    out.supported = false;
                    out.error_msg = st.error_msg;
                    return out;
                }

                if (st.has_cond_branch) {
                    BVNodeId taken = st.cond_branch_taken;
                    uint32_t target = st.cond_target_label;
                    st.has_cond_branch = false;
                    route_label(ri, level, ctx.make_binary(BVOp::And, cur_cond, taken, 1), target, st);
                    cur_cond = ctx.make_binary(BVOp::And, cur_cond, ctx.make_unary(BVOp::Not, taken, 1), 1);
                } else if (st.has_uncond_branch) {
                    uint32_t target = st.uncond_target_label;
                    st.has_uncond_branch = false;
                    route_label(ri, level, cur_cond, target, st);
                    terminated = true;
                    break;
                }
            }

            if (!terminated && postorder[ri] + 1 < lir_fn.blocks.size()) {
                route_label(ri, level, cur_cond, lir_fn.blocks[postorder[ri] + 1]->id, st);
            }
        }
    }
    return out;
}

} // namespace

EquivResult verify_arm64_isel_equivalence(
    const Function& mir_fn, const LirFunction& lir_fn, const EquivOptions& opts) {

    EquivResult res;
    BVContext ctx;
    std::vector<BVNodeId> param_vars(mir_fn.param_count(), 0);
    std::vector<std::vector<BVNodeId>> param_vec_vars(mir_fn.param_count());
    for (size_t i = 0; i < mir_fn.param_count(); ++i) {
        Type pt = mir_fn.param_type(i);
        if (pt.is_v128()) {
            uint32_t lanes = pt.vector_lanes();
            uint32_t ew = type_bv_width(pt.element_type());
            bool is_fp = pt.element_type().is_float();
            param_vec_vars[i].resize(lanes);
            for (uint32_t l = 0; l < lanes; ++l) {
                param_vec_vars[i][l] = ctx.make_var(
                    "arg" + std::to_string(i) + "_l" + std::to_string(l), ew, is_fp);
            }
        } else {
            uint32_t w = type_bv_width(pt);
            if (w == 0) {
                res.status = EquivStatus::Unsupported;
                res.diagnostic = "Unsupported parameter type in AArch64 ISel verifier";
                return res;
            }
            param_vars[i] = ctx.make_var("arg" + std::to_string(i), w, pt.is_float());
        }
    }
    if (!mir_fn.return_type().is_void() && !mir_fn.return_type().is_v128() &&
        type_bv_width(mir_fn.return_type()) == 0) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = "Unsupported return type in AArch64 ISel verifier";
        return res;
    }

    SymExecRegionConfig region_cfg;
    region_cfg.param_vec_vars = param_vec_vars;

    InitMemEnv init_mem;
    SymExecOutput out_mir = symbolically_execute_mir_region(ctx, mir_fn, param_vars, init_mem, opts, region_cfg);
    if (!out_mir.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = out_mir.error_msg;
        return res;
    }
    Arm64ExecResult out_lir = symbolically_execute_arm64_lir(
        ctx, mir_fn, lir_fn, param_vars, param_vec_vars, init_mem, opts);
    if (!out_lir.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = out_lir.error_msg;
        return res;
    }

    BVNodeId trap_diff = ctx.make_binary(BVOp::Xor, out_mir.trapped, out_lir.trapped, 1);
    BVNodeId no_trap = ctx.make_binary(
        BVOp::And, ctx.make_unary(BVOp::Not, out_mir.trapped, 1),
        ctx.make_unary(BVOp::Not, out_lir.trapped, 1), 1);
    BVNodeId both_ret = ctx.make_binary(BVOp::And, out_mir.returned, out_lir.returned, 1);
    BVNodeId can_cmp_ret = ctx.make_binary(BVOp::And, no_trap, both_ret, 1);
    BVNodeId ret_diff = ctx.make_const(0, 1);
    if (mir_fn.return_type().is_v128()) {
        if (out_mir.ret_vec.size() != out_lir.ret_vec.size()) {
            res.status = EquivStatus::Unsupported;
            res.diagnostic = "Vector return lane count mismatch in AArch64 ISel verifier";
            return res;
        }
        BVNodeId any_lane_ne = ctx.make_const(0, 1);
        for (size_t li = 0; li < out_mir.ret_vec.size(); ++li) {
            any_lane_ne = ctx.make_binary(
                BVOp::Or, any_lane_ne, ctx.make_binary(BVOp::Ne, out_mir.ret_vec[li], out_lir.ret_vec[li], 1), 1);
        }
        ret_diff = ctx.make_binary(BVOp::And, can_cmp_ret, any_lane_ne, 1);
    } else if (!mir_fn.return_type().is_void()) {
        ret_diff = ctx.make_binary(BVOp::And, can_cmp_ret,
                                   ctx.make_binary(BVOp::Ne, out_mir.ret_val, out_lir.ret_val, 1), 1);
    }

    BVNodeId mem_diff = ctx.make_const(0, 1);
    if (opts.check_memory) {
        std::vector<BVNodeId> check_addrs;
        auto add_addrs = [&](const std::vector<SymStore>& stores) {
            for (const auto& s : stores) {
                if (s.is_local_alloca) continue;
                for (uint32_t b = 0; b < s.num_bytes; ++b) {
                    BVNodeId ba = (b == 0) ? s.addr : ctx.make_binary(BVOp::Add, s.addr, ctx.make_const(b, 64), 64);
                    if (std::find(check_addrs.begin(), check_addrs.end(), ba) == check_addrs.end()) {
                        check_addrs.push_back(ba);
                    }
                }
            }
        };
        add_addrs(out_mir.stores);
        add_addrs(out_lir.stores);
        for (BVNodeId ba : check_addrs) {
            BVNodeId mb = read_byte_at(ctx, init_mem, out_mir.stores, ba);
            BVNodeId lb = read_byte_at(ctx, init_mem, out_lir.stores, ba);
            mem_diff = ctx.make_binary(BVOp::Or, mem_diff,
                ctx.make_binary(BVOp::And, no_trap, ctx.make_binary(BVOp::Ne, mb, lb, 1), 1), 1);
        }
    }

    BVNodeId mismatch = ctx.make_binary(BVOp::Or, trap_diff, ctx.make_binary(BVOp::Or, ret_diff, mem_diff, 1), 1);
    SolveOutcome solved = solve_nonzero(ctx, mismatch, opts.max_sat_conflicts);
    if (solved.status == SolveOutcome::Unsat) {
        if (out_mir.hit_truncation || out_lir.hit_truncation) {
            res.status = EquivStatus::Unknown;
            res.diagnostic = "Bounded loop unrolling truncated AArch64 ISel check";
            return res;
        }
        res.status = EquivStatus::ProvenEquivalent;
        return res;
    }
    if (solved.status == SolveOutcome::Unknown) {
        res.status = EquivStatus::Unknown;
        res.diagnostic = "Solver returned Unknown for AArch64 ISel check";
        return res;
    }

    res.status = EquivStatus::Counterexample;
    std::ostringstream diag;
    diag << "MIR != AArch64 LIR counterexample in @" << mir_fn.name()
         << " (trap_diff=" << ctx.eval_concrete(trap_diff, solved.model)
         << ", ret_diff=" << ctx.eval_concrete(ret_diff, solved.model)
         << ", mem_diff=" << ctx.eval_concrete(mem_diff, solved.model);
    if (out_mir.ret_val != 0 && out_lir.ret_val != 0) {
        diag << ", mir_ret=0x" << std::hex << ctx.eval_concrete(out_mir.ret_val, solved.model)
             << ", lir_ret=0x" << ctx.eval_concrete(out_lir.ret_val, solved.model);
    }
    diag << ")";
    res.diagnostic = diag.str();
    return res;
}

EquivResult verify_arm64_isel(const Function& mir_fn, const EquivOptions& opts) {
    Target target = Target::aarch64_macos();
    CallingConvention cc = CallingConvention::apple_aapcs64();
    aarch64::AArch64ISel isel(target, cc);
    std::unique_ptr<LirFunction> lir_fn = isel.lower(mir_fn);
    if (!lir_fn) {
        return {EquivStatus::Unsupported, {}, "AArch64ISel::lower returned null"};
    }
    return verify_arm64_isel_equivalence(mir_fn, *lir_fn, opts);
}

} // namespace brass::verify
