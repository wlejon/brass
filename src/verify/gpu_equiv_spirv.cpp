#include "gpu_equiv_spirv_internal.hpp"
#include <algorithm>
#include <unordered_map>

namespace brass::verify {

using namespace spirv_detail;

namespace spirv_detail {

namespace {

void collect_and_conjuncts(const BVContext& ctx, BVNodeId node, std::vector<BVNodeId>& out, int depth = 0) {
    if (depth > 16 || node >= ctx.nodes().size()) {
        if (std::find(out.begin(), out.end(), node) == out.end()) out.push_back(node);
        return;
    }
    const BVNode& n = bv_node(ctx, node);
    if (n.op == BVOp::And) {
        collect_and_conjuncts(ctx, n.a, out, depth + 1);
        collect_and_conjuncts(ctx, n.b, out, depth + 1);
        return;
    }
    if (n.op == BVOp::Const && n.imm == 1) return;
    if (std::find(out.begin(), out.end(), node) == out.end()) out.push_back(node);
}

bool are_complements(const BVContext& ctx, BVNodeId a, BVNodeId b) noexcept {
    if (a >= ctx.nodes().size() || b >= ctx.nodes().size()) return false;
    const BVNode& na = bv_node(ctx, a);
    const BVNode& nb = bv_node(ctx, b);
    return (na.op == BVOp::Not && na.a == b) || (nb.op == BVOp::Not && nb.a == a);
}

BVNodeId build_conjunction(BVContext& ctx, const std::vector<BVNodeId>& items) {
    if (items.empty()) return bv_true(ctx);
    BVNodeId acc = items[0];
    for (size_t i = 1; i < items.size(); ++i) acc = bv_bin(ctx, BVOp::And, 1, acc, items[i]);
    return acc;
}

BVNodeId merge_path_cond(BVContext& ctx, BVNodeId a, BVNodeId b) {
    if (a == b) return a;
    if (bv_node(ctx, a).op == BVOp::Const) return bv_node(ctx, a).imm ? a : b;
    if (bv_node(ctx, b).op == BVOp::Const) return bv_node(ctx, b).imm ? b : a;

    std::vector<BVNodeId> ca, cb;
    collect_and_conjuncts(ctx, a, ca);
    collect_and_conjuncts(ctx, b, cb);

    std::vector<BVNodeId> common, rem_a, rem_b;
    for (BVNodeId x : ca) {
        if (std::find(cb.begin(), cb.end(), x) != cb.end()) common.push_back(x);
        else rem_a.push_back(x);
    }
    for (BVNodeId y : cb) {
        if (std::find(common.begin(), common.end(), y) == common.end()) rem_b.push_back(y);
    }
    if (rem_a.empty() || rem_b.empty()) return build_conjunction(ctx, common);
    if (rem_a.size() == 1 && rem_b.size() == 1 && are_complements(ctx, rem_a[0], rem_b[0])) {
        return build_conjunction(ctx, common);
    }
    if (common.empty()) return bv_bin(ctx, BVOp::Or, 1, a, b);
    BVNodeId rem_or = bv_bin(ctx, BVOp::Or, 1, build_conjunction(ctx, rem_a), build_conjunction(ctx, rem_b));
    common.push_back(rem_or);
    return build_conjunction(ctx, common);
}

} // namespace

bool execute_spirv_function_body(
    SpirvExecContext& sc,
    const spirv::Function& fn,
    SpirvExecState init_state,
    uint32_t call_depth,
    BVNodeId& out_ret_val,
    std::vector<BVNodeId>& out_ret_comp,
    std::unordered_map<uint64_t, BVNodeId>* out_local_slots) {
    BVContext& ctx = sc.ctx;
    const spirv::Module& mod = sc.mod;
    GpuSymEnv& env = sc.env;
    GpuExecTrace& trace = sc.trace;
    SpirvModuleInfo& minfo = sc.minfo;

    if (fn.blocks.empty()) {
        trace.supported = false;
        trace.error_msg = "SPIR-V function has no basic blocks";
        return false;
    }

    std::unordered_map<spirv::Id, size_t> label_to_idx;
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        label_to_idx[fn.blocks[i].label] = i;
    }

    std::vector<bool> is_loop_branch_block(fn.blocks.size(), false);
    for (size_t bi = 0; bi < fn.blocks.size(); ++bi) {
        auto check_target = [&](spirv::Id lbl) {
            auto tit = label_to_idx.find(lbl);
            if (tit != label_to_idx.end() && tit->second <= bi) {
                is_loop_branch_block[bi] = true;
            }
        };
        for (const auto& inst : fn.blocks[bi].insts) {
            if (inst.op == spv::OpLoopMerge) {
                is_loop_branch_block[bi] = true;
            } else if (inst.op == spv::OpBranch && !inst.operands.empty()) {
                check_target(inst.operands[0].value);
            } else if (inst.op == spv::OpBranchConditional && inst.operands.size() >= 3) {
                check_target(inst.operands[1].value);
                check_target(inst.operands[2].value);
            } else if (inst.op == spv::OpSwitch && inst.operands.size() >= 2) {
                check_target(inst.operands[1].value);
                for (size_t k = 2; k + 1 < inst.operands.size(); k += 2) {
                    check_target(inst.operands[k + 1].value);
                }
            }
        }
    }

    std::vector<SpirvExecState> worklist;
    worklist.push_back(std::move(init_state));

    std::vector<uint32_t> sym_visits(fn.blocks.size(), 0);
    std::vector<uint32_t> total_visits(fn.blocks.size(), 0);
    const uint32_t max_sym_visits = sc.opts.loop_unroll_bound + 2u;
    const uint32_t max_const_visits = (sc.opts.loop_unroll_bound >= 4u)
        ? std::max<uint32_t>(max_sym_visits, 64u)
        : std::max<uint32_t>(max_sym_visits, 16u);

    auto route_edge = [&](spirv::Id from_label, spirv::Id to_label, BVNodeId edge_cond,
                          const SpirvExecState& cur_st) {
        if (bv_node(ctx, edge_cond).op == BVOp::Const && bv_node(ctx, edge_cond).imm == 0) return;
        auto bit = label_to_idx.find(to_label);
        if (bit == label_to_idx.end()) return;
        size_t dst_idx = bit->second;
        const spirv::Block& dst_bb = fn.blocks[dst_idx];

        auto next_vals = cur_st.vals;
        auto next_comps = cur_st.composites;
        for (const auto& inst : dst_bb.insts) {
            if (inst.op != spv::OpPhi) break;
            for (size_t i = 0; i + 1 < inst.operands.size(); i += 2) {
                if (inst.operands[i + 1].value == from_label) {
                    spirv::Id val_id = inst.operands[i].value;
                    if (auto vit = cur_st.vals.find(val_id); vit != cur_st.vals.end()) {
                        next_vals[inst.result] = vit->second;
                    }
                    if (auto cit = cur_st.composites.find(val_id); cit != cur_st.composites.end()) {
                        next_comps[inst.result] = cit->second;
                    }
                    break;
                }
            }
        }

        for (auto& st : worklist) {
            if (st.block_idx == dst_idx) {
                BVNodeId merged = merge_path_cond(ctx, st.cond, edge_cond);
                for (auto& [vid, vnode] : next_vals) {
                    auto it = st.vals.find(vid);
                    if (it == st.vals.end()) {
                        st.vals[vid] = vnode;
                    } else if (it->second != vnode && ctx.width(it->second) == ctx.width(vnode)) {
                        it->second = bv_ite(ctx, ctx.width(vnode), edge_cond, vnode, it->second);
                    }
                }
                for (auto& [cid, cvec] : next_comps) {
                    auto it = st.composites.find(cid);
                    if (it == st.composites.end() || it->second.size() != cvec.size()) {
                        st.composites[cid] = cvec;
                    } else {
                        for (size_t e = 0; e < cvec.size(); ++e) {
                            if (it->second[e] != cvec[e] && ctx.width(it->second[e]) == ctx.width(cvec[e])) {
                                it->second[e] = bv_ite(ctx, ctx.width(cvec[e]), edge_cond, cvec[e], it->second[e]);
                            }
                        }
                    }
                }
                for (auto& [sk, sv] : cur_st.local_slots) {
                    auto it = st.local_slots.find(sk);
                    if (it == st.local_slots.end()) {
                        st.local_slots[sk] = sv;
                    } else if (it->second != sv && ctx.width(it->second) == ctx.width(sv)) {
                        it->second = bv_ite(ctx, ctx.width(sv), edge_cond, sv, it->second);
                    }
                }
                for (const auto& [aid, ainfo] : cur_st.chains) st.chains[aid] = ainfo;
                for (const auto& [lid, linfo] : cur_st.local_vars) st.local_vars[lid] = linfo;
                st.cond = merged;
                return;
            }
        }
        SpirvExecState ns;
        ns.block_idx = dst_idx;
        ns.cond = edge_cond;
        ns.vals = std::move(next_vals);
        ns.composites = std::move(next_comps);
        ns.chains = cur_st.chains;
        ns.local_vars = cur_st.local_vars;
        ns.local_slots = cur_st.local_slots;
        worklist.push_back(std::move(ns));
    };

    out_ret_val = 0;
    out_ret_comp.clear();
    BVNodeId seen_ret = bv_false(ctx);

    while (!worklist.empty()) {
        if (++sc.total_steps > sc.step_budget) {
            trace.supported = false;
            trace.error_msg = "SPIR-V symbolic execution exceeded block step budget";
            return false;
        }
        auto min_it = std::min_element(worklist.begin(), worklist.end(),
                                       [](const SpirvExecState& a, const SpirvExecState& b) {
                                           return a.block_idx < b.block_idx;
                                       });
        SpirvExecState cur = std::move(*min_it);
        worklist.erase(min_it);
        if (bv_node(ctx, cur.cond).op == BVOp::Const && bv_node(ctx, cur.cond).imm == 0) continue;

        if (++total_visits[cur.block_idx] > max_const_visits ||
            sym_visits[cur.block_idx] >= max_sym_visits) {
            BVNodeId not_c = bv_un(ctx, BVOp::Not, 1, cur.cond);
            trace.completed = bv_bin(ctx, BVOp::And, 1, trace.completed, not_c);
            continue;
        }

        const spirv::Block& bb = fn.blocks[cur.block_idx];
        auto get_v = [&](spirv::Id id) -> BVNodeId {
            auto it = cur.vals.find(id);
            if (it != cur.vals.end()) return it->second;
            return bv_const(ctx, 32, 0);
        };

        for (const auto& inst : bb.insts) {
            if (inst.result && inst.type) minfo.value_types[inst.result] = inst.type;
            if (inst.op == spv::OpPhi || inst.op == spv::OpSelectionMerge || inst.op == spv::OpLoopMerge) {
                continue;
            }
            bool is_fl = false;
            uint32_t rw = inst.type ? spirv_type_width(mod, inst.type, &is_fl) : 32;
            const spirv::TypeInfo* rti = inst.type ? mod.type_info(inst.type) : nullptr;
            BVNodeId res = UINT32_MAX;

            switch (inst.op) {
            case spv::OpVariable: {
                if (!inst.operands.empty() && inst.operands[0].value == spv::StorageClassFunction) {
                    spirv::Id init_id = inst.operands.size() >= 2 ? inst.operands[1].value : 0u;
                    init_local_variable(sc, cur, inst.result, inst.type, init_id);
                }
                break;
            }
            case spv::OpUndef: {
                if (rti && rti->op == spv::OpTypeVector) {
                    uint32_t ew = spirv_type_width(mod, rti->elem);
                    cur.composites[inst.result] = std::vector<BVNodeId>(std::max(1u, rti->width), bv_const(ctx, ew, 0));
                } else {
                    res = bv_const(ctx, rw, 0);
                }
                break;
            }
            case spv::OpAccessChain:
            case spv::OpInBoundsAccessChain:
                handle_spirv_access_chain(sc, cur, inst, rti, res);
                break;
            case spv::OpConvertUToPtr:
                res = gpu_coerce_width(ctx, get_v(inst.operands[0].value), 64, false);
                break;
            case spv::OpConvertPtrToU:
                res = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw, false);
                break;
            case spv::OpLoad:
                handle_spirv_load(sc, cur, inst, rw, rti, res);
                break;
            case spv::OpStore:
                handle_spirv_store(sc, cur, inst);
                break;
            case spv::OpCompositeExtract: {
                spirv::Id comp_id = inst.operands[0].value;
                uint32_t idx = inst.operands.size() >= 2 ? inst.operands[1].value : 0;
                if (cur.composites.count(comp_id) && idx < cur.composites[comp_id].size()) {
                    res = cur.composites[comp_id][idx];
                }
                break;
            }
            case spv::OpCompositeConstruct: {
                std::vector<BVNodeId> elems;
                for (const auto& op : inst.operands) elems.push_back(get_v(op.value));
                cur.composites[inst.result] = std::move(elems);
                break;
            }
            case spv::OpCompositeInsert: {
                if (inst.operands.size() >= 3) {
                    std::vector<BVNodeId> elems = cur.composites.count(inst.operands[1].value)
                        ? cur.composites[inst.operands[1].value] : std::vector<BVNodeId>{};
                    uint32_t idx = inst.operands[2].value;
                    if (idx >= elems.size()) elems.resize(idx + 1, bv_const(ctx, 32, 0));
                    elems[idx] = get_v(inst.operands[0].value);
                    cur.composites[inst.result] = std::move(elems);
                }
                break;
            }
            case spv::OpVectorShuffle: {
                if (inst.operands.size() >= 2) {
                    const auto empty = std::vector<BVNodeId>{};
                    const auto& v0 = cur.composites.count(inst.operands[0].value) ? cur.composites[inst.operands[0].value] : empty;
                    const auto& v1 = cur.composites.count(inst.operands[1].value) ? cur.composites[inst.operands[1].value] : empty;
                    std::vector<BVNodeId> out_v;
                    for (size_t k = 2; k < inst.operands.size(); ++k) {
                        uint32_t sel = inst.operands[k].value;
                        if (sel < v0.size()) out_v.push_back(v0[sel]);
                        else if (sel != UINT32_MAX && sel - v0.size() < v1.size()) out_v.push_back(v1[sel - v0.size()]);
                        else out_v.push_back(bv_const(ctx, 32, 0));
                    }
                    cur.composites[inst.result] = std::move(out_v);
                }
                break;
            }
            case spv::OpVectorTimesScalar: {
                if (inst.operands.size() >= 2 && cur.composites.count(inst.operands[0].value)) {
                    const auto& v = cur.composites[inst.operands[0].value];
                    BVNodeId s = get_v(inst.operands[1].value);
                    uint32_t ew = v.empty() ? 32u : ctx.width(v[0]);
                    s = gpu_coerce_width(ctx, s, ew);
                    std::vector<BVNodeId> out_v;
                    out_v.reserve(v.size());
                    for (BVNodeId ev : v) out_v.push_back(env.eval_uf(ctx, "fp_mul", {ev, s}, ew, true));
                    cur.composites[inst.result] = std::move(out_v);
                }
                break;
            }
            case spv::OpDot: {
                if (inst.operands.size() >= 2 && cur.composites.count(inst.operands[0].value) &&
                    cur.composites.count(inst.operands[1].value)) {
                    const auto& va = cur.composites[inst.operands[0].value];
                    const auto& vb = cur.composites[inst.operands[1].value];
                    std::vector<BVNodeId> args;
                    for (size_t k = 0; k < va.size() && k < vb.size(); ++k) {
                        args.push_back(va[k]);
                        args.push_back(vb[k]);
                    }
                    res = env.eval_uf(ctx, "gpu_dot", args, rw, true);
                } else {
                    res = env.eval_uf(ctx, "gpu_dot", {}, rw, true);
                }
                break;
            }
            case spv::OpIAdd:
            case spv::OpISub:
            case spv::OpIMul: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[1].value), rw);
                if (inst.op == spv::OpIAdd && rw == 64) {
                    res = gpu_add_addr64(ctx, a, b);
                } else if (inst.op == spv::OpIMul) {
                    res = gpu_mul_or_shl(ctx, rw, a, b);
                } else if (inst.op == spv::OpISub && sc.safe_div_orig.count(b)) {
                    res = bv_bin(ctx, BVOp::SRem, rw, a, sc.safe_div_orig[b]);
                } else {
                    res = bv_bin(ctx, inst.op == spv::OpIAdd ? BVOp::Add : BVOp::Sub, rw, a, b);
                }
                if (inst.op == spv::OpIMul && sc.safe_div_orig.count(a)) {
                    sc.safe_div_orig[res] = sc.safe_div_orig[a];
                }
                break;
            }
            case spv::OpUDiv:
            case spv::OpSDiv:
            case spv::OpUMod:
            case spv::OpSRem:
            case spv::OpSMod: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[1].value), rw);
                BVNodeId orig_d = b;
                const BVNode& bn = bv_node(ctx, b);
                if (bn.op == BVOp::Ite && bv_node(ctx, bn.b).op == BVOp::Const && bv_node(ctx, bn.b).imm == 1) {
                    orig_d = bn.c;
                }
                BVNodeId d_nz = bv_bin(ctx, BVOp::Ne, 1, orig_d, bv_const(ctx, rw, 0));
                trace.domain_assumption = bv_bin(ctx, BVOp::And, 1, trace.domain_assumption,
                    bv_bin(ctx, BVOp::Or, 1, bv_un(ctx, BVOp::Not, 1, cur.cond), d_nz));
                BVOp bop = (inst.op == spv::OpUDiv) ? BVOp::UDiv
                         : (inst.op == spv::OpSDiv) ? BVOp::SDiv
                         : (inst.op == spv::OpUMod) ? BVOp::URem : BVOp::SRem;
                res = bv_bin(ctx, bop, rw, a, orig_d);
                if (orig_d != b) sc.safe_div_orig[res] = orig_d;
                break;
            }
            case spv::OpUMulExtended: {
                BVNodeId a64 = bv_ext(ctx, false, 64, gpu_coerce_width(ctx, get_v(inst.operands[0].value), 32));
                BVNodeId b64 = bv_ext(ctx, false, 64, gpu_coerce_width(ctx, get_v(inst.operands[1].value), 32));
                BVNodeId prod = bv_bin(ctx, BVOp::Mul, 64, a64, b64);
                cur.composites[inst.result] = {bv_extract(ctx, 31, 0, prod), bv_extract(ctx, 63, 32, prod)};
                break;
            }
            case spv::OpSNegate:
                res = bv_un(ctx, BVOp::Neg, rw, gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw));
                break;
            case spv::OpFNegate: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw);
                res = bv_bin(ctx, BVOp::Xor, rw, a, bv_const(ctx, rw, 1ULL << (rw - 1)));
                break;
            }
            case spv::OpFAdd:
            case spv::OpFSub:
            case spv::OpFMul:
            case spv::OpFDiv:
            case spv::OpFRem: {
                if (rti && rti->op == spv::OpTypeVector &&
                    cur.composites.count(inst.operands[0].value) && cur.composites.count(inst.operands[1].value)) {
                    const auto& va = cur.composites[inst.operands[0].value];
                    const auto& vb = cur.composites[inst.operands[1].value];
                    const char* uf = (inst.op == spv::OpFAdd) ? "fp_add" : (inst.op == spv::OpFSub) ? "fp_sub"
                                   : (inst.op == spv::OpFMul) ? "fp_mul" : (inst.op == spv::OpFDiv) ? "fp_div" : "fp_rem";
                    uint32_t ew = va.empty() ? 32u : ctx.width(va[0]);
                    std::vector<BVNodeId> ov;
                    for (size_t k = 0; k < va.size() && k < vb.size(); ++k) {
                        ov.push_back(env.eval_uf(ctx, uf, {va[k], vb[k]}, ew, true));
                    }
                    cur.composites[inst.result] = std::move(ov);
                } else {
                    BVNodeId a = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw);
                    BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[1].value), rw);
                    const char* uf = (inst.op == spv::OpFAdd) ? "fp_add" : (inst.op == spv::OpFSub) ? "fp_sub"
                                   : (inst.op == spv::OpFMul) ? "fp_mul" : (inst.op == spv::OpFDiv) ? "fp_div" : "fp_rem";
                    res = env.eval_uf(ctx, uf, {a, b}, rw, true);
                }
                break;
            }
            case spv::OpBitwiseAnd:
            case spv::OpBitwiseOr:
            case spv::OpBitwiseXor:
            case spv::OpLogicalAnd:
            case spv::OpLogicalOr: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[1].value), rw);
                BVOp bop = (inst.op == spv::OpBitwiseAnd || inst.op == spv::OpLogicalAnd) ? BVOp::And
                         : (inst.op == spv::OpBitwiseOr || inst.op == spv::OpLogicalOr) ? BVOp::Or : BVOp::Xor;
                res = bv_bin(ctx, bop, rw, a, b);
                break;
            }
            case spv::OpNot:
            case spv::OpLogicalNot:
                res = bv_un(ctx, BVOp::Not, rw, gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw));
                break;
            case spv::OpShiftLeftLogical:
            case spv::OpShiftRightLogical:
            case spv::OpShiftRightArithmetic: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[1].value), rw, false);
                BVOp bop = (inst.op == spv::OpShiftLeftLogical) ? BVOp::Shl
                         : (inst.op == spv::OpShiftRightLogical) ? BVOp::LShr : BVOp::AShr;
                res = bv_bin(ctx, bop, rw, a, b);
                break;
            }
            case spv::OpIEqual:
            case spv::OpINotEqual:
            case spv::OpLogicalEqual:
            case spv::OpLogicalNotEqual:
            case spv::OpSLessThan:
            case spv::OpSLessThanEqual:
            case spv::OpSGreaterThan:
            case spv::OpSGreaterThanEqual:
            case spv::OpULessThan:
            case spv::OpULessThanEqual:
            case spv::OpUGreaterThan:
            case spv::OpUGreaterThanEqual: {
                BVNodeId a = get_v(inst.operands[0].value);
                uint32_t ow = ctx.width(a);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[1].value), ow);
                BVOp cop = (inst.op == spv::OpIEqual || inst.op == spv::OpLogicalEqual) ? BVOp::Eq
                         : (inst.op == spv::OpINotEqual || inst.op == spv::OpLogicalNotEqual) ? BVOp::Ne
                         : (inst.op == spv::OpSLessThan) ? BVOp::Slt : (inst.op == spv::OpSLessThanEqual) ? BVOp::Sle
                         : (inst.op == spv::OpSGreaterThan) ? BVOp::Sgt : (inst.op == spv::OpSGreaterThanEqual) ? BVOp::Sge
                         : (inst.op == spv::OpULessThan) ? BVOp::Ult : (inst.op == spv::OpULessThanEqual) ? BVOp::Ule
                         : (inst.op == spv::OpUGreaterThan) ? BVOp::Ugt : BVOp::Uge;
                res = bv_bin(ctx, cop, 1, a, b);
                break;
            }
            case spv::OpIsNan:
            case spv::OpIsInf:
                res = env.eval_uf(ctx, inst.op == spv::OpIsNan ? "fp_isnan" : "fp_isinf",
                                  {get_v(inst.operands[0].value)}, 1, false);
                break;
            case spv::OpOrdered:
            case spv::OpUnordered:
            case spv::OpFOrdEqual:
            case spv::OpFUnordEqual:
            case spv::OpFUnordNotEqual:
            case spv::OpFOrdNotEqual:
            case spv::OpFOrdLessThan:
            case spv::OpFUnordLessThan:
            case spv::OpFOrdLessThanEqual:
            case spv::OpFUnordLessThanEqual:
            case spv::OpFOrdGreaterThan:
            case spv::OpFUnordGreaterThan:
            case spv::OpFOrdGreaterThanEqual:
            case spv::OpFUnordGreaterThanEqual: {
                BVNodeId a = get_v(inst.operands[0].value);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[1].value), ctx.width(a));
                const char* cname = (inst.op == spv::OpFOrdEqual || inst.op == spv::OpFUnordEqual) ? "fp_cmp_eq"
                                  : (inst.op == spv::OpFUnordNotEqual || inst.op == spv::OpFOrdNotEqual) ? "fp_cmp_ne"
                                  : (inst.op == spv::OpFOrdLessThan || inst.op == spv::OpFUnordLessThan) ? "fp_cmp_slt"
                                  : (inst.op == spv::OpFOrdLessThanEqual || inst.op == spv::OpFUnordLessThanEqual) ? "fp_cmp_sle"
                                  : (inst.op == spv::OpFOrdGreaterThan || inst.op == spv::OpFUnordGreaterThan) ? "fp_cmp_sgt"
                                  : (inst.op == spv::OpOrdered || inst.op == spv::OpUnordered) ? "fp_cmp_ord" : "fp_cmp_sge";
                res = env.eval_uf(ctx, cname, {a, b}, 1, false);
                break;
            }
            case spv::OpSelect: {
                BVNodeId c = gpu_coerce_width(ctx, get_v(inst.operands[0].value), 1);
                if (rti && rti->op == spv::OpTypeVector &&
                    cur.composites.count(inst.operands[1].value) && cur.composites.count(inst.operands[2].value)) {
                    const auto& tv = cur.composites[inst.operands[1].value];
                    const auto& fv = cur.composites[inst.operands[2].value];
                    std::vector<BVNodeId> ov;
                    for (size_t k = 0; k < tv.size() && k < fv.size(); ++k) {
                        ov.push_back(bv_ite(ctx, ctx.width(tv[k]), c, tv[k], fv[k]));
                    }
                    cur.composites[inst.result] = std::move(ov);
                } else {
                    BVNodeId tv = gpu_coerce_width(ctx, get_v(inst.operands[1].value), rw);
                    BVNodeId fv = gpu_coerce_width(ctx, get_v(inst.operands[2].value), rw);
                    if (bv_node(ctx, fv).op == BVOp::SDiv || bv_node(ctx, fv).op == BVOp::SRem) {
                        const BVNode& cn = bv_node(ctx, c);
                        if (cn.op == BVOp::Eq && bv_node(ctx, cn.b).op == BVOp::Const &&
                            bv_node(ctx, cn.b).imm == width_mask(rw)) {
                            res = fv;
                            break;
                        }
                    }
                    res = bv_ite(ctx, rw, c, tv, fv);
                }
                break;
            }
            case spv::OpSConvert:
                res = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw, true);
                break;
            case spv::OpUConvert:
                res = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw, false);
                break;
            case spv::OpBitcast:
                res = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw, false);
                break;
            case spv::OpConvertSToF:
                res = env.eval_uf(ctx, "cvt_sitofp", {get_v(inst.operands[0].value)}, rw, true);
                break;
            case spv::OpConvertUToF:
                res = env.eval_uf(ctx, "cvt_uitofp", {get_v(inst.operands[0].value)}, rw, true);
                break;
            case spv::OpConvertFToS:
                res = env.eval_uf(ctx, "cvt_fptosi", {get_v(inst.operands[0].value)}, rw, false);
                break;
            case spv::OpConvertFToU:
                res = env.eval_uf(ctx, "cvt_fptoui", {get_v(inst.operands[0].value)}, rw, false);
                break;
            case spv::OpFConvert:
                res = env.eval_uf(ctx, rw == 64 ? "cvt_fpext" : "cvt_fptrunc", {get_v(inst.operands[0].value)}, rw, true);
                break;
            case spv::OpExtInst:
                if (!handle_glsl_ext_inst(sc, cur, inst, rw, res)) {
                    trace.supported = false;
                    trace.error_msg = "Unsupported GLSL.std.450 ext inst in GPU SPIR-V tracer";
                    return false;
                }
                break;
            case spv::OpControlBarrier:
            case spv::OpMemoryBarrier:
                trace.barriers.push_back({cur.cond, bv_const(ctx, 32, 0), false, sc.barrier_seq++});
                break;
            case spv::OpGroupNonUniformShuffle: {
                BVNodeId val32 = gpu_coerce_width(ctx, get_v(inst.operands[1].value), 32);
                BVNodeId src_lane = gpu_coerce_width(ctx, get_v(inst.operands[2].value), 32);
                trace.barriers.push_back({cur.cond, bv_const(ctx, 32, 0), true, sc.barrier_seq++});
                res = env.eval_uf(ctx, "gpu_shfl", {val32, src_lane}, 32, false);
                break;
            }
            case spv::OpGroupNonUniformElect:
                trace.barriers.push_back({cur.cond, bv_const(ctx, 32, 0), true, sc.barrier_seq++});
                res = env.eval_uf(ctx, "gpu_sg_elect", {env.raw_subgroup_lane}, 1, false);
                break;
            case spv::OpGroupNonUniformIAdd:
            case spv::OpGroupNonUniformFAdd:
            case spv::OpGroupNonUniformSMin:
            case spv::OpGroupNonUniformUMin:
            case spv::OpGroupNonUniformFMin:
            case spv::OpGroupNonUniformSMax:
            case spv::OpGroupNonUniformUMax:
            case spv::OpGroupNonUniformFMax: {
                trace.barriers.push_back({cur.cond, bv_const(ctx, 32, 0), true, sc.barrier_seq++});
                BVNodeId val = inst.operands.size() >= 3
                    ? gpu_coerce_width(ctx, get_v(inst.operands[2].value), rw)
                    : bv_const(ctx, rw, 0);
                res = env.eval_uf(ctx, "gpu_sg_red_" + std::to_string(inst.op), {val}, rw, is_fl);
                break;
            }
            case spv::OpAtomicIAdd:
            case spv::OpAtomicFAddEXT:
            case spv::OpAtomicSMin:
            case spv::OpAtomicUMin:
            case spv::OpAtomicSMax:
            case spv::OpAtomicUMax:
            case spv::OpAtomicExchange: {
                spirv::Id ptr_id = inst.operands[0].value;
                BVNodeId arg = gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw);
                uint32_t nb = std::max(1u, rw / 8u);
                bool is_sh = cur.chains.count(ptr_id) && cur.chains[ptr_id].is_workgroup;
                BVNodeId caddr = 0;
                if (is_sh) {
                    const auto& ch = cur.chains[ptr_id];
                    record_shared_access(ctx, trace, cur.cond, false, ch.index, ch.elem_idx, nb);
                    record_shared_access(ctx, trace, cur.cond, true, ch.index, ch.elem_idx, nb);
                    caddr = make_shared_canonical_addr(ctx, ch.index, ch.elem_idx, nb);
                } else {
                    caddr = gpu_coerce_width(ctx, get_v(ptr_id), 64);
                    record_global_access(ctx, trace, env, cur.cond, false, caddr, nb);
                    record_global_access(ctx, trace, env, cur.cond, true, caddr, nb);
                }
                auto& st_vec = is_sh ? trace.shared_stores : trace.global_stores;
                BVNodeId old_v = env.read_mem(ctx, is_sh, st_vec, caddr, nb);
                BVNodeId new_v = arg;
                if (inst.op == spv::OpAtomicIAdd) new_v = bv_bin(ctx, BVOp::Add, rw, old_v, arg);
                else if (inst.op == spv::OpAtomicFAddEXT) new_v = env.eval_uf(ctx, "fp_add", {old_v, arg}, rw, true);
                else if (inst.op == spv::OpAtomicSMin) new_v = bv_ite(ctx, rw, bv_bin(ctx, BVOp::Slt, 1, old_v, arg), old_v, arg);
                else if (inst.op == spv::OpAtomicUMin) new_v = bv_ite(ctx, rw, bv_bin(ctx, BVOp::Ult, 1, old_v, arg), old_v, arg);
                else if (inst.op == spv::OpAtomicSMax) new_v = bv_ite(ctx, rw, bv_bin(ctx, BVOp::Sgt, 1, old_v, arg), old_v, arg);
                else if (inst.op == spv::OpAtomicUMax) new_v = bv_ite(ctx, rw, bv_bin(ctx, BVOp::Ugt, 1, old_v, arg), old_v, arg);
                st_vec.push_back({cur.cond, caddr, nb, new_v});
                res = old_v;
                break;
            }
            case spv::OpFunctionCall:
                if (!handle_spirv_function_call(sc, cur, inst, call_depth, rw, is_fl, rti, res)) return false;
                break;
            case spv::OpBranch:
                route_edge(bb.label, inst.operands[0].value, cur.cond, cur);
                break;
            case spv::OpBranchConditional: {
                BVNodeId c1 = gpu_coerce_width(ctx, get_v(inst.operands[0].value), 1);
                BVNodeId t_cond = bv_bin(ctx, BVOp::And, 1, cur.cond, c1);
                BVNodeId f_cond = bv_bin(ctx, BVOp::And, 1, cur.cond, bv_un(ctx, BVOp::Not, 1, c1));
                const bool t_zero = (bv_node(ctx, t_cond).op == BVOp::Const && bv_node(ctx, t_cond).imm == 0);
                const bool f_zero = (bv_node(ctx, f_cond).op == BVOp::Const && bv_node(ctx, f_cond).imm == 0);
                if (!t_zero && !f_zero && is_loop_branch_block[cur.block_idx]) {
                    ++sym_visits[cur.block_idx];
                }
                route_edge(bb.label, inst.operands[1].value, t_cond, cur);
                route_edge(bb.label, inst.operands[2].value, f_cond, cur);
                break;
            }
            case spv::OpSwitch: {
                if (inst.operands.size() >= 2) {
                    BVNodeId sel = get_v(inst.operands[0].value);
                    uint32_t sw = ctx.width(sel);
                    spirv::Id def_lbl = inst.operands[1].value;
                    BVNodeId rem_cond = cur.cond;
                    for (size_t k = 2; k + 1 < inst.operands.size(); k += 2) {
                        uint32_t case_lit = inst.operands[k].value;
                        spirv::Id case_lbl = inst.operands[k + 1].value;
                        BVNodeId eq_c = bv_bin(ctx, BVOp::Eq, 1, sel, bv_const(ctx, sw, case_lit));
                        BVNodeId take = bv_bin(ctx, BVOp::And, 1, rem_cond, eq_c);
                        rem_cond = bv_bin(ctx, BVOp::And, 1, rem_cond, bv_un(ctx, BVOp::Not, 1, eq_c));
                        route_edge(bb.label, case_lbl, take, cur);
                    }
                    route_edge(bb.label, def_lbl, rem_cond, cur);
                }
                break;
            }
            case spv::OpReturn:
                if (call_depth == 0) {
                    trace.returned = bv_bin(ctx, BVOp::Or, 1, trace.returned, cur.cond);
                }
                if (out_local_slots) {
                    for (const auto& [sk, sv] : cur.local_slots) {
                        auto it = out_local_slots->find(sk);
                        if (it == out_local_slots->end()) (*out_local_slots)[sk] = sv;
                        else if (it->second != sv && ctx.width(it->second) == ctx.width(sv)) {
                            it->second = bv_ite(ctx, ctx.width(sv), cur.cond, sv, it->second);
                        }
                    }
                }
                seen_ret = bv_bin(ctx, BVOp::Or, 1, seen_ret, cur.cond);
                break;
            case spv::OpReturnValue:
                if (call_depth == 0) {
                    trace.returned = bv_bin(ctx, BVOp::Or, 1, trace.returned, cur.cond);
                }
                if (out_local_slots) {
                    for (const auto& [sk, sv] : cur.local_slots) {
                        auto it = out_local_slots->find(sk);
                        if (it == out_local_slots->end()) (*out_local_slots)[sk] = sv;
                        else if (it->second != sv && ctx.width(it->second) == ctx.width(sv)) {
                            it->second = bv_ite(ctx, ctx.width(sv), cur.cond, sv, it->second);
                        }
                    }
                }
                if (!inst.operands.empty()) {
                    spirv::Id rvid = inst.operands[0].value;
                    if (cur.composites.count(rvid)) {
                        const auto& rc = cur.composites[rvid];
                        if (out_ret_comp.empty()) out_ret_comp = rc;
                        else {
                            for (size_t c = 0; c < rc.size() && c < out_ret_comp.size(); ++c) {
                                out_ret_comp[c] = bv_ite(ctx, ctx.width(rc[c]), cur.cond, rc[c], out_ret_comp[c]);
                            }
                        }
                    } else {
                        BVNodeId rv = get_v(rvid);
                        out_ret_val = (out_ret_val == 0)
                            ? rv : bv_ite(ctx, ctx.width(rv), cur.cond, rv, out_ret_val);
                    }
                }
                seen_ret = bv_bin(ctx, BVOp::Or, 1, seen_ret, cur.cond);
                break;
            case spv::OpUnreachable:
                trace.trapped = bv_bin(ctx, BVOp::Or, 1, trace.trapped, cur.cond);
                break;
            case spv::OpNop:
            case spv::OpFunctionEnd:
                break;
            default:
                trace.supported = false;
                trace.error_msg = "Unsupported SPIR-V opcode in GPU tracer: " + std::to_string(inst.op);
                return false;
            }
            if (inst.result && res != UINT32_MAX) {
                cur.vals[inst.result] = res;
            }
        }
    }
    return true;
}

} // namespace spirv_detail

GpuExecTrace execute_gpu_spirv(
    BVContext& ctx,
    const spirv::Module& mod,
    GpuSymEnv& env,
    const EquivOptions& opts) {
    GpuExecTrace trace;
    trace.completed = bv_true(ctx);
    trace.trapped = bv_false(ctx);
    trace.returned = bv_false(ctx);
    trace.domain_assumption = bv_true(ctx);

    const spirv::Function* entry_fn = find_spirv_entry_function(mod);
    if (!entry_fn || entry_fn->blocks.empty()) {
        trace.supported = false;
        trace.error_msg = "SPIR-V module has no entry function or blocks";
        return trace;
    }

    SpirvModuleInfo minfo;
    collect_spirv_module_info(ctx, mod, env, trace, minfo);

    SpirvExecState init;
    init.block_idx = 0;
    init.cond = bv_true(ctx);
    init.vals = minfo.global_vals;
    init.composites = minfo.global_composites;

    SpirvExecContext sc{ctx, mod, env, opts, minfo, trace, {}, {}, {}, 0u, 0u, 0u};
    sc.step_budget = std::max<uint32_t>(
        opts.max_paths * 32u,
        static_cast<uint32_t>(entry_fn->blocks.size()) * (opts.loop_unroll_bound + 4u) * 8u);

    BVNodeId dummy_ret = 0;
    std::vector<BVNodeId> dummy_comp;
    execute_spirv_function_body(sc, *entry_fn, std::move(init), 0u, dummy_ret, dummy_comp);
    return trace;
}

} // namespace brass::verify
