#include "gpu_equiv_internal.hpp"
#include <spirv/unified1/GLSL.std.450.h>
#include <unordered_map>

namespace brass::verify {

namespace {

uint32_t spirv_type_width(const spirv::Module& mod, spirv::Id type_id, bool* out_is_float = nullptr) {
    if (out_is_float) *out_is_float = false;
    const spirv::TypeInfo* ti = mod.type_info(type_id);
    if (!ti) return 32;
    if (ti->op == spv::OpTypeBool) return 1;
    if (ti->op == spv::OpTypeInt) return ti->width ? ti->width : 32;
    if (ti->op == spv::OpTypeFloat) {
        if (out_is_float) *out_is_float = true;
        return ti->width ? ti->width : 32;
    }
    if (ti->op == spv::OpTypePointer) return 64;
    return 32;
}

} // namespace

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

    if (mod.functions.empty() || mod.functions[0].blocks.empty()) {
        trace.supported = false;
        trace.error_msg = "SPIR-V module has no entry function or blocks";
        return trace;
    }

    std::unordered_map<spirv::Id, uint32_t> builtin_map;
    std::unordered_map<spirv::Id, uint32_t> spec_id_map;
    std::unordered_map<spirv::Id, uint32_t> location_map;
    for (const auto& ann : mod.annotations) {
        if (ann.op == spv::OpDecorate && ann.operands.size() >= 3) {
            spirv::Id target = ann.operands[0].value;
            uint32_t dec = ann.operands[1].value;
            uint32_t val = ann.operands[2].value;
            if (dec == spv::DecorationBuiltIn) builtin_map[target] = val;
            else if (dec == spv::DecorationSpecId) spec_id_map[target] = val;
            else if (dec == spv::DecorationLocation) location_map[target] = val;
        }
    }

    // If env was not initialized with param_types (e.g. standalone SPIR-V safety check),
    // infer param_types from the PushConstant struct in mod.globals.
    if (env.param_vars.empty()) {
        std::unordered_map<spirv::Id, std::vector<spirv::Id>> struct_members;
        spirv::Id pc_struct_id = 0;
        for (const auto& g : mod.globals) {
            if (g.op == spv::OpTypeStruct) {
                std::vector<spirv::Id> mems;
                for (const auto& op : g.operands) mems.push_back(op.value);
                struct_members[g.result] = std::move(mems);
            } else if (g.op == spv::OpVariable && !g.operands.empty() &&
                       g.operands[0].value == spv::StorageClassPushConstant) {
                const spirv::TypeInfo* pti = mod.type_info(g.type);
                if (pti && pti->op == spv::OpTypePointer) pc_struct_id = pti->elem;
            }
        }
        std::vector<Type> inferred;
        if (pc_struct_id && struct_members.count(pc_struct_id)) {
            for (spirv::Id mt : struct_members[pc_struct_id]) {
                bool is_fl = false;
                uint32_t w = spirv_type_width(mod, mt, &is_fl);
                if (is_fl) inferred.push_back(w == 64 ? Type::f64() : Type::f32());
                else inferred.push_back(w == 64 ? Type::ptr() : Type::i32());
            }
        }
        if (env.coord_bounds == 0) {
            init_gpu_sym_env(ctx, env, inferred);
        } else {
            for (size_t i = 0; i < inferred.size(); ++i) {
                uint32_t w = gpu_type_bv_width(inferred[i]);
                env.param_vars.push_back(ctx.make_var("gpu_param_" + std::to_string(i), w ? w : 64));
            }
            env.param_types = inferred;
        }
    }

    std::unordered_map<spirv::Id, BVNodeId> global_vals;
    std::unordered_map<spirv::Id, std::vector<BVNodeId>> global_composites;
    std::unordered_map<spirv::Id, std::vector<BVNodeId>> var_composites;
    std::unordered_map<spirv::Id, BVNodeId> var_scalars;
    std::unordered_map<spirv::Id, uint32_t> workgroup_var_idx;
    std::unordered_map<spirv::Id, uint64_t> stage_out_base;
    spirv::Id push_const_var = 0;

    for (const auto& g : mod.globals) {
        switch (g.op) {
        case spv::OpConstant: {
            uint32_t w = spirv_type_width(mod, g.type);
            uint64_t bits = 0;
            if (!g.operands.empty()) bits = g.operands[0].value;
            if (w == 64 && g.operands.size() >= 2) {
                bits |= (static_cast<uint64_t>(g.operands[1].value) << 32);
            }
            global_vals[g.result] = bv_const(ctx, w, bits);
            break;
        }
        case spv::OpConstantTrue:
            global_vals[g.result] = bv_true(ctx);
            break;
        case spv::OpConstantFalse:
            global_vals[g.result] = bv_false(ctx);
            break;
        case spv::OpConstantNull:
            global_vals[g.result] = bv_const(ctx, spirv_type_width(mod, g.type), 0);
            break;
        case spv::OpSpecConstant: {
            auto it = spec_id_map.find(g.result);
            if (it != spec_id_map.end() && it->second < 3) {
                global_vals[g.result] = env.ntid[it->second];
            } else {
                uint32_t def_v = g.operands.empty() ? 1u : g.operands[0].value;
                global_vals[g.result] = bv_const(ctx, 32, def_v);
            }
            break;
        }
        case spv::OpConstantComposite:
        case spv::OpSpecConstantComposite: {
            auto bit = builtin_map.find(g.result);
            if (bit != builtin_map.end() && bit->second == spv::BuiltInWorkgroupSize) {
                global_composites[g.result] = {env.ntid[0], env.ntid[1], env.ntid[2]};
            } else {
                std::vector<BVNodeId> parts;
                for (const auto& op : g.operands) {
                    parts.push_back(global_vals.count(op.value) ? global_vals[op.value] : bv_const(ctx, 32, 0));
                }
                global_composites[g.result] = std::move(parts);
            }
            break;
        }
        case spv::OpVariable: {
            uint32_t sc = g.operands.empty() ? 0 : g.operands[0].value;
            if (sc == spv::StorageClassInput) {
                uint32_t bi = builtin_map.count(g.result) ? builtin_map[g.result] : UINT32_MAX;
                if (bi == spv::BuiltInLocalInvocationId) {
                    var_composites[g.result] = {env.tid[0], env.tid[1], env.tid[2]};
                } else if (bi == spv::BuiltInWorkgroupId) {
                    var_composites[g.result] = {env.ctaid[0], env.ctaid[1], env.ctaid[2]};
                } else if (bi == spv::BuiltInNumWorkgroups) {
                    var_composites[g.result] = {env.nctaid[0], env.nctaid[1], env.nctaid[2]};
                } else if (bi == spv::BuiltInGlobalInvocationId) {
                    var_composites[g.result] = {env.global_tid[0], env.global_tid[1], env.global_tid[2]};
                } else if (bi == spv::BuiltInWorkgroupSize) {
                    var_composites[g.result] = {env.ntid[0], env.ntid[1], env.ntid[2]};
                } else if (bi == spv::BuiltInSubgroupLocalInvocationId) {
                    var_scalars[g.result] = env.raw_subgroup_lane;
                } else if (bi == spv::BuiltInVertexIndex) {
                    var_scalars[g.result] = env.eval_uf(ctx, "spirv_vertex_index", {}, 32, false);
                } else if (bi == spv::BuiltInInstanceIndex) {
                    var_scalars[g.result] = env.eval_uf(ctx, "spirv_instance_index", {}, 32, false);
                } else if (bi == spv::BuiltInFragCoord) {
                    var_composites[g.result] = {
                        env.eval_uf(ctx, "spirv_frag_coord", {bv_const(ctx, 32, 0)}, 32, true),
                        env.eval_uf(ctx, "spirv_frag_coord", {bv_const(ctx, 32, 1)}, 32, true),
                        env.eval_uf(ctx, "spirv_frag_coord", {bv_const(ctx, 32, 2)}, 32, true),
                        env.eval_uf(ctx, "spirv_frag_coord", {bv_const(ctx, 32, 3)}, 32, true)};
                } else if (location_map.count(g.result)) {
                    uint32_t loc = location_map[g.result];
                    var_composites[g.result] = {
                        env.eval_uf(ctx, "spirv_stage_in", {bv_const(ctx, 32, loc), bv_const(ctx, 32, 0)}, 32, true),
                        env.eval_uf(ctx, "spirv_stage_in", {bv_const(ctx, 32, loc), bv_const(ctx, 32, 1)}, 32, true),
                        env.eval_uf(ctx, "spirv_stage_in", {bv_const(ctx, 32, loc), bv_const(ctx, 32, 2)}, 32, true),
                        env.eval_uf(ctx, "spirv_stage_in", {bv_const(ctx, 32, loc), bv_const(ctx, 32, 3)}, 32, true)};
                }
            } else if (sc == spv::StorageClassOutput) {
                uint32_t bi = builtin_map.count(g.result) ? builtin_map[g.result] : UINT32_MAX;
                if (bi == spv::BuiltInPosition) {
                    stage_out_base[g.result] = 0x6000000000000000ULL;
                } else if (location_map.count(g.result)) {
                    stage_out_base[g.result] = 0x6000000000001000ULL + static_cast<uint64_t>(location_map[g.result]) * 16ULL;
                }
            } else if (sc == spv::StorageClassPushConstant) {
                push_const_var = g.result;
            } else if (sc == spv::StorageClassWorkgroup) {
                uint32_t arr_idx = static_cast<uint32_t>(trace.shared_allocs.size());
                uint32_t cnt = 256, ebytes = 4;
                const spirv::TypeInfo* pti = mod.type_info(g.type);
                if (pti && pti->op == spv::OpTypePointer) {
                    const spirv::TypeInfo* ati = mod.type_info(pti->elem);
                    if (ati && ati->op == spv::OpTypeArray) {
                        cnt = ati->width;
                        if (global_vals.count(cnt) && bv_node(ctx, global_vals[cnt]).op == BVOp::Const) {
                            cnt = static_cast<uint32_t>(bv_node(ctx, global_vals[cnt]).imm);
                        }
                        ebytes = spirv_type_width(mod, ati->elem) / 8;
                        if (ebytes == 0) ebytes = 4;
                    }
                }
                uint64_t sym_base = kGpuSharedBase0 + static_cast<uint64_t>(arr_idx) * kGpuSharedStride;
                trace.shared_allocs.push_back({arr_idx, cnt, ebytes, sym_base});
                workgroup_var_idx[g.result] = arr_idx;
            }
            break;
        }
        default:
            break;
        }
    }

    const spirv::Function& fn = mod.functions[0];
    std::unordered_map<spirv::Id, size_t> label_to_idx;
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        label_to_idx[fn.blocks[i].label] = i;
    }

    struct AccessChainInfo {
        bool is_push_const = false;
        bool is_workgroup = false;
        uint32_t index = 0;
        BVNodeId elem_idx = 0;
    };

    struct ActiveState {
        size_t block_idx = 0;
        BVNodeId cond = 0;
        std::unordered_map<spirv::Id, BVNodeId> vals = {};
        std::unordered_map<spirv::Id, std::vector<BVNodeId>> composites = {};
        std::unordered_map<spirv::Id, AccessChainInfo> chains = {};
    };

    std::vector<ActiveState> worklist;
    ActiveState init;
    init.block_idx = 0;
    init.cond = bv_true(ctx);
    init.vals = global_vals;
    init.composites = global_composites;
    worklist.push_back(std::move(init));

    std::unordered_map<size_t, uint32_t> visits;
    uint32_t max_visits = opts.loop_unroll_bound + 2;
    uint32_t total_steps = 0;
    uint32_t barrier_seq = 0;
    std::unordered_map<BVNodeId, BVNodeId> safe_div_orig;

    auto route_edge = [&](spirv::Id from_label, spirv::Id to_label, BVNodeId edge_cond,
                          const ActiveState& cur_st) {
        if (bv_node(ctx, edge_cond).op == BVOp::Const && bv_node(ctx, edge_cond).imm == 0) return;
        auto bit = label_to_idx.find(to_label);
        if (bit == label_to_idx.end()) return;
        size_t dst_idx = bit->second;
        const spirv::Block& dst_bb = fn.blocks[dst_idx];

        auto next_vals = cur_st.vals;
        for (const auto& inst : dst_bb.insts) {
            if (inst.op != spv::OpPhi) break;
            for (size_t i = 0; i + 1 < inst.operands.size(); i += 2) {
                if (inst.operands[i + 1].value == from_label) {
                    spirv::Id val_id = inst.operands[i].value;
                    auto vit = cur_st.vals.find(val_id);
                    if (vit != cur_st.vals.end()) next_vals[inst.result] = vit->second;
                    break;
                }
            }
        }

        for (auto& st : worklist) {
            if (st.block_idx == dst_idx) {
                BVNodeId merged = bv_bin(ctx, BVOp::Or, 1, st.cond, edge_cond);
                for (auto& [vid, vnode] : next_vals) {
                    auto it = st.vals.find(vid);
                    if (it == st.vals.end()) {
                        st.vals[vid] = vnode;
                    } else if (it->second != vnode && ctx.width(it->second) == ctx.width(vnode)) {
                        it->second = bv_ite(ctx, ctx.width(vnode), edge_cond, vnode, it->second);
                    }
                }
                for (auto& [cid, cvec] : cur_st.composites) st.composites[cid] = cvec;
                for (auto& [aid, ainfo] : cur_st.chains) st.chains[aid] = ainfo;
                st.cond = merged;
                return;
            }
        }
        worklist.push_back({dst_idx, edge_cond, std::move(next_vals), cur_st.composites, cur_st.chains});
    };

    while (!worklist.empty()) {
        if (++total_steps > opts.max_paths * 2) {
            trace.supported = false;
            trace.error_msg = "SPIR-V symbolic execution exceeded block step budget";
            return trace;
        }
        ActiveState cur = std::move(worklist.front());
        worklist.erase(worklist.begin());
        if (bv_node(ctx, cur.cond).op == BVOp::Const && bv_node(ctx, cur.cond).imm == 0) continue;

        if (++visits[cur.block_idx] > max_visits) {
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
            if (inst.op == spv::OpPhi || inst.op == spv::OpSelectionMerge || inst.op == spv::OpLoopMerge) {
                continue;
            }
            bool is_fl = false;
            uint32_t rw = inst.type ? spirv_type_width(mod, inst.type, &is_fl) : 32;
            BVNodeId res = 0;

            switch (inst.op) {
            case spv::OpAccessChain:
            case spv::OpInBoundsAccessChain: {
                if (inst.operands.size() >= 2) {
                    spirv::Id base_id = inst.operands[0].value;
                    spirv::Id idx_id = inst.operands[1].value;
                    if (base_id == push_const_var) {
                        BVNodeId idx_v = get_v(idx_id);
                        uint32_t pidx = (bv_node(ctx, idx_v).op == BVOp::Const)
                            ? static_cast<uint32_t>(bv_node(ctx, idx_v).imm) : 0;
                        cur.chains[inst.result] = {true, false, pidx, 0};
                    } else if (workgroup_var_idx.count(base_id)) {
                        cur.chains[inst.result] = {
                            false, true, workgroup_var_idx[base_id],
                            gpu_coerce_width(ctx, get_v(idx_id), 32, false)};
                    } else if (stage_out_base.count(base_id)) {
                        BVNodeId idx_v = get_v(idx_id);
                        uint64_t cidx = (bv_node(ctx, idx_v).op == BVOp::Const) ? bv_node(ctx, idx_v).imm : 0;
                        stage_out_base[inst.result] = stage_out_base[base_id] + cidx * 4ULL;
                    }
                }
                break;
            }
            case spv::OpConvertUToPtr:
                res = gpu_coerce_width(ctx, get_v(inst.operands[0].value), 64, false);
                break;
            case spv::OpConvertPtrToU:
                res = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw, false);
                break;
            case spv::OpLoad: {
                spirv::Id ptr_id = inst.operands[0].value;
                if (var_composites.count(ptr_id)) {
                    cur.composites[inst.result] = var_composites[ptr_id];
                } else if (var_scalars.count(ptr_id)) {
                    res = var_scalars[ptr_id];
                } else if (cur.chains.count(ptr_id)) {
                    const auto& ch = cur.chains[ptr_id];
                    if (ch.is_push_const) {
                        res = (ch.index < env.param_vars.size())
                            ? gpu_coerce_width(ctx, env.param_vars[ch.index], rw)
                            : bv_const(ctx, rw, 0);
                    } else if (ch.is_workgroup) {
                        uint32_t nb = rw / 8;
                        record_shared_access(ctx, trace, cur.cond, false, ch.index, ch.elem_idx, nb);
                        BVNodeId caddr = make_shared_canonical_addr(ctx, ch.index, ch.elem_idx, nb);
                        res = env.read_mem(ctx, true, trace.shared_stores, caddr, nb);
                    }
                } else {
                    BVNodeId addr = gpu_coerce_width(ctx, get_v(ptr_id), 64);
                    uint32_t nb = rw / 8;
                    uint32_t align = (inst.operands.size() >= 3 && (inst.operands[1].value & spv::MemoryAccessAlignedMask))
                        ? inst.operands[2].value : nb;
                    record_global_access(ctx, trace, env, cur.cond, false, addr, nb, align);
                    res = env.read_mem(ctx, false, trace.global_stores, addr, nb);
                }
                break;
            }
            case spv::OpStore: {
                spirv::Id ptr_id = inst.operands[0].value;
                spirv::Id val_id = inst.operands[1].value;
                BVNodeId val = get_v(val_id);
                uint32_t nb = ctx.width(val) / 8;
                if (nb == 0) nb = 4;
                if (stage_out_base.count(ptr_id)) {
                    uint64_t base_addr = stage_out_base[ptr_id];
                    if (cur.composites.count(val_id)) {
                        const auto& comp = cur.composites[val_id];
                        for (size_t c = 0; c < comp.size(); ++c) {
                            BVNodeId cv = gpu_coerce_width(ctx, comp[c], 32);
                            trace.global_stores.push_back({cur.cond, bv_const(ctx, 64, base_addr + c * 4), 4, cv});
                        }
                    } else {
                        BVNodeId cv = gpu_coerce_width(ctx, val, 32);
                        trace.global_stores.push_back({cur.cond, bv_const(ctx, 64, base_addr), 4, cv});
                    }
                } else if (cur.chains.count(ptr_id) && cur.chains[ptr_id].is_workgroup) {
                    const auto& ch = cur.chains[ptr_id];
                    record_shared_access(ctx, trace, cur.cond, true, ch.index, ch.elem_idx, nb);
                    BVNodeId caddr = make_shared_canonical_addr(ctx, ch.index, ch.elem_idx, nb);
                    trace.shared_stores.push_back({cur.cond, caddr, nb, val});
                } else {
                    BVNodeId addr = gpu_coerce_width(ctx, get_v(ptr_id), 64);
                    uint32_t align = (inst.operands.size() >= 4 && (inst.operands[2].value & spv::MemoryAccessAlignedMask))
                        ? inst.operands[3].value : nb;
                    record_global_access(ctx, trace, env, cur.cond, true, addr, nb, align);
                    trace.global_stores.push_back({cur.cond, addr, nb, val});
                }
                break;
            }
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
            case spv::OpIAdd:
            case spv::OpISub:
            case spv::OpIMul: {
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[1].value), rw);
                if (inst.op == spv::OpIAdd && rw == 64) {
                    res = gpu_add_addr64(ctx, a, b);
                } else if (inst.op == spv::OpIMul) {
                    res = gpu_mul_or_shl(ctx, rw, a, b);
                } else if (inst.op == spv::OpISub && safe_div_orig.count(b)) {
                    res = bv_bin(ctx, BVOp::SRem, rw, a, safe_div_orig[b]);
                } else {
                    res = bv_bin(ctx, inst.op == spv::OpIAdd ? BVOp::Add : BVOp::Sub, rw, a, b);
                }
                if (inst.op == spv::OpIMul && safe_div_orig.count(a)) {
                    safe_div_orig[res] = safe_div_orig[a];
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
                if (orig_d != b) safe_div_orig[res] = orig_d;
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
                BVNodeId a = gpu_coerce_width(ctx, get_v(inst.operands[0].value), rw);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[1].value), rw);
                const char* uf = (inst.op == spv::OpFAdd) ? "fp_add" : (inst.op == spv::OpFSub) ? "fp_sub"
                               : (inst.op == spv::OpFMul) ? "fp_mul" : (inst.op == spv::OpFDiv) ? "fp_div" : "fp_rem";
                res = env.eval_uf(ctx, uf, {a, b}, rw, true);
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
                BVOp cop = (inst.op == spv::OpIEqual) ? BVOp::Eq : (inst.op == spv::OpINotEqual) ? BVOp::Ne
                         : (inst.op == spv::OpSLessThan) ? BVOp::Slt : (inst.op == spv::OpSLessThanEqual) ? BVOp::Sle
                         : (inst.op == spv::OpSGreaterThan) ? BVOp::Sgt : (inst.op == spv::OpSGreaterThanEqual) ? BVOp::Sge
                         : (inst.op == spv::OpULessThan) ? BVOp::Ult : (inst.op == spv::OpULessThanEqual) ? BVOp::Ule
                         : (inst.op == spv::OpUGreaterThan) ? BVOp::Ugt : BVOp::Uge;
                res = bv_bin(ctx, cop, 1, a, b);
                break;
            }
            case spv::OpFOrdEqual:
            case spv::OpFUnordNotEqual:
            case spv::OpFOrdNotEqual:
            case spv::OpFOrdLessThan:
            case spv::OpFOrdLessThanEqual:
            case spv::OpFOrdGreaterThan:
            case spv::OpFOrdGreaterThanEqual: {
                BVNodeId a = get_v(inst.operands[0].value);
                BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[1].value), ctx.width(a));
                const char* cname = (inst.op == spv::OpFOrdEqual) ? "fp_cmp_eq"
                                  : (inst.op == spv::OpFUnordNotEqual || inst.op == spv::OpFOrdNotEqual) ? "fp_cmp_ne"
                                  : (inst.op == spv::OpFOrdLessThan) ? "fp_cmp_slt"
                                  : (inst.op == spv::OpFOrdLessThanEqual) ? "fp_cmp_sle"
                                  : (inst.op == spv::OpFOrdGreaterThan) ? "fp_cmp_sgt" : "fp_cmp_sge";
                res = env.eval_uf(ctx, cname, {a, b}, 1, false);
                break;
            }
            case spv::OpSelect: {
                BVNodeId c = gpu_coerce_width(ctx, get_v(inst.operands[0].value), 1);
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
            case spv::OpExtInst: {
                if (inst.operands.size() >= 3) {
                    uint32_t glsl_op = inst.operands[1].value;
                    spirv::Id arg0_id = inst.operands[2].value;
                    BVNodeId a0 = get_v(arg0_id);
                    switch (glsl_op) {
                    case GLSLstd450Fma:
                        res = env.eval_uf(ctx, "fp_fma", {
                            gpu_coerce_width(ctx, a0, rw),
                            gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw),
                            gpu_coerce_width(ctx, get_v(inst.operands[4].value), rw)}, rw, true);
                        break;
                    case GLSLstd450Sqrt:      res = env.eval_uf(ctx, "gpu_sqrt", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450InverseSqrt: res = env.eval_uf(ctx, "gpu_rsqrt", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450Floor:     res = env.eval_uf(ctx, "gpu_floor", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450Ceil:      res = env.eval_uf(ctx, "gpu_ceil", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450RoundEven: res = env.eval_uf(ctx, "gpu_round", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450FAbs:
                        res = bv_bin(ctx, BVOp::And, rw, gpu_coerce_width(ctx, a0, rw), bv_const(ctx, rw, ~(1ULL << (rw - 1))));
                        break;
                    case GLSLstd450NMin:
                    case GLSLstd450FMin:
                        res = env.eval_uf(ctx, "gpu_fmin", {gpu_coerce_width(ctx, a0, rw), gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw)}, rw, true);
                        break;
                    case GLSLstd450NMax:
                    case GLSLstd450FMax:
                        res = env.eval_uf(ctx, "gpu_fmax", {gpu_coerce_width(ctx, a0, rw), gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw)}, rw, true);
                        break;
                    case GLSLstd450Sin:  res = env.eval_uf(ctx, "gpu_sin", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450Cos:  res = env.eval_uf(ctx, "gpu_cos", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450Exp2: res = env.eval_uf(ctx, "gpu_ex2", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450Log2: res = env.eval_uf(ctx, "gpu_lg2", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450Exp:  res = env.eval_uf(ctx, "gpu_exp", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450Log:  res = env.eval_uf(ctx, "gpu_log", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450Tanh: res = env.eval_uf(ctx, "gpu_tanh", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450FSign: res = env.eval_uf(ctx, "gpu_sign", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450Fract: res = env.eval_uf(ctx, "gpu_fract", {gpu_coerce_width(ctx, a0, rw)}, rw, true); break;
                    case GLSLstd450Step:
                        res = env.eval_uf(ctx, "gpu_step", {
                            gpu_coerce_width(ctx, a0, rw),
                            gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw)}, rw, true);
                        break;
                    case GLSLstd450FClamp:
                    case GLSLstd450NClamp:
                        res = env.eval_uf(ctx, "gpu_clamp", {
                            gpu_coerce_width(ctx, a0, rw),
                            gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw),
                            gpu_coerce_width(ctx, get_v(inst.operands[4].value), rw)}, rw, true);
                        break;
                    case GLSLstd450FMix:
                        res = env.eval_uf(ctx, "gpu_mix", {
                            gpu_coerce_width(ctx, a0, rw),
                            gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw),
                            gpu_coerce_width(ctx, get_v(inst.operands[4].value), rw)}, rw, true);
                        break;
                    case GLSLstd450SmoothStep:
                        res = env.eval_uf(ctx, "gpu_smoothstep", {
                            gpu_coerce_width(ctx, a0, rw),
                            gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw),
                            gpu_coerce_width(ctx, get_v(inst.operands[4].value), rw)}, rw, true);
                        break;
                    case GLSLstd450UnpackHalf2x16: {
                        BVNodeId lo = env.eval_uf(ctx, "gpu_f16_to_f32", {gpu_coerce_width(ctx, a0, 32)}, 32, true);
                        cur.composites[inst.result] = {lo, bv_const(ctx, 32, 0)};
                        break;
                    }
                    case GLSLstd450PackHalf2x16: {
                        BVNodeId elem0 = cur.composites.count(arg0_id) && !cur.composites[arg0_id].empty()
                            ? cur.composites[arg0_id][0] : a0;
                        res = env.eval_uf(ctx, "gpu_f32_to_f16", {gpu_coerce_width(ctx, elem0, 32)}, 32, false);
                        break;
                    }
                    default:
                        break;
                    }
                }
                break;
            }
            case spv::OpControlBarrier:
                trace.barriers.push_back({cur.cond, bv_const(ctx, 32, 0), false, barrier_seq++});
                break;
            case spv::OpGroupNonUniformShuffle: {
                BVNodeId val32 = gpu_coerce_width(ctx, get_v(inst.operands[1].value), 32);
                BVNodeId src_lane = gpu_coerce_width(ctx, get_v(inst.operands[2].value), 32);
                trace.barriers.push_back({cur.cond, bv_const(ctx, 32, 0), true, barrier_seq++});
                res = env.eval_uf(ctx, "gpu_shfl", {val32, src_lane}, 32, false);
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
                uint32_t nb = rw / 8;
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
            case spv::OpBranch:
                route_edge(bb.label, inst.operands[0].value, cur.cond, cur);
                break;
            case spv::OpBranchConditional: {
                BVNodeId c1 = gpu_coerce_width(ctx, get_v(inst.operands[0].value), 1);
                BVNodeId t_cond = bv_bin(ctx, BVOp::And, 1, cur.cond, c1);
                BVNodeId f_cond = bv_bin(ctx, BVOp::And, 1, cur.cond, bv_un(ctx, BVOp::Not, 1, c1));
                route_edge(bb.label, inst.operands[1].value, t_cond, cur);
                route_edge(bb.label, inst.operands[2].value, f_cond, cur);
                break;
            }
            case spv::OpReturn:
                trace.returned = bv_bin(ctx, BVOp::Or, 1, trace.returned, cur.cond);
                break;
            case spv::OpUnreachable:
                trace.trapped = bv_bin(ctx, BVOp::Or, 1, trace.trapped, cur.cond);
                break;
            default:
                break;
            }
            if (inst.result && res != 0) {
                cur.vals[inst.result] = res;
            }
        }
    }
    return trace;
}

} // namespace brass::verify
