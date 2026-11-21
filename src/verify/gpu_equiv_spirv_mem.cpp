#include "gpu_equiv_spirv_internal.hpp"
#include <algorithm>

namespace brass::verify::spirv_detail {

namespace {

bool resolve_local_chain(
    const spirv::Module& mod,
    const SpirvExecState& cur,
    const SpirvModuleInfo& minfo,
    spirv::Id base_id,
    bool& out_is_root,
    uint64_t& out_base,
    spirv::Id& out_ty,
    BVNodeId& out_elem_idx,
    uint32_t& out_elem_bytes) {
    (void)mod;
    if (auto it = cur.local_vars.find(base_id); it != cur.local_vars.end()) {
        out_is_root = true;
        out_base = it->second.base_addr;
        out_ty = it->second.pointee_ty;
        out_elem_idx = 0;
        out_elem_bytes = it->second.elem_bytes;
        return true;
    }
    if (auto it = minfo.private_vars.find(base_id); it != minfo.private_vars.end()) {
        out_is_root = true;
        out_base = it->second.base_addr;
        out_ty = it->second.pointee_ty;
        out_elem_idx = 0;
        out_elem_bytes = it->second.elem_bytes;
        return true;
    }
    if (auto cit = cur.chains.find(base_id); cit != cur.chains.end() && cit->second.is_local) {
        out_is_root = false;
        out_base = cit->second.local_base;
        out_ty = cit->second.local_cur_ty;
        out_elem_idx = cit->second.elem_idx;
        out_elem_bytes = cit->second.local_elem_bytes;
        return true;
    }
    return false;
}

uint32_t container_dim(const SpirvModuleInfo& minfo, BVContext& ctx, spirv::Id ty_id, const spirv::TypeInfo* ti) {
    if (!ti) return 1u;
    if (ti->op == spv::OpTypeArray) return std::max(1u, minfo.resolve_array_dim(ctx, ty_id, ti));
    if (ti->op == spv::OpTypeVector || ti->op == spv::OpTypeMatrix) return std::max(1u, ti->width);
    return 1u;
}

} // namespace

void handle_spirv_access_chain(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    const spirv::Inst& inst,
    const spirv::TypeInfo* rti,
    BVNodeId& out_res) {
    if (inst.operands.size() < 2) return;
    BVContext& ctx = sc.ctx;
    const spirv::Module& mod = sc.mod;
    SpirvModuleInfo& minfo = sc.minfo;
    auto get_v = [&](spirv::Id id) -> BVNodeId {
        auto it = cur.vals.find(id);
        return (it != cur.vals.end()) ? it->second : bv_const(ctx, 32, 0);
    };

    spirv::Id base_id = inst.operands[0].value;
    spirv::Id idx_id = inst.operands[1].value;
    bool is_root = false;
    uint64_t loc_base = 0;
    spirv::Id loc_ty = 0;
    BVNodeId loc_idx = 0;
    uint32_t loc_eb = 4;

    if (base_id == minfo.push_const_var) {
        BVNodeId idx_v = get_v(idx_id);
        uint32_t pidx = (bv_node(ctx, idx_v).op == BVOp::Const)
            ? static_cast<uint32_t>(bv_node(ctx, idx_v).imm) : 0;
        SpirvAccessChainInfo info;
        info.is_push_const = true;
        info.index = pidx;
        cur.chains[inst.result] = info;
    } else if (minfo.workgroup_var_idx.count(base_id) ||
               (cur.chains.count(base_id) && cur.chains[base_id].is_workgroup)) {
        const bool wg_root = minfo.workgroup_var_idx.count(base_id) > 0;
        uint32_t wg_idx = wg_root ? minfo.workgroup_var_idx[base_id] : cur.chains[base_id].index;
        spirv::Id cur_ty = wg_root
            ? (minfo.workgroup_var_type.count(base_id) ? minfo.workgroup_var_type[base_id] : 0)
            : cur.chains[base_id].wg_cur_ty;
        BVNodeId flat_idx = wg_root ? 0 : cur.chains[base_id].elem_idx;
        bool first = wg_root;
        for (size_t k = 1; k < inst.operands.size(); ++k) {
            BVNodeId idx32 = gpu_coerce_width(ctx, get_v(inst.operands[k].value), 32, false);
            const spirv::TypeInfo* cur_ti = cur_ty ? mod.type_info(cur_ty) : nullptr;
            if (first) {
                flat_idx = idx32;
                first = false;
            } else {
                uint32_t dim = container_dim(minfo, ctx, cur_ty, cur_ti);
                flat_idx = bv_bin(ctx, BVOp::Add, 32,
                    gpu_mul_or_shl(ctx, 32, flat_idx, bv_const(ctx, 32, dim)), idx32);
            }
            cur_ty = (cur_ti && cur_ti->op == spv::OpTypeArray) ? cur_ti->elem : 0;
        }
        SpirvAccessChainInfo info;
        info.is_workgroup = true;
        info.index = wg_idx;
        info.elem_idx = flat_idx;
        info.wg_cur_ty = cur_ty;
        cur.chains[inst.result] = info;
    } else if (resolve_local_chain(mod, cur, minfo, base_id, is_root, loc_base, loc_ty, loc_idx, loc_eb)) {
        spirv::Id cur_ty = loc_ty;
        BVNodeId flat_idx = loc_idx;
        bool first = is_root;
        for (size_t k = 1; k < inst.operands.size(); ++k) {
            BVNodeId idx32 = gpu_coerce_width(ctx, get_v(inst.operands[k].value), 32, false);
            const spirv::TypeInfo* cur_ti = cur_ty ? mod.type_info(cur_ty) : nullptr;
            if (first) {
                flat_idx = idx32;
                first = false;
            } else {
                uint32_t dim = container_dim(minfo, ctx, cur_ty, cur_ti);
                flat_idx = bv_bin(ctx, BVOp::Add, 32,
                    gpu_mul_or_shl(ctx, 32, flat_idx, bv_const(ctx, 32, dim)), idx32);
            }
            cur_ty = (cur_ti && (cur_ti->op == spv::OpTypeArray || cur_ti->op == spv::OpTypeVector ||
                                 cur_ti->op == spv::OpTypeMatrix))
                ? cur_ti->elem : 0;
        }
        SpirvAccessChainInfo info;
        info.is_local = true;
        info.local_base = loc_base;
        info.local_elem_bytes = loc_eb;
        info.elem_idx = flat_idx;
        info.local_cur_ty = cur_ty;
        cur.chains[inst.result] = info;
    } else if (minfo.stage_out_base.count(base_id)) {
        BVNodeId idx_v = get_v(idx_id);
        uint64_t cidx = (bv_node(ctx, idx_v).op == BVOp::Const) ? bv_node(ctx, idx_v).imm : 0;
        minfo.stage_out_base[inst.result] = minfo.stage_out_base[base_id] + cidx * 4ULL;
    } else if (minfo.var_composites.count(base_id)) {
        BVNodeId idx_v = get_v(idx_id);
        uint64_t cidx = (bv_node(ctx, idx_v).op == BVOp::Const) ? bv_node(ctx, idx_v).imm : UINT64_MAX;
        const auto& comp = minfo.var_composites[base_id];
        if (cidx < comp.size()) {
            SpirvAccessChainInfo info;
            info.is_direct = true;
            info.direct_val = comp[static_cast<size_t>(cidx)];
            cur.chains[inst.result] = info;
        }
    } else if (rti && rti->op == spv::OpTypePointer &&
               rti->storage == spv::StorageClassPhysicalStorageBuffer) {
        BVNodeId addr = gpu_coerce_width(ctx, get_v(base_id), 64, false);
        spirv::Id base_ty = minfo.value_types.count(base_id) ? minfo.value_types[base_id] : 0;
        const spirv::TypeInfo* bti = base_ty ? mod.type_info(base_ty) : nullptr;
        spirv::Id cur_ty = (bti && bti->op == spv::OpTypePointer) ? bti->elem : 0;
        for (size_t k = 1; k < inst.operands.size(); ++k) {
            const spirv::TypeInfo* cur_ti = cur_ty ? mod.type_info(cur_ty) : nullptr;
            BVNodeId iv = get_v(inst.operands[k].value);
            if (cur_ti && cur_ti->op == spv::OpTypeStruct) {
                uint32_t m = (bv_node(ctx, iv).op == BVOp::Const)
                    ? static_cast<uint32_t>(bv_node(ctx, iv).imm) : 0u;
                auto mit = minfo.member_offsets.find((static_cast<uint64_t>(cur_ty) << 32) | m);
                if (mit != minfo.member_offsets.end() && mit->second != 0) {
                    addr = gpu_add_addr64(ctx, addr, bv_const(ctx, 64, mit->second));
                }
                auto sit = minfo.struct_members.find(cur_ty);
                cur_ty = (sit != minfo.struct_members.end() && m < sit->second.size())
                    ? sit->second[m] : 0;
            } else if (cur_ti && (cur_ti->op == spv::OpTypeRuntimeArray ||
                                  cur_ti->op == spv::OpTypeArray)) {
                auto ait = minfo.array_strides.find(cur_ty);
                uint32_t stride = (ait != minfo.array_strides.end() && ait->second != 0)
                    ? ait->second
                    : std::max(1u, spirv_type_width(mod, cur_ti->elem) / 8u);
                BVNodeId idx64 = gpu_coerce_width(ctx, iv, 64, false);
                addr = gpu_add_addr64(ctx, addr,
                    gpu_mul_or_shl(ctx, 64, idx64, bv_const(ctx, 64, stride)));
                cur_ty = cur_ti->elem;
            } else if (k + 1 == inst.operands.size()) {
                uint32_t stride = std::max(1u, spirv_type_width(mod, rti->elem) / 8u);
                BVNodeId idx64 = gpu_coerce_width(ctx, iv, 64, false);
                addr = gpu_add_addr64(ctx, addr,
                    gpu_mul_or_shl(ctx, 64, idx64, bv_const(ctx, 64, stride)));
            }
        }
        out_res = addr;
    }
}

void handle_spirv_load(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    const spirv::Inst& inst,
    uint32_t rw,
    const spirv::TypeInfo* rti,
    BVNodeId& out_res) {
    BVContext& ctx = sc.ctx;
    const spirv::Module& mod = sc.mod;
    GpuSymEnv& env = sc.env;
    SpirvModuleInfo& minfo = sc.minfo;
    GpuExecTrace& trace = sc.trace;
    auto get_v = [&](spirv::Id id) -> BVNodeId {
        auto it = cur.vals.find(id);
        return (it != cur.vals.end()) ? it->second : bv_const(ctx, 32, 0);
    };

    spirv::Id ptr_id = inst.operands[0].value;
    bool is_loc_root = false;
    uint64_t lbase = 0;
    spirv::Id lty = 0;
    BVNodeId lidx = 0;
    uint32_t leb = 4;

    if (minfo.var_composites.count(ptr_id)) {
        cur.composites[inst.result] = minfo.var_composites[ptr_id];
    } else if (minfo.var_scalars.count(ptr_id)) {
        out_res = minfo.var_scalars[ptr_id];
    } else if (resolve_local_chain(mod, cur, minfo, ptr_id, is_loc_root, lbase, lty, lidx, leb)) {
        uint32_t nb = std::max(1u, rw / 8u);
        const bool const_idx = is_loc_root || (lidx != 0 && bv_node(ctx, lidx).op == BVOp::Const);
        if (const_idx && !sc.dirty_local_bases.count(lbase)) {
            uint64_t elem_off = is_loc_root ? 0ULL : bv_node(ctx, lidx).imm;
            uint64_t skey = lbase + elem_off * leb;
            auto lsit = cur.local_slots.find(skey);
            BVNodeId lv = (lsit != cur.local_slots.end()) ? lsit->second : bv_const(ctx, rw, 0);
            if (rti && rti->op == spv::OpTypeVector) {
                uint32_t ew = spirv_type_width(mod, rti->elem);
                std::vector<BVNodeId> elems;
                for (uint32_t c = 0; c < std::max(1u, rti->width); ++c) {
                    auto eit = cur.local_slots.find(lbase + (elem_off + c) * std::max(1u, ew / 8u));
                    elems.push_back(eit != cur.local_slots.end() ? eit->second : bv_const(ctx, ew, 0));
                }
                cur.composites[inst.result] = std::move(elems);
            } else {
                out_res = gpu_coerce_width(ctx, lv, rw, false);
            }
        } else {
            BVNodeId eff_idx = is_loc_root ? bv_const(ctx, 64, 0) : gpu_coerce_width(ctx, lidx, 64, false);
            BVNodeId byte_off = gpu_mul_or_shl(ctx, 64, eff_idx, bv_const(ctx, 64, leb));
            BVNodeId addr64 = gpu_add_addr64(ctx, bv_const(ctx, 64, lbase), byte_off);
            out_res = env.read_mem(ctx, true, sc.sym_local_stores, addr64, nb);
        }
    } else if (cur.chains.count(ptr_id)) {
        const auto& ch = cur.chains[ptr_id];
        if (ch.is_push_const) {
            out_res = (ch.index < env.param_vars.size())
                ? gpu_coerce_width(ctx, env.param_vars[ch.index], rw)
                : bv_const(ctx, rw, 0);
        } else if (ch.is_workgroup) {
            uint32_t nb = std::max(1u, rw / 8u);
            record_shared_access(ctx, trace, cur.cond, false, ch.index, ch.elem_idx, nb);
            BVNodeId caddr = make_shared_canonical_addr(ctx, ch.index, ch.elem_idx, nb);
            out_res = env.read_mem(ctx, true, trace.shared_stores, caddr, nb);
        } else if (ch.is_direct) {
            out_res = gpu_coerce_width(ctx, ch.direct_val, rw);
        }
    } else {
        BVNodeId addr = gpu_coerce_width(ctx, get_v(ptr_id), 64);
        uint32_t nb = std::max(1u, rw / 8u);
        uint32_t align = (inst.operands.size() >= 3 && (inst.operands[1].value & spv::MemoryAccessAlignedMask))
            ? inst.operands[2].value : nb;
        record_global_access(ctx, trace, env, cur.cond, false, addr, nb, align);
        out_res = env.read_mem(ctx, false, trace.global_stores, addr, nb);
        if (rti && rti->op == spv::OpTypeVector && rti->width > 0) {
            uint32_t ew = spirv_type_width(mod, rti->elem);
            std::vector<BVNodeId> elems;
            elems.reserve(rti->width);
            for (uint32_t c = 0; c < rti->width; ++c) {
                elems.push_back(bv_extract(ctx, (c + 1) * ew - 1, c * ew, out_res));
            }
            cur.composites[inst.result] = std::move(elems);
        }
    }
}

void handle_spirv_store(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    const spirv::Inst& inst) {
    BVContext& ctx = sc.ctx;
    const spirv::Module& mod = sc.mod;
    GpuSymEnv& env = sc.env;
    SpirvModuleInfo& minfo = sc.minfo;
    GpuExecTrace& trace = sc.trace;
    auto get_v = [&](spirv::Id id) -> BVNodeId {
        auto it = cur.vals.find(id);
        return (it != cur.vals.end()) ? it->second : bv_const(ctx, 32, 0);
    };

    spirv::Id ptr_id = inst.operands[0].value;
    spirv::Id val_id = inst.operands[1].value;
    BVNodeId val = get_v(val_id);
    uint32_t nb = std::max(1u, ctx.width(val) / 8u);
    if (cur.composites.count(val_id) && !cur.composites[val_id].empty()) {
        const auto& comp = cur.composites[val_id];
        uint32_t total_b = 0;
        for (BVNodeId cv : comp) total_b += std::max(1u, ctx.width(cv) / 8u);
        nb = std::max(1u, total_b);
        BVNodeId packed = comp[0];
        for (size_t c = 1; c < comp.size(); ++c) {
            packed = ctx.make_concat(comp[c], packed);
        }
        val = packed;
    }

    bool is_loc_root = false;
    uint64_t lbase = 0;
    spirv::Id lty = 0;
    BVNodeId lidx = 0;
    uint32_t leb = 4;

    if (minfo.stage_out_base.count(ptr_id)) {
        uint64_t base_addr = minfo.stage_out_base[ptr_id];
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
    } else if (resolve_local_chain(mod, cur, minfo, ptr_id, is_loc_root, lbase, lty, lidx, leb)) {
        if (is_loc_root || (lidx != 0 && bv_node(ctx, lidx).op == BVOp::Const)) {
            uint64_t elem_off = is_loc_root ? 0ULL : bv_node(ctx, lidx).imm;
            if (cur.composites.count(val_id) && !cur.composites[val_id].empty()) {
                const auto& comp = cur.composites[val_id];
                uint32_t eb = std::max(1u, ctx.width(comp[0]) / 8u);
                for (size_t c = 0; c < comp.size(); ++c) {
                    uint64_t skey = lbase + (elem_off + c) * eb;
                    cur.local_slots[skey] = comp[c];
                    sc.sym_local_stores.push_back({cur.cond, bv_const(ctx, 64, skey), eb, comp[c]});
                }
            } else {
                uint64_t skey = lbase + elem_off * leb;
                cur.local_slots[skey] = val;
                sc.sym_local_stores.push_back({cur.cond, bv_const(ctx, 64, skey), nb, val});
            }
        } else {
            sc.dirty_local_bases.insert(lbase);
            BVNodeId byte_off = gpu_mul_or_shl(
                ctx, 64, gpu_coerce_width(ctx, lidx, 64, false), bv_const(ctx, 64, leb));
            BVNodeId addr64 = gpu_add_addr64(ctx, bv_const(ctx, 64, lbase), byte_off);
            sc.sym_local_stores.push_back({cur.cond, addr64, nb, val});
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
}

bool handle_spirv_function_call(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    const spirv::Inst& inst,
    uint32_t call_depth,
    uint32_t rw,
    bool is_fl,
    const spirv::TypeInfo* rti,
    BVNodeId& out_res) {
    if (inst.operands.empty()) return true;
    BVContext& ctx = sc.ctx;
    const spirv::Module& mod = sc.mod;
    GpuSymEnv& env = sc.env;
    SpirvModuleInfo& minfo = sc.minfo;
    auto get_v = [&](spirv::Id id) -> BVNodeId {
        auto it = cur.vals.find(id);
        return (it != cur.vals.end()) ? it->second : bv_const(ctx, 32, 0);
    };

    spirv::Id cid = inst.operands[0].value;
    auto fit = minfo.fn_by_id.find(cid);
    if (fit == minfo.fn_by_id.end() || !fit->second) {
        sc.trace.supported = false;
        sc.trace.error_msg = "Unresolved callee %" + std::to_string(cid) + " in SPIR-V OpFunctionCall";
        return false;
    }
    const spirv::Function& callee = *fit->second;
    std::unordered_set<spirv::Id> vis;
    size_t callee_insts = 0;
    for (const auto& cb : callee.blocks) callee_insts += cb.insts.size();
    const bool pure_scalar = callee_is_pure_scalar(mod, callee, minfo, vis);
    const bool should_abstract_as_uf =
        callee.blocks.empty() ||
        (pure_scalar && (call_depth >= 3 || callee.blocks.size() > 8 || callee_insts > 128));
    if (should_abstract_as_uf) {
        std::vector<BVNodeId> call_args;
        for (size_t k = 1; k < inst.operands.size(); ++k) {
            spirv::Id aid = inst.operands[k].value;
            if (cur.composites.count(aid)) {
                for (BVNodeId cv : cur.composites[aid]) call_args.push_back(cv);
            } else {
                call_args.push_back(get_v(aid));
            }
        }
        if (rti && rti->op == spv::OpTypeVector && rti->width > 0) {
            bool ef = false;
            uint32_t ew = spirv_type_width(mod, rti->elem, &ef);
            std::vector<BVNodeId> rets;
            for (uint32_t c = 0; c < rti->width; ++c) {
                rets.push_back(env.eval_uf(
                    ctx, "spv_call_" + std::to_string(cid) + "_" + std::to_string(c), call_args, ew, ef));
            }
            cur.composites[inst.result] = std::move(rets);
        } else if (rti && rti->op != spv::OpTypeVoid) {
            out_res = env.eval_uf(ctx, "spv_call_" + std::to_string(cid), call_args, rw, is_fl);
        }
        return true;
    }
    if (call_depth >= 3) {
        sc.trace.supported = false;
        sc.trace.error_msg = "Exceeded SPIR-V OpFunctionCall inlining depth for impure callee";
        return false;
    }

    SpirvExecState sub_st;
    sub_st.block_idx = 0;
    sub_st.cond = cur.cond;
    sub_st.vals = minfo.global_vals;
    sub_st.composites = minfo.global_composites;
    sub_st.local_vars = cur.local_vars;
    sub_st.local_slots = cur.local_slots;
    for (size_t k = 0; k < callee.params.size() && (k + 1) < inst.operands.size(); ++k) {
        spirv::Id pid = callee.params[k].result;
        spirv::Id aid = inst.operands[k + 1].value;
        if (auto vit = cur.vals.find(aid); vit != cur.vals.end()) sub_st.vals[pid] = vit->second;
        if (auto cit = cur.composites.find(aid); cit != cur.composites.end()) sub_st.composites[pid] = cit->second;
        if (auto xit = cur.chains.find(aid); xit != cur.chains.end()) sub_st.chains[pid] = xit->second;
        if (auto lit = cur.local_vars.find(aid); lit != cur.local_vars.end()) sub_st.local_vars[pid] = lit->second;
    }
    BVNodeId c_ret = 0;
    std::vector<BVNodeId> c_comp;
    std::unordered_map<uint64_t, BVNodeId> out_slots;
    if (!execute_spirv_function_body(sc, callee, std::move(sub_st), call_depth + 1, c_ret, c_comp, &out_slots)) {
        return false;
    }
    for (const auto& [sk, sv] : out_slots) cur.local_slots[sk] = sv;
    if (!c_comp.empty()) cur.composites[inst.result] = std::move(c_comp);
    else if (c_ret != 0 && rti && rti->op != spv::OpTypeVoid) out_res = c_ret;
    return true;
}

} // namespace brass::verify::spirv_detail
