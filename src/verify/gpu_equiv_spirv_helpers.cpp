#include "gpu_equiv_spirv_internal.hpp"
#include <algorithm>
#include <cstring>

namespace brass::verify::spirv_detail {

uint32_t spirv_type_width(const spirv::Module& mod, spirv::Id type_id, bool* out_is_float) {
    if (out_is_float) *out_is_float = false;
    const spirv::TypeInfo* ti = mod.type_info(type_id);
    if (!ti) return 32;
    if (ti->op == spv::OpTypeBool) return 1;
    if (ti->op == spv::OpTypeInt) return ti->width ? ti->width : 32;
    if (ti->op == spv::OpTypeFloat) {
        if (out_is_float) *out_is_float = true;
        return ti->width ? ti->width : 32;
    }
    if (ti->op == spv::OpTypeVector) {
        uint32_t ew = spirv_type_width(mod, ti->elem, out_is_float);
        return ew * std::max(1u, ti->width);
    }
    if (ti->op == spv::OpTypePointer) return 64;
    return 32;
}

uint32_t spirv_leaf_elem_bytes(const spirv::Module& mod, spirv::Id type_id) {
    spirv::Id cur = type_id;
    for (int d = 0; d < 8 && cur != 0; ++d) {
        const spirv::TypeInfo* ti = mod.type_info(cur);
        if (!ti) break;
        if (ti->op == spv::OpTypeArray || ti->op == spv::OpTypeRuntimeArray) {
            cur = ti->elem;
            continue;
        }
        if (ti->op == spv::OpTypeVector) {
            uint32_t bw = spirv_type_width(mod, ti->elem, nullptr) * std::max(1u, ti->width);
            return std::max(1u, bw / 8u);
        }
        break;
    }
    return std::max(1u, spirv_type_width(mod, cur, nullptr) / 8u);
}

uint32_t SpirvModuleInfo::resolve_array_dim(BVContext& ctx, spirv::Id ty_id, const spirv::TypeInfo* ati) const {
    if (!ati) return 1u;
    if (ati->width > 0) return ati->width;
    auto alit = array_len_id.find(ty_id);
    if (alit != array_len_id.end()) {
        auto gvit = global_vals.find(alit->second);
        if (gvit != global_vals.end() && bv_node(ctx, gvit->second).op == BVOp::Const) {
            return static_cast<uint32_t>(bv_node(ctx, gvit->second).imm);
        }
    }
    return 256u;
}

const spirv::Function* find_spirv_entry_function(const spirv::Module& mod) {
    if (mod.functions.empty()) return nullptr;
    for (const auto& ep : mod.entry_points) {
        if (ep.model == spv::ExecutionModelGLCompute) {
            for (const auto& fn : mod.functions) {
                if (fn.id == ep.function && !fn.blocks.empty()) return &fn;
            }
        }
    }
    if (!mod.entry_points.empty()) {
        spirv::Id first_ep = mod.entry_points[0].function;
        for (const auto& fn : mod.functions) {
            if (fn.id == first_ep && !fn.blocks.empty()) return &fn;
        }
    }
    for (const auto& fn : mod.functions) {
        if (!fn.blocks.empty()) return &fn;
    }
    return &mod.functions[0];
}

void collect_spirv_module_info(
    BVContext& ctx,
    const spirv::Module& mod,
    GpuSymEnv& env,
    GpuExecTrace& trace,
    SpirvModuleInfo& out) {
    for (const auto& fn : mod.functions) {
        if (fn.id != 0) out.fn_by_id[fn.id] = &fn;
    }
    for (const auto& ann : mod.annotations) {
        if (ann.op == spv::OpDecorate && ann.operands.size() >= 3) {
            spirv::Id target = ann.operands[0].value;
            uint32_t dec = ann.operands[1].value;
            uint32_t val = ann.operands[2].value;
            if (dec == spv::DecorationBuiltIn) out.builtin_map[target] = val;
            else if (dec == spv::DecorationSpecId) out.spec_id_map[target] = val;
            else if (dec == spv::DecorationLocation) out.location_map[target] = val;
            else if (dec == spv::DecorationArrayStride) out.array_strides[target] = val;
        } else if (ann.op == spv::OpMemberDecorate && ann.operands.size() >= 4 &&
                   ann.operands[2].value == spv::DecorationOffset) {
            uint64_t key = (static_cast<uint64_t>(ann.operands[0].value) << 32) | ann.operands[1].value;
            out.member_offsets[key] = ann.operands[3].value;
        }
    }

    for (const auto& g : mod.globals) {
        if (g.result && g.type) out.value_types[g.result] = g.type;
        if (g.op == spv::OpTypeStruct) {
            std::vector<spirv::Id> mems;
            for (const auto& op : g.operands) mems.push_back(op.value);
            out.struct_members[g.result] = std::move(mems);
        } else if (g.op == spv::OpVariable && !g.operands.empty() &&
                   g.operands[0].value == spv::StorageClassPushConstant) {
            const spirv::TypeInfo* pti = mod.type_info(g.type);
            if (pti && pti->op == spv::OpTypePointer) out.pc_struct_id = pti->elem;
        } else if (g.op == spv::OpTypeArray && g.operands.size() >= 2) {
            out.array_len_id[g.result] = g.operands[1].value;
        } else if (g.op == spv::OpSpecConstantComposite) {
            auto bit = out.builtin_map.find(g.result);
            if (bit != out.builtin_map.end() && bit->second == spv::BuiltInWorkgroupSize) {
                for (uint32_t axis = 0; axis < 3 && axis < g.operands.size(); ++axis) {
                    out.wg_size_spec_axis[g.operands[axis].value] = axis;
                }
            }
        }
    }

    if (env.param_vars.empty()) {
        std::vector<Type> inferred;
        if (out.pc_struct_id && out.struct_members.count(out.pc_struct_id)) {
            for (spirv::Id mt : out.struct_members[out.pc_struct_id]) {
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

    for (const auto& g : mod.globals) {
        switch (g.op) {
        case spv::OpConstant: {
            uint32_t w = spirv_type_width(mod, g.type);
            uint64_t bits = g.operands.empty() ? 0ULL : g.operands[0].value;
            if (w == 64 && g.operands.size() >= 2) {
                bits |= (static_cast<uint64_t>(g.operands[1].value) << 32);
            }
            out.global_vals[g.result] = bv_const(ctx, w, bits);
            break;
        }
        case spv::OpConstantTrue:
        case spv::OpSpecConstantTrue:
            out.global_vals[g.result] = bv_true(ctx);
            break;
        case spv::OpConstantFalse:
        case spv::OpSpecConstantFalse:
            out.global_vals[g.result] = bv_false(ctx);
            break;
        case spv::OpConstantNull:
        case spv::OpUndef: {
            const spirv::TypeInfo* gti = g.type ? mod.type_info(g.type) : nullptr;
            if (gti && gti->op == spv::OpTypeVector) {
                uint32_t ew = spirv_type_width(mod, gti->elem);
                out.global_composites[g.result] = std::vector<BVNodeId>(std::max(1u, gti->width), bv_const(ctx, ew, 0));
            } else {
                out.global_vals[g.result] = bv_const(ctx, spirv_type_width(mod, g.type), 0);
            }
            break;
        }
        case spv::OpSpecConstant: {
            auto axis_it = out.wg_size_spec_axis.find(g.result);
            if (axis_it != out.wg_size_spec_axis.end() && axis_it->second < 3) {
                out.global_vals[g.result] = env.ntid[axis_it->second];
            } else {
                uint32_t w = spirv_type_width(mod, g.type);
                uint64_t bits = g.operands.empty() ? 0ULL : g.operands[0].value;
                if (w == 64 && g.operands.size() >= 2) {
                    bits |= (static_cast<uint64_t>(g.operands[1].value) << 32);
                }
                out.global_vals[g.result] = bv_const(ctx, w, bits);
            }
            break;
        }
        case spv::OpSpecConstantOp: {
            if (!g.operands.empty()) {
                auto inner = static_cast<spv::Op>(g.operands[0].value);
                uint32_t rw = spirv_type_width(mod, g.type);
                auto gv = [&](size_t idx, uint32_t fallback_w = 32) -> BVNodeId {
                    if (idx < g.operands.size()) {
                        auto it = out.global_vals.find(g.operands[idx].value);
                        if (it != out.global_vals.end()) return it->second;
                    }
                    return bv_const(ctx, fallback_w, 0);
                };
                if (inner == spv::OpLogicalNot) {
                    out.global_vals[g.result] = bv_un(ctx, BVOp::Not, 1, gpu_coerce_width(ctx, gv(1, 1), 1));
                } else if (inner == spv::OpLogicalAnd || inner == spv::OpLogicalOr) {
                    BVNodeId a = gpu_coerce_width(ctx, gv(1, 1), 1);
                    BVNodeId b = gpu_coerce_width(ctx, gv(2, 1), 1);
                    out.global_vals[g.result] = bv_bin(ctx, inner == spv::OpLogicalAnd ? BVOp::And : BVOp::Or, 1, a, b);
                } else if (inner == spv::OpSelect) {
                    BVNodeId c = gpu_coerce_width(ctx, gv(1, 1), 1);
                    BVNodeId tv = gpu_coerce_width(ctx, gv(2, rw), rw);
                    BVNodeId fv = gpu_coerce_width(ctx, gv(3, rw), rw);
                    out.global_vals[g.result] = bv_ite(ctx, rw, c, tv, fv);
                } else if (inner == spv::OpIEqual || inner == spv::OpINotEqual ||
                           inner == spv::OpULessThan || inner == spv::OpUGreaterThan ||
                           inner == spv::OpULessThanEqual || inner == spv::OpUGreaterThanEqual ||
                           inner == spv::OpSLessThan || inner == spv::OpSGreaterThan ||
                           inner == spv::OpSLessThanEqual || inner == spv::OpSGreaterThanEqual) {
                    BVNodeId a = gv(1);
                    BVNodeId b = gpu_coerce_width(ctx, gv(2, ctx.width(a)), ctx.width(a));
                    BVOp cop = (inner == spv::OpIEqual) ? BVOp::Eq : (inner == spv::OpINotEqual) ? BVOp::Ne
                             : (inner == spv::OpSLessThan) ? BVOp::Slt : (inner == spv::OpSLessThanEqual) ? BVOp::Sle
                             : (inner == spv::OpSGreaterThan) ? BVOp::Sgt : (inner == spv::OpSGreaterThanEqual) ? BVOp::Sge
                             : (inner == spv::OpULessThan) ? BVOp::Ult : (inner == spv::OpULessThanEqual) ? BVOp::Ule
                             : (inner == spv::OpUGreaterThan) ? BVOp::Ugt : BVOp::Uge;
                    out.global_vals[g.result] = bv_bin(ctx, cop, 1, a, b);
                } else if (inner == spv::OpIAdd || inner == spv::OpISub || inner == spv::OpIMul ||
                           inner == spv::OpUDiv || inner == spv::OpUMod ||
                           inner == spv::OpShiftLeftLogical || inner == spv::OpShiftRightLogical) {
                    BVNodeId a = gpu_coerce_width(ctx, gv(1, rw), rw);
                    BVNodeId b = gpu_coerce_width(ctx, gv(2, rw), rw);
                    BVOp bop = (inner == spv::OpIAdd) ? BVOp::Add : (inner == spv::OpISub) ? BVOp::Sub
                             : (inner == spv::OpIMul) ? BVOp::Mul : (inner == spv::OpUDiv) ? BVOp::UDiv
                             : (inner == spv::OpUMod) ? BVOp::URem : (inner == spv::OpShiftLeftLogical) ? BVOp::Shl
                             : BVOp::LShr;
                    out.global_vals[g.result] = bv_bin(ctx, bop, rw, a, b);
                } else {
                    out.global_vals[g.result] = bv_const(ctx, rw, 0);
                }
            }
            break;
        }
        case spv::OpConstantComposite:
        case spv::OpSpecConstantComposite: {
            auto bit = out.builtin_map.find(g.result);
            if (bit != out.builtin_map.end() && bit->second == spv::BuiltInWorkgroupSize) {
                out.global_composites[g.result] = {env.ntid[0], env.ntid[1], env.ntid[2]};
            } else {
                std::vector<BVNodeId> parts;
                for (const auto& op : g.operands) {
                    parts.push_back(out.global_vals.count(op.value) ? out.global_vals[op.value] : bv_const(ctx, 32, 0));
                }
                out.global_composites[g.result] = std::move(parts);
            }
            break;
        }
        case spv::OpVariable: {
            uint32_t sc = g.operands.empty() ? 0 : g.operands[0].value;
            if (sc == spv::StorageClassInput) {
                uint32_t bi = out.builtin_map.count(g.result) ? out.builtin_map[g.result] : UINT32_MAX;
                if (bi == spv::BuiltInLocalInvocationId) {
                    out.var_composites[g.result] = {env.tid[0], env.tid[1], env.tid[2]};
                } else if (bi == spv::BuiltInWorkgroupId) {
                    out.var_composites[g.result] = {env.ctaid[0], env.ctaid[1], env.ctaid[2]};
                } else if (bi == spv::BuiltInNumWorkgroups) {
                    out.var_composites[g.result] = {env.nctaid[0], env.nctaid[1], env.nctaid[2]};
                } else if (bi == spv::BuiltInGlobalInvocationId) {
                    out.var_composites[g.result] = {env.global_tid[0], env.global_tid[1], env.global_tid[2]};
                } else if (bi == spv::BuiltInWorkgroupSize) {
                    out.var_composites[g.result] = {env.ntid[0], env.ntid[1], env.ntid[2]};
                } else if (bi == spv::BuiltInSubgroupLocalInvocationId) {
                    out.var_scalars[g.result] = env.raw_subgroup_lane;
                } else if (bi == spv::BuiltInSubgroupId) {
                    out.var_scalars[g.result] = bv_bin(ctx, BVOp::LShr, 32, env.tid[0], bv_const(ctx, 32, 5));
                } else if (bi == spv::BuiltInNumSubgroups) {
                    out.var_scalars[g.result] = bv_bin(
                        ctx, BVOp::LShr, 32,
                        bv_bin(ctx, BVOp::Add, 32, env.ntid[0], bv_const(ctx, 32, 31)),
                        bv_const(ctx, 32, 5));
                } else if (bi == spv::BuiltInVertexIndex) {
                    out.var_scalars[g.result] = env.eval_uf(ctx, "spirv_vertex_index", {}, 32, false);
                } else if (bi == spv::BuiltInInstanceIndex) {
                    out.var_scalars[g.result] = env.eval_uf(ctx, "spirv_instance_index", {}, 32, false);
                } else if (bi == spv::BuiltInFragCoord) {
                    out.var_composites[g.result] = {
                        env.eval_uf(ctx, "spirv_frag_coord", {bv_const(ctx, 32, 0)}, 32, true),
                        env.eval_uf(ctx, "spirv_frag_coord", {bv_const(ctx, 32, 1)}, 32, true),
                        env.eval_uf(ctx, "spirv_frag_coord", {bv_const(ctx, 32, 2)}, 32, true),
                        env.eval_uf(ctx, "spirv_frag_coord", {bv_const(ctx, 32, 3)}, 32, true)};
                } else if (out.location_map.count(g.result)) {
                    uint32_t loc = out.location_map[g.result];
                    out.var_composites[g.result] = {
                        env.eval_uf(ctx, "spirv_stage_in", {bv_const(ctx, 32, loc), bv_const(ctx, 32, 0)}, 32, true),
                        env.eval_uf(ctx, "spirv_stage_in", {bv_const(ctx, 32, loc), bv_const(ctx, 32, 1)}, 32, true),
                        env.eval_uf(ctx, "spirv_stage_in", {bv_const(ctx, 32, loc), bv_const(ctx, 32, 2)}, 32, true),
                        env.eval_uf(ctx, "spirv_stage_in", {bv_const(ctx, 32, loc), bv_const(ctx, 32, 3)}, 32, true)};
                }
            } else if (sc == spv::StorageClassOutput) {
                uint32_t bi = out.builtin_map.count(g.result) ? out.builtin_map[g.result] : UINT32_MAX;
                if (bi == spv::BuiltInPosition) {
                    out.stage_out_base[g.result] = 0x6000000000000000ULL;
                } else if (out.location_map.count(g.result)) {
                    out.stage_out_base[g.result] =
                        0x6000000000001000ULL + static_cast<uint64_t>(out.location_map[g.result]) * 16ULL;
                }
            } else if (sc == spv::StorageClassPushConstant) {
                out.push_const_var = g.result;
            } else if (sc == spv::StorageClassWorkgroup) {
                uint32_t arr_idx = static_cast<uint32_t>(trace.shared_allocs.size());
                uint64_t total_cnt = 1;
                bool saw_array = false;
                uint32_t ebytes = 4;
                const spirv::TypeInfo* pti = mod.type_info(g.type);
                if (pti && pti->op == spv::OpTypePointer) {
                    out.workgroup_var_type[g.result] = pti->elem;
                    spirv::Id cur_ty = pti->elem;
                    for (int depth = 0; depth < 8 && cur_ty != 0; ++depth) {
                        const spirv::TypeInfo* ati = mod.type_info(cur_ty);
                        if (!ati || ati->op != spv::OpTypeArray) break;
                        saw_array = true;
                        uint32_t dim = std::max(1u, out.resolve_array_dim(ctx, cur_ty, ati));
                        total_cnt *= dim;
                        if (total_cnt > 0x0FFFFFFFULL) total_cnt = 0x0FFFFFFFULL;
                        cur_ty = ati->elem;
                    }
                    if (saw_array && cur_ty != 0) {
                        ebytes = std::max(1u, spirv_type_width(mod, cur_ty) / 8u);
                    }
                }
                uint32_t cnt = saw_array ? static_cast<uint32_t>(total_cnt) : 256u;
                uint64_t sym_base = kGpuSharedBase0 + static_cast<uint64_t>(arr_idx) * kGpuSharedStride;
                trace.shared_allocs.push_back({arr_idx, cnt, ebytes, sym_base});
                out.workgroup_var_idx[g.result] = arr_idx;
            } else if (sc == spv::StorageClassPrivate) {
                const spirv::TypeInfo* pti = mod.type_info(g.type);
                spirv::Id pointee = (pti && pti->op == spv::OpTypePointer) ? pti->elem : 0;
                uint64_t lbase = kGpuLocalBase0 + (out.next_local_id++ * kGpuSharedStride);
                out.private_vars[g.result] = {lbase, pointee, spirv_leaf_elem_bytes(mod, pointee)};
            }
            break;
        }
        default:
            break;
        }
    }
}

bool callee_is_pure_scalar(
    const spirv::Module& mod,
    const spirv::Function& fn,
    const SpirvModuleInfo& minfo,
    std::unordered_set<spirv::Id>& visiting) {
    if (!visiting.insert(fn.id).second) return false;
    for (const auto& inst : fn.params) {
        const spirv::TypeInfo* pti = inst.type ? mod.type_info(inst.type) : nullptr;
        if (pti && pti->op == spv::OpTypePointer) {
            visiting.erase(fn.id);
            return false;
        }
    }
    for (const auto& bb : fn.blocks) {
        for (const auto& inst : bb.insts) {
            if (inst.op == spv::OpControlBarrier || inst.op == spv::OpMemoryBarrier ||
                inst.op == spv::OpAtomicIAdd || inst.op == spv::OpAtomicFAddEXT ||
                inst.op == spv::OpAtomicSMin || inst.op == spv::OpAtomicUMin ||
                inst.op == spv::OpAtomicSMax || inst.op == spv::OpAtomicUMax ||
                inst.op == spv::OpAtomicExchange || inst.op == spv::OpConvertUToPtr ||
                inst.op == spv::OpCooperativeMatrixLoadKHR || inst.op == spv::OpCooperativeMatrixStoreKHR) {
                visiting.erase(fn.id);
                return false;
            }
            uint32_t opv = static_cast<uint32_t>(inst.op);
            if (opv >= static_cast<uint32_t>(spv::OpGroupNonUniformElect) &&
                opv <= static_cast<uint32_t>(spv::OpGroupNonUniformQuadSwap)) {
                visiting.erase(fn.id);
                return false;
            }
            if ((inst.op == spv::OpAccessChain || inst.op == spv::OpInBoundsAccessChain) &&
                !inst.operands.empty()) {
                spirv::Id base = inst.operands[0].value;
                if (minfo.workgroup_var_idx.count(base) || minfo.stage_out_base.count(base)) {
                    visiting.erase(fn.id);
                    return false;
                }
                const spirv::TypeInfo* rti = inst.type ? mod.type_info(inst.type) : nullptr;
                if (rti && rti->op == spv::OpTypePointer &&
                    (rti->storage == spv::StorageClassPhysicalStorageBuffer ||
                     rti->storage == spv::StorageClassStorageBuffer ||
                     rti->storage == spv::StorageClassWorkgroup)) {
                    visiting.erase(fn.id);
                    return false;
                }
            }
            if (inst.op == spv::OpFunctionCall && !inst.operands.empty()) {
                auto fit = minfo.fn_by_id.find(inst.operands[0].value);
                if (fit == minfo.fn_by_id.end() ||
                    !callee_is_pure_scalar(mod, *fit->second, minfo, visiting)) {
                    visiting.erase(fn.id);
                    return false;
                }
            }
        }
    }
    visiting.erase(fn.id);
    return true;
}

void init_local_variable(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    spirv::Id var_id,
    spirv::Id ptr_ty_id,
    spirv::Id init_id) {
    const spirv::TypeInfo* pti = sc.mod.type_info(ptr_ty_id);
    spirv::Id pointee = (pti && pti->op == spv::OpTypePointer) ? pti->elem : 0;
    uint32_t ebytes = spirv_leaf_elem_bytes(sc.mod, pointee);
    uint64_t lbase = kGpuLocalBase0 + (sc.minfo.next_local_id++ * kGpuSharedStride);
    cur.local_vars[var_id] = {lbase, pointee, ebytes};
    if (init_id == 0) return;
    if (auto cit = cur.composites.find(init_id); cit != cur.composites.end()) {
        for (size_t k = 0; k < cit->second.size(); ++k) {
            cur.local_slots[lbase + static_cast<uint64_t>(k) * ebytes] = cit->second[k];
        }
    } else if (auto vit = cur.vals.find(init_id); vit != cur.vals.end()) {
        cur.local_slots[lbase] = vit->second;
    }
}

bool handle_glsl_ext_inst(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    const spirv::Inst& inst,
    uint32_t rw,
    BVNodeId& out_res) {
    if (inst.operands.size() < 3) return false;
    BVContext& ctx = sc.ctx;
    GpuSymEnv& env = sc.env;
    auto get_v = [&](spirv::Id id) -> BVNodeId {
        auto it = cur.vals.find(id);
        return (it != cur.vals.end()) ? it->second : bv_const(ctx, 32, 0);
    };
    uint32_t glsl_op = inst.operands[1].value;
    spirv::Id arg0_id = inst.operands[2].value;
    BVNodeId a0 = get_v(arg0_id);

    switch (glsl_op) {
    case GLSLstd450Fma:
        out_res = env.eval_uf(ctx, "fp_fma", {
            gpu_coerce_width(ctx, a0, rw),
            gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw),
            gpu_coerce_width(ctx, get_v(inst.operands[4].value), rw)}, rw, true);
        return true;
    case GLSLstd450Sqrt:       out_res = env.eval_uf(ctx, "gpu_sqrt", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450InverseSqrt: out_res = env.eval_uf(ctx, "gpu_rsqrt", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Floor:      out_res = env.eval_uf(ctx, "gpu_floor", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Ceil:       out_res = env.eval_uf(ctx, "gpu_ceil", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Round:
    case GLSLstd450RoundEven:  out_res = env.eval_uf(ctx, "gpu_round", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Trunc:      out_res = env.eval_uf(ctx, "gpu_trunc", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450FAbs:
        out_res = bv_bin(ctx, BVOp::And, rw, gpu_coerce_width(ctx, a0, rw), bv_const(ctx, rw, ~(1ULL << (rw - 1))));
        return true;
    case GLSLstd450SAbs: {
        BVNodeId x = gpu_coerce_width(ctx, a0, rw, true);
        BVNodeId neg = bv_un(ctx, BVOp::Neg, rw, x);
        BVNodeId is_neg = bv_bin(ctx, BVOp::Slt, 1, x, bv_const(ctx, rw, 0));
        out_res = bv_ite(ctx, rw, is_neg, neg, x);
        return true;
    }
    case GLSLstd450SMin:
    case GLSLstd450SMax:
    case GLSLstd450UMin:
    case GLSLstd450UMax: {
        bool is_signed = (glsl_op == GLSLstd450SMin || glsl_op == GLSLstd450SMax);
        bool is_min = (glsl_op == GLSLstd450SMin || glsl_op == GLSLstd450UMin);
        BVNodeId a = gpu_coerce_width(ctx, a0, rw, is_signed);
        BVNodeId b = gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw, is_signed);
        BVOp cmp = is_min ? (is_signed ? BVOp::Slt : BVOp::Ult) : (is_signed ? BVOp::Sgt : BVOp::Ugt);
        out_res = bv_ite(ctx, rw, bv_bin(ctx, cmp, 1, a, b), a, b);
        return true;
    }
    case GLSLstd450SClamp:
    case GLSLstd450UClamp: {
        if (inst.operands.size() < 5) return false;
        bool is_signed = (glsl_op == GLSLstd450SClamp);
        BVNodeId x = gpu_coerce_width(ctx, a0, rw, is_signed);
        BVNodeId lo = gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw, is_signed);
        BVNodeId hi = gpu_coerce_width(ctx, get_v(inst.operands[4].value), rw, is_signed);
        BVOp gt_op = is_signed ? BVOp::Sgt : BVOp::Ugt;
        BVOp lt_op = is_signed ? BVOp::Slt : BVOp::Ult;
        BVNodeId max_lo = bv_ite(ctx, rw, bv_bin(ctx, gt_op, 1, x, lo), x, lo);
        out_res = bv_ite(ctx, rw, bv_bin(ctx, lt_op, 1, max_lo, hi), max_lo, hi);
        return true;
    }
    case GLSLstd450NMin:
    case GLSLstd450FMin:
        out_res = env.eval_uf(ctx, "gpu_fmin", {gpu_coerce_width(ctx, a0, rw), gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw)}, rw, true);
        return true;
    case GLSLstd450NMax:
    case GLSLstd450FMax:
        out_res = env.eval_uf(ctx, "gpu_fmax", {gpu_coerce_width(ctx, a0, rw), gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw)}, rw, true);
        return true;
    case GLSLstd450Sin:  out_res = env.eval_uf(ctx, "gpu_sin", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Cos:  out_res = env.eval_uf(ctx, "gpu_cos", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Exp2: out_res = env.eval_uf(ctx, "gpu_ex2", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Log2: out_res = env.eval_uf(ctx, "gpu_lg2", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Exp:  out_res = env.eval_uf(ctx, "gpu_exp", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Log:  out_res = env.eval_uf(ctx, "gpu_log", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Tanh: out_res = env.eval_uf(ctx, "gpu_tanh", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450FSign: out_res = env.eval_uf(ctx, "gpu_sign", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Fract: out_res = env.eval_uf(ctx, "gpu_fract", {gpu_coerce_width(ctx, a0, rw)}, rw, true); return true;
    case GLSLstd450Pow:
        out_res = env.eval_uf(ctx, "gpu_pow", {gpu_coerce_width(ctx, a0, rw), gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw)}, rw, true);
        return true;
    case GLSLstd450Ldexp:
        out_res = env.eval_uf(ctx, "gpu_ldexp", {gpu_coerce_width(ctx, a0, rw), gpu_coerce_width(ctx, get_v(inst.operands[3].value), 32, true)}, rw, true);
        return true;
    case GLSLstd450FrexpStruct: {
        BVNodeId sig = env.eval_uf(ctx, "gpu_frexp_sig", {gpu_coerce_width(ctx, a0, 32)}, 32, true);
        BVNodeId exp_v = env.eval_uf(ctx, "gpu_frexp_exp", {gpu_coerce_width(ctx, a0, 32)}, 32, false);
        cur.composites[inst.result] = {sig, exp_v};
        return true;
    }
    case GLSLstd450Step:
        out_res = env.eval_uf(ctx, "gpu_step", {
            gpu_coerce_width(ctx, a0, rw),
            gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw)}, rw, true);
        return true;
    case GLSLstd450FClamp:
    case GLSLstd450NClamp:
        out_res = env.eval_uf(ctx, "gpu_clamp", {
            gpu_coerce_width(ctx, a0, rw),
            gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw),
            gpu_coerce_width(ctx, get_v(inst.operands[4].value), rw)}, rw, true);
        return true;
    case GLSLstd450FMix:
        out_res = env.eval_uf(ctx, "gpu_mix", {
            gpu_coerce_width(ctx, a0, rw),
            gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw),
            gpu_coerce_width(ctx, get_v(inst.operands[4].value), rw)}, rw, true);
        return true;
    case GLSLstd450SmoothStep:
        out_res = env.eval_uf(ctx, "gpu_smoothstep", {
            gpu_coerce_width(ctx, a0, rw),
            gpu_coerce_width(ctx, get_v(inst.operands[3].value), rw),
            gpu_coerce_width(ctx, get_v(inst.operands[4].value), rw)}, rw, true);
        return true;
    case GLSLstd450UnpackHalf2x16: {
        BVNodeId lo = env.eval_uf(ctx, "gpu_f16_to_f32", {gpu_coerce_width(ctx, a0, 32)}, 32, true);
        cur.composites[inst.result] = {lo, bv_const(ctx, 32, 0)};
        return true;
    }
    case GLSLstd450PackHalf2x16: {
        BVNodeId elem0 = cur.composites.count(arg0_id) && !cur.composites[arg0_id].empty()
            ? cur.composites[arg0_id][0] : a0;
        out_res = env.eval_uf(ctx, "gpu_f32_to_f16", {gpu_coerce_width(ctx, elem0, 32)}, 32, false);
        return true;
    }
    default:
        return false;
    }
}

} // namespace brass::verify::spirv_detail
