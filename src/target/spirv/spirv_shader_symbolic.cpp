// SPIR-V module constant extraction, CFG dominance / post-dominance /
// control-dependence analysis, and symbolic BVContext expression builder.

#include "spirv_shader_prover_internal.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace brass::spirv::detail {

namespace {

void block_successors(const Inst& term, std::vector<Id>& out) {
    out.clear();
    if (term.op == spv::OpBranch && !term.operands.empty()) {
        out.push_back(term.operands[0].value);
    } else if (term.op == spv::OpBranchConditional && term.operands.size() >= 3) {
        out.push_back(term.operands[1].value);
        out.push_back(term.operands[2].value);
    } else if (term.op == spv::OpSwitch && term.operands.size() >= 2) {
        out.push_back(term.operands[1].value);
        for (size_t i = 2; i < term.operands.size(); ++i) {
            if (term.operands[i].is_id()) out.push_back(term.operands[i].value);
        }
    }
}

} // namespace

bool is_wg_divergent_builtin(uint32_t b) {
    return b == kBuiltInLocalInvocationId || b == kBuiltInGlobalInvocationId ||
           b == kBuiltInLocalInvocationIndex || b == kBuiltInSubgroupId || b == kBuiltInSubgroupLocalInvocationId;
}

bool is_sg_divergent_builtin(uint32_t b) {
    return b == kBuiltInLocalInvocationId || b == kBuiltInGlobalInvocationId ||
           b == kBuiltInLocalInvocationIndex || b == kBuiltInSubgroupLocalInvocationId;
}

bool is_subgroup_collective_op(spv::Op op) {
    auto v = static_cast<uint32_t>(op);
    return v >= static_cast<uint32_t>(spv::OpGroupNonUniformElect) &&
           v <= static_cast<uint32_t>(spv::OpGroupNonUniformQuadSwap);
}

bool is_subgroup_uniformizing_op(const Inst& inst) {
    switch (inst.op) {
        case spv::OpGroupNonUniformAll:
        case spv::OpGroupNonUniformAny:
        case spv::OpGroupNonUniformAllEqual:
        case spv::OpGroupNonUniformBroadcast:
        case spv::OpGroupNonUniformBroadcastFirst:
            return true;
        default: break;
    }
    auto v = static_cast<uint32_t>(inst.op);
    return v >= static_cast<uint32_t>(spv::OpGroupNonUniformIAdd) &&
           v <= static_cast<uint32_t>(spv::OpGroupNonUniformLogicalXor) &&
           inst.operands.size() == 3 &&
           inst.operands[1].value == static_cast<uint32_t>(spv::GroupOperationReduce);
}

ModuleConstants collect_module_constants(const Module& mod) {
    ModuleConstants mc;
    for (const Inst& ann : mod.annotations) {
        if (ann.op == spv::OpDecorate && ann.operands.size() >= 3 &&
            ann.operands[1].value == static_cast<uint32_t>(spv::DecorationBuiltIn)) {
            mc.builtins[ann.operands[0].value] = ann.operands[2].value;
        } else if (ann.op == spv::OpMemberDecorate && ann.operands.size() >= 4 &&
                   ann.operands[2].value == static_cast<uint32_t>(spv::DecorationBuiltIn)) {
            uint64_t key = (static_cast<uint64_t>(ann.operands[0].value) << 32) | ann.operands[1].value;
            mc.member_builtins[key] = ann.operands[3].value;
        }
    }

    for (const Inst& g : mod.globals) {
        const TypeInfo* ti = g.type ? mod.type_info(g.type) : nullptr;
        if (g.op == spv::OpConstant || g.op == spv::OpSpecConstant) {
            uint64_t raw = (g.operands.size() == 1) ? g.operands[0].value :
                           (g.operands.size() >= 2 ? (static_cast<uint64_t>(g.operands[0].value) |
                                                      (static_cast<uint64_t>(g.operands[1].value) << 32)) : 0);
            if (ti && ti->op == spv::OpTypeFloat) {
                double dval = 0.0;
                if (ti->width == 32) {
                    float f = 0.0f;
                    uint32_t u = static_cast<uint32_t>(raw);
                    std::memcpy(&f, &u, 4);
                    dval = static_cast<double>(f);
                } else if (ti->width == 64) std::memcpy(&dval, &raw, 8);
                mc.float_consts[g.result] = dval;
                if (dval == 0.0) mc.zero_consts.insert(g.result);
            } else {
                mc.int_consts[g.result] = raw;
                if (raw == 0) mc.zero_consts.insert(g.result);
            }
        } else if (g.op == spv::OpConstantTrue || g.op == spv::OpSpecConstantTrue) {
            mc.int_consts[g.result] = 1;
        } else if (g.op == spv::OpConstantFalse || g.op == spv::OpSpecConstantFalse || g.op == spv::OpConstantNull) {
            mc.int_consts[g.result] = 0;
            mc.float_consts[g.result] = 0.0;
            mc.zero_consts.insert(g.result);
        } else if (g.op == spv::OpSpecConstantOp && !g.operands.empty()) {
            auto inner = static_cast<spv::Op>(g.operands[0].value);
            auto get = [&](size_t idx) -> uint64_t {
                if (idx >= g.operands.size()) return 0;
                auto it = mc.int_consts.find(g.operands[idx].value);
                return it == mc.int_consts.end() ? 0 : it->second;
            };
            uint64_t a = get(1), b = get(2), c = get(3), res = 0;
            if (inner == spv::OpIAdd) res = a + b;
            else if (inner == spv::OpISub) res = a - b;
            else if (inner == spv::OpIMul) res = a * b;
            else if (inner == spv::OpUDiv) res = b ? a / b : 0;
            else if (inner == spv::OpUMod) res = b ? a % b : 0;
            else if (inner == spv::OpShiftLeftLogical) res = (b < 64) ? (a << b) : 0;
            else if (inner == spv::OpShiftRightLogical) res = (b < 64) ? (a >> b) : 0;
            else if (inner == spv::OpSelect) res = a ? b : c;
            else if (inner == spv::OpIEqual) res = (a == b) ? 1 : 0;
            else if (inner == spv::OpINotEqual) res = (a != b) ? 1 : 0;
            else if (inner == spv::OpULessThan) res = (static_cast<uint32_t>(a) < static_cast<uint32_t>(b)) ? 1 : 0;
            else if (inner == spv::OpUGreaterThan) res = (static_cast<uint32_t>(a) > static_cast<uint32_t>(b)) ? 1 : 0;
            mc.int_consts[g.result] = res;
        } else if (g.op == spv::OpVariable && !g.operands.empty()) {
            auto sc = static_cast<spv::StorageClass>(g.operands[0].value);
            mc.var_storage[g.result] = sc;
            if (ti && ti->op == spv::OpTypePointer) {
                mc.var_pointee[g.result] = ti->elem;
            }
        }
    }
    return mc;
}

FunctionCFG build_cfg(const Function& fn) {
    FunctionCFG cfg;
    cfg.num_blocks = fn.blocks.size();
    if (cfg.num_blocks == 0) return cfg;

    for (size_t i = 0; i < cfg.num_blocks; ++i) {
        cfg.block_index[fn.blocks[i].label] = i;
    }
    cfg.succs.resize(cfg.num_blocks);
    cfg.preds.resize(cfg.num_blocks);

    std::vector<Id> tmp_succ;
    std::vector<size_t> exits;
    for (size_t i = 0; i < cfg.num_blocks; ++i) {
        const Block& b = fn.blocks[i];
        if (b.insts.empty()) continue;
        const Inst& term = b.insts.back();
        if ((term.op == spv::OpBranchConditional || term.op == spv::OpSwitch) && !term.operands.empty()) {
            cfg.branch_cond[i] = term.operands[0].value;
        }
        if (term.op == spv::OpReturn || term.op == spv::OpReturnValue ||
            term.op == spv::OpKill || term.op == spv::OpTerminateInvocation) {
            exits.push_back(i);
        }
        block_successors(term, tmp_succ);
        for (Id sid : tmp_succ) {
            auto it = cfg.block_index.find(sid);
            if (it != cfg.block_index.end() &&
                std::find(cfg.succs[i].begin(), cfg.succs[i].end(), it->second) == cfg.succs[i].end()) {
                cfg.succs[i].push_back(it->second);
                cfg.preds[it->second].push_back(i);
            }
        }
    }

    cfg.reachable.assign(cfg.num_blocks, 0);
    std::vector<size_t> stack{0};
    cfg.reachable[0] = 1;
    while (!stack.empty()) {
        size_t u = stack.back();
        stack.pop_back();
        for (size_t v : cfg.succs[u]) if (!cfg.reachable[v]) { cfg.reachable[v] = 1; stack.push_back(v); }
    }

    auto intersect_sets = [](std::unordered_set<size_t>& dst, const std::unordered_set<size_t>& src) {
        for (auto it = dst.begin(); it != dst.end();) it = src.count(*it) ? std::next(it) : dst.erase(it);
    };

    cfg.dom.resize(cfg.num_blocks);
    std::unordered_set<size_t> all_reach;
    for (size_t i = 0; i < cfg.num_blocks; ++i) if (cfg.reachable[i]) all_reach.insert(i);

    for (size_t i : all_reach) cfg.dom[i] = (i == 0) ? std::unordered_set<size_t>{0} : all_reach;
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i : all_reach) {
            if (i == 0) continue;
            std::unordered_set<size_t> nd;
            bool first = true;
            for (size_t p : cfg.preds[i]) {
                if (!cfg.reachable[p]) continue;
                if (first) { nd = cfg.dom[p]; first = false; }
                else intersect_sets(nd, cfg.dom[p]);
            }
            nd.insert(i);
            if (nd != cfg.dom[i]) { cfg.dom[i] = std::move(nd); changed = true; }
        }
    }

    cfg.pdom.resize(cfg.num_blocks);
    std::unordered_set<size_t> rev_reach;
    stack.clear();
    for (size_t e : exits) if (cfg.reachable[e]) { rev_reach.insert(e); stack.push_back(e); }
    while (!stack.empty()) {
        size_t u = stack.back();
        stack.pop_back();
        for (size_t p : cfg.preds[u]) {
            if (cfg.reachable[p] && rev_reach.insert(p).second) stack.push_back(p);
        }
    }

    std::unordered_set<size_t> exit_set(exits.begin(), exits.end());
    for (size_t i : rev_reach) cfg.pdom[i] = exit_set.count(i) ? std::unordered_set<size_t>{i} : rev_reach;
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t i : rev_reach) {
            if (exit_set.count(i)) continue;
            std::unordered_set<size_t> npd;
            bool first = true;
            for (size_t s : cfg.succs[i]) {
                if (!rev_reach.count(s)) continue;
                if (first) { npd = cfg.pdom[s]; first = false; }
                else intersect_sets(npd, cfg.pdom[s]);
            }
            npd.insert(i);
            if (npd != cfg.pdom[i]) { cfg.pdom[i] = std::move(npd); changed = true; }
        }
    }

    for (const auto& [h, cond_id] : cfg.branch_cond) {
        (void)cond_id;
        if (!cfg.reachable[h]) continue;
        std::unordered_set<size_t> region;
        for (size_t b : all_reach) {
            if (b == h || !cfg.dom[b].count(h)) continue;
            bool reconverged = false;
            for (size_t m : cfg.pdom[h]) {
                if (m != h && cfg.dom[m].count(h) && cfg.dom[b].count(m)) {
                    reconverged = true;
                    break;
                }
            }
            if (!reconverged) region.insert(b);
        }
        cfg.control_dep_region[h] = std::move(region);
    }

    return cfg;
}

verify::BVNodeId SymbolicBuilder::build(Id val_id, int depth) {
    auto cit = cache.find(val_id);
    if (cit != cache.end()) return cit->second;

    auto bit = lid_comp.find(val_id);
    if (bit != lid_comp.end() && bit->second >= 0 && bit->second < 6) {
        return builtin_vars[static_cast<size_t>(bit->second)];
    }

    auto ic = mc.int_consts.find(val_id);
    if (ic != mc.int_consts.end()) {
        const TypeInfo* ti = mod.type_info(mod.type_of_value(val_id));
        uint32_t w = (ti && ti->op == spv::OpTypeBool) ? 1u : 32u;
        return cache[val_id] = ctx.make_const(ic->second & verify::width_mask(w), w);
    }

    if (depth >= 12) {
        const TypeInfo* ti = mod.type_info(mod.type_of_value(val_id));
        uint32_t w = (ti && ti->op == spv::OpTypeBool) ? 1u : 32u;
        return cache[val_id] = ctx.make_var("v_" + std::to_string(val_id), w);
    }

    auto dit = def_inst.find(val_id);
    if (dit == def_inst.end() || !dit->second) {
        const TypeInfo* ti = mod.type_info(mod.type_of_value(val_id));
        uint32_t w = (ti && ti->op == spv::OpTypeBool) ? 1u : 32u;
        return cache[val_id] = ctx.make_var("v_" + std::to_string(val_id), w);
    }

    const Inst& inst = *dit->second;
    const TypeInfo* rti = inst.type ? mod.type_info(inst.type) : nullptr;
    uint32_t rw = (rti && rti->op == spv::OpTypeBool) ? 1u : 32u;

    if (inst.op == spv::OpPhi) {
        verify::BVNodeId sym = ctx.make_var("v_" + std::to_string(val_id), rw);
        cache[val_id] = sym;
        if (rw == 1 && inst.operands.size() == 4) {
            for (size_t k = 0; k < 4; k += 2) {
                Id v_const = inst.operands[k].value;
                Id pred_const = inst.operands[k + 1].value;
                Id v_other = inst.operands[k == 0 ? 2 : 0].value;
                Id pred_other = inst.operands[k == 0 ? 3 : 1].value;
                auto c_it = mc.int_consts.find(v_const);
                auto p_it = cfg.block_index.find(pred_const);
                auto po_it = cfg.block_index.find(pred_other);
                if (p_it != cfg.block_index.end() && po_it != cfg.block_index.end()) {
                    const Inst& pterm = fn.blocks[p_it->second].insts.back();
                    if (pterm.op == spv::OpBranchConditional && pterm.operands.size() >= 3) {
                        Id cond_id = pterm.operands[0].value;
                        auto tit = cfg.block_index.find(pterm.operands[1].value);
                        bool other_in_true = (tit != cfg.block_index.end() && cfg.dom[po_it->second].count(tit->second));
                        bool is_and = (c_it != mc.int_consts.end() && c_it->second == 0) ||
                                      (v_const == cond_id && other_in_true);
                        bool is_or = (c_it != mc.int_consts.end() && c_it->second != 0) ||
                                     (v_const == cond_id && !other_in_true);
                        if (is_and || is_or) {
                            verify::BVNodeId hc = build(cond_id, depth + 1);
                            if (ctx.width(hc) != 1) hc = ctx.make_binary(verify::BVOp::Ne, hc, ctx.make_const(0, ctx.width(hc)), 1);
                            verify::BVNodeId oc = build(v_other, depth + 1);
                            if (ctx.width(oc) != 1) oc = ctx.make_binary(verify::BVOp::Ne, oc, ctx.make_const(0, ctx.width(oc)), 1);
                            return cache[val_id] = ctx.make_binary(is_and ? verify::BVOp::And : verify::BVOp::Or, hc, oc, 1);
                        }
                    }
                }
            }
        }
        for (size_t k = 0; k + 1 < inst.operands.size(); k += 2) {
            Id incoming = inst.operands[k].value;
            auto init_it = mc.int_consts.find(incoming);
            if (init_it != mc.int_consts.end() && inst.operands.size() == 4) {
                Id other = inst.operands[k == 0 ? 2 : 0].value;
                auto oit = def_inst.find(other);
                if (oit != def_inst.end() && oit->second &&
                    oit->second->op == spv::OpShiftRightLogical &&
                    !oit->second->operands.empty() && oit->second->operands[0].value == val_id) {
                    extra_constraints.push_back(
                        ctx.make_binary(verify::BVOp::Ule, sym,
                                        ctx.make_const(init_it->second & 0xFFFFFFFFu, 32), 1));
                }
            }
        }
        return sym;
    }

    if (inst.op == spv::OpLoad && !inst.operands.empty()) {
        Id ptr = inst.operands[0].value;
        auto pbit = lid_comp.find(ptr);
        if (pbit != lid_comp.end() && pbit->second >= 0 && pbit->second < 6) {
            return cache[val_id] = builtin_vars[static_cast<size_t>(pbit->second)];
        }
        auto ssit = single_store_val.find(ptr);
        if (ssit != single_store_val.end()) {
            return cache[val_id] = build(ssit->second, depth + 1);
        }
        auto pcit = cache.find(ptr);
        if (pcit != cache.end()) return cache[val_id] = pcit->second;
        verify::BVNodeId sym = ctx.make_var("v_" + std::to_string(ptr), rw);
        cache[ptr] = sym;
        return cache[val_id] = sym;
    }

    if (inst.op == spv::OpCompositeExtract && inst.operands.size() >= 2) {
        Id vec = inst.operands[0].value;
        uint32_t comp = inst.operands[1].value;
        auto vbit = lid_comp.find(vec);
        if (vbit != lid_comp.end() && vbit->second == 10 && comp < 3) {
            return cache[val_id] = builtin_vars[comp];
        }
    }

    auto arg = [&](size_t idx) -> verify::BVNodeId {
        return build(inst.operands[idx].value, depth + 1);
    };
    auto to32 = [&](verify::BVNodeId n) -> verify::BVNodeId {
        uint32_t w = ctx.width(n);
        if (w == 32) return n;
        if (w < 32) return ctx.make_zext(n, 32 - w);
        return ctx.make_extract(n, 31, 0);
    };
    auto to1 = [&](verify::BVNodeId n) -> verify::BVNodeId {
        uint32_t w = ctx.width(n);
        if (w == 1) return n;
        return ctx.make_binary(verify::BVOp::Ne, n, ctx.make_const(0, w), 1);
    };

    auto bin32 = [&](verify::BVOp bop, uint32_t w = 32) {
        return cache[val_id] = ctx.make_binary(bop, to32(arg(0)), to32(arg(1)), w);
    };
    switch (inst.op) {
        case spv::OpIAdd: return bin32(verify::BVOp::Add);
        case spv::OpISub: return bin32(verify::BVOp::Sub);
        case spv::OpIMul: return bin32(verify::BVOp::Mul);
        case spv::OpUDiv: return bin32(verify::BVOp::UDiv);
        case spv::OpSDiv: return bin32(verify::BVOp::SDiv);
        case spv::OpUMod: return bin32(verify::BVOp::URem);
        case spv::OpSRem:
        case spv::OpSMod: return bin32(verify::BVOp::SRem);
        case spv::OpShiftLeftLogical: return bin32(verify::BVOp::Shl);
        case spv::OpShiftRightLogical: return bin32(verify::BVOp::LShr);
        case spv::OpShiftRightArithmetic: return bin32(verify::BVOp::AShr);
        case spv::OpBitwiseAnd: return bin32(verify::BVOp::And);
        case spv::OpBitwiseOr: return bin32(verify::BVOp::Or);
        case spv::OpBitwiseXor: return bin32(verify::BVOp::Xor);
        case spv::OpNot: return cache[val_id] = ctx.make_unary(verify::BVOp::Not, to32(arg(0)), 32);
        case spv::OpSNegate: return cache[val_id] = ctx.make_unary(verify::BVOp::Neg, to32(arg(0)), 32);
        case spv::OpLogicalAnd: return cache[val_id] = ctx.make_binary(verify::BVOp::And, to1(arg(0)), to1(arg(1)), 1);
        case spv::OpLogicalOr: return cache[val_id] = ctx.make_binary(verify::BVOp::Or, to1(arg(0)), to1(arg(1)), 1);
        case spv::OpLogicalNot: return cache[val_id] = ctx.make_unary(verify::BVOp::Not, to1(arg(0)), 1);
        case spv::OpIEqual:
        case spv::OpLogicalEqual: return bin32(verify::BVOp::Eq, 1);
        case spv::OpINotEqual:
        case spv::OpLogicalNotEqual: return bin32(verify::BVOp::Ne, 1);
        case spv::OpULessThan: return bin32(verify::BVOp::Ult, 1);
        case spv::OpULessThanEqual: return bin32(verify::BVOp::Ule, 1);
        case spv::OpUGreaterThan: return bin32(verify::BVOp::Ugt, 1);
        case spv::OpUGreaterThanEqual: return bin32(verify::BVOp::Uge, 1);
        case spv::OpSLessThan: return bin32(verify::BVOp::Slt, 1);
        case spv::OpSLessThanEqual: return bin32(verify::BVOp::Sle, 1);
        case spv::OpSGreaterThan: return bin32(verify::BVOp::Sgt, 1);
        case spv::OpSGreaterThanEqual: return bin32(verify::BVOp::Sge, 1);
        case spv::OpSelect:
            if (inst.operands.size() >= 3) {
                return cache[val_id] = (rw == 1)
                    ? ctx.make_ite(to1(arg(0)), to1(arg(1)), to1(arg(2)))
                    : ctx.make_ite(to1(arg(0)), to32(arg(1)), to32(arg(2)));
            }
            break;
        case spv::OpSConvert:
        case spv::OpUConvert:
        case spv::OpBitcast:
            if (!inst.operands.empty()) return cache[val_id] = to32(arg(0));
            break;
        case spv::OpExtInst:
            if (inst.operands.size() >= 4) {
                uint32_t eop = inst.operands[1].value;
                verify::BVNodeId a = to32(arg(2)), b = to32(arg(3));
                if (eop == GLSLstd450UMin) return cache[val_id] = ctx.make_ite(ctx.make_binary(verify::BVOp::Ult, a, b, 1), a, b);
                if (eop == GLSLstd450SMin) return cache[val_id] = ctx.make_ite(ctx.make_binary(verify::BVOp::Slt, a, b, 1), a, b);
                if (eop == GLSLstd450UMax) return cache[val_id] = ctx.make_ite(ctx.make_binary(verify::BVOp::Ugt, a, b, 1), a, b);
                if (eop == GLSLstd450SMax) return cache[val_id] = ctx.make_ite(ctx.make_binary(verify::BVOp::Sgt, a, b, 1), a, b);
            }
            break;
        default:
            break;
    }
    return cache[val_id] = ctx.make_var("v_" + std::to_string(val_id), rw);
}

} // namespace brass::spirv::detail
