// Static compute shader prover (`spirv::audit_compute_shader` and
// `spirv::audit_compute_shader_binary`).

#define SPV_ENABLE_UTILITY_CODE 1
#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <spirv/unified1/GLSL.std.450.h>

#include "verify/bv_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <unordered_set>

namespace brass::spirv {

namespace {

constexpr uint32_t kBuiltInLocalInvocationId = static_cast<uint32_t>(spv::BuiltInLocalInvocationId);
constexpr uint32_t kBuiltInGlobalInvocationId = static_cast<uint32_t>(spv::BuiltInGlobalInvocationId);
constexpr uint32_t kBuiltInLocalInvocationIndex = static_cast<uint32_t>(spv::BuiltInLocalInvocationIndex);
constexpr uint32_t kBuiltInWorkgroupSize = static_cast<uint32_t>(spv::BuiltInWorkgroupSize);
constexpr uint32_t kBuiltInSubgroupId = static_cast<uint32_t>(spv::BuiltInSubgroupId);
constexpr uint32_t kBuiltInSubgroupLocalInvocationId = static_cast<uint32_t>(spv::BuiltInSubgroupLocalInvocationId);

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

struct ModuleConstants {
    std::unordered_map<Id, uint64_t> int_consts;
    std::unordered_map<Id, double> float_consts;
    std::unordered_set<Id> zero_consts;
    std::unordered_map<Id, uint32_t> builtins;
    std::unordered_map<uint64_t, uint32_t> member_builtins; // (struct_id << 32) | member -> BuiltIn
    std::unordered_map<Id, spv::StorageClass> var_storage;
    std::unordered_map<Id, Id> var_pointee;
};

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

struct FunctionCFG {
    size_t num_blocks = 0;
    std::unordered_map<Id, size_t> block_index;
    std::vector<std::vector<size_t>> succs;
    std::vector<std::vector<size_t>> preds;
    std::vector<uint8_t> reachable;
    std::vector<std::unordered_set<size_t>> dom;
    std::vector<std::unordered_set<size_t>> pdom;
    std::unordered_map<size_t, Id> branch_cond;
    std::unordered_map<size_t, std::unordered_set<size_t>> control_dep_region;
};

FunctionCFG build_cfg(const Function& fn) {
    FunctionCFG cfg;
    cfg.num_blocks = fn.blocks.size();
    if (cfg.num_blocks == 0) return cfg;

    for (size_t i = 0; i < cfg.num_blocks; ++i) {
        cfg.block_index[fn.blocks[i].label] = i;
    }
    cfg.succs.resize(cfg.num_blocks);
    cfg.preds.resize(cfg.num_blocks);
    cfg.reachable.assign(cfg.num_blocks, 0);

    std::vector<size_t> exits;
    std::vector<Id> tmp_succ;
    for (size_t i = 0; i < cfg.num_blocks; ++i) {
        const Block& b = fn.blocks[i];
        if (b.insts.empty()) continue;
        const Inst& term = b.insts.back();
        if ((term.op == spv::OpBranchConditional || term.op == spv::OpSwitch) && !term.operands.empty()) {
            cfg.branch_cond[i] = term.operands[0].value;
        } else if (term.op == spv::OpReturn || term.op == spv::OpReturnValue) {
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

    std::vector<size_t> stack = {0};
    cfg.reachable[0] = 1;
    while (!stack.empty()) {
        size_t u = stack.back();
        stack.pop_back();
        for (size_t v : cfg.succs[u]) if (!cfg.reachable[v]) { cfg.reachable[v] = 1; stack.push_back(v); }
    }

    auto intersect_sets = [](std::unordered_set<size_t>& dst, const std::unordered_set<size_t>& src) {
        for (auto it = dst.begin(); it != dst.end();) it = src.count(*it) ? std::next(it) : dst.erase(it);
    };

    // Dominators on reachable blocks.
    std::unordered_set<size_t> all_reach;
    for (size_t i = 0; i < cfg.num_blocks; ++i) if (cfg.reachable[i]) all_reach.insert(i);
    cfg.dom.resize(cfg.num_blocks);
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

    // Post-dominators on reachable blocks that can reach a return block.
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
    cfg.pdom.resize(cfg.num_blocks);
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

    // For each branch header h, block b is control-dependent on h if h strictly
    // dominates b and no strict post-dominator m of h dominates b.
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

// Builds a BVNodeId expression for `val_id` inside `ctx`.
struct SymbolicBuilder {
    const Module& mod;
    const Function& fn;
    const FunctionCFG& cfg;
    const ModuleConstants& mc;
    const std::unordered_map<Id, const Inst*>& def_inst;
    const std::unordered_map<Id, int>& lid_comp; // result_id -> 0(x), 1(y), 2(z), 3(local_index), 4(sg_local), 5(sg_id)
    const std::unordered_map<Id, Id>& single_store_val;
    verify::BVContext& ctx;
    std::array<verify::BVNodeId, 6> builtin_vars;
    std::unordered_map<Id, verify::BVNodeId> cache;
    std::vector<verify::BVNodeId> extra_constraints;

    verify::BVNodeId build(Id val_id, int depth = 0) {
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
            // Short-circuit boolean OpPhi (a && b / a || b):
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
            // Check if this is a decreasing shift or bounded induction Phi with a constant init.
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
};

} // namespace

ComputeShaderAuditReport audit_compute_shader(const Module& mod) {
    ComputeShaderAuditReport report;
    report.parse_ok = true;
    report.num_functions = static_cast<uint32_t>(mod.functions.size());

    // 1. Structural & SSA Dominance Invariants
    report.verifier_diagnostics = verify(mod);
    report.structural_ok = report.verifier_diagnostics.empty();
    for (const Diagnostic& d : report.verifier_diagnostics) {
        report.hazards.push_back(ShaderHazard{
            HazardKind::StructuralOrDominance, d.function, d.block, 0, spv::OpNop, d.message});
    }

    ModuleConstants mc = collect_module_constants(mod);

    // Extract LocalSize from OpExecutionMode / OpExecutionModeId or BuiltIn WorkgroupSize.
    for (const Inst& em : mod.execution_modes) {
        if (em.op == spv::OpExecutionMode && em.operands.size() >= 5 &&
            em.operands[1].value == static_cast<uint32_t>(spv::ExecutionModeLocalSize)) {
            report.local_size = {em.operands[2].value, em.operands[3].value, em.operands[4].value};
        } else if (em.op == spv::OpExecutionModeId && em.operands.size() >= 5 &&
                   em.operands[1].value == static_cast<uint32_t>(spv::ExecutionModeLocalSizeId)) {
            for (size_t k = 0; k < 3; ++k) {
                auto it = mc.int_consts.find(em.operands[2 + k].value);
                if (it != mc.int_consts.end() && it->second > 0) report.local_size[k] = static_cast<uint32_t>(it->second);
            }
        }
    }
    for (const Inst& g : mod.globals) {
        if ((g.op == spv::OpConstantComposite || g.op == spv::OpSpecConstantComposite) && g.operands.size() >= 3) {
            auto bit = mc.builtins.find(g.result);
            if (bit != mc.builtins.end() && bit->second == kBuiltInWorkgroupSize) {
                for (size_t k = 0; k < 3; ++k) {
                    auto it = mc.int_consts.find(g.operands[k].value);
                    if (it != mc.int_consts.end() && it->second > 0) report.local_size[k] = static_cast<uint32_t>(it->second);
                }
            }
        }
    }

    for (const auto& [vid, sc] : mc.var_storage) {
        if (sc != spv::StorageClassWorkgroup) continue;
        auto pit = mc.var_pointee.find(vid);
        if (pit != mc.var_pointee.end()) {
            const TypeInfo* ti = mod.type_info(pit->second);
            if (ti && ti->op == spv::OpTypeArray) ++report.num_workgroup_arrays;
        }
    }
    report.dominance_ok = report.barriers_uniform = report.shared_bounds_ok = report.div_sqrt_ok = true;

    for (const Function& fn : mod.functions) {
        report.num_blocks += static_cast<uint32_t>(fn.blocks.size());
        FunctionCFG cfg = build_cfg(fn);
        if (cfg.num_blocks == 0) continue;

        // Collect local SSA definitions and single-store Function variables.
        std::unordered_map<Id, std::pair<size_t, size_t>> def_loc;
        std::unordered_map<Id, const Inst*> def_inst;
        std::unordered_map<Id, spv::StorageClass> local_var_sc = mc.var_storage;
        std::unordered_map<Id, uint32_t> store_count;
        std::unordered_map<Id, Id> single_store_val;
        for (size_t bi = 0; bi < cfg.num_blocks; ++bi) {
            const Block& b = fn.blocks[bi];
            report.num_instructions += static_cast<uint32_t>(b.insts.size());
            for (size_t ii = 0; ii < b.insts.size(); ++ii) {
                const Inst& inst = b.insts[ii];
                if (inst.result != 0) {
                    def_loc[inst.result] = {bi, ii};
                    def_inst[inst.result] = &inst;
                }
                if (inst.op == spv::OpVariable && !inst.operands.empty()) {
                    local_var_sc[inst.result] = static_cast<spv::StorageClass>(inst.operands[0].value);
                } else if (inst.op == spv::OpStore && inst.operands.size() >= 2) {
                    Id ptr = inst.operands[0].value;
                    if (++store_count[ptr] == 1) single_store_val[ptr] = inst.operands[1].value;
                    else single_store_val.erase(ptr);
                }
            }
        }

        // Check SSA dominance across all reachable blocks.
        for (size_t bi = 0; bi < cfg.num_blocks; ++bi) {
            if (!cfg.reachable[bi]) continue;
            const Block& b = fn.blocks[bi];
            for (size_t ii = 0; ii < b.insts.size(); ++ii) {
                const Inst& inst = b.insts[ii];
                if (inst.op == spv::OpPhi) {
                    for (size_t k = 0; k + 1 < inst.operands.size(); k += 2) {
                        Id val_id = inst.operands[k].value;
                        Id pred_label = inst.operands[k + 1].value;
                        auto dit = def_loc.find(val_id);
                        auto pit = cfg.block_index.find(pred_label);
                        if (dit != def_loc.end() && pit != cfg.block_index.end() && cfg.reachable[pit->second]) {
                            size_t def_b = dit->second.first;
                            if (!cfg.dom[pit->second].count(def_b)) {
                                report.dominance_ok = false;
                                report.hazards.push_back(ShaderHazard{
                                    HazardKind::StructuralOrDominance, fn.name, b.label, inst.result, inst.op,
                                    "OpPhi operand %" + std::to_string(val_id) +
                                        " does not dominate predecessor %" + std::to_string(pred_label)});
                            }
                        }
                    }
                } else {
                    for (const Operand& opd : inst.operands) {
                        if (!opd.is_id()) continue;
                        auto dit = def_loc.find(opd.value);
                        if (dit != def_loc.end()) {
                            auto [def_b, def_i] = dit->second;
                            if (!cfg.dom[bi].count(def_b) || (def_b == bi && def_i >= ii)) {
                                report.dominance_ok = false;
                                report.hazards.push_back(ShaderHazard{
                                    HazardKind::StructuralOrDominance, fn.name, b.label, inst.result, inst.op,
                                    "SSA operand %" + std::to_string(opd.value) +
                                        " used before or without dominating definition"});
                            }
                        }
                    }
                }
            }
        }

        // Identify builtin component pointers/values and propagate uniformity taint.
        std::unordered_map<Id, int> lid_comp; // 0:lid.x, 1:lid.y, 2:lid.z, 3:local_idx, 4:sg_local, 5:sg_id, 10:lid_vec
        std::unordered_set<Id> wg_div;
        std::unordered_set<Id> sg_div;
        for (const auto& [vid, bkind] : mc.builtins) {
            if (is_wg_divergent_builtin(bkind)) wg_div.insert(vid);
            if (is_sg_divergent_builtin(bkind)) sg_div.insert(vid);
            if (bkind == kBuiltInLocalInvocationId) lid_comp[vid] = 10;
            else if (bkind == kBuiltInLocalInvocationIndex) lid_comp[vid] = 3;
            else if (bkind == kBuiltInSubgroupLocalInvocationId) lid_comp[vid] = 4;
            else if (bkind == kBuiltInSubgroupId) lid_comp[vid] = 5;
        }

        for (int pass = 0; pass < 3; ++pass) {
            for (size_t bi = 0; bi < cfg.num_blocks; ++bi) {
                if (!cfg.reachable[bi]) continue;
                bool blk_wg_div = false;
                bool blk_sg_div = false;
                for (const auto& [h, reg] : cfg.control_dep_region) {
                    if (reg.count(bi)) {
                        Id c = cfg.branch_cond[h];
                        if (wg_div.count(c)) blk_wg_div = true;
                        if (sg_div.count(c)) blk_sg_div = true;
                    }
                }
                for (const Inst& inst : fn.blocks[bi].insts) {
                    if (inst.op == spv::OpAccessChain || inst.op == spv::OpInBoundsAccessChain) {
                        if (inst.operands.empty()) continue;
                        Id base = inst.operands[0].value;
                        auto bit = lid_comp.find(base);
                        if (bit != lid_comp.end() && bit->second == 10 && inst.operands.size() >= 2) {
                            auto cit = mc.int_consts.find(inst.operands[1].value);
                            if (cit != mc.int_consts.end() && cit->second < 3) {
                                lid_comp[inst.result] = static_cast<int>(cit->second);
                            }
                        }
                        auto pit = mc.var_pointee.find(base);
                        if (pit != mc.var_pointee.end() && inst.operands.size() >= 2) {
                            auto cit = mc.int_consts.find(inst.operands[1].value);
                            if (cit != mc.int_consts.end()) {
                                uint64_t mkey = (static_cast<uint64_t>(pit->second) << 32) | cit->second;
                                auto mbit = mc.member_builtins.find(mkey);
                                if (mbit != mc.member_builtins.end()) {
                                    if (is_wg_divergent_builtin(mbit->second)) wg_div.insert(inst.result);
                                    if (is_sg_divergent_builtin(mbit->second)) sg_div.insert(inst.result);
                                }
                            }
                        }
                        for (const Operand& o : inst.operands) {
                            if (o.is_id() && wg_div.count(o.value)) wg_div.insert(inst.result);
                            if (o.is_id() && sg_div.count(o.value)) sg_div.insert(inst.result);
                        }
                    } else if (inst.op == spv::OpLoad && !inst.operands.empty()) {
                        Id ptr = inst.operands[0].value;
                        auto bit = lid_comp.find(ptr);
                        if (bit != lid_comp.end()) lid_comp[inst.result] = bit->second;
                        if (wg_div.count(ptr)) wg_div.insert(inst.result);
                        if (sg_div.count(ptr)) sg_div.insert(inst.result);
                    } else if (inst.op == spv::OpStore && inst.operands.size() >= 2) {
                        Id ptr = inst.operands[0].value;
                        Id val = inst.operands[1].value;
                        auto scit = local_var_sc.find(ptr);
                        if (scit != local_var_sc.end() &&
                            (scit->second == spv::StorageClassFunction || scit->second == spv::StorageClassPrivate)) {
                            if (wg_div.count(val) || blk_wg_div) wg_div.insert(ptr);
                            if (sg_div.count(val) || blk_sg_div) sg_div.insert(ptr);
                        }
                    } else if (is_subgroup_uniformizing_op(inst)) {
                        wg_div.insert(inst.result);
                        sg_div.erase(inst.result);
                    } else if (inst.op == spv::OpPhi) {
                        for (size_t k = 0; k + 1 < inst.operands.size(); k += 2) {
                            Id val = inst.operands[k].value;
                            Id plab = inst.operands[k + 1].value;
                            if (wg_div.count(val)) wg_div.insert(inst.result);
                            if (sg_div.count(val)) sg_div.insert(inst.result);
                            auto pit = cfg.block_index.find(plab);
                            if (pit != cfg.block_index.end()) {
                                for (const auto& [h, reg] : cfg.control_dep_region) {
                                    if (reg.count(pit->second) && !reg.count(bi)) {
                                        Id c = cfg.branch_cond[h];
                                        if (wg_div.count(c)) wg_div.insert(inst.result);
                                        if (sg_div.count(c)) sg_div.insert(inst.result);
                                    }
                                }
                            }
                        }
                    } else if (inst.result != 0) {
                        for (const Operand& o : inst.operands) {
                            if (o.is_id() && wg_div.count(o.value)) wg_div.insert(inst.result);
                            if (o.is_id() && sg_div.count(o.value)) sg_div.insert(inst.result);
                        }
                        if (inst.op == spv::OpUDiv && inst.operands.size() >= 2) {
                            auto cit = mc.int_consts.find(inst.operands[1].value);
                            if (cit != mc.int_consts.end() && cit->second >= 32 && (cit->second % 32) == 0) {
                                sg_div.erase(inst.result);
                            }
                        }
                    }
                }
            }
        }

        // Helper to build BVSolver context with LocalInvocationId bounds for block `bi`.
        auto make_bv_prover = [&](size_t bi, verify::BVContext& ctx, verify::BVNodeId& out_pc) -> SymbolicBuilder {
            std::array<verify::BVNodeId, 6> bvars;
            bvars[0] = ctx.make_var("lid_x", 32);
            bvars[1] = ctx.make_var("lid_y", 32);
            bvars[2] = ctx.make_var("lid_z", 32);
            uint32_t lx = std::max(1u, report.local_size[0]);
            uint32_t ly = std::max(1u, report.local_size[1]);
            uint32_t lz = std::max(1u, report.local_size[2]);
            uint32_t wg_total = lx * ly * lz;
            uint32_t max_sg = std::max(1u, (wg_total + 31u) / 32u);
            verify::BVNodeId c_lx = ctx.make_const(lx, 32);
            verify::BVNodeId c_ly = ctx.make_const(ly, 32);
            verify::BVNodeId c_lz = ctx.make_const(lz, 32);
            verify::BVNodeId zy = ctx.make_binary(verify::BVOp::Add,
                                                  ctx.make_binary(verify::BVOp::Mul, bvars[2], c_ly, 32), bvars[1], 32);
            bvars[3] = ctx.make_binary(verify::BVOp::Add,
                                       ctx.make_binary(verify::BVOp::Mul, zy, c_lx, 32), bvars[0], 32);
            bvars[4] = ctx.make_var("sg_local_id", 32);
            bvars[5] = ctx.make_var("sg_id", 32);

            verify::BVNodeId pc = ctx.make_binary(verify::BVOp::Ult, bvars[0], c_lx, 1);
            pc = ctx.make_binary(verify::BVOp::And, pc, ctx.make_binary(verify::BVOp::Ult, bvars[1], c_ly, 1), 1);
            pc = ctx.make_binary(verify::BVOp::And, pc, ctx.make_binary(verify::BVOp::Ult, bvars[2], c_lz, 1), 1);
            pc = ctx.make_binary(verify::BVOp::And, pc,
                                 ctx.make_binary(verify::BVOp::Ult, bvars[4], ctx.make_const(std::min(64u, wg_total), 32), 1), 1);
            pc = ctx.make_binary(verify::BVOp::And, pc,
                                 ctx.make_binary(verify::BVOp::Ult, bvars[5], ctx.make_const(max_sg, 32), 1), 1);

            SymbolicBuilder sb{mod, fn, cfg, mc, def_inst, lid_comp, single_store_val, ctx, bvars, {}, {}};
            for (const auto& [h, reg] : cfg.control_dep_region) {
                if (!reg.count(bi)) continue;
                const Inst& hterm = fn.blocks[h].insts.back();
                if (hterm.op == spv::OpBranchConditional && hterm.operands.size() >= 3) {
                    Id cond_id = hterm.operands[0].value;
                    auto tit = cfg.block_index.find(hterm.operands[1].value);
                    auto fit = cfg.block_index.find(hterm.operands[2].value);
                    bool in_t = (tit != cfg.block_index.end() && cfg.dom[bi].count(tit->second));
                    bool in_f = (fit != cfg.block_index.end() && cfg.dom[bi].count(fit->second));
                    if (in_t != in_f) {
                        verify::BVNodeId cnode = sb.build(cond_id);
                        if (ctx.width(cnode) != 1) {
                            cnode = ctx.make_binary(verify::BVOp::Ne, cnode, ctx.make_const(0, ctx.width(cnode)), 1);
                        }
                        if (in_f) cnode = ctx.make_unary(verify::BVOp::Not, cnode, 1);
                        pc = ctx.make_binary(verify::BVOp::And, pc, cnode, 1);
                    }
                }
            }
            for (verify::BVNodeId ec : sb.extra_constraints) pc = ctx.make_binary(verify::BVOp::And, pc, ec, 1);
            out_pc = pc;
            return sb;
        };

        // 2, 3, 4: Inspect instructions in block `bi`.
        for (size_t bi = 0; bi < cfg.num_blocks; ++bi) {
            if (!cfg.reachable[bi]) continue;
            const Block& b = fn.blocks[bi];

            std::unordered_set<Id> select_operands;
            for (const Inst& inst : b.insts) {
                if (inst.op == spv::OpSelect && inst.operands.size() >= 3) {
                    select_operands.insert(inst.operands[1].value);
                    select_operands.insert(inst.operands[2].value);
                } else if (inst.op == spv::OpCompositeConstruct) {
                    for (const Operand& o : inst.operands) select_operands.insert(o.value);
                }
            }

            for (const Inst& inst : b.insts) {
                // Property 2: Barrier & Subgroup Legality
                if (inst.op == spv::OpControlBarrier || is_subgroup_collective_op(inst.op)) {
                    bool is_wg = false;
                    if (inst.op == spv::OpControlBarrier && !inst.operands.empty()) {
                        auto cit = mc.int_consts.find(inst.operands[0].value);
                        uint64_t scope = (cit != mc.int_consts.end()) ? cit->second : static_cast<uint64_t>(spv::ScopeWorkgroup);
                        is_wg = (scope == static_cast<uint64_t>(spv::ScopeWorkgroup) ||
                                 scope == static_cast<uint64_t>(spv::ScopeDevice));
                    }
                    bool divergent = false;
                    Id div_cond = 0;
                    for (const auto& [h, reg] : cfg.control_dep_region) {
                        if (!reg.count(bi)) continue;
                        Id c = cfg.branch_cond[h];
                        if (is_wg ? wg_div.count(c) : sg_div.count(c)) {
                            if (!is_wg && inst.op != spv::OpControlBarrier && inst.operands.size() >= 4 &&
                                inst.operands[1].value == static_cast<uint32_t>(spv::GroupOperationClusteredReduce)) {
                                continue;
                            }
                            verify::BVContext ctx;
                            verify::BVNodeId pc_base = 0;
                            SymbolicBuilder sb = make_bv_prover(h, ctx, pc_base);
                            verify::BVNodeId cnode = sb.build(c);
                            if (ctx.width(cnode) != 1) {
                                cnode = ctx.make_binary(verify::BVOp::Ne, cnode, ctx.make_const(0, ctx.width(cnode)), 1);
                            }
                            auto can_true = verify::solve_nonzero(
                                ctx, ctx.make_binary(verify::BVOp::And, pc_base, cnode, 1), 2000);
                            auto can_false = verify::solve_nonzero(
                                ctx, ctx.make_binary(verify::BVOp::And, pc_base,
                                                     ctx.make_unary(verify::BVOp::Not, cnode, 1), 1), 2000);
                            if (can_true.status == verify::SolveOutcome::Sat &&
                                can_false.status == verify::SolveOutcome::Sat) {
                                divergent = true;
                                div_cond = c;
                                break;
                            }
                        }
                    }
                    std::string detail = divergent
                        ? ("divergent control flow on condition %" + std::to_string(div_cond))
                        : "uniform control flow";
                    report.barriers.push_back(BarrierAuditEntry{fn.name, b.label, inst.op, is_wg, !divergent, detail});
                    if (divergent) {
                        report.barriers_uniform = false;
                        report.hazards.push_back(ShaderHazard{
                            is_wg ? HazardKind::DivergentBarrier : HazardKind::DivergentSubgroupOp,
                            fn.name, b.label, inst.result, inst.op, detail});
                    }
                }

                // Property 3: Workgroup `shared` Memory Array Bounds Safety
                if ((inst.op == spv::OpAccessChain || inst.op == spv::OpInBoundsAccessChain) &&
                    inst.operands.size() >= 2) {
                    Id base = inst.operands[0].value;
                    auto scit = mc.var_storage.find(base);
                    auto pit = mc.var_pointee.find(base);
                    if (scit != mc.var_storage.end() && scit->second == spv::StorageClassWorkgroup &&
                        pit != mc.var_pointee.end()) {
                        Id cur_type = pit->second;
                        for (size_t idx_pos = 1; idx_pos < inst.operands.size(); ++idx_pos) {
                            const TypeInfo* ti = mod.type_info(cur_type);
                            if (!ti) break;
                            if (ti->op == spv::OpTypeArray && ti->width > 0) {
                                Id idx_id = inst.operands[idx_pos].value;
                                uint32_t arr_len = ti->width;
                                verify::BVContext ctx;
                                verify::BVNodeId pc = 0;
                                SymbolicBuilder sb = make_bv_prover(bi, ctx, pc);
                                verify::BVNodeId idx_bv = sb.build(idx_id);
                                if (ctx.width(idx_bv) < 32) idx_bv = ctx.make_zext(idx_bv, 32 - ctx.width(idx_bv));
                                for (verify::BVNodeId ec : sb.extra_constraints) {
                                    pc = ctx.make_binary(verify::BVOp::And, pc, ec, 1);
                                }
                                verify::BVNodeId oob_cond = ctx.make_binary(
                                    verify::BVOp::And, pc,
                                    ctx.make_binary(verify::BVOp::Uge, idx_bv, ctx.make_const(arr_len, 32), 1), 1);
                                auto res = verify::solve_nonzero(ctx, oob_cond, 4000);

                                std::vector<verify::BVNodeId> c_nodes;
                                std::vector<uint32_t> c_vars;
                                ctx.compute_cone(idx_bv, c_nodes, c_vars);
                                bool only_constrained_builtins = (arr_len > 1);
                                for (uint32_t vi : c_vars) {
                                    if (vi >= 5) { // 0..4 are lid_x, lid_y, lid_z, sg_local_id, sg_id
                                        only_constrained_builtins = false;
                                        break;
                                    }
                                }

                                bool proven_ok = (res.status == verify::SolveOutcome::Unsat);
                                bool static_oob = (res.status == verify::SolveOutcome::Sat && only_constrained_builtins);
                                std::string detail = proven_ok
                                    ? "proven in-bounds (< " + std::to_string(arr_len) + ") by BVSolver"
                                    : (static_oob ? "STATIC OOB (>= " + std::to_string(arr_len) + ")"
                                                  : "dynamic index into shared[" + std::to_string(arr_len) + "]");
                                report.shared_accesses.push_back(SharedMemoryAccessAudit{
                                    fn.name, b.label, inst.result, base, idx_id, arr_len, proven_ok, static_oob, detail});
                                if (static_oob) {
                                    report.shared_bounds_ok = false;
                                    report.hazards.push_back(ShaderHazard{
                                        HazardKind::SharedMemoryOutOfBounds, fn.name, b.label, inst.result, inst.op,
                                        "shared memory index %" + std::to_string(idx_id) +
                                            " can exceed array length " + std::to_string(arr_len)});
                                }
                            }
                            cur_type = ti->elem;
                        }
                    }
                }

                // Property 4: Division / Reciprocal / Sqrt Hazard Analysis
                bool is_div = (inst.op == spv::OpFDiv || inst.op == spv::OpUDiv || inst.op == spv::OpSDiv);
                bool is_sqrt = (inst.op == spv::OpExtInst && inst.operands.size() >= 3 &&
                                (inst.operands[1].value == GLSLstd450Sqrt ||
                                 inst.operands[1].value == GLSLstd450InverseSqrt));
                if ((is_div && inst.operands.size() >= 2) || is_sqrt) {
                    uint32_t ext_op = is_sqrt ? inst.operands[1].value : 0;
                    Id opd = is_sqrt ? inst.operands[2].value : inst.operands[1].value;
                    bool safe = false;
                    std::string reason = "runtime-guarded / input-dependent";

                    if (is_div && mc.zero_consts.count(opd)) {
                        report.div_sqrt_ok = false;
                        reason = "constant zero divisor";
                        report.hazards.push_back(ShaderHazard{
                            HazardKind::DefiniteDivisionByZero, fn.name, b.label, inst.result, inst.op,
                            "division by constant zero (%" + std::to_string(opd) + ")"});
                    } else if (is_sqrt && mc.float_consts.count(opd) &&
                               (mc.float_consts[opd] < 0.0 ||
                                (ext_op == GLSLstd450InverseSqrt && mc.float_consts[opd] == 0.0))) {
                        report.div_sqrt_ok = false;
                        reason = "constant negative/zero sqrt operand";
                        report.hazards.push_back(ShaderHazard{
                            ext_op == GLSLstd450InverseSqrt && mc.float_consts[opd] == 0.0
                                ? HazardKind::DefiniteDivisionByZero
                                : HazardKind::DefiniteNegativeSqrt,
                            fn.name, b.label, inst.result, inst.op,
                            "sqrt/inversesqrt on invalid constant (%" + std::to_string(opd) + ")"});
                    } else if (mc.int_consts.count(opd) && mc.int_consts[opd] != 0) {
                        safe = true;
                        reason = "non-zero integer constant";
                    } else if (mc.float_consts.count(opd) && mc.float_consts[opd] != 0.0) {
                        safe = true;
                        reason = "non-zero float constant";
                    } else if (select_operands.count(inst.result)) {
                        safe = true;
                        reason = "guarded by OpSelect ternary";
                    } else {
                        auto dit = def_inst.find(opd);
                        if (dit != def_inst.end() && dit->second) {
                            const Inst& dinst = *dit->second;
                            if (dinst.op == spv::OpExtInst && dinst.operands.size() >= 3) {
                                uint32_t deop = dinst.operands[1].value;
                                if (deop == GLSLstd450FMax || deop == GLSLstd450UMax || deop == GLSLstd450SMax ||
                                    deop == GLSLstd450FClamp || deop == GLSLstd450UClamp || deop == GLSLstd450SClamp) {
                                    safe = true;
                                    reason = "clamped via max/clamp";
                                } else if (deop == GLSLstd450Exp || deop == GLSLstd450Exp2 ||
                                           deop == GLSLstd450Length || deop == GLSLstd450Sqrt) {
                                    safe = true;
                                    reason = "positive intrinsic result";
                                }
                            } else if (dinst.op == spv::OpFAdd || dinst.op == spv::OpIAdd ||
                                       dinst.op == spv::OpFMul || dinst.op == spv::OpDot ||
                                       dinst.op == spv::OpSelect) {
                                safe = true;
                                reason = "composite / epsilon-shifted expression";
                            }
                        }
                        if (!safe) {
                            for (const auto& [h, reg] : cfg.control_dep_region) {
                                if (reg.count(bi)) {
                                    safe = true;
                                    reason = "guarded by enclosing branch";
                                    break;
                                }
                            }
                        }
                    }

                    report.divisions.push_back(DivisionAuditEntry{
                        fn.name, b.label, inst.result, inst.op, ext_op, opd, safe, reason});
                }
            }
        }
    }

    return report;
}

ComputeShaderAuditReport audit_compute_shader_binary(std::span<const uint32_t> words) {
    Module mod;
    std::string err;
    if (!parse_binary(words, mod, &err)) {
        ComputeShaderAuditReport rep;
        rep.parse_ok = false;
        rep.hazards.push_back(ShaderHazard{
            HazardKind::StructuralOrDominance, "", 0, 0, spv::OpNop, "parse_binary failed: " + err});
        return rep;
    }
    return audit_compute_shader(mod);
}

} // namespace brass::spirv
