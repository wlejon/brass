#include <brass/mir/egraph.hpp>
#include <brass/mir/inst_opt_model.hpp>
#include "../verify/bv_solver.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/uses.hpp>
#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

enum class ExtractCostMode : uint8_t {
    StaticOps = 0,
    LatencyDepth = 1,
    MulPenaltyLatency = 2
};

uint32_t eop_mode_cost(EOp op, ExtractCostMode mode) noexcept {
    if (mode == ExtractCostMode::StaticOps) return eop_cost(op);
    if (mode == ExtractCostMode::LatencyDepth) {
        switch (op) {
            case EOp::Var: case EOp::Const: return 0;
            case EOp::Select: return 2;
            case EOp::Mul: return 3;
            case EOp::SDiv: case EOp::UDiv: case EOp::SMod: case EOp::UMod: return 10;
            default: return 1;
        }
    }
    switch (op) {
        case EOp::Var: case EOp::Const: return 0;
        case EOp::Select: return 2;
        case EOp::Mul: return 6;
        case EOp::SDiv: case EOp::UDiv: case EOp::SMod: case EOp::UMod: return 16;
        default: return 1;
    }
}

struct ModeTable {
    std::vector<uint32_t> best_metric;
    std::vector<uint32_t> best_lat;
    std::vector<uint32_t> static_cost;
    std::vector<ENode> best_node;
    std::vector<uint8_t> has_best;
};

ModeTable compute_mode_table(const EGraph& eg, ExtractCostMode mode) {
    const auto& classes = eg.classes();
    const size_t n_classes = classes.size();
    ModeTable tbl;
    tbl.best_metric.assign(n_classes, UINT32_MAX);
    tbl.best_lat.assign(n_classes, UINT32_MAX);
    tbl.static_cost.assign(n_classes, UINT32_MAX);
    tbl.best_node.resize(n_classes);
    tbl.has_best.assign(n_classes, 0);

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < n_classes; ++i) {
            const EClassId cid = eg.find(static_cast<EClassId>(i));
            if (cid != static_cast<EClassId>(i)) continue;
            for (const ENode& raw_n : classes[cid].nodes) {
                const ENode n = eg.canonicalize(raw_n);
                const uint32_t ar = eop_arity(n.op);
                uint64_t scost = eop_cost(n.op);
                uint32_t max_child_lat = 0;
                uint64_t sum_child_s = 0;
                bool ok = true;
                const EClassId ch[3] = {n.a, n.b, n.c};
                for (uint32_t k = 0; k < ar; ++k) {
                    if (ch[k] == cid || ch[k] >= n_classes || tbl.best_metric[ch[k]] == UINT32_MAX) {
                        ok = false;
                        break;
                    }
                    max_child_lat = (std::max)(max_child_lat, tbl.best_lat[ch[k]]);
                    sum_child_s += tbl.static_cost[ch[k]];
                }
                if (!ok) continue;
                scost = (std::min)(scost + sum_child_s, static_cast<uint64_t>(UINT32_MAX - 1u));
                uint64_t lat = scost;
                uint64_t metric = scost;
                if (mode != ExtractCostMode::StaticOps) {
                    lat = (std::min)(static_cast<uint64_t>(max_child_lat) + eop_mode_cost(n.op, mode),
                                     static_cast<uint64_t>(UINT32_MAX - 1u));
                    metric = (std::min)(lat * static_cast<uint64_t>(256u) + (std::min)(scost, static_cast<uint64_t>(255u)),
                                        static_cast<uint64_t>(UINT32_MAX - 1u));
                }

                if (metric < tbl.best_metric[cid] ||
                    (metric == tbl.best_metric[cid] && tbl.has_best[cid] && n.op == EOp::Const && tbl.best_node[cid].op != EOp::Const)) {
                    tbl.best_metric[cid] = static_cast<uint32_t>(metric);
                    tbl.best_lat[cid] = static_cast<uint32_t>(lat);
                    tbl.static_cost[cid] = static_cast<uint32_t>(scost);
                    tbl.best_node[cid] = n;
                    tbl.has_best[cid] = 1;
                    changed = true;
                }
            }
        }
    }
    return tbl;
}

bool build_expr_from_table(const EGraph& eg,
                           EClassId cid,
                           const ModeTable& tbl,
                           uint32_t depth,
                           std::vector<uint8_t>& visiting,
                           ExtractedExpr& out) {
    cid = eg.find(cid);
    const size_t n_classes = eg.classes().size();
    if (cid >= n_classes || !tbl.has_best[cid] || visiting[cid] || depth > 24) {
        return false;
    }
    visiting[cid] = 1;
    out = ExtractedExpr{cid, tbl.static_cost[cid], tbl.best_node[cid], nullptr, {}};
    out.leaf_value = (out.node.op == EOp::Var) ? eg.var_value(static_cast<uint32_t>(out.node.imm)) : nullptr;
    if (!out.leaf_value) out.leaf_value = eg.class_value(cid);
    const uint32_t ar = eop_arity(out.node.op);
    if (ar > 0) {
        out.children.resize(ar);
        const EClassId ch[3] = {out.node.a, out.node.b, out.node.c};
        for (uint32_t k = 0; k < ar; ++k) {
            if (!build_expr_from_table(eg, ch[k], tbl, depth + 1, visiting, out.children[k])) {
                visiting[cid] = 0;
                return false;
            }
        }
    }
    visiting[cid] = 0;
    return true;
}

std::string extracted_expr_sig(const ExtractedExpr& e, uint32_t depth = 0) {
    if (depth > 24) return "?";
    if (e.node.op == EOp::Var) return "V" + std::to_string(e.node.imm);
    if (e.node.op == EOp::Const) return "C" + std::to_string(e.node.imm);
    std::string s = "(" + std::to_string(static_cast<uint32_t>(e.node.op)) + " ";
    for (size_t i = 0; i < e.children.size(); ++i) {
        if (i > 0) s.push_back(' ');
        s += extracted_expr_sig(e.children[i], depth + 1);
    }
    s.push_back(')');
    return s;
}

std::string mir_expr_sig(const Value* v, const BasicBlock* bb, uint32_t depth = 0) {
    if (!v) return "?";
    if (!v->is_instruction() || depth >= 8) return "V" + std::to_string(v->id());
    const Instruction* inst = v->defining_instruction();
    if (!inst || inst->parent() != bb) return "V" + std::to_string(v->id());
    const Opcode oc = inst->opcode();
    if (oc == Opcode::iconst_i32) return "C" + std::to_string(static_cast<uint32_t>(inst->imm_i32()));
    if (oc == Opcode::iconst_i64) return "C" + std::to_string(static_cast<uint64_t>(inst->imm_i64()));
    EOp eop = EOp::Var;
    bool sw = false;
    // Forward-declared helper resolved below via manual mapping or identical signature helper
    if (inst->has_side_effects() || !inst->produces_value()) return "V" + std::to_string(v->id());
    switch (oc) {
        case Opcode::add:  eop = EOp::Add;  break;
        case Opcode::sub:  eop = EOp::Sub;  break;
        case Opcode::mul:  eop = EOp::Mul;  break;
        case Opcode::sdiv: eop = EOp::SDiv; break;
        case Opcode::udiv: eop = EOp::UDiv; break;
        case Opcode::smod: eop = EOp::SMod; break;
        case Opcode::umod: eop = EOp::UMod; break;
        case Opcode::and_: eop = EOp::And;  break;
        case Opcode::or_:  eop = EOp::Or;   break;
        case Opcode::xor_: eop = EOp::Xor;  break;
        case Opcode::shl:  eop = EOp::Shl;  break;
        case Opcode::lshr: eop = EOp::LShr; break;
        case Opcode::ashr: eop = EOp::AShr; break;
        case Opcode::neg:  eop = EOp::Neg;  break;
        case Opcode::not_: eop = EOp::Not;  break;
        case Opcode::select:    eop = EOp::Select;  break;
        case Opcode::sext_i64:  eop = EOp::SExt64;  break;
        case Opcode::zext_i64:  eop = EOp::ZExt64;  break;
        case Opcode::trunc_i32: eop = EOp::Trunc32; break;
        case Opcode::eq: eop = EOp::Eq; break;
        case Opcode::ne: eop = EOp::Ne; break;
        case Opcode::slt: eop = EOp::Slt; break;
        case Opcode::sgt: eop = EOp::Slt; sw = true; break;
        case Opcode::ult: eop = EOp::Ult; break;
        case Opcode::ugt: eop = EOp::Ult; sw = true; break;
        case Opcode::sle: eop = EOp::Sle; break;
        case Opcode::sge: eop = EOp::Sle; sw = true; break;
        case Opcode::ule: eop = EOp::Ule; break;
        case Opcode::uge: eop = EOp::Ule; sw = true; break;
        default: return "V" + std::to_string(v->id());
    }
    const uint32_t ar = eop_arity(eop);
    std::string ch0 = (ar >= 1 && inst->operand_count() >= 1) ? mir_expr_sig(inst->operand(0), bb, depth + 1) : "";
    std::string ch1 = (ar >= 2 && inst->operand_count() >= 2) ? mir_expr_sig(inst->operand(1), bb, depth + 1) : "";
    std::string ch2 = (ar >= 3 && inst->operand_count() >= 3) ? mir_expr_sig(inst->operand(2), bb, depth + 1) : "";
    if (sw) std::swap(ch0, ch1);
    std::string s = "(" + std::to_string(static_cast<uint32_t>(eop));
    if (ar >= 1) s += " " + ch0;
    if (ar >= 2) s += " " + ch1;
    if (ar >= 3) s += " " + ch2;
    s.push_back(')');
    return s;
}

} // namespace

bool extract_min_cost(const EGraph& eg, EClassId root, ExtractedExpr& out) {
    const EClassId canon_root = eg.find(root);
    const auto& classes = eg.classes();
    if (canon_root >= classes.size()) return false;
    const ModeTable tbl = compute_mode_table(eg, ExtractCostMode::StaticOps);
    if (!tbl.has_best[canon_root] || tbl.static_cost[canon_root] == UINT32_MAX) return false;
    std::vector<uint8_t> visiting(classes.size(), 0);
    return build_expr_from_table(eg, canon_root, tbl, 0, visiting, out);
}

bool extract_top_k_candidates(
    const EGraph& eg,
    EClassId root,
    uint32_t max_k,
    std::vector<ExtractedExpr>& out) {
    out.clear();
    if (max_k == 0) return false;
    const EClassId canon_root = eg.find(root);
    const auto& classes = eg.classes();
    if (canon_root >= classes.size()) return false;

    const ModeTable tables[3] = {
        compute_mode_table(eg, ExtractCostMode::StaticOps),
        compute_mode_table(eg, ExtractCostMode::LatencyDepth),
        compute_mode_table(eg, ExtractCostMode::MulPenaltyLatency)
    };
    if (!tables[0].has_best[canon_root]) return false;

    std::unordered_set<std::string> seen;
    auto push_unique = [&](ExtractedExpr&& cand) {
        if (out.size() >= max_k) return;
        std::string sig = extracted_expr_sig(cand);
        if (seen.insert(std::move(sig)).second) {
            out.push_back(std::move(cand));
        }
    };

    const size_t n_classes = classes.size();
    std::vector<uint8_t> visiting(n_classes, uint8_t{0});
    for (int m = 0; m < 3 && out.size() < max_k; ++m) {
        if (tables[m].has_best[canon_root]) {
            std::fill(visiting.begin(), visiting.end(), uint8_t{0});
            ExtractedExpr cand;
            if (build_expr_from_table(eg, canon_root, tables[m], 0, visiting, cand)) {
                push_unique(std::move(cand));
            }
        }
    }

    for (int m = 0; m < 2 && out.size() < max_k; ++m) {
        const ModeTable& tbl = tables[m];
        for (const ENode& raw_n : classes[canon_root].nodes) {
            if (out.size() >= max_k) break;
            const ENode n = eg.canonicalize(raw_n);
            const uint32_t ar = eop_arity(n.op);
            if ((ar >= 1 && (n.a == canon_root || n.a >= n_classes || !tbl.has_best[n.a])) ||
                (ar >= 2 && (n.b == canon_root || n.b >= n_classes || !tbl.has_best[n.b])) ||
                (ar >= 3 && (n.c == canon_root || n.c >= n_classes || !tbl.has_best[n.c]))) {
                continue;
            }
            uint64_t scost = eop_cost(n.op);
            if (ar >= 1) scost += tbl.static_cost[n.a];
            if (ar >= 2) scost += tbl.static_cost[n.b];
            if (ar >= 3) scost += tbl.static_cost[n.c];
            ExtractedExpr cand{canon_root, static_cast<uint32_t>(scost), n, nullptr, {}};
            cand.leaf_value = (n.op == EOp::Var) ? eg.var_value(static_cast<uint32_t>(n.imm)) : nullptr;
            if (!cand.leaf_value) cand.leaf_value = eg.class_value(canon_root);
            cand.children.resize(ar);
            bool ch_ok = true;
            const EClassId ch[3] = {n.a, n.b, n.c};
            for (uint32_t k = 0; k < ar; ++k) {
                std::fill(visiting.begin(), visiting.end(), uint8_t{0});
                visiting[canon_root] = 1;
                if (!build_expr_from_table(eg, ch[k], tbl, 1, visiting, cand.children[k])) {
                    ch_ok = false;
                    break;
                }
            }
            if (ch_ok) push_unique(std::move(cand));
        }
    }
    return !out.empty();
}

bool extract_min_cost_learned(
    const EGraph& eg,
    EClassId root,
    ExtractedExpr& out,
    const InstOptModel& model,
    InstOptTarget target,
    bool has_runtime_guards,
    const OnlineCostCalibrator* cal,
    float* out_predicted_cost) {
    std::vector<ExtractedExpr> cands;
    if (!extract_top_k_candidates(eg, root, 12, cands) || cands.empty()) return false;
    size_t best_idx = 0;
    float best_pred = 1e30f;
    for (size_t i = 0; i < cands.size(); ++i) {
        const InstOptFeatures feat = extract_inst_opt_features(cands[i], target, has_runtime_guards);
        const float p = model.predict_cost(feat, cal);
        if (p < best_pred || (p == best_pred && cands[i].cost < cands[best_idx].cost)) {
            best_pred = p;
            best_idx = i;
        }
    }
    out = std::move(cands[best_idx]);
    if (out_predicted_cost) *out_predicted_cost = best_pred;
    return true;
}

namespace {

using verify::BVContext;
using verify::BVNodeId;
using verify::BVOp;
using verify::SolveOutcome;

constexpr BVNodeId kInvalidBV = UINT32_MAX;

uint32_t int_bv_width(Type t) noexcept {
    switch (t.kind()) {
        case TypeKind::I8: return 8;
        case TypeKind::I16:
        case TypeKind::F16:
        case TypeKind::BF16: return 16;
        case TypeKind::I32: return 32;
        case TypeKind::I64: case TypeKind::Ptr: case TypeKind::GCRef: case TypeKind::Tagged: return 64;
        default: return 0;
    }
}

BVNodeId coerce_bv(BVContext& ctx, BVNodeId n, uint32_t target_w, bool sign_ext) {
    const uint32_t w = ctx.width(n);
    if (w == target_w) return n;
    if (w > target_w) return ctx.make_extract(n, target_w - 1, 0);
    return sign_ext ? ctx.make_sext(n, target_w - w) : ctx.make_zext(n, target_w - w);
}

BVNodeId eval_eop_bv(BVContext& ctx, EOp op, uint32_t w, BVNodeId a, BVNodeId b, BVNodeId c) {
    switch (op) {
        case EOp::SExt64:  return (a == kInvalidBV) ? kInvalidBV : coerce_bv(ctx, a, 64, true);
        case EOp::ZExt64:  return (a == kInvalidBV) ? kInvalidBV : coerce_bv(ctx, a, 64, false);
        case EOp::Trunc32: return (a == kInvalidBV) ? kInvalidBV : coerce_bv(ctx, a, 32, false);
        case EOp::Neg:
        case EOp::Not:
            return (a == kInvalidBV) ? kInvalidBV : ctx.make_unary(op == EOp::Neg ? BVOp::Neg : BVOp::Not, coerce_bv(ctx, a, w, false), w);
        case EOp::Add: case EOp::Sub: case EOp::Mul: case EOp::SDiv: case EOp::UDiv: case EOp::SMod: case EOp::UMod:
        case EOp::And: case EOp::Or: case EOp::Xor: case EOp::Shl: case EOp::LShr: case EOp::AShr: {
            if (a == kInvalidBV || b == kInvalidBV) return kInvalidBV;
            const BVOp bop = (op == EOp::Add)  ? BVOp::Add  : (op == EOp::Sub)  ? BVOp::Sub  : (op == EOp::Mul)  ? BVOp::Mul
                           : (op == EOp::SDiv) ? BVOp::SDiv : (op == EOp::UDiv) ? BVOp::UDiv : (op == EOp::SMod) ? BVOp::SRem
                           : (op == EOp::UMod) ? BVOp::URem : (op == EOp::And)  ? BVOp::And  : (op == EOp::Or)   ? BVOp::Or
                           : (op == EOp::Xor)  ? BVOp::Xor  : (op == EOp::Shl)  ? BVOp::Shl  : (op == EOp::LShr) ? BVOp::LShr : BVOp::AShr;
            return ctx.make_binary(bop, coerce_bv(ctx, a, w, true), coerce_bv(ctx, b, w, true), w);
        }
        case EOp::Eq: case EOp::Ne: case EOp::Slt: case EOp::Ult: case EOp::Sle: case EOp::Ule: {
            if (a == kInvalidBV || b == kInvalidBV) return kInvalidBV;
            const uint32_t cw = std::max(ctx.width(a), ctx.width(b));
            const BVOp cop = (op == EOp::Eq) ? BVOp::Eq : (op == EOp::Ne) ? BVOp::Ne : (op == EOp::Slt) ? BVOp::Slt
                           : (op == EOp::Ult) ? BVOp::Ult : (op == EOp::Sle) ? BVOp::Sle : BVOp::Ule;
            return ctx.make_zext(ctx.make_binary(cop, coerce_bv(ctx, a, cw, false), coerce_bv(ctx, b, cw, false), 1), w - 1);
        }
        case EOp::Select: {
            if (a == kInvalidBV || b == kInvalidBV || c == kInvalidBV) return kInvalidBV;
            const BVNodeId c_bit = ctx.make_binary(BVOp::Ne, a, ctx.make_const(0, ctx.width(a)), 1);
            return ctx.make_ite(c_bit, coerce_bv(ctx, b, w, false), coerce_bv(ctx, c, w, false));
        }
        default: return kInvalidBV;
    }
}

bool map_inst_to_eop(const Instruction* inst, EOp& out_op, bool& out_swap_operands) noexcept {
    out_swap_operands = false;
    if (!inst || inst->has_side_effects() || !inst->produces_value()) return false;
    const Type res_t = inst->type();
    if (res_t != Type::i32() && res_t != Type::i64()) return false;
    const Opcode oc = inst->opcode();
    if (oc == Opcode::add)  { out_op = EOp::Add;  return true; }
    if (oc == Opcode::sub)  { out_op = EOp::Sub;  return true; }
    if (oc == Opcode::mul)  { out_op = EOp::Mul;  return true; }
    if (oc == Opcode::sdiv) { out_op = EOp::SDiv; return true; }
    if (oc == Opcode::udiv) { out_op = EOp::UDiv; return true; }
    if (oc == Opcode::smod) { out_op = EOp::SMod; return true; }
    if (oc == Opcode::umod) { out_op = EOp::UMod; return true; }
    if (oc == Opcode::and_) { out_op = EOp::And;  return true; }
    if (oc == Opcode::or_)  { out_op = EOp::Or;   return true; }
    if (oc == Opcode::xor_) { out_op = EOp::Xor;  return true; }
    if (oc == Opcode::shl)  { out_op = EOp::Shl;  return true; }
    if (oc == Opcode::lshr) { out_op = EOp::LShr; return true; }
    if (oc == Opcode::ashr) { out_op = EOp::AShr; return true; }
    if (oc == Opcode::neg)  { out_op = EOp::Neg;  return true; }
    if (oc == Opcode::not_) { out_op = EOp::Not;  return true; }
    if (oc == Opcode::select)    { out_op = EOp::Select;  return inst->operand(0) && inst->operand(0)->type() == Type::i32(); }
    if (oc == Opcode::sext_i64)  { out_op = EOp::SExt64;  return inst->operand(0) && inst->operand(0)->type() == Type::i32(); }
    if (oc == Opcode::zext_i64)  { out_op = EOp::ZExt64;  return inst->operand(0) && inst->operand(0)->type() == Type::i32(); }
    if (oc == Opcode::trunc_i32) { out_op = EOp::Trunc32; return inst->operand(0) && inst->operand(0)->type() == Type::i64(); }
    if (oc == Opcode::eq || oc == Opcode::ne || oc == Opcode::slt || oc == Opcode::ult || oc == Opcode::sle || oc == Opcode::ule ||
        oc == Opcode::sgt || oc == Opcode::ugt || oc == Opcode::sge || oc == Opcode::uge) {
        if (!inst->operand(0) || !inst->operand(1)) return false;
        const Type op_t = inst->operand(0)->type();
        if ((op_t != Type::i32() && op_t != Type::i64()) || inst->operand(1)->type() != op_t) return false;
        out_swap_operands = (oc == Opcode::sgt || oc == Opcode::ugt || oc == Opcode::sge || oc == Opcode::uge);
        out_op = (oc == Opcode::eq) ? EOp::Eq : (oc == Opcode::ne) ? EOp::Ne
               : (oc == Opcode::slt || oc == Opcode::sgt) ? EOp::Slt
               : (oc == Opcode::ult || oc == Opcode::ugt) ? EOp::Ult
               : (oc == Opcode::sle || oc == Opcode::sge) ? EOp::Sle : EOp::Ule;
        return true;
    }
    return false;
}

struct EGraphBVEncoder {
    BVContext& ctx;
    const RangeAnalysis& ra;
    const BasicBlock* cur_bb;
    uint32_t max_depth;
    std::unordered_map<const Value*, BVNodeId> cache;
    std::vector<BVNodeId> range_assumptions;

    void add_range_constraint(const Value* v, BVNodeId node) {
        if (!v || !cur_bb || node == kInvalidBV) return;
        const uint32_t w = ctx.width(node);
        if (w != 32 && w != 64) return;
        const ValueRange r = ra.get_range_at(v, cur_bb);
        if (r.is_empty()) return;
        const int64_t tmin = (w == 32) ? static_cast<int64_t>(INT32_MIN) : INT64_MIN;
        const int64_t tmax = (w == 32) ? static_cast<int64_t>(INT32_MAX) : INT64_MAX;
        if (r.min_val > tmin && r.min_val <= tmax) range_assumptions.push_back(ctx.make_binary(BVOp::Sge, node, ctx.make_const(static_cast<uint64_t>(r.min_val), w), 1));
        if (r.max_val < tmax && r.max_val >= tmin) range_assumptions.push_back(ctx.make_binary(BVOp::Sle, node, ctx.make_const(static_cast<uint64_t>(r.max_val), w), 1));
    }

    BVNodeId encode(const Value* v, uint32_t depth = 0) {
        if (!v) return kInvalidBV;
        if (auto it = cache.find(v); it != cache.end()) return it->second;
        const uint32_t w = int_bv_width(v->type());
        if (w == 0) return kInvalidBV;
        auto make_leaf = [&]() -> BVNodeId {
            const BVNodeId var = ctx.make_var("v" + std::to_string(v->id()), w, false);
            cache[v] = var;
            add_range_constraint(v, var);
            return var;
        };
        if (!v->is_instruction() || depth >= max_depth || !v->defining_instruction()) return make_leaf();
        const Instruction* inst = v->defining_instruction();
        const Opcode op = inst->opcode();
        BVNodeId res = kInvalidBV;
        if (op == Opcode::iconst_i32) res = ctx.make_const(static_cast<uint32_t>(inst->imm_i32()), 32);
        else if (op == Opcode::iconst_i64) res = ctx.make_const(static_cast<uint64_t>(inst->imm_i64()), 64);
        else {
            EOp eop = EOp::Var; bool sw = false;
            if (!map_inst_to_eop(inst, eop, sw)) return make_leaf();
            const uint32_t ar = eop_arity(eop);
            BVNodeId a = (ar >= 1) ? encode(inst->operand(0), depth + 1) : kInvalidBV;
            BVNodeId b = (ar >= 2) ? encode(inst->operand(1), depth + 1) : kInvalidBV;
            const BVNodeId c = (ar >= 3) ? encode(inst->operand(2), depth + 1) : kInvalidBV;
            if (sw) std::swap(a, b);
            res = eval_eop_bv(ctx, eop, w, a, b, c);
            if (res == kInvalidBV) return make_leaf();
        }
        cache[v] = res;
        add_range_constraint(v, res);
        return res;
    }

    BVNodeId encode_bool(const Value* cond_val) {
        const BVNodeId bv = encode(cond_val, 0);
        return (bv == kInvalidBV || ctx.width(bv) == 1) ? bv : ctx.make_binary(BVOp::Ne, bv, ctx.make_const(0, ctx.width(bv)), 1);
    }
};

BVNodeId encode_extracted_bv(
    BVContext& ctx, EGraphBVEncoder& enc, const ExtractedExpr& expr, const std::unordered_map<uint32_t, Value*>& var_map) {
    const uint32_t w = expr.node.width;
    if (expr.node.op == EOp::Var) {
        if (expr.leaf_value) return enc.encode(expr.leaf_value, 0);
        auto it = var_map.find(static_cast<uint32_t>(expr.node.imm));
        return (it != var_map.end()) ? enc.encode(it->second, 0) : kInvalidBV;
    }
    if (expr.node.op == EOp::Const) return ctx.make_const(expr.node.imm, w);
    const uint32_t ar = eop_arity(expr.node.op);
    if (expr.children.size() < ar) return kInvalidBV;
    return eval_eop_bv(ctx, expr.node.op, w,
        (ar >= 1) ? encode_extracted_bv(ctx, enc, expr.children[0], var_map) : kInvalidBV,
        (ar >= 2) ? encode_extracted_bv(ctx, enc, expr.children[1], var_map) : kInvalidBV,
        (ar >= 3) ? encode_extracted_bv(ctx, enc, expr.children[2], var_map) : kInvalidBV);
}

uint32_t current_mir_cost(const Value* v, const BasicBlock* bb, uint32_t depth = 0) noexcept {
    if (!v || !v->is_instruction() || depth > 8) return 0;
    const Instruction* inst = v->defining_instruction();
    if (!inst || inst->parent() != bb) return 0;
    const Opcode oc = inst->opcode();
    if (oc == Opcode::iconst_i32 || oc == Opcode::iconst_i64 || oc == Opcode::patchable_const_i32 || oc == Opcode::patchable_const_i64) return 0;
    EOp eop = EOp::Var; bool sw = false;
    if (!map_inst_to_eop(inst, eop, sw)) return 0;
    uint32_t cost = eop_cost(eop);
    for (size_t i = 0; i < inst->operand_count(); ++i) cost += current_mir_cost(inst->operand(i), bb, depth + 1);
    return cost;
}

bool expr_leaves_dominate(
    const ExtractedExpr& expr, const std::unordered_map<uint32_t, Value*>& var_map, const std::unordered_set<const Value*>& dom_vals) {
    if (expr.node.op == EOp::Var) {
        const Value* v = expr.leaf_value;
        if (!v) { if (auto it = var_map.find(static_cast<uint32_t>(expr.node.imm)); it != var_map.end()) v = it->second; }
        return v != nullptr && dom_vals.find(v) != dom_vals.end();
    }
    for (const ExtractedExpr& ch : expr.children) { if (!expr_leaves_dominate(ch, var_map, dom_vals)) return false; }
    return true;
}

Value* materialize_extracted_mir(
    Builder& b, const ExtractedExpr& expr, const Instruction* before_inst,
    const std::unordered_map<uint32_t, Value*>& var_map, std::unordered_map<EClassId, Value*>& available_ssa) {
    const Type t = (expr.node.width == 32) ? Type::i32() : Type::i64();
    if (expr.node.op == EOp::Var) {
        if (expr.leaf_value && expr.leaf_value->type() == t) return expr.leaf_value;
        auto it = var_map.find(static_cast<uint32_t>(expr.node.imm));
        return (it != var_map.end() && it->second->type() == t) ? it->second : nullptr;
    }
    if (expr.node.op == EOp::Const) {
        return (t == Type::i32()) ? b.build_iconst_i32(static_cast<int32_t>(expr.node.imm))
                                  : b.build_iconst_i64(static_cast<int64_t>(expr.node.imm));
    }
    if (expr.eclass != kInvalidEClass) {
        auto it = available_ssa.find(expr.eclass);
        if (it != available_ssa.end() && it->second && it->second != before_inst->result() &&
            it->second->type() == t && current_mir_cost(it->second, before_inst->parent()) <= expr.cost) {
            return it->second;
        }
    }
    const uint32_t ar = eop_arity(expr.node.op);
    if (expr.children.size() < ar) return nullptr;
    Value* v0 = (ar >= 1) ? materialize_extracted_mir(b, expr.children[0], before_inst, var_map, available_ssa) : nullptr;
    Value* v1 = (ar >= 2) ? materialize_extracted_mir(b, expr.children[1], before_inst, var_map, available_ssa) : nullptr;
    Value* v2 = (ar >= 3) ? materialize_extracted_mir(b, expr.children[2], before_inst, var_map, available_ssa) : nullptr;
    if ((ar >= 1 && !v0) || (ar >= 2 && !v1) || (ar >= 3 && !v2)) return nullptr;

    Value* res = nullptr;
    const bool same_t = (ar >= 2 && v0->type() == t && v1->type() == t);
    const bool cmp_ok = (ar >= 2 && v0->type() == v1->type() && t == Type::i32());
    const EOp op = expr.node.op;
    if (same_t) {
        if      (op == EOp::Add)  res = b.build_add(v0, v1);
        else if (op == EOp::Sub)  res = b.build_sub(v0, v1);
        else if (op == EOp::Mul)  res = b.build_mul(v0, v1);
        else if (op == EOp::SDiv) res = b.build_sdiv(v0, v1);
        else if (op == EOp::UDiv) res = b.build_udiv(v0, v1);
        else if (op == EOp::SMod) res = b.build_smod(v0, v1);
        else if (op == EOp::UMod) res = b.build_umod(v0, v1);
        else if (op == EOp::And)  res = b.build_and(v0, v1);
        else if (op == EOp::Or)   res = b.build_or(v0, v1);
        else if (op == EOp::Xor)  res = b.build_xor(v0, v1);
        else if (op == EOp::Shl)  res = b.build_shl(v0, v1);
        else if (op == EOp::LShr) res = b.build_lshr(v0, v1);
        else if (op == EOp::AShr) res = b.build_ashr(v0, v1);
    }
    if (!res && cmp_ok) {
        if      (op == EOp::Eq)  res = b.build_eq(v0, v1);
        else if (op == EOp::Ne)  res = b.build_ne(v0, v1);
        else if (op == EOp::Slt) res = b.build_slt(v0, v1);
        else if (op == EOp::Ult) res = b.build_ult(v0, v1);
        else if (op == EOp::Sle) res = b.build_sle(v0, v1);
        else if (op == EOp::Ule) res = b.build_ule(v0, v1);
    }
    if (!res) {
        if      (op == EOp::Neg && v0->type() == t) res = b.build_neg(v0);
        else if (op == EOp::Not && v0->type() == t) res = b.build_not(v0);
        else if (op == EOp::Select && v0->type() == Type::i32() && v1->type() == t && v2->type() == t) res = b.build_select(v0, v1, v2);
        else if (op == EOp::SExt64 && t == Type::i64() && v0->type() == Type::i32()) res = b.build_sext_i64(v0);
        else if (op == EOp::ZExt64 && t == Type::i64() && v0->type() == Type::i32()) res = b.build_zext_i64(v0);
        else if (op == EOp::Trunc32 && t == Type::i32() && v0->type() == Type::i64()) res = b.build_trunc_i32(v0);
    }
    if (res && expr.eclass != kInvalidEClass) available_ssa[expr.eclass] = res;
    return res;
}

void remove_dead_pure_insts(Function& fn) {
    for (bool progress = true; progress;) {
        progress = false;
        auto uses = compute_use_counts(fn);
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (Instruction* cur = bb->head(); cur;) {
                Instruction* next = cur->next();
                if (!cur->has_side_effects() && cur->produces_value() && uses[cur->result()] == 0) {
                    bb->remove_instruction(cur); progress = true;
                }
                cur = next;
            }
        }
    }
}

} // namespace

bool run_mir_egraph_superopt(Function& fn, Module& mod, const SuperoptOptions& opts, const EGraphLimits& limits) {
    if (fn.blocks().empty() || !fn.entry_block()) return false;
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);
    RangeAnalysis ra(fn, dom, loops);
    bool changed = false;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        EGraph eg;
        std::unordered_map<const Value*, EClassId> val_to_eclass;
        std::unordered_map<uint32_t, Value*> var_map;
        std::unordered_map<EClassId, Value*> pre_block_ssa;

        auto get_or_add_leaf = [&](Value* v) -> EClassId {
            if (!v) return kInvalidEClass;
            if (auto it = val_to_eclass.find(v); it != val_to_eclass.end()) return eg.find(it->second);
            if (eg.num_enodes() >= limits.max_enodes || eg.num_eclasses() >= limits.max_eclasses) return kInvalidEClass;
            const Type vt = v->type();
            if (vt != Type::i32() && vt != Type::i64()) return kInvalidEClass;
            if (v->is_instruction()) {
                const Instruction* def = v->defining_instruction();
                if (def && def->opcode() == Opcode::iconst_i32) return val_to_eclass[v] = eg.add_const(static_cast<uint32_t>(def->imm_i32()), 32);
                if (def && def->opcode() == Opcode::iconst_i64) return val_to_eclass[v] = eg.add_const(static_cast<uint64_t>(def->imm_i64()), 64);
            }
            const EClassId cid = eg.add_var(v->id(), (vt == Type::i32()) ? 32u : 64u, v);
            val_to_eclass[v] = cid; var_map[v->id()] = v;
            if (!v->is_instruction() || (v->defining_instruction() && v->defining_instruction()->parent() != bb)) pre_block_ssa[cid] = v;
            return cid;
        };

        for (Instruction* inst : *bb) {
            if (!inst || inst->has_side_effects() || !inst->produces_value()) continue;
            if (eg.num_enodes() >= limits.max_enodes || eg.num_eclasses() >= limits.max_eclasses) break;
            Value* res_v = inst->result();
            if (!res_v || (res_v->type() != Type::i32() && res_v->type() != Type::i64())) continue;
            const uint32_t w = (res_v->type() == Type::i32()) ? 32u : 64u;
            const Opcode oc = inst->opcode();
            if (oc == Opcode::iconst_i32 || oc == Opcode::iconst_i64) {
                (void)get_or_add_leaf(res_v); continue;
            }
            EOp eop = EOp::Var; bool swap_ops = false;
            if (!map_inst_to_eop(inst, eop, swap_ops)) { (void)get_or_add_leaf(res_v); continue; }
            const uint32_t ar = eop_arity(eop);
            EClassId ca = (ar >= 1) ? get_or_add_leaf(inst->operand(0)) : kInvalidEClass;
            EClassId cb = (ar >= 2) ? get_or_add_leaf(inst->operand(1)) : kInvalidEClass;
            const EClassId cc = (ar >= 3) ? get_or_add_leaf(inst->operand(2)) : kInvalidEClass;
            if ((ar >= 1 && ca == kInvalidEClass) || (ar >= 2 && cb == kInvalidEClass) || (ar >= 3 && cc == kInvalidEClass)) continue;
            if (swap_ops) std::swap(ca, cb);
            val_to_eclass[res_v] = eg.add_node(ENode{eop, w, ca, cb, cc, 0}, res_v);
        }
        if (eg.num_enodes() == 0) continue;
        EGraphStats eg_stats;
        eg.saturate(limits, &eg_stats);

        std::vector<const BasicBlock*> dom_chain;
        for (const BasicBlock* d : fn.blocks()) { if (d && d != bb && dom.dominates(d, bb)) dom_chain.push_back(d); }

        BVContext ctx;
        EGraphBVEncoder enc{ctx, ra, bb, std::max<uint32_t>(opts.max_window_depth, 8u), {}, {}};
        std::vector<BVNodeId> assumptions;
        for (const BasicBlock* dbb : dom_chain) {
            for (const Instruction* inst : *dbb) {
                if (inst && inst->opcode() == Opcode::guard && inst->operand(0)) {
                    if (const BVNodeId g = enc.encode_bool(inst->operand(0)); g != kInvalidBV) assumptions.push_back(g);
                }
            }
        }
        std::vector<const BasicBlock*> all_doms = dom_chain;
        all_doms.push_back(bb);
        for (const BasicBlock* anc : all_doms) {
            if (anc->predecessors().size() != 1) continue;
            const BasicBlock* pred = anc->predecessors()[0];
            const Instruction* term = pred ? pred->terminator() : nullptr;
            if (term && term->opcode() == Opcode::br_if && term->true_target().block != term->false_target().block && term->operand(0)) {
                if (const BVNodeId c_bit = enc.encode_bool(term->operand(0)); c_bit != kInvalidBV) {
                    if (term->true_target().block == anc) assumptions.push_back(c_bit);
                    else if (term->false_target().block == anc) assumptions.push_back(ctx.make_unary(BVOp::Not, c_bit, 1));
                }
            }
        }

        auto build_path_cond = [&](BVNodeId orig_bv, BVNodeId cand_bv) -> BVNodeId {
            verify::BVCone& sc = ctx.scratch_cone();
            ctx.compute_cone(ctx.make_binary(BVOp::Xor, orig_bv, cand_bv, ctx.width(orig_bv)), sc.nodes, sc.vars);
            if (sc.vars.empty()) return ctx.make_const(1, 1);
            sc.var_seen.assign(ctx.vars().size(), 0);
            for (uint32_t vi : sc.vars) { if (vi < sc.var_seen.size()) sc.var_seen[vi] = 1; }
            BVNodeId pc = ctx.make_const(1, 1);
            auto try_add_fact = [&](BVNodeId fact) {
                std::vector<BVNodeId> fnodes; std::vector<uint32_t> fvars;
                ctx.compute_cone(fact, fnodes, fvars);
                for (uint32_t fvi : fvars) {
                    if (fvi < sc.var_seen.size() && sc.var_seen[fvi]) { pc = ctx.make_binary(BVOp::And, pc, fact, 1); return; }
                }
            };
            for (BVNodeId a : assumptions) try_add_fact(a);
            for (BVNodeId r : enc.range_assumptions) try_add_fact(r);
            return pc;
        };

        std::unordered_map<EClassId, Value*> available_ssa;
        std::unordered_set<const Value*> dominating_vals;
        for (const auto& [cid, v] : pre_block_ssa) { available_ssa[eg.find(cid)] = v; dominating_vals.insert(v); }

        for (Instruction* cur = bb->head(); cur;) {
            Instruction* next = cur->next();
            if (cur->opcode() == Opcode::guard && cur->operand(0)) {
                if (const BVNodeId g = enc.encode_bool(cur->operand(0)); g != kInvalidBV) assumptions.push_back(g);
                cur = next; continue;
            }
            if (cur->has_side_effects() || !cur->produces_value()) {
                if (cur->produces_value() && cur->result()) dominating_vals.insert(cur->result());
                cur = next; continue;
            }
            const Opcode op = cur->opcode();
            if (op == Opcode::iconst_i32 || op == Opcode::iconst_i64) {
                dominating_vals.insert(cur->result()); cur = next; continue;
            }
            Value* res_v = cur->result();
            auto it_ec = val_to_eclass.find(res_v);
            if (it_ec == val_to_eclass.end()) { dominating_vals.insert(res_v); cur = next; continue; }
            const EClassId cid = eg.find(it_ec->second);
            const uint32_t orig_cost = current_mir_cost(res_v, bb);

            ExtractedExpr best_expr;
            bool accept_candidate = false;
            bool used_learned_model = false;
            if (opts.use_inst_opt_model && orig_cost > 0) {
                const InstOptModel& mdl = opts.inst_opt_model ? *opts.inst_opt_model : InstOptModel::builtin();
                const InstOptTarget tgt = opts.enable_gpu_superopt ? InstOptTarget::GPU_PTX : InstOptTarget::X64;
                const bool has_guards = !assumptions.empty();
                const float orig_pred = mdl.predict_cost(
                    extract_inst_opt_features(cur, bb, tgt, has_guards), opts.cost_calibrator);
                float cand_pred = 0.0f;
                if (extract_min_cost_learned(eg, cid, best_expr, mdl, tgt, has_guards, opts.cost_calibrator, &cand_pred) &&
                    ((cand_pred < orig_pred * 0.985f) || (cand_pred <= orig_pred && best_expr.cost < orig_cost)) &&
                    extracted_expr_sig(best_expr) != mir_expr_sig(res_v, bb)) {
                    accept_candidate = true;
                    used_learned_model = true;
                }
            } else if (orig_cost > 0 && extract_min_cost(eg, cid, best_expr) && best_expr.cost < orig_cost) {
                accept_candidate = true;
            }
            if (accept_candidate &&
                expr_leaves_dominate(best_expr, var_map, dominating_vals)) {
                const BVNodeId orig_bv = enc.encode(res_v, 0);
                const BVNodeId cand_bv = encode_extracted_bv(ctx, enc, best_expr, var_map);
                if (orig_bv != kInvalidBV && cand_bv != kInvalidBV && ctx.width(orig_bv) == ctx.width(cand_bv)) {
                    const BVNodeId pc = build_path_cond(orig_bv, cand_bv);
                    bool trap_safe = true;
                    if (op == Opcode::sdiv || op == Opcode::udiv || op == Opcode::smod || op == Opcode::umod) {
                        const uint32_t dw = ctx.width(orig_bv);
                        const BVNodeId lhs_bv = enc.encode(cur->operand(0), 0);
                        const BVNodeId rhs_bv = enc.encode(cur->operand(1), 0);
                        if (lhs_bv == kInvalidBV || rhs_bv == kInvalidBV) {
                            trap_safe = false;
                        } else {
                            BVNodeId trap_cond = ctx.make_binary(BVOp::Eq, rhs_bv, ctx.make_const(0, dw), 1);
                            if (op == Opcode::sdiv || op == Opcode::smod) {
                                const uint64_t smin = (dw == 32) ? 0x80000000ULL : 0x8000000000000000ULL;
                                const uint64_t neg1 = (dw == 32) ? 0xFFFFFFFFULL : ~0ULL;
                                BVNodeId ovf = ctx.make_binary(
                                    BVOp::And,
                                    ctx.make_binary(BVOp::Eq, lhs_bv, ctx.make_const(smin, dw), 1),
                                    ctx.make_binary(BVOp::Eq, rhs_bv, ctx.make_const(neg1, dw), 1), 1);
                                trap_cond = ctx.make_binary(BVOp::Or, trap_cond, ovf, 1);
                            }
                            const BVNodeId trap_viol = ctx.make_binary(BVOp::And, pc, trap_cond, 1);
                            if (opts.stats) opts.stats->solver_queries++;
                            trap_safe = (verify::solve_nonzero(ctx, trap_viol, limits.max_sat_conflicts).status == SolveOutcome::Unsat);
                        }
                    }
                    const BVNodeId violated = ctx.make_binary(BVOp::And, pc, ctx.make_binary(BVOp::Ne, orig_bv, cand_bv, 1), 1);
                    if (opts.stats) opts.stats->solver_queries++;
                    if (trap_safe && verify::solve_nonzero(ctx, violated, limits.max_sat_conflicts).status == SolveOutcome::Unsat) {
                        verify::debug_verify_unsat_samples(ctx, violated);
                        Builder b(mod); b.set_function(&fn); b.position_before(cur);
                        if (Value* repl = materialize_extracted_mir(b, best_expr, cur, var_map, available_ssa)) {
                            enc.cache[repl] = cand_bv;
                            fn.retarget_guard_elim_witness(res_v, repl);
                            replace_all_uses(fn, res_v, repl);
                            available_ssa[cid] = repl;
                            dominating_vals.insert(repl);
                            if (opts.stats) {
                                opts.stats->egraph_rewrites++;
                                opts.stats->dags_synthesized++;
                                if (used_learned_model) opts.stats->model_guided_rewrites++;
                                if ((op == Opcode::and_ || op == Opcode::or_) && cur->operand_count() == 2 &&
                                    (repl == cur->operand(0) || repl == cur->operand(1))) {
                                    opts.stats->masks_elided++;
                                }
                            }
                            changed = true; cur = next; continue;
                        }
                    }
                }
            }
            dominating_vals.insert(res_v);
            if (available_ssa.find(cid) == available_ssa.end()) available_ssa[cid] = res_v;
            cur = next;
        }
    }

    if (changed) {
        remove_dead_pure_insts(fn);
        fn.reconcile_guard_elim_certs();
    }
    return changed;
}

} // namespace brass
