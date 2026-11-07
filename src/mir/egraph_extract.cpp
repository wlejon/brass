#include <brass/mir/egraph.hpp>
#include "../verify/bv_solver.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/uses.hpp>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

bool extract_min_cost(const EGraph& eg, EClassId root, ExtractedExpr& out) {
    const EClassId canon_root = eg.find(root);
    const auto& classes = eg.classes();
    if (canon_root >= classes.size()) return false;

    const size_t n_classes = classes.size();
    std::vector<uint32_t> best_cost(n_classes, UINT32_MAX);
    std::vector<ENode> best_node(n_classes);
    std::vector<uint8_t> has_best(n_classes, 0);

    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < n_classes; ++i) {
            const EClassId cid = eg.find(static_cast<EClassId>(i));
            if (cid != static_cast<EClassId>(i)) continue;
            for (const ENode& raw_n : classes[cid].nodes) {
                const ENode n = eg.canonicalize(raw_n);
                const uint32_t ar = eop_arity(n.op);
                uint64_t cost = eop_cost(n.op);
                if (ar >= 1) { if (n.a >= n_classes || best_cost[n.a] == UINT32_MAX) continue; cost += best_cost[n.a]; }
                if (ar >= 2) { if (n.b >= n_classes || best_cost[n.b] == UINT32_MAX) continue; cost += best_cost[n.b]; }
                if (ar >= 3) { if (n.c >= n_classes || best_cost[n.c] == UINT32_MAX) continue; cost += best_cost[n.c]; }

                if (cost < best_cost[cid] ||
                    (cost == best_cost[cid] && has_best[cid] && n.op == EOp::Const && best_node[cid].op != EOp::Const)) {
                    best_cost[cid] = static_cast<uint32_t>(cost);
                    best_node[cid] = n;
                    has_best[cid] = 1;
                    changed = true;
                }
            }
        }
    }
    if (!has_best[canon_root] || best_cost[canon_root] == UINT32_MAX) return false;

    auto build_tree = [&](auto& self, EClassId cid, uint32_t depth) -> ExtractedExpr {
        cid = eg.find(cid);
        ExtractedExpr expr{cid, best_cost[cid], best_node[cid], nullptr, {}};
        expr.leaf_value = (expr.node.op == EOp::Var) ? eg.var_value(static_cast<uint32_t>(expr.node.imm)) : nullptr;
        if (!expr.leaf_value) expr.leaf_value = eg.class_value(cid);
        const uint32_t ar = eop_arity(expr.node.op);
        if (ar == 0 || depth > n_classes) return expr;
        expr.children.reserve(ar);
        if (ar >= 1) expr.children.push_back(self(self, expr.node.a, depth + 1));
        if (ar >= 2) expr.children.push_back(self(self, expr.node.b, depth + 1));
        if (ar >= 3) expr.children.push_back(self(self, expr.node.c, depth + 1));
        return expr;
    };
    out = build_tree(build_tree, canon_root, 0);
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
        case TypeKind::I16: return 16;
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
        if (op == Opcode::iconst_i32 || op == Opcode::patchable_const_i32) res = ctx.make_const(static_cast<uint32_t>(inst->imm_i32()), 32);
        else if (op == Opcode::iconst_i64 || op == Opcode::patchable_const_i64) res = ctx.make_const(static_cast<uint64_t>(inst->imm_i64()), 64);
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
                if (def && (def->opcode() == Opcode::iconst_i32 || def->opcode() == Opcode::patchable_const_i32)) return val_to_eclass[v] = eg.add_const(static_cast<uint32_t>(def->imm_i32()), 32);
                if (def && (def->opcode() == Opcode::iconst_i64 || def->opcode() == Opcode::patchable_const_i64)) return val_to_eclass[v] = eg.add_const(static_cast<uint64_t>(def->imm_i64()), 64);
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
            if (oc == Opcode::iconst_i32 || oc == Opcode::patchable_const_i32 || oc == Opcode::iconst_i64 || oc == Opcode::patchable_const_i64) {
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
            if (op == Opcode::iconst_i32 || op == Opcode::iconst_i64 || op == Opcode::patchable_const_i32 || op == Opcode::patchable_const_i64) {
                dominating_vals.insert(cur->result()); cur = next; continue;
            }
            Value* res_v = cur->result();
            auto it_ec = val_to_eclass.find(res_v);
            if (it_ec == val_to_eclass.end()) { dominating_vals.insert(res_v); cur = next; continue; }
            const EClassId cid = eg.find(it_ec->second);
            const uint32_t orig_cost = current_mir_cost(res_v, bb);

            ExtractedExpr best_expr;
            if (orig_cost > 0 && extract_min_cost(eg, cid, best_expr) && best_expr.cost < orig_cost &&
                expr_leaves_dominate(best_expr, var_map, dominating_vals)) {
                const BVNodeId orig_bv = enc.encode(res_v, 0);
                const BVNodeId cand_bv = encode_extracted_bv(ctx, enc, best_expr, var_map);
                if (orig_bv != kInvalidBV && cand_bv != kInvalidBV && ctx.width(orig_bv) == ctx.width(cand_bv)) {
                    const BVNodeId violated = ctx.make_binary(BVOp::And, build_path_cond(orig_bv, cand_bv), ctx.make_binary(BVOp::Ne, orig_bv, cand_bv, 1), 1);
                    if (opts.stats) opts.stats->solver_queries++;
                    if (verify::solve_nonzero(ctx, violated, limits.max_sat_conflicts).status == SolveOutcome::Unsat) {
                        verify::debug_verify_unsat_samples(ctx, violated);
                        Builder b(mod); b.set_function(&fn); b.position_before(cur);
                        if (Value* repl = materialize_extracted_mir(b, best_expr, cur, var_map, available_ssa)) {
                            enc.cache[repl] = cand_bv;
                            replace_all_uses(fn, res_v, repl);
                            available_ssa[cid] = repl;
                            dominating_vals.insert(repl);
                            if (opts.stats) { opts.stats->egraph_rewrites++; opts.stats->dags_synthesized++; }
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

    if (changed) remove_dead_pure_insts(fn);
    return changed;
}

} // namespace brass
