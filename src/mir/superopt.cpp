#include <brass/mir/superopt.hpp>
#include "../verify/bv_solver.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/uses.hpp>
#include <brass/verify/smt_equiv.hpp>
#include <algorithm>
#include <bit>
#include <unordered_map>
#include <vector>

namespace brass {

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
        case TypeKind::I64:
        case TypeKind::Ptr:
        case TypeKind::GCRef:
        case TypeKind::Tagged: return 64;
        default: return 0;
    }
}

BVNodeId coerce_bv(BVContext& ctx, BVNodeId n, uint32_t target_w, bool sign_ext) {
    uint32_t w = ctx.width(n);
    if (w == target_w) return n;
    if (w > target_w) return ctx.make_extract(n, target_w - 1, 0);
    return sign_ext ? ctx.make_sext(n, target_w - w) : ctx.make_zext(n, target_w - w);
}

bool get_const_imm(const Value* v, int64_t& out_imm) noexcept {
    if (!v || !v->is_instruction()) return false;
    const Instruction* inst = v->defining_instruction();
    if (!inst) return false;
    if (inst->opcode() == Opcode::iconst_i32 || inst->opcode() == Opcode::patchable_const_i32) {
        out_imm = inst->imm_i32();
        return true;
    }
    if (inst->opcode() == Opcode::iconst_i64 || inst->opcode() == Opcode::patchable_const_i64) {
        out_imm = inst->imm_i64();
        return true;
    }
    return false;
}

Value* build_typed_const(Builder& b, Type t, uint64_t bits) {
    if (t == Type::i32()) return b.build_iconst_i32(static_cast<int32_t>(bits));
    if (t == Type::i64()) return b.build_iconst_i64(static_cast<int64_t>(bits));
    return nullptr;
}

struct SuperoptEncoder {
    BVContext& ctx;
    const RangeAnalysis& ra;
    const BasicBlock* cur_bb;
    uint32_t max_depth;
    std::unordered_map<const Value*, BVNodeId> cache;
    std::vector<BVNodeId> range_assumptions;

    void add_range_constraint(const Value* v, BVNodeId node) {
        if (!v || !cur_bb || node == kInvalidBV) return;
        uint32_t w = ctx.width(node);
        if (w != 32 && w != 64) return;
        ValueRange r = ra.get_range_at(v, cur_bb);
        if (r.is_empty()) return;
        int64_t type_min = (w == 32) ? static_cast<int64_t>(INT32_MIN) : INT64_MIN;
        int64_t type_max = (w == 32) ? static_cast<int64_t>(INT32_MAX) : INT64_MAX;
        if (r.min_val > type_min && r.min_val <= type_max) {
            BVNodeId lo = ctx.make_const(static_cast<uint64_t>(r.min_val), w);
            range_assumptions.push_back(ctx.make_binary(BVOp::Sge, node, lo, 1));
        }
        if (r.max_val < type_max && r.max_val >= type_min) {
            BVNodeId hi = ctx.make_const(static_cast<uint64_t>(r.max_val), w);
            range_assumptions.push_back(ctx.make_binary(BVOp::Sle, node, hi, 1));
        }
    }

    BVNodeId encode(const Value* v, uint32_t depth = 0) {
        if (!v) return kInvalidBV;
        auto it = cache.find(v);
        if (it != cache.end()) return it->second;

        uint32_t w = int_bv_width(v->type());
        if (w == 0) return kInvalidBV;

        auto make_leaf = [&]() -> BVNodeId {
            BVNodeId var = ctx.make_var("v" + std::to_string(v->id()), w, false);
            cache[v] = var;
            add_range_constraint(v, var);
            return var;
        };

        if (!v->is_instruction() || depth >= max_depth) {
            return make_leaf();
        }
        const Instruction* inst = v->defining_instruction();
        if (!inst) return make_leaf();

        Opcode op = inst->opcode();
        BVNodeId res = kInvalidBV;

        switch (op) {
            case Opcode::iconst_i32:
            case Opcode::patchable_const_i32:
                res = ctx.make_const(static_cast<uint32_t>(inst->imm_i32()), 32);
                break;
            case Opcode::iconst_i64:
            case Opcode::patchable_const_i64:
                res = ctx.make_const(static_cast<uint64_t>(inst->imm_i64()), 64);
                break;
            case Opcode::sext_i64: {
                BVNodeId a = encode(inst->operand(0), depth + 1);
                if (a == kInvalidBV) return make_leaf();
                res = coerce_bv(ctx, a, 64, true);
                break;
            }
            case Opcode::zext_i64: {
                BVNodeId a = encode(inst->operand(0), depth + 1);
                if (a == kInvalidBV) return make_leaf();
                res = coerce_bv(ctx, a, 64, false);
                break;
            }
            case Opcode::trunc_i32: {
                BVNodeId a = encode(inst->operand(0), depth + 1);
                if (a == kInvalidBV) return make_leaf();
                res = coerce_bv(ctx, a, 32, false);
                break;
            }
            case Opcode::trunc_i8: {
                BVNodeId a = encode(inst->operand(0), depth + 1);
                if (a == kInvalidBV) return make_leaf();
                res = coerce_bv(ctx, a, 8, false);
                break;
            }
            case Opcode::add: case Opcode::sub: case Opcode::mul:
            case Opcode::sdiv: case Opcode::udiv: case Opcode::smod: case Opcode::umod:
            case Opcode::and_: case Opcode::or_: case Opcode::xor_:
            case Opcode::shl: case Opcode::lshr: case Opcode::ashr: {
                if (inst->type().is_float()) return make_leaf();
                BVNodeId a = encode(inst->operand(0), depth + 1);
                BVNodeId b = encode(inst->operand(1), depth + 1);
                if (a == kInvalidBV || b == kInvalidBV) return make_leaf();
                a = coerce_bv(ctx, a, w, true);
                b = coerce_bv(ctx, b, w, true);
                BVOp bop = (op == Opcode::add)  ? BVOp::Add  : (op == Opcode::sub)  ? BVOp::Sub
                         : (op == Opcode::mul)  ? BVOp::Mul  : (op == Opcode::sdiv) ? BVOp::SDiv
                         : (op == Opcode::udiv) ? BVOp::UDiv : (op == Opcode::smod) ? BVOp::SRem
                         : (op == Opcode::umod) ? BVOp::URem : (op == Opcode::and_) ? BVOp::And
                         : (op == Opcode::or_)  ? BVOp::Or   : (op == Opcode::xor_) ? BVOp::Xor
                         : (op == Opcode::shl)  ? BVOp::Shl  : (op == Opcode::lshr) ? BVOp::LShr : BVOp::AShr;
                res = ctx.make_binary(bop, a, b, w);
                break;
            }
            case Opcode::neg: case Opcode::not_:
            case Opcode::clz: case Opcode::ctz: case Opcode::popcnt: {
                if (inst->type().is_float()) return make_leaf();
                BVNodeId a = encode(inst->operand(0), depth + 1);
                if (a == kInvalidBV) return make_leaf();
                a = coerce_bv(ctx, a, w, false);
                BVOp uop = (op == Opcode::neg)  ? BVOp::Neg : (op == Opcode::not_) ? BVOp::Not
                         : (op == Opcode::clz)  ? BVOp::Clz : (op == Opcode::ctz)  ? BVOp::Ctz : BVOp::Popcnt;
                res = ctx.make_unary(uop, a, w);
                break;
            }
            case Opcode::eq: case Opcode::ne:
            case Opcode::slt: case Opcode::ult: case Opcode::sle: case Opcode::ule:
            case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge: {
                if (!inst->operand(0) || inst->operand(0)->type().is_float()) return make_leaf();
                BVNodeId a = encode(inst->operand(0), depth + 1);
                BVNodeId b = encode(inst->operand(1), depth + 1);
                if (a == kInvalidBV || b == kInvalidBV) return make_leaf();
                uint32_t cw = std::max(ctx.width(a), ctx.width(b));
                a = coerce_bv(ctx, a, cw, false);
                b = coerce_bv(ctx, b, cw, false);
                BVOp cop = (op == Opcode::eq)  ? BVOp::Eq  : (op == Opcode::ne)  ? BVOp::Ne
                         : (op == Opcode::slt) ? BVOp::Slt : (op == Opcode::ult) ? BVOp::Ult
                         : (op == Opcode::sle) ? BVOp::Sle : (op == Opcode::ule) ? BVOp::Ule
                         : (op == Opcode::sgt) ? BVOp::Sgt : (op == Opcode::ugt) ? BVOp::Ugt
                         : (op == Opcode::sge) ? BVOp::Sge : BVOp::Uge;
                res = ctx.make_zext(ctx.make_binary(cop, a, b, 1), 31);
                break;
            }
            case Opcode::select: {
                if (inst->type().is_float()) return make_leaf();
                BVNodeId c = encode(inst->operand(0), depth + 1);
                BVNodeId t = encode(inst->operand(1), depth + 1);
                BVNodeId f = encode(inst->operand(2), depth + 1);
                if (c == kInvalidBV || t == kInvalidBV || f == kInvalidBV) return make_leaf();
                BVNodeId c_bit = ctx.make_binary(BVOp::Ne, c, ctx.make_const(0, ctx.width(c)), 1);
                res = ctx.make_ite(c_bit, coerce_bv(ctx, t, w, false), coerce_bv(ctx, f, w, false));
                break;
            }
            default:
                return make_leaf();
        }

        cache[v] = res;
        add_range_constraint(v, res);
        return res;
    }

    BVNodeId encode_bool(const Value* cond_val) {
        BVNodeId bv = encode(cond_val, 0);
        if (bv == kInvalidBV) return kInvalidBV;
        if (ctx.width(bv) == 1) return bv;
        return ctx.make_binary(BVOp::Ne, bv, ctx.make_const(0, ctx.width(bv)), 1);
    }
};

bool is_proven_equal(
    BVContext& ctx,
    BVNodeId path_cond,
    BVNodeId a,
    BVNodeId b,
    uint32_t max_conflicts,
    SuperoptStats* stats) {

    if (a == kInvalidBV || b == kInvalidBV || ctx.width(a) != ctx.width(b)) return false;
    if (a == b) return true;
    BVNodeId diff = ctx.make_binary(BVOp::Ne, a, b, 1);
    BVNodeId violated = ctx.make_binary(BVOp::And, path_cond, diff, 1);
    if (stats) stats->solver_queries++;
    SolveOutcome res = verify::solve_nonzero(ctx, violated, max_conflicts);
    return res.status == SolveOutcome::Unsat;
}

void collect_window_values(const Value* root, uint32_t depth, uint32_t max_depth, std::vector<Value*>& out) {
    if (!root || depth > max_depth) return;
    if (depth > 0) {
        int64_t dummy = 0;
        if (!get_const_imm(root, dummy)) {
            if (std::find(out.begin(), out.end(), root) == out.end()) {
                out.push_back(const_cast<Value*>(root));
            }
        }
    }
    if (!root->is_instruction() || depth == max_depth) return;
    const Instruction* inst = root->defining_instruction();
    if (!inst) return;
    for (size_t i = 0; i < inst->operand_count(); ++i) {
        collect_window_values(inst->operand(i), depth + 1, max_depth, out);
    }
}

void remove_dead_pure_instructions(Function& fn) {
    bool progress = true;
    while (progress) {
        progress = false;
        auto uses = compute_use_counts(fn);
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (!cur->has_side_effects() && cur->produces_value() && uses[cur->result()] == 0) {
                    bb->remove_instruction(cur);
                    progress = true;
                }
                cur = next;
            }
        }
    }
}

} // namespace

bool run_mir_superopt(
    Function& fn,
    Module& mod,
    const SuperoptOptions& opts) {

    if (fn.blocks().empty() || !fn.entry_block()) return false;

    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);
    RangeAnalysis ra(fn, dom, loops);

    bool changed = false;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        std::vector<const BasicBlock*> dom_chain;
        for (const BasicBlock* d : fn.blocks()) {
            if (d && d != bb && dom.dominates(d, bb)) {
                dom_chain.push_back(d);
            }
        }

        BVContext ctx;
        SuperoptEncoder enc{ctx, ra, bb, opts.max_window_depth, {}, {}};
        std::vector<BVNodeId> assumptions;

        // Dominating guards & branch edge conditions
        for (const BasicBlock* dbb : dom_chain) {
            for (const Instruction* inst : *dbb) {
                if (inst && inst->opcode() == Opcode::guard && inst->operand(0)) {
                    BVNodeId g = enc.encode_bool(inst->operand(0));
                    if (g != kInvalidBV) assumptions.push_back(g);
                }
            }
        }
        std::vector<const BasicBlock*> all_doms = dom_chain;
        all_doms.push_back(bb);
        for (const BasicBlock* anc : all_doms) {
            if (anc->predecessors().size() == 1) {
                const BasicBlock* pred = anc->predecessors()[0];
                const Instruction* term = pred ? pred->terminator() : nullptr;
                if (term && term->opcode() == Opcode::br_if &&
                    term->true_target().block != term->false_target().block &&
                    term->operand(0)) {
                    BVNodeId c_bit = enc.encode_bool(term->operand(0));
                    if (c_bit != kInvalidBV) {
                        if (term->true_target().block == anc) {
                            assumptions.push_back(c_bit);
                        } else if (term->false_target().block == anc) {
                            assumptions.push_back(ctx.make_unary(BVOp::Not, c_bit, 1));
                        }
                    }
                }
            }
        }

        auto filter_path_cond = [&](BVNodeId target_bv) -> BVNodeId {
            verify::BVCone& sc = ctx.scratch_cone();
            ctx.compute_cone(target_bv, sc.nodes, sc.vars);
            if (sc.vars.empty()) return ctx.make_const(1, 1);
            auto& vseen = sc.var_seen;
            vseen.assign(ctx.vars().size(), 0);
            for (uint32_t vi : sc.vars) {
                if (vi < vseen.size()) vseen[vi] = 1;
            }
            BVNodeId pc = ctx.make_const(1, 1);
            auto try_add_fact = [&](BVNodeId fact) {
                std::vector<BVNodeId> f_nodes;
                std::vector<uint32_t> f_vars;
                ctx.compute_cone(fact, f_nodes, f_vars);
                for (uint32_t fvi : f_vars) {
                    if (fvi < vseen.size() && vseen[fvi]) {
                        pc = ctx.make_binary(BVOp::And, pc, fact, 1);
                        return;
                    }
                }
            };
            for (BVNodeId a : assumptions) try_add_fact(a);
            for (BVNodeId r : enc.range_assumptions) try_add_fact(r);
            return pc;
        };

        Instruction* cur = bb->head();
        uint32_t block_queries = 0;
        constexpr uint32_t kMaxBlockQueries = 24;
        while (cur && block_queries < kMaxBlockQueries) {
            Instruction* next = cur->next();

            auto build_path_cond = [&](BVNodeId target_bv) -> BVNodeId {
                return filter_path_cond(target_bv);
            };

            auto check_equal = [&](BVNodeId path_cond, BVNodeId a, BVNodeId b) -> bool {
                if (block_queries >= kMaxBlockQueries) return false;
                if (a != b) ++block_queries;
                return is_proven_equal(ctx, path_cond, a, b, opts.max_sat_conflicts, opts.stats);
            };

            if (cur->opcode() == Opcode::guard && cur->operand(0)) {
                BVNodeId g = enc.encode_bool(cur->operand(0));
                if (g != kInvalidBV) assumptions.push_back(g);
                cur = next;
                continue;
            }

            if (cur->has_side_effects() || !cur->produces_value()) {
                cur = next;
                continue;
            }

            Opcode op = cur->opcode();
            if (op == Opcode::iconst_i32 || op == Opcode::iconst_i64 ||
                op == Opcode::patchable_const_i32 || op == Opcode::patchable_const_i64) {
                cur = next;
                continue;
            }

            Type res_t = cur->type();
            if (res_t != Type::i32() && res_t != Type::i64()) {
                cur = next;
                continue;
            }
            uint32_t w = (res_t == Type::i32()) ? 32 : 64;

            BVNodeId orig_bv = enc.encode(cur->result(), 0);
            if (orig_bv == kInvalidBV || ctx.width(orig_bv) != w) {
                cur = next;
                continue;
            }
            BVNodeId path_cond = build_path_cond(orig_bv);

            // ---------------------------------------------------------------
            // 1. Solver-Proven Redundant Bitmask Elision
            // ---------------------------------------------------------------
            if (op == Opcode::and_ || op == Opcode::or_) {
                Value* op0 = cur->operand(0);
                Value* op1 = cur->operand(1);
                BVNodeId b0 = enc.encode(op0);
                BVNodeId b1 = enc.encode(op1);
                int64_t c0 = 0, c1 = 0;
                bool is_c0 = get_const_imm(op0, c0);
                bool is_c1 = get_const_imm(op1, c1);
                if (!is_c0 && (orig_bv == b0 || (op1 && (op1->is_instruction() || !assumptions.empty())))) {
                    if (check_equal(path_cond, orig_bv, b0)) {
                        replace_all_uses(fn, cur->result(), op0);
                        if (opts.stats) opts.stats->masks_elided++;
                        changed = true;
                        cur = next;
                        continue;
                    }
                }
                if (!is_c1 && (orig_bv == b1 || (op0 && (op0->is_instruction() || !assumptions.empty())))) {
                    if (check_equal(path_cond, orig_bv, b1)) {
                        replace_all_uses(fn, cur->result(), op1);
                        if (opts.stats) opts.stats->masks_elided++;
                        changed = true;
                        cur = next;
                        continue;
                    }
                }
            }

            // ---------------------------------------------------------------
            // 2. Solver-Proven Redundant Sign/Zero-Extension & Truncation Elision
            // ---------------------------------------------------------------
            if (op == Opcode::sext_i64 || op == Opcode::zext_i64 || op == Opcode::trunc_i32) {
                Value* inner = cur->operand(0);
                if (inner && inner->is_instruction()) {
                    Instruction* inner_inst = inner->defining_instruction();
                    if (inner_inst && inner_inst->operand_count() == 1) {
                        Value* src = inner_inst->operand(0);
                        if (src && src->type() == res_t) {
                            BVNodeId src_bv = enc.encode(src);
                            if (check_equal(path_cond, orig_bv, src_bv)) {
                                replace_all_uses(fn, cur->result(), src);
                                if (opts.stats) opts.stats->extensions_elided++;
                                changed = true;
                                cur = next;
                                continue;
                            }
                        }
                    }
                }
            }

            // ---------------------------------------------------------------
            // 3. Solver-Proven Constant Synthesis
            // ---------------------------------------------------------------
            if (ctx.nodes()[orig_bv].op == BVOp::Const) {
                uint64_t c_imm = ctx.nodes()[orig_bv].imm;
                if (check_equal(path_cond, orig_bv, ctx.make_const(c_imm, w))) {
                    Builder b(mod);
                    b.set_function(&fn);
                    b.position_before(cur);
                    if (Value* cv = build_typed_const(b, res_t, c_imm)) {
                        replace_all_uses(fn, cur->result(), cv);
                        if (opts.stats) opts.stats->constants_synthesized++;
                        changed = true;
                        cur = next;
                        continue;
                    }
                }
            }

            // ---------------------------------------------------------------
            // 4. Solver-Proven Single-Variable Identity / Window Collapse
            // ---------------------------------------------------------------
            bool has_inst_op = false;
            for (size_t oi = 0; oi < cur->operand_count(); ++oi) {
                if (cur->operand(oi) && cur->operand(oi)->is_instruction()) {
                    has_inst_op = true;
                    break;
                }
            }
            if (has_inst_op) {
                std::vector<Value*> window_vals;
                collect_window_values(cur->result(), 0, std::min<uint32_t>(opts.max_window_depth, 3u), window_vals);
                bool collapsed = false;
                for (Value* cand_v : window_vals) {
                    if (!cand_v || cand_v == cur->result() || cand_v->type() != res_t) continue;
                    // Do not compare against a direct operand unless algebraic canonicalization already simplified it
                    const bool is_direct_op =
                        (cand_v == cur->operand(0)) || (cur->operand_count() > 1 && cand_v == cur->operand(1));
                    BVNodeId cand_bv = enc.encode(cand_v);
                    if (is_direct_op && orig_bv != cand_bv && assumptions.empty()) {
                        continue;
                    }
                    if (check_equal(path_cond, orig_bv, cand_bv)) {
                        replace_all_uses(fn, cur->result(), cand_v);
                        if (opts.stats) opts.stats->dags_synthesized++;
                        changed = true;
                        collapsed = true;
                        break;
                    }
                }
                if (collapsed) {
                    cur = next;
                    continue;
                }
            }

            // ---------------------------------------------------------------
            // 5. Solver-Proven Signed/Unsigned Division & Modulo Strength Reduction
            // ---------------------------------------------------------------
            if (op == Opcode::sdiv || op == Opcode::udiv || op == Opcode::smod || op == Opcode::umod) {
                Value* lhs = cur->operand(0);
                int64_t div_imm = 0;
                if (lhs && get_const_imm(cur->operand(1), div_imm) && div_imm > 1) {
                    uint64_t udiv_imm = static_cast<uint64_t>(div_imm);
                    if (std::has_single_bit(udiv_imm)) {
                        uint32_t sh = static_cast<uint32_t>(std::countr_zero(udiv_imm));
                        BVNodeId lhs_bv = enc.encode(lhs);
                        if (op == Opcode::sdiv || op == Opcode::udiv) {
                            BVNodeId cand_bv = ctx.make_binary(BVOp::LShr, lhs_bv, ctx.make_const(sh, w), w);
                            if (check_equal(path_cond, orig_bv, cand_bv)) {
                                Builder b(mod);
                                b.set_function(&fn);
                                b.position_before(cur);
                                Value* sh_v = build_typed_const(b, res_t, sh);
                                Value* repl = b.build_lshr(lhs, sh_v);
                                enc.cache[repl] = cand_bv;
                                replace_all_uses(fn, cur->result(), repl);
                                if (opts.stats) opts.stats->dags_synthesized++;
                                changed = true;
                                cur = next;
                                continue;
                            }
                        } else {
                            uint64_t mask_bits = udiv_imm - 1;
                            BVNodeId cand_bv = ctx.make_binary(BVOp::And, lhs_bv, ctx.make_const(mask_bits, w), w);
                            if (check_equal(path_cond, orig_bv, cand_bv)) {
                                Builder b(mod);
                                b.set_function(&fn);
                                b.position_before(cur);
                                Value* mask_v = build_typed_const(b, res_t, mask_bits);
                                Value* repl = b.build_and(lhs, mask_v);
                                enc.cache[repl] = cand_bv;
                                replace_all_uses(fn, cur->result(), repl);
                                if (opts.stats) opts.stats->dags_synthesized++;
                                changed = true;
                                cur = next;
                                continue;
                            }
                        }
                    }
                }
            }

            // ---------------------------------------------------------------
            // 6. Solver-Proven Multi-Instruction Bitwise & Arithmetic Synthesis
            // ---------------------------------------------------------------
            if (cur->operand_count() == 2) {
                Value* op0 = cur->operand(0);
                Value* op1 = cur->operand(1);
                Instruction* d0 = (op0 && op0->is_instruction()) ? op0->defining_instruction() : nullptr;
                Instruction* d1 = (op1 && op1->is_instruction()) ? op1->defining_instruction() : nullptr;

                // 6a. Associative constant combination: (x op c1) op c2 -> x op (c1 op c2)
                int64_t c1 = 0, c2 = 0;
                if (d0 && d0->opcode() == op && (op == Opcode::and_ || op == Opcode::or_ ||
                    op == Opcode::xor_ || op == Opcode::add) &&
                    get_const_imm(d0->operand(1), c1) && get_const_imm(op1, c2)) {
                    Value* x = d0->operand(0);
                    BVNodeId x_bv = enc.encode(x);
                    uint64_t comb = (op == Opcode::and_) ? (static_cast<uint64_t>(c1) & static_cast<uint64_t>(c2))
                                  : (op == Opcode::or_)  ? (static_cast<uint64_t>(c1) | static_cast<uint64_t>(c2))
                                  : (op == Opcode::xor_) ? (static_cast<uint64_t>(c1) ^ static_cast<uint64_t>(c2))
                                                         : (static_cast<uint64_t>(c1) + static_cast<uint64_t>(c2));
                    BVOp bop = (op == Opcode::and_) ? BVOp::And : (op == Opcode::or_) ? BVOp::Or
                             : (op == Opcode::xor_) ? BVOp::Xor : BVOp::Add;
                    BVNodeId cand_bv = ctx.make_binary(bop, x_bv, ctx.make_const(comb, w), w);
                    if (check_equal(path_cond, orig_bv, cand_bv)) {
                        Builder b(mod);
                        b.set_function(&fn);
                        b.position_before(cur);
                        Value* cv = build_typed_const(b, res_t, comb);
                        Value* repl = (op == Opcode::and_) ? b.build_and(x, cv)
                                    : (op == Opcode::or_)  ? b.build_or(x, cv)
                                    : (op == Opcode::xor_) ? b.build_xor(x, cv) : b.build_add(x, cv);
                        enc.cache[repl] = cand_bv;
                        replace_all_uses(fn, cur->result(), repl);
                        if (opts.stats) opts.stats->dags_synthesized++;
                        changed = true;
                        cur = next;
                        continue;
                    }
                }

                // 6b. Distributive factoring: (x & y) | (x & z) -> x & (y | z),
                //     (x | y) & (x | z) -> x | (y & z), (x * y) + (x * z) -> x * (y + z), etc.
                if (d0 && d1 && d0->opcode() == d1->opcode() && d0->operand_count() == 2 && d1->operand_count() == 2) {
                    Opcode inner_op = d0->opcode();
                    bool valid_pair =
                        (inner_op == Opcode::and_ && (op == Opcode::or_ || op == Opcode::xor_)) ||
                        (inner_op == Opcode::or_  && op == Opcode::and_) ||
                        (inner_op == Opcode::mul  && (op == Opcode::add || op == Opcode::sub));
                    if (valid_pair) {
                        Value* a0 = d0->operand(0);
                        Value* a1 = d0->operand(1);
                        Value* b0 = d1->operand(0);
                        Value* b1 = d1->operand(1);
                        Value* shared = nullptr;
                        Value* rem0 = nullptr;
                        Value* rem1 = nullptr;
                        if (a0 == b0)      { shared = a0; rem0 = a1; rem1 = b1; }
                        else if (a0 == b1) { shared = a0; rem0 = a1; rem1 = b0; }
                        else if (a1 == b0) { shared = a1; rem0 = a0; rem1 = b1; }
                        else if (a1 == b1) { shared = a1; rem0 = a0; rem1 = b0; }

                        if (shared && rem0 && rem1) {
                            BVOp outer_bop = (op == Opcode::or_)  ? BVOp::Or  : (op == Opcode::xor_) ? BVOp::Xor
                                           : (op == Opcode::and_) ? BVOp::And : (op == Opcode::add)  ? BVOp::Add : BVOp::Sub;
                            BVOp inner_bop = (inner_op == Opcode::and_) ? BVOp::And
                                           : (inner_op == Opcode::or_)  ? BVOp::Or : BVOp::Mul;
                            BVNodeId sh_bv = enc.encode(shared);
                            BVNodeId r0_bv = enc.encode(rem0);
                            BVNodeId r1_bv = enc.encode(rem1);
                            BVNodeId comb_bv = ctx.make_binary(outer_bop, r0_bv, r1_bv, w);
                            BVNodeId cand_bv = ctx.make_binary(inner_bop, sh_bv, comb_bv, w);
                            if (check_equal(path_cond, orig_bv, cand_bv)) {
                                Builder b(mod);
                                b.set_function(&fn);
                                b.position_before(cur);
                                Value* comb_v = (op == Opcode::or_)  ? b.build_or(rem0, rem1)
                                              : (op == Opcode::xor_) ? b.build_xor(rem0, rem1)
                                              : (op == Opcode::and_) ? b.build_and(rem0, rem1)
                                              : (op == Opcode::add)  ? b.build_add(rem0, rem1)
                                                                     : b.build_sub(rem0, rem1);
                                Value* repl = (inner_op == Opcode::and_) ? b.build_and(shared, comb_v)
                                            : (inner_op == Opcode::or_)  ? b.build_or(shared, comb_v)
                                                                         : b.build_mul(shared, comb_v);
                                enc.cache[comb_v] = comb_bv;
                                enc.cache[repl] = cand_bv;
                                replace_all_uses(fn, cur->result(), repl);
                                if (opts.stats) opts.stats->dags_synthesized++;
                                changed = true;
                                cur = next;
                                continue;
                            }
                        }
                    }
                }

                // 6c. De Morgan's laws: (~x & ~y) -> ~(x | y) and (~x | ~y) -> ~(x & y)
                if ((op == Opcode::and_ || op == Opcode::or_) &&
                    d0 && d1 && d0->opcode() == Opcode::not_ && d1->opcode() == Opcode::not_) {
                    Value* x = d0->operand(0);
                    Value* y = d1->operand(0);
                    BVOp inner_bop = (op == Opcode::and_) ? BVOp::Or : BVOp::And;
                    BVNodeId inner_bv = ctx.make_binary(inner_bop, enc.encode(x), enc.encode(y), w);
                    BVNodeId cand_bv = ctx.make_unary(BVOp::Not, inner_bv, w);
                    if (check_equal(path_cond, orig_bv, cand_bv)) {
                        Builder b(mod);
                        b.set_function(&fn);
                        b.position_before(cur);
                        Value* inner_v = (op == Opcode::and_) ? b.build_or(x, y) : b.build_and(x, y);
                        Value* repl = b.build_not(inner_v);
                        enc.cache[inner_v] = inner_bv;
                        enc.cache[repl] = cand_bv;
                        replace_all_uses(fn, cur->result(), repl);
                        if (opts.stats) opts.stats->dags_synthesized++;
                        changed = true;
                        cur = next;
                        continue;
                    }
                }
            }

            cur = next;
        }
    }

    if (changed) {
        remove_dead_pure_instructions(fn);
    }
    bool is_gpu_fn = opts.enable_gpu_superopt;
    if (!is_gpu_fn) {
        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (inst && inst->opcode() == Opcode::call && inst->symbol().substr(0, 4) == "ptx_") {
                    is_gpu_fn = true;
                    break;
                }
            }
            if (is_gpu_fn) break;
        }
    }
    if (is_gpu_fn) {
        SuperoptOptions gpu_opts = opts;
        gpu_opts.enable_gpu_superopt = true;
        changed |= run_gpu_mir_superopt(fn, mod, gpu_opts);
    }
    return changed;
}

bool run_mir_superopt(
    Module& mod,
    const SuperoptOptions& opts) {

    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn && !fn->blocks().empty()) {
            changed |= run_mir_superopt(*fn, mod, opts);
        }
    }
    return changed;
}

bool run_x64_lir_superopt(
    codegen::LirBlock& block,
    const SuperoptOptions& opts) {

    using codegen::LirBlock;
    using codegen::LirInst;
    using codegen::LirOpcode;
    using codegen::LirOperand;

    auto clone_block = [](const LirBlock& src) -> LirBlock {
        LirBlock dst(src.id, src.name);
        dst.loop_depth = src.loop_depth;
        dst.x64_label = src.x64_label;
        dst.is_landing_pad = src.is_landing_pad;
        dst.instructions.reserve(src.instructions.size());
        for (const auto& inst : src.instructions) {
            if (inst) {
                dst.instructions.push_back(std::make_unique<LirInst>(*inst));
            }
        }
        return dst;
    };

    auto is_pure_alu_or_mov = [](LirOpcode op) noexcept {
        switch (op) {
            case LirOpcode::Mov:
            case LirOpcode::Mov32:
            case LirOpcode::Lea:
            case LirOpcode::Add:
            case LirOpcode::Add32:
            case LirOpcode::Sub:
            case LirOpcode::Sub32:
            case LirOpcode::Imul:
            case LirOpcode::Imul32:
            case LirOpcode::And:
            case LirOpcode::And32:
            case LirOpcode::Or:
            case LirOpcode::Or32:
            case LirOpcode::Xor:
            case LirOpcode::Xor32:
            case LirOpcode::Shl:
            case LirOpcode::Shl32:
            case LirOpcode::Shr:
            case LirOpcode::Shr32:
            case LirOpcode::Sar:
            case LirOpcode::Sar32:
                return true;
            default:
                return false;
        }
    };

    bool any_changed = false;
    bool progress = true;
    const uint16_t gpr_mask = static_cast<uint16_t>(opts.lir_live_out_gpr_mask & 0xFFFFu);

    while (progress) {
        progress = false;

        // 1. Try 2-instruction window collapse `[i, i + 1]` -> single instruction
        for (size_t i = 0; i + 1 < block.instructions.size(); ++i) {
            const LirInst* i0 = block.instructions[i].get();
            const LirInst* i1 = block.instructions[i + 1].get();
            if (!i0 || !i1 || !is_pure_alu_or_mov(i0->opcode) || !is_pure_alu_or_mov(i1->opcode)) continue;
            if (i0->defs.size() != 1 || i1->defs.size() != 1) continue;
            const LirOperand& d0 = i0->defs[0];
            const LirOperand& d1 = i1->defs[0];
            if (!d0.is_preg() || !d1.is_preg() || d0.preg_val != d1.preg_val) continue;

            std::vector<LirInst> candidates;
            // (a) Overwritten Mov/ALU where i1 is Mov(r, imm/reg) not reading r
            if (i1->opcode == LirOpcode::Mov && i1->uses.size() == 1 &&
                (!i1->uses[0].is_preg() || i1->uses[0].preg_val != d0.preg_val)) {
                candidates.push_back(*i1);
            }
            // (b) Mov(r, imm1); Add/Sub(r, r, imm2) -> Mov(r, imm1 +/- imm2)
            if (i0->opcode == LirOpcode::Mov && i0->uses.size() == 1 && i0->uses[0].is_imm_int() &&
                (i1->opcode == LirOpcode::Add || i1->opcode == LirOpcode::Sub) &&
                i1->uses.size() == 2 && i1->uses[0].is_preg() && i1->uses[0].preg_val == d0.preg_val &&
                i1->uses[1].is_imm_int()) {
                LirInst folded = *i0;
                uint64_t v0 = static_cast<uint64_t>(i0->uses[0].imm_int);
                uint64_t v1 = static_cast<uint64_t>(i1->uses[1].imm_int);
                uint64_t res = (i1->opcode == LirOpcode::Add) ? (v0 + v1) : (v0 - v1);
                folded.uses[0] = LirOperand::imm(static_cast<int64_t>(res), 8);
                candidates.push_back(std::move(folded));
            }

            bool accepted = false;
            for (const LirInst& cand_inst : candidates) {
                LirBlock cand_block = clone_block(block);
                *cand_block.instructions[i] = cand_inst;
                cand_block.instructions.erase(cand_block.instructions.begin() + static_cast<std::ptrdiff_t>(i + 1));
                if (opts.stats) opts.stats->solver_queries++;
                auto eq = verify::verify_x64_lir_block_equivalence(
                    block, cand_block, gpr_mask, 0u, opts.lir_check_flags);
                if (eq.status == verify::EquivStatus::ProvenEquivalent) {
                    block.instructions = std::move(cand_block.instructions);
                    if (opts.stats) opts.stats->lir_windows_optimized++;
                    any_changed = true;
                    progress = true;
                    accepted = true;
                    break;
                }
            }
            if (accepted) break;
        }
        if (progress) continue;

        // 2. Try single-instruction redundant instruction elision (`[i]` -> empty)
        for (size_t i = 0; i < block.instructions.size(); ++i) {
            const LirInst* inst = block.instructions[i].get();
            if (!inst || !is_pure_alu_or_mov(inst->opcode) || inst->defs.size() != 1) continue;
            const LirOperand& dst = inst->defs[0];
            if (!dst.is_preg()) continue;

            bool plausible = false;
            if (inst->opcode == LirOpcode::Mov && inst->uses.size() == 1 &&
                inst->uses[0].is_preg() && inst->uses[0].preg_val == dst.preg_val) {
                plausible = true;
            } else if ((inst->opcode == LirOpcode::Add || inst->opcode == LirOpcode::Sub ||
                        inst->opcode == LirOpcode::Or || inst->opcode == LirOpcode::Xor ||
                        inst->opcode == LirOpcode::Shl || inst->opcode == LirOpcode::Shr ||
                        inst->opcode == LirOpcode::Sar) &&
                       inst->uses.size() == 2 && inst->uses[1].is_imm_int() && inst->uses[1].imm_int == 0) {
                plausible = true;
            } else if (inst->opcode == LirOpcode::Imul && inst->uses.size() == 2 &&
                       inst->uses[1].is_imm_int() && inst->uses[1].imm_int == 1) {
                plausible = true;
            } else if (inst->opcode == LirOpcode::And && inst->uses.size() == 2 &&
                       inst->uses[1].is_imm_int() && inst->uses[1].imm_int == -1) {
                plausible = true;
            }

            if (!plausible) continue;

            LirBlock cand_block = clone_block(block);
            cand_block.instructions.erase(cand_block.instructions.begin() + static_cast<std::ptrdiff_t>(i));
            if (opts.stats) opts.stats->solver_queries++;
            auto eq = verify::verify_x64_lir_block_equivalence(
                block, cand_block, gpr_mask, 0u, opts.lir_check_flags);
            if (eq.status == verify::EquivStatus::ProvenEquivalent) {
                block.instructions = std::move(cand_block.instructions);
                if (opts.stats) opts.stats->lir_windows_optimized++;
                any_changed = true;
                progress = true;
                break;
            }
        }
    }

    return any_changed;
}

} // namespace brass
