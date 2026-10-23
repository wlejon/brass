#include <brass/mir/smt_guard_elim.hpp>
#include "../verify/bv_solver.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/proof_cert.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/uses.hpp>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
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

struct SliceEncoder {
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
                BVNodeId bit = ctx.make_binary(cop, a, b, 1);
                res = ctx.make_zext(bit, 31);
                break;
            }
            case Opcode::sadd_overflow: case Opcode::ssub_overflow: case Opcode::smul_overflow:
            case Opcode::uadd_overflow: case Opcode::usub_overflow: case Opcode::umul_overflow: {
                BVNodeId a = encode(inst->operand(0), depth + 1);
                BVNodeId b = encode(inst->operand(1), depth + 1);
                if (a == kInvalidBV || b == kInvalidBV) return make_leaf();
                uint32_t cw = std::max(ctx.width(a), ctx.width(b));
                a = coerce_bv(ctx, a, cw, false);
                b = coerce_bv(ctx, b, cw, false);
                BVOp ovf = (op == Opcode::sadd_overflow) ? BVOp::SAddOvf
                         : (op == Opcode::ssub_overflow) ? BVOp::SSubOvf
                         : (op == Opcode::smul_overflow) ? BVOp::SMulOvf
                         : (op == Opcode::uadd_overflow) ? BVOp::UAddOvf
                         : (op == Opcode::usub_overflow) ? BVOp::USubOvf : BVOp::UMulOvf;
                res = ctx.make_zext(ctx.make_binary(ovf, a, b, 1), 31);
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

// Prove whether `implication_bit == 1` holds under `path_cond` (i.e. `path_cond & ~implication_bit` is UNSAT).
bool is_proven_true(BVContext& ctx, BVNodeId path_cond, BVNodeId implication_bit, uint32_t max_conflicts,
                    SmtGuardElimStats* stats) {
    BVNodeId violated = ctx.make_binary(BVOp::And, path_cond, ctx.make_unary(BVOp::Not, implication_bit, 1), 1);
    if (stats) stats->queries_solved++;
    SolveOutcome res = verify::solve_nonzero(ctx, violated, max_conflicts);
    return res.status == SolveOutcome::Unsat;
}

// Nonlinear 2D strided indexing theorem:
// Proves `0 <=s (i + stride * j)` and `(i + stride * j) <s len` (or `<u len`) when:
//   1) `0 <=s i && i <s stride`
//   2) `0 <=s j && j <s height`
//   3) `stride * height <=s len` (and `stride * height` does not signed-overflow, either via an explicit
//      `smul_overflow(stride, height) == 0` / `umul_overflow(stride, height) == 0` guard or bounded factors).
bool try_prove_2d_strided_bound(
    SliceEncoder& enc,
    BVNodeId path_cond,
    const Instruction* cmp_inst,
    const std::vector<const Value*>& dom_guard_conds,
    uint32_t max_conflicts,
    SmtGuardElimStats* stats) {

    if (!cmp_inst || cmp_inst->operand_count() != 2) return false;
    Opcode cop = cmp_inst->opcode();
    const Value* lhs = cmp_inst->operand(0);
    const Value* rhs = cmp_inst->operand(1);
    if (!lhs || !rhs || !lhs->is_instruction()) return false;

    // Strip sext_i64 / zext_i64 on index expression if needed
    const Value* idx_val = lhs;
    while (idx_val && idx_val->is_instruction()) {
        const Instruction* d = idx_val->defining_instruction();
        if (d && (d->opcode() == Opcode::sext_i64 || d->opcode() == Opcode::zext_i64)) {
            idx_val = d->operand(0);
        } else {
            break;
        }
    }
    if (!idx_val || !idx_val->is_instruction()) return false;
    const Instruction* add_inst = idx_val->defining_instruction();
    if (!add_inst || add_inst->opcode() != Opcode::add) return false;

    // Match add(i, mul(stride, j)) or add(mul(stride, j), i)
    const Value* op0 = add_inst->operand(0);
    const Value* op1 = add_inst->operand(1);
    const Instruction* mul_inst = nullptr;
    const Value* i_val = nullptr;
    if (op1 && op1->is_instruction() && op1->defining_instruction()->opcode() == Opcode::mul) {
        mul_inst = op1->defining_instruction();
        i_val = op0;
    } else if (op0 && op0->is_instruction() && op0->defining_instruction()->opcode() == Opcode::mul) {
        mul_inst = op0->defining_instruction();
        i_val = op1;
    }
    if (!mul_inst || !i_val) return false;

    BVContext& ctx = enc.ctx;
    BVNodeId i_bv = enc.encode(i_val);
    BVNodeId m0_bv = enc.encode(mul_inst->operand(0));
    BVNodeId m1_bv = enc.encode(mul_inst->operand(1));
    if (i_bv == kInvalidBV || m0_bv == kInvalidBV || m1_bv == kInvalidBV) return false;
    uint32_t w = ctx.width(i_bv);
    if (ctx.width(m0_bv) != w || ctx.width(m1_bv) != w) return false;

    // Identify which of (m0, m1) is `stride` (by checking if `0 <=s i <s m0` or `0 <=s i <s m1` is proven)
    BVNodeId zero = ctx.make_const(0, w);
    if (!is_proven_true(ctx, path_cond, ctx.make_binary(BVOp::Sge, i_bv, zero, 1), max_conflicts, stats)) {
        return false;
    }

    const Value* stride_val = nullptr;
    const Value* j_val = nullptr;
    BVNodeId stride_bv = kInvalidBV;
    BVNodeId j_bv = kInvalidBV;

    if (is_proven_true(ctx, path_cond, ctx.make_binary(BVOp::Slt, i_bv, m0_bv, 1), max_conflicts, stats)) {
        stride_val = mul_inst->operand(0);
        j_val = mul_inst->operand(1);
        stride_bv = m0_bv;
        j_bv = m1_bv;
    } else if (is_proven_true(ctx, path_cond, ctx.make_binary(BVOp::Slt, i_bv, m1_bv, 1), max_conflicts, stats)) {
        stride_val = mul_inst->operand(1);
        j_val = mul_inst->operand(0);
        stride_bv = m1_bv;
        j_bv = m0_bv;
    } else {
        return false;
    }

    if (!is_proven_true(ctx, path_cond, ctx.make_binary(BVOp::Sge, j_bv, zero, 1), max_conflicts, stats)) {
        return false;
    }

    // Find a height bound `H` from dominating guards/facts where `mul(stride, H)` is compared against `len`
    // or where `j <s H` holds.
    std::vector<const Value*> candidate_heights;
    for (const Value* gc : dom_guard_conds) {
        if (!gc || !gc->is_instruction()) continue;
        const Instruction* gd = gc->defining_instruction();
        if (!gd) continue;
        if ((gd->opcode() == Opcode::slt || gd->opcode() == Opcode::ult) && gd->operand(0) == j_val) {
            candidate_heights.push_back(gd->operand(1));
        }
        if ((gd->opcode() == Opcode::sgt || gd->opcode() == Opcode::ugt) && gd->operand(1) == j_val) {
            candidate_heights.push_back(gd->operand(0));
        }
    }

    // Check if there is a non-overflow fact or bounded range on `stride * H`
    for (const Value* h_val : candidate_heights) {
        BVNodeId h_bv = enc.encode(h_val);
        if (h_bv == kInvalidBV || ctx.width(h_bv) != w) continue;
        if (!is_proven_true(ctx, path_cond, ctx.make_binary(BVOp::Slt, j_bv, h_bv, 1), max_conflicts, stats)) {
            continue;
        }
        BVNodeId prod_sh = ctx.make_binary(BVOp::Mul, stride_bv, h_bv, w);
        BVNodeId ovf_sh = ctx.make_binary(BVOp::SMulOvf, stride_bv, h_bv, 1);
        bool no_overflow = is_proven_true(ctx, path_cond, ctx.make_unary(BVOp::Not, ovf_sh, 1), max_conflicts, stats);
        if (!no_overflow) {
            // Also check if an explicit mul(stride, h) <=s rhs guard exists in dom_guard_conds with rhs > 0
            for (const Value* gc : dom_guard_conds) {
                if (!gc || !gc->is_instruction()) continue;
                const Instruction* gd = gc->defining_instruction();
                if (!gd || (gd->opcode() != Opcode::sle && gd->opcode() != Opcode::ule &&
                            gd->opcode() != Opcode::slt && gd->opcode() != Opcode::ult)) continue;
                const Value* gl = gd->operand(0);
                if (gl && gl->is_instruction() && gl->defining_instruction()->opcode() == Opcode::mul) {
                    const Instruction* gm = gl->defining_instruction();
                    if ((gm->operand(0) == stride_val && gm->operand(1) == h_val) ||
                        (gm->operand(0) == h_val && gm->operand(1) == stride_val)) {
                        no_overflow = true;
                        break;
                    }
                }
            }
        }
        if (!no_overflow) continue;

        // Under 0 <= i < stride, 0 <= j < H, and no overflow on stride * H:
        //   0 <= i + stride * j < stride * H
        if (cop == Opcode::sge || cop == Opcode::uge) {
            BVNodeId rhs_bv = enc.encode(rhs);
            if (rhs_bv != kInvalidBV && is_proven_true(ctx, path_cond,
                    ctx.make_binary(BVOp::Sle, coerce_bv(ctx, rhs_bv, w, true), zero, 1), max_conflicts, stats)) {
                return true;
            }
        }
        if (cop == Opcode::slt || cop == Opcode::ult) {
            BVNodeId rhs_bv = enc.encode(rhs);
            if (rhs_bv != kInvalidBV && is_proven_true(ctx, path_cond,
                    ctx.make_binary(BVOp::Sle, prod_sh, coerce_bv(ctx, rhs_bv, w, true), 1), max_conflicts, stats)) {
                return true;
            }
        }
    }
    return false;
}

void collect_loop_iv_assumptions(
    SliceEncoder& enc,
    const LoopAnalysis& loops,
    const BasicBlock* bb,
    std::vector<BVNodeId>& assumptions) {

    for (LoopInfo* loop : loops.post_order_loops()) {
        if (!loop || !loop->contains(bb)) continue;
        const BasicBlock* hdr = loop->header();
        if (!hdr) continue;

        // Identify single preheader and single latch target into hdr
        const BasicBlock* preheader = nullptr;
        const BasicBlock* latch = nullptr;
        for (const BasicBlock* pred : hdr->predecessors()) {
            if (!pred) continue;
            if (loop->contains(pred)) {
                if (latch) { latch = nullptr; break; }
                latch = pred;
            } else {
                if (preheader) { preheader = nullptr; break; }
                preheader = pred;
            }
        }
        if (!preheader || !latch) continue;

        auto find_edge_args = [](const BasicBlock* from, const BasicBlock* to) -> const std::vector<Value*>* {
            const Instruction* term = from->terminator();
            if (!term) return nullptr;
            if (term->opcode() == Opcode::br && term->branch_target().block == to) {
                return &term->branch_target().args;
            }
            if (term->opcode() == Opcode::br_if) {
                if (term->true_target().block == to && term->false_target().block != to) {
                    return &term->true_target().args;
                }
                if (term->false_target().block == to && term->true_target().block != to) {
                    return &term->false_target().args;
                }
            }
            return nullptr;
        };

        const auto* pre_args = find_edge_args(preheader, hdr);
        const auto* latch_args = find_edge_args(latch, hdr);
        if (!pre_args || !latch_args) continue;
        if (pre_args->size() != hdr->param_count() || latch_args->size() != hdr->param_count()) continue;

        for (size_t pi = 0; pi < hdr->param_count(); ++pi) {
            const Value* iv = hdr->param(pi);
            const Value* init_v = (*pre_args)[pi];
            const Value* next_v = (*latch_args)[pi];
            if (!iv || !init_v || !next_v || !next_v->is_instruction()) continue;
            const Instruction* step_inst = next_v->defining_instruction();
            if (!step_inst || step_inst->opcode() != Opcode::add) continue;

            const Value* step_op = (step_inst->operand(0) == iv) ? step_inst->operand(1)
                                 : (step_inst->operand(1) == iv) ? step_inst->operand(0) : nullptr;
            if (!step_op || !step_op->is_instruction()) continue;
            const Instruction* step_const = step_op->defining_instruction();
            if (!step_const) continue;
            int64_t step_imm = 0;
            if (step_const->opcode() == Opcode::iconst_i32) step_imm = step_const->imm_i32();
            else if (step_const->opcode() == Opcode::iconst_i64) step_imm = step_const->imm_i64();
            if (step_imm <= 0) continue;

            // Check if the loop has an upper bound test on `iv` or `next_v`
            BVNodeId iv_bv = enc.encode(iv);
            BVNodeId init_bv = enc.encode(init_v);
            if (iv_bv == kInvalidBV || init_bv == kInvalidBV || enc.ctx.width(iv_bv) != enc.ctx.width(init_bv)) continue;

            // Check if the latch or header branches on iv < bound or next_v < bound
            const Instruction* l_term = latch->terminator();
            const Instruction* h_term = hdr->terminator();
            bool has_upper_bound = false;
            auto check_term_bound = [&](const Instruction* term, const BasicBlock* in_loop_succ) {
                if (!term || term->opcode() != Opcode::br_if) return;
                const Value* c = term->operand(0);
                if (!c || !c->is_instruction()) return;
                const Instruction* cmp = c->defining_instruction();
                if (!cmp) return;
                bool on_true = (term->true_target().block == in_loop_succ);
                Opcode cop = cmp->opcode();
                if (on_true && (cop == Opcode::slt || cop == Opcode::sle || cop == Opcode::ult || cop == Opcode::ule)) {
                    if (cmp->operand(0) == iv || cmp->operand(0) == next_v) {
                        has_upper_bound = true;
                        if (cmp->operand(0) == next_v && term == l_term) {
                            // Bottom-tested loop: next_v < bound on latch; if init_v < bound is also guarded, iv < bound!
                            BVNodeId b_bv = enc.encode(cmp->operand(1));
                            if (b_bv != kInvalidBV && enc.ctx.width(b_bv) == enc.ctx.width(iv_bv)) {
                                BVOp bop = (cop == Opcode::slt) ? BVOp::Slt : (cop == Opcode::sle) ? BVOp::Sle
                                         : (cop == Opcode::ult) ? BVOp::Ult : BVOp::Ule;
                                BVNodeId init_ok = enc.ctx.make_binary(bop, init_bv, b_bv, 1);
                                BVNodeId iv_ok = enc.ctx.make_binary(bop, iv_bv, b_bv, 1);
                                assumptions.push_back(enc.ctx.make_ite(init_ok, iv_ok, enc.ctx.make_const(1, 1)));
                            }
                        }
                    }
                }
            };
            check_term_bound(h_term, (h_term && h_term->opcode() == Opcode::br_if &&
                                      loop->contains(h_term->true_target().block))
                                     ? h_term->true_target().block
                                     : (h_term && h_term->opcode() == Opcode::br_if ? h_term->false_target().block : nullptr));
            check_term_bound(l_term, hdr);

            if (has_upper_bound) {
                assumptions.push_back(enc.ctx.make_binary(BVOp::Sge, iv_bv, init_bv, 1));
            }
        }
    }
}

} // namespace

bool run_smt_guard_elimination(
    Function& fn,
    Module& mod,
    const SmtGuardElimOptions& opts) {

    if (fn.blocks().empty() || !fn.entry_block()) return false;

    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);
    RangeAnalysis ra(fn, dom, loops);

    bool changed = false;
    bool branches_folded = false;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        // Collect all strict dominators of `bb`
        std::vector<const BasicBlock*> dom_chain;
        for (const BasicBlock* d : fn.blocks()) {
            if (d && d != bb && dom.dominates(d, bb)) {
                dom_chain.push_back(d);
            }
        }

        BVContext ctx;
        SliceEncoder enc{ctx, ra, bb, opts.max_slice_depth, {}, {}};

        std::vector<BVNodeId> assumptions;
        std::vector<const Value*> dom_guard_conds;

        // 1. Dominating guards & dominating branch edge conditions
        for (const BasicBlock* dbb : dom_chain) {
            for (const Instruction* inst : *dbb) {
                if (inst && inst->opcode() == Opcode::guard && inst->operand(0)) {
                    dom_guard_conds.push_back(inst->operand(0));
                    BVNodeId g_bit = enc.encode_bool(inst->operand(0));
                    if (g_bit != kInvalidBV) assumptions.push_back(g_bit);
                }
            }
        }

        // Branch edge conditions on any dominator block `anc` (including `bb` itself)
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
                            dom_guard_conds.push_back(term->operand(0));
                        } else if (term->false_target().block == anc) {
                            assumptions.push_back(ctx.make_unary(BVOp::Not, c_bit, 1));
                        }
                    }
                }
            }
        }

        // 2. Inductive loop IV bounds for enclosing loops
        collect_loop_iv_assumptions(enc, loops, bb, assumptions);

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
            std::vector<BVNodeId> f_nodes;
            std::vector<uint32_t> f_vars;
            // Fixed-point expansion (2 rounds) to capture transitive relational chains
            for (int round = 0; round < 2; ++round) {
                auto try_add_fact = [&](BVNodeId fact) {
                    ctx.compute_cone(fact, f_nodes, f_vars);
                    bool relevant = false;
                    for (uint32_t fvi : f_vars) {
                        if (fvi < vseen.size() && vseen[fvi]) {
                            relevant = true;
                            break;
                        }
                    }
                    if (relevant) {
                        for (uint32_t fvi : f_vars) {
                            if (fvi < vseen.size()) vseen[fvi] = 1;
                        }
                        if (round == 1) {
                            pc = ctx.make_binary(BVOp::And, pc, fact, 1);
                        }
                    }
                };
                for (BVNodeId a : assumptions) try_add_fact(a);
                for (BVNodeId r : enc.range_assumptions) try_add_fact(r);
            }
            return pc;
        };

        // Walk instructions in `bb`
        Instruction* cur = bb->head();
        uint32_t block_queries = 0;
        constexpr uint32_t kMaxBlockQueries = 24;
        while (cur && block_queries < kMaxBlockQueries) {
            Instruction* next = cur->next();

            auto build_path_cond = [&](BVNodeId target_bv) -> BVNodeId {
                return filter_path_cond(target_bv);
            };

            if (cur->opcode() == Opcode::guard && cur->operand(0)) {
                ++block_queries;
                const Value* cond_val = cur->operand(0);
                BVNodeId cond_bit = enc.encode_bool(cond_val);
                if (cond_bit != kInvalidBV) {
                    BVNodeId path_cond = build_path_cond(cond_bit);
                    bool proven = is_proven_true(ctx, path_cond, cond_bit, opts.max_sat_conflicts, opts.stats);
                    if (!proven && cond_val->is_instruction()) {
                        proven = try_prove_2d_strided_bound(
                            enc, path_cond, cond_val->defining_instruction(),
                            dom_guard_conds, opts.max_sat_conflicts, opts.stats);
                    }
                    if (proven) {
                        GuardElimCert cert;
                        cert.eliminated_resume_id = cur->resume_id();
                        cert.block_id = bb->id();
                        cert.reason = GuardElimCert::Reason::SmtProven;
                        cert.witness_inst_or_val_id = 0;
                        fn.add_guard_elim_cert(cert);

                        bb->remove_instruction(cur);
                        if (opts.stats) opts.stats->guards_proven_eliminated++;
                        changed = true;
                        cur = next;
                        continue;
                    }
                    assumptions.push_back(cond_bit);
                    dom_guard_conds.push_back(cond_val);
                }
            } else if (opts.fold_implied_comparisons && cur->produces_value() &&
                       is_comparison(cur->opcode()) && cur->operand_count() == 2 &&
                       cur->operand(0) && !cur->operand(0)->type().is_float() &&
                       !(next && (next->opcode() == Opcode::guard || next->opcode() == Opcode::br_if) &&
                         next->operand_count() > 0 && next->operand(0) == cur->result())) {
                ++block_queries;
                BVNodeId cmp_bit = enc.encode_bool(cur->result());
                if (cmp_bit != kInvalidBV) {
                    BVNodeId path_cond = build_path_cond(cmp_bit);
                    bool proven_true = is_proven_true(ctx, path_cond, cmp_bit, opts.max_sat_conflicts, opts.stats);
                    if (!proven_true) {
                        proven_true = try_prove_2d_strided_bound(
                            enc, path_cond, cur, dom_guard_conds, opts.max_sat_conflicts, opts.stats);
                    }
                    if (proven_true) {
                        Builder b(mod);
                        b.set_function(&fn);
                        b.position_before(cur);
                        Value* one = b.build_iconst_i32(1);
                        replace_all_uses(fn, cur->result(), one);
                        if (opts.stats) opts.stats->comparisons_proven_true++;
                        changed = true;
                    } else if (is_proven_true(ctx, path_cond, ctx.make_unary(BVOp::Not, cmp_bit, 1),
                                              opts.max_sat_conflicts, opts.stats)) {
                        Builder b(mod);
                        b.set_function(&fn);
                        b.position_before(cur);
                        Value* zero = b.build_iconst_i32(0);
                        replace_all_uses(fn, cur->result(), zero);
                        if (opts.stats) opts.stats->comparisons_proven_false++;
                        changed = true;
                    }
                }
            } else if (opts.fold_branches && cur->opcode() == Opcode::br_if && cur->operand(0)) {
                ++block_queries;
                BVNodeId br_bit = enc.encode_bool(cur->operand(0));
                if (br_bit != kInvalidBV) {
                    BVNodeId path_cond = build_path_cond(br_bit);
                    if (is_proven_true(ctx, path_cond, br_bit, opts.max_sat_conflicts, opts.stats)) {
                        BranchTarget kept = cur->true_target();
                        cur->set_opcode(Opcode::br);
                        cur->operands().clear();
                        cur->true_target() = BranchTarget();
                        cur->false_target() = BranchTarget();
                        cur->set_branch_target(std::move(kept));
                        if (opts.stats) opts.stats->branches_folded++;
                        branches_folded = true;
                        changed = true;
                    } else if (is_proven_true(ctx, path_cond, ctx.make_unary(BVOp::Not, br_bit, 1),
                                              opts.max_sat_conflicts, opts.stats)) {
                        BranchTarget kept = cur->false_target();
                        cur->set_opcode(Opcode::br);
                        cur->operands().clear();
                        cur->true_target() = BranchTarget();
                        cur->false_target() = BranchTarget();
                        cur->set_branch_target(std::move(kept));
                        if (opts.stats) opts.stats->branches_folded++;
                        branches_folded = true;
                        changed = true;
                    }
                }
            }

            cur = next;
        }
    }

    if (branches_folded) {
        fn.rebuild_cfg_predecessors();
        cfg_simplify_function(fn);
        fn.rebuild_cfg_predecessors();
    }

    // Remove dead pure comparison instructions whose uses were folded
    if (changed) {
        bool dce_progress = true;
        while (dce_progress) {
            dce_progress = false;
            auto uses = compute_use_counts(fn);
            for (BasicBlock* bb : fn.blocks()) {
                if (!bb) continue;
                Instruction* cur = bb->head();
                while (cur) {
                    Instruction* next = cur->next();
                    if (!cur->has_side_effects() && cur->produces_value() && uses[cur->result()] == 0) {
                        bb->remove_instruction(cur);
                        dce_progress = true;
                    }
                    cur = next;
                }
            }
        }
    }

    return changed;
}

bool run_smt_guard_elimination(
    Module& mod,
    const SmtGuardElimOptions& opts) {

    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn && !fn->blocks().empty()) {
            changed |= run_smt_guard_elimination(*fn, mod, opts);
        }
    }
    return changed;
}

} // namespace brass
