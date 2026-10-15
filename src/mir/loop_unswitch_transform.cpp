#include "loop_unswitch_internal.hpp"
#include "ir_clone.hpp"
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/range_analysis.hpp>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

Instruction* make_branch(Function& fn, Opcode op) {
    return fn.parent() ? fn.parent()->arena().make<Instruction>(op, Type::void_type())
                       : new Instruction(op, Type::void_type());
}

bool get_const_int(const Value* v, int64_t& out) {
    if (!v || !v->is_instruction()) return false;
    const Instruction* d = v->defining_instruction();
    if (!d) return false;
    if (d->opcode() == Opcode::iconst_i32 || d->opcode() == Opcode::patchable_const_i32) {
        out = static_cast<int64_t>(d->imm_i32());
        return true;
    }
    if (d->opcode() == Opcode::iconst_i64 || d->opcode() == Opcode::patchable_const_i64) {
        out = d->imm_i64();
        return true;
    }
    return false;
}

Opcode invert_cmp(Opcode op) noexcept {
    switch (op) {
        case Opcode::slt: return Opcode::sge;
        case Opcode::sle: return Opcode::sgt;
        case Opcode::sgt: return Opcode::sle;
        case Opcode::sge: return Opcode::slt;
        case Opcode::ult: return Opcode::uge;
        case Opcode::ule: return Opcode::ugt;
        case Opcode::ugt: return Opcode::ule;
        case Opcode::uge: return Opcode::ult;
        case Opcode::eq:  return Opcode::ne;
        case Opcode::ne:  return Opcode::eq;
        default: return op;
    }
}

Opcode commute_cmp(Opcode op) noexcept {
    switch (op) {
        case Opcode::slt: return Opcode::sgt;
        case Opcode::sle: return Opcode::sge;
        case Opcode::sgt: return Opcode::slt;
        case Opcode::sge: return Opcode::sle;
        case Opcode::ult: return Opcode::ugt;
        case Opcode::ule: return Opcode::uge;
        case Opcode::ugt: return Opcode::ult;
        case Opcode::uge: return Opcode::ule;
        default: return op;
    }
}

bool is_pure_unswitch_inst(const Instruction* inst) {
    if (!inst || !inst->produces_value() || is_derived_gcref(inst->result())) return false;
    switch (inst->opcode()) {
        case Opcode::iconst_i32: case Opcode::iconst_i64: case Opcode::fconst_f64:
        case Opcode::patchable_const_i32: case Opcode::patchable_const_i64:
        case Opcode::sext_i64: case Opcode::zext_i64: case Opcode::trunc_i32:
        case Opcode::fptosi_i32: case Opcode::fptosi_i64:
        case Opcode::sitofp_f64_i32: case Opcode::sitofp_f64_i64:
        case Opcode::ceil_f64: case Opcode::floor_f64:
        case Opcode::bitcast_i64_f64: case Opcode::bitcast_f64_i64:
        case Opcode::bitcast_i32_f32: case Opcode::bitcast_f32_i32:
        case Opcode::add: case Opcode::sub: case Opcode::mul: case Opcode::neg:
        case Opcode::and_: case Opcode::or_: case Opcode::xor_:
        case Opcode::shl: case Opcode::lshr: case Opcode::ashr: case Opcode::not_:
        case Opcode::eq: case Opcode::ne:
        case Opcode::slt: case Opcode::ult: case Opcode::sle: case Opcode::ule:
        case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge:
        case Opcode::select:
            return true;
        default:
            return false;
    }
}

struct AffineIvExpr {
    int64_t scale = 1;
    Value* inv_offset = nullptr;
    int64_t const_offset = 0;
    int32_t outer_shift = 0;
};

} // namespace

// The false copy of the loop must be a closed region: every block that can
// see a value defined in the loop has to be copied with it, or the copy's
// values would not dominate their uses. The blocks dominated by the header
// are exactly that set — every predecessor of such a block (other than the
// header's own entry edge) is itself dominated by the header, and a block
// outside it can only receive loop values through block parameters.
bool transform_unswitch_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    BasicBlock* cond_bb,
    Instruction* br_if_inst,
    const LoopUnswitchOptions& opts
) {
    (void)dom;
    if (!cond_bb || !br_if_inst || br_if_inst->opcode() != Opcode::br_if) return false;
    Value* cond = br_if_inst->operand(0);
    BasicBlock* header = loop.header();
    if (!cond || !header) return false;

    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader || !preheader->terminator()) return false;
    Instruction* ph_term = preheader->terminator();
    if (ph_term->opcode() != Opcode::br || ph_term->branch_target().block != header) return false;

    // ensure_preheader may have changed the CFG since `dom` was built.
    fn.rebuild_cfg_predecessors();
    DominatorTree cur_dom(fn);

    const BasicBlock* cond_def = cond->is_instruction() ? cond->defining_instruction()->parent()
                               : (cond->is_block_param() ? cond->defining_block() : nullptr);
    if (cond_def && !cur_dom.dominates(cond_def, preheader)) return false;

    std::vector<BasicBlock*> region;
    size_t region_instructions = 0;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || !cur_dom.dominates(header, bb)) continue;
        region.push_back(bb);
        for (Instruction* inst : *bb) {
            if (inst) ++region_instructions;
        }
    }
    // The loop itself was already bounded by max_loop_instructions; allow the
    // code after it to cost at most as much again.
    if (region_instructions > 2 * opts.max_loop_instructions) return false;

    const std::vector<Value*> init_args = ph_term->branch_target().args;

    ir::BlockMap blocks;
    ir::ValueMap values;
    for (BasicBlock* bb : region) {
        BasicBlock* copy = ir::new_block(fn, std::string(bb->name()) + "_unsw_f");
        blocks[bb] = copy;
        for (size_t i = 0; i < bb->param_count(); ++i) {
            values[bb->param(i)] = ir::new_block_param(fn, copy, bb->param(i)->type());
        }
    }

    // Shells first so every result exists before any operand is remapped.
    std::vector<std::pair<Instruction*, Instruction*>> pairs;
    for (BasicBlock* bb : region) {
        BasicBlock* copy = blocks[bb];
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            if (inst == br_if_inst) {
                Instruction* br = make_branch(fn, Opcode::br);
                copy->append_instruction(br);
                pairs.emplace_back(inst, br);
                continue;
            }
            Instruction* shell = ir::clone_shell(fn, *inst, values);
            copy->append_instruction(shell);
            pairs.emplace_back(inst, shell);
        }
    }
    for (auto& [orig, copy] : pairs) {
        if (orig == br_if_inst) {
            // The false copy always takes the false edge.
            const BranchTarget& f = orig->false_target();
            BranchTarget t(f.block, f.args);
            if (auto it = blocks.find(t.block); it != blocks.end()) t.block = it->second;
            for (Value*& a : t.args) {
                if (auto it = values.find(a); it != values.end()) a = it->second;
            }
            copy->set_branch_target(std::move(t));
            continue;
        }
        ir::clone_uses(*orig, *copy, values, blocks);
    }

    // The original copy always takes the true edge.
    Instruction* br_true = make_branch(fn, Opcode::br);
    br_true->set_branch_target(br_if_inst->true_target());
    cond_bb->remove_instruction(br_if_inst);
    cond_bb->append_instruction(br_true);

    Instruction* ph_br_if = make_branch(fn, Opcode::br_if);
    ph_br_if->add_operand(cond);
    ph_br_if->set_true_target(BranchTarget(header, init_args));
    ph_br_if->set_false_target(BranchTarget(blocks[header], init_args));
    preheader->remove_instruction(ph_term);
    preheader->append_instruction(ph_br_if);

    fn.rebuild_cfg_predecessors();
    CfgSimplifyOptions cfg_opts;
    cfg_opts.enable_param_elimination = false;
    cfg_simplify_function(fn, cfg_opts);
    return true;
}

bool unswitch_fast_slow_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopUnswitchOptions& opts
) {
    (void)dom;
    if (!fn.parent() || loop.depth() > opts.max_unswitch_depth) return false;
    BasicBlock* header = loop.header();
    if (!header || loop.latches().size() != 1) return false;
    if (header->name().find("_unsw_") != std::string_view::npos) return false;
    BasicBlock* latch = loop.latches()[0];
    if (!latch) return false;

    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader || !preheader->terminator()) return false;
    Instruction* ph_term = preheader->terminator();
    if (ph_term->opcode() != Opcode::br || ph_term->branch_target().block != header) return false;
    if (ph_term->branch_target().args.size() != header->param_count()) return false;

    Instruction* latch_term = latch->terminator();
    Instruction* hdr_term = header->terminator();
    if (!latch_term || latch_term->opcode() != Opcode::br || latch_term->branch_target().block != header) return false;
    if (latch_term->branch_target().args.size() != header->param_count()) return false;
    if (!hdr_term || hdr_term->opcode() != Opcode::br_if) return false;

    bool body_is_true = loop.contains(hdr_term->true_target().block);
    bool body_is_false = loop.contains(hdr_term->false_target().block);
    if (body_is_true == body_is_false) return false;

    fn.rebuild_cfg_predecessors();
    DominatorTree cur_dom(fn);
    LoopAnalysis cur_loops(fn, cur_dom);

    // Identify slow-path fallback blocks (do not dominate latch, contain calls/throws)
    std::unordered_set<BasicBlock*> slow_blocks;
    for (BasicBlock* bb : loop.blocks()) {
        if (!bb || bb == header || bb == latch || cur_dom.dominates(bb, latch)) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            if (is_call(inst->opcode()) || inst->opcode() == Opcode::throw_ || inst->opcode() == Opcode::unreachable) {
                slow_blocks.insert(bb);
                break;
            }
        }
    }
    if (slow_blocks.empty()) return false;

    bool added_slow = true;
    while (added_slow) {
        added_slow = false;
        for (BasicBlock* bb : loop.blocks()) {
            if (!bb || bb == header || bb == latch || cur_dom.dominates(bb, latch) || slow_blocks.count(bb)) continue;
            const auto& preds = bb->predecessors();
            if (!preds.empty() && std::all_of(preds.begin(), preds.end(), [&](BasicBlock* p) { return slow_blocks.count(p) > 0; })) {
                slow_blocks.insert(bb);
                added_slow = true;
            }
        }
    }

    size_t fast_inst_count = 0;
    for (BasicBlock* bb : loop.blocks()) {
        if (!bb || slow_blocks.count(bb)) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            ++fast_inst_count;
            if (is_call(inst->opcode()) || is_coro_op(inst->opcode()) ||
                inst->opcode() == Opcode::throw_ || inst->opcode() == Opcode::invoke) {
                return false;
            }
        }
    }
    const size_t max_fast_insts = std::max<size_t>(opts.max_loop_instructions * 4, 400);
    if (fast_inst_count > max_fast_insts) return false;

    // Identify canonical primary induction variable
    Value* hdr_cond = hdr_term->operand(0);
    if (!hdr_cond || !hdr_cond->is_instruction()) return false;
    Instruction* hdr_cmp = hdr_cond->defining_instruction();
    if (!hdr_cmp || !is_comparison(hdr_cmp->opcode())) return false;
    Opcode hdr_op = body_is_true ? hdr_cmp->opcode() : invert_cmp(hdr_cmp->opcode());

    auto dominates_preheader = [&](const Value* v) -> bool {
        if (!v) return false;
        if (v->is_block_param()) {
            const BasicBlock* b = v->defining_block();
            return b && cur_dom.dominates(b, preheader);
        }
        if (v->is_instruction()) {
            const Instruction* d = v->defining_instruction();
            return d && d->parent() && cur_dom.dominates(d->parent(), preheader);
        }
        return true;
    };

    AliasAnalysis aa(fn);
    RangeAnalysis ra(fn, cur_dom, cur_loops);

    auto can_materialize = [&](auto& self, Value* v, bool& has_load, int depth) -> bool {
        if (!v || depth > 12) return false;
        if (dominates_preheader(v)) return true;
        if (!v->is_instruction()) return false;
        Instruction* def = v->defining_instruction();
        if (!def || !def->parent()) return false;
        if (def->operand_count() == 0 && is_pure_unswitch_inst(def)) return true;
        if (!loop.contains(def->parent()) || slow_blocks.count(def->parent())) return false;
        if (def->opcode() == Opcode::load && !is_derived_gcref(v) && def->offset() >= 0 && def->offset() <= 64) {
            if (!self(self, def->operand(0), has_load, depth + 1)) return false;
            for (BasicBlock* lbb : loop.blocks()) {
                if (!lbb || slow_blocks.count(lbb)) continue;
                for (const Instruction* w : *lbb) {
                    if (w && aa.can_clobber(w, def)) return false;
                }
            }
            has_load = true;
            return true;
        }
        if (is_pure_unswitch_inst(def)) {
            for (Value* op : def->operands()) {
                if (!self(self, op, has_load, depth + 1)) return false;
            }
            return true;
        }
        return false;
    };

    Value* iv_param = nullptr;
    Value* init_v = nullptr;
    Value* limit_v = nullptr;
    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* p = header->param(i);
        if (!p || (p->type() != Type::i32() && p->type() != Type::i64())) continue;
        Opcode cand_op = hdr_op;
        Value* cand_lim = nullptr;
        if (hdr_cmp->operand(0) == p && loop.is_loop_invariant(hdr_cmp->operand(1))) {
            cand_lim = hdr_cmp->operand(1);
        } else if (hdr_cmp->operand(1) == p && loop.is_loop_invariant(hdr_cmp->operand(0))) {
            cand_lim = hdr_cmp->operand(0);
            cand_op = commute_cmp(hdr_op);
        } else {
            continue;
        }
        if (cand_op != Opcode::slt && cand_op != Opcode::ult && cand_op != Opcode::sle && cand_op != Opcode::ule) continue;
        bool lim_load = false;
        if (!can_materialize(can_materialize, cand_lim, lim_load, 0) || lim_load) continue;
        Value* step_v = latch_term->branch_target().args[i];
        if (!step_v || !step_v->is_instruction()) continue;
        Instruction* sdef = step_v->defining_instruction();
        if (!sdef || sdef->opcode() != Opcode::add) continue;
        Value* step_c_v = (sdef->operand(0) == p) ? sdef->operand(1) : ((sdef->operand(1) == p) ? sdef->operand(0) : nullptr);
        int64_t step_c = 0;
        if (!step_c_v || !get_const_int(step_c_v, step_c) || step_c != 1) continue;
        iv_param = p;
        init_v = ph_term->branch_target().args[i];
        limit_v = cand_lim;
        hdr_op = cand_op;
        break;
    }
    if (!iv_param || !init_v || !limit_v || !dominates_preheader(init_v)) return false;
    const bool hdr_is_strict = (hdr_op == Opcode::slt || hdr_op == Opcode::ult);
    const ValueRange init_r = ra.get_range(init_v);

    // Collect all fast/slow branches in fast-path blocks
    struct FastBranch {
        BasicBlock* bb;
        Instruction* term;
        bool fast_on_true;
    };
    std::vector<FastBranch> fast_branches;
    for (BasicBlock* bb : loop.blocks()) {
        if (!bb || slow_blocks.count(bb)) continue;
        Instruction* term = bb->terminator();
        if (!term || term->opcode() != Opcode::br_if) continue;
        BasicBlock* t_bb = term->true_target().block;
        BasicBlock* f_bb = term->false_target().block;
        if (bb == header && (!loop.contains(t_bb) || !loop.contains(f_bb))) continue;
        bool t_slow = slow_blocks.count(t_bb) > 0;
        bool f_slow = slow_blocks.count(f_bb) > 0;
        if (t_slow == f_slow) return false;
        fast_branches.push_back({bb, term, !t_slow});
    }
    if (fast_branches.empty()) return false;

    bool need_limit_24bit = false;
    auto match_iv_affine = [&](Value* v, AffineIvExpr& out) -> bool {
        out = AffineIvExpr{};
        auto strip_ext = [&](Value* x) -> Value* {
            while (x && x->is_instruction()) {
                Instruction* d = x->defining_instruction();
                if (!d) break;
                if (d->opcode() == Opcode::sext_i64 || d->opcode() == Opcode::zext_i64) {
                    x = d->operand(0);
                } else if (d->opcode() == Opcode::trunc_i32) {
                    need_limit_24bit = true;
                    x = d->operand(0);
                } else {
                    break;
                }
            }
            return x;
        };
        Value* cur = strip_ext(v);
        if (cur && cur->is_instruction()) {
            Instruction* d = cur->defining_instruction();
            int64_t sc = 0;
            if (d->opcode() == Opcode::shl && get_const_int(d->operand(1), sc) && sc >= 0 && sc <= 4) {
                out.outer_shift = static_cast<int32_t>(sc);
                cur = strip_ext(d->operand(0));
            }
        }
        if (cur && cur->is_instruction() && cur->defining_instruction()->opcode() == Opcode::add) {
            Instruction* d = cur->defining_instruction();
            Value* op0 = strip_ext(d->operand(0));
            Value* op1 = strip_ext(d->operand(1));
            int64_t c = 0;
            bool dummy = false;
            if (get_const_int(op1, c) && c >= 0) {
                out.const_offset = c;
                cur = op0;
            } else if (get_const_int(op0, c) && c >= 0) {
                out.const_offset = c;
                cur = op1;
            } else if (can_materialize(can_materialize, d->operand(0), dummy, 0)) {
                out.inv_offset = d->operand(0);
                cur = op1;
            } else if (can_materialize(can_materialize, d->operand(1), dummy, 0)) {
                out.inv_offset = d->operand(1);
                cur = op0;
            }
        }
        if (cur && cur->is_instruction()) {
            Instruction* d = cur->defining_instruction();
            int64_t m = 0;
            if (d->opcode() == Opcode::mul) {
                if (get_const_int(d->operand(1), m) && m >= 1 && m <= 64) {
                    out.scale = m;
                    cur = strip_ext(d->operand(0));
                } else if (get_const_int(d->operand(0), m) && m >= 1 && m <= 64) {
                    out.scale = m;
                    cur = strip_ext(d->operand(1));
                }
            } else if (d->opcode() == Opcode::shl && out.outer_shift == 0 && get_const_int(d->operand(1), m) && m >= 0 && m <= 4) {
                out.outer_shift = static_cast<int32_t>(m);
                cur = strip_ext(d->operand(0));
            }
        }
        return cur == iv_param;
    };

    struct InvClause {
        Value* val;
        bool want_true;
    };
    struct BoundClause {
        Opcode op;
        AffineIvExpr iv_expr;
        Value* rhs;
    };
    std::vector<InvClause> stage1_clauses;
    std::vector<InvClause> stage2_clauses;
    std::vector<BoundClause> bound_clauses;
    bool need_iv_non_neg = (init_r.min_val < 0);

    auto is_bool_tree = [](auto& self, const Value* v, int depth) -> bool {
        if (!v || !v->is_instruction() || depth > 5) return false;
        const Instruction* d = v->defining_instruction();
        if (!d) return false;
        if (is_comparison(d->opcode())) return true;
        int64_t c = 0;
        if (get_const_int(v, c)) return c == 0 || c == 1;
        if ((d->opcode() == Opcode::and_ || d->opcode() == Opcode::or_) && d->operand_count() == 2) {
            return self(self, d->operand(0), depth + 1) && self(self, d->operand(1), depth + 1);
        }
        return false;
    };

    std::vector<InvClause> raw_clauses;
    auto collect_clauses = [&](auto& self, Value* v, bool want_true, int depth) -> void {
        if (!v || depth > 8) return;
        if (v->is_instruction()) {
            Instruction* d = v->defining_instruction();
            if (d && d->opcode() == Opcode::and_ && want_true && is_bool_tree(is_bool_tree, v, 0)) {
                self(self, d->operand(0), true, depth + 1);
                self(self, d->operand(1), true, depth + 1);
                return;
            }
            if (d && d->opcode() == Opcode::or_ && !want_true) {
                self(self, d->operand(0), false, depth + 1);
                self(self, d->operand(1), false, depth + 1);
                return;
            }
        }
        raw_clauses.push_back({v, want_true});
    };
    for (const auto& fb : fast_branches) {
        collect_clauses(collect_clauses, fb.term->operand(0), fb.fast_on_true, 0);
    }

    auto match_sext_trunc = [](Value* ext_v, Value* orig_v) -> bool {
        if (!ext_v || !orig_v || !ext_v->is_instruction()) return false;
        Instruction* d = ext_v->defining_instruction();
        if (!d || d->opcode() != Opcode::sext_i64 || !d->operand(0) || !d->operand(0)->is_instruction()) return false;
        Instruction* td = d->operand(0)->defining_instruction();
        return td && td->opcode() == Opcode::trunc_i32 && td->operand(0) == orig_v;
    };

    for (const auto& rc : raw_clauses) {
        int64_t c = 0;
        if (get_const_int(rc.val, c)) {
            if ((c != 0) == rc.want_true) continue;
            return false;
        }
        bool has_load = false;
        if (can_materialize(can_materialize, rc.val, has_load, 0)) {
            if (rc.val->type() != Type::i32() && rc.val->type() != Type::i64()) return false;
            if (has_load) stage2_clauses.push_back(rc);
            else stage1_clauses.push_back(rc);
            continue;
        }
        if (!rc.val || !rc.val->is_instruction()) return false;
        Instruction* cmp = rc.val->defining_instruction();
        if (!cmp || !is_comparison(cmp->opcode()) || cmp->operand_count() != 2) return false;
        Opcode op = rc.want_true ? cmp->opcode() : invert_cmp(cmp->opcode());
        Value* lhs = cmp->operand(0);
        Value* rhs = cmp->operand(1);
        if (!lhs || !rhs || (lhs->type() != Type::i32() && lhs->type() != Type::i64())) return false;

        if (op == Opcode::eq) {
            Value* idx_v = match_sext_trunc(lhs, rhs) ? rhs : (match_sext_trunc(rhs, lhs) ? lhs : nullptr);
            AffineIvExpr iv_expr;
            if (idx_v && match_iv_affine(idx_v, iv_expr)) {
                need_limit_24bit = true;
                if (init_r.min_val < 0 || iv_expr.inv_offset != nullptr) need_iv_non_neg = true;
                continue;
            }
        }

        bool lhs_mat = can_materialize(can_materialize, lhs, has_load, 0);
        bool rhs_mat = can_materialize(can_materialize, rhs, has_load, 0);
        if (lhs_mat && !rhs_mat) {
            std::swap(lhs, rhs);
            op = commute_cmp(op);
            rhs_mat = true;
        }
        if (!rhs_mat) return false;

        AffineIvExpr iv_expr;
        if (!match_iv_affine(lhs, iv_expr)) return false;

        int64_t rhs_c = 0;
        if ((op == Opcode::sge && get_const_int(rhs, rhs_c) && rhs_c == 0) ||
            (op == Opcode::sgt && get_const_int(rhs, rhs_c) && rhs_c == -1)) {
            if (init_r.min_val < 0 || iv_expr.inv_offset != nullptr) need_iv_non_neg = true;
            continue;
        }
        if (op == Opcode::slt || op == Opcode::ult || op == Opcode::sle || op == Opcode::ule) {
            bound_clauses.push_back({op, iv_expr, rhs});
            continue;
        }
        return false;
    }

    // Check region size before cloning
    std::vector<BasicBlock*> region;
    size_t region_instructions = 0;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || !cur_dom.dominates(header, bb)) continue;
        region.push_back(bb);
        for (Instruction* inst : *bb) {
            if (inst) ++region_instructions;
        }
    }
    if (region_instructions > 3 * max_fast_insts) return false;

    const std::vector<Value*> init_args = ph_term->branch_target().args;

    // Clone region for the fallback loop (preserving all original checks in the fallback loop)
    ir::BlockMap blocks;
    ir::ValueMap values;
    for (BasicBlock* bb : region) {
        BasicBlock* copy = ir::new_block(fn, std::string(bb->name()) + "_unsw_f");
        blocks[bb] = copy;
        for (size_t i = 0; i < bb->param_count(); ++i) {
            values[bb->param(i)] = ir::new_block_param(fn, copy, bb->param(i)->type());
        }
    }
    std::vector<std::pair<Instruction*, Instruction*>> pairs;
    for (BasicBlock* bb : region) {
        BasicBlock* copy = blocks[bb];
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            Instruction* shell = ir::clone_shell(fn, *inst, values);
            copy->append_instruction(shell);
            pairs.emplace_back(inst, shell);
        }
    }
    for (auto& [orig, copy] : pairs) {
        ir::clone_uses(*orig, *copy, values, blocks);
    }

    // Fold all fast_branches in the fast loop to their fast target
    for (const auto& fb : fast_branches) {
        BranchTarget kept = fb.fast_on_true ? fb.term->true_target() : fb.term->false_target();
        Instruction* br_fast = make_branch(fn, Opcode::br);
        br_fast->set_branch_target(std::move(kept));
        fb.bb->remove_instruction(fb.term);
        fb.bb->append_instruction(br_fast);
    }

    // In the fast loop, any sext_i64(trunc_i32(x)) of an affine i64 IV expression
    // is guaranteed equal to x under the preheader/guard bounds, so forward x directly.
    for (BasicBlock* bb : loop.blocks()) {
        if (!bb || slow_blocks.count(bb)) continue;
        for (Instruction* inst = bb->head(); inst != nullptr;) {
            Instruction* next = inst->next();
            if (inst->opcode() == Opcode::sext_i64 && inst->operand(0) && inst->operand(0)->is_instruction()) {
                Instruction* td = inst->operand(0)->defining_instruction();
                AffineIvExpr dummy_expr;
                if (td && td->opcode() == Opcode::trunc_i32 && td->operand(0) &&
                    td->operand(0)->type() == Type::i64() && match_iv_affine(td->operand(0), dummy_expr)) {
                    replace_all_uses(fn, inst->result(), td->operand(0));
                    bb->remove_instruction(inst);
                }
            }
            inst = next;
        }
    }

    Builder b(*fn.parent());
    b.set_function(&fn);

    // Materializer for pure/load instructions into a target block. The returned
    // lambda owns its block and its memo: they must outlive make_materializer's
    // frame (captured by reference they dangled, and the memo lookups read a
    // dead stack slot).
    auto make_materializer = [&](BasicBlock* target_bb) {
        auto materialize = [&, target_bb, mat_map = std::unordered_map<const Value*, Value*>{}](
                               auto& self, Value* v) mutable -> Value* {
            if (!v) return nullptr;
            if (dominates_preheader(v)) return v;
            auto it = mat_map.find(v);
            if (it != mat_map.end()) return it->second;
            Instruction* def = v->defining_instruction();
            std::vector<Value*> m_ops;
            m_ops.reserve(def->operand_count());
            for (Value* op : def->operands()) {
                m_ops.push_back(self(self, op));
            }
            // Deduplicate identical instruction in target_bb
            for (Instruction* prev = target_bb->head(); prev != nullptr; prev = prev->next()) {
                if (prev->opcode() == def->opcode() && prev->type() == def->type() &&
                    prev->memory_type() == def->memory_type() && prev->imm_i64() == def->imm_i64() &&
                    prev->operand_count() == m_ops.size()) {
                    bool same = true;
                    for (size_t k = 0; k < m_ops.size(); ++k) {
                        if (prev->operand(k) != m_ops[k]) { same = false; break; }
                    }
                    if (same) {
                        mat_map[v] = prev->result();
                        return prev->result();
                    }
                }
            }
            ir::ValueMap tmp_map;
            Instruction* clone = ir::clone_shell(fn, *def, tmp_map);
            for (size_t k = 0; k < m_ops.size(); ++k) {
                clone->add_operand(m_ops[k]);
            }
            target_bb->append_instruction(clone);
            mat_map[v] = clone->result();
            return clone->result();
        };
        return materialize;
    };

    auto to_bool_i32 = [&](Value* cv, bool want_true) -> Value* {
        if (want_true && cv->type() == Type::i32() && cv->is_instruction() &&
            is_comparison(cv->defining_instruction()->opcode())) {
            return cv;
        }
        Value* z = (cv->type() == Type::i32()) ? b.build_iconst_i32(0) : b.build_iconst_i64(0);
        return want_true ? b.build_ne(cv, z) : b.build_eq(cv, z);
    };

    // Build stage 1 condition in preheader (before ph_term)
    preheader->remove_instruction(ph_term);
    auto mat_stage1 = make_materializer(preheader);
    Value* mat_limit_v = mat_stage1(mat_stage1, limit_v);
    std::vector<Value*> stage1_vals;
    for (const auto& sc : stage1_clauses) {
        Value* mv = mat_stage1(mat_stage1, sc.val);
        stage1_vals.push_back(mv);
    }

    b.position_at_end(preheader);
    Value* stage1_cond = (hdr_op == Opcode::slt) ? b.build_slt(init_v, mat_limit_v)
                       : (hdr_op == Opcode::ult) ? b.build_ult(init_v, mat_limit_v)
                       : (hdr_op == Opcode::sle) ? b.build_sle(init_v, mat_limit_v)
                                                 : b.build_ule(init_v, mat_limit_v);
    if (need_iv_non_neg) {
        Value* zero_iv = (iv_param->type() == Type::i32()) ? b.build_iconst_i32(0) : b.build_iconst_i64(0);
        stage1_cond = b.build_and(stage1_cond, b.build_sge(init_v, zero_iv));
    }
    for (size_t idx = 0; idx < stage1_clauses.size(); ++idx) {
        Value* cv = to_bool_i32(stage1_vals[idx], stage1_clauses[idx].want_true);
        stage1_cond = b.build_and(stage1_cond, cv);
    }

    // Create guard_bb right before header
    BasicBlock* guard_bb = ir::new_block(fn, std::string(header->name()) + "_unsw_guard");
    {
        auto& fn_blks = fn.blocks();
        auto it_hdr = std::find(fn_blks.begin(), fn_blks.end(), header);
        if (it_hdr != fn_blks.end()) {
            fn_blks.erase(std::find(fn_blks.begin(), fn_blks.end(), guard_bb));
            it_hdr = std::find(fn_blks.begin(), fn_blks.end(), header);
            fn_blks.insert(it_hdr, guard_bb);
        }
    }

    Instruction* ph_br = make_branch(fn, Opcode::br_if);
    ph_br->add_operand(stage1_cond);
    ph_br->set_true_target(BranchTarget(guard_bb, {}));
    ph_br->set_false_target(BranchTarget(blocks[header], init_args));
    preheader->append_instruction(ph_br);

    // Build stage 2 conditions and monotonic bound checks in guard_bb
    auto mat_stage2 = make_materializer(guard_bb);
    std::vector<Value*> stage2_vals;
    for (const auto& sc : stage2_clauses) {
        stage2_vals.push_back(mat_stage2(mat_stage2, sc.val));
    }
    struct MatBound {
        BoundClause bc;
        Value* rhs_m;
        Value* inv_off_m;
    };
    std::vector<MatBound> mat_bounds;
    for (const auto& bc : bound_clauses) {
        Value* rm = mat_stage2(mat_stage2, bc.rhs);
        Value* iom = bc.iv_expr.inv_offset ? mat_stage2(mat_stage2, bc.iv_expr.inv_offset) : nullptr;
        mat_bounds.push_back({bc, rm, iom});
    }

    b.position_at_end(guard_bb);
    Value* stage2_cond = nullptr;
    auto add_s2 = [&](Value* v) {
        stage2_cond = stage2_cond ? b.build_and(stage2_cond, v) : v;
    };
    for (size_t idx = 0; idx < stage2_clauses.size(); ++idx) {
        add_s2(to_bool_i32(stage2_vals[idx], stage2_clauses[idx].want_true));
    }

    auto to_i64 = [&](Value* v) -> Value* {
        if (v->type() == Type::i32()) return b.build_sext_i64(v);
        return v;
    };
    Value* limit_i64 = to_i64(mat_limit_v);
    if (need_limit_24bit) {
        add_s2(b.build_ult(limit_i64, b.build_iconst_i64(1LL << 24)));
    }
    for (const auto& mb : mat_bounds) {
        const auto& bc = mb.bc;
        Value* rhs_i64 = to_i64(mb.rhs_m);
        if ((bc.op == Opcode::slt || bc.op == Opcode::sle) && ra.get_range(bc.rhs).min_val < 0) {
            add_s2(b.build_sge(rhs_i64, b.build_iconst_i64(0)));
        }
        Value* bound_rhs = (bc.iv_expr.outer_shift > 0)
            ? b.build_lshr(rhs_i64, b.build_iconst_i64(bc.iv_expr.outer_shift))
            : rhs_i64;
        const bool check_strict = (bc.op == Opcode::slt || bc.op == Opcode::ult || bc.iv_expr.outer_shift > 0);
        if (bc.iv_expr.scale == 1 && !mb.inv_off_m && bc.iv_expr.const_offset == 0 && hdr_is_strict && check_strict) {
            if (bc.iv_expr.outer_shift > 0 && iv_param->type() == Type::i32()) {
                add_s2(b.build_ult(limit_i64, b.build_iconst_i64(1LL << 26)));
            }
            add_s2(b.build_ule(limit_i64, bound_rhs));
        } else {
            add_s2(b.build_ult(limit_i64, b.build_iconst_i64(1LL << 24)));
            Value* max_iv = hdr_is_strict ? b.build_sub(limit_i64, b.build_iconst_i64(1)) : limit_i64;
            Value* max_idx = max_iv;
            if (bc.iv_expr.scale > 1) {
                max_idx = b.build_mul(max_idx, b.build_iconst_i64(bc.iv_expr.scale));
            }
            if (mb.inv_off_m) {
                Value* io64 = to_i64(mb.inv_off_m);
                add_s2(b.build_ult(io64, b.build_iconst_i64(1LL << 24)));
                max_idx = b.build_add(max_idx, io64);
            }
            if (bc.iv_expr.const_offset > 0) {
                max_idx = b.build_add(max_idx, b.build_iconst_i64(bc.iv_expr.const_offset));
            }
            add_s2(check_strict ? b.build_ult(max_idx, bound_rhs) : b.build_ule(max_idx, bound_rhs));
        }
    }

    if (stage2_cond) {
        Instruction* g_br = make_branch(fn, Opcode::br_if);
        g_br->add_operand(stage2_cond);
        g_br->set_true_target(BranchTarget(header, init_args));
        g_br->set_false_target(BranchTarget(blocks[header], init_args));
        guard_bb->append_instruction(g_br);
    } else {
        Instruction* g_br = make_branch(fn, Opcode::br);
        g_br->set_branch_target(BranchTarget(header, init_args));
        guard_bb->append_instruction(g_br);
    }

    fn.rebuild_cfg_predecessors();
    CfgSimplifyOptions cfg_opts;
    cfg_opts.enable_param_elimination = false;
    cfg_simplify_function(fn, cfg_opts);
    return true;
}

} // namespace brass
