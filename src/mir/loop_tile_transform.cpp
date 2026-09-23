// Tiling of the outer two loops of a nest, done in place:
//
//   for i in [i0, N):                 for it = i0; it < N; it = iend:
//     pre(i)                            iend = min(it + Ti, N)
//     for j in [j0, M):        ==>      for jt = j0; jt < M; jt = jend:
//       body(i, j)                        jend = min(jt + Tj, M)
//     post(i)                             for i in [it, iend):
//                                           pre(i)
//                                           for j in [jt, jend): body(i, j)
//                                           post(i)
//
// The original loops keep their blocks; only their bounds and the outer
// loop's exit change. That is exact when:
//   - both loops count up by 1 under slt/ult with bounds fixed for the nest,
//     carry nothing but their induction variable, and leave only through
//     their header (so the inner loop runs at most once per outer iteration);
//   - everything the outer loop does outside the inner loop, and both
//     headers, is pure and cannot trap, so running it once per (i, tile)
//     instead of once per i changes nothing;
//   - the inner loop's body has no effect but plain loads and stores, and no
//     two iterations whose order tiling swaps (i1 < i2 with j1 > j2) touch
//     the same bytes where one of them writes.
// Loops deeper in the nest run whole inside each (i, j) iteration.

#include "loop_tile_transform.hpp"
#include "ir_clone.hpp"
#include "loop_affine.hpp"
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/opcodes.hpp>
#include <cstdlib>
#include <limits>
#include <string>

namespace brass {

namespace {

struct CountedLoop {
    LoopInfo* loop = nullptr;
    BasicBlock* header = nullptr;
    BasicBlock* preheader = nullptr;
    Instruction* hdr_term = nullptr;
    Value* iv = nullptr;
    Value* init = nullptr;
    Value* limit = nullptr;
    Opcode pred = Opcode::slt;
};

bool available_for(const LoopInfo& nest, const Value* v) {
    return affine::defined_outside(nest, v) || affine::is_plain_constant(v);
}

// A loop `for (iv = init; iv pred limit; iv += 1)` with one entry and one
// exit (the header's false edge), carrying nothing but iv.
bool match_counted(LoopInfo& loop, const LoopInfo& nest, CountedLoop& out) {
    BasicBlock* header = loop.header();
    if (!header || loop.latches().size() != 1 || header->param_count() != 1) return false;
    Instruction* latch_term = loop.latches()[0]->terminator();
    if (!latch_term || latch_term->opcode() != Opcode::br || latch_term->branch_target().block != header) return false;

    BasicBlock* preheader = nullptr;
    for (BasicBlock* pred : header->predecessors()) {
        if (!pred || loop.contains(pred)) continue;
        if (preheader && preheader != pred) return false;
        preheader = pred;
    }
    Instruction* ph_term = preheader ? preheader->terminator() : nullptr;
    if (!ph_term || ph_term->opcode() != Opcode::br || ph_term->branch_target().block != header) return false;

    Instruction* hdr_term = header->terminator();
    if (!hdr_term || hdr_term->opcode() != Opcode::br_if) return false;
    BasicBlock* exit = hdr_term->false_target().block;
    if (!loop.contains(hdr_term->true_target().block) || !exit || loop.contains(exit)) return false;
    for (BasicBlock* bb : loop.blocks()) {
        for (BasicBlock* succ : bb->successors()) {
            if (!loop.contains(succ) && !(bb == header && succ == exit)) return false;
        }
    }

    Value* iv = header->param(0);
    if (iv->type() != Type::i32() && iv->type() != Type::i64()) return false;
    Value* cond = hdr_term->operand(0);
    Instruction* cmp = cond && cond->is_instruction() ? cond->defining_instruction() : nullptr;
    if (!cmp || cmp->parent() != header || (cmp->opcode() != Opcode::slt && cmp->opcode() != Opcode::ult) ||
        cmp->operand(0) != iv || !available_for(nest, cmp->operand(1))) {
        return false;
    }

    Value* next = latch_term->branch_target().args[0];
    Instruction* inc = next && next->is_instruction() ? next->defining_instruction() : nullptr;
    int64_t step = 0;
    if (!inc || inc->opcode() != Opcode::add ||
        !((inc->operand(0) == iv && affine::is_int_constant(inc->operand(1), step)) ||
          (inc->operand(1) == iv && affine::is_int_constant(inc->operand(0), step))) ||
        step != 1) {
        return false;
    }
    Value* init = ph_term->branch_target().args[0];
    if (!available_for(nest, init)) return false;

    out.loop = &loop;
    out.header = header;
    out.preheader = preheader;
    out.hdr_term = hdr_term;
    out.iv = iv;
    out.init = init;
    out.limit = cmp->operand(1);
    out.pred = cmp->opcode();
    return true;
}

bool is_zero(const Value* v) {
    int64_t c = 0;
    return affine::is_int_constant(v, c) && c == 0;
}

// The coefficient `c * (limit of `loop`)` a row index needs for the address
// to be (row * range + col) * c with col in [0, range).
bool is_row_stride(const std::pair<const Value*, int64_t>& term, const CountedLoop& col, int64_t c) {
    int64_t lim = 0;
    if (affine::is_int_constant(col.limit, lim)) {
        int64_t want = 0;
        return term.first == nullptr && affine::checked_mul(c, lim, want) && term.second == want;
    }
    return term.first == col.limit && term.second == c;
}

// True when the address differs between any two iterations (i1, j1) and
// (i2, j2) with i1 != i2 and j1 != j2 by at least the access size: then the
// iterations tiling reorders never touch the same bytes.
bool distinct_on_swapped_pairs(const affine::Form& f, const CountedLoop& outer, const CountedLoop& inner) {
    std::vector<std::pair<const Value*, int64_t>> ti, tj;
    for (const auto& [key, coeff] : f.terms) {
        if (key.first == outer.iv) ti.emplace_back(key.second, coeff);
        else if (key.first == inner.iv) tj.emplace_back(key.second, coeff);
        else return false;
    }
    if (ti.size() > 1 || tj.size() > 1 || (ti.empty() && tj.empty())) return false;
    const int64_t size = static_cast<int64_t>(f.size);
    auto unit = [size](const std::pair<const Value*, int64_t>& t) {
        return t.first == nullptr && std::llabs(t.second) >= size;
    };
    if (tj.empty()) return unit(ti[0]);
    if (ti.empty()) return unit(tj[0]);
    // Row-major over (i, j), or column-major (j selects the row).
    if (unit(tj[0]) && is_zero(inner.init) && is_row_stride(ti[0], inner, tj[0].second)) return true;
    if (unit(ti[0]) && is_zero(outer.init) && is_row_stride(tj[0], outer, ti[0].second)) return true;
    return false;
}

bool region_instruction_ok(const Instruction& inst) {
    if (affine::is_pure_nontrapping(inst)) return true;
    const Opcode op = inst.opcode();
    return affine::is_plain_memory_access(op) || op == Opcode::br || op == Opcode::br_if || op == Opcode::switch_;
}

bool control_or_pure(const Instruction& inst) {
    const Opcode op = inst.opcode();
    return affine::is_pure_nontrapping(inst) || op == Opcode::br || op == Opcode::br_if || op == Opcode::switch_;
}

bool legal(Function& fn, const CountedLoop& outer, const CountedLoop& inner) {
    LoopInfo& l0 = *outer.loop;
    LoopInfo& l1 = *inner.loop;

    for (BasicBlock* bb : l0.blocks()) {
        const bool in_region = l1.contains(bb) && bb != inner.header;
        for (Instruction* inst : *bb) {
            if (in_region ? !region_instruction_ok(*inst) : !control_or_pure(*inst)) return false;
        }
    }

    // What leaves each loop must not depend on how its iterations were cut.
    if (affine::used_outside(fn, l1, inner.iv)) return false;
    for (Value* a : inner.hdr_term->false_target().args) {
        if (!affine::defined_outside(l1, a) && !affine::is_plain_constant(a)) return false;
    }
    for (Instruction* inst : *inner.header) {
        if (inst->result() && affine::used_outside(fn, l1, inst->result())) return false;
    }
    for (Value* a : outer.hdr_term->false_target().args) {
        if (a != outer.iv && !available_for(l0, a)) return false;
    }
    for (Instruction* inst : *outer.header) {
        if (inst->result() && affine::used_outside(fn, l0, inst->result())) return false;
    }

    const std::vector<const Value*> ivs = {outer.iv, inner.iv};
    const affine::InvariantFn invariant = [&l0](const Value* v) { return affine::defined_outside(l0, v); };
    std::vector<affine::Form> forms;
    for (BasicBlock* bb : l1.blocks()) {
        for (Instruction* inst : *bb) {
            if (affine::is_plain_memory_access(inst->opcode())) forms.push_back(affine::address_form(*inst, ivs, invariant));
        }
    }
    AliasAnalysis aa(fn);
    for (size_t a = 0; a < forms.size(); ++a) {
        for (size_t b = a; b < forms.size(); ++b) {
            if (!forms[a].is_store && !forms[b].is_store) continue;
            if (affine::same_form(forms[a], forms[b]) && distinct_on_swapped_pairs(forms[a], outer, inner)) continue;
            if (affine::distinct_objects(aa, forms[a].pointer, forms[b].pointer)) continue;
            return false;
        }
    }
    return true;
}

Value* available_value(Builder& b, const LoopInfo& nest, Value* v) {
    if (affine::defined_outside(nest, v)) return v;
    const Instruction* def = v->defining_instruction();
    switch (def->opcode()) {
        case Opcode::iconst_i32: return b.build_iconst_i32(def->imm_i32());
        case Opcode::iconst_i64: return b.build_iconst_i64(def->imm_i64());
        default: return b.build_fconst_f64(def->imm_f64());
    }
}

Value* build_cmp(Builder& b, Opcode pred, Value* lhs, Value* rhs) {
    return pred == Opcode::slt ? b.build_slt(lhs, rhs) : b.build_ult(lhs, rhs);
}

// min(start + tile, limit) for start below limit, without overflowing:
// the remaining distance limit - start is exact as an unsigned value.
Value* build_tile_end(Builder& b, Value* start, Value* limit, Type t, int64_t tile) {
    Value* tile_c = t == Type::i32() ? b.build_iconst_i32(static_cast<int32_t>(tile)) : b.build_iconst_i64(tile);
    Value* remaining = b.build_sub(limit, start);
    Value* more = b.build_ugt(remaining, tile_c);
    return b.build_select(more, b.build_add(start, tile_c), limit);
}

bool tile_sizes_ok(const LoopTileOptions& options, Type ti, Type tj) {
    auto fits = [](size_t s, Type t) {
        const size_t max = t == Type::i32() ? static_cast<size_t>(std::numeric_limits<int32_t>::max())
                                            : static_cast<size_t>(std::numeric_limits<int64_t>::max());
        return s >= 2 && s <= max;
    };
    return fits(options.tile_size_i, ti) && fits(options.tile_size_j, tj);
}

} // namespace

bool tile_2d_loop_nest(Function& fn, LoopInfo& outer_loop, const LoopTileOptions& options) {
    if (outer_loop.sub_loops().size() != 1) return false;
    LoopInfo& inner_loop = *outer_loop.sub_loops()[0];
    if (outer_loop.header() && outer_loop.header()->name().find("_2dt") != std::string_view::npos) return false;

    CountedLoop outer, inner;
    if (!match_counted(outer_loop, outer_loop, outer) || !match_counted(inner_loop, outer_loop, inner)) return false;
    if (!tile_sizes_ok(options, outer.iv->type(), inner.iv->type())) return false;
    if (!legal(fn, outer, inner)) return false;

    Module* mod = fn.parent();
    if (!mod) return false;
    Builder b(*mod);
    b.set_function(&fn);

    const std::string pfx = std::string(outer.header->name()) + "_2dt";
    BasicBlock* ti = ir::new_block(fn, pfx + "_i");
    BasicBlock* ti_body = ir::new_block(fn, pfx + "_i_body");
    BasicBlock* tj = ir::new_block(fn, pfx + "_j");
    BasicBlock* tj_body = ir::new_block(fn, pfx + "_j_body");
    BasicBlock* tj_latch = ir::new_block(fn, pfx + "_j_latch");
    BasicBlock* ti_latch = ir::new_block(fn, pfx + "_i_latch");
    const Type ty0 = outer.iv->type();
    const Type ty1 = inner.iv->type();
    Value* it = ir::new_block_param(fn, ti, ty0);
    Value* jt = ir::new_block_param(fn, tj, ty1);

    // Enter the tile loop instead of the outer header.
    outer.preheader->terminator()->branch_target().block = ti;

    // ti: it < N, else leave the nest the way the outer header did.
    b.position_at_end(ti);
    Value* n = available_value(b, *outer.loop, outer.limit);
    BranchTarget exit_bt;
    exit_bt.block = outer.hdr_term->false_target().block;
    for (Value* a : outer.hdr_term->false_target().args) {
        exit_bt.args.push_back(a == outer.iv ? it : available_value(b, *outer.loop, a));
    }
    b.build_br_if(build_cmp(b, outer.pred, it, n), ti_body, {}, exit_bt.block, exit_bt.args);

    b.position_at_end(ti_body);
    Value* iend = build_tile_end(b, it, n, ty0, static_cast<int64_t>(options.tile_size_i));
    b.build_br(tj, {available_value(b, *outer.loop, inner.init)});

    b.position_at_end(tj);
    Value* m = available_value(b, *outer.loop, inner.limit);
    b.build_br_if(build_cmp(b, inner.pred, jt, m), tj_body, {}, ti_latch, {});

    b.position_at_end(tj_body);
    Value* jend = build_tile_end(b, jt, m, ty1, static_cast<int64_t>(options.tile_size_j));
    b.build_br(outer.header, {it});

    // The point loops run over the current tile.
    b.position_before(outer.hdr_term);
    outer.hdr_term->set_operand(0, build_cmp(b, outer.pred, outer.iv, iend));
    BranchTarget to_tj_latch;
    to_tj_latch.block = tj_latch;
    outer.hdr_term->set_false_target(to_tj_latch);

    inner.preheader->terminator()->branch_target().args[0] = jt;
    b.position_before(inner.hdr_term);
    inner.hdr_term->set_operand(0, build_cmp(b, inner.pred, inner.iv, jend));

    b.position_at_end(tj_latch);
    b.build_br(tj, {jend});
    b.position_at_end(ti_latch);
    b.build_br(ti, {iend});

    // After the nest the outer induction variable is the tile loop's, which
    // ends on the same value (N, or i0 when the loop never ran).
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || outer.loop->contains(bb) || bb == ti) continue;
        for (Instruction* inst : *bb) {
            for_each_use_slot(*inst, [&](Value*& v) {
                if (v == outer.iv) v = it;
            });
        }
    }

    fn.rebuild_cfg_predecessors();
    return true;
}

} // namespace brass
