#include "f64_demote_internal.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/opcodes.hpp>
#include <string_view>
#include <unordered_set>
#include <vector>

// A loop's counter stepped by a value the loop does not change — `i += len`,
// `i += stride` — is bounded only if the step is positive: the header's
// `i < n` caps the counter from above, and a step that can be zero or
// negative lets it fall without limit, past 2^53 and off what an i64 computes
// the same. Where the step is a loop value of its own its range is often all
// the int32s (`len <<= 1` wraps negative once len passes 2^30), so the bounds
// cannot clear the counter, it stays an f64, and so does every index made from
// it: the FFT's `real[i + k + half]` converted to an integer and checked
// back, per access.
//
// So the loop is versioned on its step: the preheader tests 1 <= step <=
// 2^31 and enters the loop as written, or else a copy of it. The test is a
// fact the bounds read (f64_demote_bounds.cpp, `collect_branch_facts`) in
// the blocks the loop's header dominates, where it then bounds the counter
// by the limit plus one step; the copy, for every other step, is the loop as
// it was and keeps its f64s.
namespace brass {

namespace {

constexpr double kStepMax = 2147483648.0;  // 2^31: the counter stays within the limit plus this

bool is_copy(const BasicBlock* bb) {
    const std::string_view n = bb->name();
    return n.find("_f64c") != std::string_view::npos || n.find("_f64v") != std::string_view::npos;
}

const BasicBlock* defining_block(const Value* v) {
    if (v->is_block_param()) return v->defining_block();
    if (v->is_instruction()) return v->defining_instruction()->parent();
    return nullptr;
}

// The argument a block parameter takes from its block's one predecessor, when
// that predecessor only jumps there; otherwise the value itself.
const Value* through_jumps(const Value* v) {
    for (int hop = 0; hop < 4 && v && v->is_block_param(); ++hop) {
        const BasicBlock* bb = v->defining_block();
        if (!bb || bb->predecessors().size() != 1) break;
        const Instruction* term = bb->predecessors()[0]->terminator();
        if (!term || term->opcode() != Opcode::br || term->branch_target().block != bb) break;
        const auto& args = term->branch_target().args;
        if (v->param_index() >= args.size()) break;
        v = args[v->param_index()];
    }
    return v;
}

// Whether `v` holds an integer by how it is computed, as far as a quick look
// sees: only whether versioning could pay. Whether the counter is exact and
// bounded is the demotion's own analysis, which runs on the versioned loop.
bool looks_integral(const Value* v, int depth, std::unordered_set<const Value*>& seen) {
    if (!v) return false;
    if (v->type().is_integer()) return true;
    if (v->type() != Type::f64() || depth > 8) return false;
    if (!seen.insert(v).second) return true;  // a cycle adds nothing new
    if (v->is_block_param()) {
        const BasicBlock* bb = v->defining_block();
        if (!bb) return false;
        if (bb->predecessors().empty()) return true;  // an entry parameter: the entry check tests it
        for (const BasicBlock* pred : bb->predecessors()) {
            const Instruction* term = pred ? pred->terminator() : nullptr;
            if (!term) return false;
            for (const BranchTarget* bt : {&term->branch_target(), &term->true_target(), &term->false_target()}) {
                if (bt->block != bb) continue;
                if (v->param_index() >= bt->args.size() ||
                    !looks_integral(bt->args[v->param_index()], depth + 1, seen)) {
                    return false;
                }
            }
        }
        return true;
    }
    if (!v->is_instruction()) return false;
    const Instruction* d = v->defining_instruction();
    switch (d->opcode()) {
        case Opcode::fconst_f64: return is_safe_integer_f64(d->imm_f64());
        case Opcode::sitofp_f64_i32: case Opcode::sitofp_f64_i64: return true;
        case Opcode::add: case Opcode::sub: case Opcode::mul:
            return looks_integral(d->operand(0), depth + 1, seen) && looks_integral(d->operand(1), depth + 1, seen);
        case Opcode::neg: return looks_integral(d->operand(0), depth + 1, seen);
        case Opcode::select:
            return looks_integral(d->operand(1), depth + 1, seen) && looks_integral(d->operand(2), depth + 1, seen);
        default: return false;
    }
}

// The header parameters its exit test bounds from above: `p < limit` (or <=)
// on the edge that stays in the loop.
std::vector<const Value*> upper_guarded_params(const LoopInfo& loop) {
    std::vector<const Value*> out;
    const BasicBlock* header = loop.header();
    const Instruction* term = header->terminator();
    if (!term || term->opcode() != Opcode::br_if || !term->operand(0) || !term->operand(0)->is_instruction()) {
        return out;
    }
    const bool stay_true = loop.contains(term->true_target().block) && !loop.contains(term->false_target().block);
    const bool stay_false = loop.contains(term->false_target().block) && !loop.contains(term->true_target().block);
    if (!stay_true && !stay_false) return out;
    const Instruction* cmp = term->operand(0)->defining_instruction();
    if (cmp->opcode() == Opcode::and_ && cmp->operand_count() == 2) {
        for (size_t side = 0; side < 2; ++side) {
            const Value* m = cmp->operand(side);
            const Value* inner = cmp->operand(1 - side);
            if (m && m->is_instruction() && m->defining_instruction()->opcode() == Opcode::iconst_i32 &&
                m->defining_instruction()->imm_i32() == 1 && inner && inner->is_instruction()) {
                cmp = inner->defining_instruction();
                break;
            }
        }
    }
    if (cmp->operand_count() != 2) return out;
    bool less;
    switch (cmp->opcode()) {
        case Opcode::slt: case Opcode::sle: less = true; break;
        case Opcode::sgt: case Opcode::sge: less = false; break;
        default: return out;
    }
    if (stay_false) less = !less;
    for (size_t side = 0; side < 2; ++side) {
        const Value* p = cmp->operand(side);
        if (p && p->is_block_param() && p->defining_block() == header && ((side == 0) == less)) out.push_back(p);
    }
    return out;
}

bool loop_converts_to_integer(const LoopInfo& loop) {
    for (const BasicBlock* bb : loop.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst && (inst->opcode() == Opcode::fptosi_i64 || inst->opcode() == Opcode::fptosi_i32)) return true;
        }
    }
    return false;
}

// The step of one loop the bounds would need positive, or null.
const Value* unbounded_step(const LoopInfo& loop, const DominatorTree& dom, BasicBlock*& preheader) {
    BasicBlock* header = loop.header();
    if (!header || is_copy(header) || loop.latches().size() != 1) return nullptr;
    preheader = nullptr;
    for (BasicBlock* pred : header->predecessors()) {
        if (!pred || loop.contains(pred)) continue;
        if (preheader) return nullptr;
        preheader = pred;
    }
    const Instruction* ph_term = preheader ? preheader->terminator() : nullptr;
    if (!ph_term || ph_term->opcode() != Opcode::br || ph_term->branch_target().block != header) return nullptr;
    if (!loop_converts_to_integer(loop)) return nullptr;

    const Instruction* latch_term = loop.latches()[0]->terminator();
    if (!latch_term) return nullptr;
    for (const Value* p : upper_guarded_params(loop)) {
        if (p->type() != Type::f64()) continue;
        const size_t idx = p->param_index();
        const Value* next = nullptr;
        const Value* init = nullptr;
        for (const BranchTarget* bt : {&latch_term->branch_target(), &latch_term->true_target(), &latch_term->false_target()}) {
            if (bt->block == header && idx < bt->args.size()) next = bt->args[idx];
        }
        if (idx < ph_term->branch_target().args.size()) init = ph_term->branch_target().args[idx];
        next = through_jumps(next);
        if (!next || !next->is_instruction() || next->defining_instruction()->opcode() != Opcode::add) continue;
        const Instruction* add = next->defining_instruction();
        const Value* a = through_jumps(add->operand(0));
        const Value* b = through_jumps(add->operand(1));
        const Value* step = a == p ? add->operand(1) : (b == p ? add->operand(0) : nullptr);
        if (!step || step->type() != Type::f64()) continue;
        if (step->is_instruction() && step->defining_instruction()->opcode() == Opcode::fconst_f64) continue;
        const BasicBlock* def = defining_block(step);
        if (!def || !dom.dominates(def, preheader)) continue;
        std::unordered_set<const Value*> seen;
        if (!looks_integral(step, 0, seen) || !looks_integral(init, 0, seen)) continue;
        return step;
    }
    return nullptr;
}

} // namespace

bool version_loops_on_step(Function& fn) {
    constexpr int kMaxVersions = 8;
    std::unordered_set<const BasicBlock*> tried;
    bool changed = false;
    for (int round = 0; round < kMaxVersions; ++round) {
        fn.rebuild_cfg_predecessors();
        const DominatorTree dom(fn);
        const LoopAnalysis loops(fn, dom);
        BasicBlock* preheader = nullptr;
        const LoopInfo* site_loop = nullptr;
        const Value* step = nullptr;
        for (const LoopInfo* loop : loops.post_order_loops()) {
            if (!loop || !loop->header() || tried.count(loop->header())) continue;
            step = unbounded_step(*loop, dom, preheader);
            if (step) { site_loop = loop; break; }
        }
        if (!site_loop) break;
        BasicBlock* header = site_loop->header();
        tried.insert(header);

        Builder b(*fn.parent());
        b.set_function(&fn);
        b.position_before(preheader->terminator());
        Value* s = const_cast<Value*>(step);
        Value* fits = b.build_and(b.build_sge(s, b.build_fconst_f64(1.0)), b.build_sle(s, b.build_fconst_f64(kStepMax)));
        ClampedLimit site;
        site.preheader = preheader;
        site.header = header;
        site.fits = fits;
        if (version_loop_on_clamp(fn, site, {})) changed = true;
    }
    if (changed) fn.rebuild_cfg_predecessors();
    return changed;
}

} // namespace brass
