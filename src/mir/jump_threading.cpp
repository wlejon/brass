#include <brass/mir/jump_threading.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include "int_fold.hpp"
#include "ir_clone.hpp"
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

// Recursion bound for evaluating B's instructions along one incoming edge.
constexpr int kMaxEvalDepth = 8;

std::optional<int64_t> const_int(const Value* val) {
    if (!val || !val->is_instruction()) return std::nullopt;
    const Instruction* def = val->defining_instruction();
    if (!def) return std::nullopt;
    if (def->opcode() == Opcode::iconst_i32) return static_cast<int64_t>(def->imm_i32());
    if (def->opcode() == Opcode::iconst_i64) return def->imm_i64();
    return std::nullopt;
}

// Value of `val` when B is entered with `incoming_args`, if that follows
// from constants alone. Arithmetic goes through the width-correct folder so
// the answer is the one the machine computes.
std::optional<int64_t> resolve_along_edge(
    const BasicBlock* B,
    const Value* val,
    const std::vector<Value*>& incoming_args,
    int depth
) {
    if (!val || depth > kMaxEvalDepth) return std::nullopt;
    if (auto c = const_int(val)) return c;

    if (val->is_block_param() && val->defining_block() == B) {
        const uint32_t idx = val->param_index();
        if (idx < incoming_args.size()) return const_int(incoming_args[idx]);
        return std::nullopt;
    }

    if (!val->is_instruction()) return std::nullopt;
    const Instruction* def = val->defining_instruction();
    if (!def || def->parent() != B || def->operand_count() != 2) return std::nullopt;
    const Value* lhs = def->operand(0);
    if (!lhs) return std::nullopt;
    const unsigned width = int_fold::width_of(is_comparison(def->opcode()) ? lhs->type() : def->type());
    if (width == 0 || lhs->type().is_pointer_or_gcref()) return std::nullopt;

    auto a = resolve_along_edge(B, lhs, incoming_args, depth + 1);
    if (!a) return std::nullopt;
    auto b = resolve_along_edge(B, def->operand(1), incoming_args, depth + 1);
    if (!b) return std::nullopt;
    switch (def->opcode()) {
        case Opcode::add: case Opcode::sub: case Opcode::mul:
        case Opcode::and_: case Opcode::or_: case Opcode::xor_:
        case Opcode::eq: case Opcode::ne:
        case Opcode::slt: case Opcode::sle: case Opcode::sgt: case Opcode::sge:
        case Opcode::ult: case Opcode::ule: case Opcode::ugt: case Opcode::uge:
            return int_fold::binary(def->opcode(), width, *a, *b);
        default:
            return std::nullopt;
    }
}

// 1 or 0 when B's branch condition is known on the edge from P, -1 otherwise.
int evaluate_condition_along_edge(
    const BasicBlock* B,
    const Value* cond,
    const std::vector<Value*>& incoming_args,
    const BasicBlock* P,
    bool is_true_edge,
    bool is_false_edge
) {
    if (!cond) return -1;
    if (auto v = resolve_along_edge(B, cond, incoming_args, 0)) return *v != 0 ? 1 : 0;

    // P branched on the same SSA value to reach B, so its outcome is known.
    if (P && P->terminator() && P->terminator()->opcode() == Opcode::br_if &&
        P->terminator()->operand(0) == cond) {
        if (is_true_edge) return 1;
        if (is_false_edge) return 0;
    }
    return -1;
}

std::unordered_set<const BasicBlock*> reachable_from(const BasicBlock* start) {
    std::unordered_set<const BasicBlock*> seen{start};
    std::vector<const BasicBlock*> work{start};
    while (!work.empty()) {
        const BasicBlock* bb = work.back();
        work.pop_back();
        for (const BasicBlock* s : bb->successors()) {
            if (s && seen.insert(s).second) work.push_back(s);
        }
    }
    return seen;
}

// Threading adds an edge into `dest` that skips B. A value B defines stays
// dominated where it is used unless the use is reachable from `dest`; those
// uses are allowed only inside dest's dominance region, where the caller
// routes the value through a new parameter of `dest`. Collects the values
// that need that parameter; false when some use cannot be repaired.
bool collect_escaping_defs(
    const Function& fn,
    const BasicBlock* B,
    const BasicBlock* dest,
    const DominatorTree& dom,
    std::vector<Value*>& escaping
) {
    std::unordered_set<const Value*> defs(B->params().begin(), B->params().end());
    for (const Instruction* inst : *B) {
        if (inst && inst->result()) defs.insert(inst->result());
    }
    const std::unordered_set<const BasicBlock*> reach = reachable_from(dest);
    std::unordered_set<const Value*> seen;
    bool ok = true;
    for (const BasicBlock* Z : fn.blocks()) {
        if (!Z || Z == B || !reach.count(Z)) continue;
        const bool in_region = Z == dest || dom.dominates(dest, Z);
        for (const Instruction* inst : *Z) {
            if (!inst) continue;
            for_each_use(*inst, [&](Value* v) {
                if (!defs.count(v)) return;
                if (!in_region) ok = false;
                else if (seen.insert(v).second) escaping.push_back(v);
            });
            if (!ok) return false;
        }
    }
    return true;
}

void replace_in_region(
    const std::vector<BasicBlock*>& blocks,
    const BasicBlock* dest,
    const DominatorTree& dom,
    const std::unordered_map<const Value*, Value*>& repl
) {
    auto sub = [&](Value*& v) {
        auto it = repl.find(v);
        if (it != repl.end()) v = it->second;
    };
    for (BasicBlock* Z : blocks) {
        if (!Z || (Z != dest && !dom.dominates(dest, Z))) continue;
        for (Instruction* inst : *Z) {
            if (!inst) continue;
            for_each_use_slot(*inst, sub);
        }
    }
}

// A back edge enters B from a block B dominates.
bool is_loop_header(const BasicBlock* B, const DominatorTree& dom) {
    for (const BasicBlock* pred : B->predecessors()) {
        if (pred && dom.dominates(B, pred)) return true;
    }
    return false;
}

void retarget_edge(Instruction* p_term, bool is_true_edge, bool is_false_edge, BranchTarget target) {
    if (p_term->opcode() == Opcode::br) {
        p_term->set_branch_target(std::move(target));
    } else if (is_true_edge) {
        p_term->set_true_target(std::move(target));
    } else if (is_false_edge) {
        p_term->set_false_target(std::move(target));
    }
}

bool try_thread_edge(
    Function& fn,
    BasicBlock* B,
    BasicBlock* P,
    const DominatorTree& dom
) {
    if (!B || !P || B == P) return false;

    // Skipping a loop header from outside the loop would make the loop
    // irreducible; from inside it is a back edge. Neither is threaded.
    if (dom.dominates(B, P) || is_loop_header(B, dom)) return false;

    Instruction* b_term = B->terminator();
    if (!b_term || b_term->opcode() != Opcode::br_if) return false;

    Value* cond = b_term->operand(0);
    const BranchTarget& tgt_true = b_term->true_target();
    const BranchTarget& tgt_false = b_term->false_target();
    if (!tgt_true.block || !tgt_false.block) return false;

    Instruction* p_term = P->terminator();
    if (!p_term) return false;

    std::vector<Value*> incoming_args;
    bool is_true_edge = false;
    bool is_false_edge = false;

    if (p_term->opcode() == Opcode::br && p_term->branch_target().block == B) {
        incoming_args = p_term->branch_target().args;
    } else if (p_term->opcode() == Opcode::br_if) {
        const bool m_true = (p_term->true_target().block == B);
        const bool m_false = (p_term->false_target().block == B);
        if (m_true && !m_false) {
            incoming_args = p_term->true_target().args;
            is_true_edge = true;
        } else if (m_false && !m_true) {
            incoming_args = p_term->false_target().args;
            is_false_edge = true;
        } else {
            return false;
        }
    } else {
        return false;
    }

    if (incoming_args.size() != B->param_count()) return false;

    const int known_val = evaluate_condition_along_edge(B, cond, incoming_args, P, is_true_edge, is_false_edge);
    if (known_val != 0 && known_val != 1) return false;

    const BranchTarget& chosen_tgt = (known_val == 1) ? tgt_true : tgt_false;
    BasicBlock* dest = chosen_tgt.block;
    if (!dest || dest == B) return false;

    // B's body is re-executed (or skipped) on the threaded path, so it must
    // be free of effects that observe or change the world.
    for (Instruction* inst : *B) {
        if (!inst || inst->is_terminator()) break;
        if (inst->has_side_effects() || inst->is_call()) return false;
    }

    std::vector<Value*> escaping;
    if (!collect_escaping_defs(fn, B, dest, dom, escaping)) return false;
    if (!escaping.empty()) {
        // The repair gives dest one parameter per escaping value, fed by
        // each incoming edge; that needs B to be dest's only way in so far.
        const BranchTarget& other_tgt = (known_val == 1) ? tgt_false : tgt_true;
        if (other_tgt.block == dest) return false;
        for (const BasicBlock* pred : dest->predecessors()) {
            if (pred != B) return false;
        }
    }

    bool args_use_b_insts = false;
    for (Value* a : chosen_tgt.args) {
        if (a && a->is_instruction() && a->defining_instruction()->parent() == B) {
            args_use_b_insts = true;
            break;
        }
    }

    if (!args_use_b_insts && escaping.empty()) {
        // The successor's arguments are B's parameters or outside values:
        // redirect P straight to it.
        std::vector<Value*> redirected_args;
        redirected_args.reserve(chosen_tgt.args.size());
        for (Value* a : chosen_tgt.args) {
            if (a && a->is_block_param() && a->defining_block() == B) {
                redirected_args.push_back(incoming_args[a->param_index()]);
            } else {
                redirected_args.push_back(a);
            }
        }
        retarget_edge(p_term, is_true_edge, is_false_edge, BranchTarget(dest, std::move(redirected_args)));
        return true;
    }

    // The arguments are computed in B, or B's values live on past dest:
    // duplicate B's body into a block that serves only the edge from P and
    // jumps to the known successor.
    const std::vector<BasicBlock*> blocks_before = fn.blocks();
    BasicBlock* thread_bb = ir::new_block(fn, std::string(B->name()) + "_thread_" + std::string(P->name()));

    ir::ValueMap thread_val_map;
    for (size_t p_i = 0; p_i < B->param_count(); ++p_i) {
        thread_val_map[B->param(p_i)] = incoming_args[p_i];
    }
    const ir::BlockMap no_blocks;
    for (Instruction* inst : *B) {
        if (!inst || inst->is_terminator()) break;
        thread_bb->append_instruction(ir::clone_instruction(fn, *inst, thread_val_map, no_blocks));
    }

    std::vector<Value*> thread_dest_args;
    thread_dest_args.reserve(chosen_tgt.args.size());
    for (Value* a : chosen_tgt.args) {
        auto it = thread_val_map.find(a);
        thread_dest_args.push_back(it != thread_val_map.end() ? it->second : a);
    }

    if (!escaping.empty()) {
        // dest now merges B's values with their copies from the threaded
        // path; uses in dest's region read the merged parameter instead.
        std::unordered_map<const Value*, Value*> repl;
        BranchTarget& b_edge = (known_val == 1) ? b_term->true_target() : b_term->false_target();
        for (Value* v : escaping) {
            repl[v] = ir::new_block_param(fn, dest, v->type());
            b_edge.args.push_back(v);
            auto it = thread_val_map.find(v);
            thread_dest_args.push_back(it != thread_val_map.end() ? it->second : v);
        }
        replace_in_region(blocks_before, dest, dom, repl);
    }

    Instruction* br_dest = fn.parent()
        ? fn.parent()->arena().make<Instruction>(Opcode::br, Type::void_type())
        : new Instruction(Opcode::br, Type::void_type());
    br_dest->set_branch_target(BranchTarget(dest, std::move(thread_dest_args)));
    thread_bb->append_instruction(br_dest);

    retarget_edge(p_term, is_true_edge, is_false_edge, BranchTarget(thread_bb, {}));
    return true;
}

// BRANCH TAIL DUPLICATION. A join whose only work is a test of what came in
// — `merge(%c): %n = eq %c, 0; br_if %n, T, F` — costs every predecessor
// that computed %c as a comparison a setcc, a zero extension, the jump and a
// test: the comparison's flags are gone by the time the join tests its
// value. Copying the join's few instructions and its branch into such a
// predecessor P (one that ends in `br merge(...)`) puts the comparison and
// the test in one block, where instruction selection branches on the flags
// (x64_isel_branch.cpp). The join stays for its other predecessors.
//
// The copy computes what the join computed, from P's arguments in place of
// its parameters, so only pure instructions are copied, and only when no
// value the join defines is read outside it (nothing past the join needs a
// merge of the copies). The new edges P -> T and P -> F must not enter a
// loop from inside (T or F dominating P), and the join must not be a loop
// header; the join itself must not dominate P.
constexpr size_t kMaxDuplicated = 4;

// Whether `cond`, followed through B's 0/1 glue (and / eq / ne / xor with a
// constant, zero extension, truncation), is a parameter of B whose argument
// from P is a comparison: the case where the copy fuses.
bool condition_is_incoming_compare(const BasicBlock* B, const Value* cond, const std::vector<Value*>& args) {
    for (int depth = 0; depth < 6 && cond; ++depth) {
        if (cond->is_block_param() && cond->defining_block() == B) {
            const uint32_t idx = cond->param_index();
            const Value* a = idx < args.size() ? args[idx] : nullptr;
            const Instruction* d = a && a->is_instruction() ? a->defining_instruction() : nullptr;
            return d && is_comparison(d->opcode());
        }
        const Instruction* def = cond->is_instruction() ? cond->defining_instruction() : nullptr;
        if (!def || def->parent() != B) return false;
        switch (def->opcode()) {
            case Opcode::and_: case Opcode::eq: case Opcode::ne: case Opcode::xor_:
                if (def->operand_count() != 2) return false;
                if (const_int(def->operand(1))) cond = def->operand(0);
                else if (const_int(def->operand(0))) cond = def->operand(1);
                else return false;
                break;
            case Opcode::zext_i64: case Opcode::trunc_i32:
                cond = def->operand(0);
                break;
            default:
                return false;
        }
    }
    return false;
}

// Values read outside the block that defines them (a block parameter's
// block, an instruction's parent).
std::unordered_set<const Value*> values_used_outside(const Function& fn) {
    std::unordered_set<const Value*> out;
    for (const BasicBlock* Z : fn.blocks()) {
        if (!Z) continue;
        for (const Instruction* inst : *Z) {
            if (!inst) continue;
            for_each_use(*inst, [&](Value* v) {
                const BasicBlock* home = v->is_block_param() ? v->defining_block()
                                       : v->is_instruction() && v->defining_instruction()
                                           ? v->defining_instruction()->parent() : nullptr;
                if (home && home != Z) out.insert(v);
            });
        }
    }
    return out;
}

bool try_duplicate_branch_block(Function& fn, BasicBlock* B, BasicBlock* P, const DominatorTree& dom,
                                std::optional<std::unordered_set<const Value*>>& outside) {
    if (!B || !P || B == P || B == fn.entry_block()) return false;
    Instruction* p_term = P->terminator();
    if (!p_term || p_term->opcode() != Opcode::br || p_term->branch_target().block != B) return false;
    Instruction* b_term = B->terminator();
    if (!b_term || b_term->opcode() != Opcode::br_if) return false;
    BasicBlock* T = b_term->true_target().block;
    BasicBlock* F = b_term->false_target().block;
    if (!T || !F || T == B || F == B || T == F) return false;
    const std::vector<Value*>& incoming = p_term->branch_target().args;
    if (incoming.size() != B->param_count()) return false;
    if (!condition_is_incoming_compare(B, b_term->operand(0), incoming)) return false;
    if (dom.dominates(B, P) || is_loop_header(B, dom) || dom.dominates(T, P) || dom.dominates(F, P)) return false;

    size_t n = 0;
    for (Instruction* inst : *B) {
        if (!inst || inst->is_terminator()) break;
        if (inst->has_side_effects() || inst->is_call() || ++n > kMaxDuplicated) return false;
    }
    if (!outside) outside = values_used_outside(fn);
    for (const Value* p : B->params()) {
        if (outside->count(p)) return false;
    }
    for (const Instruction* inst : *B) {
        if (inst && inst->result() && outside->count(inst->result())) return false;
    }

    ir::ValueMap vals;
    for (size_t i = 0; i < B->param_count(); ++i) vals[B->param(i)] = incoming[i];
    const ir::BlockMap no_blocks;
    for (Instruction* inst : *B) {
        if (!inst || inst->is_terminator()) break;
        P->insert_before(ir::clone_instruction(fn, *inst, vals, no_blocks), p_term);
    }
    P->insert_before(ir::clone_instruction(fn, *b_term, vals, no_blocks), p_term);
    P->remove_instruction(p_term);
    return true;
}

} // namespace

bool run_jump_threading(Function& fn, const JumpThreadingOptions& opts, JumpThreadingStats* stats) {
    if (fn.name().starts_with("__wrapper_")) return false;
    bool changed = false;
    // Each sweep visits every block once and takes every edge it can; a
    // change recomputes the predecessors and dominators the next decision
    // reads, so a function with many joins is not limited to a few edges.
    for (size_t iter = 0; iter < opts.max_iterations; ++iter) {
        fn.rebuild_cfg_predecessors();
        auto dom = std::make_unique<DominatorTree>(fn);
        std::optional<std::unordered_set<const Value*>> outside;

        bool iter_changed = false;
        std::vector<BasicBlock*> blocks = fn.blocks();

        for (BasicBlock* B : blocks) {
            bool progress = true;
            while (progress) {
                progress = false;
                if (!B || B == fn.entry_block() || B->predecessors().empty()) break;
                if (B->instruction_count() > opts.max_block_size) break;
                Instruction* term = B->terminator();
                if (!term || term->opcode() != Opcode::br_if) break;

                std::vector<BasicBlock*> preds = B->predecessors();
                for (BasicBlock* P : preds) {
                    const bool threaded = try_thread_edge(fn, B, P, *dom);
                    if (!threaded && !try_duplicate_branch_block(fn, B, P, *dom, outside)) continue;
                    iter_changed = changed = progress = true;
                    if (stats && threaded) stats->edges_threaded++;
                    // The CFG changed: what the next decision reads is
                    // recomputed.
                    fn.rebuild_cfg_predecessors();
                    dom = std::make_unique<DominatorTree>(fn);
                    outside.reset();
                    break;
                }
            }
        }

        if (!iter_changed) break;
    }

    return changed;
}

bool run_jump_threading(Function& fn) {
    return run_jump_threading(fn, JumpThreadingOptions(), nullptr);
}

bool jump_thread_module(Module& mod, const JumpThreadingOptions& opts, JumpThreadingStats* stats) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= run_jump_threading(*fn, opts, stats);
        }
    }
    return changed;
}

bool jump_thread_module(Module& mod) {
    return jump_thread_module(mod, JumpThreadingOptions(), nullptr);
}

} // namespace brass
