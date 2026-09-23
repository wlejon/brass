#include <brass/mir/jump_threading.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include "int_fold.hpp"
#include "ir_clone.hpp"
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

} // namespace

bool run_jump_threading(Function& fn, const JumpThreadingOptions& opts, JumpThreadingStats* stats) {
    if (fn.name().starts_with("__wrapper_")) return false;
    bool changed = false;
    for (size_t iter = 0; iter < opts.max_iterations; ++iter) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);

        bool iter_changed = false;
        std::vector<BasicBlock*> blocks = fn.blocks();

        for (BasicBlock* B : blocks) {
            if (!B || B == fn.entry_block() || B->predecessors().empty()) continue;
            if (B->instruction_count() > opts.max_block_size) continue;

            Instruction* term = B->terminator();
            if (!term || term->opcode() != Opcode::br_if) continue;

            std::vector<BasicBlock*> preds = B->predecessors();
            for (BasicBlock* P : preds) {
                if (try_thread_edge(fn, B, P, dom)) {
                    iter_changed = true;
                    changed = true;
                    if (stats) {
                        stats->edges_threaded++;
                    }
                    break; // The CFG changed: dominators must be recomputed.
                }
            }

            if (iter_changed) break;
        }

        if (iter_changed) {
            fn.rebuild_cfg_predecessors();
        } else {
            break;
        }
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
