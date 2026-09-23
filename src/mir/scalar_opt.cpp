#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/runtime_symbols.hpp>
#include <brass/mir/uses.hpp>
#include "int_fold.hpp"
#include <cstring>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

bool get_const_int(const Value* val, int64_t& out_val) {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i32) {
        out_val = static_cast<int64_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == Opcode::iconst_i64) {
        out_val = def->imm_i64();
        return true;
    }
    return false;
}

// Pure and safe to execute speculatively: LICM hoists these out of loops
// and CSE merges them.
bool is_pure_instruction(const Instruction* inst) {
    if (!inst) return false;
    // Pure, but tied to its block: hoisting or merging a derived gcref could
    // keep it live across a GC point (gc_refs.hpp).
    if (is_derived_gcref(inst->result())) return false;
    Opcode op = inst->opcode();
    switch (op) {
        case Opcode::iconst_i32: case Opcode::iconst_i64: case Opcode::fconst_f64:
        case Opcode::sext_i64: case Opcode::zext_i64: case Opcode::trunc_i32:
        case Opcode::fptosi_i32: case Opcode::fptosi_i64: case Opcode::sitofp_f64_i32: case Opcode::sitofp_f64_i64:
        case Opcode::bitcast_i64_f64: case Opcode::bitcast_f64_i64:
        case Opcode::add: case Opcode::sub: case Opcode::mul: case Opcode::neg:
        case Opcode::and_: case Opcode::or_: case Opcode::xor_:
        case Opcode::shl: case Opcode::lshr: case Opcode::ashr: case Opcode::not_:
        case Opcode::clz: case Opcode::ctz: case Opcode::popcnt:
        case Opcode::eq: case Opcode::ne: case Opcode::slt: case Opcode::ult:
        case Opcode::sle: case Opcode::ule: case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge:
        case Opcode::select: return true;
        case Opcode::call: return callee_has_role(*inst, SymbolRole::Pure);
        case Opcode::sdiv: case Opcode::udiv: case Opcode::smod: case Opcode::umod: {
            // LICM executes pure instructions speculatively, so a division is
            // pure only when its constant divisor rules out every trap.
            if (inst->operand_count() < 2 || !inst->operand(1)) return false;
            int64_t divisor = 0;
            if (!get_const_int(inst->operand(1), divisor)) return false;
            const unsigned width = int_fold::width_of(inst->type());
            return width != 0 && !int_fold::division_may_trap(op, width, divisor);
        }
        default: return false;
    }
}

void replace_and_remove(Function& fn, BasicBlock* bb, Instruction* inst, Value* replacement) {
    replace_all_uses(fn, inst->result(), replacement);
    bb->remove_instruction(inst);
}

// The key two pure instructions share exactly when they compute the same
// value: opcode, type, immediate, callee and every operand.
struct ExprKey {
    Opcode op = Opcode::iconst_i64;
    Type type;
    const Value* op0 = nullptr;
    const Value* op1 = nullptr;
    const Value* op2 = nullptr;
    uint64_t imm_bits = 0;
    std::string_view symbol;

    bool operator==(const ExprKey& o) const noexcept {
        return op == o.op && type == o.type && op0 == o.op0 && op1 == o.op1 && op2 == o.op2 &&
               imm_bits == o.imm_bits && symbol == o.symbol;
    }
};

struct ExprKeyHash {
    size_t operator()(const ExprKey& k) const noexcept {
        size_t h = static_cast<size_t>(k.op);
        h = h * 31 + std::hash<const void*>()(k.op0);
        h = h * 31 + std::hash<const void*>()(k.op1);
        h = h * 31 + std::hash<const void*>()(k.op2);
        h = h * 31 + static_cast<size_t>(k.imm_bits);
        h = h * 31 + std::hash<std::string_view>()(k.symbol);
        return h;
    }
};

bool is_commutative(Opcode op) {
    return op == Opcode::add || op == Opcode::mul || op == Opcode::and_ ||
           op == Opcode::or_ || op == Opcode::xor_ || op == Opcode::eq || op == Opcode::ne;
}

// Builds the CSE key of `inst`; nullopt when the key could not tell it apart
// from a different computation (more operands than the key holds).
std::optional<ExprKey> expr_key(const Instruction* inst) {
    if (inst->operand_count() > 3) return std::nullopt;
    ExprKey key;
    key.op = inst->opcode();
    key.type = inst->type();
    if (inst->opcode() == Opcode::fconst_f64) {
        const double fval = inst->imm_f64();
        std::memcpy(&key.imm_bits, &fval, sizeof(double));
    } else {
        key.imm_bits = static_cast<uint64_t>(inst->imm_i64());
    }
    if (is_call(inst->opcode())) key.symbol = inst->symbol();
    if (inst->operand_count() >= 1) key.op0 = inst->operand(0);
    if (inst->operand_count() >= 2) key.op1 = inst->operand(1);
    if (inst->operand_count() >= 3) key.op2 = inst->operand(2);
    if (is_commutative(key.op) && std::less<const Value*>()(key.op1, key.op0)) std::swap(key.op0, key.op1);
    return key;
}

bool licm_loop(Function& fn, LoopInfo& loop) {
    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader) return false;
    Instruction* ph_term = preheader->terminator();
    if (!ph_term) return false;

    bool any_hoisted = false;
    bool hoisted_in_round = true;
    while (hoisted_in_round) {
        hoisted_in_round = false;
        for (BasicBlock* bb : loop.blocks()) {
            if (!bb || bb == preheader) continue;
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (is_pure_instruction(cur) && cur->produces_value()) {
                    bool all_operands_invariant = true;
                    for (Value* op : cur->operands()) {
                        if (!op) continue;
                        if (op->is_block_param()) {
                            if (loop.contains(op->defining_block())) {
                                all_operands_invariant = false;
                                break;
                            }
                        } else if (op->is_instruction()) {
                            Instruction* def_inst = op->defining_instruction();
                            if (def_inst && loop.contains(def_inst->parent()) && def_inst->parent() != preheader) {
                                all_operands_invariant = false;
                                break;
                            }
                        }
                    }
                    if (all_operands_invariant) {
                        bb->remove_instruction(cur);
                        preheader->insert_before(cur, ph_term);
                        hoisted_in_round = true;
                        any_hoisted = true;
                    }
                }
                cur = next;
            }
        }
    }
    return any_hoisted;
}

} // namespace

bool fold_constants(Function& fn) {
    if (!fn.parent()) return false;
    bool changed = false;
    Builder b(*fn.parent());
    b.set_function(&fn);

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        Instruction* cur = bb->head();
        while (cur) {
            Instruction* next = cur->next();
            if (!is_pure_instruction(cur) || !cur->produces_value()) {
                cur = next;
                continue;
            }

            Opcode op = cur->opcode();
            Value* replacement = nullptr;
            if (cur->operand_count() == 2) {
                Value* op0 = cur->operand(0);
                Value* op1 = cur->operand(1);
                int64_t c0 = 0, c1 = 0;
                bool has_c0 = get_const_int(op0, c0);
                bool has_c1 = get_const_int(op1, c1);
                Type res_type = cur->type();
                // Identities below only hold for integer arithmetic, and only a
                // result of the same type as the replacement may stand in for it.
                const unsigned width = op0 ? int_fold::width_of(op0->type()) : 0;
                const bool int_result = res_type == Type::i32() || res_type == Type::i64();
                auto same_type = [&](const Value* v) { return v && v->type() == res_type; };
                auto build_int = [&](int64_t v) -> Value* {
                    b.position_before(cur);
                    return (res_type == Type::i32()) ? b.build_iconst_i32(static_cast<int32_t>(v)) : b.build_iconst_i64(v);
                };

                if (has_c0 && has_c1) {
                    if (int_result && width != 0) {
                        if (auto folded = int_fold::binary(op, width, c0, c1)) replacement = build_int(*folded);
                    }
                } else if (has_c1 && width != 0) {
                    if ((op == Opcode::add || op == Opcode::sub || op == Opcode::or_ || op == Opcode::xor_ ||
                         op == Opcode::shl || op == Opcode::ashr || op == Opcode::lshr) && c1 == 0) {
                        if (same_type(op0)) replacement = op0;
                    } else if (op == Opcode::mul && c1 == 1) {
                        if (same_type(op0)) replacement = op0;
                    } else if (op == Opcode::mul && c1 == 0 && int_result) {
                        replacement = build_int(0);
                    }
                } else if (has_c0 && op1 && int_fold::width_of(op1->type()) != 0) {
                    if ((op == Opcode::add || op == Opcode::or_ || op == Opcode::xor_) && c0 == 0) {
                        if (same_type(op1)) replacement = op1;
                    } else if (op == Opcode::mul && c0 == 1) {
                        if (same_type(op1)) replacement = op1;
                    } else if (op == Opcode::mul && c0 == 0 && int_result) {
                        replacement = build_int(0);
                    }
                } else if (op0 == op1 && (op == Opcode::sub || op == Opcode::xor_)) {
                    if (int_result && width != 0) {
                        replacement = build_int(0);
                    } else if (op == Opcode::sub && res_type == Type::f64() && fn.allow_fp_reassociation()) {
                        b.position_before(cur);
                        replacement = b.build_fconst_f64(0.0);
                    }
                }
            } else if (cur->operand_count() == 1 && !is_call(op)) {
                Value* op0 = cur->operand(0);
                int64_t c0 = 0;
                if (get_const_int(op0, c0)) {
                    std::optional<int64_t> folded;
                    if (op == Opcode::sext_i64 || op == Opcode::zext_i64 || op == Opcode::trunc_i32) {
                        folded = int_fold::convert(op, op0->type(), c0);
                    } else if (cur->type() == Type::i32() || cur->type() == Type::i64()) {
                        folded = int_fold::unary(op, int_fold::width_of(cur->type()), c0);
                    }
                    if (folded) {
                        b.position_before(cur);
                        replacement = (cur->type() == Type::i32()) ? b.build_iconst_i32(static_cast<int32_t>(*folded))
                                                                   : b.build_iconst_i64(*folded);
                    }
                }
            } else if (cur->operand_count() == 3 && op == Opcode::select) {
                Value* cond = cur->operand(0);
                Value* true_v = cur->operand(1);
                Value* false_v = cur->operand(2);
                int64_t cond_c = 0;
                if (get_const_int(cond, cond_c)) {
                    replacement = (cond_c != 0) ? true_v : false_v;
                } else if (true_v == false_v) {
                    replacement = true_v;
                }
            }
            if (replacement) {
                replace_and_remove(fn, bb, cur, replacement);
                changed = true;
            }
            cur = next;
        }
    }
    return changed;
}

bool dominator_cse(Function& fn) {
    if (!fn.entry_block()) return false;
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    bool changed = false;
    std::unordered_map<ExprKey, Value*, ExprKeyHash> expr_map;

    // An explicit stack: a deep dominator tree must not overflow the C++ one.
    struct Frame {
        BasicBlock* bb;
        std::vector<ExprKey> added;
        bool entered = false;
    };
    std::vector<Frame> stack;
    stack.push_back({fn.entry_block(), {}, false});
    while (!stack.empty()) {
        if (stack.back().entered) {
            for (const ExprKey& key : stack.back().added) expr_map.erase(key);
            stack.pop_back();
            continue;
        }
        stack.back().entered = true;
        BasicBlock* bb = stack.back().bb;
        std::vector<ExprKey> added;
        Instruction* cur = bb->head();
        while (cur) {
            Instruction* next = cur->next();
            if (is_pure_instruction(cur) && cur->produces_value()) {
                if (std::optional<ExprKey> key = expr_key(cur)) {
                    auto it = expr_map.find(*key);
                    if (it != expr_map.end()) {
                        replace_and_remove(fn, bb, cur, it->second);
                        changed = true;
                    } else {
                        expr_map.emplace(*key, cur->result());
                        added.push_back(*key);
                    }
                }
            }
            cur = next;
        }
        stack.back().added = std::move(added);
        const auto& children = dom.children(bb);
        for (auto it = children.rbegin(); it != children.rend(); ++it) {
            if (*it) stack.push_back({const_cast<BasicBlock*>(*it), {}, false});
        }
    }
    return changed;
}

bool eliminate_dead_code(Function& fn) {
    bool changed = false;
    bool progress = true;
    while (progress) {
        progress = false;
        auto use_counts = compute_use_counts(fn);
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (!cur->has_side_effects() && cur->produces_value()) {
                    Value* res = cur->result();
                    if (!res || use_counts[res] == 0) {
                        bb->remove_instruction(cur);
                        progress = true;
                        changed = true;
                    }
                }
                cur = next;
            }
        }

        if (progress) use_counts = compute_use_counts(fn);
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb || bb == fn.entry_block()) continue;
            size_t p_i = 0;
            while (p_i < bb->param_count()) {
                Value* p = bb->param(p_i);
                if (p && use_counts[p] == 0) {
                    remove_block_param(*bb, p_i);
                    progress = true;
                    changed = true;
                    use_counts = compute_use_counts(fn);
                } else {
                    ++p_i;
                }
            }
        }
    }
    return changed;
}

bool eliminate_dead_induction_cycles(Function& fn) {
    bool changed = false;
    bool progress = true;
    while (progress) {
        progress = false;
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb || bb == fn.entry_block()) continue;
            size_t p_i = 0;
            while (p_i < bb->param_count()) {
                Value* p = bb->param(p_i);
                if (!p) { ++p_i; continue; }

                // The values reachable from `p` through add/sub; the cycle is
                // dead when they reach nothing else and flow back only into
                // `p` itself.
                std::unordered_set<Instruction*> cycle_insts;
                std::vector<Value*> worklist = {p};
                std::unordered_set<Value*> visited = {p};
                bool is_pure_cycle = true;
                while (!worklist.empty() && is_pure_cycle) {
                    Value* cur_v = worklist.back();
                    worklist.pop_back();
                    for (BasicBlock* u_bb : fn.blocks()) {
                        if (!u_bb || !is_pure_cycle) continue;
                        for (Instruction* inst : *u_bb) {
                            if (!inst) continue;
                            for (Value* sv : inst->state_map()) {
                                if (sv == cur_v) is_pure_cycle = false;
                            }
                            for (Value* op : inst->operands()) {
                                if (op != cur_v) continue;
                                if (inst->has_side_effects() || !inst->produces_value() ||
                                    (inst->opcode() != Opcode::add && inst->opcode() != Opcode::sub)) {
                                    is_pure_cycle = false;
                                    break;
                                }
                                cycle_insts.insert(inst);
                                Value* res = inst->result();
                                if (res && visited.insert(res).second) worklist.push_back(res);
                            }
                            for_each_edge(*inst, [&](const BranchTarget& bt) {
                                for (size_t arg_idx = 0; arg_idx < bt.args.size(); ++arg_idx) {
                                    if (bt.args[arg_idx] == cur_v && !(bt.block == bb && arg_idx == p_i)) {
                                        is_pure_cycle = false;
                                    }
                                }
                            });
                            if (!is_pure_cycle) break;
                        }
                    }
                }

                if (is_pure_cycle && !cycle_insts.empty()) {
                    for (Instruction* inst : cycle_insts) {
                        if (inst && inst->parent()) inst->parent()->remove_instruction(inst);
                    }
                    remove_block_param(*bb, p_i);
                    progress = true;
                    changed = true;
                } else {
                    ++p_i;
                }
            }
        }
    }
    return changed;
}

bool scalar_cleanup(Function& fn) {
    bool changed = fold_constants(fn);
    changed |= dominator_cse(fn);
    changed |= eliminate_dead_code(fn);
    return changed;
}

bool hoist_loop_invariants(Function& fn) {
    if (!fn.entry_block()) return false;
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis loops(fn, dom);
    bool changed = false;
    for (LoopInfo* loop : loops.post_order_loops()) {
        if (loop) changed |= licm_loop(fn, *loop);
    }
    if (changed) fn.rebuild_cfg_predecessors();
    return changed;
}

bool loop_cleanup(Function& fn, const LoopCleanupOptions& options) {
    bool any_changed = false;
    auto cleanups = [&] {
        bool c = fold_constants(fn);
        c |= dominator_cse(fn);
        c |= eliminate_dead_code(fn);
        c |= eliminate_dead_induction_cycles(fn);
        c |= eliminate_dead_code(fn);
        return c;
    };
    for (size_t iter = 0; iter < options.max_iterations; ++iter) {
        bool iter_changed = false;
        if (options.fold_and_dce) iter_changed |= cleanups();
        if (options.licm) iter_changed |= hoist_loop_invariants(fn);
        if (options.licm && options.fold_and_dce) iter_changed |= cleanups();
        fn.rebuild_cfg_predecessors();
        any_changed |= iter_changed;
        // Without LICM a second round would find nothing the first missed.
        if (!iter_changed || !options.licm) break;
    }
    return any_changed;
}

} // namespace brass
