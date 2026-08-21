#include <brass/mir/loop_opt.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/loop_unroll.hpp>
#include <brass/mir/select_opt.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <functional>
#include <cstring>

namespace brass {

namespace {

struct BasicIV {
    Value* param = nullptr;
    size_t param_idx = 0;
    Value* init = nullptr;
    Value* step = nullptr;
    Instruction* step_inst = nullptr;
    bool is_sub = false;
};

struct PairHash {
    size_t operator()(const std::pair<const Value*, uint8_t>& p) const noexcept {
        return std::hash<const void*>()(p.first) ^ (static_cast<size_t>(p.second) << 3);
    }
};

bool is_pure_instruction(const Instruction* inst) {
    if (!inst) return false;
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
        case Opcode::select:
            return true;
        case Opcode::sdiv: case Opcode::udiv: case Opcode::smod: case Opcode::umod: {
            if (inst->operand_count() < 2 || !inst->operand(1)) return false;
            const Value* denom = inst->operand(1);
            if (denom->is_instruction()) {
                const Instruction* ddef = denom->defining_instruction();
                if (ddef && (ddef->opcode() == Opcode::iconst_i32 || ddef->opcode() == Opcode::iconst_i64)) {
                    return ddef->imm_i64() != 0;
                }
            }
            return false;
        }
        default: return false;
    }
}

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

Value* build_smart_const_i64(Builder& b, int64_t val) {
    return b.build_iconst_i64(val);
}

Value* build_smart_mul(Builder& b, Value* lhs, Value* rhs) {
    int64_t c0, c1;
    bool has_c0 = get_const_int(lhs, c0);
    bool has_c1 = get_const_int(rhs, c1);
    if (has_c0 && has_c1) return build_smart_const_i64(b, c0 * c1);
    if (has_c0) {
        if (c0 == 0) return build_smart_const_i64(b, 0);
        if (c0 == 1) return rhs;
    }
    if (has_c1) {
        if (c1 == 0) return build_smart_const_i64(b, 0);
        if (c1 == 1) return lhs;
    }
    return b.build_mul(lhs, rhs);
}

Value* build_smart_add(Builder& b, Value* lhs, Value* rhs) {
    int64_t c0, c1;
    bool has_c0 = get_const_int(lhs, c0);
    bool has_c1 = get_const_int(rhs, c1);
    if (has_c0 && has_c1) return build_smart_const_i64(b, c0 + c1);
    if (has_c0 && c0 == 0) return rhs;
    if (has_c1 && c1 == 0) return lhs;
    return b.build_add(lhs, rhs);
}

void replace_all_uses(Function& fn, Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == old_val) inst->set_operand(i, new_val);
            }
            for (size_t i = 0; i < inst->branch_target().args.size(); ++i) {
                if (inst->branch_target().args[i] == old_val) inst->branch_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->true_target().args.size(); ++i) {
                if (inst->true_target().args[i] == old_val) inst->true_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->false_target().args.size(); ++i) {
                if (inst->false_target().args[i] == old_val) inst->false_target().args[i] = new_val;
            }
            for (size_t i = 0; i < inst->default_target().args.size(); ++i) {
                if (inst->default_target().args[i] == old_val) inst->default_target().args[i] = new_val;
            }
            for (auto& sc : inst->switch_cases()) {
                for (size_t i = 0; i < sc.target.args.size(); ++i) {
                    if (sc.target.args[i] == old_val) sc.target.args[i] = new_val;
                }
            }
            for (size_t i = 0; i < inst->state_map().size(); ++i) {
                if (inst->state_map()[i] == old_val) inst->state_map()[i] = new_val;
            }
        }
    }
}

std::unordered_map<const Value*, uint32_t> compute_use_counts(const Function& fn) {
    std::unordered_map<const Value*, uint32_t> counts;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            for (const Value* op : inst->operands()) if (op) counts[op]++;
            for (const Value* arg : inst->branch_target().args) if (arg) counts[arg]++;
            for (const Value* arg : inst->true_target().args) if (arg) counts[arg]++;
            for (const Value* arg : inst->false_target().args) if (arg) counts[arg]++;
            for (const Value* arg : inst->default_target().args) if (arg) counts[arg]++;
            for (const auto& sc : inst->switch_cases()) {
                for (const Value* arg : sc.target.args) if (arg) counts[arg]++;
            }
            for (const Value* sv : inst->state_map()) if (sv) counts[sv]++;
        }
    }
    return counts;
}

bool constant_folding_pass(Function& fn) {
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
            if (cur->operand_count() == 2) {
                Value* op0 = cur->operand(0);
                Value* op1 = cur->operand(1);
                int64_t c0, c1;
                bool has_c0 = get_const_int(op0, c0);
                bool has_c1 = get_const_int(op1, c1);
                Type res_type = cur->type();
                Value* replacement = nullptr;

                if (has_c0 && has_c1) {
                    int64_t result = 0;
                    bool can_fold = true;
                    switch (op) {
                        case Opcode::add: result = c0 + c1; break;
                        case Opcode::sub: result = c0 - c1; break;
                        case Opcode::mul: result = c0 * c1; break;
                        case Opcode::and_: result = c0 & c1; break;
                        case Opcode::or_: result = c0 | c1; break;
                        case Opcode::xor_: result = c0 ^ c1; break;
                        case Opcode::shl: result = c0 << (c1 & 63); break;
                        case Opcode::lshr: result = static_cast<int64_t>(static_cast<uint64_t>(c0) >> (c1 & 63)); break;
                        case Opcode::ashr: result = c0 >> (c1 & 63); break;
                        case Opcode::slt: result = (c0 < c1) ? 1 : 0; break;
                        case Opcode::sle: result = (c0 <= c1) ? 1 : 0; break;
                        case Opcode::sgt: result = (c0 > c1) ? 1 : 0; break;
                        case Opcode::sge: result = (c0 >= c1) ? 1 : 0; break;
                        case Opcode::ult: result = (static_cast<uint64_t>(c0) < static_cast<uint64_t>(c1)) ? 1 : 0; break;
                        case Opcode::ule: result = (static_cast<uint64_t>(c0) <= static_cast<uint64_t>(c1)) ? 1 : 0; break;
                        case Opcode::ugt: result = (static_cast<uint64_t>(c0) > static_cast<uint64_t>(c1)) ? 1 : 0; break;
                        case Opcode::uge: result = (static_cast<uint64_t>(c0) >= static_cast<uint64_t>(c1)) ? 1 : 0; break;
                        case Opcode::eq: result = (c0 == c1) ? 1 : 0; break;
                        case Opcode::ne: result = (c0 != c1) ? 1 : 0; break;
                        case Opcode::sdiv: if (c1 != 0) result = c0 / c1; else can_fold = false; break;
                        case Opcode::udiv: if (c1 != 0) result = static_cast<int64_t>(static_cast<uint64_t>(c0) / static_cast<uint64_t>(c1)); else can_fold = false; break;
                        case Opcode::smod: if (c1 != 0) result = c0 % c1; else can_fold = false; break;
                        case Opcode::umod: if (c1 != 0) result = static_cast<int64_t>(static_cast<uint64_t>(c0) % static_cast<uint64_t>(c1)); else can_fold = false; break;
                        default: can_fold = false; break;
                    }
                    if (can_fold) {
                        b.position_before(cur);
                        replacement = (res_type == Type::i32()) ? b.build_iconst_i32(static_cast<int32_t>(result)) : b.build_iconst_i64(result);
                    }
                } else if (has_c1) {
                    if (op == Opcode::add && c1 == 0) replacement = op0;
                    else if (op == Opcode::sub && c1 == 0) replacement = op0;
                    else if (op == Opcode::mul && c1 == 1) replacement = op0;
                    else if (op == Opcode::mul && c1 == 0) {
                        b.position_before(cur);
                        replacement = (res_type == Type::i32()) ? b.build_iconst_i32(0) : b.build_iconst_i64(0);
                    } else if ((op == Opcode::shl || op == Opcode::ashr || op == Opcode::lshr) && c1 == 0) replacement = op0;
                    else if (op == Opcode::or_ && c1 == 0) replacement = op0;
                    else if (op == Opcode::xor_ && c1 == 0) replacement = op0;
                } else if (has_c0) {
                    if (op == Opcode::add && c0 == 0) replacement = op1;
                    else if (op == Opcode::mul && c0 == 1) replacement = op1;
                    else if (op == Opcode::mul && c0 == 0) {
                        b.position_before(cur);
                        replacement = (res_type == Type::i32()) ? b.build_iconst_i32(0) : b.build_iconst_i64(0);
                    } else if (op == Opcode::or_ && c0 == 0) replacement = op1;
                    else if (op == Opcode::xor_ && c0 == 0) replacement = op1;
                } else if (op0 == op1 && (op == Opcode::sub || op == Opcode::xor_)) {
                    b.position_before(cur);
                    replacement = (res_type == Type::i32()) ? b.build_iconst_i32(0) : b.build_iconst_i64(0);
                }

                if (replacement) {
                    replace_all_uses(fn, cur->result(), replacement);
                    bb->remove_instruction(cur);
                    changed = true;
                }
            } else if (cur->operand_count() == 1) {
                Value* op0 = cur->operand(0);
                int64_t c0;
                if (get_const_int(op0, c0)) {
                    Value* replacement = nullptr;
                    b.position_before(cur);
                    if (op == Opcode::sext_i64) replacement = b.build_iconst_i64(static_cast<int64_t>(static_cast<int32_t>(c0)));
                    else if (op == Opcode::zext_i64) replacement = b.build_iconst_i64(static_cast<int64_t>(static_cast<uint64_t>(static_cast<uint32_t>(c0))));
                    else if (op == Opcode::trunc_i32) replacement = b.build_iconst_i32(static_cast<int32_t>(c0));

                    if (replacement) {
                        replace_all_uses(fn, cur->result(), replacement);
                        bb->remove_instruction(cur);
                        changed = true;
                    }
                }
            } else if (cur->operand_count() == 3 && cur->opcode() == Opcode::select) {
                Value* cond = cur->operand(0);
                Value* true_v = cur->operand(1);
                Value* false_v = cur->operand(2);
                Value* replacement = nullptr;
                int64_t cond_c;
                if (get_const_int(cond, cond_c)) {
                    replacement = (cond_c != 0) ? true_v : false_v;
                } else if (true_v == false_v) {
                    replacement = true_v;
                }
                if (replacement) {
                    replace_all_uses(fn, cur->result(), replacement);
                    bb->remove_instruction(cur);
                    changed = true;
                }
            }
            cur = next;
        }
    }
    return changed;
}

struct ExprKey {
    Opcode op;
    Type type;
    const Value* op0 = nullptr;
    const Value* op1 = nullptr;
    const Value* op2 = nullptr;
    uint64_t imm_bits = 0;

    bool operator==(const ExprKey& o) const noexcept {
        return op == o.op && type == o.type && op0 == o.op0 && op1 == o.op1 && op2 == o.op2 && imm_bits == o.imm_bits;
    }
};

struct ExprKeyHash {
    size_t operator()(const ExprKey& k) const noexcept {
        size_t h = static_cast<size_t>(k.op);
        h = h * 31 + std::hash<const void*>()(k.op0);
        h = h * 31 + std::hash<const void*>()(k.op1);
        h = h * 31 + std::hash<const void*>()(k.op2);
        h = h * 31 + static_cast<size_t>(k.imm_bits);
        return h;
    }
};

bool cse_pass(Function& fn, DominatorTree& dom) {
    bool changed = false;
    std::unordered_map<ExprKey, Value*, ExprKeyHash> expr_map;

    auto is_commutative = [](Opcode op) {
        return op == Opcode::add || op == Opcode::mul || op == Opcode::and_ ||
               op == Opcode::or_ || op == Opcode::xor_ || op == Opcode::eq || op == Opcode::ne;
    };

    std::function<void(BasicBlock*)> visit_block = [&](BasicBlock* bb) {
        if (!bb) return;
        std::vector<ExprKey> added_keys;

        Instruction* cur = bb->head();
        while (cur) {
            Instruction* next = cur->next();
            if (is_pure_instruction(cur) && cur->produces_value()) {
                ExprKey key;
                key.op = cur->opcode();
                key.type = cur->type();
                if (cur->opcode() == Opcode::fconst_f64) {
                    double fval = cur->imm_f64();
                    std::memcpy(&key.imm_bits, &fval, sizeof(double));
                } else {
                    key.imm_bits = static_cast<uint64_t>(cur->imm_i64());
                }
                if (cur->operand_count() >= 1) key.op0 = cur->operand(0);
                if (cur->operand_count() >= 2) key.op1 = cur->operand(1);
                if (cur->operand_count() >= 3) key.op2 = cur->operand(2);

                if (is_commutative(key.op) && key.op0 > key.op1) {
                    std::swap(key.op0, key.op1);
                }

                auto it = expr_map.find(key);
                if (it != expr_map.end()) {
                    replace_all_uses(fn, cur->result(), it->second);
                    bb->remove_instruction(cur);
                    changed = true;
                } else {
                    expr_map[key] = cur->result();
                    added_keys.push_back(key);
                }
            }
            cur = next;
        }

        for (const BasicBlock* child : dom.children(bb)) {
            if (child) visit_block(const_cast<BasicBlock*>(child));
        }

        for (const auto& key : added_keys) {
            expr_map.erase(key);
        }
    };

    if (fn.entry_block()) {
        visit_block(fn.entry_block());
    }

    return changed;
}

bool dead_code_elimination_pass(Function& fn) {
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
                    auto& params = bb->params();
                    params.erase(params.begin() + static_cast<std::ptrdiff_t>(p_i));
                    for (size_t k = p_i; k < params.size(); ++k) {
                        params[k]->set_block_param(bb, static_cast<uint32_t>(k));
                    }

                    for (BasicBlock* pred : bb->predecessors()) {
                        if (!pred) continue;
                        Instruction* term = pred->terminator();
                        if (!term) continue;
                        if (term->opcode() == Opcode::br && term->branch_target().block == bb) {
                            auto& args = term->branch_target().args;
                            if (p_i < args.size()) args.erase(args.begin() + static_cast<std::ptrdiff_t>(p_i));
                        } else if (term->opcode() == Opcode::br_if) {
                            if (term->true_target().block == bb) {
                                auto& args = term->true_target().args;
                                if (p_i < args.size()) args.erase(args.begin() + static_cast<std::ptrdiff_t>(p_i));
                            }
                            if (term->false_target().block == bb) {
                                auto& args = term->false_target().args;
                                if (p_i < args.size()) args.erase(args.begin() + static_cast<std::ptrdiff_t>(p_i));
                            }
                        } else if (term->opcode() == Opcode::switch_) {
                            if (term->default_target().block == bb) {
                                auto& args = term->default_target().args;
                                if (p_i < args.size()) args.erase(args.begin() + static_cast<std::ptrdiff_t>(p_i));
                            }
                            for (auto& sc : term->switch_cases()) {
                                if (sc.target.block == bb) {
                                    auto& args = sc.target.args;
                                    if (p_i < args.size()) args.erase(args.begin() + static_cast<std::ptrdiff_t>(p_i));
                                }
                            }
                        }
                    }
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

bool licm_pass(Function& fn, LoopInfo& loop, DominatorTree& dom) {
    (void)dom;
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

bool ivsr_pass(Function& fn, LoopInfo& loop, DominatorTree& dom) {
    (void)dom;
    BasicBlock* header = loop.header();
    BasicBlock* preheader = loop.preheader();
    if (!header || !preheader || loop.latches().size() != 1) return false;

    BasicBlock* latch = loop.latches()[0];
    if (!latch) return false;

    Instruction* ph_term = preheader->terminator();
    Instruction* latch_term = latch->terminator();
    if (!ph_term || !latch_term) return false;

    BranchTarget* ph_bt = nullptr;
    if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == header) ph_bt = &ph_term->branch_target();
    else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == header) ph_bt = &ph_term->true_target();
        else if (ph_term->false_target().block == header) ph_bt = &ph_term->false_target();
    }
    if (!ph_bt || ph_bt->args.size() != header->param_count()) return false;

    BranchTarget* latch_bt = nullptr;
    if (latch_term->opcode() == Opcode::br && latch_term->branch_target().block == header) latch_bt = &latch_term->branch_target();
    else if (latch_term->opcode() == Opcode::br_if) {
        if (latch_term->true_target().block == header) latch_bt = &latch_term->true_target();
        else if (latch_term->false_target().block == header) latch_bt = &latch_term->false_target();
    }
    if (!latch_bt || latch_bt->args.size() != header->param_count()) return false;

    std::vector<BasicIV> bivs;
    std::unordered_map<const Value*, size_t> param_to_biv;

    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* param = header->param(i);
        if (!param || !param->type().is_integer()) continue;
        Value* latch_arg = latch_bt->args[i];
        if (!latch_arg || !latch_arg->is_instruction()) continue;
        Instruction* def = latch_arg->defining_instruction();
        if (!def) continue;

        if (def->opcode() == Opcode::add) {
            Value* op0 = def->operand(0);
            Value* op1 = def->operand(1);
            if (op0 == param && loop.is_loop_invariant(op1)) {
                param_to_biv[param] = bivs.size();
                bivs.push_back({param, i, ph_bt->args[i], op1, def, false});
            } else if (op1 == param && loop.is_loop_invariant(op0)) {
                param_to_biv[param] = bivs.size();
                bivs.push_back({param, i, ph_bt->args[i], op0, def, false});
            }
        } else if (def->opcode() == Opcode::sub && def->operand(0) == param && loop.is_loop_invariant(def->operand(1))) {
            param_to_biv[param] = bivs.size();
            bivs.push_back({param, i, ph_bt->args[i], def->operand(1), def, true});
        }
    }

    if (bivs.empty()) return false;

    Builder b_ph(*fn.parent());
    b_ph.set_function(&fn);
    b_ph.position_before(ph_term);

    Builder b_latch(*fn.parent());
    b_latch.set_function(&fn);
    b_latch.position_before(latch_term);

    bool changed = false;
    std::unordered_map<std::pair<const Value*, uint8_t>, Value*, PairHash> biv_scaled_map;

    auto get_or_create_scaled_biv = [&](const BasicIV& biv, uint8_t scale) -> Value* {
        if (scale == 1) return biv.param;
        auto key = std::make_pair(biv.param, scale);
        auto it = biv_scaled_map.find(key);
        if (it != biv_scaled_map.end()) return it->second;

        Value* scale_val = build_smart_const_i64(b_ph, scale);
        Value* init_bytes = build_smart_mul(b_ph, biv.init, scale_val);
        Value* step_bytes = build_smart_mul(b_ph, biv.step, scale_val);

        Value* biv_bytes_p = fn.parent()->arena().make<Value>(fn.next_value_id(), Type::i64(), ValueKind::BlockParam);
        header->add_param(biv_bytes_p);
        ph_bt->args.push_back(init_bytes);

        Value* next_biv_bytes = b_latch.build_add(biv_bytes_p, step_bytes);
        latch_bt->args.push_back(next_biv_bytes);

        biv_scaled_map[key] = biv_bytes_p;
        return biv_bytes_p;
    };

    for (BasicBlock* bb : loop.blocks()) {
        if (!bb || bb == preheader) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            if (inst->opcode() == Opcode::load_indexed || inst->opcode() == Opcode::store_indexed) {
                Value* base = inst->operand(0);
                Value* index = inst->operand(1);
                uint8_t scale = inst->scale();
                int32_t offset = inst->offset();

                if (loop.is_loop_invariant(base)) {
                    if (index && index->is_instruction()) {
                        Instruction* idx_def = index->defining_instruction();
                        if (idx_def && idx_def->opcode() == Opcode::add) {
                            Value* a = idx_def->operand(0);
                            Value* b = idx_def->operand(1);
                            const BasicIV* matched_biv = nullptr;
                            Value* inv_offset_val = nullptr;

                            if (param_to_biv.count(a) && loop.is_loop_invariant(b)) {
                                matched_biv = &bivs[param_to_biv[a]];
                                inv_offset_val = b;
                            } else if (param_to_biv.count(b) && loop.is_loop_invariant(a)) {
                                matched_biv = &bivs[param_to_biv[b]];
                                inv_offset_val = a;
                            }

                            if (matched_biv && inv_offset_val) {
                                Value* scale_val = build_smart_const_i64(b_ph, scale);
                                Value* row_bytes = build_smart_mul(b_ph, inv_offset_val, scale_val);
                                Value* new_base = build_smart_add(b_ph, base, row_bytes);
                                if (offset != 0) new_base = build_smart_add(b_ph, new_base, build_smart_const_i64(b_ph, offset));

                                Value* biv_bytes_p = get_or_create_scaled_biv(*matched_biv, scale);
                                inst->set_operand(0, new_base);
                                inst->set_operand(1, biv_bytes_p);
                                inst->set_scale(1);
                                inst->set_offset(0);
                                changed = true;
                            }
                        }
                    }

                    if (index && param_to_biv.count(index) && scale > 1) {
                        const BasicIV& matched_biv = bivs[param_to_biv[index]];
                        Value* new_base = base;
                        if (offset != 0) new_base = build_smart_add(b_ph, base, build_smart_const_i64(b_ph, offset));

                        Value* biv_bytes_p = get_or_create_scaled_biv(matched_biv, scale);
                        inst->set_operand(0, new_base);
                        inst->set_operand(1, biv_bytes_p);
                        inst->set_scale(1);
                        inst->set_offset(0);
                        changed = true;
                    }
                }
            }
        }
    }

    for (const auto& [key, scaled_p] : biv_scaled_map) {
        const Value* biv_param = key.first;
        uint8_t sc = key.second;
        int64_t sc_int = static_cast<int64_t>(sc);
        int shl_k = (sc == 8 ? 3 : (sc == 4 ? 2 : (sc == 2 ? 1 : 0)));

        for (BasicBlock* bb : loop.blocks()) {
            if (!bb || bb == preheader) continue;
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (cur->opcode() == Opcode::mul && cur->produces_value()) {
                    Value* op0 = cur->operand(0);
                    Value* op1 = cur->operand(1);
                    int64_t c0, c1;
                    bool has_c0 = get_const_int(op0, c0);
                    bool has_c1 = get_const_int(op1, c1);
                    if ((op0 == biv_param && has_c1 && c1 == sc_int) || (op1 == biv_param && has_c0 && c0 == sc_int)) {
                        replace_all_uses(fn, cur->result(), scaled_p);
                        bb->remove_instruction(cur);
                        changed = true;
                    }
                } else if (cur->opcode() == Opcode::shl && cur->produces_value() && shl_k > 0) {
                    Value* op0 = cur->operand(0);
                    Value* op1 = cur->operand(1);
                    int64_t c1;
                    if (op0 == biv_param && get_const_int(op1, c1) && c1 == shl_k) {
                        replace_all_uses(fn, cur->result(), scaled_p);
                        bb->remove_instruction(cur);
                        changed = true;
                    }
                }
                cur = next;
            }
        }
    }

    Instruction* hdr_term = header->terminator();
    if (hdr_term && hdr_term->opcode() == Opcode::br_if) {
        Value* cond_val = hdr_term->operand(0);
        if (cond_val && cond_val->is_instruction()) {
            Instruction* cond_inst = cond_val->defining_instruction();
            if (cond_inst && cond_inst->parent() == header && is_comparison(cond_inst->opcode())) {
                Value* lhs = cond_inst->operand(0);
                Value* rhs = cond_inst->operand(1);
                if (param_to_biv.count(lhs) && loop.is_loop_invariant(rhs)) {
                    const BasicIV& matched_biv = bivs[param_to_biv[lhs]];
                    for (uint8_t sc : {uint8_t(8), uint8_t(4), uint8_t(2)}) {
                        auto key = std::make_pair(matched_biv.param, sc);
                        if (biv_scaled_map.count(key)) {
                            Value* biv_bytes_p = biv_scaled_map[key];
                            Value* scale_val = build_smart_const_i64(b_ph, sc);
                            Value* limit_bytes = build_smart_mul(b_ph, rhs, scale_val);
                            cond_inst->set_operand(0, biv_bytes_p);
                            cond_inst->set_operand(1, limit_bytes);
                            changed = true;
                            break;
                        }
                    }
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

                std::unordered_set<Instruction*> cycle_insts;
                std::vector<Value*> worklist = {p};
                std::unordered_set<Value*> visited = {p};
                bool is_pure_cycle = true;

                while (!worklist.empty()) {
                    Value* cur_v = worklist.back();
                    worklist.pop_back();

                    for (BasicBlock* u_bb : fn.blocks()) {
                        if (!u_bb) continue;
                        for (Instruction* inst : *u_bb) {
                            if (!inst) continue;
                            for (Value* sv : inst->state_map()) {
                                if (sv == cur_v) { is_pure_cycle = false; break; }
                            }
                            if (!is_pure_cycle) break;

                            for (Value* op : inst->operands()) {
                                if (op == cur_v) {
                                    if (inst->has_side_effects() || !inst->produces_value()) { is_pure_cycle = false; break; }
                                    if (inst->opcode() != Opcode::add && inst->opcode() != Opcode::sub) { is_pure_cycle = false; break; }
                                    cycle_insts.insert(inst);
                                    Value* res = inst->result();
                                    if (res && !visited.count(res)) {
                                        visited.insert(res);
                                        worklist.push_back(res);
                                    }
                                }
                            }
                            if (!is_pure_cycle) break;

                            auto check_bt = [&](const BranchTarget& bt) {
                                for (size_t arg_idx = 0; arg_idx < bt.args.size(); ++arg_idx) {
                                    if (bt.args[arg_idx] == cur_v) {
                                        if (bt.block == bb && arg_idx == p_i) {}
                                        else { is_pure_cycle = false; }
                                    }
                                }
                            };

                            if (inst->opcode() == Opcode::br) check_bt(inst->branch_target());
                            else if (inst->opcode() == Opcode::br_if) {
                                check_bt(inst->true_target());
                                check_bt(inst->false_target());
                            }
                            if (!is_pure_cycle) break;
                        }
                        if (!is_pure_cycle) break;
                    }
                    if (!is_pure_cycle) break;
                }

                if (is_pure_cycle && !cycle_insts.empty()) {
                    for (Instruction* inst : cycle_insts) {
                        if (inst && inst->parent()) inst->parent()->remove_instruction(inst);
                    }
                    auto& params = bb->params();
                    params.erase(params.begin() + static_cast<std::ptrdiff_t>(p_i));
                    for (size_t k = p_i; k < params.size(); ++k) {
                        params[k]->set_block_param(bb, static_cast<uint32_t>(k));
                    }

                    for (BasicBlock* pred : bb->predecessors()) {
                        if (!pred) continue;
                        Instruction* term = pred->terminator();
                        if (!term) continue;
                        if (term->opcode() == Opcode::br && term->branch_target().block == bb) {
                            auto& args = term->branch_target().args;
                            if (p_i < args.size()) args.erase(args.begin() + static_cast<std::ptrdiff_t>(p_i));
                        } else if (term->opcode() == Opcode::br_if) {
                            if (term->true_target().block == bb) {
                                auto& args = term->true_target().args;
                                if (p_i < args.size()) args.erase(args.begin() + static_cast<std::ptrdiff_t>(p_i));
                            }
                            if (term->false_target().block == bb) {
                                auto& args = term->false_target().args;
                                if (p_i < args.size()) args.erase(args.begin() + static_cast<std::ptrdiff_t>(p_i));
                            }
                        }
                    }
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
} // namespace

bool optimize_function_loops(Function& fn, const LoopOptOptions& options) {
    bool any_changed = false;

    if (options.enable_diamond_select) {
        any_changed |= simplify_cfg_diamonds(fn);
    }

    for (size_t iter = 0; iter < options.max_iterations; ++iter) {
        bool iter_changed = false;
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopAnalysis loops(fn, dom);

        if (options.enable_dce) {
            iter_changed |= constant_folding_pass(fn);
            iter_changed |= cse_pass(fn, dom);
            iter_changed |= dead_code_elimination_pass(fn);
            iter_changed |= eliminate_dead_induction_cycles(fn);
            iter_changed |= dead_code_elimination_pass(fn);
        }

        std::vector<LoopInfo*> post_order = loops.post_order_loops();
        if (post_order.empty()) {
            if (iter_changed) any_changed = true;
            break;
        }

        for (LoopInfo* loop : post_order) {
            if (!loop) continue;
            if (options.enable_licm) iter_changed |= licm_pass(fn, *loop, dom);
            if (options.enable_ivsr) iter_changed |= ivsr_pass(fn, *loop, dom);
        }

        if (options.enable_dce) {
            iter_changed |= constant_folding_pass(fn);
            iter_changed |= cse_pass(fn, dom);
            iter_changed |= dead_code_elimination_pass(fn);
            iter_changed |= eliminate_dead_induction_cycles(fn);
            iter_changed |= dead_code_elimination_pass(fn);
        }

        fn.rebuild_cfg_predecessors();
        if (iter_changed) any_changed = true;
        else break;
    }

    if (options.enable_unroll) {
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        bool allow_fp = options.enable_fp_reassociation || fn.allow_fp_reassociation() || (fn.parent() && fn.parent()->allow_fp_reassociation());
        if (loop_unroll_pass(fn, dom, {options.unroll_factor, true, allow_fp, true})) {
            any_changed = true;
            fn.rebuild_cfg_predecessors();
            DominatorTree dom_after(fn);
            constant_folding_pass(fn);
            cse_pass(fn, dom_after);
            dead_code_elimination_pass(fn);
        }
    }

    if (options.enable_diamond_select) {
        any_changed |= simplify_cfg_diamonds(fn);
    }

    return any_changed;
}

bool optimize_module_loops(Module& mod, const LoopOptOptions& options) {
    bool changed = false;
    LoopOptOptions mod_opts = options;
    if (mod.allow_fp_reassociation()) {
        mod_opts.enable_fp_reassociation = true;
    }
    for (Function* fn : mod.functions()) {
        if (fn) changed |= optimize_function_loops(*fn, mod_opts);
    }
    return changed;
}

} // namespace brass
