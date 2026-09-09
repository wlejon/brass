#include <brass/mir/jump_threading.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
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

bool resolve_val_to_int(
    const BasicBlock* B,
    const Value* val,
    const std::vector<Value*>& incoming_args,
    int64_t& out_val
) {
    if (!val) return false;
    if (get_const_int(val, out_val)) return true;

    if (val->is_block_param() && val->defining_block() == B) {
        uint32_t idx = val->param_index();
        if (idx < incoming_args.size()) {
            return get_const_int(incoming_args[idx], out_val);
        }
    }

    if (val->is_instruction() && val->defining_instruction()->parent() == B) {
        const Instruction* def = val->defining_instruction();
        if (def->opcode() == Opcode::iconst_i32 || def->opcode() == Opcode::iconst_i64) {
            return get_const_int(val, out_val);
        }
        if (def->operand_count() == 2) {
            int64_t s0 = 0, s1 = 0;
            if (resolve_val_to_int(B, def->operand(0), incoming_args, s0) &&
                resolve_val_to_int(B, def->operand(1), incoming_args, s1)) {
                switch (def->opcode()) {
                    case Opcode::add: out_val = s0 + s1; return true;
                    case Opcode::sub: out_val = s0 - s1; return true;
                    case Opcode::mul: out_val = s0 * s1; return true;
                    case Opcode::and_: out_val = s0 & s1; return true;
                    case Opcode::or_: out_val = s0 | s1; return true;
                    case Opcode::xor_: out_val = s0 ^ s1; return true;
                    default: break;
                }
            }
        }
    }

    return false;
}

int evaluate_condition_along_edge(
    const BasicBlock* B,
    const Value* cond,
    const std::vector<Value*>& incoming_args,
    const BasicBlock* P,
    bool is_true_edge,
    bool is_false_edge
) {
    if (!cond) return -1;

    // Direct constant
    int64_t cval = 0;
    if (get_const_int(cond, cval)) {
        return cval != 0 ? 1 : 0;
    }

    // Case A: cond is a block parameter of B
    if (cond->is_block_param() && cond->defining_block() == B) {
        uint32_t idx = cond->param_index();
        if (idx < incoming_args.size()) {
            if (get_const_int(incoming_args[idx], cval)) {
                return cval != 0 ? 1 : 0;
            }
        }
    }

    // Case B: cond is computed in B via comparison or pure op
    if (cond->is_instruction() && cond->defining_instruction()->parent() == B) {
        const Instruction* c_inst = cond->defining_instruction();
        Opcode op = c_inst->opcode();

        if (c_inst->operand_count() == 2) {
            int64_t v0 = 0, v1 = 0;
            if (resolve_val_to_int(B, c_inst->operand(0), incoming_args, v0) &&
                resolve_val_to_int(B, c_inst->operand(1), incoming_args, v1)) {
                bool res = false;
                switch (op) {
                    case Opcode::eq: res = (v0 == v1); break;
                    case Opcode::ne: res = (v0 != v1); break;
                    case Opcode::slt: res = (v0 < v1); break;
                    case Opcode::ult: res = (static_cast<uint64_t>(v0) < static_cast<uint64_t>(v1)); break;
                    case Opcode::sle: res = (v0 <= v1); break;
                    case Opcode::ule: res = (static_cast<uint64_t>(v0) <= static_cast<uint64_t>(v1)); break;
                    case Opcode::sgt: res = (v0 > v1); break;
                    case Opcode::ugt: res = (static_cast<uint64_t>(v0) > static_cast<uint64_t>(v1)); break;
                    case Opcode::sge: res = (v0 >= v1); break;
                    case Opcode::uge: res = (static_cast<uint64_t>(v0) >= static_cast<uint64_t>(v1)); break;
                    default: return -1;
                }
                return res ? 1 : 0;
            }
        }
    }

    // Case C: Edge-correlated condition from predecessor P
    if (P && P->terminator() && P->terminator()->opcode() == Opcode::br_if) {
        const Instruction* p_term = P->terminator();
        const Value* p_cond = p_term->operand(0);
        if (p_cond == cond) {
            if (is_true_edge) return 1;
            if (is_false_edge) return 0;
        }
    }

    return -1;
}

bool has_external_instruction_uses(const Function& fn, const BasicBlock* B) {
    for (const Instruction* inst : *B) {
        if (!inst || !inst->produces_value()) continue;
        const Value* res = inst->result();

        for (const BasicBlock* other_bb : fn.blocks()) {
            if (!other_bb || other_bb == B) continue;
            for (const Instruction* other_i : *other_bb) {
                if (!other_i) continue;
                for (size_t op_i = 0; op_i < other_i->operand_count(); ++op_i) {
                    if (other_i->operand(op_i) == res) return true;
                }
                for (size_t sm_i = 0; sm_i < other_i->state_map().size(); ++sm_i) {
                    if (other_i->state_map()[sm_i] == res) return true;
                }
                for (const Value* ba : other_i->branch_target().args) {
                    if (ba == res) return true;
                }
                for (const Value* ba : other_i->true_target().args) {
                    if (ba == res) return true;
                }
                for (const Value* ba : other_i->false_target().args) {
                    if (ba == res) return true;
                }
            }
        }
    }
    return false;
}

bool try_thread_edge(
    Function& fn,
    BasicBlock* B,
    BasicBlock* P,
    const DominatorTree& dom,
    const JumpThreadingOptions& opts
) {
    (void)opts;
    if (!B || !P || B == P) return false;

    // Safety: Do not thread back-edges to prevent irreducible CFG
    if (dom.dominates(B, P)) {
        return false;
    }

    Instruction* b_term = B->terminator();
    if (!b_term || b_term->opcode() != Opcode::br_if) return false;

    Value* cond = b_term->operand(0);
    const BranchTarget& tgt_true = b_term->true_target();
    const BranchTarget& tgt_false = b_term->false_target();
    if (!tgt_true.block || !tgt_false.block) {
        return false;
    }

    Instruction* p_term = P->terminator();
    if (!p_term) return false;

    std::vector<Value*> incoming_args;
    bool is_true_edge = false;
    bool is_false_edge = false;

    if (p_term->opcode() == Opcode::br && p_term->branch_target().block == B) {
        incoming_args = p_term->branch_target().args;
    } else if (p_term->opcode() == Opcode::br_if) {
        bool m_true = (p_term->true_target().block == B);
        bool m_false = (p_term->false_target().block == B);
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

    if (incoming_args.size() != B->param_count()) {
        return false;
    }

    int known_val = evaluate_condition_along_edge(B, cond, incoming_args, P, is_true_edge, is_false_edge);
    if (known_val != 0 && known_val != 1) {
        return false;
    }

    const BranchTarget& chosen_tgt = (known_val == 1) ? tgt_true : tgt_false;
    BasicBlock* dest = chosen_tgt.block;
    if (!dest || dest == B) return false;

    // Ensure non-terminator instructions have no side effects
    for (Instruction* inst : *B) {
        if (!inst || inst->is_terminator()) break;
        if (inst->has_side_effects() || inst->is_call()) {
            return false;
        }
    }

    // Safety: ensure values defined in B are not used outside B without parameters
    if (has_external_instruction_uses(fn, B)) {
        return false;
    }

    bool args_use_b_insts = false;
    for (Value* a : chosen_tgt.args) {
        if (a && a->is_instruction() && a->defining_instruction()->parent() == B) {
            args_use_b_insts = true;
            break;
        }
    }

    if (!args_use_b_insts) {
        // Trivial redirection: map block parameters to incoming arguments
        std::vector<Value*> redirected_args;
        redirected_args.reserve(chosen_tgt.args.size());
        for (Value* a : chosen_tgt.args) {
            if (a && a->is_block_param() && a->defining_block() == B) {
                uint32_t idx = a->param_index();
                redirected_args.push_back(idx < incoming_args.size() ? incoming_args[idx] : a);
            } else {
                redirected_args.push_back(a);
            }
        }

        if (p_term->opcode() == Opcode::br) {
            p_term->set_branch_target(BranchTarget(dest, std::move(redirected_args)));
        } else if (p_term->opcode() == Opcode::br_if) {
            if (is_true_edge) {
                p_term->set_true_target(BranchTarget(dest, std::move(redirected_args)));
            } else if (is_false_edge) {
                p_term->set_false_target(BranchTarget(dest, std::move(redirected_args)));
            }
        }
        return true;
    }

    // Non-trivial: duplicate B into a specialized thread block for P
    std::string thread_name = std::string(B->name()) + "_thread_" + std::string(P->name());
    BasicBlock* thread_bb = fn.parent()
        ? fn.parent()->arena().make<BasicBlock>(fn.next_block_id(), fn.parent()->string_pool().intern(thread_name))
        : new BasicBlock(fn.next_block_id(), thread_name);
    thread_bb->set_parent(&fn);
    fn.append_block(thread_bb);

    std::unordered_map<const Value*, Value*> thread_val_map;
    for (size_t p_i = 0; p_i < B->param_count(); ++p_i) {
        if (p_i < incoming_args.size()) {
            thread_val_map[B->param(p_i)] = incoming_args[p_i];
        }
    }

    for (Instruction* inst : *B) {
        if (!inst || inst->is_terminator()) break;
        Instruction* cloned = fn.parent()
            ? fn.parent()->arena().make<Instruction>(inst->opcode(), inst->type())
            : new Instruction(inst->opcode(), inst->type());

        cloned->set_imm_i64(inst->imm_i64());
        cloned->set_imm_f64(inst->imm_f64());
        cloned->set_scale(inst->scale());
        cloned->set_offset(inst->offset());
        cloned->set_memory_type(inst->memory_type());
        if (!inst->symbol().empty() && fn.parent()) {
            cloned->set_symbol(fn.parent()->string_pool().intern(inst->symbol()));
        }

        for (Value* op : inst->operands()) {
            if (!op) continue;
            auto it = thread_val_map.find(op);
            cloned->add_operand(it != thread_val_map.end() ? it->second : op);
        }

        if (inst->produces_value()) {
            Value* res = fn.parent()
                ? fn.parent()->arena().make<Value>(fn.next_value_id(), inst->type(), ValueKind::InstructionResult)
                : new Value(fn.next_value_id(), inst->type(), ValueKind::InstructionResult);
            res->set_defining_instruction(cloned);
            cloned->set_result(res);
            thread_val_map[inst->result()] = res;
        }
        thread_bb->append_instruction(cloned);
    }

    std::vector<Value*> thread_dest_args;
    thread_dest_args.reserve(chosen_tgt.args.size());
    for (Value* a : chosen_tgt.args) {
        if (!a) continue;
        auto it = thread_val_map.find(a);
        thread_dest_args.push_back(it != thread_val_map.end() ? it->second : a);
    }

    Instruction* br_dest = fn.parent()
        ? fn.parent()->arena().make<Instruction>(Opcode::br, Type::void_type())
        : new Instruction(Opcode::br, Type::void_type());
    br_dest->set_branch_target(BranchTarget(dest, std::move(thread_dest_args)));
    thread_bb->append_instruction(br_dest);

    if (p_term->opcode() == Opcode::br) {
        p_term->set_branch_target(BranchTarget(thread_bb, {}));
    } else if (p_term->opcode() == Opcode::br_if) {
        if (is_true_edge) {
            p_term->set_true_target(BranchTarget(thread_bb, {}));
        } else if (is_false_edge) {
            p_term->set_false_target(BranchTarget(thread_bb, {}));
        }
    }

    return true;
}

} // namespace

bool run_jump_threading(Function& fn, const JumpThreadingOptions& opts, JumpThreadingStats* stats) {
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
                if (try_thread_edge(fn, B, P, dom, opts)) {
                    iter_changed = true;
                    changed = true;
                    if (stats) {
                        stats->edges_threaded++;
                    }
                    break; // Restart block inspection after edge mutation
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
