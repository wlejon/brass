#include <brass/mir/sccp.hpp>
#include "sccp_lattice.hpp"
#include <brass/mir/builder.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace brass {

namespace {

struct Edge {
    BasicBlock* from = nullptr;
    BasicBlock* to = nullptr;

    bool operator==(const Edge& o) const noexcept {
        return from == o.from && to == o.to;
    }
};

struct EdgeHash {
    size_t operator()(const Edge& e) const noexcept {
        return std::hash<void*>()(e.from) ^ (std::hash<void*>()(e.to) << 1);
    }
};

struct PhiArgUse {
    BasicBlock* from_bb = nullptr;
    BasicBlock* dest_bb = nullptr;
    size_t param_idx = 0;
};

void replace_all_uses(Function& fn, Value* old_val, Value* new_val) {
    if (!old_val || !new_val || old_val == new_val) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == old_val) {
                    inst->set_operand(i, new_val);
                }
            }
            if (inst->opcode() == Opcode::br) {
                for (size_t i = 0; i < inst->branch_target().args.size(); ++i) {
                    if (inst->branch_target().args[i] == old_val) {
                        inst->branch_target().args[i] = new_val;
                    }
                }
            } else if (inst->opcode() == Opcode::br_if) {
                for (size_t i = 0; i < inst->true_target().args.size(); ++i) {
                    if (inst->true_target().args[i] == old_val) {
                        inst->true_target().args[i] = new_val;
                    }
                }
                for (size_t i = 0; i < inst->false_target().args.size(); ++i) {
                    if (inst->false_target().args[i] == old_val) {
                        inst->false_target().args[i] = new_val;
                    }
                }
            } else if (inst->opcode() == Opcode::switch_) {
                for (size_t i = 0; i < inst->default_target().args.size(); ++i) {
                    if (inst->default_target().args[i] == old_val) {
                        inst->default_target().args[i] = new_val;
                    }
                }
                for (auto& sc : inst->switch_cases()) {
                    for (size_t i = 0; i < sc.target.args.size(); ++i) {
                        if (sc.target.args[i] == old_val) {
                            sc.target.args[i] = new_val;
                        }
                    }
                }
            }
            for (size_t i = 0; i < inst->state_map().size(); ++i) {
                if (inst->state_map()[i] == old_val) {
                    inst->state_map()[i] = new_val;
                }
            }
        }
    }
}

Value* get_or_create_constant(Function& fn, const LatticeValue& lat) {
    BasicBlock* entry = fn.entry_block();
    if (!entry) return nullptr;

    // Search existing constants in entry block
    for (Instruction* inst : *entry) {
        if (!inst) continue;
        if (lat.type() == Type::i32() && inst->opcode() == Opcode::iconst_i32) {
            if (inst->imm_i32() == lat.as_i32()) return inst->result();
        } else if (lat.type() == Type::i64() && inst->opcode() == Opcode::iconst_i64) {
            if (inst->imm_i64() == lat.as_i64()) return inst->result();
        } else if (lat.type() == Type::f64() && inst->opcode() == Opcode::fconst_f64) {
            if (inst->imm_f64() == lat.as_f64()) return inst->result();
        }
    }

    Builder b(*fn.parent());
    b.position_before(entry->head());
    if (lat.type() == Type::i32()) {
        return b.build_iconst_i32(lat.as_i32());
    }
    if (lat.type() == Type::i64()) {
        return b.build_iconst_i64(lat.as_i64());
    }
    if (lat.type() == Type::ptr()) {
        return b.build_iconst_i64(static_cast<int64_t>(lat.as_ptr()));
    }
    if (lat.type() == Type::f64()) {
        return b.build_fconst_f64(lat.as_f64());
    }
    if (lat.type() == Type::f32()) {
        return b.build_fconst_f64(static_cast<double>(lat.as_f32()));
    }

    return nullptr;
}

class SccpEngine {
public:
    SccpEngine(Function& fn, const SccpOptions& options)
        : fn_(fn), options_(options) {}

    bool run() {
        if (fn_.blocks().empty() || !fn_.entry_block()) {
            return false;
        }

        build_def_use_info();
        initialize_worklists();
        solve_fixed_point();
        return materialize();
    }

private:
    Function& fn_;
    const SccpOptions& options_;

    std::vector<Edge> cfg_worklist_;
    std::unordered_set<Edge, EdgeHash> executable_edges_;
    std::unordered_set<BasicBlock*> executable_blocks_;

    std::vector<Value*> ssa_worklist_;
    std::unordered_set<Value*> in_ssa_worklist_;
    std::unordered_map<const Value*, LatticeValue> lattice_;

    std::unordered_map<const Value*, std::vector<Instruction*>> users_;
    std::unordered_map<const Value*, std::vector<PhiArgUse>> phi_arg_uses_;
    std::unordered_set<BasicBlock*> resume_targets_;

    LatticeValue get_lattice(const Value* v) const {
        if (!v) return LatticeValue::make_top();
        auto it = lattice_.find(v);
        if (it != lattice_.end()) {
            return it->second;
        }
        return LatticeValue::make_top();
    }

    void set_lattice(Value* v, const LatticeValue& new_val) {
        if (!v) return;
        LatticeValue old_val = get_lattice(v);
        if (old_val != new_val) {
            lattice_[v] = new_val;
            if (in_ssa_worklist_.insert(v).second) {
                ssa_worklist_.push_back(v);
            }
        }
    }

    bool is_edge_executable(BasicBlock* from, BasicBlock* to) const {
        return executable_edges_.find(Edge{from, to}) != executable_edges_.end();
    }

    void mark_edge_executable(BasicBlock* from, BasicBlock* to) {
        if (!to) return;
        Edge edge{from, to};
        if (executable_edges_.insert(edge).second) {
            cfg_worklist_.push_back(edge);
        }
    }

    void build_def_use_info() {
        for (const auto& rp : fn_.resume_points()) {
            if (rp.second) {
                resume_targets_.insert(rp.second);
            }
        }

        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                if (!inst) continue;
                for (size_t i = 0; i < inst->operand_count(); ++i) {
                    Value* op = inst->operand(i);
                    if (op) users_[op].push_back(inst);
                }
                for (Value* sv : inst->state_map()) {
                    if (sv) users_[sv].push_back(inst);
                }

                auto record_target_args = [&](const BranchTarget& bt) {
                    if (!bt.block) return;
                    for (size_t i = 0; i < bt.args.size(); ++i) {
                        Value* arg = bt.args[i];
                        if (arg) {
                            phi_arg_uses_[arg].push_back(PhiArgUse{bb, bt.block, i});
                        }
                    }
                };

                if (inst->opcode() == Opcode::br) {
                    record_target_args(inst->branch_target());
                } else if (inst->opcode() == Opcode::br_if) {
                    record_target_args(inst->true_target());
                    record_target_args(inst->false_target());
                } else if (inst->opcode() == Opcode::switch_) {
                    record_target_args(inst->default_target());
                    for (const auto& sc : inst->switch_cases()) {
                        record_target_args(sc.target);
                    }
                }
            }
        }
    }

    void initialize_worklists() {
        BasicBlock* entry = fn_.entry_block();
        // Function parameters on entry block start at Bottom
        for (Value* param : entry->params()) {
            if (param) {
                set_lattice(param, LatticeValue::make_bottom(param->type()));
            }
        }

        // Resume point targets are external entry points; their params start at Bottom
        for (BasicBlock* r_target : resume_targets_) {
            for (Value* param : r_target->params()) {
                if (param) {
                    set_lattice(param, LatticeValue::make_bottom(param->type()));
                }
            }
            mark_edge_executable(nullptr, r_target);
        }

        mark_edge_executable(nullptr, entry);
    }

    void evaluate_block_param(BasicBlock* dest, size_t idx) {
        if (dest == fn_.entry_block() || resume_targets_.find(dest) != resume_targets_.end()) {
            return;
        }
        if (idx >= dest->param_count()) return;

        Value* param = dest->param(idx);
        LatticeValue merged = LatticeValue::make_top();

        for (BasicBlock* pred : dest->predecessors()) {
            if (!is_edge_executable(pred, dest)) {
                continue;
            }
            Instruction* term = pred->terminator();
            if (!term) continue;

            auto check_bt = [&](const BranchTarget& bt) {
                if (bt.block == dest && idx < bt.args.size()) {
                    Value* arg = bt.args[idx];
                    LatticeValue arg_lat = get_lattice(arg);
                    merged = merged.meet(arg_lat);
                }
            };

            if (term->opcode() == Opcode::br) {
                check_bt(term->branch_target());
            } else if (term->opcode() == Opcode::br_if) {
                check_bt(term->true_target());
                check_bt(term->false_target());
            } else if (term->opcode() == Opcode::switch_) {
                check_bt(term->default_target());
                for (const auto& sc : term->switch_cases()) {
                    check_bt(sc.target);
                }
            }
        }

        set_lattice(param, merged);
    }

    void evaluate_block_params(BasicBlock* dest) {
        for (size_t i = 0; i < dest->param_count(); ++i) {
            evaluate_block_param(dest, i);
        }
    }

    void evaluate_instruction(Instruction* inst) {
        Opcode op = inst->opcode();

        if (op == Opcode::iconst_i32) {
            set_lattice(inst->result(), LatticeValue::make_i32(inst->imm_i32()));
            return;
        }
        if (op == Opcode::iconst_i64) {
            set_lattice(inst->result(), LatticeValue::make_i64(inst->imm_i64()));
            return;
        }
        if (op == Opcode::fconst_f64) {
            set_lattice(inst->result(), LatticeValue::make_f64(inst->imm_f64()));
            return;
        }

        if (op == Opcode::br) {
            mark_edge_executable(inst->parent(), inst->branch_target().block);
            return;
        }

        if (op == Opcode::br_if) {
            Value* cond = inst->operand(0);
            LatticeValue c_lat = get_lattice(cond);
            if (c_lat.is_constant()) {
                if (!c_lat.is_int_zero()) {
                    mark_edge_executable(inst->parent(), inst->true_target().block);
                } else {
                    mark_edge_executable(inst->parent(), inst->false_target().block);
                }
            } else if (c_lat.is_bottom()) {
                mark_edge_executable(inst->parent(), inst->true_target().block);
                mark_edge_executable(inst->parent(), inst->false_target().block);
            }
            return;
        }

        if (op == Opcode::switch_) {
            Value* sel = inst->operand(0);
            LatticeValue s_lat = get_lattice(sel);
            if (s_lat.is_constant()) {
                int64_t val = s_lat.as_i64();
                bool matched = false;
                for (const auto& sc : inst->switch_cases()) {
                    if (sc.value == val) {
                        mark_edge_executable(inst->parent(), sc.target.block);
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    mark_edge_executable(inst->parent(), inst->default_target().block);
                }
            } else if (s_lat.is_bottom()) {
                mark_edge_executable(inst->parent(), inst->default_target().block);
                for (const auto& sc : inst->switch_cases()) {
                    mark_edge_executable(inst->parent(), sc.target.block);
                }
            }
            return;
        }

        if (op == Opcode::select) {
            LatticeValue cond = get_lattice(inst->operand(0));
            LatticeValue tv = get_lattice(inst->operand(1));
            LatticeValue fv = get_lattice(inst->operand(2));
            set_lattice(inst->result(), evaluate_select(cond, tv, fv));
            return;
        }

        if (is_arithmetic(op) || is_bitwise(op) || is_comparison(op)) {
            if (inst->operand_count() == 1) {
                LatticeValue op0 = get_lattice(inst->operand(0));
                set_lattice(inst->result(), evaluate_unary(op, inst->type(), op0));
            } else if (inst->operand_count() == 2) {
                LatticeValue op0 = get_lattice(inst->operand(0));
                LatticeValue op1 = get_lattice(inst->operand(1));
                set_lattice(inst->result(), evaluate_binary(op, inst->type(), op0, op1));
            }
            return;
        }

        if (is_conversion(op)) {
            LatticeValue op0 = get_lattice(inst->operand(0));
            set_lattice(inst->result(), evaluate_conversion(op, inst->type(), op0));
            return;
        }

        if (op == Opcode::guard || op == Opcode::resume_point) {
            return;
        }

        if (inst->produces_value()) {
            set_lattice(inst->result(), LatticeValue::make_bottom(inst->type()));
        }
    }

    void solve_fixed_point() {
        while (!cfg_worklist_.empty() || !ssa_worklist_.empty()) {
            while (!cfg_worklist_.empty()) {
                Edge edge = cfg_worklist_.back();
                cfg_worklist_.pop_back();

                BasicBlock* dest = edge.to;
                bool first_visit = executable_blocks_.insert(dest).second;

                if (first_visit) {
                    evaluate_block_params(dest);
                    for (Instruction* inst : *dest) {
                        evaluate_instruction(inst);
                    }
                } else {
                    evaluate_block_params(dest);
                }
            }

            while (!ssa_worklist_.empty()) {
                Value* val = ssa_worklist_.back();
                ssa_worklist_.pop_back();
                in_ssa_worklist_.erase(val);

                auto it = users_.find(val);
                if (it != users_.end()) {
                    for (Instruction* user_inst : it->second) {
                        if (executable_blocks_.find(user_inst->parent()) != executable_blocks_.end()) {
                            evaluate_instruction(user_inst);
                        }
                    }
                }

                auto it_arg = phi_arg_uses_.find(val);
                if (it_arg != phi_arg_uses_.end()) {
                    for (const auto& entry : it_arg->second) {
                        if (is_edge_executable(entry.from_bb, entry.dest_bb)) {
                            evaluate_block_param(entry.dest_bb, entry.param_idx);
                        }
                    }
                }
            }
        }
    }

    bool materialize() {
        bool changed = false;

        // 1. Guard Elimination
        if (options_.enable_guard_elim) {
            for (BasicBlock* bb : fn_.blocks()) {
                if (!bb || executable_blocks_.find(bb) == executable_blocks_.end()) {
                    continue;
                }
                Instruction* cur = bb->head();
                while (cur) {
                    Instruction* next = cur->next();
                    if (cur->opcode() == Opcode::guard) {
                        Value* cond = cur->operand(0);
                        LatticeValue c_lat = get_lattice(cond);
                        if (c_lat.is_constant()) {
                            if (c_lat.as_i32() != 0) {
                                // Guard provably succeeds: remove guard completely
                                bb->remove_instruction(cur);
                                if (options_.stats) options_.stats->guards_eliminated++;
                                changed = true;
                            } else {
                                // Guard provably fails: remove subsequent instructions and insert unreachable
                                Instruction* rem = cur->next();
                                while (rem) {
                                    Instruction* rem_next = rem->next();
                                    bb->remove_instruction(rem);
                                    rem = rem_next;
                                }
                                Builder b(*fn_.parent());
                                b.position_at_end(bb);
                                b.build_unreachable();
                                if (options_.stats) options_.stats->guards_always_failing++;
                                changed = true;
                                break;
                            }
                        }
                    }
                    cur = next;
                }
            }
        }

        // 2. Branch Folding
        if (options_.enable_branch_folding) {
            for (BasicBlock* bb : fn_.blocks()) {
                if (!bb || executable_blocks_.find(bb) == executable_blocks_.end()) {
                    continue;
                }
                Instruction* term = bb->terminator();
                if (!term) continue;

                if (term->opcode() == Opcode::br_if) {
                    Value* cond = term->operand(0);
                    LatticeValue c_lat = get_lattice(cond);
                    if (c_lat.is_constant()) {
                        BranchTarget target = (c_lat.as_i32() != 0) ? term->true_target() : term->false_target();
                        term->set_opcode(Opcode::br);
                        term->set_branch_target(std::move(target));
                        term->operands().clear();
                        if (options_.stats) options_.stats->branches_folded++;
                        changed = true;
                    }
                } else if (term->opcode() == Opcode::switch_) {
                    Value* sel = term->operand(0);
                    LatticeValue s_lat = get_lattice(sel);
                    if (s_lat.is_constant()) {
                        int64_t val = s_lat.as_i64();
                        BranchTarget chosen = term->default_target();
                        for (const auto& sc : term->switch_cases()) {
                            if (sc.value == val) {
                                chosen = sc.target;
                                break;
                            }
                        }
                        term->set_opcode(Opcode::br);
                        term->set_branch_target(std::move(chosen));
                        term->switch_cases().clear();
                        term->operands().clear();
                        if (options_.stats) options_.stats->branches_folded++;
                        changed = true;
                    }
                }
            }
        }

        // 3. Constant Value Propagation & Instruction Folding
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb || executable_blocks_.find(bb) == executable_blocks_.end()) {
                continue;
            }
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (cur->produces_value() && !is_constant(cur->opcode())) {
                    LatticeValue lat = get_lattice(cur->result());
                    if (lat.is_constant()) {
                        Value* c_val = get_or_create_constant(fn_, lat);
                        if (c_val) {
                            replace_all_uses(fn_, cur->result(), c_val);
                            bb->remove_instruction(cur);
                            if (options_.stats) {
                                options_.stats->constants_propagated++;
                                options_.stats->instructions_folded++;
                            }
                            changed = true;
                        }
                    }
                }
                cur = next;
            }
        }

        // 4. Constant Block Parameter Replacement
        for (BasicBlock* bb : fn_.blocks()) {
            if (!bb || executable_blocks_.find(bb) == executable_blocks_.end()) {
                continue;
            }
            if (bb == fn_.entry_block() || resume_targets_.find(bb) != resume_targets_.end()) {
                continue;
            }
            for (size_t i = 0; i < bb->param_count(); ++i) {
                Value* p = bb->param(i);
                if (!p) continue;
                LatticeValue lat = get_lattice(p);
                if (lat.is_constant()) {
                    Value* c_val = get_or_create_constant(fn_, lat);
                    if (c_val) {
                        replace_all_uses(fn_, p, c_val);
                        if (options_.stats) options_.stats->constants_propagated++;
                        changed = true;
                    }
                }
            }
        }

        if (changed) {
            fn_.rebuild_cfg_predecessors();
        }

        return changed;
    }
};

} // namespace

bool sccp_function(Function& fn) {
    SccpOptions opts;
    return sccp_function(fn, opts);
}

bool sccp_function(Function& fn, const SccpOptions& options) {
    SccpEngine engine(fn, options);
    return engine.run();
}

bool sccp_module(Module& mod) {
    SccpOptions opts;
    return sccp_module(mod, opts);
}

bool sccp_module(Module& mod, const SccpOptions& options) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn) {
            changed |= sccp_function(*fn, options);
        }
    }
    return changed;
}

} // namespace brass
