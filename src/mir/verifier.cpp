#include <brass/mir/verifier.hpp>
#include "verifier_vec.hpp"
#include "verifier_exceptions.hpp"
#include "verifier_coro.hpp"
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <sstream>
#include <vector>

namespace brass {

namespace {

class DominanceCalculator {
public:
    explicit DominanceCalculator(const Function& fn) {
        build(fn);
    }

    bool is_reachable(const BasicBlock* bb) const {
        return reachable_.find(bb) != reachable_.end();
    }

    bool dominates(const BasicBlock* a, const BasicBlock* b) const {
        if (!a || !b) return false;
        if (a == b) return true;
        if (!is_reachable(a) || !is_reachable(b)) {
            return false;
        }

        auto it_b = block_idx_.find(b);
        auto it_a = block_idx_.find(a);
        if (it_b == block_idx_.end() || it_a == block_idx_.end()) {
            return false;
        }

        int curr = it_b->second;
        int target = it_a->second;
        int entry = block_idx_.at(entry_block_);

        while (curr != -1 && curr != entry) {
            curr = idom_[curr];
            if (curr == target) {
                return true;
            }
        }
        return false;
    }

private:
    void build(const Function& fn) {
        entry_block_ = fn.entry_block();
        if (!entry_block_) return;

        // 1. Find reachable blocks via BFS
        std::queue<const BasicBlock*> q;
        q.push(entry_block_);
        reachable_.insert(entry_block_);

        std::vector<const BasicBlock*> reachable_blocks;
        while (!q.empty()) {
            const BasicBlock* curr = q.front();
            q.pop();
            reachable_blocks.push_back(curr);

            for (const BasicBlock* succ : curr->successors()) {
                if (succ && reachable_.insert(succ).second) {
                    q.push(succ);
                }
            }
        }

        int n = static_cast<int>(reachable_blocks.size());
        for (int i = 0; i < n; ++i) {
            block_idx_[reachable_blocks[i]] = i;
        }

        // 2. Compute DFS post-order
        std::vector<bool> visited(n, false);
        std::vector<int> post_order;
        std::vector<int> rpo_num(n, 0);

        auto dfs = [&](auto& self, int u) -> void {
            visited[u] = true;
            for (const BasicBlock* succ : reachable_blocks[u]->successors()) {
                if (!succ) continue;
                auto it = block_idx_.find(succ);
                if (it != block_idx_.end() && !visited[it->second]) {
                    self(self, it->second);
                }
            }
            post_order.push_back(u);
        };

        dfs(dfs, block_idx_[entry_block_]);

        int rpo_count = n;
        for (int u : post_order) {
            rpo_num[u] = --rpo_count;
        }

        // 3. Iterative Dominators (Cooper-Harvey-Kennedy)
        idom_.assign(n, -1);
        int entry_idx = block_idx_[entry_block_];
        idom_[entry_idx] = entry_idx;

        auto intersect = [&](int b1, int b2) -> int {
            int finger1 = b1;
            int finger2 = b2;
            while (finger1 != finger2) {
                while (rpo_num[finger1] > rpo_num[finger2]) {
                    finger1 = idom_[finger1];
                }
                while (rpo_num[finger2] > rpo_num[finger1]) {
                    finger2 = idom_[finger2];
                }
            }
            return finger1;
        };

        bool changed = true;
        while (changed) {
            changed = false;
            // Iterate in Reverse Post-Order (skip entry)
            for (int i = n - 1; i >= 0; --i) {
                int b = post_order[i];
                if (b == entry_idx) continue;

                int new_idom = -1;
                // Find first processed predecessor
                for (const BasicBlock* pred : reachable_blocks[b]->predecessors()) {
                    if (!pred) continue;
                    auto it = block_idx_.find(pred);
                    if (it != block_idx_.end()) {
                        int p = it->second;
                        if (idom_[p] != -1) {
                            new_idom = p;
                            break;
                        }
                    }
                }

                if (new_idom == -1) continue;

                for (const BasicBlock* pred : reachable_blocks[b]->predecessors()) {
                    if (!pred) continue;
                    auto it = block_idx_.find(pred);
                    if (it != block_idx_.end()) {
                        int p = it->second;
                        if (p != new_idom && idom_[p] != -1) {
                            new_idom = intersect(p, new_idom);
                        }
                    }
                }

                if (idom_[b] != new_idom) {
                    idom_[b] = new_idom;
                    changed = true;
                }
            }
        }
    }

    const BasicBlock* entry_block_ = nullptr;
    std::unordered_set<const BasicBlock*> reachable_;
    std::unordered_map<const BasicBlock*, int> block_idx_;
    std::vector<int> idom_;
};

} // namespace

void Verifier::report_error(std::string msg) {
    has_error_ = true;
    if (diag_) {
        diag_->error(std::move(msg));
    }
}

void Verifier::report_warning(std::string msg) {
    if (diag_) {
        diag_->warning(std::move(msg));
    }
}

bool Verifier::verify_module(const Module& mod) {
    has_error_ = false;

    std::unordered_set<std::string_view> fn_names;
    for (const Function* fn : mod.functions()) {
        if (!fn) {
            report_error("Module contains a null function pointer.");
            continue;
        }
        if (!fn_names.insert(fn->name()).second) {
            report_error("Duplicate function name in module: '" + std::string(fn->name()) + "'");
        }
        verify_function(*fn);
    }

    return !has_error_;
}

bool Verifier::verify_function(const Function& fn) {
    const std::string fn_prefix = "Function '" + std::string(fn.name()) + "': ";

    if (fn.blocks().empty()) {
        report_error(fn_prefix + "Function has no basic blocks.");
        return false;
    }

    const BasicBlock* entry = fn.entry_block();
    if (!entry) {
        report_error(fn_prefix + "Function has no entry basic block.");
        return false;
    }

    // Check entry block predecessors
    if (!entry->predecessors().empty()) {
        report_error(fn_prefix + "Entry block '" + std::string(entry->name()) +
                     "' must have no predecessors, but has " +
                     std::to_string(entry->predecessors().size()) + ".");
    }

    // Check entry block parameters match function signature
    if (entry->param_count() != fn.param_count()) {
        report_error(fn_prefix + "Entry block parameter count (" +
                     std::to_string(entry->param_count()) + ") does not match function parameter count (" +
                     std::to_string(fn.param_count()) + ").");
    } else {
        for (size_t i = 0; i < fn.param_count(); ++i) {
            const Value* param = entry->param(i);
            if (!param) {
                report_error(fn_prefix + "Entry block parameter " + std::to_string(i) + " is null.");
            } else if (param->type() != fn.param_type(i)) {
                report_error(fn_prefix + "Entry block parameter " + std::to_string(i) + " type (" +
                             std::string(param->type().name()) + ") does not match signature (" +
                             std::string(fn.param_type(i).name()) + ").");
            }
        }
    }

    // Index blocks and instruction positions
    std::unordered_set<const BasicBlock*> block_set;
    std::unordered_map<const Instruction*, size_t> inst_index;

    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) {
            report_error(fn_prefix + "Contains null basic block pointer.");
            continue;
        }
        if (!block_set.insert(bb).second) {
            report_error(fn_prefix + "Duplicate basic block in block list.");
        }

        size_t idx = 0;
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst) {
                inst_index[inst] = idx++;
            }
        }
    }

    // Compute Dominance
    DominanceCalculator dom(fn);

    // Verify each block and instruction
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        const std::string bb_prefix = fn_prefix + "Block '" + std::string(bb->name()) + "': ";

        if (bb->is_empty()) {
            report_error(bb_prefix + "Basic block is empty (missing terminator).");
            continue;
        }

        // Check terminator rules
        const Instruction* term = bb->terminator();
        if (!term) {
            report_error(bb_prefix + "Basic block does not end with a valid terminator instruction.");
        }

        // Ensure no non-tail instruction is a terminator
        for (const Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            if (inst != bb->tail() && inst->is_terminator()) {
                report_error(bb_prefix + "Terminator instruction '" +
                             std::string(opcode_name(inst->opcode())) + "' is not at the end of the block.");
            }
        }

        // Verify block parameters
        for (size_t i = 0; i < bb->param_count(); ++i) {
            const Value* param = bb->param(i);
            if (!param) {
                report_error(bb_prefix + "Block parameter " + std::to_string(i) + " is null.");
            } else if (param->type().is_void()) {
                report_error(bb_prefix + "Block parameter " + std::to_string(i) + " cannot be void.");
            }
        }

        // Verify each instruction in block
        for (const Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            const std::string inst_prefix = bb_prefix + "Instruction '" +
                                            std::string(opcode_name(inst->opcode())) + "': ";

            // Check operands & SSA Dominance
            auto check_value_dom = [&](const Value* val, const std::string& desc) {
                if (!val) {
                    report_error(inst_prefix + desc + " operand is null.");
                    return;
                }
                if (val->is_block_param()) {
                    const BasicBlock* def_bb = val->defining_block();
                    if (!def_bb) {
                        report_error(inst_prefix + desc + " block parameter has null defining block.");
                    } else if (dom.is_reachable(bb) && !dom.dominates(def_bb, bb)) {
                        report_error(inst_prefix + "SSA Dominance violation: " + desc +
                                     " block parameter of block '" + std::string(def_bb->name()) +
                                     "' does not dominate use in block '" + std::string(bb->name()) + "'.");
                    }
                } else if (val->is_instruction()) {
                    const Instruction* def_inst = val->defining_instruction();
                    if (!def_inst) {
                        report_error(inst_prefix + desc + " instruction result has null defining instruction.");
                    } else {
                        const BasicBlock* def_bb = def_inst->parent();
                        if (!def_bb) {
                            report_error(inst_prefix + desc + " defining instruction has null parent block.");
                        } else if (def_bb == bb) {
                            auto it_def = inst_index.find(def_inst);
                            auto it_use = inst_index.find(inst);
                            if (it_def != inst_index.end() && it_use != inst_index.end()) {
                                if (it_def->second >= it_use->second) {
                                    report_error(inst_prefix + "SSA Dominance violation: " + desc +
                                                 " is used before or at its definition in the same block.");
                                }
                            }
                        } else if (dom.is_reachable(bb) && !dom.dominates(def_bb, bb)) {
                            report_error(inst_prefix + "SSA Dominance violation: " + desc +
                                         " defined in block '" + std::string(def_bb->name()) +
                                         "' does not dominate use in block '" + std::string(bb->name()) + "'.");
                        }
                    }
                }
            };

            // 1. Check general operands dominance
            for (size_t op_i = 0; op_i < inst->operand_count(); ++op_i) {
                check_value_dom(inst->operand(op_i), "operand " + std::to_string(op_i));
            }

            // 2. Check state map dominance
            for (size_t sm_i = 0; sm_i < inst->state_map().size(); ++sm_i) {
                check_value_dom(inst->state_map()[sm_i], "state map operand " + std::to_string(sm_i));
            }

            // 3. Opcode-specific type & semantic rules
            Opcode op = inst->opcode();
            switch (op) {
                case Opcode::iconst_i32:
                    if (inst->type() != Type::i32()) {
                        report_error(inst_prefix + "Result type must be i32, got " + std::string(inst->type().name()) + ".");
                    }
                    break;
                case Opcode::iconst_i64:
                    if (inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Result type must be i64, got " + std::string(inst->type().name()) + ".");
                    }
                    break;
                case Opcode::fconst_f64:
                    if (inst->type() != Type::f64()) {
                        report_error(inst_prefix + "Result type must be f64, got " + std::string(inst->type().name()) + ".");
                    }
                    break;
                case Opcode::patchable_const_i32:
                    if (inst->type() != Type::i32()) {
                        report_error(inst_prefix + "Result type must be i32.");
                    }
                    if (inst->symbol().empty()) {
                        report_error(inst_prefix + "Patchable const requires non-empty symbol.");
                    }
                    break;
                case Opcode::patchable_const_i64:
                    if (inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Result type must be i64.");
                    }
                    if (inst->symbol().empty()) {
                        report_error(inst_prefix + "Patchable const requires non-empty symbol.");
                    }
                    break;

                case Opcode::sext_i64:
                case Opcode::zext_i64:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i32()) {
                        report_error(inst_prefix + "Requires 1 i32 operand.");
                    }
                    if (inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Result type must be i64.");
                    }
                    break;
                case Opcode::trunc_i32:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i64()) {
                        report_error(inst_prefix + "Requires 1 i64 operand.");
                    }
                    if (inst->type() != Type::i32()) {
                        report_error(inst_prefix + "Result type must be i32.");
                    }
                    break;
                case Opcode::fptosi_i32:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::f64()) {
                        report_error(inst_prefix + "Requires 1 f64 operand.");
                    }
                    if (inst->type() != Type::i32()) {
                        report_error(inst_prefix + "Result type must be i32.");
                    }
                    break;
                case Opcode::fptosi_i64:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::f64()) {
                        report_error(inst_prefix + "Requires 1 f64 operand.");
                    }
                    if (inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Result type must be i64.");
                    }
                    break;
                case Opcode::sitofp_f64_i32:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i32()) {
                        report_error(inst_prefix + "Requires 1 i32 operand.");
                    }
                    if (inst->type() != Type::f64()) {
                        report_error(inst_prefix + "Result type must be f64.");
                    }
                    break;
                case Opcode::sitofp_f64_i64:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i64()) {
                        report_error(inst_prefix + "Requires 1 i64 operand.");
                    }
                    if (inst->type() != Type::f64()) {
                        report_error(inst_prefix + "Result type must be f64.");
                    }
                    break;
                case Opcode::bitcast_i64_f64:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::f64()) {
                        report_error(inst_prefix + "Requires 1 f64 operand.");
                    }
                    if (inst->type() != Type::i64()) {
                        report_error(inst_prefix + "Result type must be i64.");
                    }
                    break;
                case Opcode::bitcast_f64_i64:
                    if (inst->operand_count() != 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i64()) {
                        report_error(inst_prefix + "Requires 1 i64 operand.");
                    }
                    if (inst->type() != Type::f64()) {
                        report_error(inst_prefix + "Result type must be f64.");
                    }
                    break;

                case Opcode::add:
                case Opcode::sub: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Requires 2 operands.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        if (t0.is_pointer_or_gcref() && t1.is_integer()) {
                            if (inst->type() != t0) {
                                report_error(inst_prefix + "Pointer arithmetic result type must match base pointer type.");
                            }
                        } else if (op == Opcode::sub && t0.is_pointer_or_gcref() && t0 == t1) {
                            if (inst->type() != Type::i64()) {
                                report_error(inst_prefix + "Pointer difference result type must be i64.");
                            }
                        } else if (t0 != t1) {
                            report_error(inst_prefix + "Operand types mismatch: " +
                                         std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                        } else if (!t0.is_numeric()) {
                            report_error(inst_prefix + "Operands must be numeric or pointer.");
                        } else if (inst->type() != t0) {
                            report_error(inst_prefix + "Result type (" + std::string(inst->type().name()) +
                                         ") must match operand type (" + std::string(t0.name()) + ").");
                        }
                    }
                    break;
                }

                case Opcode::mul:
                case Opcode::sdiv:
                case Opcode::udiv:
                case Opcode::smod:
                case Opcode::umod: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Requires 2 operands.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        if (t0 != t1) {
                            report_error(inst_prefix + "Operand types mismatch: " +
                                         std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                        } else if (!t0.is_numeric()) {
                            report_error(inst_prefix + "Operands must be numeric.");
                        } else if (inst->type() != t0) {
                            report_error(inst_prefix + "Result type (" + std::string(inst->type().name()) +
                                         ") must match operand type (" + std::string(t0.name()) + ").");
                        }
                    }
                    break;
                }

                case Opcode::fma_f32: {
                    if (inst->operand_count() != 3 || !inst->operand(0) || !inst->operand(1) || !inst->operand(2)) {
                        report_error(inst_prefix + "Requires 3 operands.");
                    } else if (inst->operand(0)->type() != Type::f32() ||
                               inst->operand(1)->type() != Type::f32() ||
                               inst->operand(2)->type() != Type::f32() ||
                               inst->type() != Type::f32()) {
                        report_error(inst_prefix + "fma_f32 operands and result must be f32.");
                    }
                    break;
                }

                case Opcode::fma_f64: {
                    if (inst->operand_count() != 3 || !inst->operand(0) || !inst->operand(1) || !inst->operand(2)) {
                        report_error(inst_prefix + "Requires 3 operands.");
                    } else if (inst->operand(0)->type() != Type::f64() ||
                               inst->operand(1)->type() != Type::f64() ||
                               inst->operand(2)->type() != Type::f64() ||
                               inst->type() != Type::f64()) {
                        report_error(inst_prefix + "fma_f64 operands and result must be f64.");
                    }
                    break;
                }

                case Opcode::neg: {
                    if (inst->operand_count() != 1 || !inst->operand(0)) {
                        report_error(inst_prefix + "Requires 1 operand.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        if (!t0.is_numeric()) {
                            report_error(inst_prefix + "Operand must be numeric.");
                        } else if (inst->type() != t0) {
                            report_error(inst_prefix + "Result type must match operand type.");
                        }
                    }
                    break;
                }

                case Opcode::and_:
                case Opcode::or_:
                case Opcode::xor_:
                case Opcode::shl:
                case Opcode::lshr:
                case Opcode::ashr: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Requires 2 operands.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        if (!t0.is_integer() || !t1.is_integer()) {
                            report_error(inst_prefix + "Bitwise/shift operands must be integer.");
                        } else if (inst->type() != t0) {
                            report_error(inst_prefix + "Result type must match first operand type.");
                        }
                    }
                    break;
                }

                case Opcode::not_:
                case Opcode::clz:
                case Opcode::ctz:
                case Opcode::popcnt: {
                    if (inst->operand_count() != 1 || !inst->operand(0)) {
                        report_error(inst_prefix + "Requires 1 operand.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        if (!t0.is_integer()) {
                            report_error(inst_prefix + "Operand must be integer.");
                        } else if (inst->type() != t0) {
                            report_error(inst_prefix + "Result type must match operand type.");
                        }
                    }
                    break;
                }

                case Opcode::eq:
                case Opcode::ne:
                case Opcode::slt:
                case Opcode::ult:
                case Opcode::sle:
                case Opcode::ule:
                case Opcode::sgt:
                case Opcode::ugt:
                case Opcode::sge:
                case Opcode::uge: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Requires 2 operands.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        if (t0 != t1) {
                            report_error(inst_prefix + "Comparison operand types mismatch: " +
                                         std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                        } else if (t0.is_void()) {
                            report_error(inst_prefix + "Cannot compare void types.");
                        }
                        if (inst->type() != Type::i32()) {
                            report_error(inst_prefix + "Comparison result type must be i32.");
                        }
                    }
                    break;
                }

                case Opcode::sadd_overflow:
                case Opcode::ssub_overflow:
                case Opcode::smul_overflow:
                case Opcode::uadd_overflow:
                case Opcode::usub_overflow:
                case Opcode::umul_overflow: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Requires 2 operands.");
                    } else {
                        Type t0 = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        if (!t0.is_integer() || !t1.is_integer()) {
                            report_error(inst_prefix + "Overflow arithmetic operands must be integer.");
                        } else if (t0 != t1) {
                            report_error(inst_prefix + "Overflow arithmetic operand types mismatch: " +
                                         std::string(t0.name()) + " vs " + std::string(t1.name()) + ".");
                        }
                        if (inst->type() != Type::i32()) {
                            report_error(inst_prefix + "Overflow arithmetic result type must be i32.");
                        }
                    }
                    break;
                }

                case Opcode::select: {
                    if (inst->operand_count() != 3 || !inst->operand(0) || !inst->operand(1) || !inst->operand(2)) {
                        report_error(inst_prefix + "Select requires 3 operands (cond, true_val, false_val).");
                    } else {
                        Type cond_t = inst->operand(0)->type();
                        Type t1 = inst->operand(1)->type();
                        Type t2 = inst->operand(2)->type();
                        if (cond_t != Type::i32()) {
                            report_error(inst_prefix + "Select condition type must be i32, got " + std::string(cond_t.name()) + ".");
                        }
                        if (t1.is_void() || t2.is_void()) {
                            report_error(inst_prefix + "Select operands cannot be void.");
                        } else if (t1 != t2) {
                            report_error(inst_prefix + "Select value types mismatch: " +
                                         std::string(t1.name()) + " vs " + std::string(t2.name()) + ".");
                        } else if (inst->type() != t1) {
                            report_error(inst_prefix + "Select result type (" + std::string(inst->type().name()) +
                                         ") must match operand type (" + std::string(t1.name()) + ").");
                        }
                    }
                    break;
                }

                case Opcode::load: {
                    if (inst->operand_count() != 1 || !inst->operand(0)) {
                        report_error(inst_prefix + "Load requires 1 base operand.");
                    } else {
                        Type base_t = inst->operand(0)->type();
                        if (!base_t.is_pointer_or_gcref()) {
                            report_error(inst_prefix + "Load base must be ptr or gcref, got " +
                                         std::string(base_t.name()) + ".");
                        }
                        if (inst->memory_type().is_void()) {
                            report_error(inst_prefix + "Load memory type cannot be void.");
                        } else if (inst->type() != inst->memory_type()) {
                            report_error(inst_prefix + "Load result type must match memory type.");
                        }
                    }
                    break;
                }

                case Opcode::store: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Store requires 2 operands (base, val).");
                    } else {
                        Type base_t = inst->operand(0)->type();
                        Type val_t = inst->operand(1)->type();
                        if (!base_t.is_pointer_or_gcref()) {
                            report_error(inst_prefix + "Store base must be ptr or gcref, got " +
                                         std::string(base_t.name()) + ".");
                        }
                        if (inst->memory_type().is_void()) {
                            report_error(inst_prefix + "Store memory type cannot be void.");
                        } else if (val_t != inst->memory_type()) {
                            report_error(inst_prefix + "Store value type (" + std::string(val_t.name()) +
                                         ") must match memory type (" + std::string(inst->memory_type().name()) + ").");
                        }
                    }
                    break;
                }

                case Opcode::load_indexed: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Load indexed requires 2 operands (base, index).");
                    } else {
                        Type base_t = inst->operand(0)->type();
                        Type idx_t = inst->operand(1)->type();
                        if (!base_t.is_pointer_or_gcref()) {
                            report_error(inst_prefix + "Load indexed base must be ptr or gcref.");
                        }
                        if (!idx_t.is_integer()) {
                            report_error(inst_prefix + "Load indexed index must be integer.");
                        }
                        uint8_t sc = inst->scale();
                        if (sc != 1 && sc != 2 && sc != 4 && sc != 8) {
                            report_error(inst_prefix + "Scale must be 1, 2, 4, or 8, got " + std::to_string(sc) + ".");
                        }
                        if (inst->memory_type().is_void() || inst->type() != inst->memory_type()) {
                            report_error(inst_prefix + "Load indexed result type must match memory type.");
                        }
                    }
                    break;
                }

                case Opcode::store_indexed: {
                    if (inst->operand_count() != 3 || !inst->operand(0) || !inst->operand(1) || !inst->operand(2)) {
                        report_error(inst_prefix + "Store indexed requires 3 operands (base, index, val).");
                    } else {
                        Type base_t = inst->operand(0)->type();
                        Type idx_t = inst->operand(1)->type();
                        Type val_t = inst->operand(2)->type();
                        if (!base_t.is_pointer_or_gcref()) {
                            report_error(inst_prefix + "Store indexed base must be ptr or gcref.");
                        }
                        if (!idx_t.is_integer()) {
                            report_error(inst_prefix + "Store indexed index must be integer.");
                        }
                        uint8_t sc = inst->scale();
                        if (sc != 1 && sc != 2 && sc != 4 && sc != 8) {
                            report_error(inst_prefix + "Scale must be 1, 2, 4, or 8, got " + std::to_string(sc) + ".");
                        }
                        if (inst->memory_type().is_void() || val_t != inst->memory_type()) {
                            report_error(inst_prefix + "Store indexed value type must match memory type.");
                        }
                    }
                    break;
                }

                case Opcode::write_barrier: {
                    if (inst->operand_count() != 2 || !inst->operand(0) || !inst->operand(1)) {
                        report_error(inst_prefix + "Write barrier requires 2 operands (obj, val).");
                    } else {
                        Type obj_t = inst->operand(0)->type();
                        if (!obj_t.is_pointer_or_gcref() && !obj_t.is_integer()) {
                            report_error(inst_prefix + "Write barrier obj must be ptr, gcref, or int.");
                        }
                    }
                    break;
                }

                case Opcode::call: {
                    if (inst->symbol().empty()) {
                        report_error(inst_prefix + "Call requires a non-empty callee symbol.");
                    }
                    break;
                }

                case Opcode::call_indirect: {
                    if (inst->operand_count() < 1 || !inst->operand(0) || !inst->operand(0)->type().is_pointer()) {
                        report_error(inst_prefix + "Call indirect requires first operand to be ptr.");
                    }
                    break;
                }

                case Opcode::patchable_call: {
                    if (inst->symbol().empty() || inst->extra_symbol().empty()) {
                        report_error(inst_prefix + "Patchable call requires site symbol and callee symbol.");
                    }
                    break;
                }

                case Opcode::safepoint:
                    break;

                case Opcode::guard: {
                    if (inst->operand_count() < 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i32()) {
                        report_error(inst_prefix + "Guard requires i32 condition operand.");
                    }
                    if (inst->symbol().empty()) {
                        report_error(inst_prefix + "Guard requires non-empty exit label.");
                    }
                    break;
                }

                case Opcode::resume_point:
                case Opcode::osr_entry:
                    break;

                case Opcode::br: {
                    const BranchTarget& target = inst->branch_target();
                    if (!target.block) {
                        report_error(inst_prefix + "Branch has null target block.");
                    } else {
                        if (block_set.find(target.block) == block_set.end()) {
                            report_error(inst_prefix + "Branch target block does not belong to function.");
                        }
                        if (target.args.size() != target.block->param_count()) {
                            report_error(inst_prefix + "Branch passes " + std::to_string(target.args.size()) +
                                         " arguments, but target block '" + std::string(target.block->name()) +
                                         "' expects " + std::to_string(target.block->param_count()) + ".");
                        } else {
                            for (size_t a_i = 0; a_i < target.args.size(); ++a_i) {
                                check_value_dom(target.args[a_i], "branch arg " + std::to_string(a_i));
                                if (target.args[a_i] && target.args[a_i]->type() != target.block->param(a_i)->type()) {
                                    report_error(inst_prefix + "Branch argument " + std::to_string(a_i) +
                                                 " type (" + std::string(target.args[a_i]->type().name()) +
                                                 ") does not match target parameter type (" +
                                                 std::string(target.block->param(a_i)->type().name()) + ").");
                                }
                            }
                        }
                    }
                    break;
                }

                case Opcode::br_if: {
                    if (inst->operand_count() < 1 || !inst->operand(0) || inst->operand(0)->type() != Type::i32()) {
                        report_error(inst_prefix + "br_if condition must be i32.");
                    }
                    auto check_target = [&](const BranchTarget& target, const std::string& label) {
                        if (!target.block) {
                            report_error(inst_prefix + label + " target block is null.");
                        } else {
                            if (block_set.find(target.block) == block_set.end()) {
                                report_error(inst_prefix + label + " target block does not belong to function.");
                            }
                            if (target.args.size() != target.block->param_count()) {
                                report_error(inst_prefix + label + " branch passes " + std::to_string(target.args.size()) +
                                             " arguments, but target expects " + std::to_string(target.block->param_count()) + ".");
                            } else {
                                for (size_t a_i = 0; a_i < target.args.size(); ++a_i) {
                                    check_value_dom(target.args[a_i], label + " arg " + std::to_string(a_i));
                                    if (target.args[a_i] && target.args[a_i]->type() != target.block->param(a_i)->type()) {
                                        report_error(inst_prefix + label + " argument " + std::to_string(a_i) +
                                                     " type (" + std::string(target.args[a_i]->type().name()) +
                                                     ") does not match target parameter (" +
                                                     std::string(target.block->param(a_i)->type().name()) + ").");
                                    }
                                }
                            }
                        }
                    };
                    check_target(inst->true_target(), "true");
                    check_target(inst->false_target(), "false");
                    break;
                }

                case Opcode::switch_: {
                    if (inst->operand_count() != 1 || !inst->operand(0) || !inst->operand(0)->type().is_integer()) {
                        report_error(inst_prefix + "switch requires 1 integer condition operand.");
                    }
                    auto check_target = [&](const BranchTarget& target, const std::string& label) {
                        if (!target.block) {
                            report_error(inst_prefix + label + " target block is null.");
                        } else {
                            if (block_set.find(target.block) == block_set.end()) {
                                report_error(inst_prefix + label + " target block does not belong to function.");
                            }
                            if (target.args.size() != target.block->param_count()) {
                                report_error(inst_prefix + label + " switch passes " + std::to_string(target.args.size()) +
                                             " arguments, but target expects " + std::to_string(target.block->param_count()) + ".");
                            } else {
                                for (size_t a_i = 0; a_i < target.args.size(); ++a_i) {
                                    check_value_dom(target.args[a_i], label + " arg " + std::to_string(a_i));
                                    if (target.args[a_i] && target.args[a_i]->type() != target.block->param(a_i)->type()) {
                                        report_error(inst_prefix + label + " argument " + std::to_string(a_i) +
                                                     " type (" + std::string(target.args[a_i]->type().name()) +
                                                     ") does not match target parameter (" +
                                                     std::string(target.block->param(a_i)->type().name()) + ").");
                                    }
                                }
                            }
                        }
                    };
                    check_target(inst->default_target(), "default");
                    const auto& cases = inst->switch_cases();
                    for (size_t c_i = 0; c_i < cases.size(); ++c_i) {
                        check_target(cases[c_i].target, "case " + std::to_string(cases[c_i].value));
                    }
                    break;
                }

                case Opcode::ret: {
                    if (fn.return_type().is_void()) {
                        if (inst->operand_count() > 0 && inst->operand(0) != nullptr && !inst->operand(0)->type().is_void()) {
                            report_error(inst_prefix + "Void function return instruction must not have a return value.");
                        }
                    } else {
                        if (inst->operand_count() != 1 || !inst->operand(0)) {
                            report_error(inst_prefix + "Non-void function must return a value of type " +
                                         std::string(fn.return_type().name()) + ".");
                        } else if (inst->operand(0)->type() != fn.return_type()) {
                            report_error(inst_prefix + "Return value type (" +
                                         std::string(inst->operand(0)->type().name()) +
                                         ") does not match function return type (" +
                                         std::string(fn.return_type().name()) + ").");
                        }
                    }
                    break;
                }

                case Opcode::unreachable:
                    break;
                default:
                    if (is_vector_op(inst->opcode())) {
                        verify_vector_instruction(inst, inst_prefix, [this](const std::string& msg) { report_error(msg); });
                    } else if (is_coro_op(inst->opcode())) {
                        verify_coro_instruction(inst, bb, inst_prefix, [this](const std::string& msg) { report_error(msg); });
                    } else if (!verify_exception_instruction(inst, bb, inst_prefix, [this](const std::string& msg) { report_error(msg); })) {
                        report_error(inst_prefix + "Unhandled or invalid instruction opcode: " + std::string(opcode_name(inst->opcode())));
                    }
                    break;
            }
        }
    }

    // Verify resume points
    for (const auto& entry_pair : fn.resume_points()) {
        if (!entry_pair.second) {
            report_error(fn_prefix + "Resume point " + std::to_string(entry_pair.first) + " has null target block.");
        } else if (block_set.find(entry_pair.second) == block_set.end()) {
            report_error(fn_prefix + "Resume point " + std::to_string(entry_pair.first) +
                         " points to block not in function.");
        }
    }

    return !has_error_;
}

bool verify_module(const Module& mod, DiagnosticReporter* diag) {
    Verifier v(diag);
    return v.verify_module(mod);
}

bool verify_function(const Function& fn, DiagnosticReporter* diag) {
    Verifier v(diag);
    return v.verify_function(fn);
}

} // namespace brass
