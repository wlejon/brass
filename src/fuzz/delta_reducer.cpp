#include <brass/fuzz/delta_reducer.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/loop_opt.hpp>

#include <unordered_set>
#include <fstream>
#include <iostream>

namespace brass::fuzz {

static size_t count_instructions(const Function& fn) {
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb) count += bb->instruction_count();
    }
    return count;
}

ReductionResult DeltaReducer::reduce(const Module& initial_mod, std::string_view fn_name,
                                    const OraclePredicate& oracle) {
    ReductionResult result;
    const Function* initial_fn = initial_mod.get_function(fn_name);
    if (!initial_fn) {
        return result;
    }

    result.initial_instructions = count_instructions(*initial_fn);
    result.initial_blocks = initial_fn->block_count();

    auto cur_mod = clone_module(initial_mod);
    if (!cur_mod || !oracle(*cur_mod, fn_name)) {
        return result;
    }

    for (size_t pass = 0; pass < options_.max_passes; ++pass) {
        bool changed = false;

        if (options_.prune_instructions) {
            changed |= prune_unused_instructions(cur_mod, fn_name, oracle);
        }
        if (options_.replace_with_constants) {
            changed |= replace_with_constants(cur_mod, fn_name, oracle);
        }
        if (options_.remove_redundant_blocks) {
            changed |= simplify_control_flow(cur_mod, fn_name, oracle);
        }

        if (!changed) {
            break; // Fixed point reached
        }
    }

    const Function* final_fn = cur_mod->get_function(fn_name);
    if (final_fn) {
        result.final_instructions = count_instructions(*final_fn);
        result.final_blocks = final_fn->block_count();
    }

    result.success = true;
    result.minimized_module = std::move(cur_mod);
    return result;
}

bool DeltaReducer::prune_unused_instructions(std::unique_ptr<Module>& mod, std::string_view fn_name,
                                             const OraclePredicate& oracle) {
    Function* fn = mod->get_function(fn_name);
    if (!fn) return false;

    // Collect all used values
    std::unordered_set<const Value*> used;
    for (const BasicBlock* bb : fn->blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i)) used.insert(inst->operand(i));
            }
            for (const Value* arg : inst->branch_target().args) {
                if (arg) used.insert(arg);
            }
            for (const Value* arg : inst->true_target().args) {
                if (arg) used.insert(arg);
            }
            for (const Value* arg : inst->false_target().args) {
                if (arg) used.insert(arg);
            }
        }
    }

    // Try removing an unused instruction
    for (BasicBlock* bb : fn->blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst || inst->is_terminator() || inst->opcode() == Opcode::landing_pad) continue;
            Value* res = inst->result();
            if (res && used.find(res) == used.end()) {
                auto test_mod = clone_module(*mod);
                Function* test_fn = test_mod->get_function(fn_name);
                if (!test_fn) continue;

                BasicBlock* test_bb = test_fn->get_block_by_id(bb->id());
                if (!test_bb) continue;
                Instruction* match_inst = nullptr;
                for (Instruction* ti : *test_bb) {
                    if (ti && ti->opcode() == inst->opcode() && ti->result() && ti->result()->id() == res->id()) {
                        match_inst = ti;
                        break;
                    }
                }

                if (match_inst) {
                    test_bb->remove_instruction(match_inst);
                    test_fn->rebuild_cfg_predecessors();
                    if (verify_function(*test_fn) && oracle(*test_mod, fn_name)) {
                        mod = std::move(test_mod);
                        return true;
                    }
                }
            }
        }
    }
    return false;
}

bool DeltaReducer::replace_with_constants(std::unique_ptr<Module>& mod, std::string_view fn_name,
                                          const OraclePredicate& oracle) {
    Function* fn = mod->get_function(fn_name);
    if (!fn) return false;

    for (BasicBlock* bb : fn->blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (!inst || is_constant(inst->opcode()) || inst->is_terminator() || inst->opcode() == Opcode::landing_pad) continue;
            Value* res = inst->result();
            if (!res) continue;

            auto test_mod = clone_module(*mod);
            Function* test_fn = test_mod->get_function(fn_name);
            if (!test_fn) continue;
            BasicBlock* test_bb = test_fn->get_block_by_id(bb->id());
            if (!test_bb) continue;

            Instruction* match_inst = nullptr;
            for (Instruction* ti : *test_bb) {
                if (ti && ti->result() && ti->result()->id() == res->id()) {
                    match_inst = ti;
                    break;
                }
            }
            if (!match_inst) continue;

            Builder b(*test_mod);
            b.set_function(test_fn);
            b.position_before(match_inst);
            Value* c_val = nullptr;
            if (res->type() == Type::i64()) c_val = b.build_iconst_i64(0);
            else if (res->type() == Type::i32()) c_val = b.build_iconst_i32(0);
            else if (res->type() == Type::f64()) c_val = b.build_fconst_f64(0.0);
            else continue;

            Value* old_res = match_inst->result();
            for (BasicBlock* bbi : test_fn->blocks()) {
                if (!bbi) continue;
                for (Instruction* cur : *bbi) {
                    if (!cur) continue;
                    for (size_t i = 0; i < cur->operand_count(); ++i) {
                        if (cur->operand(i) == old_res) cur->set_operand(i, c_val);
                    }
                    for (size_t i = 0; i < cur->branch_target().args.size(); ++i) {
                        if (cur->branch_target().args[i] == old_res) cur->branch_target().args[i] = c_val;
                    }
                    for (size_t i = 0; i < cur->true_target().args.size(); ++i) {
                        if (cur->true_target().args[i] == old_res) cur->true_target().args[i] = c_val;
                    }
                    for (size_t i = 0; i < cur->false_target().args.size(); ++i) {
                        if (cur->false_target().args[i] == old_res) cur->false_target().args[i] = c_val;
                    }
                }
            }

            test_bb->remove_instruction(match_inst);
            test_fn->rebuild_cfg_predecessors();

            if (verify_function(*test_fn) && oracle(*test_mod, fn_name)) {
                mod = std::move(test_mod);
                return true;
            }
        }
    }
    return false;
}

bool DeltaReducer::simplify_control_flow(std::unique_ptr<Module>& mod, std::string_view fn_name,
                                         const OraclePredicate& oracle) {
    Function* fn = mod->get_function(fn_name);
    if (!fn) return false;

    for (BasicBlock* bb : fn->blocks()) {
        if (!bb) continue;
        Instruction* term = bb->terminator();
        if (!term || term->opcode() != Opcode::br_if) continue;

        for (int branch = 0; branch < 2; ++branch) {
            auto test_mod = clone_module(*mod);
            Function* test_fn = test_mod->get_function(fn_name);
            if (!test_fn) continue;
            BasicBlock* test_bb = test_fn->get_block_by_id(bb->id());
            if (!test_bb) continue;
            Instruction* test_term = test_bb->terminator();
            if (!test_term || test_term->opcode() != Opcode::br_if) continue;

            BranchTarget bt = (branch == 0) ? test_term->true_target() : test_term->false_target();
            test_bb->remove_instruction(test_term);

            Builder b(*test_mod);
            b.set_function(test_fn);
            b.position_at_end(test_bb);
            b.build_br(bt.block, bt.args);

            test_fn->rebuild_cfg_predecessors();

            std::vector<BasicBlock*> to_remove;
            for (BasicBlock* blk : test_fn->blocks()) {
                if (blk != test_fn->entry_block() && blk->predecessors().empty()) {
                    to_remove.push_back(blk);
                }
            }
            for (BasicBlock* blk : to_remove) {
                test_fn->remove_block(blk);
            }
            test_fn->rebuild_cfg_predecessors();

            if (verify_function(*test_fn) && oracle(*test_mod, fn_name)) {
                mod = std::move(test_mod);
                return true;
            }
        }
    }
    return false;
}

bool DeltaReducer::export_mir(const Module& mod, const std::string& path) {
    std::ofstream os(path);
    if (!os.is_open()) return false;
    print_module(mod, os);
    return true;
}

bool DeltaReducer::export_il(const Module& mod, const std::string& path) {
    std::ofstream os(path);
    if (!os.is_open()) return false;

    os << "module " << mod.name() << "\n\n";
    for (const Function* fn : mod.functions()) {
        if (!fn) continue;
        os << "func " << fn->name() << "(";
        for (size_t i = 0; i < fn->param_count(); ++i) {
            if (i > 0) os << ", ";
            os << "%" << i << ": " << to_string(fn->param_type(i));
        }
        os << ") -> " << to_string(fn->return_type()) << " {\n";

        for (const BasicBlock* bb : fn->blocks()) {
            if (!bb) continue;
            os << "  " << bb->name();
            if (bb->param_count() > 0) {
                os << "(";
                for (size_t p = 0; p < bb->param_count(); ++p) {
                    if (p > 0) os << ", ";
                    os << "%" << bb->param(p)->id() << ": " << to_string(bb->param(p)->type());
                }
                os << ")";
            }
            os << ":\n";

            for (const Instruction* inst : *bb) {
                if (!inst) continue;
                os << "    ";
                if (inst->result() && !inst->result()->type().is_void()) {
                    os << "%" << inst->result()->id() << ": " << to_string(inst->result()->type()) << " = ";
                }
                os << opcode_name(inst->opcode());
                for (size_t op = 0; op < inst->operand_count(); ++op) {
                    if (inst->operand(op)) {
                        os << " %" << inst->operand(op)->id();
                    }
                }
                if (is_constant(inst->opcode())) {
                    if (inst->opcode() == Opcode::iconst_i64) os << " " << inst->imm_i64();
                    else if (inst->opcode() == Opcode::iconst_i32) os << " " << inst->imm_i32();
                    else if (inst->opcode() == Opcode::fconst_f64) os << " " << inst->imm_f64();
                }
                if (inst->opcode() == Opcode::br && inst->branch_target().block) {
                    os << " " << inst->branch_target().block->name();
                } else if (inst->opcode() == Opcode::br_if && inst->true_target().block && inst->false_target().block) {
                    os << ", " << inst->true_target().block->name() << ", " << inst->false_target().block->name();
                }
                os << "\n";
            }
        }
        os << "}\n\n";
    }
    return true;
}

} // namespace brass::fuzz
