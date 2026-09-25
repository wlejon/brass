#include <brass/mir/loop_opt.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/core/arena.hpp>
#include <brass/core/string_pool.hpp>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

Function* clone_function(const Function& src, Module& dst_mod) {
    std::vector<Type> params = src.param_types();
    Function* dst_fn = dst_mod.create_function(src.name(), src.return_type(), params);

    uint32_t max_block_id = 0;
    uint32_t max_val_id = 0;
    for (const BasicBlock* src_bb : src.blocks()) {
        if (!src_bb) continue;
        if (src_bb->id() > max_block_id) max_block_id = src_bb->id();
        for (size_t i = 0; i < src_bb->param_count(); ++i) {
            if (src_bb->param(i) && src_bb->param(i)->id() > max_val_id) max_val_id = src_bb->param(i)->id();
        }
        for (const Instruction* src_inst : *src_bb) {
            if (src_inst && src_inst->result() && src_inst->result()->id() > max_val_id) max_val_id = src_inst->result()->id();
        }
    }
    dst_fn->set_next_block_id(max_block_id + 1);
    dst_fn->set_next_value_id(max_val_id + 1);

    std::unordered_map<const BasicBlock*, BasicBlock*> block_map;
    std::unordered_map<const Value*, Value*> value_map;

    for (const BasicBlock* src_bb : src.blocks()) {
        if (!src_bb) continue;
        BasicBlock* dst_bb = dst_fn->parent()->arena().make<BasicBlock>(src_bb->id(), dst_fn->parent()->string_pool().intern(src_bb->name()));
        dst_bb->set_parent(dst_fn);
        dst_fn->append_block(dst_bb);
        block_map[src_bb] = dst_bb;

        for (size_t i = 0; i < src_bb->param_count(); ++i) {
            const Value* src_p = src_bb->param(i);
            Value* dst_p = dst_fn->parent()->arena().make<Value>(src_p->id(), src_p->type(), ValueKind::BlockParam);
            dst_bb->add_param(dst_p);
            value_map[src_p] = dst_p;
        }
    }

    auto map_value = [&](const Value* v) -> Value* {
        if (!v) return nullptr;
        auto it = value_map.find(v);
        if (it != value_map.end()) return it->second;
        return const_cast<Value*>(v);
    };

    auto map_target = [&](const BranchTarget& src_bt) -> BranchTarget {
        BranchTarget dst_bt;
        if (src_bt.block) dst_bt.block = block_map[src_bt.block];
        for (const Value* arg : src_bt.args) dst_bt.args.push_back(map_value(arg));
        return dst_bt;
    };

    std::unordered_map<const Instruction*, Instruction*> inst_map;

    // Pass 1: Create all instructions and their result values
    for (const BasicBlock* src_bb : src.blocks()) {
        if (!src_bb) continue;
        BasicBlock* dst_bb = block_map[src_bb];

        for (const Instruction* src_inst : *src_bb) {
            if (!src_inst) continue;
            Instruction* dst_inst = dst_fn->parent()->arena().make<Instruction>(src_inst->opcode(), src_inst->type());
            dst_inst->set_imm_i64(src_inst->imm_i64());
            dst_inst->set_imm_f64(src_inst->imm_f64());
            dst_inst->set_scale(src_inst->scale());
            dst_inst->set_offset(src_inst->offset());
            dst_inst->set_memory_type(src_inst->memory_type());
            dst_inst->set_loc(src_inst->loc());
            if (!src_inst->symbol().empty()) dst_inst->set_symbol(dst_fn->parent()->string_pool().intern(src_inst->symbol()));
            if (!src_inst->extra_symbol().empty()) dst_inst->set_extra_symbol(dst_fn->parent()->string_pool().intern(src_inst->extra_symbol()));

            if (src_inst->produces_value()) {
                const Value* src_res = src_inst->result();
                Value* dst_res = dst_fn->parent()->arena().make<Value>(src_res->id(), src_res->type(), ValueKind::InstructionResult);
                dst_res->set_defining_instruction(dst_inst);
                dst_inst->set_result(dst_res);
                value_map[src_res] = dst_res;
            }

            dst_bb->append_instruction(dst_inst);
            inst_map[src_inst] = dst_inst;
        }
    }

    // Pass 2: Connect operands and control-flow targets
    for (const BasicBlock* src_bb : src.blocks()) {
        if (!src_bb) continue;
        for (const Instruction* src_inst : *src_bb) {
            if (!src_inst) continue;
            Instruction* dst_inst = inst_map[src_inst];

            for (const Value* op : src_inst->operands()) dst_inst->add_operand(map_value(op));
            for (const Value* sv : src_inst->state_map()) dst_inst->add_state_value(map_value(sv));

            dst_inst->set_branch_target(map_target(src_inst->branch_target()));
            dst_inst->set_true_target(map_target(src_inst->true_target()));
            dst_inst->set_false_target(map_target(src_inst->false_target()));
            dst_inst->set_default_target(map_target(src_inst->default_target()));
            dst_inst->set_normal_target(map_target(src_inst->normal_target()));
            dst_inst->set_unwind_target(map_target(src_inst->unwind_target()));
            for (const auto& sc : src_inst->switch_cases()) {
                dst_inst->add_switch_case(sc.value, map_target(sc.target));
            }
        }
    }

    for (const auto& rp : src.resume_points()) {
        if (rp.second) dst_fn->add_resume_point(rp.first, block_map[rp.second]);
    }

    dst_fn->set_allow_fp_reassociation(src.allow_fp_reassociation());
    dst_fn->rebuild_cfg_predecessors();
    return dst_fn;
}

void clone_callee_closure(const Module& src, Module& dst, const std::vector<const BasicBlock*>& blocks,
                          const std::vector<std::string>& also, const Function* exclude) {
    // The functions the blocks name directly and those of `also`, with
    // their bodies; and, for every body copied, the exit stubs its guards
    // name, which a guard's code is compiled together with.
    std::vector<const Function*> bodies;
    std::unordered_set<const Function*> copied;
    if (exclude) copied.insert(exclude);
    auto copy = [&](std::string_view name) {
        if (name.empty()) return;
        const Function* f = src.get_function(name);
        if (f && f->block_count() != 0 && !dst.get_function(f->name()) && copied.insert(f).second) {
            bodies.push_back(f);
        }
    };
    for (const BasicBlock* bb : blocks) {
        if (!bb) continue;
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (!inst) continue;
            copy(inst->symbol());
            copy(inst->extra_symbol());
        }
    }
    for (const std::string& name : also) copy(name);
    for (size_t i = 0; i < bodies.size(); ++i) {
        for (const BasicBlock* bb : bodies[i]->blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (inst && inst->opcode() == Opcode::guard) copy(inst->symbol());
            }
        }
    }
    for (const Function* f : bodies) clone_function(*f, dst);

    // Every other module function the blocks and those bodies name becomes
    // an external symbol of the copy, in the order it is first named.
    std::unordered_set<const Function*> seen;
    auto note = [&](std::string_view sym) {
        if (sym.empty()) return;
        const Function* f = src.get_function(sym);
        if (f && !dst.get_function(f->name()) && seen.insert(f).second) dst.add_external_symbol(f->name());
    };
    auto note_block = [&](const BasicBlock* bb) {
        if (!bb) return;
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (!inst) continue;
            note(inst->symbol());
            note(inst->extra_symbol());
        }
    };
    for (const BasicBlock* bb : blocks) note_block(bb);
    for (const Function* f : bodies) {
        for (const BasicBlock* bb : f->blocks()) note_block(bb);
    }
}

std::unique_ptr<Module> clone_function_module(const Module& src, std::string_view fn_name,
                                              const std::vector<std::string>& also) {
    const Function* target = src.get_function(fn_name);
    if (!target) return nullptr;
    auto dst = std::make_unique<Module>(src.name());
    dst->set_allow_fp_reassociation(src.allow_fp_reassociation());
    dst->set_pinned_tls_register(src.pinned_tls_register());
    dst->copy_declarations_from(src);
    clone_function(*target, *dst);
    const std::vector<const BasicBlock*> blocks(target->blocks().begin(), target->blocks().end());
    clone_callee_closure(src, *dst, blocks, also, target);
    return dst;
}

std::unique_ptr<Module> clone_module(const Module& src) {
    auto dst = std::make_unique<Module>(src.name());
    dst->set_allow_fp_reassociation(src.allow_fp_reassociation());
    dst->set_pinned_tls_register(src.pinned_tls_register());
    dst->copy_declarations_from(src);
    for (const Function* fn : src.functions()) if (fn) clone_function(*fn, *dst);
    return dst;
}

} // namespace brass
