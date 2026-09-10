#include <brass/mir/inline_transform.hpp>
#include <brass/mir/module.hpp>
#include <unordered_map>
#include <string>
#include <algorithm>

namespace brass {

namespace {

void replace_all_uses_in_fn(Function& fn, Value* old_val, Value* new_val) {
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

} // namespace

InlineResult inline_call_site(Function& caller, Instruction* call_inst, const Function& callee, DebugContext* dbg_ctx) {
    InlineResult result;
    if (!call_inst || &caller == &callee) return result;

    BasicBlock* caller_bb = call_inst->parent();
    if (!caller_bb || caller_bb->parent() != &caller) return result;
    if (callee.blocks().empty() || !callee.entry_block()) return result;
    if (!caller.parent()) return result;

    // Check parameter count
    if (call_inst->operand_count() != callee.param_count()) return result;

    Module* mod = caller.parent();
    if (!dbg_ctx && mod) {
        dbg_ctx = &mod->debug_context();
    }
    uint32_t inline_scope_id = 0;
    if (dbg_ctx) {
        DebugLoc callsite_loc = call_inst->loc();
        inline_scope_id = dbg_ctx->record_inlined_scope(
            std::string(callee.name()),
            callsite_loc,
            callsite_loc.inlined_at_id
        );
    }
    BasicBlock* split_head = caller_bb;
    result.split_head = split_head;

    // 1. Create split_tail block
    std::string tail_name = std::string(caller_bb->name()) + ".split_tail";
    BasicBlock* split_tail = mod->arena().make<BasicBlock>(
        caller.next_block_id(),
        mod->string_pool().intern(tail_name)
    );
    split_tail->set_parent(&caller);
    result.split_tail = split_tail;

    // 2. Move instructions strictly after call_inst to split_tail
    std::vector<Instruction*> tail_insts;
    Instruction* cur = call_inst->next();
    while (cur) {
        tail_insts.push_back(cur);
        cur = cur->next();
    }
    for (Instruction* ti : tail_insts) {
        caller_bb->remove_instruction(ti);
        split_tail->append_instruction(ti);
    }

    // 3. Setup return value forwarding via split_tail block parameter
    Value* old_call_res = call_inst->result();
    Value* ret_param = nullptr;
    if (call_inst->produces_value() && !callee.return_type().is_void()) {
        ret_param = mod->arena().make<Value>(
            caller.next_value_id(),
            call_inst->type(),
            ValueKind::BlockParam
        );
        split_tail->add_param(ret_param);
        result.return_value = ret_param;
    }

    // Detach call_inst from split_head
    caller_bb->remove_instruction(call_inst);

    // 4. Map callee parameters to caller call arguments
    std::unordered_map<const Value*, Value*> value_map;
    const BasicBlock* callee_entry = callee.entry_block();
    for (size_t i = 0; i < callee.param_count(); ++i) {
        value_map[callee_entry->param(i)] = call_inst->operand(i);
    }

    // 5. Clone callee basic blocks
    std::unordered_map<const BasicBlock*, BasicBlock*> block_map;
    std::vector<BasicBlock*> cloned_blocks;
    for (const BasicBlock* src_bb : callee.blocks()) {
        if (!src_bb) continue;

        std::string cloned_name = std::string(callee.name()) + "." + std::string(src_bb->name());
        BasicBlock* cloned_bb = mod->arena().make<BasicBlock>(
            caller.next_block_id(),
            mod->string_pool().intern(cloned_name)
        );
        cloned_bb->set_parent(&caller);
        block_map[src_bb] = cloned_bb;
        cloned_blocks.push_back(cloned_bb);

        if (src_bb != callee_entry) {
            for (size_t i = 0; i < src_bb->param_count(); ++i) {
                const Value* src_p = src_bb->param(i);
                Value* dst_p = mod->arena().make<Value>(
                    caller.next_value_id(),
                    src_p->type(),
                    ValueKind::BlockParam
                );
                cloned_bb->add_param(dst_p);
                value_map[src_p] = dst_p;
            }
        }
    }

    auto map_val = [&](const Value* v) -> Value* {
        if (!v) return nullptr;
        auto it = value_map.find(v);
        return (it != value_map.end()) ? it->second : const_cast<Value*>(v);
    };

    auto map_target = [&](const BranchTarget& src_bt) -> BranchTarget {
        BranchTarget dst_bt;
        if (src_bt.block) {
            auto it = block_map.find(src_bt.block);
            dst_bt.block = (it != block_map.end()) ? it->second : src_bt.block;
        }
        for (const Value* arg : src_bt.args) {
            dst_bt.args.push_back(map_val(arg));
        }
        return dst_bt;
    };

    auto wrap_loc = [&](DebugLoc src_loc) -> DebugLoc {
        if (!src_loc.is_valid()) return src_loc;
        if (!dbg_ctx || inline_scope_id == 0) return src_loc;
        DebugLoc wrapped = src_loc;
        if (src_loc.inlined_at_id == 0) {
            wrapped.inlined_at_id = inline_scope_id;
        } else {
            wrapped.inlined_at_id = dbg_ctx->wrap_inlined_scope(src_loc.inlined_at_id, inline_scope_id);
        }
        return wrapped;
    };

    // 6. Clone callee instructions into cloned blocks
    for (const BasicBlock* src_bb : callee.blocks()) {
        if (!src_bb) continue;
        BasicBlock* dst_bb = block_map[src_bb];

        for (const Instruction* src_inst : *src_bb) {
            if (!src_inst) continue;

            if (src_inst->opcode() == Opcode::ret) {
                // Remap ret to br split_tail(...)
                Instruction* br_tail = mod->arena().make<Instruction>(Opcode::br, Type::void_type());
                br_tail->set_loc(wrap_loc(src_inst->loc()));
                std::vector<Value*> br_args;
                if (!callee.return_type().is_void() && src_inst->operand_count() > 0) {
                    br_args.push_back(map_val(src_inst->operand(0)));
                }
                br_tail->set_branch_target(BranchTarget(split_tail, std::move(br_args)));
                dst_bb->append_instruction(br_tail);
                continue;
            }

            Instruction* dst_inst = mod->arena().make<Instruction>(src_inst->opcode(), src_inst->type());
            dst_inst->set_loc(wrap_loc(src_inst->loc()));
            dst_inst->set_imm_i64(src_inst->imm_i64());
            dst_inst->set_imm_f64(src_inst->imm_f64());
            dst_inst->set_scale(src_inst->scale());
            dst_inst->set_offset(src_inst->offset());
            dst_inst->set_memory_type(src_inst->memory_type());
            if (!src_inst->symbol().empty()) {
                dst_inst->set_symbol(mod->string_pool().intern(src_inst->symbol()));
            }
            if (!src_inst->extra_symbol().empty()) {
                dst_inst->set_extra_symbol(mod->string_pool().intern(src_inst->extra_symbol()));
            }

            for (const Value* op : src_inst->operands()) {
                dst_inst->add_operand(map_val(op));
            }
            for (const Value* sv : src_inst->state_map()) {
                dst_inst->add_state_value(map_val(sv));
            }

            if (src_inst->produces_value()) {
                const Value* src_res = src_inst->result();
                Value* dst_res = mod->arena().make<Value>(
                    caller.next_value_id(),
                    src_res->type(),
                    ValueKind::InstructionResult
                );
                dst_res->set_defining_instruction(dst_inst);
                dst_inst->set_result(dst_res);
                value_map[src_res] = dst_res;
            }

            dst_inst->set_branch_target(map_target(src_inst->branch_target()));
            dst_inst->set_true_target(map_target(src_inst->true_target()));
            dst_inst->set_false_target(map_target(src_inst->false_target()));
            dst_inst->set_default_target(map_target(src_inst->default_target()));
            for (const auto& sc : src_inst->switch_cases()) {
                dst_inst->add_switch_case(sc.value, map_target(sc.target));
            }

            dst_bb->append_instruction(dst_inst);
        }
    }

    // 7. Connect split_head -> cloned callee entry block
    Instruction* br_callee_entry = mod->arena().make<Instruction>(Opcode::br, Type::void_type());
    br_callee_entry->set_loc(call_inst->loc());
    br_callee_entry->set_branch_target(BranchTarget(block_map[callee_entry], {}));
    split_head->append_instruction(br_callee_entry);

    // 8. Insert cloned blocks and split_tail into caller.blocks() right after split_head
    auto it = std::find(caller.blocks().begin(), caller.blocks().end(), split_head);
    if (it != caller.blocks().end()) {
        auto tail_it = caller.blocks().insert(it + 1, split_tail);
        caller.blocks().insert(tail_it, cloned_blocks.begin(), cloned_blocks.end());
    } else {
        for (BasicBlock* cb : cloned_blocks) caller.append_block(cb);
        caller.append_block(split_tail);
    }

    // 9. Forward resume points from callee if any
    for (const auto& rp : callee.resume_points()) {
        if (rp.second) {
            BasicBlock* mapped_bb = block_map[rp.second];
            if (mapped_bb) {
                caller.add_resume_point(rp.first, mapped_bb);
            }
        }
    }

    // 10. Replace all uses of old_call_res with ret_param
    if (old_call_res && ret_param) {
        replace_all_uses_in_fn(caller, old_call_res, ret_param);
        for (Instruction* ti : *split_tail) {
            if (!ti) continue;
            for (size_t i = 0; i < ti->operand_count(); ++i) {
                if (ti->operand(i) == old_call_res) ti->set_operand(i, ret_param);
            }
            for (size_t i = 0; i < ti->branch_target().args.size(); ++i) {
                if (ti->branch_target().args[i] == old_call_res) ti->branch_target().args[i] = ret_param;
            }
            for (size_t i = 0; i < ti->true_target().args.size(); ++i) {
                if (ti->true_target().args[i] == old_call_res) ti->true_target().args[i] = ret_param;
            }
            for (size_t i = 0; i < ti->false_target().args.size(); ++i) {
                if (ti->false_target().args[i] == old_call_res) ti->false_target().args[i] = ret_param;
            }
            for (size_t i = 0; i < ti->default_target().args.size(); ++i) {
                if (ti->default_target().args[i] == old_call_res) ti->default_target().args[i] = ret_param;
            }
            for (auto& sc : ti->switch_cases()) {
                for (size_t i = 0; i < sc.target.args.size(); ++i) {
                    if (sc.target.args[i] == old_call_res) sc.target.args[i] = ret_param;
                }
            }
            for (size_t i = 0; i < ti->state_map().size(); ++i) {
                if (ti->state_map()[i] == old_call_res) ti->state_map()[i] = ret_param;
            }
        }
    }

    // 11. Rebuild CFG predecessors
    caller.rebuild_cfg_predecessors();

    result.inlined_blocks = std::move(cloned_blocks);
    result.success = true;
    return result;
}

} // namespace brass
