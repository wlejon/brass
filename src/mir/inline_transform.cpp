#include <brass/mir/inline_transform.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/uses.hpp>
#include <unordered_map>
#include <string>
#include <algorithm>

namespace brass {

InlineResult inline_call_site(Function& caller, Instruction* call_inst, const Function& callee, DebugContext* dbg_ctx) {
    InlineResult result;
    if (!call_inst || &caller == &callee) return result;

    BasicBlock* caller_bb = call_inst->parent();
    if (!caller_bb || caller_bb->parent() != &caller) return result;
    if (callee.blocks().empty() || !callee.entry_block()) return result;
    if (!caller.parent()) return result;

    // Check parameter count
    if (call_inst->operand_count() != callee.param_count()) return result;

    // A guard deopts the function it is in: its exit stub's result is that
    // function's result and its resume id names that function's Tier-0
    // guard. Copied into the caller it would finish the caller with the
    // callee's exit; instead the copy branches, where it fails, to a call of
    // the exit stub, whose result is the inlined call's. A guard with no
    // stub (one resuming in a Tier-0 frame) keeps its callee out of line, as
    // does any guard at an invoke site (the stub may raise).
    const bool is_invoke_site = call_inst->opcode() == Opcode::invoke;
    for (const BasicBlock* bb : callee.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst && inst->opcode() == Opcode::guard && (is_invoke_site || !callee.guard_exit_stub(*inst))) {
                return result;
            }
        }
    }

    // At an invoke site every way the callee can raise must end at the
    // invoke's handler. Plain calls are rewritten into invokes below; a
    // throw, a resume, or any other kind of call would unwind straight past
    // the handler, so such callees are not inlined there.
    if (call_inst->opcode() == Opcode::invoke) {
        for (const BasicBlock* bb : callee.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (!inst) continue;
                const Opcode op = inst->opcode();
                if (op == Opcode::throw_ || op == Opcode::resume ||
                    (is_call(op) && op != Opcode::call && op != Opcode::invoke)) {
                    return result;
                }
            }
        }
    }

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

    bool is_invoke = (call_inst->opcode() == Opcode::invoke);
    BranchTarget normal_target = is_invoke ? call_inst->normal_target() : BranchTarget{};
    BranchTarget unwind_target = is_invoke ? call_inst->unwind_target() : BranchTarget{};

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

    if (is_invoke) {
        Instruction* br_norm = mod->arena().make<Instruction>(Opcode::br, Type::void_type());
        br_norm->set_loc(call_inst->loc());
        br_norm->set_branch_target(normal_target);
        split_tail->append_instruction(br_norm);
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
        cloned_bb->set_cold(src_bb->is_cold());
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

    // 6. Clone callee instructions into cloned blocks (two passes to resolve any out-of-order block references)
    std::vector<std::pair<const Instruction*, Instruction*>> inst_pairs;
    // Each guard copied as a branch and a call of its exit stub, whose
    // operands (the condition, the state) are mapped with the rest below.
    struct GuardExit {
        const Instruction* guard;
        Instruction* branch;
        Instruction* stub_call;
    };
    std::vector<GuardExit> guard_exits;
    std::unordered_map<const BasicBlock*, BasicBlock*> final_block_map;
    for (const BasicBlock* src_bb : callee.blocks()) {
        if (!src_bb) continue;
        BasicBlock* dst_bb = block_map[src_bb];

        for (const Instruction* src_inst : *src_bb) {
            if (!src_inst) continue;

            if (src_inst->opcode() == Opcode::ret) {
                continue;
            }

            if (is_invoke && src_inst->opcode() == Opcode::call) {
                // If inlining an invoke call site, calls inside callee can throw and
                // must unwind to caller's invoke landing pad (unwind_target).
                BasicBlock* cont_bb = mod->arena().make<BasicBlock>(
                    caller.next_block_id(),
                    mod->string_pool().intern(std::string(dst_bb->name()) + ".call_cont")
                );
                cont_bb->set_parent(&caller);
                cont_bb->set_cold(dst_bb->is_cold());
                cloned_blocks.push_back(cont_bb);

                Instruction* dst_inst = mod->arena().make<Instruction>(Opcode::invoke, src_inst->type());
                dst_inst->set_loc(wrap_loc(src_inst->loc()));
                dst_inst->copy_attributes_from(*src_inst);
                if (!src_inst->symbol().empty()) {
                    dst_inst->set_symbol(mod->string_pool().intern(src_inst->symbol()));
                }
                if (!src_inst->extra_symbol().empty()) {
                    dst_inst->set_extra_symbol(mod->string_pool().intern(src_inst->extra_symbol()));
                }
                dst_inst->set_normal_target(BranchTarget(cont_bb, {}));
                dst_inst->set_unwind_target(unwind_target);

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

                dst_bb->append_instruction(dst_inst);
                inst_pairs.emplace_back(src_inst, dst_inst);

                dst_bb = cont_bb;
                continue;
            }

            if (src_inst->opcode() == Opcode::guard) {
                // br_if %cond, cont, exit; exit: the stub's result is the call's.
                auto make_block = [&](std::string_view suffix) {
                    BasicBlock* nb = mod->arena().make<BasicBlock>(
                        caller.next_block_id(),
                        mod->string_pool().intern(std::string(dst_bb->name()) + std::string(suffix)));
                    nb->set_parent(&caller);
                    cloned_blocks.push_back(nb);
                    return nb;
                };
                BasicBlock* cont_bb = make_block(".guard_ok");
                cont_bb->set_cold(dst_bb->is_cold());
                BasicBlock* exit_bb = make_block(".guard_exit");
                exit_bb->set_cold();
                Instruction* br = mod->arena().make<Instruction>(Opcode::br_if, Type::void_type());
                br->set_loc(wrap_loc(src_inst->loc()));
                br->set_true_target(BranchTarget(cont_bb, {}));
                br->set_false_target(BranchTarget(exit_bb, {}));
                dst_bb->append_instruction(br);

                const Function* stub = callee.guard_exit_stub(*src_inst);
                Instruction* stub_call = mod->arena().make<Instruction>(Opcode::call, stub->return_type());
                stub_call->set_loc(wrap_loc(src_inst->loc()));
                stub_call->set_symbol(mod->string_pool().intern(stub->name()));
                std::vector<Value*> tail_args;
                if (!stub->return_type().is_void()) {
                    Value* r = mod->arena().make<Value>(caller.next_value_id(), stub->return_type(),
                                                        ValueKind::InstructionResult);
                    r->set_defining_instruction(stub_call);
                    stub_call->set_result(r);
                    if (ret_param) tail_args.push_back(r);
                }
                exit_bb->append_instruction(stub_call);
                Instruction* to_tail = mod->arena().make<Instruction>(Opcode::br, Type::void_type());
                to_tail->set_loc(wrap_loc(src_inst->loc()));
                to_tail->set_branch_target(BranchTarget(split_tail, std::move(tail_args)));
                exit_bb->append_instruction(to_tail);
                guard_exits.push_back({src_inst, br, stub_call});

                dst_bb = cont_bb;
                continue;
            }

            Instruction* dst_inst = mod->arena().make<Instruction>(src_inst->opcode(), src_inst->type());
            dst_inst->set_loc(wrap_loc(src_inst->loc()));
            dst_inst->copy_attributes_from(*src_inst);
            if (!src_inst->symbol().empty()) {
                dst_inst->set_symbol(mod->string_pool().intern(src_inst->symbol()));
            }
            if (!src_inst->extra_symbol().empty()) {
                dst_inst->set_extra_symbol(mod->string_pool().intern(src_inst->extra_symbol()));
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

            // The callee's frame slots (its entry's allocas) become the
            // caller's: allocated once, in its entry, not per call.
            if (src_inst->opcode() == Opcode::alloca_ && src_bb == callee_entry) {
                caller.entry_block()->prepend_instruction(dst_inst);
            } else {
                dst_bb->append_instruction(dst_inst);
            }
            inst_pairs.emplace_back(src_inst, dst_inst);
        }
        final_block_map[src_bb] = dst_bb;
    }

    // 6b. Second pass: map all operands, state maps, and branch targets
    for (const auto& [src_inst, dst_inst] : inst_pairs) {
        for (const Value* op : src_inst->operands()) {
            dst_inst->add_operand(map_val(op));
        }
        for (const Value* sv : src_inst->state_map()) {
            dst_inst->add_state_value(map_val(sv));
        }

        if (dst_inst->opcode() == Opcode::invoke && src_inst->opcode() == Opcode::call) {
            continue;
        }

        if (src_inst->opcode() == Opcode::br) {
            dst_inst->set_branch_target(map_target(src_inst->branch_target()));
        } else if (src_inst->opcode() == Opcode::br_if) {
            dst_inst->set_true_target(map_target(src_inst->true_target()));
            dst_inst->set_false_target(map_target(src_inst->false_target()));
        } else if (src_inst->opcode() == Opcode::switch_) {
            dst_inst->set_default_target(map_target(src_inst->default_target()));
            for (const auto& sc : src_inst->switch_cases()) {
                dst_inst->add_switch_case(sc.value, map_target(sc.target));
            }
        } else if (src_inst->opcode() == Opcode::invoke) {
            dst_inst->set_normal_target(map_target(src_inst->normal_target()));
            dst_inst->set_unwind_target(map_target(src_inst->unwind_target()));
        }
    }
    for (const GuardExit& ge : guard_exits) {
        ge.branch->add_operand(map_val(ge.guard->operand(0)));
        for (const Value* sv : ge.guard->state_map()) ge.stub_call->add_operand(map_val(sv));
    }

    // 6c. Remap ret instructions to br split_tail(...)
    for (const BasicBlock* src_bb : callee.blocks()) {
        if (!src_bb) continue;
        BasicBlock* dst_bb = final_block_map.count(src_bb) ? final_block_map[src_bb] : block_map[src_bb];
        for (const Instruction* src_inst : *src_bb) {
            if (src_inst && src_inst->opcode() == Opcode::ret) {
                Instruction* br_tail = mod->arena().make<Instruction>(Opcode::br, Type::void_type());
                br_tail->set_loc(wrap_loc(src_inst->loc()));
                std::vector<Value*> br_args;
                if (!callee.return_type().is_void() && src_inst->operand_count() > 0) {
                    br_args.push_back(map_val(src_inst->operand(0)));
                }
                br_tail->set_branch_target(BranchTarget(split_tail, std::move(br_args)));
                dst_bb->append_instruction(br_tail);
            }
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
        replace_all_uses(caller, old_call_res, ret_param);
        // The split tail may not be in the caller's block list yet.
        replace_uses_in(*split_tail, old_call_res, ret_param);
    }

    // 11. Rebuild CFG predecessors
    caller.rebuild_cfg_predecessors();

    result.inlined_blocks = std::move(cloned_blocks);
    result.success = true;
    return result;
}

} // namespace brass
