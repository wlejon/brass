#include <brass/mir/speculative_inliner.hpp>
#include <brass/mir/inline_transform.hpp>
#include <brass/mir/verifier.hpp>
#include <algorithm>
#include <vector>
#include <unordered_set>

namespace brass {

namespace {

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

bool devirtualize_monomorphic_call(
    Function& fn,
    Module& mod,
    Instruction* call_inst,
    const runtime::FeedbackSlot* slot,
    const SpeculativeInlinerOptions& opts
) {
    if (!call_inst || !slot) return false;
    const runtime::CallFeedback* target = slot->get_monomorphic_target();
    if (!target || target->target_name.empty()) return false;

    std::string callee_name = target->target_name;
    if (callee_name.starts_with('@')) {
        callee_name = callee_name.substr(1);
    }
    Function* callee_fn = mod.get_function(callee_name);
    if (!callee_fn) return false;

    BasicBlock* cur_bb = call_inst->parent();
    if (!cur_bb || cur_bb->parent() != &fn) return false;

    if (call_inst->opcode() == Opcode::call_indirect) {
        if (call_inst->operand_count() < 1) return false;
        size_t call_args_count = call_inst->operand_count() - 1;
        if (callee_fn->param_count() != call_args_count) return false;
        if (callee_fn->return_type() != call_inst->type()) return false;

        Value* callee_val = call_inst->operand(0);

        // 1. %expected_fn = func_addr @callee_fn
        Instruction* expected_fn = mod.arena().make<Instruction>(Opcode::func_addr, Type::ptr());
        expected_fn->set_symbol(mod.string_pool().intern(callee_name));
        expected_fn->set_loc(call_inst->loc());
        Value* expected_res = mod.arena().make<Value>(fn.next_value_id(), Type::ptr(), ValueKind::InstructionResult);
        expected_res->set_defining_instruction(expected_fn);
        expected_fn->set_result(expected_res);
        cur_bb->insert_before(expected_fn, call_inst);

        // 2. %cond = eq.ptr %callee_val, %expected_fn
        Instruction* cond_inst = mod.arena().make<Instruction>(Opcode::eq, Type::i32());
        cond_inst->add_operand(callee_val);
        cond_inst->add_operand(expected_res);
        cond_inst->set_loc(call_inst->loc());
        Value* cond_res = mod.arena().make<Value>(fn.next_value_id(), Type::i32(), ValueKind::InstructionResult);
        cond_res->set_defining_instruction(cond_inst);
        cond_inst->set_result(cond_res);
        cur_bb->insert_before(cond_inst, call_inst);

        // 3. guard %cond, @deopt_slow_call
        Instruction* guard_inst = mod.arena().make<Instruction>(Opcode::guard, Type::void_type());
        guard_inst->add_operand(cond_res);
        guard_inst->set_loc(call_inst->loc());
        std::string deopt_label = opts.deopt_stub_prefix.empty() ? "@deopt_slow_call" : opts.deopt_stub_prefix;
        guard_inst->set_symbol(mod.string_pool().intern(deopt_label));
        for (Value* sv : call_inst->state_map()) {
            guard_inst->add_state_value(sv);
        }
        for (size_t i = 1; i < call_inst->operand_count(); ++i) {
            guard_inst->add_state_value(call_inst->operand(i));
        }
        cur_bb->insert_before(guard_inst, call_inst);

        // 4. Convert call_indirect to direct call
        call_inst->set_opcode(Opcode::call);
        call_inst->set_symbol(mod.string_pool().intern(callee_name));
        call_inst->operands().erase(call_inst->operands().begin());

        // 5. Speculative inlining if requested and callee satisfies heuristics
        if (opts.enable_inlining && callee_fn != &fn && !callee_fn->blocks().empty() && callee_fn->entry_block()) {
            size_t callee_size = 0;
            for (const BasicBlock* b : callee_fn->blocks()) {
                if (b) callee_size += b->instruction_count();
            }
            if (callee_size <= opts.max_callee_instruction_count) {
                inline_call_site(fn, call_inst, *callee_fn);
            }
        }
        return true;
    } else if (call_inst->opcode() == Opcode::patchable_call) {
        if (callee_fn->param_count() != call_inst->operand_count()) return false;
        if (callee_fn->return_type() != call_inst->type()) return false;

        call_inst->set_opcode(Opcode::call);
        call_inst->set_symbol(mod.string_pool().intern(callee_name));
        call_inst->set_extra_symbol("");

        if (opts.enable_inlining && callee_fn != &fn && !callee_fn->blocks().empty() && callee_fn->entry_block()) {
            size_t callee_size = 0;
            for (const BasicBlock* b : callee_fn->blocks()) {
                if (b) callee_size += b->instruction_count();
            }
            if (callee_size <= opts.max_callee_instruction_count) {
                inline_call_site(fn, call_inst, *callee_fn);
            }
        }
        return true;
    }
    return false;
}

bool specialize_polymorphic_call(
    Function& fn,
    Module& mod,
    Instruction* call_inst,
    const runtime::FeedbackSlot* slot,
    const SpeculativeInlinerOptions& opts
) {
    if (!call_inst || !slot || call_inst->opcode() != Opcode::call_indirect) return false;
    if (slot->targets.size() < 2) return false;

    // Specialize Degree 2
    const auto& target0 = slot->targets[0];
    const auto& target1 = slot->targets[1];

    std::string nameA = target0.target_name;
    if (nameA.starts_with('@')) nameA = nameA.substr(1);
    std::string nameB = target1.target_name;
    if (nameB.starts_with('@')) nameB = nameB.substr(1);

    Function* fnA = mod.get_function(nameA);
    Function* fnB = mod.get_function(nameB);
    if (!fnA || !fnB) return false;

    size_t call_args_count = call_inst->operand_count() > 0 ? call_inst->operand_count() - 1 : 0;
    if (fnA->param_count() != call_args_count || fnB->param_count() != call_args_count) return false;
    if (fnA->return_type() != call_inst->type() || fnB->return_type() != call_inst->type()) return false;

    BasicBlock* cur_bb = call_inst->parent();
    if (!cur_bb || cur_bb->parent() != &fn) return false;

    Value* callee_val = call_inst->operand(0);
    std::vector<Value*> call_args;
    call_args.reserve(call_args_count);
    for (size_t i = 1; i < call_inst->operand_count(); ++i) {
        call_args.push_back(call_inst->operand(i));
    }
    Type ret_type = call_inst->type();
    Value* old_res = call_inst->result();

    // 1. Create split_tail block
    std::string tail_name = std::string(cur_bb->name()) + ".poly_tail";
    BasicBlock* split_tail = mod.arena().make<BasicBlock>(
        fn.next_block_id(),
        mod.string_pool().intern(tail_name)
    );
    split_tail->set_parent(&fn);

    // Move instructions strictly after call_inst to split_tail
    std::vector<Instruction*> tail_insts;
    Instruction* cur = call_inst->next();
    while (cur) {
        tail_insts.push_back(cur);
        cur = cur->next();
    }
    for (Instruction* ti : tail_insts) {
        cur_bb->remove_instruction(ti);
        split_tail->append_instruction(ti);
    }

    // Return value block parameter on split_tail if non-void
    Value* merged_ret = nullptr;
    if (call_inst->produces_value()) {
        merged_ret = mod.arena().make<Value>(
            fn.next_value_id(),
            ret_type,
            ValueKind::BlockParam
        );
        split_tail->add_param(merged_ret);
    }

    // Detach call_inst from cur_bb
    cur_bb->remove_instruction(call_inst);

    // 2. Create bb_fnA
    std::string fnA_block_name = std::string(cur_bb->name()) + ".target_A";
    BasicBlock* bb_fnA = mod.arena().make<BasicBlock>(
        fn.next_block_id(),
        mod.string_pool().intern(fnA_block_name)
    );
    bb_fnA->set_parent(&fn);

    Instruction* callA = mod.arena().make<Instruction>(Opcode::call, ret_type);
    callA->set_symbol(mod.string_pool().intern(nameA));
    callA->set_loc(call_inst->loc());
    for (Value* arg : call_args) {
        callA->add_operand(arg);
    }
    Value* resA = nullptr;
    if (!ret_type.is_void()) {
        resA = mod.arena().make<Value>(fn.next_value_id(), ret_type, ValueKind::InstructionResult);
        resA->set_defining_instruction(callA);
        callA->set_result(resA);
    }
    bb_fnA->append_instruction(callA);

    Instruction* br_tailA = mod.arena().make<Instruction>(Opcode::br, Type::void_type());
    br_tailA->set_loc(call_inst->loc());
    std::vector<Value*> brA_args;
    if (resA) brA_args.push_back(resA);
    br_tailA->set_branch_target(BranchTarget(split_tail, std::move(brA_args)));
    bb_fnA->append_instruction(br_tailA);

    // 3. Create bb_checkB
    std::string checkB_block_name = std::string(cur_bb->name()) + ".check_B";
    BasicBlock* bb_checkB = mod.arena().make<BasicBlock>(
        fn.next_block_id(),
        mod.string_pool().intern(checkB_block_name)
    );
    bb_checkB->set_parent(&fn);

    // 4. Create bb_fnB
    std::string fnB_block_name = std::string(cur_bb->name()) + ".target_B";
    BasicBlock* bb_fnB = mod.arena().make<BasicBlock>(
        fn.next_block_id(),
        mod.string_pool().intern(fnB_block_name)
    );
    bb_fnB->set_parent(&fn);

    Instruction* callB = mod.arena().make<Instruction>(Opcode::call, ret_type);
    callB->set_symbol(mod.string_pool().intern(nameB));
    callB->set_loc(call_inst->loc());
    for (Value* arg : call_args) {
        callB->add_operand(arg);
    }
    Value* resB = nullptr;
    if (!ret_type.is_void()) {
        resB = mod.arena().make<Value>(fn.next_value_id(), ret_type, ValueKind::InstructionResult);
        resB->set_defining_instruction(callB);
        callB->set_result(resB);
    }
    bb_fnB->append_instruction(callB);

    Instruction* br_tailB = mod.arena().make<Instruction>(Opcode::br, Type::void_type());
    br_tailB->set_loc(call_inst->loc());
    std::vector<Value*> brB_args;
    if (resB) brB_args.push_back(resB);
    br_tailB->set_branch_target(BranchTarget(split_tail, std::move(brB_args)));
    bb_fnB->append_instruction(br_tailB);

    // 5. Create bb_fallback
    std::string fb_block_name = std::string(cur_bb->name()) + ".fallback";
    BasicBlock* bb_fallback = mod.arena().make<BasicBlock>(
        fn.next_block_id(),
        mod.string_pool().intern(fb_block_name)
    );
    bb_fallback->set_parent(&fn);

    Instruction* callFallback = mod.arena().make<Instruction>(Opcode::call_indirect, ret_type);
    callFallback->set_site_id(UINT32_MAX);
    callFallback->set_loc(call_inst->loc());
    callFallback->add_operand(callee_val);
    for (Value* arg : call_args) {
        callFallback->add_operand(arg);
    }
    Value* resFallback = nullptr;
    if (!ret_type.is_void()) {
        resFallback = mod.arena().make<Value>(fn.next_value_id(), ret_type, ValueKind::InstructionResult);
        resFallback->set_defining_instruction(callFallback);
        callFallback->set_result(resFallback);
    }
    bb_fallback->append_instruction(callFallback);

    Instruction* br_tailFb = mod.arena().make<Instruction>(Opcode::br, Type::void_type());
    br_tailFb->set_loc(call_inst->loc());
    std::vector<Value*> brFb_args;
    if (resFallback) brFb_args.push_back(resFallback);
    br_tailFb->set_branch_target(BranchTarget(split_tail, std::move(brFb_args)));
    bb_fallback->append_instruction(br_tailFb);

    // 6. Connect cur_bb -> if (callee_val == addrA) bb_fnA else bb_checkB
    Instruction* addrA = mod.arena().make<Instruction>(Opcode::func_addr, Type::ptr());
    addrA->set_symbol(mod.string_pool().intern(nameA));
    addrA->set_loc(call_inst->loc());
    Value* resAddrA = mod.arena().make<Value>(fn.next_value_id(), Type::ptr(), ValueKind::InstructionResult);
    resAddrA->set_defining_instruction(addrA);
    addrA->set_result(resAddrA);
    cur_bb->append_instruction(addrA);

    Instruction* eqA = mod.arena().make<Instruction>(Opcode::eq, Type::i32());
    eqA->add_operand(callee_val);
    eqA->add_operand(resAddrA);
    eqA->set_loc(call_inst->loc());
    Value* resCondA = mod.arena().make<Value>(fn.next_value_id(), Type::i32(), ValueKind::InstructionResult);
    resCondA->set_defining_instruction(eqA);
    eqA->set_result(resCondA);
    cur_bb->append_instruction(eqA);

    Instruction* brIfA = mod.arena().make<Instruction>(Opcode::br_if, Type::void_type());
    brIfA->add_operand(resCondA);
    brIfA->set_loc(call_inst->loc());
    brIfA->set_true_target(BranchTarget(bb_fnA, {}));
    brIfA->set_false_target(BranchTarget(bb_checkB, {}));
    cur_bb->append_instruction(brIfA);

    // 7. Connect bb_checkB -> if (callee_val == addrB) bb_fnB else bb_fallback
    Instruction* addrB = mod.arena().make<Instruction>(Opcode::func_addr, Type::ptr());
    addrB->set_symbol(mod.string_pool().intern(nameB));
    addrB->set_loc(call_inst->loc());
    Value* resAddrB = mod.arena().make<Value>(fn.next_value_id(), Type::ptr(), ValueKind::InstructionResult);
    resAddrB->set_defining_instruction(addrB);
    addrB->set_result(resAddrB);
    bb_checkB->append_instruction(addrB);

    Instruction* eqB = mod.arena().make<Instruction>(Opcode::eq, Type::i32());
    eqB->add_operand(callee_val);
    eqB->add_operand(resAddrB);
    eqB->set_loc(call_inst->loc());
    Value* resCondB = mod.arena().make<Value>(fn.next_value_id(), Type::i32(), ValueKind::InstructionResult);
    resCondB->set_defining_instruction(eqB);
    eqB->set_result(resCondB);
    bb_checkB->append_instruction(eqB);

    Instruction* brIfB = mod.arena().make<Instruction>(Opcode::br_if, Type::void_type());
    brIfB->add_operand(resCondB);
    brIfB->set_loc(call_inst->loc());
    brIfB->set_true_target(BranchTarget(bb_fnB, {}));
    brIfB->set_false_target(BranchTarget(bb_fallback, {}));
    bb_checkB->append_instruction(brIfB);

    // 8. Insert new blocks into caller.blocks()
    auto it = std::find(fn.blocks().begin(), fn.blocks().end(), cur_bb);
    if (it != fn.blocks().end()) {
        std::vector<BasicBlock*> new_blocks = {bb_fnA, bb_checkB, bb_fnB, bb_fallback, split_tail};
        fn.blocks().insert(it + 1, new_blocks.begin(), new_blocks.end());
    } else {
        fn.append_block(bb_fnA);
        fn.append_block(bb_checkB);
        fn.append_block(bb_fnB);
        fn.append_block(bb_fallback);
        fn.append_block(split_tail);
    }

    // 9. Replace old result uses with merged_ret
    if (old_res && merged_ret) {
        replace_all_uses(fn, old_res, merged_ret);
    }

    // 10. Rebuild CFG predecessors
    fn.rebuild_cfg_predecessors();

    // 11. Optionally inline fnA and fnB
    if (opts.enable_inlining) {
        if (fnA != &fn && !fnA->blocks().empty() && fnA->entry_block()) {
            size_t sizeA = 0;
            for (const BasicBlock* b : fnA->blocks()) if (b) sizeA += b->instruction_count();
            if (sizeA <= opts.max_callee_instruction_count) {
                inline_call_site(fn, callA, *fnA);
            }
        }
        if (fnB != &fn && !fnB->blocks().empty() && fnB->entry_block()) {
            size_t sizeB = 0;
            for (const BasicBlock* b : fnB->blocks()) if (b) sizeB += b->instruction_count();
            if (sizeB <= opts.max_callee_instruction_count) {
                inline_call_site(fn, callB, *fnB);
            }
        }
    }

    return true;
}

} // namespace

bool run_speculative_devirtualization(
    Function& fn,
    Module& mod,
    const runtime::TypeFeedbackVector* tfv,
    const SpeculativeInlinerOptions& opts
) {
    if (fn.blocks().empty() || !fn.entry_block()) return false;
    if (!tfv || tfv->slots().empty()) return false;

    bool any_changed = false;
    std::unordered_set<uint32_t> processed_slots;
    constexpr size_t MAX_ROUNDS = 16;
    for (size_t round = 0; round < MAX_ROUNDS; ++round) {
        bool changed_in_round = false;
        fn.rebuild_cfg_predecessors();

        uint32_t call_counter = 0;
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (Instruction* inst : *bb) {
                if (!inst) continue;
                if (inst->opcode() != Opcode::call_indirect && inst->opcode() != Opcode::patchable_call) {
                    continue;
                }
                if (inst->site_id() == UINT32_MAX) {
                    continue;
                }
                call_counter++;
                uint32_t sid = inst->site_id();
                const runtime::FeedbackSlot* slot = nullptr;
                if (sid != 0) {
                    slot = tfv->find_slot(sid);
                }
                if (!slot) {
                    slot = tfv->find_slot(call_counter);
                }
                if (!slot) {
                    slot = tfv->find_slot(call_counter - 1);
                }
                if (!slot || slot->total_invocations < opts.min_invocations) {
                    continue;
                }
                if (processed_slots.contains(slot->site_id)) {
                    continue;
                }

                if (slot->is_monomorphic()) {
                    if (devirtualize_monomorphic_call(fn, mod, inst, slot, opts)) {
                        processed_slots.insert(slot->site_id);
                        changed_in_round = true;
                        any_changed = true;
                        break;
                    }
                } else if (slot->is_polymorphic() && opts.enable_polymorphic) {
                    if (specialize_polymorphic_call(fn, mod, inst, slot, opts)) {
                        processed_slots.insert(slot->site_id);
                        changed_in_round = true;
                        any_changed = true;
                        break;
                    }
                }
            }
            if (changed_in_round) break;
        }

        if (!changed_in_round) break;
    }

    if (any_changed) {
        fn.rebuild_cfg_predecessors();
    }
    return any_changed;
}

bool run_speculative_devirtualization(
    Module& mod,
    const runtime::FeedbackRegistry& registry,
    const SpeculativeInlinerOptions& opts
) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (!fn) continue;
        const runtime::TypeFeedbackVector* tfv = registry.find(fn->name());
        if (!tfv) continue;
        changed |= run_speculative_devirtualization(*fn, mod, tfv, opts);
    }
    return changed;
}

bool run_speculative_devirtualization(
    Module& mod,
    const SpeculativeInlinerOptions& opts
) {
    return run_speculative_devirtualization(mod, runtime::FeedbackRegistry::instance(), opts);
}

} // namespace brass
