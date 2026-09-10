#include <brass/mir/coro_transform.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/runtime/coroutine.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <string>

namespace brass {

namespace {

struct SuspendPoint {
    Instruction* inst = nullptr;
    BasicBlock* block = nullptr;
    uint32_t state_id = 0;
    Value* yield_val = nullptr;
    BasicBlock* resume_bb = nullptr;
    std::vector<Value*> live_values;
};

Value* make_val(Function& fn, Arena& arena, Type type, ValueKind kind = ValueKind::InstructionResult) {
    return arena.make<Value>(fn.next_value_id(), type, kind);
}

Instruction* make_store(Arena& arena, Value* base, int32_t offset, Value* val) {
    Instruction* inst = arena.make<Instruction>(Opcode::store, Type::void_type());
    inst->add_operand(base);
    inst->add_operand(val);
    inst->set_offset(offset);
    inst->set_memory_type(val->type());
    return inst;
}

Instruction* make_load(Function& fn, Arena& arena, Type type, Value* base, int32_t offset) {
    Instruction* inst = arena.make<Instruction>(Opcode::load, type);
    inst->add_operand(base);
    inst->set_offset(offset);
    inst->set_memory_type(type);
    Value* res = make_val(fn, arena, type);
    inst->set_result(res);
    res->set_defining_instruction(inst);
    return inst;
}

void replace_uses_in_region(
    BasicBlock* start_bb,
    Value* old_val,
    Value* new_val,
    const std::unordered_set<BasicBlock*>& stop_bbs
) {
    if (!old_val || !new_val || old_val == new_val) return;
    std::unordered_set<BasicBlock*> visited;
    std::vector<BasicBlock*> worklist = {start_bb};
    visited.insert(start_bb);

    while (!worklist.empty()) {
        BasicBlock* bb = worklist.back();
        worklist.pop_back();

        for (Instruction* inst : *bb) {
            if (inst == new_val->defining_instruction()) continue;
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i) == old_val) {
                    inst->set_operand(i, new_val);
                }
            }
            if (inst->opcode() == Opcode::switch_) {
                for (auto& sc : inst->switch_cases()) {
                    for (auto& arg : sc.target.args) {
                        if (arg == old_val) arg = new_val;
                    }
                }
                for (auto& arg : inst->default_target().args) {
                    if (arg == old_val) arg = new_val;
                }
            }
            if (inst->opcode() == Opcode::br) {
                for (auto& arg : inst->branch_target().args) {
                    if (arg == old_val) arg = new_val;
                }
            } else if (inst->opcode() == Opcode::br_if) {
                for (auto& arg : inst->true_target().args) {
                    if (arg == old_val) arg = new_val;
                }
                for (auto& arg : inst->false_target().args) {
                    if (arg == old_val) arg = new_val;
                }
            }
        }

        for (BasicBlock* succ : bb->successors()) {
            if (stop_bbs.find(succ) == stop_bbs.end() && visited.find(succ) == visited.end()) {
                visited.insert(succ);
                worklist.push_back(succ);
            }
        }
    }
}

std::unordered_set<BasicBlock*> compute_non_dominated_blocks(BasicBlock* entry_bb, BasicBlock* dom_bb) {
    std::unordered_set<BasicBlock*> reachable;
    std::vector<BasicBlock*> q;
    if (entry_bb && entry_bb != dom_bb) {
        reachable.insert(entry_bb);
        q.push_back(entry_bb);
    }
    while (!q.empty()) {
        BasicBlock* curr = q.back();
        q.pop_back();
        for (BasicBlock* succ : curr->successors()) {
            if (succ && succ != dom_bb && reachable.insert(succ).second) {
                q.push_back(succ);
            }
        }
    }
    return reachable;
}

} // namespace

bool CoroTransformPass::run_on_function(Function& fn) {
    if (fn.blocks().empty()) return false;
    BasicBlock* orig_entry = fn.entry_block();
    if (!orig_entry || orig_entry->name() == "bb_coro_entry") return false;

    // 1. Identify all coro_suspend instructions
    std::vector<SuspendPoint> suspends;
    uint32_t max_state_id = 0;

    for (BasicBlock* bb : fn.blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::coro_suspend) {
                SuspendPoint sp;
                sp.inst = inst;
                sp.block = bb;
                sp.state_id = inst->resume_id();
                if (sp.state_id > max_state_id) max_state_id = sp.state_id;
                sp.yield_val = (inst->operand_count() > 0) ? inst->operand(0) : nullptr;
                suspends.push_back(sp);
            }
        }
    }

    if (suspends.empty()) return false;

    if (options_.stats) {
        options_.stats->coroutines_transformed++;
        options_.stats->suspend_points_transformed += suspends.size();
    }

    // Assign unique 1-based state_ids
    uint32_t next_state = (max_state_id > 0) ? (max_state_id + 1) : 1;
    for (auto& sp : suspends) {
        if (sp.state_id == 0) {
            sp.state_id = next_state++;
            sp.inst->set_resume_id(sp.state_id);
        }
    }

    // 2. Ensure orig_entry has frame_param
    auto& arena = fn.parent() ? fn.parent()->arena() : *new Arena();
    std::vector<Value*>& orig_params = orig_entry->params();
    Value* orig_frame_val = nullptr;

    if (orig_params.empty() || !orig_params[0]->type().is_pointer_or_gcref()) {
        orig_frame_val = make_val(fn, arena, Type::gcref(), ValueKind::BlockParam);
        orig_params.insert(orig_params.begin(), orig_frame_val);
        orig_frame_val->set_block_param(orig_entry, 0);
        for (size_t i = 1; i < orig_params.size(); ++i) {
            orig_params[i]->set_block_param(orig_entry, static_cast<uint32_t>(i));
        }
        auto pts = fn.param_types();
        pts.insert(pts.begin(), Type::gcref());
        fn.set_param_types(std::move(pts));
    } else {
        orig_frame_val = orig_params[0];
    }

    // 3. Compute live SSA variables across suspends
    std::unordered_map<BasicBlock*, std::unordered_set<Value*>> def_map;
    std::unordered_map<BasicBlock*, std::unordered_set<Value*>> use_map;

    for (BasicBlock* bb : fn.blocks()) {
        for (Value* p : bb->params()) {
            def_map[bb].insert(p);
        }
        for (Instruction* inst : *bb) {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                Value* op = inst->operand(i);
                if (op && def_map[bb].find(op) == def_map[bb].end()) {
                    use_map[bb].insert(op);
                }
            }
            if (inst->result()) {
                def_map[bb].insert(inst->result());
            }
        }
    }

    std::unordered_map<BasicBlock*, std::unordered_set<Value*>> live_in;
    std::unordered_map<BasicBlock*, std::unordered_set<Value*>> live_out;
    bool changed = true;
    while (changed) {
        changed = false;
        for (BasicBlock* bb : fn.blocks()) {
            std::unordered_set<Value*> new_out;
            for (BasicBlock* succ : bb->successors()) {
                const auto& in_succ = live_in[succ];
                new_out.insert(in_succ.begin(), in_succ.end());
            }
            if (new_out != live_out[bb]) {
                live_out[bb] = std::move(new_out);
                changed = true;
            }

            std::unordered_set<Value*> new_in = use_map[bb];
            for (Value* v : live_out[bb]) {
                if (def_map[bb].find(v) == def_map[bb].end()) {
                    new_in.insert(v);
                }
            }
            if (new_in != live_in[bb]) {
                live_in[bb] = std::move(new_in);
                changed = true;
            }
        }
    }

    // Backward pass inside each block containing suspend
    for (auto& sp : suspends) {
        std::unordered_set<Value*> cur_live = live_out[sp.block];
        Instruction* inst = sp.block->tail();
        while (inst && inst != sp.inst) {
            if (inst->result()) {
                cur_live.erase(inst->result());
            }
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                if (inst->operand(i)) cur_live.insert(inst->operand(i));
            }
            inst = inst->prev();
        }

        cur_live.erase(orig_frame_val);
        if (sp.inst->result()) {
            cur_live.erase(sp.inst->result());
        }

        for (Value* v : cur_live) {
            sp.live_values.push_back(v);
        }
    }

    // 4. Slot allocation
    std::unordered_map<Value*, uint32_t> slot_map;
    uint32_t next_slot = options_.first_slot_index;

    for (auto& sp : suspends) {
        for (Value* v : sp.live_values) {
            if (slot_map.find(v) == slot_map.end()) {
                slot_map[v] = next_slot++;
                if (options_.stats) options_.stats->variables_spilled++;
            }
        }
    }

    // 5. Prepend new entry block with dispatch switch
    std::string_view entry_name = fn.parent()
        ? fn.parent()->string_pool().intern("bb_coro_entry")
        : std::string_view("bb_coro_entry");
    BasicBlock* entry_bb = arena.make<BasicBlock>(fn.next_block_id(), entry_name);
    fn.prepend_block(entry_bb);
    Value* global_frame_param = make_val(fn, arena, Type::gcref(), ValueKind::BlockParam);
    entry_bb->add_param(global_frame_param);

    Instruction* ld_state = make_load(fn, arena, Type::i32(), global_frame_param, runtime::CORO_OFFSET_STATE_ID);
    entry_bb->append_instruction(ld_state);

    Instruction* sw_inst = arena.make<Instruction>(Opcode::switch_, Type::void_type());
    sw_inst->add_operand(ld_state->result());
    BranchTarget def_target(orig_entry);
    if (orig_entry->param_count() > 0) {
        def_target.args.push_back(global_frame_param);
    }
    sw_inst->set_default_target(def_target);

    // 6. Split basic blocks at each coro_suspend
    std::unordered_set<BasicBlock*> resume_bbs_set;
    for (auto& sp : suspends) {
        std::string res_name = "bb_resume_" + std::to_string(sp.state_id);
        std::string_view res_name_view = fn.parent()
            ? fn.parent()->string_pool().intern(res_name)
            : std::string_view("bb_resume");
        BasicBlock* resume_bb = arena.make<BasicBlock>(fn.next_block_id(), res_name_view);
        fn.append_block(resume_bb);
        sp.resume_bb = resume_bb;
        resume_bbs_set.insert(resume_bb);

        sw_inst->add_switch_case(static_cast<int64_t>(sp.state_id), resume_bb);
        fn.add_resume_point(sp.state_id, resume_bb);
    }
    entry_bb->append_instruction(sw_inst);
    std::unordered_set<Instruction*> suspend_rets;

    for (auto& sp : suspends) {
        Instruction* susp = sp.inst;
        BasicBlock* cur_bb = susp->parent();
        BasicBlock* resume_bb = sp.resume_bb;

        // Move instructions after susp to resume_bb
        Instruction* mover = susp->next();
        while (mover) {
            Instruction* nxt = mover->next();
            cur_bb->remove_instruction(mover);
            resume_bb->append_instruction(mover);
            mover = nxt;
        }

        // Before susp: store live values to frame
        for (Value* v : sp.live_values) {
            uint32_t slot = slot_map[v];
            int32_t offset = static_cast<int32_t>(runtime::CORO_OFFSET_SLOTS + slot * 8);
            Instruction* st = make_store(arena, global_frame_param, offset, v);
            cur_bb->insert_before(st, susp);
        }

        // Store state_id
        Instruction* state_c = arena.make<Instruction>(Opcode::iconst_i32, Type::i32());
        state_c->set_imm_i64(sp.state_id);
        Value* state_c_val = make_val(fn, arena, Type::i32());
        state_c->set_result(state_c_val);
        state_c_val->set_defining_instruction(state_c);
        cur_bb->insert_before(state_c, susp);

        Instruction* st_state = make_store(arena, global_frame_param, runtime::CORO_OFFSET_STATE_ID, state_c_val);
        cur_bb->insert_before(st_state, susp);

        // Store yield_val
        if (sp.yield_val) {
            Instruction* st_yield = make_store(arena, global_frame_param, runtime::CORO_OFFSET_YIELD_VAL, sp.yield_val);
            cur_bb->insert_before(st_yield, susp);
        }

        // Replace susp with ret
        Instruction* ret_inst = arena.make<Instruction>(Opcode::ret, Type::void_type());
        if (sp.yield_val) {
            ret_inst->add_operand(sp.yield_val);
        }
        cur_bb->insert_before(ret_inst, susp);
        suspend_rets.insert(ret_inst);
        cur_bb->remove_instruction(susp);

        std::unordered_set<BasicBlock*> stop_bbs = compute_non_dominated_blocks(entry_bb, resume_bb);
        stop_bbs.insert(orig_entry);
        for (BasicBlock* rbb : resume_bbs_set) {
            if (rbb != resume_bb) stop_bbs.insert(rbb);
        }

        // Inside resume_bb: load resume_arg if needed
        if (susp->result()) {
            Value* orig_res = susp->result();
            Instruction* ld_arg = make_load(fn, arena, orig_res->type(), global_frame_param, runtime::CORO_OFFSET_RESUME_ARG);
            resume_bb->prepend_instruction(ld_arg);

            replace_uses_in_region(resume_bb, orig_res, ld_arg->result(), stop_bbs);
        }

        // Inside resume_bb: restore live values
        for (Value* v : sp.live_values) {
            uint32_t slot = slot_map[v];
            int32_t offset = static_cast<int32_t>(runtime::CORO_OFFSET_SLOTS + slot * 8);
            Instruction* ld_v = make_load(fn, arena, v->type(), global_frame_param, offset);
            resume_bb->prepend_instruction(ld_v);

            replace_uses_in_region(resume_bb, v, ld_v->result(), stop_bbs);
        }
    }

    // 7. Update return instructions to mark is_done = 1 and state_id = ~0U
    for (BasicBlock* bb : fn.blocks()) {
        if (bb == entry_bb) continue;
        Instruction* term = bb->terminator();
        if (term && term->opcode() == Opcode::ret && suspend_rets.find(term) == suspend_rets.end()) {
            Instruction* c_one = arena.make<Instruction>(Opcode::iconst_i32, Type::i32());
            c_one->set_imm_i64(1);
            Value* one_val = make_val(fn, arena, Type::i32());
            c_one->set_result(one_val);
            one_val->set_defining_instruction(c_one);
            bb->insert_before(c_one, term);

            Instruction* st_done = make_store(arena, global_frame_param, runtime::CORO_OFFSET_IS_DONE, one_val);
            bb->insert_before(st_done, term);

            Instruction* c_term_state = arena.make<Instruction>(Opcode::iconst_i32, Type::i32());
            c_term_state->set_imm_i64(static_cast<int64_t>(0xFFFFFFFF));
            Value* term_state_val = make_val(fn, arena, Type::i32());
            c_term_state->set_result(term_state_val);
            term_state_val->set_defining_instruction(c_term_state);
            bb->insert_before(c_term_state, term);

            Instruction* st_tstate = make_store(arena, global_frame_param, runtime::CORO_OFFSET_STATE_ID, term_state_val);
            bb->insert_before(st_tstate, term);

            if (term->operand_count() > 0 && term->operand(0)) {
                Instruction* st_y = make_store(arena, global_frame_param, runtime::CORO_OFFSET_YIELD_VAL, term->operand(0));
                bb->insert_before(st_y, term);
            }
        }
    }

    if (orig_frame_val != global_frame_param) {
        replace_uses_in_region(orig_entry, orig_frame_val, global_frame_param, {});
    }

    fn.rebuild_cfg_predecessors();
    return true;
}

bool CoroTransformPass::run_on_module(Module& mod) {
    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (fn && run_on_function(*fn)) {
            changed = true;
        }
    }
    return changed;
}

} // namespace brass
