#include <brass/mir/coro_transform.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/uses.hpp>
#include <brass/runtime/coroutine.hpp>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace brass {

namespace {

// A frame slot's pointer bit is bit (slot + 5) of the frame object's header
// mask (the slots follow five header words), and bit 63 of that mask means
// "every field from 63 on". A gcref slot past this one cannot be described
// precisely.
constexpr uint32_t kMaxPointerSlot = 63 - static_cast<uint32_t>(runtime::CORO_OFFSET_SLOTS / 8) - 1;

struct SuspendPoint {
    Instruction* inst = nullptr;
    BasicBlock* block = nullptr;
    uint32_t state_id = 0;
    Value* yield_val = nullptr;
    BasicBlock* resume_bb = nullptr;
};

Value* make_val(Function& fn, Arena& arena, Type type, ValueKind kind = ValueKind::InstructionResult) {
    return arena.make<Value>(fn.next_value_id(), type, kind);
}

// Vector values move through the frame with vload/vstore, which every tier
// lowers to a full-width unaligned access; scalars use load/store.
Instruction* make_store(Arena& arena, Value* base, int32_t offset, Value* val) {
    const Opcode op = val->type().is_vector() ? Opcode::vstore : Opcode::store;
    Instruction* inst = arena.make<Instruction>(op, Type::void_type());
    inst->add_operand(base);
    inst->add_operand(val);
    inst->set_offset(offset);
    inst->set_memory_type(val->type());
    return inst;
}

// A load whose result is `res` (a fresh value when null).
Instruction* make_load(Function& fn, Arena& arena, Type type, Value* base, int32_t offset, Value* res = nullptr) {
    Instruction* inst = arena.make<Instruction>(type.is_vector() ? Opcode::vload : Opcode::load, type);
    inst->add_operand(base);
    inst->set_offset(offset);
    inst->set_memory_type(type);
    if (!res) res = make_val(fn, arena, type);
    inst->set_result(res);
    res->set_defining_instruction(inst);
    return inst;
}

Instruction* make_i32(Function& fn, Arena& arena, int64_t imm) {
    Instruction* c = arena.make<Instruction>(Opcode::iconst_i32, Type::i32());
    c->set_imm_i64(imm);
    Value* v = make_val(fn, arena, Type::i32());
    c->set_result(v);
    v->set_defining_instruction(c);
    return c;
}

int32_t slot_offset(uint32_t slot) {
    return static_cast<int32_t>(runtime::CORO_OFFSET_SLOTS + slot * 8);
}

BasicBlock* def_block_of(const Value* v) {
    if (v->is_block_param()) return v->defining_block();
    Instruction* d = v->defining_instruction();
    return d ? d->parent() : nullptr;
}

bool block_uses(BasicBlock* bb, const Value* v) {
    for (Instruction* inst : *bb) {
        if (uses_value(*inst, v)) return true;
    }
    return false;
}

std::string fn_desc(const Function& fn) {
    return "coroutine lowering of @" + std::string(fn.name()) + ": ";
}

} // namespace

// The lowered body runs once per resume, entering at bb_coro_entry, which
// dispatches on the frame's state to the original entry or to the resume
// block of the suspend that paused it. A value live across a suspend
// therefore reaches its uses from a different invocation than the one that
// defined it; such a value lives in a frame slot:
// - it is stored to its slot where it is defined (a coroutine argument is
//   already in its argument slot, put there by coro_create), so the slot
//   always holds its latest definition, as SSA's dominance guarantees for
//   any use;
// - every block that uses it and is not dominated by its definition in the
//   lowered CFG (a resume block, or a merge point such as a loop header that
//   a resume path re-enters) reloads it from the slot on entry.
// Blocks the definition dominates keep the register value.
bool CoroTransformPass::run_on_function(Function& fn, bool force, int64_t create_arg_count) {
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

    if (suspends.empty() && !force) return false;

    // The yielded value, the resume argument and the return value travel
    // through the frame header's 8-byte yield/resume fields.
    for (const auto& sp : suspends) {
        if (sp.yield_val && sp.yield_val->type().size_in_bytes() > 8) {
            throw std::logic_error(fn_desc(fn) + "a coro_suspend yields a " +
                                   std::to_string(sp.yield_val->type().size_in_bytes()) +
                                   "-byte value; yielded values must fit the frame's 8-byte yield field");
        }
        if (sp.inst->result() && sp.inst->result()->type().size_in_bytes() > 8) {
            throw std::logic_error(fn_desc(fn) + "a coro_suspend result is " +
                                   std::to_string(sp.inst->result()->type().size_in_bytes()) +
                                   " bytes; resume arguments must fit the frame's 8-byte resume field");
        }
    }
    for (BasicBlock* bb : fn.blocks()) {
        Instruction* term = bb->terminator();
        if (term && term->opcode() == Opcode::ret && term->operand_count() > 0 && term->operand(0) &&
            term->operand(0)->type().size_in_bytes() > 8) {
            throw std::logic_error(fn_desc(fn) + "returns a " +
                                   std::to_string(term->operand(0)->type().size_in_bytes()) +
                                   "-byte value; a coroutine's return value must fit the frame's 8-byte yield field");
        }
    }

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

    // A coroutine whose first argument is a gcref must not have that
    // argument taken for the frame: coro_create's argument count decides.
    bool has_frame_param = !orig_params.empty() && orig_params[0]->type().is_pointer_or_gcref();
    if (create_arg_count >= 0) {
        const auto n = static_cast<int64_t>(orig_params.size());
        if (n == create_arg_count) {
            has_frame_param = false;
        } else if (n == create_arg_count + 1 && has_frame_param) {
            has_frame_param = true;
        } else {
            throw std::logic_error(fn_desc(fn) + "coro_create passes " + std::to_string(create_arg_count) +
                                   " argument(s) to a body with " + std::to_string(n) + " parameter(s)");
        }
    }
    if (!has_frame_param) {
        orig_frame_val = make_val(fn, arena, Type::gcref(), ValueKind::BlockParam);
        orig_params.insert(orig_params.begin(), orig_frame_val);
        orig_frame_val->set_block_param(orig_entry, 0);
        for (size_t i = 1; i < orig_params.size(); ++i) {
            orig_params[i]->set_block_param(orig_entry, static_cast<uint32_t>(i));
        }
    } else {
        orig_frame_val = orig_params[0];
    }
    // Every other original parameter is a coroutine argument: coro_create
    // stores argument i in frame slot i, and the lowered body takes only the
    // frame.
    const uint32_t arg_count = static_cast<uint32_t>(orig_params.size() - 1);
    fn.set_param_types({orig_frame_val->type()});

    // 3. Liveness over the original CFG. A use is any value slot of an
    // instruction: operands, deopt state and the arguments of every edge.
    std::unordered_map<BasicBlock*, std::unordered_set<Value*>> def_map;
    std::unordered_map<BasicBlock*, std::unordered_set<Value*>> use_map;

    for (BasicBlock* bb : fn.blocks()) {
        auto& defs = def_map[bb];
        auto& uses = use_map[bb];
        for (Value* p : bb->params()) defs.insert(p);
        for (Instruction* inst : *bb) {
            for_each_use(*inst, [&](Value* op) {
                if (defs.find(op) == defs.end()) uses.insert(op);
            });
            if (inst->result()) defs.insert(inst->result());
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

    // Values live just after each suspend: they cross it.
    std::unordered_set<Value*> crossing;
    for (auto& sp : suspends) {
        std::unordered_set<Value*> cur_live = live_out[sp.block];
        for (Instruction* inst = sp.block->tail(); inst && inst != sp.inst; inst = inst->prev()) {
            if (inst->result()) cur_live.erase(inst->result());
            for_each_use(*inst, [&](Value* op) { cur_live.insert(op); });
        }
        if (sp.inst->result()) cur_live.erase(sp.inst->result());
        cur_live.erase(orig_frame_val);
        crossing.insert(cur_live.begin(), cur_live.end());
    }

    // 4. Slot allocation. Arguments keep their argument slots; every other
    // crossing value gets its own slot, gcrefs first so they stay within
    // the frame's precise pointer mask. Sorted by id for a stable layout.
    // A value wider than 8 bytes spans consecutive slots (coro_slot_count).
    std::unordered_map<Value*, uint32_t> slot_map;
    std::vector<Value*> spilled;
    std::vector<uint32_t> arg_slot(arg_count);
    uint32_t arg_slot_end = 0;
    for (uint32_t i = 0; i < arg_count; ++i) {
        arg_slot[i] = arg_slot_end;
        arg_slot_end += coro_slot_count(orig_params[i + 1]->type());
        if (crossing.count(orig_params[i + 1])) slot_map[orig_params[i + 1]] = arg_slot[i];
    }
    for (Value* v : crossing) {
        if (!slot_map.count(v)) spilled.push_back(v);
    }
    std::sort(spilled.begin(), spilled.end(), [](const Value* a, const Value* b) {
        const bool ga = a->type().is_pointer_or_gcref(), gb = b->type().is_pointer_or_gcref();
        if (ga != gb) return ga;
        return a->id() < b->id();
    });
    uint32_t next_slot = std::max(options_.first_slot_index, arg_slot_end);
    for (Value* v : spilled) {
        slot_map[v] = next_slot;
        next_slot += coro_slot_count(v->type());
        if (options_.stats) options_.stats->variables_spilled++;
    }
    for (const auto& [v, slot] : slot_map) {
        if (v->type().is_pointer_or_gcref() && slot > kMaxPointerSlot) {
            throw std::logic_error(fn_desc(fn) + "a gcref live across a suspend needs frame slot " +
                                   std::to_string(slot) + ", past the last slot the frame's pointer mask covers (" +
                                   std::to_string(kMaxPointerSlot) + ")");
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
    def_target.args.push_back(global_frame_param);
    for (uint32_t i = 0; i < arg_count; ++i) {
        Instruction* ld_arg = make_load(fn, arena, orig_params[i + 1]->type(), global_frame_param, slot_offset(arg_slot[i]));
        entry_bb->append_instruction(ld_arg);
        def_target.args.push_back(ld_arg->result());
    }
    sw_inst->set_default_target(def_target);

    // 6. Split basic blocks at each coro_suspend
    for (auto& sp : suspends) {
        std::string res_name = "bb_resume_" + std::to_string(sp.state_id);
        std::string_view res_name_view = fn.parent()
            ? fn.parent()->string_pool().intern(res_name)
            : std::string_view("bb_resume");
        BasicBlock* resume_bb = arena.make<BasicBlock>(fn.next_block_id(), res_name_view);
        fn.append_block(resume_bb);
        sp.resume_bb = resume_bb;

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

        Instruction* state_c = make_i32(fn, arena, sp.state_id);
        cur_bb->insert_before(state_c, susp);
        cur_bb->insert_before(make_store(arena, global_frame_param, runtime::CORO_OFFSET_STATE_ID,
                                         state_c->result()), susp);
        if (sp.yield_val) {
            cur_bb->insert_before(make_store(arena, global_frame_param, runtime::CORO_OFFSET_YIELD_VAL,
                                             sp.yield_val), susp);
        }

        // Replace susp with ret
        Instruction* ret_inst = arena.make<Instruction>(Opcode::ret, Type::void_type());
        if (sp.yield_val) ret_inst->add_operand(sp.yield_val);
        cur_bb->insert_before(ret_inst, susp);
        suspend_rets.insert(ret_inst);
        cur_bb->remove_instruction(susp);

        // The suspend's result becomes the resume block's load of the
        // resume argument: the same value, now defined there.
        if (Value* orig_res = susp->result()) {
            Instruction* ld_arg = make_load(fn, arena, orig_res->type(), global_frame_param,
                                            runtime::CORO_OFFSET_RESUME_ARG, orig_res);
            resume_bb->prepend_instruction(ld_arg);
        }
    }

    // 7. Store each spilled value to its slot where it is defined.
    for (Value* v : spilled) {
        Instruction* st = make_store(arena, global_frame_param, slot_offset(slot_map[v]), v);
        if (v->is_block_param()) {
            v->defining_block()->prepend_instruction(st);
            continue;
        }
        Instruction* d = v->defining_instruction();
        if (!d || !d->parent()) {
            throw std::logic_error(fn_desc(fn) + "a value live across a suspend has no definition");
        }
        if (d->is_terminator()) {
            throw std::logic_error(fn_desc(fn) + "the result of a terminator (" +
                                   std::string(opcode_name(d->opcode())) +
                                   ") is live across a suspend, which is not supported");
        }
        d->parent()->insert_after(st, d);
    }

    // 8. Update return instructions to mark is_done = 1 and state_id = ~0U
    for (BasicBlock* bb : fn.blocks()) {
        if (bb == entry_bb) continue;
        Instruction* term = bb->terminator();
        if (term && term->opcode() == Opcode::ret && suspend_rets.find(term) == suspend_rets.end()) {
            Instruction* c_one = make_i32(fn, arena, 1);
            bb->insert_before(c_one, term);
            bb->insert_before(make_store(arena, global_frame_param, runtime::CORO_OFFSET_IS_DONE,
                                         c_one->result()), term);

            Instruction* c_term_state = make_i32(fn, arena, static_cast<int64_t>(0xFFFFFFFF));
            bb->insert_before(c_term_state, term);
            bb->insert_before(make_store(arena, global_frame_param, runtime::CORO_OFFSET_STATE_ID,
                                         c_term_state->result()), term);

            if (term->operand_count() > 0 && term->operand(0)) {
                bb->insert_before(make_store(arena, global_frame_param, runtime::CORO_OFFSET_YIELD_VAL,
                                             term->operand(0)), term);
            }
        }
    }

    if (orig_frame_val != global_frame_param) {
        for (BasicBlock* bb : fn.blocks()) {
            if (bb != entry_bb) replace_uses_in(*bb, orig_frame_val, global_frame_param);
        }
    }

    fn.rebuild_cfg_predecessors();

    // 9. Reload each crossing value on entry to every block that uses it
    // and that its definition does not dominate in the lowered CFG.
    DominatorTree dom(fn);
    for (const auto& [v, slot] : slot_map) {
        BasicBlock* def_bb = def_block_of(v);
        if (!def_bb) {
            throw std::logic_error(fn_desc(fn) + "a value live across a suspend has no defining block");
        }
        for (BasicBlock* bb : fn.blocks()) {
            if (bb == def_bb || bb == entry_bb) continue;
            if (dom.is_reachable(bb) && dom.dominates(def_bb, bb)) continue;
            if (!block_uses(bb, v)) continue;
            Instruction* ld = make_load(fn, arena, v->type(), global_frame_param, slot_offset(slot));
            bb->prepend_instruction(ld);
            replace_uses_in(*bb, v, ld->result());
        }
    }

    return true;
}

bool CoroTransformPass::run_on_module(Module& mod) {
    // Every coro_create target is a coroutine body, suspends or not; every
    // coro_create of one body passes the same number of arguments.
    std::unordered_map<std::string_view, int64_t> coro_targets;
    for (Function* fn : mod.functions()) {
        if (!fn) continue;
        for (BasicBlock* bb : fn->blocks()) {
            for (Instruction* inst : *bb) {
                if (inst->opcode() != Opcode::coro_create) continue;
                const auto n = static_cast<int64_t>(inst->operand_count());
                auto [it, fresh] = coro_targets.emplace(inst->symbol(), n);
                if (!fresh && it->second != n) {
                    throw std::logic_error("coroutine lowering: coro_create @" + std::string(inst->symbol()) +
                                           " is passed both " + std::to_string(it->second) + " and " +
                                           std::to_string(n) + " arguments");
                }
            }
        }
    }

    bool changed = false;
    for (Function* fn : mod.functions()) {
        if (!fn) continue;
        auto it = coro_targets.find(fn->name());
        const bool target = it != coro_targets.end();
        if (run_on_function(*fn, target, target ? it->second : -1)) {
            changed = true;
        }
    }
    return changed;
}

bool is_lowered_coro_body(const Function& fn) {
    if (fn.param_count() != 1 || !fn.param_type(0).is_pointer_or_gcref()) return false;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::coro_suspend) return false;
        }
    }
    return true;
}

uint32_t coro_slot_count(Type t) {
    const size_t bytes = t.size_in_bytes();
    return bytes <= 8 ? 1U : static_cast<uint32_t>((bytes + 7) / 8);
}

uint32_t coro_create_slot_count(const Instruction& create) {
    uint32_t n = 0;
    for (size_t i = 0; i < create.operand_count(); ++i) n += coro_slot_count(create.operand(i)->type());
    return n;
}

CoroFrameLayout compute_coro_frame_layout(const Function& fn) {
    CoroFrameLayout layout;
    const BasicBlock* entry = fn.entry_block();
    const Value* frame = (entry && entry->param_count() > 0) ? entry->param(0) : nullptr;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            const Opcode op = inst->opcode();
            if (op != Opcode::store && op != Opcode::load && op != Opcode::vstore && op != Opcode::vload) continue;
            if (!frame || inst->operand(0) != frame || inst->offset() < runtime::CORO_OFFSET_SLOTS) continue;
            uint32_t slot = static_cast<uint32_t>((inst->offset() - runtime::CORO_OFFSET_SLOTS) / 8);
            layout.slot_count = std::max(layout.slot_count, slot + coro_slot_count(inst->memory_type()));
            if (inst->memory_type().is_pointer_or_gcref()) {
                if (slot > kMaxPointerSlot) {
                    throw std::logic_error("coroutine frame of @" + std::string(fn.name()) + ": gcref slot " +
                                           std::to_string(slot) + " is past the last slot the frame's pointer "
                                           "mask covers (" + std::to_string(kMaxPointerSlot) + ")");
                }
                layout.pointer_mask |= (1ULL << slot);
            }
        }
    }
    layout.slot_count = std::max(layout.slot_count, 1U);
    return layout;
}

} // namespace brass
