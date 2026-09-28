#include <brass/mir/coro_transform.hpp>
#include <brass/mir/block_liveness.hpp>
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

bool is_ref_type(Type t) {
    return t.is_pointer_or_gcref() || t.is_tagged();
}

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

// Whether `v` is defined where `at` executes: in a block dominating `at`'s,
// or earlier in the same block.
bool available_at(const Value* v, const Instruction* at, const DominatorTree& dom) {
    const BasicBlock* db = def_block_of(v);
    const BasicBlock* ab = at->parent();
    if (!db || !ab) return false;
    if (db != ab) return dom.is_reachable(ab) && dom.dominates(db, ab);
    if (v->is_block_param()) return true;
    for (const Instruction* i = ab->head(); i && i != at; i = i->next()) {
        if (i->result() == v) return true;
    }
    return false;
}

// A guard that resumes at a block of this body (no exit stub) gives a lower
// tier every value that block reads (verify_guard_resume_state). Lowering
// makes the body read values the front end's state never named: the frame,
// which each return, spill store and reload now addresses, and values the
// resume block's successors reload. Each one the target reads and the state
// lacks is appended, in id order, to the Tier-0 guard; a tier-2 copy is
// taken of the lowered body, so its guard of the same resume id carries the
// same state in the same order.
//
// A guard in a block the lowered body cannot reach (code after a `throw`, as
// a front end may leave it) never runs in any tier, and dominance says
// nothing about what is defined there, so no state can be proven for it: it
// is removed, with its resume point when no other guard names that id. Its
// resume target stays a block of the body, now reachable from nothing.
void complete_guard_states(Function& fn, const DominatorTree& dom) {
    std::vector<Instruction*> guards;
    std::vector<Instruction*> dead;
    for (BasicBlock* bb : fn.blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::guard && !fn.guard_exit_stub(*inst) &&
                fn.get_resume_target(inst->resume_id())) {
                (dom.is_reachable(bb) ? guards : dead).push_back(inst);
            }
        }
    }
    for (Instruction* g : dead) {
        const uint32_t id = g->resume_id();
        g->parent()->remove_instruction(g);
        if (!fn.find_guard(id)) fn.remove_resume_point(id);
    }
    if (guards.empty()) return;
    std::vector<const BasicBlock*> targets;
    targets.reserve(guards.size());
    for (const Instruction* g : guards) targets.push_back(fn.get_resume_target(g->resume_id()));
    const auto live_in = block_live_ins(fn, targets);
    for (Instruction* g : guards) {
        auto it = live_in.find(fn.get_resume_target(g->resume_id()));
        if (it == live_in.end()) continue;
        const std::unordered_set<const Value*> have(g->state_map().begin(), g->state_map().end());
        std::vector<Value*> add;
        for (const Value* v : it->second) {
            if (!have.count(v)) add.push_back(const_cast<Value*>(v));
        }
        std::sort(add.begin(), add.end(), [](const Value* a, const Value* b) { return a->id() < b->id(); });
        for (Value* v : add) {
            if (!available_at(v, g, dom)) {
                throw std::logic_error(fn_desc(fn) + "guard (resume id " + std::to_string(g->resume_id()) +
                                       ") resumes at a block that reads %" + std::to_string(v->id()) +
                                       ", which is not defined at the guard");
            }
            g->state_map().push_back(v);
        }
    }
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

    // An alloca is stack memory of one invocation of the body, and each
    // resume is a new invocation: the entry block's allocas are re-created by
    // bb_coro_entry on every resume (step 5), so their addresses are never
    // spilled. Their contents do not survive a suspend.
    std::vector<Instruction*> entry_allocas;
    for (Instruction* inst : *orig_entry) {
        if (inst->opcode() == Opcode::alloca_) entry_allocas.push_back(inst);
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
    for (Instruction* a : entry_allocas) crossing.erase(a->result());
    for (Value* v : crossing) {
        Instruction* d = v->defining_instruction();
        if (d && d->opcode() == Opcode::alloca_) {
            throw std::logic_error(fn_desc(fn) + "an alloca outside the entry block is live across a suspend; "
                                                 "a body's allocas belong in its entry block");
        }
    }

    // 4. Slot allocation. Arguments keep their argument slots; every other
    // crossing value gets its own slot, references first (they then fit the
    // fixed-code entry's 64-bit mask as long as they can). Sorted by id for
    // a stable layout. A value wider than 8 bytes spans consecutive slots
    // (coro_slot_count).
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
        const bool ga = is_ref_type(a->type());
        const bool gb = is_ref_type(b->type());
        if (ga != gb) return ga;
        return a->id() < b->id();
    });
    uint32_t next_slot = std::max(options_.first_slot_index, arg_slot_end);
    for (Value* v : spilled) {
        slot_map[v] = next_slot;
        next_slot += coro_slot_count(v->type());
        if (options_.stats) options_.stats->variables_spilled++;
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

        // Only the dispatch switch enters a suspend's resume block. Its entry
        // in the resume table (which keeps the passes that assume one entry
        // off the body, as a mid-body entry needs) is numbered apart from
        // the guards' resume ids, which the front end picks and a deopt looks
        // its target up by: a state id is not a resume id.
        sw_inst->add_switch_case(static_cast<int64_t>(sp.state_id), resume_bb);
        fn.add_resume_point(coro_suspend_resume_id(sp.state_id), resume_bb);
    }
    entry_bb->append_instruction(sw_inst);
    for (auto it = entry_allocas.rbegin(); it != entry_allocas.rend(); ++it) {
        orig_entry->remove_instruction(*it);
        entry_bb->prepend_instruction(*it);
    }
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

    // 7. Store each spilled value to its slot where it is defined. A
    // reference's store carries its write barrier: a frame that survived a
    // collection since it was created may be old, and its slots are found
    // by a minor collection only through its card.
    for (Value* v : spilled) {
        Instruction* st = make_store(arena, global_frame_param, slot_offset(slot_map[v]), v);
        Instruction* wb = nullptr;
        if (is_ref_type(v->type())) {
            wb = arena.make<Instruction>(Opcode::write_barrier, Type::void_type());
            wb->add_operand(global_frame_param);
            wb->add_operand(v);
        }
        if (v->is_block_param()) {
            if (wb) v->defining_block()->prepend_instruction(wb);
            v->defining_block()->prepend_instruction(st);
            continue;
        }
        Instruction* d = v->defining_instruction();
        if (!d || !d->parent()) {
            throw std::logic_error(fn_desc(fn) + "a value live across a suspend has no definition");
        }
        if (d->opcode() == Opcode::invoke) {
            // The result exists only on the normal edge: the store goes on a
            // block of its own there, which the invoke's block dominates.
            std::string edge_name = "bb_invoke_spill_" + std::to_string(v->id());
            std::string_view edge_view = fn.parent() ? fn.parent()->string_pool().intern(edge_name)
                                                     : std::string_view("bb_invoke_spill");
            BasicBlock* edge = arena.make<BasicBlock>(fn.next_block_id(), edge_view);
            fn.append_block(edge);
            Instruction* br = arena.make<Instruction>(Opcode::br, Type::void_type());
            br->set_branch_target(d->normal_target());
            d->set_normal_target(BranchTarget(edge));
            edge->append_instruction(st);
            if (wb) edge->append_instruction(wb);
            edge->append_instruction(br);
            continue;
        }
        if (d->is_terminator()) {
            throw std::logic_error(fn_desc(fn) + "the result of a terminator (" +
                                   std::string(opcode_name(d->opcode())) +
                                   ") is live across a suspend, which is not supported");
        }
        if (wb) d->parent()->insert_after(wb, d);
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
            // A landing pad stays its block's first instruction.
            Instruction* head = bb->head();
            if (head && head->opcode() == Opcode::landing_pad) {
                bb->insert_after(ld, head);
            } else {
                bb->prepend_instruction(ld);
            }
            replace_uses_in(*bb, v, ld->result());
        }
    }

    // 10. Guards resuming in this body (step 9 already renamed a reloaded
    // value in their state, a state map being a use).
    complete_guard_states(fn, dom);

    record_coro_frame_layout(fn);
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

bool coro_body_resumes_mid_body(const Function& fn) {
    const BasicBlock* entry = fn.entry_block();
    if (!entry || entry->name() != "bb_coro_entry") return false;
    const Instruction* term = entry->terminator();
    return term && term->opcode() == Opcode::switch_ && !term->switch_cases().empty();
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

namespace {

void finish_layout(CoroFrameLayout& layout) {
    layout.slot_count = std::max(layout.slot_count, 1U);
    layout.pointer_mask = layout.ref_bits.empty() ? 0 : layout.ref_bits[0];
    layout.fits_pointer_mask = true;
    for (size_t w = 1; w < layout.ref_bits.size(); ++w) {
        if (layout.ref_bits[w]) layout.fits_pointer_mask = false;
    }
}

// The slots the body's frame accesses name, as lowering left them.
CoroFrameLayout scan_coro_frame_layout(const Function& fn) {
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
            if (is_ref_type(inst->memory_type())) {
                if (layout.ref_bits.size() <= slot / 64) layout.ref_bits.resize(slot / 64 + 1, 0);
                layout.ref_bits[slot / 64] |= uint64_t{1} << (slot % 64);
            }
        }
    }
    finish_layout(layout);
    return layout;
}

} // namespace

void record_coro_frame_layout(Function& fn) {
    const CoroFrameLayout layout = scan_coro_frame_layout(fn);
    fn.set_coro_frame_layout(layout.slot_count, layout.ref_bits);
}

CoroFrameLayout compute_coro_frame_layout(const Function& fn) {
    if (!fn.has_coro_frame_layout()) return scan_coro_frame_layout(fn);
    CoroFrameLayout layout;
    layout.slot_count = fn.coro_frame_slot_count();
    layout.ref_bits = fn.coro_frame_ref_bits();
    finish_layout(layout);
    return layout;
}

bool lower_coroutines(Module& mod) {
    // Cheap when there is nothing to lower: a scan for the coroutine ops.
    bool any = false;
    for (const Function* fn : mod.functions()) {
        if (!fn) continue;
        for (const BasicBlock* bb : fn->blocks()) {
            for (const Instruction* inst : *bb) {
                if (inst->opcode() == Opcode::coro_suspend || inst->opcode() == Opcode::coro_create) {
                    any = true;
                    break;
                }
            }
            if (any) break;
        }
        if (any) break;
    }
    return any && CoroTransformPass().run_on_module(mod);
}

} // namespace brass
