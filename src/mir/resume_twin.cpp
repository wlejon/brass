// Resume twins (resume_twin.hpp): the twin built from a site-form function,
// the entry that runs it from the start, and the retargeting of a
// speculated copy's guards.

#include <brass/mir/resume_twin.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/uses.hpp>
#include <brass/mir/verifier.hpp>
#include "ir_clone.hpp"

#include <algorithm>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

bool fail(std::string* why, std::string msg) {
    if (why) *why = std::move(msg);
    return false;
}

// A guard that resumes in its own function's resume table: the site form's.
bool resumes_in_table(const Function& fn, const Instruction& inst) {
    return inst.opcode() == Opcode::guard && !fn.guard_exit_stub(inst) &&
           fn.get_resume_target(inst.resume_id()) != nullptr;
}

const BasicBlock* defining_block(const Value* v) {
    if (v->is_block_param()) return v->defining_block();
    const Instruction* def = v->defining_instruction();
    return def ? def->parent() : nullptr;
}

struct TwinEntry {
    uint32_t id = 0;
    const BasicBlock* target = nullptr;
    // What the entry reads from the state block, in order: the target's
    // parameters first, then values the code from the target on reads under
    // their own names.
    std::vector<const Value*> state;
};

} // namespace

Function* build_resume_twin(const Function& fn, Module& dst, std::string_view name,
                            const ResumeTwinOptions& options, std::string* why) {
    if (fn.has_coro_frame_layout()) {
        fail(why, "a coroutine body");
        return nullptr;
    }
    if (fn.block_count() == 0) {
        fail(why, "no body");
        return nullptr;
    }

    // The entries: one per resume id a site-form guard names. Guards sharing
    // an id are copies of one guard (the verifier holds them to one state
    // signature); the first gives the entry its state.
    std::map<uint32_t, TwinEntry> entries;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (!inst || !resumes_in_table(fn, *inst)) continue;
            auto [it, fresh] = entries.try_emplace(inst->resume_id());
            if (!fresh) continue;
            TwinEntry& e = it->second;
            e.id = inst->resume_id();
            e.target = fn.get_resume_target(e.id);
            for (const Value* v : inst->state_map()) {
                if (!v) {
                    fail(why, "guard " + std::to_string(e.id) + " has an empty state value");
                    return nullptr;
                }
                if (v->type().is_vector()) {
                    fail(why, "guard " + std::to_string(e.id) + " has a vector state value");
                    return nullptr;
                }
                e.state.push_back(v);
            }
            if (e.target->param_count() > e.state.size()) {
                fail(why, "resume block of guard " + std::to_string(e.id) + " has more parameters than state values");
                return nullptr;
            }
        }
    }
    if (options.entry_id) {
        if (entries.count(*options.entry_id)) {
            fail(why, "entry id " + std::to_string(*options.entry_id) + " is a guard's resume id");
            return nullptr;
        }
        TwinEntry& e = entries[*options.entry_id];
        e.id = *options.entry_id;
        e.target = fn.entry_block();
        for (const Value* p : e.target->params()) {
            if (p->type().is_vector()) {
                fail(why, "a vector parameter");
                return nullptr;
            }
            e.state.push_back(p);
        }
    }
    if (entries.empty()) {
        fail(why, "no guard resumes in the function");
        return nullptr;
    }

    // The region: everything reachable from an entry's target.
    std::unordered_set<const BasicBlock*> reach;
    std::vector<const BasicBlock*> work;
    for (const auto& [id, e] : entries) {
        if (reach.insert(e.target).second) work.push_back(e.target);
    }
    while (!work.empty()) {
        const BasicBlock* bb = work.back();
        work.pop_back();
        for (const BasicBlock* s : bb->successors()) {
            if (s && reach.insert(s).second) work.push_back(s);
        }
        // A site-form guard is a branch to its resume block here.
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (!inst || !resumes_in_table(fn, *inst)) continue;
            const BasicBlock* t = fn.get_resume_target(inst->resume_id());
            if (reach.insert(t).second) work.push_back(t);
        }
    }
    std::vector<const BasicBlock*> region;
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb && reach.count(bb)) region.push_back(bb);
    }

    const std::vector<Type> params{Type::i32(), Type::ptr()};
    Function* out = dst.create_function(name, fn.return_type(), params);
    out->set_resume_twin();
    out->set_allow_fp_reassociation(fn.allow_fp_reassociation());
    out->set_requires_barrier_certs(fn.requires_barrier_certs());
    // New blocks and values are numbered past fn's.
    uint32_t max_block = 0;
    uint32_t max_value = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        max_block = std::max(max_block, bb->id());
        for (const Value* p : bb->params()) max_value = std::max(max_value, p->id());
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst && inst->result()) max_value = std::max(max_value, inst->result()->id());
        }
    }
    out->set_next_block_id(max_block + 1);
    out->set_next_value_id(max_value + 1);

    Builder b(dst);
    b.set_function(out);
    BasicBlock* entry = b.append_block("twin.entry");
    Value* resume_id = b.add_block_param(entry, Type::i32());
    Value* state = b.add_block_param(entry, Type::ptr());
    b.position_at_end(entry);

    // A state value that is not a parameter of its entry's target is read
    // under its own name by the code after it, and may be defined there too
    // (or supplied by another entry): each one goes through a stack slot,
    // stored by every entry that carries it and at its definition, which
    // SROA turns back into SSA form below.
    std::vector<const Value*> slotted;
    {
        std::unordered_set<const Value*> seen;
        for (const auto& [id, e] : entries) {
            for (size_t i = e.target->param_count(); i < e.state.size(); ++i) {
                if (seen.insert(e.state[i]).second) slotted.push_back(e.state[i]);
            }
        }
        std::sort(slotted.begin(), slotted.end(), [](const Value* a, const Value* c) { return a->id() < c->id(); });
    }
    std::unordered_map<const Value*, Value*> slots;  // fn's value -> its slot
    for (const Value* v : slotted) {
        // A tagged value's slot is a root, as its value is.
        slots.emplace(v, v->type().is_tagged() ? b.build_alloca_tagged(1) : b.build_alloca(8, 8));
    }

    // The region's blocks and their parameters, then each entry.
    ir::ValueMap values;
    ir::BlockMap blocks;
    for (const BasicBlock* src_bb : region) {
        BasicBlock* bb = ir::new_block(*out, src_bb->name());
        bb->set_cold(src_bb->is_cold());
        blocks[src_bb] = bb;
        for (const Value* p : src_bb->params()) values[p] = ir::new_block_param(*out, bb, p->type());
    }
    BasicBlock* bad = ir::new_block(*out, "twin.bad_id");
    bad->set_cold();
    std::vector<SwitchCase> cases;
    for (const auto& [id, e] : entries) {
        BasicBlock* load = ir::new_block(*out, "twin.r" + std::to_string(id));
        cases.emplace_back(static_cast<int64_t>(id), load);
        b.position_at_end(load);
        std::vector<Value*> args;
        std::vector<Type> types;
        for (size_t i = 0; i < e.state.size(); ++i) {
            const Type t = e.state[i]->type();
            types.push_back(t);
            Value* v = b.build_load(t, state, static_cast<int32_t>(i * 8));
            if (i < e.target->param_count()) {
                args.push_back(v);
            } else {
                b.build_store(t, slots.at(e.state[i]), 0, v);
            }
        }
        b.build_br(blocks.at(e.target), Span<Value* const>(args.data(), args.size()));
        out->set_twin_state_types(id, std::move(types));
    }
    b.position_at_end(entry);
    b.build_switch(resume_id, bad, Span<const SwitchCase>(cases.data(), cases.size()));
    b.position_at_end(bad);
    b.build_unreachable();

    // The region's instructions. A site-form guard ends its block with a
    // branch: on to the rest of the block, or to the resume block.
    struct GuardBranch {
        const Instruction* guard;
        BasicBlock* from;
        BasicBlock* ok;
    };
    std::vector<GuardBranch> guard_branches;
    std::vector<std::pair<const Instruction*, Instruction*>> cloned;
    for (const BasicBlock* src_bb : region) {
        BasicBlock* bb = blocks.at(src_bb);
        for (const Instruction* inst : *const_cast<BasicBlock*>(src_bb)) {
            if (!inst) continue;
            if (resumes_in_table(fn, *inst)) {
                BasicBlock* ok = ir::new_block(*out, std::string(src_bb->name()) + ".g" +
                                                         std::to_string(inst->resume_id()));
                ok->set_cold(src_bb->is_cold());
                guard_branches.push_back({inst, bb, ok});
                bb = ok;
                continue;
            }
            Instruction* copy = ir::clone_shell(*out, *inst, values);
            bb->append_instruction(copy);
            copy->set_parent(bb);
            cloned.emplace_back(inst, copy);
        }
    }
    for (const auto& [src_inst, copy] : cloned) ir::clone_uses(*src_inst, *copy, values, blocks);
    auto mapped = [&](const Value* v) -> Value* {
        auto it = values.find(v);
        return it != values.end() ? it->second : const_cast<Value*>(v);
    };
    for (const GuardBranch& gb : guard_branches) {
        const BasicBlock* target = fn.get_resume_target(gb.guard->resume_id());
        std::vector<Value*> args;
        for (size_t i = 0; i < target->param_count(); ++i) args.push_back(mapped(gb.guard->state_map()[i]));
        b.position_at_end(gb.from);
        b.build_br_if(mapped(gb.guard->operand(0)), gb.ok, Span<Value* const>(), blocks.at(target),
                      Span<Value* const>(args.data(), args.size()));
    }

    // Slotted values: every read in the region loads the slot just before
    // it, and a definition in the region stores to it. A value defined
    // outside the region has no copy; its reads still name fn's value.
    if (!slots.empty()) {
        std::unordered_map<const Value*, std::pair<Value*, Type>> by_name;  // the value as read -> (slot, type)
        for (const auto& [v, slot] : slots) by_name.emplace(mapped(v), std::make_pair(slot, v->type()));
        std::unordered_set<const BasicBlock*> region_copies;
        for (const auto& [src_bb, bb] : blocks) region_copies.insert(bb);
        for (const GuardBranch& gb : guard_branches) region_copies.insert(gb.ok);
        std::vector<Instruction*> readers;
        for (BasicBlock* bb : out->blocks()) {
            if (!region_copies.count(bb)) continue;
            for (Instruction* inst : *bb) {
                bool reads = false;
                for_each_use(*inst, [&](Value* u) { reads |= by_name.count(u) != 0; });
                if (reads) readers.push_back(inst);
            }
        }
        for (Instruction* inst : readers) {
            for_each_use_slot(*inst, [&](Value*& u) {
                auto it = u ? by_name.find(u) : by_name.end();
                if (it == by_name.end()) return;
                b.position_before(inst);
                u = b.build_load(it->second.second, it->second.first);
            });
        }
        for (const auto& [v, slot] : slots) {
            const BasicBlock* def_block = defining_block(v);
            if (!def_block || !reach.count(def_block)) continue;
            Value* def = values.at(v);
            if (def->is_block_param()) {
                BasicBlock* bb = def->defining_block();
                if (bb->head()) {
                    b.position_before(bb->head());
                } else {
                    b.position_at_end(bb);
                }
            } else {
                Instruction* d = def->defining_instruction();
                if (d->is_terminator()) {
                    // An invoke's result exists only on its normal edge: the
                    // store goes there, in a block of its own.
                    BranchTarget& normal = d->normal_target();
                    BasicBlock* on_edge = ir::new_block(*out, "twin.def");
                    b.position_at_end(on_edge);
                    b.build_br(normal.block, Span<Value* const>(normal.args.data(), normal.args.size()));
                    normal = BranchTarget(on_edge);
                    b.position_before(on_edge->head());
                } else if (d->next()) {
                    b.position_before(d->next());
                } else {
                    b.position_at_end(d->parent());
                }
            }
            b.build_store(v->type(), slot, 0, def);
        }
    }

    out->rebuild_cfg_predecessors();
    if (!slots.empty()) sroa_function(*out);
    // A reference must not sit in a frame word the collector does not know:
    // every slot of one has to be gone again.
    std::unordered_set<const Value*> ref_slots;
    for (const auto& [v, slot] : slots) {
        if (v->type().is_gcref()) ref_slots.insert(slot);
    }
    for (const BasicBlock* bb : out->blocks()) {
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst && inst->result() && ref_slots.count(inst->result())) {
                fail(why, "a gcref state value's slot could not be turned back into a value");
                return nullptr;
            }
        }
    }
    DiagnosticReporter diag;
    if (!verify_function(*out, &diag)) {
        fail(why, "the resume twin does not verify:\n" + diag.format_all());
        return nullptr;
    }
    return out;
}

Function* build_twin_entry(const Function& fn, const Function& twin, uint32_t entry_id, std::string_view name,
                           std::string* why) {
    Module* mod = twin.parent();
    const std::vector<Type>* want = twin.twin_state_types(entry_id);
    if (!mod || !twin.is_resume_twin() || !want || *want != fn.param_types() ||
        twin.return_type() != fn.return_type()) {
        fail(why, "'" + std::string(twin.name()) + "' has no entry " + std::to_string(entry_id) + " taking '" +
                      std::string(fn.name()) + "' arguments");
        return nullptr;
    }
    Function* out = mod->create_function(name, fn.return_type(), fn.param_types());
    Builder b(*mod);
    b.set_function(out);
    BasicBlock* entry = b.append_block("entry");
    std::vector<Value*> args;
    for (Type t : fn.param_types()) args.push_back(b.add_block_param(entry, t));
    b.position_at_end(entry);
    // The twin reads the words before its first GC point, so a reference in
    // one needs no root here.
    Value* words = b.build_alloca(static_cast<uint32_t>(std::max<size_t>(args.size(), 1) * 8), 8);
    for (size_t i = 0; i < args.size(); ++i) {
        b.build_store(args[i]->type(), words, static_cast<int32_t>(i * 8), args[i]);
    }
    Value* r = b.build_call(twin.name(), fn.return_type(), {b.build_iconst_i32(static_cast<int32_t>(entry_id)), words});
    if (fn.return_type().is_void()) {
        b.build_ret_void();
    } else {
        b.build_ret(r);
    }
    out->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    if (!verify_function(*out, &diag)) {
        fail(why, "the twin entry does not verify:\n" + diag.format_all());
        return nullptr;
    }
    return out;
}

bool speculate_against_twin(Function& fn, const Function& twin, std::string* why) {
    Module* mod = fn.parent();
    if (!mod || twin.parent() != mod || !twin.is_resume_twin()) {
        return fail(why, "'" + std::string(twin.name()) + "' is not a resume twin of the function's module");
    }
    std::vector<Instruction*> guards;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (inst && resumes_in_table(fn, *inst)) guards.push_back(inst);
        }
    }
    const std::string_view twin_name = mod->string_pool().intern(twin.name());
    for (Instruction* guard : guards) {
        std::string_view old = guard->symbol();
        guard->set_symbol(twin_name);
        std::string mismatch;
        if (!fn.guard_exit_stub_matches(*guard, twin, mismatch)) {
            guard->set_symbol(old);
            return fail(why, "guard " + std::to_string(guard->resume_id()) + ": " + mismatch);
        }
    }
    // The twin holds the resume blocks now. Resume points left are other
    // code's (a coroutine's suspends), and stay.
    for (Instruction* guard : guards) fn.remove_resume_point(guard->resume_id());
    fn.rebuild_cfg_predecessors();
    CfgSimplifyOptions prune;
    prune.enable_block_merge = false;
    prune.enable_param_elimination = false;
    prune.enable_trampoline_elimination = false;
    prune.max_iterations = 2;
    cfg_simplify_function(fn, prune);
    DiagnosticReporter diag;
    if (!verify_function(fn, &diag)) {
        return fail(why, "the speculated function does not verify:\n" + diag.format_all());
    }
    return true;
}

std::string resume_twin_name(std::string_view function_name) { return std::string(function_name) + "__twin"; }

bool speculate_module_against_twins(Module& mod, std::string* why) {
    std::vector<Function*> sited;
    for (Function* fn : mod.functions()) {
        if (!fn || fn->is_resume_twin()) continue;
        bool has = false;
        for (const BasicBlock* bb : fn->blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) has |= inst && resumes_in_table(*fn, *inst);
        }
        if (has) sited.push_back(fn);
    }
    for (Function* fn : sited) {
        std::string reason;
        const Function* twin = build_resume_twin(*fn, mod, resume_twin_name(fn->name()), {}, &reason);
        if (!twin || !speculate_against_twin(*fn, *twin, &reason)) {
            return fail(why, "'" + std::string(fn->name()) + "': " + reason);
        }
    }
    return true;
}

} // namespace brass
