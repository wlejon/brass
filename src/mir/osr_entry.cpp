// OSR entry functions (osr_entry.hpp): the plan (liveness at the entry
// block, the region reachable from it) and the function built from it.

#include <brass/mir/osr_entry.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/uses.hpp>
#include <brass/mir/verifier.hpp>
#include "ir_clone.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace brass {

namespace {

using ValueSet = std::unordered_set<const Value*>;

bool fail(std::string* why, std::string msg) {
    if (why) *why = std::move(msg);
    return false;
}

// The values live into each block of `fn` (block parameters are defined by
// their block, so never live into it): the usual backward dataflow, with an
// edge's arguments read by its source block.
std::unordered_map<const BasicBlock*, ValueSet> live_in_sets(const Function& fn) {
    struct BlockInfo {
        ValueSet use;
        ValueSet def;
        std::vector<const BasicBlock*> succs;
    };
    std::unordered_map<const BasicBlock*, BlockInfo> info;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        BlockInfo& bi = info[bb];
        for (const Value* p : bb->params()) bi.def.insert(p);
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (!inst) continue;
            for_each_use(*inst, [&](Value* v) {
                // Only SSA values are live; a constant is defined like any
                // other instruction, so it is one too.
                if (!bi.def.count(v)) bi.use.insert(v);
            });
            if (inst->result()) bi.def.insert(inst->result());
        }
        for (const BasicBlock* s : bb->successors()) {
            if (s) bi.succs.push_back(s);
        }
    }
    std::unordered_map<const BasicBlock*, ValueSet> live_in;
    for (auto& [bb, bi] : info) live_in[bb] = bi.use;
    bool changed = true;
    while (changed) {
        changed = false;
        for (auto it = fn.blocks().rbegin(); it != fn.blocks().rend(); ++it) {
            const BasicBlock* bb = *it;
            if (!bb) continue;
            BlockInfo& bi = info[bb];
            ValueSet& in = live_in[bb];
            for (const BasicBlock* s : bi.succs) {
                for (const Value* v : live_in[s]) {
                    if (!bi.def.count(v) && in.insert(v).second) changed = true;
                }
            }
        }
    }
    return live_in;
}

bool is_rematerializable(const Value* v) {
    const Instruction* def = v->is_instruction() ? v->defining_instruction() : nullptr;
    if (!def || def->operand_count() != 0) return false;
    return is_constant(def->opcode()) || def->opcode() == Opcode::func_addr;
}

} // namespace

std::optional<OsrEntryPlan> plan_osr_entry(const Function& fn, const BasicBlock& block, std::string* why) {
    if (!fn.resume_points().empty()) {
        fail(why, "a coroutine (it has resume points)");
        return std::nullopt;
    }
    if (fn.entry_block() == &block) {
        fail(why, "the entry block is not an OSR entry");
        return std::nullopt;
    }

    OsrEntryPlan plan;
    plan.function = &fn;
    plan.block = &block;

    // The region: everything reachable from the block.
    std::unordered_set<const BasicBlock*> reach{&block};
    std::vector<const BasicBlock*> work{&block};
    while (!work.empty()) {
        const BasicBlock* bb = work.back();
        work.pop_back();
        for (const BasicBlock* s : bb->successors()) {
            if (s && reach.insert(s).second) work.push_back(s);
        }
    }
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb && reach.count(bb)) plan.region.push_back(bb);
    }

    const auto live_in = live_in_sets(fn);
    std::vector<const Value*> live(live_in.at(&block).begin(), live_in.at(&block).end());
    std::sort(live.begin(), live.end(), [](const Value* a, const Value* b) { return a->id() < b->id(); });

    auto add = [&](const Value* v) -> bool {
        const Type t = v->type();
        if (t.is_vector()) return fail(why, "value %" + std::to_string(v->id()) + " is a vector");
        if (t.is_gcref()) return fail(why, "value %" + std::to_string(v->id()) + " is a gcref");
        plan.live_ins.push_back({v, is_rematerializable(v)});
        return true;
    };
    for (const Value* p : block.params()) {
        if (!add(p)) return std::nullopt;
    }
    for (const Value* v : live) {
        if (!add(v)) return std::nullopt;
    }
    return plan;
}

Function* build_osr_entry_function(const OsrEntryPlan& plan, Module& dst, std::string_view name, std::string* why) {
    const Function& fn = *plan.function;
    const std::vector<Type> params{Type::ptr()};
    Function* out = dst.create_function(name, fn.return_type(), params);
    out->set_allow_fp_reassociation(fn.allow_fp_reassociation());
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

    std::unordered_set<const BasicBlock*> in_region(plan.region.begin(), plan.region.end());

    // The entry: every live value, loaded or recomputed.
    Builder b(dst);
    b.set_function(out);
    BasicBlock* entry = b.append_block("osr.entry");
    Value* buffer = b.add_block_param(entry, Type::ptr());
    b.position_at_end(entry);
    ir::ValueMap values;
    ir::BlockMap blocks;
    std::vector<std::pair<const Value*, Value*>> incoming;  // live value -> its value on entry
    for (size_t i = 0; i < plan.live_ins.size(); ++i) {
        const auto& li = plan.live_ins[i];
        Value* v = nullptr;
        if (li.rematerialize) {
            Instruction* copy = ir::clone_shell(*out, *li.value->defining_instruction(), values);
            entry->append_instruction(copy);
            copy->set_parent(entry);
            v = copy->result();
        } else {
            v = b.build_load(li.value->type(), buffer, static_cast<int32_t>(i * 8));
        }
        incoming.emplace_back(li.value, v);
    }

    // A live value defined in the region (an enclosing loop's, reached again
    // through its latch) has two definitions in the entry function: the
    // entry's and its own. It goes through a stack slot, stored at both.
    std::unordered_map<const Value*, Value*> slots;  // fn's value -> slot
    std::vector<Value*> header_args;
    for (size_t i = 0; i < incoming.size(); ++i) {
        const Value* v = incoming[i].first;
        const bool header_param = i < plan.block->param_count();
        if (header_param) {
            header_args.push_back(incoming[i].second);
            continue;
        }
        const BasicBlock* def_block = v->is_block_param() ? v->defining_block()
                                      : v->defining_instruction() ? v->defining_instruction()->parent()
                                                                  : nullptr;
        if (def_block && in_region.count(def_block) && !plan.live_ins[i].rematerialize) {
            // A tagged value's slot is a root, as its value is.
            Value* slot = v->type().is_tagged() ? b.build_alloca_tagged(1) : b.build_alloca(8, 8);
            b.build_store(v->type(), slot, 0, incoming[i].second);
            slots.emplace(v, slot);
        } else {
            values[v] = incoming[i].second;
        }
    }

    // The region, cloned: blocks and their parameters first, then the
    // instructions, then their uses.
    for (const BasicBlock* src_bb : plan.region) {
        BasicBlock* bb = ir::new_block(*out, src_bb->name());
        blocks[src_bb] = bb;
        for (const Value* p : src_bb->params()) values[p] = ir::new_block_param(*out, bb, p->type());
    }
    b.build_br(blocks.at(plan.block), Span<Value* const>(header_args.data(), header_args.size()));

    std::vector<std::pair<const Instruction*, Instruction*>> cloned;
    for (const BasicBlock* src_bb : plan.region) {
        BasicBlock* bb = blocks.at(src_bb);
        for (const Instruction* inst : *const_cast<BasicBlock*>(src_bb)) {
            if (!inst) continue;
            Instruction* copy = ir::clone_shell(*out, *inst, values);
            bb->append_instruction(copy);
            copy->set_parent(bb);
            cloned.emplace_back(inst, copy);
        }
    }
    for (const auto& [src_inst, copy] : cloned) ir::clone_uses(*src_inst, *copy, values, blocks);

    if (!slots.empty()) {
        // Every read of a slotted value loads the slot just before it, and
        // each of its definitions in the region stores to it; SROA then
        // rebuilds the value's SSA form with block parameters.
        std::unordered_map<const Value*, std::pair<Value*, Type>> by_clone;  // clone -> (slot, type)
        for (const auto& [v, slot] : slots) by_clone.emplace(values.at(v), std::make_pair(slot, v->type()));
        std::vector<Instruction*> readers;
        for (BasicBlock* bb : out->blocks()) {
            if (bb == entry) continue;
            for (Instruction* inst : *bb) {
                bool reads = false;
                for_each_use(*inst, [&](Value* u) { reads |= by_clone.count(u) != 0; });
                if (reads) readers.push_back(inst);
            }
        }
        for (Instruction* inst : readers) {
            for_each_use_slot(*inst, [&](Value*& u) {
                auto it = u ? by_clone.find(u) : by_clone.end();
                if (it == by_clone.end()) return;
                b.position_before(inst);
                u = b.build_load(it->second.second, it->second.first);
            });
        }
        for (const auto& [clone, st] : by_clone) {
            Value* def = const_cast<Value*>(clone);
            if (def->is_block_param()) {
                BasicBlock* bb = def->defining_block();
                if (bb->head()) {
                    b.position_before(bb->head());
                } else {
                    b.position_at_end(bb);
                }
            } else {
                Instruction* d = def->defining_instruction();
                if (d->next()) {
                    b.position_before(d->next());
                } else {
                    b.position_at_end(d->parent());
                }
            }
            b.build_store(st.second, st.first, 0, def);
        }
    }

    out->rebuild_cfg_predecessors();
    if (!slots.empty()) sroa_function(*out);
    DiagnosticReporter diag;
    if (!verify_function(*out, &diag)) {
        fail(why, "the OSR entry function does not verify:\n" + diag.format_all());
        return nullptr;
    }
    return out;
}

} // namespace brass
