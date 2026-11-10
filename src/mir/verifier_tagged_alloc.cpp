// A tagged allocation's payload is written before anything can look at it
// (Module::tagged_allocator, gc_contract.md 2.13).
#include "verifier_gc.hpp"
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/tagged_alloc.hpp>
#include <brass/mir/uses.hpp>

#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

// Payloads past this many words are checked by their producer alone.
constexpr int64_t kCheckedWords = 64;

// The words stored so far on every path from the call; all ones is the
// unvisited state.
using Words = uint64_t;

} // namespace

void verify_tagged_allocations(const Function& fn, const std::string& fn_prefix,
                               const std::function<void(const std::string&)>& report_error) {
    const Module* mod = fn.parent();
    if (!mod || !mod->tagged_allocator()) return;
    TaggedAllocations allocs(fn);
    for (const TaggedAllocSite& site : allocs.sites()) {
        const int64_t words = site.bytes / 8;
        if (site.bytes < 0 || words > kCheckedWords) continue;
        const Words full = words == 64 ? ~Words{0} : ((Words{1} << words) - 1);
        std::unordered_set<const Instruction*> inits;
        for (const TaggedUse& u : site.uses) {
            if (u.kind != TaggedUseKind::Store || u.offset < 0 || u.offset % 8 != 0 || u.offset >= site.bytes) continue;
            if (u.inst->memory_type().size_in_bytes() == 8) inits.insert(u.inst);
        }
        auto report = [&](const BasicBlock* bb, const Instruction* at) {
            report_error(fn_prefix + "Block '" + std::string(bb->name()) + "': Instruction '" +
                         std::string(opcode_name(at->opcode())) +
                         "': reached from a tagged allocation before every word of its payload is stored "
                         "(a GC point, a return or a throw must see the object whole).");
        };
        // Forward from the call: words meet by intersection, and every GC
        // point and exit must find them all.
        std::unordered_map<const BasicBlock*, Words> in;
        std::vector<const BasicBlock*> work;
        auto walk = [&](const BasicBlock* bb, const Instruction* from, Words have) {
            for (const Instruction* inst = from; inst; inst = inst->next()) {
                if (inst == site.call) return;  // the next object
                if (inits.count(inst)) {
                    for (const TaggedUse& u : site.uses) {
                        if (u.inst == inst) have |= Words{1} << (u.offset / 8);
                    }
                }
                const Opcode op = inst->opcode();
                const bool exits = op == Opcode::ret || op == Opcode::throw_ || op == Opcode::resume;
                if ((may_trigger_gc(*inst) || exits) && (have & full) != full) {
                    report(bb, inst);
                    return;
                }
            }
            if (const Instruction* term = bb->terminator()) {
                for_each_edge(*term, [&](const BranchTarget& t) {
                    auto [it, fresh] = in.try_emplace(t.block, ~Words{0});
                    const Words next = it->second & have;
                    if (fresh || next != it->second) {
                        it->second = next;
                        work.push_back(t.block);
                    }
                });
            }
        };
        walk(site.call->parent(), site.call->next(), 0);
        while (!work.empty()) {
            const BasicBlock* bb = work.back();
            work.pop_back();
            walk(bb, bb->head(), in.at(bb));
        }
    }
}

} // namespace brass
