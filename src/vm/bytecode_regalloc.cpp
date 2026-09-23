#include "bytecode_regalloc.hpp"

#include <algorithm>
#include <queue>
#include <stdexcept>
#include <string>

namespace brass::detail {

std::vector<const BranchTarget*> branch_targets_of(const Instruction& inst) {
    std::vector<const BranchTarget*> out;
    switch (inst.opcode()) {
        case Opcode::br:
            out.push_back(&inst.branch_target());
            break;
        case Opcode::br_if:
            out.push_back(&inst.true_target());
            out.push_back(&inst.false_target());
            break;
        case Opcode::switch_:
            for (const auto& sc : inst.switch_cases()) out.push_back(&sc.target);
            out.push_back(&inst.default_target());
            break;
        case Opcode::invoke:
            out.push_back(&inst.normal_target());
            out.push_back(&inst.unwind_target());
            break;
        default:
            break;
    }
    return out;
}

BlockLayout build_block_layout(const Function& fn) {
    BlockLayout layout;
    const BasicBlock* entry = fn.entry_block();
    if (entry) layout.order.push_back(entry);
    for (const BasicBlock* bb : fn.blocks()) {
        if (bb && bb != entry) layout.order.push_back(bb);
    }
    for (uint32_t i = 0; i < layout.order.size(); ++i) {
        layout.index.emplace(layout.order[i], i);
    }

    auto block_index = [&](const BasicBlock* bb) -> uint32_t {
        auto it = layout.index.find(bb);
        if (it == layout.index.end()) {
            throw std::runtime_error("Function @" + std::string(fn.name()) +
                                     ": branch to a block that is not in the function");
        }
        return it->second;
    };

    std::unordered_map<uint32_t, const BasicBlock*> resume_targets;
    for (const auto& [id, target] : fn.resume_points()) resume_targets.emplace(id, target);

    const size_t n = layout.order.size();
    layout.succs.assign(n, {});
    layout.preds.assign(n, {});
    for (uint32_t bi = 0; bi < n; ++bi) {
        auto add_edge = [&](const BasicBlock* to) {
            if (!to) return;
            uint32_t ti = block_index(to);
            auto& s = layout.succs[bi];
            if (std::find(s.begin(), s.end(), ti) == s.end()) {
                s.push_back(ti);
                layout.preds[ti].push_back(bi);
            }
        };
        for (const Instruction* inst : *layout.order[bi]) {
            if (!inst) continue;
            for (const BranchTarget* t : branch_targets_of(*inst)) add_edge(t->block);
            if (inst->opcode() == Opcode::guard) {
                // A failing guard may re-enter this frame at its resume block.
                auto it = resume_targets.find(inst->resume_id());
                if (it != resume_targets.end()) add_edge(it->second);
            }
        }
    }
    return layout;
}

namespace {

struct ValueInfo {
    const Value* value = nullptr;
    uint32_t def_block = 0;
    uint32_t def_pos = 0;
    uint32_t lo = 0;
    uint32_t hi = 0;
    int32_t fixed_reg = -1; // entry block parameter i: register i
};

size_t type_class(Type t) { return static_cast<size_t>(t.kind()); }

} // namespace

RegisterAssignment allocate_bytecode_registers(const Function& fn, const BlockLayout& layout) {
    const size_t nblocks = layout.order.size();
    std::vector<uint32_t> block_start(nblocks, 0), block_end(nblocks, 0);

    std::vector<ValueInfo> values;
    std::unordered_map<const Value*, uint32_t> vindex;
    auto define = [&](const Value* v, uint32_t block, uint32_t pos, int32_t fixed) {
        if (!v) return;
        if (!vindex.emplace(v, static_cast<uint32_t>(values.size())).second) {
            throw std::runtime_error("Function @" + std::string(fn.name()) + ": value %" +
                                     std::to_string(v->id()) + " is defined twice");
        }
        ValueInfo vi;
        vi.value = v;
        vi.def_block = block;
        vi.def_pos = pos;
        vi.lo = pos;
        vi.hi = pos;
        vi.fixed_reg = fixed;
        values.push_back(vi);
    };

    // Positions: one for each block start (where its parameters are
    // defined) and one for each instruction.
    uint32_t pos = 0;
    for (uint32_t bi = 0; bi < nblocks; ++bi) {
        const BasicBlock* bb = layout.order[bi];
        block_start[bi] = pos;
        for (size_t p = 0; p < bb->param_count(); ++p) define(bb->param(p), bi, pos, bi == 0 ? static_cast<int32_t>(p) : -1);
        ++pos;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            if (inst->produces_value() && inst->result()) define(inst->result(), bi, pos, -1);
            ++pos;
        }
        block_end[bi] = pos - 1;
    }

    // Uses, and the live-range hull of every value.
    std::vector<uint32_t> stamp(nblocks, 0);
    std::vector<uint32_t> worklist;
    auto use = [&](const Value* v, uint32_t bi, uint32_t at) {
        if (!v) return;
        auto it = vindex.find(v);
        if (it == vindex.end()) {
            throw std::runtime_error("Function @" + std::string(fn.name()) + ": value %" +
                                     std::to_string(v->id()) + " is used but never defined");
        }
        ValueInfo& vi = values[it->second];
        vi.lo = std::min(vi.lo, at);
        vi.hi = std::max(vi.hi, at);
        if (bi == vi.def_block) return;
        // Walk back from the use to the definition: the value is live into
        // every block on the way and live out of each of their predecessors.
        const uint32_t tag = it->second + 1;
        auto live_in = [&](uint32_t b) {
            if (stamp[b] == tag) return;
            stamp[b] = tag;
            worklist.push_back(b);
        };
        live_in(bi);
        while (!worklist.empty()) {
            uint32_t b = worklist.back();
            worklist.pop_back();
            vi.lo = std::min(vi.lo, block_start[b]);
            vi.hi = std::max(vi.hi, block_start[b]);
            for (uint32_t p : layout.preds[b]) {
                vi.lo = std::min(vi.lo, block_end[p]);
                vi.hi = std::max(vi.hi, block_end[p]);
                if (p != vi.def_block) live_in(p);
            }
        }
    };
    // Each use walk must see fresh stamps for its value only; stamps are
    // tagged by value, and repeated walks of one value may share them.
    for (uint32_t bi = 0; bi < nblocks; ++bi) {
        uint32_t at = block_start[bi];
        for (const Instruction* inst : *layout.order[bi]) {
            if (!inst) continue;
            ++at;
            for (size_t i = 0; i < inst->operand_count(); ++i) use(inst->operand(i), bi, at);
            for (const Value* v : inst->state_map()) use(v, bi, at);
            for (const BranchTarget* t : branch_targets_of(*inst)) {
                for (const Value* v : t->args) use(v, bi, at);
            }
        }
    }

    // Linear scan over the hulls, reusing registers per type.
    RegisterAssignment out;
    out.num_values = static_cast<uint32_t>(values.size());
    std::vector<uint32_t> order(values.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return values[a].lo < values[b].lo;
    });

    using Active = std::pair<uint32_t, uint32_t>; // (hi, value index)
    std::priority_queue<Active, std::vector<Active>, std::greater<Active>> active;
    std::vector<std::vector<BcReg>> free_regs;
    uint32_t next_reg = 0;

    auto fresh = [&](Type t) -> BcReg {
        if (next_reg >= kMaxBytecodeRegisters - 2) {
            throw std::runtime_error("Function @" + std::string(fn.name()) + " needs more than " +
                                     std::to_string(kMaxBytecodeRegisters - 2) +
                                     " simultaneously live bytecode registers");
        }
        out.register_types.push_back(t);
        return static_cast<BcReg>(next_reg++);
    };

    // Entry parameters hold the arguments in registers 0..n-1, including
    // parameters that are absent or never used.
    const BasicBlock* entry = nblocks ? layout.order[0] : nullptr;
    const size_t nparams = entry ? entry->param_count() : fn.param_count();
    for (size_t p = 0; p < nparams; ++p) {
        const Value* v = entry ? entry->param(p) : nullptr;
        Type t = v ? v->type() : (p < fn.param_count() ? fn.param_type(p) : Type::i64());
        fresh(t);
    }

    for (uint32_t vi_idx : order) {
        const ValueInfo& vi = values[vi_idx];
        while (!active.empty() && active.top().first < vi.lo) {
            const ValueInfo& done = values[active.top().second];
            BcReg r = out.reg.at(done.value);
            size_t cls = type_class(done.value->type());
            if (free_regs.size() <= cls) free_regs.resize(cls + 1);
            free_regs[cls].push_back(r);
            active.pop();
        }
        BcReg r = 0;
        if (vi.fixed_reg >= 0) {
            r = static_cast<BcReg>(vi.fixed_reg);
        } else {
            size_t cls = type_class(vi.value->type());
            if (cls < free_regs.size() && !free_regs[cls].empty()) {
                r = free_regs[cls].back();
                free_regs[cls].pop_back();
            } else {
                r = fresh(vi.value->type());
            }
        }
        out.reg.emplace(vi.value, r);
        active.emplace(vi.hi, vi_idx);
    }
    out.num_registers = next_reg;
    return out;
}

} // namespace brass::detail
