#include <brass/mir/code_sinking.hpp>

#include <brass/mir/critical_edge.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/uses.hpp>

#include "ir_clone.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

bool is_gc_typed(const Value* v) { return v && v->type().is_gc_root(); }

// `gc_after`: a GC point lies between `inst` and the end of its block.
bool sinkable(const Instruction& inst, bool gc_after) {
    if (!inst.produces_value() || inst.has_side_effects()) return false;
    const Opcode op = inst.opcode();
    // Bits made a tagged value: before a GC point the collector keeps the
    // value current, after one the bits would be what it was.
    const bool boxes = op == Opcode::bitcast_tagged_i64;
    if (boxes && gc_after) return false;
    if (!(boxes || is_arithmetic(op) || is_bitwise(op) || is_comparison(op) || is_conversion(op) || is_select(op))) {
        return false;
    }
    switch (op) {
        case Opcode::sadd_overflow:
        case Opcode::ssub_overflow:
        case Opcode::smul_overflow:
        case Opcode::uadd_overflow:
        case Opcode::usub_overflow:
        case Opcode::umul_overflow:
            return false;
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::smod:
        case Opcode::umod:
            if (!inst.type().is_float()) return false;
            break;
        default:
            break;
    }
    if ((!boxes && is_gc_typed(inst.result())) || is_derived_gcref(inst.result())) return false;
    if (!inst.state_map().empty()) return false;
    for (const Value* v : inst.operands()) {
        if (!v || is_gc_typed(v) || is_derived_gcref(v)) return false;
    }
    return true;
}

// More exits than this reading one value and it stays where it is.
constexpr size_t kMaxCopies = 4;

// The one successor of `bb` an instruction of it may sink into when every
// use lies there: a block entered only from `bb`, by one edge that carries no
// arguments. A deoptimization's resume target (`resumed`) is entered from
// outside the graph as well.
bool edge_takes(const BasicBlock* bb, const BasicBlock* succ, const std::unordered_set<const BasicBlock*>& resumed) {
    if (succ == bb || succ->param_count() != 0 || resumed.count(succ) != 0) return false;
    if (succ->head() && succ->head()->opcode() == Opcode::landing_pad) return false;
    const auto& preds = succ->predecessors();
    if (preds.size() != 1 || preds[0] != bb) return false;
    size_t edges = 0;
    for_each_edge(*bb->terminator(), [&](const BranchTarget& t) { edges += t.block == succ; });
    return edges == 1;
}

// A block entered by one edge needs no parameters: each is the argument the
// edge passes. A split edge's block is left with them (critical_edge.hpp),
// and with them it is a block nothing can be moved into ahead of its branch.
void forward_single_edge_params(Function& fn, const std::unordered_set<const BasicBlock*>& resumed) {
    std::unordered_map<const Value*, Value*> is;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || bb == fn.entry_block() || bb->param_count() == 0 || bb->predecessors().size() != 1) continue;
        if (resumed.count(bb) != 0) continue;
        Instruction* term = bb->predecessors()[0]->terminator();
        if (!term || term->opcode() == Opcode::invoke) continue;
        BranchTarget* edge = nullptr;
        size_t edges = 0;
        for_each_edge(*term, [&](BranchTarget& t) {
            if (t.block == bb) {
                edge = &t;
                ++edges;
            }
        });
        if (edges != 1 || edge->args.size() != bb->param_count()) continue;
        for (size_t i = 0; i < bb->param_count(); ++i) is[bb->param(i)] = edge->args[i];
        edge->args.clear();
        bb->params().clear();
    }
    if (is.empty()) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            for_each_use_slot(*inst, [&](Value*& v) {
                // An argument may itself be a forwarded parameter.
                for (auto it = is.find(v); it != is.end(); it = is.find(v)) v = it->second;
            });
        }
    }
}

// More blocks than this between a value and the join that reads it and it
// stays where it is.
constexpr size_t kMaxJoinRegion = 16;

// `users` all lie in one block that some path out of `bb` does not reach,
// with no GC point and no loop between: that block, else null. The blocks
// between are the ones that reach it without passing `bb` (which dominates
// it); an edge out of them or out of `bb` to anywhere else is a path that
// goes without the value.
BasicBlock* join_that_takes(const BasicBlock* bb, const std::vector<Instruction*>& users,
                            const std::unordered_set<const BasicBlock*>& resumed) {
    BasicBlock* join = users.front()->parent();
    for (const Instruction* u : users) {
        if (u->parent() != join) return nullptr;
    }
    if (join == bb || resumed.count(join) != 0) return nullptr;
    if (join->head() && join->head()->opcode() == Opcode::landing_pad) return nullptr;
    std::vector<const BasicBlock*> between;
    std::vector<const BasicBlock*> work{join};
    while (!work.empty()) {
        const BasicBlock* at = work.back();
        work.pop_back();
        for (const BasicBlock* p : at->predecessors()) {
            if (p == join) return nullptr;
            if (p == bb || std::find(between.begin(), between.end(), p) != between.end()) continue;
            if (between.size() == kMaxJoinRegion || resumed.count(p) != 0) return nullptr;
            for (const Instruction* inst : *p) {
                if (may_trigger_gc(*inst)) return nullptr;
            }
            between.push_back(p);
            work.push_back(p);
        }
    }
    auto leaves = [&](const BasicBlock* from) {
        for (const BasicBlock* s : from->successors()) {
            if (s != join && std::find(between.begin(), between.end(), s) == between.end()) return true;
        }
        return false;
    };
    if (leaves(bb) || std::any_of(between.begin(), between.end(), leaves)) return join;
    return nullptr;
}

} // namespace

size_t sink_code(Function& fn) {
    if (fn.blocks().empty() || !fn.entry_block()) return 0;
    split_critical_edges(fn);
    fn.rebuild_cfg_predecessors();
    std::unordered_set<const BasicBlock*> resumed;
    for (const auto& rp : fn.resume_points()) {
        if (rp.second) resumed.insert(rp.second);
    }
    forward_single_edge_params(fn, resumed);
    const DominatorTree dom(fn);

    std::unordered_map<const Value*, std::vector<Instruction*>> users;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (inst) for_each_use(*inst, [&](Value* v) { users[v].push_back(inst); });
        }
    }

    size_t moved = 0;
    // Dominator preorder: what a block receives is weighed again against its
    // own branch. Within a block last to first, so a chain follows its end.
    for (const BasicBlock* cbb : dom.reachable_blocks()) {
        BasicBlock* bb = const_cast<BasicBlock*>(cbb);
        Instruction* term = bb->terminator();
        if (!term) continue;
        std::vector<BasicBlock*> targets;
        for_each_edge(*term, [&](const BranchTarget& t) {
            if (edge_takes(bb, t.block, resumed)) targets.push_back(t.block);
        });
        if (bb->successors().size() < 2) continue;

        bool gc_after = may_trigger_gc(*term);
        for (Instruction* inst = term->prev(); inst;) {
            Instruction* prev = inst->prev();
            const auto it = users.find(inst->result());
            if (sinkable(*inst, gc_after) && it != users.end() && !it->second.empty()) {
                // Each use's edge: the target that dominates it.
                std::vector<std::pair<BasicBlock*, std::vector<Instruction*>>> groups;
                bool all = true;
                for (Instruction* u : it->second) {
                    BasicBlock* at = nullptr;
                    if (u->parent() != bb) {
                        for (BasicBlock* t : targets) {
                            if (dom.dominates(t, u->parent())) {
                                at = t;
                                break;
                            }
                        }
                    }
                    if (!at) {
                        all = false;
                        break;
                    }
                    auto g = std::find_if(groups.begin(), groups.end(), [&](const auto& e) { return e.first == at; });
                    if (g == groups.end()) g = groups.insert(groups.end(), {at, {}});
                    g->second.push_back(u);
                }
                // Copies pay when an edge goes without one, or when one lands
                // in a block that only passes it on, to be weighed again
                // there; a copy for every edge, each read at once, is the
                // same work in more code.
                bool pays = all && groups.size() <= kMaxCopies;
                if (pays && groups.size() > 1 && groups.size() == bb->successors().size()) {
                    pays = false;
                    for (const auto& [at, us] : groups) {
                        const BasicBlock* t = at;
                        pays = pays || std::none_of(us.begin(), us.end(),
                                                    [&](const Instruction* u) { return u->parent() == t; });
                    }
                }
                if (pays) {
                    // One edge is taken, so one copy runs where one ran.
                    for (size_t i = 1; i < groups.size(); ++i) {
                        ir::ValueMap values;
                        Instruction* copy = ir::clone_instruction(fn, *inst, values, {});
                        groups[i].first->prepend_instruction(copy);
                        for (Instruction* u : groups[i].second) replace_uses_in(*u, inst->result(), copy->result());
                        users[copy->result()] = std::move(groups[i].second);
                        for (Value* v : inst->operands()) users[v].push_back(copy);
                    }
                    bb->remove_instruction(inst);
                    groups[0].first->prepend_instruction(inst);
                    users[inst->result()] = std::move(groups[0].second);
                    moved += groups.size();
                } else if (BasicBlock* join = gc_after ? nullptr : join_that_takes(bb, it->second, resumed)) {
                    bb->remove_instruction(inst);
                    join->prepend_instruction(inst);
                    ++moved;
                }
            }
            if (inst->parent() == bb && may_trigger_gc(*inst)) gc_after = true;
            inst = prev;
        }
    }
    return moved;
}

} // namespace brass
