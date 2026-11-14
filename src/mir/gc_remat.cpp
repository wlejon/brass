// Computing a value again after a point that made it stale (gc_remat.hpp).
#include "gc_remat.hpp"
#include "ir_clone.hpp"

#include <brass/mir/block.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/uses.hpp>

#include <algorithm>
#include <functional>
#include <unordered_set>
#include <utility>

namespace brass {

bool is_rematerializable(const Instruction& inst) {
    if (!inst.produces_value()) return false;
    const Opcode op = inst.opcode();
    return op == Opcode::bitcast_i64_tagged || op == Opcode::load || op == Opcode::load_indexed ||
           is_pure_instruction(&inst);
}

int GcSensitive::state(const Value* v) {
    // Only a word can hold an address. A narrower value made of a tagged
    // value's bits (a tag test, an int32 payload, a Number) reads nothing a
    // move changes: a collection keeps tags and identity.
    if (!v || v->type().is_tagged() || !(v->type().is_i64() || v->type().is_pointer())) return 0;
    if (const auto it = memo_.find(v); it != memo_.end()) return it->second;
    if (v->is_block_param()) {
        const int r = param_sensitive(v) ? 2 : 0;
        memo_[v] = r;
        return r;
    }
    const Instruction* d = v->is_instruction() ? v->defining_instruction() : nullptr;
    if (!d || !is_rematerializable(*d)) return memo_[v] = 0;
    if (d->opcode() == Opcode::bitcast_i64_tagged) return memo_[v] = 1;
    // A chain of pure instructions can be long: past this depth the value
    // counts as sensitive and not clonable, which moves nothing.
    if (depth_ > 1000) return 2;
    ++depth_;
    int r = 0;
    for (const Value* op : d->operands()) r = std::max(r, state(op));
    --depth_;
    return memo_[v] = r;
}

bool GcSensitive::param_sensitive(const Value* param) {
    std::vector<const Value*> work{param};
    std::unordered_set<const Value*> seen;
    while (!work.empty()) {
        const Value* p = work.back();
        work.pop_back();
        if (!seen.insert(p).second) continue;
        const BasicBlock* bb = p->defining_block();
        if (!bb) continue;
        const uint32_t idx = p->param_index();
        for (BasicBlock* pred : bb->predecessors()) {
            Instruction* term = pred ? pred->terminator() : nullptr;
            if (!term) continue;
            bool found = false;
            for_each_edge(*term, [&](BranchTarget& t) {
                if (t.block != bb || idx >= t.args.size() || !t.args[idx]) return;
                const Value* a = t.args[idx];
                if (a->type().is_tagged()) return;
                if (a->is_block_param()) {
                    work.push_back(a);
                } else if (const auto it = memo_.find(a); it != memo_.end() ? it->second != 0 : state(a) != 0) {
                    found = true;
                }
            });
            if (found) return true;
        }
    }
    return false;
}

namespace {

// The edges of `term` into `target`.
template <typename F>
void edges_into(Instruction& term, const BasicBlock* target, F&& f) {
    for_each_edge(term, [&](BranchTarget& t) {
        if (t.block == target) f(t);
    });
}

std::vector<BasicBlock*> unique_preds(const BasicBlock* bb) {
    std::vector<BasicBlock*> out;
    for (BasicBlock* p : bb->predecessors()) {
        if (p && std::find(out.begin(), out.end(), p) == out.end()) out.push_back(p);
    }
    return out;
}

} // namespace

size_t rematerialize_after(Function& fn, const std::vector<Instruction*>& vars, const std::vector<RematPoint>& points) {
    if (vars.empty() || points.empty() || !fn.entry_block()) return 0;
    fn.rebuild_cfg_predecessors();
    const DominatorTree dom(fn);

    std::unordered_map<const BasicBlock*, size_t> block_index;
    const std::vector<BasicBlock*>& blocks = fn.blocks();
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i]) block_index[blocks[i]] = i;
    }
    const size_t nb = blocks.size();
    const size_t nv = vars.size();
    std::unordered_map<const Value*, size_t> var_of;
    for (size_t i = 0; i < nv; ++i) var_of[vars[i]->result()] = i;
    auto var_index = [&](const Value* v) -> long {
        const auto it = v ? var_of.find(v) : var_of.end();
        return it == var_of.end() ? -1 : static_cast<long>(it->second);
    };
    std::unordered_map<const Instruction*, std::vector<const RematPoint*>> points_at;
    for (const RematPoint& p : points) {
        if (p.after && !p.vars.empty()) points_at[p.after].push_back(&p);
    }

    // Where each var is defined (its definition and its points) and used
    // before any definition in a block (upward exposed).
    std::vector<std::vector<char>> defs(nv, std::vector<char>(nb, 0));
    std::vector<std::vector<char>> exposed(nv, std::vector<char>(nb, 0));
    for (size_t bi = 0; bi < nb; ++bi) {
        BasicBlock* bb = blocks[bi];
        if (!bb || !dom.is_reachable(bb)) continue;
        auto use = [&](const Value* u) {
            const long vi = var_index(u);
            if (vi >= 0 && !defs[vi][bi]) exposed[vi][bi] = 1;
        };
        for (Instruction* inst : *bb) {
            for_each_use(*inst, use);
            if (const long vi = var_index(inst->result()); vi >= 0) defs[vi][bi] = 1;
            const auto it = points_at.find(inst);
            if (it == points_at.end()) continue;
            for (const RematPoint* p : it->second) {
                for (const size_t vi : p->vars) {
                    for (const Value* op : vars[vi]->operands()) use(op);
                    defs[vi][bi] = 1;
                }
            }
        }
    }

    // Dominance frontiers.
    std::vector<std::vector<size_t>> frontier(nb);
    for (size_t bi = 0; bi < nb; ++bi) {
        const BasicBlock* bb = blocks[bi];
        if (!bb || !dom.is_reachable(bb)) continue;
        const std::vector<BasicBlock*> preds = unique_preds(bb);
        if (preds.size() < 2) continue;
        const BasicBlock* idom = dom.immediate_dominator(bb);
        for (const BasicBlock* p : preds) {
            if (!dom.is_reachable(p)) continue;
            for (const BasicBlock* runner = p; runner && runner != idom; runner = dom.immediate_dominator(runner)) {
                std::vector<size_t>& f = frontier[block_index[runner]];
                if (std::find(f.begin(), f.end(), bi) == f.end()) f.push_back(bi);
                if (runner == fn.entry_block()) break;
            }
        }
    }

    // Per var: liveness, then parameters at the iterated frontier of its
    // definitions where it is live.
    std::vector<std::vector<std::pair<size_t, Value*>>> phis(nb);  // (var, param) per block
    for (size_t vi = 0; vi < nv; ++vi) {
        std::vector<char> live_in(nb, 0);
        std::vector<size_t> work;
        for (size_t bi = 0; bi < nb; ++bi) {
            if (exposed[vi][bi]) {
                live_in[bi] = 1;
                work.push_back(bi);
            }
        }
        while (!work.empty()) {
            const size_t bi = work.back();
            work.pop_back();
            for (const BasicBlock* p : blocks[bi]->predecessors()) {
                if (!p || !dom.is_reachable(p)) continue;
                const size_t pi = block_index[p];
                if (!defs[vi][pi] && !live_in[pi]) {
                    live_in[pi] = 1;
                    work.push_back(pi);
                }
            }
        }
        std::vector<char> has_phi(nb, 0);
        std::vector<char> queued(nb, 0);
        for (size_t bi = 0; bi < nb; ++bi) {
            if (defs[vi][bi]) {
                queued[bi] = 1;
                work.push_back(bi);
            }
        }
        while (!work.empty()) {
            const size_t bi = work.back();
            work.pop_back();
            for (const size_t yi : frontier[bi]) {
                if (has_phi[yi] || !live_in[yi]) continue;
                has_phi[yi] = 1;
                BasicBlock* y = blocks[yi];
                Value* param = ir::new_block_param(fn, y, vars[vi]->result()->type());
                phis[yi].push_back({vi, param});
                // A placeholder the renaming below replaces by the version
                // that reaches the end of each predecessor.
                for (BasicBlock* p : unique_preds(y)) {
                    if (Instruction* term = p->terminator()) {
                        edges_into(*term, y, [&](BranchTarget& t) { t.args.push_back(vars[vi]->result()); });
                    }
                }
                if (!queued[yi]) {
                    queued[yi] = 1;
                    work.push_back(yi);
                }
            }
        }
    }

    // Renaming over the dominator tree.
    GcSensitive sensitive;
    std::vector<std::vector<Value*>> stack(nv);
    std::vector<std::vector<size_t>> pushed;  // vars pushed per open block
    size_t copies = 0;
    auto top = [&](Value* v) -> Value* {
        const long vi = var_index(v);
        if (vi < 0 || stack[vi].empty()) return v;
        return stack[vi].back();
    };
    walk_dominator_tree(
        dom, fn.entry_block(),
        [&](const BasicBlock* cbb) {
            BasicBlock* bb = const_cast<BasicBlock*>(cbb);
            const size_t bi = block_index[bb];
            pushed.emplace_back();
            std::vector<size_t>& mine = pushed.back();
            for (const auto& [vi, param] : phis[bi]) {
                stack[vi].push_back(param);
                mine.push_back(vi);
            }
            for (Instruction* inst = bb->head(); inst;) {
                for_each_use_slot(*inst, [&](Value*& slot) { slot = top(slot); });
                if (const long vi = var_index(inst->result()); vi >= 0) {
                    stack[vi].push_back(inst->result());
                    mine.push_back(static_cast<size_t>(vi));
                }
                Instruction* last = inst;
                const auto it = points_at.find(inst);
                if (it != points_at.end()) {
                    const bool collects = may_trigger_gc(*inst);
                    ir::ValueMap fresh;  // what this point copies from the tagged values up
                    std::unordered_set<size_t> done;  // the vars this point has computed
                    std::function<Value*(Value*)> operand = [&](Value* op) -> Value* {
                        const long ov = var_index(op);
                        if (ov >= 0 && done.count(static_cast<size_t>(ov))) return top(op);
                        if (!collects || !op || !op->is_instruction() || !sensitive.clonable(op)) return top(op);
                        if (const auto f = fresh.find(op); f != fresh.end()) return f->second;
                        const Instruction* d = op->defining_instruction();
                        std::vector<Value*> ops;
                        for (Value* o : d->operands()) ops.push_back(operand(o));
                        ir::ValueMap map;
                        Instruction* copy = ir::clone_shell(fn, *d, map);
                        for (Value* o : ops) copy->add_operand(o);
                        bb->insert_after(copy, last);
                        last = copy;
                        ++copies;
                        fresh[op] = copy->result();
                        return copy->result();
                    };
                    for (const RematPoint* p : it->second) {
                        for (const size_t vi : p->vars) {
                            const Instruction* e = vars[vi];
                            std::vector<Value*> ops;
                            for (Value* o : e->operands()) ops.push_back(operand(o));
                            ir::ValueMap map;
                            Instruction* copy = ir::clone_shell(fn, *e, map);
                            for (Value* o : ops) copy->add_operand(o);
                            bb->insert_after(copy, last);
                            last = copy;
                            ++copies;
                            stack[vi].push_back(copy->result());
                            mine.push_back(vi);
                            done.insert(vi);
                        }
                    }
                }
                inst = last->next();
            }
        },
        [&](const BasicBlock*) {
            for (const size_t vi : pushed.back()) stack[vi].pop_back();
            pushed.pop_back();
        });
    return copies;
}

} // namespace brass
