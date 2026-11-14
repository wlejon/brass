// Interrupt polls on the back edges of compiled code
// (include/brass/mir/interrupt_poll.hpp).
#include <brass/mir/interrupt_poll.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/gc_refs.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/uses.hpp>
#include "gc_remat.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace brass {

namespace {

using Edge = std::pair<BasicBlock*, BasicBlock*>; // (source, loop header)

// The back edges of a depth-first walk: an edge to a block still on the
// walk's stack. Every cycle of the CFG holds one, whatever the walk's roots
// (the entry, then every block a resume point or nothing else reaches).
// An edge into a landing pad is left out: nothing may run before the pad.
std::vector<Edge> back_edges(const Function& fn) {
    std::vector<Edge> edges;
    const auto& blocks = fn.blocks();
    std::unordered_map<const BasicBlock*, uint8_t> state; // 1 on the stack, 2 done
    state.reserve(blocks.size());
    struct Frame {
        BasicBlock* bb;
        SuccessorList succs;
        size_t next;
    };
    std::vector<Frame> stack;
    auto walk = [&](BasicBlock* root) {
        if (!root || state.count(root)) return;
        state[root] = 1;
        stack.push_back({root, root->successors(), 0});
        while (!stack.empty()) {
            Frame& f = stack.back();
            if (f.next == f.succs.size()) {
                state[f.bb] = 2;
                stack.pop_back();
                continue;
            }
            BasicBlock* succ = f.succs[f.next++];
            if (!succ) continue;
            auto it = state.find(succ);
            if (it == state.end()) {
                state[succ] = 1;
                stack.push_back({succ, succ->successors(), 0});
            } else if (it->second == 1) {
                if (!(succ->head() && succ->head()->opcode() == Opcode::landing_pad)) {
                    edges.emplace_back(f.bb, succ);
                }
            }
        }
    };
    walk(const_cast<Function&>(fn).entry_block());
    for (BasicBlock* bb : blocks) walk(bb);
    // A source with two edges to one header (both arms of a br_if) is one
    // pair: its terminator is rewritten once.
    std::vector<Edge> unique;
    unique.reserve(edges.size());
    for (const Edge& e : edges) {
        bool seen = false;
        for (const Edge& u : unique) seen = seen || (u == e);
        if (!seen) unique.push_back(e);
    }
    return unique;
}

std::vector<Value*> params_of(BasicBlock* bb) {
    return std::vector<Value*>(bb->params().begin(), bb->params().end());
}

// sp < [tls + offset], built at the builder's position.
Value* build_poll_condition(Builder& b, int32_t offset) {
    Value* tls = b.build_pinned_tls_read();
    Value* limit = b.build_load(Type::i64(), tls, offset);
    Value* sp = b.build_read_sp();
    return b.build_ult(sp, limit);
}

// A loop the parallel transform outlined (loop_parallel_transform.cpp) runs
// its chunk of a counted range on a parallel runtime worker, entered with no
// pinned TLS register loaded, and has nowhere to yield to: it is not polled.
// The function that dispatches it polls its own back edges.
bool is_parallel_kernel(const Function& fn) {
    return fn.name().find("_par_k_") != std::string_view::npos;
}

bool polled(const Function& fn) {
    const Module* mod = fn.parent();
    // A `nocollect` function reaches no GC point by its contract, and a poll
    // is one: its loops go unpolled.
    return mod && mod->interrupt_poll() && mod->pinned_tls_register() && !is_parallel_kernel(fn) &&
           !mod->has_symbol_role(fn.name(), SymbolRole::NoCollect);
}


// A handler may collect, and a collection moves objects: what the loop
// holds across the handler's call that is made of a tagged value's bits
// (gc_remat.hpp GcSensitive), a value defined above the loop's header and
// used at or below it (LICM's hoisted invariants), the handler block
// computes again after it, from the tagged values up, when every other
// operand of the instructions that make it reaches the call;
// rematerialize_after carries the new versions to their uses. A value it
// cannot make is left as it is: no pass makes one (LICM hoists only
// clonable values, IVSR steps no pointer made of one).
void recompute_after_handlers(Function& fn, const std::vector<std::pair<BasicBlock*, BasicBlock*>>& handlers) {
    fn.rebuild_cfg_predecessors();
    const DominatorTree dom(fn);
    GcSensitive sensitive;
    std::vector<Instruction*> vars;
    std::unordered_map<const Value*, size_t> var_of;
    std::vector<RematPoint> points;
    for (const auto& [header, cold] : handlers) {
        Instruction* call = cold->head();
        if (!call || !call->is_call()) continue;
        const BasicBlock* at = call->parent();
        // Whether `e` computed again from the tagged values up has every
        // other operand defined where the call is.
        auto leaves_reach = [&](const Instruction* e) {
            std::vector<const Instruction*> work{e};
            std::unordered_set<const Instruction*> seen;
            while (!work.empty()) {
                const Instruction* i = work.back();
                work.pop_back();
                if (!seen.insert(i).second) continue;
                for (const Value* op : i->operands()) {
                    if (!op) continue;
                    if (op->is_instruction() && sensitive.clonable(op)) {
                        work.push_back(op->defining_instruction());
                        continue;
                    }
                    const BasicBlock* def = op->is_instruction() ? op->defining_instruction()->parent()
                                                                 : op->is_block_param() ? op->defining_block() : nullptr;
                    if (def && (def == at || !dom.dominates(def, at))) return false;
                }
            }
            return true;
        };
        RematPoint point{call, {}};
        auto add = [&](Value* v) {
            auto it = var_of.find(v);
            if (it == var_of.end()) {
                if (!v->is_instruction() || !sensitive.clonable(v)) return;
                Instruction* d = v->defining_instruction();
                if (!leaves_reach(d)) return;
                it = var_of.emplace(v, vars.size()).first;
                vars.push_back(d);
            }
            if (std::find(point.vars.begin(), point.vars.end(), it->second) == point.vars.end()) {
                point.vars.push_back(it->second);
            }
        };
        // What is defined above the header and used at or below it.
        for (BasicBlock* bb : fn.blocks()) {
            if (!bb || !dom.is_reachable(bb) || !dom.dominates(header, bb)) continue;
            for (Instruction* inst : *bb) {
                for_each_use(*inst, [&](Value* u) {
                    if (u->type().is_tagged() || !sensitive(u)) return;
                    const BasicBlock* def = u->is_instruction() ? u->defining_instruction()->parent()
                                                                : u->is_block_param() ? u->defining_block() : nullptr;
                    if (!def || dom.dominates(header, def)) return;
                    add(u);
                });
            }
        }
        if (!point.vars.empty()) points.push_back(std::move(point));
    }
    rematerialize_after(fn, vars, points);
}

} // namespace

bool needs_interrupt_polls(const Function& fn) {
    return polled(fn) && !back_edges(fn).empty();
}

size_t insert_interrupt_polls(Function& fn) {
    Module* mod = fn.parent();
    if (!polled(fn)) return 0;
    const Module::InterruptPoll poll = mod->interrupt_poll();
    const std::vector<Edge> edges = back_edges(fn);
    if (edges.empty()) return 0;

    Builder b(fn);
    // One cold block per header: call the handler, then run the header.
    std::unordered_map<BasicBlock*, BasicBlock*> handler_block;
    // One polling block per header, for back edges a conditional branch,
    // switch or invoke takes: the poll, then the header or the handler.
    std::unordered_map<BasicBlock*, BasicBlock*> poll_block;

    auto cold_for = [&](BasicBlock* header) {
        auto it = handler_block.find(header);
        if (it != handler_block.end()) return it->second;
        BasicBlock* cold = b.create_block();
        cold->set_cold();
        for (Value* p : header->params()) b.add_block_param(cold, p->type());
        fn.append_block(cold);
        b.position_at_end(cold);
        b.clear_current_loc();
        if (Instruction* first = header->head()) b.set_current_loc(first->loc());
        b.build_call(poll.handler, Type::void_type());
        const std::vector<Value*> args = params_of(cold);
        b.build_br(header, Span<Value* const>(args.data(), args.size()));
        handler_block[header] = cold;
        return cold;
    };
    auto poll_for = [&](BasicBlock* header) {
        auto it = poll_block.find(header);
        if (it != poll_block.end()) return it->second;
        BasicBlock* cold = cold_for(header);
        BasicBlock* pb = b.create_block();
        for (Value* p : header->params()) b.add_block_param(pb, p->type());
        fn.append_block(pb);
        b.position_at_end(pb);
        b.clear_current_loc();
        if (Instruction* first = header->head()) b.set_current_loc(first->loc());
        Value* cond = build_poll_condition(b, poll.tls_offset);
        const std::vector<Value*> args = params_of(pb);
        const Span<Value* const> span(args.data(), args.size());
        b.build_br_if(cond, cold, span, header, span);
        poll_block[header] = pb;
        return pb;
    };

    size_t polled = 0;
    for (const auto& [src, header] : edges) {
        Instruction* term = src->terminator();
        if (!term) continue;
        if (term->opcode() == Opcode::br && term->branch_target().block == header) {
            // The common latch: the poll goes in before the jump, which
            // becomes a branch to the header or the handler.
            BasicBlock* cold = cold_for(header);
            std::vector<Value*> args = term->branch_target().args;
            b.position_before(term);
            b.set_current_loc(term->loc());
            Value* cond = build_poll_condition(b, poll.tls_offset);
            const Span<Value* const> span(args.data(), args.size());
            b.build_br_if(cond, cold, span, header, span);
            src->remove_instruction(term);
            ++polled;
            continue;
        }
        BasicBlock* pb = poll_for(header);
        bool redirected = false;
        auto redirect = [&](BranchTarget& t) {
            if (t.block == header) {
                t.block = pb;
                redirected = true;
            }
        };
        switch (term->opcode()) {
            case Opcode::br:
                redirect(term->branch_target());
                break;
            case Opcode::br_if:
                redirect(term->true_target());
                redirect(term->false_target());
                break;
            case Opcode::switch_:
                redirect(term->default_target());
                for (auto& sc : term->switch_cases()) redirect(sc.target);
                break;
            case Opcode::invoke:
                // Its unwind edge goes to a landing pad, never polled.
                redirect(term->normal_target());
                break;
            default:
                break;
        }
        if (redirected) ++polled;
    }
    std::vector<std::pair<BasicBlock*, BasicBlock*>> handlers(handler_block.begin(), handler_block.end());
    std::sort(handlers.begin(), handlers.end(),
              [](const auto& a, const auto& b) { return a.first->id() < b.first->id(); });
    recompute_after_handlers(fn, handlers);
    fn.rebuild_cfg_predecessors();
    return polled;
}

} // namespace brass
