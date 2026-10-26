// Interrupt polls on the back edges of compiled code
// (include/brass/mir/interrupt_poll.hpp).
#include <brass/mir/interrupt_poll.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>

#include <unordered_map>
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

} // namespace

bool needs_interrupt_polls(const Function& fn) {
    const Module* mod = fn.parent();
    if (!mod || !mod->interrupt_poll() || !mod->pinned_tls_register()) return false;
    return !back_edges(fn).empty();
}

size_t insert_interrupt_polls(Function& fn) {
    Module* mod = fn.parent();
    if (!mod || !mod->interrupt_poll() || !mod->pinned_tls_register()) return 0;
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
    fn.rebuild_cfg_predecessors();
    return polled;
}

} // namespace brass
