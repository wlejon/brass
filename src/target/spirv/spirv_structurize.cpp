// Structurizer: MIR CFG -> SPIR-V structured control flow (see
// spirv_structurize.hpp and docs/spirv_backend_design.md).
//
// 1. Build one node per reachable MIR block plus the prologue; a br_if with
//    both edges to one block gets a forward node on its false edge.
// 2. Find back edges (DFS retreating edges); a retreating edge whose target
//    does not dominate its source makes the CFG irreducible -> diagnostic.
// 3. Every loop head h gets a header node H (a forward node in front of h,
//    taking all of h's incoming edges) and a continue node C (taking every
//    back edge, branching to H). The MIR head becomes an ordinary block
//    inside the loop, so its br_if is a selection like any other.
// 4. Headers are visited in reverse postorder (outer before inner), each
//    choosing its merge:
//      loop       the unique target of its exit edges (none: an unreachable
//                 merge; several: diagnostic);
//      selection  edges to the innermost loop's continue/merge are
//                 continue/break and need no merge; with two ordinary arms
//                 the merge is the first block in RPO reachable from both
//                 (no such block: the arm reaching more blocks), with one it
//                 is that arm, with none an unreachable merge.
//    A merge that another header already claimed, or that the header does
//    not dominate, is replaced by a fresh forward node taking the edges into
//    it from the header's region -- that is how an inner if whose arms jump
//    straight to an outer join gets its own merge.

#include "spirv_structurize.hpp"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <unordered_map>

namespace brass::spirv {

std::string block_name(const BasicBlock* bb) {
    if (!bb) return "<null>";
    std::string s = "bb" + std::to_string(bb->id());
    if (!bb->name().empty()) s += " ('" + std::string(bb->name()) + "')";
    return s;
}

namespace {

class Structurizer {
public:
    explicit Structurizer(const brass::Function& fn) : fn_(fn) {}

    StructuredCfg run() {
        build();
        compute_order();
        make_loops();
        compute_order();
        place_merges();
        g_.order = rpo_;
        for (size_t n = 0; n < g_.nodes.size(); ++n) {
            if (g_.nodes[n].kind == CfgNode::Kind::unreachable) g_.order.push_back(n);
        }
        return std::move(g_);
    }

private:
    const brass::Function& fn_;
    StructuredCfg g_;
    std::vector<size_t> rpo_;
    std::vector<size_t> rpo_index_; // kNoNode when unreachable
    std::vector<size_t> idom_;
    std::vector<size_t> loops_;     // loop header nodes, outer before inner
    std::unordered_map<size_t, std::vector<size_t>> absorbed_; // loop header -> returning exit regions taken into the loop
    std::vector<bool> claimed_;

    [[noreturn]] void fail(const std::string& what) const {
        throw std::runtime_error("SpirvISel: kernel '" + std::string(fn_.name()) + "': " + what);
    }

    std::string node_name(size_t n) const { return g_.nodes[n].name; }

    // ---- graph editing ----------------------------------------------------

    size_t add_node(CfgNode::Kind kind, const BasicBlock* block, const BasicBlock* params, std::string name) {
        CfgNode node;
        node.kind = kind;
        node.block = block;
        node.params = params;
        node.name = std::move(name);
        g_.nodes.push_back(std::move(node));
        claimed_.push_back(false);
        return g_.nodes.size() - 1;
    }

    size_t add_edge(size_t from, size_t to, const BranchTarget* target) {
        g_.edges.push_back(CfgEdge{from, to, target});
        size_t e = g_.edges.size() - 1;
        g_.nodes[from].out.push_back(e);
        g_.nodes[to].in.push_back(e);
        return e;
    }

    void redirect(size_t e, size_t to) {
        auto& old_in = g_.nodes[g_.edges[e].to].in;
        old_in.erase(std::remove(old_in.begin(), old_in.end(), e), old_in.end());
        g_.edges[e].to = to;
        g_.nodes[to].in.push_back(e);
    }

    // A forward node in front of `target`: it carries target's phis and
    // branches to it. Edges are redirected into it by the caller.
    size_t make_forward(size_t target, CfgNode::Kind kind, const std::string& prefix) {
        size_t f = add_node(kind, nullptr, g_.nodes[target].params, prefix + "_" + g_.nodes[target].name);
        add_edge(f, target, nullptr);
        return f;
    }

    // ---- 1. build -----------------------------------------------------------

    bool branches_to(const BasicBlock* target) const {
        for (const BasicBlock* bb : fn_.blocks()) {
            const brass::Instruction* t = bb->terminator();
            if (!t) continue;
            if (t->branch_target().block == target || t->true_target().block == target ||
                t->false_target().block == target) {
                return true;
            }
        }
        return false;
    }

    void build() {
        const BasicBlock* entry = fn_.entry_block();
        if (!entry) fail("function has no entry block");
        add_node(CfgNode::Kind::prologue, nullptr, nullptr, "prologue");

        std::unordered_map<const BasicBlock*, size_t> node_of;
        std::vector<const BasicBlock*> work = {entry};
        std::vector<const BasicBlock*> reachable;
        while (!work.empty()) {
            const BasicBlock* bb = work.back();
            work.pop_back();
            if (node_of.count(bb)) continue;
            // The entry carries phis only when something branches back to
            // it; otherwise its parameters are the prologue's loads.
            const BasicBlock* params = bb;
            if (bb == entry && !branches_to(entry)) params = nullptr;
            std::string name = "bb" + std::to_string(bb->id());
            if (!bb->name().empty()) name += "_" + std::string(bb->name());
            node_of[bb] = add_node(CfgNode::Kind::block, bb, params, name);
            reachable.push_back(bb);
            const brass::Instruction* term = bb->terminator();
            if (!term) fail("block " + block_name(bb) + " has no terminator");
            if (term->opcode() == brass::Opcode::br) work.push_back(term->branch_target().block);
            if (term->opcode() == brass::Opcode::br_if) {
                work.push_back(term->true_target().block);
                work.push_back(term->false_target().block);
            }
        }

        add_edge(0, node_of.at(entry), nullptr);
        for (const BasicBlock* bb : reachable) {
            size_t n = node_of.at(bb);
            const brass::Instruction* term = bb->terminator();
            switch (term->opcode()) {
                case brass::Opcode::br:
                    add_edge(n, node_of.at(term->branch_target().block), &term->branch_target());
                    break;
                case brass::Opcode::br_if: {
                    size_t t = node_of.at(term->true_target().block);
                    size_t f = node_of.at(term->false_target().block);
                    add_edge(n, t, &term->true_target());
                    if (t == f) f = make_forward(f, CfgNode::Kind::forward, "split");
                    add_edge(n, f, &term->false_target());
                    break;
                }
                case brass::Opcode::ret:
                case brass::Opcode::unreachable:
                    break;
                default:
                    fail("block " + block_name(bb) + " ends in '" + std::string(brass::opcode_name(term->opcode())) +
                         "', which the SPIR-V target does not lower (only br, br_if, ret, unreachable)");
            }
        }
    }

    // ---- orders and dominators ------------------------------------------------

    void compute_order() {
        size_t n = g_.nodes.size();
        std::vector<int> state(n, 0);
        std::vector<size_t> post;
        std::vector<std::pair<size_t, size_t>> stack = {{0, 0}};
        state[0] = 1;
        while (!stack.empty()) {
            auto& [node, next] = stack.back();
            if (next < g_.nodes[node].out.size()) {
                size_t to = g_.edges[g_.nodes[node].out[next++]].to;
                if (state[to] == 0) {
                    state[to] = 1;
                    stack.emplace_back(to, 0);
                }
            } else {
                post.push_back(node);
                stack.pop_back();
            }
        }
        rpo_.assign(post.rbegin(), post.rend());
        rpo_index_.assign(n, kNoNode);
        for (size_t i = 0; i < rpo_.size(); ++i) rpo_index_[rpo_[i]] = i;

        // Cooper, Harvey, Kennedy: "A Simple, Fast Dominance Algorithm".
        idom_.assign(n, kNoNode);
        idom_[0] = 0;
        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t i = 1; i < rpo_.size(); ++i) {
                size_t b = rpo_[i];
                size_t new_idom = kNoNode;
                for (size_t e : g_.nodes[b].in) {
                    size_t p = g_.edges[e].from;
                    if (idom_[p] == kNoNode) continue;
                    new_idom = new_idom == kNoNode ? p : intersect(p, new_idom);
                }
                if (new_idom != idom_[b]) {
                    idom_[b] = new_idom;
                    changed = true;
                }
            }
        }
    }

    size_t intersect(size_t a, size_t b) const {
        while (a != b) {
            while (rpo_index_[a] > rpo_index_[b]) a = idom_[a];
            while (rpo_index_[b] > rpo_index_[a]) b = idom_[b];
        }
        return a;
    }

    bool dominates(size_t a, size_t b) const {
        if (rpo_index_[b] == kNoNode || rpo_index_[a] == kNoNode) return false;
        while (true) {
            if (a == b) return true;
            if (b == 0) return false;
            b = idom_[b];
        }
    }

    // ---- 2./3. loops ------------------------------------------------------------

    void make_loops() {
        // Retreating edges of a DFS from the prologue.
        size_t n = g_.nodes.size();
        std::vector<int> state(n, 0);
        std::vector<size_t> back_edges;
        std::vector<std::pair<size_t, size_t>> stack = {{0, 0}};
        state[0] = 1;
        while (!stack.empty()) {
            auto& [node, next] = stack.back();
            if (next < g_.nodes[node].out.size()) {
                size_t e = g_.nodes[node].out[next++];
                size_t to = g_.edges[e].to;
                if (state[to] == 1) back_edges.push_back(e);
                else if (state[to] == 0) {
                    state[to] = 1;
                    stack.emplace_back(to, 0);
                }
            } else {
                state[node] = 2;
                stack.pop_back();
            }
        }

        std::vector<size_t> heads;
        std::unordered_map<size_t, std::vector<size_t>> latches; // head -> back edges
        for (size_t e : back_edges) {
            size_t from = g_.edges[e].from;
            size_t to = g_.edges[e].to;
            if (!dominates(to, from)) {
                fail("irreducible control flow: the edge " + node_name(from) + " -> " + node_name(to) +
                     " enters a cycle at a block that does not dominate it; SPIR-V needs structured, reducible loops");
            }
            if (!latches.count(to)) heads.push_back(to);
            latches[to].push_back(e);
        }
        std::sort(heads.begin(), heads.end(), [&](size_t a, size_t b) { return rpo_index_[a] < rpo_index_[b]; });

        for (size_t h : heads) {
            std::vector<size_t> incoming = g_.nodes[h].in;
            size_t header = make_forward(h, CfgNode::Kind::forward, "loop");
            for (size_t e : incoming) redirect(e, header);
            size_t cont = add_node(CfgNode::Kind::loop_continue, nullptr, g_.nodes[h].params,
                                   "continue_" + g_.nodes[h].name);
            add_edge(cont, header, nullptr);
            for (size_t e : latches[h]) redirect(e, cont);
            g_.nodes[header].cont = cont;
            loops_.push_back(header);
        }
    }

    // The loop's blocks: its header plus everything that reaches the
    // continue node backward without passing the header.
    std::vector<bool> loop_body(size_t header) const {
        std::vector<bool> in(g_.nodes.size(), false);
        in[header] = true;
        std::vector<size_t> work = {g_.nodes[header].cont};
        while (!work.empty()) {
            size_t n = work.back();
            work.pop_back();
            if (in[n]) continue;
            in[n] = true;
            for (size_t e : g_.nodes[n].in) work.push_back(g_.edges[e].from);
        }
        auto it = absorbed_.find(header);
        if (it != absorbed_.end()) {
            for (size_t n : it->second) in[n] = true;
        }
        return in;
    }

    // Nodes reachable from `start` without taking a back edge.
    std::vector<bool> forward(size_t start) const {
        std::vector<bool> seen(g_.nodes.size(), false);
        std::vector<size_t> work = {start};
        while (!work.empty()) {
            size_t n = work.back();
            work.pop_back();
            if (seen[n]) continue;
            seen[n] = true;
            if (g_.nodes[n].kind == CfgNode::Kind::loop_continue) continue;
            for (size_t e : g_.nodes[n].out) work.push_back(g_.edges[e].to);
        }
        return seen;
    }

    // A loop that returns from inside (`if (x) return;` in the body) has the
    // returning block among its exit targets. Such a target, whose region
    // the header dominates and that shares no block with another target's
    // region, is part of the loop construct (it leaves by returning), so it
    // is absorbed into the loop and only the remaining target becomes the
    // merge. With every target returning, the one with the larger region
    // stays the merge (on a tie, the one the earliest block exits to).
    void absorb_returning_exits(size_t h, std::vector<size_t>& targets,
                                const std::vector<std::pair<size_t, size_t>>& exit_edges) {
        std::vector<std::vector<bool>> regions;
        for (size_t t : targets) regions.push_back(forward(t));
        std::vector<size_t> live;
        std::vector<size_t> dead;
        for (size_t i = 0; i < targets.size(); ++i) {
            bool returning = true;
            for (size_t n = 0; n < g_.nodes.size() && returning; ++n) {
                if (!regions[i][n]) continue;
                if (!dominates(h, n)) returning = false;
                for (size_t j = 0; j < targets.size() && returning; ++j) {
                    if (j != i && regions[j][n]) returning = false;
                }
            }
            (returning ? dead : live).push_back(i);
        }
        if (live.size() > 1) return; // a real multi-exit loop: the caller reports it
        size_t keep = live.empty() ? dead.front() : live.front();
        if (live.empty()) {
            auto size = [&](size_t i) { return std::count(regions[i].begin(), regions[i].end(), true); };
            auto first_source = [&](size_t i) {
                size_t best = kNoNode;
                for (const auto& [from, to] : exit_edges) {
                    if (to == targets[i]) best = std::min(best, rpo_index_[from]);
                }
                return best;
            };
            for (size_t i : dead) {
                if (size(i) > size(keep) || (size(i) == size(keep) && first_source(i) < first_source(keep))) keep = i;
            }
        }
        for (size_t i : dead) {
            if (i == keep) continue;
            for (size_t n = 0; n < g_.nodes.size(); ++n) {
                if (regions[i][n]) absorbed_[h].push_back(n);
            }
        }
        targets = {targets[keep]};
    }

    // ---- 4. merges ----------------------------------------------------------------

    void place_merges() {
        std::vector<size_t> headers;
        for (size_t n : rpo_) {
            bool loop = std::find(loops_.begin(), loops_.end(), n) != loops_.end();
            if (loop || g_.nodes[n].out.size() == 2) headers.push_back(n);
        }
        for (size_t h : headers) {
            if (g_.nodes[h].cont != kNoNode) place_loop_merge(h);
            else place_selection_merge(h);
        }
    }

    size_t unreachable_merge(size_t header) {
        size_t u = add_node(CfgNode::Kind::unreachable, nullptr, nullptr, "unreachable_" + node_name(header));
        claimed_[u] = true;
        compute_order(); // keeps the per-node tables sized
        return u;
    }

    // Claims `m` as the merge of `header`. If that is not possible, a fresh
    // forward node takes the edges into `m` that come from the header's
    // region (`region(from)`), and becomes the merge instead.
    void claim(size_t header, size_t m, const std::function<bool(size_t)>& region) {
        if (claimed_[m] || !dominates(header, m)) {
            size_t f = make_forward(m, CfgNode::Kind::forward, "merge");
            std::vector<size_t> into = g_.nodes[m].in;
            for (size_t e : into) {
                size_t from = g_.edges[e].from;
                if (from != f && region(from)) redirect(e, f);
            }
            m = f;
            compute_order();
        }
        claimed_[m] = true;
        g_.nodes[header].merge = m;
    }

    void place_loop_merge(size_t h) {
        std::vector<bool> body = loop_body(h);
        std::vector<size_t> targets;
        std::vector<std::pair<size_t, size_t>> exit_edges;
        for (size_t n = 0; n < g_.nodes.size(); ++n) {
            if (!body[n]) continue;
            for (size_t e : g_.nodes[n].out) {
                size_t to = g_.edges[e].to;
                if (body[to]) continue;
                exit_edges.emplace_back(n, to);
                if (std::find(targets.begin(), targets.end(), to) == targets.end()) targets.push_back(to);
            }
        }
        if (targets.size() > 1) {
            absorb_returning_exits(h, targets, exit_edges);
            body = loop_body(h);
        }
        if (targets.empty()) {
            g_.nodes[h].merge = unreachable_merge(h);
            return;
        }
        if (targets.size() > 1) {
            std::string list;
            for (size_t t : targets) list += (list.empty() ? "" : ", ") + node_name(t);
            fail("the loop headed by " + node_name(g_.edges[g_.nodes[h].out[0]].to) + " exits to more than one block (" +
                 list + "); a SPIR-V loop has a single merge block, so route every break through one exit block");
        }
        claim(h, targets[0], [&](size_t from) { return body[from]; });
    }

    // The innermost loop (header node) whose body contains `n`, or kNoNode.
    size_t innermost_loop(size_t n, std::vector<bool>* body_out) const {
        size_t best = kNoNode;
        size_t best_size = 0;
        for (size_t h : loops_) {
            std::vector<bool> body = loop_body(h);
            if (!body[n]) continue;
            size_t size = static_cast<size_t>(std::count(body.begin(), body.end(), true));
            if (best == kNoNode || size < best_size) {
                best = h;
                best_size = size;
                *body_out = std::move(body);
            }
        }
        return best;
    }

    void place_selection_merge(size_t x) {
        std::vector<bool> body;
        size_t loop = innermost_loop(x, &body);
        auto terminal = [&](size_t to) {
            return loop != kNoNode && (to == g_.nodes[loop].cont || to == g_.nodes[loop].merge);
        };

        std::vector<size_t> arms;
        for (size_t e : g_.nodes[x].out) {
            size_t to = g_.edges[e].to;
            if (terminal(to)) continue;
            if (loop != kNoNode && !body[to]) {
                fail("the branch " + node_name(x) + " -> " + node_name(to) +
                     " leaves its loop other than through the loop's merge block (a multi-level break or continue)");
            }
            arms.push_back(to);
        }

        size_t m = kNoNode;
        if (arms.empty()) {
            g_.nodes[x].merge = unreachable_merge(x);
            return;
        }
        if (arms.size() == 1) {
            m = arms[0];
        } else {
            std::vector<bool> ra = reach(arms[0], loop, body);
            std::vector<bool> rb = reach(arms[1], loop, body);
            for (size_t n : rpo_) {
                if (ra[n] && rb[n]) { m = n; break; }
            }
            if (m == kNoNode) {
                auto count = [](const std::vector<bool>& r) { return std::count(r.begin(), r.end(), true); };
                m = count(ra) > count(rb) ? arms[0] : arms[1];
            }
        }
        size_t merge_target = m;
        claim(x, m, [&](size_t from) { return dominates(x, from) && !dominates(merge_target, from); });
    }

    // Nodes reachable from `start` without taking a back edge, and, inside a
    // loop, without leaving it through its continue or merge node.
    std::vector<bool> reach(size_t start, size_t loop, const std::vector<bool>& body) const {
        std::vector<bool> seen(g_.nodes.size(), false);
        std::vector<size_t> work = {start};
        while (!work.empty()) {
            size_t n = work.back();
            work.pop_back();
            if (seen[n]) continue;
            seen[n] = true;
            if (g_.nodes[n].kind == CfgNode::Kind::loop_continue) continue;
            for (size_t e : g_.nodes[n].out) {
                size_t to = g_.edges[e].to;
                if (loop != kNoNode && (to == g_.nodes[loop].cont || to == g_.nodes[loop].merge || !body[to])) continue;
                work.push_back(to);
            }
        }
        return seen;
    }
};

} // namespace

StructuredCfg structurize(const brass::Function& fn) {
    return Structurizer(fn).run();
}

} // namespace brass::spirv
