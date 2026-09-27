#include <brass/mir/critical_edge.hpp>
#include <brass/mir/builder.hpp>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace brass {

bool is_critical_edge(const BasicBlock* src, const BasicBlock* dst) {
    if (!src || !dst) return false;

    // Do not split critical edges to exception landing pads with plain branch
    // instructions that violate the exception verifier (which requires landing_pad as first instruction).
    if (dst->head() && dst->head()->opcode() == Opcode::landing_pad) {
        return false;
    }

    // Count unique successors of src
    auto succs = src->successors();
    std::vector<const BasicBlock*> unique_succs;
    for (const BasicBlock* s : succs) {
        if (s && std::find(unique_succs.begin(), unique_succs.end(), s) == unique_succs.end()) {
            unique_succs.push_back(s);
        }
    }
    if (unique_succs.size() <= 1) return false;
    if (std::find(unique_succs.begin(), unique_succs.end(), dst) == unique_succs.end()) return false;

    // Count unique predecessors of dst
    std::vector<const BasicBlock*> unique_preds;
    for (const BasicBlock* p : dst->predecessors()) {
        if (p && std::find(unique_preds.begin(), unique_preds.end(), p) == unique_preds.end()) {
            unique_preds.push_back(p);
        }
    }
    return unique_preds.size() > 1;
}

namespace {

// Splits (src, dst), a critical edge. With `placed` null the split block is
// placed right after src and the function's predecessor lists are rebuilt;
// otherwise the block is only recorded there, for the caller to place and
// rebuild once after every split (split_critical_edges). Doing both per
// split made splitting all of a function's edges quadratic in its blocks:
// a bundle's top level of 30k blocks spent most of a minute here.
BasicBlock* split_edge(Function& fn, BasicBlock* src, BasicBlock* dst, CriticalEdgeStats* stats,
                       std::vector<std::pair<BasicBlock*, BasicBlock*>>* placed) {
    if (!src || !dst) return nullptr;
    if (!is_critical_edge(src, dst)) return nullptr;

    Instruction* term = src->terminator();
    if (!term) return nullptr;

    Builder b(fn);
    std::string src_name = src->name().empty() ? ("b" + std::to_string(src->id())) : std::string(src->name());
    std::string dst_name = dst->name().empty() ? ("b" + std::to_string(dst->id())) : std::string(dst->name());
    std::string split_name = src_name + "_" + dst_name + "_crit";

    BasicBlock* split_bb = b.create_block(split_name);
    split_bb->set_parent(&fn);

    // Insert split_bb immediately after src in the block list for fall-through locality
    if (placed) {
        placed->emplace_back(src, split_bb);
    } else {
        auto& blks = fn.blocks();
        auto it = std::find(blks.begin(), blks.end(), src);
        if (it != blks.end()) {
            blks.insert(it + 1, split_bb);
        } else {
            fn.append_block(split_bb);
        }
    }

    // Forward block parameters: split_bb accepts same parameter types as dst
    for (size_t i = 0; i < dst->param_count(); ++i) {
        Value* p = dst->param(i);
        b.add_block_param(split_bb, p ? p->type() : Type::void_type());
    }

    // Terminate split_bb by jumping to dst with its block parameters forwarded
    b.position_at_end(split_bb);
    b.build_br(dst, Span<Value* const>(split_bb->params().data(), split_bb->params().size()));

    // Redirect src's branch targets pointing to dst to point to split_bb instead
    if (term->opcode() == Opcode::br_if) {
        if (term->true_target().block == dst) {
            term->true_target().block = split_bb;
        }
        if (term->false_target().block == dst) {
            term->false_target().block = split_bb;
        }
    } else if (term->opcode() == Opcode::switch_) {
        if (term->default_target().block == dst) {
            term->default_target().block = split_bb;
        }
        for (auto& sc : term->switch_cases()) {
            if (sc.target.block == dst) {
                sc.target.block = split_bb;
            }
        }
    } else if (term->opcode() == Opcode::invoke) {
        if (term->normal_target().block == dst) {
            term->normal_target().block = split_bb;
        }
        if (term->unwind_target().block == dst) {
            term->unwind_target().block = split_bb;
        }
    }

    if (!placed) fn.rebuild_cfg_predecessors();

    if (stats) {
        stats->critical_edges_split++;
    }

    return split_bb;
}

} // namespace

BasicBlock* split_critical_edge(Function& fn, BasicBlock* src, BasicBlock* dst, CriticalEdgeStats* stats) {
    return split_edge(fn, src, dst, stats, nullptr);
}

// One pass over the blocks there were at the start. Splitting (src, dst)
// changes no other edge's criticality: src keeps as many distinct
// successors (the split block stands in for dst) and dst as many distinct
// predecessors (the split block stands in for src), and a split block has
// one successor. So the predecessor lists, stale in which block they name
// but not in how many, answer every later test, and are rebuilt once at
// the end. Each split block lands right after its src with that src's
// later splits before its earlier ones, the order splitting one edge at a
// time gave.
bool split_critical_edges(Function& fn, CriticalEdgeStats* stats) {
    fn.rebuild_cfg_predecessors();
    std::vector<std::pair<BasicBlock*, BasicBlock*>> placed;

    const std::vector<BasicBlock*> original = fn.blocks();
    for (BasicBlock* src : original) {
        if (!src) continue;

        auto succs = src->successors();
        std::vector<BasicBlock*> unique_succs;
        for (BasicBlock* s : succs) {
            if (s && std::find(unique_succs.begin(), unique_succs.end(), s) == unique_succs.end()) {
                unique_succs.push_back(s);
            }
        }
        if (unique_succs.size() <= 1) continue;

        for (BasicBlock* dst : unique_succs) {
            // An edge into a landing pad stays whole (is_critical_edge).
            split_edge(fn, src, dst, stats, &placed);
        }
    }
    if (placed.empty()) return false;

    // The blocks the pass created, each after its src (placed is in
    // creation order, grouped by src since src is visited once).
    std::vector<BasicBlock*>& blks = fn.blocks();
    std::vector<BasicBlock*> order;
    order.reserve(blks.size() + placed.size());
    size_t next = 0;
    for (BasicBlock* bb : blks) {
        order.push_back(bb);
        size_t end = next;
        while (end < placed.size() && placed[end].first == bb) ++end;
        for (size_t i = end; i > next; --i) order.push_back(placed[i - 1].second);
        next = end;
    }
    // A src no longer in the list (none should be) keeps its splits at the end.
    for (; next < placed.size(); ++next) order.push_back(placed[next].second);
    blks = std::move(order);
    fn.rebuild_cfg_predecessors();
    return true;
}

} // namespace brass
