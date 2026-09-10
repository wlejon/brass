#include <brass/mir/critical_edge.hpp>
#include <brass/mir/builder.hpp>
#include <algorithm>
#include <string>
#include <vector>

namespace brass {

bool is_critical_edge(const BasicBlock* src, const BasicBlock* dst) {
    if (!src || !dst) return false;

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

BasicBlock* split_critical_edge(Function& fn, BasicBlock* src, BasicBlock* dst, CriticalEdgeStats* stats) {
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
    auto& blks = fn.blocks();
    auto it = std::find(blks.begin(), blks.end(), src);
    if (it != blks.end()) {
        blks.insert(it + 1, split_bb);
    } else {
        fn.append_block(split_bb);
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

    fn.rebuild_cfg_predecessors();

    if (stats) {
        stats->critical_edges_split++;
    }

    return split_bb;
}

bool split_critical_edges(Function& fn, CriticalEdgeStats* stats) {
    fn.rebuild_cfg_predecessors();
    bool any_changed = false;

    bool changed = true;
    while (changed) {
        changed = false;
        for (BasicBlock* src : fn.blocks()) {
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
                std::vector<BasicBlock*> unique_preds;
                for (BasicBlock* p : dst->predecessors()) {
                    if (p && std::find(unique_preds.begin(), unique_preds.end(), p) == unique_preds.end()) {
                        unique_preds.push_back(p);
                    }
                }

                if (unique_preds.size() > 1) {
                    split_critical_edge(fn, src, dst, stats);
                    changed = true;
                    any_changed = true;
                    break;
                }
            }

            if (changed) break;
        }
    }

    return any_changed;
}

} // namespace brass
