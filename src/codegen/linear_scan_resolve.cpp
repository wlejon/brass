#include <brass/codegen/linear_scan.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>

// Where a split value's pieces meet, its value moves from one location to
// the other: inside a block, before the instruction after the split; at a
// block boundary, on every control-flow edge whose ends see it in different
// places. An edge's moves go at the end of its source when control leaves
// that block only for this edge, at the start of its target when control
// enters that block only by it, and otherwise in a new block on the edge.
//
// Edges the moves cannot be placed on keep the values live across them in
// one location (unsplittable): a landing pad is entered from inside a call,
// a resume point from the resume dispatch, and a guard exit reads its state
// in place at the Jcc that enters it.

namespace brass::codegen {

namespace {

bool is_unconditional_end(const LirInst& inst) {
    return inst.opcode == LirOpcode::Jmp || inst.opcode == LirOpcode::Ret || inst.opcode == LirOpcode::Trap ||
           inst.opcode == LirOpcode::GuardExit;
}

bool names_label(const LirInst& inst, uint32_t id) {
    for (const auto& u : inst.uses) {
        if (u.is_label() && u.label_id == id) return true;
    }
    for (const auto& d : inst.defs) {
        if (d.is_label() && d.label_id == id) return true;
    }
    return false;
}

bool same_location(const LirOperand& a, const LirOperand& b) {
    if (a.is_preg() && b.is_preg()) return a.preg_val == b.preg_val;
    if (a.is_spill_slot() && b.is_spill_slot()) return a.spill_slot == b.spill_slot;
    return false;
}

} // namespace

bool LinearScanAllocator::is_guard_exit(const LirBlock* b) const {
    return liveness_.guard_exits().count(b->id) != 0;
}

bool LinearScanAllocator::is_entered_implicitly(const LirBlock* b) const {
    return std::binary_search(landing_pads_.begin(), landing_pads_.end(), b->id) ||
           std::binary_search(resume_targets_.begin(), resume_targets_.end(), b->id);
}

LinearScanAllocator::EdgePlace LinearScanAllocator::classify_edge(const LirBlock& pred, const LirBlock& succ,
                                                                  const LirBlock* next_in_order,
                                                                  size_t succ_pred_count) const {
    const size_t n = pred.instructions.size();
    if (n == 0) return EdgePlace::None;
    // How control goes from pred to a block: by a Jcc, by the trailing Jmp,
    // or by falling through (a block not ending in a jump falls into the
    // next one in order).
    auto transfers = [&](const LirBlock& to, bool& by_jcc, bool& unplaceable) {
        by_jcc = false;
        int ways = 0;
        for (size_t i = 0; i < n; ++i) {
            const LirInst& inst = *pred.instructions[i];
            if (!names_label(inst, to.id)) continue;
            if (inst.opcode == LirOpcode::Jcc) by_jcc = true;
            else if (!(inst.opcode == LirOpcode::Jmp && i + 1 == n)) unplaceable = true;
            ++ways;
        }
        if (!is_unconditional_end(*pred.instructions.back()) && next_in_order == &to) ++ways;
        // Taken from two places in the block, the edge sees the value where
        // each of them is: a value moved between them has no one location.
        if (ways > 1) unplaceable = true;
        return ways > 0;
    };
    bool jcc = false;
    bool unplaceable = false;
    if (!transfers(succ, jcc, unplaceable) || unplaceable) return EdgePlace::None;

    size_t leaving = 0;
    for (const LirBlock* s : pred.successors) {
        if (!s || is_guard_exit(s)) continue;
        bool j = false;
        bool u = false;
        if (transfers(*s, j, u) || u) ++leaving;
    }
    if (leaving == 1 && !jcc) return EdgePlace::EndOfPred;
    if (succ_pred_count == 1 && !is_entered_implicitly(&succ)) return EdgePlace::StartOfSucc;
    return EdgePlace::Split;
}

void LinearScanAllocator::mark_unsplittable_live_ins() {
    landing_pads_.clear();
    resume_targets_.clear();
    guard_exit_at_.clear();
    const auto& exits = liveness_.guard_exits();
    std::vector<uint32_t> shared_exits;
    for (const auto& block : fn_.blocks) {
        for (const auto& inst : block->instructions) {
            if (inst->is_invoke && inst->unwind_block_id != UINT32_MAX) landing_pads_.push_back(inst->unwind_block_id);
            if (inst->opcode != LirOpcode::Jcc) continue;
            for (const auto& u : inst->uses) {
                if (!u.is_label() || !exits.count(u.label_id)) continue;
                if (!guard_exit_at_.emplace(u.label_id, inst->id).second) shared_exits.push_back(u.label_id);
            }
        }
    }
    for (const auto& rp : fn_.resume_entries) resume_targets_.push_back(rp.second);
    std::sort(landing_pads_.begin(), landing_pads_.end());
    std::sort(resume_targets_.begin(), resume_targets_.end());

    auto mark = [this](VReg v) {
        if (v.is_valid() && v.id < unsplittable_.size()) unsplittable_[v.id] = 1;
    };
    auto mark_all = [&](const std::vector<VReg>& vs) {
        for (VReg v : vs) mark(v);
    };
    // A guard exit entered by more than one Jcc reads its state where each
    // of them is: one place for all of them.
    for (uint32_t id : shared_exits) {
        guard_exit_at_.erase(id);
        const LirInst* exit = exits.at(id);
        for (const auto& u : exit->uses) {
            if (u.is_vreg()) mark(u.vreg_val);
            else if (u.is_mem()) {
                mark(u.mem_val.base_vreg);
                mark(u.mem_val.index_vreg);
            }
        }
    }

    std::unordered_map<const LirBlock*, size_t> pred_count;
    for (const auto& block : fn_.blocks) {
        for (const LirBlock* s : block->successors) {
            if (s) ++pred_count[s];
        }
    }
    const size_t nb = fn_.blocks.size();
    for (size_t i = 0; i < nb; ++i) {
        const LirBlock* block = fn_.blocks[i].get();
        const BlockLiveness& bl = liveness_.block_liveness(block);
        if (block->instructions.empty()) {
            mark_all(bl.live_in);
            mark_all(bl.live_out);
            continue;
        }
        if (is_entered_implicitly(block)) mark_all(bl.live_in);
        if (is_guard_exit(block)) continue;
        const LirBlock* next = i + 1 < nb ? fn_.blocks[i + 1].get() : nullptr;
        for (const LirBlock* s : block->successors) {
            if (!s || is_guard_exit(s) || is_entered_implicitly(s)) continue;
            if (classify_edge(*block, *s, next, pred_count[s]) == EdgePlace::None) {
                mark_all(liveness_.block_liveness(s).live_in);
            }
        }
    }
}

void LinearScanAllocator::collect_resolution_moves() {
    auto piece_location = [this](const AllocPiece& p, uint8_t size) {
        return p.spilled ? LirOperand::slot(vreg_slot_[p.vreg().id], size) : LirOperand::preg(p.reg, size);
    };
    auto move_size = [](VReg v) -> uint8_t { return v.is_gpr() ? uint8_t{8} : v.size; };
    auto is_block_start = [this](uint32_t pos) {
        auto it = std::lower_bound(spans_.begin(), spans_.end(), pos,
            [](const BlockSpan& s, uint32_t v) { return s.start < v; });
        return it != spans_.end() && it->start == pos;
    };

    // Every move, with where it goes and how deep in loops it runs, before
    // any is placed: a value stored back to its slot is looked at whole.
    enum class Place : uint8_t { Before, AtStart, AtEnd, OnEdge };
    struct Candidate {
        PendingMove move;
        uint32_t vreg = 0;
        uint32_t depth = 0;
        Place place = Place::Before;
        uint32_t key = 0;   // instruction id, block id, or edge index
    };
    std::vector<Candidate> candidates;
    std::vector<std::pair<LirBlock*, LirBlock*>> edges;
    auto depth_of = [](const LirBlock* b) { return b->loop_depth; };

    // 1. Inside blocks.
    for (size_t id = 0; id < vreg_pieces_.size(); ++id) {
        const auto& list = vreg_pieces_[id];
        if (list.size() < 2) continue;
        const VReg v = fn_.vreg_table[id].vreg;
        const uint8_t sz = move_size(v);
        for (size_t k = 1; k < list.size(); ++k) {
            const AllocPiece& left = pieces_[list[k - 1]];
            const AllocPiece& right = pieces_[list[k]];
            const uint32_t q = right.from;
            if (!left.base->covers(q) || is_block_start(q + 1)) continue;
            LirOperand src = piece_location(left, sz);
            LirOperand dst = piece_location(right, sz);
            if (same_location(src, dst)) continue;
            candidates.push_back(Candidate{PendingMove{dst, src, v.is_xmm(), sz}, static_cast<uint32_t>(id),
                                           liveness_.get_loop_depth_at(q + 1), Place::Before, q + 1});
        }
    }

    // 2. On edges.
    std::unordered_map<const LirBlock*, size_t> pred_count;
    for (const auto& block : fn_.blocks) {
        for (const LirBlock* s : block->successors) {
            if (s) ++pred_count[s];
        }
    }
    const size_t nb = fn_.blocks.size();
    for (size_t i = 0; i < nb; ++i) {
        LirBlock* pred = fn_.blocks[i].get();
        if (pred->instructions.empty() || is_guard_exit(pred)) continue;
        const LirBlock* next = i + 1 < nb ? fn_.blocks[i + 1].get() : nullptr;
        for (LirBlock* succ : pred->successors) {
            if (!succ || succ->instructions.empty() || is_guard_exit(succ) || is_entered_implicitly(succ)) continue;
            const uint32_t succ_start = succ->instructions.front()->id;
            // Where control leaves for succ: its Jcc, or the block's end.
            uint32_t pred_end = pred->instructions.back()->id;
            for (const auto& inst : pred->instructions) {
                if (inst->opcode == LirOpcode::Jcc && names_label(*inst, succ->id)) {
                    pred_end = inst->id;
                    break;
                }
            }
            const size_t first = candidates.size();
            for (VReg v : liveness_.block_liveness(succ).live_in) {
                if (!v.is_valid() || v.id >= vreg_pieces_.size() || vreg_pieces_[v.id].size() < 2) continue;
                const VReg full = fn_.vreg_table[v.id].vreg;
                const uint8_t sz = move_size(full);
                LirOperand src = location_at(v, pred_end, sz);
                LirOperand dst = location_at(v, succ_start, sz);
                if (src.is_none() || dst.is_none() || same_location(src, dst)) continue;
                candidates.push_back(Candidate{PendingMove{dst, src, full.is_xmm(), sz}, v.id,
                                               std::min(depth_of(pred), depth_of(succ)), Place::OnEdge, 0});
            }
            if (candidates.size() == first) continue;
            Place place = Place::OnEdge;
            uint32_t key = static_cast<uint32_t>(edges.size());
            switch (classify_edge(*pred, *succ, next, pred_count[succ])) {
                case EdgePlace::EndOfPred:
                    place = Place::AtEnd;
                    key = pred->id;
                    break;
                case EdgePlace::StartOfSucc:
                    place = Place::AtStart;
                    key = succ->id;
                    break;
                case EdgePlace::Split:
                    edges.emplace_back(pred, succ);
                    break;
                case EdgePlace::None:
                    throw std::logic_error("register allocation: a split value changes location on an edge "
                                           "no move can be placed on, in " + fn_.name);
            }
            for (size_t k = first; k < candidates.size(); ++k) {
                candidates[k].place = place;
                candidates[k].key = key;
            }
        }
    }

    // 3. A value defined once is stored to its slot once, right after its
    // definition, rather than each time it leaves a register (a value
    // reloaded for a loop and evicted after it would be stored back
    // unchanged): when the definition runs no deeper in loops than any of
    // the stores it replaces. A definition that writes the slot itself
    // (its piece spilled) needs no store at all.
    std::vector<uint32_t> store_depth(vreg_pieces_.size(), UINT32_MAX);
    for (const auto& c : candidates) {
        if (c.move.src.is_preg() && c.move.dst.is_spill_slot()) {
            store_depth[c.vreg] = std::min(store_depth[c.vreg], c.depth);
        }
    }
    std::vector<uint8_t> drop_stores(vreg_pieces_.size(), 0);
    for (size_t id = 0; id < vreg_pieces_.size(); ++id) {
        const uint32_t def = def_pos_[id];
        if (store_depth[id] == UINT32_MAX || def == UINT32_MAX) continue;
        const VReg v = fn_.vreg_table[id].vreg;
        const uint8_t sz = move_size(v);
        LirOperand at_def = location_at(v, def, sz);
        if (at_def.is_spill_slot()) {
            drop_stores[id] = 1;
        } else if (at_def.is_preg() && liveness_.get_loop_depth_at(def) <= store_depth[id]) {
            drop_stores[id] = 1;
            moves_after_[def].push_back(PendingMove{LirOperand::slot(vreg_slot_[id], sz), at_def, v.is_xmm(), sz});
        }
    }

    for (auto& c : candidates) {
        if (drop_stores[c.vreg] && c.move.src.is_preg() && c.move.dst.is_spill_slot()) continue;
        switch (c.place) {
            case Place::Before: moves_before_[c.key].push_back(c.move); break;
            case Place::AtStart: moves_at_start_[c.key].push_back(c.move); break;
            case Place::AtEnd: moves_at_end_[c.key].push_back(c.move); break;
            case Place::OnEdge: {
                if (edge_splits_.size() <= c.key) edge_splits_.resize(c.key + 1);
                EdgeSplit& e = edge_splits_[c.key];
                e.pred = edges[c.key].first;
                e.succ = edges[c.key].second;
                e.moves.push_back(c.move);
                break;
            }
        }
    }
    // An edge whose moves were all dropped stays as it is.
    edge_splits_.erase(std::remove_if(edge_splits_.begin(), edge_splits_.end(),
                                      [](const EdgeSplit& e) { return e.moves.empty(); }),
                       edge_splits_.end());
}

// One parallel move as a sequence: a move whose destination no other move
// still reads goes first; what is left then is cycles of registers, which a
// ParallelCopy resolves through its scratch register.
void LinearScanAllocator::emit_moves(std::vector<PendingMove>& moves, std::vector<std::unique_ptr<LirInst>>& out) {
    auto make = [](const PendingMove& m) {
        const bool reg_to_reg = m.dst.is_preg() && m.src.is_preg();
        LirOpcode op = LirOpcode::Mov;
        if (m.is_xmm) {
            if (m.size == 32) op = reg_to_reg ? LirOpcode::Vmovaps : LirOpcode::Vmovups;
            else if (m.size == 16 || reg_to_reg) op = LirOpcode::Movaps;
            else op = m.size == 4 ? LirOpcode::Movss : LirOpcode::Movsd;
        }
        auto inst = std::make_unique<LirInst>(op);
        inst->add_def(m.dst);
        inst->add_use(m.src);
        return inst;
    };
    while (!moves.empty()) {
        bool progress = false;
        for (size_t k = 0; k < moves.size(); ++k) {
            bool read_later = false;
            for (size_t j = 0; j < moves.size() && !read_later; ++j) {
                if (j != k && same_location(moves[j].src, moves[k].dst)) read_later = true;
            }
            if (read_later) continue;
            out.push_back(make(moves[k]));
            moves.erase(moves.begin() + static_cast<std::ptrdiff_t>(k));
            progress = true;
            break;
        }
        if (progress) continue;
        auto pc = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
        for (const auto& m : moves) {
            pc->add_def(m.dst);
            pc->add_use(m.src);
        }
        out.push_back(std::move(pc));
        emitted_parallel_copy_ = true;
        moves.clear();
    }
}

// Each critical edge with moves gets a block of its own: the moves and a
// jump to the target. The source's branches to the target go to the new
// block instead, and when the source fell through to the target the new
// block is placed right after it.
void LinearScanAllocator::split_critical_edges() {
    if (edge_splits_.empty()) return;
    uint32_t next_id = 0;
    for (const auto& b : fn_.blocks) next_id = std::max(next_id, b->id + 1);
    for (auto& e : edge_splits_) {
        LirBlock* pred = e.pred;
        LirBlock* succ = e.succ;
        size_t pred_pos = 0;
        while (pred_pos < fn_.blocks.size() && fn_.blocks[pred_pos].get() != pred) ++pred_pos;
        const bool falls = !pred->instructions.empty() && !is_unconditional_end(*pred->instructions.back()) &&
                           pred_pos + 1 < fn_.blocks.size() && fn_.blocks[pred_pos + 1].get() == succ;

        LirBlock* nb = fn_.create_block_with_id(next_id++, pred->name + ".edge");
        nb->loop_depth = std::min(pred->loop_depth, succ->loop_depth);
        emit_moves(e.moves, nb->instructions);
        auto jmp = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp->add_use(LirOperand::label(succ->id));
        nb->append_inst(std::move(jmp));

        for (auto& inst : pred->instructions) {
            if (inst->opcode != LirOpcode::Jcc && inst->opcode != LirOpcode::Jmp) continue;
            for (auto& u : inst->uses) {
                if (u.is_label() && u.label_id == succ->id) u.label_id = nb->id;
            }
        }
        std::replace(pred->successors.begin(), pred->successors.end(), succ, nb);
        std::replace(succ->predecessors.begin(), succ->predecessors.end(), pred, nb);
        nb->successors.push_back(succ);
        nb->predecessors.push_back(pred);

        // Place it: after pred when pred fell through to succ; otherwise
        // after a block control never falls out of.
        std::unique_ptr<LirBlock> owned = std::move(fn_.blocks.back());
        fn_.blocks.pop_back();
        size_t at = fn_.blocks.size();
        if (falls) {
            at = pred_pos + 1;
        } else if (fn_.blocks.empty() || fn_.blocks.back()->instructions.empty() ||
                   !is_unconditional_end(*fn_.blocks.back()->instructions.back())) {
            for (size_t i = 0; i < fn_.blocks.size(); ++i) {
                const auto& b = fn_.blocks[i];
                if (!b->instructions.empty() && is_unconditional_end(*b->instructions.back())) {
                    at = i + 1;
                    break;
                }
            }
        }
        fn_.blocks.insert(fn_.blocks.begin() + static_cast<std::ptrdiff_t>(at), std::move(owned));
    }
}

} // namespace brass::codegen
