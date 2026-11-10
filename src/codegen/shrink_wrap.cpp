#include <brass/codegen/shrink_wrap.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/core/phase_times.hpp>
#include <algorithm>
#include <unordered_map>

// Region choice (choose_frame_regions) and the split before register
// allocation (split_for_shrink_wrap); shrink_wrap.hpp says what both are for.

namespace brass::codegen {

namespace {

constexpr uint32_t kUndef = UINT32_MAX;

struct DomInfo {
    std::vector<uint32_t> rpo;
    std::vector<uint32_t> rpo_index;  // kUndef: unreachable
    std::vector<std::vector<uint32_t>> preds;  // reachable predecessors
    std::vector<uint32_t> idom;
    std::vector<uint8_t> in_loop;
};

void compute_rpo(const FrameGraph& g, DomInfo& d) {
    const size_t n = g.succs.size();
    d.rpo_index.assign(n, kUndef);
    std::vector<uint8_t> seen(n, 0);
    std::vector<uint32_t> post;
    std::vector<std::pair<uint32_t, size_t>> stack{{0u, 0}};
    seen[0] = 1;
    while (!stack.empty()) {
        auto& [v, i] = stack.back();
        if (i < g.succs[v].size()) {
            const uint32_t s = g.succs[v][i++];
            if (!seen[s]) {
                seen[s] = 1;
                stack.push_back({s, 0});
            }
            continue;
        }
        post.push_back(v);
        stack.pop_back();
    }
    d.rpo.assign(post.rbegin(), post.rend());
    for (size_t i = 0; i < d.rpo.size(); ++i) d.rpo_index[d.rpo[i]] = static_cast<uint32_t>(i);
    d.preds.assign(n, {});
    for (uint32_t v : d.rpo) {
        for (uint32_t s : g.succs[v]) d.preds[s].push_back(v);
    }
}

// Cooper, Harvey and Kennedy's iteration over reverse postorder.
void compute_idom(DomInfo& d) {
    const size_t n = d.rpo_index.size();
    d.idom.assign(n, kUndef);
    d.idom[0] = 0;
    auto intersect = [&](uint32_t a, uint32_t b) {
        while (a != b) {
            while (d.rpo_index[a] > d.rpo_index[b]) a = d.idom[a];
            while (d.rpo_index[b] > d.rpo_index[a]) b = d.idom[b];
        }
        return a;
    };
    for (bool changed = true; changed;) {
        changed = false;
        for (size_t k = 1; k < d.rpo.size(); ++k) {
            const uint32_t b = d.rpo[k];
            uint32_t nd = kUndef;
            for (uint32_t p : d.preds[b]) {
                if (d.idom[p] == kUndef) continue;
                nd = nd == kUndef ? p : intersect(p, nd);
            }
            if (nd != d.idom[b]) {
                d.idom[b] = nd;
                changed = true;
            }
        }
    }
}

// A node is in a loop when it is in a strongly connected component of more
// than one node, or branches to itself (Tarjan, iterative).
void compute_loops(const FrameGraph& g, DomInfo& d) {
    const size_t n = g.succs.size();
    d.in_loop.assign(n, 0);
    std::vector<uint32_t> index(n, kUndef), low(n, 0);
    std::vector<uint8_t> on_stack(n, 0);
    std::vector<uint32_t> scc_stack;
    std::vector<std::pair<uint32_t, size_t>> call{{0u, 0}};
    uint32_t next = 0;
    index[0] = low[0] = next++;
    scc_stack.push_back(0);
    on_stack[0] = 1;
    while (!call.empty()) {
        auto& [v, i] = call.back();
        if (i < g.succs[v].size()) {
            const uint32_t w = g.succs[v][i++];
            if (w == v) d.in_loop[v] = 1;
            if (index[w] == kUndef) {
                index[w] = low[w] = next++;
                scc_stack.push_back(w);
                on_stack[w] = 1;
                call.push_back({w, 0});
            } else if (on_stack[w]) {
                low[v] = std::min(low[v], index[w]);
            }
            continue;
        }
        const uint32_t done = v;
        call.pop_back();
        if (!call.empty()) low[call.back().first] = std::min(low[call.back().first], low[done]);
        if (low[done] != index[done]) continue;
        size_t members = 0;
        size_t top = scc_stack.size();
        while (true) {
            const uint32_t w = scc_stack[--top];
            ++members;
            if (w == done) break;
        }
        for (size_t k = top; k < scc_stack.size(); ++k) {
            on_stack[scc_stack[k]] = 0;
            if (members > 1) d.in_loop[scc_stack[k]] = 1;
        }
        scc_stack.resize(top);
    }
}

} // namespace

FrameRegions choose_frame_regions(const FrameGraph& g) {
    FrameRegions out;
    const size_t n = g.succs.size();
    if (n == 0 || g.needs_frame[0]) return out;
    DomInfo d;
    compute_rpo(g, d);
    compute_idom(d);
    compute_loops(g, d);

    // Each block that needs the frame starts at itself and climbs its
    // dominators out of any loop, and into a cold dominator (one region for
    // a cold path rather than one per call on it); never into the entry.
    std::vector<uint8_t> is_save(n, 0);
    for (uint32_t x : d.rpo) {
        if (!g.needs_frame[x]) continue;
        uint32_t s = x;
        while (s != 0 && d.idom[s] != 0 && (d.in_loop[s] || g.cold[d.idom[s]])) s = d.idom[s];
        if (s == 0 || d.in_loop[s]) return out;
        is_save[s] = 1;
    }

    // A save point in another's region goes (reverse postorder visits a
    // dominator first). One entered from another region is entered after
    // that region's teardown, on the edge.
    std::vector<uint32_t> region(n, FrameRegions::kNone);
    for (uint32_t b : d.rpo) {
        if (b == 0) continue;
        if (is_save[b] && region[d.idom[b]] != FrameRegions::kNone) is_save[b] = 0;
        region[b] = is_save[b] ? b : region[d.idom[b]];
    }

    for (uint32_t b : d.rpo) {
        if (is_save[b]) out.saves.push_back(b);
    }
    if (out.saves.empty() || out.saves.size() > kMaxFrameRegions) return out;

    // Worth it only where some return is reached without a frame.
    std::vector<uint8_t> seen(n, 0);
    std::vector<uint32_t> work{0};
    seen[0] = 1;
    bool frameless_return = false;
    while (!work.empty() && !frameless_return) {
        const uint32_t v = work.back();
        work.pop_back();
        if (g.returns[v]) frameless_return = true;
        for (uint32_t s : g.succs[v]) {
            if (seen[s] || region[s] != FrameRegions::kNone) continue;
            seen[s] = 1;
            work.push_back(s);
        }
    }
    if (!frameless_return) {
        out.saves.clear();
        return out;
    }
    out.region = std::move(region);
    out.wrapped = true;
    return out;
}

bool shrink_wrap_shape_ok(const LirFunction& fn, const CallingConvention& cc) {
    if (!cc.target().is_x64() || cc.kind() == CallingConvKind::Win64) return false;
    if (!fn.resume_entries.empty() || fn.forced_saved_gprs != 0) return false;
    if (!fn.frame.tagged_locals.empty() || fn.frame.local_frame_bytes != 0 ||
        fn.frame.has_stack_args)
        return false;
    if (fn.frame.saved_callee_xmms != 0 || fn.blocks.empty()) return false;
    for (const auto& b : fn.blocks) {
        if (b->is_landing_pad) return false;
        for (const auto& inst : b->instructions) {
            if (inst->is_invoke || inst->unwind_block_id != UINT32_MAX) return false;
            if (inst->opcode == LirOpcode::Push || inst->opcode == LirOpcode::Pop) return false;
            for (const auto& u : inst->uses) {
                if (u.is_label() && !inst->is_branch()) return false;
            }
            for (const auto& dd : inst->defs) {
                if (dd.is_label()) return false;
            }
        }
    }
    return true;
}

namespace {

bool names_frame_preg(const PReg& p, uint32_t mask) {
    return p.is_gpr() && p.code < 16 && (mask & (1u << p.code)) != 0;
}

bool mem_needs_frame(const LirMem& m, uint32_t mask) {
    if (!m.has_base() && !m.has_index()) return true;  // rbp-relative
    const uint32_t stack = mask | (1u << static_cast<int>(x64::GPR::RSP));
    return names_frame_preg(m.base_preg, stack) || names_frame_preg(m.index_preg, stack);
}

} // namespace

uint32_t frame_gpr_mask(const LirFunction& fn, const CallingConvention& cc) {
    uint32_t mask = 0;
    for (int i = 0; i < 16; ++i) {
        const auto g = static_cast<x64::GPR>(i);
        if (g != x64::GPR::RSP && cc.is_callee_saved(g)) mask |= 1u << i;
    }
    return (mask & ~fn.reserved_gprs) | (1u << static_cast<int>(x64::GPR::RBP));
}

bool inst_needs_frame(const LirInst& inst, uint32_t mask, const std::vector<int32_t>& free_slots) {
    switch (inst.opcode) {
        case LirOpcode::Call: case LirOpcode::CallIndirect: case LirOpcode::Safepoint:
        case LirOpcode::GuardExit: case LirOpcode::Push: case LirOpcode::Pop:
        case LirOpcode::WriteBarrier:
            return true;
        default:
            break;
    }
    if (inst.clobbered_gprs & mask) return true;
    for (const auto* ops : {&inst.defs, &inst.uses}) {
        for (const auto& op : *ops) {
            if (op.is_spill_slot() &&
                !std::binary_search(free_slots.begin(), free_slots.end(), op.spill_slot)) {
                return true;
            }
            if (op.is_local_slot()) return true;
            if (op.is_mem() && mem_needs_frame(op.mem_val, mask)) return true;
            if (op.is_preg() && names_frame_preg(op.preg_val, mask)) return true;
        }
    }
    for (const auto& op : inst.defs) {
        if (op.is_preg() && op.preg_val == PReg::gpr(x64::GPR::RSP)) return true;
    }
    for (const auto& c : inst.use_constraints) {
        if (c.has_fixed_preg && names_frame_preg(c.fixed_preg, mask)) return true;
    }
    for (const auto& c : inst.def_constraints) {
        if (c.has_fixed_preg && names_frame_preg(c.fixed_preg, mask)) return true;
    }
    return false;
}

bool reads_rsp(const LirInst& inst) {
    const PReg rsp = PReg::gpr(x64::GPR::RSP);
    for (const auto& op : inst.uses) {
        if (op.is_preg() && op.preg_val == rsp) return true;
    }
    return false;
}

namespace {

bool needs_frame_pre_ra(const LirBlock& b, uint32_t mask) {
    if (b.is_landing_pad) return true;
    static const std::vector<int32_t> kNoSlots;
    for (const auto& inst : b.instructions) {
        if (inst_needs_frame(*inst, mask, kNoSlots)) return true;
    }
    return false;
}

// The vregs an operand names.
template <typename F>
void for_each_vreg(LirOperand& op, F&& f) {
    if (op.is_vreg()) {
        f(op.vreg_val);
    } else if (op.is_mem()) {
        if (op.mem_val.base_vreg.is_valid()) f(op.mem_val.base_vreg);
        if (op.mem_val.index_vreg.is_valid()) f(op.mem_val.index_vreg);
    }
}

std::unique_ptr<LirInst> make_copies(const std::vector<std::pair<VReg, VReg>>& moves) {
    auto pc = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
    for (const auto& [dst, src] : moves) {
        pc->add_def(LirOperand::vreg(dst, dst.size));
        pc->add_use(LirOperand::vreg(src, src.size));
    }
    return pc;
}

void replace_ptr(std::vector<LirBlock*>& v, LirBlock* from, LirBlock* to) {
    for (auto& p : v) {
        if (p == from) p = to;
    }
}

} // namespace

bool split_for_shrink_wrap(LirFunction& fn, const CallingConvention& cc) {
    if (!shrink_wrap_shape_ok(fn, cc)) return false;
    const size_t n = fn.blocks.size();
    const uint32_t mask = frame_gpr_mask(fn, cc);
    std::unordered_map<const LirBlock*, uint32_t> index;
    index.reserve(n);
    for (size_t i = 0; i < n; ++i) index[fn.blocks[i].get()] = static_cast<uint32_t>(i);

    FrameGraph g;
    g.succs.resize(n);
    g.cold.assign(n, 0);
    g.needs_frame.assign(n, 0);
    g.returns.assign(n, 0);
    for (size_t i = 0; i < n; ++i) {
        const LirBlock& b = *fn.blocks[i];
        for (const LirBlock* s : b.successors) {
            auto it = index.find(s);
            if (it == index.end()) return false;
            g.succs[i].push_back(it->second);
        }
        g.cold[i] = b.cold;
        g.needs_frame[i] = needs_frame_pre_ra(b, mask);
        for (const auto& inst : b.instructions) g.returns[i] |= inst->opcode == LirOpcode::Ret;
    }
    const FrameRegions regions = choose_frame_regions(g);
    if (!regions.wrapped) return false;

    // Every exit edge must take copies: its source jumps by label, with no
    // fall-through, to a target its branches name.
    struct Exit {
        uint32_t from, to;
    };
    std::vector<Exit> exits;
    for (uint32_t u = 0; u < n; ++u) {
        if (regions.region[u] == FrameRegions::kNone) continue;
        const LirBlock& b = *fn.blocks[u];
        for (uint32_t t : g.succs[u]) {
            if (regions.region[t] == regions.region[u]) continue;
            if (b.instructions.empty() || b.instructions.back()->opcode != LirOpcode::Jmp) return false;
            bool named = false;
            for (const auto& inst : b.instructions) {
                if (!inst->is_branch()) continue;
                for (const auto& op : inst->uses) named |= op.is_label() && op.label_id == fn.blocks[t]->id;
            }
            if (!named) return false;
            exits.push_back({u, t});
        }
    }

    LivenessAnalysis liveness(fn);
    liveness.run();
    const auto& guard_exits = liveness.guard_exits();
    // Live-in sets by position, read before anything changes.
    std::vector<std::vector<VReg>> live_in(n);
    for (size_t i = 0; i < n; ++i) {
        const auto& li = liveness.block_liveness(fn.blocks[i].get()).live_in;
        live_in[i].assign(li.begin(), li.end());
    }
    std::vector<LirBlock*> by_pos(n);
    for (size_t i = 0; i < n; ++i) by_pos[i] = fn.blocks[i].get();

    bool changed = false;
    const size_t num_vregs = fn.vreg_table.size();
    for (uint32_t s : regions.saves) {
        // A guard's exit block reads its state where the guard's Jcc is (its
        // uses are counted there): its values are not live into it, and
        // renaming them would read nothing.
        if (guard_exits.count(by_pos[s]->id)) continue;
        std::vector<uint32_t> renamed(num_vregs, VReg::kInvalid);
        auto name_in_region = [&](VReg v) {
            if (renamed[v.id] == VReg::kInvalid) {
                const VReg canon = fn.vreg_table[v.id].vreg;
                renamed[v.id] = fn.allocate_vreg(canon.reg_class, canon.size, canon.is_gcref, canon.is_tagged).id;
            }
            return renamed[v.id];
        };
        std::vector<std::pair<VReg, VReg>> entry_copies;
        for (VReg v : live_in[s]) {
            const VReg canon = fn.vreg_table[v.id].vreg;
            VReg inner = canon;
            inner.id = name_in_region(canon);
            entry_copies.push_back({inner, canon});
        }
        for (uint32_t b = 0; b < n; ++b) {
            if (regions.region[b] != s) continue;
            for (auto& inst : by_pos[b]->instructions) {
                auto rename = [&](VReg& v) {
                    if (v.id < num_vregs) v.id = name_in_region(v);
                };
                for (auto& op : inst->defs) for_each_vreg(op, rename);
                for (auto& op : inst->uses) for_each_vreg(op, rename);
                for (VReg& v : inst->live_gcrefs) rename(v);
            }
        }
        if (!entry_copies.empty()) {
            auto& ins = by_pos[s]->instructions;
            ins.insert(ins.begin(), make_copies(entry_copies));
        }
        for (const Exit& e : exits) {
            if (regions.region[e.from] != s) continue;
            std::vector<std::pair<VReg, VReg>> back;
            for (VReg v : live_in[e.to]) {
                if (v.id >= num_vregs || renamed[v.id] == VReg::kInvalid) continue;
                const VReg canon = fn.vreg_table[v.id].vreg;
                VReg inner = canon;
                inner.id = renamed[v.id];
                back.push_back({canon, inner});
            }
            if (back.empty()) continue;
            LirBlock* u = by_pos[e.from];
            LirBlock* t = by_pos[e.to];
            if (u->successors.size() == 1) {
                u->instructions.insert(u->instructions.end() - 1, make_copies(back));
                continue;
            }
            // The edge gets a block of its own, inside the region, right
            // after its source (which ends in a jump, so nothing falls in).
            LirBlock* edge = fn.create_block(u->name + ".sw_exit");
            edge->cold = u->cold || t->cold;
            edge->append_inst(make_copies(back));
            auto jmp = std::make_unique<LirInst>(LirOpcode::Jmp);
            jmp->add_use(LirOperand::label(t->id));
            edge->append_inst(std::move(jmp));
            for (auto& inst : u->instructions) {
                if (!inst->is_branch()) continue;
                for (auto& op : inst->uses) {
                    if (op.is_label() && op.label_id == t->id) op.label_id = edge->id;
                }
            }
            replace_ptr(u->successors, t, edge);
            replace_ptr(t->predecessors, u, edge);
            edge->predecessors.push_back(u);
            edge->successors.push_back(t);
            auto owned = std::move(fn.blocks.back());
            fn.blocks.pop_back();
            auto at = std::find_if(fn.blocks.begin(), fn.blocks.end(), [&](const auto& p) { return p.get() == u; });
            fn.blocks.insert(at + 1, std::move(owned));
        }
        changed = true;
    }
    if (!changed) return false;
    // What the blocks without a frame name stays out of callee-saved
    // registers a hint would give it (lir.hpp, frameless_vregs).
    fn.frameless_vregs.assign(fn.vreg_table.size(), 0);
    for (uint32_t b = 0; b < n; ++b) {
        if (regions.region[b] != FrameRegions::kNone) continue;
        for (auto& inst : by_pos[b]->instructions) {
            auto mark = [&](VReg& v) { fn.frameless_vregs[v.id] = 1; };
            for (auto& op : inst->defs) for_each_vreg(op, mark);
            for (auto& op : inst->uses) for_each_vreg(op, mark);
        }
    }
    if (phase_times_enabled()) record_phase_time("x64 shrink-wrap: functions split before allocation", 0);
    return changed;
}

} // namespace brass::codegen
