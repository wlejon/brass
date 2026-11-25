#include <brass/mir/gpu_portability.hpp>
#include <brass/mir/uses.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/gpu_intrinsics.hpp>

#include <algorithm>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brass {

namespace {

bool is_divergent_intrinsic(std::string_view sym) noexcept {
    if (sym == "ptx_tid_x" || sym == "ptx_tid_y" || sym == "ptx_tid_z" ||
        sym == "ptx_global_tid_x" || sym == "ptx_global_id_x" ||
        sym == "ptx_laneid" || sym == "ptx_lane_id" ||
        sym == "ptx_subgroup_id" || sym == "ptx_warpid" || sym == "ptx_warp_id" ||
        sym == "ptx_coop_get_acc" ||
        sym == "ptx_clock" || sym == "ptx_clock64" || sym == "ptx_globaltimer") {
        return true;
    }
    if (sym.rfind("ptx_shfl_", 0) == 0 || sym.rfind("ptx_atom_", 0) == 0) {
        return true;
    }
    return false;
}

bool is_non_portable_intrinsic(std::string_view sym) noexcept {
    return sym == "ptx_warpid" || sym == "ptx_warp_id" || sym == "ptx_nwarpid" ||
           sym == "ptx_smid" || sym == "ptx_nsmid" || sym == "ptx_clock" ||
           sym == "ptx_clock64" || sym == "ptx_globaltimer" ||
           sym == "ptx_bar_sync_count";
}

bool is_fp64_type(Type t) noexcept {
    const TypeKind k = t.kind();
    return k == TypeKind::F64 || k == TypeKind::F64x2 || k == TypeKind::F64x4;
}

bool is_fp64_intrinsic(std::string_view sym) noexcept {
    return sym == "ptx_f32_to_f64" || sym == "ptx_f64_to_f32" ||
           sym == "ptx_shared_alloc_f64" || sym == "ptx_shared_load_f64" ||
           sym == "ptx_shared_load_f64_indexed" || sym == "ptx_shared_store_f64" ||
           sym == "ptx_shared_store_f64_indexed";
}

bool is_i64_atomic_intrinsic(std::string_view sym) noexcept {
    if (sym.rfind("ptx_atom_", 0) != 0) return false;
    return sym.size() >= 4 &&
           (sym.compare(sym.size() - 4, 4, "_i64") == 0 ||
            sym.compare(sym.size() - 4, 4, "_u64") == 0);
}

bool is_uniform_barrier_or_coop_intrinsic(std::string_view sym) noexcept {
    return sym == "ptx_sync" || sym == "ptx_bar_sync" || sym == "ptx_subgroup_barrier" ||
           sym == "ptx_coop_fill_acc" || sym == "ptx_coop_load_a" ||
           sym == "ptx_coop_load_b" || sym == "ptx_coop_load_acc" ||
           sym == "ptx_coop_mma" || sym == "ptx_coop_store_acc" ||
           sym == "ptx_coop_scale_acc" || sym == "ptx_coop_add_acc";
}

bool is_shared_alloc_sym(std::string_view sym) noexcept {
    return sym == "ptx_shared_alloc_f32" || sym == "ptx_shared_alloc_i32" ||
           sym == "ptx_shared_alloc_f64" || sym == "ptx_shared_alloc_i64" ||
           sym == "ptx_shared_alloc_i32x4" || sym == "ptx_shared_alloc_u16";
}

std::string_view expected_alloc_for_shared_access(std::string_view sym) noexcept {
    if (sym == "ptx_shared_load_f32_from_v4" ||
        sym == "ptx_shared_load_i32x4_indexed" || sym == "ptx_shared_store_i32x4_indexed") {
        return "ptx_shared_alloc_i32x4";
    }
    if (sym == "ptx_shared_load_f32" || sym == "ptx_shared_load_f32_indexed" ||
        sym == "ptx_shared_store_f32" || sym == "ptx_shared_store_f32_indexed") {
        return "ptx_shared_alloc_f32";
    }
    if (sym == "ptx_shared_load_i32" || sym == "ptx_shared_load_i32_indexed" ||
        sym == "ptx_shared_store_i32" || sym == "ptx_shared_store_i32_indexed") {
        return "ptx_shared_alloc_i32";
    }
    if (sym == "ptx_shared_load_f64" || sym == "ptx_shared_load_f64_indexed" ||
        sym == "ptx_shared_store_f64" || sym == "ptx_shared_store_f64_indexed") {
        return "ptx_shared_alloc_f64";
    }
    if (sym == "ptx_shared_load_i64" || sym == "ptx_shared_load_i64_indexed" ||
        sym == "ptx_shared_store_i64" || sym == "ptx_shared_store_i64_indexed") {
        return "ptx_shared_alloc_i64";
    }
    if (sym == "ptx_shared_load_u16_indexed" || sym == "ptx_shared_store_u16_indexed") {
        return "ptx_shared_alloc_u16";
    }
    return {};
}

// Cooper-Harvey-Kennedy immediate post-dominators over reachable blocks that can
// reach a function terminator (`ret`/`unreachable`/empty-succs). Nodes stuck in
// endless loops map to the virtual exit node (`kVirtualExit`).
struct PostDominators {
    std::unordered_map<const BasicBlock*, int> bb_to_idx;
    std::vector<const BasicBlock*> idx_to_bb;
    std::vector<int> ipdom; // virtual exit node is index `exit_idx`

    explicit PostDominators(const Function& fn) { build(fn); }

    const BasicBlock* immediate_post_dominator(const BasicBlock* bb) const noexcept {
        auto it = bb_to_idx.find(bb);
        if (it == bb_to_idx.end()) return nullptr;
        const int p = ipdom[it->second];
        if (p < 0 || p >= static_cast<int>(idx_to_bb.size())) return nullptr;
        return idx_to_bb[p];
    }

private:
    void build(const Function& fn) {
        const BasicBlock* entry = fn.entry_block();
        if (!entry) return;

        // Forward reachable blocks from entry.
        std::queue<const BasicBlock*> q;
        bb_to_idx[entry] = 0;
        idx_to_bb.push_back(entry);
        q.push(entry);
        while (!q.empty()) {
            const BasicBlock* cur = q.front();
            q.pop();
            for (const BasicBlock* s : cur->successors()) {
                if (s && bb_to_idx.emplace(s, static_cast<int>(idx_to_bb.size())).second) {
                    idx_to_bb.push_back(s);
                    q.push(s);
                }
            }
        }

        const int n = static_cast<int>(idx_to_bb.size());
        const int exit_idx = n;
        // Reverse edges: rev_succs[u] holds CFG predecessors of u (or nodes exiting to exit_idx).
        std::vector<std::vector<int>> fwd(n + 1);
        std::vector<std::vector<int>> rev(n + 1);
        for (int i = 0; i < n; ++i) {
            const SuccessorList succs = idx_to_bb[i]->successors();
            bool has_cfg_succ = false;
            for (const BasicBlock* s : succs) {
                if (!s) continue;
                auto it = bb_to_idx.find(s);
                if (it != bb_to_idx.end()) {
                    fwd[i].push_back(it->second);
                    rev[it->second].push_back(i);
                    has_cfg_succ = true;
                }
            }
            if (!has_cfg_succ) {
                fwd[i].push_back(exit_idx);
                rev[exit_idx].push_back(i);
            }
        }

        // Reverse-CFG DFS post-order starting from exit_idx.
        std::vector<bool> vis(n + 1, false);
        std::vector<int> post_order;
        struct Frame {
            int u;
            size_t next;
        };
        std::vector<Frame> st;
        vis[exit_idx] = true;
        st.push_back({exit_idx, 0});
        while (!st.empty()) {
            Frame& f = st.back();
            if (f.next == rev[f.u].size()) {
                post_order.push_back(f.u);
                st.pop_back();
                continue;
            }
            int v = rev[f.u][f.next++];
            if (!vis[v]) {
                vis[v] = true;
                st.push_back({v, 0});
            }
        }

        std::vector<int> rpo_num(n + 1, n + 1);
        int rpo = 0;
        for (int i = static_cast<int>(post_order.size()) - 1; i >= 0; --i) {
            rpo_num[post_order[i]] = rpo++;
        }

        ipdom.assign(n + 1, -1);
        ipdom[exit_idx] = exit_idx;

        auto intersect = [&](int b1, int b2) -> int {
            int f1 = b1;
            int f2 = b2;
            while (f1 != f2) {
                while (rpo_num[f1] > rpo_num[f2]) f1 = ipdom[f1];
                while (rpo_num[f2] > rpo_num[f1]) f2 = ipdom[f2];
            }
            return f1;
        };

        bool changed = true;
        while (changed) {
            changed = false;
            for (int i = static_cast<int>(post_order.size()) - 1; i >= 0; --i) {
                const int b = post_order[i];
                if (b == exit_idx) continue;
                int new_ipdom = -1;
                for (int s : fwd[b]) {
                    if (ipdom[s] != -1) {
                        new_ipdom = (new_ipdom == -1) ? s : intersect(s, new_ipdom);
                    }
                }
                if (new_ipdom != -1 && new_ipdom != ipdom[b]) {
                    ipdom[b] = new_ipdom;
                    changed = true;
                }
            }
        }

        // Any reachable block with no path to exit gets exit_idx as ipdom.
        for (int i = 0; i < n; ++i) {
            if (ipdom[i] == -1) ipdom[i] = exit_idx;
        }
    }
};

void report(DiagnosticReporter* diag, std::string msg) {
    if (diag) diag->error(std::move(msg));
}

} // namespace

GpuDivergenceAnalysis::GpuDivergenceAnalysis(const Function& fn) {
    analyze(fn);
}

bool GpuDivergenceAnalysis::is_divergent(const Value* v) const noexcept {
    return v && divergent_values_.count(v) > 0;
}

bool GpuDivergenceAnalysis::is_divergent_branch(const BasicBlock* bb) const noexcept {
    return bb && divergent_branches_.count(bb) > 0;
}

bool GpuDivergenceAnalysis::is_under_divergent_control(const BasicBlock* bb) const noexcept {
    return bb && under_divergent_control_.count(bb) > 0;
}

void GpuDivergenceAnalysis::analyze(const Function& fn) {
    if (fn.blocks().empty() || !fn.entry_block()) return;

    // 1. Seed inherently lane/thread-dependent intrinsics.
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call && inst->result()) {
                const std::string_view sym = target::canonical_gpu_intrinsic(inst->symbol());
                if (is_divergent_intrinsic(sym)) {
                    divergent_values_.insert(inst->result());
                }
            }
        }
    }

    PostDominators pdom(fn);

    // 2. Fixed-point propagation across dataflow, block-param edges, and
    //    control-dependent merges up to each divergent branch's immediate post-dominator.
    bool changed = true;
    while (changed) {
        changed = false;

        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (inst->result() && !divergent_values_.count(inst->result())) {
                    for (const Value* op : inst->operands()) {
                        if (op && divergent_values_.count(op)) {
                            divergent_values_.insert(inst->result());
                            changed = true;
                            break;
                        }
                    }
                }
                if (inst->is_terminator()) {
                    for_each_edge(*inst, [&](const BranchTarget& bt) {
                        if (!bt.block) return;
                        const size_t n = std::min(bt.args.size(), bt.block->param_count());
                        for (size_t k = 0; k < n; ++k) {
                            if (bt.args[k] && divergent_values_.count(bt.args[k]) &&
                                divergent_values_.insert(bt.block->param(k)).second) {
                                changed = true;
                            }
                        }
                    });
                    const Opcode top = inst->opcode();
                    if ((top == Opcode::br_if || top == Opcode::switch_) &&
                        inst->operand_count() > 0 && inst->operand(0) &&
                        divergent_values_.count(inst->operand(0))) {
                        if (divergent_branches_.insert(bb).second) {
                            changed = true;
                        }
                    }
                }
            }
        }

        // Propagate divergent control dependence & phi value divergence.
        for (const BasicBlock* br_bb : divergent_branches_) {
            const BasicBlock* ipdom = pdom.immediate_post_dominator(br_bb);
            std::unordered_set<const BasicBlock*> region;
            std::queue<const BasicBlock*> q;
            for (const BasicBlock* s : br_bb->successors()) {
                if (s && s != ipdom && region.insert(s).second) {
                    q.push(s);
                }
            }
            while (!q.empty()) {
                const BasicBlock* cur = q.front();
                q.pop();
                if (under_divergent_control_.insert(cur).second) {
                    changed = true;
                }
                for (const BasicBlock* s : cur->successors()) {
                    if (s && s != ipdom && region.insert(s).second) {
                        q.push(s);
                    }
                }
            }
            // Any block parameter at the post-dominator merge (or internal merge)
            // whose value can depend on which arm ran becomes divergent.
            auto mark_merge_params = [&](const BasicBlock* merge_bb) {
                if (!merge_bb || merge_bb->param_count() == 0) return;
                for (size_t k = 0; k < merge_bb->param_count(); ++k) {
                    const Value* pv = merge_bb->param(k);
                    if (divergent_values_.count(pv)) continue;
                    const Value* seen = nullptr;
                    bool differs = false;
                    for (const BasicBlock* pred : fn.blocks()) {
                        if (!pred || (pred != br_bb && !region.count(pred))) continue;
                        const Instruction* term = pred->terminator();
                        if (!term) continue;
                        for_each_edge(*term, [&](const BranchTarget& bt) {
                            if (bt.block != merge_bb || k >= bt.args.size()) return;
                            if (!seen) seen = bt.args[k];
                            else if (bt.args[k] != seen) differs = true;
                        });
                        if (differs) break;
                    }
                    if (differs && divergent_values_.insert(pv).second) {
                        changed = true;
                    }
                }
            };
            mark_merge_params(ipdom);
            for (const BasicBlock* rbb : region) mark_merge_params(rbb);
        }
    }
}

bool verify_gpu_portability(const Function& fn,
                            const GpuPortabilityOptions& opts,
                            DiagnosticReporter* diag) {
    bool ok = true;
    const std::string prefix = "Function '" + std::string(fn.name()) + "': ";

    if (opts.check_ssa) {
        Verifier v(diag);
        if (!v.verify_function(fn)) {
            ok = false;
        }
    }

    if (!opts.allow_fp64) {
        if (is_fp64_type(fn.return_type())) {
            report(diag, prefix + "FP64 return type is disallowed when allow_fp64=false.");
            ok = false;
        }
        for (size_t i = 0; i < fn.param_count(); ++i) {
            if (is_fp64_type(fn.param_type(i))) {
                report(diag, prefix + "FP64 parameter " + std::to_string(i) +
                                 " is disallowed when allow_fp64=false.");
                ok = false;
            }
        }
    }

    // Shared memory provenance lattice across pointers.
    enum class PtrOrigin : uint8_t { Unknown = 0, Global = 1, Shared = 2, Conflict = 3 };
    std::unordered_map<const Value*, PtrOrigin> ptr_origin;
    std::unordered_map<const Value*, std::string_view> shared_alloc;

    auto join_origin = [](PtrOrigin a, PtrOrigin b) noexcept -> PtrOrigin {
        if (a == PtrOrigin::Unknown) return b;
        if (b == PtrOrigin::Unknown) return a;
        if (a == b) return a;
        return PtrOrigin::Conflict;
    };

    if (opts.check_shared_provenance) {
        if (const BasicBlock* entry = fn.entry_block()) {
            for (size_t i = 0; i < entry->param_count(); ++i) {
                const Value* pv = entry->param(i);
                if (pv && pv->type().is_pointer()) {
                    ptr_origin[pv] = PtrOrigin::Global;
                }
            }
        }
        for (const BasicBlock* bb : fn.blocks()) {
            if (!bb) continue;
            for (const Instruction* inst : *bb) {
                if (inst->opcode() == Opcode::call && inst->result()) {
                    const std::string_view sym = target::canonical_gpu_intrinsic(inst->symbol());
                    if (is_shared_alloc_sym(sym)) {
                        ptr_origin[inst->result()] = PtrOrigin::Shared;
                        shared_alloc[inst->result()] = sym;
                        if (inst->operand_count() != 1 || !inst->operand(0) ||
                            !inst->operand(0)->is_instruction() ||
                            inst->operand(0)->defining_instruction()->opcode() != Opcode::iconst_i32 ||
                            inst->operand(0)->defining_instruction()->imm_i32() <= 0) {
                            report(diag, prefix + "'" + std::string(inst->symbol()) +
                                             "' requires a positive iconst_i32 element count.");
                            ok = false;
                        }
                    } else if (inst->type().is_pointer()) {
                        ptr_origin[inst->result()] = PtrOrigin::Global;
                    }
                }
            }
        }

        bool pchanged = true;
        while (pchanged) {
            pchanged = false;
            for (const BasicBlock* bb : fn.blocks()) {
                if (!bb) continue;
                for (const Instruction* inst : *bb) {
                    if (inst->result() && inst->type().is_pointer()) {
                        const Value* res = inst->result();
                        PtrOrigin next_orig = ptr_origin[res];
                        std::string_view next_alloc = shared_alloc[res];
                        if ((inst->opcode() == Opcode::add || inst->opcode() == Opcode::sub) &&
                            inst->operand_count() >= 1) {
                            for (const Value* op : inst->operands()) {
                                if (op && op->type().is_pointer()) {
                                    next_orig = join_origin(next_orig, ptr_origin[op]);
                                    if (next_alloc.empty()) next_alloc = shared_alloc[op];
                                }
                            }
                        } else if (inst->opcode() == Opcode::select && inst->operand_count() >= 3) {
                            const PtrOrigin o1 = ptr_origin[inst->operand(1)];
                            const PtrOrigin o2 = ptr_origin[inst->operand(2)];
                            next_orig = join_origin(next_orig, join_origin(o1, o2));
                            const auto a1 = shared_alloc[inst->operand(1)];
                            const auto a2 = shared_alloc[inst->operand(2)];
                            if (!a1.empty() && a1 == a2) next_alloc = a1;
                        }
                        if (next_orig != ptr_origin[res]) {
                            ptr_origin[res] = next_orig;
                            pchanged = true;
                        }
                        if (next_alloc != shared_alloc[res]) {
                            shared_alloc[res] = next_alloc;
                            pchanged = true;
                        }
                    }
                    if (inst->is_terminator()) {
                        for_each_edge(*inst, [&](const BranchTarget& bt) {
                            if (!bt.block) return;
                            const size_t n = std::min(bt.args.size(), bt.block->param_count());
                            for (size_t k = 0; k < n; ++k) {
                                const Value* pv = bt.block->param(k);
                                const Value* av = bt.args[k];
                                if (!pv || !pv->type().is_pointer() || !av) continue;
                                const PtrOrigin merged = join_origin(ptr_origin[pv], ptr_origin[av]);
                                if (merged != ptr_origin[pv]) {
                                    ptr_origin[pv] = merged;
                                    pchanged = true;
                                }
                                if (!shared_alloc[av].empty() && shared_alloc[pv].empty()) {
                                    shared_alloc[pv] = shared_alloc[av];
                                    pchanged = true;
                                }
                            }
                        });
                    }
                }
            }
        }
    }

    const GpuDivergenceAnalysis div(fn);

    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        if (opts.check_shared_provenance) {
            for (size_t k = 0; k < bb->param_count(); ++k) {
                const Value* pv = bb->param(k);
                if (pv && pv->type().is_pointer() && ptr_origin[pv] == PtrOrigin::Conflict) {
                    report(diag, prefix + "Block '" + std::string(bb->name()) +
                                     "' parameter " + std::to_string(k) +
                                     " mixes shared and global pointer provenance.");
                    ok = false;
                }
            }
        }

        for (const Instruction* inst : *bb) {
            if (!opts.allow_fp64) {
                if (is_fp64_type(inst->type())) {
                    report(diag, prefix + "FP64 instruction '" +
                                     std::string(opcode_name(inst->opcode())) +
                                     "' is disallowed when allow_fp64=false.");
                    ok = false;
                }
                for (const Value* op : inst->operands()) {
                    if (op && is_fp64_type(op->type())) {
                        report(diag, prefix + "FP64 operand in '" +
                                         std::string(opcode_name(inst->opcode())) +
                                         "' is disallowed when allow_fp64=false.");
                        ok = false;
                        break;
                    }
                }
            }

            if (opts.check_shared_provenance) {
                if (inst->result() && inst->type().is_pointer() &&
                    ptr_origin[inst->result()] == PtrOrigin::Conflict) {
                    report(diag, prefix + "Pointer '" +
                                     std::string(opcode_name(inst->opcode())) +
                                     "' mixes shared and global address spaces.");
                    ok = false;
                }
                const Opcode op = inst->opcode();
                const Value* stored_val = nullptr;
                if ((op == Opcode::store || op == Opcode::vstore) && inst->operand_count() >= 2) {
                    stored_val = inst->operand(1);
                } else if (op == Opcode::store_indexed && inst->operand_count() >= 3) {
                    stored_val = inst->operand(2);
                }
                if (stored_val && stored_val->type().is_pointer() &&
                    ptr_origin[stored_val] == PtrOrigin::Shared) {
                    report(diag, prefix + "Shared memory pointer cannot escape via memory store.");
                    ok = false;
                }
            }

            if (inst->opcode() != Opcode::call) continue;
            const std::string_view raw = inst->symbol();
            const std::string_view sym = target::canonical_gpu_intrinsic(raw);

            if (opts.reject_non_portable_intrinsics && is_non_portable_intrinsic(sym)) {
                report(diag, prefix + "Non-portable GPU intrinsic '" + std::string(raw) +
                                 "' is unsupported across SPIR-V and Metal MSL.");
                ok = false;
            }
            if (!opts.allow_fp64 && is_fp64_intrinsic(sym)) {
                report(diag, prefix + "FP64 intrinsic '" + std::string(raw) +
                                 "' is disallowed when allow_fp64=false.");
                ok = false;
            }
            if (!opts.allow_i64_atomics && is_i64_atomic_intrinsic(sym)) {
                report(diag, prefix + "64-bit atomic intrinsic '" + std::string(raw) +
                                 "' is disallowed on Metal MSL.");
                ok = false;
            }
            if (opts.check_barrier_uniformity && is_uniform_barrier_or_coop_intrinsic(sym) &&
                div.is_under_divergent_control(bb)) {
                report(diag, prefix + "Collective/barrier intrinsic '" + std::string(raw) +
                                 "' in block '" + std::string(bb->name()) +
                                 "' lies under divergent control flow.");
                ok = false;
            }
            if (opts.check_shared_provenance) {
                const std::string_view exp_alloc = expected_alloc_for_shared_access(sym);
                if (!exp_alloc.empty() && inst->operand_count() >= 1) {
                    const Value* sptr = inst->operand(0);
                    if (ptr_origin[sptr] != PtrOrigin::Shared) {
                        report(diag, prefix + "Shared access '" + std::string(raw) +
                                         "' operates on a non-shared pointer.");
                        ok = false;
                    } else if (!shared_alloc[sptr].empty() && shared_alloc[sptr] != exp_alloc) {
                        report(diag, prefix + "Shared access '" + std::string(raw) +
                                         "' mismatches allocation '" +
                                         std::string(shared_alloc[sptr]) + "'.");
                        ok = false;
                    }
                }
            }
        }
    }

    return ok;
}

} // namespace brass
