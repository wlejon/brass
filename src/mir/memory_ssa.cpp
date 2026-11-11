#include <brass/mir/memory_ssa.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace brass {

std::string_view memory_access_kind_name(MemoryAccessKind kind) noexcept {
    switch (kind) {
        case MemoryAccessKind::LiveOnEntry: return "LiveOnEntry";
        case MemoryAccessKind::Use: return "MemoryUse";
        case MemoryAccessKind::Def: return "MemoryDef";
        case MemoryAccessKind::Phi: return "MemoryPhi";
    }
    return "Unknown";
}

MemoryUse* MemoryAccess::as_use() noexcept {
    return (kind_ == MemoryAccessKind::Use) ? static_cast<MemoryUse*>(this) : nullptr;
}

const MemoryUse* MemoryAccess::as_use() const noexcept {
    return (kind_ == MemoryAccessKind::Use) ? static_cast<const MemoryUse*>(this) : nullptr;
}

MemoryDef* MemoryAccess::as_def() noexcept {
    return (kind_ == MemoryAccessKind::Def) ? static_cast<MemoryDef*>(this) : nullptr;
}

const MemoryDef* MemoryAccess::as_def() const noexcept {
    return (kind_ == MemoryAccessKind::Def) ? static_cast<const MemoryDef*>(this) : nullptr;
}

MemoryPhi* MemoryAccess::as_phi() noexcept {
    return (kind_ == MemoryAccessKind::Phi) ? static_cast<MemoryPhi*>(this) : nullptr;
}

const MemoryPhi* MemoryAccess::as_phi() const noexcept {
    return (kind_ == MemoryAccessKind::Phi) ? static_cast<const MemoryPhi*>(this) : nullptr;
}

void MemoryPhi::add_incoming(BasicBlock* pred, MemoryAccess* access) {
    if (!pred) return;
    for (auto& entry : incoming_) {
        if (entry.first == pred) {
            entry.second = access;
            return;
        }
    }
    incoming_.emplace_back(pred, access);
}

MemoryAccess* MemoryPhi::get_incoming(const BasicBlock* pred) const {
    for (const auto& entry : incoming_) {
        if (entry.first == pred) {
            return entry.second;
        }
    }
    return nullptr;
}

void MemoryPhi::set_incoming(BasicBlock* pred, MemoryAccess* access) {
    add_incoming(pred, access);
}

MemorySSA::MemorySSA(const Function& fn, const DominatorTree& dom, const AliasAnalysis& aa)
    : fn_(&fn), dom_(&dom), aa_(&aa) {
    build();
}

MemorySSA::~MemorySSA() = default;
MemorySSA::MemorySSA(MemorySSA&&) noexcept = default;
MemorySSA& MemorySSA::operator=(MemorySSA&&) noexcept = default;

MemoryAccess* MemorySSA::get_memory_access(const Instruction* inst) const {
    if (!inst) return nullptr;
    auto it = inst_to_access_.find(inst);
    return (it != inst_to_access_.end()) ? it->second : nullptr;
}

MemoryUse* MemorySSA::get_memory_use(const Instruction* inst) const {
    MemoryAccess* acc = get_memory_access(inst);
    return acc ? acc->as_use() : nullptr;
}

MemoryDef* MemorySSA::get_memory_def(const Instruction* inst) const {
    MemoryAccess* acc = get_memory_access(inst);
    return acc ? acc->as_def() : nullptr;
}

MemoryPhi* MemorySSA::get_memory_phi(const BasicBlock* bb) const {
    if (!bb) return nullptr;
    auto it = block_to_phi_.find(bb);
    return (it != block_to_phi_.end()) ? it->second : nullptr;
}

namespace {

// One query's walk up the def chain, through the merges it can see past.
//
// A MemoryPhi stands for "some write happened on some path into this block".
// For one load that is often not so: the writes on those paths are stores the
// load cannot overlap (another alias class, another field), and then every
// incoming path carries the same state up from one access above the merge,
// which is the load's real clobber. `resolve` answers that access for a phi:
// each incoming chain is walked to its first clobber, and when all of them
// end at one access the phi is transparent and the walk goes on from there.
// A chain that comes back to the phi being resolved went round a cycle with
// no clobber on it and says nothing; one that ends at another phi still being
// resolved ends there, which is sound (nothing clobbers between the two) and
// only stops short. The steps are bounded, so a function of many merges costs
// a load a fixed amount; out of steps, the phi stands as the clobber.
struct ClobberWalk {
    const AliasAnalysis& aa;
    const Instruction* load;
    uint32_t steps_left = 256;
    std::unordered_map<const MemoryPhi*, MemoryAccess*> resolved;  // nullptr: in progress

    MemoryAccess* from(MemoryAccess* cur) {
        while (cur) {
            if (cur->is_def()) {
                MemoryDef* def = cur->as_def();
                if (steps_left == 0 || aa.can_clobber(def->origin_instruction(), load)) return cur;
                --steps_left;
                cur = def->defining_access();
            } else if (cur->is_phi()) {
                MemoryAccess* above = resolve(cur->as_phi());
                if (above == cur) return cur;
                cur = above;
            } else {
                return cur;
            }
        }
        return cur;
    }

    // The access every path into `phi` carries, or `phi` itself.
    MemoryAccess* resolve(MemoryPhi* phi) {
        auto [it, fresh] = resolved.try_emplace(phi, nullptr);
        if (!fresh) return it->second ? it->second : phi;  // in progress: the walk ends here
        MemoryAccess* common = nullptr;
        bool transparent = steps_left != 0;
        for (const auto& [pred, incoming] : phi->incoming()) {
            if (!transparent) break;
            --steps_left;
            MemoryAccess* end = from(incoming);
            if (end == phi) continue;  // round a clobber-free cycle
            if (steps_left == 0 || !end || (common && end != common)) transparent = false;
            common = end;
        }
        MemoryAccess* answer = transparent && common ? common : phi;
        resolved[phi] = answer;
        return answer;
    }
};

}  // namespace

MemoryAccess* MemorySSA::find_clobbering_access(const Instruction* load_inst, MemoryAccess* starting_access) const {
    if (!load_inst || !starting_access) return starting_access;
    ClobberWalk walk{*aa_, load_inst};
    return walk.from(starting_access);
}

void MemorySSA::compute_dominance_frontiers(std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>>& df) {
    for (const BasicBlock* bb : fn_->blocks()) {
        if (!bb || !dom_->is_reachable(bb) || bb->predecessors().size() < 2) continue;
        const BasicBlock* idom_bb = dom_->immediate_dominator(bb);

        for (BasicBlock* pred : bb->predecessors()) {
            const BasicBlock* runner = pred;
            while (runner && runner != idom_bb && dom_->is_reachable(runner)) {
                auto& list = df[runner];
                if (std::find(list.begin(), list.end(), bb) == list.end()) {
                    list.push_back(const_cast<BasicBlock*>(bb));
                }
                runner = dom_->immediate_dominator(runner);
            }
        }
    }
}

void MemorySSA::build() {
    live_on_entry_ = std::make_unique<MemoryLiveOnEntry>(0);

    // 1. Identify blocks containing memory definitions
    std::unordered_set<const BasicBlock*> def_blocks;
    for (const BasicBlock* bb : fn_->blocks()) {
        if (!bb || !dom_->is_reachable(bb)) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            Opcode op = inst->opcode();
            if (op == Opcode::store || op == Opcode::store_indexed || op == Opcode::vstore) {
                def_blocks.insert(bb);
                break;
            }
            if ((is_call(op) && !is_allocation_call(inst)) || is_coro_op(op)) {
                def_blocks.insert(bb);
                break;
            }
        }
    }

    // 2. Compute dominance frontiers
    std::unordered_map<const BasicBlock*, std::vector<BasicBlock*>> df;
    compute_dominance_frontiers(df);

    // 3. Place MemoryPhi nodes at iterated dominance frontiers
    std::vector<const BasicBlock*> worklist(def_blocks.begin(), def_blocks.end());
    std::unordered_set<const BasicBlock*> has_phi;

    while (!worklist.empty()) {
        const BasicBlock* x = worklist.back();
        worklist.pop_back();

        auto it = df.find(x);
        if (it != df.end()) {
            for (BasicBlock* y : it->second) {
                if (has_phi.insert(y).second) {
                    auto phi = std::make_unique<MemoryPhi>(next_access_id_++, y);
                    block_to_phi_[y] = phi.get();
                    all_accesses_.push_back(std::move(phi));
                    worklist.push_back(y);
                }
            }
        }
    }

    // 4. Renaming Walk over the DominatorTree
    const BasicBlock* entry = fn_->entry_block();
    if (!entry || !dom_->is_reachable(entry)) return;

    std::vector<MemoryAccess*> version_stack;
    version_stack.push_back(live_on_entry_.get());
    std::vector<MemoryUse*> uses;

    // Each open block's version_stack height on entry; leaving it pops back to that.
    std::vector<size_t> version_marks;
    auto rename_block = [&](const BasicBlock* bb) -> void {
        version_marks.push_back(version_stack.size());

        auto phi_it = block_to_phi_.find(bb);
        if (phi_it != block_to_phi_.end()) {
            version_stack.push_back(phi_it->second);
        }

        for (Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (!inst) continue;
            Opcode op = inst->opcode();

            if (op == Opcode::load || op == Opcode::load_indexed || op == Opcode::vload) {
                // The reaching def for now; narrowed to the load's clobber
                // below, once every phi has its incoming accesses.
                auto use = std::make_unique<MemoryUse>(next_access_id_++, const_cast<BasicBlock*>(bb), inst,
                                                       version_stack.back());
                inst_to_access_[inst] = use.get();
                uses.push_back(use.get());
                all_accesses_.push_back(std::move(use));
            } else if (op == Opcode::store || op == Opcode::store_indexed || op == Opcode::vstore) {
                MemoryAccess* current_def = version_stack.back();
                auto def = std::make_unique<MemoryDef>(next_access_id_++, const_cast<BasicBlock*>(bb), inst, current_def);
                inst_to_access_[inst] = def.get();
                version_stack.push_back(def.get());
                all_accesses_.push_back(std::move(def));
            } else if (is_call(op) || is_coro_op(op)) {
                // Coroutine operations run the coroutine body, which may write anything.
                if (is_coro_op(op) || !is_allocation_call(inst)) {
                    MemoryAccess* current_def = version_stack.back();
                    auto def = std::make_unique<MemoryDef>(next_access_id_++, const_cast<BasicBlock*>(bb), inst, current_def);
                    inst_to_access_[inst] = def.get();
                    version_stack.push_back(def.get());
                    all_accesses_.push_back(std::move(def));
                }
            }
        }

        // Populate MemoryPhi in CFG successors
        for (BasicBlock* succ : bb->successors()) {
            if (!succ) continue;
            auto succ_phi_it = block_to_phi_.find(succ);
            if (succ_phi_it != block_to_phi_.end()) {
                succ_phi_it->second->add_incoming(const_cast<BasicBlock*>(bb), version_stack.back());
            }
        }
    };

    // Dominated children are renamed between a block's entry and its leave,
    // which pops the versions the block pushed.
    walk_dominator_tree(*dom_, entry, rename_block, [&](const BasicBlock*) {
        version_stack.resize(version_marks.back());
        version_marks.pop_back();
    });

    // Fill in default LiveOnEntry for any unvisited phi predecessors
    for (auto& [b, phi] : block_to_phi_) {
        for (BasicBlock* pred : b->predecessors()) {
            if (!phi->get_incoming(pred)) {
                phi->add_incoming(pred, live_on_entry_.get());
            }
        }
    }

    // Each use's defining access is its load's clobber (find_clobbering_access).
    for (MemoryUse* use : uses) {
        use->set_defining_access(find_clobbering_access(use->origin_instruction(), use->defining_access()));
    }
}

} // namespace brass
