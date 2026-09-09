#include <brass/mir/memory_ssa.hpp>
#include <brass/mir/escape_analysis.hpp>
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

MemoryAccess* MemorySSA::find_clobbering_access(const Instruction* load_inst, MemoryAccess* starting_access) const {
    if (!load_inst || !starting_access) return starting_access;

    MemoryAccess* cur = starting_access;
    while (cur && cur->is_def()) {
        MemoryDef* def = cur->as_def();
        const Instruction* write_inst = def->origin_instruction();
        if (aa_->can_clobber(write_inst, load_inst)) {
            return cur;
        }
        cur = def->defining_access();
    }
    return cur;
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
            if (is_call(op) && !is_allocation_callee(inst->symbol())) {
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

    auto rename_block = [&](auto& self, const BasicBlock* bb) -> void {
        size_t pushed_count = 0;

        auto phi_it = block_to_phi_.find(bb);
        if (phi_it != block_to_phi_.end()) {
            version_stack.push_back(phi_it->second);
            pushed_count++;
        }

        for (Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (!inst) continue;
            Opcode op = inst->opcode();

            if (op == Opcode::load || op == Opcode::load_indexed || op == Opcode::vload) {
                MemoryAccess* starting_def = version_stack.back();
                // Optimize use using alias analysis
                MemoryAccess* clobber = find_clobbering_access(inst, starting_def);
                auto use = std::make_unique<MemoryUse>(next_access_id_++, const_cast<BasicBlock*>(bb), inst, clobber);
                inst_to_access_[inst] = use.get();
                all_accesses_.push_back(std::move(use));
            } else if (op == Opcode::store || op == Opcode::store_indexed || op == Opcode::vstore) {
                MemoryAccess* current_def = version_stack.back();
                auto def = std::make_unique<MemoryDef>(next_access_id_++, const_cast<BasicBlock*>(bb), inst, current_def);
                inst_to_access_[inst] = def.get();
                version_stack.push_back(def.get());
                pushed_count++;
                all_accesses_.push_back(std::move(def));
            } else if (is_call(op)) {
                if (!is_allocation_callee(inst->symbol())) {
                    MemoryAccess* current_def = version_stack.back();
                    auto def = std::make_unique<MemoryDef>(next_access_id_++, const_cast<BasicBlock*>(bb), inst, current_def);
                    inst_to_access_[inst] = def.get();
                    version_stack.push_back(def.get());
                    pushed_count++;
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

        // Recurse to dominated children in the DominatorTree
        for (const BasicBlock* child : dom_->children(bb)) {
            if (child) {
                self(self, child);
            }
        }

        // Pop versions pushed by this block
        for (size_t i = 0; i < pushed_count; ++i) {
            version_stack.pop_back();
        }
    };

    rename_block(rename_block, entry);

    // Fill in default LiveOnEntry for any unvisited phi predecessors
    for (auto& [b, phi] : block_to_phi_) {
        for (BasicBlock* pred : b->predecessors()) {
            if (!phi->get_incoming(pred)) {
                phi->add_incoming(pred, live_on_entry_.get());
            }
        }
    }
}

} // namespace brass
