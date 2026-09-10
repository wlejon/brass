#include <brass/mir/partial_escape.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <algorithm>
#include <iomanip>
#include <ostream>
#include <queue>
#include <sstream>

namespace brass {

const std::unordered_set<const Value*> PartialEscapeAnalysis::empty_aliases_{};

std::string_view object_state_name(ObjectState state) noexcept {
    switch (state) {
        case ObjectState::Virtual: return "Virtual";
        case ObjectState::Materialized: return "Materialized";
        case ObjectState::Unknown: return "Unknown";
    }
    return "Unknown";
}

std::ostream& operator<<(std::ostream& os, ObjectState state) {
    return os << object_state_name(state);
}

void VirtualObject::set_field(int32_t offset, Type type, Value* val) {
    fields_[offset] = VirtualField{offset, type, val};
}

Value* VirtualObject::get_field(int32_t offset) const {
    auto it = fields_.find(offset);
    return (it != fields_.end()) ? it->second.value : nullptr;
}

Type VirtualObject::get_field_type(int32_t offset) const {
    auto it = fields_.find(offset);
    return (it != fields_.end()) ? it->second.type : Type::void_type();
}

bool VirtualObject::has_field(int32_t offset) const {
    return fields_.find(offset) != fields_.end();
}

bool VirtualObject::equals_fields(const VirtualObject& other) const noexcept {
    return fields_ == other.fields_;
}

std::string PartialEscapeStats::format_report() const {
    std::ostringstream ss;
    ss << "=== Partial Escape Analysis & Allocation Sinking Statistics ===\n"
       << "  Virtual Allocations:       " << virtual_allocations << "\n"
       << "  Materialized Allocations:  " << materialized_allocations << "\n"
       << "  Materialization Edges:     " << materialization_edges << "\n"
       << "  Sunk Allocations:          " << sunk_allocations << "\n"
       << "  Scalarized Loads:          " << scalarized_loads << "\n"
       << "  Scalarized Stores:         " << scalarized_stores << "\n";
    return ss.str();
}

PartialEscapeAnalysis::PartialEscapeAnalysis(const Function& fn, const PartialEscapeOptions& options)
    : fn_(&fn), options_(options) {
    analyze();
}

PartialEscapeAnalysis::~PartialEscapeAnalysis() = default;
PartialEscapeAnalysis::PartialEscapeAnalysis(PartialEscapeAnalysis&&) noexcept = default;
PartialEscapeAnalysis& PartialEscapeAnalysis::operator=(PartialEscapeAnalysis&&) noexcept = default;

bool PartialEscapeAnalysis::is_candidate(const Value* alloc_val) const {
    if (!alloc_val) return false;
    for (const Value* cand : candidates_) {
        if (cand == alloc_val) return true;
    }
    return false;
}

ObjectState PartialEscapeAnalysis::get_block_state(const BasicBlock* bb, const Value* alloc_val) const {
    if (!bb || !alloc_val) return ObjectState::Unknown;
    auto it_alloc = block_state_.find(alloc_val);
    if (it_alloc == block_state_.end()) return ObjectState::Unknown;
    auto it_bb = it_alloc->second.find(bb);
    if (it_bb == it_alloc->second.end()) return ObjectState::Unknown;
    return it_bb->second.state();
}

const VirtualObject* PartialEscapeAnalysis::get_virtual_object(const BasicBlock* bb, const Value* alloc_val) const {
    if (!bb || !alloc_val) return nullptr;
    auto it_alloc = block_state_.find(alloc_val);
    if (it_alloc == block_state_.end()) return nullptr;
    auto it_bb = it_alloc->second.find(bb);
    if (it_bb == it_alloc->second.end()) return nullptr;
    return &it_bb->second;
}

bool PartialEscapeAnalysis::is_virtual_in_loop(const Value* alloc_val, const LoopInfo& loop) const {
    if (!alloc_val) return false;
    if (!is_candidate(alloc_val)) return false;

    // The allocation must not escape along any block inside the loop
    for (const BasicBlock* bb : loop.blocks()) {
        if (!bb) continue;
        ObjectState st = get_block_state(bb, alloc_val);
        if (st == ObjectState::Materialized) {
            return false;
        }
    }

    // Materialization frontier edges must not be internal to the loop
    auto frontier = get_materialization_frontier(alloc_val);
    for (const auto& edge : frontier) {
        if (edge.from && edge.to) {
            if (loop.contains(edge.from) && loop.contains(edge.to)) {
                return false;
            }
        }
    }

    return true;
}

bool PartialEscapeAnalysis::escapes_in_loop(const Value* alloc_val, const LoopInfo& loop) const {
    return !is_virtual_in_loop(alloc_val, loop);
}

std::vector<CFGEdge> PartialEscapeAnalysis::get_materialization_frontier(const Value* alloc_val) const {
    if (!alloc_val) return {};
    auto it = frontiers_.find(alloc_val);
    if (it != frontiers_.end()) {
        return it->second;
    }
    return {};
}

const std::unordered_set<const Value*>& PartialEscapeAnalysis::get_aliases(const Value* alloc_val) const {
    if (!alloc_val) return empty_aliases_;
    auto it = aliases_map_.find(alloc_val);
    if (it != aliases_map_.end()) {
        return it->second;
    }
    return empty_aliases_;
}

namespace {

void for_each_branch_target_const(const Instruction* term, auto&& fn) {
    if (!term) return;
    if (term->opcode() == Opcode::br) {
        fn(term->branch_target());
    } else if (term->opcode() == Opcode::br_if) {
        fn(term->true_target());
        fn(term->false_target());
    } else if (term->opcode() == Opcode::switch_) {
        fn(term->default_target());
        for (const auto& sc : term->switch_cases()) {
            fn(sc.target);
        }
    }
}

} // namespace

void PartialEscapeAnalysis::analyze() {
    if (!fn_) return;

    for (const BasicBlock* bb : fn_->blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            if (inst->opcode() == Opcode::call && is_allocation_callee(inst->symbol())) {
                const Value* res = inst->result();
                if (res) {
                    analyze_allocation(res);
                }
            }
        }
    }
}

void PartialEscapeAnalysis::analyze_allocation(const Value* alloc_val) {
    if (!alloc_val || !alloc_val->is_instruction()) return;
    Instruction* alloc_inst = alloc_val->defining_instruction();
    if (!alloc_inst) return;
    BasicBlock* alloc_bb = alloc_inst->parent();
    if (!alloc_bb) return;

    // 1. Gather all aliases of alloc_val across block parameters
    std::unordered_set<const Value*> aliases;
    aliases.insert(alloc_val);

    bool alias_changed = true;
    while (alias_changed) {
        alias_changed = false;
        for (const BasicBlock* bb : fn_->blocks()) {
            if (!bb) continue;
            const Instruction* term = bb->terminator();
            if (!term) continue;

            for_each_branch_target_const(term, [&](const BranchTarget& bt) {
                if (!bt.block) return;
                for (size_t i = 0; i < bt.args.size(); ++i) {
                    if (aliases.count(bt.args[i]) > 0 && i < bt.block->param_count()) {
                        const Value* param = bt.block->param(i);
                        if (param && aliases.insert(param).second) {
                            alias_changed = true;
                        }
                    }
                }
            });
        }
    }
    aliases_map_[alloc_val] = aliases;

    // 2. Validate operations on aliases
    std::map<int32_t, Type> fields;
    std::unordered_set<const Instruction*> escaping_uses;

    for (const BasicBlock* bb : fn_->blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst || inst == alloc_inst) continue;

            for (size_t i = 0; i < inst->operand_count(); ++i) {
                const Value* op = inst->operand(i);
                if (aliases.count(op) == 0) continue;

                if (inst->opcode() == Opcode::load && i == 0) {
                    if (inst->offset() < 0) return;
                    auto [it, ins] = fields.insert({inst->offset(), inst->type()});
                    if (!ins && it->second != inst->type()) return;
                } else if (inst->opcode() == Opcode::store && i == 0) {
                    if (inst->offset() < 0) return;
                    const Value* stored = inst->operand(1);
                    if (aliases.count(stored) > 0) {
                        return; // Self-referential store
                    }
                    Type stored_type = stored ? stored->type() : inst->memory_type();
                    auto [it, ins] = fields.insert({inst->offset(), stored_type});
                    if (!ins && it->second != stored_type) return;
                } else {
                    escaping_uses.insert(inst);
                }
            }

            // Check branch targets: if an alias is passed to non-alias block param, it escapes
            for_each_branch_target_const(inst, [&](const BranchTarget& bt) {
                if (!bt.block) return;
                for (size_t i = 0; i < bt.args.size(); ++i) {
                    if (aliases.count(bt.args[i]) > 0) {
                        if (i >= bt.block->param_count() || aliases.count(bt.block->param(i)) == 0) {
                            escaping_uses.insert(inst);
                        }
                    }
                }
            });
        }
    }

    if (fields.size() > options_.max_fields) {
        return;
    }

    // 3. Forward dataflow analysis on CFG
    std::unordered_map<const BasicBlock*, VirtualObject> in_state;
    std::unordered_map<const BasicBlock*, VirtualObject> out_state;

    // Simulate alloc_bb
    VirtualObject alloc_bb_out(alloc_val->id(), alloc_inst);
    alloc_bb_out.set_state(ObjectState::Virtual);
    for (const auto& [off, type] : fields) {
        alloc_bb_out.set_field(off, type, nullptr);
    }

    bool started = false;
    for (Instruction* inst : *alloc_bb) {
        if (inst == alloc_inst) {
            started = true;
            continue;
        }
        if (!started) continue;

        if (inst->opcode() == Opcode::store && aliases.count(inst->operand(0)) > 0) {
            alloc_bb_out.set_field(inst->offset(), inst->operand(1)->type(), inst->operand(1));
        } else if (escaping_uses.count(inst) > 0) {
            alloc_bb_out.set_state(ObjectState::Materialized);
            break;
        }
    }
    out_state[alloc_bb] = alloc_bb_out;

    // Propagate forward to reachable blocks
    std::queue<const BasicBlock*> worklist;
    std::unordered_set<const BasicBlock*> in_queue;

    for (const BasicBlock* succ : alloc_bb->successors()) {
        if (succ && succ != alloc_bb) {
            worklist.push(succ);
            in_queue.insert(succ);
        }
    }

    std::vector<CFGEdge> candidate_frontier;

    while (!worklist.empty()) {
        const BasicBlock* bb = worklist.front();
        worklist.pop();
        in_queue.erase(bb);

        // Merge incoming states from predecessors
        VirtualObject in_vobj(alloc_val->id(), alloc_inst);
        bool has_pred = false;
        bool all_virtual = true;

        for (const BasicBlock* pred : bb->predecessors()) {
            auto it_p = out_state.find(pred);
            if (it_p == out_state.end()) continue;
            has_pred = true;
            const VirtualObject& p_out = it_p->second;

            if (p_out.state() == ObjectState::Materialized) {
                all_virtual = false;
            } else if (p_out.state() == ObjectState::Virtual) {
                for (const auto& [off, f] : p_out.fields()) {
                    if (!in_vobj.has_field(off)) {
                        in_vobj.set_field(off, f.type, f.value);
                    } else if (in_vobj.get_field(off) != f.value) {
                        in_vobj.set_field(off, f.type, nullptr); // phi merge
                    }
                }
            }
        }

        if (!has_pred) continue;

        if (!all_virtual) {
            in_vobj.set_state(ObjectState::Materialized);
        } else {
            in_vobj.set_state(ObjectState::Virtual);
        }

        in_state[bb] = in_vobj;

        // Transfer function across bb
        VirtualObject bb_out = in_vobj;
        if (bb_out.state() == ObjectState::Virtual) {
            for (const Instruction* inst : *bb) {
                if (inst->opcode() == Opcode::store && aliases.count(inst->operand(0)) > 0) {
                    bb_out.set_field(inst->offset(), inst->operand(1)->type(), inst->operand(1));
                } else if (escaping_uses.count(inst) > 0) {
                    bb_out.set_state(ObjectState::Materialized);
                    break;
                }
            }
        }

        auto it_prev = out_state.find(bb);
        bool changed = (it_prev == out_state.end() ||
                        it_prev->second.state() != bb_out.state() ||
                        !it_prev->second.equals_fields(bb_out));

        if (changed) {
            out_state[bb] = bb_out;
            for (const BasicBlock* succ : bb->successors()) {
                if (succ && in_queue.insert(succ).second) {
                    worklist.push(succ);
                }
            }
        }
    }

    // 4. Compute Minimal Materialization Frontier
    std::unordered_set<const BasicBlock*> visited;
    std::vector<CFGEdge> frontier;

    for (const auto& [bb, vout] : out_state) {
        if (vout.state() != ObjectState::Virtual) continue;

        for (const BasicBlock* succ : bb->successors()) {
            if (!succ) continue;
            auto it_sin = in_state.find(succ);
            bool succ_materialized = (it_sin != in_state.end() && it_sin->second.state() == ObjectState::Materialized);

            // Check if succ has escaping uses or transitions to materialized
            bool edge_escapes = succ_materialized;
            if (!edge_escapes) {
                for (const Instruction* inst : *succ) {
                    if (escaping_uses.count(inst) > 0) {
                        edge_escapes = true;
                        break;
                    }
                }
            }

            if (edge_escapes) {
                CFGEdge edge{bb, succ};
                if (std::find(frontier.begin(), frontier.end(), edge) == frontier.end()) {
                    frontier.push_back(edge);
                }
            }
        }
    }

    std::sort(frontier.begin(), frontier.end());

    // 5. Qualification: Allocation is a candidate if it starts virtual and has at least one virtual path
    block_state_[alloc_val] = std::move(out_state);
    if (alloc_bb_out.state() == ObjectState::Virtual) {
        candidates_.push_back(alloc_val);
        frontiers_[alloc_val] = std::move(frontier);
    }
}

void PartialEscapeAnalysis::dump(std::ostream& os) const {
    os << "Partial Escape Analysis for Function '" << (fn_ ? fn_->name() : "<null>") << "': "
       << candidates_.size() << " candidate allocations\n";
    for (const Value* cand : candidates_) {
        if (!cand) continue;
        auto frontier = get_materialization_frontier(cand);
        os << "  alloc %" << cand->id() << ": "
           << "frontier size=" << frontier.size() << " edges [";
        for (size_t i = 0; i < frontier.size(); ++i) {
            if (i > 0) os << ", ";
            os << (frontier[i].from ? frontier[i].from->name() : "<null>")
               << " -> "
               << (frontier[i].to ? frontier[i].to->name() : "<null>");
        }
        os << "]\n";
    }
}

} // namespace brass
