#include <brass/mir/escape_analysis.hpp>
#include "escape_graph.hpp"
#include <ostream>

namespace brass {

std::string_view escape_state_name(EscapeState state) noexcept {
    switch (state) {
        case EscapeState::NoEscape: return "NoEscape";
        case EscapeState::ArgEscape: return "ArgEscape";
        case EscapeState::GlobalEscape: return "GlobalEscape";
    }
    return "Unknown";
}

std::ostream& operator<<(std::ostream& os, EscapeState state) {
    return os << escape_state_name(state);
}

bool is_allocation_callee(std::string_view symbol) noexcept {
    return symbol == "brass_gc_alloc" ||
           symbol == "host_gc_alloc" ||
           symbol == "alloc_obj" ||
           symbol == "make_array" ||
           symbol == "create_array" ||
           symbol == "alloc_array" ||
           symbol == "malloc" ||
           symbol == "calloc" ||
           symbol.starts_with("alloc_") ||
           symbol.ends_with("_alloc");
}

EscapeAnalysis::EscapeAnalysis(const Function& fn)
    : EscapeAnalysis(fn, EscapeAnalysisOptions{}) {}

EscapeAnalysis::EscapeAnalysis(const Function& fn, const EscapeAnalysisOptions& options)
    : fn_(&fn), options_(options) {
    run();
}

EscapeAnalysis::~EscapeAnalysis() = default;
EscapeAnalysis::EscapeAnalysis(EscapeAnalysis&&) noexcept = default;
EscapeAnalysis& EscapeAnalysis::operator=(EscapeAnalysis&&) noexcept = default;

bool EscapeAnalysis::does_escape(const Value* val) const {
    return get_escape_state(val) != EscapeState::NoEscape;
}

EscapeState EscapeAnalysis::get_escape_state(const Value* val) const {
    if (!val) return EscapeState::GlobalEscape;
    auto it = escape_states_.find(val);
    if (it != escape_states_.end()) {
        return it->second;
    }
    if (graph_) {
        return graph_->get_escape_state(val);
    }
    return EscapeState::GlobalEscape;
}

bool EscapeAnalysis::is_allocation(const Value* val) const {
    if (!val || !val->is_instruction()) return false;
    const Instruction* inst = val->defining_instruction();
    if (!inst) return false;
    return inst->opcode() == Opcode::call && is_allocation_callee(inst->symbol());
}

namespace {

struct StoreOp {
    const Value* base = nullptr;
    int32_t offset = 0;
    const Value* val = nullptr;
};

struct LoadOp {
    const Value* result = nullptr;
    const Value* base = nullptr;
    int32_t offset = 0;
};

bool is_pointer_like(const Value* val) {
    if (!val) return false;
    Type t = val->type();
    return t.is_pointer_or_gcref();
}

} // namespace

void EscapeAnalysis::run() {
    if (!fn_) return;
    graph_ = std::make_unique<ConnectionGraph>();

    std::vector<StoreOp> stores;
    std::vector<LoadOp> loads;

    // 1. Process entry block parameters (function arguments)
    const BasicBlock* entry = fn_->entry_block();
    if (entry) {
        for (const Value* param : entry->params()) {
            if (param && is_pointer_like(param)) {
                CGNode* phantom = graph_->create_phantom_object(EscapeState::ArgEscape);
                CGNode* ref = graph_->get_or_create_ref_node(param);
                graph_->add_points_to(ref, phantom);
                graph_->mark_escape_state(ref, EscapeState::ArgEscape);
            }
        }
    }

    // 2. Scan all blocks and instructions
    for (const BasicBlock* bb : fn_->blocks()) {
        if (!bb) continue;

        // Block parameters for non-entry blocks
        if (bb != entry) {
            for (const Value* param : bb->params()) {
                if (param && is_pointer_like(param)) {
                    graph_->get_or_create_ref_node(param);
                }
            }
        }

        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            Opcode op = inst->opcode();

            switch (op) {
                case Opcode::call: {
                    std::string_view callee = inst->symbol();
                    if (is_allocation_callee(callee)) {
                        const Value* res = inst->result();
                        if (res) {
                            CGNode* obj = graph_->create_object_node(res, EscapeState::NoEscape);
                            CGNode* ref = graph_->get_or_create_ref_node(res);
                            graph_->add_points_to(ref, obj);
                            allocations_.push_back(res);
                        }
                    } else {
                        bool is_arg_escape = options_.arg_escape_callees.count(std::string(callee)) > 0 ||
                                             callee.starts_with("arg_escape") ||
                                             !options_.treat_unhandled_calls_as_global;
                        EscapeState call_escape = is_arg_escape ? EscapeState::ArgEscape : EscapeState::GlobalEscape;
                        for (size_t i = 0; i < inst->operand_count(); ++i) {
                            const Value* arg = inst->operand(i);
                            if (arg && is_pointer_like(arg)) {
                                CGNode* ref = graph_->get_or_create_ref_node(arg);
                                graph_->mark_escape_state(ref, call_escape);
                            }
                        }
                    }
                    break;
                }

                case Opcode::call_indirect: {
                    for (size_t i = 0; i < inst->operand_count(); ++i) {
                        const Value* op_val = inst->operand(i);
                        if (op_val && is_pointer_like(op_val)) {
                            CGNode* ref = graph_->get_or_create_ref_node(op_val);
                            graph_->mark_escape_state(ref, EscapeState::GlobalEscape);
                        }
                    }
                    break;
                }

                case Opcode::ret: {
                    if (inst->operand_count() > 0) {
                        const Value* ret_val = inst->operand(0);
                        if (ret_val && is_pointer_like(ret_val)) {
                            CGNode* ref = graph_->get_or_create_ref_node(ret_val);
                            graph_->mark_escape_state(ref, EscapeState::GlobalEscape);
                        }
                    }
                    break;
                }

                case Opcode::store: {
                    const Value* base = inst->operand(0);
                    const Value* val = inst->operand(1);
                    stores.push_back({base, inst->offset(), val});
                    break;
                }

                case Opcode::load: {
                    const Value* base = inst->operand(0);
                    const Value* res = inst->result();
                    if (res && is_pointer_like(res)) {
                        loads.push_back({res, base, inst->offset()});
                    }
                    break;
                }

                case Opcode::select: {
                    const Value* res = inst->result();
                    if (res && is_pointer_like(res)) {
                        CGNode* res_ref = graph_->get_or_create_ref_node(res);
                        const Value* true_val = inst->operand(1);
                        const Value* false_val = inst->operand(2);
                        if (true_val && is_pointer_like(true_val)) {
                            CGNode* t_ref = graph_->get_or_create_ref_node(true_val);
                            graph_->add_deferred(t_ref, res_ref);
                        }
                        if (false_val && is_pointer_like(false_val)) {
                            CGNode* f_ref = graph_->get_or_create_ref_node(false_val);
                            graph_->add_deferred(f_ref, res_ref);
                        }
                    }
                    break;
                }

                case Opcode::br: {
                    const BranchTarget& target = inst->branch_target();
                    if (target.block) {
                        for (size_t i = 0; i < target.args.size(); ++i) {
                            const Value* arg = target.args[i];
                            const Value* param = target.block->param(i);
                            if (arg && param && is_pointer_like(arg) && is_pointer_like(param)) {
                                CGNode* arg_ref = graph_->get_or_create_ref_node(arg);
                                CGNode* param_ref = graph_->get_or_create_ref_node(param);
                                graph_->add_deferred(arg_ref, param_ref);
                            }
                        }
                    }
                    break;
                }

                case Opcode::br_if: {
                    auto handle_target = [&](const BranchTarget& target) {
                        if (target.block) {
                            for (size_t i = 0; i < target.args.size(); ++i) {
                                const Value* arg = target.args[i];
                                const Value* param = target.block->param(i);
                                if (arg && param && is_pointer_like(arg) && is_pointer_like(param)) {
                                    CGNode* arg_ref = graph_->get_or_create_ref_node(arg);
                                    CGNode* param_ref = graph_->get_or_create_ref_node(param);
                                    graph_->add_deferred(arg_ref, param_ref);
                                }
                            }
                        }
                    };
                    handle_target(inst->true_target());
                    handle_target(inst->false_target());
                    break;
                }

                case Opcode::switch_: {
                    auto handle_target = [&](const BranchTarget& target) {
                        if (target.block) {
                            for (size_t i = 0; i < target.args.size(); ++i) {
                                const Value* arg = target.args[i];
                                const Value* param = target.block->param(i);
                                if (arg && param && is_pointer_like(arg) && is_pointer_like(param)) {
                                    CGNode* arg_ref = graph_->get_or_create_ref_node(arg);
                                    CGNode* param_ref = graph_->get_or_create_ref_node(param);
                                    graph_->add_deferred(arg_ref, param_ref);
                                }
                            }
                        }
                    };
                    handle_target(inst->default_target());
                    for (const auto& sc : inst->switch_cases()) {
                        handle_target(sc.target);
                    }
                    break;
                }

                case Opcode::load_indexed:
                case Opcode::store_indexed: {
                    // Non-constant / indexed access forces base to GlobalEscape
                    const Value* base = inst->operand(0);
                    if (base && is_pointer_like(base)) {
                        CGNode* ref = graph_->get_or_create_ref_node(base);
                        graph_->mark_escape_state(ref, EscapeState::GlobalEscape);
                    }
                    if (op == Opcode::store_indexed) {
                        const Value* stored = inst->operand(2);
                        if (stored && is_pointer_like(stored)) {
                            CGNode* ref = graph_->get_or_create_ref_node(stored);
                            graph_->mark_escape_state(ref, EscapeState::GlobalEscape);
                        }
                    }
                    break;
                }

                default: {
                    // Check if pointer is used in an unhandled instruction
                    for (size_t i = 0; i < inst->operand_count(); ++i) {
                        const Value* op_val = inst->operand(i);
                        if (op_val && is_pointer_like(op_val)) {
                            CGNode* ref = graph_->get_or_create_ref_node(op_val);
                            graph_->mark_escape_state(ref, EscapeState::GlobalEscape);
                        }
                    }
                    break;
                }
            }
        }
    }

    // 3. Iterative points-to and field-sensitive connection
    bool changed = true;
    while (changed) {
        changed = false;
        graph_->propagate();

        // Connect stores
        for (const auto& st : stores) {
            CGNode* base_ref = graph_->find_ref_node(st.base);
            CGNode* val_ref = is_pointer_like(st.val) ? graph_->find_ref_node(st.val) : nullptr;

            if (!base_ref || base_ref->points_to.empty()) {
                // Storing into an unknown or external pointer escapes the value globally
                if (val_ref) {
                    graph_->mark_escape_state(val_ref, EscapeState::GlobalEscape);
                }
                continue;
            }

            for (CGNode* obj : base_ref->points_to) {
                if (!obj) continue;
                CGNode* field = graph_->get_or_create_field_node(obj, st.offset);
                if (val_ref) {
                    for (CGNode* target_obj : val_ref->points_to) {
                        if (target_obj && field->points_to.count(target_obj) == 0) {
                            graph_->add_points_to(field, target_obj);
                            changed = true;
                        }
                    }
                    if (obj->state > EscapeState::NoEscape) {
                        graph_->mark_escape_state(val_ref, obj->state);
                    }
                }
            }
        }

        // Connect loads
        for (const auto& ld : loads) {
            CGNode* base_ref = graph_->find_ref_node(ld.base);
            CGNode* res_ref = graph_->get_or_create_ref_node(ld.result);
            if (!base_ref || !res_ref) continue;

            for (CGNode* obj : base_ref->points_to) {
                if (!obj) continue;
                CGNode* field = graph_->get_or_create_field_node(obj, ld.offset);
                for (CGNode* target_obj : field->points_to) {
                    if (target_obj && res_ref->points_to.count(target_obj) == 0) {
                        graph_->add_points_to(res_ref, target_obj);
                        changed = true;
                    }
                }
                if (field->state > EscapeState::NoEscape) {
                    graph_->mark_escape_state(res_ref, field->state);
                }
            }
        }
    }

    // Final propagation
    graph_->propagate();

    // 4. Record allocation escape states
    for (const Value* alloc_val : allocations_) {
        EscapeState st = graph_->get_escape_state(alloc_val);
        escape_states_[alloc_val] = st;
        if (st == EscapeState::NoEscape) {
            non_escaping_allocations_.push_back(alloc_val);
        }
    }
}

} // namespace brass
