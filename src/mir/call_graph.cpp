#include <brass/mir/call_graph.hpp>
#include <algorithm>
#include <stack>

namespace brass {

CallGraph::CallGraph(Module& module)
    : module_(module) {
    build();
}

CallGraphNode* CallGraph::get_node(const Function* fn) const noexcept {
    if (!fn) return nullptr;
    auto it = node_map_.find(fn);
    return (it != node_map_.end()) ? it->second.get() : nullptr;
}

CallGraphNode* CallGraph::get_node(std::string_view name) const noexcept {
    auto it = name_map_.find(name);
    return (it != name_map_.end()) ? it->second : nullptr;
}

bool CallGraph::is_recursive(const Function* fn) const noexcept {
    const CallGraphNode* node = get_node(fn);
    return node ? node->is_recursive : false;
}

bool CallGraph::is_leaf(const Function* fn) const noexcept {
    const CallGraphNode* node = get_node(fn);
    return node ? node->is_leaf : true;
}

size_t CallGraph::instruction_count(const Function* fn) const noexcept {
    const CallGraphNode* node = get_node(fn);
    return node ? node->instruction_count : 0;
}

void CallGraph::rebuild() {
    ordered_nodes_.clear();
    node_map_.clear();
    name_map_.clear();
    bottom_up_order_.clear();
    sccs_.clear();
    build();
}

void CallGraph::build() {
    // 1. Create nodes for each function in the module
    for (Function* fn : module_.functions()) {
        if (!fn) continue;
        auto node = std::make_unique<CallGraphNode>();
        node->function = fn;
        node->name = fn->name();

        size_t inst_count = 0;
        for (const BasicBlock* bb : fn->blocks()) {
            if (bb) {
                inst_count += bb->instruction_count();
            }
        }
        node->instruction_count = inst_count;

        name_map_[fn->name()] = node.get();
        ordered_nodes_.push_back(node.get());
        node_map_[fn] = std::move(node);
    }

    // 2. Discover call sites and connect edges
    for (CallGraphNode* node : ordered_nodes_) {
        Function* caller_fn = node->function;
        if (!caller_fn) continue;

        for (BasicBlock* bb : caller_fn->blocks()) {
            if (!bb) continue;

            for (Instruction* inst : *bb) {
                if (!inst) continue;

                if (inst->opcode() == Opcode::call) {
                    std::string_view callee_name = inst->symbol();
                    Function* callee_fn = module_.get_function(callee_name);

                    CallSite cs;
                    cs.instruction = inst;
                    cs.caller_block = bb;
                    cs.caller = caller_fn;
                    cs.callee = callee_fn;
                    cs.callee_name = callee_name;
                    cs.is_patchable = false;
                    cs.is_indirect = false;
                    node->call_sites.push_back(cs);

                    if (callee_fn) {
                        CallGraphNode* callee_node = get_node(callee_fn);
                        if (callee_node) {
                            if (std::find(node->callees.begin(), node->callees.end(), callee_node) == node->callees.end()) {
                                node->callees.push_back(callee_node);
                            }
                            if (std::find(callee_node->callers.begin(), callee_node->callers.end(), node) == callee_node->callers.end()) {
                                callee_node->callers.push_back(node);
                            }
                        }
                    }
                } else if (inst->opcode() == Opcode::patchable_call) {
                    std::string_view callee_name = inst->extra_symbol().empty() ? inst->symbol() : inst->extra_symbol();
                    Function* callee_fn = module_.get_function(callee_name);

                    CallSite cs;
                    cs.instruction = inst;
                    cs.caller_block = bb;
                    cs.caller = caller_fn;
                    cs.callee = callee_fn;
                    cs.callee_name = callee_name;
                    cs.is_patchable = true;
                    cs.is_indirect = false;
                    node->call_sites.push_back(cs);

                    if (callee_fn) {
                        CallGraphNode* callee_node = get_node(callee_fn);
                        if (callee_node) {
                            if (std::find(node->callees.begin(), node->callees.end(), callee_node) == node->callees.end()) {
                                node->callees.push_back(callee_node);
                            }
                            if (std::find(callee_node->callers.begin(), callee_node->callers.end(), node) == callee_node->callers.end()) {
                                callee_node->callers.push_back(node);
                            }
                        }
                    }
                } else if (inst->opcode() == Opcode::call_indirect) {
                    CallSite cs;
                    cs.instruction = inst;
                    cs.caller_block = bb;
                    cs.caller = caller_fn;
                    cs.callee = nullptr;
                    cs.callee_name = "";
                    cs.is_patchable = false;
                    cs.is_indirect = true;
                    node->call_sites.push_back(cs);
                }
            }
        }

        node->is_leaf = node->callees.empty();
    }

    // 3. Compute SCCs and bottom-up order
    compute_sccs_and_order();
}

void CallGraph::compute_sccs_and_order() {
    int current_index = 0;
    std::unordered_map<const CallGraphNode*, int> indices;
    std::unordered_map<const CallGraphNode*, int> lowlinks;
    std::unordered_map<const CallGraphNode*, bool> on_stack;
    std::vector<CallGraphNode*> stack;

    auto strongconnect = [&](auto& self, CallGraphNode* v) -> void {
        indices[v] = current_index;
        lowlinks[v] = current_index;
        ++current_index;
        stack.push_back(v);
        on_stack[v] = true;

        for (CallGraphNode* w : v->callees) {
            if (!w) continue;
            if (indices.find(w) == indices.end()) {
                self(self, w);
                lowlinks[v] = std::min(lowlinks[v], lowlinks[w]);
            } else if (on_stack[w]) {
                lowlinks[v] = std::min(lowlinks[v], indices[w]);
            }
        }

        if (lowlinks[v] == indices[v]) {
            std::vector<Function*> scc;
            while (!stack.empty()) {
                CallGraphNode* w = stack.back();
                stack.pop_back();
                on_stack[w] = false;
                if (w->function) {
                    scc.push_back(w->function);
                }
                if (w == v) break;
            }

            bool is_rec = false;
            if (scc.size() > 1) {
                is_rec = true;
            } else if (scc.size() == 1) {
                CallGraphNode* single_node = get_node(scc.front());
                if (single_node) {
                    for (CallGraphNode* c : single_node->callees) {
                        if (c == single_node) {
                            is_rec = true;
                            break;
                        }
                    }
                }
            }

            for (Function* f : scc) {
                CallGraphNode* n = get_node(f);
                if (n) {
                    n->is_recursive = is_rec;
                }
                bottom_up_order_.push_back(f);
            }

            sccs_.push_back(std::move(scc));
        }
    };

    for (CallGraphNode* node : ordered_nodes_) {
        if (indices.find(node) == indices.end()) {
            strongconnect(strongconnect, node);
        }
    }
}

} // namespace brass
