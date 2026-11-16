#include <brass/mir/call_graph.hpp>
#include <brass/core/parallel_for.hpp>
#include <algorithm>
#include <stack>
#include <unordered_set>

namespace brass {

CallGraph::CallGraph(Module& module, size_t threads)
    : module_(module), threads_(threads) {
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
        name_map_[fn->name()] = node.get();
        ordered_nodes_.push_back(node.get());
        node_map_[fn] = std::move(node);
    }

    // 2. Each function's size, call sites and callees (each once, in the
    // order first called), functions in parallel: a scan writes only its own
    // node and reads the maps step 1 filled.
    std::vector<size_t> weights(ordered_nodes_.size());
    size_t total = 0;
    for (size_t i = 0; i < ordered_nodes_.size(); ++i) {
        weights[i] = ordered_nodes_[i]->function->block_count();
        total += weights[i];
    }
    const size_t n = threads_for_work(threads_, ordered_nodes_.size(), total, kMinBlocksPerThread);
    parallel_for_each(heaviest_first(weights), n, [this](size_t i) { scan(*ordered_nodes_[i]); });

    // 3. Callers, each once per callee, in module order.
    for (CallGraphNode* node : ordered_nodes_) {
        for (CallGraphNode* callee_node : node->callees) {
            if (callee_node->callers.empty() || callee_node->callers.back() != node) {
                callee_node->callers.push_back(node);
            }
        }
    }

    // 4. Compute SCCs and bottom-up order
    compute_sccs_and_order();
}

void CallGraph::scan(CallGraphNode& node) {
    Function* caller_fn = node.function;
    std::unordered_set<const CallGraphNode*> linked;
    auto add_site = [&](Instruction* inst, BasicBlock* bb, Function* callee_fn, std::string_view callee_name,
                        bool patchable, bool indirect) {
        CallSite cs;
        cs.instruction = inst;
        cs.caller_block = bb;
        cs.caller = caller_fn;
        cs.callee = callee_fn;
        cs.callee_name = callee_name;
        cs.is_patchable = patchable;
        cs.is_indirect = indirect;
        node.call_sites.push_back(cs);
        if (!callee_fn) return;
        CallGraphNode* callee_node = get_node(callee_fn);
        if (callee_node && linked.insert(callee_node).second) node.callees.push_back(callee_node);
    };

    size_t inst_count = 0;
    for (BasicBlock* bb : caller_fn->blocks()) {
        if (!bb) continue;
        inst_count += bb->instruction_count();
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            if (inst->opcode() == Opcode::call) {
                std::string_view callee_name = inst->symbol();
                auto it = name_map_.find(callee_name);
                Function* callee_fn = (it != name_map_.end()) ? it->second->function : nullptr;
                add_site(inst, bb, callee_fn, callee_name, false, false);
            } else if (inst->opcode() == Opcode::patchable_call) {
                std::string_view callee_name = inst->extra_symbol().empty() ? inst->symbol() : inst->extra_symbol();
                add_site(inst, bb, module_.get_function(callee_name), callee_name, true, false);
            } else if (inst->opcode() == Opcode::call_indirect) {
                add_site(inst, bb, nullptr, "", false, true);
            }
        }
    }
    node.instruction_count = inst_count;
    node.is_leaf = node.call_sites.empty();
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
