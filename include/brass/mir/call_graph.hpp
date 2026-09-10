#pragma once

#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <memory>

namespace brass {

struct CallSite {
    Instruction* instruction = nullptr;
    BasicBlock* caller_block = nullptr;
    Function* caller = nullptr;
    Function* callee = nullptr;         // nullptr if external or indirect
    std::string_view callee_name;
    bool is_patchable = false;
    bool is_indirect = false;
};

struct CallGraphNode {
    Function* function = nullptr;
    std::string_view name;
    size_t instruction_count = 0;
    bool is_leaf = false;               // True if no direct calls to functions defined in module
    bool is_recursive = false;          // True if part of recursive SCC or self-loop

    std::vector<CallSite> call_sites;
    std::vector<CallGraphNode*> callees;
    std::vector<CallGraphNode*> callers;
};

class CallGraph {
public:
    explicit CallGraph(Module& module);
    ~CallGraph() = default;

    CallGraph(const CallGraph&) = delete;
    CallGraph& operator=(const CallGraph&) = delete;
    CallGraph(CallGraph&&) noexcept = default;
    CallGraph& operator=(CallGraph&&) noexcept = delete;

    Module& module() const noexcept { return module_; }

    const std::vector<CallGraphNode*>& nodes() const noexcept { return ordered_nodes_; }
    CallGraphNode* get_node(const Function* fn) const noexcept;
    CallGraphNode* get_node(std::string_view name) const noexcept;

    // Bottom-up (leaf-first) ordering of functions in module
    const std::vector<Function*>& bottom_up_order() const noexcept { return bottom_up_order_; }

    // Strongly Connected Components (cycles). Each component is a list of Functions.
    const std::vector<std::vector<Function*>>& sccs() const noexcept { return sccs_; }

    bool is_recursive(const Function* fn) const noexcept;
    bool is_leaf(const Function* fn) const noexcept;
    size_t instruction_count(const Function* fn) const noexcept;

    void rebuild();

private:
    Module& module_;
    std::vector<CallGraphNode*> ordered_nodes_;
    std::unordered_map<const Function*, std::unique_ptr<CallGraphNode>> node_map_;
    std::unordered_map<std::string_view, CallGraphNode*> name_map_;
    std::vector<Function*> bottom_up_order_;
    std::vector<std::vector<Function*>> sccs_;

    void build();
    void compute_sccs_and_order();
};

} // namespace brass
