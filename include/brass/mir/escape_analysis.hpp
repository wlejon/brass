#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/instruction.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <iosfwd>

namespace brass {

enum class EscapeState : uint8_t {
    NoEscape = 0,     // Allocation never escapes the local function activation
    ArgEscape = 1,    // Passed to callees as argument but not stored globally or returned
    GlobalEscape = 2  // Escapes via return, store into escaping object, or unhandled call
};

std::string_view escape_state_name(EscapeState state) noexcept;
std::ostream& operator<<(std::ostream& os, EscapeState state);

struct EscapeAnalysisOptions {
    bool track_fields = true;
    bool treat_unhandled_calls_as_global = true;
    std::unordered_set<std::string> arg_escape_callees;
};

class ConnectionGraph;

class EscapeAnalysis {
public:
    explicit EscapeAnalysis(const Function& fn);
    EscapeAnalysis(const Function& fn, const EscapeAnalysisOptions& options);
    ~EscapeAnalysis();

    EscapeAnalysis(const EscapeAnalysis&) = delete;
    EscapeAnalysis& operator=(const EscapeAnalysis&) = delete;
    EscapeAnalysis(EscapeAnalysis&&) noexcept;
    EscapeAnalysis& operator=(EscapeAnalysis&&) noexcept;

    // Analysis Query API
    bool does_escape(const Value* val) const;
    EscapeState get_escape_state(const Value* val) const;

    // Allocation queries
    bool is_allocation(const Value* val) const;
    const std::vector<const Value*>& allocations() const noexcept { return allocations_; }
    const std::vector<const Value*>& non_escaping_allocations() const noexcept { return non_escaping_allocations_; }

    const Function& function() const noexcept { return *fn_; }
    const ConnectionGraph* connection_graph() const noexcept { return graph_.get(); }

private:
    void run();

    const Function* fn_ = nullptr;
    EscapeAnalysisOptions options_;
    std::unique_ptr<ConnectionGraph> graph_;
    std::vector<const Value*> allocations_;
    std::vector<const Value*> non_escaping_allocations_;
    std::unordered_map<const Value*, EscapeState> escape_states_;
};

// True for the allocators whose contract brass itself knows: each call
// returns memory no other live pointer refers to and writes nothing else the
// caller can observe. Alias analysis, memory SSA and escape analysis build
// on that promise, so it is never inferred from a symbol's spelling.
bool is_allocation_callee(std::string_view symbol) noexcept;

// True when `inst` is a direct call to such an allocator, or to a function
// its module declares as one (Module::add_allocation_function).
bool is_allocation_call(const Instruction* inst) noexcept;

} // namespace brass
