#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/dominators.hpp>
#include <string>
#include <vector>
#include <unordered_map>

namespace brass {

enum class DemoteRefusalReason {
    None,                           // Demoted successfully
    NonIntegralStep,                // non-integral step / induction variable
    EscapingValue,                  // escaping value / phi used outside loop
    UnprovenRangeOrOverflow,        // unproven range / value >= 2^53 bound
    UnprovenDivision,               // unproven division
    CallInBody,                     // call in body
    NonIntegralConstantOrFloatOp,   // non-integral constant / float operations
    DynamicOrNonF64State,           // dynamic / non-f64 state
    Unprofitable,                   // unprofitable
    Other                           // other refusal reason
};

const char* demote_refusal_reason_string(DemoteRefusalReason reason);

struct LoopDemoteRecord {
    std::string function_name;
    std::string loop_header_block;
    bool demoted = false;
    DemoteRefusalReason refusal_reason = DemoteRefusalReason::None;
    std::string detail;
};

struct DemoteStats {
    std::string module_name;
    std::vector<LoopDemoteRecord> loop_records;
    size_t total_loops_considered = 0;
    size_t loops_demoted = 0;
    size_t loops_refused = 0;
    std::unordered_map<DemoteRefusalReason, size_t> refusal_counts;

    void add_record(LoopDemoteRecord rec);
    std::string format_report() const;
};

struct F64DemoteOptions {
    bool enable_exact_div = true;
    bool enable_entry_param_demote = true;
    DemoteStats* stats = nullptr;
};

// Analyzes and demotes exact-integer f64 SSA values, block parameters, and instructions
// within a single function to i64, inserting boundary conversions where necessary.
bool f64_demote_pass(Function& fn, const F64DemoteOptions& options = {});

// Runs the f64 demotion pass across all functions in a module.
bool f64_demote_module_pass(Module& mod, const F64DemoteOptions& options = {});

} // namespace brass
