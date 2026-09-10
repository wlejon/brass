#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <vector>
#include <string>
#include <string_view>
#include <memory>
#include <cstdint>

namespace brass {

struct SubscriptExpr {
    Value* base = nullptr;
    Value* iv = nullptr;
    int64_t stride = 1;
    int64_t offset = 0;
    uint8_t scale = 1;
    bool is_valid = false;
};

struct ParallelMemAccess {
    Instruction* inst = nullptr;
    bool is_store = false;
    Type elem_type;
    SubscriptExpr expr;
};

enum class DependenceKind : uint8_t {
    None,
    RAW, // Flow dependence: Read-After-Write
    WAR, // Anti dependence: Write-After-Read
    WAW  // Output dependence: Write-After-Write
};

enum class DependenceDir : uint8_t {
    Equal,    // distance == 0 (loop-independent)
    Forward,  // distance > 0 (loop-carried forward)
    Backward, // distance < 0 (loop-carried backward)
    Any       // variable or indeterminate distance
};

struct ParallelDependence {
    DependenceKind kind = DependenceKind::None;
    DependenceDir dir = DependenceDir::Equal;
    int64_t distance = 0;
    bool has_distance = false;
    bool is_loop_carried = false;
    const Instruction* src = nullptr;
    const Instruction* dst = nullptr;
};

enum class LoopParallelKind : uint8_t {
    Sequential,
    DOALL,
    Reduction
};

struct ParallelLoopInfo {
    LoopParallelKind kind = LoopParallelKind::Sequential;
    LoopInfo* loop = nullptr;
    BasicBlock* header = nullptr;
    BasicBlock* preheader = nullptr;
    BasicBlock* body = nullptr;
    BasicBlock* latch = nullptr;
    BasicBlock* exit_bb = nullptr;
    bool exit_on_false = true;

    // Primary Induction Variable
    size_t iv_param_index = 0;
    Value* iv_param = nullptr;
    Type iv_type = Type::i64();
    Value* init_iv = nullptr;
    Value* limit_val = nullptr;
    int64_t step = 1;
    Opcode cmp_opcode = Opcode::slt;

    // Cost model estimates
    bool has_const_trip_count = false;
    uint64_t const_trip_count = 0;
    uint32_t instruction_count = 0;

    // Reduction info
    bool has_reduction = false;
    runtime::ReductionKind reduction_kind = runtime::ReductionKind::None;
    size_t reduction_param_index = 0;
    Value* reduction_param = nullptr;
    Value* reduction_init_val = nullptr;
    Instruction* reduction_op_inst = nullptr;
    Type reduction_type = Type::i64();

    std::vector<ParallelMemAccess> accesses;
    std::vector<ParallelDependence> dependences;
    std::string rejection_reason;

    bool is_parallelizable() const noexcept {
        return kind == LoopParallelKind::DOALL || kind == LoopParallelKind::Reduction;
    }
};

struct ParallelLoopStats {
    uint32_t loops_analyzed = 0;
    uint32_t doall_loops_found = 0;
    uint32_t reduction_loops_found = 0;
    uint32_t parallel_loops_transformed = 0;
    uint32_t loops_rejected_carried_dependence = 0;
    uint32_t loops_rejected_uncontrolled_effects = 0;
    uint32_t loops_rejected_cost = 0;

    std::string format_report() const;
};

struct ParallelLoopOptions {
    uint64_t parallel_threshold = 1000;
    uint32_t parallel_workers = 0;
    bool allow_fp_reassociation = false;
    ParallelLoopStats* stats = nullptr;
};

// Subscript expression parser
bool parse_subscript_expression(
    const Instruction* inst,
    const Value* iv_val,
    SubscriptExpr& out_expr
);

// Dependence analysis between two memory accesses
ParallelDependence check_subscript_dependence(
    const ParallelMemAccess& a1,
    const ParallelMemAccess& a2
);

// Analyze loop for parallelism
bool analyze_parallel_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    ParallelLoopInfo& pli,
    const ParallelLoopOptions& options = {}
);

// Auto-parallelization transformation pass
bool auto_parallelize_function(
    Function& fn,
    const DominatorTree& dom,
    const ParallelLoopOptions& options = {}
);

} // namespace brass