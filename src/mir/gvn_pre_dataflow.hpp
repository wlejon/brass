#pragma once

#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/memory_ssa.hpp>
#include <vector>
#include <unordered_map>
#include <string_view>
#include <cstdint>

namespace brass {

bool is_pre_commutative_op(Opcode op, Type type) noexcept;
bool is_pre_candidate_op(const Instruction* inst) noexcept;

struct PreExpression {
    Opcode opcode = Opcode::unreachable;
    Type type = Type::void_type();
    const Value* op0 = nullptr;
    const Value* op1 = nullptr;
    const Value* op2 = nullptr;
    uint64_t imm_bits = 0;
    int32_t offset = 0;
    uint8_t scale = 1;
    Type memory_type = Type::void_type();
    std::string_view symbol;

    bool is_load() const noexcept {
        return opcode == Opcode::load || opcode == Opcode::vload;
    }

    static PreExpression from_instruction(
        const Instruction* inst,
        const std::unordered_map<const Value*, const Value*>& leaders
    );

    bool operator==(const PreExpression& other) const noexcept;
};

struct PreExprHash {
    size_t operator()(const PreExpression& expr) const noexcept;
};

struct BlockLocalInfo {
    std::vector<Instruction*> evaluations;
    bool ant_loc = false;
    bool transp = true;
    bool avail_loc = false;
    Value* avail_val = nullptr;
};

// One walk of the function, shared by every expression the pass analyses in
// an iteration. Block-local information for an expression depends on three
// kinds of instruction only — its evaluations, the definitions of its
// operands, and (for a load) the instructions that may write memory — so
// with their positions indexed up front, an expression's local info costs
// its own events rather than a scan of every instruction in the function.
// Without this the pass was quadratic: a candidate count times the
// instruction count, on every iteration, and a flattened module top level
// has thousands of both.
//
// Blocks are numbered here too (their order in Function::blocks, nulls
// skipped), so the per-block state the dataflow keeps is a vector indexed by
// ordinal rather than a hash map keyed by pointer; with a thousand blocks and
// a thousand expressions that indexing is most of the pass.
struct PreFunctionIndex {
    struct Position {
        uint32_t block = 0;  // ordinal into `blocks`
        uint32_t index = 0;  // ordinal within the block
    };
    struct Writer {
        Instruction* inst;
        uint32_t index;
    };

    std::vector<BasicBlock*> blocks;
    std::unordered_map<const BasicBlock*, uint32_t> block_ordinal;
    // Where every instruction sits.
    std::unordered_map<const Instruction*, Position> positions;
    // Per block ordinal, in program order, the instructions
    // AliasAnalysis::can_clobber can answer true for: stores and calls.
    std::vector<std::vector<Writer>> memory_writers;
    // Per candidate expression, its evaluations in program order.
    std::unordered_map<PreExpression, std::vector<Instruction*>, PreExprHash> evaluations;
    // Candidate expressions in first-seen order, with the first instruction
    // evaluating each — the order and exemplars the pass always used.
    std::vector<PreExpression> candidates;
    std::unordered_map<PreExpression, const Instruction*, PreExprHash> exemplars;
    // The memory types some store in the function writes. A load whose type
    // no store writes can never become available through store-to-load
    // forwarding, which is the one way a load evaluated once can still be a
    // PRE target.
    std::vector<Type> stored_memory_types;

    void build(
        Function& fn,
        const std::unordered_map<const Value*, const Value*>& leaders,
        bool include_load_candidates
    );

    bool has_store_of_type(Type memory_type) const noexcept;
    const BasicBlock* block_of(const Instruction* inst) const noexcept;
};

class PreDataflow {
public:
    PreDataflow(
        Function& fn,
        const DominatorTree& dom,
        const AliasAnalysis& aa,
        const PreFunctionIndex& index
    );

    // Block-local information for `expr` from its events in the index, then
    // anticipation and availability over the CFG.
    void analyze_expression(const PreExpression& expr, const Instruction* exemplar);

    bool is_anticipated_at_entry(const BasicBlock* bb) const;
    bool is_anticipated_at_exit(const BasicBlock* bb) const;
    Value* available_at_exit(const BasicBlock* bb) const;

    bool can_evaluate_at_end(
        const BasicBlock* bb,
        const PreExpression& expr,
        const Instruction* exemplar
    ) const;

    // The block-local facts for a block with an event, or the default for one
    // without. Its `transp` is only meaningful for the former: ask
    // `is_transparent`, which also answers for a block a load's writers make
    // opaque without an event of its own.
    const BlockLocalInfo& get_local_info(const BasicBlock* bb) const;
    bool is_transparent(const BasicBlock* bb) const;

    // The blocks the current expression is anticipated at the entry of, in
    // block order: every block the partial-redundancy step can act at.
    std::vector<BasicBlock*> anticipated_blocks() const;
    // The blocks holding an evaluation of the current expression.
    std::vector<BasicBlock*> evaluation_blocks() const;

private:
    void compute_local_info(const PreExpression& expr, const Instruction* exemplar);
    void compute_anticipation();
    void compute_availability();

    // The ordinal of a block, or `kNoBlock` for a block the index does not
    // know (never one of the function's own).
    static constexpr uint32_t kNoBlock = ~uint32_t{0};
    uint32_t ordinal(const BasicBlock* bb) const noexcept;

    // Whether the current expression passes through block `o` unchanged:
    // the replayed facts for a block with an event, and otherwise — for a
    // load — whether none of the block's writers can clobber it, found on the
    // first question and remembered for the rest of the expression.
    bool transparent(uint32_t o) const;
    static constexpr uint8_t kOpacityUnknown = 0;
    static constexpr uint8_t kTransparent = 1;
    static constexpr uint8_t kOpaque = 2;

    Function& fn_;
    const DominatorTree& dom_;
    const AliasAnalysis& aa_;
    const PreFunctionIndex& index_;

    // Per block ordinal, and SPARSE: every vector below is reset only where
    // the previous expression set it (`touched_`, `ant_set_`, `avail_set_`),
    // and the dataflow visits only the blocks an expression's events can
    // reach. Sweeping every block per expression made the pass
    // candidates x blocks, which on a function of eight thousand blocks and
    // eighty thousand instructions — a bundled library's top level — was
    // twenty-five seconds of a one-minute compile.
    std::vector<BlockLocalInfo> local_info_;
    std::vector<uint32_t> touched_;
    std::vector<uint8_t> is_touched_;
    mutable std::vector<uint8_t> opacity_;
    mutable std::vector<uint32_t> opacity_set_;
    bool track_memory_ = false;
    const Instruction* exemplar_ = nullptr;
    // Per underlying base, the blocks holding a store off it, in block order:
    // the only stores that can MUST-alias a load off the same base.
    std::unordered_map<const Value*, std::vector<uint32_t>> stores_by_base_;
    std::vector<uint8_t> ant_in_;
    std::vector<uint32_t> ant_set_;
    std::vector<Value*> avail_at_exit_;
    std::vector<uint32_t> avail_set_;
    // Blocks from which no path reaches a block without successors. The
    // greatest fixpoint anticipates a transparent one of these vacuously, so
    // they join every expression's candidate set; there are few.
    std::vector<uint32_t> no_exit_blocks_;
    // Per block ordinal, the successor and predecessor ordinals, so the
    // fixpoint sweeps index vectors and never touch a pointer map.
    std::vector<std::vector<uint32_t>> succs_;
    std::vector<std::vector<uint32_t>> preds_;
    // `succs_` reversed: the blocks listing this one as a successor. What
    // anticipation walks backwards over, so it reads the edges it propagates
    // along rather than the predecessor lists availability uses.
    std::vector<std::vector<uint32_t>> succ_of_;
    std::vector<bool> pred_has_unknown_;  // a null predecessor: never all-same
    BlockLocalInfo default_local_info_;
};

} // namespace brass
