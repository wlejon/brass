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

bool is_pre_commutative_op(Opcode op) noexcept;
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

class PreDataflow {
public:
    PreDataflow(
        Function& fn,
        const DominatorTree& dom,
        const AliasAnalysis& aa
    );

    void analyze_expression(
        const PreExpression& expr,
        const Instruction* exemplar,
        const std::unordered_map<const Value*, const Value*>& leaders = {}
    );

    bool is_anticipated_at_entry(const BasicBlock* bb) const;
    bool is_anticipated_at_exit(const BasicBlock* bb) const;
    Value* available_at_exit(const BasicBlock* bb) const;

    bool can_evaluate_at_end(
        const BasicBlock* bb,
        const PreExpression& expr,
        const Instruction* exemplar
    ) const;

    const BlockLocalInfo& get_local_info(const BasicBlock* bb) const;

private:
    void compute_local_info(
        const PreExpression& expr,
        const Instruction* exemplar,
        const std::unordered_map<const Value*, const Value*>& leaders
    );
    void compute_anticipation();
    void compute_availability();

    Function& fn_;
    const DominatorTree& dom_;
    const AliasAnalysis& aa_;

    std::unordered_map<const BasicBlock*, BlockLocalInfo> local_info_;
    std::unordered_map<const BasicBlock*, bool> ant_in_;
    std::unordered_map<const BasicBlock*, bool> ant_out_;
    std::unordered_map<const BasicBlock*, Value*> avail_at_exit_;
    BlockLocalInfo default_local_info_;
};

} // namespace brass
