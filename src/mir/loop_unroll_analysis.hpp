#pragma once

#include <brass/mir/loop_unroll.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_info.hpp>
#include <vector>
#include <unordered_set>

namespace brass {

bool get_const_int(const Value* val, int64_t& out_val);
Value* make_smart_const_int(Builder& b, Type t, int64_t val);
Value* make_smart_mul(Builder& b, Type t, Value* val, int64_t mul_factor);
Value* make_smart_add(Builder& b, Type t, Value* lhs, Value* rhs);
Value* get_invariant_val(Builder& b, Type t, Value* val);

enum class ParamRole {
    BasicIV,
    DerivedIV,
    Invariant,
    ReductionAcc,
    SerialReductionAcc
};

struct ParamAnalysis {
    ParamRole role = ParamRole::Invariant;
    Value* header_param = nullptr;
    size_t param_index = 0;
    Value* ph_init_val = nullptr;
    Value* latch_next_val = nullptr;
    Value* step_val = nullptr;
    Instruction* update_inst = nullptr;
    bool is_sub = false;
    Type type = Type::void_type();
};

struct CountedLoopAnalysis {
    bool is_counted = false;
    BasicBlock* header = nullptr;
    BasicBlock* body = nullptr;
    BasicBlock* latch = nullptr;
    BasicBlock* preheader = nullptr;
    BasicBlock* exit_bb = nullptr;

    size_t primary_iv_index = 0;
    Opcode cmp_opcode = Opcode::slt;
    Value* limit_val = nullptr;
    bool exit_on_false = true; // br_if cond, body, exit

    std::vector<ParamAnalysis> params;
    std::vector<size_t> reduction_indices;
};

bool analyze_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    CountedLoopAnalysis& cla,
    const LoopUnrollOptions& options
);

} // namespace brass
