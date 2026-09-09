#pragma once

#include <brass/mir/loop_vectorize.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <vector>

namespace brass {

struct VectorizableMemOp {
    Instruction* inst = nullptr;
    bool is_store = false;
    Value* base = nullptr;
    Value* index = nullptr;
    uint8_t scale = 0;
    int32_t offset = 0;
    Type elem_type;
};

struct VectorizableLoopInfo {
    bool is_vectorizable = false;
    BasicBlock* header = nullptr;
    BasicBlock* body = nullptr;
    BasicBlock* latch = nullptr;
    BasicBlock* preheader = nullptr;
    BasicBlock* exit_bb = nullptr;
    bool exit_on_false = true;

    size_t primary_iv_index = 0;
    Type iv_type = Type::i64();
    Value* init_iv = nullptr;
    Value* limit_val = nullptr;
    Opcode cmp_opcode = Opcode::slt;

    uint32_t vector_width = 4;
    Type elem_type = Type::f32();
    Type vec_type = Type::f32x4();

    bool has_reduction = false;
    size_t reduction_param_index = 0;
    Type reduction_type = Type::f32();
    Instruction* reduction_update_inst = nullptr;
    Value* reduction_init_val = nullptr;

    std::vector<VectorizableMemOp> mem_ops;
};

bool analyze_vectorizable_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    VectorizableLoopInfo& vli,
    const LoopVectorizeOptions& options
);

} // namespace brass