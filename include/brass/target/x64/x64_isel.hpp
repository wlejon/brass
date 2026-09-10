#pragma once

#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/mir/function.hpp>
#include <brass/codegen/lir.hpp>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace brass::x64 {

class X64ISel {
public:
    X64ISel();
    explicit X64ISel(const Target& target);
    X64ISel(const Target& target, const CallingConvention& cc);

    std::unique_ptr<codegen::LirFunction> lower(const Function& mir_fn);

    const Target& target() const noexcept { return target_; }
    const CallingConvention& calling_conv() const noexcept { return cc_; }

private:
    Target target_;
    CallingConvention cc_;

    codegen::LirFunction* lir_fn_ = nullptr;
    std::unordered_map<const Value*, codegen::VReg> val_to_vreg_;
    std::unordered_map<const Value*, uint32_t> use_count_;
    std::unordered_set<const Instruction*> skipped_insts_;

    struct ImmIntInfo {
        bool is_imm = false;
        int64_t val = 0;
        bool fits_i32 = false;
        const Instruction* def_inst = nullptr;
    };

    struct MemFold {
        const Value* base_val = nullptr;
        const Value* index_val = nullptr;
        Scale scale = Scale::One;
        int32_t disp = 0;
        std::vector<const Instruction*> folded_instructions;
    };

    ImmIntInfo get_imm_int_info(const Value* val) const;
    MemFold match_address(const Value* ptr, int32_t offset) const;
    MemFold match_indexed_address(const Value* base, const Value* index, Scale scale, int32_t offset) const;
    bool can_fuse_load(const Instruction* load_inst, const Instruction* user_inst) const;
    codegen::LirOperand get_load_mem_operand(const Instruction* load_inst) const;
    bool is_value_dead_after(const Function& mir_fn, const BasicBlock& bb, const Instruction* inst, const Value* val) const;
    void analyze_function(const Function& mir_fn);

    codegen::VReg get_or_alloc_vreg(const Value* val);
    codegen::VReg get_vreg(const Value* val) const;

    void lower_entry_parameters(const Function& mir_fn);
    void lower_block(const BasicBlock& bb);
    void lower_instruction(const Instruction& inst, codegen::LirBlock& lir_bb);

    void lower_binary_alu(const Instruction& inst, codegen::LirBlock& lir_bb, codegen::LirOpcode op32, codegen::LirOpcode op64, codegen::LirOpcode op_f64, codegen::LirOpcode op_f32);
    void lower_div_mod(const Instruction& inst, codegen::LirBlock& lir_bb, bool is_signed, bool is_mod);
    void lower_shift(const Instruction& inst, codegen::LirBlock& lir_bb, codegen::LirOpcode op32, codegen::LirOpcode op64);
    void lower_comparison(const Instruction& inst, codegen::LirBlock& lir_bb, Condition cond, Condition float_cond);
    void lower_select(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_call(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_invoke(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_throw(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_resume(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_landing_pad(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_branch(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_branch_if(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_switch(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_overflow_check(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_return(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_load(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_store(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_load_indexed(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_store_indexed(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_safepoint(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_guard(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_vector_instruction(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_coro(const Instruction& inst, codegen::LirBlock& lir_bb);
};

std::unique_ptr<codegen::LirFunction> lower_to_x64_lir(
    const Function& mir_fn,
    const Target& target,
    const CallingConvention& cc
);

} // namespace brass::x64
