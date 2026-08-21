#pragma once

#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/mir/function.hpp>
#include <brass/codegen/lir.hpp>
#include <memory>
#include <unordered_map>

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

    codegen::VReg get_or_alloc_vreg(const Value* val);
    codegen::VReg get_vreg(const Value* val) const;

    void lower_entry_parameters(const Function& mir_fn);
    void lower_block(const BasicBlock& bb);
    void lower_instruction(const Instruction& inst, codegen::LirBlock& lir_bb);

    void lower_binary_alu(const Instruction& inst, codegen::LirBlock& lir_bb, codegen::LirOpcode op32, codegen::LirOpcode op64, codegen::LirOpcode op_f64);
    void lower_div_mod(const Instruction& inst, codegen::LirBlock& lir_bb, bool is_signed, bool is_mod);
    void lower_shift(const Instruction& inst, codegen::LirBlock& lir_bb, codegen::LirOpcode op32, codegen::LirOpcode op64);
    void lower_comparison(const Instruction& inst, codegen::LirBlock& lir_bb, Condition cond, Condition float_cond);
    void lower_call(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_branch(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_branch_if(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_return(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_load(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_store(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_load_indexed(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_store_indexed(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_safepoint(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_guard(const Instruction& inst, codegen::LirBlock& lir_bb);
};

std::unique_ptr<codegen::LirFunction> lower_to_x64_lir(
    const Function& mir_fn,
    const Target& target,
    const CallingConvention& cc
);

} // namespace brass::x64
