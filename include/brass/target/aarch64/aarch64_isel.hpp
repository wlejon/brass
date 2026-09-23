#pragma once

#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/target/x64/x64_operands.hpp>
#include <brass/target/aarch64/aarch64_registers.hpp>
#include <brass/mir/function.hpp>
#include <brass/codegen/lir.hpp>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace brass {
struct OsrTarget;
}

namespace brass::aarch64 {

// The register a module with Module::pinned_tls_register() keeps its
// thread-local block in (callee-saved under AAPCS64). See x64::kPinnedTlsGpr.
inline constexpr GPR kPinnedTlsGpr = GPR::X28;

class AArch64ISel {
public:
    AArch64ISel();
    explicit AArch64ISel(const Target& target);
    AArch64ISel(const Target& target, const CallingConvention& cc);

    std::unique_ptr<codegen::LirFunction> lower(const Function& mir_fn);

    void set_osr_target(const OsrTarget* target) noexcept { osr_target_ = target; }
    const OsrTarget* osr_target() const noexcept { return osr_target_; }

    // The module coro_create callees are looked up in, when the function
    // being lowered is a clone in a scratch module (default: its parent).
    void set_callee_module(const Module* mod) noexcept { callee_module_ = mod; }

    const Target& target() const noexcept { return target_; }
    const CallingConvention& calling_conv() const noexcept { return cc_; }

private:
    Target target_;
    CallingConvention cc_;
    const OsrTarget* osr_target_ = nullptr;
    const Module* callee_module_ = nullptr;

    struct VRegPair {
        codegen::VReg lo;
        codegen::VReg hi;
    };

    codegen::LirFunction* lir_fn_ = nullptr;
    const Function* mir_fn_ = nullptr;
    std::unordered_map<const Value*, codegen::VReg> val_to_vreg_;
    std::unordered_map<const Value*, VRegPair> val_to_vreg_pair_;
    std::unordered_map<const Value*, uint32_t> use_count_;
    std::unordered_set<const Instruction*> skipped_insts_;
    // LIR the selector may delete once selection is done if nothing reads
    // what it defines (see eliminate_dead_materializations).
    std::unordered_set<const codegen::LirInst*> elidable_insts_;

    struct ImmIntInfo {
        bool is_imm = false;
        int64_t val = 0;
        bool fits_i32 = false;
        const Instruction* def_inst = nullptr;
    };

    struct MemFold {
        const Value* base_val = nullptr;
        const Value* index_val = nullptr;
        x64::Scale scale = x64::Scale::One;
        int32_t disp = 0;
        std::vector<const Instruction*> folded_instructions;
    };

    ImmIntInfo get_imm_int_info(const Value* val) const;
    MemFold match_address(const Value* ptr, int32_t offset) const;
    MemFold match_indexed_address(const Value* base, const Value* index, x64::Scale scale, int32_t offset) const;
    bool can_fuse_load(const Instruction* load_inst, const Instruction* user_inst) const;
    codegen::LirOperand get_load_mem_operand(const Instruction* load_inst) const;
    bool is_value_dead_after(const Function& mir_fn, const BasicBlock& bb, const Instruction* inst, const Value* val) const;
    void analyze_function(const Function& mir_fn);
    bool is_elidable_materialization(const Instruction& inst) const;
    void eliminate_dead_materializations();

    codegen::VReg get_or_alloc_vreg(const Value* val);
    codegen::VReg get_vreg(const Value* val) const;
    VRegPair get_or_alloc_vreg_pair(const Value* val);
    VRegPair get_vreg_pair(const Value* val) const;

    void lower_entry_parameters(const Function& mir_fn);
    void lower_block(const BasicBlock& bb);
    void lower_instruction(const Instruction& inst, codegen::LirBlock& lir_bb);

    void lower_binary_alu(const Instruction& inst, codegen::LirBlock& lir_bb, codegen::LirOpcode op32, codegen::LirOpcode op64, codegen::LirOpcode op_f64, codegen::LirOpcode op_f32);
    void lower_div_mod(const Instruction& inst, codegen::LirBlock& lir_bb, bool is_signed, bool is_mod);
    void lower_shift(const Instruction& inst, codegen::LirBlock& lir_bb, codegen::LirOpcode op32, codegen::LirOpcode op64);
    void lower_comparison(const Instruction& inst, codegen::LirBlock& lir_bb, x64::Condition cond, x64::Condition float_cond);
    void lower_select(const Instruction& inst, codegen::LirBlock& lir_bb);
    x64::Condition emit_fused_compare(const Instruction& cmp_inst, codegen::LirBlock& lir_bb);
    void lower_pinned_tls_read(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_pinned_tls_write(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_read_sp(const Instruction& inst, codegen::LirBlock& lir_bb);
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
    void lower_alloca(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_load(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_store(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_load_indexed(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_store_indexed(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_write_barrier(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_safepoint(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_guard(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_vector_instruction(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_fp_instruction(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_coro(const Instruction& inst, codegen::LirBlock& lir_bb);
};

std::unique_ptr<codegen::LirFunction> lower_to_aarch64_lir(
    const Function& mir_fn,
    const Target& target,
    const CallingConvention& cc
);

} // namespace brass::aarch64
