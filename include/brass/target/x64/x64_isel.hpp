#pragma once

#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/mir/function.hpp>
#include <brass/codegen/lir.hpp>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace brass {
struct OsrTarget;
}

namespace brass::x64 {

// The register a module with Module::pinned_tls_register() keeps its
// thread-local block in: callee-saved on both x64 conventions, so a C++
// helper called from generated code hands it back untouched.
inline constexpr GPR kPinnedTlsGpr = GPR::R13;

class X64ISel {
public:
    X64ISel();
    explicit X64ISel(const Target& target);
    X64ISel(const Target& target, const CallingConvention& cc);

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

    codegen::LirFunction* lir_fn_ = nullptr;
    const Function* mir_fn_ = nullptr;
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
    // The memory operand for a matched address. Its base and index values
    // must have registers: one without would silently drop out of the
    // address, so that is a hard error.
    codegen::LirOperand mem_operand(const MemFold& mf, uint8_t size) const;

    ImmIntInfo get_imm_int_info(const Value* val) const;
    // True when lower_div_mod turns division by this constant into shifts
    // and masks, so the divisor needs no register. The use analysis and the
    // lowering must agree on this, or the divisor is left without one.
    static bool divisor_folds(const ImmIntInfo& divisor, bool is_mod) noexcept;
    // Likewise for integer multiplication by a constant (lower_binary_alu).
    static bool mul_imm_folds(const ImmIntInfo& factor) noexcept;
    // A floating-point comparison fused into a branch, select or guard is
    // one ucomis and one flags condition. That is exact only for the ordered
    // relations: a < b is ucomis(b, a) + A, a <= b is ucomis(b, a) + AE, and
    // so on, all false on unordered operands. Equality needs the parity
    // flag as well, so eq / ne (and the unsigned predicates) are not fused.
    static bool fused_float_compare(Opcode cmp, Condition& cond, bool& swap_operands) noexcept;
    // Whether analyze_function may fold this comparison into its single
    // branch / select / guard user.
    static bool comparison_fusible(const Instruction& cmp) noexcept;
    // Emits the ucomis of a fused floating-point comparison and returns the
    // condition that holds exactly when the comparison is true.
    void append_fused_float_compare(const Instruction& cmp, codegen::LirBlock& lir_bb, Condition& cond);
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
    // Marks `call` as clobbering the caller-saved registers and defining the
    // return register (RAX, or XMM0 for float/vector) for a non-void `ret_t`.
    // Every Call the isel emits goes through this.
    void finish_call(codegen::LirInst& call, Type ret_t) const;
    void lower_invoke(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_throw(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_resume(const Instruction& inst, codegen::LirBlock& lir_bb);
    void append_noreturn_trap(codegen::LirBlock& lir_bb);
    void lower_landing_pad(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_branch(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_branch_if(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_switch(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_overflow_check(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_return(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_alloca(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_pinned_tls_read(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_pinned_tls_write(const Instruction& inst, codegen::LirBlock& lir_bb);
    void lower_read_sp(const Instruction& inst, codegen::LirBlock& lir_bb);
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

std::unique_ptr<codegen::LirFunction> lower_to_x64_lir(
    const Function& mir_fn,
    const Target& target,
    const CallingConvention& cc
);

} // namespace brass::x64
