#pragma once

// PtxISel: MIR -> ptx::Function. The only place that decides how a MIR
// instruction becomes PTX. Suffixes come from the ptx_ir tables (type_for,
// signed_type_for, bit_type_for), comparisons produce Pred registers, block
// arguments go through a parallel-copy resolver, and builtin calls are looked
// up in a name -> lowering table (ptx_isel_intrinsics.cpp).
//
// The lowered function is not verified here; PtxTarget runs ptx::verify on
// the result and refuses to print anything that fails.
//
// See docs/ptx_backend_design.md ("PtxISel").

#include <brass/mir/function.hpp>
#include <brass/target/ptx/ptx_ir.hpp>

#include <string_view>
#include <unordered_map>
#include <vector>

namespace brass::ptx {

class PtxISel {
public:
    PtxISel() = default;

    // Throws std::runtime_error for non-void kernels, unsupported opcodes and
    // malformed instructions. The message always names the offending item.
    Function lower(const brass::Function& mir_fn);

    // Intrinsic table: MIR builtin callee name -> lowering rule. A callee that
    // is not in the table is emitted as a plain `call`.
    using IntrinsicLowering = void (*)(PtxISel&, const brass::Instruction&);
    static IntrinsicLowering find_intrinsic(std::string_view callee) noexcept;
    static bool is_intrinsic(std::string_view callee) noexcept { return find_intrinsic(callee) != nullptr; }
    static std::vector<std::string_view> intrinsic_names(); // every table key, sorted

private:
    struct Intrinsics; // the lowering rules behind the table
    friend struct Intrinsics;

    // One pending register-to-register move of a block-argument edge.
    struct Copy {
        Reg dst;
        Reg src;
        Type type; // mov suffix (bit_type_for the element type)
    };

    // ---- state ------------------------------------------------------------
    Function* fn_ = nullptr;
    Block* bb_ = nullptr;
    Block* prologue_ = nullptr;                                    // $L_params (created on demand)
    const brass::Instruction* origin_ = nullptr;
    std::unordered_map<const Value*, std::vector<Reg>> regs_;     // value -> register run (1 for scalars)
    std::unordered_map<const Value*, Reg> preds_;                  // comparison result -> pred
    std::unordered_map<const BasicBlock*, Block*> blocks_;
    std::unordered_map<const Value*, uint32_t> value_uses_;        // uses other than as a br_if/select condition
    std::unordered_map<uint8_t, Reg> special_cache_;               // invariant SpecialReg -> register read in the prologue

    // ---- setup (ptx_isel.cpp) ---------------------------------------------
    void analyze_uses(const brass::Function& mir_fn);
    void allocate_registers(const brass::Function& mir_fn);
    void lower_params(const brass::Function& mir_fn);
    void lower_block(const BasicBlock& bb);
    void lower_instruction(const brass::Instruction& inst);
    // The fall-through prologue block ahead of the MIR entry block: ld.param
    // loads and the cached special-register reads live here.
    Block* prologue_block();

    // ---- value access -----------------------------------------------------
    const std::vector<Reg>& regs_of(const Value* v, const char* what) const;
    Reg reg_of(const Value* v, const char* what) const;
    Reg result_reg(const brass::Instruction& inst) const;
    const std::vector<Reg>& result_regs(const brass::Instruction& inst) const;
    Reg pred_of(const Value* v) const; // comparison results only
    bool has_value_uses(const Value* v) const;

    // Source operand for position `src_index` of `op`, whose slot has type
    // `t`: an immediate when the value is a MIR constant (iconst/fconst), the
    // rule table (ptx_ir: allows_immediate) permits one there and it fits
    // `t`; the value's register otherwise. Float constants become 0f/0d
    // literals of the slot width, integer constants print in decimal.
    Operand operand_of(const Value* v, Opcode op, size_t src_index, Type t, const char* what);

    // Returns a Pred holding `cond != 0`; comparison results are used directly.
    Reg materialize_pred(const Value* cond);
    // PTX shift counts are 32-bit: a constant becomes a .u32 immediate, a
    // 64-bit register is narrowed with cvt.u32.u64.
    Operand shift_amount(const Value* amt);
    Operand label_of(const BasicBlock* bb) const;

    // Register holding a special register. Invariant ones (is_invariant) are
    // read once, in the prologue, and the same register is returned for every
    // later read; volatile ones (%clock, %warpid, ...) are read at the use.
    Reg special_register(SpecialReg s);

    Inst& emit(Inst inst);
    [[noreturn]] void malformed(const brass::Instruction& inst, const char* what) const;

    // ---- alu (ptx_isel_alu.cpp) -------------------------------------------
    void lower_iconst(const brass::Instruction& inst, Type t);
    void lower_fconst(const brass::Instruction& inst);
    void lower_binary(const brass::Instruction& inst, Opcode op);
    void lower_mul(const brass::Instruction& inst);
    void lower_div(const brass::Instruction& inst, bool is_unsigned);
    void lower_rem(const brass::Instruction& inst, bool is_unsigned);
    void lower_int_div_rem(const brass::Instruction& inst, Opcode op, Type t, bool is_unsigned);
    void lower_fma(const brass::Instruction& inst);
    void lower_neg(const brass::Instruction& inst);
    void lower_bitwise(const brass::Instruction& inst, Opcode op);
    void lower_not(const brass::Instruction& inst);
    void lower_shift(const brass::Instruction& inst, Opcode op, Type t);
    void lower_comparison(const brass::Instruction& inst, CmpOp cmp, bool is_unsigned);
    void lower_select(const brass::Instruction& inst);
    void lower_cvt(const brass::Instruction& inst, Type dst, Type src, Rounding rnd = Rounding::none);
    void lower_sitofp(const brass::Instruction& inst, Type src);
    void lower_fptosi(const brass::Instruction& inst, Type dst);
    void lower_bitcast(const brass::Instruction& inst);

    // ---- memory (ptx_isel_mem.cpp) ----------------------------------------
    void lower_load(const brass::Instruction& inst);
    void lower_store(const brass::Instruction& inst);
    void lower_vload(const brass::Instruction& inst);
    void lower_vstore(const brass::Instruction& inst);
    void lower_load_indexed(const brass::Instruction& inst);
    void lower_store_indexed(const brass::Instruction& inst);
    Reg indexed_address(const brass::Instruction& inst, const Value* base, const Value* index);
    // A vector value is loaded/stored as one or more {..} tuples: v4 for
    // 4-lane 32-bit, v2 for 2-lane 64-bit; 8-lane / 4-lane-64-bit types
    // are two tuples 16 bytes apart.
    void emit_vector_access(Opcode op, StateSpace space, brass::Type vt, Reg base, int32_t disp,
                            const std::vector<Reg>& lanes);

    // ---- vectors (ptx_isel_vec.cpp) ----------------------------------------
    // Vector MIR ops lower to one scalar PTX instruction per lane over the
    // contiguous register run of the vector value.
    void lower_vector_binary(const brass::Instruction& inst, Opcode op);
    void lower_vector_unary(const brass::Instruction& inst, Opcode op);
    void lower_vector_fma(const brass::Instruction& inst);
    void lower_vector_bitwise(const brass::Instruction& inst, Opcode op);
    void lower_vector_not(const brass::Instruction& inst);
    void lower_vbroadcast(const brass::Instruction& inst);
    void lower_vextract_lane(const brass::Instruction& inst);
    void lower_vinsert_lane(const brass::Instruction& inst);
    void lower_vshuffle(const brass::Instruction& inst);
    void lower_vzero(const brass::Instruction& inst);
    Type vector_elem_type(const brass::Instruction& inst, brass::Type vt) const;

    // ---- intrinsic helpers (ptx_isel_intrinsics.cpp) ----------------------
    // Compile-time integer constant behind a value (iconst_i32/iconst_i64
    // result), used for barrier ids, lane deltas, shared array sizes and
    // immediate byte offsets; const_float is the fconst (f32 or f64) form.
    bool const_int(const Value* v, int64_t* out) const;
    bool const_float(const Value* v, double* out) const;
    // `[base + disp]` for a pointer value and an optional byte-offset value:
    // constant offsets fold into the displacement, others are added into a
    // fresh 64-bit register.
    Operand address_operand(const Value* ptr, const Value* offset, const char* what);
    // `[base + index * elem_size]` for an integer index value (i32 or i64).
    Operand indexed_operand(const Value* ptr, const Value* index, uint32_t elem_size, const char* what);
    // A B32 lane operand for shfl/bar: immediate when constant, else the register.
    Operand lane_operand(const Value* v, const char* what);

    // ---- control flow (ptx_isel_control.cpp) ------------------------------
    void lower_branch(const brass::Instruction& inst);
    void lower_branch_if(const brass::Instruction& inst);
    void lower_return(const brass::Instruction& inst);
    void lower_unreachable(const brass::Instruction& inst);
    std::vector<Copy> edge_copies(const brass::Instruction& inst, const BranchTarget& target) const;
    void emit_parallel_copies(std::vector<Copy> copies, const Reg* guard);

    // ---- calls and intrinsics (ptx_isel_intrinsics*.cpp) ------------------
    void lower_call(const brass::Instruction& inst);
    void lower_plain_call(const brass::Instruction& inst);
};

} // namespace brass::ptx
