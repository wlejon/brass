#pragma once

// Private to the PTX ISel: the lowering rules behind the intrinsic table.
// Each rule is a static function `void(PtxISel&, const brass::Instruction&)`
// that reads its call operands with `isel.reg_of(inst.operand(i), "...")`
// (or `isel.const_int` for immediates), allocates scratch with
// `isel.fn_->new_*()`, writes `isel.result_reg(inst)` when the call has a
// result and emits through `isel.emit`. The name -> rule table lives in
// ptx_isel_intrinsics.cpp; the rules are spread over
//   ptx_isel_intrinsics.cpp       special registers, math, conversions
//   ptx_isel_intrinsics_warp.cpp  bar.sync, shfl, atomics, mul.wide/mad
//   ptx_isel_intrinsics_mem.cpp   shared memory, narrow global loads/stores
// The MIR-level signatures are documented in docs/ptx_backend_design.md
// ("Stage 4 implementation notes").

#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/mir/instruction.hpp>

#include <string_view>
#include <unordered_map>

namespace brass::ptx {

struct PtxISel::Intrinsics {
    using Table = std::unordered_map<std::string_view, IntrinsicLowering>;
    static const Table& table();

    // ---- special registers / math / conversions (ptx_isel_intrinsics.cpp) --
    // Invariant special registers are read once in the prologue
    // (PtxISel::special_register) and copied into the result; the copy is
    // coalesced by the cleanup pass. Volatile ones are read at the use.
    template <SpecialReg S>
    static void special(PtxISel& isel, const brass::Instruction& inst) {
        Type t = reg_class_for(S) == RegClass::B64 ? Type::u64 : Type::u32;
        if (is_invariant(S)) {
            isel.emit(Inst::make(Opcode::mov, t).dst(isel.result_reg(inst)).src(isel.special_register(S)));
        } else {
            isel.emit(Inst::make(Opcode::mov, t).dst(isel.result_reg(inst)).src(Operand::special(S)));
        }
    }
    template <Opcode Op>
    static void approx_f32(PtxISel& isel, const brass::Instruction& inst) {
        isel.emit(Inst::make(Op, Type::f32).approx()
                      .dst(isel.result_reg(inst))
                      .src(isel.operand_of(inst.operand(0), Op, 0, Type::f32, "argument 0")));
    }
    template <Opcode Op>
    static void binary_float(PtxISel& isel, const brass::Instruction& inst); // min/max, type from arg 0
    static void global_tid_x(PtxISel& isel, const brass::Instruction& inst);
    static void exp_f32(PtxISel& isel, const brass::Instruction& inst);
    static void log_f32(PtxISel& isel, const brass::Instruction& inst);
    static void sqrt_rn(PtxISel& isel, const brass::Instruction& inst);
    static void abs_float(PtxISel& isel, const brass::Instruction& inst);
    static void div_approx_f32(PtxISel& isel, const brass::Instruction& inst);
    static void cvt(PtxISel& isel, const brass::Instruction& inst, Type dst, Type src, Rounding rnd);
    static void i32_to_f32(PtxISel& isel, const brass::Instruction& inst);
    static void u32_to_f32(PtxISel& isel, const brass::Instruction& inst);
    static void u64_to_f32(PtxISel& isel, const brass::Instruction& inst);
    static void i64_to_f32(PtxISel& isel, const brass::Instruction& inst);
    static void f32_to_i32(PtxISel& isel, const brass::Instruction& inst);
    static void f32_to_u32(PtxISel& isel, const brass::Instruction& inst);
    static void f16_to_f32(PtxISel& isel, const brass::Instruction& inst);
    static void f32_to_f16(PtxISel& isel, const brass::Instruction& inst);
    static void f32_to_f64(PtxISel& isel, const brass::Instruction& inst);
    static void f64_to_f32(PtxISel& isel, const brass::Instruction& inst);

    // ---- barriers, shuffles, atomics, mul.wide/mad (ptx_isel_intrinsics_warp.cpp)
    static void bar_sync(PtxISel& isel, const brass::Instruction& inst);
    static void bar_sync_id(PtxISel& isel, const brass::Instruction& inst);
    static void bar_sync_count(PtxISel& isel, const brass::Instruction& inst);
    static void shfl_down_sync_f32(PtxISel& isel, const brass::Instruction& inst);
    template <ShflMode M>
    static void shfl(PtxISel& isel, const brass::Instruction& inst);
    template <AtomOp A, StateSpace S>
    static void atom(PtxISel& isel, const brass::Instruction& inst);
    static void mul_wide_u32(PtxISel& isel, const brass::Instruction& inst);
    static void mul_wide_s32(PtxISel& isel, const brass::Instruction& inst);
    static void mad_lo_u32(PtxISel& isel, const brass::Instruction& inst);
    static void mul_hi_u32(PtxISel& isel, const brass::Instruction& inst);

    // ---- shared memory and narrow loads/stores (ptx_isel_intrinsics_mem.cpp)
    template <Type T>
    static void shared_alloc(PtxISel& isel, const brass::Instruction& inst);
    template <Type T>
    static void shared_load(PtxISel& isel, const brass::Instruction& inst);
    template <Type T>
    static void shared_load_indexed(PtxISel& isel, const brass::Instruction& inst);
    template <Type T>
    static void shared_store(PtxISel& isel, const brass::Instruction& inst);
    template <Type T>
    static void shared_store_indexed(PtxISel& isel, const brass::Instruction& inst);
    template <Type T>
    static void load_narrow(PtxISel& isel, const brass::Instruction& inst);
    template <Type T>
    static void store_narrow(PtxISel& isel, const brass::Instruction& inst);
};

} // namespace brass::ptx
