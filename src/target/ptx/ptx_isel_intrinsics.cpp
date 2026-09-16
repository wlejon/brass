// PtxISel: builtin calls. `call` instructions whose callee is in the table
// below are lowered inline; anything else becomes a plain PTX `call`.
//
// Adding an intrinsic:
//   1. add a static rule to PtxISel::Intrinsics (ptx_isel_intrinsics.hpp;
//      reuse a template where one fits),
//   2. add one row per callee name/alias to Intrinsics::table() below,
//   3. add the signature to the table in tests/unit/test_ptx_intrinsics.cpp
//      (its coverage check fails for names it does not know).
// This file holds the table and the special-register / math / conversion
// rules; ptx_isel_intrinsics_warp.cpp and ptx_isel_intrinsics_mem.cpp hold
// the rest.

#include "ptx_isel_intrinsics.hpp"

#include <algorithm>
#include <string>

namespace brass::ptx {

// ---------------------------------------------------------------------------
// Special registers
// ---------------------------------------------------------------------------

// Global thread id along x: ctaid.x * ntid.x + tid.x, from the cached
// prologue reads of the three special registers.
void PtxISel::Intrinsics::global_tid_x(PtxISel& isel, const brass::Instruction& inst) {
    Reg tid = isel.special_register(SpecialReg::tid_x);
    Reg ctaid = isel.special_register(SpecialReg::ctaid_x);
    Reg ntid = isel.special_register(SpecialReg::ntid_x);
    isel.emit(Inst::make(Opcode::mad, Type::s32).lo()
                  .dst(isel.result_reg(inst)).src(ctaid).src(ntid).src(tid));
}

// ---------------------------------------------------------------------------
// Math
// ---------------------------------------------------------------------------

// min.f32 / max.f32 (or .f64 when the arguments are f64).
template <Opcode Op>
void PtxISel::Intrinsics::binary_float(PtxISel& isel, const brass::Instruction& inst) {
    const Value* a = inst.operand(0);
    if (!a) isel.malformed(inst, "missing argument 0");
    Type t = type_for(a->type());
    if (!is_float(t)) isel.malformed(inst, "argument 0 is not a float");
    isel.emit(Inst::make(Op, t).dst(isel.result_reg(inst))
                  .src(isel.operand_of(a, Op, 0, t, "argument 0"))
                  .src(isel.operand_of(inst.operand(1), Op, 1, t, "argument 1")));
}
template void PtxISel::Intrinsics::binary_float<Opcode::min>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::binary_float<Opcode::max>(PtxISel&, const brass::Instruction&);

// e^x = 2^(x * log2(e)); log2(e) = 1.44269504f.
void PtxISel::Intrinsics::exp_f32(PtxISel& isel, const brass::Instruction& inst) {
    Reg scaled = isel.fn_->new_f32();
    isel.emit(Inst::make(Opcode::mul, Type::f32).dst(scaled)
                  .src(isel.operand_of(inst.operand(0), Opcode::mul, 0, Type::f32, "argument 0"))
                  .src(Operand::imm_f32(1.44269504f)));
    isel.emit(Inst::make(Opcode::ex2, Type::f32).approx().dst(isel.result_reg(inst)).src(scaled));
}

// ln(x) = log2(x) * ln(2); ln(2) = 0.69314718f.
void PtxISel::Intrinsics::log_f32(PtxISel& isel, const brass::Instruction& inst) {
    Reg lg = isel.fn_->new_f32();
    isel.emit(Inst::make(Opcode::lg2, Type::f32).approx().dst(lg)
                  .src(isel.reg_of(inst.operand(0), "argument 0")));
    isel.emit(Inst::make(Opcode::mul, Type::f32).dst(isel.result_reg(inst))
                  .src(lg).src(Operand::imm_f32(0.69314718f)));
}

// sqrt.rn.f32 / sqrt.rn.f64: the IEEE-rounded square root (sqrt.approx is
// the `ptx_sqrt` family).
void PtxISel::Intrinsics::sqrt_rn(PtxISel& isel, const brass::Instruction& inst) {
    const Value* a = inst.operand(0);
    if (!a) isel.malformed(inst, "missing argument 0");
    Type t = type_for(a->type());
    if (!is_float(t)) isel.malformed(inst, "argument 0 is not a float");
    isel.emit(Inst::make(Opcode::sqrt, t).rnd(Rounding::rn)
                  .dst(isel.result_reg(inst)).src(isel.reg_of(a, "argument 0")));
}

// abs.f32 / abs.f64 from the argument type.
void PtxISel::Intrinsics::abs_float(PtxISel& isel, const brass::Instruction& inst) {
    const Value* a = inst.operand(0);
    if (!a) isel.malformed(inst, "missing argument 0");
    Type t = type_for(a->type());
    if (!is_float(t)) isel.malformed(inst, "argument 0 is not a float");
    isel.emit(Inst::make(Opcode::abs, t).dst(isel.result_reg(inst)).src(isel.reg_of(a, "argument 0")));
}

// div.approx.f32 a, b
void PtxISel::Intrinsics::div_approx_f32(PtxISel& isel, const brass::Instruction& inst) {
    isel.emit(Inst::make(Opcode::div, Type::f32).approx()
                  .dst(isel.result_reg(inst))
                  .src(isel.operand_of(inst.operand(0), Opcode::div, 0, Type::f32, "argument 0"))
                  .src(isel.operand_of(inst.operand(1), Opcode::div, 1, Type::f32, "argument 1")));
}

// ---------------------------------------------------------------------------
// Conversions (cvt.<rnd>.dst.src of argument 0 into the result)
// ---------------------------------------------------------------------------

void PtxISel::Intrinsics::cvt(PtxISel& isel, const brass::Instruction& inst, Type dst, Type src, Rounding rnd) {
    isel.emit(Inst::make(Opcode::cvt, dst).from(src).rnd(rnd)
                  .dst(isel.result_reg(inst))
                  .src(isel.reg_of(inst.operand(0), "argument 0")));
}

void PtxISel::Intrinsics::i32_to_f32(PtxISel& isel, const brass::Instruction& inst) { cvt(isel, inst, Type::f32, Type::s32, Rounding::rn); }
void PtxISel::Intrinsics::u32_to_f32(PtxISel& isel, const brass::Instruction& inst) { cvt(isel, inst, Type::f32, Type::u32, Rounding::rn); }
void PtxISel::Intrinsics::u64_to_f32(PtxISel& isel, const brass::Instruction& inst) { cvt(isel, inst, Type::f32, Type::u64, Rounding::rn); }
void PtxISel::Intrinsics::i64_to_f32(PtxISel& isel, const brass::Instruction& inst) { cvt(isel, inst, Type::f32, Type::s64, Rounding::rn); }
void PtxISel::Intrinsics::f32_to_i32(PtxISel& isel, const brass::Instruction& inst) { cvt(isel, inst, Type::s32, Type::f32, Rounding::rzi); }
void PtxISel::Intrinsics::f32_to_u32(PtxISel& isel, const brass::Instruction& inst) { cvt(isel, inst, Type::u32, Type::f32, Rounding::rzi); }
void PtxISel::Intrinsics::f16_to_f32(PtxISel& isel, const brass::Instruction& inst) { cvt(isel, inst, Type::f32, Type::f16, Rounding::none); }
void PtxISel::Intrinsics::f32_to_f16(PtxISel& isel, const brass::Instruction& inst) { cvt(isel, inst, Type::f16, Type::f32, Rounding::rn); }
void PtxISel::Intrinsics::f32_to_f64(PtxISel& isel, const brass::Instruction& inst) { cvt(isel, inst, Type::f64, Type::f32, Rounding::none); }
void PtxISel::Intrinsics::f64_to_f32(PtxISel& isel, const brass::Instruction& inst) { cvt(isel, inst, Type::f32, Type::f64, Rounding::rn); }

// ---------------------------------------------------------------------------
// The table. Every alias the old string-matching emitter accepted is kept.
// ---------------------------------------------------------------------------

const PtxISel::Intrinsics::Table& PtxISel::Intrinsics::table() {
    static const Table kIntrinsics = {
        // Thread / block / grid indices                                -> i32
        { "ptx_tid_x",         &special<SpecialReg::tid_x> },
        { "ptx_tid_y",         &special<SpecialReg::tid_y> },
        { "ptx_tid_z",         &special<SpecialReg::tid_z> },
        { "ptx_ctaid_x",       &special<SpecialReg::ctaid_x> },
        { "ptx_ctaid_y",       &special<SpecialReg::ctaid_y> },
        { "ptx_ctaid_z",       &special<SpecialReg::ctaid_z> },
        { "ptx_ntid_x",        &special<SpecialReg::ntid_x> },
        { "ptx_ntid_y",        &special<SpecialReg::ntid_y> },
        { "ptx_ntid_z",        &special<SpecialReg::ntid_z> },
        { "ptx_nctaid_x",      &special<SpecialReg::nctaid_x> },
        { "ptx_nctaid_y",      &special<SpecialReg::nctaid_y> },
        { "ptx_nctaid_z",      &special<SpecialReg::nctaid_z> },
        { "ptx_laneid",        &special<SpecialReg::laneid> },
        { "ptx_lane_id",       &special<SpecialReg::laneid> },
        { "ptx_warpid",        &special<SpecialReg::warpid> },
        { "ptx_warp_id",       &special<SpecialReg::warpid> },
        { "ptx_nwarpid",       &special<SpecialReg::nwarpid> },
        { "ptx_smid",          &special<SpecialReg::smid> },
        { "ptx_nsmid",         &special<SpecialReg::nsmid> },
        { "ptx_clock",         &special<SpecialReg::clock> },
        { "ptx_global_tid_x",  &global_tid_x },
        { "ptx_global_id_x",   &global_tid_x },
        // Timers                                                       -> i64
        { "ptx_clock64",       &special<SpecialReg::clock64> },
        { "ptx_globaltimer",   &special<SpecialReg::globaltimer> },

        // Approximate math                                       (f32) -> f32
        { "rsqrtf",           &approx_f32<Opcode::rsqrt> },
        { "rsqrt",            &approx_f32<Opcode::rsqrt> },
        { "ptx_rsqrt",        &approx_f32<Opcode::rsqrt> },
        { "sqrtf",            &approx_f32<Opcode::sqrt> },
        { "sqrt",             &approx_f32<Opcode::sqrt> },
        { "ptx_sqrt",         &approx_f32<Opcode::sqrt> },
        { "sinf",             &approx_f32<Opcode::sin> },
        { "sin",              &approx_f32<Opcode::sin> },
        { "ptx_sin",          &approx_f32<Opcode::sin> },
        { "cosf",             &approx_f32<Opcode::cos> },
        { "cos",              &approx_f32<Opcode::cos> },
        { "ptx_cos",          &approx_f32<Opcode::cos> },
        { "ex2f",             &approx_f32<Opcode::ex2> },
        { "ex2",              &approx_f32<Opcode::ex2> },
        { "ptx_ex2",          &approx_f32<Opcode::ex2> },
        { "lg2f",             &approx_f32<Opcode::lg2> },
        { "ptx_lg2",          &approx_f32<Opcode::lg2> },
        { "ptx_rcp",          &approx_f32<Opcode::rcp> },
        { "ptx_rcp_approx",   &approx_f32<Opcode::rcp> },
        { "expf",             &exp_f32 },
        { "exp",              &exp_f32 },
        { "ptx_exp",          &exp_f32 },
        { "logf",             &log_f32 },
        { "log",              &log_f32 },
        { "ptx_log",          &log_f32 },
        // Exact / IEEE math                                   (f32|f64) -> same
        { "ptx_sqrt_rn",      &sqrt_rn },
        { "fabsf",            &abs_float },
        { "fabs",             &abs_float },
        { "ptx_fabs",         &abs_float },
        // Binary float math                              (f32|f64, same) -> same
        { "fminf",            &binary_float<Opcode::min> },
        { "fmin",             &binary_float<Opcode::min> },
        { "ptx_fmin",         &binary_float<Opcode::min> },
        { "fmaxf",            &binary_float<Opcode::max> },
        { "fmax",             &binary_float<Opcode::max> },
        { "ptx_fmax",         &binary_float<Opcode::max> },
        { "ptx_div_approx",   &div_approx_f32 },            // (f32, f32) -> f32

        // Conversions
        { "i32_to_f32",       &i32_to_f32 },   // (i32) -> f32   cvt.rn.f32.s32
        { "ptx_i32_to_f32",   &i32_to_f32 },
        { "ptx_u32_to_f32",   &u32_to_f32 },   // (i32) -> f32   cvt.rn.f32.u32
        { "ptx_u64_to_f32",   &u64_to_f32 },   // (i64) -> f32   cvt.rn.f32.u64
        { "ptx_i64_to_f32",   &i64_to_f32 },   // (i64) -> f32   cvt.rn.f32.s64
        { "ptx_f32_to_i32",   &f32_to_i32 },   // (f32) -> i32   cvt.rzi.s32.f32
        { "ptx_f32_to_u32",   &f32_to_u32 },   // (f32) -> i32   cvt.rzi.u32.f32
        { "ptx_f16_to_f32",   &f16_to_f32 },   // (i32 bits) -> f32   cvt.f32.f16
        { "ptx_f32_to_f16",   &f32_to_f16 },   // (f32) -> i32 bits   cvt.rn.f16.f32
        { "ptx_f32_to_f64",   &f32_to_f64 },   // (f32) -> f64   cvt.f64.f32
        { "ptx_f64_to_f32",   &f64_to_f32 },   // (f64) -> f32   cvt.rn.f32.f64

        // Barriers                                                   -> void
        { "bar.sync",            &bar_sync },        // ()          bar.sync 0
        { "ptx_sync",            &bar_sync },
        { "ptx_bar_sync",        &bar_sync_id },     // (i32 id)    bar.sync id
        { "ptx_bar_sync_count",  &bar_sync_count },  // (i32 id, i32 nthreads)

        // Warp shuffles (value, lane delta / source lane) -> same type as value
        { "ptx_shfl_down_sync_f32", &shfl_down_sync_f32 }, // (i32 mask, f32, i32)
        { "shfl_down_sync_f32",     &shfl_down_sync_f32 },
        { "ptx_shfl_down_f32",  &shfl<ShflMode::down> },
        { "ptx_shfl_up_f32",    &shfl<ShflMode::up> },
        { "ptx_shfl_bfly_f32",  &shfl<ShflMode::bfly> },
        { "ptx_shfl_xor_f32",   &shfl<ShflMode::bfly> },
        { "ptx_shfl_idx_f32",   &shfl<ShflMode::idx> },
        { "ptx_shfl_down_i32",  &shfl<ShflMode::down> },
        { "ptx_shfl_up_i32",    &shfl<ShflMode::up> },
        { "ptx_shfl_bfly_i32",  &shfl<ShflMode::bfly> },
        { "ptx_shfl_xor_i32",   &shfl<ShflMode::bfly> },
        { "ptx_shfl_idx_i32",   &shfl<ShflMode::idx> },

        // Atomics (ptr, value) -> old value
        { "ptx_atom_add_f32",        &atom<AtomOp::add, StateSpace::global> },
        { "ptx_atom_add_i32",        &atom<AtomOp::add, StateSpace::global> },
        { "ptx_atom_add_u32",        &atom<AtomOp::add, StateSpace::global> },
        { "ptx_atom_add_i64",        &atom<AtomOp::add, StateSpace::global> },
        { "ptx_atom_min_i32",        &atom<AtomOp::min, StateSpace::global> },
        { "ptx_atom_max_i32",        &atom<AtomOp::max, StateSpace::global> },
        { "ptx_atom_exch_i32",       &atom<AtomOp::exch, StateSpace::global> },
        { "ptx_atom_shared_add_f32", &atom<AtomOp::add, StateSpace::shared> },
        { "ptx_atom_shared_add_i32", &atom<AtomOp::add, StateSpace::shared> },

        // Wide / fused integer multiplies
        { "ptx_mul_wide_u32", &mul_wide_u32 },  // (i32, i32) -> i64      mul.wide.u32
        { "ptx_mul_wide_s32", &mul_wide_s32 },  // (i32, i32) -> i64      mul.wide.s32
        { "ptx_mul_hi_u32",   &mul_hi_u32 },    // (i32, i32) -> i32      mul.hi.u32
        { "ptx_mad_lo_u32",   &mad_lo_u32 },    // (i32, i32, i32) -> i32 mad.lo.u32

        // Shared memory (see ptx_isel_intrinsics_mem.cpp)
        { "ptx_shared_alloc_f32", &shared_alloc<Type::f32> },   // (const i32 count) -> ptr
        { "ptx_shared_alloc_i32", &shared_alloc<Type::u32> },
        { "ptx_shared_alloc_f64", &shared_alloc<Type::f64> },
        { "ptx_shared_alloc_i64", &shared_alloc<Type::u64> },
        { "ptx_shared_load_f32",  &shared_load<Type::f32> },    // (ptr[, i32/i64 byte offset]) -> f32
        { "ptx_shared_load_i32",  &shared_load<Type::u32> },
        { "ptx_shared_load_f64",  &shared_load<Type::f64> },
        { "ptx_shared_load_i64",  &shared_load<Type::u64> },
        { "ptx_shared_load_f32_indexed", &shared_load_indexed<Type::f32> }, // (ptr, i32/i64 index) -> f32
        { "ptx_shared_load_i32_indexed", &shared_load_indexed<Type::u32> },
        { "ptx_shared_load_f64_indexed", &shared_load_indexed<Type::f64> },
        { "ptx_shared_load_i64_indexed", &shared_load_indexed<Type::u64> },
        { "ptx_shared_store_f32", &shared_store<Type::f32> },   // (ptr, value[, byte offset]) -> void
        { "ptx_shared_store_i32", &shared_store<Type::u32> },
        { "ptx_shared_store_f64", &shared_store<Type::f64> },
        { "ptx_shared_store_i64", &shared_store<Type::u64> },
        { "ptx_shared_store_f32_indexed", &shared_store_indexed<Type::f32> }, // (ptr, index, value) -> void
        { "ptx_shared_store_i32_indexed", &shared_store_indexed<Type::u32> },
        { "ptx_shared_store_f64_indexed", &shared_store_indexed<Type::f64> },
        { "ptx_shared_store_i64_indexed", &shared_store_indexed<Type::u64> },

        // Narrow global loads/stores (ptr[, byte offset]) -> i32 / (ptr, i32 value[, byte offset])
        { "ptx_load_u8",   &load_narrow<Type::u8> },
        { "ptx_load_s8",   &load_narrow<Type::s8> },
        { "ptx_load_u16",  &load_narrow<Type::u16> },
        { "ptx_load_s16",  &load_narrow<Type::s16> },
        { "ptx_store_u8",  &store_narrow<Type::u8> },
        { "ptx_store_u16", &store_narrow<Type::u16> },
    };
    return kIntrinsics;
}

PtxISel::IntrinsicLowering PtxISel::find_intrinsic(std::string_view callee) noexcept {
    const auto& table = Intrinsics::table();
    auto it = table.find(callee);
    return it == table.end() ? nullptr : it->second;
}

std::vector<std::string_view> PtxISel::intrinsic_names() {
    const auto& table = Intrinsics::table();
    std::vector<std::string_view> names;
    names.reserve(table.size());
    for (const auto& entry : table) names.push_back(entry.first);
    std::sort(names.begin(), names.end());
    return names;
}

// ---------------------------------------------------------------------------
// call
// ---------------------------------------------------------------------------

void PtxISel::lower_call(const brass::Instruction& inst) {
    if (IntrinsicLowering rule = find_intrinsic(inst.symbol())) {
        rule(*this, inst);
        return;
    }
    lower_plain_call(inst);
}

// call (%ret), callee, (args);   or   call callee, (args);
void PtxISel::lower_plain_call(const brass::Instruction& inst) {
    Inst call = Inst::make(Opcode::call);
    if (!inst.type().is_void() && inst.result()) call.dst(result_reg(inst));
    call.src(Operand::symbol(std::string(inst.symbol())));
    for (size_t i = 0; i < inst.operand_count(); ++i) {
        call.src(reg_of(inst.operand(i), "call argument"));
    }
    emit(std::move(call));
}

} // namespace brass::ptx
