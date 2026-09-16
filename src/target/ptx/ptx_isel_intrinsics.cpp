// PtxISel: builtin calls. `call` instructions whose callee is in the table
// below are lowered inline; anything else becomes a plain PTX `call`.
//
// Adding an intrinsic (Stage 4: bar.sync ids, shared memory, rcp.approx,
// div.approx, f16 -> f32, v4 u32 loads, mad.lo, ...):
//   1. add a static rule to PtxISel::Intrinsics (or reuse a template below),
//   2. add one row per callee name/alias to Intrinsics::table().
// A rule receives the ISel (for registers and emit) and the MIR call; it must
// write the call's result register when the call has a result.

#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/mir/instruction.hpp>

#include <algorithm>
#include <string>
#include <unordered_map>

namespace brass::ptx {

struct PtxISel::Intrinsics {
    using Table = std::unordered_map<std::string_view, IntrinsicLowering>;
    static const Table& table();

    // mov.u32 %r, %special
    template <SpecialReg S>
    static void special(PtxISel& isel, const brass::Instruction& inst) {
        isel.emit(Inst::make(Opcode::mov, Type::u32).dst(isel.result_reg(inst)).src(Operand::special(S)));
    }

    // <op>.approx.f32 %f, %f   (rsqrt, sqrt, sin, cos, ex2)
    template <Opcode Op>
    static void approx_f32(PtxISel& isel, const brass::Instruction& inst) {
        isel.emit(Inst::make(Op, Type::f32).approx()
                      .dst(isel.result_reg(inst))
                      .src(isel.reg_of(inst.operand(0), "argument 0")));
    }

    // Global thread id along x: ctaid.x * ntid.x + tid.x.
    static void global_tid_x(PtxISel& isel, const brass::Instruction& inst) {
        Reg tid = isel.fn_->new_b32();
        Reg ctaid = isel.fn_->new_b32();
        Reg ntid = isel.fn_->new_b32();
        isel.emit(Inst::make(Opcode::mov, Type::u32).dst(tid).src(Operand::special(SpecialReg::tid_x)));
        isel.emit(Inst::make(Opcode::mov, Type::u32).dst(ctaid).src(Operand::special(SpecialReg::ctaid_x)));
        isel.emit(Inst::make(Opcode::mov, Type::u32).dst(ntid).src(Operand::special(SpecialReg::ntid_x)));
        isel.emit(Inst::make(Opcode::mad, Type::s32).lo()
                      .dst(isel.result_reg(inst)).src(ctaid).src(ntid).src(tid));
    }

    // e^x = 2^(x * log2(e)); log2(e) = 1.44269504f = 0f3FB8AA3B.
    static void exp_f32(PtxISel& isel, const brass::Instruction& inst) {
        Reg scaled = isel.fn_->new_f32();
        isel.emit(Inst::make(Opcode::mul, Type::f32).dst(scaled)
                      .src(isel.reg_of(inst.operand(0), "argument 0"))
                      .src(Operand::imm_f32(1.44269504f)));
        isel.emit(Inst::make(Opcode::ex2, Type::f32).approx().dst(isel.result_reg(inst)).src(scaled));
    }

    // cvt.rn.f32.s32
    static void i32_to_f32(PtxISel& isel, const brass::Instruction& inst) {
        isel.emit(Inst::make(Opcode::cvt, Type::f32).from(Type::s32).rnd(Rounding::rn)
                      .dst(isel.result_reg(inst))
                      .src(isel.reg_of(inst.operand(0), "argument 0")));
    }

    // bar.sync 0
    static void bar_sync(PtxISel& isel, const brass::Instruction&) {
        isel.emit(Inst::make(Opcode::bar).sync().src(Operand::imm(0)));
    }

    // shfl.sync.down.b32 %f, value, delta, 0x1f, mask   for (mask, value, delta)
    static void shfl_down_sync_f32(PtxISel& isel, const brass::Instruction& inst) {
        isel.emit(Inst::make(Opcode::shfl, Type::b32).sync().shfl(ShflMode::down)
                      .dst(isel.result_reg(inst))
                      .src(isel.reg_of(inst.operand(1), "value"))
                      .src(isel.reg_of(inst.operand(2), "lane delta"))
                      .src(Operand::imm(0x1f))
                      .src(isel.reg_of(inst.operand(0), "member mask")));
    }

    // shfl.sync.down.b32 %f, value, delta, 0x1f, 0xffffffff   for (value, delta)
    static void shfl_down_f32(PtxISel& isel, const brass::Instruction& inst) {
        isel.emit(Inst::make(Opcode::shfl, Type::b32).sync().shfl(ShflMode::down)
                      .dst(isel.result_reg(inst))
                      .src(isel.reg_of(inst.operand(0), "value"))
                      .src(isel.reg_of(inst.operand(1), "lane delta"))
                      .src(Operand::imm(0x1f))
                      .src(Operand::imm(0xffffffffLL)));
    }
};

// The table. Every alias the old string-matching emitter accepted is kept.
const PtxISel::Intrinsics::Table& PtxISel::Intrinsics::table() {
    static const Table kIntrinsics = {
        // Thread / block / grid indices
        { "ptx_tid_x",         &special<SpecialReg::tid_x> },
        { "ptx_tid_y",         &special<SpecialReg::tid_y> },
        { "ptx_tid_z",         &special<SpecialReg::tid_z> },
        { "ptx_ctaid_x",       &special<SpecialReg::ctaid_x> },
        { "ptx_ctaid_y",       &special<SpecialReg::ctaid_y> },
        { "ptx_ctaid_z",       &special<SpecialReg::ctaid_z> },
        { "ptx_ntid_x",        &special<SpecialReg::ntid_x> },
        { "ptx_ntid_y",        &special<SpecialReg::ntid_y> },
        { "ptx_ntid_z",        &special<SpecialReg::ntid_z> },
        { "ptx_laneid",        &special<SpecialReg::laneid> },
        { "ptx_lane_id",       &special<SpecialReg::laneid> },
        { "ptx_warpid",        &special<SpecialReg::warpid> },
        { "ptx_warp_id",       &special<SpecialReg::warpid> },
        { "ptx_global_tid_x",  &global_tid_x },
        { "ptx_global_id_x",   &global_tid_x },

        // Approximate math
        { "rsqrtf",    &approx_f32<Opcode::rsqrt> },
        { "rsqrt",     &approx_f32<Opcode::rsqrt> },
        { "ptx_rsqrt", &approx_f32<Opcode::rsqrt> },
        { "sqrtf",     &approx_f32<Opcode::sqrt> },
        { "sqrt",      &approx_f32<Opcode::sqrt> },
        { "ptx_sqrt",  &approx_f32<Opcode::sqrt> },
        { "sinf",      &approx_f32<Opcode::sin> },
        { "sin",       &approx_f32<Opcode::sin> },
        { "ptx_sin",   &approx_f32<Opcode::sin> },
        { "cosf",      &approx_f32<Opcode::cos> },
        { "cos",       &approx_f32<Opcode::cos> },
        { "ptx_cos",   &approx_f32<Opcode::cos> },
        { "ex2f",      &approx_f32<Opcode::ex2> },
        { "ex2",       &approx_f32<Opcode::ex2> },
        { "ptx_ex2",   &approx_f32<Opcode::ex2> },
        { "expf",      &exp_f32 },
        { "exp",       &exp_f32 },
        { "ptx_exp",   &exp_f32 },

        // Conversions
        { "i32_to_f32", &i32_to_f32 },

        // Synchronization and warp shuffles
        { "bar.sync",               &bar_sync },
        { "ptx_sync",               &bar_sync },
        { "ptx_shfl_down_sync_f32", &shfl_down_sync_f32 },
        { "shfl_down_sync_f32",     &shfl_down_sync_f32 },
        { "ptx_shfl_down_f32",      &shfl_down_f32 },
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
