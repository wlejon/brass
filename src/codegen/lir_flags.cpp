#include <brass/codegen/lir_flags.hpp>

// Each switch below names every LirOpcode and has no default, so a new
// opcode is a -Wswitch warning (an error under -Werror) here until someone
// states what it does to the flags on each target.
//
// The table describes what the emitters actually emit:
//   x64      src/codegen/emit_*.cpp
//   AArch64  src/target/aarch64/aarch64_emit_*.cpp
// A lowering that starts or stops using a flag-setting instruction has to be
// reflected here.

namespace brass::codegen {

namespace {

FlagEffect x64_flag_effect(LirOpcode op) noexcept {
    switch (op) {
        // Full, deterministic definitions of ZF/SF/CF/OF (and PF for the
        // ucomis compares, which are what the FP condition codes read).
        case LirOpcode::Cmp: case LirOpcode::Cmp32:
        case LirOpcode::Test: case LirOpcode::Test32:
        case LirOpcode::Add: case LirOpcode::Add32:
        case LirOpcode::Sub: case LirOpcode::Sub32:
        case LirOpcode::And: case LirOpcode::And32:
        case LirOpcode::Or: case LirOpcode::Or32:
        case LirOpcode::Xor: case LirOpcode::Xor32:
        case LirOpcode::Neg: case LirOpcode::Neg32:
        case LirOpcode::Ucomisd: case LirOpcode::Ucomiss:
            return FlagEffect::Defines;

        // Write some flags, or leave them undefined, or write them only
        // conditionally: imul (SF/ZF undefined), idiv/div (all undefined),
        // shifts (unchanged for a zero count), popcnt/lzcnt/tzcnt/bsr/bsf
        // (partial or undefined), the one-operand mul behind smulh/umulh.
        case LirOpcode::Imul: case LirOpcode::Imul32:
        case LirOpcode::Smulh: case LirOpcode::Umulh:
        case LirOpcode::Idiv: case LirOpcode::Idiv32:
        case LirOpcode::Div: case LirOpcode::Div32:
        case LirOpcode::Shl: case LirOpcode::Shl32:
        case LirOpcode::Shr: case LirOpcode::Shr32:
        case LirOpcode::Sar: case LirOpcode::Sar32:
        case LirOpcode::Popcnt: case LirOpcode::Popcnt32:
        case LirOpcode::Lzcnt: case LirOpcode::Lzcnt32:
        case LirOpcode::Tzcnt: case LirOpcode::Tzcnt32:
        case LirOpcode::Bsr: case LirOpcode::Bsr32:
        case LirOpcode::Bsf: case LirOpcode::Bsf32:
        // Adds/Subs are AArch64-only in practice; if x64 ever sees them they
        // lower to add/sub, which at least write the flags.
        case LirOpcode::Adds: case LirOpcode::Adds32:
        case LirOpcode::Subs: case LirOpcode::Subs32:
        // Runtime sequences: polls, barrier fast paths and guard checks
        // compare and branch internally, and calls clobber everything.
        case LirOpcode::Call: case LirOpcode::CallIndirect:
        case LirOpcode::Safepoint:
        case LirOpcode::WriteBarrier:
        case LirOpcode::GuardExit:
            return FlagEffect::Clobbers;

        case LirOpcode::Nop: case LirOpcode::KeepAlive:
        case LirOpcode::Mov: case LirOpcode::Mov32: case LirOpcode::Movabs:
        case LirOpcode::Movsx8: case LirOpcode::Movsx16: case LirOpcode::Movsxd:
        case LirOpcode::Movzx8: case LirOpcode::Movzx16:
        case LirOpcode::Cdq: case LirOpcode::Cqo:
        case LirOpcode::Not: case LirOpcode::Not32:
        case LirOpcode::Setcc: case LirOpcode::Cmovcc:
        case LirOpcode::Movsd: case LirOpcode::Movss:
        case LirOpcode::Movq_gx: case LirOpcode::Movq_xg:
        case LirOpcode::Addsd: case LirOpcode::Addss:
        case LirOpcode::Subsd: case LirOpcode::Subss:
        case LirOpcode::Mulsd: case LirOpcode::Mulss:
        case LirOpcode::Divsd: case LirOpcode::Divss:
        case LirOpcode::Sqrtsd: case LirOpcode::Sqrtss:
        case LirOpcode::Xorpd:
        case LirOpcode::Fneg: case LirOpcode::Fneg32:
        case LirOpcode::Cvtsi2sd: case LirOpcode::Cvtsi2sd32:
        case LirOpcode::Cvttsd2si: case LirOpcode::Cvttsd2si32:
        case LirOpcode::Cvtsi2ss: case LirOpcode::Cvtsi2ss32:
        case LirOpcode::Cvttss2si: case LirOpcode::Cvttss2si32:
        case LirOpcode::Cvtsd2ss: case LirOpcode::Cvtss2sd:
        case LirOpcode::Floor32: case LirOpcode::Floor64:
        case LirOpcode::Ceil32: case LirOpcode::Ceil64:
        case LirOpcode::Round32: case LirOpcode::Round64:
        case LirOpcode::Fabs32: case LirOpcode::Fabs64:
        case LirOpcode::Minss: case LirOpcode::Minsd:
        case LirOpcode::Maxss: case LirOpcode::Maxsd:
        case LirOpcode::Movaps: case LirOpcode::Movups:
        case LirOpcode::Movd_xg: case LirOpcode::Movd_gx:
        case LirOpcode::Addps: case LirOpcode::Subps: case LirOpcode::Mulps:
        case LirOpcode::Divps: case LirOpcode::Minps: case LirOpcode::Maxps:
        case LirOpcode::Sqrtps:
        case LirOpcode::Addpd: case LirOpcode::Subpd: case LirOpcode::Mulpd:
        case LirOpcode::Divpd: case LirOpcode::Minpd: case LirOpcode::Maxpd:
        case LirOpcode::Sqrtpd:
        case LirOpcode::Fneg4s: case LirOpcode::Fneg2d:
        case LirOpcode::Paddd: case LirOpcode::Psubd: case LirOpcode::Pmulld:
        case LirOpcode::Pminsd: case LirOpcode::Pmaxsd:
        case LirOpcode::Paddq: case LirOpcode::Psubq:
        case LirOpcode::Pand: case LirOpcode::Por: case LirOpcode::Pxor:
        case LirOpcode::Pandn: case LirOpcode::Pnot:
        case LirOpcode::Pcmpeqd: case LirOpcode::Pslld: case LirOpcode::Psllq:
        case LirOpcode::Shufps: case LirOpcode::Shufpd: case LirOpcode::Pshufd:
        case LirOpcode::Movddup:
        case LirOpcode::Pinsrd: case LirOpcode::Pextrd:
        case LirOpcode::Pinsrq: case LirOpcode::Pextrq:
        case LirOpcode::Insertps: case LirOpcode::Extractps:
        case LirOpcode::Xorps:
        case LirOpcode::Vmovaps: case LirOpcode::Vmovups:
        case LirOpcode::Vaddps: case LirOpcode::Vsubps: case LirOpcode::Vmulps:
        case LirOpcode::Vdivps: case LirOpcode::Vminps: case LirOpcode::Vmaxps:
        case LirOpcode::Vaddpd: case LirOpcode::Vsubpd: case LirOpcode::Vmulpd:
        case LirOpcode::Vdivpd: case LirOpcode::Vminpd: case LirOpcode::Vmaxpd:
        case LirOpcode::Vsqrtps: case LirOpcode::Vsqrtpd:
        case LirOpcode::Vextractf128: case LirOpcode::Vinsertf128:
        case LirOpcode::Vpaddd: case LirOpcode::Vpsubd: case LirOpcode::Vpmulld:
        case LirOpcode::Vpaddq: case LirOpcode::Vpsubq:
        case LirOpcode::Vandps: case LirOpcode::Vorps: case LirOpcode::Vxorps:
        case LirOpcode::Vandpd: case LirOpcode::Vorpd: case LirOpcode::Vxorpd:
        case LirOpcode::Vpand: case LirOpcode::Vpor: case LirOpcode::Vpxor:
        case LirOpcode::Vbroadcastss: case LirOpcode::Vbroadcastsd:
        case LirOpcode::Vpbroadcastd: case LirOpcode::Vpbroadcastq:
        case LirOpcode::Vfmadd213ps: case LirOpcode::Vfmadd231ps:
        case LirOpcode::Vfmadd213pd: case LirOpcode::Vfmadd231pd:
        case LirOpcode::Vfmadd213ss: case LirOpcode::Vfmadd231ss:
        case LirOpcode::Vfmadd213sd: case LirOpcode::Vfmadd231sd:
        case LirOpcode::Jmp: case LirOpcode::Jcc: case LirOpcode::Ret:
        case LirOpcode::Push: case LirOpcode::Pop: case LirOpcode::Lea:
        case LirOpcode::ParallelCopy:
        case LirOpcode::Trap:
            return FlagEffect::None;
    }
    return FlagEffect::Clobbers;  // out-of-range value: assume the worst
}

FlagEffect aarch64_flag_effect(LirOpcode op) noexcept {
    switch (op) {
        // cmp/cmn, tst, adds/subs, fcmp: every NZCV bit, deterministically.
        case LirOpcode::Cmp: case LirOpcode::Cmp32:
        case LirOpcode::Test: case LirOpcode::Test32:
        case LirOpcode::Adds: case LirOpcode::Adds32:
        case LirOpcode::Subs: case LirOpcode::Subs32:
        case LirOpcode::Ucomisd: case LirOpcode::Ucomiss:
            return FlagEffect::Defines;

        case LirOpcode::Call: case LirOpcode::CallIndirect:
        case LirOpcode::Safepoint:
        case LirOpcode::WriteBarrier:
        case LirOpcode::GuardExit:
            return FlagEffect::Clobbers;

        // AArch64 arithmetic does not touch NZCV unless it is the S form:
        // add/sub/and/orr/eor/neg, mul/smulh/umulh, sdiv/udiv (the divide by
        // zero check is a cbz), lsl/lsr/asr, clz/rbit/cnt all leave it alone.
        case LirOpcode::Add: case LirOpcode::Add32:
        case LirOpcode::Sub: case LirOpcode::Sub32:
        case LirOpcode::Imul: case LirOpcode::Imul32:
        case LirOpcode::Smulh: case LirOpcode::Umulh:
        case LirOpcode::Idiv: case LirOpcode::Idiv32:
        case LirOpcode::Div: case LirOpcode::Div32:
        case LirOpcode::And: case LirOpcode::And32:
        case LirOpcode::Or: case LirOpcode::Or32:
        case LirOpcode::Xor: case LirOpcode::Xor32:
        case LirOpcode::Neg: case LirOpcode::Neg32:
        case LirOpcode::Shl: case LirOpcode::Shl32:
        case LirOpcode::Shr: case LirOpcode::Shr32:
        case LirOpcode::Sar: case LirOpcode::Sar32:
        case LirOpcode::Popcnt: case LirOpcode::Popcnt32:
        case LirOpcode::Lzcnt: case LirOpcode::Lzcnt32:
        case LirOpcode::Tzcnt: case LirOpcode::Tzcnt32:
        case LirOpcode::Bsr: case LirOpcode::Bsr32:
        case LirOpcode::Bsf: case LirOpcode::Bsf32:
        case LirOpcode::Nop: case LirOpcode::KeepAlive:
        case LirOpcode::Mov: case LirOpcode::Mov32: case LirOpcode::Movabs:
        case LirOpcode::Movsx8: case LirOpcode::Movsx16: case LirOpcode::Movsxd:
        case LirOpcode::Movzx8: case LirOpcode::Movzx16:
        case LirOpcode::Cdq: case LirOpcode::Cqo:
        case LirOpcode::Not: case LirOpcode::Not32:
        case LirOpcode::Setcc: case LirOpcode::Cmovcc:
        case LirOpcode::Movsd: case LirOpcode::Movss:
        case LirOpcode::Movq_gx: case LirOpcode::Movq_xg:
        case LirOpcode::Addsd: case LirOpcode::Addss:
        case LirOpcode::Subsd: case LirOpcode::Subss:
        case LirOpcode::Mulsd: case LirOpcode::Mulss:
        case LirOpcode::Divsd: case LirOpcode::Divss:
        case LirOpcode::Sqrtsd: case LirOpcode::Sqrtss:
        case LirOpcode::Xorpd:
        case LirOpcode::Fneg: case LirOpcode::Fneg32:
        case LirOpcode::Cvtsi2sd: case LirOpcode::Cvtsi2sd32:
        case LirOpcode::Cvttsd2si: case LirOpcode::Cvttsd2si32:
        case LirOpcode::Cvtsi2ss: case LirOpcode::Cvtsi2ss32:
        case LirOpcode::Cvttss2si: case LirOpcode::Cvttss2si32:
        case LirOpcode::Cvtsd2ss: case LirOpcode::Cvtss2sd:
        case LirOpcode::Floor32: case LirOpcode::Floor64:
        case LirOpcode::Ceil32: case LirOpcode::Ceil64:
        case LirOpcode::Round32: case LirOpcode::Round64:
        case LirOpcode::Fabs32: case LirOpcode::Fabs64:
        case LirOpcode::Minss: case LirOpcode::Minsd:
        case LirOpcode::Maxss: case LirOpcode::Maxsd:
        case LirOpcode::Movaps: case LirOpcode::Movups:
        case LirOpcode::Movd_xg: case LirOpcode::Movd_gx:
        case LirOpcode::Addps: case LirOpcode::Subps: case LirOpcode::Mulps:
        case LirOpcode::Divps: case LirOpcode::Minps: case LirOpcode::Maxps:
        case LirOpcode::Sqrtps:
        case LirOpcode::Addpd: case LirOpcode::Subpd: case LirOpcode::Mulpd:
        case LirOpcode::Divpd: case LirOpcode::Minpd: case LirOpcode::Maxpd:
        case LirOpcode::Sqrtpd:
        case LirOpcode::Fneg4s: case LirOpcode::Fneg2d:
        case LirOpcode::Paddd: case LirOpcode::Psubd: case LirOpcode::Pmulld:
        case LirOpcode::Pminsd: case LirOpcode::Pmaxsd:
        case LirOpcode::Paddq: case LirOpcode::Psubq:
        case LirOpcode::Pand: case LirOpcode::Por: case LirOpcode::Pxor:
        case LirOpcode::Pandn: case LirOpcode::Pnot:
        case LirOpcode::Pcmpeqd: case LirOpcode::Pslld: case LirOpcode::Psllq:
        case LirOpcode::Shufps: case LirOpcode::Shufpd: case LirOpcode::Pshufd:
        case LirOpcode::Movddup:
        case LirOpcode::Pinsrd: case LirOpcode::Pextrd:
        case LirOpcode::Pinsrq: case LirOpcode::Pextrq:
        case LirOpcode::Insertps: case LirOpcode::Extractps:
        case LirOpcode::Xorps:
        case LirOpcode::Vmovaps: case LirOpcode::Vmovups:
        case LirOpcode::Vaddps: case LirOpcode::Vsubps: case LirOpcode::Vmulps:
        case LirOpcode::Vdivps: case LirOpcode::Vminps: case LirOpcode::Vmaxps:
        case LirOpcode::Vaddpd: case LirOpcode::Vsubpd: case LirOpcode::Vmulpd:
        case LirOpcode::Vdivpd: case LirOpcode::Vminpd: case LirOpcode::Vmaxpd:
        case LirOpcode::Vsqrtps: case LirOpcode::Vsqrtpd:
        case LirOpcode::Vextractf128: case LirOpcode::Vinsertf128:
        case LirOpcode::Vpaddd: case LirOpcode::Vpsubd: case LirOpcode::Vpmulld:
        case LirOpcode::Vpaddq: case LirOpcode::Vpsubq:
        case LirOpcode::Vandps: case LirOpcode::Vorps: case LirOpcode::Vxorps:
        case LirOpcode::Vandpd: case LirOpcode::Vorpd: case LirOpcode::Vxorpd:
        case LirOpcode::Vpand: case LirOpcode::Vpor: case LirOpcode::Vpxor:
        case LirOpcode::Vbroadcastss: case LirOpcode::Vbroadcastsd:
        case LirOpcode::Vpbroadcastd: case LirOpcode::Vpbroadcastq:
        case LirOpcode::Vfmadd213ps: case LirOpcode::Vfmadd231ps:
        case LirOpcode::Vfmadd213pd: case LirOpcode::Vfmadd231pd:
        case LirOpcode::Vfmadd213ss: case LirOpcode::Vfmadd231ss:
        case LirOpcode::Vfmadd213sd: case LirOpcode::Vfmadd231sd:
        case LirOpcode::Jmp: case LirOpcode::Jcc: case LirOpcode::Ret:
        case LirOpcode::Push: case LirOpcode::Pop: case LirOpcode::Lea:
        case LirOpcode::ParallelCopy:
        case LirOpcode::Trap:
            return FlagEffect::None;
    }
    return FlagEffect::Clobbers;  // out-of-range value: assume the worst
}

} // namespace

FlagEffect lir_flag_effect(const LirInst& inst, Arch arch) noexcept {
    // A call-shaped instruction (whatever its opcode) runs code that is free
    // to use the flags.
    if (inst.is_call()) return FlagEffect::Clobbers;
    switch (arch) {
        case Arch::x64:     return x64_flag_effect(inst.opcode);
        case Arch::aarch64: return aarch64_flag_effect(inst.opcode);
    }
    return FlagEffect::Clobbers;
}

bool lir_reads_flags(const LirInst& inst, Arch arch) noexcept {
    (void)arch;  // jcc/setcc/cmovcc and b.cond/cset/csel: the same opcodes
    switch (inst.opcode) {
        case LirOpcode::Jcc:
        case LirOpcode::Setcc:
        case LirOpcode::Cmovcc:
            return true;
        default:
            return false;
    }
}

} // namespace brass::codegen
