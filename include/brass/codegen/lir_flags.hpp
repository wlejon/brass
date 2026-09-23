#pragma once

#include <brass/codegen/lir.hpp>
#include <brass/target/target.hpp>
#include <cstdint>

// The one table of what each LIR instruction does to the condition flags
// (x64 EFLAGS, AArch64 NZCV) as the target's emitter actually lowers it.
// The instruction scheduler (ordering around flag producers and consumers)
// and the peephole optimizer (flag liveness before rewriting an instruction
// into a flag-clobbering one) both ask here, so a new opcode, or a lowering
// that starts using a flag-setting instruction, is described once.
namespace brass::codegen {

enum class FlagEffect : uint8_t {
    None,      // leaves every flag as it was
    Clobbers,  // writes some flags, or leaves them undefined, or writes them
               // only sometimes (x64 shift by CL = 0, bsf of 0, idiv, calls)
    Defines,   // sets every condition flag a consumer can read, deterministically
};

// What `inst` does to the flags when lowered for `arch`.
FlagEffect lir_flag_effect(const LirInst& inst, Arch arch) noexcept;

// True if `inst` reads the flags (jcc / setcc / cmovcc; b.cond / cset / csel).
bool lir_reads_flags(const LirInst& inst, Arch arch) noexcept;

// Writes any flag at all: what the scheduler orders against.
inline bool lir_writes_flags(const LirInst& inst, Arch arch) noexcept {
    return lir_flag_effect(inst, arch) != FlagEffect::None;
}

// Ends the life of whatever flags were live before it: what a liveness scan
// may stop at. Only a full, unconditional definition qualifies.
inline bool lir_kills_flags(const LirInst& inst, Arch arch) noexcept {
    return lir_flag_effect(inst, arch) == FlagEffect::Defines;
}

// The architecture a LIR function was selected for (its calling convention
// carries the target).
inline Arch lir_arch(const LirFunction& fn) noexcept {
    return fn.calling_conv.target().arch();
}

} // namespace brass::codegen
