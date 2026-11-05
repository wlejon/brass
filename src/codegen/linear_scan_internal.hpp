#pragma once

#include <cstdint>

namespace brass::codegen {

// LinearScanAllocator::only_copied_ holds these bits per vreg.
// Every read of the value is a block argument's copy or takes no register.
inline constexpr uint8_t kOnlyCopied = 1;
// The value is a GPR that a loop it is invariant in reads only by moving it
// across to an XMM register (or copying it into a value read only so), each
// such read folded into a load from its slot (mark_folded_uses).
inline constexpr uint8_t kXmmReadInLoop = 2;

} // namespace brass::codegen
