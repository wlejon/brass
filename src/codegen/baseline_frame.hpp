#pragma once

// What the baseline tier's two backends (x64: baseline_emit*.cpp, AArch64:
// target/aarch64/aarch64_baseline_emit*.cpp) share: the slot conventions and
// the frame layout. Target-neutral, so it names no encoder.

#include <brass/mir/function.hpp>
#include <brass/mir/types.hpp>
#include <cstdint>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace brass::codegen {

// Slot conventions of the baseline tier:
//   - an I8/I16/I32 value lives in the low 4 bytes of its slot and is
//     operated on at 32 bits (the interpreter models all three as i32);
//   - an F32 lives in the low 4 bytes, an F64 / I64 / Ptr / GCRef in all 8.
//   - a 128-bit vector owns a 16-byte, 16-byte-aligned slot. 256-bit
//     vectors are not compiled by this tier.
inline bool bl_is_int32(Type t) { return t.is_integer() && t.size_in_bytes() <= 4; }
inline bool bl_is_f32(Type t) { return t.kind() == TypeKind::F32; }
inline bool bl_is_v128(Type t) { return t.is_v128(); }

// The frame below the frame pointer (baseline_frame.cpp): each value's slot
// and each alloca's buffer as a positive offset down from it, the GC-ref
// slots (stack-map roots), and the bytes used, starting after
// `start_offset`. Values whose live ranges do not overlap share a slot.
// A function the layout cannot place is rejected (UnsupportedOperation)
// under `stage`.
struct BaselineFrameLayout {
    std::unordered_map<const Value*, int32_t> slot_map;
    std::unordered_map<const Instruction*, int32_t> alloca_offsets;
    std::vector<int32_t> gcref_slots;
    int32_t size = 0;
};
BaselineFrameLayout layout_baseline_frame(const Function& fn, int32_t start_offset, std::string_view stage);

} // namespace brass::codegen
