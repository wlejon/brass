#pragma once

// The Windows unwind tables (.pdata RUNTIME_FUNCTIONs) brass hands to
// RtlAddFunctionTable, kept a second time where a sampling profiler can read
// them without taking ntdll's dynamic-function-table lock.
//
// A sampler suspends a thread and walks its stack. For a pc in JIT code the
// walk needs the function's RUNTIME_FUNCTION, and RtlLookupFunctionEntry finds
// it only under the lock RtlAddFunctionTable holds exclusively; if the
// suspended thread is itself inside RtlAddFunctionTable the lookup never
// returns and the process hangs. find_jit_unwind_entry_nonblocking answers the
// same question from brass's own copy, and gives up (false) rather than wait
// when another thread holds that copy's lock. RtlVirtualUnwind, given the
// entry, takes no lock.
//
// Every RtlAddFunctionTable in brass registers here too, and every
// RtlDeleteFunctionTable unregisters first. Elsewhere than Windows x64/ARM64
// the functions are no-ops that find nothing.

#include <cstddef>
#include <cstdint>

namespace brass::debug {

// `table` is `count` RUNTIME_FUNCTIONs whose addresses are relative to
// `image_base`, as passed to RtlAddFunctionTable.
void register_jit_unwind_table(const void* table, uint32_t count, uintptr_t image_base) noexcept;
void unregister_jit_unwind_table(const void* table) noexcept;

// The RUNTIME_FUNCTION covering `ip` and the image base its addresses are
// relative to. False when `ip` is in no registered table, or when the
// registry is busy on another thread (no waiting, no allocation).
bool find_jit_unwind_entry_nonblocking(uintptr_t ip, const void** function_entry, uintptr_t* image_base) noexcept;

}  // namespace brass::debug
