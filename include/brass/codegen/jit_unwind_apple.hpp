#pragma once

#include <cstddef>
#include <cstdint>

namespace brass::codegen {

#if defined(__APPLE__)

// Returns true if the host libunwind supports dynamic unwind section registration
// (__unw_add_find_dynamic_unwind_sections).
bool has_apple_dynamic_unwind() noexcept;

// Registers a JIT code range and its corresponding .eh_frame section with libunwind.
// Returns true on success, false if dynamic unwind sections is unsupported.
bool register_apple_dynamic_unwind(const void* code_start, size_t code_size,
                                   const void* eh_frame, size_t eh_frame_size) noexcept;

// Unregisters a previously registered JIT code range by its code_start pointer or eh_frame pointer.
void unregister_apple_dynamic_unwind(const void* ptr) noexcept;

#else

inline bool has_apple_dynamic_unwind() noexcept { return false; }
inline bool register_apple_dynamic_unwind(const void*, size_t, const void*, size_t) noexcept { return false; }
inline void unregister_apple_dynamic_unwind(const void*) noexcept {}

#endif

} // namespace brass::codegen
