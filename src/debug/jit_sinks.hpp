#pragma once

// The native-tool sinks the JIT code registry fans registrations out to.
// Each is called only when its knob is on, under the registry's sink mutex.

#include <brass/debug/jit_code_registry.hpp>
#include <cstdint>
#include <string>
#include <vector>

namespace brass::debug::detail {

unsigned long current_pid() noexcept;

// "<name> [tier N]", the name every tool shows.
std::string tool_name(std::string_view name, JitTier tier);

// perf map: one "START SIZE name" line per range, appended.
void perf_map_append(const JitProfilerConfig& cfg, JitTier tier, const JitCodeRange* ranges, size_t count);

// jitdump: a JIT_CODE_LOAD record (with the code bytes) per range.
void jitdump_append(const JitProfilerConfig& cfg, JitTier tier, const JitCodeRange* ranges, size_t count);

// The in-memory ELF symbol file the GDB JIT interface is handed for a batch:
// one SHT_NOBITS .text spanning the batch and an STT_FUNC per range.
std::vector<uint8_t> build_gdb_symfile(JitTier tier, const JitCodeRange* ranges, size_t count);
// Links a symfile into __jit_debug_descriptor and tells the debugger; the
// returned entry owns the symfile until gdb_jit_unregister.
void* gdb_jit_register(std::vector<uint8_t> symfile);
void gdb_jit_unregister(void* entry) noexcept;

// VTune's JIT profiling API, loaded from INTEL_JIT_PROFILER64. Returns the
// method ids it assigned (empty when VTune is not collecting).
std::vector<uint32_t> vtune_register(JitTier tier, const JitCodeRange* ranges, size_t count);
void vtune_unregister(const std::vector<uint32_t>& ids) noexcept;

// Installs the platform fault handler behind install_jit_crash_report.
void install_crash_handler();

} // namespace brass::debug::detail
