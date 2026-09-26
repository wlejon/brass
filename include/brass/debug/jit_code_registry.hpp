#pragma once

// JIT code as named functions, for tools outside brass and for brass itself.
//
// Every piece of machine code a JIT tier installs is registered here with its
// name and tier, and unregistered when the code is retired (the handle a
// registration returns is released before the code's memory is freed). The
// registry answers "which function, which tier" for an instruction address
// (brass's symbolizer and crash report use it), and fans each registration
// out to the native tools that were asked for:
//
//   BRASS_PERF_MAP=1     perf map, <dir>/perf-<pid>.map  (perf, bpftrace...)
//   BRASS_JITDUMP=1      jitdump, <dir>/jit-<pid>.dump   (perf inject --jit)
//   BRASS_PERF_DIR=<d>   <dir> for both (default /tmp; %TEMP% on Windows)
//   BRASS_GDB_JIT=1      the GDB JIT interface (__jit_debug_register_code)
//   INTEL_JIT_PROFILER64 set by VTune: its JIT profiling API (Windows/Linux)
//   BRASS_JIT_CRASH_REPORT=0/1  the named crash report (on by default on
//                        Windows, a vectored handler; opt-in on POSIX, signals)
//
// The knobs are read once, at the first registration. With none of them set a
// registration is one map insert under a mutex, on the compile path only.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brass::debug {

enum class JitTier : uint8_t {
    Stub = 0,       // trampolines and lazy-link stubs
    Baseline = 1,   // the baseline (template) tier
    Optimized = 2,  // the optimizing tier (and a plain JitExecutionEngine)
    Osr = 3,        // optimizing-tier code entered mid-loop (on-stack replacement)
};

// "tier 1", "tier 2", "tier 2 osr", "stub".
const char* jit_tier_label(JitTier tier) noexcept;

struct JitCodeRange {
    std::string_view name;
    const void* code = nullptr;
    size_t size = 0;
};

struct JitCodeInfo {
    std::string name;
    uintptr_t start = 0;
    size_t size = 0;
    JitTier tier = JitTier::Optimized;
};

// Registers `ranges` (empty ranges are skipped) as code of `tier`. The code
// must already be in place (the jitdump copies its bytes). The returned handle
// unregisters every range when the last copy of it is released; release it
// before the code's memory is freed. Null when nothing was registered.
[[nodiscard]] std::shared_ptr<const void> register_jit_code(JitTier tier, const JitCodeRange* ranges, size_t count);
[[nodiscard]] std::shared_ptr<const void> register_jit_code(JitTier tier, std::string_view name, const void* code,
                                                            size_t size);

// The registered function containing `ip`.
bool find_jit_code(uintptr_t ip, JitCodeInfo* out);
// "<name> [tier N]+0x<off>", or "" when `ip` is not in registered JIT code.
std::string describe_jit_address(uintptr_t ip);
// For a fault handler: no allocation, no blocking (a registry held by the
// faulting thread yields false). Writes the NUL-terminated name into `name`.
bool find_jit_code_nonblocking(uintptr_t ip, char* name, size_t name_cap, JitTier* tier, uintptr_t* start) noexcept;

// Registered functions right now.
size_t jit_code_count();

// One native frame of a stack walk: JIT frames carry their function and tier.
struct NativeFrame {
    uintptr_t ip = 0;
    bool is_jit = false;
    JitCodeInfo jit;       // when is_jit
    std::string module;    // otherwise: the image holding ip ("" if none), and
    uintptr_t module_offset = 0;
};

// The calling thread's native stack, innermost first, starting at the caller
// of this function. JIT frames unwind through the unwind data brass registers
// (Windows); elsewhere through the frame-pointer chain.
std::vector<NativeFrame> capture_native_stack(size_t max_frames = 64);

// "#0 <name> [tier 2]+0x1c" / "#1 brass.exe+0x1234" lines, then, when a
// coroutine runs on this thread, its async stack ("async #0 <body> state 3").
std::string format_native_stack(const std::vector<NativeFrame>& frames, bool with_async_stack = true);

// The crash report's handler, installed on demand by the first registration
// (see BRASS_JIT_CRASH_REPORT); tests call these to install it explicitly.
// A fault whose address is in JIT code prints the named stack to stderr and
// is passed on unhandled.
void install_jit_crash_report();
bool jit_crash_report_installed() noexcept;

// Knob state, read once. Tests may re-read after changing the environment,
// before code they want reported is registered.
struct JitProfilerConfig {
    bool perf_map = false;
    bool jitdump = false;
    bool gdb_jit = false;
    bool vtune = false;
    bool crash_report = false;
    std::string perf_dir;
};
const JitProfilerConfig& jit_profiler_config();
void reload_jit_profiler_config();
// Where this process's perf map / jitdump are (or would be) written.
std::string perf_map_path();
std::string jitdump_path();

} // namespace brass::debug

// The GDB JIT interface (gdb/Documentation "JIT Compilation Interface"): the
// debugger sets a breakpoint on __jit_debug_register_code and reads
// __jit_debug_descriptor. brass hands it one in-memory ELF symbol file per
// registration when BRASS_GDB_JIT=1.
extern "C" {
enum brass_jit_actions_t { BRASS_JIT_NOACTION = 0, BRASS_JIT_REGISTER_FN, BRASS_JIT_UNREGISTER_FN };
struct jit_code_entry {
    jit_code_entry* next_entry;
    jit_code_entry* prev_entry;
    const char* symfile_addr;
    uint64_t symfile_size;
};
struct jit_descriptor {
    uint32_t version;
    uint32_t action_flag;
    jit_code_entry* relevant_entry;
    jit_code_entry* first_entry;
};
extern jit_descriptor __jit_debug_descriptor;
void __jit_debug_register_code(void);
}
