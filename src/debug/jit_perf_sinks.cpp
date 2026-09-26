// Linux perf's two ways of learning about JIT code, written from any host (on
// Windows the files only serve tests and manual lookup):
//
//   perf map  <dir>/perf-<pid>.map, "START SIZE name" lines (hex, no 0x).
//             perf report / perf top / bpftrace read it for the pid.
//   jitdump   <dir>/jit-<pid>.dump, the binary format of
//             tools/perf/Documentation/jitdump-specification.txt, one
//             JIT_CODE_LOAD record per function with its code bytes. The
//             file is mapped executable once so that `perf record -k 1`
//             sees it; `perf inject --jit` then makes a .so per function.
//
// Neither format can say that code went away: a reused address gets a later
// line / record, and perf takes the newest (jitdump by timestamp).

#include "jit_sinks.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <share.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#endif

namespace brass::debug::detail {

unsigned long current_pid() noexcept {
#if defined(_WIN32)
    return GetCurrentProcessId();
#else
    return static_cast<unsigned long>(getpid());
#endif
}

namespace {

unsigned long current_tid() noexcept {
#if defined(_WIN32)
    return GetCurrentThreadId();
#elif defined(__linux__)
    return static_cast<unsigned long>(syscall(SYS_gettid));
#else
    return 0;
#endif
}

// The clock `perf record -k 1` (CLOCK_MONOTONIC) stamps samples with.
uint64_t timestamp_ns() noexcept {
#if defined(_WIN32)
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
#else
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
#endif
}

uint32_t elf_machine() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    return 183;   // EM_AARCH64
#else
    return 62;    // EM_X86_64
#endif
}

FILE* open_append(const std::string& path, const char* mode) {
#if defined(_MSC_VER)
    // Shared, so tools can read the file while this process writes it
    // (fopen_s opens files exclusively).
    return _fsopen(path.c_str(), mode, _SH_DENYNO);
#else
    return std::fopen(path.c_str(), mode);
#endif
}

// Files stay open for the process; a changed directory (tests) reopens.
struct OpenFile {
    FILE* f = nullptr;
    std::string path;
};

FILE* file_for(OpenFile& of, const std::string& path, const char* mode, bool* fresh) {
    *fresh = false;
    if (of.f && of.path == path) return of.f;
    if (of.f) std::fclose(of.f);
    of.f = open_append(path, mode);
    of.path = path;
    *fresh = of.f != nullptr;
    return of.f;
}

struct JitDumpHeader {
    uint32_t magic = 0x4A695444;   // "JiTD" as a little-endian u32
    uint32_t version = 1;
    uint32_t total_size = sizeof(JitDumpHeader);
    uint32_t elf_mach = 0;
    uint32_t pad1 = 0;
    uint32_t pid = 0;
    uint64_t timestamp = 0;
    uint64_t flags = 0;
};
static_assert(sizeof(JitDumpHeader) == 40);

struct JitCodeLoadRecord {
    uint32_t id = 0;               // JIT_CODE_LOAD
    uint32_t total_size = 0;
    uint64_t timestamp = 0;
    uint32_t pid = 0;
    uint32_t tid = 0;
    uint64_t vma = 0;
    uint64_t code_addr = 0;
    uint64_t code_size = 0;
    uint64_t code_index = 0;
};
static_assert(sizeof(JitCodeLoadRecord) == 56);

} // namespace

void perf_map_append(const JitProfilerConfig& cfg, JitTier tier, const JitCodeRange* ranges, size_t count) {
    static OpenFile file;
    bool fresh = false;
    FILE* f = file_for(file, cfg.perf_dir + "/perf-" + std::to_string(current_pid()) + ".map", "a", &fresh);
    if (!f) return;
    for (size_t i = 0; i < count; ++i) {
        if (!ranges[i].code || !ranges[i].size) continue;
        std::fprintf(f, "%llx %llx %s\n", static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(ranges[i].code)),
                     static_cast<unsigned long long>(ranges[i].size), tool_name(ranges[i].name, tier).c_str());
    }
    std::fflush(f);
}

void jitdump_append(const JitProfilerConfig& cfg, JitTier tier, const JitCodeRange* ranges, size_t count) {
    static OpenFile file;
    static uint64_t code_index = 0;
    bool fresh = false;
    const std::string path = cfg.perf_dir + "/jit-" + std::to_string(current_pid()) + ".dump";
    FILE* f = file_for(file, path, "wb", &fresh);
    if (!f) return;
    if (fresh) {
        JitDumpHeader h;
        h.elf_mach = elf_machine();
        h.pid = static_cast<uint32_t>(current_pid());
        h.timestamp = timestamp_ns();
        std::fwrite(&h, sizeof h, 1, f);
        std::fflush(f);
#if !defined(_WIN32)
        // The marker perf record looks for: an executable mapping of the file.
        const int fd = fileno(f);
        const long page = sysconf(_SC_PAGESIZE);
        void* marker = mmap(nullptr, static_cast<size_t>(page), PROT_READ | PROT_EXEC, MAP_PRIVATE, fd, 0);
        (void)marker;   // kept for the life of the process
#endif
    }
    for (size_t i = 0; i < count; ++i) {
        if (!ranges[i].code || !ranges[i].size) continue;
        const std::string name = tool_name(ranges[i].name, tier);
        JitCodeLoadRecord r;
        r.total_size = static_cast<uint32_t>(sizeof r + name.size() + 1 + ranges[i].size);
        r.timestamp = timestamp_ns();
        r.pid = static_cast<uint32_t>(current_pid());
        r.tid = static_cast<uint32_t>(current_tid());
        r.vma = reinterpret_cast<uintptr_t>(ranges[i].code);
        r.code_addr = r.vma;
        r.code_size = ranges[i].size;
        r.code_index = code_index++;
        std::fwrite(&r, sizeof r, 1, f);
        std::fwrite(name.c_str(), name.size() + 1, 1, f);
        std::fwrite(ranges[i].code, ranges[i].size, 1, f);
    }
    std::fflush(f);
}

} // namespace brass::debug::detail
