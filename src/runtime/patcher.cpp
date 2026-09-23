#include <brass/runtime/patcher.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <atomic>
#include <iostream>
#include <mutex>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <libkern/OSCacheControl.h>
#include <pthread.h>
#endif
#endif

namespace {
static inline void memory_fence() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_mfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

static inline void flush_code_cache(void* addr, size_t size) noexcept {
#if defined(__APPLE__)
    sys_dcache_flush(addr, size);
    sys_icache_invalidate(addr, size);
#elif defined(_WIN32)
    FlushInstructionCache(GetCurrentProcess(), addr, size);
#elif defined(__GNUC__) || defined(__clang__)
    char* begin = static_cast<char*>(addr);
    __builtin___clear_cache(begin, begin + size);
#endif
}

template <typename T>
static inline void atomic_store_release(T* ptr, T val) noexcept {
#if defined(__cpp_lib_atomic_ref)
    std::atomic_ref<T> ref(*ptr);
    ref.store(val, std::memory_order_release);
#elif defined(__GNUC__) || defined(__clang__)
    __atomic_store_n(ptr, val, __ATOMIC_RELEASE);
#else
    std::atomic_ref<T> ref(*ptr);
    ref.store(val, std::memory_order_release);
#endif
}

// Every patch takes this lock for the whole flip-write-restore sequence. Two
// patchers on the same page would otherwise race on its protection: on
// Windows the second one's VirtualProtect records the first one's temporary
// protection as the "old" one and restores it last, leaving the page
// writable and executable for good; with mprotect the first one's restore
// can land between the second one's flip and its write, which then faults.
std::mutex& code_patch_mutex() {
    static auto* m = new std::mutex();
    return *m;
}

// Makes the pages of a live JIT code range writable for one patch. The page
// keeps its execute permission while the patch is written: other threads may
// be running code on it, and taking execute away would fault them. This
// transient, lock-serialised window on the patched pages only is the one
// place code is writable and executable; JIT memory is otherwise W^X (see
// JitMemoryBlock). Callers hold code_patch_mutex().
class ScopedCodeWrite {
public:
    ScopedCodeWrite(void* addr, size_t size) {
        if (!addr || size == 0) return;
#if defined(__APPLE__) && defined(__aarch64__)
        pthread_jit_write_protect_np(0);
        active_ = true;
        writable_ = true;
        return;
#endif
        if (!brass::codegen::is_jit_code_address(addr)) {
            // Not a JIT code page (e.g. stack buffer or data memory). Already writable!
            writable_ = true;
            return;
        }
#if defined(_WIN32)
        DWORD old_protect = 0;
        if (VirtualProtect(addr, size, PAGE_EXECUTE_READWRITE, &old_protect)) {
            addr_ = addr;
            size_ = size;
            old_protect_ = old_protect;
            active_ = true;
        }
#else
        long page_size = sysconf(_SC_PAGESIZE);
        if (page_size <= 0) page_size = 4096;
        uintptr_t start = reinterpret_cast<uintptr_t>(addr);
        uintptr_t page_start = start & ~static_cast<uintptr_t>(page_size - 1);
        uintptr_t page_end = (start + size + page_size - 1) & ~static_cast<uintptr_t>(page_size - 1);
        page_addr_ = reinterpret_cast<void*>(page_start);
        page_len_ = page_end - page_start;

        if (mprotect(page_addr_, page_len_, PROT_READ | PROT_WRITE | PROT_EXEC) == 0) {
            active_ = true;
        }
#endif
    }

    ~ScopedCodeWrite() {
#if defined(__APPLE__) && defined(__aarch64__)
        if (active_) {
            pthread_jit_write_protect_np(1);
        }
#else
        if (!active_) return;
#if defined(_WIN32)
        DWORD dummy = 0;
        VirtualProtect(addr_, size_, old_protect_, &dummy);
#else
        mprotect(page_addr_, page_len_, PROT_READ | PROT_EXEC);
#endif
#endif
    }

    bool is_writable() const noexcept { return active_ || writable_; }

private:
    bool active_ = false;
    bool writable_ = false;
#if defined(_WIN32)
    void* addr_ = nullptr;
    size_t size_ = 0;
    DWORD old_protect_ = 0;
#else
    void* page_addr_ = nullptr;
    size_t page_len_ = 0;
#endif
};

} // anonymous namespace

namespace brass::runtime {

std::string_view to_string(PatchKind kind) noexcept {
    switch (kind) {
        case PatchKind::Const32: return "Const32";
        case PatchKind::Const64: return "Const64";
        case PatchKind::Call: return "Call";
        default: return "Unknown";
    }
}

std::ostream& operator<<(std::ostream& os, PatchKind kind) {
    return os << to_string(kind);
}

} // namespace brass::runtime

namespace brass::runtime {

namespace {

// B/BL imm26: the whole instruction is rewritten (opcode kept).
bool patch_aarch64_branch(void* call_site_addr, const void* new_target) {
    if ((reinterpret_cast<uintptr_t>(call_site_addr) & 3) != 0) return false;
    if (!is_cache_line_safe(call_site_addr, sizeof(uint32_t))) return false;
    std::lock_guard<std::mutex> lock(code_patch_mutex());
    const uint32_t current_inst = *reinterpret_cast<uint32_t*>(call_site_addr);
    const uint32_t opcode = current_inst & 0xFC000000u;
    if (opcode != 0x94000000u && opcode != 0x14000000u) return false;   // not BL / B
    const int64_t disp = reinterpret_cast<intptr_t>(new_target) - reinterpret_cast<intptr_t>(call_site_addr);
    if ((disp & 3) != 0) return false;
    const int64_t disp_words = disp >> 2;
    if (disp_words < -33554432 || disp_words > 33554431) return false; // +-128MB
    const uint32_t new_inst = opcode | (static_cast<uint32_t>(disp_words) & 0x03FFFFFFu);
    ScopedCodeWrite write_guard(call_site_addr, sizeof(uint32_t));
    if (!write_guard.is_writable()) return false;
    atomic_store_release(reinterpret_cast<uint32_t*>(call_site_addr), new_inst);
    memory_fence();
    flush_code_cache(call_site_addr, sizeof(uint32_t));
    return true;
}

// CALL/JMP rel32 (E8 / E9): the displacement after the opcode byte.
bool patch_x64_call(void* call_site_addr, const void* new_target) {
    uint8_t* inst = static_cast<uint8_t*>(call_site_addr);
    uint8_t* disp_ptr = inst + 1;
    if (!is_cache_line_safe(disp_ptr, sizeof(int32_t))) return false;
    std::lock_guard<std::mutex> lock(code_patch_mutex());
    if (*inst != 0xE8 && *inst != 0xE9) return false;
    const int64_t disp = reinterpret_cast<intptr_t>(new_target) - reinterpret_cast<intptr_t>(inst + 5);
    if (disp < INT32_MIN || disp > INT32_MAX) return false;
    ScopedCodeWrite write_guard(disp_ptr, sizeof(int32_t));
    if (!write_guard.is_writable()) return false;
    atomic_store_release(reinterpret_cast<int32_t*>(disp_ptr), static_cast<int32_t>(disp));
    memory_fence();
    flush_code_cache(disp_ptr, sizeof(int32_t));
    return true;
}

} // namespace

bool patch_call_site(CodeArch arch, void* call_site_addr, const void* new_target) {
    if (!call_site_addr || !new_target) return false;
    // The instruction set is the caller's to state: a byte pattern alone
    // cannot tell an x64 call from an AArch64 branch (E8 xx xx 94 is both a
    // plausible CALL and, read as a word, a BL).
    switch (arch) {
        case CodeArch::X64:     return patch_x64_call(call_site_addr, new_target);
        case CodeArch::AArch64: return patch_aarch64_branch(call_site_addr, new_target);
    }
    return false;
}

} // namespace brass::runtime

extern "C" {

bool brass_patch_const32(void* code_addr, int32_t new_val) {
    if (!code_addr) return false;
    if (!brass::runtime::is_cache_line_safe(code_addr, sizeof(int32_t))) return false;
    std::lock_guard<std::mutex> lock(code_patch_mutex());
    ScopedCodeWrite write_guard(code_addr, sizeof(int32_t));
    if (!write_guard.is_writable()) return false;
    auto* target_ptr = reinterpret_cast<int32_t*>(code_addr);
    atomic_store_release(target_ptr, new_val);
    memory_fence();
    flush_code_cache(code_addr, sizeof(int32_t));
    return true;
}

bool brass_patch_const64(void* code_addr, int64_t new_val) {
    if (!code_addr) return false;
    if (!brass::runtime::is_cache_line_safe(code_addr, sizeof(int64_t))) return false;
    std::lock_guard<std::mutex> lock(code_patch_mutex());
    ScopedCodeWrite write_guard(code_addr, sizeof(int64_t));
    if (!write_guard.is_writable()) return false;
    auto* target_ptr = reinterpret_cast<int64_t*>(code_addr);
    atomic_store_release(target_ptr, new_val);
    memory_fence();
    flush_code_cache(code_addr, sizeof(int64_t));
    return true;
}

bool brass_patch_call(void* call_site_addr, const void* new_target) {
    // In-process code is host code.
    return brass::runtime::patch_call_site(brass::runtime::host_code_arch(), call_site_addr, new_target);
}

} // extern "C"

namespace brass::runtime {

void PatchRegistry::register_site(PatchSite site) {
    auto it = site_map_.find(site.name);
    if (it != site_map_.end()) {
        sites_[it->second] = std::move(site);
        return;
    }
    site_map_[site.name] = sites_.size();
    sites_.push_back(std::move(site));
}

const PatchSite* PatchRegistry::find_site(std::string_view name) const noexcept {
    auto it = site_map_.find(std::string(name));
    if (it != site_map_.end()) {
        return &sites_[it->second];
    }
    return nullptr;
}

PatchSite* PatchRegistry::find_site(std::string_view name) noexcept {
    auto it = site_map_.find(std::string(name));
    if (it != site_map_.end()) {
        return &sites_[it->second];
    }
    return nullptr;
}

bool PatchRegistry::has_site(std::string_view name) const noexcept {
    return site_map_.find(std::string(name)) != site_map_.end();
}

bool PatchRegistry::patch_const32(void* fn_base, std::string_view name, int32_t new_val) {
    const auto* site = find_site(name);
    if (!site || site->kind != PatchKind::Const32 || !fn_base) {
        return false;
    }
    void* imm_addr = static_cast<uint8_t*>(fn_base) + site->code_offset + site->imm_offset;
    return brass_patch_const32(imm_addr, new_val);
}

bool PatchRegistry::patch_const64(void* fn_base, std::string_view name, int64_t new_val) {
    const auto* site = find_site(name);
    if (!site || site->kind != PatchKind::Const64 || !fn_base) {
        return false;
    }
    void* imm_addr = static_cast<uint8_t*>(fn_base) + site->code_offset + site->imm_offset;
    return brass_patch_const64(imm_addr, new_val);
}

bool PatchRegistry::patch_call(CodeArch arch, void* fn_base, std::string_view name, const void* new_target) {
    const auto* site = find_site(name);
    if (!site || site->kind != PatchKind::Call || !fn_base) {
        return false;
    }
    void* call_site = static_cast<uint8_t*>(fn_base) + site->code_offset;
    return patch_call_site(arch, call_site, new_target);
}

} // namespace brass::runtime
