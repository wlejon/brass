#include <brass/runtime/patcher.hpp>
#include <atomic>
#include <iostream>

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
#endif

namespace {
static inline void memory_fence() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_mfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
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

extern "C" {

bool brass_patch_const32(void* code_addr, int32_t new_val) {
    if (!code_addr) return false;
    auto* target_ptr = reinterpret_cast<int32_t*>(code_addr);
    atomic_store_release(target_ptr, new_val);
    memory_fence();
#if defined(_WIN32)
    FlushInstructionCache(GetCurrentProcess(), code_addr, sizeof(int32_t));
#elif defined(__GNUC__) || defined(__clang__)
    char* begin = static_cast<char*>(code_addr);
    __builtin___clear_cache(begin, begin + sizeof(int32_t));
#endif
    return true;
}

bool brass_patch_const64(void* code_addr, int64_t new_val) {
    if (!code_addr) return false;
    auto* target_ptr = reinterpret_cast<int64_t*>(code_addr);
    atomic_store_release(target_ptr, new_val);
    memory_fence();
#if defined(_WIN32)
    FlushInstructionCache(GetCurrentProcess(), code_addr, sizeof(int64_t));
#elif defined(__GNUC__) || defined(__clang__)
    char* begin = static_cast<char*>(code_addr);
    __builtin___clear_cache(begin, begin + sizeof(int64_t));
#endif
    return true;
}

bool brass_patch_call(void* call_site_addr, const void* new_target) {
    if (!call_site_addr || !new_target) return false;
    uint8_t* inst = static_cast<uint8_t*>(call_site_addr);

    uint8_t* disp_ptr = inst;
    uint8_t* next_ip = inst + 5;
    if (*inst == 0xE8) {
        disp_ptr = inst + 1;
        next_ip = inst + 5;
    } else {
        next_ip = inst + 4;
    }

    intptr_t target_int = reinterpret_cast<intptr_t>(new_target);
    intptr_t next_ip_int = reinterpret_cast<intptr_t>(next_ip);
    int64_t disp = static_cast<int64_t>(target_int - next_ip_int);

    if (disp < INT32_MIN || disp > INT32_MAX) {
        return false;
    }

    int32_t disp32 = static_cast<int32_t>(disp);
    auto* target_ptr = reinterpret_cast<int32_t*>(disp_ptr);
    atomic_store_release(target_ptr, disp32);
    memory_fence();
#if defined(_WIN32)
    FlushInstructionCache(GetCurrentProcess(), disp_ptr, sizeof(int32_t));
#elif defined(__GNUC__) || defined(__clang__)
    char* begin = reinterpret_cast<char*>(disp_ptr);
    __builtin___clear_cache(begin, begin + sizeof(int32_t));
#endif
    return true;
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

bool PatchRegistry::patch_call(void* fn_base, std::string_view name, const void* new_target) {
    const auto* site = find_site(name);
    if (!site || site->kind != PatchKind::Call || !fn_base) {
        return false;
    }
    void* call_site = static_cast<uint8_t*>(fn_base) + site->code_offset;
    return brass_patch_call(call_site, new_target);
}

} // namespace brass::runtime
