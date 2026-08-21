#include <brass/runtime/patcher.hpp>
#include <atomic>
#include <iostream>

#if defined(_MSC_VER)
#include <intrin.h>
#include <immintrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace {
static inline void memory_fence() noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    _mm_mfence();
#else
    std::atomic_thread_fence(std::memory_order_seq_cst);
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
    std::atomic_ref<int32_t> ref(*target_ptr);
    ref.store(new_val, std::memory_order_release);
    memory_fence();
    return true;
}

bool brass_patch_const64(void* code_addr, int64_t new_val) {
    if (!code_addr) return false;
    auto* target_ptr = reinterpret_cast<int64_t*>(code_addr);
    std::atomic_ref<int64_t> ref(*target_ptr);
    ref.store(new_val, std::memory_order_release);
    memory_fence();
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
    std::atomic_ref<int32_t> ref(*target_ptr);
    ref.store(disp32, std::memory_order_release);
    memory_fence();
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
