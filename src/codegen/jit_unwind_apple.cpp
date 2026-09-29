#include <brass/codegen/jit_unwind_apple.hpp>

#if defined(__APPLE__)
#include <dlfcn.h>
#include <mach-o/loader.h>
#include <mutex>
#include <vector>
#include <algorithm>
#include <cstring>

extern "C" const struct mach_header* _NSGetMachExecuteHeader();

namespace brass::codegen {

namespace {

struct unw_dynamic_unwind_sections {
    uintptr_t dso_base;
    uintptr_t dwarf_section;
    size_t dwarf_section_length;
    uintptr_t compact_unwind_section;
    size_t compact_unwind_section_length;
};

typedef int (*unw_find_dynamic_unwind_sections)(uintptr_t addr, struct unw_dynamic_unwind_sections *info);
typedef int (*unw_add_find_dynamic_unwind_sections_fn)(unw_find_dynamic_unwind_sections);
typedef int (*unw_remove_find_dynamic_unwind_sections_fn)(unw_find_dynamic_unwind_sections);

struct DynamicUnwindEntry {
    uintptr_t code_start = 0;
    uintptr_t code_end = 0;
    uintptr_t eh_frame = 0;
    size_t eh_frame_length = 0;
};

struct DynamicUnwindRegistry {
    std::mutex mutex;
    std::vector<DynamicUnwindEntry> entries;
    bool callback_registered = false;
    unw_add_find_dynamic_unwind_sections_fn add_fn = nullptr;
    unw_remove_find_dynamic_unwind_sections_fn remove_fn = nullptr;
};

DynamicUnwindRegistry& registry() {
    static auto* r = new DynamicUnwindRegistry();
    return *r;
}

int brass_find_dynamic_unwind_sections_callback(uintptr_t addr, struct unw_dynamic_unwind_sections* info) {
    if (!info) return 0;
    DynamicUnwindRegistry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    for (const auto& entry : r.entries) {
        if (addr >= entry.code_start && addr < entry.code_end) {
            info->dso_base = reinterpret_cast<uintptr_t>(_NSGetMachExecuteHeader());
            info->dwarf_section = entry.eh_frame;
            info->dwarf_section_length = entry.eh_frame_length;
            info->compact_unwind_section = 0;
            info->compact_unwind_section_length = 0;
            return 1;
        }
    }
    return 0;
}

size_t scan_eh_frame_length(const uint8_t* p) {
    const uint8_t* start = p;
    for (;;) {
        uint32_t len = 0;
        std::memcpy(&len, p, 4);
        if (len == 0) {
            p += 4;
            break;
        }
        if (len == 0xFFFFFFFFu) break;
        p += 4 + len;
    }
    return static_cast<size_t>(p - start);
}

} // namespace

bool has_apple_dynamic_unwind() noexcept {
    static const bool supported = []() {
        void* add_sym = dlsym(RTLD_DEFAULT, "__unw_add_find_dynamic_unwind_sections");
        void* rem_sym = dlsym(RTLD_DEFAULT, "__unw_remove_find_dynamic_unwind_sections");
        if (add_sym && rem_sym) {
            DynamicUnwindRegistry& r = registry();
            r.add_fn = reinterpret_cast<unw_add_find_dynamic_unwind_sections_fn>(add_sym);
            r.remove_fn = reinterpret_cast<unw_remove_find_dynamic_unwind_sections_fn>(rem_sym);
            return true;
        }
        return false;
    }();
    return supported;
}

bool register_apple_dynamic_unwind(const void* code_start, size_t code_size,
                                   const void* eh_frame, size_t eh_frame_size) noexcept {
    if (!code_start || code_size == 0 || !eh_frame) return false;
    if (!has_apple_dynamic_unwind()) return false;

    DynamicUnwindRegistry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);

    if (!r.callback_registered) {
        if (r.add_fn(brass_find_dynamic_unwind_sections_callback) != 0) {
            return false;
        }
        r.callback_registered = true;
    }

    if (eh_frame_size == 0) {
        eh_frame_size = scan_eh_frame_length(static_cast<const uint8_t*>(eh_frame));
    }

    DynamicUnwindEntry entry;
    entry.code_start = reinterpret_cast<uintptr_t>(code_start);
    entry.code_end = entry.code_start + code_size;
    entry.eh_frame = reinterpret_cast<uintptr_t>(eh_frame);
    entry.eh_frame_length = eh_frame_size;
    r.entries.push_back(entry);
    return true;
}

void unregister_apple_dynamic_unwind(const void* ptr) noexcept {
    if (!ptr) return;
    DynamicUnwindRegistry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    r.entries.erase(
        std::remove_if(r.entries.begin(), r.entries.end(),
                       [p](const DynamicUnwindEntry& e) {
                           return e.code_start == p || e.eh_frame == p;
                       }),
        r.entries.end());
}

} // namespace brass::codegen
#endif
