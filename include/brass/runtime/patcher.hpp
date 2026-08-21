#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <iosfwd>

namespace brass::runtime {

enum class PatchKind : uint8_t {
    Const32,
    Const64,
    Call
};

std::string_view to_string(PatchKind kind) noexcept;
std::ostream& operator<<(std::ostream& os, PatchKind kind);

struct PatchSite {
    std::string name;
    PatchKind kind = PatchKind::Const32;
    size_t code_offset = 0;   // Offset in function code where instruction starts
    size_t imm_offset = 0;    // Offset from instruction start to immediate/displacement bytes
    size_t inst_size = 0;     // Total size of instruction
    int64_t initial_value = 0;
    std::string target_symbol;

    constexpr PatchSite() noexcept = default;
    PatchSite(std::string n, PatchKind k, size_t off, size_t imm_off, size_t sz, int64_t init_val = 0, std::string target = "")
        : name(std::move(n)), kind(k), code_offset(off), imm_offset(imm_off), inst_size(sz), initial_value(init_val), target_symbol(std::move(target)) {}

    bool operator==(const PatchSite& other) const noexcept = default;
};

inline size_t compute_cache_line_padding(size_t current_offset, size_t imm_offset, size_t imm_size) noexcept {
    size_t imm_pos = current_offset + imm_offset;
    size_t align = (imm_size >= 8) ? 8 : 4;
    size_t rem = imm_pos % align;
    return (rem == 0) ? 0 : (align - rem);
}

inline bool is_cache_line_safe(const void* addr, size_t size) noexcept {
    if (!addr || size == 0) return true;
    uintptr_t start = reinterpret_cast<uintptr_t>(addr);
    uintptr_t end = start + size - 1;
    return (start / 64) == (end / 64);
}

} // namespace brass::runtime

extern "C" {
    bool brass_patch_const32(void* code_addr, int32_t new_val);
    bool brass_patch_const64(void* code_addr, int64_t new_val);
    bool brass_patch_call(void* call_site_addr, const void* new_target);
}

namespace brass::runtime {

using ::brass_patch_const32;
using ::brass_patch_const64;
using ::brass_patch_call;

class PatchRegistry {
public:
    PatchRegistry() = default;

    void register_site(PatchSite site);
    const PatchSite* find_site(std::string_view name) const noexcept;
    PatchSite* find_site(std::string_view name) noexcept;
    bool has_site(std::string_view name) const noexcept;

    bool patch_const32(void* fn_base, std::string_view name, int32_t new_val);
    bool patch_const64(void* fn_base, std::string_view name, int64_t new_val);
    bool patch_call(void* fn_base, std::string_view name, const void* new_target);

    const std::vector<PatchSite>& sites() const noexcept { return sites_; }
    size_t size() const noexcept { return sites_.size(); }
    bool empty() const noexcept { return sites_.empty(); }
    void clear() noexcept {
        sites_.clear();
        site_map_.clear();
    }

private:
    std::vector<PatchSite> sites_;
    std::unordered_map<std::string, size_t> site_map_;
};

} // namespace brass::runtime
