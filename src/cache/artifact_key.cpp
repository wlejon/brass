// Cache keys: named, length-prefixed fields under two seeds, and what brass
// itself contributes to a key (artifact_cache.hpp, docs/artifact_cache.md).
#include <brass/brass.hpp>
#include <brass/cache/artifact_cache.hpp>
#include <brass/target/host_cpu.hpp>

#include "cache/artifact_format.hpp"

#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace brass::cache {

namespace {

// The two seeds: `id` names the file, `check` sits in its header. Any two
// distinct constants do; these spell "brassid" and "brasschk".
constexpr uint64_t kIdSeed = 0x0064697373617262ULL;
constexpr uint64_t kCheckSeed = 0x6b68637373617262ULL;

std::filesystem::path module_of_this_code() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&module_of_this_code), &module)) {
        return {};
    }
    std::wstring buf(1024, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(module, buf.data(), static_cast<DWORD>(buf.size()));
        if (n == 0) return {};
        if (n < buf.size()) {
            buf.resize(n);
            return std::filesystem::path(buf);
        }
        buf.resize(buf.size() * 2);
    }
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&module_of_this_code), &info) && info.dli_fname && info.dli_fname[0]) {
        return std::filesystem::path(info.dli_fname);
    }
    return {};
#endif
}

std::string measure_binary_identity() {
    const std::filesystem::path self = module_of_this_code();
    if (self.empty()) return {};
    std::error_code ec;
    const std::filesystem::path abs = std::filesystem::absolute(self, ec);
    const std::filesystem::path& p = ec ? self : abs;
    const auto size = std::filesystem::file_size(p, ec);
    if (ec) return {};
    const auto mtime = std::filesystem::last_write_time(p, ec);
    if (ec) return {};
    return p.generic_string() + "|" + std::to_string(size) + "|" +
           std::to_string(static_cast<long long>(mtime.time_since_epoch().count()));
}

} // namespace

const std::string& brass_binary_identity() {
    static const std::string* const identity = new std::string(measure_binary_identity());
    return *identity;
}

KeyBuilder::KeyBuilder(std::string_view domain) : id_(kIdSeed), check_(kCheckSeed) {
    add("domain", domain);
}

void KeyBuilder::field(std::string_view name, const void* data, size_t size) {
    for (Hasher128* h : {&id_, &check_}) {
        h->update_u64(name.size());
        h->update(name);
        h->update_u64(size);
        h->update(data, size);
    }
}

KeyBuilder& KeyBuilder::add(std::string_view name, std::string_view value) {
    field(name, value.data(), value.size());
    return *this;
}

KeyBuilder& KeyBuilder::add_bytes(std::string_view name, std::span<const uint8_t> value) {
    field(name, value.data(), value.size());
    return *this;
}

KeyBuilder& KeyBuilder::add_u64(std::string_view name, uint64_t value) {
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(value >> (i * 8));
    field(name, b, 8);
    return *this;
}

KeyBuilder& KeyBuilder::add_bool(std::string_view name, bool value) {
    const unsigned char b = value ? 1 : 0;
    field(name, &b, 1);
    return *this;
}

KeyBuilder& KeyBuilder::add_digest(std::string_view name, const Digest128& value) {
    unsigned char b[16];
    for (int i = 0; i < 8; ++i) {
        b[i] = static_cast<unsigned char>(value.lo >> (i * 8));
        b[8 + i] = static_cast<unsigned char>(value.hi >> (i * 8));
    }
    field(name, b, 16);
    return *this;
}

KeyBuilder& KeyBuilder::add_key(std::string_view name, const CacheKey& value) {
    if (!value.valid) valid_ = false;
    add_digest(name, value.id);
    add_digest(name, value.check);
    return *this;
}

KeyBuilder& KeyBuilder::add_brass_codegen(const Target& target) {
    add_u64("brass.cache_format", kFormatVersion);
    add("brass.version", version_string());
    add("brass.target.arch", to_string(target.arch()));
    add("brass.target.os", to_string(target.os()));
    add("brass.target.object", to_string(target.object_format()));
    add_u64("brass.target.pointer", target.pointer_size());
    add_u64("brass.target.stack_align", target.stack_alignment());
    // What codegen for this machine reads from the CPU: the JIT and an AOT
    // build for the host use AVX scalar forms, and the MIR pipeline
    // vectorizes for AVX2 and fuses for FMA, only when the CPU has them.
    // Code for another target assumes none of them.
    const bool host = target == Target::host();
    add_bool("brass.cpu.avx", host && Target::host_has_avx());
    add_bool("brass.cpu.avx2", host && host_has_avx2());
    add_bool("brass.cpu.fma", host && host_has_fma());
    return *this;
}

KeyBuilder& KeyBuilder::add_brass_build(const Target& target) {
    add_brass_codegen(target);
    const std::string& identity = brass_binary_identity();
    if (identity.empty()) valid_ = false;
    add("brass.binary", identity);
    return *this;
}

CacheKey KeyBuilder::finish() const noexcept {
    CacheKey key;
    key.id = id_.finish();
    key.check = check_.finish();
    key.valid = valid_;
    return key;
}

} // namespace brass::cache
