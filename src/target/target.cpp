#include <brass/target/target.hpp>
#include <ostream>
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#include <immintrin.h>
#endif

namespace brass {

Target Target::host() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(_WIN32) || defined(_WIN64)
    return Target::aarch64_windows();
#elif defined(__APPLE__)
    return Target::aarch64_macos();
#else
    return Target::aarch64_linux();
#endif
#else
#if defined(_WIN32) || defined(_WIN64)
    return Target::x64_windows();
#elif defined(__APPLE__)
    return Target::x64_macos();
#else
    return Target::x64_linux();
#endif
#endif
}

bool Target::host_has_avx() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
    // libgcc's check includes OSXSAVE and XCR0, not just the CPUID bit.
    static const bool supported = __builtin_cpu_supports("avx") != 0;
    return supported;
#elif defined(_MSC_VER)
    static const bool supported = []() noexcept {
        int info[4] = {0, 0, 0, 0};
        __cpuid(info, 1);
        const bool osxsave = (info[2] & (1 << 27)) != 0;
        const bool avx = (info[2] & (1 << 28)) != 0;
        if (!osxsave || !avx) return false;
        return (_xgetbv(0) & 0x6ULL) == 0x6ULL;
    }();
    return supported;
#else
    return false;
#endif
#else
    return false;
#endif
}

std::string_view to_string(Arch arch) noexcept {
    switch (arch) {
    case Arch::x64: return "x64";
    case Arch::aarch64: return "aarch64";
    default: return "unknown";
    }
}

std::string_view to_string(OperatingSystem os) noexcept {
    switch (os) {
    case OperatingSystem::Windows: return "Windows";
    case OperatingSystem::Linux: return "Linux";
    case OperatingSystem::macOS: return "macOS";
    default: return "unknown";
    }
}

std::string_view to_string(ObjectFormat fmt) noexcept {
    switch (fmt) {
    case ObjectFormat::COFF: return "COFF";
    case ObjectFormat::ELF64: return "ELF64";
    case ObjectFormat::MachO: return "MachO";
    default: return "unknown";
    }
}

std::ostream& operator<<(std::ostream& os, Arch arch) {
    return os << to_string(arch);
}

std::ostream& operator<<(std::ostream& os, OperatingSystem s) {
    return os << to_string(s);
}

std::ostream& operator<<(std::ostream& os, ObjectFormat fmt) {
    return os << to_string(fmt);
}

} // namespace brass
