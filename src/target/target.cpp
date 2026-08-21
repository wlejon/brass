#include <brass/target/target.hpp>

namespace brass {

Target Target::host() noexcept {
#if defined(_WIN32) || defined(_WIN64)
    return Target::x64_windows();
#elif defined(__APPLE__)
    return Target::x64_macos();
#else
    return Target::x64_linux();
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

} // namespace brass
