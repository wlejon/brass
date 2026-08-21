#pragma once

#include <cstdint>
#include <string_view>
#include <iosfwd>

namespace brass {

enum class Arch : uint8_t {
    x64,
    aarch64, // reserved
};

enum class OperatingSystem : uint8_t {
    Windows,
    Linux,
    macOS, // reserved
};

enum class ObjectFormat : uint8_t {
    COFF,
    ELF64,
    MachO, // reserved
};

class Target {
public:
    constexpr Target(Arch target_arch, OperatingSystem target_os, ObjectFormat target_obj_format) noexcept
        : arch_(target_arch), os_(target_os), obj_format_(target_obj_format), pointer_size_(8), stack_alignment_(16) {}

    constexpr Target(Arch target_arch, OperatingSystem target_os, ObjectFormat target_obj_format, uint8_t ptr_size, uint8_t stack_align) noexcept
        : arch_(target_arch), os_(target_os), obj_format_(target_obj_format), pointer_size_(ptr_size), stack_alignment_(stack_align) {}

    constexpr Arch arch() const noexcept { return arch_; }
    constexpr OperatingSystem os() const noexcept { return os_; }
    constexpr ObjectFormat object_format() const noexcept { return obj_format_; }
    constexpr uint8_t pointer_size() const noexcept { return pointer_size_; }
    constexpr uint8_t stack_alignment() const noexcept { return stack_alignment_; }

    constexpr bool is_x64() const noexcept { return arch_ == Arch::x64; }
    constexpr bool is_aarch64() const noexcept { return arch_ == Arch::aarch64; }
    constexpr bool is_windows() const noexcept { return os_ == OperatingSystem::Windows; }
    constexpr bool is_linux() const noexcept { return os_ == OperatingSystem::Linux; }
    constexpr bool is_macos() const noexcept { return os_ == OperatingSystem::macOS; }

    constexpr bool is_64bit() const noexcept { return pointer_size_ == 8; }

    static constexpr Target x64_windows() noexcept {
        return Target(Arch::x64, OperatingSystem::Windows, ObjectFormat::COFF, 8, 16);
    }

    static constexpr Target x64_linux() noexcept {
        return Target(Arch::x64, OperatingSystem::Linux, ObjectFormat::ELF64, 8, 16);
    }

    static constexpr Target x64_macos() noexcept {
        return Target(Arch::x64, OperatingSystem::macOS, ObjectFormat::MachO, 8, 16);
    }

    static Target host() noexcept;

    constexpr bool operator==(const Target& other) const noexcept = default;
    constexpr bool operator!=(const Target& other) const noexcept = default;

private:
    Arch arch_ = Arch::x64;
    OperatingSystem os_ = OperatingSystem::Windows;
    ObjectFormat obj_format_ = ObjectFormat::COFF;
    uint8_t pointer_size_ = 8;
    uint8_t stack_alignment_ = 16;
};

std::string_view to_string(Arch arch) noexcept;
std::string_view to_string(OperatingSystem os) noexcept;
std::string_view to_string(ObjectFormat fmt) noexcept;

std::ostream& operator<<(std::ostream& os, Arch arch);
std::ostream& operator<<(std::ostream& os, OperatingSystem s);
std::ostream& operator<<(std::ostream& os, ObjectFormat fmt);

} // namespace brass
