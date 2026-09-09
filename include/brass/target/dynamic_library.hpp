#pragma once

#include <string>
#include <memory>
#include <cstdint>

namespace brass::target {

class DynamicLibrary {
public:
    static std::unique_ptr<DynamicLibrary> open(const std::string& path, std::string* error_out);
    static std::unique_ptr<DynamicLibrary> open(const std::string& path);

    ~DynamicLibrary();

    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    DynamicLibrary(DynamicLibrary&& other) noexcept;
    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept;

    void* get_symbol(const std::string& name) const;

    template <typename Fn>
    Fn get_function(const std::string& name) const {
        return reinterpret_cast<Fn>(get_symbol(name));
    }

    bool is_valid() const noexcept { return handle_ != nullptr; }
    const std::string& path() const noexcept { return path_; }
    void* native_handle() const noexcept { return handle_; }

private:
    DynamicLibrary(void* handle, std::string path);

    void* handle_ = nullptr;
    std::string path_;
};

} // namespace brass::target
