#include <brass/target/dynamic_library.hpp>

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

namespace brass::target {

DynamicLibrary::DynamicLibrary(void* handle, std::string path)
    : handle_(handle), path_(std::move(path)) {}

DynamicLibrary::~DynamicLibrary() {
    if (handle_) {
#if defined(_WIN32)
        FreeLibrary(static_cast<HMODULE>(handle_));
#else
        dlclose(handle_);
#endif
        handle_ = nullptr;
    }
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : handle_(other.handle_), path_(std::move(other.path_)) {
    other.handle_ = nullptr;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        if (handle_) {
#if defined(_WIN32)
            FreeLibrary(static_cast<HMODULE>(handle_));
#else
            dlclose(handle_);
#endif
        }
        handle_ = other.handle_;
        path_ = std::move(other.path_);
        other.handle_ = nullptr;
    }
    return *this;
}

std::unique_ptr<DynamicLibrary> DynamicLibrary::open(const std::string& path, std::string* error_out) {
#if defined(_WIN32)
    HMODULE mod = LoadLibraryA(path.c_str());
    if (!mod) {
        if (error_out) {
            DWORD err = GetLastError();
            *error_out = "LoadLibraryA failed with error code " + std::to_string(err);
        }
        return nullptr;
    }
    return std::unique_ptr<DynamicLibrary>(new DynamicLibrary(reinterpret_cast<void*>(mod), path));
#else
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        if (error_out) {
            const char* err = dlerror();
            *error_out = err ? std::string(err) : "Unknown dlopen error";
        }
        return nullptr;
    }
    return std::unique_ptr<DynamicLibrary>(new DynamicLibrary(handle, path));
#endif
}

std::unique_ptr<DynamicLibrary> DynamicLibrary::open(const std::string& path) {
    return open(path, nullptr);
}

void* DynamicLibrary::get_symbol(const std::string& name) const {
    if (!handle_) return nullptr;
#if defined(_WIN32)
    FARPROC proc = GetProcAddress(static_cast<HMODULE>(handle_), name.c_str());
    return reinterpret_cast<void*>(proc);
#else
    void* sym = dlsym(handle_, name.c_str());
#if defined(__APPLE__)
    if (!sym && !name.empty() && name[0] != '_') {
        std::string under = "_" + name;
        sym = dlsym(handle_, under.c_str());
    }
#endif
    return sym;
#endif
}

} // namespace brass::target
