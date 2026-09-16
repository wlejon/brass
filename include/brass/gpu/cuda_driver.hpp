#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Dynamically-loaded CUDA driver runtime. brass has no link-time dependency on
// libcuda / CUDA toolkit: everything is resolved with dlopen at first use and
// every entry point degrades gracefully to "unavailable" when no driver or
// device is present.
namespace brass::gpu {

struct CudaDeviceInfo {
    int index = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    size_t total_memory = 0;
};

// True when libcuda can be loaded and a context can be created on a device.
bool cuda_available();

// Human-readable description of the last driver failure on this thread.
std::string cuda_last_error();

// Enumerate usable CUDA devices. Empty when the driver is unavailable.
std::vector<CudaDeviceInfo> cuda_devices();

// Driver version reported by cuDriverGetVersion (e.g. 12040), or 0.
int cuda_driver_version();

// Owning device-memory allocation. Movable, non-copyable.
class CudaBuffer {
public:
    CudaBuffer() = default;
    ~CudaBuffer();

    CudaBuffer(CudaBuffer&& other) noexcept;
    CudaBuffer& operator=(CudaBuffer&& other) noexcept;
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    static CudaBuffer alloc(size_t bytes);

    void reset();
    bool valid() const noexcept { return ptr_ != nullptr; }
    size_t size() const noexcept { return size_; }
    void* device_ptr() const noexcept { return ptr_; }

    bool upload(const void* host, size_t bytes, size_t offset = 0);
    bool download(void* host, size_t bytes, size_t offset = 0) const;
    bool zero();

private:
    void* ptr_ = nullptr;
    size_t size_ = 0;
};

// A PTX module JIT-compiled by the CUDA driver.
class CudaModule {
public:
    CudaModule() = default;
    ~CudaModule();

    CudaModule(CudaModule&& other) noexcept;
    CudaModule& operator=(CudaModule&& other) noexcept;
    CudaModule(const CudaModule&) = delete;
    CudaModule& operator=(const CudaModule&) = delete;

    static CudaModule load(const std::string& ptx, std::string* error = nullptr);

    bool valid() const noexcept { return module_ != nullptr; }
    void reset();
    bool has_function(std::string_view name) const;

    // Launch `entry` with an argument vector in kernel-parameter order. Each
    // element of `kernel_args` points at the actual argument value (device
    // pointers are passed as a `void*`/64-bit value).
    bool launch(std::string_view entry,
                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                uint32_t block_x, uint32_t block_y, uint32_t block_z,
                void** kernel_args, size_t shared_bytes = 0,
                std::string* error = nullptr) const;

    // Convenience overload for the common 1-D grid/block case.
    bool launch_1d(std::string_view entry,
                   uint32_t grid, uint32_t block,
                   void** kernel_args, size_t shared_bytes = 0,
                   std::string* error = nullptr) const;

private:
    void* module_ = nullptr;
};

bool cuda_synchronize(std::string* error = nullptr);

} // namespace brass::gpu
