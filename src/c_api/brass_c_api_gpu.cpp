#include "brass_c_api_internal.hpp"

#include <brass/gpu/cuda_driver.hpp>

#include <cstring>
#include <string>

using namespace brass;

namespace {

void copy_string(const std::string& src, char* out, size_t out_len) {
    if (!out || out_len == 0) return;
    size_t n = src.size();
    if (n >= out_len) n = out_len - 1;
    std::memcpy(out, src.data(), n);
    out[n] = '\0';
}

} // namespace

extern "C" {

int brass_gpu_available(void) {
    return gpu::cuda_available() ? 1 : 0;
}

int brass_gpu_device_count(void) {
    return static_cast<int>(gpu::cuda_devices().size());
}

BrassStatus brass_gpu_device_name(int index, char* out, size_t out_len) {
    if (!out || out_len == 0) return BRASS_ERR_INVALID_ARGUMENT;
    auto devices = gpu::cuda_devices();
    if (index < 0 || static_cast<size_t>(index) >= devices.size()) {
        out[0] = '\0';
        return BRASS_ERR_NOT_FOUND;
    }
    copy_string(devices[static_cast<size_t>(index)].name, out, out_len);
    return BRASS_OK;
}

BrassStatus brass_gpu_last_error(char* out, size_t out_len) {
    if (!out || out_len == 0) return BRASS_ERR_INVALID_ARGUMENT;
    copy_string(gpu::cuda_last_error(), out, out_len);
    return BRASS_OK;
}

BrassGpuBuffer brass_gpu_buffer_alloc(size_t bytes) {
    try {
        auto* buf = new BrassGpuBuffer_T();
        buf->buffer = gpu::CudaBuffer::alloc(bytes);
        if (!buf->buffer.valid() && bytes > 0) {
            delete buf;
            return nullptr;
        }
        return buf;
    } catch (...) {
        return nullptr;
    }
}

void brass_gpu_buffer_destroy(BrassGpuBuffer buf) {
    delete buf;
}

BrassStatus brass_gpu_buffer_upload(BrassGpuBuffer buf, const void* host, size_t bytes, size_t offset) {
    if (!buf || !host) return BRASS_ERR_INVALID_ARGUMENT;
    return buf->buffer.upload(host, bytes, offset) ? BRASS_OK : BRASS_ERR_GENERIC;
}

BrassStatus brass_gpu_buffer_download(BrassGpuBuffer buf, void* host, size_t bytes, size_t offset) {
    if (!buf || !host) return BRASS_ERR_INVALID_ARGUMENT;
    return buf->buffer.download(host, bytes, offset) ? BRASS_OK : BRASS_ERR_GENERIC;
}

void* brass_gpu_buffer_device_ptr(BrassGpuBuffer buf) {
    return buf ? buf->buffer.device_ptr() : nullptr;
}

BrassStatus brass_gpu_module_load(const char* ptx, BrassGpuModule* out_module) {
    if (!ptx || !out_module) return BRASS_ERR_INVALID_ARGUMENT;
    *out_module = nullptr;
    try {
        // CudaModule::load records the JIT log in the thread-local error,
        // retrievable through brass_gpu_last_error().
        gpu::CudaModule mod = gpu::CudaModule::load(ptx);
        if (!mod.valid()) return BRASS_ERR_COMPILE_FAILED;
        auto* handle = new BrassGpuModule_T();
        handle->module = std::move(mod);
        *out_module = handle;
        return BRASS_OK;
    } catch (...) {
        return BRASS_ERR_GENERIC;
    }
}

void brass_gpu_module_destroy(BrassGpuModule mod) {
    delete mod;
}

int brass_gpu_module_has_function(BrassGpuModule mod, const char* entry) {
    if (!mod || !entry) return 0;
    return mod->module.has_function(entry) ? 1 : 0;
}

BrassStatus brass_gpu_module_launch(
    BrassGpuModule mod,
    const char* entry,
    uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z,
    void** kernel_args, size_t shared_bytes) {
    if (!mod || !entry || block_x == 0 || block_y == 0 || block_z == 0) {
        return BRASS_ERR_INVALID_ARGUMENT;
    }
    return mod->module.launch(entry, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                              kernel_args, shared_bytes)
               ? BRASS_OK
               : BRASS_ERR_GENERIC;
}

BrassStatus brass_gpu_synchronize(void) {
    return gpu::cuda_synchronize() ? BRASS_OK : BRASS_ERR_GENERIC;
}

} /* extern "C" */
