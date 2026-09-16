#include <brass/gpu/cuda_driver.hpp>

#include <atomic>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>

#if defined(__linux__)
#  include <dlfcn.h>
#  define BRASS_GPU_HAS_DLOPEN 1
#endif

namespace brass::gpu {

namespace {

// Minimal CUDA driver ABI. We intentionally avoid including <cuda.h> so that
// building brass never requires a CUDA toolkit.
using CUresult = int;
using CUdevice = int;
using CUdeviceptr = unsigned long long;
struct CUctx_st;        using CUcontext = CUctx_st*;
struct CUmod_st;        using CUmodule = CUmod_st*;
struct CUfunc_st;       using CUfunction = CUfunc_st*;
struct CUstream_st;     using CUstream = CUstream_st*;

constexpr CUresult CUDA_SUCCESS = 0;

// cuDeviceGetAttribute attribute ids
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
constexpr int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;

// cuModuleLoadDataEx option ids
constexpr int CU_JIT_ERROR_LOG_BUFFER = 5;
constexpr int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;

struct DriverApi {
    void* handle = nullptr;
    bool available = false;

    CUresult (*cuInit)(unsigned int) = nullptr;
    CUresult (*cuDriverGetVersion)(int*) = nullptr;
    CUresult (*cuDeviceGetCount)(int*) = nullptr;
    CUresult (*cuDeviceGet)(CUdevice*, int) = nullptr;
    CUresult (*cuDeviceGetName)(char*, int, CUdevice) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice) = nullptr;
    CUresult (*cuCtxCreate_v2)(CUcontext*, unsigned int, CUdevice) = nullptr;
    CUresult (*cuCtxDestroy_v2)(CUcontext) = nullptr;
    CUresult (*cuCtxSetCurrent)(CUcontext) = nullptr;
    CUresult (*cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, int*, void**) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int, unsigned int,
                               CUstream, void**, void**) = nullptr;
    CUresult (*cuMemAlloc_v2)(CUdeviceptr*, size_t) = nullptr;
    CUresult (*cuMemFree_v2)(CUdeviceptr) = nullptr;
    CUresult (*cuMemcpyHtoD_v2)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult (*cuMemcpyDtoH_v2)(void*, CUdeviceptr, size_t) = nullptr;
    CUresult (*cuMemsetD8_v2)(CUdeviceptr, unsigned char, size_t) = nullptr;
    CUresult (*cuMemGetInfo_v2)(size_t*, size_t*) = nullptr;
    CUresult (*cuCtxSynchronize)() = nullptr;
    CUresult (*cuGetErrorString)(CUresult, const char**) = nullptr;
};

DriverApi& api() {
    static DriverApi a;
    return a;
}

std::recursive_mutex& api_mutex() {
    static std::recursive_mutex m;
    return m;
}

thread_local std::string tls_last_error;

void set_error(const std::string& msg) { tls_last_error = msg; }

bool init_driver() {
    std::lock_guard<std::recursive_mutex> lock(api_mutex());
    DriverApi& a = api();
    if (a.handle) return a.available;
#if defined(BRASS_GPU_HAS_DLOPEN)
    const char* names[] = { "libcuda.so.1", "libcuda.so", nullptr };
    for (int i = 0; names[i]; ++i) {
        a.handle = dlopen(names[i], RTLD_NOW | RTLD_GLOBAL);
        if (a.handle) break;
    }
    if (!a.handle) {
        set_error("libcuda could not be loaded (no NVIDIA driver)");
        return false;
    }

    auto sym = [&](const char* name) -> void* { return dlsym(a.handle, name); };

    a.cuInit = reinterpret_cast<CUresult(*)(unsigned int)>(sym("cuInit"));
    a.cuDriverGetVersion = reinterpret_cast<CUresult(*)(int*)>(sym("cuDriverGetVersion"));
    a.cuDeviceGetCount = reinterpret_cast<CUresult(*)(int*)>(sym("cuDeviceGetCount"));
    a.cuDeviceGet = reinterpret_cast<CUresult(*)(CUdevice*, int)>(sym("cuDeviceGet"));
    a.cuDeviceGetName = reinterpret_cast<CUresult(*)(char*, int, CUdevice)>(sym("cuDeviceGetName"));
    a.cuDeviceGetAttribute = reinterpret_cast<CUresult(*)(int*, int, CUdevice)>(sym("cuDeviceGetAttribute"));
    a.cuCtxCreate_v2 = reinterpret_cast<CUresult(*)(CUcontext*, unsigned int, CUdevice)>(sym("cuCtxCreate_v2"));
    a.cuCtxDestroy_v2 = reinterpret_cast<CUresult(*)(CUcontext)>(sym("cuCtxDestroy_v2"));
    a.cuCtxSetCurrent = reinterpret_cast<CUresult(*)(CUcontext)>(sym("cuCtxSetCurrent"));
    a.cuModuleLoadDataEx = reinterpret_cast<CUresult(*)(CUmodule*, const void*, unsigned int, int*, void**)>(sym("cuModuleLoadDataEx"));
    a.cuModuleUnload = reinterpret_cast<CUresult(*)(CUmodule)>(sym("cuModuleUnload"));
    a.cuModuleGetFunction = reinterpret_cast<CUresult(*)(CUfunction*, CUmodule, const char*)>(sym("cuModuleGetFunction"));
    a.cuLaunchKernel = reinterpret_cast<CUresult(*)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, CUstream, void**, void**)>(sym("cuLaunchKernel"));
    a.cuMemAlloc_v2 = reinterpret_cast<CUresult(*)(CUdeviceptr*, size_t)>(sym("cuMemAlloc_v2"));
    a.cuMemFree_v2 = reinterpret_cast<CUresult(*)(CUdeviceptr)>(sym("cuMemFree_v2"));
    a.cuMemcpyHtoD_v2 = reinterpret_cast<CUresult(*)(CUdeviceptr, const void*, size_t)>(sym("cuMemcpyHtoD_v2"));
    a.cuMemcpyDtoH_v2 = reinterpret_cast<CUresult(*)(void*, CUdeviceptr, size_t)>(sym("cuMemcpyDtoH_v2"));
    a.cuMemsetD8_v2 = reinterpret_cast<CUresult(*)(CUdeviceptr, unsigned char, size_t)>(sym("cuMemsetD8_v2"));
    a.cuMemGetInfo_v2 = reinterpret_cast<CUresult(*)(size_t*, size_t*)>(sym("cuMemGetInfo_v2"));
    a.cuCtxSynchronize = reinterpret_cast<CUresult(*)()>(sym("cuCtxSynchronize"));
    a.cuGetErrorString = reinterpret_cast<CUresult(*)(CUresult, const char**)>(sym("cuGetErrorString"));

    if (!a.cuInit || !a.cuDeviceGetCount || !a.cuDeviceGet || !a.cuCtxCreate_v2 ||
        !a.cuModuleLoadDataEx || !a.cuModuleGetFunction || !a.cuLaunchKernel ||
        !a.cuMemAlloc_v2 || !a.cuMemFree_v2 || !a.cuMemcpyHtoD_v2 || !a.cuMemcpyDtoH_v2) {
        set_error("libcuda is missing required driver entry points");
        return false;
    }

    CUresult r = a.cuInit(0);
    if (r != CUDA_SUCCESS) {
        set_error("cuInit failed");
        return false;
    }
    a.available = true;
    return true;
#else
    set_error("GPU execution is only supported on Linux in this build");
    return false;
#endif
}

std::string error_string(CUresult r) {
    DriverApi& a = api();
    if (a.cuGetErrorString && r != CUDA_SUCCESS) {
        const char* s = nullptr;
        if (a.cuGetErrorString(r, &s) == CUDA_SUCCESS && s) return s;
    }
    return "CUDA error " + std::to_string(r);
}

// Lazily-created process-wide context, made current before every driver call so
// that modules/buffers created on one thread can be used from another.
struct Context {
    CUcontext ctx = nullptr;
    bool tried = false;
    bool ok = false;
};

Context& context() {
    static Context c;
    return c;
}

bool ensure_context() {
    std::lock_guard<std::recursive_mutex> lock(api_mutex());
    Context& c = context();
    if (c.tried) {
        if (c.ok && api().cuCtxSetCurrent) api().cuCtxSetCurrent(c.ctx);
        return c.ok;
    }
    c.tried = true;
    if (!init_driver()) return false;

    int count = 0;
    if (api().cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) {
        set_error("no CUDA devices found");
        c.ok = false;
        return false;
    }

    CUdevice dev = 0;
    if (api().cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        set_error("cuDeviceGet failed");
        c.ok = false;
        return false;
    }

    CUcontext ctx = nullptr;
    if (api().cuCtxCreate_v2(&ctx, 0, dev) != CUDA_SUCCESS) {
        set_error("cuCtxCreate failed");
        c.ok = false;
        return false;
    }
    c.ctx = ctx;
    c.ok = true;
    return true;
}

bool make_current() {
    if (!ensure_context()) return false;
    // cuCtxCreate makes the context current on the creating thread only.
    if (api().cuCtxSetCurrent && api().cuCtxSetCurrent(context().ctx) != CUDA_SUCCESS) {
        set_error("cuCtxSetCurrent failed");
        return false;
    }
    return true;
}

} // namespace

bool cuda_available() {
    return ensure_context();
}

std::string cuda_last_error() { return tls_last_error; }

int cuda_driver_version() {
    if (!init_driver()) return 0;
    int v = 0;
    if (api().cuDriverGetVersion) api().cuDriverGetVersion(&v);
    return v;
}

std::vector<CudaDeviceInfo> cuda_devices() {
    std::vector<CudaDeviceInfo> out;
    if (!init_driver()) return out;

    int count = 0;
    if (api().cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) return out;
    out.reserve(static_cast<size_t>(count));

    for (int i = 0; i < count; ++i) {
        CUdevice dev = 0;
        if (api().cuDeviceGet(&dev, i) != CUDA_SUCCESS) continue;
        CudaDeviceInfo info;
        info.index = i;
        char name[256] = {0};
        if (api().cuDeviceGetName && api().cuDeviceGetName(name, sizeof(name), dev) == CUDA_SUCCESS) {
            info.name = name;
        }
        if (api().cuDeviceGetAttribute) {
            api().cuDeviceGetAttribute(&info.compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
            api().cuDeviceGetAttribute(&info.compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
        }
        out.push_back(std::move(info));
    }

    // Total memory is per-context; report it for the first device only if we
    // can query it without disturbing the primary context.
    if (ensure_context() && api().cuMemGetInfo_v2 && !out.empty()) {
        size_t free_b = 0, total_b = 0;
        if (api().cuMemGetInfo_v2(&free_b, &total_b) == CUDA_SUCCESS) out[0].total_memory = total_b;
    }
    return out;
}

// ---------------------------------------------------------------------------
// CudaBuffer
// ---------------------------------------------------------------------------

CudaBuffer::~CudaBuffer() { reset(); }

CudaBuffer::CudaBuffer(CudaBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

CudaBuffer& CudaBuffer::operator=(CudaBuffer&& other) noexcept {
    if (this != &other) {
        reset();
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

CudaBuffer CudaBuffer::alloc(size_t bytes) {
    CudaBuffer buf;
    if (bytes == 0) return buf;
    if (!make_current()) return buf;
    CUdeviceptr dptr = 0;
    if (api().cuMemAlloc_v2(&dptr, bytes) != CUDA_SUCCESS) {
        set_error("cuMemAlloc failed for " + std::to_string(bytes) + " bytes");
        return buf;
    }
    buf.ptr_ = reinterpret_cast<void*>(dptr);
    buf.size_ = bytes;
    return buf;
}

void CudaBuffer::reset() {
    if (ptr_) {
        if (make_current()) {
            CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr_);
            api().cuMemFree_v2(dptr);
        }
        ptr_ = nullptr;
        size_ = 0;
    }
}

bool CudaBuffer::upload(const void* host, size_t bytes, size_t offset) {
    if (!ptr_ || offset + bytes > size_) {
        set_error("CudaBuffer::upload out of range");
        return false;
    }
    if (!make_current()) return false;
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr_) + offset;
    if (api().cuMemcpyHtoD_v2(dptr, host, bytes) != CUDA_SUCCESS) {
        set_error("cuMemcpyHtoD failed");
        return false;
    }
    return true;
}

bool CudaBuffer::download(void* host, size_t bytes, size_t offset) const {
    if (!ptr_ || offset + bytes > size_) {
        set_error("CudaBuffer::download out of range");
        return false;
    }
    if (!make_current()) return false;
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr_) + offset;
    if (api().cuMemcpyDtoH_v2(host, dptr, bytes) != CUDA_SUCCESS) {
        set_error("cuMemcpyDtoH failed");
        return false;
    }
    return true;
}

bool CudaBuffer::zero() {
    if (!ptr_) return false;
    if (!make_current()) return false;
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr_);
    if (!api().cuMemsetD8_v2 || api().cuMemsetD8_v2(dptr, 0, size_) != CUDA_SUCCESS) {
        set_error("cuMemsetD8 failed");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// CudaModule
// ---------------------------------------------------------------------------

CudaModule::~CudaModule() { reset(); }

CudaModule::CudaModule(CudaModule&& other) noexcept : module_(other.module_) {
    other.module_ = nullptr;
}

CudaModule& CudaModule::operator=(CudaModule&& other) noexcept {
    if (this != &other) {
        reset();
        module_ = other.module_;
        other.module_ = nullptr;
    }
    return *this;
}

CudaModule CudaModule::load(const std::string& ptx, std::string* error) {
    CudaModule out;
    if (!make_current()) {
        if (error) *error = cuda_last_error();
        return out;
    }

    char error_log[8192] = {0};
    int option_ids[2] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    void* option_vals[2];
    option_vals[0] = static_cast<void*>(error_log);
    option_vals[1] = reinterpret_cast<void*>(static_cast<uintptr_t>(sizeof(error_log)));

    CUmodule mod = nullptr;
    CUresult r = api().cuModuleLoadDataEx(&mod, ptx.c_str(), 2, option_ids, option_vals);
    if (r != CUDA_SUCCESS || mod == nullptr) {
        std::string msg = error_string(r);
        if (error_log[0]) {
            msg += ": ";
            msg += error_log;
        }
        set_error("PTX module load failed: " + msg);
        if (error) *error = msg;
        return out;
    }
    out.module_ = reinterpret_cast<void*>(mod);
    return out;
}

void CudaModule::reset() {
    if (module_) {
        if (make_current()) {
            api().cuModuleUnload(reinterpret_cast<CUmodule>(module_));
        }
        module_ = nullptr;
    }
}

bool CudaModule::has_function(std::string_view name) const {
    if (!module_) return false;
    if (!make_current()) return false;
    std::string n(name);
    CUfunction fn = nullptr;
    return api().cuModuleGetFunction(&fn, reinterpret_cast<CUmodule>(module_), n.c_str()) == CUDA_SUCCESS && fn != nullptr;
}

bool CudaModule::launch(std::string_view entry,
                        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                        uint32_t block_x, uint32_t block_y, uint32_t block_z,
                        void** kernel_args, size_t shared_bytes,
                        std::string* error) const {
    if (!module_) {
        set_error("launch on invalid module");
        if (error) *error = cuda_last_error();
        return false;
    }
    if (!make_current()) {
        if (error) *error = cuda_last_error();
        return false;
    }
    std::string n(entry);
    CUfunction fn = nullptr;
    CUresult r = api().cuModuleGetFunction(&fn, reinterpret_cast<CUmodule>(module_), n.c_str());
    if (r != CUDA_SUCCESS || fn == nullptr) {
        std::string msg = "kernel '" + n + "' not found: " + error_string(r);
        set_error(msg);
        if (error) *error = msg;
        return false;
    }
    r = api().cuLaunchKernel(fn, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                             static_cast<unsigned int>(shared_bytes), nullptr,
                             kernel_args, nullptr);
    if (r != CUDA_SUCCESS) {
        std::string msg = "cuLaunchKernel failed: " + error_string(r);
        set_error(msg);
        if (error) *error = msg;
        return false;
    }
    return true;
}

bool CudaModule::launch_1d(std::string_view entry, uint32_t grid, uint32_t block,
                           void** kernel_args, size_t shared_bytes,
                           std::string* error) const {
    return launch(entry, grid, 1, 1, block, 1, 1, kernel_args, shared_bytes, error);
}

bool cuda_synchronize(std::string* error) {
    if (!make_current()) {
        if (error) *error = cuda_last_error();
        return false;
    }
    CUresult r = api().cuCtxSynchronize();
    if (r != CUDA_SUCCESS) {
        std::string msg = "cuCtxSynchronize failed: " + error_string(r);
        set_error(msg);
        if (error) *error = msg;
        return false;
    }
    return true;
}

} // namespace brass::gpu
