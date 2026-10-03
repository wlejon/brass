#pragma once

// Minimal Vulkan compute runtime for SpirvTarget kernels: the test and
// benchmark harness for the SPIR-V backend, and the reference for how an
// embedder's own Vulkan runtime drives these kernels (the contract is in
// docs/spirv_backend_design.md, "Kernel ABI" and "Vulkan runtime").
//
// Like cuda_driver, brass has no link-time dependency on Vulkan: libvulkan is
// dlopen'ed at first use, and every entry point degrades to "unavailable"
// when there is no loader or no device (or when brass was built without the
// Vulkan headers). One process-wide device, one compute queue, synchronous
// submissions -- not a general runtime.
//
// Memory model: every buffer is device-local with SHADER_DEVICE_ADDRESS usage
// (memory allocated with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT); a kernel's
// `ptr` parameters are the buffers' device addresses in the push-constant
// block. Uploads and downloads go through a temporary host-visible staging
// buffer and vkCmdCopyBuffer.

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace brass::target {
struct SpirvKernel;
}

namespace brass::gpu {

struct VulkanDeviceInfo {
    uint32_t index = 0;          // position in vkEnumeratePhysicalDevices order
    std::string name;
    std::string type;            // "discrete", "integrated", "virtual", "cpu", "other"
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    uint32_t api_version = 0;    // VK_MAKE_API_VERSION packed
};

// What the selected device can do, and what the runtime enabled. A kernel's
// SpirvKernel::capabilities / extensions are checked against this before a
// pipeline is created (`missing_for`).
struct VulkanDeviceCaps {
    VulkanDeviceInfo info;
    std::string driver_name;     // VkPhysicalDeviceDriverProperties::driverName ("radv", ...)
    std::string driver_info;

    // Features (each enabled on the logical device when supported).
    bool buffer_device_address = false;
    bool shader_int64 = false;
    bool shader_float64 = false;
    bool shader_int16 = false;
    bool storage_buffer_8bit = false;
    bool storage_buffer_16bit = false;
    bool subgroup_basic = false;      // compute stage, VK_SUBGROUP_FEATURE_BASIC_BIT
    bool subgroup_shuffle = false;    // compute stage, VK_SUBGROUP_FEATURE_SHUFFLE_BIT
    bool buffer_float32_atomic_add = false;  // VK_EXT_shader_atomic_float
    bool shared_float32_atomic_add = false;
    bool buffer_float64_atomic_add = false;
    bool buffer_int64_atomics = false;
    bool shared_int64_atomics = false;
    bool subgroup_clock = false;      // VK_KHR_shader_clock
    bool subgroup_size_control = false;   // VK_EXT_subgroup_size_control / Vulkan 1.3
    bool compute_full_subgroups = false;

    // Limits.
    uint32_t subgroup_size = 0;           // the driver's default
    uint32_t min_subgroup_size = 0;
    uint32_t max_subgroup_size = 0;
    bool can_require_subgroup_size_in_compute = false;
    uint32_t max_push_constant_bytes = 0;
    uint32_t max_shared_bytes = 0;
    uint32_t max_workgroup_invocations = 0;
    uint32_t max_workgroup_size[3] = {0, 0, 0};
    double timestamp_period_ns = 0.0;     // 0 when the queue has no timestamps

    // Empty when the device can run `k`; otherwise one line per missing
    // feature or extension, naming the SPIR-V capability that needs it.
    std::string missing_for(const brass::target::SpirvKernel& k) const;
};

// True when libvulkan loads and a device with bufferDeviceAddress and
// shaderInt64 (the floor every SpirvTarget kernel needs) can be opened.
// The device is the first discrete GPU, else the first integrated one, else
// the first of any type; BRASS_VULKAN_DEVICE=<index> overrides.
bool vulkan_available();

// Human-readable description of the last failure on this thread.
std::string vulkan_last_error();

// Every physical device (empty without a loader). Does not open a device.
std::vector<VulkanDeviceInfo> vulkan_devices();

// The opened device's capabilities, or nullptr when unavailable.
const VulkanDeviceCaps* vulkan_device_caps();

// Owning device-local buffer with a device address. Movable, non-copyable.
class VulkanBuffer {
public:
    VulkanBuffer() = default;
    ~VulkanBuffer();
    VulkanBuffer(VulkanBuffer&& other) noexcept;
    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept;
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    // Rounds `bytes` up to 4 (vkCmdFillBuffer granularity); 0 allocates 4.
    static VulkanBuffer alloc(size_t bytes);

    void reset();
    bool valid() const noexcept { return buffer_ != 0; }
    size_t size() const noexcept { return size_; }
    uint64_t device_address() const noexcept { return address_; }

    // Synchronous transfers through a staging buffer. `bytes` and `offset`
    // are any byte counts within size().
    bool upload(const void* host, size_t bytes, size_t offset = 0);
    bool download(void* host, size_t bytes, size_t offset = 0) const;
    bool fill(uint32_t word);
    bool zero() { return fill(0); }

    uint64_t handle() const noexcept { return buffer_; }   // VkBuffer

private:
    uint64_t buffer_ = 0;   // VkBuffer
    uint64_t memory_ = 0;   // VkDeviceMemory
    size_t size_ = 0;
    uint64_t address_ = 0;
};

// One kernel argument, packed into the push-constant block at the offset
// SpirvKernel::params gives. A buffer stands for its device address.
class VulkanArg {
public:
    enum class Kind : uint8_t { U64, U32, F32, F64 };
    VulkanArg(const VulkanBuffer& b) : kind_(Kind::U64), bits_(b.device_address()) {}  // NOLINT
    VulkanArg(uint64_t v) : kind_(Kind::U64), bits_(v) {}                                // NOLINT
    VulkanArg(int64_t v) : kind_(Kind::U64), bits_(static_cast<uint64_t>(v)) {}         // NOLINT
    VulkanArg(uint32_t v) : kind_(Kind::U32), bits_(v) {}                                // NOLINT
    VulkanArg(int32_t v) : kind_(Kind::U32), bits_(static_cast<uint32_t>(v)) {}          // NOLINT
    VulkanArg(float v);                                                                  // NOLINT
    VulkanArg(double v);                                                                 // NOLINT

    Kind kind() const noexcept { return kind_; }
    uint64_t bits() const noexcept { return bits_; }

private:
    Kind kind_;
    uint64_t bits_;
};

struct VulkanModuleOptions {
    // Subgroup size for the pipeline: 32 (the PTX warp the kernels are
    // written for; the default), 64, or 0 for the driver's choice. Honoured
    // with VK_EXT_subgroup_size_control (core in 1.3) when the device can
    // require that size in compute; otherwise the driver's choice is used
    // (the kernels are correct for 32 and 64, see the design doc).
    uint32_t subgroup_size = 32;
};

struct VulkanDispatch {
    uint32_t grid[3] = {1, 1, 1};
    uint32_t block[3] = {256, 1, 1};   // spec constants 0..2 (must match a fixed LocalSize)
    // Record the dispatch `repeat` times in one command buffer, with a
    // compute->compute memory barrier between them (benchmarking).
    uint32_t repeat = 1;
    // When set, receives the GPU time of all `repeat` dispatches in
    // milliseconds (timestamp queries), or -1 without timestamp support.
    double* gpu_ms = nullptr;
};

// A SpirvKernel as a compute pipeline (one per distinct workgroup size,
// created on first use and cached). Movable, non-copyable.
class VulkanModule {
public:
    VulkanModule();
    ~VulkanModule();
    VulkanModule(VulkanModule&& other) noexcept;
    VulkanModule& operator=(VulkanModule&& other) noexcept;
    VulkanModule(const VulkanModule&) = delete;
    VulkanModule& operator=(const VulkanModule&) = delete;

    // Fails (empty module, `error` set) when Vulkan is unavailable, the
    // device lacks a feature the kernel's capabilities need, the push
    // constants exceed the device limit, or vkCreateShaderModule fails.
    static VulkanModule load(const brass::target::SpirvKernel& kernel, std::string* error = nullptr,
                             const VulkanModuleOptions& opts = {});

    bool valid() const noexcept;
    void reset();
    const std::string& entry() const;

    // Packs `args` per the kernel's parameter layout (count and kinds must
    // match: ptr/i64 take U64, i32 U32, f32 F32, f64 F64), dispatches and
    // waits on a fence.
    bool launch(const VulkanDispatch& d, const std::vector<VulkanArg>& args, std::string* error = nullptr) const;
    bool launch_1d(uint32_t grid, uint32_t block, std::initializer_list<VulkanArg> args,
                   std::string* error = nullptr) const;

    // The subgroup size the pipelines were created with (0 = driver's choice).
    uint32_t subgroup_size() const noexcept;

    struct State;

private:
    std::unique_ptr<State> s_;
};

} // namespace brass::gpu
