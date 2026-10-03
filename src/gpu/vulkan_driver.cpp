// Minimal dlopen'ed Vulkan compute runtime for SpirvTarget kernels; see
// include/brass/gpu/vulkan_driver.hpp and docs/spirv_backend_design.md
// ("Vulkan runtime"). This file: loader, device selection, feature
// negotiation, buffers and one-shot command submission. Pipelines and
// dispatch are in vulkan_driver_module.cpp.

#include "vulkan_driver_internal.hpp"

#if BRASS_VULKAN_HEADERS

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace brass::gpu::vk {

namespace {

thread_local std::string tls_last_error;

void* open_loader() {
#if defined(_WIN32)
    return reinterpret_cast<void*>(LoadLibraryA("vulkan-1.dll"));
#elif defined(__APPLE__)
    const char* names[] = {"libvulkan.1.dylib", "libvulkan.dylib", "libMoltenVK.dylib", nullptr};
    for (int i = 0; names[i]; ++i)
        if (void* h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL)) return h;
    return nullptr;
#else
    const char* names[] = {"libvulkan.so.1", "libvulkan.so", nullptr};
    for (int i = 0; names[i]; ++i)
        if (void* h = dlopen(names[i], RTLD_NOW | RTLD_LOCAL)) return h;
    return nullptr;
#endif
}

PFN_vkGetInstanceProcAddr loader_entry(void* lib) {
#if defined(_WIN32)
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), "vkGetInstanceProcAddr")));
#else
    return reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
#endif
}

const char* type_name(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}

bool has_ext(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (const VkExtensionProperties& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

// Instance + the instance-level entry points; no device yet.
bool open_instance(Runtime& r) {
    r.lib = open_loader();
    if (!r.lib) return r.fail("libvulkan could not be loaded");
    r.vkGetInstanceProcAddr = loader_entry(r.lib);
    if (!r.vkGetInstanceProcAddr) return r.fail("libvulkan has no vkGetInstanceProcAddr");
    auto global = [&](const char* n) { return r.vkGetInstanceProcAddr(VK_NULL_HANDLE, n); };
    auto create = reinterpret_cast<PFN_vkCreateInstance>(global("vkCreateInstance"));
    auto version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(global("vkEnumerateInstanceVersion"));
    if (!create) return r.fail("libvulkan has no vkCreateInstance");
    uint32_t loader_version = VK_API_VERSION_1_0;
    if (version) version(&loader_version);
    if (loader_version < VK_API_VERSION_1_2) return r.fail("the Vulkan loader is older than 1.2");

    auto app = vk::make<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    app.pApplicationName = "brass";
    app.pEngineName = "brass";
    app.apiVersion = std::min<uint32_t>(loader_version, VK_API_VERSION_1_3);
    auto ci = vk::make<VkInstanceCreateInfo>(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    ci.pApplicationInfo = &app;
    if (create(&ci, nullptr, &r.instance) != VK_SUCCESS) return r.fail("vkCreateInstance failed");
    r.instance_version = app.apiVersion;

#define BRASS_VK_INSTANCE_FN(name) \
    r.name = reinterpret_cast<PFN_##name>(r.vkGetInstanceProcAddr(r.instance, #name)); \
    if (!r.name) return r.fail("missing instance entry point " #name);
    BRASS_VK_INSTANCE_FNS(BRASS_VK_INSTANCE_FN)
#undef BRASS_VK_INSTANCE_FN
    return true;
}

// Pick a device: BRASS_VULKAN_DEVICE=<index>, else discrete, integrated, any.
bool pick_device(Runtime& r) {
    uint32_t n = 0;
    r.vkEnumeratePhysicalDevices(r.instance, &n, nullptr);
    if (n == 0) return r.fail("no Vulkan physical devices");
    std::vector<VkPhysicalDevice> devs(n);
    r.vkEnumeratePhysicalDevices(r.instance, &n, devs.data());
    std::vector<VkPhysicalDeviceProperties> props(n);
    for (uint32_t i = 0; i < n; ++i) r.vkGetPhysicalDeviceProperties(devs[i], &props[i]);

    int chosen = -1;
    if (const char* env = std::getenv("BRASS_VULKAN_DEVICE"); env && *env) {
        int idx = std::atoi(env);
        if (idx < 0 || static_cast<uint32_t>(idx) >= n)
            return r.fail("BRASS_VULKAN_DEVICE=" + std::string(env) + " is out of range (" + std::to_string(n) + " devices)");
        chosen = idx;
    } else {
        for (VkPhysicalDeviceType want : {VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU}) {
            for (uint32_t i = 0; i < n && chosen < 0; ++i)
                if (props[i].deviceType == want && props[i].apiVersion >= VK_API_VERSION_1_2) chosen = static_cast<int>(i);
            if (chosen >= 0) break;
        }
        for (uint32_t i = 0; i < n && chosen < 0; ++i)
            if (props[i].apiVersion >= VK_API_VERSION_1_2) chosen = static_cast<int>(i);
    }
    if (chosen < 0) return r.fail("no Vulkan 1.2 device");
    r.phys = devs[static_cast<size_t>(chosen)];
    const VkPhysicalDeviceProperties& p = props[static_cast<size_t>(chosen)];
    if (p.apiVersion < VK_API_VERSION_1_2) return r.fail(std::string(p.deviceName) + " is not a Vulkan 1.2 device");
    r.device_version = std::min(p.apiVersion, r.instance_version);
    VulkanDeviceInfo& info = r.caps.info;
    info.index = static_cast<uint32_t>(chosen);
    info.name = p.deviceName;
    info.type = type_name(p.deviceType);
    info.vendor_id = p.vendorID;
    info.device_id = p.deviceID;
    info.api_version = p.apiVersion;
    return true;
}

// Query what the device supports, fill caps, and build the enable chain.
bool open_device(Runtime& r) {
    uint32_t next = 0;
    r.vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &next, nullptr);
    std::vector<VkExtensionProperties> exts(next);
    r.vkEnumerateDeviceExtensionProperties(r.phys, nullptr, &next, exts.data());
    const bool v13 = r.device_version >= VK_API_VERSION_1_3;
    const bool ext_atomic_float = has_ext(exts, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    const bool ext_clock = has_ext(exts, VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
    const bool ext_sg_size = v13 || has_ext(exts, VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);

    // --- properties ---
    auto sg_size_props = vk::make<VkPhysicalDeviceSubgroupSizeControlProperties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES);
    auto sg_props = vk::make<VkPhysicalDeviceSubgroupProperties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES);
    auto drv = vk::make<VkPhysicalDeviceDriverProperties>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
    sg_props.pNext = &drv;
    if (ext_sg_size) drv.pNext = &sg_size_props;
    auto props2 = vk::make<VkPhysicalDeviceProperties2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2);
    props2.pNext = &sg_props;
    r.vkGetPhysicalDeviceProperties2(r.phys, &props2);
    const VkPhysicalDeviceLimits& lim = props2.properties.limits;

    // --- features ---
    auto f11 = vk::make<VkPhysicalDeviceVulkan11Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    auto f12 = vk::make<VkPhysicalDeviceVulkan12Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    auto fsg = vk::make<VkPhysicalDeviceSubgroupSizeControlFeatures>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES);
    auto faf = vk::make<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT);
    auto fclk = vk::make<VkPhysicalDeviceShaderClockFeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR);
    auto f2 = vk::make<VkPhysicalDeviceFeatures2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
    f2.pNext = &f11;
    f11.pNext = &f12;
    void** tail = &f12.pNext;
    if (ext_sg_size) { *tail = &fsg; tail = &fsg.pNext; }
    if (ext_atomic_float) { *tail = &faf; tail = &faf.pNext; }
    if (ext_clock) { *tail = &fclk; tail = &fclk.pNext; }
    r.vkGetPhysicalDeviceFeatures2(r.phys, &f2);

    VulkanDeviceCaps& c = r.caps;
    c.driver_name = drv.driverName;
    c.driver_info = drv.driverInfo;
    c.buffer_device_address = f12.bufferDeviceAddress;
    c.shader_int64 = f2.features.shaderInt64;
    c.shader_float64 = f2.features.shaderFloat64;
    c.shader_int16 = f2.features.shaderInt16;
    c.storage_buffer_8bit = f12.storageBuffer8BitAccess;
    c.storage_buffer_16bit = f11.storageBuffer16BitAccess;
    const bool compute_sg = (sg_props.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    c.subgroup_basic = compute_sg && (sg_props.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT);
    c.subgroup_shuffle = compute_sg && (sg_props.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT);
    c.buffer_int64_atomics = f12.shaderBufferInt64Atomics;
    c.shared_int64_atomics = f12.shaderSharedInt64Atomics;
    if (ext_atomic_float) {
        c.buffer_float32_atomic_add = faf.shaderBufferFloat32AtomicAdd;
        c.shared_float32_atomic_add = faf.shaderSharedFloat32AtomicAdd;
        c.buffer_float64_atomic_add = faf.shaderBufferFloat64AtomicAdd;
    }
    c.subgroup_clock = ext_clock && fclk.shaderSubgroupClock;
    if (ext_sg_size) {
        c.subgroup_size_control = fsg.subgroupSizeControl;
        c.compute_full_subgroups = fsg.computeFullSubgroups;
        c.min_subgroup_size = sg_size_props.minSubgroupSize;
        c.max_subgroup_size = sg_size_props.maxSubgroupSize;
        c.can_require_subgroup_size_in_compute =
            c.subgroup_size_control && (sg_size_props.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT);
    } else {
        c.min_subgroup_size = c.max_subgroup_size = sg_props.subgroupSize;
    }
    c.subgroup_size = sg_props.subgroupSize;
    c.max_push_constant_bytes = lim.maxPushConstantsSize;
    c.max_shared_bytes = lim.maxComputeSharedMemorySize;
    c.max_workgroup_invocations = lim.maxComputeWorkGroupInvocations;
    for (int i = 0; i < 3; ++i) c.max_workgroup_size[i] = lim.maxComputeWorkGroupSize[i];

    if (!c.buffer_device_address) return r.fail(c.info.name + ": no bufferDeviceAddress");
    if (!c.shader_int64) return r.fail(c.info.name + ": no shaderInt64");

    // --- queue: a compute queue, preferring one without graphics ---
    uint32_t nq = 0;
    r.vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    r.vkGetPhysicalDeviceQueueFamilyProperties(r.phys, &nq, qf.data());
    int family = -1;
    for (uint32_t i = 0; i < nq && family < 0; ++i)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) family = static_cast<int>(i);
    for (uint32_t i = 0; i < nq && family < 0; ++i)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) family = static_cast<int>(i);
    if (family < 0) return r.fail(c.info.name + ": no compute queue");
    r.queue_family = static_cast<uint32_t>(family);
    c.timestamp_period_ns = qf[r.queue_family].timestampValidBits ? static_cast<double>(lim.timestampPeriod) : 0.0;

    // --- enable exactly what the kernels use ---
    auto e2 = vk::make<VkPhysicalDeviceFeatures2>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2);
    e2.features.shaderInt64 = VK_TRUE;
    e2.features.shaderFloat64 = c.shader_float64;
    e2.features.shaderInt16 = c.shader_int16;
    auto e11 = vk::make<VkPhysicalDeviceVulkan11Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES);
    e11.storageBuffer16BitAccess = c.storage_buffer_16bit;
    auto e12 = vk::make<VkPhysicalDeviceVulkan12Features>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES);
    e12.bufferDeviceAddress = VK_TRUE;
    e12.storageBuffer8BitAccess = c.storage_buffer_8bit;
    e12.shaderBufferInt64Atomics = c.buffer_int64_atomics;
    e12.shaderSharedInt64Atomics = c.shared_int64_atomics;
    auto esg = vk::make<VkPhysicalDeviceSubgroupSizeControlFeatures>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES);
    esg.subgroupSizeControl = c.subgroup_size_control;
    esg.computeFullSubgroups = c.compute_full_subgroups;
    auto eaf = vk::make<VkPhysicalDeviceShaderAtomicFloatFeaturesEXT>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT);
    eaf.shaderBufferFloat32AtomicAdd = c.buffer_float32_atomic_add;
    eaf.shaderSharedFloat32AtomicAdd = c.shared_float32_atomic_add;
    eaf.shaderBufferFloat64AtomicAdd = c.buffer_float64_atomic_add;
    auto eclk = vk::make<VkPhysicalDeviceShaderClockFeaturesKHR>(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR);
    eclk.shaderSubgroupClock = c.subgroup_clock;

    std::vector<const char*> enable_exts;
    e2.pNext = &e11;
    e11.pNext = &e12;
    tail = &e12.pNext;
    if (c.subgroup_size_control) {
        *tail = &esg; tail = &esg.pNext;
        if (!v13) enable_exts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    }
    if (c.buffer_float32_atomic_add || c.shared_float32_atomic_add) {
        *tail = &eaf; tail = &eaf.pNext;
        enable_exts.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    }
    if (c.subgroup_clock) {
        *tail = &eclk; tail = &eclk.pNext;
        enable_exts.push_back(VK_KHR_SHADER_CLOCK_EXTENSION_NAME);
    }
    r.enabled_extensions.assign(enable_exts.begin(), enable_exts.end());

    float prio = 1.0f;
    auto qci = vk::make<VkDeviceQueueCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
    qci.queueFamilyIndex = r.queue_family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    auto dci = vk::make<VkDeviceCreateInfo>(VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    dci.pNext = &e2;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(enable_exts.size());
    dci.ppEnabledExtensionNames = enable_exts.data();
    if (r.vkCreateDevice(r.phys, &dci, nullptr, &r.device) != VK_SUCCESS)
        return r.fail(c.info.name + ": vkCreateDevice failed");

#define BRASS_VK_DEVICE_FN(name) \
    r.name = reinterpret_cast<PFN_##name>(r.vkGetDeviceProcAddr(r.device, #name)); \
    if (!r.name) return r.fail("missing device entry point " #name);
    BRASS_VK_DEVICE_FNS(BRASS_VK_DEVICE_FN)
#undef BRASS_VK_DEVICE_FN

    r.vkGetDeviceQueue(r.device, r.queue_family, 0, &r.queue);
    r.vkGetPhysicalDeviceMemoryProperties(r.phys, &r.mem_props);

    auto pci = vk::make<VkCommandPoolCreateInfo>(VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = r.queue_family;
    if (r.vkCreateCommandPool(r.device, &pci, nullptr, &r.pool) != VK_SUCCESS) return r.fail("vkCreateCommandPool failed");
    auto cai = vk::make<VkCommandBufferAllocateInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    cai.commandPool = r.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    if (r.vkAllocateCommandBuffers(r.device, &cai, &r.cmd) != VK_SUCCESS) return r.fail("vkAllocateCommandBuffers failed");
    auto fci = vk::make<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    if (r.vkCreateFence(r.device, &fci, nullptr, &r.fence) != VK_SUCCESS) return r.fail("vkCreateFence failed");
    return true;
}

} // namespace

bool Runtime::fail(const std::string& msg) {
    error = msg;
    tls_last_error = msg;
    return false;
}

Runtime* runtime() {
    static Runtime r;
    static std::once_flag once;
    std::call_once(once, [] { r.ok = open_instance(r) && pick_device(r) && open_device(r); });
    if (!r.ok) tls_last_error = r.error;
    return r.ok ? &r : nullptr;
}

void set_error(const std::string& msg) { tls_last_error = msg; }
const std::string& last_error() { return tls_last_error; }

uint32_t Runtime::memory_type(uint32_t bits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) const {
    for (VkMemoryPropertyFlags want : {required | preferred, required}) {
        for (uint32_t i = 0; i < mem_props.memoryTypeCount; ++i)
            if ((bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & want) == want) return i;
    }
    return UINT32_MAX;
}

bool Runtime::make_buffer(size_t bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags required,
                          VkMemoryPropertyFlags preferred, bool device_address, VkBuffer& buf, VkDeviceMemory& mem) {
    auto bci = vk::make<VkBufferCreateInfo>(VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    bci.size = bytes;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bci, nullptr, &buf) != VK_SUCCESS) return fail("vkCreateBuffer failed");
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buf, &req);
    uint32_t type = memory_type(req.memoryTypeBits, required, preferred);
    if (type == UINT32_MAX) {
        vkDestroyBuffer(device, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return fail("no memory type for a " + std::to_string(bytes) + "-byte buffer");
    }
    auto flags = vk::make<VkMemoryAllocateFlagsInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO);
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    auto mai = vk::make<VkMemoryAllocateInfo>(VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    mai.pNext = device_address ? &flags : nullptr;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = type;
    if (vkAllocateMemory(device, &mai, nullptr, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(device, buf, nullptr);
        buf = VK_NULL_HANDLE;
        return fail("vkAllocateMemory of " + std::to_string(req.size) + " bytes failed");
    }
    vkBindBufferMemory(device, buf, mem, 0);
    return true;
}

void Runtime::full_barrier(VkCommandBuffer cb) const {
    auto mb = vk::make<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0,
                         nullptr, 0, nullptr);
}

bool Runtime::submit(const std::function<void(VkCommandBuffer)>& record) {
    std::lock_guard<std::mutex> lock(submit_mutex);
    vkResetCommandBuffer(cmd, 0);
    auto bi = vk::make<VkCommandBufferBeginInfo>(VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return fail("vkBeginCommandBuffer failed");
    // Submission order is not a memory dependency: make every earlier
    // device write visible to this submission and this one's to the host.
    full_barrier(cmd);
    record(cmd);
    auto mb = vk::make<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
    mb.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT | VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return fail("vkEndCommandBuffer failed");
    auto si = vk::make<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkResetFences(device, 1, &fence);
    VkResult res = vkQueueSubmit(queue, 1, &si, fence);
    if (res != VK_SUCCESS) return fail("vkQueueSubmit failed (" + std::to_string(res) + ")");
    res = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (res != VK_SUCCESS)
        return fail(res == VK_ERROR_DEVICE_LOST ? "device lost (a kernel faulted or hung)"
                                                : "vkWaitForFences failed (" + std::to_string(res) + ")");
    return true;
}

// A host-visible coherent staging buffer, mapped; destroyed by the caller.
struct Staging {
    Runtime& r;
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* map = nullptr;
    Staging(Runtime& rt, size_t bytes) : r(rt) {
        if (!r.make_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           VK_MEMORY_PROPERTY_HOST_CACHED_BIT, false, buf, mem))
            return;
        if (r.vkMapMemory(r.device, mem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) map = nullptr;
    }
    ~Staging() {
        if (map) r.vkUnmapMemory(r.device, mem);
        if (buf) r.vkDestroyBuffer(r.device, buf, nullptr);
        if (mem) r.vkFreeMemory(r.device, mem, nullptr);
    }
    Staging(const Staging&) = delete;
    Staging& operator=(const Staging&) = delete;
};

} // namespace brass::gpu::vk

namespace brass::gpu {

using vk::Runtime;

bool vulkan_available() { return vk::runtime() != nullptr; }

std::string vulkan_last_error() { return vk::last_error(); }

const VulkanDeviceCaps* vulkan_device_caps() {
    Runtime* r = vk::runtime();
    return r ? &r->caps : nullptr;
}

std::vector<VulkanDeviceInfo> vulkan_devices() {
    std::vector<VulkanDeviceInfo> out;
    Runtime* r = vk::runtime();
    if (!r) return out;
    uint32_t n = 0;
    r->vkEnumeratePhysicalDevices(r->instance, &n, nullptr);
    std::vector<VkPhysicalDevice> devs(n);
    r->vkEnumeratePhysicalDevices(r->instance, &n, devs.data());
    for (uint32_t i = 0; i < n; ++i) {
        VkPhysicalDeviceProperties p;
        r->vkGetPhysicalDeviceProperties(devs[i], &p);
        VulkanDeviceInfo info;
        info.index = i;
        info.name = p.deviceName;
        info.type = vk::type_name(p.deviceType);
        info.vendor_id = p.vendorID;
        info.device_id = p.deviceID;
        info.api_version = p.apiVersion;
        out.push_back(std::move(info));
    }
    return out;
}

// --- VulkanBuffer ---------------------------------------------------------

VulkanBuffer::~VulkanBuffer() { reset(); }

VulkanBuffer::VulkanBuffer(VulkanBuffer&& o) noexcept
    : buffer_(o.buffer_), memory_(o.memory_), size_(o.size_), address_(o.address_) {
    o.buffer_ = o.memory_ = 0;
    o.size_ = 0;
    o.address_ = 0;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& o) noexcept {
    if (this != &o) {
        reset();
        buffer_ = o.buffer_; memory_ = o.memory_; size_ = o.size_; address_ = o.address_;
        o.buffer_ = o.memory_ = 0;
        o.size_ = 0;
        o.address_ = 0;
    }
    return *this;
}

void VulkanBuffer::reset() {
    if (!buffer_) return;
    if (Runtime* r = vk::runtime()) {
        r->vkDestroyBuffer(r->device, vk::as_buffer(buffer_), nullptr);
        r->vkFreeMemory(r->device, vk::as_memory(memory_), nullptr);
    }
    buffer_ = memory_ = 0;
    size_ = 0;
    address_ = 0;
}

VulkanBuffer VulkanBuffer::alloc(size_t bytes) {
    VulkanBuffer out;
    Runtime* r = vk::runtime();
    if (!r) return out;
    size_t size = std::max<size_t>(4, (bytes + 3) & ~size_t{3});
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (!r->make_buffer(size,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, true, buf, mem))
        return out;
    auto ai = vk::make<VkBufferDeviceAddressInfo>(VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO);
    ai.buffer = buf;
    out.buffer_ = vk::as_u64(buf);
    out.memory_ = vk::as_u64(mem);
    out.size_ = size;
    out.address_ = r->vkGetBufferDeviceAddress(r->device, &ai);
    return out;
}

bool VulkanBuffer::upload(const void* host, size_t bytes, size_t offset) {
    Runtime* r = vk::runtime();
    if (!r || !valid()) return false;
    if (offset + bytes > size_) { vk::set_error("upload past the end of the buffer"); return false; }
    if (bytes == 0) return true;
    vk::Staging st(*r, bytes);
    if (!st.map) return false;
    std::memcpy(st.map, host, bytes);
    VkBuffer dst = vk::as_buffer(buffer_);
    return r->submit([&](VkCommandBuffer cb) {
        VkBufferCopy region{0, offset, bytes};
        r->vkCmdCopyBuffer(cb, st.buf, dst, 1, &region);
    });
}

bool VulkanBuffer::download(void* host, size_t bytes, size_t offset) const {
    Runtime* r = vk::runtime();
    if (!r || !valid()) return false;
    if (offset + bytes > size_) { vk::set_error("download past the end of the buffer"); return false; }
    if (bytes == 0) return true;
    vk::Staging st(*r, bytes);
    if (!st.map) return false;
    VkBuffer src = vk::as_buffer(buffer_);
    bool ok = r->submit([&](VkCommandBuffer cb) {
        VkBufferCopy region{offset, 0, bytes};
        r->vkCmdCopyBuffer(cb, src, st.buf, 1, &region);
    });
    if (ok) std::memcpy(host, st.map, bytes);
    return ok;
}

bool VulkanBuffer::fill(uint32_t word) {
    Runtime* r = vk::runtime();
    if (!r || !valid()) return false;
    VkBuffer dst = vk::as_buffer(buffer_);
    return r->submit([&](VkCommandBuffer cb) { r->vkCmdFillBuffer(cb, dst, 0, VK_WHOLE_SIZE, word); });
}

} // namespace brass::gpu

#else // !BRASS_VULKAN_HEADERS: built without <vulkan/vulkan.h>, nothing is available

namespace brass::gpu {

namespace {
const char* kNoHeaders = "brass was built without the Vulkan headers (vulkan/vulkan.h)";
}

bool vulkan_available() { return false; }
std::string vulkan_last_error() { return kNoHeaders; }
std::vector<VulkanDeviceInfo> vulkan_devices() { return {}; }
const VulkanDeviceCaps* vulkan_device_caps() { return nullptr; }

VulkanBuffer::~VulkanBuffer() = default;
VulkanBuffer::VulkanBuffer(VulkanBuffer&&) noexcept {}
VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&&) noexcept { return *this; }
void VulkanBuffer::reset() {}
VulkanBuffer VulkanBuffer::alloc(size_t) { return {}; }
bool VulkanBuffer::upload(const void*, size_t, size_t) { return false; }
bool VulkanBuffer::download(void*, size_t, size_t) const { return false; }
bool VulkanBuffer::fill(uint32_t) { return false; }

} // namespace brass::gpu

#endif
