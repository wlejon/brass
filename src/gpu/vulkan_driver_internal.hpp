#pragma once

// Internals shared by vulkan_driver.cpp (loader, device, buffers) and
// vulkan_driver_module.cpp (pipelines, dispatch). Not a public header.

#include <brass/gpu/vulkan_driver.hpp>

#if defined(__has_include)
#  if __has_include(<vulkan/vulkan.h>)
#    define VK_NO_PROTOTYPES 1
#    include <vulkan/vulkan.h>
#  endif
#endif

// The implementation needs Vulkan 1.3 headers (it runs on 1.2 devices).
#if defined(VK_VERSION_1_3)
#  define BRASS_VULKAN_HEADERS 1
#else
#  define BRASS_VULKAN_HEADERS 0
#endif

#if BRASS_VULKAN_HEADERS

#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace brass::gpu::vk {

#define BRASS_VK_INSTANCE_FNS(X)                    \
    X(vkDestroyInstance)                            \
    X(vkEnumeratePhysicalDevices)                   \
    X(vkGetPhysicalDeviceProperties)                \
    X(vkGetPhysicalDeviceProperties2)               \
    X(vkGetPhysicalDeviceFeatures2)                 \
    X(vkGetPhysicalDeviceQueueFamilyProperties)     \
    X(vkGetPhysicalDeviceMemoryProperties)          \
    X(vkEnumerateDeviceExtensionProperties)         \
    X(vkCreateDevice)                               \
    X(vkGetDeviceProcAddr)

#define BRASS_VK_DEVICE_FNS(X)          \
    X(vkDestroyDevice)                  \
    X(vkGetDeviceQueue)                 \
    X(vkCreateCommandPool)              \
    X(vkAllocateCommandBuffers)         \
    X(vkResetCommandBuffer)             \
    X(vkBeginCommandBuffer)             \
    X(vkEndCommandBuffer)               \
    X(vkQueueSubmit)                    \
    X(vkCreateFence)                    \
    X(vkResetFences)                    \
    X(vkWaitForFences)                  \
    X(vkCreateBuffer)                   \
    X(vkDestroyBuffer)                  \
    X(vkGetBufferMemoryRequirements)    \
    X(vkAllocateMemory)                 \
    X(vkFreeMemory)                     \
    X(vkBindBufferMemory)               \
    X(vkMapMemory)                      \
    X(vkUnmapMemory)                    \
    X(vkGetBufferDeviceAddress)         \
    X(vkCmdCopyBuffer)                  \
    X(vkCmdFillBuffer)                  \
    X(vkCmdPipelineBarrier)             \
    X(vkCreateShaderModule)             \
    X(vkDestroyShaderModule)            \
    X(vkCreatePipelineLayout)           \
    X(vkDestroyPipelineLayout)          \
    X(vkCreateComputePipelines)         \
    X(vkDestroyPipeline)                \
    X(vkCmdBindPipeline)                \
    X(vkCmdPushConstants)               \
    X(vkCmdDispatch)                    \
    X(vkCreateQueryPool)                \
    X(vkDestroyQueryPool)               \
    X(vkCmdResetQueryPool)              \
    X(vkCmdWriteTimestamp)              \
    X(vkGetQueryPoolResults)

// The process-wide device. Created once (runtime()), never destroyed: the
// driver reclaims it at exit, and buffers held in statics stay valid.
struct Runtime {
    bool ok = false;
    std::string error;
    void* lib = nullptr;
    PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    uint32_t instance_version = 0;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    uint32_t device_version = 0;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mem_props{};
    std::vector<std::string> enabled_extensions;
    VulkanDeviceCaps caps;
    std::mutex submit_mutex;

#define BRASS_VK_DECLARE_FN(name) PFN_##name name = nullptr;
    BRASS_VK_INSTANCE_FNS(BRASS_VK_DECLARE_FN)
    BRASS_VK_DEVICE_FNS(BRASS_VK_DECLARE_FN)
#undef BRASS_VK_DECLARE_FN

    bool fail(const std::string& msg);
    uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred) const;
    bool make_buffer(size_t bytes, VkBufferUsageFlags usage, VkMemoryPropertyFlags required,
                     VkMemoryPropertyFlags preferred, bool device_address, VkBuffer& buf, VkDeviceMemory& mem);
    void full_barrier(VkCommandBuffer cb) const;
    // Record with `record`, submit on the one queue and wait (serialized).
    bool submit(const std::function<void(VkCommandBuffer)>& record);
};

// nullptr when Vulkan is unavailable (the reason is in last_error()).
Runtime* runtime();
void set_error(const std::string& msg);
const std::string& last_error();

// A zeroed Vulkan struct with its sType.
template <typename T>
inline T make(VkStructureType type) {
    T s{};
    s.sType = type;
    return s;
}

// Non-dispatchable handles are pointers on 64-bit and uint64_t on 32-bit.
template <typename H>
inline uint64_t as_u64(H h) {
    if constexpr (std::is_pointer_v<H>) return reinterpret_cast<uint64_t>(h);
    else return static_cast<uint64_t>(h);
}
template <typename H>
inline H from_u64(uint64_t v) {
    if constexpr (std::is_pointer_v<H>) return reinterpret_cast<H>(v);
    else return static_cast<H>(v);
}
inline VkBuffer as_buffer(uint64_t v) { return from_u64<VkBuffer>(v); }
inline VkDeviceMemory as_memory(uint64_t v) { return from_u64<VkDeviceMemory>(v); }

} // namespace brass::gpu::vk

#endif // BRASS_VULKAN_HEADERS
