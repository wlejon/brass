// VulkanModule: a SpirvKernel as compute pipelines (one per workgroup size,
// specialization constants 0..2), a push-constant-only pipeline layout, and
// dispatch with the kernel's push-constant block packed from VulkanArgs.
// The device-feature check against SpirvKernel::capabilities lives here too.

#include "vulkan_driver_internal.hpp"

#include <brass/target/spirv_target.hpp>

#include <array>
#include <cstring>
#include <map>
#include <mutex>
#include <sstream>

namespace brass::gpu {

VulkanArg::VulkanArg(float v) : kind_(Kind::F32), bits_(0) {
    uint32_t w;
    std::memcpy(&w, &v, 4);
    bits_ = w;
}

VulkanArg::VulkanArg(double v) : kind_(Kind::F64), bits_(0) { std::memcpy(&bits_, &v, 8); }

// SPIR-V capability (as SpirvKernel lists it) -> the device feature it needs.
std::string VulkanDeviceCaps::missing_for(const brass::target::SpirvKernel& k) const {
    struct Need { const char* capability; bool VulkanDeviceCaps::*feature; const char* feature_name; };
    static const Need table[] = {
        {"PhysicalStorageBufferAddresses", &VulkanDeviceCaps::buffer_device_address, "bufferDeviceAddress"},
        {"Int64", &VulkanDeviceCaps::shader_int64, "shaderInt64"},
        {"Float64", &VulkanDeviceCaps::shader_float64, "shaderFloat64"},
        {"StorageBuffer8BitAccess", &VulkanDeviceCaps::storage_buffer_8bit, "storageBuffer8BitAccess"},
        {"StorageBuffer16BitAccess", &VulkanDeviceCaps::storage_buffer_16bit, "storageBuffer16BitAccess"},
        {"GroupNonUniform", &VulkanDeviceCaps::subgroup_basic, "subgroup BASIC operations in compute"},
        {"GroupNonUniformShuffle", &VulkanDeviceCaps::subgroup_shuffle, "subgroup SHUFFLE operations in compute"},
        {"AtomicFloat32AddEXT", &VulkanDeviceCaps::buffer_float32_atomic_add,
         "shaderBufferFloat32AtomicAdd (VK_EXT_shader_atomic_float)"},
        {"AtomicFloat64AddEXT", &VulkanDeviceCaps::buffer_float64_atomic_add,
         "shaderBufferFloat64AtomicAdd (VK_EXT_shader_atomic_float)"},
        {"Int64Atomics", &VulkanDeviceCaps::buffer_int64_atomics, "shaderBufferInt64Atomics"},
        {"ShaderClockKHR", &VulkanDeviceCaps::subgroup_clock, "shaderSubgroupClock (VK_KHR_shader_clock)"},
    };
    std::ostringstream out;
    for (const std::string& cap : k.capabilities) {
        if (cap == "Shader") continue;
        bool known = false;
        for (const Need& n : table) {
            if (cap != n.capability) continue;
            known = true;
            if (!(this->*n.feature))
                out << "capability " << cap << " needs " << n.feature_name << ", which " << info.name << " lacks\n";
        }
        if (!known) out << "capability " << cap << " is not one the Vulkan runtime knows how to enable\n";
    }
    if (k.push_constant_bytes > max_push_constant_bytes)
        out << "push constants take " << k.push_constant_bytes << " bytes; the device allows " << max_push_constant_bytes
            << "\n";
    if (k.shared_bytes > max_shared_bytes)
        out << "shared memory takes " << k.shared_bytes << " bytes; the device allows " << max_shared_bytes << "\n";
    return out.str();
}

} // namespace brass::gpu

#if BRASS_VULKAN_HEADERS

namespace brass::gpu {

using vk::Runtime;

struct VulkanModule::State {
    brass::target::SpirvKernel kernel;
    uint32_t subgroup_size = 0;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    mutable std::mutex mutex;
    mutable std::map<std::array<uint32_t, 3>, VkPipeline> pipelines;

    ~State() {
        Runtime* r = vk::runtime();
        if (!r) return;
        for (auto& [key, p] : pipelines) r->vkDestroyPipeline(r->device, p, nullptr);
        if (layout) r->vkDestroyPipelineLayout(r->device, layout, nullptr);
        if (shader) r->vkDestroyShaderModule(r->device, shader, nullptr);
    }

    // The pipeline for one workgroup size, created on first use.
    VkPipeline pipeline(Runtime& r, const uint32_t block[3], std::string& err) const {
        std::array<uint32_t, 3> key{block[0], block[1], block[2]};
        std::lock_guard<std::mutex> lock(mutex);
        if (auto it = pipelines.find(key); it != pipelines.end()) return it->second;

        VkSpecializationMapEntry entries[3];
        for (uint32_t i = 0; i < 3; ++i) entries[i] = {i, i * 4, 4};
        VkSpecializationInfo spec{3, entries, sizeof(key), key.data()};

        auto req = vk::make<VkPipelineShaderStageRequiredSubgroupSizeCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO);
        req.requiredSubgroupSize = subgroup_size;
        auto stage = vk::make<VkPipelineShaderStageCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shader;
        stage.pName = kernel.entry.c_str();
        stage.pSpecializationInfo = kernel.local_size_spec_constants ? &spec : nullptr;
        if (subgroup_size) {
            stage.pNext = &req;
            // Full subgroups keep `SubgroupLocalInvocationId == LocalInvocationId.x % size`
            // exact; allowed only when x is a multiple of the subgroup size.
            if (r.caps.compute_full_subgroups && block[0] % subgroup_size == 0)
                stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        }
        auto ci = vk::make<VkComputePipelineCreateInfo>(VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO);
        ci.stage = stage;
        ci.layout = layout;
        VkPipeline p = VK_NULL_HANDLE;
        VkResult res = r.vkCreateComputePipelines(r.device, VK_NULL_HANDLE, 1, &ci, nullptr, &p);
        if (res != VK_SUCCESS) {
            err = "vkCreateComputePipelines failed for '" + kernel.entry + "' (" + std::to_string(res) + ")";
            return VK_NULL_HANDLE;
        }
        pipelines.emplace(key, p);
        return p;
    }
};

VulkanModule::VulkanModule() = default;
VulkanModule::~VulkanModule() = default;
VulkanModule::VulkanModule(VulkanModule&&) noexcept = default;
VulkanModule& VulkanModule::operator=(VulkanModule&&) noexcept = default;

bool VulkanModule::valid() const noexcept { return s_ != nullptr; }
void VulkanModule::reset() { s_.reset(); }
uint32_t VulkanModule::subgroup_size() const noexcept { return s_ ? s_->subgroup_size : 0; }

const std::string& VulkanModule::entry() const {
    static const std::string empty;
    return s_ ? s_->kernel.entry : empty;
}

VulkanModule VulkanModule::load(const brass::target::SpirvKernel& kernel, std::string* error,
                                const VulkanModuleOptions& opts) {
    VulkanModule out;
    auto fail = [&](const std::string& msg) {
        vk::set_error(msg);
        if (error) *error = msg;
        return VulkanModule{};
    };
    Runtime* r = vk::runtime();
    if (!r) return fail(vk::last_error());
    if (std::string missing = r->caps.missing_for(kernel); !missing.empty())
        return fail("kernel '" + kernel.entry + "' cannot run on " + r->caps.info.name + ":\n" + missing);
    if (kernel.words.empty()) return fail("kernel '" + kernel.entry + "' has no SPIR-V words");

    auto s = std::make_unique<State>();
    s->kernel = kernel;
    if (opts.subgroup_size && r->caps.can_require_subgroup_size_in_compute &&
        opts.subgroup_size >= r->caps.min_subgroup_size && opts.subgroup_size <= r->caps.max_subgroup_size)
        s->subgroup_size = opts.subgroup_size;

    auto smi = vk::make<VkShaderModuleCreateInfo>(VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
    smi.codeSize = kernel.words.size() * 4;
    smi.pCode = kernel.words.data();
    if (r->vkCreateShaderModule(r->device, &smi, nullptr, &s->shader) != VK_SUCCESS)
        return fail("vkCreateShaderModule failed for '" + kernel.entry + "'");

    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, kernel.push_constant_bytes};
    auto lci = vk::make<VkPipelineLayoutCreateInfo>(VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
    lci.pushConstantRangeCount = kernel.push_constant_bytes ? 1u : 0u;
    lci.pPushConstantRanges = &range;
    if (r->vkCreatePipelineLayout(r->device, &lci, nullptr, &s->layout) != VK_SUCCESS)
        return fail("vkCreatePipelineLayout failed for '" + kernel.entry + "'");

    // Create the default-size pipeline now, so a driver compile failure
    // surfaces at load like a CUDA module JIT error.
    std::string err;
    if (s->pipeline(*r, kernel.local_size, err) == VK_NULL_HANDLE) return fail(err);
    out.s_ = std::move(s);
    return out;
}

namespace {

// Packs the push-constant block; "" or the mismatch.
std::string pack_args(const brass::target::SpirvKernel& k, const std::vector<VulkanArg>& args,
                      std::vector<uint8_t>& block) {
    if (args.size() != k.params.size())
        return "kernel '" + k.entry + "' takes " + std::to_string(k.params.size()) + " arguments, got " +
               std::to_string(args.size());
    block.assign(k.push_constant_bytes, 0);
    for (size_t i = 0; i < args.size(); ++i) {
        const brass::target::SpirvParam& p = k.params[i];
        VulkanArg::Kind want;
        if (p.type.is_pointer() || p.type.is_i64()) want = VulkanArg::Kind::U64;
        else if (p.type.is_i32()) want = VulkanArg::Kind::U32;
        else if (p.type.kind() == TypeKind::F32) want = VulkanArg::Kind::F32;
        else want = VulkanArg::Kind::F64;
        if (args[i].kind() != want)
            return "kernel '" + k.entry + "' argument " + std::to_string(i) + " has the wrong kind for its " +
                   std::to_string(p.size) + "-byte parameter";
        uint64_t bits = args[i].bits();
        std::memcpy(block.data() + p.offset, &bits, p.size);  // little-endian
    }
    return "";
}

} // namespace

bool VulkanModule::launch(const VulkanDispatch& d, const std::vector<VulkanArg>& args, std::string* error) const {
    auto fail = [&](const std::string& msg) {
        vk::set_error(msg);
        if (error) *error = msg;
        return false;
    };
    Runtime* r = vk::runtime();
    if (!r) return fail(vk::last_error());
    if (!s_) return fail("launch on an empty VulkanModule");
    const brass::target::SpirvKernel& k = s_->kernel;
    if (!k.local_size_spec_constants &&
        (d.block[0] != k.local_size[0] || d.block[1] != k.local_size[1] || d.block[2] != k.local_size[2]))
        return fail("kernel '" + k.entry + "' has a fixed LocalSize; the dispatch block must match it");
    uint64_t threads = uint64_t{d.block[0]} * d.block[1] * d.block[2];
    if (threads == 0 || threads > r->caps.max_workgroup_invocations || d.block[0] > r->caps.max_workgroup_size[0] ||
        d.block[1] > r->caps.max_workgroup_size[1] || d.block[2] > r->caps.max_workgroup_size[2])
        return fail("workgroup size " + std::to_string(d.block[0]) + "x" + std::to_string(d.block[1]) + "x" +
                    std::to_string(d.block[2]) + " exceeds the device limits");
    if (d.grid[0] == 0 || d.grid[1] == 0 || d.grid[2] == 0) return true;

    std::vector<uint8_t> block;
    if (std::string bad = pack_args(k, args, block); !bad.empty()) return fail(bad);
    std::string err;
    VkPipeline pipe = s_->pipeline(*r, d.block, err);
    if (!pipe) return fail(err);

    const bool timed = d.gpu_ms && r->caps.timestamp_period_ns > 0.0;
    VkQueryPool qp = VK_NULL_HANDLE;
    if (timed) {
        auto qci = vk::make<VkQueryPoolCreateInfo>(VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO);
        qci.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qci.queryCount = 2;
        if (r->vkCreateQueryPool(r->device, &qci, nullptr, &qp) != VK_SUCCESS) return fail("vkCreateQueryPool failed");
    }
    uint32_t repeat = d.repeat ? d.repeat : 1;
    bool ok = r->submit([&](VkCommandBuffer cb) {
        r->vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        if (!block.empty())
            r->vkCmdPushConstants(cb, s_->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(block.size()),
                                  block.data());
        if (timed) {
            r->vkCmdResetQueryPool(cb, qp, 0, 2);
            r->vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, 0);
        }
        for (uint32_t i = 0; i < repeat; ++i) {
            if (i) {
                auto mb = vk::make<VkMemoryBarrier>(VK_STRUCTURE_TYPE_MEMORY_BARRIER);
                mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                r->vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                        0, 1, &mb, 0, nullptr, 0, nullptr);
            }
            r->vkCmdDispatch(cb, d.grid[0], d.grid[1], d.grid[2]);
        }
        if (timed) r->vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, 1);
    });
    if (timed) {
        uint64_t ts[2] = {0, 0};
        if (ok && r->vkGetQueryPoolResults(r->device, qp, 0, 2, sizeof(ts), ts, 8,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) == VK_SUCCESS)
            *d.gpu_ms = static_cast<double>(ts[1] - ts[0]) * r->caps.timestamp_period_ns * 1e-6;
        else
            *d.gpu_ms = -1.0;
        r->vkDestroyQueryPool(r->device, qp, nullptr);
    } else if (d.gpu_ms) {
        *d.gpu_ms = -1.0;
    }
    if (!ok) return fail("kernel '" + k.entry + "': " + vk::last_error());
    return true;
}

bool VulkanModule::launch_1d(uint32_t grid, uint32_t block, std::initializer_list<VulkanArg> args,
                             std::string* error) const {
    VulkanDispatch d;
    d.grid[0] = grid;
    d.block[0] = block;
    return launch(d, std::vector<VulkanArg>(args), error);
}

} // namespace brass::gpu

#else // !BRASS_VULKAN_HEADERS

namespace brass::gpu {

struct VulkanModule::State {};
VulkanModule::VulkanModule() = default;
VulkanModule::~VulkanModule() = default;
VulkanModule::VulkanModule(VulkanModule&&) noexcept = default;
VulkanModule& VulkanModule::operator=(VulkanModule&&) noexcept = default;
bool VulkanModule::valid() const noexcept { return false; }
void VulkanModule::reset() {}
uint32_t VulkanModule::subgroup_size() const noexcept { return 0; }
const std::string& VulkanModule::entry() const {
    static const std::string empty;
    return empty;
}
VulkanModule VulkanModule::load(const brass::target::SpirvKernel&, std::string* error, const VulkanModuleOptions&) {
    if (error) *error = vulkan_last_error();
    return {};
}
bool VulkanModule::launch(const VulkanDispatch&, const std::vector<VulkanArg>&, std::string* error) const {
    if (error) *error = vulkan_last_error();
    return false;
}
bool VulkanModule::launch_1d(uint32_t, uint32_t, std::initializer_list<VulkanArg>, std::string* error) const {
    if (error) *error = vulkan_last_error();
    return false;
}

} // namespace brass::gpu

#endif
