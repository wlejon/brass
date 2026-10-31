#pragma once

// SpirvTarget: MIR kernels -> SPIR-V 1.5 compute shaders for Vulkan 1.2+.
// The same MIR that PtxTarget lowers for NVIDIA lowers here through
// spirv::SpirvISel (structurizer + instruction selection) and the
// spirv::verify structural check. See docs/spirv_backend_design.md.
//
// Kernel ABI (the contract with the Vulkan runtime):
//   - pointers are buffer device addresses (VK_KHR_buffer_device_address,
//     core in Vulkan 1.2): every kernel parameter -- ptr, i32, i64, f32,
//     f64 -- is a member of one push-constant block, in declaration order,
//     each at the next offset aligned to its own size (`SpirvKernel::params`
//     lists the offsets); there are no descriptor sets;
//   - the workgroup size is SpecId 0/1/2 (default `local_size_*`) when
//     `local_size_spec_constants` is on, else the fixed LocalSize;
//   - shared arrays are Workgroup variables (static; nothing at dispatch).

#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace brass::target {

enum class SpirvStage : uint8_t {
    Compute = 0,
    Vertex = 1,
    Fragment = 2,
};

// A descriptor a kernel or stage reads (KernelBuilder's texture, image and
// buffer calls): a combined image sampler, an image sampled through a
// separate sampler, that sampler, a storage image, a uniform buffer or a
// storage buffer.
enum class SpirvDescriptorKind : uint8_t {
    CombinedImageSampler = 0,
    SampledImage = 1,
    Sampler = 2,
    StorageImage = 3,
    UniformBuffer = 4,
    StorageBuffer = 5,
};

// An image's shape (SPIR-V Dim; a cube's coordinates are a direction).
enum class SpirvImageDim : uint8_t { D1 = 0, D2 = 1, D3 = 2, Cube = 3 };
// What its texels read as: f32 lanes, or u32 / i32 lanes of an integer format.
enum class SpirvImageScalar : uint8_t { F32 = 0, U32 = 1, I32 = 2 };

// A stage input's or output's description (spirv_stage_in / _out): its
// lanes (1..4) in bits 0-3, its interpolation as a fragment input, and
// whether its lanes are 32-bit integers rather than f32.
constexpr uint32_t kSpirvStageFlat = 0x10, kSpirvStageNoPerspective = 0x20, kSpirvStageInt = 0x100;

struct SpirvDescriptorBinding {
    uint32_t set = 0;
    uint32_t binding = 0;
    SpirvDescriptorKind kind = SpirvDescriptorKind::StorageImage;
    // The image's (none for a Sampler).
    SpirvImageDim dim = SpirvImageDim::D2;
    bool arrayed = false;
    bool depth = false;                       // compared against (a shadow map)
    SpirvImageScalar scalar = SpirvImageScalar::F32;
    uint32_t format = 0;                      // a storage image's spv::ImageFormat; 0 (Unknown) when sampled
    uint32_t bytes = 0;                       // a uniform buffer's block size (a storage buffer's is unbounded)

    bool operator==(const SpirvDescriptorBinding& o) const noexcept {
        return set == o.set && binding == o.binding && kind == o.kind && dim == o.dim && arrayed == o.arrayed &&
               depth == o.depth && scalar == o.scalar && format == o.format && bytes == o.bytes;
    }
};

struct SpirvOptions {
    SpirvStage stage = SpirvStage::Compute;
    uint32_t local_size_x = 256;
    uint32_t local_size_y = 1;
    uint32_t local_size_z = 1;
    // Workgroup size as specialization constants (SpecId 0, 1, 2 with the
    // values above as defaults) so the runtime picks the block size at
    // pipeline creation, as a PTX launch does; off fixes it at emit time.
    bool local_size_spec_constants = true;
    // SPIR-V version word: 1.5 (Vulkan 1.2) or 1.6 (Vulkan 1.3). Earlier
    // versions are rejected (PhysicalStorageBuffer and 8/16-bit storage are
    // core from 1.5).
    uint32_t spirv_version = 0x00010500;
    // Every float add / sub / mul / div / negate / rem is NoContraction, so
    // the driver rounds each as MIR does (no FMA formation, and NVIDIA's
    // compiler then also keeps a*b + a*c from becoming a*(b + c)): the
    // kernel computes what the CPU JIT computes, at the cost of the fusions.
    // An explicit fma is still fused.
    bool precise_float = false;
};

struct SpirvParam {
    Type type;           // MIR parameter type (ptr is a 64-bit device address)
    uint32_t offset = 0; // byte offset in the push-constant block
    uint32_t size = 0;
};

// One compiled kernel plus what the runtime needs to launch it.
struct SpirvKernel {
    std::string entry;                     // OpEntryPoint name (the MIR function name)
    SpirvStage stage = SpirvStage::Compute;
    std::vector<uint32_t> words;           // the SPIR-V binary
    std::vector<SpirvParam> params;        // push-constant layout, in parameter order
    std::vector<SpirvDescriptorBinding> descriptors; // descriptor-bound images/samplers
    uint32_t push_constant_bytes = 0;
    uint32_t shared_bytes = 0;             // total Workgroup storage
    uint32_t local_size[3] = {1, 1, 1};       // the LocalSize default
    bool local_size_spec_constants = true;    // SpecId 0..2 override local_size
    std::vector<std::string> capabilities; // e.g. "Int64", "Float64", "GroupNonUniformShuffle"
    std::vector<std::string> extensions;   // e.g. "SPV_EXT_shader_atomic_float_add"
};

class SpirvTarget {
public:
    // One kernel per module; throws std::runtime_error naming the kernel and
    // block for anything the target cannot lower (and, as a compiler bug, if
    // the lowered module fails spirv::verify).
    static std::vector<uint32_t> emit_function(const Function& fn, const SpirvOptions& opts = {});
    // Every function of `mod` as an entry point of one SPIR-V module.
    static std::vector<uint32_t> emit_module(const Module& mod, const SpirvOptions& opts = {});
    static SpirvKernel compile(const Function& fn, const SpirvOptions& opts = {});
    // Text listing of the lowered, verified module (spirv-dis-like ids and
    // opcode names) for tests and diagnostics; needs no external tools.
    static std::string dump_function(const Function& fn, const SpirvOptions& opts = {});
};

} // namespace brass::target
