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

struct SpirvOptions {
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
};

struct SpirvParam {
    Type type;           // MIR parameter type (ptr is a 64-bit device address)
    uint32_t offset = 0; // byte offset in the push-constant block
    uint32_t size = 0;
};

// One compiled kernel plus what the runtime needs to launch it.
struct SpirvKernel {
    std::string entry;                     // OpEntryPoint name (the MIR function name)
    std::vector<uint32_t> words;           // the SPIR-V binary
    std::vector<SpirvParam> params;        // push-constant layout, in parameter order
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
