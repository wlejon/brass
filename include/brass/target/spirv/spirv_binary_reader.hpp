#pragma once

// Binary SPIR-V reader (`parse_binary`) and static compute shader prover
// (`audit_compute_shader` / `audit_compute_shader_binary`).
//
// Lifts standard SPIR-V 1.0..1.6 word streams (glslang, shaderc/glslc, or
// brass's SpirvTarget) into a typed `spirv::Module` and verifies:
//   1. SPIR-V structural & SSA dominance invariants
//   2. Uniform control-flow barrier & subgroup collective legality
//   3. Workgroup `shared` memory array bounds safety (via BVSolver)
//   4. Division / reciprocal / sqrt hazard analysis

#include <brass/target/spirv/spirv_ir.hpp>

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace brass::spirv {

// Parses a binary SPIR-V word stream (little-endian 32-bit words, starting
// with 0x07230203) into `out_mod`. Returns true on success; on failure, writes
// a human-readable error message to `*err_out` (when non-null) and returns
// false.
bool parse_binary(std::span<const uint32_t> words, Module& out_mod, std::string* err_out);

enum class HazardKind : uint8_t {
    StructuralOrDominance,
    DivergentBarrier,
    DivergentSubgroupOp,
    SharedMemoryOutOfBounds,
    DefiniteDivisionByZero,
    DefiniteNegativeSqrt,
};

struct ShaderHazard {
    HazardKind kind = HazardKind::StructuralOrDominance;
    std::string function;
    Id block = 0;
    Id inst_result = 0;
    spv::Op op = spv::OpNop;
    std::string message;
};

struct DivisionAuditEntry {
    std::string function;
    Id block = 0;
    Id result = 0;
    spv::Op op = spv::OpNop;     // OpFDiv, OpUDiv, OpSDiv, or OpExtInst (Sqrt / InverseSqrt)
    uint32_t ext_op = 0;         // GLSL.std.450 opcode when op == OpExtInst
    Id operand = 0;              // divisor or sqrt argument id
    bool proven_safe = false;    // guarded by constant, max(..., eps), + eps, exp, branch guard, etc.
    std::string guard_reason;    // how safety was established (or why unguarded)
};

struct SharedMemoryAccessAudit {
    std::string function;
    Id block = 0;
    Id access_chain = 0;
    Id variable = 0;
    Id index_id = 0;
    uint32_t array_length = 0;
    bool proven_in_bounds = false;
    bool statically_out_of_bounds = false;
    std::string detail;
};

struct BarrierAuditEntry {
    std::string function;
    Id block = 0;
    spv::Op op = spv::OpNop;     // OpControlBarrier or OpGroupNonUniform*
    bool is_workgroup_scope = false;
    bool uniform_control_flow = true;
    std::string detail;
};

struct ComputeShaderAuditReport {
    bool parse_ok = false;
    bool structural_ok = false;
    bool dominance_ok = false;
    bool barriers_uniform = false;
    bool shared_bounds_ok = false;
    bool div_sqrt_ok = false;

    std::array<uint32_t, 3> local_size = {1, 1, 1};
    uint32_t num_functions = 0;
    uint32_t num_blocks = 0;
    uint32_t num_instructions = 0;
    uint32_t num_workgroup_arrays = 0;

    std::vector<Diagnostic> verifier_diagnostics;
    std::vector<ShaderHazard> hazards;
    std::vector<BarrierAuditEntry> barriers;
    std::vector<SharedMemoryAccessAudit> shared_accesses;
    std::vector<DivisionAuditEntry> divisions;

    bool ok() const {
        return parse_ok && structural_ok && dominance_ok &&
               barriers_uniform && shared_bounds_ok && div_sqrt_ok &&
               hazards.empty();
    }
};

ComputeShaderAuditReport audit_compute_shader(const Module& mod);
ComputeShaderAuditReport audit_compute_shader_binary(std::span<const uint32_t> words);

} // namespace brass::spirv
