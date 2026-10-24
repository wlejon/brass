#pragma once

#define SPV_ENABLE_UTILITY_CODE 1
#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <spirv/unified1/GLSL.std.450.h>

#include "verify/bv_solver.hpp"

#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass::spirv::detail {

inline constexpr uint32_t kBuiltInLocalInvocationId = static_cast<uint32_t>(spv::BuiltInLocalInvocationId);
inline constexpr uint32_t kBuiltInGlobalInvocationId = static_cast<uint32_t>(spv::BuiltInGlobalInvocationId);
inline constexpr uint32_t kBuiltInLocalInvocationIndex = static_cast<uint32_t>(spv::BuiltInLocalInvocationIndex);
inline constexpr uint32_t kBuiltInWorkgroupSize = static_cast<uint32_t>(spv::BuiltInWorkgroupSize);
inline constexpr uint32_t kBuiltInSubgroupId = static_cast<uint32_t>(spv::BuiltInSubgroupId);
inline constexpr uint32_t kBuiltInSubgroupLocalInvocationId = static_cast<uint32_t>(spv::BuiltInSubgroupLocalInvocationId);

bool is_wg_divergent_builtin(uint32_t b);
bool is_sg_divergent_builtin(uint32_t b);
bool is_subgroup_collective_op(spv::Op op);
bool is_subgroup_uniformizing_op(const Inst& inst);

struct ModuleConstants {
    std::unordered_map<Id, uint64_t> int_consts;
    std::unordered_map<Id, double> float_consts;
    std::unordered_set<Id> zero_consts;
    std::unordered_map<Id, uint32_t> builtins;
    std::unordered_map<uint64_t, uint32_t> member_builtins; // (struct_id << 32) | member -> BuiltIn
    std::unordered_map<Id, spv::StorageClass> var_storage;
    std::unordered_map<Id, Id> var_pointee;
};

ModuleConstants collect_module_constants(const Module& mod);

struct FunctionCFG {
    size_t num_blocks = 0;
    std::unordered_map<Id, size_t> block_index;
    std::vector<std::vector<size_t>> succs;
    std::vector<std::vector<size_t>> preds;
    std::vector<uint8_t> reachable;
    std::vector<std::unordered_set<size_t>> dom;
    std::vector<std::unordered_set<size_t>> pdom;
    std::unordered_map<size_t, Id> branch_cond;
    std::unordered_map<size_t, std::unordered_set<size_t>> control_dep_region;
};

FunctionCFG build_cfg(const Function& fn);

struct SymbolicBuilder {
    const Module& mod;
    const Function& fn;
    const FunctionCFG& cfg;
    const ModuleConstants& mc;
    const std::unordered_map<Id, const Inst*>& def_inst;
    const std::unordered_map<Id, int>& lid_comp; // result_id -> 0(x), 1(y), 2(z), 3(local_index), 4(sg_local), 5(sg_id)
    const std::unordered_map<Id, Id>& single_store_val;
    verify::BVContext& ctx;
    std::array<verify::BVNodeId, 6> builtin_vars;
    std::unordered_map<Id, verify::BVNodeId> cache;
    std::vector<verify::BVNodeId> extra_constraints;

    verify::BVNodeId build(Id val_id, int depth = 0);
};

} // namespace brass::spirv::detail
