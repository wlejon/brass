#pragma once

#define SPV_ENABLE_UTILITY_CODE 1
#include <brass/target/spirv/spirv_binary_reader.hpp>
#include <spirv/unified1/GLSL.std.450.h>

#include "verify/bv_solver.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
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
    static constexpr size_t kNoBlock = SIZE_MAX;

    size_t num_blocks = 0;
    std::unordered_map<Id, size_t> block_index;
    std::vector<std::vector<size_t>> succs;
    std::vector<std::vector<size_t>> preds;
    std::vector<uint8_t> reachable;
    std::vector<size_t> idom;
    std::vector<size_t> ipdom;
    std::vector<uint32_t> dom_tin;
    std::vector<uint32_t> dom_tout;
    std::vector<Id> branch_cond;
    // Per block `b`, sorted list of conditional branch headers `h` on which `b` is control-dependent.
    std::vector<std::vector<size_t>> control_dep_headers;

    bool dominates(size_t dom_blk, size_t sub_blk) const noexcept {
        return dom_blk < num_blocks && sub_blk < num_blocks &&
               reachable[dom_blk] && reachable[sub_blk] &&
               dom_tin[dom_blk] <= dom_tin[sub_blk] &&
               dom_tout[sub_blk] <= dom_tout[dom_blk];
    }

    bool is_control_dep_on(size_t blk, size_t header) const noexcept {
        if (blk >= control_dep_headers.size()) return false;
        const auto& hs = control_dep_headers[blk];
        return std::binary_search(hs.begin(), hs.end(), header);
    }
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
