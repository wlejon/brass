#pragma once

#include "gpu_equiv_internal.hpp"
#include <spirv/unified1/GLSL.std.450.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass::verify::spirv_detail {

constexpr uint64_t kGpuLocalBase0 = 0x5000000000000000ULL;

struct SpirvAccessChainInfo {
    bool is_push_const = false;
    bool is_workgroup = false;
    bool is_local = false;
    uint32_t index = 0;       // push-constant member index or workgroup array index
    BVNodeId elem_idx = 0;
    bool is_direct = false;
    BVNodeId direct_val = 0;
    spirv::Id wg_cur_ty = 0;
    uint64_t local_base = 0;
    uint32_t local_elem_bytes = 4;
    spirv::Id local_cur_ty = 0;
};

struct SpirvLocalVarInfo {
    uint64_t base_addr = 0;
    spirv::Id pointee_ty = 0;
    uint32_t elem_bytes = 4;
};

struct SpirvModuleInfo {
    std::unordered_map<spirv::Id, uint32_t> builtin_map;
    std::unordered_map<spirv::Id, uint32_t> spec_id_map;
    std::unordered_map<spirv::Id, uint32_t> location_map;
    std::unordered_map<spirv::Id, uint32_t> array_strides;
    std::unordered_map<uint64_t, uint32_t> member_offsets;
    std::unordered_map<spirv::Id, std::vector<spirv::Id>> struct_members;
    std::unordered_map<spirv::Id, spirv::Id> value_types;
    spirv::Id pc_struct_id = 0;
    spirv::Id push_const_var = 0;

    std::unordered_map<spirv::Id, BVNodeId> global_vals;
    std::unordered_map<spirv::Id, std::vector<BVNodeId>> global_composites;
    std::unordered_map<spirv::Id, std::vector<BVNodeId>> var_composites;
    std::unordered_map<spirv::Id, BVNodeId> var_scalars;
    std::unordered_map<spirv::Id, uint32_t> workgroup_var_idx;
    std::unordered_map<spirv::Id, spirv::Id> workgroup_var_type;
    std::unordered_map<spirv::Id, SpirvLocalVarInfo> private_vars;
    std::unordered_map<spirv::Id, spirv::Id> array_len_id;
    std::unordered_map<spirv::Id, uint32_t> wg_size_spec_axis;
    std::unordered_map<spirv::Id, uint64_t> stage_out_base;
    std::unordered_map<spirv::Id, const spirv::Function*> fn_by_id;
    uint64_t next_local_id = 0;

    uint32_t resolve_array_dim(BVContext& ctx, spirv::Id ty_id, const spirv::TypeInfo* ati) const;
};

struct SpirvExecState {
    size_t block_idx = 0;
    BVNodeId cond = 0;
    std::unordered_map<spirv::Id, BVNodeId> vals = {};
    std::unordered_map<spirv::Id, std::vector<BVNodeId>> composites = {};
    std::unordered_map<spirv::Id, SpirvAccessChainInfo> chains = {};
    std::unordered_map<spirv::Id, SpirvLocalVarInfo> local_vars = {};
    std::unordered_map<uint64_t, BVNodeId> local_slots = {};
};

struct SpirvExecContext {
    BVContext& ctx;
    const spirv::Module& mod;
    GpuSymEnv& env;
    const EquivOptions& opts;
    SpirvModuleInfo& minfo;
    GpuExecTrace& trace;
    std::unordered_map<BVNodeId, BVNodeId> safe_div_orig;
    std::vector<GpuSymStore> sym_local_stores;
    std::unordered_set<uint64_t> dirty_local_bases;
    uint32_t barrier_seq = 0;
    uint32_t total_steps = 0;
    uint32_t step_budget = 512;
};

uint32_t spirv_type_width(const spirv::Module& mod, spirv::Id type_id, bool* out_is_float = nullptr);
uint32_t spirv_leaf_elem_bytes(const spirv::Module& mod, spirv::Id type_id);
const spirv::Function* find_spirv_entry_function(const spirv::Module& mod);

void collect_spirv_module_info(
    BVContext& ctx,
    const spirv::Module& mod,
    GpuSymEnv& env,
    GpuExecTrace& trace,
    SpirvModuleInfo& out);

bool callee_is_pure_scalar(
    const spirv::Module& mod,
    const spirv::Function& fn,
    const SpirvModuleInfo& minfo,
    std::unordered_set<spirv::Id>& visiting);

void init_local_variable(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    spirv::Id var_id,
    spirv::Id ptr_ty_id,
    spirv::Id init_id);

bool handle_glsl_ext_inst(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    const spirv::Inst& inst,
    uint32_t rw,
    BVNodeId& out_res);

bool execute_spirv_function_body(
    SpirvExecContext& sc,
    const spirv::Function& fn,
    SpirvExecState init_state,
    uint32_t call_depth,
    BVNodeId& out_ret_val,
    std::vector<BVNodeId>& out_ret_comp,
    std::unordered_map<uint64_t, BVNodeId>* out_local_slots = nullptr);

void handle_spirv_access_chain(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    const spirv::Inst& inst,
    const spirv::TypeInfo* rti,
    BVNodeId& out_res);

void handle_spirv_load(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    const spirv::Inst& inst,
    uint32_t rw,
    const spirv::TypeInfo* rti,
    BVNodeId& out_res);

void handle_spirv_store(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    const spirv::Inst& inst);

bool handle_spirv_function_call(
    SpirvExecContext& sc,
    SpirvExecState& cur,
    const spirv::Inst& inst,
    uint32_t call_depth,
    uint32_t rw,
    bool is_fl,
    const spirv::TypeInfo* rti,
    BVNodeId& out_res);

} // namespace brass::verify::spirv_detail
