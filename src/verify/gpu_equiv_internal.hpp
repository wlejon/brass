#pragma once

#include "bv_solver.hpp"
#include <brass/verify/gpu_equiv.hpp>
#include <brass/verify/gpu_kernel_prover.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace brass::verify {

constexpr uint64_t kGpuSharedBase0 = 0x3000000000000000ULL;
constexpr uint64_t kGpuSharedStride = 0x10000000ULL;
constexpr uint64_t kGpuSharedWindowHalf = 0x08000000ULL;

struct GpuSharedAllocInfo {
    uint32_t array_idx = 0;
    uint32_t elem_count = 0;
    uint32_t elem_bytes = 4;
    uint64_t sym_base_addr = 0;
};

struct GpuSymStore {
    BVNodeId guard = 0;
    BVNodeId addr = 0;
    uint32_t num_bytes = 0;
    BVNodeId val = 0;
};

struct GpuSymAccess {
    BVNodeId guard = 0;
    bool is_shared = false;
    bool is_store = false;
    uint32_t region_index = UINT32_MAX;
    BVNodeId elem_index_32 = 0;
    BVNodeId byte_offset_64 = 0;
    BVNodeId full_addr_64 = 0;
    uint32_t access_bytes = 4;
    uint32_t required_align = 4;
    uint32_t shared_elem_count = 0;
    uint32_t shared_elem_bytes = 4;
};

struct GpuSymBarrier {
    BVNodeId reach_cond = 0;
    BVNodeId barrier_id = 0;
    bool is_subgroup_collective = false;
    uint32_t inst_index = 0;
};

struct GpuExecTrace {
    bool supported = true;
    std::string error_msg;
    BVNodeId completed = 0;
    BVNodeId trapped = 0;
    BVNodeId returned = 0;
    BVNodeId domain_assumption = 0;
    std::vector<GpuSharedAllocInfo> shared_allocs;
    std::vector<GpuSymStore> global_stores;
    std::vector<GpuSymStore> shared_stores;
    std::vector<GpuSymAccess> accesses;
    std::vector<GpuSymBarrier> barriers;
};

struct GpuSymEnv {
    std::vector<BVNodeId> param_vars;
    std::vector<Type> param_types;
    BVNodeId tid[3] = {0, 0, 0};
    BVNodeId ntid[3] = {0, 0, 0};
    BVNodeId ctaid[3] = {0, 0, 0};
    BVNodeId nctaid[3] = {0, 0, 0};
    BVNodeId raw_subgroup_lane = 0;
    BVNodeId laneid = 0;
    BVNodeId lanebase = 0;
    BVNodeId global_tid[3] = {0, 0, 0};
    BVNodeId coord_bounds = 0;

    struct ByteEntry {
        BVNodeId addr = 0;
        BVNodeId val = 0;
    };
    std::vector<ByteEntry> global_init_bytes;
    std::vector<ByteEntry> shared_init_bytes;

    struct UfEntry {
        std::string sym;
        std::vector<BVNodeId> args;
        BVNodeId ret_val = 0;
    };
    std::vector<UfEntry> uf_calls;

    BVNodeId read_init_byte(BVContext& ctx, std::vector<ByteEntry>& table, std::string_view prefix, BVNodeId byte_addr);
    BVNodeId read_mem(BVContext& ctx, bool is_shared, const std::vector<GpuSymStore>& stores, BVNodeId addr, uint32_t num_bytes);
    BVNodeId eval_uf(BVContext& ctx, std::string_view sym, const std::vector<BVNodeId>& args, uint32_t ret_width, bool is_float = false);
    BVNodeId eval_shuffle_mode(BVContext& ctx, uint32_t mode, BVNodeId val32, BVNodeId delta32);
};

uint32_t gpu_type_bv_width(Type t) noexcept;
BVNodeId gpu_coerce_width(BVContext& ctx, BVNodeId node, uint32_t target_w, bool sign_ext = false);
uint32_t shift_for_elem_bytes(uint32_t elem_bytes) noexcept;
BVNodeId extract_shared_elem_index_32(BVContext& ctx, BVNodeId byte_off_64, uint32_t shift);
BVNodeId gpu_add_addr64(BVContext& ctx, BVNodeId a, BVNodeId b);
BVNodeId gpu_mul_or_shl(BVContext& ctx, uint32_t w, BVNodeId a, BVNodeId b);

inline const BVNode& bv_node(const BVContext& ctx, BVNodeId id) noexcept { return ctx.nodes()[id]; }
inline BVNodeId bv_true(BVContext& ctx) { return ctx.make_const(1, 1); }
inline BVNodeId bv_false(BVContext& ctx) { return ctx.make_const(0, 1); }
inline BVNodeId bv_const(BVContext& ctx, uint32_t w, uint64_t v) { return ctx.make_const(v, w); }
inline BVNodeId bv_bin(BVContext& ctx, BVOp op, uint32_t w, BVNodeId a, BVNodeId b) { return ctx.make_binary(op, a, b, w); }
inline BVNodeId bv_un(BVContext& ctx, BVOp op, uint32_t w, BVNodeId a) { return ctx.make_unary(op, a, w); }
inline BVNodeId bv_ite(BVContext& ctx, uint32_t /*w*/, BVNodeId c, BVNodeId t, BVNodeId f) { return ctx.make_ite(c, t, f); }
inline BVNodeId bv_ext(BVContext& ctx, bool is_signed, uint32_t target_w, BVNodeId a) {
    uint32_t cw = ctx.width(a);
    if (cw >= target_w) return a;
    return is_signed ? ctx.make_sext(a, target_w - cw) : ctx.make_zext(a, target_w - cw);
}
inline BVNodeId bv_extract(BVContext& ctx, uint32_t hi, uint32_t lo, BVNodeId a) { return ctx.make_extract(a, hi, lo); }

void record_global_access(
    BVContext& ctx,
    GpuExecTrace& trace,
    const GpuSymEnv& env,
    BVNodeId guard,
    bool is_store,
    BVNodeId full_addr,
    uint32_t access_bytes,
    uint32_t required_align = 0);

void record_shared_access(
    BVContext& ctx,
    GpuExecTrace& trace,
    BVNodeId guard,
    bool is_store,
    uint32_t array_idx,
    BVNodeId elem_idx_32,
    uint32_t access_bytes);

void init_gpu_sym_env(
    BVContext& ctx,
    GpuSymEnv& env,
    const std::vector<Type>& param_types,
    std::string_view thread_suffix = "",
    const uint32_t* fixed_ntid = nullptr,
    const uint32_t* max_tid_bounds = nullptr,
    const uint32_t* max_ctaid_bounds = nullptr);

bool resolve_shared_address(
    BVContext& ctx,
    BVNodeId addr_64,
    const std::vector<GpuSharedAllocInfo>& shared_allocs,
    uint32_t& out_array_idx,
    BVNodeId& out_byte_offset_64,
    BVNodeId& out_elem_index_32);

bool resolve_global_address(
    BVContext& ctx,
    BVNodeId addr_64,
    const GpuSymEnv& env,
    uint32_t& out_param_idx,
    BVNodeId& out_byte_offset_64);

BVNodeId make_shared_canonical_addr(
    BVContext& ctx,
    uint32_t array_idx,
    BVNodeId elem_index_32,
    uint32_t elem_bytes);

GpuExecTrace execute_gpu_mir(
    BVContext& ctx,
    const Function& fn,
    GpuSymEnv& env,
    const EquivOptions& opts);

GpuExecTrace execute_gpu_spirv(
    BVContext& ctx,
    const spirv::Module& mod,
    GpuSymEnv& env,
    const EquivOptions& opts);

GpuExecTrace execute_gpu_ptx(
    BVContext& ctx,
    const ptx::Function& fn,
    GpuSymEnv& env,
    const EquivOptions& opts);

EquivResult compare_gpu_traces(
    BVContext& ctx,
    GpuSymEnv& env,
    const GpuExecTrace& trace_a,
    const GpuExecTrace& trace_b,
    std::string_view label_a,
    std::string_view label_b,
    std::string* err_out,
    const EquivOptions& opts);

} // namespace brass::verify
