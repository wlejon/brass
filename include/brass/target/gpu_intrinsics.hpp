#pragma once

#include <string_view>
#include <unordered_map>

namespace brass::target {

// Maps backend-neutral `gpu_*` intrinsic names to their canonical internal
// lowering name (`ptx_*`). Returns `sym` unchanged when `sym` is already
// canonical or not a `gpu_*` intrinsic alias.
inline std::string_view canonical_gpu_intrinsic(std::string_view sym) noexcept {
    if (sym.rfind("gpu_", 0) != 0) return sym;
    static const std::unordered_map<std::string_view, std::string_view> kAliases = {
        // Thread & grid geometry
        {"gpu_tid_x", "ptx_tid_x"}, {"gpu_tid_y", "ptx_tid_y"}, {"gpu_tid_z", "ptx_tid_z"},
        {"gpu_ntid_x", "ptx_ntid_x"}, {"gpu_ntid_y", "ptx_ntid_y"}, {"gpu_ntid_z", "ptx_ntid_z"},
        {"gpu_ctaid_x", "ptx_ctaid_x"}, {"gpu_ctaid_y", "ptx_ctaid_y"}, {"gpu_ctaid_z", "ptx_ctaid_z"},
        {"gpu_nctaid_x", "ptx_nctaid_x"}, {"gpu_nctaid_y", "ptx_nctaid_y"}, {"gpu_nctaid_z", "ptx_nctaid_z"},
        {"gpu_global_tid_x", "ptx_global_tid_x"}, {"gpu_global_id_x", "ptx_global_id_x"},
        {"gpu_laneid", "ptx_laneid"}, {"gpu_lane_id", "ptx_lane_id"},
        {"gpu_subgroup_id", "ptx_subgroup_id"},
        {"gpu_warpid", "ptx_warpid"}, {"gpu_warp_id", "ptx_warp_id"},
        {"gpu_nwarpid", "ptx_nwarpid"}, {"gpu_smid", "ptx_smid"}, {"gpu_nsmid", "ptx_nsmid"},
        {"gpu_clock", "ptx_clock"}, {"gpu_clock64", "ptx_clock64"}, {"gpu_globaltimer", "ptx_globaltimer"},

        // Barriers & shuffles
        {"gpu_sync", "ptx_sync"},
        {"gpu_bar_sync", "ptx_bar_sync"},
        {"gpu_bar_sync_count", "ptx_bar_sync_count"},
        {"gpu_subgroup_barrier", "ptx_subgroup_barrier"},
        {"gpu_shfl_down_f32", "ptx_shfl_down_f32"},
        {"gpu_shfl_down_sync_f32", "ptx_shfl_down_sync_f32"},
        {"gpu_shfl_up_f32", "ptx_shfl_up_f32"},
        {"gpu_shfl_bfly_f32", "ptx_shfl_bfly_f32"},
        {"gpu_shfl_xor_f32", "ptx_shfl_xor_f32"},
        {"gpu_shfl_idx_f32", "ptx_shfl_idx_f32"},
        {"gpu_shfl_down_i32", "ptx_shfl_down_i32"},
        {"gpu_shfl_up_i32", "ptx_shfl_up_i32"},
        {"gpu_shfl_bfly_i32", "ptx_shfl_bfly_i32"},
        {"gpu_shfl_xor_i32", "ptx_shfl_xor_i32"},
        {"gpu_shfl_idx_i32", "ptx_shfl_idx_i32"},

        // Fast math & conversions
        {"gpu_rcp", "ptx_rcp"}, {"gpu_rcp_approx", "ptx_rcp_approx"}, {"gpu_div_approx", "ptx_div_approx"},
        {"gpu_rsqrt", "ptx_rsqrt"}, {"gpu_sqrt", "ptx_sqrt"}, {"gpu_sqrt_rn", "ptx_sqrt_rn"},
        {"gpu_exp", "ptx_exp"}, {"gpu_log", "ptx_log"}, {"gpu_ex2", "ptx_ex2"}, {"gpu_lg2", "ptx_lg2"},
        {"gpu_sin", "ptx_sin"}, {"gpu_cos", "ptx_cos"}, {"gpu_tan", "ptx_tan"},
        {"gpu_asin", "ptx_asin"}, {"gpu_acos", "ptx_acos"}, {"gpu_atan", "ptx_atan"}, {"gpu_atan2", "ptx_atan2"},
        {"gpu_sinh", "ptx_sinh"}, {"gpu_cosh", "ptx_cosh"}, {"gpu_tanh", "ptx_tanh"},
        {"gpu_pow", "ptx_pow"}, {"gpu_fabs", "ptx_fabs"}, {"gpu_fmin", "ptx_fmin"}, {"gpu_fmax", "ptx_fmax"},
        {"gpu_clamp", "ptx_clamp"}, {"gpu_mix", "ptx_mix"}, {"gpu_step", "ptx_step"},
        {"gpu_smoothstep", "ptx_smoothstep"}, {"gpu_sign", "ptx_sign"}, {"gpu_fract", "ptx_fract"},
        {"gpu_fma", "ptx_fma"}, {"gpu_ldexp_f32", "ptx_ldexp_f32"},
        {"gpu_f16_to_f32", "ptx_f16_to_f32"}, {"gpu_f16_hi_to_f32", "ptx_f16_hi_to_f32"},
        {"gpu_f32_to_f16", "ptx_f32_to_f16"}, {"gpu_pack_f16x2", "ptx_pack_f16x2"},
        {"gpu_bf16_to_f32", "ptx_bf16_to_f32"}, {"gpu_f32_to_bf16", "ptx_f32_to_bf16"},
        {"gpu_f32_to_f64", "ptx_f32_to_f64"}, {"gpu_f64_to_f32", "ptx_f64_to_f32"},
        {"gpu_u32_to_f32", "ptx_u32_to_f32"}, {"gpu_i32_to_f32", "ptx_i32_to_f32"},
        {"gpu_i64_to_f32", "ptx_i64_to_f32"}, {"gpu_u64_to_f32", "ptx_u64_to_f32"},
        {"gpu_f32_to_u32", "ptx_f32_to_u32"}, {"gpu_f32_to_i32", "ptx_f32_to_i32"},
        {"gpu_sdot4", "ptx_sdot4"},
        {"gpu_mul_wide_u32", "ptx_mul_wide_u32"}, {"gpu_mul_wide_s32", "ptx_mul_wide_s32"},
        {"gpu_mul_hi_u32", "ptx_mul_hi_u32"}, {"gpu_mad_lo_u32", "ptx_mad_lo_u32"},

        // Shared memory
        {"gpu_shared_alloc_f32", "ptx_shared_alloc_f32"}, {"gpu_shared_alloc_i32", "ptx_shared_alloc_i32"},
        {"gpu_shared_alloc_f64", "ptx_shared_alloc_f64"}, {"gpu_shared_alloc_i64", "ptx_shared_alloc_i64"},
        {"gpu_shared_alloc_i32x4", "ptx_shared_alloc_i32x4"}, {"gpu_shared_alloc_u16", "ptx_shared_alloc_u16"},
        {"gpu_shared_load_f32", "ptx_shared_load_f32"}, {"gpu_shared_load_i32", "ptx_shared_load_i32"},
        {"gpu_shared_load_f64", "ptx_shared_load_f64"}, {"gpu_shared_load_i64", "ptx_shared_load_i64"},
        {"gpu_shared_load_f32_indexed", "ptx_shared_load_f32_indexed"},
        {"gpu_shared_load_i32_indexed", "ptx_shared_load_i32_indexed"},
        {"gpu_shared_load_f64_indexed", "ptx_shared_load_f64_indexed"},
        {"gpu_shared_load_i64_indexed", "ptx_shared_load_i64_indexed"},
        {"gpu_shared_load_i32x4_indexed", "ptx_shared_load_i32x4_indexed"},
        {"gpu_shared_load_u16_indexed", "ptx_shared_load_u16_indexed"},
        {"gpu_shared_load_f32_from_v4", "ptx_shared_load_f32_from_v4"},
        {"gpu_shared_store_f32", "ptx_shared_store_f32"}, {"gpu_shared_store_i32", "ptx_shared_store_i32"},
        {"gpu_shared_store_f64", "ptx_shared_store_f64"}, {"gpu_shared_store_i64", "ptx_shared_store_i64"},
        {"gpu_shared_store_f32_indexed", "ptx_shared_store_f32_indexed"},
        {"gpu_shared_store_i32_indexed", "ptx_shared_store_i32_indexed"},
        {"gpu_shared_store_f64_indexed", "ptx_shared_store_f64_indexed"},
        {"gpu_shared_store_i64_indexed", "ptx_shared_store_i64_indexed"},
        {"gpu_shared_store_i32x4_indexed", "ptx_shared_store_i32x4_indexed"},
        {"gpu_shared_store_u16_indexed", "ptx_shared_store_u16_indexed"},

        // Narrow memory, read-only cache loads (.nc) & atomics
        {"gpu_ldg_f32", "ptx_ldg_f32"}, {"gpu_ldg_i32", "ptx_ldg_i32"},
        {"gpu_ldg_i64", "ptx_ldg_i64"}, {"gpu_ldg_f32x4", "ptx_ldg_f32x4"},
        {"gpu_ldg_i32x4", "ptx_ldg_i32x4"},
        {"gpu_load_u8", "ptx_load_u8"}, {"gpu_load_s8", "ptx_load_s8"},
        {"gpu_load_u16", "ptx_load_u16"}, {"gpu_load_s16", "ptx_load_s16"},
        {"gpu_store_u8", "ptx_store_u8"}, {"gpu_store_u16", "ptx_store_u16"},
        {"gpu_atom_add_f32", "ptx_atom_add_f32"}, {"gpu_atom_add_i32", "ptx_atom_add_i32"},
        {"gpu_atom_add_u32", "ptx_atom_add_u32"}, {"gpu_atom_add_i64", "ptx_atom_add_i64"},
        {"gpu_atom_min_i32", "ptx_atom_min_i32"}, {"gpu_atom_min_u32", "ptx_atom_min_u32"},
        {"gpu_atom_max_i32", "ptx_atom_max_i32"}, {"gpu_atom_max_u32", "ptx_atom_max_u32"},
        {"gpu_atom_exch_i32", "ptx_atom_exch_i32"}, {"gpu_atom_exch_u32", "ptx_atom_exch_u32"},
        {"gpu_atom_exch_f32", "ptx_atom_exch_f32"},
        {"gpu_atom_and_i32", "ptx_atom_and_i32"}, {"gpu_atom_and_u32", "ptx_atom_and_u32"},
        {"gpu_atom_or_i32", "ptx_atom_or_i32"}, {"gpu_atom_or_u32", "ptx_atom_or_u32"},
        {"gpu_atom_xor_i32", "ptx_atom_xor_i32"}, {"gpu_atom_xor_u32", "ptx_atom_xor_u32"},
        {"gpu_atom_cas_i32", "ptx_atom_cas_i32"}, {"gpu_atom_cas_u32", "ptx_atom_cas_u32"},
        {"gpu_atom_shared_add_f32", "ptx_atom_shared_add_f32"},
        {"gpu_atom_shared_add_i32", "ptx_atom_shared_add_i32"},
        {"gpu_atom_shared_add_u32", "ptx_atom_shared_add_u32"},
        {"gpu_atom_shared_min_i32", "ptx_atom_shared_min_i32"},
        {"gpu_atom_shared_min_u32", "ptx_atom_shared_min_u32"},
        {"gpu_atom_shared_max_i32", "ptx_atom_shared_max_i32"},
        {"gpu_atom_shared_max_u32", "ptx_atom_shared_max_u32"},
        {"gpu_atom_shared_exch_i32", "ptx_atom_shared_exch_i32"},
        {"gpu_atom_shared_exch_u32", "ptx_atom_shared_exch_u32"},
        {"gpu_atom_shared_exch_f32", "ptx_atom_shared_exch_f32"},
        {"gpu_atom_shared_and_i32", "ptx_atom_shared_and_i32"},
        {"gpu_atom_shared_and_u32", "ptx_atom_shared_and_u32"},
        {"gpu_atom_shared_or_i32", "ptx_atom_shared_or_i32"},
        {"gpu_atom_shared_or_u32", "ptx_atom_shared_or_u32"},
        {"gpu_atom_shared_xor_i32", "ptx_atom_shared_xor_i32"},
        {"gpu_atom_shared_xor_u32", "ptx_atom_shared_xor_u32"},
        {"gpu_atom_shared_cas_i32", "ptx_atom_shared_cas_i32"},
        {"gpu_atom_shared_cas_u32", "ptx_atom_shared_cas_u32"},

        // Cooperative matrix 16x16x16
        {"gpu_coop_alloc_a", "ptx_coop_alloc_a"}, {"gpu_coop_alloc_b", "ptx_coop_alloc_b"},
        {"gpu_coop_alloc_acc", "ptx_coop_alloc_acc"}, {"gpu_coop_fill_acc", "ptx_coop_fill_acc"},
        {"gpu_coop_load_a", "ptx_coop_load_a"}, {"gpu_coop_load_b", "ptx_coop_load_b"},
        {"gpu_coop_load_acc", "ptx_coop_load_acc"}, {"gpu_coop_mma", "ptx_coop_mma"},
        {"gpu_coop_store_acc", "ptx_coop_store_acc"}, {"gpu_coop_scale_acc", "ptx_coop_scale_acc"},
        {"gpu_coop_add_acc", "ptx_coop_add_acc"}, {"gpu_coop_acc_len", "ptx_coop_acc_len"},
        {"gpu_coop_get_acc", "ptx_coop_get_acc"}, {"gpu_coop_set_acc", "ptx_coop_set_acc"},
    };
    auto it = kAliases.find(sym);
    return it == kAliases.end() ? sym : it->second;
}

} // namespace brass::target
