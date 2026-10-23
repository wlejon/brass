#include "gpu_equiv_internal.hpp"
#include <brass/target/ptx/ptx_cleanup.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/spirv/spirv_isel.hpp>
#include <brass/verify/gpu_equiv.h>
#include <algorithm>
#include <cstring>
#include <sstream>

namespace brass::verify {

uint32_t gpu_type_bv_width(Type t) noexcept {
    if (t.is_void()) return 0;
    if (t.is_pointer() || t.is_i64() || t.kind() == TypeKind::F64) return 64;
    if (t.is_i8()) return 8;
    if (t.is_i16()) return 16;
    return 32;
}

BVNodeId gpu_coerce_width(BVContext& ctx, BVNodeId node, uint32_t target_w, bool sign_ext) {
    if (target_w == 0) return 0;
    uint32_t cur_w = ctx.width(node);
    if (cur_w == target_w) return node;
    if (cur_w < target_w) return bv_ext(ctx, sign_ext, target_w, node);
    return bv_extract(ctx, target_w - 1, 0, node);
}

uint32_t shift_for_elem_bytes(uint32_t elem_bytes) noexcept {
    return (elem_bytes == 8) ? 3 : (elem_bytes == 4) ? 2 : (elem_bytes == 2) ? 1 : 0;
}

BVNodeId GpuSymEnv::read_init_byte(
    BVContext& ctx,
    std::vector<ByteEntry>& table,
    std::string_view prefix,
    BVNodeId byte_addr) {
    for (const auto& entry : table) {
        if (entry.addr == byte_addr) return entry.val;
    }
    std::string vname = std::string(prefix) + "_" + std::to_string(table.size());
    BVNodeId fresh = ctx.make_var(vname, 8);
    for (const auto& entry : table) {
        BVNodeId same = bv_bin(ctx, BVOp::Eq, 1, byte_addr, entry.addr);
        fresh = bv_ite(ctx, 8, same, entry.val, fresh);
    }
    table.push_back({byte_addr, fresh});
    return fresh;
}

BVNodeId GpuSymEnv::read_mem(
    BVContext& ctx,
    bool is_shared,
    const std::vector<GpuSymStore>& stores,
    BVNodeId addr,
    uint32_t num_bytes) {
    addr = gpu_coerce_width(ctx, addr, 64, false);
    auto& init_table = is_shared ? shared_init_bytes : global_init_bytes;
    std::string_view prefix = is_shared ? "gpu_smem_b" : "gpu_gmem_b";
    BVNodeId result = 0;
    for (uint32_t b = 0; b < num_bytes; ++b) {
        BVNodeId b_addr = (b == 0)
            ? addr
            : gpu_add_addr64(ctx, addr, bv_const(ctx, 64, b));
        BVNodeId byte_val = read_init_byte(ctx, init_table, prefix, b_addr);
        for (const auto& st : stores) {
            for (uint32_t sb = 0; sb < st.num_bytes; ++sb) {
                BVNodeId sb_addr = (sb == 0)
                    ? st.addr
                    : gpu_add_addr64(ctx, st.addr, bv_const(ctx, 64, sb));
                BVNodeId addr_eq = bv_bin(ctx, BVOp::Eq, 1, b_addr, sb_addr);
                BVNodeId hit = bv_bin(ctx, BVOp::And, 1, st.guard, addr_eq);
                if (bv_node(ctx, hit).op == BVOp::Const && bv_node(ctx, hit).imm == 0) continue;
                BVNodeId st_byte = bv_extract(ctx, sb * 8 + 7, sb * 8, st.val);
                byte_val = bv_ite(ctx, 8, hit, st_byte, byte_val);
            }
        }
        result = (b == 0) ? byte_val : ctx.make_concat(byte_val, result);
    }
    return result;
}

BVNodeId GpuSymEnv::eval_uf(
    BVContext& ctx,
    std::string_view sym,
    const std::vector<BVNodeId>& args,
    uint32_t ret_width,
    bool is_float) {
    bool all_const = true;
    for (BVNodeId a : args) {
        if (bv_node(ctx, a).op != BVOp::Const) { all_const = false; break; }
    }
    if (all_const && is_float && args.size() == 2 && ret_width == 32) {
        uint32_t u0 = static_cast<uint32_t>(bv_node(ctx, args[0]).imm);
        uint32_t u1 = static_cast<uint32_t>(bv_node(ctx, args[1]).imm);
        float f0 = 0.0f, f1 = 0.0f;
        std::memcpy(&f0, &u0, 4);
        std::memcpy(&f1, &u1, 4);
        float fres = 0.0f;
        bool folded = true;
        if (sym == "fp_add") fres = f0 + f1;
        else if (sym == "fp_sub") fres = f0 - f1;
        else if (sym == "fp_mul") fres = f0 * f1;
        else if (sym == "fp_div") fres = f0 / f1;
        else folded = false;
        if (folded) {
            uint32_t ures = 0;
            std::memcpy(&ures, &fres, 4);
            return bv_const(ctx, 32, ures);
        }
    }
    for (const auto& entry : uf_calls) {
        if (entry.sym == sym && entry.args == args && ctx.width(entry.ret_val) == ret_width) {
            return entry.ret_val;
        }
    }
    std::string vname = "uf_" + std::string(sym) + "_" + std::to_string(uf_calls.size());
    BVNodeId res = ctx.make_var(vname, ret_width);
    for (const auto& entry : uf_calls) {
        if (entry.sym != sym || entry.args.size() != args.size() || ctx.width(entry.ret_val) != ret_width) {
            continue;
        }
        BVNodeId args_eq = bv_true(ctx);
        for (size_t i = 0; i < args.size(); ++i) {
            if (ctx.width(args[i]) != ctx.width(entry.args[i])) {
                args_eq = bv_false(ctx);
                break;
            }
            BVNodeId eq_i = bv_bin(ctx, BVOp::Eq, 1, args[i], entry.args[i]);
            args_eq = bv_bin(ctx, BVOp::And, 1, args_eq, eq_i);
        }
        res = bv_ite(ctx, ret_width, args_eq, entry.ret_val, res);
    }
    uf_calls.push_back({std::string(sym), args, res});
    return res;
}

BVNodeId GpuSymEnv::eval_shuffle_mode(BVContext& ctx, uint32_t mode, BVNodeId val32, BVNodeId delta32) {
    val32 = gpu_coerce_width(ctx, val32, 32, false);
    delta32 = gpu_coerce_width(ctx, delta32, 32, false);
    BVNodeId c31 = bv_const(ctx, 32, 31);
    BVNodeId src = 0;
    if (mode == 0) { // down
        BVNodeId j = bv_bin(ctx, BVOp::Add, 32, raw_subgroup_lane, delta32);
        BVNodeId seg_plus_d = bv_bin(ctx, BVOp::Add, 32, laneid, delta32);
        BVNodeId ok = bv_bin(ctx, BVOp::Ule, 1, seg_plus_d, c31);
        src = bv_ite(ctx, 32, ok, j, raw_subgroup_lane);
    } else if (mode == 1) { // up
        BVNodeId j = bv_bin(ctx, BVOp::Sub, 32, raw_subgroup_lane, delta32);
        BVNodeId ok = bv_bin(ctx, BVOp::Uge, 1, laneid, delta32);
        src = bv_ite(ctx, 32, ok, j, raw_subgroup_lane);
    } else if (mode == 2) { // bfly / xor
        BVNodeId d_masked = bv_bin(ctx, BVOp::And, 32, delta32, c31);
        BVNodeId x = bv_bin(ctx, BVOp::Xor, 32, laneid, d_masked);
        src = bv_bin(ctx, BVOp::Or, 32, lanebase, x);
    } else { // idx
        BVNodeId d_masked = bv_bin(ctx, BVOp::And, 32, delta32, c31);
        src = bv_bin(ctx, BVOp::Or, 32, lanebase, d_masked);
    }
    return eval_uf(ctx, "gpu_shfl", {val32, src}, 32, false);
}

void init_gpu_sym_env(
    BVContext& ctx,
    GpuSymEnv& env,
    const std::vector<Type>& param_types,
    std::string_view thread_suffix,
    const uint32_t* fixed_ntid,
    const uint32_t* max_tid_bounds,
    const uint32_t* max_ctaid_bounds) {
    env.param_types = param_types;
    env.param_vars.clear();
    for (size_t i = 0; i < param_types.size(); ++i) {
        uint32_t w = gpu_type_bv_width(param_types[i]);
        if (w == 0) w = 64;
        env.param_vars.push_back(ctx.make_var("gpu_param_" + std::to_string(i), w));
    }
    std::string suf(thread_suffix);
    const char* dims[3] = {"x", "y", "z"};
    BVNodeId bounds = bv_true(ctx);
    for (int d = 0; d < 3; ++d) {
        env.tid[d] = ctx.make_var(std::string("gpu_tid_") + dims[d] + suf, 32);
        env.ctaid[d] = ctx.make_var(std::string("gpu_ctaid_") + dims[d] + suf, 32);
        if (fixed_ntid) {
            env.ntid[d] = bv_const(ctx, 32, fixed_ntid[d]);
        } else {
            env.ntid[d] = ctx.make_var(std::string("gpu_ntid_") + dims[d], 32);
        }
        env.nctaid[d] = ctx.make_var(std::string("gpu_nctaid_") + dims[d], 32);
        env.global_tid[d] = bv_bin(
            ctx, BVOp::Add, 32,
            bv_bin(ctx, BVOp::Mul, 32, env.ctaid[d], env.ntid[d]),
            env.tid[d]);
        uint32_t tid_lim = max_tid_bounds ? max_tid_bounds[d] : (d == 0 ? 1024u : 64u);
        uint32_t cta_lim = max_ctaid_bounds ? max_ctaid_bounds[d] : 65535u;
        bounds = bv_bin(ctx, BVOp::And, 1, bounds,
            bv_bin(ctx, BVOp::Ult, 1, env.tid[d], bv_const(ctx, 32, tid_lim)));
        bounds = bv_bin(ctx, BVOp::And, 1, bounds,
            bv_bin(ctx, BVOp::Ult, 1, env.ctaid[d], bv_const(ctx, 32, cta_lim)));
        if (!fixed_ntid) {
            bounds = bv_bin(ctx, BVOp::And, 1, bounds,
                bv_bin(ctx, BVOp::Ugt, 1, env.ntid[d], bv_const(ctx, 32, 0)));
            bounds = bv_bin(ctx, BVOp::And, 1, bounds,
                bv_bin(ctx, BVOp::Ule, 1, env.ntid[d], bv_const(ctx, 32, 1024)));
            bounds = bv_bin(ctx, BVOp::And, 1, bounds,
                bv_bin(ctx, BVOp::Ult, 1, env.tid[d], env.ntid[d]));
        }
    }
    env.raw_subgroup_lane = ctx.make_var("gpu_raw_lane" + suf, 32);
    env.laneid = bv_bin(ctx, BVOp::And, 32, env.raw_subgroup_lane, bv_const(ctx, 32, 31));
    env.lanebase = bv_bin(ctx, BVOp::And, 32, env.raw_subgroup_lane, bv_const(ctx, 32, ~31u));
    bounds = bv_bin(ctx, BVOp::And, 1, bounds,
        bv_bin(ctx, BVOp::Ult, 1, env.raw_subgroup_lane, bv_const(ctx, 32, 64)));
    env.coord_bounds = bounds;
}

BVNodeId extract_shared_elem_index_32(BVContext& ctx, BVNodeId byte_off_64, uint32_t shift) {
    if (ctx.width(byte_off_64) == 32) {
        return shift == 0 ? byte_off_64
                          : bv_bin(ctx, BVOp::LShr, 32, byte_off_64, bv_const(ctx, 32, shift));
    }
    const BVNode& n = bv_node(ctx, byte_off_64);
    if (n.op == BVOp::Const) {
        auto signed_off = static_cast<int64_t>(n.imm);
        return bv_const(ctx, 32, static_cast<uint32_t>(signed_off >> shift));
    }
    if (n.op == BVOp::Add) {
        BVNodeId lhs = extract_shared_elem_index_32(ctx, n.a, shift);
        BVNodeId rhs = extract_shared_elem_index_32(ctx, n.b, shift);
        return bv_bin(ctx, BVOp::Add, 32, lhs, rhs);
    }
    if (n.op == BVOp::Shl) {
        const BVNode& sh = bv_node(ctx, n.b);
        if (sh.op == BVOp::Const && sh.imm == shift) {
            const BVNode& inner = bv_node(ctx, n.a);
            if ((inner.op == BVOp::SExt || inner.op == BVOp::ZExt) && ctx.width(inner.a) == 32) {
                return inner.a;
            }
            return bv_extract(ctx, 31, 0, n.a);
        }
    }
    if (n.op == BVOp::Mul) {
        const BVNode& mul_r = bv_node(ctx, n.b);
        if (mul_r.op == BVOp::Const && mul_r.imm == (1ULL << shift)) {
            const BVNode& inner = bv_node(ctx, n.a);
            if ((inner.op == BVOp::SExt || inner.op == BVOp::ZExt) && ctx.width(inner.a) == 32) {
                return inner.a;
            }
            return bv_extract(ctx, 31, 0, n.a);
        }
    }
    if ((n.op == BVOp::SExt || n.op == BVOp::ZExt) && ctx.width(n.a) == 32) {
        const BVNode& inner = bv_node(ctx, n.a);
        if (inner.op == BVOp::Shl && bv_node(ctx, inner.b).op == BVOp::Const &&
            bv_node(ctx, inner.b).imm == shift) {
            return inner.a;
        }
        return shift == 0 ? n.a
                          : bv_bin(ctx, BVOp::LShr, 32, n.a, bv_const(ctx, 32, shift));
    }
    BVNodeId shifted = shift == 0
        ? byte_off_64
        : bv_bin(ctx, BVOp::LShr, 64, byte_off_64, bv_const(ctx, 64, shift));
    return bv_extract(ctx, 31, 0, shifted);
}

namespace {

bool split_base_from_expr(
    BVContext& ctx,
    BVNodeId expr,
    BVNodeId target_base,
    BVNodeId& out_rem) {
    if (expr == target_base) {
        out_rem = bv_const(ctx, 64, 0);
        return true;
    }
    const BVNode& n = bv_node(ctx, expr);
    if (n.op == BVOp::Const && bv_node(ctx, target_base).op == BVOp::Const) {
        uint64_t base_c = bv_node(ctx, target_base).imm;
        int64_t diff = static_cast<int64_t>(n.imm - base_c);
        if (diff >= -static_cast<int64_t>(kGpuSharedWindowHalf) &&
            diff < static_cast<int64_t>(kGpuSharedWindowHalf)) {
            out_rem = bv_const(ctx, 64, static_cast<uint64_t>(diff));
            return true;
        }
    }
    if (n.op == BVOp::Add) {
        BVNodeId sub_rem = 0;
        if (split_base_from_expr(ctx, n.a, target_base, sub_rem)) {
            out_rem = gpu_add_addr64(ctx, sub_rem, n.b);
            return true;
        }
        if (split_base_from_expr(ctx, n.b, target_base, sub_rem)) {
            out_rem = gpu_add_addr64(ctx, n.a, sub_rem);
            return true;
        }
    }
    return false;
}

} // namespace

bool resolve_shared_address(
    BVContext& ctx,
    BVNodeId addr_64,
    const std::vector<GpuSharedAllocInfo>& shared_allocs,
    uint32_t& out_array_idx,
    BVNodeId& out_byte_offset_64,
    BVNodeId& out_elem_index_32) {
    addr_64 = gpu_coerce_width(ctx, addr_64, 64, false);
    for (const auto& alloc : shared_allocs) {
        BVNodeId base_node = bv_const(ctx, 64, alloc.sym_base_addr);
        BVNodeId rem = 0;
        if (split_base_from_expr(ctx, addr_64, base_node, rem)) {
            out_array_idx = alloc.array_idx;
            out_byte_offset_64 = rem;
            uint32_t shift = shift_for_elem_bytes(alloc.elem_bytes);
            out_elem_index_32 = extract_shared_elem_index_32(ctx, rem, shift);
            return true;
        }
    }
    return false;
}

bool resolve_global_address(
    BVContext& ctx,
    BVNodeId addr_64,
    const GpuSymEnv& env,
    uint32_t& out_param_idx,
    BVNodeId& out_byte_offset_64) {
    addr_64 = gpu_coerce_width(ctx, addr_64, 64, false);
    for (size_t i = 0; i < env.param_vars.size(); ++i) {
        if (ctx.width(env.param_vars[i]) != 64) continue;
        BVNodeId rem = 0;
        if (split_base_from_expr(ctx, addr_64, env.param_vars[i], rem)) {
            out_param_idx = static_cast<uint32_t>(i);
            out_byte_offset_64 = rem;
            return true;
        }
    }
    return false;
}

BVNodeId make_shared_canonical_addr(
    BVContext& ctx,
    uint32_t array_idx,
    BVNodeId elem_index_32,
    uint32_t elem_bytes) {
    elem_index_32 = gpu_coerce_width(ctx, elem_index_32, 32, false);
    uint32_t shift = shift_for_elem_bytes(elem_bytes);
    uint64_t base_addr = kGpuSharedBase0 + static_cast<uint64_t>(array_idx) * kGpuSharedStride;
    BVNodeId idx64 = bv_ext(ctx, false, 64, elem_index_32);
    BVNodeId byte_off = (shift == 0)
        ? idx64
        : bv_bin(ctx, BVOp::Shl, 64, idx64, bv_const(ctx, 64, shift));
    return bv_bin(ctx, BVOp::Add, 64, bv_const(ctx, 64, base_addr), byte_off);
}

BVNodeId gpu_add_addr64(BVContext& ctx, BVNodeId a, BVNodeId b) {
    a = gpu_coerce_width(ctx, a, 64, false);
    b = gpu_coerce_width(ctx, b, 64, true);
    const BVNode& na = bv_node(ctx, a);
    const BVNode& nb = bv_node(ctx, b);
    if (na.op == BVOp::Add && bv_node(ctx, na.b).op == BVOp::Const && nb.op == BVOp::Const) {
        return bv_bin(ctx, BVOp::Add, 64, na.a, bv_const(ctx, 64, bv_node(ctx, na.b).imm + nb.imm));
    }
    return bv_bin(ctx, BVOp::Add, 64, a, b);
}

BVNodeId gpu_mul_or_shl(BVContext& ctx, uint32_t w, BVNodeId a, BVNodeId b) {
    const BVNode& nb = bv_node(ctx, b);
    if (nb.op == BVOp::Const && nb.imm > 1 && (nb.imm & (nb.imm - 1)) == 0) {
        uint32_t sh = 0;
        while ((1ULL << sh) < nb.imm && sh < w) ++sh;
        if ((1ULL << sh) == nb.imm) {
            return bv_bin(ctx, BVOp::Shl, w, a, bv_const(ctx, w, sh));
        }
    }
    return bv_bin(ctx, BVOp::Mul, w, a, b);
}

void record_global_access(
    BVContext& ctx,
    GpuExecTrace& trace,
    const GpuSymEnv& env,
    BVNodeId guard,
    bool is_store,
    BVNodeId full_addr,
    uint32_t access_bytes,
    uint32_t required_align) {
    GpuSymAccess acc;
    acc.guard = guard;
    acc.is_shared = false;
    acc.is_store = is_store;
    acc.full_addr_64 = full_addr;
    acc.access_bytes = access_bytes;
    acc.required_align = required_align ? required_align : access_bytes;
    resolve_global_address(ctx, full_addr, env, acc.region_index, acc.byte_offset_64);
    trace.accesses.push_back(acc);
}

void record_shared_access(
    BVContext& ctx,
    GpuExecTrace& trace,
    BVNodeId guard,
    bool is_store,
    uint32_t array_idx,
    BVNodeId elem_idx_32,
    uint32_t access_bytes) {
    GpuSymAccess acc;
    acc.guard = guard;
    acc.is_shared = true;
    acc.is_store = is_store;
    acc.region_index = array_idx;
    acc.elem_index_32 = gpu_coerce_width(ctx, elem_idx_32, 32, false);
    uint32_t shift = shift_for_elem_bytes(access_bytes);
    acc.byte_offset_64 = bv_bin(
        ctx, BVOp::Shl, 64, bv_ext(ctx, true, 64, acc.elem_index_32), bv_const(ctx, 64, shift));
    acc.full_addr_64 = make_shared_canonical_addr(ctx, array_idx, acc.elem_index_32, access_bytes);
    acc.access_bytes = access_bytes;
    acc.required_align = access_bytes;
    if (array_idx < trace.shared_allocs.size()) {
        acc.shared_elem_count = trace.shared_allocs[array_idx].elem_count;
        acc.shared_elem_bytes = trace.shared_allocs[array_idx].elem_bytes;
    }
    trace.accesses.push_back(acc);
}

namespace {

bool stores_identically_equal(const std::vector<GpuSymStore>& a, const std::vector<GpuSymStore>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].guard != b[i].guard || a[i].addr != b[i].addr ||
            a[i].num_bytes != b[i].num_bytes || a[i].val != b[i].val) {
            return false;
        }
    }
    return true;
}

} // namespace

EquivResult compare_gpu_traces(
    BVContext& ctx,
    GpuSymEnv& env,
    const GpuExecTrace& trace_a,
    const GpuExecTrace& trace_b,
    std::string_view label_a,
    std::string_view label_b,
    std::string* err_out,
    const EquivOptions& opts) {
    if (!trace_a.supported || !trace_b.supported) {
        std::string msg = !trace_a.supported ? trace_a.error_msg : trace_b.error_msg;
        if (err_out) *err_out = msg;
        return EquivResult{EquivStatus::Unsupported, {}, msg};
    }

    BVNodeId domain = bv_bin(ctx, BVOp::And, 1, env.coord_bounds,
        bv_bin(ctx, BVOp::And, 1,
            bv_bin(ctx, BVOp::And, 1, trace_a.completed, trace_b.completed),
            bv_bin(ctx, BVOp::And, 1, trace_a.domain_assumption, trace_b.domain_assumption)));

    BVNodeId trap_diff = bv_bin(ctx, BVOp::Xor, 1, trace_a.trapped, trace_b.trapped);
    BVNodeId no_trap = bv_un(ctx, BVOp::Not, 1,
        bv_bin(ctx, BVOp::Or, 1, trace_a.trapped, trace_b.trapped));

    BVNodeId bar_diff = bv_false(ctx);
    size_t min_bars = std::min(trace_a.barriers.size(), trace_b.barriers.size());
    bool compare_bar_id = (label_a != "SPIR-V" && label_b != "SPIR-V");
    for (size_t i = 0; i < min_bars; ++i) {
        const auto& ba = trace_a.barriers[i];
        const auto& bb = trace_b.barriers[i];
        if (ba.is_subgroup_collective != bb.is_subgroup_collective) {
            bar_diff = bv_bin(ctx, BVOp::Or, 1, bar_diff,
                bv_bin(ctx, BVOp::Or, 1, ba.reach_cond, bb.reach_cond));
        } else {
            bar_diff = bv_bin(ctx, BVOp::Or, 1, bar_diff,
                bv_bin(ctx, BVOp::Xor, 1, ba.reach_cond, bb.reach_cond));
            if (compare_bar_id && !ba.is_subgroup_collective) {
                BVNodeId id_ne = bv_bin(ctx, BVOp::Ne, 1, ba.barrier_id, bb.barrier_id);
                bar_diff = bv_bin(ctx, BVOp::Or, 1, bar_diff,
                    bv_bin(ctx, BVOp::And, 1, ba.reach_cond, id_ne));
            }
        }
    }
    for (size_t i = min_bars; i < trace_a.barriers.size(); ++i) {
        bar_diff = bv_bin(ctx, BVOp::Or, 1, bar_diff, trace_a.barriers[i].reach_cond);
    }
    for (size_t i = min_bars; i < trace_b.barriers.size(); ++i) {
        bar_diff = bv_bin(ctx, BVOp::Or, 1, bar_diff, trace_b.barriers[i].reach_cond);
    }

    BVNodeId gmem_diff = bv_false(ctx);
    BVNodeId probe_gaddr = 0;
    if (!stores_identically_equal(trace_a.global_stores, trace_b.global_stores)) {
        probe_gaddr = ctx.make_var("gpu_probe_gaddr", 64);
        BVNodeId ga = env.read_mem(ctx, false, trace_a.global_stores, probe_gaddr, 1);
        BVNodeId gb = env.read_mem(ctx, false, trace_b.global_stores, probe_gaddr, 1);
        gmem_diff = bv_bin(ctx, BVOp::Ne, 1, ga, gb);
    }

    BVNodeId smem_diff = bv_false(ctx);
    BVNodeId probe_saddr = 0;
    if (!stores_identically_equal(trace_a.shared_stores, trace_b.shared_stores)) {
        probe_saddr = ctx.make_var("gpu_probe_saddr", 64);
        BVNodeId sa = env.read_mem(ctx, true, trace_a.shared_stores, probe_saddr, 1);
        BVNodeId sb = env.read_mem(ctx, true, trace_b.shared_stores, probe_saddr, 1);
        smem_diff = bv_bin(ctx, BVOp::Ne, 1, sa, sb);
    }

    BVNodeId post_trap_diff = bv_bin(ctx, BVOp::And, 1, no_trap,
        bv_bin(ctx, BVOp::Or, 1, bar_diff,
            bv_bin(ctx, BVOp::Or, 1, gmem_diff, smem_diff)));
    BVNodeId any_diff = bv_bin(ctx, BVOp::And, 1, domain,
        bv_bin(ctx, BVOp::Or, 1, trap_diff, post_trap_diff));

    SolveOutcome out = solve_nonzero(ctx, any_diff, opts.max_sat_conflicts);
    if (out.status == SolveOutcome::Unsat) {
        return EquivResult{EquivStatus::ProvenEquivalent, {}, {}};
    }
    if (out.status == SolveOutcome::Unknown) {
        std::string msg = "SAT solver conflict budget exhausted comparing " +
                          std::string(label_a) + " vs " + std::string(label_b);
        if (err_out) *err_out = msg;
        return EquivResult{EquivStatus::Unknown, {}, msg};
    }

    std::vector<uint64_t> node_vals;
    ctx.evaluate_all(out.model, node_vals);
    auto mval = [&](BVNodeId id) -> uint64_t {
        return id < node_vals.size() ? node_vals[id] : 0;
    };

    std::ostringstream oss;
    oss << "Counterexample (" << label_a << " != " << label_b << "): "
        << "tid=(" << mval(env.tid[0]) << "," << mval(env.tid[1]) << "," << mval(env.tid[2]) << ") "
        << "ctaid=(" << mval(env.ctaid[0]) << "," << mval(env.ctaid[1]) << "," << mval(env.ctaid[2]) << ") "
        << "laneid=" << mval(env.laneid);
    std::vector<RuntimeValue> cex;
    for (size_t i = 0; i < env.param_vars.size(); ++i) {
        uint64_t pv = mval(env.param_vars[i]);
        oss << " param_" << i << "=0x" << std::hex << pv << std::dec;
        cex.push_back(RuntimeValue::from_i64(static_cast<int64_t>(pv)));
    }
    if (mval(trap_diff) != 0) {
        oss << " [trap mismatch: " << label_a << "=" << mval(trace_a.trapped)
            << " vs " << label_b << "=" << mval(trace_b.trapped) << "]";
    } else if (mval(bar_diff) != 0) {
        oss << " [barrier/collective divergence]";
    } else if (probe_gaddr && mval(gmem_diff) != 0) {
        oss << " [global mem mismatch at 0x" << std::hex << mval(probe_gaddr) << std::dec << "]";
    } else if (probe_saddr && mval(smem_diff) != 0) {
        oss << " [shared mem mismatch at 0x" << std::hex << mval(probe_saddr) << std::dec << "]";
    }
    std::string diag = oss.str();
    if (err_out) *err_out = diag;
    return EquivResult{EquivStatus::Counterexample, std::move(cex), diag};
}

namespace {

bool extract_spirv_fixed_local_size(const spirv::Module& mod, uint32_t out_ntid[3]) {
    bool has_spec_wg = false;
    for (const auto& ann : mod.annotations) {
        if (ann.op == spv::OpDecorate && ann.operands.size() >= 3 &&
            ann.operands[1].value == spv::DecorationBuiltIn &&
            ann.operands[2].value == spv::BuiltInWorkgroupSize) {
            has_spec_wg = true;
            break;
        }
    }
    if (has_spec_wg) return false;
    for (const auto& em : mod.execution_modes) {
        if (em.op == spv::OpExecutionMode && em.operands.size() >= 5 &&
            em.operands[1].value == spv::ExecutionModeLocalSize) {
            out_ntid[0] = em.operands[2].value;
            out_ntid[1] = em.operands[3].value;
            out_ntid[2] = em.operands[4].value;
            return true;
        }
    }
    return false;
}

} // namespace

EquivResult verify_spirv_equivalence(
    const Function& mir_fn,
    const spirv::Module& spv_mod,
    std::string* err_out,
    const EquivOptions& opts) {
    BVContext ctx;
    GpuSymEnv env;
    uint32_t fixed_ntid[3] = {256, 1, 1};
    const uint32_t* ntid_ptr = extract_spirv_fixed_local_size(spv_mod, fixed_ntid) ? fixed_ntid : nullptr;
    init_gpu_sym_env(ctx, env, mir_fn.param_types(), "", ntid_ptr);

    GpuExecTrace mir_trace = execute_gpu_mir(ctx, mir_fn, env, opts);
    GpuExecTrace spv_trace = execute_gpu_spirv(ctx, spv_mod, env, opts);
    return compare_gpu_traces(ctx, env, mir_trace, spv_trace, "MIR", "SPIR-V", err_out, opts);
}

EquivResult verify_spirv_equivalence(
    const Function& mir_fn,
    const target::SpirvOptions& spv_opts,
    std::string* err_out,
    const EquivOptions& opts) {
    try {
        spirv::Module mod(spv_opts.spirv_version);
        spirv::SpirvISel isel(mod, spv_opts);
        isel.lower(mir_fn);
        return verify_spirv_equivalence(mir_fn, mod, err_out, opts);
    } catch (const std::exception& ex) {
        if (err_out) *err_out = ex.what();
        return EquivResult{EquivStatus::Unsupported, {}, ex.what()};
    }
}

EquivResult verify_spirv_ptx_equivalence(
    const Function& mir_fn,
    const spirv::Module& spv_mod,
    const ptx::Function& ptx_mod,
    std::string* err_out,
    const EquivOptions& opts) {
    BVContext ctx;
    GpuSymEnv env;
    uint32_t fixed_ntid[3] = {256, 1, 1};
    const uint32_t* ntid_ptr = extract_spirv_fixed_local_size(spv_mod, fixed_ntid) ? fixed_ntid : nullptr;
    init_gpu_sym_env(ctx, env, mir_fn.param_types(), "", ntid_ptr);

    GpuExecTrace mir_trace = execute_gpu_mir(ctx, mir_fn, env, opts);
    GpuExecTrace spv_trace = execute_gpu_spirv(ctx, spv_mod, env, opts);
    GpuExecTrace ptx_trace = execute_gpu_ptx(ctx, ptx_mod, env, opts);

    EquivResult r1 = compare_gpu_traces(ctx, env, mir_trace, spv_trace, "MIR", "SPIR-V", err_out, opts);
    if (!r1.is_equivalent()) return r1;
    EquivResult r2 = compare_gpu_traces(ctx, env, mir_trace, ptx_trace, "MIR", "PTX", err_out, opts);
    if (!r2.is_equivalent()) return r2;
    return compare_gpu_traces(ctx, env, spv_trace, ptx_trace, "SPIR-V", "PTX", err_out, opts);
}

EquivResult verify_spirv_ptx_equivalence(
    const Function& mir_fn,
    std::string* err_out,
    const EquivOptions& opts) {
    try {
        target::SpirvOptions spv_opts;
        spirv::Module spv_mod(spv_opts.spirv_version);
        spirv::SpirvISel spv_isel(spv_mod, spv_opts);
        spv_isel.lower(mir_fn);

        ptx::PtxISel ptx_isel;
        ptx::Function ptx_fn = ptx_isel.lower(mir_fn);
        ptx::cleanup(ptx_fn);

        return verify_spirv_ptx_equivalence(mir_fn, spv_mod, ptx_fn, err_out, opts);
    } catch (const std::exception& ex) {
        if (err_out) *err_out = ex.what();
        return EquivResult{EquivStatus::Unsupported, {}, ex.what()};
    }
}

} // namespace brass::verify

extern "C" {

static void copy_err_str(const std::string& s, char* buf, size_t cap) {
    if (!buf || cap == 0) return;
    size_t n = std::min(s.size(), cap - 1);
    std::memcpy(buf, s.data(), n);
    buf[n] = '\0';
}

int32_t brass_verify_spirv_equivalence(
    const struct BrassFunction_T* mir_fn,
    char* err_buf,
    size_t err_buf_cap) {
    if (!mir_fn) return -3;
    std::string err;
    auto res = brass::verify::verify_spirv_equivalence(
        *reinterpret_cast<const brass::Function*>(mir_fn),
        brass::target::SpirvOptions{},
        &err);
    copy_err_str(err, err_buf, err_buf_cap);
    return static_cast<int32_t>(res.status);
}

int32_t brass_verify_ptx_equivalence(
    const struct BrassFunction_T* mir_fn,
    char* err_buf,
    size_t err_buf_cap) {
    if (!mir_fn) return -3;
    std::string err;
    auto res = brass::verify::verify_ptx_equivalence(
        *reinterpret_cast<const brass::Function*>(mir_fn),
        brass::target::PtxOptions{},
        &err);
    copy_err_str(err, err_buf, err_buf_cap);
    return static_cast<int32_t>(res.status);
}

int32_t brass_verify_spirv_ptx_equivalence(
    const struct BrassFunction_T* mir_fn,
    char* err_buf,
    size_t err_buf_cap) {
    if (!mir_fn) return -3;
    std::string err;
    auto res = brass::verify::verify_spirv_ptx_equivalence(
        *reinterpret_cast<const brass::Function*>(mir_fn),
        &err);
    copy_err_str(err, err_buf, err_buf_cap);
    return static_cast<int32_t>(res.status);
}

} // extern "C"
