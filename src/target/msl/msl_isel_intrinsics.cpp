#include "msl_builder.hpp"

namespace brass::msl {

void MslFnBuilder::lower_call(const Instruction& inst, int d) {
    std::string_view sym = inst.symbol();
    if (is_shared_alloc(sym) || is_coop_alloc(sym)) return;
    if (sym.rfind("ptx_coop_", 0) == 0) {
        lower_coop_call(inst, d);
        return;
    }

    auto arg = [&](size_t i) { return val_expr(inst.operand(i)); };

    if (sym == "ptx_tid_x" || sym == "ptx_tid_y" || sym == "ptx_tid_z") {
        need_tid_ = true;
        char ax = sym.back();
        emit_assign(inst, std::string("static_cast<int32_t>(_tid.") + ax + ")", d);
        return;
    }
    if (sym == "ptx_ntid_x" || sym == "ptx_ntid_y" || sym == "ptx_ntid_z") {
        need_ntid_ = true;
        char ax = sym.back();
        emit_assign(inst, std::string("static_cast<int32_t>(_ntid.") + ax + ")", d);
        return;
    }
    if (sym == "ptx_ctaid_x" || sym == "ptx_ctaid_y" || sym == "ptx_ctaid_z") {
        need_ctaid_ = true;
        char ax = sym.back();
        emit_assign(inst, std::string("static_cast<int32_t>(_ctaid.") + ax + ")", d);
        return;
    }
    if (sym == "ptx_nctaid_x" || sym == "ptx_nctaid_y" || sym == "ptx_nctaid_z") {
        need_nctaid_ = true;
        char ax = sym.back();
        emit_assign(inst, std::string("static_cast<int32_t>(_nctaid.") + ax + ")", d);
        return;
    }
    if (sym == "ptx_global_tid_x" || sym == "ptx_global_id_x") {
        need_ctaid_ = need_ntid_ = need_tid_ = true;
        emit_assign(inst, "static_cast<int32_t>(_ctaid.x * _ntid.x + _tid.x)", d);
        return;
    }
    if (sym == "ptx_laneid" || sym == "ptx_lane_id") {
        need_lane_ = true;
        emit_assign(inst, "static_cast<int32_t>(_lane)", d);
        return;
    }
    if (sym == "ptx_subgroup_id") {
        need_sg_ = true;
        emit_assign(inst, "static_cast<int32_t>(_sg)", d);
        return;
    }

    if (sym == "bar.sync" || sym == "ptx_sync" || sym == "ptx_bar_sync" || sym == "ptx_bar_sync_count") {
        indent_line(d);
        body_ << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        return;
    }
    if (sym == "ptx_subgroup_barrier") {
        indent_line(d);
        body_ << "simdgroup_barrier(mem_flags::mem_threadgroup);\n";
        return;
    }

    if (sym == "ptx_shfl_xor_f32" || sym == "ptx_shfl_bfly_f32" ||
        sym == "ptx_shfl_xor_i32" || sym == "ptx_shfl_bfly_i32") {
        emit_assign(inst, "simd_shuffle_xor(" + arg(0) + ", static_cast<uint16_t>(" + arg(1) + "))", d);
        return;
    }
    if (sym == "ptx_shfl_down_f32" || sym == "ptx_shfl_down_i32") {
        emit_assign(inst, "simd_shuffle_down(" + arg(0) + ", static_cast<uint16_t>(" + arg(1) + "))", d);
        return;
    }
    if (sym == "ptx_shfl_up_f32" || sym == "ptx_shfl_up_i32") {
        emit_assign(inst, "simd_shuffle_up(" + arg(0) + ", static_cast<uint16_t>(" + arg(1) + "))", d);
        return;
    }
    if (sym == "ptx_shfl_idx_f32" || sym == "ptx_shfl_idx_i32") {
        emit_assign(inst, "simd_shuffle(" + arg(0) + ", static_cast<uint16_t>(" + arg(1) + "))", d);
        return;
    }

    if (sym == "ptx_shared_load_f32" || sym == "ptx_shared_load_i32") {
        std::string mt = (sym == "ptx_shared_load_i32") ? "int32_t" : "float";
        emit_assign(inst, "(*reinterpret_cast<const threadgroup " + mt + "*>(" + arg(0) + "))", d);
        return;
    }
    if (sym == "ptx_shared_load_f32_indexed" || sym == "ptx_shared_load_i32_indexed" ||
        sym == "ptx_shared_load_i32x4_indexed") {
        std::string mt = (sym == "ptx_shared_load_i32_indexed")   ? "int32_t"
                       : (sym == "ptx_shared_load_i32x4_indexed") ? "int4"
                                                                  : "float";
        emit_assign(inst, "(reinterpret_cast<const threadgroup " + mt + "*>(" + arg(0) + ")[" + arg(1) + "])", d);
        return;
    }
    if (sym == "ptx_shared_load_f32_from_v4") {
        emit_assign(inst, "(reinterpret_cast<const threadgroup float*>(" + arg(0) + ")[" + arg(1) + "])", d);
        return;
    }
    if (sym == "ptx_shared_load_u16_indexed") {
        emit_assign(inst, "static_cast<int32_t>(reinterpret_cast<const threadgroup uint16_t*>(" + arg(0) + ")[" + arg(1) + "])", d);
        return;
    }
    if (sym == "ptx_shared_store_f32" || sym == "ptx_shared_store_i32") {
        std::string mt = (sym == "ptx_shared_store_i32") ? "int32_t" : "float";
        indent_line(d);
        body_ << "*reinterpret_cast<threadgroup " << mt << "*>(" << arg(0) << ") = " << arg(1) << ";\n";
        return;
    }
    if (sym == "ptx_shared_store_f32_indexed" || sym == "ptx_shared_store_i32_indexed" ||
        sym == "ptx_shared_store_i32x4_indexed") {
        std::string mt = (sym == "ptx_shared_store_i32_indexed")   ? "int32_t"
                       : (sym == "ptx_shared_store_i32x4_indexed") ? "int4"
                                                                   : "float";
        indent_line(d);
        body_ << "reinterpret_cast<threadgroup " << mt << "*>(" << arg(0) << ")[" << arg(1) << "] = " << arg(2) << ";\n";
        return;
    }
    if (sym == "ptx_shared_store_u16_indexed") {
        indent_line(d);
        body_ << "reinterpret_cast<threadgroup uint16_t*>(" << arg(0) << ")[" << arg(1) << "] = static_cast<uint16_t>(" << arg(2) << ");\n";
        return;
    }

    if (sym == "ptx_load_u8")  { emit_assign(inst, "static_cast<int32_t>(*reinterpret_cast<const device uint8_t*>(" + arg(0) + "))", d); return; }
    if (sym == "ptx_load_s8")  { emit_assign(inst, "static_cast<int32_t>(*reinterpret_cast<const device int8_t*>(" + arg(0) + "))", d); return; }
    if (sym == "ptx_load_u16") { emit_assign(inst, "static_cast<int32_t>(*reinterpret_cast<const device uint16_t*>(" + arg(0) + "))", d); return; }
    if (sym == "ptx_load_s16") { emit_assign(inst, "static_cast<int32_t>(*reinterpret_cast<const device int16_t*>(" + arg(0) + "))", d); return; }
    if (sym == "ptx_store_u8") { indent_line(d); body_ << "*reinterpret_cast<device uint8_t*>(" << arg(0) << ") = static_cast<uint8_t>(" << arg(1) << ");\n"; return; }
    if (sym == "ptx_store_u16") { indent_line(d); body_ << "*reinterpret_cast<device uint16_t*>(" << arg(0) << ") = static_cast<uint16_t>(" << arg(1) << ");\n"; return; }

    if (sym == "ptx_mul_hi_u32") {
        emit_assign(inst, "as_type<int32_t>(mulhi(as_type<uint32_t>(" + arg(0) + "), as_type<uint32_t>(" + arg(1) + ")))", d);
        return;
    }
    if (sym == "ptx_mul_wide_u32") {
        emit_assign(inst, "static_cast<int64_t>(static_cast<uint64_t>(as_type<uint32_t>(" + arg(0) + ")) * static_cast<uint64_t>(as_type<uint32_t>(" + arg(1) + ")))", d);
        return;
    }
    if (sym == "ptx_mul_wide_s32") {
        emit_assign(inst, "(static_cast<int64_t>(" + arg(0) + ") * static_cast<int64_t>(" + arg(1) + "))", d);
        return;
    }
    if (sym == "ptx_mad_lo_u32") {
        emit_assign(inst, "as_type<int32_t>(as_type<uint32_t>(" + arg(0) + ") * as_type<uint32_t>(" + arg(1) + ") + as_type<uint32_t>(" + arg(2) + "))", d);
        return;
    }

    if (sym == "rsqrtf" || sym == "rsqrt" || sym == "ptx_rsqrt") { emit_assign(inst, "rsqrt(" + arg(0) + ")", d); return; }
    if (sym == "sqrtf" || sym == "sqrt" || sym == "ptx_sqrt" || sym == "ptx_sqrt_rn") { emit_assign(inst, "sqrt(" + arg(0) + ")", d); return; }
    if (sym == "sinf" || sym == "sin" || sym == "ptx_sin") { emit_assign(inst, "sin(" + arg(0) + ")", d); return; }
    if (sym == "cosf" || sym == "cos" || sym == "ptx_cos") { emit_assign(inst, "cos(" + arg(0) + ")", d); return; }
    if (sym == "tanhf" || sym == "ptx_tanh") { emit_assign(inst, "tanh(" + arg(0) + ")", d); return; }
    if (sym == "ex2f" || sym == "ex2" || sym == "ptx_ex2") { emit_assign(inst, "exp2(" + arg(0) + ")", d); return; }
    if (sym == "lg2f" || sym == "ptx_lg2") { emit_assign(inst, "log2(" + arg(0) + ")", d); return; }
    if (sym == "expf" || sym == "exp" || sym == "ptx_exp") { emit_assign(inst, "exp(" + arg(0) + ")", d); return; }
    if (sym == "logf" || sym == "log" || sym == "ptx_log") { emit_assign(inst, "log(" + arg(0) + ")", d); return; }
    if (sym == "ptx_rcp" || sym == "ptx_rcp_approx") { emit_assign(inst, "(1.0f / " + arg(0) + ")", d); return; }
    if (sym == "ptx_div_approx") { emit_assign(inst, "(" + arg(0) + " / " + arg(1) + ")", d); return; }
    if (sym == "fabsf" || sym == "fabs" || sym == "ptx_fabs") { emit_assign(inst, "fabs(" + arg(0) + ")", d); return; }
    if (sym == "fminf" || sym == "fmin" || sym == "ptx_fmin") { emit_assign(inst, "fmin(" + arg(0) + ", " + arg(1) + ")", d); return; }
    if (sym == "fmaxf" || sym == "fmax" || sym == "ptx_fmax") { emit_assign(inst, "fmax(" + arg(0) + ", " + arg(1) + ")", d); return; }
    if (sym == "clampf" || sym == "ptx_clamp") { emit_assign(inst, "clamp(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")", d); return; }
    if (sym == "mixf" || sym == "ptx_mix") { emit_assign(inst, "mix(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")", d); return; }
    if (sym == "ptx_fma") { emit_assign(inst, "fma(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")", d); return; }

    if (sym == "i32_to_f32" || sym == "ptx_i32_to_f32") { emit_assign(inst, "static_cast<float>(" + arg(0) + ")", d); return; }
    if (sym == "ptx_u32_to_f32") { emit_assign(inst, "static_cast<float>(as_type<uint32_t>(" + arg(0) + "))", d); return; }
    if (sym == "ptx_f32_to_i32") { emit_assign(inst, "static_cast<int32_t>(" + arg(0) + ")", d); return; }
    if (sym == "ptx_f32_to_u32") { emit_assign(inst, "as_type<int32_t>(static_cast<uint32_t>(" + arg(0) + "))", d); return; }
    if (sym == "ptx_f16_to_f32") {
        emit_assign(inst, "static_cast<float>(as_type<half>(static_cast<uint16_t>(as_type<uint32_t>(" + arg(0) + ") & 0xFFFFu)))", d);
        return;
    }
    if (sym == "ptx_f16_hi_to_f32") {
        emit_assign(inst, "static_cast<float>(as_type<half2>(" + arg(0) + ").y)", d);
        return;
    }
    if (sym == "ptx_f32_to_f16") {
        emit_assign(inst, "static_cast<int32_t>(static_cast<uint32_t>(as_type<uint16_t>(static_cast<half>(" + arg(0) + "))))", d);
        return;
    }
    if (sym == "ptx_pack_f16x2") {
        emit_assign(inst, "as_type<int32_t>(half2(" + arg(0) + ", " + arg(1) + "))", d);
        return;
    }
    if (sym == "ptx_f32_to_bf16") {
        res_.needs_bf16_helper = true;
        emit_assign(inst, "_brass_f32_to_bf16(" + arg(0) + ")", d);
        return;
    }
    if (sym == "ptx_bf16_to_f32") {
        emit_assign(inst, "as_type<float>(as_type<uint32_t>(" + arg(0) + ") << 16u)", d);
        return;
    }
    if (sym == "ptx_ldexp_f32") {
        emit_assign(inst, "ldexp(" + arg(0) + ", " + arg(1) + ")", d);
        return;
    }
    if (sym == "ptx_sdot4") {
        res_.needs_sdot4_helper = true;
        emit_assign(inst, "_brass_sdot4(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")", d);
        return;
    }

    fail_msl(fn_, "unsupported intrinsic '" + std::string(sym) + "'");
}

} // namespace brass::msl
