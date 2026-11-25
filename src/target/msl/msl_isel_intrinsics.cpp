#include "msl_builder.hpp"

namespace brass::msl {

void MslFnBuilder::lower_call(const Instruction& inst, int d) {
    std::string_view sym = target::canonical_gpu_intrinsic(inst.symbol());
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
    if (sym == "ptx_shfl_down_sync_f32" || sym == "shfl_down_sync_f32") {
        emit_assign(inst, "simd_shuffle_down(" + arg(1) + ", static_cast<uint16_t>(" + arg(2) + "))", d);
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

    if (sym == "ptx_shared_load_f32" || sym == "ptx_shared_load_i32" || sym == "ptx_shared_load_i64") {
        std::string mt = (sym == "ptx_shared_load_i32") ? "int32_t"
                       : (sym == "ptx_shared_load_i64") ? "int64_t"
                                                        : "float";
        std::string addr = arg(0);
        if (inst.operand_count() > 1) addr = "(" + addr + " + static_cast<int64_t>(" + arg(1) + "))";
        emit_assign(inst, "(*reinterpret_cast<const threadgroup " + mt + "*>(" + addr + "))", d);
        return;
    }
    if (sym == "ptx_shared_load_f32_indexed" || sym == "ptx_shared_load_i32_indexed" ||
        sym == "ptx_shared_load_i64_indexed" || sym == "ptx_shared_load_i32x4_indexed") {
        std::string mt = (sym == "ptx_shared_load_i32_indexed")   ? "int32_t"
                       : (sym == "ptx_shared_load_i64_indexed")   ? "int64_t"
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
    if (sym == "ptx_shared_store_f32" || sym == "ptx_shared_store_i32" || sym == "ptx_shared_store_i64") {
        std::string mt = (sym == "ptx_shared_store_i32") ? "int32_t"
                       : (sym == "ptx_shared_store_i64") ? "int64_t"
                                                         : "float";
        std::string addr = arg(0);
        if (inst.operand_count() > 2) addr = "(" + addr + " + static_cast<int64_t>(" + arg(2) + "))";
        indent_line(d);
        body_ << "*reinterpret_cast<threadgroup " << mt << "*>(" << addr << ") = " << arg(1) << ";\n";
        return;
    }
    if (sym == "ptx_shared_store_f32_indexed" || sym == "ptx_shared_store_i32_indexed" ||
        sym == "ptx_shared_store_i64_indexed" || sym == "ptx_shared_store_i32x4_indexed") {
        std::string mt = (sym == "ptx_shared_store_i32_indexed")   ? "int32_t"
                       : (sym == "ptx_shared_store_i64_indexed")   ? "int64_t"
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

    if (sym == "ptx_load_u8" || sym == "ptx_load_s8" || sym == "ptx_load_u16" || sym == "ptx_load_s16") {
        std::string sp = ptr_space(inst.operand(0));
        const char* mt = (sym == "ptx_load_u8")  ? "uint8_t"
                       : (sym == "ptx_load_s8")  ? "int8_t"
                       : (sym == "ptx_load_u16") ? "uint16_t"
                                                 : "int16_t";
        std::string addr = arg(0);
        if (inst.operand_count() > 1) addr = "(" + addr + " + static_cast<int64_t>(" + arg(1) + "))";
        emit_assign(inst, "static_cast<int32_t>(*reinterpret_cast<const " + sp + " " + mt + "*>(" + addr + "))", d);
        return;
    }
    if (sym == "ptx_store_u8" || sym == "ptx_store_u16") {
        std::string sp = ptr_space(inst.operand(0));
        const char* mt = (sym == "ptx_store_u8") ? "uint8_t" : "uint16_t";
        std::string addr = arg(0);
        if (inst.operand_count() > 2) addr = "(" + addr + " + static_cast<int64_t>(" + arg(2) + "))";
        indent_line(d);
        body_ << "*reinterpret_cast<" << sp << " " << mt << "*>(" << addr << ") = static_cast<" << mt << ">(" << arg(1) << ");\n";
        return;
    }

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
    if (sym == "tanf" || sym == "tan" || sym == "ptx_tan") { emit_assign(inst, "tan(" + arg(0) + ")", d); return; }
    if (sym == "asinf" || sym == "asin" || sym == "ptx_asin") { emit_assign(inst, "asin(" + arg(0) + ")", d); return; }
    if (sym == "acosf" || sym == "acos" || sym == "ptx_acos") { emit_assign(inst, "acos(" + arg(0) + ")", d); return; }
    if (sym == "atanf" || sym == "atan" || sym == "ptx_atan") { emit_assign(inst, "atan(" + arg(0) + ")", d); return; }
    if (sym == "sinhf" || sym == "sinh" || sym == "ptx_sinh") { emit_assign(inst, "sinh(" + arg(0) + ")", d); return; }
    if (sym == "coshf" || sym == "cosh" || sym == "ptx_cosh") { emit_assign(inst, "cosh(" + arg(0) + ")", d); return; }
    if (sym == "atan2f" || sym == "atan2" || sym == "ptx_atan2") { emit_assign(inst, "atan2(" + arg(0) + ", " + arg(1) + ")", d); return; }
    if (sym == "powf" || sym == "pow" || sym == "ptx_pow") { emit_assign(inst, "pow(" + arg(0) + ", " + arg(1) + ")", d); return; }
    if (sym == "tanhf" || sym == "ptx_tanh") { emit_assign(inst, "tanh(" + arg(0) + ")", d); return; }
    if (sym == "signf" || sym == "ptx_sign") { emit_assign(inst, "sign(" + arg(0) + ")", d); return; }
    if (sym == "fractf" || sym == "ptx_fract") { emit_assign(inst, "fract(" + arg(0) + ")", d); return; }
    if (sym == "stepf" || sym == "ptx_step") { emit_assign(inst, "step(" + arg(0) + ", " + arg(1) + ")", d); return; }
    if (sym == "smoothstepf" || sym == "ptx_smoothstep") { emit_assign(inst, "smoothstep(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")", d); return; }
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
    if (sym == "ptx_i64_to_f32") { emit_assign(inst, "static_cast<float>(" + arg(0) + ")", d); return; }
    if (sym == "ptx_u64_to_f32") { emit_assign(inst, "static_cast<float>(as_type<uint64_t>(" + arg(0) + "))", d); return; }
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

    if (sym.rfind("ptx_atom_", 0) == 0) {
        const bool is_shared = sym.rfind("ptx_atom_shared_", 0) == 0;
        const std::string_view op = sym.substr(is_shared ? 16 : 9);
        const std::string sp = is_shared ? "threadgroup" : "device";
        if (op == "add_i64") fail_msl(fn_, "64-bit atomic add is not supported in Metal MSL");
        if (op == "add_f32") {
            res_.needs_atom_f32_helper = true;
            emit_assign(inst, "_brass_atom_add_f32(reinterpret_cast<" + sp + " atomic_uint*>(" + arg(0) + "), " + arg(1) + ")", d);
            return;
        }
        if (op == "exch_f32") {
            emit_assign(inst, "as_type<float>(atomic_exchange_explicit(reinterpret_cast<" + sp + " atomic_uint*>(" +
                               arg(0) + "), as_type<uint32_t>(" + arg(1) + "), memory_order_relaxed))", d);
            return;
        }
        if (op == "cas_i32" || op == "cas_u32") {
            res_.needs_atom_cas_helper = true;
            emit_assign(inst, "_brass_atom_cas_i32(reinterpret_cast<" + sp + " atomic_int*>(" +
                               arg(0) + "), " + arg(1) + ", " + arg(2) + ")", d);
            return;
        }
        if (op == "add_u32" || op == "min_u32" || op == "max_u32") {
            const char* fn_name = (op == "add_u32") ? "atomic_fetch_add_explicit"
                                : (op == "min_u32") ? "atomic_fetch_min_explicit"
                                                    : "atomic_fetch_max_explicit";
            emit_assign(inst, "as_type<int32_t>(" + std::string(fn_name) + "(reinterpret_cast<" + sp + " atomic_uint*>(" +
                               arg(0) + "), as_type<uint32_t>(" + arg(1) + "), memory_order_relaxed))", d);
            return;
        }
        const char* int_fn = (op == "add_i32")                    ? "atomic_fetch_add_explicit"
                           : (op == "min_i32")                    ? "atomic_fetch_min_explicit"
                           : (op == "max_i32")                    ? "atomic_fetch_max_explicit"
                           : (op == "and_i32" || op == "and_u32") ? "atomic_fetch_and_explicit"
                           : (op == "or_i32"  || op == "or_u32")  ? "atomic_fetch_or_explicit"
                           : (op == "xor_i32" || op == "xor_u32") ? "atomic_fetch_xor_explicit"
                           : (op == "exch_i32"|| op == "exch_u32")? "atomic_exchange_explicit"
                                                                  : nullptr;
        if (int_fn) {
            emit_assign(inst, std::string(int_fn) + "(reinterpret_cast<" + sp + " atomic_int*>(" +
                               arg(0) + "), " + arg(1) + ", memory_order_relaxed)", d);
            return;
        }
    }

    fail_msl(fn_, "unsupported intrinsic '" + std::string(sym) + "'");
}

} // namespace brass::msl
