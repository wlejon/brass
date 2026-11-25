#include "msl_builder.hpp"

namespace brass::msl {

void MslFnBuilder::lower_coop_call(const Instruction& inst, int d) {
    std::string_view sym = target::canonical_gpu_intrinsic(inst.symbol());
    auto arg = [&](size_t i) { return val_expr(inst.operand(i)); };
    auto coop_slot = [&](const Value* v, const char* what) -> const std::pair<std::string, int>& {
        auto it = coop_vars_.find(v);
        if (it == coop_vars_.end()) {
            fail_msl(fn_, std::string(what) + " must come from ptx_coop_alloc_{a,b,acc}");
        }
        return it->second;
    };
    auto const_flag = [&](const Value* v) -> int32_t {
        if (!v || !v->is_instruction() || v->defining_instruction()->opcode() != Opcode::iconst_i32) {
            fail_msl(fn_, "coop flag must be a compile-time constant");
        }
        return v->defining_instruction()->imm_i32();
    };

    if (sym == "ptx_coop_fill_acc") {
        const auto& [name, use] = coop_slot(inst.operand(0), "ptx_coop_fill_acc target");
        if (use != 2) fail_msl(fn_, "ptx_coop_fill_acc requires an accumulator slot");
        std::string val = arg(1);
        for (int k = 0; k < 4; ++k) {
            indent_line(d);
            body_ << name << "[" << k << "] = make_filled_simdgroup_matrix<float, 8, 8>(" << val << ");\n";
        }
        return;
    }

    if (sym == "ptx_coop_load_a" || sym == "ptx_coop_load_b" || sym == "ptx_coop_load_acc") {
        const auto& [name, use] = coop_slot(inst.operand(0), "coop_load target");
        const int want_use = (sym == "ptx_coop_load_a") ? 0 : (sym == "ptx_coop_load_b") ? 1 : 2;
        if (use != want_use) fail_msl(fn_, "coop_load slot mismatch");
        bool col_major = const_flag(inst.operand(3)) != 0;
        bool is_f16 = (want_use != 2) || (inst.operand_count() > 4 && inst.operand(4) && const_flag(inst.operand(4)) != 0);
        std::string sp = ptr_space(inst.operand(1));
        std::string mtype = is_f16 ? "half" : "float";
        std::string base = "reinterpret_cast<const " + sp + " " + mtype + "*>(" + arg(1) + ")";
        std::string stride = "static_cast<ulong>(" + arg(2) + ")";
        for (int tr = 0; tr < 2; ++tr) {
            for (int tc = 0; tc < 2; ++tc) {
                int idx = tr * 2 + tc;
                int major = col_major ? (tc * 8) : (tr * 8);
                int minor = col_major ? (tr * 8) : (tc * 8);
                std::string ptr_expr = base + " + " + std::to_string(major) + "UL * " + stride + " + " + std::to_string(minor) + "UL";
                indent_line(d);
                if (want_use == 2 && is_f16) {
                    body_ << "{ simdgroup_half8x8 _cm_h; simdgroup_load(_cm_h, " << ptr_expr << ", " << stride
                          << ", ulong2(0, 0), " << (col_major ? "true" : "false") << "); "
                          << name << "[" << idx << "].thread_elements()[0] = float(_cm_h.thread_elements()[0]); "
                          << name << "[" << idx << "].thread_elements()[1] = float(_cm_h.thread_elements()[1]); }\n";
                } else {
                    body_ << "simdgroup_load(" << name << "[" << idx << "], " << ptr_expr << ", " << stride
                          << ", ulong2(0, 0), " << (col_major ? "true" : "false") << ");\n";
                }
            }
        }
        return;
    }

    if (sym == "ptx_coop_mma") {
        const auto& [acc_n, acc_u] = coop_slot(inst.operand(0), "ptx_coop_mma acc");
        const auto& [a_n, a_u]     = coop_slot(inst.operand(1), "ptx_coop_mma a");
        const auto& [b_n, b_u]     = coop_slot(inst.operand(2), "ptx_coop_mma b");
        if (acc_u != 2 || a_u != 0 || b_u != 1) fail_msl(fn_, "ptx_coop_mma requires (acc, a, b)");
        for (int tr = 0; tr < 2; ++tr) {
            for (int tc = 0; tc < 2; ++tc) {
                for (int tk = 0; tk < 2; ++tk) {
                    indent_line(d);
                    body_ << "simdgroup_multiply_accumulate("
                          << acc_n << "[" << (tr * 2 + tc) << "], "
                          << a_n << "[" << (tr * 2 + tk) << "], "
                          << b_n << "[" << (tk * 2 + tc) << "], "
                          << acc_n << "[" << (tr * 2 + tc) << "]);\n";
                }
            }
        }
        return;
    }

    if (sym == "ptx_coop_store_acc") {
        const auto& [name, use] = coop_slot(inst.operand(0), "ptx_coop_store_acc src");
        if (use != 2) fail_msl(fn_, "ptx_coop_store_acc requires an accumulator slot");
        bool col_major = const_flag(inst.operand(3)) != 0;
        bool is_f16 = (inst.operand_count() > 4 && inst.operand(4) && const_flag(inst.operand(4)) != 0);
        std::string sp = ptr_space(inst.operand(1));
        std::string mtype = is_f16 ? "half" : "float";
        std::string base = "reinterpret_cast<" + sp + " " + mtype + "*>(" + arg(1) + ")";
        std::string stride = "static_cast<ulong>(" + arg(2) + ")";
        for (int tr = 0; tr < 2; ++tr) {
            for (int tc = 0; tc < 2; ++tc) {
                int idx = tr * 2 + tc;
                int major = col_major ? (tc * 8) : (tr * 8);
                int minor = col_major ? (tr * 8) : (tc * 8);
                std::string ptr_expr = base + " + " + std::to_string(major) + "UL * " + stride + " + " + std::to_string(minor) + "UL";
                indent_line(d);
                if (is_f16) {
                    body_ << "{ simdgroup_half8x8 _cm_h; "
                          << "_cm_h.thread_elements()[0] = half(" << name << "[" << idx << "].thread_elements()[0]); "
                          << "_cm_h.thread_elements()[1] = half(" << name << "[" << idx << "].thread_elements()[1]); "
                          << "simdgroup_store(_cm_h, " << ptr_expr << ", " << stride
                          << ", ulong2(0, 0), " << (col_major ? "true" : "false") << "); }\n";
                } else {
                    body_ << "simdgroup_store(" << name << "[" << idx << "], " << ptr_expr << ", " << stride
                          << ", ulong2(0, 0), " << (col_major ? "true" : "false") << ");\n";
                }
            }
        }
        return;
    }

    if (sym == "ptx_coop_scale_acc") {
        const auto& [name, use] = coop_slot(inst.operand(0), "ptx_coop_scale_acc");
        if (use != 2) fail_msl(fn_, "ptx_coop_scale_acc requires an accumulator slot");
        std::string s = arg(1);
        for (int k = 0; k < 4; ++k) {
            indent_line(d);
            body_ << name << "[" << k << "].thread_elements() *= (" << s << ");\n";
        }
        return;
    }

    if (sym == "ptx_coop_add_acc") {
        const auto& [dst_n, dst_u] = coop_slot(inst.operand(0), "ptx_coop_add_acc dst");
        const auto& [rhs_n, rhs_u] = coop_slot(inst.operand(1), "ptx_coop_add_acc rhs");
        if (dst_u != 2 || rhs_u != 2) fail_msl(fn_, "ptx_coop_add_acc requires accumulator slots");
        for (int k = 0; k < 4; ++k) {
            indent_line(d);
            body_ << dst_n << "[" << k << "].thread_elements() += " << rhs_n << "[" << k << "].thread_elements();\n";
        }
        return;
    }

    if (sym == "ptx_coop_acc_len") {
        const auto& [name, use] = coop_slot(inst.operand(0), "ptx_coop_acc_len");
        (void)name;
        if (use != 2) fail_msl(fn_, "ptx_coop_acc_len requires an accumulator slot");
        emit_assign(inst, "8", d);
        return;
    }

    if (sym == "ptx_coop_get_acc") {
        const auto& [name, use] = coop_slot(inst.operand(0), "ptx_coop_get_acc");
        if (use != 2) fail_msl(fn_, "ptx_coop_get_acc requires an accumulator slot");
        const Value* idx_v = inst.operand(1);
        if (idx_v && idx_v->is_instruction() && idx_v->defining_instruction()->opcode() == Opcode::iconst_i32) {
            int32_t k = idx_v->defining_instruction()->imm_i32();
            emit_assign(inst, name + "[" + std::to_string((k >> 1) & 3) + "].thread_elements()[" + std::to_string(k & 1) + "]", d);
            return;
        }
        std::string iv = arg(1);
        emit_assign(inst, name + "[((" + iv + ") >> 1) & 3].thread_elements()[(" + iv + ") & 1]", d);
        return;
    }

    if (sym == "ptx_coop_set_acc") {
        const auto& [name, use] = coop_slot(inst.operand(0), "ptx_coop_set_acc");
        if (use != 2) fail_msl(fn_, "ptx_coop_set_acc requires an accumulator slot");
        const Value* idx_v = inst.operand(1);
        std::string val = arg(2);
        indent_line(d);
        if (idx_v && idx_v->is_instruction() && idx_v->defining_instruction()->opcode() == Opcode::iconst_i32) {
            int32_t k = idx_v->defining_instruction()->imm_i32();
            body_ << name << "[" << ((k >> 1) & 3) << "].thread_elements()[" << (k & 1) << "] = " << val << ";\n";
        } else {
            std::string iv = arg(1);
            body_ << name << "[((" << iv << ") >> 1) & 3].thread_elements()[(" << iv << ") & 1] = " << val << ";\n";
        }
        return;
    }

    fail_msl(fn_, "unsupported coop intrinsic '" + std::string(sym) + "'");
}

} // namespace brass::msl
