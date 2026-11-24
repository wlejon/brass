#include "msl_builder.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <set>

namespace brass::msl {

namespace {

std::string hex_u32(uint32_t v) {
    std::ostringstream os;
    os << "0x" << std::hex << v << "u";
    return os.str();
}

std::string f32_literal(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, 4);
    return "as_type<float>(" + hex_u32(bits) + ")";
}

std::string i64_literal(int64_t v) {
    if (v == INT64_MIN) return "(-9223372036854775807LL - 1LL)";
    return std::to_string(v) + "LL";
}

std::string sanitize_ident(std::string_view sv) {
    std::string out;
    out.reserve(sv.size() + 2);
    for (char c : sv) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
            (c >= '0' && c <= '9') || c == '_') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) return "kernel_fn";
    return out;
}

} // namespace

MslEmitter::KernelResult MslFnBuilder::run() {
        if (!fn_.return_type().is_void()) {
            fail_msl(fn_, "compute kernel must return void");
        }
        if (!fn_.entry_block()) {
            fail_msl(fn_, "function has no entry block");
        }

        find_shared_allocs();
        StructuredCfg cfg = brass::spirv::structurize(fn_);
        cfg_ = &cfg;

        std::string entry_ident = sanitize_ident(fn_.name());

        // Classify pointer buffers vs packed scalar params.
        const auto& params = fn_.param_types();
        const auto* entry = fn_.entry_block();
        std::vector<std::string> param_decls;
        std::vector<size_t> scalar_indices;
        uint32_t buf_idx = 0;
        uint32_t scalar_off = 0;
        uint32_t max_align = 1;

        for (size_t i = 0; i < params.size(); ++i) {
            Type pt = params[i];
            if (pt.is_pointer()) {
                std::string pn = "_b" + std::to_string(buf_idx);
                param_decls.push_back("device uint8_t* " + pn + " [[buffer(" + std::to_string(buf_idx) + ")]]");
                if (entry && i < entry->param_count()) {
                    val_names_[entry->param(i)] = pn;
                }
                ++buf_idx;
            } else {
                uint32_t sz = scalar_param_size(pt);
                uint32_t al = sz >= 8 ? 8u : 4u;
                scalar_off = (scalar_off + al - 1u) & ~(al - 1u);
                res_.scalar_param_offsets.push_back(scalar_off);
                scalar_indices.push_back(i);
                max_align = std::max(max_align, al);
                scalar_off += sz;
            }
        }
        res_.num_buffers = buf_idx;

        if (!scalar_indices.empty()) {
            uint32_t aligned_total = (scalar_off + max_align - 1u) & ~(max_align - 1u);
            res_.scalar_params_bytes = aligned_total;
            std::ostringstream ss;
            ss << "struct " << entry_ident << "_Params {\n";
            for (size_t si : scalar_indices) {
                Type pt = params[si];
                ss << "    " << msl_param_type(pt) << " _p" << si << ";\n";
                if (entry && si < entry->param_count()) {
                    val_names_[entry->param(si)] = "_params._p" + std::to_string(si);
                }
            }
            ss << "};\n";
            res_.params_struct_code = ss.str();
            param_decls.push_back("constant " + entry_ident + "_Params& _params [[buffer(" +
                                  std::to_string(buf_idx) + ")]]");
        }

        // Pre-allocate shared arrays, block phi variables, and SSA result variables.
        std::ostringstream decls;
        for (const auto* bb : fn_.blocks()) {
            for (const auto* inst : *bb) {
                if (inst->opcode() == Opcode::call && inst->result() && is_shared_alloc(inst->symbol())) {
                    uint32_t count = const_i32_arg(inst->operand(0), *inst);
                    Type et = shared_elem_type(inst->symbol());
                    std::string sname = "_smem_" + std::to_string(shared_count_++);
                    decls << "    threadgroup " << shared_decl_elem_type(inst->symbol(), *this)
                          << " " << sname << "[" << count << "];\n";
                    val_names_[inst->result()] = "((threadgroup uint8_t*)" + sname + ")";
                    shared_ptrs_.insert(inst->result());
                    res_.uses_shared_memory = true;
                    res_.shared_memory_bytes += count * static_cast<uint32_t>(et.size_in_bytes());
                } else if (inst->opcode() == Opcode::call && inst->result() && is_coop_alloc(inst->symbol())) {
                    int use = coop_use_kind(inst->symbol());
                    std::string cname = "_cm_" + std::to_string(coop_count_++);
                    decls << "    " << (use == 2 ? "simdgroup_float8x8" : "simdgroup_half8x8")
                          << " " << cname << "[4];\n";
                    coop_vars_[inst->result()] = {cname, use};
                    res_.uses_coopmat = true;
                } else if (inst->result() && !inst->type().is_void()) {
                    std::string vn = "v_" + std::to_string(inst->result()->id());
                    val_names_[inst->result()] = vn;
                    decls << "    " << msl_val_type(inst->type(), inst->result()) << " " << vn << " = "
                          << zero_init(inst->type(), inst->result()) << ";\n";
                }
            }
        }

        for (size_t ni = 0; ni < cfg.nodes.size(); ++ni) {
            const CfgNode& node = cfg.nodes[ni];
            if (!node.params) continue;
            for (size_t k = 0; k < node.params->param_count(); ++k) {
                const Value* pv = node.params->param(k);
                std::string pn = phi_name(ni, k);
                decls << "    " << msl_val_type(pv->type(), pv) << " " << pn << " = "
                      << zero_init(pv->type(), pv) << ";\n";
                if (node.kind == CfgNode::Kind::block) {
                    val_names_[pv] = pn;
                }
            }
        }

        // Lower body starting from prologue -> node 0's out[0]
        if (!cfg.nodes[0].out.empty()) {
            emit_edge_args(cfg.nodes[0].out[0], 1);
            emit_node(cfg.edges[cfg.nodes[0].out[0]].to, kNoNode, kNoNode, kNoNode, 1);
        }

        // Assemble complete kernel function after knowing which thread builtins were referenced.
        std::ostringstream fn_ss;
        fn_ss << "kernel void " << entry_ident << "(\n";
        for (size_t i = 0; i < param_decls.size(); ++i) {
            fn_ss << "    " << param_decls[i];
            if (i + 1 < param_decls.size() || need_tid_ || need_ntid_ || need_ctaid_ ||
                need_nctaid_ || need_lane_ || need_sg_) {
                fn_ss << ",";
            }
            fn_ss << "\n";
        }
        std::vector<std::string> builtins;
        if (need_tid_)    builtins.push_back("uint3 _tid [[thread_position_in_threadgroup]]");
        if (need_ntid_)   builtins.push_back("uint3 _ntid [[threads_per_threadgroup]]");
        if (need_ctaid_)  builtins.push_back("uint3 _ctaid [[threadgroup_position_in_grid]]");
        if (need_nctaid_) builtins.push_back("uint3 _nctaid [[threadgroups_per_grid]]");
        if (need_lane_)   builtins.push_back("uint _lane [[thread_index_in_simdgroup]]");
        if (need_sg_)     builtins.push_back("uint _sg [[simdgroup_index_in_threadgroup]]");
        for (size_t i = 0; i < builtins.size(); ++i) {
            fn_ss << "    " << builtins[i] << (i + 1 < builtins.size() ? "," : "") << "\n";
        }
        fn_ss << ") {\n" << decls.str() << body_.str() << "}\n";
        res_.function_code = fn_ss.str();
        return res_;
    }

uint32_t MslFnBuilder::const_i32_arg(const Value* v, const Instruction& inst) const {
    if (!v || !v->is_instruction()) fail_msl(fn_, "shared alloc count must be constant");
    const auto* ci = v->defining_instruction();
    if (ci->opcode() != Opcode::iconst_i32 || ci->imm_i32() <= 0) {
        fail_msl(fn_, "shared alloc count must be positive constant in " + std::string(inst.symbol()));
    }
    return static_cast<uint32_t>(ci->imm_i32());
}

void MslFnBuilder::find_shared_allocs() {
    for (const auto* bb : fn_.blocks()) {
        for (const auto* inst : *bb) {
            if (inst->opcode() == Opcode::call && inst->result() && is_shared_alloc(inst->symbol())) {
                shared_ptrs_.insert(inst->result());
            }
        }
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto* bb : fn_.blocks()) {
            for (const auto* inst : *bb) {
                if (!inst->result() || !inst->type().is_pointer() || shared_ptrs_.count(inst->result())) continue;
                for (size_t i = 0; i < inst->operand_count(); ++i) {
                    if (shared_ptrs_.count(inst->operand(i))) {
                        shared_ptrs_.insert(inst->result());
                        changed = true;
                        break;
                    }
                }
            }
        }
    }
}

std::string MslFnBuilder::msl_type(Type t) const {
        switch (t.kind()) {
            case TypeKind::I8:     return "int8_t";
            case TypeKind::I16:    return "int16_t";
            case TypeKind::I32:    return "int32_t";
            case TypeKind::I64:    return "int64_t";
            case TypeKind::F32:    return "float";
            case TypeKind::F32x4:  return "float4";
            case TypeKind::I32x4:  return "int4";
            case TypeKind::I64x2:  return "long2";
            case TypeKind::F64:
            case TypeKind::F64x2:
            case TypeKind::F32x8:
            case TypeKind::F64x4:
            case TypeKind::I32x8:
            case TypeKind::I64x4:
                fail_msl(fn_, "unsupported type in Metal MSL: " + std::string(t.name()));
            case TypeKind::Ptr:
            case TypeKind::GCRef:
                return "device uint8_t*";
            default:
                return "int32_t";
        }
    }

std::string MslFnBuilder::msl_utype(Type t) const {
    switch (t.kind()) {
        case TypeKind::I8:    return "uint8_t";
        case TypeKind::I16:   return "uint16_t";
        case TypeKind::I32:   return "uint32_t";
        case TypeKind::I64:   return "uint64_t";
        case TypeKind::I32x4: return "uint4";
        case TypeKind::I64x2: return "ulong2";
        default:              return msl_type(t);
    }
}

std::string MslFnBuilder::msl_param_type(Type t) const {
    if (t == Type::i64()) return "int64_t";
    if (t == Type::f32()) return "float";
    return "int32_t";
}

uint32_t MslFnBuilder::scalar_param_size(Type t) const {
    if (t == Type::i64()) return 8;
    if (t == Type::i32() || t == Type::f32() || t == Type::i8() || t == Type::i16()) return 4;
    fail_msl(fn_, "unsupported scalar parameter type: " + std::string(t.name()));
}

std::string MslFnBuilder::msl_val_type(Type t, const Value* v) const {
    if (t.is_pointer()) return shared_ptrs_.count(v) ? "threadgroup uint8_t*" : "device uint8_t*";
    return msl_type(t);
}

std::string MslFnBuilder::zero_init(Type t, const Value* v) const {
    if (t.is_pointer()) return "nullptr";
    switch (t.kind()) {
        case TypeKind::F32:   return "0.0f";
        case TypeKind::F32x4: return "float4(0.0f)";
        case TypeKind::I32x4: return "int4(0)";
        case TypeKind::I64x2: return "long2(0)";
        case TypeKind::I64:   return "0LL";
        default:              return "0";
    }
}

std::string MslFnBuilder::val_expr(const Value* v) const {
    auto it = val_names_.find(v);
    if (it == val_names_.end()) fail_msl(fn_, "use of undefined value %" + std::to_string(v ? v->id() : 0));
    return it->second;
}

void MslFnBuilder::emit_edge_args(size_t edge_idx, int d) {
        const auto& e = cfg_->edges[edge_idx];
        const auto& dst = cfg_->nodes[e.to];
        if (!dst.params || dst.params->param_count() == 0) return;
        const auto& src = cfg_->nodes[e.from];
        std::vector<std::string> rhs;
        if (e.target) {
            for (const Value* v : e.target->args) rhs.push_back(val_expr(v));
        } else if (src.kind == CfgNode::Kind::prologue) {
            const auto* entry = fn_.entry_block();
            for (size_t k = 0; k < entry->param_count(); ++k) rhs.push_back(val_expr(entry->param(k)));
        } else {
            for (size_t k = 0; k < src.params->param_count(); ++k) rhs.push_back(phi_name(e.from, k));
        }
        const size_t n = std::min(rhs.size(), dst.params->param_count());
        if (n <= 1) {
            for (size_t k = 0; k < n; ++k) {
                indent_line(d);
                body_ << phi_name(e.to, k) << " = " << rhs[k] << ";\n";
            }
            return;
        }
        std::vector<std::string> tmps(n);
        for (size_t k = 0; k < n; ++k) {
            const Value* pv = dst.params->param(k);
            tmps[k] = "_ptmp_" + std::to_string(tmp_seq_++);
            indent_line(d);
            body_ << msl_val_type(pv->type(), pv) << " " << tmps[k] << " = " << rhs[k] << ";\n";
        }
        for (size_t k = 0; k < n; ++k) {
            indent_line(d);
            body_ << phi_name(e.to, k) << " = " << tmps[k] << ";\n";
        }
    }

void MslFnBuilder::emit_node(size_t n, size_t stop_node, size_t loop_merge, size_t loop_cont, int d) {
        if (n == kNoNode) return;
        if (n == loop_cont) {
            if (!cfg_->nodes[n].out.empty()) emit_edge_args(cfg_->nodes[n].out[0], d);
            indent_line(d);
            body_ << "continue;\n";
            return;
        }
        if (n == loop_merge) {
            indent_line(d);
            body_ << "break;\n";
            return;
        }
        if (n == stop_node) return;

        const CfgNode& node = cfg_->nodes[n];
        if (node.kind == CfgNode::Kind::unreachable) return;

        if (node.cont != kNoNode) {
            indent_line(d);
            body_ << "while (true) {\n";
            if (!node.out.empty()) {
                emit_edge_args(node.out[0], d + 1);
                emit_node(cfg_->edges[node.out[0]].to, node.merge, node.merge, node.cont, d + 1);
            }
            indent_line(d);
            body_ << "}\n";
            if (node.merge != kNoNode && node.merge != stop_node) {
                emit_node(node.merge, stop_node, loop_merge, loop_cont, d);
            }
            return;
        }

        if (node.kind == CfgNode::Kind::forward) {
            if (!node.out.empty()) {
                emit_edge_args(node.out[0], d);
                emit_node(cfg_->edges[node.out[0]].to, stop_node, loop_merge, loop_cont, d);
            }
            return;
        }

        if (node.kind == CfgNode::Kind::block && node.block) {
            const Instruction* term = node.block->terminator();
            for (const Instruction* inst : *node.block) {
                if (inst == term) continue;
                lower_inst(*inst, d);
            }
            if (!term) return;
            switch (term->opcode()) {
                case Opcode::ret:
                    indent_line(d);
                    body_ << "return;\n";
                    return;
                case Opcode::unreachable:
                    return;
                case Opcode::br:
                    emit_edge_args(node.out.at(0), d);
                    emit_node(cfg_->edges[node.out.at(0)].to, stop_node, loop_merge, loop_cont, d);
                    return;
                case Opcode::br_if: {
                    size_t te = node.out.at(0), fe = node.out.at(1);
                    indent_line(d);
                    body_ << "if (" << val_expr(term->operand(0)) << " != 0) {\n";
                    emit_edge_args(te, d + 1);
                    emit_node(cfg_->edges[te].to, node.merge, loop_merge, loop_cont, d + 1);
                    indent_line(d);
                    body_ << "} else {\n";
                    emit_edge_args(fe, d + 1);
                    emit_node(cfg_->edges[fe].to, node.merge, loop_merge, loop_cont, d + 1);
                    indent_line(d);
                    body_ << "}\n";
                    if (node.merge != kNoNode && node.merge != stop_node &&
                        node.merge != loop_merge && node.merge != loop_cont) {
                        emit_node(node.merge, stop_node, loop_merge, loop_cont, d);
                    }
                    return;
                }
                case Opcode::switch_: {
                    const auto& cases = term->switch_cases();
                    indent_line(d);
                    body_ << "switch (" << val_expr(term->operand(0)) << ") {\n";
                    for (size_t ci = 0; ci < cases.size(); ++ci) {
                        indent_line(d + 1);
                        body_ << "case " << cases[ci].value << ": {\n";
                        emit_edge_args(node.out.at(ci), d + 2);
                        emit_node(cfg_->edges[node.out.at(ci)].to, node.merge, loop_merge, loop_cont, d + 2);
                        indent_line(d + 2);
                        body_ << "break;\n";
                        indent_line(d + 1);
                        body_ << "}\n";
                    }
                    size_t de = node.out.back();
                    indent_line(d + 1);
                    body_ << "default: {\n";
                    emit_edge_args(de, d + 2);
                    emit_node(cfg_->edges[de].to, node.merge, loop_merge, loop_cont, d + 2);
                    indent_line(d + 2);
                    body_ << "break;\n";
                    indent_line(d + 1);
                    body_ << "}\n";
                    indent_line(d);
                    body_ << "}\n";
                    if (node.merge != kNoNode && node.merge != stop_node &&
                        node.merge != loop_merge && node.merge != loop_cont) {
                        emit_node(node.merge, stop_node, loop_merge, loop_cont, d);
                    }
                    return;
                }
                default:
                    fail_msl(fn_, "unsupported terminator " + std::string(opcode_name(term->opcode())));
            }
        }
    }

void MslFnBuilder::emit_assign(const Instruction& inst, const std::string& expr, int d) {
    if (!inst.result() || inst.type().is_void()) {
        indent_line(d);
        body_ << expr << ";\n";
        return;
    }
    indent_line(d);
    body_ << val_expr(inst.result()) << " = " << expr << ";\n";
}

void MslFnBuilder::lower_inst(const Instruction& inst, int d) {
        const Opcode op = inst.opcode();
        auto arg = [&](size_t i) { return val_expr(inst.operand(i)); };
        auto t = inst.type();

        switch (op) {
            case Opcode::iconst_i32: emit_assign(inst, std::to_string(inst.imm_i32()), d); return;
            case Opcode::iconst_i64: emit_assign(inst, i64_literal(inst.imm_i64()), d); return;
            case Opcode::fconst_f64:
                if (t.kind() == TypeKind::F32) {
                    emit_assign(inst, f32_literal(static_cast<float>(inst.imm_f64())), d);
                    return;
                }
                fail_msl(fn_, "unsupported f64 constant in Metal MSL");
            case Opcode::sext_i64:  emit_assign(inst, "static_cast<int64_t>(" + arg(0) + ")", d); return;
            case Opcode::zext_i64:
                emit_assign(inst, "static_cast<int64_t>(static_cast<" + msl_utype(inst.operand(0)->type()) + ">(" + arg(0) + "))", d);
                return;
            case Opcode::trunc_i32: emit_assign(inst, "static_cast<int32_t>(" + arg(0) + ")", d); return;
            case Opcode::trunc_i8:  emit_assign(inst, "static_cast<int8_t>(" + arg(0) + ")", d); return;
            case Opcode::fptosi_i32:
            case Opcode::fptosi_i32_f32: emit_assign(inst, "static_cast<int32_t>(" + arg(0) + ")", d); return;
            case Opcode::fptosi_i64:
            case Opcode::fptosi_i64_f32: emit_assign(inst, "static_cast<int64_t>(" + arg(0) + ")", d); return;
            case Opcode::sitofp_f32_i32:
            case Opcode::sitofp_f32_i64: emit_assign(inst, "static_cast<float>(" + arg(0) + ")", d); return;
            case Opcode::bitcast_i32_f32: emit_assign(inst, "as_type<int32_t>(" + arg(0) + ")", d); return;
            case Opcode::bitcast_f32_i32: emit_assign(inst, "as_type<float>(" + arg(0) + ")", d); return;

            case Opcode::add:
                if (t.is_pointer()) {
                    std::string idx = inst.operand(1)->type().is_pointer() ? arg(0) : arg(1);
                    std::string ptr = inst.operand(0)->type().is_pointer() ? arg(0) : arg(1);
                    emit_assign(inst, "(" + ptr + " + static_cast<size_t>(" + idx + "))", d);
                    return;
                }
                if (t.is_integer()) {
                    std::string ut = msl_utype(t);
                    emit_assign(inst, "as_type<" + msl_type(t) + ">(as_type<" + ut + ">(" + arg(0) + ") + as_type<" + ut + ">(" + arg(1) + "))", d);
                } else {
                    emit_assign(inst, "(" + arg(0) + " + " + arg(1) + ")", d);
                }
                return;
            case Opcode::sub:
                if (t.is_pointer()) {
                    emit_assign(inst, "(" + arg(0) + " - static_cast<size_t>(" + arg(1) + "))", d);
                } else if (t.is_integer()) {
                    std::string ut = msl_utype(t);
                    emit_assign(inst, "as_type<" + msl_type(t) + ">(as_type<" + ut + ">(" + arg(0) + ") - as_type<" + ut + ">(" + arg(1) + "))", d);
                } else {
                    emit_assign(inst, "(" + arg(0) + " - " + arg(1) + ")", d);
                }
                return;
            case Opcode::mul:
                if (t.is_integer()) {
                    std::string ut = msl_utype(t);
                    emit_assign(inst, "as_type<" + msl_type(t) + ">(as_type<" + ut + ">(" + arg(0) + ") * as_type<" + ut + ">(" + arg(1) + "))", d);
                } else {
                    emit_assign(inst, "(" + arg(0) + " * " + arg(1) + ")", d);
                }
                return;
            case Opcode::sdiv: emit_assign(inst, "(" + arg(0) + " / " + arg(1) + ")", d); return;
            case Opcode::udiv: {
                std::string ut = msl_utype(t);
                emit_assign(inst, "as_type<" + msl_type(t) + ">(as_type<" + ut + ">(" + arg(0) + ") / as_type<" + ut + ">(" + arg(1) + "))", d);
                return;
            }
            case Opcode::smod: emit_assign(inst, "(" + arg(0) + " % " + arg(1) + ")", d); return;
            case Opcode::umod: {
                std::string ut = msl_utype(t);
                emit_assign(inst, "as_type<" + msl_type(t) + ">(as_type<" + ut + ">(" + arg(0) + ") % as_type<" + ut + ">(" + arg(1) + "))", d);
                return;
            }
            case Opcode::neg:      emit_assign(inst, "(-" + arg(0) + ")", d); return;
            case Opcode::fma_f32:  emit_assign(inst, "fma(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")", d); return;
            case Opcode::fabs_f32: emit_assign(inst, "fabs(" + arg(0) + ")", d); return;
            case Opcode::sqrt_f32: emit_assign(inst, "sqrt(" + arg(0) + ")", d); return;
            case Opcode::floor_f32: emit_assign(inst, "floor(" + arg(0) + ")", d); return;
            case Opcode::ceil_f32:  emit_assign(inst, "ceil(" + arg(0) + ")", d); return;
            case Opcode::round_f32: emit_assign(inst, "rint(" + arg(0) + ")", d); return;
            case Opcode::fmin_f32:  emit_assign(inst, "fmin(" + arg(0) + ", " + arg(1) + ")", d); return;
            case Opcode::fmax_f32:  emit_assign(inst, "fmax(" + arg(0) + ", " + arg(1) + ")", d); return;

            case Opcode::and_: emit_assign(inst, "(" + arg(0) + " & " + arg(1) + ")", d); return;
            case Opcode::or_:  emit_assign(inst, "(" + arg(0) + " | " + arg(1) + ")", d); return;
            case Opcode::xor_: emit_assign(inst, "(" + arg(0) + " ^ " + arg(1) + ")", d); return;
            case Opcode::not_: emit_assign(inst, "(~" + arg(0) + ")", d); return;
            case Opcode::shl: {
                std::string ut = msl_utype(t);
                emit_assign(inst, "as_type<" + msl_type(t) + ">(as_type<" + ut + ">(" + arg(0) + ") << as_type<" + ut + ">(" + arg(1) + "))", d);
                return;
            }
            case Opcode::lshr: {
                std::string ut = msl_utype(t);
                emit_assign(inst, "as_type<" + msl_type(t) + ">(as_type<" + ut + ">(" + arg(0) + ") >> as_type<" + ut + ">(" + arg(1) + "))", d);
                return;
            }
            case Opcode::ashr: emit_assign(inst, "(" + arg(0) + " >> " + arg(1) + ")", d); return;
            case Opcode::clz:  emit_assign(inst, "static_cast<" + msl_type(t) + ">(clz(as_type<" + msl_utype(t) + ">(" + arg(0) + ")))", d); return;
            case Opcode::ctz:  emit_assign(inst, "static_cast<" + msl_type(t) + ">(ctz(as_type<" + msl_utype(t) + ">(" + arg(0) + ")))", d); return;
            case Opcode::popcnt: emit_assign(inst, "static_cast<" + msl_type(t) + ">(popcount(as_type<" + msl_utype(t) + ">(" + arg(0) + ")))", d); return;

            case Opcode::eq: case Opcode::ne:
            case Opcode::slt: case Opcode::sle: case Opcode::sgt: case Opcode::sge:
            case Opcode::ult: case Opcode::ule: case Opcode::ugt: case Opcode::uge: {
                bool is_u = (op == Opcode::ult || op == Opcode::ule || op == Opcode::ugt || op == Opcode::uge);
                Type ot = inst.operand(0)->type();
                std::string ut = is_u ? msl_utype(ot) : msl_type(ot);
                std::string a = is_u ? ("as_type<" + ut + ">(" + arg(0) + ")") : arg(0);
                std::string b = is_u ? ("as_type<" + ut + ">(" + arg(1) + ")") : arg(1);
                const char* cmp = (op == Opcode::eq) ? " == " : (op == Opcode::ne) ? " != "
                                : (op == Opcode::slt || op == Opcode::ult) ? " < "
                                : (op == Opcode::sle || op == Opcode::ule) ? " <= "
                                : (op == Opcode::sgt || op == Opcode::ugt) ? " > " : " >= ";
                emit_assign(inst, "((" + a + cmp + b + ") ? 1 : 0)", d);
                return;
            }
            case Opcode::select:
                emit_assign(inst, "((" + arg(0) + " != 0) ? " + arg(1) + " : " + arg(2) + ")", d);
                return;

            case Opcode::load:
            case Opcode::vload: {
                std::string sp = ptr_space(inst.operand(0));
                std::string addr = arg(0);
                if (inst.offset() != 0) addr += " + " + std::to_string(inst.offset());
                emit_assign(inst, "(*reinterpret_cast<const " + sp + " " + msl_type(t) + "*>(" + addr + "))", d);
                return;
            }
            case Opcode::store:
            case Opcode::vstore: {
                std::string sp = ptr_space(inst.operand(0));
                Type vt = inst.operand(1)->type();
                std::string addr = arg(0);
                if (inst.offset() != 0) addr += " + " + std::to_string(inst.offset());
                indent_line(d);
                body_ << "*reinterpret_cast<" << sp << " " << msl_type(vt) << "*>(" << addr << ") = " << arg(1) << ";\n";
                return;
            }
            case Opcode::load_indexed: {
                std::string sp = ptr_space(inst.operand(0));
                std::string off = "static_cast<int64_t>(" + arg(1) + ") * " + std::to_string(inst.scale()) + "LL";
                if (inst.offset() != 0) off += " + " + std::to_string(inst.offset()) + "LL";
                emit_assign(inst, "(*reinterpret_cast<const " + sp + " " + msl_type(t) + "*>(" +
                            arg(0) + " + static_cast<size_t>(" + off + ")))", d);
                return;
            }
            case Opcode::store_indexed: {
                std::string sp = ptr_space(inst.operand(0));
                Type vt = inst.operand(2)->type();
                std::string off = "static_cast<int64_t>(" + arg(1) + ") * " + std::to_string(inst.scale()) + "LL";
                if (inst.offset() != 0) off += " + " + std::to_string(inst.offset()) + "LL";
                indent_line(d);
                body_ << "*reinterpret_cast<" << sp << " " << msl_type(vt) << "*>(" <<
                         arg(0) + " + static_cast<size_t>(" + off + ")) = " << arg(2) << ";\n";
                return;
            }

            case Opcode::vzero: emit_assign(inst, zero_init(t, inst.result()), d); return;
            case Opcode::vbroadcast: emit_assign(inst, msl_type(t) + "(" + arg(0) + ")", d); return;
            case Opcode::vextract_lane:
                emit_assign(inst, "(" + arg(0) + ")[" + std::to_string(inst.lane()) + "]", d);
                return;
            case Opcode::vinsert_lane: {
                std::string vn = val_expr(inst.result());
                indent_line(d);
                body_ << vn << " = " << arg(0) << "; " << vn << "[" << inst.lane() << "] = " << arg(1) << ";\n";
                return;
            }
            case Opcode::vadd:  emit_assign(inst, "(" + arg(0) + " + " + arg(1) + ")", d); return;
            case Opcode::vsub:  emit_assign(inst, "(" + arg(0) + " - " + arg(1) + ")", d); return;
            case Opcode::vmul:  emit_assign(inst, "(" + arg(0) + " * " + arg(1) + ")", d); return;
            case Opcode::vdiv:  emit_assign(inst, "(" + arg(0) + " / " + arg(1) + ")", d); return;
            case Opcode::vfma:  emit_assign(inst, "fma(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")", d); return;
            case Opcode::vneg:  emit_assign(inst, "(-" + arg(0) + ")", d); return;
            case Opcode::vsqrt: emit_assign(inst, "sqrt(" + arg(0) + ")", d); return;
            case Opcode::vmin:  emit_assign(inst, "min(" + arg(0) + ", " + arg(1) + ")", d); return;
            case Opcode::vmax:  emit_assign(inst, "max(" + arg(0) + ", " + arg(1) + ")", d); return;
            case Opcode::vfloor: emit_assign(inst, "floor(" + arg(0) + ")", d); return;
            case Opcode::vceil:  emit_assign(inst, "ceil(" + arg(0) + ")", d); return;
            case Opcode::vtrunc: emit_assign(inst, "trunc(" + arg(0) + ")", d); return;
            case Opcode::vround: emit_assign(inst, "rint(" + arg(0) + ")", d); return;
            case Opcode::vand:  emit_assign(inst, "(" + arg(0) + " & " + arg(1) + ")", d); return;
            case Opcode::vor:   emit_assign(inst, "(" + arg(0) + " | " + arg(1) + ")", d); return;
            case Opcode::vxor:  emit_assign(inst, "(" + arg(0) + " ^ " + arg(1) + ")", d); return;
            case Opcode::vnot:  emit_assign(inst, "(~" + arg(0) + ")", d); return;

        case Opcode::call:
            lower_call(inst, d);
            return;

        default:
            fail_msl(fn_, "unsupported opcode '" + std::string(opcode_name(op)) + "'");
    }
}

MslEmitter::KernelResult MslEmitter::lower(const Function& fn, const target::MslOptions& opts) {
    MslFnBuilder builder(fn, opts);
    return builder.run();
}

} // namespace brass::msl
