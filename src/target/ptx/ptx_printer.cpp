#include <brass/target/ptx/ptx_printer.hpp>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace brass::ptx {

// ---------------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------------

std::string format_f32_hex(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    std::ostringstream ss;
    ss << "0f" << std::uppercase << std::hex << std::setfill('0') << std::setw(8) << u;
    return ss.str();
}

std::string format_f64_hex(double d) {
    uint64_t u = 0;
    std::memcpy(&u, &d, sizeof(u));
    std::ostringstream ss;
    ss << "0d" << std::uppercase << std::hex << std::setfill('0') << std::setw(16) << u;
    return ss.str();
}

// ---------------------------------------------------------------------------
// Operands
// ---------------------------------------------------------------------------

namespace {

// ptxas accepts "[%rd0 + -2]" but not "[%rd0 - 2]".
void print_disp(std::ostringstream& ss, int32_t disp) {
    if (disp != 0) ss << " + " << disp;
}

} // namespace

std::string to_string(const Operand& op) {
    std::ostringstream ss;
    switch (op.kind) {
        case OperandKind::None:
            ss << "<none>";
            break;
        case OperandKind::Reg:
            ss << to_string(op.reg_val);
            break;
        case OperandKind::ImmInt:
            ss << op.imm_int;
            break;
        case OperandKind::ImmFloat:
            ss << (op.imm_is_f32 ? format_f32_hex(static_cast<float>(op.imm_float))
                                 : format_f64_hex(op.imm_float));
            break;
        case OperandKind::Addr:
            ss << "[";
            if (op.addr_base.valid()) ss << to_string(op.addr_base);
            else ss << op.addr_symbol;
            print_disp(ss, op.disp);
            ss << "]";
            break;
        case OperandKind::Vector:
            ss << "{";
            for (size_t i = 0; i < op.elems.size(); ++i) {
                if (i) ss << ", ";
                ss << to_string(op.elems[i]);
            }
            ss << "}";
            break;
        case OperandKind::Label:
        case OperandKind::Symbol:
            ss << op.name;
            break;
        case OperandKind::Param:
            ss << "[" << op.name << "]";
            break;
        case OperandKind::Special:
            ss << to_string(op.special_reg);
            break;
    }
    return ss.str();
}

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

std::string mnemonic(const Inst& inst) {
    std::string s(to_string(inst.op));
    auto suffix = [&](std::string_view part) {
        if (!part.empty()) { s += '.'; s += part; }
    };

    if (inst.op == Opcode::wmma_load || inst.op == Opcode::wmma_store) {
        suffix(to_string(inst.wmma_matrix));
        suffix("sync");
        suffix("aligned");
        suffix(to_string(inst.wmma_layout));
        suffix("m16n16k16");
        suffix(to_string(inst.state_space));
        suffix(to_string(inst.type));
        return s;
    }
    if (inst.op == Opcode::wmma_mma) {
        suffix("sync");
        suffix("aligned");
        suffix(to_string(inst.wmma_layout));
        suffix(to_string(inst.wmma_layout_b));
        suffix("m16n16k16");
        suffix(to_string(inst.type));
        suffix(to_string(inst.wmma_c_type));
        return s;
    }
    if (inst.op == Opcode::cvta) {
        if (inst.is_to_space) suffix("to");
        suffix(to_string(inst.state_space));
        suffix(to_string(inst.type));
        return s;
    }
    if (inst.op == Opcode::cp_async) {
        suffix(inst.is_cg ? "cg" : "ca");
        suffix("shared");
        suffix("global");
        if (inst.has_cache_hint) suffix("L2::cache_hint");
        return s;
    }
    if (inst.op == Opcode::cp_async_commit ||
        inst.op == Opcode::cp_async_wait_group ||
        inst.op == Opcode::cp_async_wait_all) {
        return s;
    }
    if (inst.op == Opcode::ldmatrix) {
        suffix("sync");
        suffix("aligned");
        suffix("m8n8");
        suffix(inst.ldmatrix_num == 4 ? "x4" : (inst.ldmatrix_num == 2 ? "x2" : "x1"));
        if (inst.is_trans) suffix("trans");
        suffix("shared");
        suffix("b16");
        return s;
    }
    if (inst.op == Opcode::mma_sync) {
        suffix("aligned");
        suffix("m16n8k16");
        suffix("row");
        suffix("col");
        suffix(to_string(inst.type));
        suffix(to_string(inst.mma_ab_type));
        suffix(to_string(inst.mma_ab_type));
        suffix(to_string(inst.wmma_c_type));
        return s;
    }

    switch (inst.op) {
        case Opcode::shfl:
            if (inst.is_sync) suffix("sync");
            suffix(to_string(inst.shfl_mode));
            break;
        case Opcode::bar:
            if (inst.is_warp) suffix("warp");
            if (inst.is_sync) suffix("sync");
            break;
        case Opcode::bra:
        case Opcode::ret:
            if (inst.is_uni) suffix("uni");
            break;
        case Opcode::setp:
            suffix(to_string(inst.cmp_op));
            break;
        case Opcode::vote:
            if (inst.is_sync) suffix("sync");
            suffix("ballot");
            break;
        default:
            break;
    }

    // atom{.space}.op.type ; everything else: {.space}{.nc}{.rnd|.approx}{.ftz}{.sat}{.lo|.hi|.wide}{.vN}.type{.atype}
    suffix(to_string(inst.state_space));
    if (inst.op == Opcode::ld && inst.is_nc) suffix("nc");
    if (inst.op == Opcode::atom) suffix(to_string(inst.atom_op));
    suffix(to_string(inst.rounding));
    if (inst.is_approx) suffix("approx");
    if (inst.is_ftz) suffix("ftz");
    if (inst.is_sat) suffix("sat");
    suffix(to_string(inst.mul_mode));
    suffix(to_string(inst.vec_width));
    suffix(to_string(inst.type));
    suffix(to_string(inst.src_type));
    return s;
}

std::string to_string(const Inst& inst) {
    std::ostringstream ss;
    if (inst.has_guard) {
        ss << "@" << (inst.guard_negated ? "!" : "") << to_string(inst.guard_reg) << " ";
    }
    if (inst.op == Opcode::createpolicy_evict_first) {
        ss << "createpolicy.fractional.L2::evict_first.b64 " << to_string(inst.dsts[0]) << ", 1.0;";
        return ss.str();
    }
    ss << mnemonic(inst);

    if (inst.op == Opcode::call) {
        // call (dst), callee, (args);  or  call callee, (args);
        ss << " ";
        if (!inst.dsts.empty()) {
            ss << "(";
            for (size_t i = 0; i < inst.dsts.size(); ++i) {
                if (i) ss << ", ";
                ss << to_string(inst.dsts[i]);
            }
            ss << "), ";
        }
        if (!inst.srcs.empty()) ss << to_string(inst.srcs[0]);
        ss << ", (";
        for (size_t i = 1; i < inst.srcs.size(); ++i) {
            if (i > 1) ss << ", ";
            ss << to_string(inst.srcs[i]);
        }
        ss << ");";
        return ss.str();
    }

    bool first = true;
    auto emit = [&](const Operand& o) {
        ss << (first ? " " : ", ") << to_string(o);
        first = false;
    };
    for (const auto& d : inst.dsts) emit(d);
    for (const auto& s : inst.srcs) emit(s);
    ss << ";";
    return ss.str();
}

// ---------------------------------------------------------------------------
// Header, signature, declarations, blocks
// ---------------------------------------------------------------------------

std::string print_header(const target::PtxOptions& opts) {
    uint32_t major = opts.ptx_version_major;
    uint32_t minor = opts.ptx_version_minor;
    const int sm = (opts.sm_arch.rfind("sm_", 0) == 0) ? std::atoi(opts.sm_arch.c_str() + 3) : 0;
    // sm_80+ instructions (cp.async, ldmatrix, bf16 mma/cvt, createpolicy) require PTX 7.4+;
    // sm_89 / sm_90 require PTX 7.8+.
    if (sm >= 89 && (major < 7 || (major == 7 && minor < 8))) {
        major = 7;
        minor = 8;
    } else if (sm >= 80 && (major < 7 || (major == 7 && minor < 4))) {
        major = 7;
        minor = 4;
    }
    std::ostringstream ss;
    ss << "// Generated by Brass PTX Target Emitter\n"
       << ".version " << major << "." << minor << "\n"
       << ".target " << opts.sm_arch << "\n"
       << ".address_size 64\n\n";
    return ss.str();
}

PtxIsaRequirement required_isa(const Function& fn) noexcept {
    PtxIsaRequirement req;
    auto bump = [&](uint32_t sm, uint32_t ptx_maj, uint32_t ptx_min) {
        if (sm > req.min_sm) req.min_sm = sm;
        if (ptx_maj > req.min_ptx_major ||
            (ptx_maj == req.min_ptx_major && ptx_min > req.min_ptx_minor)) {
            req.min_ptx_major = ptx_maj;
            req.min_ptx_minor = ptx_min;
        }
    };
    for (const auto& block : fn.blocks) {
        for (const auto& inst : block->insts) {
            if (inst.op == Opcode::createpolicy_evict_first ||
                (inst.op == Opcode::cp_async && inst.has_cache_hint)) {
                bump(80, 7, 4);
            } else if (inst.op == Opcode::cp_async ||
                       inst.op == Opcode::cp_async_commit ||
                       inst.op == Opcode::cp_async_wait_group ||
                       inst.op == Opcode::cp_async_wait_all ||
                       inst.op == Opcode::ldmatrix ||
                       inst.op == Opcode::mma_sync ||
                       (inst.op == Opcode::cvt && (inst.type == Type::bf16 || inst.src_type == Type::bf16))) {
                bump(80, 7, 0);
            }
        }
    }
    return req;
}

namespace {

int parse_sm_version(const std::string& sm_arch) {
    if (sm_arch.rfind("sm_", 0) != 0) return 999;
    return std::atoi(sm_arch.c_str() + 3);
}

// tanh.approx.f32 is sm_75 and later.
bool has_native_tanh(const std::string& sm_arch) {
    return parse_sm_version(sm_arch) >= 75;
}

// Native bf16 converts (cvt.rn.bf16.f32, cvt.f32.bf16) require sm_80+.
bool has_native_bf16(const std::string& sm_arch) {
    return parse_sm_version(sm_arch) >= 80;
}

void print_native_f32_to_bf16(std::ostringstream& ss, const Inst& inst) {
    std::string guard;
    if (inst.has_guard) guard = "@" + std::string(inst.guard_negated ? "!" : "") + to_string(inst.guard_reg) + " ";
    const std::string dst = to_string(inst.dsts[0]);
    const std::string src = to_string(inst.srcs[0]);
    ss << "    {\n"
       << "    .reg .b16 %bf_tmp;\n"
       << "    " << guard << "cvt.rn.bf16.f32 %bf_tmp, " << src << ";\n"
       << "    " << guard << "cvt.u32.u16 " << dst << ", %bf_tmp;\n"
       << "    }\n";
}

void print_f32_to_bf16_expansion(std::ostringstream& ss, const Inst& inst) {
    std::string guard;
    if (inst.has_guard) guard = "@" + std::string(inst.guard_negated ? "!" : "") + to_string(inst.guard_reg) + " ";
    const std::string dst = to_string(inst.dsts[0]);
    const std::string src = to_string(inst.srcs[0]);
    ss << "    {\n"
       << "    .reg .b32 %bf_r<10>;\n"
       << "    .reg .pred %bf_p<2>;\n"
       << "    " << guard << "mov.b32 %bf_r0, " << src << ";\n"
       << "    " << guard << "shr.u32 %bf_r1, %bf_r0, 16;\n"
       << "    " << guard << "and.b32 %bf_r2, %bf_r1, 1;\n"
       << "    " << guard << "add.u32 %bf_r3, %bf_r0, 32767;\n"
       << "    " << guard << "add.u32 %bf_r4, %bf_r3, %bf_r2;\n"
       << "    " << guard << "shr.u32 %bf_r5, %bf_r4, 16;\n"
       << "    " << guard << "and.b32 %bf_r6, %bf_r0, 2139095040;\n"
       << "    " << guard << "and.b32 %bf_r7, %bf_r0, 8388607;\n"
       << "    " << guard << "setp.eq.u32 %bf_p0, %bf_r6, 2139095040;\n"
       << "    " << guard << "setp.ne.u32 %bf_p1, %bf_r7, 0;\n"
       << "    " << guard << "or.b32 %bf_r8, %bf_r1, 64;\n"
       << "    " << guard << "selp.b32 %bf_r9, %bf_r8, %bf_r5, %bf_p1;\n"
       << "    " << guard << "selp.b32 " << dst << ", %bf_r9, %bf_r5, %bf_p0;\n"
       << "    }\n";
}

void print_bf16_to_f32_expansion(std::ostringstream& ss, const Inst& inst) {
    std::string guard;
    if (inst.has_guard) guard = "@" + std::string(inst.guard_negated ? "!" : "") + to_string(inst.guard_reg) + " ";
    const std::string dst = to_string(inst.dsts[0]);
    const std::string src = to_string(inst.srcs[0]);
    ss << "    {\n"
       << "    .reg .b32 %bf_s;\n"
       << "    " << guard << "shl.b32 %bf_s, " << src << ", 16;\n"
       << "    " << guard << "mov.b32 " << dst << ", %bf_s;\n"
       << "    }\n";
}

// tanh(x) on a target without tanh.approx: 1 - 2 / (2^(2x log2 e) + 1), from
// ex2.approx and rcp.approx (sm_20 and later), in a block of its own so the
// temporaries need no declaration in the function. It saturates as tanh does:
// a large x gives 2^inf = inf, 1 - 2/inf = 1; a large -x gives 1 - 2/1 = -1.
void print_tanh_expansion(std::ostringstream& ss, const Inst& inst) {
    std::string guard;
    if (inst.has_guard) guard = "@" + std::string(inst.guard_negated ? "!" : "") + to_string(inst.guard_reg) + " ";
    const std::string ftz = inst.is_ftz ? ".ftz" : "";
    const std::string dst = to_string(inst.dsts[0]);
    const std::string src = to_string(inst.srcs[0]);
    ss << "    {\n"
       << "    .reg .f32 %tanh_t<2>;\n"
       << "    " << guard << "mul" << ftz << ".f32 %tanh_t0, " << src << ", 0f4038AA3B;\n"   // 2 log2(e)
       << "    " << guard << "ex2.approx" << ftz << ".f32 %tanh_t0, %tanh_t0;\n"
       << "    " << guard << "add" << ftz << ".f32 %tanh_t0, %tanh_t0, 0f3F800000;\n"
       << "    " << guard << "rcp.approx" << ftz << ".f32 %tanh_t1, %tanh_t0;\n"
       << "    " << guard << "fma.rn" << ftz << ".f32 " << dst << ", %tanh_t1, 0fC0000000, 0f3F800000;\n"
       << "    }\n";
}

} // namespace

std::string print_body(const Function& fn) {
    return print_body(fn, true, true);
}

std::string print_body(const Function& fn, bool native_tanh) {
    return print_body(fn, native_tanh, false);
}

std::string print_body(const Function& fn, bool native_tanh, bool native_bf16) {
    std::ostringstream ss;

    ss << ".visible " << (fn.is_entry ? ".entry " : ".func ") << fn.name << "(";
    if (!fn.params.empty()) ss << "\n";
    for (size_t i = 0; i < fn.params.size(); ++i) {
        ss << "    .param ." << to_string(fn.params[i].type) << " " << fn.params[i].name;
        ss << (i + 1 < fn.params.size() ? ",\n" : "\n");
    }
    ss << ")\n";
    if (fn.is_entry) {
        if (fn.maxntid[0] > 0) {
            const uint32_t ny = fn.maxntid[1] > 0 ? fn.maxntid[1] : 1;
            const uint32_t nz = fn.maxntid[2] > 0 ? fn.maxntid[2] : 1;
            ss << ".maxntid " << fn.maxntid[0] << ", " << ny << ", " << nz << "\n";
        }
        if (fn.minnctapersm > 0) {
            ss << ".minnctapersm " << fn.minnctapersm << "\n";
        }
        if (fn.maxnreg > 0) {
            ss << ".maxnreg " << fn.maxnreg << "\n";
        }
    }
    ss << "{\n";

    for (size_t c = 0; c < kRegClassCount; ++c) {
        uint32_t n = fn.reg_counts[c];
        if (n == 0) continue;
        auto rc = static_cast<RegClass>(c);
        ss << "    .reg " << reg_decl_type(rc) << " " << reg_prefix(rc) << "<" << n << ">;\n";
    }
    for (const auto& sh : fn.shared) {
        ss << "    .shared .align " << sh.align << " ." << to_string(sh.type) << " "
           << sh.name << "[" << sh.count << "];\n";
    }
    ss << "\n";

    for (const auto& block : fn.blocks) {
        ss << block->label << ":\n";
        for (const auto& inst : block->insts) {
            if (inst.op == Opcode::tanh && !native_tanh) {
                print_tanh_expansion(ss, inst);
                continue;
            }
            if (inst.op == Opcode::cvt) {
                if (inst.type == Type::bf16 && inst.src_type == Type::f32) {
                    if (native_bf16) print_native_f32_to_bf16(ss, inst);
                    else print_f32_to_bf16_expansion(ss, inst);
                    continue;
                }
                if (inst.type == Type::f32 && inst.src_type == Type::bf16 && !native_bf16) {
                    print_bf16_to_f32_expansion(ss, inst);
                    continue;
                }
            }
            ss << "    " << to_string(inst) << "\n";
        }
    }

    ss << "}\n";
    return ss.str();
}

std::string print(const Function& fn, const target::PtxOptions& opts) {
    std::string out = print_header(opts);
    if (fn.has_dynamic_shared) out += ".extern .shared .align 128 .b8 smem_dyn[];\n\n";
    return out + print_body(fn, has_native_tanh(opts.sm_arch), has_native_bf16(opts.sm_arch));
}

std::string print_module(const std::vector<const Function*>& fns, const target::PtxOptions& opts) {
    std::string out = print_header(opts);
    bool any_dyn = false;
    for (const Function* f : fns) {
        if (f && f->has_dynamic_shared) { any_dyn = true; break; }
    }
    if (any_dyn) out += ".extern .shared .align 128 .b8 smem_dyn[];\n\n";
    const bool nt = has_native_tanh(opts.sm_arch);
    const bool nb = has_native_bf16(opts.sm_arch);
    for (size_t i = 0; i < fns.size(); ++i) {
        if (i) out += "\n";
        out += print_body(*fns[i], nt, nb);
    }
    return out;
}

} // namespace brass::ptx
