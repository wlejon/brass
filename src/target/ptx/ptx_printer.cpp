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

    // atom{.space}.op.type ; everything else: {.space}{.rnd|.approx}{.ftz}{.sat}{.lo|.hi|.wide}{.vN}.type{.atype}
    suffix(to_string(inst.state_space));
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
    // sm_89 / sm_90 were introduced in PTX ISA 7.8; older headers are rejected by ptxas.
    if ((opts.sm_arch == "sm_89" || opts.sm_arch == "sm_90") &&
        (major < 7 || (major == 7 && minor < 8))) {
        major = 7;
        minor = 8;
    }
    std::ostringstream ss;
    ss << "// Generated by Brass PTX Target Emitter\n"
       << ".version " << major << "." << minor << "\n"
       << ".target " << opts.sm_arch << "\n"
       << ".address_size 64\n\n";
    return ss.str();
}

namespace {

// tanh.approx.f32 is sm_75 and later.
bool has_native_tanh(const std::string& sm_arch) {
    if (sm_arch.rfind("sm_", 0) != 0) return true;
    return std::atoi(sm_arch.c_str() + 3) >= 75;
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
    return print_body(fn, true);
}

std::string print_body(const Function& fn, bool native_tanh) {
    std::ostringstream ss;

    ss << ".visible " << (fn.is_entry ? ".entry " : ".func ") << fn.name << "(";
    if (!fn.params.empty()) ss << "\n";
    for (size_t i = 0; i < fn.params.size(); ++i) {
        ss << "    .param ." << to_string(fn.params[i].type) << " " << fn.params[i].name;
        ss << (i + 1 < fn.params.size() ? ",\n" : "\n");
    }
    ss << ")\n{\n";

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
            ss << "    " << to_string(inst) << "\n";
        }
    }

    ss << "}\n";
    return ss.str();
}

std::string print(const Function& fn, const target::PtxOptions& opts) {
    return print_header(opts) + print_body(fn, has_native_tanh(opts.sm_arch));
}

std::string print_module(const std::vector<const Function*>& fns, const target::PtxOptions& opts) {
    std::string out = print_header(opts);
    for (size_t i = 0; i < fns.size(); ++i) {
        if (i) out += "\n";
        out += print_body(*fns[i], has_native_tanh(opts.sm_arch));
    }
    return out;
}

} // namespace brass::ptx
