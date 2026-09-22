#include <brass/target/ptx/ptx_printer.hpp>

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

    switch (inst.op) {
        case Opcode::shfl:
            if (inst.is_sync) suffix("sync");
            suffix(to_string(inst.shfl_mode));
            break;
        case Opcode::bar:
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

std::string print_body(const Function& fn) {
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
            ss << "    " << to_string(inst) << "\n";
        }
    }

    ss << "}\n";
    return ss.str();
}

std::string print(const Function& fn, const target::PtxOptions& opts) {
    return print_header(opts) + print_body(fn);
}

std::string print_module(const std::vector<const Function*>& fns, const target::PtxOptions& opts) {
    std::string out = print_header(opts);
    for (size_t i = 0; i < fns.size(); ++i) {
        if (i) out += "\n";
        out += print_body(*fns[i]);
    }
    return out;
}

} // namespace brass::ptx
