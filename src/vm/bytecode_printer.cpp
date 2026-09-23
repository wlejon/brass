#include <brass/vm/bytecode.hpp>
#include <sstream>
#include <iomanip>

namespace brass {

namespace {

std::string reg_name(uint32_t r) {
    return r == kNoReg ? std::string("_") : "r" + std::to_string(r);
}

void print_target(std::ostream& os, size_t pc, int64_t offset) {
    os << std::showpos << offset << std::noshowpos << " -> [" << std::setfill('0') << std::setw(4)
       << static_cast<int64_t>(pc) + offset << std::setfill(' ') << "]";
}

void print_call_site(std::ostream& os, const BytecodeFunction& fn, uint32_t idx) {
    os << ", site[" << idx << "]";
    if (idx >= fn.call_sites.size()) return;
    const auto& cs = fn.call_sites[idx];
    os << " (";
    if (cs.callee_reg != kNoReg) {
        os << "[" << reg_name(cs.callee_reg) << "]";
    } else {
        os << "@" << cs.callee;
    }
    os << " args: [";
    for (size_t a = 0; a < cs.arg_regs.size(); ++a) {
        if (a > 0) os << ", ";
        os << reg_name(cs.arg_regs[a]);
    }
    os << "])";
}

bool is_unary(BytecodeOp op) {
    switch (op) {
        case BytecodeOp::mov: case BytecodeOp::vmov: case BytecodeOp::sext64: case BytecodeOp::zext64:
        case BytecodeOp::trunc32: case BytecodeOp::trunc8: case BytecodeOp::fptosi32: case BytecodeOp::fptosi64:
        case BytecodeOp::fptosi32_f32: case BytecodeOp::fptosi64_f32: case BytecodeOp::sitofp_f64:
        case BytecodeOp::sitofp_f32: case BytecodeOp::sitofp_f64_i64: case BytecodeOp::sitofp_f32_i64:
        case BytecodeOp::fptrunc_f32: case BytecodeOp::fpext_f64: case BytecodeOp::bitcast_i64_f64:
        case BytecodeOp::bitcast_f64_i64: case BytecodeOp::neg_i32: case BytecodeOp::neg_i64:
        case BytecodeOp::neg_f32: case BytecodeOp::neg_f64: case BytecodeOp::not_i32: case BytecodeOp::not_i64:
        case BytecodeOp::clz_i32: case BytecodeOp::clz_i64: case BytecodeOp::ctz_i32: case BytecodeOp::ctz_i64:
        case BytecodeOp::popcnt_i32: case BytecodeOp::popcnt_i64: case BytecodeOp::sqrt_f32:
        case BytecodeOp::sqrt_f64: case BytecodeOp::fabs_f32: case BytecodeOp::fabs_f64:
        case BytecodeOp::floor_f32: case BytecodeOp::floor_f64: case BytecodeOp::ceil_f32:
        case BytecodeOp::ceil_f64: case BytecodeOp::round_f32: case BytecodeOp::round_f64:
        case BytecodeOp::vbroadcast: case BytecodeOp::vneg: case BytecodeOp::vsqrt: case BytecodeOp::vnot:
            return true;
        default:
            return false;
    }
}

// Prints the operands of the instruction at `pc` and returns the number of
// code words it occupies.
size_t print_operands(std::ostream& os, const BytecodeFunction& fn, size_t pc) {
    const BytecodeWord w = fn.code[pc];
    const BytecodeOp op = decode_op(w);
    const uint32_t a = decode_a(w);
    const uint32_t b = decode_b(w);
    const uint32_t c = decode_c(w);
    const uint32_t u = decode_uimm32(w);
    const uint64_t next = pc + 1 < fn.code.size() ? fn.code[pc + 1] : 0;

    if (is_unary(op)) {
        os << " " << reg_name(a) << ", " << reg_name(b);
        return 1;
    }
    switch (op) {
        case BytecodeOp::nop: case BytecodeOp::unreachable: case BytecodeOp::ret_void:
        case BytecodeOp::safepoint: case BytecodeOp::resume_point: case BytecodeOp::osr_entry:
            break;
        case BytecodeOp::ret: case BytecodeOp::throw_: case BytecodeOp::resume: case BytecodeOp::landing_pad:
        case BytecodeOp::pinned_tls_read: case BytecodeOp::read_sp: case BytecodeOp::vzero:
            os << " " << reg_name(a);
            break;
        case BytecodeOp::pinned_tls_write: case BytecodeOp::coro_destroy:
            os << " " << reg_name(b);
            break;
        case BytecodeOp::write_barrier:
            os << " " << reg_name(b) << ", " << reg_name(c);
            break;
        case BytecodeOp::mov_imm:
        case BytecodeOp::iconst32:
            os << " " << reg_name(a) << ", " << decode_imm32(w);
            break;
        case BytecodeOp::load_const:
            os << " " << reg_name(a) << ", const[" << u << "]";
            if (u < fn.constants.size()) os << " (=" << static_cast<int64_t>(fn.constants[u]) << ")";
            break;
        case BytecodeOp::patchable_const32:
        case BytecodeOp::patchable_const64:
            os << " " << reg_name(a) << ", patch[" << u << "]";
            if (u < fn.patch_consts.size()) {
                os << " (@" << fn.patch_consts[u].symbol << " default " << fn.patch_consts[u].default_value << ")";
            }
            break;
        case BytecodeOp::func_addr:
            os << " " << reg_name(a) << ", string[" << u << "]";
            if (u < fn.string_pool.size()) os << " (@" << fn.string_pool[u] << ")";
            break;
        case BytecodeOp::call: case BytecodeOp::patchable_call: case BytecodeOp::call_indirect:
        case BytecodeOp::invoke: case BytecodeOp::coro_create:
            os << " " << reg_name(a);
            print_call_site(os, fn, u);
            break;
        case BytecodeOp::guard:
            os << " " << reg_name(a) << ", guard[" << u << "]";
            if (u < fn.guards.size()) {
                os << " (resume_id=" << fn.guards[u].resume_id << " stub=" << fn.guards[u].exit_stub << ")";
            }
            break;
        case BytecodeOp::switch_:
            os << " " << reg_name(a) << ", table[" << u << "]";
            break;
        case BytecodeOp::jump:
            os << " ";
            print_target(os, pc, decode_imm32(w));
            break;
        case BytecodeOp::jump_if: case BytecodeOp::jump_if_not:
            os << " " << reg_name(a) << ", ";
            print_target(os, pc, decode_imm32(w));
            break;
        case BytecodeOp::br_eq_i32: case BytecodeOp::br_ne_i32: case BytecodeOp::br_slt_i32:
        case BytecodeOp::br_sle_i32: case BytecodeOp::br_ult_i32: case BytecodeOp::br_ule_i32:
        case BytecodeOp::br_eq_i64: case BytecodeOp::br_ne_i64: case BytecodeOp::br_slt_i64:
        case BytecodeOp::br_sle_i64: case BytecodeOp::br_ult_i64: case BytecodeOp::br_ule_i64:
            os << " " << reg_name(a) << ", " << reg_name(b) << ", ";
            print_target(os, pc, decode_imm24(w));
            break;
        case BytecodeOp::add_imm_i32: case BytecodeOp::add_imm_i64:
            os << " " << reg_name(a) << ", " << reg_name(b) << ", " << decode_imm24(w);
            break;
        case BytecodeOp::load8: case BytecodeOp::load16: case BytecodeOp::load32: case BytecodeOp::load64:
        case BytecodeOp::vload:
            os << " " << reg_name(a) << ", [" << reg_name(b) << " + " << decode_imm24(w) << "]";
            break;
        case BytecodeOp::store8: case BytecodeOp::store16: case BytecodeOp::store32: case BytecodeOp::store64:
        case BytecodeOp::vstore:
            os << " [" << reg_name(b) << " + " << decode_imm24(w) << "], " << reg_name(a);
            break;
        case BytecodeOp::alloca_:
            os << " " << reg_name(a) << ", size=" << next << ", align=" << decode_d(w);
            break;
        case BytecodeOp::coro_suspend:
            os << " " << reg_name(a) << ", " << reg_name(b) << ", resume_id=" << next;
            break;
        case BytecodeOp::vshuffle:
            os << " " << reg_name(a) << ", " << reg_name(b) << ", " << reg_name(c) << ", mask=0x" << std::hex << next
               << std::dec;
            break;
        case BytecodeOp::index_addr:
            os << " " << reg_name(a) << ", " << reg_name(b) << " + " << reg_name(c) << " * " << decode_d(w);
            break;
        case BytecodeOp::vextract_lane:
            os << " " << reg_name(a) << ", " << reg_name(b) << ", lane " << decode_d(w);
            break;
        case BytecodeOp::vinsert_lane:
            os << " " << reg_name(a) << ", " << reg_name(b) << ", " << reg_name(c) << ", lane " << decode_d(w);
            break;
        default:
            os << " " << reg_name(a) << ", " << reg_name(b) << ", " << reg_name(c);
            break;
    }
    return bytecode_inst_words(op);
}

} // namespace

void dump(const BytecodeFunction& fn, std::ostream& os) {
    os << "function @" << (fn.name.empty() ? "anonymous" : fn.name) << "(";
    for (size_t i = 0; i < fn.param_types.size(); ++i) {
        if (i > 0) os << ", ";
        os << "r" << i << ": " << fn.param_types[i];
    }
    os << ") -> " << fn.return_type << " {\n";
    os << "  .registers " << fn.num_registers << "\n";
    os << "  .params    " << fn.num_params << "\n";

    if (!fn.constants.empty()) {
        os << "  .constants {\n";
        for (size_t i = 0; i < fn.constants.size(); ++i) {
            os << "    [" << std::setw(3) << i << "] = 0x" << std::hex << fn.constants[i] << std::dec;
            os << " (" << static_cast<int64_t>(fn.constants[i]) << ")\n";
        }
        os << "  }\n";
    }

    if (!fn.string_pool.empty()) {
        os << "  .strings {\n";
        for (size_t i = 0; i < fn.string_pool.size(); ++i) {
            os << "    [" << std::setw(3) << i << "] = \"" << fn.string_pool[i] << "\"\n";
        }
        os << "  }\n";
    }

    for (size_t pc = 0; pc < fn.code.size();) {
        const BytecodeOp op = decode_op(fn.code[pc]);
        os << "  " << std::setfill('0') << std::setw(4) << pc << ": " << std::setfill(' ');
        os << std::left << std::setw(18) << bytecode_op_name(op) << std::right;
        const size_t words = print_operands(os, fn, pc);

        DebugLoc loc = fn.get_line_info(static_cast<uint32_t>(pc));
        if (loc.is_valid()) {
            os << "  ; line " << loc.line;
        }
        os << "\n";
        pc += words;
    }

    for (size_t t = 0; t < fn.switch_tables.size(); ++t) {
        const auto& table = fn.switch_tables[t];
        os << "  .switch_table " << t << (table.is_i32 ? " (i32)" : "") << " {\n";
        for (const auto& [value, target] : table.cases) {
            os << "    " << value << " -> [" << std::setfill('0') << std::setw(4) << target << std::setfill(' ') << "]\n";
        }
        os << "    default -> [" << std::setfill('0') << std::setw(4) << table.default_offset << std::setfill(' ')
           << "]\n  }\n";
    }

    if (!fn.exception_table.empty()) {
        os << "  .exception_table {\n";
        for (const auto& ee : fn.exception_table) {
            os << "    [" << std::setfill('0') << std::setw(4) << ee.start_pc << ".."
               << std::setw(4) << ee.end_pc << "] -> handler ["
               << std::setw(4) << ee.handler_pc << "]\n" << std::setfill(' ');
        }
        os << "  }\n";
    }

    if (!fn.resume_points.empty()) {
        os << "  .resume_points {\n";
        for (const auto& rp : fn.resume_points) {
            os << "    resume_id " << rp.resume_id << " -> [" << std::setfill('0') << std::setw(4)
               << rp.target_pc << "]\n" << std::setfill(' ');
        }
        os << "  }\n";
    }

    os << "}\n";
}

void dump(const BytecodeModule& mod, std::ostream& os) {
    os << "module @" << (mod.name().empty() ? "anonymous" : mod.name()) << " {\n";
    for (const auto& fn : mod.functions()) {
        if (fn) {
            dump(*fn, os);
        }
    }
    os << "}\n";
}

std::string disassemble(const BytecodeFunction& fn) {
    std::ostringstream ss;
    dump(fn, ss);
    return ss.str();
}

std::string disassemble(const BytecodeModule& mod) {
    std::ostringstream ss;
    dump(mod, ss);
    return ss.str();
}

} // namespace brass
