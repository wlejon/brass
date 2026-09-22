#include <brass/vm/bytecode.hpp>
#include <sstream>
#include <iomanip>

namespace brass {

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

    for (size_t pc = 0; pc < fn.code.size(); ++pc) {
        uint32_t raw = fn.code[pc];
        BytecodeOp op = decode_op(raw);
        uint8_t dst = decode_dst(raw);
        uint8_t src1 = decode_src1(raw);
        uint8_t src2 = decode_src2(raw);
        uint16_t u16 = decode_u16(raw);
        int16_t s16 = decode_s16(raw);

        os << "  " << std::setfill('0') << std::setw(4) << pc << ": " << std::setfill(' ');
        os << std::left << std::setw(18) << bytecode_op_name(op) << std::right;

        switch (op) {
            case BytecodeOp::nop:
            case BytecodeOp::unreachable:
            case BytecodeOp::ret_void:
            case BytecodeOp::safepoint:
                break;

            case BytecodeOp::ret:
            case BytecodeOp::throw_:
            case BytecodeOp::resume:
                os << " r" << static_cast<uint32_t>(dst);
                break;

            case BytecodeOp::landing_pad:
                os << " r" << static_cast<uint32_t>(dst);
                break;

            case BytecodeOp::mov_imm:
                os << " r" << static_cast<uint32_t>(dst) << ", " << s16;
                break;

            case BytecodeOp::alloca_:
                os << " r" << static_cast<uint32_t>(dst) << ", size=" << u16;
                break;

            case BytecodeOp::load_const:
            case BytecodeOp::iconst32:
            case BytecodeOp::iconst64:
            case BytecodeOp::fconst32:
            case BytecodeOp::fconst64:
            case BytecodeOp::patchable_const32:
            case BytecodeOp::patchable_const64:
                os << " r" << static_cast<uint32_t>(dst) << ", const[" << u16 << "]";
                if (u16 < fn.constants.size()) {
                    os << " (=" << static_cast<int64_t>(fn.constants[u16]) << ")";
                }
                break;

            case BytecodeOp::jump: {
                int32_t target_pc = static_cast<int32_t>(pc) + s16;
                os << " " << std::showpos << s16 << std::noshowpos
                   << " -> [" << std::setfill('0') << std::setw(4) << target_pc << std::setfill(' ') << "]";
                break;
            }

            case BytecodeOp::jump_if:
            case BytecodeOp::jump_if_not: {
                int32_t target_pc = static_cast<int32_t>(pc) + s16;
                os << " r" << static_cast<uint32_t>(dst) << ", " << std::showpos << s16 << std::noshowpos
                   << " -> [" << std::setfill('0') << std::setw(4) << target_pc << std::setfill(' ') << "]";
                break;
            }

            case BytecodeOp::switch_: {
                os << " r" << static_cast<uint32_t>(dst) << ", table[" << u16 << "]";
                break;
            }

            case BytecodeOp::func_addr: {
                os << " r" << static_cast<uint32_t>(dst) << ", string[" << u16 << "]";
                if (u16 < fn.string_pool.size()) {
                    os << " (@" << fn.string_pool[u16] << ")";
                }
                break;
            }

            case BytecodeOp::call:
            case BytecodeOp::patchable_call: {
                os << " r" << static_cast<uint32_t>(dst) << ", site[" << u16 << "]";
                if (u16 < fn.call_sites.size()) {
                    const auto& cs = fn.call_sites[u16];
                    os << " (@" << cs.callee << " args: [";
                    for (size_t a = 0; a < cs.arg_regs.size(); ++a) {
                        if (a > 0) os << ", ";
                        os << "r" << static_cast<uint32_t>(cs.arg_regs[a]);
                    }
                    os << "])";
                }
                break;
            }

            case BytecodeOp::call_indirect: {
                os << " r" << static_cast<uint32_t>(dst) << ", site[" << u16 << "]";
                if (u16 < fn.call_sites.size()) {
                    const auto& cs = fn.call_sites[u16];
                    os << " ([r" << static_cast<uint32_t>(cs.callee_reg) << "] args: [";
                    for (size_t a = 0; a < cs.arg_regs.size(); ++a) {
                        if (a > 0) os << ", ";
                        os << "r" << static_cast<uint32_t>(cs.arg_regs[a]);
                    }
                    os << "])";
                }
                break;
            }

            case BytecodeOp::guard: {
                os << " r" << static_cast<uint32_t>(dst) << ", guard[" << u16 << "]";
                if (u16 < fn.guards.size()) {
                    const auto& g = fn.guards[u16];
                    os << " (resume_id=" << g.resume_id << " stub=" << g.exit_stub << ")";
                }
                break;
            }

            case BytecodeOp::mov:
            case BytecodeOp::sext64:
            case BytecodeOp::zext64:
            case BytecodeOp::trunc32:
            case BytecodeOp::trunc8:
            case BytecodeOp::fptosi32:
            case BytecodeOp::fptosi64:
            case BytecodeOp::fptosi32_f32:
            case BytecodeOp::fptosi64_f32:
            case BytecodeOp::sitofp_f64:
            case BytecodeOp::sitofp_f32:
            case BytecodeOp::sitofp_f64_i64:
            case BytecodeOp::sitofp_f32_i64:
            case BytecodeOp::fptrunc_f32:
            case BytecodeOp::fpext_f64:
            case BytecodeOp::bitcast_i64_f64:
            case BytecodeOp::bitcast_f64_i64:
            case BytecodeOp::neg_i32:
            case BytecodeOp::neg_i64:
            case BytecodeOp::neg_f32:
            case BytecodeOp::neg_f64:
            case BytecodeOp::not_i32:
            case BytecodeOp::not_i64:
            case BytecodeOp::clz_i32:
            case BytecodeOp::clz_i64:
            case BytecodeOp::ctz_i32:
            case BytecodeOp::ctz_i64:
            case BytecodeOp::popcnt_i32:
            case BytecodeOp::popcnt_i64:
            case BytecodeOp::sqrt_f32:
            case BytecodeOp::sqrt_f64:
            case BytecodeOp::fabs_f32:
            case BytecodeOp::fabs_f64:
            case BytecodeOp::floor_f32:
            case BytecodeOp::floor_f64:
            case BytecodeOp::ceil_f32:
            case BytecodeOp::ceil_f64:
            case BytecodeOp::round_f32:
            case BytecodeOp::round_f64:
                os << " r" << static_cast<uint32_t>(dst) << ", r" << static_cast<uint32_t>(src1);
                break;

            case BytecodeOp::load8:
            case BytecodeOp::load16:
            case BytecodeOp::load32:
            case BytecodeOp::load64:
                os << " r" << static_cast<uint32_t>(dst) << ", [r" << static_cast<uint32_t>(src1) << " + "
                   << static_cast<uint32_t>(src2) << "]";
                break;

            case BytecodeOp::store8:
            case BytecodeOp::store16:
            case BytecodeOp::store32:
            case BytecodeOp::store64:
                os << " [r" << static_cast<uint32_t>(src1) << " + " << static_cast<uint32_t>(src2)
                   << "], r" << static_cast<uint32_t>(dst);
                break;

            default:
                // Standard 3-register ABC format
                os << " r" << static_cast<uint32_t>(dst) << ", r" << static_cast<uint32_t>(src1)
                   << ", r" << static_cast<uint32_t>(src2);
                break;
        }

        DebugLoc loc = fn.get_line_info(static_cast<uint32_t>(pc));
        if (loc.is_valid()) {
            os << "  ; line " << loc.line;
        }

        os << "\n";
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
