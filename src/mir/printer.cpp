#include <brass/mir/printer.hpp>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <charconv>

namespace brass {

namespace {

std::string format_float(double val) {
    if (std::isnan(val)) return "nan";
    if (std::isinf(val)) return val < 0 ? "-inf" : "inf";
    char buf[64];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), val);
    *ptr = '\0';
    std::string s(buf, ptr - buf);
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
        s += ".0";
    }
    return s;
}

class FunctionPrinter {
public:
    FunctionPrinter(const Function& fn, std::ostream& os) : fn_(fn), os_(os) {
        build_name_maps();
    }

    void print() {
        // Function signature: func @name(%0: i32, %1: ptr) -> ret_type {
        os_ << "func @" << fn_.name() << "(";
        const BasicBlock* entry = fn_.entry_block();
        if (entry && entry->param_count() == fn_.param_count()) {
            for (size_t i = 0; i < entry->param_count(); ++i) {
                if (i > 0) os_ << ", ";
                os_ << value_name(entry->param(i)) << ": " << entry->param(i)->type().name();
            }
        } else {
            for (size_t i = 0; i < fn_.param_count(); ++i) {
                if (i > 0) os_ << ", ";
                os_ << "%" << i << ": " << fn_.param_type(i).name();
            }
        }
        os_ << ") -> " << fn_.return_type().name() << " {\n";

        // Print basic blocks
        for (size_t b_idx = 0; b_idx < fn_.block_count(); ++b_idx) {
            const BasicBlock* bb = fn_.blocks()[b_idx];
            if (!bb) continue;

            if (b_idx > 0) {
                os_ << "\n";
            }

            // Block label
            os_ << block_name(bb);
            if (b_idx > 0 && bb->param_count() > 0) {
                os_ << "(";
                for (size_t p_i = 0; p_i < bb->param_count(); ++p_i) {
                    if (p_i > 0) os_ << ", ";
                    os_ << value_name(bb->param(p_i)) << ": " << bb->param(p_i)->type().name();
                }
                os_ << ")";
            }
            os_ << ":\n";

            // Block instructions
            for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
                if (inst) {
                    print_inst(*inst);
                }
            }
        }

        // Resume table
        if (!fn_.resume_points().empty()) {
            os_ << "\nresume_table {\n";
            std::vector<std::pair<uint32_t, BasicBlock*>> sorted_rp = fn_.resume_points();
            std::sort(sorted_rp.begin(), sorted_rp.end(), [](const auto& a, const auto& b) {
                return a.first < b.first;
            });

            for (const auto& entry_rp : sorted_rp) {
                os_ << "  entry " << entry_rp.first << " -> " << block_name(entry_rp.second) << "\n";
            }
            os_ << "}\n";
        }

        os_ << "}\n";
    }

    void print_inst(const Instruction& inst) {
        os_ << "  ";

        if (inst.produces_value() && inst.result()) {
            if (inst.type().is_vector() || inst.opcode() == Opcode::vextract_lane) {
                os_ << value_name(inst.result()) << ": " << inst.type().name() << " = ";
            } else {
                os_ << value_name(inst.result()) << " = ";
            }
        }

        Opcode op = inst.opcode();
        switch (op) {
            case Opcode::iconst_i32:
                os_ << "iconst.i32 " << inst.imm_i32();
                break;
            case Opcode::iconst_i64:
                os_ << "iconst.i64 " << inst.imm_i64();
                break;
            case Opcode::fconst_f64:
                os_ << "fconst.f64 " << format_float(inst.imm_f64());
                break;
            case Opcode::patchable_const_i32:
                os_ << "patchable_const.i32 @" << inst.symbol() << ", " << inst.imm_i32();
                break;
            case Opcode::patchable_const_i64:
                os_ << "patchable_const.i64 @" << inst.symbol() << ", " << inst.imm_i64();
                break;

            case Opcode::sext_i64:
                os_ << "sext.i64 " << value_name(inst.operand(0));
                break;
            case Opcode::zext_i64:
                os_ << "zext.i64 " << value_name(inst.operand(0));
                break;
            case Opcode::trunc_i32:
                os_ << "trunc.i32 " << value_name(inst.operand(0));
                break;
            case Opcode::fptosi_i32:
                os_ << "fptosi.i32 " << value_name(inst.operand(0));
                break;
            case Opcode::fptosi_i64:
                os_ << "fptosi.i64 " << value_name(inst.operand(0));
                break;
            case Opcode::sitofp_f64_i32:
                os_ << "sitofp.f64.i32 " << value_name(inst.operand(0));
                break;
            case Opcode::sitofp_f64_i64:
                os_ << "sitofp.f64.i64 " << value_name(inst.operand(0));
                break;
            case Opcode::bitcast_i64_f64:
                os_ << "bitcast.i64.f64 " << value_name(inst.operand(0));
                break;
            case Opcode::bitcast_f64_i64:
                os_ << "bitcast.f64.i64 " << value_name(inst.operand(0));
                break;

            case Opcode::add:
            case Opcode::sub:
            case Opcode::mul:
            case Opcode::sdiv:
            case Opcode::udiv:
            case Opcode::smod:
            case Opcode::umod:
            case Opcode::shl:
            case Opcode::lshr:
            case Opcode::ashr:
                os_ << opcode_name(op) << "." << inst.type().name() << " "
                    << value_name(inst.operand(0)) << ", " << value_name(inst.operand(1));
                break;

            case Opcode::and_:
                os_ << "and." << inst.type().name() << " "
                    << value_name(inst.operand(0)) << ", " << value_name(inst.operand(1));
                break;
            case Opcode::or_:
                os_ << "or." << inst.type().name() << " "
                    << value_name(inst.operand(0)) << ", " << value_name(inst.operand(1));
                break;
            case Opcode::xor_:
                os_ << "xor." << inst.type().name() << " "
                    << value_name(inst.operand(0)) << ", " << value_name(inst.operand(1));
                break;

            case Opcode::neg:
            case Opcode::clz:
            case Opcode::ctz:
            case Opcode::popcnt:
                os_ << opcode_name(op) << "." << inst.type().name() << " "
                    << value_name(inst.operand(0));
                break;

            case Opcode::not_:
                os_ << "not." << inst.type().name() << " "
                    << value_name(inst.operand(0));
                break;

            case Opcode::eq:
            case Opcode::ne:
            case Opcode::slt:
            case Opcode::ult:
            case Opcode::sle:
            case Opcode::ule:
            case Opcode::sgt:
            case Opcode::ugt:
            case Opcode::sge:
            case Opcode::uge: {
                std::string_view op_t = inst.operand(0) ? inst.operand(0)->type().name() : "i32";
                os_ << opcode_name(op) << "." << op_t << " "
                    << value_name(inst.operand(0)) << ", " << value_name(inst.operand(1));
                break;
            }

            case Opcode::sadd_overflow:
            case Opcode::ssub_overflow:
            case Opcode::smul_overflow:
            case Opcode::uadd_overflow:
            case Opcode::usub_overflow:
            case Opcode::umul_overflow: {
                std::string_view op_t = inst.operand(0) ? inst.operand(0)->type().name() : "i32";
                os_ << opcode_name(op) << "." << op_t << " "
                    << value_name(inst.operand(0)) << ", " << value_name(inst.operand(1));
                break;
            }

            case Opcode::select:
                os_ << "select." << inst.type().name() << " "
                    << value_name(inst.operand(0)) << ", "
                    << value_name(inst.operand(1)) << ", "
                    << value_name(inst.operand(2));
                break;

            case Opcode::load:
                os_ << "load." << inst.memory_type().name() << " " << value_name(inst.operand(0));
                if (inst.offset() != 0) {
                    os_ << ", " << inst.offset();
                }
                break;

            case Opcode::store:
                os_ << "store." << inst.memory_type().name() << " " << value_name(inst.operand(0));
                if (inst.offset() != 0) {
                    os_ << ", " << inst.offset();
                }
                os_ << ", " << value_name(inst.operand(1));
                break;

            case Opcode::load_indexed:
                os_ << "load_indexed." << inst.memory_type().name() << " "
                    << value_name(inst.operand(0)) << ", " << value_name(inst.operand(1)) << ", "
                    << static_cast<uint32_t>(inst.scale());
                if (inst.offset() != 0) {
                    os_ << ", " << inst.offset();
                }
                break;

            case Opcode::store_indexed:
                os_ << "store_indexed." << inst.memory_type().name() << " "
                    << value_name(inst.operand(0)) << ", " << value_name(inst.operand(1)) << ", "
                    << static_cast<uint32_t>(inst.scale());
                if (inst.offset() != 0) {
                    os_ << ", " << inst.offset();
                }
                os_ << ", " << value_name(inst.operand(2));
                break;

            case Opcode::write_barrier:
                os_ << "write_barrier " << value_name(inst.operand(0)) << ", " << value_name(inst.operand(1));
                break;

            case Opcode::call:
                if (!inst.type().is_void()) {
                    os_ << "call." << inst.type().name() << " ";
                } else {
                    os_ << "call ";
                }
                os_ << "@" << inst.symbol() << "(";
                for (size_t i = 0; i < inst.operand_count(); ++i) {
                    if (i > 0) os_ << ", ";
                    os_ << value_name(inst.operand(i));
                }
                os_ << ")";
                break;

            case Opcode::call_indirect:
                if (!inst.type().is_void()) {
                    os_ << "call_indirect." << inst.type().name() << " ";
                } else {
                    os_ << "call_indirect ";
                }
                os_ << value_name(inst.operand(0)) << "(";
                for (size_t i = 1; i < inst.operand_count(); ++i) {
                    if (i > 1) os_ << ", ";
                    os_ << value_name(inst.operand(i));
                }
                os_ << ")";
                break;

            case Opcode::patchable_call:
                if (!inst.type().is_void()) {
                    os_ << "patchable_call." << inst.type().name() << " ";
                } else {
                    os_ << "patchable_call ";
                }
                os_ << "@" << inst.symbol() << ", @" << inst.extra_symbol() << "(";
                for (size_t i = 0; i < inst.operand_count(); ++i) {
                    if (i > 0) os_ << ", ";
                    os_ << value_name(inst.operand(i));
                }
                os_ << ")";
                break;

            case Opcode::safepoint:
                os_ << "safepoint";
                break;

            case Opcode::guard:
                os_ << "guard " << value_name(inst.operand(0)) << ", @" << inst.symbol();
                if (!inst.state_map().empty()) {
                    os_ << ", [";
                    for (size_t i = 0; i < inst.state_map().size(); ++i) {
                        if (i > 0) os_ << ", ";
                        os_ << value_name(inst.state_map()[i]);
                    }
                    os_ << "]";
                }
                break;

            case Opcode::resume_point:
                os_ << "resume_point " << inst.resume_id();
                break;

            case Opcode::br:
                os_ << "br " << format_branch_target(inst.branch_target());
                break;

            case Opcode::br_if:
                os_ << "br_if " << value_name(inst.operand(0)) << ", "
                    << format_branch_target(inst.true_target()) << ", "
                    << format_branch_target(inst.false_target());
                break;

            case Opcode::switch_: {
                std::string_view op_t = inst.operand(0) ? inst.operand(0)->type().name() : "i32";
                os_ << "switch." << op_t << " " << value_name(inst.operand(0))
                    << ", default: " << format_branch_target(inst.default_target()) << ", [";
                const auto& cases = inst.switch_cases();
                for (size_t i = 0; i < cases.size(); ++i) {
                    if (i > 0) os_ << ", ";
                    os_ << cases[i].value << ": " << format_branch_target(cases[i].target);
                }
                os_ << "]";
                break;
            }

            case Opcode::throw_:
                os_ << "throw " << value_name(inst.operand(0));
                break;

            case Opcode::invoke:
                if (!inst.type().is_void()) {
                    os_ << "invoke." << inst.type().name() << " ";
                } else {
                    os_ << "invoke ";
                }
                os_ << "@" << inst.symbol() << "(";
                for (size_t i = 0; i < inst.operand_count(); ++i) {
                    if (i > 0) os_ << ", ";
                    os_ << value_name(inst.operand(i));
                }
                os_ << "), " << format_branch_target(inst.normal_target()) << ", "
                    << format_branch_target(inst.unwind_target());
                break;

            case Opcode::landing_pad:
                if (!inst.type().is_void() && inst.type() != Type::i64()) {
                    os_ << "landing_pad." << inst.type().name();
                } else {
                    os_ << "landing_pad";
                }
                break;

            case Opcode::resume:
                os_ << "resume";
                if (inst.operand_count() > 0 && inst.operand(0)) {
                    os_ << " " << value_name(inst.operand(0));
                }
                break;

            case Opcode::coro_create: {
                os_ << "coro_create @" << inst.symbol() << "(";
                for (size_t i = 0; i < inst.operand_count(); ++i) {
                    if (i > 0) os_ << ", ";
                    os_ << value_name(inst.operand(i));
                }
                os_ << ")";
                break;
            }

            case Opcode::coro_suspend: {
                if (!inst.type().is_void()) {
                    os_ << "coro_suspend." << inst.type().name() << " ";
                } else {
                    os_ << "coro_suspend ";
                }
                if (inst.operand_count() > 0 && inst.operand(0)) {
                    os_ << value_name(inst.operand(0));
                } else {
                    os_ << "0";
                }
                if (inst.resume_id() != 0) {
                    os_ << ", " << inst.resume_id();
                }
                break;
            }

            case Opcode::coro_resume: {
                if (!inst.type().is_void()) {
                    os_ << "coro_resume." << inst.type().name() << " ";
                } else {
                    os_ << "coro_resume ";
                }
                os_ << value_name(inst.operand(0));
                if (inst.operand_count() > 1 && inst.operand(1)) {
                    os_ << ", " << value_name(inst.operand(1));
                }
                break;
            }

            case Opcode::coro_destroy: {
                os_ << "coro_destroy " << value_name(inst.operand(0));
                break;
            }

            case Opcode::vadd:
            case Opcode::vsub:
            case Opcode::vmul:
            case Opcode::vdiv:
            case Opcode::vmin:
            case Opcode::vmax:
            case Opcode::vand:
            case Opcode::vor:
            case Opcode::vxor:
                os_ << opcode_name(op) << " "
                    << value_name(inst.operand(0)) << ", " << value_name(inst.operand(1));
                break;

            case Opcode::vneg:
            case Opcode::vsqrt:
            case Opcode::vnot:
                os_ << opcode_name(op) << " " << value_name(inst.operand(0));
                break;

            case Opcode::vload:
                os_ << "vload." << inst.memory_type().name() << " "
                    << value_name(inst.operand(0)) << ", " << inst.offset();
                break;

            case Opcode::vstore:
                os_ << "vstore." << inst.memory_type().name() << " "
                    << value_name(inst.operand(0)) << ", " << inst.offset() << ", "
                    << value_name(inst.operand(1));
                break;

            case Opcode::vbroadcast:
                os_ << "vbroadcast." << inst.type().name() << " " << value_name(inst.operand(0));
                break;

            case Opcode::vextract_lane:
                os_ << "vextract_lane " << value_name(inst.operand(0)) << ", " << inst.lane();
                break;

            case Opcode::vinsert_lane:
                os_ << "vinsert_lane " << value_name(inst.operand(0)) << ", "
                    << value_name(inst.operand(1)) << ", " << inst.lane();
                break;

            case Opcode::vshuffle: {
                std::stringstream hex_ss;
                hex_ss << "0x" << std::uppercase << std::hex << inst.shuffle_mask();
                os_ << "vshuffle " << value_name(inst.operand(0)) << ", "
                    << value_name(inst.operand(1)) << ", " << hex_ss.str();
                break;
            }

            case Opcode::vzero:
                os_ << "vzero." << inst.type().name();
                break;

            case Opcode::ret:
                os_ << "ret";
                if (inst.operand_count() > 0 && inst.operand(0)) {
                    os_ << " " << value_name(inst.operand(0));
                }
                break;

            case Opcode::unreachable:
                os_ << "unreachable";
                break;
        }

        os_ << "\n";
    }

    std::string value_name(const Value* val) const {
        if (!val) return "%null";
        auto it = val_names_.find(val);
        if (it != val_names_.end()) {
            return it->second;
        }
        return "%v" + std::to_string(val->id());
    }

    std::string block_name(const BasicBlock* bb) const {
        if (!bb) return "bb_null";
        auto it = bb_names_.find(bb);
        if (it != bb_names_.end()) {
            return it->second;
        }
        return bb->name().empty() ? ("bb" + std::to_string(bb->id())) : std::string(bb->name());
    }

    std::string format_branch_target(const BranchTarget& target) const {
        std::string s = block_name(target.block);
        if (!target.args.empty()) {
            s += "(";
            for (size_t i = 0; i < target.args.size(); ++i) {
                if (i > 0) s += ", ";
                s += value_name(target.args[i]);
            }
            s += ")";
        }
        return s;
    }

private:
    void build_name_maps() {
        size_t next_v_id = 0;

        for (size_t b_idx = 0; b_idx < fn_.block_count(); ++b_idx) {
            const BasicBlock* bb = fn_.blocks()[b_idx];
            if (!bb) continue;

            if (!bb->name().empty()) {
                bb_names_[bb] = std::string(bb->name());
            } else {
                bb_names_[bb] = "bb" + std::to_string(b_idx);
            }

            for (const Value* param : bb->params()) {
                if (param && val_names_.find(param) == val_names_.end()) {
                    val_names_[param] = "%" + std::to_string(next_v_id++);
                }
            }

            for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
                if (inst && inst->produces_value() && inst->result()) {
                    if (val_names_.find(inst->result()) == val_names_.end()) {
                        val_names_[inst->result()] = "%" + std::to_string(next_v_id++);
                    }
                }
            }
        }
    }

    const Function& fn_;
    std::ostream& os_;
    std::unordered_map<const Value*, std::string> val_names_;
    std::unordered_map<const BasicBlock*, std::string> bb_names_;
};

} // namespace

void print_function(const Function& fn, std::ostream& os) {
    FunctionPrinter printer(fn, os);
    printer.print();
}

std::string to_string(const Function& fn) {
    std::ostringstream oss;
    print_function(fn, oss);
    return oss.str();
}

void print_block(const BasicBlock& bb, std::ostream& os) {
    if (bb.parent()) {
        FunctionPrinter printer(*bb.parent(), os);
        os << printer.block_name(&bb) << ":\n";
        for (const Instruction* inst : const_cast<BasicBlock&>(bb)) {
            if (inst) {
                printer.print_inst(*inst);
            }
        }
    } else {
        os << (bb.name().empty() ? ("bb" + std::to_string(bb.id())) : std::string(bb.name())) << ":\n";
    }
}

std::string to_string(const BasicBlock& bb) {
    std::ostringstream oss;
    print_block(bb, oss);
    return oss.str();
}

void print_instruction(const Instruction& inst, std::ostream& os) {
    if (inst.parent() && inst.parent()->parent()) {
        FunctionPrinter printer(*inst.parent()->parent(), os);
        printer.print_inst(inst);
    } else {
        os << opcode_name(inst.opcode()) << "\n";
    }
}

std::string to_string(const Instruction& inst) {
    std::ostringstream oss;
    print_instruction(inst, oss);
    return oss.str();
}

void print_module(const Module& mod, std::ostream& os) {
    if (!mod.name().empty()) {
        os << "module @" << mod.name() << "\n\n";
    }

    if (!mod.external_symbols().empty()) {
        for (std::string_view sym : mod.external_symbols()) {
            os << "extern @" << sym << "\n";
        }
        if (!mod.functions().empty()) {
            os << "\n";
        }
    }

    for (size_t i = 0; i < mod.function_count(); ++i) {
        const Function* fn = mod.functions()[i];
        if (fn) {
            if (i > 0) {
                os << "\n";
            }
            print_function(*fn, os);
        }
    }
}

std::string to_string(const Module& mod) {
    std::ostringstream oss;
    print_module(mod, oss);
    return oss.str();
}

} // namespace brass
