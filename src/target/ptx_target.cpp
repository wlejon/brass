#include <brass/target/ptx_target.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/opcodes.hpp>

#include <sstream>
#include <iomanip>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace brass::target {

namespace {

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

bool is_64bit_int(Type t) {
    return t == Type::i64() || t == Type::ptr() || t == Type::gcref();
}

class PtxEmitter {
public:
    PtxEmitter(const Function& fn, const PtxOptions& opts)
        : fn_(fn), opts_(opts) {}

    std::string emit() {
        // PTX `ret` takes no operand and `.entry` kernels cannot return values;
        // results must be written through pointer parameters.
        if (!fn_.return_type().is_void()) {
            throw std::runtime_error("PtxTarget: kernel '" + std::string(fn_.name()) +
                                     "' has a non-void return type; .entry kernels cannot return values");
        }
        assign_registers();

        // Emit the whole body first so that any registers allocated lazily
        // (indexed memory addressing, builtin call temporaries) are accounted
        // for before the .reg declarations are printed.
        std::ostringstream body;

        if (const auto* entry = fn_.entry_block()) {
            for (size_t i = 0; i < entry->param_count(); ++i) {
                const Value* param_val = entry->param(i);
                if (!param_val) {
                    throw std::runtime_error("PtxTarget: entry block has a null parameter at index " +
                                             std::to_string(i));
                }
                std::string reg = get_reg(param_val);
                body << "    ld.param." << ptx_type_suffix(param_val->type()) << " " << reg
                     << ", [param_" << i << "];\n";
            }
        }
        body << "\n";

        for (size_t b_idx = 0; b_idx < fn_.blocks().size(); ++b_idx) {
            const BasicBlock* bb = fn_.blocks()[b_idx];
            body << block_label(bb) << ":\n";
            for (const auto* inst_ptr : *bb) {
                emit_instruction(*inst_ptr, body);
            }
        }

        std::ostringstream ss;
        ss << ".visible .entry " << fn_.name() << "(\n";
        const auto& params = fn_.param_types();
        for (size_t i = 0; i < params.size(); ++i) {
            ss << "    .param " << param_type_name(params[i]) << " param_" << i;
            if (i + 1 < params.size()) ss << ",\n";
            else ss << "\n";
        }
        ss << ")\n{\n";

        if (num_pred_ > 0) ss << "    .reg .pred %p<" << num_pred_ << ">;\n";
        if (num_u32_ > 0)  ss << "    .reg .b32 %r<" << num_u32_ << ">;\n";
        if (num_u64_ > 0)  ss << "    .reg .b64 %rd<" << num_u64_ << ">;\n";
        if (num_f32_ > 0)  ss << "    .reg .f32 %f<" << num_f32_ << ">;\n";
        if (num_f64_ > 0)  ss << "    .reg .f64 %fd<" << num_f64_ << ">;\n";
        ss << "\n";
        ss << body.str();
        ss << "}\n";
        return ss.str();
    }

private:
    const Function& fn_;
    const PtxOptions& opts_;

    uint32_t num_pred_ = 0;
    uint32_t num_u32_ = 0;
    uint32_t num_u64_ = 0;
    uint32_t num_f32_ = 0;
    uint32_t num_f64_ = 0;

    std::unordered_map<const Value*, std::string> reg_map_;
    std::unordered_map<const Value*, std::string> pred_map_; // For comparison instructions

    std::string block_label(const BasicBlock* bb) {
        return "$L_bb_" + std::to_string(bb->id());
    }

    std::string param_type_name(Type t) {
        if (t == Type::f32()) return ".f32";
        if (t == Type::f64()) return ".f64";
        if (t == Type::i32()) return ".u32";
        return ".u64"; // ptr, i64
    }

    std::string ptx_type_suffix(Type t) {
        if (t == Type::f32()) return "f32";
        if (t == Type::f64()) return "f64";
        if (t == Type::i32()) return "u32";
        return "u64";
    }

    std::string alloc_reg(Type t) {
        if (t == Type::f32()) return "%f" + std::to_string(num_f32_++);
        if (t == Type::f64()) return "%fd" + std::to_string(num_f64_++);
        if (t == Type::i32()) return "%r" + std::to_string(num_u32_++);
        if (t == Type::f32x4()) {
            uint32_t base = num_f32_;
            num_f32_ += 4;
            return "%f" + std::to_string(base);
        }
        if (t == Type::f32x8()) {
            uint32_t base = num_f32_;
            num_f32_ += 8;
            return "%f" + std::to_string(base);
        }
        if (t == Type::f64x2()) {
            uint32_t base = num_f64_;
            num_f64_ += 2;
            return "%fd" + std::to_string(base);
        }
        if (t == Type::f64x4()) {
            uint32_t base = num_f64_;
            num_f64_ += 4;
            return "%fd" + std::to_string(base);
        }
        return "%rd" + std::to_string(num_u64_++);
    }

    void assign_registers() {
        // Reserve at least 1 register of each type to be safe
        num_pred_ = 1;
        num_u32_ = 1;
        num_u64_ = 1;
        num_f32_ = 1;
        num_f64_ = 1;

        if (const auto* entry = fn_.entry_block()) {
            for (size_t i = 0; i < entry->param_count(); ++i) {
                const Value* param = entry->param(i);
                if (param) reg_map_[param] = alloc_reg(param->type());
            }
        }

        for (const auto* bb : fn_.blocks()) {
            for (const auto* param : bb->params()) {
                if (param && reg_map_.find(param) == reg_map_.end()) {
                    reg_map_[param] = alloc_reg(param->type());
                }
            }

            for (const auto* inst : *bb) {
                if (inst->result() && inst->type() != Type::void_type()) {
                    reg_map_[inst->result()] = alloc_reg(inst->type());
                }
                if (brass::is_comparison(inst->opcode()) && inst->result()) {
                    pred_map_[inst->result()] = "%p" + std::to_string(num_pred_++);
                }
            }
        }
    }

    std::string get_reg(const Value* v) {
        if (!v) return "%r0";
        auto it = reg_map_.find(v);
        if (it != reg_map_.end()) return it->second;
        return "%r0";
    }

    std::string get_reg(const Instruction* inst) {
        return inst ? get_reg(inst->result()) : "%r0";
    }

    std::string get_pred(const Value* v) {
        if (!v) return "%p0";
        auto it = pred_map_.find(v);
        if (it != pred_map_.end()) return it->second;
        return "%p0";
    }

    std::string get_pred(const Instruction* inst) {
        return inst ? get_pred(inst->result()) : "%p0";
    }

    // Return a predicate register that holds `cond != 0`, emitting a setp if
    // the condition was not itself produced by a comparison.
    std::string materialize_pred(const Value* cond, std::ostringstream& ss) {
        if (!cond) return "%p0";
        auto it = pred_map_.find(cond);
        if (it != pred_map_.end()) return it->second;

        std::string p = "%p" + std::to_string(num_pred_++);
        Type t = cond->type();
        if (t == Type::f32()) {
            ss << "    setp.neu.f32 " << p << ", " << get_reg(cond) << ", 0f00000000;\n";
        } else if (t == Type::f64()) {
            ss << "    setp.neu.f64 " << p << ", " << get_reg(cond) << ", 0d0000000000000000;\n";
        } else if (t == Type::i64() || t == Type::ptr() || t == Type::gcref()) {
            ss << "    setp.ne.u64 " << p << ", " << get_reg(cond) << ", 0;\n";
        } else {
            ss << "    setp.ne.u32 " << p << ", " << get_reg(cond) << ", 0;\n";
        }
        return p;
    }

    // Bit width suffix for integer shifts/bitwise instructions.
    std::string int_width_suffix(Type t) {
        return is_64bit_int(t) ? "b64" : "b32";
    }

    // PTX shift instructions take a 32-bit shift amount for both b32 and b64
    // data. Convert a 64-bit count into a fresh u32 register when necessary.
    std::string shift_amount_reg(const Value* amt, std::ostringstream& ss) {
        require(amt != nullptr, "shift missing amount");
        if (is_64bit_int(amt->type())) {
            std::string tmp = "%r" + std::to_string(num_u32_++);
            ss << "    cvt.u32.u64 " << tmp << ", " << get_reg(amt) << ";\n";
            return tmp;
        }
        return get_reg(amt);
    }

    std::string cmp_type_suffix(Opcode op, Type t) {
        if (t == Type::f32()) return "f32";
        if (t == Type::f64()) return "f64";
        bool width64 = is_64bit_int(t);
        bool is_unsigned = (op == Opcode::ult || op == Opcode::ule ||
                            op == Opcode::ugt || op == Opcode::uge);
        if (is_unsigned) return width64 ? "u64" : "u32";
        return width64 ? "s64" : "s32";
    }

    void require(bool cond, const char* what) {
        if (!cond) {
            throw std::runtime_error(std::string("PtxTarget: malformed instruction (") + what + ")");
        }
    }

    void emit_instruction(const Instruction& inst, std::ostringstream& ss) {
        switch (inst.opcode()) {
            case Opcode::iconst_i32: {
                int32_t val = inst.imm_i32();
                ss << "    mov.b32 " << get_reg(&inst) << ", " << val << ";\n";
                break;
            }
            case Opcode::iconst_i64: {
                int64_t val = inst.imm_i64();
                ss << "    mov.b64 " << get_reg(&inst) << ", " << val << ";\n";
                break;
            }
            case Opcode::fconst_f64: {
                if (inst.type() == Type::f32()) {
                    float f = static_cast<float>(inst.imm_f64());
                    ss << "    mov.f32 " << get_reg(&inst) << ", " << format_f32_hex(f) << ";\n";
                } else {
                    double d = inst.imm_f64();
                    ss << "    mov.f64 " << get_reg(&inst) << ", " << format_f64_hex(d) << ";\n";
                }
                break;
            }

            case Opcode::add: {
                Type t = inst.type();
                if (t == Type::f32()) {
                    ss << "    add.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else if (t == Type::f64()) {
                    ss << "    add.f64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else if (t == Type::i32()) {
                    ss << "    add.s32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else {
                    ss << "    add.s64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                }
                break;
            }
            case Opcode::sub: {
                Type t = inst.type();
                if (t == Type::f32()) {
                    ss << "    sub.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else if (t == Type::f64()) {
                    ss << "    sub.f64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else if (t == Type::i32()) {
                    ss << "    sub.s32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else {
                    ss << "    sub.s64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                }
                break;
            }
            case Opcode::mul: {
                Type t = inst.type();
                if (t == Type::f32()) {
                    ss << "    mul.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else if (t == Type::f64()) {
                    ss << "    mul.f64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else if (t == Type::i32()) {
                    ss << "    mul.lo.s32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else {
                    ss << "    mul.lo.s64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                }
                break;
            }
            case Opcode::fma_f32: {
                ss << "    fma.rn.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                   << ", " << get_reg(inst.operand(1)) << ", " << get_reg(inst.operand(2)) << ";\n";
                break;
            }
            case Opcode::fma_f64: {
                ss << "    fma.rn.f64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                   << ", " << get_reg(inst.operand(1)) << ", " << get_reg(inst.operand(2)) << ";\n";
                break;
            }
            case Opcode::sdiv:
            case Opcode::udiv: {
                Type t = inst.type();
                if (t == Type::f32()) {
                    ss << "    div.rn.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else if (t == Type::f64()) {
                    ss << "    div.rn.f64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else if (t == Type::i32()) {
                    const char* s = (inst.opcode() == Opcode::udiv) ? "u32" : "s32";
                    ss << "    div." << s << " " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else {
                    const char* s = (inst.opcode() == Opcode::udiv) ? "u64" : "s64";
                    ss << "    div." << s << " " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                }
                break;
            }
            case Opcode::neg: {
                Type t = inst.type();
                if (t == Type::f32()) {
                    ss << "    neg.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else if (t == Type::f64()) {
                    ss << "    neg.f64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else if (t == Type::i32()) {
                    ss << "    neg.s32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else {
                    ss << "    neg.s64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                }
                break;
            }

            case Opcode::and_: {
                ss << "    and." << int_width_suffix(inst.type()) << " " << get_reg(&inst) << ", "
                   << get_reg(inst.operand(0)) << ", " << get_reg(inst.operand(1)) << ";\n";
                break;
            }
            case Opcode::or_: {
                ss << "    or." << int_width_suffix(inst.type()) << " " << get_reg(&inst) << ", "
                   << get_reg(inst.operand(0)) << ", " << get_reg(inst.operand(1)) << ";\n";
                break;
            }
            case Opcode::xor_: {
                ss << "    xor." << int_width_suffix(inst.type()) << " " << get_reg(&inst) << ", "
                   << get_reg(inst.operand(0)) << ", " << get_reg(inst.operand(1)) << ";\n";
                break;
            }
            case Opcode::shl: {
                std::string amt = shift_amount_reg(inst.operand(1), ss);
                ss << "    shl." << int_width_suffix(inst.type()) << " " << get_reg(&inst) << ", "
                   << get_reg(inst.operand(0)) << ", " << amt << ";\n";
                break;
            }
            case Opcode::lshr: {
                bool w64 = is_64bit_int(inst.type());
                std::string amt = shift_amount_reg(inst.operand(1), ss);
                ss << "    shr.u" << (w64 ? "64" : "32") << " " << get_reg(&inst) << ", "
                   << get_reg(inst.operand(0)) << ", " << amt << ";\n";
                break;
            }
            case Opcode::ashr: {
                bool w64 = is_64bit_int(inst.type());
                std::string amt = shift_amount_reg(inst.operand(1), ss);
                ss << "    shr.s" << (w64 ? "64" : "32") << " " << get_reg(&inst) << ", "
                   << get_reg(inst.operand(0)) << ", " << amt << ";\n";
                break;
            }

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
                std::string cmp = "eq";
                if (inst.opcode() == Opcode::ne) cmp = "ne";
                else if (inst.opcode() == Opcode::slt || inst.opcode() == Opcode::ult) cmp = "lt";
                else if (inst.opcode() == Opcode::sle || inst.opcode() == Opcode::ule) cmp = "le";
                else if (inst.opcode() == Opcode::sgt || inst.opcode() == Opcode::ugt) cmp = "gt";
                else if (inst.opcode() == Opcode::sge || inst.opcode() == Opcode::uge) cmp = "ge";

                Type opnd_t = inst.operand(0) ? inst.operand(0)->type() : Type::i32();
                std::string type_s = cmp_type_suffix(inst.opcode(), opnd_t);

                std::string p_reg = get_pred(&inst);
                ss << "    setp." << cmp << "." << type_s << " " << p_reg << ", "
                   << get_reg(inst.operand(0)) << ", " << get_reg(inst.operand(1)) << ";\n";

                // Materialize boolean int in result register
                ss << "    selp.u32 " << get_reg(&inst) << ", 1, 0, " << p_reg << ";\n";
                break;
            }

            case Opcode::select: {
                std::string p_cond = materialize_pred(inst.operand(0), ss);
                Type t = inst.type();
                std::string sfx = (t == Type::f32()) ? "f32"
                                : (t == Type::f64()) ? "f64"
                                : (is_64bit_int(t)) ? "b64"
                                : "b32";
                ss << "    selp." << sfx << " " << get_reg(&inst) << ", "
                   << get_reg(inst.operand(1)) << ", " << get_reg(inst.operand(2))
                   << ", " << p_cond << ";\n";
                break;
            }

            case Opcode::load: {
                require(inst.operand(0) != nullptr, "load missing pointer");
                Type t = inst.type();
                std::string sfx = ptx_type_suffix(t);
                int32_t off = inst.offset();
                if (off != 0) {
                    ss << "    ld.global." << sfx << " " << get_reg(&inst) << ", ["
                       << get_reg(inst.operand(0)) << " + " << off << "];\n";
                } else {
                    ss << "    ld.global." << sfx << " " << get_reg(&inst) << ", ["
                       << get_reg(inst.operand(0)) << "];\n";
                }
                break;
            }

            case Opcode::store: {
                require(inst.operand(0) != nullptr && inst.operand(1) != nullptr,
                        "store missing operand");
                Type t = inst.operand(1)->type();
                std::string sfx = ptx_type_suffix(t);
                int32_t off = inst.offset();
                if (off != 0) {
                    ss << "    st.global." << sfx << " [" << get_reg(inst.operand(0))
                       << " + " << off << "], " << get_reg(inst.operand(1)) << ";\n";
                } else {
                    ss << "    st.global." << sfx << " [" << get_reg(inst.operand(0))
                       << "], " << get_reg(inst.operand(1)) << ";\n";
                }
                break;
            }

            case Opcode::vload: {
                require(inst.operand(0) != nullptr, "vload missing pointer");
                Type vt = inst.type();
                std::string base_reg = get_reg(&inst);
                int32_t off = inst.offset();
                if (vt == Type::f32x4()) {
                    int r_num = std::stoi(base_reg.substr(2));
                    ss << "    ld.global.v4.f32 {"
                       << "%f" << r_num << ", %f" << (r_num + 1) << ", %f" << (r_num + 2)
                       << ", %f" << (r_num + 3) << "}, [" << get_reg(inst.operand(0));
                    if (off != 0) ss << " + " << off;
                    ss << "];\n";
                } else if (vt == Type::f64x2()) {
                    int r_num = std::stoi(base_reg.substr(3));
                    ss << "    ld.global.v2.f64 {"
                       << "%fd" << r_num << ", %fd" << (r_num + 1) << "}, ["
                       << get_reg(inst.operand(0));
                    if (off != 0) ss << " + " << off;
                    ss << "];\n";
                } else {
                    throw std::runtime_error(
                        "PtxTarget: unsupported vload vector type (only f32x4/f64x2 are supported)");
                }
                break;
            }

            case Opcode::vstore: {
                require(inst.operand(0) != nullptr && inst.operand(1) != nullptr,
                        "vstore missing operand");
                Type vt = inst.operand(1)->type();
                std::string base_reg = get_reg(inst.operand(1));
                int32_t off = inst.offset();
                if (vt == Type::f32x4()) {
                    int r_num = std::stoi(base_reg.substr(2));
                    ss << "    st.global.v4.f32 [" << get_reg(inst.operand(0));
                    if (off != 0) ss << " + " << off;
                    ss << "], {" << "%f" << r_num << ", %f" << (r_num + 1) << ", %f"
                       << (r_num + 2) << ", %f" << (r_num + 3) << "};\n";
                } else if (vt == Type::f64x2()) {
                    int r_num = std::stoi(base_reg.substr(3));
                    ss << "    st.global.v2.f64 [" << get_reg(inst.operand(0));
                    if (off != 0) ss << " + " << off;
                    ss << "], {" << "%fd" << r_num << ", %fd" << (r_num + 1) << "};\n";
                } else {
                    throw std::runtime_error(
                        "PtxTarget: unsupported vstore vector type (only f32x4/f64x2 are supported)");
                }
                break;
            }

            case Opcode::load_indexed: {
                require(inst.operand(0) != nullptr && inst.operand(1) != nullptr,
                        "load_indexed missing operand");
                Type t = inst.type();
                uint8_t scale = inst.scale();
                int32_t off = inst.offset();
                std::string addr_reg = "%rd" + std::to_string(num_u64_++);
                std::string idx_reg = get_reg(inst.operand(1));

                if (inst.operand(1)->type() == Type::i32()) {
                    std::string wide_idx = "%rd" + std::to_string(num_u64_++);
                    ss << "    cvt.s64.s32 " << wide_idx << ", " << idx_reg << ";\n";
                    idx_reg = wide_idx;
                }

                if (scale > 1) {
                    std::string scaled_reg = "%rd" + std::to_string(num_u64_++);
                    uint32_t shift = (scale == 8) ? 3 : ((scale == 4) ? 2 : 1);
                    ss << "    shl.b64 " << scaled_reg << ", " << idx_reg << ", " << shift << ";\n";
                    ss << "    add.s64 " << addr_reg << ", " << get_reg(inst.operand(0))
                       << ", " << scaled_reg << ";\n";
                } else {
                    ss << "    add.s64 " << addr_reg << ", " << get_reg(inst.operand(0))
                       << ", " << idx_reg << ";\n";
                }

                std::string sfx = ptx_type_suffix(t);
                if (off != 0) {
                    ss << "    ld.global." << sfx << " " << get_reg(&inst) << ", [" << addr_reg
                       << " + " << off << "];\n";
                } else {
                    ss << "    ld.global." << sfx << " " << get_reg(&inst) << ", [" << addr_reg
                       << "];\n";
                }
                break;
            }

            case Opcode::store_indexed: {
                require(inst.operand(0) != nullptr && inst.operand(1) != nullptr &&
                            inst.operand(2) != nullptr,
                        "store_indexed missing operand");
                Type t = inst.operand(2)->type();
                uint8_t scale = inst.scale();
                int32_t off = inst.offset();
                std::string addr_reg = "%rd" + std::to_string(num_u64_++);
                std::string idx_reg = get_reg(inst.operand(1));

                if (inst.operand(1)->type() == Type::i32()) {
                    std::string wide_idx = "%rd" + std::to_string(num_u64_++);
                    ss << "    cvt.s64.s32 " << wide_idx << ", " << idx_reg << ";\n";
                    idx_reg = wide_idx;
                }

                if (scale > 1) {
                    std::string scaled_reg = "%rd" + std::to_string(num_u64_++);
                    uint32_t shift = (scale == 8) ? 3 : ((scale == 4) ? 2 : 1);
                    ss << "    shl.b64 " << scaled_reg << ", " << idx_reg << ", " << shift << ";\n";
                    ss << "    add.s64 " << addr_reg << ", " << get_reg(inst.operand(0))
                       << ", " << scaled_reg << ";\n";
                } else {
                    ss << "    add.s64 " << addr_reg << ", " << get_reg(inst.operand(0))
                       << ", " << idx_reg << ";\n";
                }

                std::string sfx = ptx_type_suffix(t);
                if (off != 0) {
                    ss << "    st.global." << sfx << " [" << addr_reg << " + " << off << "], "
                       << get_reg(inst.operand(2)) << ";\n";
                } else {
                    ss << "    st.global." << sfx << " [" << addr_reg << "], "
                       << get_reg(inst.operand(2)) << ";\n";
                }
                break;
            }

            case Opcode::call: {
                std::string_view callee = inst.symbol();
                if (callee == "ptx_tid_x") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %tid.x;\n";
                } else if (callee == "ptx_tid_y") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %tid.y;\n";
                } else if (callee == "ptx_tid_z") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %tid.z;\n";
                } else if (callee == "ptx_ctaid_x") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %ctaid.x;\n";
                } else if (callee == "ptx_ctaid_y") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %ctaid.y;\n";
                } else if (callee == "ptx_ctaid_z") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %ctaid.z;\n";
                } else if (callee == "ptx_ntid_x") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %ntid.x;\n";
                } else if (callee == "ptx_ntid_y") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %ntid.y;\n";
                } else if (callee == "ptx_ntid_z") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %ntid.z;\n";
                } else if (callee == "ptx_global_tid_x" || callee == "ptx_global_id_x") {
                    std::string r_tid = "%r" + std::to_string(num_u32_++);
                    std::string r_ctaid = "%r" + std::to_string(num_u32_++);
                    std::string r_ntid = "%r" + std::to_string(num_u32_++);
                    ss << "    mov.u32 " << r_tid << ", %tid.x;\n";
                    ss << "    mov.u32 " << r_ctaid << ", %ctaid.x;\n";
                    ss << "    mov.u32 " << r_ntid << ", %ntid.x;\n";
                    ss << "    mad.lo.s32 " << get_reg(&inst) << ", " << r_ctaid << ", " << r_ntid
                       << ", " << r_tid << ";\n";
                } else if (callee == "rsqrtf" || callee == "rsqrt" || callee == "ptx_rsqrt") {
                    ss << "    rsqrt.approx.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else if (callee == "sqrtf" || callee == "sqrt" || callee == "ptx_sqrt") {
                    ss << "    sqrt.approx.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else if (callee == "sinf" || callee == "sin" || callee == "ptx_sin") {
                    ss << "    sin.approx.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else if (callee == "cosf" || callee == "cos" || callee == "ptx_cos") {
                    ss << "    cos.approx.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else if (callee == "ex2f" || callee == "ex2" || callee == "ptx_ex2") {
                    ss << "    ex2.approx.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else if (callee == "expf" || callee == "exp" || callee == "ptx_exp") {
                    // e^x = 2^(x * log2(e)); log2(e) = 1.44269504f = 0f3FB8AA3B
                    std::string f_scaled = "%f" + std::to_string(num_f32_++);
                    ss << "    mul.f32 " << f_scaled << ", " << get_reg(inst.operand(0)) << ", 0f3FB8AA3B;\n";
                    ss << "    ex2.approx.f32 " << get_reg(&inst) << ", " << f_scaled << ";\n";
                } else if (callee == "i32_to_f32") {
                    ss << "    cvt.rn.f32.s32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else if (callee == "bar.sync" || callee == "ptx_sync") {
                    ss << "    bar.sync 0;\n";
                } else if (callee == "ptx_shfl_down_sync_f32" || callee == "shfl_down_sync_f32") {
                    ss << "    shfl.sync.down.b32 " << get_reg(&inst) << ", " << get_reg(inst.operand(1))
                       << ", " << get_reg(inst.operand(2)) << ", 0x1f, " << get_reg(inst.operand(0)) << ";\n";
                } else if (callee == "ptx_shfl_down_f32") {
                    ss << "    shfl.sync.down.b32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ", 0x1f, 0xffffffff;\n";
                } else if (callee == "ptx_laneid" || callee == "ptx_lane_id") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %laneid;\n";
                } else if (callee == "ptx_warpid" || callee == "ptx_warp_id") {
                    ss << "    mov.u32 " << get_reg(&inst) << ", %warpid;\n";
                } else {
                    // Regular function call
                    if (inst.type().is_void() || !inst.result()) {
                        ss << "    call " << callee << ", (";
                    } else {
                        ss << "    call (" << get_reg(&inst) << "), " << callee << ", (";
                    }
                    for (size_t i = 0; i < inst.operand_count(); ++i) {
                        if (i > 0) ss << ", ";
                        ss << get_reg(inst.operand(i));
                    }
                    ss << ");\n";
                }
                break;
            }

            case Opcode::sext_i64: {
                ss << "    cvt.s64.s32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                break;
            }
            case Opcode::zext_i64: {
                ss << "    cvt.u64.u32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                break;
            }
            case Opcode::trunc_i32: {
                ss << "    cvt.u32.u64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                break;
            }
            case Opcode::sitofp_f64_i32: {
                if (inst.type() == Type::f32()) {
                    ss << "    cvt.rn.f32.s32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else {
                    ss << "    cvt.rn.f64.s32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                }
                break;
            }
            case Opcode::fptosi_i32: {
                if (inst.operand(0)->type() == Type::f32()) {
                    ss << "    cvt.rzi.s32.f32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                } else {
                    ss << "    cvt.rzi.s32.f64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                }
                break;
            }

            case Opcode::unreachable: {
                ss << "    trap;\n";
                break;
            }

            case Opcode::not_: {
                Type t = inst.type();
                std::string sfx = is_64bit_int(t) ? "b64" : "b32";
                ss << "    not." << sfx << " " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                break;
            }

            case Opcode::smod: {
                Type t = inst.type();
                std::string sfx = (t == Type::i64()) ? "s64" : "s32";
                ss << "    rem." << sfx << " " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                   << ", " << get_reg(inst.operand(1)) << ";\n";
                break;
            }

            case Opcode::umod: {
                Type t = inst.type();
                std::string sfx = (t == Type::i64()) ? "u64" : "u32";
                ss << "    rem." << sfx << " " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                   << ", " << get_reg(inst.operand(1)) << ";\n";
                break;
            }

            case Opcode::bitcast_i64_f64:
            case Opcode::bitcast_f64_i64: {
                ss << "    mov.b64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0)) << ";\n";
                break;
            }

            case Opcode::safepoint:
            case Opcode::write_barrier:
            case Opcode::resume_point:
            case Opcode::osr_entry:
                // No-op in GPU kernel execution
                break;

            case Opcode::br: {
                const BasicBlock* target = inst.branch_target().block;
                require(target != nullptr, "br missing target");
                emit_phi_copies(target, inst.branch_target().args, ss);
                ss << "    bra " << block_label(target) << ";\n";
                break;
            }
            case Opcode::br_if: {
                require(inst.operand(0) != nullptr, "br_if missing condition");
                std::string p_cond = materialize_pred(inst.operand(0), ss);

                const BasicBlock* true_target = inst.true_target().block;
                const BasicBlock* false_target = inst.false_target().block;
                require(true_target != nullptr && false_target != nullptr, "br_if missing target");

                if (!inst.true_target().args.empty()) {
                    for (size_t i = 0; i < true_target->param_count() && i < inst.true_target().args.size(); ++i) {
                        const Value* param = true_target->param(i);
                        const Value* arg = inst.true_target().args[i];
                        if (param && arg && param != arg) {
                            Type t = param->type();
                            std::string sfx = (t == Type::f32()) ? "f32"
                                            : (t == Type::f64()) ? "f64"
                                            : (is_64bit_int(t)) ? "b64"
                                            : "b32";
                            ss << "    @" << p_cond << " mov." << sfx << " " << get_reg(param)
                               << ", " << get_reg(arg) << ";\n";
                        }
                    }
                }
                ss << "    @" << p_cond << " bra " << block_label(true_target) << ";\n";

                if (!inst.false_target().args.empty()) {
                    emit_phi_copies(false_target, inst.false_target().args, ss);
                }
                ss << "    bra " << block_label(false_target) << ";\n";
                break;
            }
            case Opcode::ret: {
                ss << "    ret;\n";
                break;
            }

            default:
                throw std::runtime_error("PtxTarget: unsupported opcode in PTX emission: " + std::string(opcode_name(inst.opcode())));
        }
    }

    void emit_phi_copies(const BasicBlock* target, const std::vector<Value*>& args, std::ostringstream& ss) {
        if (!target || args.empty() || target->params().empty()) return;
        for (size_t i = 0; i < target->params().size() && i < args.size(); ++i) {
            const Value* param = target->param(i);
            const Value* arg = args[i];
            if (param && arg && param != arg) {
                Type t = param->type();
                std::string sfx = (t == Type::f32()) ? "f32"
                                : (t == Type::f64()) ? "f64"
                                : (is_64bit_int(t)) ? "b64"
                                : "b32";
                ss << "    mov." << sfx << " " << get_reg(param) << ", " << get_reg(arg) << ";\n";
            }
        }
    }
};

} // anonymous namespace

std::string PtxTarget::emit_function(const Function& fn, const PtxOptions& opts) {
    uint32_t major = opts.ptx_version_major;
    uint32_t minor = opts.ptx_version_minor;
    if (opts.sm_arch == "sm_89" && (major < 7 || (major == 7 && minor < 8))) {
        major = 7;
        minor = 8;
    }
    std::ostringstream ss;
    ss << "// Generated by Brass PTX Target Emitter\n"
       << ".version " << major << "." << minor << "\n"
       << ".target " << opts.sm_arch << "\n"
       << ".address_size 64\n\n";

    PtxEmitter emitter(fn, opts);
    ss << emitter.emit();
    return ss.str();
}

std::string PtxTarget::emit_module(const Module& mod, const PtxOptions& opts) {
    uint32_t major = opts.ptx_version_major;
    uint32_t minor = opts.ptx_version_minor;
    if (opts.sm_arch == "sm_89" && (major < 7 || (major == 7 && minor < 8))) {
        major = 7;
        minor = 8;
    }
    std::ostringstream ss;
    ss << "// Generated by Brass PTX Target Emitter\n"
       << ".version " << major << "." << minor << "\n"
       << ".target " << opts.sm_arch << "\n"
       << ".address_size 64\n\n";

    for (const auto& fn_ptr : mod.functions()) {
        PtxEmitter emitter(*fn_ptr, opts);
        ss << emitter.emit() << "\n";
    }

    return ss.str();
}

} // namespace brass::target
