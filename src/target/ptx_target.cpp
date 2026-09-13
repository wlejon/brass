#include <brass/target/ptx_target.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>

#include <sstream>
#include <iomanip>
#include <cstring>
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

class PtxEmitter {
public:
    PtxEmitter(const Function& fn, const PtxOptions& opts)
        : fn_(fn), opts_(opts) {}

    std::string emit() {
        assign_registers();

        std::ostringstream ss;
        // Function declaration
        ss << ".visible .entry " << fn_.name() << "(\n";
        for (size_t i = 0; i < fn_.param_types().size(); ++i) {
            ss << "    .param " << param_type_name(fn_.param_types()[i])
               << " param_" << i;
            if (i + 1 < fn_.param_types().size()) ss << ",\n";
            else ss << "\n";
        }
        ss << ")\n{\n";

        // Register allocations
        if (num_pred_ > 0) ss << "    .reg .pred %p<" << num_pred_ << ">;\n";
        if (num_u32_ > 0)  ss << "    .reg .b32 %r<" << num_u32_ << ">;\n";
        if (num_u64_ > 0)  ss << "    .reg .b64 %rd<" << num_u64_ << ">;\n";
        if (num_f32_ > 0)  ss << "    .reg .f32 %f<" << num_f32_ << ">;\n";
        if (num_f64_ > 0)  ss << "    .reg .f64 %fd<" << num_f64_ << ">;\n";
        ss << "\n";

        // Load parameters
        if (const auto* entry = fn_.entry_block()) {
            for (size_t i = 0; i < entry->param_count(); ++i) {
                const Value* param_val = entry->param(i);
                std::string reg = get_reg(param_val);
                ss << "    ld.param." << ptx_type_suffix(param_val->type()) << " " << reg
                   << ", [param_" << i << "];\n";
            }
        }
        ss << "\n";

        // Emit blocks
        for (size_t b_idx = 0; b_idx < fn_.blocks().size(); ++b_idx) {
            const BasicBlock* bb = fn_.blocks()[b_idx];
            ss << block_label(bb) << ":\n";

            for (const auto* inst_ptr : *bb) {
                emit_instruction(*inst_ptr, ss);
            }
        }

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
                reg_map_[param] = alloc_reg(param->type());
            }
        }

        for (const auto* bb : fn_.blocks()) {
            for (const auto* param : bb->params()) {
                if (reg_map_.find(param) == reg_map_.end()) {
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
                    ss << "    div.s32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                       << ", " << get_reg(inst.operand(1)) << ";\n";
                } else {
                    ss << "    div.s64 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
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
                ss << "    and.b32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                   << ", " << get_reg(inst.operand(1)) << ";\n";
                break;
            }
            case Opcode::or_: {
                ss << "    or.b32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                   << ", " << get_reg(inst.operand(1)) << ";\n";
                break;
            }
            case Opcode::xor_: {
                ss << "    xor.b32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                   << ", " << get_reg(inst.operand(1)) << ";\n";
                break;
            }
            case Opcode::shl: {
                ss << "    shl.b32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                   << ", " << get_reg(inst.operand(1)) << ";\n";
                break;
            }
            case Opcode::lshr: {
                ss << "    shr.u32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                   << ", " << get_reg(inst.operand(1)) << ";\n";
                break;
            }
            case Opcode::ashr: {
                ss << "    shr.s32 " << get_reg(&inst) << ", " << get_reg(inst.operand(0))
                   << ", " << get_reg(inst.operand(1)) << ";\n";
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

                Type opnd_t = inst.operand(0)->type();
                std::string type_s = "s32";
                if (opnd_t == Type::f32()) type_s = "f32";
                else if (opnd_t == Type::f64()) type_s = "f64";
                else if (opnd_t == Type::i64() || opnd_t == Type::ptr()) type_s = "s64";

                std::string p_reg = get_pred(&inst);
                ss << "    setp." << cmp << "." << type_s << " " << p_reg << ", "
                   << get_reg(inst.operand(0)) << ", " << get_reg(inst.operand(1)) << ";\n";

                // Materialize boolean int in result register
                ss << "    selp.u32 " << get_reg(&inst) << ", 1, 0, " << p_reg << ";\n";
                break;
            }

            case Opcode::select: {
                std::string p_cond = get_pred(inst.operand(0));
                Type t = inst.type();
                std::string sfx = (t == Type::f32()) ? "f32" : ((t == Type::f64()) ? "f64" : "b32");
                ss << "    selp." << sfx << " " << get_reg(&inst) << ", "
                   << get_reg(inst.operand(1)) << ", " << get_reg(inst.operand(2))
                   << ", " << p_cond << ";\n";
                break;
            }

            case Opcode::load: {
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

            case Opcode::load_indexed: {
                // operand(0) = ptr, operand(1) = index
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
                    ss << "    add.s64 " << addr_reg << ", " << get_reg(inst.operand(0)) << ", " << scaled_reg << ";\n";
                } else {
                    ss << "    add.s64 " << addr_reg << ", " << get_reg(inst.operand(0)) << ", " << idx_reg << ";\n";
                }

                std::string sfx = ptx_type_suffix(t);
                if (off != 0) {
                    ss << "    ld.global." << sfx << " " << get_reg(&inst) << ", [" << addr_reg << " + " << off << "];\n";
                } else {
                    ss << "    ld.global." << sfx << " " << get_reg(&inst) << ", [" << addr_reg << "];\n";
                }
                break;
            }

            case Opcode::store_indexed: {
                // operand(0) = ptr, operand(1) = index, operand(2) = value
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
                    ss << "    add.s64 " << addr_reg << ", " << get_reg(inst.operand(0)) << ", " << scaled_reg << ";\n";
                } else {
                    ss << "    add.s64 " << addr_reg << ", " << get_reg(inst.operand(0)) << ", " << idx_reg << ";\n";
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
                    ss << "    mad.lo.s32 " << get_reg(&inst) << ", " << r_ctaid << ", " << r_ntid << ", " << r_tid << ";\n";
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
                } else {
                    // Fallback to standard PTX call if regular function
                    ss << "    // call " << callee << "\n";
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

            case Opcode::br: {
                const BasicBlock* target = inst.branch_target().block;
                emit_phi_copies(target, inst.branch_target().args, ss);
                ss << "    bra " << block_label(target) << ";\n";
                break;
            }
            case Opcode::br_if: {
                const Value* cond = inst.operand(0);
                std::string p_cond = get_pred(cond);
                if (pred_map_.find(cond) == pred_map_.end()) {
                    // Condition is an integer value, test it: setp.ne.s32 %p_tmp, %r_cond, 0
                    p_cond = "%p" + std::to_string(num_pred_++);
                    ss << "    setp.ne.s32 " << p_cond << ", " << get_reg(cond) << ", 0;\n";
                }

                const BasicBlock* true_target = inst.true_target().block;
                const BasicBlock* false_target = inst.false_target().block;

                if (true_target && !inst.true_target().args.empty()) {
                    for (size_t i = 0; i < true_target->param_count() && i < inst.true_target().args.size(); ++i) {
                        const Value* param = true_target->param(i);
                        const Value* arg = inst.true_target().args[i];
                        if (param != arg) {
                            Type t = param->type();
                            std::string sfx = (t == Type::f32()) ? "f32" : ((t == Type::f64()) ? "f64" : ((t == Type::i64() || t == Type::ptr()) ? "b64" : "b32"));
                            ss << "    @" << p_cond << " mov." << sfx << " " << get_reg(param) << ", " << get_reg(arg) << ";\n";
                        }
                    }
                }
                ss << "    @" << p_cond << " bra " << block_label(true_target) << ";\n";

                if (false_target && !inst.false_target().args.empty()) {
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
                break;
        }
    }

    void emit_phi_copies(const BasicBlock* target, const std::vector<Value*>& args, std::ostringstream& ss) {
        if (!target || args.empty() || target->params().empty()) return;
        for (size_t i = 0; i < target->params().size() && i < args.size(); ++i) {
            const Value* param = target->param(i);
            const Value* arg = args[i];
            if (param != arg) {
                Type t = param->type();
                std::string sfx = (t == Type::f32()) ? "f32" : ((t == Type::f64()) ? "f64" : ((t == Type::i64() || t == Type::ptr()) ? "b64" : "b32"));
                ss << "    mov." << sfx << " " << get_reg(param) << ", " << get_reg(arg) << ";\n";
            }
        }
    }
};

} // anonymous namespace

std::string PtxTarget::emit_function(const Function& fn, const PtxOptions& opts) {
    std::ostringstream ss;
    ss << "// Generated by Brass PTX Target Emitter\n"
       << ".version " << opts.ptx_version_major << "." << opts.ptx_version_minor << "\n"
       << ".target " << opts.sm_arch << "\n"
       << ".address_size 64\n\n";

    PtxEmitter emitter(fn, opts);
    ss << emitter.emit();
    return ss.str();
}

std::string PtxTarget::emit_module(const Module& mod, const PtxOptions& opts) {
    std::ostringstream ss;
    ss << "// Generated by Brass PTX Target Emitter\n"
       << ".version " << opts.ptx_version_major << "." << opts.ptx_version_minor << "\n"
       << ".target " << opts.sm_arch << "\n"
       << ".address_size 64\n\n";

    for (const auto& fn_ptr : mod.functions()) {
        PtxEmitter emitter(*fn_ptr, opts);
        ss << emitter.emit() << "\n";
    }

    return ss.str();
}

} // namespace brass::target
