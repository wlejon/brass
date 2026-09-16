#include <brass/target/ptx/ptx_verifier.hpp>
#include <brass/target/ptx/ptx_printer.hpp>

#include <sstream>
#include <unordered_set>

namespace brass::ptx {

std::string to_string(const Diagnostic& d) {
    std::ostringstream ss;
    ss << d.function;
    if (!d.block.empty()) ss << ":" << d.block;
    if (d.inst_index != Diagnostic::kNoInst) ss << "[" << d.inst_index << "]";
    if (!d.inst_text.empty()) ss << " '" << d.inst_text << "'";
    ss << ": " << d.message;
    return ss.str();
}

std::string format_diagnostics(const std::vector<Diagnostic>& diags) {
    std::string out;
    for (const auto& d : diags) {
        out += to_string(d);
        out += '\n';
    }
    return out;
}

namespace {

// Register/type compatibility. Stricter than ptxas: .f32 needs an F32
// register and .u32/.s32 need B32, but bit types accept either register file
// of the same width (shfl.b32 of an f32 value, mov.b64 bitcasts).
bool reg_compatible(RegClass rc, Type t) {
    switch (t) {
        case Type::none: return false;
        case Type::pred: return rc == RegClass::Pred;
        case Type::f32:  return rc == RegClass::F32;
        case Type::f64:  return rc == RegClass::F64;
        case Type::b32:  return rc == RegClass::B32 || rc == RegClass::F32;
        case Type::b64:  return rc == RegClass::B64 || rc == RegClass::F64;
        case Type::u64:
        case Type::s64:  return rc == RegClass::B64;
        default:         return rc == RegClass::B32; // 8/16/32-bit ints, f16
    }
}

std::string required_class_text(Type t) {
    switch (t) {
        case Type::b32: return "a register of class b32 or f32";
        case Type::b64: return "a register of class b64 or f64";
        default: return "a register of class " + std::string(to_string(reg_class_for(t)));
    }
}

bool is_unary_math(Opcode op) {
    switch (op) {
        case Opcode::rsqrt: case Opcode::sqrt: case Opcode::sin: case Opcode::cos:
        case Opcode::ex2: case Opcode::lg2: case Opcode::rcp:
            return true;
        default:
            return false;
    }
}

bool is_binary_alu(Opcode op) {
    switch (op) {
        case Opcode::add: case Opcode::sub: case Opcode::mul: case Opcode::div:
        case Opcode::rem: case Opcode::min: case Opcode::max:
        case Opcode::and_: case Opcode::or_: case Opcode::xor_:
            return true;
        default:
            return false;
    }
}

bool takes_type(Opcode op) {
    switch (op) {
        case Opcode::bra: case Opcode::ret: case Opcode::call:
        case Opcode::trap: case Opcode::exit: case Opcode::bar:
            return false;
        default:
            return true;
    }
}

class Verifier {
public:
    explicit Verifier(const Function& fn) : fn_(fn) {}

    std::vector<Diagnostic> run() {
        verify_function();
        for (const auto& block : fn_.blocks) {
            block_ = block.get();
            for (size_t i = 0; i < block->insts.size(); ++i) {
                index_ = i;
                inst_ = &block->insts[i];
                verify_inst();
            }
        }
        return std::move(diags_);
    }

private:
    const Function& fn_;
    const Block* block_ = nullptr;
    size_t index_ = 0;
    const Inst* inst_ = nullptr;
    std::vector<Diagnostic> diags_;
    std::unordered_set<std::string> labels_;

    void fn_error(std::string msg) {
        Diagnostic d;
        d.function = fn_.name;
        d.message = std::move(msg);
        diags_.push_back(std::move(d));
    }

    void error(std::string msg) {
        Diagnostic d;
        d.function = fn_.name;
        d.block = block_->label;
        d.inst_index = index_;
        d.inst_text = to_string(*inst_);
        d.message = std::move(msg);
        diags_.push_back(std::move(d));
    }

    // ---- function level ---------------------------------------------------

    void verify_function() {
        if (fn_.name.empty()) fn_error("function has no name");
        if (fn_.blocks.empty()) fn_error("function has no blocks");

        std::unordered_set<std::string> names;
        for (const auto& p : fn_.params) {
            if (p.name.empty()) fn_error("param has an empty name");
            if (p.type == Type::none || p.type == Type::pred)
                fn_error("param '" + p.name + "' has invalid type ." + std::string(to_string(p.type)));
            if (!names.insert(p.name).second) fn_error("duplicate param name '" + p.name + "'");
        }
        std::unordered_set<std::string> shared_names;
        for (const auto& s : fn_.shared) {
            if (s.name.empty()) fn_error("shared array has an empty name");
            if (s.count == 0) fn_error("shared array '" + s.name + "' has zero elements");
            if (s.type == Type::none || s.type == Type::pred)
                fn_error("shared array '" + s.name + "' has invalid type ." + std::string(to_string(s.type)));
            if (s.align == 0 || (s.align & (s.align - 1)) != 0)
                fn_error("shared array '" + s.name + "' alignment must be a power of two");
            if (!shared_names.insert(s.name).second) fn_error("duplicate shared array name '" + s.name + "'");
            if (names.count(s.name)) fn_error("shared array '" + s.name + "' collides with a param name");
        }
        for (const auto& b : fn_.blocks) {
            if (b->label.empty()) fn_error("block has an empty label");
            if (!labels_.insert(b->label).second) fn_error("duplicate block label '" + b->label + "'");
        }
    }

    // ---- operand helpers ---------------------------------------------------

    bool check_reg_range(Reg r) {
        if (!r.valid()) {
            error(std::string("uses an invalid ") + std::string(to_string(r.cls)) + " register");
            return false;
        }
        uint32_t declared = fn_.reg_count(r.cls);
        if (r.index >= declared) {
            error(to_string(r) + " is out of range (declared " + std::string(reg_prefix(r.cls)) +
                  "<" + std::to_string(declared) + ">)");
            return false;
        }
        return true;
    }

    // Walk every register mentioned by the operand and range-check it.
    void check_operand_regs(const Operand& o) {
        switch (o.kind) {
            case OperandKind::Reg: check_reg_range(o.reg_val); break;
            case OperandKind::Addr: if (o.addr_base.valid()) check_reg_range(o.addr_base); break;
            case OperandKind::Vector: for (Reg r : o.elems) check_reg_range(r); break;
            default: break;
        }
    }

    std::string slot(const char* which, size_t i) {
        return std::string(which) + " " + std::to_string(i);
    }

    // Register operand whose class must match `t`.
    void expect_reg(const Operand& o, Type t, const std::string& what) {
        if (!o.is_reg()) {
            error(what + " must be a register, got " + std::string(to_string(o.kind)) + " '" + to_string(o) + "'");
            return;
        }
        if (!reg_compatible(o.reg_val.cls, t)) {
            error(what + " is " + to_string(o.reg_val) + " (" + std::string(to_string(o.reg_val.cls)) +
                  ") but ." + std::string(to_string(t)) + " requires " + required_class_text(t));
        }
    }

    void expect_reg_class(const Operand& o, RegClass rc, const std::string& what) {
        if (!o.is_reg()) {
            error(what + " must be a " + std::string(to_string(rc)) + " register, got " +
                  std::string(to_string(o.kind)) + " '" + to_string(o) + "'");
            return;
        }
        if (o.reg_val.cls != rc) {
            error(what + " is " + to_string(o.reg_val) + " but must be a " + std::string(to_string(rc)) + " register");
        }
    }

    // Register or immediate compatible with `t`.
    void expect_value(const Operand& o, Type t, const std::string& what) {
        switch (o.kind) {
            case OperandKind::Reg:
                expect_reg(o, t, what);
                return;
            case OperandKind::ImmInt:
                if (is_float(t)) error(what + " is an integer immediate but ." + std::string(to_string(t)) + " requires a float immediate (0f.../0d...)");
                if (t == Type::pred) error(what + " cannot be an immediate for .pred");
                return;
            case OperandKind::ImmFloat:
                if (!is_float(t)) {
                    error(what + " is a float immediate but ." + std::string(to_string(t)) + " requires an integer immediate");
                } else if (t == Type::f32 && !o.imm_is_f32) {
                    error(what + " is an f64 immediate on an .f32 instruction");
                } else if (t == Type::f64 && o.imm_is_f32) {
                    error(what + " is an f32 immediate on an .f64 instruction");
                }
                return;
            default:
                error(what + " must be a register or immediate, got " + std::string(to_string(o.kind)) + " '" + to_string(o) + "'");
                return;
        }
    }

    // B32 register or integer immediate (shift counts, shfl lanes, bar ids).
    void expect_b32_or_imm(const Operand& o, const std::string& what) {
        if (o.is_imm_int()) return;
        if (!o.is_reg()) {
            error(what + " must be a b32 register or integer immediate, got " +
                  std::string(to_string(o.kind)) + " '" + to_string(o) + "'");
            return;
        }
        if (o.reg_val.cls != RegClass::B32) {
            error(what + " is " + to_string(o.reg_val) + " but must be a b32 register (PTX shift/lane counts are 32-bit)");
        }
    }

    // Scalar register or vector tuple according to the instruction's .vN.
    void expect_data(const Operand& o, Type t, const std::string& what, bool allow_imm) {
        auto width = static_cast<size_t>(inst_->vec_width);
        if (width == 1) {
            if (o.is_vec()) {
                error(what + " is a vector tuple but the instruction has no .v2/.v4 modifier");
                return;
            }
            if (allow_imm) expect_value(o, t, what);
            else expect_reg(o, t, what);
            return;
        }
        if (!o.is_vec()) {
            error(what + " must be a {..} vector tuple for ." + std::string(to_string(inst_->vec_width)));
            return;
        }
        if (o.elems.size() != width) {
            error(what + " has " + std::to_string(o.elems.size()) + " elements but ." +
                  std::string(to_string(inst_->vec_width)) + " requires " + std::to_string(width));
        }
        for (size_t i = 0; i < o.elems.size(); ++i) {
            if (!reg_compatible(o.elems[i].cls, t)) {
                error(what + " element " + std::to_string(i) + " is " + to_string(o.elems[i]) +
                      " but ." + std::string(to_string(t)) + " requires " + required_class_text(t));
            }
        }
    }

    void expect_addr(const Operand& o, const std::string& what) {
        if (!o.is_addr()) {
            error(what + " must be an address [reg + disp] or [symbol + disp], got " +
                  std::string(to_string(o.kind)) + " '" + to_string(o) + "'");
            return;
        }
        StateSpace space = inst_->state_space;
        if (o.addr_base.valid()) {
            RegClass rc = o.addr_base.cls;
            bool ok = (space == StateSpace::shared) ? (rc == RegClass::B32 || rc == RegClass::B64)
                                                    : (rc == RegClass::B64);
            if (!ok) {
                error(what + " base " + to_string(o.addr_base) + " must be a " +
                      (space == StateSpace::shared ? "b32 or b64" : "b64") + " register");
            }
            return;
        }
        if (o.addr_symbol.empty()) {
            error(what + " has neither a base register nor a symbol");
            return;
        }
        if (space == StateSpace::param) {
            if (!fn_.find_param(o.addr_symbol))
                error(what + " names '" + o.addr_symbol + "' which is not a declared param");
        } else if (space == StateSpace::shared) {
            if (!fn_.find_shared(o.addr_symbol))
                error(what + " names '" + o.addr_symbol + "' which is not a declared .shared array");
        } else {
            error(what + " uses symbol '" + o.addr_symbol + "' but only .param and .shared addressing may use symbols");
        }
    }

    bool check_arity(size_t ndst, size_t nsrc_min, size_t nsrc_max) {
        bool ok = true;
        if (inst_->dsts.size() != ndst) {
            error(std::string(to_string(inst_->op)) + " takes " + std::to_string(ndst) +
                  " destination(s), got " + std::to_string(inst_->dsts.size()));
            ok = false;
        }
        size_t n = inst_->srcs.size();
        if (n < nsrc_min || n > nsrc_max) {
            std::string want = (nsrc_min == nsrc_max) ? std::to_string(nsrc_min)
                                                      : std::to_string(nsrc_min) + ".." + std::to_string(nsrc_max);
            error(std::string(to_string(inst_->op)) + " takes " + want + " source(s), got " + std::to_string(n));
            ok = false;
        }
        return ok;
    }

    // ---- instruction level -------------------------------------------------

    void verify_inst() {
        const Inst& I = *inst_;

        if (I.has_guard) {
            if (I.guard_reg.cls != RegClass::Pred)
                error("guard " + to_string(I.guard_reg) + " is not a pred register");
            check_reg_range(I.guard_reg);
        }
        for (const auto& o : I.dsts) check_operand_regs(o);
        for (const auto& o : I.srcs) check_operand_regs(o);
        for (const auto& o : I.dsts) {
            if (o.is_imm() || o.is_label() || o.is_special() || o.is_symbol() || o.is_param())
                error("destination '" + to_string(o) + "' is not a register");
        }

        if (takes_type(I.op)) {
            if (I.type == Type::none) { error("missing type suffix"); return; }
        } else if (I.type != Type::none) {
            error(std::string(to_string(I.op)) + " does not take a type suffix");
        }
        if (I.op != Opcode::cvt && I.src_type != Type::none)
            error("only cvt takes a second (source) type suffix");
        if (I.vec_width != VecWidth::v1 && I.op != Opcode::ld && I.op != Opcode::st)
            error("only ld/st take a .v2/.v4 modifier");
        if (I.mul_mode != MulMode::none && I.op != Opcode::mul && I.op != Opcode::mad)
            error("only mul/mad take .lo/.hi/.wide");
        if (I.cmp_op != CmpOp::none && I.op != Opcode::setp)
            error("only setp takes a comparison operator");
        if (I.state_space != StateSpace::none && I.op != Opcode::ld && I.op != Opcode::st && I.op != Opcode::atom)
            error("only ld/st/atom take a state space");

        switch (I.op) {
            case Opcode::ld:    verify_ld(); break;
            case Opcode::st:    verify_st(); break;
            case Opcode::mov:   verify_mov(); break;
            case Opcode::cvt:   verify_cvt(); break;
            case Opcode::mad:   verify_mad(); break;
            case Opcode::fma:   verify_fma(); break;
            case Opcode::neg: case Opcode::abs: case Opcode::not_:
                if (check_arity(1, 1, 1)) { expect_reg(I.dsts[0], I.type, "destination"); expect_value(I.srcs[0], I.type, "source 0"); }
                break;
            case Opcode::shl: case Opcode::shr: verify_shift(); break;
            case Opcode::setp:  verify_setp(); break;
            case Opcode::selp:  verify_selp(); break;
            case Opcode::bra:   verify_bra(); break;
            case Opcode::ret:
                if (!I.srcs.empty() || !I.dsts.empty())
                    error(fn_.is_entry ? "ret inside .entry takes no operand (results go through pointer params)"
                                       : "ret takes no operand");
                break;
            case Opcode::call:  verify_call(); break;
            case Opcode::trap: case Opcode::exit: check_arity(0, 0, 0); break;
            case Opcode::shfl:  verify_shfl(); break;
            case Opcode::bar:   verify_bar(); break;
            case Opcode::atom:  verify_atom(); break;
            default:
                if (is_binary_alu(I.op)) verify_binary();
                else if (is_unary_math(I.op)) verify_unary_math();
                else error("unhandled opcode in verifier");
                break;
        }
    }

    void verify_ld() {
        const Inst& I = *inst_;
        if (!check_arity(1, 1, 1)) return;
        expect_data(I.dsts[0], I.type, "destination", false);
        const Operand& src = I.srcs[0];
        if (I.state_space == StateSpace::param) {
            const Param* p = nullptr;
            if (src.is_param()) {
                p = fn_.find_param(src.name);
                if (!p) error("ld.param source names '" + src.name + "' which is not a declared param");
            } else if (src.is_addr() && !src.addr_base.valid() && !src.addr_symbol.empty()) {
                p = fn_.find_param(src.addr_symbol);
                if (!p) error("ld.param source names '" + src.addr_symbol + "' which is not a declared param");
            } else {
                error("ld.param source must be a [param] operand, got " + std::string(to_string(src.kind)) + " '" + to_string(src) + "'");
            }
            if (p && reg_class_for(p->type) != reg_class_for(I.type)) {
                error("ld.param." + std::string(to_string(I.type)) + " does not match param '" + p->name +
                      "' declared as ." + std::string(to_string(p->type)));
            }
            return;
        }
        if (I.state_space == StateSpace::none) error("ld requires a state space (.global/.shared/.param/.local)");
        if (src.is_param()) { error("[param] operands are only valid with ld.param"); return; }
        expect_addr(src, "address");
    }

    void verify_st() {
        const Inst& I = *inst_;
        if (!check_arity(0, 2, 2)) return;
        if (I.state_space == StateSpace::none) error("st requires a state space (.global/.shared/.local)");
        if (I.state_space == StateSpace::param) error("st.param is not valid inside a kernel");
        expect_addr(I.srcs[0], "address");
        expect_data(I.srcs[1], I.type, "value", true);
    }

    void verify_mov() {
        const Inst& I = *inst_;
        if (!check_arity(1, 1, 1)) return;
        expect_reg(I.dsts[0], I.type, "destination");
        const Operand& s = I.srcs[0];
        if (s.is_special()) {
            if (!(is_integer(I.type) || is_bits(I.type)))
                error("special register source requires an integer type");
            if (I.dsts[0].is_reg() && I.dsts[0].reg_val.cls != reg_class_for(s.special_reg))
                error("special register " + to_string(s) + " must be moved into a " +
                      std::string(to_string(reg_class_for(s.special_reg))) + " register");
            return;
        }
        if (s.is_symbol()) {
            if (!fn_.find_shared(s.name))
                error("symbol '" + s.name + "' is not a declared .shared array");
            if (!(is_integer(I.type) || is_bits(I.type)))
                error("address-of-symbol move requires an integer type");
            return;
        }
        expect_value(s, I.type, "source 0");
    }

    void verify_cvt() {
        const Inst& I = *inst_;
        if (I.src_type == Type::none) { error("cvt requires a source type (cvt.dtype.atype)"); return; }
        if (!check_arity(1, 1, 1)) return;
        expect_reg(I.dsts[0], I.type, "destination");
        expect_reg(I.srcs[0], I.src_type, "source 0");
        if (I.type == I.src_type && I.rounding == Rounding::none && !I.is_sat && !I.is_ftz)
            error("cvt between identical types needs a rounding/sat/ftz modifier");
        if (is_float(I.type) && is_integer(I.src_type) && I.rounding == Rounding::none)
            error("integer -> float cvt requires a rounding modifier (.rn/.rz/.rm/.rp)");
        if (is_integer(I.type) && is_float(I.src_type) && I.rounding == Rounding::none)
            error("float -> integer cvt requires an integer rounding modifier (.rni/.rzi/.rmi/.rpi)");
        if (I.type == Type::f32 && I.src_type == Type::f64 && I.rounding == Rounding::none)
            error("f64 -> f32 cvt requires a rounding modifier");
    }

    void verify_binary() {
        const Inst& I = *inst_;
        if (!check_arity(1, 2, 2)) return;
        Type dst_t = I.type;
        if (I.op == Opcode::mul) {
            if (is_float(I.type) && I.mul_mode != MulMode::none) error(".lo/.hi/.wide are only valid for integer mul");
            if (I.mul_mode == MulMode::wide) {
                dst_t = wide_type_for(I.type);
                if (dst_t == Type::none) { error("mul.wide is only valid for 16/32-bit integer types"); return; }
            } else if (is_integer(I.type) && I.mul_mode == MulMode::none) {
                error("integer mul requires .lo, .hi or .wide");
            }
        }
        if ((I.op == Opcode::and_ || I.op == Opcode::or_ || I.op == Opcode::xor_) && !is_bits(I.type) && I.type != Type::pred)
            error(std::string(to_string(I.op)) + " requires a bit type (.b16/.b32/.b64) or .pred");
        if (I.op == Opcode::div && is_float(I.type) && !I.is_approx && I.rounding == Rounding::none)
            error("float div requires .approx or a rounding modifier (.rn/.rz/.rm/.rp)");
        if (I.op == Opcode::rem && !is_integer(I.type))
            error("rem requires an integer type");
        expect_reg(I.dsts[0], dst_t, "destination");
        expect_value(I.srcs[0], I.type, "source 0");
        expect_value(I.srcs[1], I.type, "source 1");
    }

    void verify_mad() {
        const Inst& I = *inst_;
        if (!check_arity(1, 3, 3)) return;
        Type acc_t = I.type;
        if (is_float(I.type)) {
            if (I.mul_mode != MulMode::none) error(".lo/.hi/.wide are only valid for integer mad");
        } else if (I.mul_mode == MulMode::none) {
            error("integer mad requires .lo, .hi or .wide");
        } else if (I.mul_mode == MulMode::wide) {
            acc_t = wide_type_for(I.type);
            if (acc_t == Type::none) { error("mad.wide is only valid for 16/32-bit integer types"); return; }
        }
        expect_reg(I.dsts[0], acc_t, "destination");
        expect_value(I.srcs[0], I.type, "source 0");
        expect_value(I.srcs[1], I.type, "source 1");
        expect_value(I.srcs[2], acc_t, "source 2");
    }

    void verify_fma() {
        const Inst& I = *inst_;
        if (!is_float(I.type)) error("fma requires a float type");
        if (I.rounding == Rounding::none) error("fma requires a rounding modifier (.rn/.rz/.rm/.rp)");
        if (!check_arity(1, 3, 3)) return;
        expect_reg(I.dsts[0], I.type, "destination");
        for (size_t i = 0; i < 3; ++i) expect_value(I.srcs[i], I.type, slot("source", i));
    }

    void verify_shift() {
        const Inst& I = *inst_;
        if (I.op == Opcode::shl && !is_bits(I.type)) error("shl requires a bit type (.b16/.b32/.b64)");
        if (I.op == Opcode::shr && !(is_bits(I.type) || is_integer(I.type))) error("shr requires an integer or bit type");
        if (!check_arity(1, 2, 2)) return;
        expect_reg(I.dsts[0], I.type, "destination");
        expect_value(I.srcs[0], I.type, "source 0");
        expect_b32_or_imm(I.srcs[1], "shift amount");
    }

    void verify_setp() {
        const Inst& I = *inst_;
        if (I.cmp_op == CmpOp::none) error("setp requires a comparison operator");
        if (I.type == Type::pred) error("setp cannot compare .pred values");
        if (!check_arity(1, 2, 2)) return;
        expect_reg_class(I.dsts[0], RegClass::Pred, "setp destination");
        expect_value(I.srcs[0], I.type, "source 0");
        expect_value(I.srcs[1], I.type, "source 1");
    }

    void verify_selp() {
        const Inst& I = *inst_;
        if (!check_arity(1, 3, 3)) return;
        expect_reg(I.dsts[0], I.type, "destination");
        expect_value(I.srcs[0], I.type, "source 0");
        expect_value(I.srcs[1], I.type, "source 1");
        expect_reg_class(I.srcs[2], RegClass::Pred, "selp condition");
    }

    void verify_bra() {
        const Inst& I = *inst_;
        if (!check_arity(0, 1, 1)) return;
        const Operand& t = I.srcs[0];
        if (!t.is_label()) { error("bra target must be a label, got " + std::string(to_string(t.kind)) + " '" + to_string(t) + "'"); return; }
        if (!labels_.count(t.name)) error("bra target label '" + t.name + "' does not exist");
    }

    void verify_call() {
        const Inst& I = *inst_;
        if (I.srcs.empty() || !I.srcs[0].is_symbol()) { error("call requires a callee symbol as source 0"); return; }
        for (size_t i = 0; i < I.dsts.size(); ++i)
            if (!I.dsts[i].is_reg()) error(slot("return value", i) + " must be a register");
        for (size_t i = 1; i < I.srcs.size(); ++i)
            if (!I.srcs[i].is_reg() && !I.srcs[i].is_imm()) error(slot("argument", i - 1) + " must be a register or immediate");
    }

    void verify_shfl() {
        const Inst& I = *inst_;
        if (!I.is_sync) error("shfl requires .sync (the non-sync form is deprecated)");
        if (I.shfl_mode == ShflMode::none) error("shfl requires a mode (.up/.down/.bfly/.idx)");
        if (I.type != Type::b32) error("shfl requires .b32");
        if (!check_arity(1, 4, 4)) return;
        expect_reg(I.dsts[0], Type::b32, "destination");
        expect_reg(I.srcs[0], Type::b32, "source 0");
        expect_b32_or_imm(I.srcs[1], "lane delta (source 1)");
        expect_b32_or_imm(I.srcs[2], "clamp (source 2)");
        expect_b32_or_imm(I.srcs[3], "member mask (source 3)");
    }

    void verify_bar() {
        const Inst& I = *inst_;
        if (!I.is_sync) error("bar requires .sync");
        if (!check_arity(0, 1, 2)) return;
        for (size_t i = 0; i < I.srcs.size(); ++i) expect_b32_or_imm(I.srcs[i], slot("barrier operand", i));
    }

    void verify_atom() {
        const Inst& I = *inst_;
        if (I.atom_op == AtomOp::none) error("atom requires an operation (.add/.min/...)");
        if (I.state_space != StateSpace::global && I.state_space != StateSpace::shared)
            error("atom requires .global or .shared");
        size_t nsrc = (I.atom_op == AtomOp::cas) ? 3 : 2;
        if (!check_arity(1, nsrc, nsrc)) return;
        expect_reg(I.dsts[0], I.type, "destination");
        expect_addr(I.srcs[0], "address");
        for (size_t i = 1; i < nsrc; ++i) expect_value(I.srcs[i], I.type, slot("source", i));
    }

    void verify_unary_math() {
        const Inst& I = *inst_;
        if (!is_float(I.type)) error(std::string(to_string(I.op)) + " requires a float type");
        bool needs_approx = (I.op != Opcode::sqrt && I.op != Opcode::rcp);
        if (needs_approx && !I.is_approx)
            error(std::string(to_string(I.op)) + " only exists in the .approx form");
        if (!needs_approx && !I.is_approx && I.rounding == Rounding::none)
            error(std::string(to_string(I.op)) + " requires .approx or a rounding modifier");
        if (!check_arity(1, 1, 1)) return;
        expect_reg(I.dsts[0], I.type, "destination");
        expect_value(I.srcs[0], I.type, "source 0");
    }
};

} // namespace

std::vector<Diagnostic> verify(const Function& fn) {
    return Verifier(fn).run();
}

} // namespace brass::ptx
