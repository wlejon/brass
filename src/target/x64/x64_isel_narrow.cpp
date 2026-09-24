// i8 / i16 operands (docs/semantics.md, "Narrow integers").
//
// Tier 2 keeps a narrow value in a GPR whose bits above the value's width
// are unspecified: producers (add, mul, shl, trunc_i8, ...) do not mask.
// Every consumer whose result depends on those bits sees the operand
// through a sign or zero extension to 64 bits instead, made here before
// the instruction is lowered. Stores write only the value's width and
// zext_i64 masks it, so they need nothing.
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>

namespace brass::x64 {

using namespace brass::codegen;

namespace {

bool is_narrow(Type t) noexcept { return t == Type::i8() || t == Type::i16(); }

// Counting leading / trailing zeros and overflow checks depend on the width
// itself, not only on the bits of an extended operand.
bool is_width_dependent(Opcode op) noexcept {
    switch (op) {
        case Opcode::clz: case Opcode::ctz:
        case Opcode::sadd_overflow: case Opcode::ssub_overflow: case Opcode::smul_overflow:
        case Opcode::uadd_overflow: case Opcode::usub_overflow: case Opcode::umul_overflow:
            return true;
        default:
            return false;
    }
}

enum class Ext { None, Sign, Zero };

// How operand `i` of `inst` is widened, when it is narrow.
Ext operand_extension(const Instruction& inst, size_t i) noexcept {
    switch (inst.opcode()) {
        case Opcode::slt: case Opcode::sle: case Opcode::sgt: case Opcode::sge:
        case Opcode::sdiv: case Opcode::smod:
        case Opcode::switch_:
            return Ext::Sign;
        case Opcode::ashr:
            return i == 0 ? Ext::Sign : Ext::Zero;
        case Opcode::eq: case Opcode::ne:
        case Opcode::ult: case Opcode::ule: case Opcode::ugt: case Opcode::uge:
        case Opcode::udiv: case Opcode::umod:
        case Opcode::lshr: case Opcode::popcnt:
            return Ext::Zero;
        default:
            return Ext::None;
    }
}

} // namespace

bool X64ISel::narrow_compare(const Instruction& cmp) noexcept {
    return is_comparison(cmp.opcode()) && cmp.operand_count() >= 1 && cmp.operand(0) &&
           is_narrow(cmp.operand(0)->type());
}

std::vector<std::pair<const Value*, VReg>> X64ISel::widen_narrow_operands(const Instruction& inst,
                                                                          LirBlock& lir_bb) {
    std::vector<std::pair<const Value*, VReg>> saved;
    bool any_narrow = false;
    for (size_t i = 0; i < inst.operand_count(); ++i) {
        if (inst.operand(i) && is_narrow(inst.operand(i)->type())) any_narrow = true;
    }
    if (!any_narrow) return saved;

    if (is_width_dependent(inst.opcode())) {
        // lower_narrow_width_op lowers these; widening the operands alone
        // would compute them at the wrong width.
        throw std::logic_error("x64 isel: narrow " + std::string(opcode_name(inst.opcode())) +
                               " reached the operand widening");
    }

    for (size_t i = 0; i < inst.operand_count(); ++i) {
        const Value* v = inst.operand(i);
        if (!v || !is_narrow(v->type())) continue;
        const Ext ext = operand_extension(inst, i);
        if (ext == Ext::None) continue;
        bool already = false;
        for (const auto& s : saved) already = already || s.first == v;
        if (already) continue;
        auto it = val_to_vreg_.find(v);
        if (it == val_to_vreg_.end()) {
            throw std::logic_error("x64 isel: narrow operand of " + std::string(opcode_name(inst.opcode())) +
                                   " has no register");
        }
        const uint8_t w = static_cast<uint8_t>(v->type().size_in_bytes());
        const LirOpcode op = ext == Ext::Sign ? (w == 1 ? LirOpcode::Movsx8 : LirOpcode::Movsx16)
                                              : (w == 1 ? LirOpcode::Movzx8 : LirOpcode::Movzx16);
        VReg wide = lir_fn_->allocate_vreg(RegClass::GPR, 8);
        auto x = std::make_unique<LirInst>(op);
        // movsx is emitted 64-bit; movzx 32-bit, which clears the upper half.
        x->add_def(LirOperand::vreg(wide, ext == Ext::Sign ? 8 : 4));
        x->add_use(LirOperand::vreg(it->second, w));
        lir_bb.append_inst(std::move(x));
        saved.emplace_back(v, it->second);
        it->second = wide;
    }
    return saved;
}

void X64ISel::restore_narrow_operands(const std::vector<std::pair<const Value*, VReg>>& saved) {
    for (const auto& [v, r] : saved) val_to_vreg_[v] = r;
}

bool X64ISel::lower_narrow_width_op(const Instruction& inst, LirBlock& lir_bb) {
    if (!is_width_dependent(inst.opcode()) || inst.operand_count() < 1 || !inst.operand(0) ||
        !is_narrow(inst.operand(0)->type())) {
        return false;
    }
    const Type t = inst.operand(0)->type();
    const unsigned bits = t == Type::i8() ? 8u : 16u;
    const Opcode op = inst.opcode();

    auto append = [&](LirOpcode lop, std::initializer_list<LirOperand> defs, std::initializer_list<LirOperand> uses) {
        auto x = std::make_unique<LirInst>(lop);
        for (const auto& d : defs) x->add_def(d);
        for (const auto& u : uses) x->add_use(u);
        x->mir_origin = &inst;
        lir_bb.append_inst(std::move(x));
    };
    auto operand = [&](size_t i) {
        VReg r = get_vreg(inst.operand(i));
        if (!r.is_valid()) {
            throw std::logic_error("x64 isel: narrow operand of " + std::string(opcode_name(op)) + " has no register");
        }
        return r;
    };
    // The operand's `bits` low bits sign- or zero-extended to 64 bits.
    auto extend = [&](VReg src, bool sign) {
        VReg wide = lir_fn_->allocate_vreg(RegClass::GPR, 8);
        const uint8_t w = static_cast<uint8_t>(bits / 8);
        const LirOpcode xop = sign ? (bits == 8 ? LirOpcode::Movsx8 : LirOpcode::Movsx16)
                                   : (bits == 8 ? LirOpcode::Movzx8 : LirOpcode::Movzx16);
        // movsx is emitted 64-bit; movzx 32-bit, which clears the upper half.
        append(xop, {LirOperand::vreg(wide, sign ? 8 : 4)}, {LirOperand::vreg(src, w)});
        return wide;
    };
    const VReg dst = get_vreg(inst.result());

    switch (op) {
        case Opcode::clz: {
            // The value in the top `bits` of a 32-bit word with a stop bit
            // just below it: its 32-bit leading-zero count is the narrow
            // one, the width for zero, and the word is never zero.
            VReg w = extend(operand(0), false);
            append(LirOpcode::Shl32, {LirOperand::vreg(w, 4)}, {LirOperand::vreg(w, 4), LirOperand::imm(32 - bits, 1)});
            append(LirOpcode::Or32, {LirOperand::vreg(w, 4)},
                   {LirOperand::vreg(w, 4), LirOperand::imm(int64_t{1} << (31 - bits), 4)});
            // clz32 = 31 - bsr.
            VReg idx = lir_fn_->allocate_vreg(RegClass::GPR, 8);
            append(LirOpcode::Bsr32, {LirOperand::vreg(idx, 4)}, {LirOperand::vreg(w, 4)});
            append(LirOpcode::Mov32, {LirOperand::vreg(dst, 4)}, {LirOperand::imm(31, 4)});
            append(LirOpcode::Sub32, {LirOperand::vreg(dst, 4)}, {LirOperand::vreg(dst, 4), LirOperand::vreg(idx, 4)});
            return true;
        }
        case Opcode::ctz: {
            // A bit just above the width makes ctz of zero the width.
            VReg w = extend(operand(0), false);
            append(LirOpcode::Or32, {LirOperand::vreg(w, 4)},
                   {LirOperand::vreg(w, 4), LirOperand::imm(int64_t{1} << bits, 4)});
            append(LirOpcode::Bsf32, {LirOperand::vreg(dst, 4)}, {LirOperand::vreg(w, 4)});
            return true;
        }
        default:
            break;
    }

    // Overflow: the exact result of the extended operands (it fits in 64
    // bits) differs from the extension of its own low `bits`.
    const bool sign = op == Opcode::sadd_overflow || op == Opcode::ssub_overflow || op == Opcode::smul_overflow;
    const LirOpcode alu = (op == Opcode::sadd_overflow || op == Opcode::uadd_overflow)   ? LirOpcode::Add
                          : (op == Opcode::ssub_overflow || op == Opcode::usub_overflow) ? LirOpcode::Sub
                                                                                         : LirOpcode::Imul;
    VReg a = extend(operand(0), sign);
    VReg b = extend(operand(1), sign);
    append(alu, {LirOperand::vreg(a, 8)}, {LirOperand::vreg(a, 8), LirOperand::vreg(b, 8)});
    VReg re = extend(a, sign);
    append(LirOpcode::Cmp, {}, {LirOperand::vreg(a, 8), LirOperand::vreg(re, 8)});
    auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
    setcc->condition = Condition::NE;
    setcc->add_def(LirOperand::vreg(dst, 1));
    setcc->mir_origin = &inst;
    lir_bb.append_inst(std::move(setcc));
    append(LirOpcode::Movzx8, {LirOperand::vreg(dst, 4)}, {LirOperand::vreg(dst, 1)});
    return true;
}

} // namespace brass::x64
