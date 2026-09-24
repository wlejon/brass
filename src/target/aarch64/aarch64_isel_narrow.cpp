// i8 / i16 operands (docs/semantics.md, "Narrow integers").
//
// Tier 2 keeps a narrow value in a GPR whose bits above the value's width
// are unspecified: producers (add, mul, shl, trunc_i8, ...) do not mask.
// Every consumer whose result depends on those bits sees the operand
// through a sign or zero extension to 64 bits instead, made here before
// the instruction is lowered. Stores write only the value's width and
// zext_i64 masks it, so they need nothing.
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/unsupported_operation.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

namespace {

bool is_narrow(Type t) noexcept { return t == Type::i8() || t == Type::i16(); }

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

bool AArch64ISel::narrow_compare(const Instruction& cmp) noexcept {
    return is_comparison(cmp.opcode()) && cmp.operand_count() >= 1 && cmp.operand(0) &&
           is_narrow(cmp.operand(0)->type());
}

std::vector<std::pair<const Value*, VReg>> AArch64ISel::widen_narrow_operands(const Instruction& inst,
                                                                              LirBlock& lir_bb) {
    std::vector<std::pair<const Value*, VReg>> saved;
    bool any_narrow = false;
    for (size_t i = 0; i < inst.operand_count(); ++i) {
        if (inst.operand(i) && is_narrow(inst.operand(i)->type())) any_narrow = true;
    }
    if (!any_narrow) return saved;

    switch (inst.opcode()) {
        // Counting leading / trailing zeros and overflow checks depend on
        // the width itself, not only on the bits of an extended operand.
        case Opcode::clz: case Opcode::ctz:
        case Opcode::sadd_overflow: case Opcode::ssub_overflow: case Opcode::smul_overflow:
        case Opcode::uadd_overflow: case Opcode::usub_overflow: case Opcode::umul_overflow:
            throw_unsupported("aarch64 isel", std::string(opcode_name(inst.opcode())) + " on " +
                                              std::string(inst.operand(0)->type().name()));
        default:
            break;
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
            throw std::logic_error("aarch64 isel: narrow operand of " + std::string(opcode_name(inst.opcode())) +
                                   " has no register");
        }
        const uint8_t w = static_cast<uint8_t>(v->type().size_in_bytes());
        const LirOpcode op = ext == Ext::Sign ? (w == 1 ? LirOpcode::Movsx8 : LirOpcode::Movsx16)
                                              : (w == 1 ? LirOpcode::Movzx8 : LirOpcode::Movzx16);
        VReg wide = lir_fn_->allocate_vreg(RegClass::GPR, 8);
        auto x = std::make_unique<LirInst>(op);
        // movsx is emitted 64-bit; movzx 32-bit, which clears the upper half in AArch64.
        x->add_def(LirOperand::vreg(wide, ext == Ext::Sign ? 8 : 4));
        x->add_use(LirOperand::vreg(it->second, w));
        lir_bb.append_inst(std::move(x));
        saved.emplace_back(v, it->second);
        it->second = wide;
    }
    return saved;
}

void AArch64ISel::restore_narrow_operands(const std::vector<std::pair<const Value*, VReg>>& saved) {
    for (const auto& [v, r] : saved) val_to_vreg_[v] = r;
}

} // namespace brass::aarch64
