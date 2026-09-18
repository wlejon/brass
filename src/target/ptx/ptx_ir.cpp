#include <brass/target/ptx/ptx_ir.hpp>

namespace brass::ptx {

// ---------------------------------------------------------------------------
// Register classes
// ---------------------------------------------------------------------------

std::string_view to_string(RegClass rc) noexcept {
    switch (rc) {
        case RegClass::Pred: return "pred";
        case RegClass::B32:  return "b32";
        case RegClass::B64:  return "b64";
        case RegClass::F32:  return "f32";
        case RegClass::F64:  return "f64";
    }
    return "?";
}

std::string_view reg_prefix(RegClass rc) noexcept {
    switch (rc) {
        case RegClass::Pred: return "%p";
        case RegClass::B32:  return "%r";
        case RegClass::B64:  return "%rd";
        case RegClass::F32:  return "%f";
        case RegClass::F64:  return "%fd";
    }
    return "%?";
}

std::string_view reg_decl_type(RegClass rc) noexcept {
    switch (rc) {
        case RegClass::Pred: return ".pred";
        case RegClass::B32:  return ".b32";
        case RegClass::B64:  return ".b64";
        case RegClass::F32:  return ".f32";
        case RegClass::F64:  return ".f64";
    }
    return ".?";
}

std::string to_string(Reg r) {
    if (!r.valid()) return std::string(reg_prefix(r.cls)) + "<invalid>";
    return std::string(reg_prefix(r.cls)) + std::to_string(r.index);
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

std::string_view to_string(Type t) noexcept {
    switch (t) {
        case Type::none: return "";
        case Type::pred: return "pred";
        case Type::b8:   return "b8";
        case Type::b16:  return "b16";
        case Type::b32:  return "b32";
        case Type::b64:  return "b64";
        case Type::u8:   return "u8";
        case Type::u16:  return "u16";
        case Type::u32:  return "u32";
        case Type::u64:  return "u64";
        case Type::s8:   return "s8";
        case Type::s16:  return "s16";
        case Type::s32:  return "s32";
        case Type::s64:  return "s64";
        case Type::f16:  return "f16";
        case Type::f32:  return "f32";
        case Type::f64:  return "f64";
    }
    return "?";
}

uint32_t bit_width(Type t) noexcept {
    switch (t) {
        case Type::none: return 0;
        case Type::pred: return 1;
        case Type::b8: case Type::u8: case Type::s8: return 8;
        case Type::b16: case Type::u16: case Type::s16: case Type::f16: return 16;
        case Type::b32: case Type::u32: case Type::s32: case Type::f32: return 32;
        case Type::b64: case Type::u64: case Type::s64: case Type::f64: return 64;
    }
    return 0;
}

Type type_for(brass::Type t) noexcept {
    switch (t.kind()) {
        case TypeKind::I8:    return Type::u8;
        case TypeKind::I16:   return Type::u16;
        case TypeKind::I32:   return Type::u32;
        case TypeKind::I64:   return Type::u64;
        case TypeKind::F32:   return Type::f32;
        case TypeKind::F64:   return Type::f64;
        case TypeKind::Ptr:   return Type::u64;
        case TypeKind::GCRef: return Type::u64;
        case TypeKind::Void:  return Type::none;
        case TypeKind::F32x4: case TypeKind::F32x8: return Type::f32;
        case TypeKind::F64x2: case TypeKind::F64x4: return Type::f64;
        case TypeKind::I32x4: case TypeKind::I32x8: return Type::u32;
        case TypeKind::I64x2: case TypeKind::I64x4: return Type::u64;
    }
    return Type::none;
}

Type signed_type_for(brass::Type t) noexcept {
    Type u = type_for(t);
    switch (u) {
        case Type::u32: return Type::s32;
        case Type::u64: return Type::s64;
        default:        return u;
    }
}

Type bit_type_for(brass::Type t) noexcept {
    Type u = type_for(t);
    switch (u) {
        case Type::u32: return Type::b32;
        case Type::u64: return Type::b64;
        default:        return u;
    }
}

Type wide_type_for(Type t) noexcept {
    switch (t) {
        case Type::u16: return Type::u32;
        case Type::s16: return Type::s32;
        case Type::u32: return Type::u64;
        case Type::s32: return Type::s64;
        default:        return Type::none;
    }
}

RegClass reg_class_for(Type t) noexcept {
    switch (t) {
        case Type::pred: return RegClass::Pred;
        case Type::f32:  return RegClass::F32;
        case Type::f64:  return RegClass::F64;
        case Type::b64: case Type::u64: case Type::s64: return RegClass::B64;
        default: return RegClass::B32; // 8/16/32-bit ints, f16, none
    }
}

RegClass reg_class_for(brass::Type t) noexcept {
    return reg_class_for(type_for(t));
}

// ---------------------------------------------------------------------------
// Opcodes and modifiers
// ---------------------------------------------------------------------------

std::string_view to_string(Opcode op) noexcept {
    switch (op) {
        case Opcode::ld:    return "ld";
        case Opcode::st:    return "st";
        case Opcode::mov:   return "mov";
        case Opcode::cvt:   return "cvt";
        case Opcode::add:   return "add";
        case Opcode::sub:   return "sub";
        case Opcode::mul:   return "mul";
        case Opcode::mad:   return "mad";
        case Opcode::fma:   return "fma";
        case Opcode::div:   return "div";
        case Opcode::rem:   return "rem";
        case Opcode::neg:   return "neg";
        case Opcode::abs:   return "abs";
        case Opcode::min:   return "min";
        case Opcode::max:   return "max";
        case Opcode::and_:  return "and";
        case Opcode::or_:   return "or";
        case Opcode::xor_:  return "xor";
        case Opcode::not_:  return "not";
        case Opcode::shl:   return "shl";
        case Opcode::shr:   return "shr";
        case Opcode::setp:  return "setp";
        case Opcode::selp:  return "selp";
        case Opcode::bra:   return "bra";
        case Opcode::ret:   return "ret";
        case Opcode::call:  return "call";
        case Opcode::trap:  return "trap";
        case Opcode::exit:  return "exit";
        case Opcode::shfl:  return "shfl";
        case Opcode::bar:   return "bar";
        case Opcode::atom:  return "atom";
        case Opcode::rsqrt: return "rsqrt";
        case Opcode::sqrt:  return "sqrt";
        case Opcode::sin:   return "sin";
        case Opcode::cos:   return "cos";
        case Opcode::ex2:   return "ex2";
        case Opcode::lg2:   return "lg2";
        case Opcode::rcp:   return "rcp";
    }
    return "?";
}

std::string_view to_string(Rounding r) noexcept {
    switch (r) {
        case Rounding::none: return "";
        case Rounding::rn:  return "rn";
        case Rounding::rz:  return "rz";
        case Rounding::rm:  return "rm";
        case Rounding::rp:  return "rp";
        case Rounding::rni: return "rni";
        case Rounding::rzi: return "rzi";
        case Rounding::rmi: return "rmi";
        case Rounding::rpi: return "rpi";
    }
    return "?";
}

std::string_view to_string(StateSpace s) noexcept {
    switch (s) {
        case StateSpace::none:   return "";
        case StateSpace::global: return "global";
        case StateSpace::shared: return "shared";
        case StateSpace::param:  return "param";
        case StateSpace::local:  return "local";
        case StateSpace::const_: return "const";
    }
    return "?";
}

std::string_view to_string(VecWidth v) noexcept {
    switch (v) {
        case VecWidth::v1: return "";
        case VecWidth::v2: return "v2";
        case VecWidth::v4: return "v4";
    }
    return "?";
}

std::string_view to_string(MulMode m) noexcept {
    switch (m) {
        case MulMode::none: return "";
        case MulMode::lo:   return "lo";
        case MulMode::hi:   return "hi";
        case MulMode::wide: return "wide";
    }
    return "?";
}

std::string_view to_string(CmpOp c) noexcept {
    switch (c) {
        case CmpOp::none: return "";
        case CmpOp::eq:  return "eq";
        case CmpOp::ne:  return "ne";
        case CmpOp::lt:  return "lt";
        case CmpOp::le:  return "le";
        case CmpOp::gt:  return "gt";
        case CmpOp::ge:  return "ge";
        case CmpOp::lo:  return "lo";
        case CmpOp::ls:  return "ls";
        case CmpOp::hi:  return "hi";
        case CmpOp::hs:  return "hs";
        case CmpOp::equ: return "equ";
        case CmpOp::neu: return "neu";
        case CmpOp::ltu: return "ltu";
        case CmpOp::leu: return "leu";
        case CmpOp::gtu: return "gtu";
        case CmpOp::geu: return "geu";
        case CmpOp::num: return "num";
        case CmpOp::nan: return "nan";
    }
    return "?";
}

std::string_view to_string(ShflMode m) noexcept {
    switch (m) {
        case ShflMode::none: return "";
        case ShflMode::up:   return "up";
        case ShflMode::down: return "down";
        case ShflMode::bfly: return "bfly";
        case ShflMode::idx:  return "idx";
    }
    return "?";
}

std::string_view to_string(AtomOp a) noexcept {
    switch (a) {
        case AtomOp::none: return "";
        case AtomOp::add:  return "add";
        case AtomOp::min:  return "min";
        case AtomOp::max:  return "max";
        case AtomOp::inc:  return "inc";
        case AtomOp::dec:  return "dec";
        case AtomOp::and_: return "and";
        case AtomOp::or_:  return "or";
        case AtomOp::xor_: return "xor";
        case AtomOp::exch: return "exch";
        case AtomOp::cas:  return "cas";
    }
    return "?";
}

std::string_view to_string(SpecialReg s) noexcept {
    switch (s) {
        case SpecialReg::tid_x:       return "%tid.x";
        case SpecialReg::tid_y:       return "%tid.y";
        case SpecialReg::tid_z:       return "%tid.z";
        case SpecialReg::ntid_x:      return "%ntid.x";
        case SpecialReg::ntid_y:      return "%ntid.y";
        case SpecialReg::ntid_z:      return "%ntid.z";
        case SpecialReg::ctaid_x:     return "%ctaid.x";
        case SpecialReg::ctaid_y:     return "%ctaid.y";
        case SpecialReg::ctaid_z:     return "%ctaid.z";
        case SpecialReg::nctaid_x:    return "%nctaid.x";
        case SpecialReg::nctaid_y:    return "%nctaid.y";
        case SpecialReg::nctaid_z:    return "%nctaid.z";
        case SpecialReg::laneid:      return "%laneid";
        case SpecialReg::warpid:      return "%warpid";
        case SpecialReg::nwarpid:     return "%nwarpid";
        case SpecialReg::smid:        return "%smid";
        case SpecialReg::nsmid:       return "%nsmid";
        case SpecialReg::clock:       return "%clock";
        case SpecialReg::clock64:     return "%clock64";
        case SpecialReg::globaltimer: return "%globaltimer";
    }
    return "%?";
}

RegClass reg_class_for(SpecialReg s) noexcept {
    return (s == SpecialReg::clock64 || s == SpecialReg::globaltimer) ? RegClass::B64 : RegClass::B32;
}

bool is_invariant(SpecialReg s) noexcept {
    switch (s) {
        case SpecialReg::warpid:
        case SpecialReg::smid:
        case SpecialReg::clock:
        case SpecialReg::clock64:
        case SpecialReg::globaltimer:
            return false;
        default:
            return true;
    }
}

// ---------------------------------------------------------------------------
// Operand rules
// ---------------------------------------------------------------------------

bool allows_immediate(Opcode op, size_t src_index) noexcept {
    switch (op) {
        case Opcode::mov:
        case Opcode::neg: case Opcode::abs: case Opcode::not_:
        case Opcode::rsqrt: case Opcode::sqrt: case Opcode::sin: case Opcode::cos:
        case Opcode::ex2: case Opcode::lg2: case Opcode::rcp:
            return src_index == 0;
        case Opcode::add: case Opcode::sub: case Opcode::mul: case Opcode::div: case Opcode::rem:
        case Opcode::min: case Opcode::max:
        case Opcode::and_: case Opcode::or_: case Opcode::xor_:
        case Opcode::shl: case Opcode::shr:
        case Opcode::selp:                       // sources 0 and 1; source 2 is the predicate
            return src_index <= 1;
        case Opcode::mad: case Opcode::fma:
            return src_index <= 2;
        case Opcode::setp:                       // second source only
            return src_index == 1;
        case Opcode::st:                         // the stored value
            return src_index == 1;
        case Opcode::atom:                       // value (and the cas compare value)
            return src_index == 1 || src_index == 2;
        case Opcode::shfl:                       // lane/delta, clamp, member mask
            return src_index >= 1 && src_index <= 3;
        case Opcode::bar:                        // barrier id, thread count
            return src_index <= 1;
        case Opcode::call:                       // arguments (source 0 is the callee)
            return src_index >= 1;
        case Opcode::ld: case Opcode::cvt:
        case Opcode::bra: case Opcode::ret: case Opcode::trap: case Opcode::exit:
            return false;
    }
    return false;
}

bool imm_fits(Type t, int64_t v) noexcept {
    switch (bit_width(t)) {
        case 64: return true;
        case 32: return v >= INT32_MIN && v <= static_cast<int64_t>(UINT32_MAX);
        case 16: return v >= INT16_MIN && v <= static_cast<int64_t>(UINT16_MAX);
        case 8:  return v >= INT8_MIN && v <= static_cast<int64_t>(UINT8_MAX);
        default: return false; // pred / none: no integer immediates
    }
}

std::string_view to_string(OperandKind k) noexcept {
    switch (k) {
        case OperandKind::None:     return "none";
        case OperandKind::Reg:      return "reg";
        case OperandKind::ImmInt:   return "imm";
        case OperandKind::ImmFloat: return "fimm";
        case OperandKind::Addr:     return "addr";
        case OperandKind::Vector:   return "vector";
        case OperandKind::Label:    return "label";
        case OperandKind::Param:    return "param";
        case OperandKind::Special:  return "special";
        case OperandKind::Symbol:   return "symbol";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// Operands
// ---------------------------------------------------------------------------

Operand Operand::reg(Reg r) { return Operand(r); }

Operand Operand::imm(int64_t v) {
    Operand o;
    o.kind = OperandKind::ImmInt;
    o.imm_int = v;
    return o;
}

Operand Operand::imm_f32(float v) {
    Operand o;
    o.kind = OperandKind::ImmFloat;
    o.imm_float = static_cast<double>(v);
    o.imm_is_f32 = true;
    return o;
}

Operand Operand::imm_f64(double v) {
    Operand o;
    o.kind = OperandKind::ImmFloat;
    o.imm_float = v;
    o.imm_is_f32 = false;
    return o;
}

Operand Operand::addr(Reg base, int32_t disp) {
    Operand o;
    o.kind = OperandKind::Addr;
    o.addr_base = base;
    o.disp = disp;
    return o;
}

Operand Operand::addr(std::string symbol, int32_t disp) {
    Operand o;
    o.kind = OperandKind::Addr;
    o.addr_symbol = std::move(symbol);
    o.disp = disp;
    return o;
}

Operand Operand::vec(std::vector<Reg> regs) {
    Operand o;
    o.kind = OperandKind::Vector;
    o.elems = std::move(regs);
    return o;
}

Operand Operand::label(std::string name) {
    Operand o;
    o.kind = OperandKind::Label;
    o.name = std::move(name);
    return o;
}

Operand Operand::param(std::string name) {
    Operand o;
    o.kind = OperandKind::Param;
    o.name = std::move(name);
    return o;
}

Operand Operand::special(SpecialReg s) {
    Operand o;
    o.kind = OperandKind::Special;
    o.special_reg = s;
    return o;
}

Operand Operand::symbol(std::string name) {
    Operand o;
    o.kind = OperandKind::Symbol;
    o.name = std::move(name);
    return o;
}

// ---------------------------------------------------------------------------
// Function
// ---------------------------------------------------------------------------

Reg Function::new_reg(RegClass rc) {
    uint32_t& count = reg_counts[static_cast<size_t>(rc)];
    return Reg(rc, count++);
}

std::vector<Reg> Function::new_regs(RegClass rc, size_t n) {
    std::vector<Reg> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) out.push_back(new_reg(rc));
    return out;
}

Operand Function::add_param(Type t, std::string param_name) {
    params.push_back(Param{t, param_name});
    return Operand::param(std::move(param_name));
}

const Param* Function::find_param(std::string_view param_name) const noexcept {
    for (const auto& p : params) {
        if (p.name == param_name) return &p;
    }
    return nullptr;
}

SharedDecl& Function::add_shared(Type t, std::string shared_name, uint32_t count, uint32_t align) {
    shared.push_back(SharedDecl{t, std::move(shared_name), count, align});
    return shared.back();
}

const SharedDecl* Function::find_shared(std::string_view shared_name) const noexcept {
    for (const auto& s : shared) {
        if (s.name == shared_name) return &s;
    }
    return nullptr;
}

Block* Function::add_block(std::string label) {
    blocks.push_back(std::make_unique<Block>(std::move(label)));
    return blocks.back().get();
}

Block* Function::find_block(std::string_view label) const noexcept {
    for (const auto& b : blocks) {
        if (b->label == label) return b.get();
    }
    return nullptr;
}

std::string Function::unique_label(std::string_view hint) {
    return "$L_" + std::string(hint) + "_" + std::to_string(next_label_id_++);
}

} // namespace brass::ptx
