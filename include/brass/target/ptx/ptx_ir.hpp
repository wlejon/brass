#pragma once

// Typed PTX IR. This is the data model between PtxISel (MIR -> ptx::Function)
// and the PtxPrinter (ptx::Function -> text). Every PTX construct the backend
// emits is representable here as typed fields; nothing in this layer is a
// string except names (params, labels, shared arrays, call targets).
//
// See docs/ptx_backend_design.md for the contract.

#include <brass/mir/types.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brass {
class Instruction;
}

namespace brass::ptx {

// ---------------------------------------------------------------------------
// Registers
// ---------------------------------------------------------------------------

// One virtual register file per class. The printer names them %p, %r, %rd,
// %f and %fd respectively, matching the historical emitter.
enum class RegClass : uint8_t { Pred, B32, B64, F32, F64 };
constexpr size_t kRegClassCount = 5;

std::string_view to_string(RegClass rc) noexcept;   // "pred", "b32", ...
std::string_view reg_prefix(RegClass rc) noexcept;  // "%p", "%r", "%rd", "%f", "%fd"
std::string_view reg_decl_type(RegClass rc) noexcept; // ".pred", ".b32", ".b64", ".f32", ".f64"

struct Reg {
    static constexpr uint32_t kInvalid = UINT32_MAX;

    RegClass cls = RegClass::B32;
    uint32_t index = kInvalid;

    constexpr Reg() noexcept = default;
    constexpr Reg(RegClass c, uint32_t i) noexcept : cls(c), index(i) {}

    constexpr bool valid() const noexcept { return index != kInvalid; }
    constexpr bool operator==(const Reg&) const noexcept = default;
    constexpr bool operator!=(const Reg&) const noexcept = default;
};

std::string to_string(Reg r);

// ---------------------------------------------------------------------------
// Types (the PTX suffix set)
// ---------------------------------------------------------------------------

enum class Type : uint8_t {
    none,
    pred,
    b8, b16, b32, b64,
    u8, u16, u32, u64,
    s8, s16, s32, s64,
    f16, f32, f64
};

std::string_view to_string(Type t) noexcept; // "f32", "u64", ... ("" for none)

constexpr bool is_float(Type t) noexcept { return t == Type::f16 || t == Type::f32 || t == Type::f64; }
constexpr bool is_signed(Type t) noexcept { return t == Type::s8 || t == Type::s16 || t == Type::s32 || t == Type::s64; }
constexpr bool is_unsigned(Type t) noexcept { return t == Type::u8 || t == Type::u16 || t == Type::u32 || t == Type::u64; }
constexpr bool is_bits(Type t) noexcept { return t == Type::b8 || t == Type::b16 || t == Type::b32 || t == Type::b64; }
constexpr bool is_integer(Type t) noexcept { return is_signed(t) || is_unsigned(t); }

// Bit width of a suffix (pred -> 1, none -> 0).
uint32_t bit_width(Type t) noexcept;

// The single source of truth for suffix selection.
//   type_for:        data/unsigned suffix used by ld/st/param/cvt (i32 -> u32,
//                    i64/ptr/gcref -> u64, f32 -> f32, f64 -> f64). Vector MIR
//                    types map to their element type.
//   signed_type_for: s32/s64 for integer MIR types (floats unchanged).
//   bit_type_for:    b32/b64 for integer/pointer MIR types (floats unchanged);
//                    used by mov/selp/and/or/xor/not/shl.
//   wide_type_for:   the double-width type for mul.wide / mad.wide results.
Type type_for(brass::Type t) noexcept;
Type signed_type_for(brass::Type t) noexcept;
Type bit_type_for(brass::Type t) noexcept;
Type wide_type_for(Type t) noexcept;

// Register class that holds a value of the given suffix. Sub-32-bit types and
// f16 live in B32 registers (PTX permits wider registers for ld/st/cvt of
// narrow types, and the hand-written kernels rely on it).
RegClass reg_class_for(Type t) noexcept;
RegClass reg_class_for(brass::Type t) noexcept;

// ---------------------------------------------------------------------------
// Opcodes and modifiers
// ---------------------------------------------------------------------------

enum class Opcode : uint8_t {
    ld, st, mov, cvt,
    add, sub, mul, mad, fma, div, rem, neg, abs, min, max,
    and_, or_, xor_, not_, shl, shr,
    setp, selp,
    bra, ret, call, trap, exit,
    shfl, bar, atom,
    rsqrt, sqrt, sin, cos, ex2, lg2, rcp
};

std::string_view to_string(Opcode op) noexcept; // mnemonic ("and", "or", ...)

enum class Rounding : uint8_t { none, rn, rz, rm, rp, rni, rzi, rmi, rpi };
enum class StateSpace : uint8_t { none, global, shared, param, local, const_ };
enum class VecWidth : uint8_t { v1 = 1, v2 = 2, v4 = 4 };
enum class MulMode : uint8_t { none, lo, hi, wide };
enum class CmpOp : uint8_t {
    none,
    eq, ne, lt, le, gt, ge,      // signed / unsigned / bit compare
    lo, ls, hi, hs,              // unsigned aliases
    equ, neu, ltu, leu, gtu, geu, num, nan // float, unordered variants
};
enum class ShflMode : uint8_t { none, up, down, bfly, idx };
enum class AtomOp : uint8_t { none, add, min, max, inc, dec, and_, or_, xor_, exch, cas };

std::string_view to_string(Rounding r) noexcept;
std::string_view to_string(StateSpace s) noexcept;
std::string_view to_string(VecWidth v) noexcept;
std::string_view to_string(MulMode m) noexcept;
std::string_view to_string(CmpOp c) noexcept;
std::string_view to_string(ShflMode m) noexcept;
std::string_view to_string(AtomOp a) noexcept;

// Special (read-only) registers. Printed as %tid.x, %ctaid.x, ...
enum class SpecialReg : uint8_t {
    tid_x, tid_y, tid_z,
    ntid_x, ntid_y, ntid_z,
    ctaid_x, ctaid_y, ctaid_z,
    nctaid_x, nctaid_y, nctaid_z,
    laneid, warpid, nwarpid, smid, nsmid,
    clock, clock64, globaltimer
};

std::string_view to_string(SpecialReg s) noexcept;
RegClass reg_class_for(SpecialReg s) noexcept; // B32, or B64 for clock64/globaltimer

// ---------------------------------------------------------------------------
// Operands
// ---------------------------------------------------------------------------

enum class OperandKind : uint8_t {
    None,
    Reg,        // %r3
    ImmInt,     // 42
    ImmFloat,   // 0f3F800000 / 0d3FF0000000000000 (printer owns the formatting)
    Addr,       // [%rd1 + 8] or [symbol + 8]
    Vector,     // {%f0, %f1, %f2, %f3}
    Label,      // $L_loop
    Param,      // [param_name] (ld.param source)
    Special,    // %tid.x
    Symbol      // smem (shared array name) or a call target
};

std::string_view to_string(OperandKind k) noexcept;

struct Operand {
    OperandKind kind = OperandKind::None;
    Reg reg_val;                 // Reg
    int64_t imm_int = 0;         // ImmInt
    double imm_float = 0.0;      // ImmFloat
    bool imm_is_f32 = false;     // ImmFloat: print as 0f... (else 0d...)
    Reg addr_base;               // Addr: base register (may be invalid when symbol-based)
    std::string addr_symbol;     // Addr: symbol base (param or shared name)
    int32_t disp = 0;            // Addr: byte displacement
    std::vector<Reg> elems;      // Vector
    std::string name;            // Label / Param / Symbol
    SpecialReg special_reg = SpecialReg::tid_x; // Special

    Operand() = default;
    Operand(Reg r) : kind(OperandKind::Reg), reg_val(r) {} // NOLINT: implicit by design

    static Operand reg(Reg r);
    static Operand imm(int64_t v);
    static Operand imm_f32(float v);
    static Operand imm_f64(double v);
    static Operand addr(Reg base, int32_t disp = 0);
    static Operand addr(std::string symbol, int32_t disp = 0);
    static Operand vec(std::vector<Reg> regs);
    static Operand label(std::string name);
    static Operand param(std::string name);
    static Operand special(SpecialReg s);
    static Operand symbol(std::string name);

    bool is_none() const noexcept { return kind == OperandKind::None; }
    bool is_reg() const noexcept { return kind == OperandKind::Reg; }
    bool is_imm_int() const noexcept { return kind == OperandKind::ImmInt; }
    bool is_imm_float() const noexcept { return kind == OperandKind::ImmFloat; }
    bool is_imm() const noexcept { return is_imm_int() || is_imm_float(); }
    bool is_addr() const noexcept { return kind == OperandKind::Addr; }
    bool is_vec() const noexcept { return kind == OperandKind::Vector; }
    bool is_label() const noexcept { return kind == OperandKind::Label; }
    bool is_param() const noexcept { return kind == OperandKind::Param; }
    bool is_special() const noexcept { return kind == OperandKind::Special; }
    bool is_symbol() const noexcept { return kind == OperandKind::Symbol; }
};

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

struct Inst {
    Opcode op;
    Type type = Type::none;       // instruction type (cvt: destination type)
    Type src_type = Type::none;   // cvt only: source type
    Rounding rounding = Rounding::none;
    bool is_approx = false;
    bool is_ftz = false;
    bool is_sat = false;
    bool is_sync = false;         // shfl.sync / bar.sync
    bool is_uni = false;          // bra.uni / ret.uni
    StateSpace state_space = StateSpace::none;
    VecWidth vec_width = VecWidth::v1;
    MulMode mul_mode = MulMode::none;
    CmpOp cmp_op = CmpOp::none;
    ShflMode shfl_mode = ShflMode::none;
    AtomOp atom_op = AtomOp::none;

    bool has_guard = false;
    bool guard_negated = false;
    Reg guard_reg;

    std::vector<Operand> dsts;
    std::vector<Operand> srcs;

    const brass::Instruction* mir_origin = nullptr;

    explicit Inst(Opcode o, Type t = Type::none) : op(o), type(t) {}

    // Builder. Each setter returns *this so ISel reads as
    //   Inst::make(Opcode::add, Type::f32).dst(r).src(a).src(b)
    static Inst make(Opcode o, Type t = Type::none) { return Inst(o, t); }

    Inst& dst(Operand o) { dsts.push_back(std::move(o)); return *this; }
    Inst& src(Operand o) { srcs.push_back(std::move(o)); return *this; }
    Inst& guard(Reg p, bool negated = false) {
        has_guard = true; guard_reg = p; guard_negated = negated; return *this;
    }
    Inst& from(Type t) { src_type = t; return *this; }       // cvt source type
    Inst& rnd(Rounding r) { rounding = r; return *this; }
    Inst& approx() { is_approx = true; return *this; }
    Inst& ftz() { is_ftz = true; return *this; }
    Inst& sat() { is_sat = true; return *this; }
    Inst& sync() { is_sync = true; return *this; }
    Inst& uni() { is_uni = true; return *this; }
    Inst& space(StateSpace s) { state_space = s; return *this; }
    Inst& vec(VecWidth w) { vec_width = w; return *this; }
    Inst& lo() { mul_mode = MulMode::lo; return *this; }
    Inst& hi() { mul_mode = MulMode::hi; return *this; }
    Inst& wide() { mul_mode = MulMode::wide; return *this; }
    Inst& cmp(CmpOp c) { cmp_op = c; return *this; }
    Inst& shfl(ShflMode m) { shfl_mode = m; return *this; }
    Inst& atom(AtomOp a) { atom_op = a; return *this; }
    Inst& origin(const brass::Instruction* i) { mir_origin = i; return *this; }

    bool is_terminator() const noexcept {
        return (op == Opcode::bra || op == Opcode::ret || op == Opcode::exit || op == Opcode::trap) &&
               !has_guard;
    }
};

// ---------------------------------------------------------------------------
// Blocks and functions
// ---------------------------------------------------------------------------

struct Block {
    std::string label;
    std::vector<Inst> insts;

    Block() = default;
    explicit Block(std::string l) : label(std::move(l)) {}

    Inst& append(Inst i) { insts.push_back(std::move(i)); return insts.back(); }
};

struct Param {
    Type type = Type::none;
    std::string name;
};

struct SharedDecl {
    Type type = Type::none;
    std::string name;
    uint32_t count = 0;
    uint32_t align = 4;
};

class Function {
public:
    std::string name;
    bool is_entry = true;
    std::vector<Param> params;
    std::vector<SharedDecl> shared;
    std::array<uint32_t, kRegClassCount> reg_counts{};
    std::vector<std::unique_ptr<Block>> blocks;

    Function() = default;
    explicit Function(std::string n, bool entry = true) : name(std::move(n)), is_entry(entry) {}

    // Registers are allocated by class; counts feed the .reg declarations.
    Reg new_reg(RegClass rc);
    std::vector<Reg> new_regs(RegClass rc, size_t n); // contiguous indices
    Reg new_pred() { return new_reg(RegClass::Pred); }
    Reg new_b32() { return new_reg(RegClass::B32); }
    Reg new_b64() { return new_reg(RegClass::B64); }
    Reg new_f32() { return new_reg(RegClass::F32); }
    Reg new_f64() { return new_reg(RegClass::F64); }
    uint32_t reg_count(RegClass rc) const noexcept { return reg_counts[static_cast<size_t>(rc)]; }

    // Params and shared arrays. add_param returns the operand to use as a
    // ld.param source.
    Operand add_param(Type t, std::string param_name);
    const Param* find_param(std::string_view param_name) const noexcept;
    SharedDecl& add_shared(Type t, std::string shared_name, uint32_t count, uint32_t align = 4);
    const SharedDecl* find_shared(std::string_view shared_name) const noexcept;

    // Blocks are appended in layout order. Pointers stay valid for the life of
    // the function. Labels must be unique; unique_label() derives one from a hint.
    Block* add_block(std::string label);
    Block* find_block(std::string_view label) const noexcept;
    Block* entry_block() const noexcept { return blocks.empty() ? nullptr : blocks.front().get(); }
    std::string unique_label(std::string_view hint);

private:
    uint32_t next_label_id_ = 0;
};

} // namespace brass::ptx
