#pragma once

#include <brass/target/x64/x64_registers.hpp>
#include <brass/target/x64/x64_operands.hpp>
#include <brass/target/x64/code_buffer.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/mir/types.hpp>
#include <brass/mir/instruction.hpp>
#include <cstdint>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <iosfwd>

namespace brass::codegen {

enum class RegClass : uint8_t {
    GPR,
    XMM
};

std::string_view to_string(RegClass rc) noexcept;

struct VReg {
    static constexpr uint32_t kInvalid = UINT32_MAX;

    uint32_t id = kInvalid;
    RegClass reg_class = RegClass::GPR;
    uint8_t size = 8; // in bytes: 4 or 8
    bool is_gcref = false;

    constexpr bool is_valid() const noexcept { return id != kInvalid; }
    constexpr bool is_gpr() const noexcept { return reg_class == RegClass::GPR; }
    constexpr bool is_xmm() const noexcept { return reg_class == RegClass::XMM; }
    constexpr bool is_32() const noexcept { return size == 4; }
    constexpr bool is_64() const noexcept { return size == 8; }
    constexpr bool is_128() const noexcept { return size == 16; }

    constexpr bool operator==(const VReg& other) const noexcept = default;
    constexpr bool operator!=(const VReg& other) const noexcept = default;
};

struct PReg {
    static constexpr uint8_t kInvalid = 0xFF;

    RegClass reg_class = RegClass::GPR;
    uint8_t code = kInvalid; // 0..15

    constexpr bool is_valid() const noexcept { return code != kInvalid; }
    constexpr bool is_gpr() const noexcept { return reg_class == RegClass::GPR && is_valid(); }
    constexpr bool is_xmm() const noexcept { return reg_class == RegClass::XMM && is_valid(); }

    constexpr x64::GPR as_gpr() const noexcept {
        return is_gpr() ? static_cast<x64::GPR>(code) : x64::GPR::None;
    }
    constexpr x64::XMM as_xmm() const noexcept {
        return is_xmm() ? static_cast<x64::XMM>(code) : x64::XMM::None;
    }

    static constexpr PReg gpr(x64::GPR r) noexcept {
        PReg p;
        p.reg_class = RegClass::GPR;
        p.code = static_cast<uint8_t>(r);
        return p;
    }

    static constexpr PReg xmm(x64::XMM r) noexcept {
        PReg p;
        p.reg_class = RegClass::XMM;
        p.code = static_cast<uint8_t>(r);
        return p;
    }

    constexpr bool operator==(const PReg& other) const noexcept = default;
    constexpr bool operator!=(const PReg& other) const noexcept = default;
};

enum class LirOperandKind : uint8_t {
    None,
    VReg,
    PReg,
    ImmInt,
    ImmFloat,
    Mem,
    SpillSlot,
    Label,
    Symbol,
    Condition
};

std::string_view to_string(LirOperandKind kind) noexcept;

struct LirMem {
    VReg base_vreg;
    PReg base_preg;
    VReg index_vreg;
    PReg index_preg;
    x64::Scale scale = x64::Scale::One;
    int32_t disp = 0;

    bool has_base() const noexcept { return base_vreg.is_valid() || base_preg.is_valid(); }
    bool has_index() const noexcept { return index_vreg.is_valid() || index_preg.is_valid(); }
    bool operator==(const LirMem& other) const noexcept = default;
};

struct LirOperand {
    LirOperandKind kind = LirOperandKind::None;
    VReg vreg_val;
    PReg preg_val;
    int64_t imm_int = 0;
    double imm_float = 0.0;
    LirMem mem_val;
    int32_t spill_slot = -1;
    uint32_t label_id = 0;
    std::string symbol_name;
    x64::Condition cond = x64::Condition::None;
    uint8_t size = 8; // operand size in bytes

    static LirOperand vreg(VReg v, uint8_t sz = 8);
    static LirOperand preg(PReg p, uint8_t sz = 8);
    static LirOperand preg_gpr(x64::GPR g, uint8_t sz = 8);
    static LirOperand preg_xmm(x64::XMM x, uint8_t sz = 8);
    static LirOperand imm(int64_t v, uint8_t sz = 8);
    static LirOperand imm_f64(double v);
    static LirOperand mem(VReg base, int32_t disp = 0, uint8_t sz = 8);
    static LirOperand mem(PReg base, int32_t disp = 0, uint8_t sz = 8);
    static LirOperand mem(VReg base, VReg index, x64::Scale scale = x64::Scale::One, int32_t disp = 0, uint8_t sz = 8);
    static LirOperand mem(PReg base, PReg index, x64::Scale scale = x64::Scale::One, int32_t disp = 0, uint8_t sz = 8);
    static LirOperand mem_custom(const LirMem& m, uint8_t sz = 8);
    static LirOperand slot(int32_t slot_idx, uint8_t sz = 8);
    static LirOperand label(uint32_t id);
    static LirOperand symbol(std::string name);
    static LirOperand condition(x64::Condition c);

    bool is_none() const noexcept { return kind == LirOperandKind::None; }
    bool is_vreg() const noexcept { return kind == LirOperandKind::VReg; }
    bool is_preg() const noexcept { return kind == LirOperandKind::PReg; }
    bool is_imm_int() const noexcept { return kind == LirOperandKind::ImmInt; }
    bool is_imm_float() const noexcept { return kind == LirOperandKind::ImmFloat; }
    bool is_mem() const noexcept { return kind == LirOperandKind::Mem; }
    bool is_spill_slot() const noexcept { return kind == LirOperandKind::SpillSlot; }
    bool is_label() const noexcept { return kind == LirOperandKind::Label; }
    bool is_symbol() const noexcept { return kind == LirOperandKind::Symbol; }
    bool is_condition() const noexcept { return kind == LirOperandKind::Condition; }

    bool is_reg() const noexcept { return is_vreg() || is_preg(); }
};

std::string to_string(const LirOperand& op);

enum class LirOpcode : uint16_t {
    Nop,
    // Moves
    Mov,
    Mov32,
    Movabs,
    Movsx8,
    Movsx16,
    Movsxd,
    Movzx8,
    Movzx16,
    // ALU 64 & 32
    Add,
    Add32,
    Sub,
    Sub32,
    Imul,
    Imul32,
    Idiv,
    Idiv32,
    Div,
    Div32,
    Cdq,
    Cqo,
    And,
    And32,
    Or,
    Or32,
    Xor,
    Xor32,
    Not,
    Not32,
    Neg,
    Neg32,
    Shl,
    Shl32,
    Shr,
    Shr32,
    Sar,
    Sar32,
    Popcnt,
    Popcnt32,
    Lzcnt,
    Lzcnt32,
    Tzcnt,
    Tzcnt32,
    Bsr,
    Bsr32,
    Bsf,
    Bsf32,
    // Compare & Test
    Cmp,
    Cmp32,
    Test,
    Test32,
    Setcc,
    Cmovcc,
    // SSE
    Movsd,
    Movss,
    Movq_gx,  // GPR -> XMM
    Movq_xg,  // XMM -> GPR
    Addsd,
    Addss,
    Subsd,
    Subss,
    Mulsd,
    Mulss,
    Divsd,
    Divss,
    Sqrtsd,
    Sqrtss,
    Ucomisd,
    Ucomiss,
    Xorpd,
    Cvtsi2sd,
    Cvtsi2sd32,
    Cvttsd2si,
    Cvttsd2si32,
    // 128-bit SIMD Vector
    Movaps,
    Movups,
    Movd_xg,
    Movd_gx,
    Addps,
    Subps,
    Mulps,
    Divps,
    Minps,
    Maxps,
    Sqrtps,
    Addpd,
    Subpd,
    Mulpd,
    Divpd,
    Minpd,
    Maxpd,
    Sqrtpd,
    Paddd,
    Psubd,
    Pmulld,
    Pminsd,
    Pmaxsd,
    Paddq,
    Psubq,
    Pand,
    Por,
    Pxor,
    Pandn,
    Pcmpeqd,
    Pslld,
    Psllq,
    Shufps,
    Shufpd,
    Pshufd,
    Movddup,
    Pinsrd,
    Pextrd,
    Pinsrq,
    Pextrq,
    Insertps,
    Extractps,
    Xorps,
    // Branches & Calls
    Jmp,
    Jcc,
    Call,
    CallIndirect,
    Ret,
    Push,
    Pop,
    Lea,
    // Parallel Copy & Runtime
    ParallelCopy,
    Safepoint,
    GuardExit
};

std::string_view to_string(LirOpcode op) noexcept;

struct FixedConstraint {
    bool has_fixed_preg = false;
    PReg fixed_preg;

    static constexpr FixedConstraint none() noexcept {
        return FixedConstraint{false, PReg{}};
    }
    static constexpr FixedConstraint preg(PReg p) noexcept {
        return FixedConstraint{true, p};
    }
    static constexpr FixedConstraint gpr(x64::GPR g) noexcept {
        return FixedConstraint{true, PReg::gpr(g)};
    }
    static constexpr FixedConstraint xmm(x64::XMM x) noexcept {
        return FixedConstraint{true, PReg::xmm(x)};
    }
};

class LirInst {
public:
    uint32_t id = 0;
    LirOpcode opcode = LirOpcode::Nop;
    std::vector<LirOperand> defs;
    std::vector<LirOperand> uses;
    std::vector<FixedConstraint> def_constraints;
    std::vector<FixedConstraint> use_constraints;
    x64::RegMask clobbered_gprs = 0;
    x64::RegMask clobbered_xmms = 0;
    x64::Condition condition = x64::Condition::None;
    const Instruction* mir_origin = nullptr;
    uint32_t safepoint_id = 0;
    uint32_t resume_id = 0;
    uint32_t deopt_reason = 0;
    std::string exit_symbol;
    bool is_patchable = false;
    std::string patch_symbol;
    std::string callee_symbol;
    std::vector<VReg> live_gcrefs;

    LirInst() = default;
    explicit LirInst(LirOpcode op) : opcode(op) {}

    bool is_terminator() const noexcept;
    bool is_branch() const noexcept;
    bool is_call() const noexcept;

    void add_def(LirOperand op, FixedConstraint constraint = FixedConstraint::none()) {
        defs.push_back(std::move(op));
        def_constraints.push_back(constraint);
    }

    void add_use(LirOperand op, FixedConstraint constraint = FixedConstraint::none()) {
        uses.push_back(std::move(op));
        use_constraints.push_back(constraint);
    }
};

std::string to_string(const LirInst& inst);

class LirBlock {
public:
    uint32_t id = 0;
    std::string name;
    std::vector<std::unique_ptr<LirInst>> instructions;
    std::vector<LirBlock*> predecessors;
    std::vector<LirBlock*> successors;
    uint32_t loop_depth = 0;
    x64::Label x64_label;

    LirBlock() = default;
    LirBlock(uint32_t id, std::string name) : id(id), name(std::move(name)) {}

    void append_inst(std::unique_ptr<LirInst> inst) {
        instructions.push_back(std::move(inst));
    }
};

std::string to_string(const LirBlock& block);

struct VRegInfo {
    VReg vreg;
    PReg assigned_preg;
    int32_t assigned_spill_slot = -1;
    bool is_spilled = false;
};

struct FrameInfo {
    size_t num_spill_slots = 0;
    std::vector<bool> spill_slot_is_gcref;
    x64::RegMask saved_callee_gprs = 0;
    x64::RegMask saved_callee_xmms = 0;
    size_t outgoing_arg_space = 0;
    size_t total_frame_size = 0;
    bool has_calls = false;
    bool is_leaf = false;

    int32_t spill_slot_offset(int32_t slot_idx) const noexcept;
};

class LirFunction {
public:
    std::string name;
    Type return_type = Type::void_type();
    CallingConvention calling_conv;
    std::vector<std::unique_ptr<LirBlock>> blocks;
    std::vector<VRegInfo> vreg_table;
    FrameInfo frame;
    std::vector<std::pair<uint32_t, uint32_t>> resume_entries;

    LirFunction() = default;

    VReg allocate_vreg(RegClass rc, uint8_t size, bool is_gcref = false);
    LirBlock* create_block(std::string name = "");
    LirBlock* entry_block() const { return blocks.empty() ? nullptr : blocks.front().get(); }
    LirBlock* get_block_by_id(uint32_t id) const;

    const VRegInfo& get_vreg_info(VReg v) const;
    VRegInfo& get_vreg_info(VReg v);
};

std::string to_string(const LirFunction& fn);

} // namespace brass::codegen
