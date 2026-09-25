#include <brass/codegen/lir.hpp>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <unordered_map>

namespace brass::codegen {

std::string_view to_string(RegClass rc) noexcept {
    switch (rc) {
        case RegClass::GPR: return "GPR";
        case RegClass::XMM: return "XMM";
    }
    return "Unknown";
}

std::string_view to_string(LirOperandKind kind) noexcept {
    switch (kind) {
        case LirOperandKind::None: return "none";
        case LirOperandKind::VReg: return "vreg";
        case LirOperandKind::PReg: return "preg";
        case LirOperandKind::ImmInt: return "imm_int";
        case LirOperandKind::ImmFloat: return "imm_float";
        case LirOperandKind::Mem: return "mem";
        case LirOperandKind::SpillSlot: return "spill_slot";
        case LirOperandKind::LocalSlot: return "local_slot";
        case LirOperandKind::Label: return "label";
        case LirOperandKind::Symbol: return "symbol";
        case LirOperandKind::Condition: return "cond";
    }
    return "unknown";
}

std::string_view to_string(LirOpcode op) noexcept {
    switch (op) {
        case LirOpcode::Nop: return "nop";
        case LirOpcode::KeepAlive: return "keep_alive";
        case LirOpcode::Mov: return "mov";
        case LirOpcode::Mov32: return "mov32";
        case LirOpcode::Movabs: return "movabs";
        case LirOpcode::Movsx8: return "movsx8";
        case LirOpcode::Movsx16: return "movsx16";
        case LirOpcode::Movsxd: return "movsxd";
        case LirOpcode::Movzx8: return "movzx8";
        case LirOpcode::Movzx16: return "movzx16";
        case LirOpcode::Add: return "add";
        case LirOpcode::Add32: return "add32";
        case LirOpcode::Adds: return "adds";
        case LirOpcode::Adds32: return "adds32";
        case LirOpcode::Sub: return "sub";
        case LirOpcode::Sub32: return "sub32";
        case LirOpcode::Subs: return "subs";
        case LirOpcode::Subs32: return "subs32";
        case LirOpcode::Imul: return "imul";
        case LirOpcode::Imul32: return "imul32";
        case LirOpcode::Smulh: return "smulh";
        case LirOpcode::Umulh: return "umulh";
        case LirOpcode::Idiv: return "idiv";
        case LirOpcode::Idiv32: return "idiv32";
        case LirOpcode::Div: return "div";
        case LirOpcode::Div32: return "div32";
        case LirOpcode::Cdq: return "cdq";
        case LirOpcode::Cqo: return "cqo";
        case LirOpcode::And: return "and";
        case LirOpcode::And32: return "and32";
        case LirOpcode::Or: return "or";
        case LirOpcode::Or32: return "or32";
        case LirOpcode::Xor: return "xor";
        case LirOpcode::Xor32: return "xor32";
        case LirOpcode::Not: return "not";
        case LirOpcode::Not32: return "not32";
        case LirOpcode::Neg: return "neg";
        case LirOpcode::Neg32: return "neg32";
        case LirOpcode::Shl: return "shl";
        case LirOpcode::Shl32: return "shl32";
        case LirOpcode::Shr: return "shr";
        case LirOpcode::Shr32: return "shr32";
        case LirOpcode::Sar: return "sar";
        case LirOpcode::Sar32: return "sar32";
        case LirOpcode::Popcnt: return "popcnt";
        case LirOpcode::Popcnt32: return "popcnt32";
        case LirOpcode::Lzcnt: return "lzcnt";
        case LirOpcode::Lzcnt32: return "lzcnt32";
        case LirOpcode::Tzcnt: return "tzcnt";
        case LirOpcode::Tzcnt32: return "tzcnt32";
        case LirOpcode::Bsr: return "bsr";
        case LirOpcode::Bsr32: return "bsr32";
        case LirOpcode::Bsf: return "bsf";
        case LirOpcode::Bsf32: return "bsf32";
        case LirOpcode::Cmp: return "cmp";
        case LirOpcode::Cmp32: return "cmp32";
        case LirOpcode::Test: return "test";
        case LirOpcode::Test32: return "test32";
        case LirOpcode::Setcc: return "setcc";
        case LirOpcode::Cmovcc: return "cmovcc";
        case LirOpcode::Movsd: return "movsd";
        case LirOpcode::Movss: return "movss";
        case LirOpcode::Movq_gx: return "movq_gx";
        case LirOpcode::Movq_xg: return "movq_xg";
        case LirOpcode::Addsd: return "addsd";
        case LirOpcode::Addss: return "addss";
        case LirOpcode::Subsd: return "subsd";
        case LirOpcode::Subss: return "subss";
        case LirOpcode::Mulsd: return "mulsd";
        case LirOpcode::Mulss: return "mulss";
        case LirOpcode::Divsd: return "divsd";
        case LirOpcode::Divss: return "divss";
        case LirOpcode::Sqrtsd: return "sqrtsd";
        case LirOpcode::Sqrtss: return "sqrtss";
        case LirOpcode::Ucomisd: return "ucomisd";
        case LirOpcode::Ucomiss: return "ucomiss";
        case LirOpcode::Xorpd: return "xorpd";
        case LirOpcode::Fneg: return "fneg";
        case LirOpcode::Fneg32: return "fneg32";
        case LirOpcode::Cvtsi2sd: return "cvtsi2sd";
        case LirOpcode::Cvtsi2sd32: return "cvtsi2sd32";
        case LirOpcode::Cvttsd2si: return "cvttsd2si";
        case LirOpcode::Cvttsd2si32: return "cvttsd2si32";
        case LirOpcode::Cvtsi2ss: return "cvtsi2ss";
        case LirOpcode::Cvtsi2ss32: return "cvtsi2ss32";
        case LirOpcode::Cvttss2si: return "cvttss2si";
        case LirOpcode::Cvttss2si32: return "cvttss2si32";
        case LirOpcode::Cvtsd2ss: return "cvtsd2ss";
        case LirOpcode::Cvtss2sd: return "cvtss2sd";
        case LirOpcode::Floor32: return "floor32";
        case LirOpcode::Floor64: return "floor64";
        case LirOpcode::Ceil32: return "ceil32";
        case LirOpcode::Ceil64: return "ceil64";
        case LirOpcode::Round32: return "round32";
        case LirOpcode::Round64: return "round64";
        case LirOpcode::Fabs32: return "fabs32";
        case LirOpcode::Fabs64: return "fabs64";
        case LirOpcode::Minss: return "minss";
        case LirOpcode::Minsd: return "minsd";
        case LirOpcode::Maxss: return "maxss";
        case LirOpcode::Maxsd: return "maxsd";
        case LirOpcode::Movaps: return "movaps";
        case LirOpcode::Movups: return "movups";
        case LirOpcode::Movd_xg: return "movd_xg";
        case LirOpcode::Movd_gx: return "movd_gx";
        case LirOpcode::Addps: return "addps";
        case LirOpcode::Subps: return "subps";
        case LirOpcode::Mulps: return "mulps";
        case LirOpcode::Divps: return "divps";
        case LirOpcode::Minps: return "minps";
        case LirOpcode::Maxps: return "maxps";
        case LirOpcode::Sqrtps: return "sqrtps";
        case LirOpcode::Addpd: return "addpd";
        case LirOpcode::Subpd: return "subpd";
        case LirOpcode::Mulpd: return "mulpd";
        case LirOpcode::Divpd: return "divpd";
        case LirOpcode::Minpd: return "minpd";
        case LirOpcode::Maxpd: return "maxpd";
        case LirOpcode::Sqrtpd: return "sqrtpd";
        case LirOpcode::Fneg4s: return "fneg4s";
        case LirOpcode::Fneg2d: return "fneg2d";
        case LirOpcode::Paddd: return "paddd";
        case LirOpcode::Psubd: return "psubd";
        case LirOpcode::Pmulld: return "pmulld";
        case LirOpcode::Pminsd: return "pminsd";
        case LirOpcode::Pmaxsd: return "pmaxsd";
        case LirOpcode::Paddq: return "paddq";
        case LirOpcode::Psubq: return "psubq";
        case LirOpcode::Pand: return "pand";
        case LirOpcode::Por: return "por";
        case LirOpcode::Pxor: return "pxor";
        case LirOpcode::Pandn: return "pandn";
        case LirOpcode::Pnot: return "pnot";
        case LirOpcode::Pcmpeqd: return "pcmpeqd";
        case LirOpcode::Pslld: return "pslld";
        case LirOpcode::Psllq: return "psllq";
        case LirOpcode::Shufps: return "shufps";
        case LirOpcode::Shufpd: return "shufpd";
        case LirOpcode::Pshufd: return "pshufd";
        case LirOpcode::Movddup: return "movddup";
        case LirOpcode::Pinsrd: return "pinsrd";
        case LirOpcode::Pextrd: return "pextrd";
        case LirOpcode::Pinsrq: return "pinsrq";
        case LirOpcode::Pextrq: return "pextrq";
        case LirOpcode::Insertps: return "insertps";
        case LirOpcode::Extractps: return "extractps";
        case LirOpcode::Xorps: return "xorps";
        case LirOpcode::Vmovaps: return "vmovaps";
        case LirOpcode::Vmovups: return "vmovups";
        case LirOpcode::Vaddps: return "vaddps";
        case LirOpcode::Vsubps: return "vsubps";
        case LirOpcode::Vmulps: return "vmulps";
        case LirOpcode::Vdivps: return "vdivps";
        case LirOpcode::Vminps: return "vminps";
        case LirOpcode::Vmaxps: return "vmaxps";
        case LirOpcode::Vaddpd: return "vaddpd";
        case LirOpcode::Vsubpd: return "vsubpd";
        case LirOpcode::Vmulpd: return "vmulpd";
        case LirOpcode::Vdivpd: return "vdivpd";
        case LirOpcode::Vminpd: return "vminpd";
        case LirOpcode::Vmaxpd: return "vmaxpd";
        case LirOpcode::Vsqrtps: return "vsqrtps";
        case LirOpcode::Vsqrtpd: return "vsqrtpd";
        case LirOpcode::Vextractf128: return "vextractf128";
        case LirOpcode::Vinsertf128: return "vinsertf128";
        case LirOpcode::Vpaddd: return "vpaddd";
        case LirOpcode::Vpsubd: return "vpsubd";
        case LirOpcode::Vpmulld: return "vpmulld";
        case LirOpcode::Vpaddq: return "vpaddq";
        case LirOpcode::Vpsubq: return "vpsubq";
        case LirOpcode::Vandps: return "vandps";
        case LirOpcode::Vorps: return "vorps";
        case LirOpcode::Vxorps: return "vxorps";
        case LirOpcode::Vandpd: return "vandpd";
        case LirOpcode::Vorpd: return "vorpd";
        case LirOpcode::Vxorpd: return "vxorpd";
        case LirOpcode::Vpand: return "vpand";
        case LirOpcode::Vpor: return "vpor";
        case LirOpcode::Vpxor: return "vpxor";
        case LirOpcode::Vbroadcastss: return "vbroadcastss";
        case LirOpcode::Vbroadcastsd: return "vbroadcastsd";
        case LirOpcode::Vpbroadcastd: return "vpbroadcastd";
        case LirOpcode::Vpbroadcastq: return "vpbroadcastq";
        case LirOpcode::Vfmadd213ps: return "vfmadd213ps";
        case LirOpcode::Vfmadd231ps: return "vfmadd231ps";
        case LirOpcode::Vfmadd213pd: return "vfmadd213pd";
        case LirOpcode::Vfmadd231pd: return "vfmadd231pd";
        case LirOpcode::Vfmadd213ss: return "vfmadd213ss";
        case LirOpcode::Vfmadd231ss: return "vfmadd231ss";
        case LirOpcode::Vfmadd213sd: return "vfmadd213sd";
        case LirOpcode::Vfmadd231sd: return "vfmadd231sd";
        case LirOpcode::Jmp: return "jmp";
        case LirOpcode::Jcc: return "jcc";
        case LirOpcode::Call: return "call";
        case LirOpcode::CallIndirect: return "call_ind";
        case LirOpcode::Ret: return "ret";
        case LirOpcode::Push: return "push";
        case LirOpcode::Pop: return "pop";
        case LirOpcode::Lea: return "lea";
        case LirOpcode::ParallelCopy: return "parallel_copy";
        case LirOpcode::Safepoint: return "safepoint";
        case LirOpcode::WriteBarrier: return "write_barrier";
        case LirOpcode::GuardExit: return "guard_exit";
        case LirOpcode::Trap: return "trap";
    }
    return "unknown";
}

LirOperand LirOperand::vreg(VReg v, uint8_t sz) {
    LirOperand op;
    op.kind = LirOperandKind::VReg;
    op.vreg_val = v;
    op.size = sz;
    return op;
}

LirOperand LirOperand::preg(PReg p, uint8_t sz) {
    LirOperand op;
    op.kind = LirOperandKind::PReg;
    op.preg_val = p;
    op.size = sz;
    return op;
}

LirOperand LirOperand::preg_gpr(x64::GPR g, uint8_t sz) {
    return preg(PReg::gpr(g), sz);
}

LirOperand LirOperand::preg_xmm(x64::XMM x, uint8_t sz) {
    return preg(PReg::xmm(x), sz);
}

LirOperand LirOperand::preg_aarch64_gpr(aarch64::GPR g, uint8_t sz) {
    return preg(PReg::aarch64_gpr(g), sz);
}

LirOperand LirOperand::preg_aarch64_fpr(aarch64::FPR f, uint8_t sz) {
    return preg(PReg::aarch64_fpr(f), sz);
}

LirOperand LirOperand::imm(int64_t v, uint8_t sz) {
    LirOperand op;
    op.kind = LirOperandKind::ImmInt;
    op.imm_int = v;
    op.size = sz;
    return op;
}

LirOperand LirOperand::imm_f64(double v) {
    LirOperand op;
    op.kind = LirOperandKind::ImmFloat;
    op.imm_float = v;
    op.size = 8;
    return op;
}

LirOperand LirOperand::mem(VReg base, int32_t disp, uint8_t sz) {
    LirOperand op;
    op.kind = LirOperandKind::Mem;
    op.mem_val.base_vreg = base;
    op.mem_val.disp = disp;
    op.size = sz;
    return op;
}

LirOperand LirOperand::mem(PReg base, int32_t disp, uint8_t sz) {
    LirOperand op;
    op.kind = LirOperandKind::Mem;
    op.mem_val.base_preg = base;
    op.mem_val.disp = disp;
    op.size = sz;
    return op;
}

LirOperand LirOperand::mem(VReg base, VReg index, x64::Scale scale, int32_t disp, uint8_t sz) {
    LirOperand op;
    op.kind = LirOperandKind::Mem;
    op.mem_val.base_vreg = base;
    op.mem_val.index_vreg = index;
    op.mem_val.scale = scale;
    op.mem_val.disp = disp;
    op.size = sz;
    return op;
}

LirOperand LirOperand::mem(PReg base, PReg index, x64::Scale scale, int32_t disp, uint8_t sz) {
    LirOperand op;
    op.kind = LirOperandKind::Mem;
    op.mem_val.base_preg = base;
    op.mem_val.index_preg = index;
    op.mem_val.scale = scale;
    op.mem_val.disp = disp;
    op.size = sz;
    return op;
}

LirOperand LirOperand::mem_custom(const LirMem& m, uint8_t sz) {
    LirOperand op;
    op.kind = LirOperandKind::Mem;
    op.mem_val = m;
    op.size = sz;
    return op;
}

LirOperand LirOperand::slot(int32_t slot_idx, uint8_t sz) {
    LirOperand op;
    op.kind = LirOperandKind::SpillSlot;
    op.spill_slot = slot_idx;
    op.size = sz;
    return op;
}

LirOperand LirOperand::local_slot(int32_t offset, uint8_t sz) {
    LirOperand op;
    op.kind = LirOperandKind::LocalSlot;
    op.local_offset = offset;
    op.size = sz;
    return op;
}

LirOperand LirOperand::label(uint32_t id) {
    LirOperand op;
    op.kind = LirOperandKind::Label;
    op.label_id = id;
    return op;
}

LirOperand LirOperand::symbol(std::string name) {
    LirOperand op;
    op.kind = LirOperandKind::Symbol;
    op.symbol_name = std::move(name);
    return op;
}

LirOperand LirOperand::condition(x64::Condition c) {
    LirOperand op;
    op.kind = LirOperandKind::Condition;
    op.cond = c;
    return op;
}

std::string to_string(const LirOperand& op) {
    std::ostringstream ss;
    switch (op.kind) {
        case LirOperandKind::None:
            ss << "<none>";
            break;
        case LirOperandKind::VReg:
            if (op.vreg_val.is_xmm()) {
                ss << "%vx" << op.vreg_val.id;
            } else {
                ss << "%v" << op.vreg_val.id;
            }
            if (op.vreg_val.is_gcref) ss << "(gc)";
            break;
        case LirOperandKind::PReg:
            if (op.preg_val.is_gpr()) {
                if (op.preg_val.code < 16) {
                    ss << x64::to_string(op.preg_val.as_gpr(),
                        op.size == 4 ? x64::OperandSize::Dword : x64::OperandSize::Qword);
                } else {
                    ss << aarch64::to_string(op.preg_val.as_aarch64_gpr(),
                        op.size == 4 ? aarch64::OperandSize::Word : aarch64::OperandSize::Xword);
                }
            } else if (op.preg_val.is_xmm()) {
                if (op.preg_val.code < 16) {
                    ss << x64::to_string(op.preg_val.as_xmm());
                } else {
                    ss << aarch64::to_string(op.preg_val.as_aarch64_fpr(),
                        op.size == 4 ? aarch64::OperandSize::Word : aarch64::OperandSize::Xword);
                }
            } else {
                ss << "preg(" << static_cast<int>(op.preg_val.code) << ")";
            }
            break;
        case LirOperandKind::ImmInt:
            ss << "$" << op.imm_int;
            break;
        case LirOperandKind::ImmFloat:
            ss << "$" << op.imm_float;
            break;
        case LirOperandKind::Mem: {
            ss << "[";
            bool has_prev = false;
            if (op.mem_val.base_preg.is_valid()) {
                if (op.mem_val.base_preg.code < 16) {
                    ss << x64::to_string(op.mem_val.base_preg.as_gpr());
                } else {
                    ss << aarch64::to_string(op.mem_val.base_preg.as_aarch64_gpr());
                }
                has_prev = true;
            } else if (op.mem_val.base_vreg.is_valid()) {
                ss << "%v" << op.mem_val.base_vreg.id;
                has_prev = true;
            }
            if (op.mem_val.index_preg.is_valid()) {
                if (has_prev) ss << " + ";
                if (op.mem_val.index_preg.code < 16) {
                    ss << x64::to_string(op.mem_val.index_preg.as_gpr());
                } else {
                    ss << aarch64::to_string(op.mem_val.index_preg.as_aarch64_gpr());
                }
                if (op.mem_val.scale != x64::Scale::One) {
                    ss << "*" << static_cast<int>(op.mem_val.scale);
                }
                has_prev = true;
            } else if (op.mem_val.index_vreg.is_valid()) {
                if (has_prev) ss << " + ";
                ss << "%v" << op.mem_val.index_vreg.id;
                if (op.mem_val.scale != x64::Scale::One) {
                    ss << "*" << static_cast<int>(op.mem_val.scale);
                }
                has_prev = true;
            }
            if (op.mem_val.disp != 0 || !has_prev) {
                if (has_prev) {
                    if (op.mem_val.disp > 0) ss << " + " << op.mem_val.disp;
                    else ss << " - " << (-op.mem_val.disp);
                } else {
                    ss << op.mem_val.disp;
                }
            }
            ss << "]";
            break;
        }
        case LirOperandKind::SpillSlot:
            ss << "slot(" << op.spill_slot << ")";
            break;
        case LirOperandKind::LocalSlot:
            ss << "local(" << op.local_offset << ")";
            break;
        case LirOperandKind::Label:
            ss << "L" << op.label_id;
            break;
        case LirOperandKind::Symbol:
            ss << "@" << op.symbol_name;
            break;
        case LirOperandKind::Condition:
            ss << x64::to_string(op.cond);
            break;
    }
    return ss.str();
}

bool LirInst::is_terminator() const noexcept {
    return opcode == LirOpcode::Ret || opcode == LirOpcode::Jmp ||
           opcode == LirOpcode::Jcc || opcode == LirOpcode::GuardExit ||
           opcode == LirOpcode::Trap;
}

bool LirInst::is_branch() const noexcept {
    return opcode == LirOpcode::Jmp || opcode == LirOpcode::Jcc;
}

bool LirInst::is_call() const noexcept {
    return opcode == LirOpcode::Call || opcode == LirOpcode::CallIndirect;
}

std::string to_string(const LirInst& inst) {
    std::ostringstream ss;
    ss << std::setw(4) << inst.id << ": ";
    if (!inst.defs.empty()) {
        for (size_t i = 0; i < inst.defs.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << to_string(inst.defs[i]);
            if (i < inst.def_constraints.size() && inst.def_constraints[i].has_fixed_preg) {
                ss << "=" << to_string(LirOperand::preg(inst.def_constraints[i].fixed_preg));
            }
        }
        ss << " = ";
    }
    ss << to_string(inst.opcode);
    if (inst.condition != x64::Condition::None) {
        ss << "." << x64::to_string(inst.condition);
    }
    if (!inst.uses.empty()) {
        ss << " ";
        for (size_t i = 0; i < inst.uses.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << to_string(inst.uses[i]);
            if (i < inst.use_constraints.size() && inst.use_constraints[i].has_fixed_preg) {
                ss << "=" << to_string(LirOperand::preg(inst.use_constraints[i].fixed_preg));
            }
        }
    }
    if (inst.opcode == LirOpcode::Safepoint) {
        ss << " (id=" << inst.safepoint_id << ", live_gc=[";
        for (size_t i = 0; i < inst.live_gcrefs.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << "%v" << inst.live_gcrefs[i].id;
        }
        ss << "])";
    } else if (inst.opcode == LirOpcode::GuardExit) {
        ss << " (resume_id=" << inst.resume_id << ")";
    }
    return ss.str();
}

std::string to_string(const LirBlock& block) {
    std::ostringstream ss;
    ss << "L" << block.id << " (" << (block.name.empty() ? "block" : block.name) << "):\n";
    for (const auto& inst : block.instructions) {
        ss << "    " << to_string(*inst) << "\n";
    }
    return ss.str();
}

int32_t FrameInfo::spill_slot_offset(int32_t slot_idx) const noexcept {
    // Callee-saved GPRs are saved at [rbp - 8], [rbp - 16], ...
    size_t num_callee_gprs = 0;
    for (int i = 0; i < 16; ++i) {
        if (saved_callee_gprs & (1u << i)) {
            if (i != static_cast<int>(x64::GPR::RSP) && i != static_cast<int>(x64::GPR::RBP)) {
                num_callee_gprs++;
            }
        }
    }
    size_t num_callee_xmms = 0;
    for (int i = 0; i < 16; ++i) {
        if (saved_callee_xmms & (1u << i)) num_callee_xmms++;
    }

    bool needs_gpr_align = (num_callee_xmms > 0) || (num_spill_slots > 0) || (local_frame_bytes > 0);
    size_t gpr_bytes = needs_gpr_align ? ((num_callee_gprs * 8 + 15) & ~size_t(15)) : (num_callee_gprs * 8);
    int32_t callee_offset = static_cast<int32_t>(gpr_bytes + num_callee_xmms * 16);
    return -(callee_offset + static_cast<int32_t>((slot_idx + 1) * 8));
}

VReg LirFunction::allocate_vreg(RegClass rc, uint8_t size, bool is_gcref, bool is_tagged) {
    uint32_t id = static_cast<uint32_t>(vreg_table.size());
    VReg v{id, rc, size, is_gcref || is_tagged, is_tagged};
    VRegInfo info;
    info.vreg = v;
    vreg_table.push_back(info);
    return v;
}

void link_blocks(LirBlock& from, LirBlock& to) {
    if (std::find(from.successors.begin(), from.successors.end(), &to) == from.successors.end()) {
        from.successors.push_back(&to);
    }
    if (std::find(to.predecessors.begin(), to.predecessors.end(), &from) == to.predecessors.end()) {
        to.predecessors.push_back(&from);
    }
}

LirBlock* LirFunction::create_block_with_id(uint32_t id, std::string block_name) {
    if (id >= next_block_id_) {
        next_block_id_ = id + 1;
    }
    auto blk = std::make_unique<LirBlock>(id, std::move(block_name));
    LirBlock* ptr = blk.get();
    blocks.push_back(std::move(blk));
    return ptr;
}

LirBlock* LirFunction::create_block(std::string block_name) {
    for (const auto& b : blocks) {
        if (b && b->id >= next_block_id_) {
            next_block_id_ = b->id + 1;
        }
    }
    uint32_t id = next_block_id_++;
    auto blk = std::make_unique<LirBlock>(id, std::move(block_name));
    LirBlock* ptr = blk.get();
    blocks.push_back(std::move(blk));
    return ptr;
}

LirBlock* LirFunction::get_block_by_id(uint32_t id) const {
    if (id < blocks.size() && blocks[id] && blocks[id]->id == id) {
        return blocks[id].get();
    }
    // Off the fast path the lookup was a scan of every block, and instruction
    // selection asks once per edge: quadratic on a function whose MIR block
    // ids are sparse, which every optimized function's are.
    auto indexed = [&]() -> LirBlock* {
        const auto it = block_index_.find(id);
        if (it == block_index_.end()) return nullptr;
        const size_t pos = it->second;
        if (pos < blocks.size() && blocks[pos] && blocks[pos]->id == id) return blocks[pos].get();
        return nullptr;
    };
    if (LirBlock* b = indexed()) return b;
    block_index_.clear();
    block_index_.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i]) block_index_.emplace(blocks[i]->id, i);
    }
    return indexed();
}

const VRegInfo& LirFunction::get_vreg_info(VReg v) const {
    if (v.id < vreg_table.size()) {
        return vreg_table[v.id];
    }
    throw std::out_of_range("Invalid VReg id in get_vreg_info");
}

VRegInfo& LirFunction::get_vreg_info(VReg v) {
    if (v.id < vreg_table.size()) {
        return vreg_table[v.id];
    }
    throw std::out_of_range("Invalid VReg id in get_vreg_info");
}

void LirFunction::sort_blocks_rpo() {
    if (blocks.size() <= 1) return;
    LirBlock* entry = entry_block();
    if (!entry) return;

    std::unordered_map<const LirBlock*, size_t> block_idx;
    block_idx.reserve(blocks.size());
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (blocks[i]) block_idx[blocks[i].get()] = i;
    }

    std::vector<bool> visited(blocks.size(), false);
    std::vector<LirBlock*> entry_po;
    entry_po.reserve(blocks.size());

    // An explicit stack, not recursion: a chain of many thousands of blocks
    // overflowed the native stack.
    std::vector<std::pair<LirBlock*, size_t>> stack;
    auto dfs = [&](LirBlock* root, std::vector<LirBlock*>& po) {
        auto enter = [&](LirBlock* b) {
            auto it = block_idx.find(b);
            if (it == block_idx.end() || visited[it->second]) return;
            visited[it->second] = true;
            stack.push_back({b, 0});
        };
        enter(root);
        while (!stack.empty()) {
            auto& [b, next] = stack.back();
            if (next == b->successors.size()) {
                po.push_back(b);
                stack.pop_back();
                continue;
            }
            LirBlock* succ = b->successors[next++];
            if (succ) enter(succ);
        }
    };

    dfs(entry, entry_po);
    std::reverse(entry_po.begin(), entry_po.end());

    std::vector<LirBlock*> other_po;
    for (const auto& entry_pair : resume_entries) {
        LirBlock* rb = get_block_by_id(entry_pair.second);
        if (rb) dfs(rb, other_po);
    }

    for (const auto& b : blocks) {
        if (b) dfs(b.get(), other_po);
    }

    std::reverse(other_po.begin(), other_po.end());

    std::vector<LirBlock*> full_order = std::move(entry_po);
    full_order.insert(full_order.end(), other_po.begin(), other_po.end());

    std::vector<std::unique_ptr<LirBlock>> new_blocks;
    new_blocks.reserve(blocks.size());
    std::unordered_map<LirBlock*, std::unique_ptr<LirBlock>> map;
    for (auto& b : blocks) {
        map[b.get()] = std::move(b);
    }
    for (LirBlock* ptr : full_order) {
        auto it = map.find(ptr);
        if (it != map.end() && it->second) {
            new_blocks.push_back(std::move(it->second));
        }
    }
    blocks = std::move(new_blocks);
}

std::string to_string(const LirFunction& fn) {
    std::ostringstream ss;
    ss << "function " << fn.name << "() -> " << fn.return_type.name() << " {\n";
    for (const auto& block : fn.blocks) {
        ss << to_string(*block);
    }
    ss << "}\n";
    return ss.str();
}

} // namespace brass::codegen

