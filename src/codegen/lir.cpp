#include <brass/codegen/lir.hpp>
#include <sstream>
#include <iomanip>
#include <stdexcept>

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
        case LirOperandKind::Label: return "label";
        case LirOperandKind::Symbol: return "symbol";
        case LirOperandKind::Condition: return "cond";
    }
    return "unknown";
}

std::string_view to_string(LirOpcode op) noexcept {
    switch (op) {
        case LirOpcode::Nop: return "nop";
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
        case LirOpcode::Sub: return "sub";
        case LirOpcode::Sub32: return "sub32";
        case LirOpcode::Imul: return "imul";
        case LirOpcode::Imul32: return "imul32";
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
        case LirOpcode::Subsd: return "subsd";
        case LirOpcode::Mulsd: return "mulsd";
        case LirOpcode::Divsd: return "divsd";
        case LirOpcode::Sqrtsd: return "sqrtsd";
        case LirOpcode::Ucomisd: return "ucomisd";
        case LirOpcode::Xorpd: return "xorpd";
        case LirOpcode::Cvtsi2sd: return "cvtsi2sd";
        case LirOpcode::Cvtsi2sd32: return "cvtsi2sd32";
        case LirOpcode::Cvttsd2si: return "cvttsd2si";
        case LirOpcode::Cvttsd2si32: return "cvttsd2si32";
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
        case LirOpcode::GuardExit: return "guard_exit";
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
                ss << x64::to_string(op.preg_val.as_gpr(),
                    op.size == 4 ? x64::OperandSize::Dword : x64::OperandSize::Qword);
            } else if (op.preg_val.is_xmm()) {
                ss << x64::to_string(op.preg_val.as_xmm());
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
                ss << x64::to_string(op.mem_val.base_preg.as_gpr());
                has_prev = true;
            } else if (op.mem_val.base_vreg.is_valid()) {
                ss << "%v" << op.mem_val.base_vreg.id;
                has_prev = true;
            }
            if (op.mem_val.index_preg.is_valid()) {
                if (has_prev) ss << " + ";
                ss << x64::to_string(op.mem_val.index_preg.as_gpr());
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
           opcode == LirOpcode::Jcc || opcode == LirOpcode::GuardExit;
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

    int32_t callee_offset = static_cast<int32_t>(num_callee_gprs * 8 + num_callee_xmms * 16);
    return -(callee_offset + static_cast<int32_t>((slot_idx + 1) * 8));
}

VReg LirFunction::allocate_vreg(RegClass rc, uint8_t size, bool is_gcref) {
    uint32_t id = static_cast<uint32_t>(vreg_table.size());
    VReg v{id, rc, size, is_gcref};
    VRegInfo info;
    info.vreg = v;
    vreg_table.push_back(info);
    return v;
}

LirBlock* LirFunction::create_block(std::string block_name) {
    uint32_t id = static_cast<uint32_t>(blocks.size());
    auto blk = std::make_unique<LirBlock>(id, std::move(block_name));
    LirBlock* ptr = blk.get();
    blocks.push_back(std::move(blk));
    return ptr;
}

LirBlock* LirFunction::get_block_by_id(uint32_t id) const {
    if (id < blocks.size() && blocks[id] && blocks[id]->id == id) {
        return blocks[id].get();
    }
    for (const auto& b : blocks) {
        if (b && b->id == id) return b.get();
    }
    return nullptr;
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
