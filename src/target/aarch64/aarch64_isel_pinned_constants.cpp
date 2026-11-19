#include <brass/codegen/lir.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace brass::aarch64 {

using namespace brass::codegen;

// Host-pinned 64-bit constants live at fixed offsets from kPinnedTlsGpr (X28).
// On AArch64, ALU ops do not accept memory operands, so:
//   - Movq_xg reading a pinned constant vreg folds directly into Movsd [X28, #off]
//     (single 4-byte ldur Dd, [X28, #off]), dropping the GPR load.
//   - Multi-instruction Movabs constants kept in GPRs are loaded from [X28, #off]
//     at emission time (`aarch64_emit_mov.cpp`).
void fold_pinned_constants(LirFunction& fn) {
    const uint32_t pinned = reg_mask(kPinnedTlsGpr);
    if (fn.pinned_constants.empty() || !(fn.reserved_gprs & pinned) || (fn.forced_saved_gprs & pinned)) {
        return;
    }

    auto offset_of = [&](int64_t imm) -> std::optional<int32_t> {
        // Immediates with a 32-bit encoding stay inline.
        if (imm >= INT32_MIN && imm <= static_cast<int64_t>(UINT32_MAX)) return std::nullopt;
        for (const auto& [v, off] : fn.pinned_constants) {
            if (v == static_cast<uint64_t>(imm)) return off;
        }
        return std::nullopt;
    };

    struct Constant {
        int32_t offset = 0;
        LirBlock* block = nullptr;
        size_t pos = 0;
        bool kept = false;
    };

    std::unordered_map<uint32_t, Constant> constants;
    std::unordered_map<uint32_t, uint32_t> def_count;

    for (const auto& block : fn.blocks) {
        for (size_t i = 0; i < block->instructions.size(); ++i) {
            const LirInst& in = *block->instructions[i];
            for (const auto& d : in.defs) {
                if (d.is_vreg()) def_count[d.vreg_val.id]++;
                if (d.is_preg() && d.preg_val.is_gpr() && d.preg_val.as_aarch64_gpr() == kPinnedTlsGpr) {
                    return;
                }
            }
            if ((in.opcode != LirOpcode::Movabs && in.opcode != LirOpcode::Mov) || in.is_patchable ||
                in.defs.size() != 1 || in.uses.size() != 1 || !in.uses[0].is_imm_int()) {
                continue;
            }
            const LirOperand& d = in.defs[0];
            if (!d.is_vreg() || d.size != 8 || !d.vreg_val.is_gpr() || d.vreg_val.is_gcref) continue;
            if (const std::optional<int32_t> off = offset_of(in.uses[0].imm_int)) {
                constants[d.vreg_val.id] = Constant{*off, block.get(), i, false};
            }
        }
    }

    for (auto it = constants.begin(); it != constants.end();) {
        it = (def_count[it->first] == 1) ? std::next(it) : constants.erase(it);
    }
    if (constants.empty()) return;

    for (const auto& block : fn.blocks) {
        for (const auto& inst_ptr : block->instructions) {
            LirInst& in = *inst_ptr;
            auto keep = [&](VReg v) {
                if (!v.is_valid()) return;
                auto it = constants.find(v.id);
                if (it != constants.end()) it->second.kept = true;
            };
            for (const LirOperand& d : in.defs) {
                if (!d.is_mem()) continue;
                keep(d.mem_val.base_vreg);
                keep(d.mem_val.index_vreg);
            }
            for (size_t k = 0; k < in.uses.size(); ++k) {
                LirOperand& u = in.uses[k];
                if (u.is_mem()) {
                    keep(u.mem_val.base_vreg);
                    keep(u.mem_val.index_vreg);
                    continue;
                }
                if (!u.is_vreg()) continue;
                auto it = constants.find(u.vreg_val.id);
                if (it == constants.end()) continue;

                const bool fold_movq_xg =
                    (in.opcode == LirOpcode::Movq_xg && k == 0 && in.uses.size() == 1 &&
                     in.defs.size() == 1 && in.defs[0].is_vreg() && u.size == 8);
                if (!fold_movq_xg) {
                    it->second.kept = true;
                    continue;
                }
                in.opcode = LirOpcode::Movsd;
                u = LirOperand::mem(PReg::aarch64_gpr(kPinnedTlsGpr), it->second.offset, 8);
                if (k < in.use_constraints.size()) in.use_constraints[k] = FixedConstraint::none();
            }
        }
    }

    for (const auto& [id, c] : constants) {
        if (!c.kept) c.block->instructions[c.pos].reset();
    }
    for (const auto& block : fn.blocks) {
        auto& ins = block->instructions;
        ins.erase(std::remove(ins.begin(), ins.end(), nullptr), ins.end());
    }
}

void fold_reserved_register_copies(LirFunction& fn, uint32_t pinned_mask) {
    struct Copy {
        PReg reg;
        const LirBlock* block = nullptr;
        size_t pos = 0;
        uint32_t defs = 0;
        bool kept = false;
    };
    std::unordered_map<uint32_t, Copy> copies;
    std::unordered_map<uint32_t, uint32_t> def_count;

    // A pinned register written by the function (e.g. coroutine / runtime entry)
    // cannot be folded across reads.
    for (const auto& block : fn.blocks) {
        for (const auto& in : block->instructions) {
            for (const auto& d : in->defs) {
                if (d.is_preg() && d.preg_val.is_gpr()) {
                    const uint32_t bit = reg_mask(d.preg_val.as_aarch64_gpr());
                    if (bit != 0) pinned_mask &= ~bit;
                }
            }
        }
    }

    for (const auto& block : fn.blocks) {
        for (size_t i = 0; i < block->instructions.size(); ++i) {
            const LirInst& in = *block->instructions[i];
            for (const auto& d : in.defs) {
                if (d.is_vreg()) def_count[d.vreg_val.id]++;
            }
            if (in.opcode != LirOpcode::Mov || in.defs.size() != 1 || in.uses.size() != 1) continue;
            const LirOperand& d = in.defs[0];
            const LirOperand& u = in.uses[0];
            if (!d.is_vreg() || d.size != 8 || !u.is_preg() || !u.preg_val.is_gpr() || u.size != 8) continue;
            const GPR g = u.preg_val.as_aarch64_gpr();
            const bool pinned = (reg_mask(g) & pinned_mask) != 0;
            if (!pinned && g != GPR::SP) continue;
            copies[d.vreg_val.id] = Copy{u.preg_val, block.get(), i, 0, false};
        }
    }
    if (copies.empty()) return;
    for (auto& [id, c] : copies) c.defs = def_count[id];

    auto copy_of = [&](VReg v) -> Copy* {
        if (!v.is_valid()) return nullptr;
        auto it = copies.find(v.id);
        return (it == copies.end() || it->second.defs != 1) ? nullptr : &it->second;
    };

    auto sp_unmoved = [](const Copy& c, const LirBlock* block, size_t pos) {
        if (block != c.block || pos <= c.pos) return false;
        for (size_t k = c.pos + 1; k < pos; ++k) {
            const LirInst& m = *block->instructions[k];
            if (m.is_call() || m.opcode == LirOpcode::Push || m.opcode == LirOpcode::Pop) return false;
            for (const auto& d : m.defs) {
                if (d.is_preg() && d.preg_val.is_gpr() && d.preg_val.as_aarch64_gpr() == GPR::SP) return false;
            }
        }
        return true;
    };

    for (const auto& block : fn.blocks) {
        for (size_t i = 0; i < block->instructions.size(); ++i) {
            LirInst& in = *block->instructions[i];
            const bool sp_safe = !in.is_call() && in.opcode != LirOpcode::GuardExit;
            auto mem = [&](LirOperand& op) {
                LirMem& m = op.mem_val;
                if (Copy* c = copy_of(m.base_vreg)) {
                    if (c->reg.as_aarch64_gpr() != GPR::SP || (sp_safe && sp_unmoved(*c, block.get(), i))) {
                        m.base_preg = c->reg;
                        m.base_vreg = VReg{};
                    } else {
                        c->kept = true;
                    }
                }
                if (Copy* c = copy_of(m.index_vreg)) {
                    if (c->reg.as_aarch64_gpr() != GPR::SP) {
                        m.index_preg = c->reg;
                        m.index_vreg = VReg{};
                    } else {
                        c->kept = true;
                    }
                }
            };
            for (auto& d : in.defs) {
                if (d.is_mem()) mem(d);
            }
            const bool is_tls_readable_op =
                in.opcode == LirOpcode::Cmp || in.opcode == LirOpcode::Test || in.opcode == LirOpcode::Cmovcc ||
                in.opcode == LirOpcode::Csinc || in.opcode == LirOpcode::Csneg || in.opcode == LirOpcode::Csinv ||
                in.opcode == LirOpcode::Add || in.opcode == LirOpcode::Sub;
            for (size_t k = 0; k < in.uses.size(); ++k) {
                LirOperand& u = in.uses[k];
                if (u.is_mem()) {
                    mem(u);
                    continue;
                }
                if (!u.is_vreg()) continue;
                Copy* c = copy_of(u.vreg_val);
                if (!c) continue;
                if (c->block == block.get() && c->pos == i) continue;
                const bool fixed = k < in.use_constraints.size() && in.use_constraints[k].has_fixed_preg;
                const bool is_sp = (c->reg.as_aarch64_gpr() == GPR::SP);
                const bool can_fold = is_sp ? (in.opcode == LirOpcode::Cmp && k == 0 && !fixed && u.size == 8 &&
                                               sp_unmoved(*c, block.get(), i))
                                            : (is_tls_readable_op && !fixed && u.size == 8);
                if (can_fold) {
                    u = LirOperand::preg(c->reg, 8);
                } else {
                    c->kept = true;
                }
            }
        }
    }

    for (const auto& [id, c] : copies) {
        if (c.defs != 1 || c.kept) continue;
        const_cast<LirBlock*>(c.block)->instructions[c.pos].reset();
    }
    for (const auto& block : fn.blocks) {
        auto& ins = block->instructions;
        ins.erase(std::remove(ins.begin(), ins.end(), nullptr), ins.end());
    }
}

} // namespace brass::aarch64
