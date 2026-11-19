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

} // namespace brass::aarch64
