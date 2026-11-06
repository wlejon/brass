#include <brass/codegen/lir.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <cstdint>
#include <optional>
#include <unordered_map>

// The host keeps some 64-bit constants at fixed offsets from the pinned TLS
// register (Module::pinned_constants). A constant the isel made as a
// `movabs v, C` is then read where it is used: `xor rax, [r13 - 0x80]`
// rather than `movabs r9, C; xor rax, r9`, or a register held across the
// function for it. The read is four bytes, a byte more than the
// register-register form, but the constant takes no register, so the values
// it displaced stay out of spill slots, and no 10-byte movabs or 7-byte pool
// load makes it. Only uses an x64 instruction can take from memory fold: an
// ALU operation's source, a compare's right operand, a move into a register;
// the rest keep the movabs, which the emitter turns into a load from the
// same word.

namespace brass::x64 {

using namespace brass::codegen;

namespace {

// The use of `in` at `k` may become a memory operand: the instruction's only
// one, in a position its x64 encoding reads from memory.
bool foldable_use(const LirInst& in, size_t k) {
    switch (in.opcode) {
        case LirOpcode::And:
        case LirOpcode::Or:
        case LirOpcode::Xor:
            // dst = dst op src: the source is the last of two uses.
            if (in.uses.size() != 2 || k != 1 || in.defs.size() != 1 || !in.defs[0].is_vreg()) return false;
            break;
        case LirOpcode::Cmp:
            if (in.uses.size() != 2 || k != 1 || !in.defs.empty()) return false;
            break;
        case LirOpcode::Mov:
            if (in.uses.size() != 1 || in.defs.size() != 1 || !in.defs[0].is_vreg() || in.is_patchable) return false;
            break;
        default:
            return false;
    }
    for (size_t j = 0; j < in.uses.size(); ++j) {
        const LirOperand& u = in.uses[j];
        if (j == k) continue;
        if (u.is_mem() || u.is_spill_slot() || u.is_local_slot()) return false;
        // The same constant twice (`cmp v, v`) would be two memory operands.
        if (u.is_vreg() && in.uses[k].is_vreg() && u.vreg_val == in.uses[k].vreg_val) return false;
    }
    return in.uses[k].size == 8;
}

} // namespace

void fold_pinned_constants(LirFunction& fn) {
    const uint32_t pinned = reg_mask(kPinnedTlsGpr);
    if (fn.pinned_constants.empty() || !(fn.reserved_gprs & pinned) || (fn.forced_saved_gprs & pinned)) return;
    auto offset_of = [&](int64_t imm) -> std::optional<int32_t> {
        // A value with a 32-bit encoding is better as an immediate.
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
                // A function that writes the register reads the caller's
                // value before the write.
                if (d.is_preg() && d.preg_val.is_gpr() && d.preg_val.as_gpr() == kPinnedTlsGpr) return;
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
        it = def_count[it->first] == 1 ? std::next(it) : constants.erase(it);
    }
    if (constants.empty()) return;
    for (const auto& block : fn.blocks) {
        for (size_t i = 0; i < block->instructions.size(); ++i) {
            LirInst& in = *block->instructions[i];
            auto keep = [&](VReg v) {
                auto it = constants.find(v.id);
                if (it != constants.end()) it->second.kept = true;
            };
            // An address a store writes through names its registers too.
            for (const LirOperand& d : in.defs) {
                if (!d.is_mem()) continue;
                if (d.mem_val.base_vreg.is_valid()) keep(d.mem_val.base_vreg);
                if (d.mem_val.index_vreg.is_valid()) keep(d.mem_val.index_vreg);
            }
            for (size_t k = 0; k < in.uses.size(); ++k) {
                LirOperand& u = in.uses[k];
                if (u.is_mem()) {
                    if (u.mem_val.base_vreg.is_valid()) keep(u.mem_val.base_vreg);
                    if (u.mem_val.index_vreg.is_valid()) keep(u.mem_val.index_vreg);
                    continue;
                }
                if (!u.is_vreg()) continue;
                auto it = constants.find(u.vreg_val.id);
                if (it == constants.end()) continue;
                if (!foldable_use(in, k)) {
                    it->second.kept = true;
                    continue;
                }
                u = LirOperand::mem(PReg::gpr(kPinnedTlsGpr), it->second.offset, 8);
                if (k < in.use_constraints.size()) in.use_constraints[k] = FixedConstraint::none();
            }
        }
    }
    // A constant every use of which now reads memory needs no register.
    for (const auto& [id, c] : constants) {
        if (!c.kept) c.block->instructions[c.pos].reset();
    }
    for (const auto& block : fn.blocks) {
        auto& ins = block->instructions;
        size_t out = 0;
        for (size_t i = 0; i < ins.size(); ++i) {
            if (ins[i]) ins[out++] = std::move(ins[i]);
        }
        ins.resize(out);
    }
}

} // namespace brass::x64
