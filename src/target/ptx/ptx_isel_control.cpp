// PtxISel: terminators and block-argument edges.
//
// MIR blocks take parameters and branch targets carry arguments, so every
// edge is a parallel copy (all sources read before any destination is
// written). The resolver below sequentializes it: copies whose destination
// nobody else still reads go first; when only cycles remain, one source is
// saved to a scratch register of the same class to break the cycle.
//
// For br_if the taken edge's copies are guarded on the branch predicate and
// emitted before the guarded `bra`; the fall-through edge's copies follow
// unguarded, so each edge's copies only execute when that edge is taken.

#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace brass::ptx {

// ---------------------------------------------------------------------------
// Edge copies
// ---------------------------------------------------------------------------

std::vector<PtxISel::Copy> PtxISel::edge_copies(const brass::Instruction& inst,
                                                const BranchTarget& target) const {
    std::vector<Copy> copies;
    if (!target.block) malformed(inst, "missing branch target");
    if (target.args.empty()) return copies;

    const auto& params = target.block->params();
    if (params.size() != target.args.size()) {
        throw std::runtime_error("PtxISel: " + std::string(brass::opcode_name(inst.opcode())) + " to block " +
                                 std::to_string(target.block->id()) + " passes " +
                                 std::to_string(target.args.size()) + " argument(s) but the block takes " +
                                 std::to_string(params.size()));
    }

    for (size_t i = 0; i < params.size(); ++i) {
        const Value* param = params[i];
        const Value* arg = target.args[i];
        if (!param || !arg) malformed(inst, "null block parameter or argument");
        if (param == arg) continue;
        const auto& dst = regs_of(param, "block parameter");
        const auto& src = regs_of(arg, "branch argument");
        if (dst.size() != src.size()) malformed(inst, "branch argument width does not match the block parameter");
        Type t = bit_type_for(param->type());
        for (size_t lane = 0; lane < dst.size(); ++lane) {
            if (dst[lane] == src[lane]) continue;
            copies.push_back(Copy{dst[lane], src[lane], t});
        }
    }
    return copies;
}

void PtxISel::emit_parallel_copies(std::vector<Copy> pending, const Reg* guard) {
    auto mov = [&](Reg dst, Reg src, Type t) {
        Inst m = Inst::make(Opcode::mov, t).dst(dst).src(src);
        if (guard) m.guard(*guard);
        emit(std::move(m));
    };

    auto is_still_read = [&](Reg r) {
        return std::any_of(pending.begin(), pending.end(), [&](const Copy& c) { return c.src == r; });
    };

    while (!pending.empty()) {
        // A copy whose destination is no longer read by a pending copy is safe.
        auto ready = std::find_if(pending.begin(), pending.end(),
                                  [&](const Copy& c) { return !is_still_read(c.dst); });
        if (ready != pending.end()) {
            mov(ready->dst, ready->src, ready->type);
            pending.erase(ready);
            continue;
        }

        // Only cycles remain: park one source in a scratch register and
        // redirect every reader of it there, which frees that register to
        // be written and unblocks the cycle.
        Copy& victim = pending.front();
        Reg scratch = fn_->new_reg(victim.src.cls);
        Reg old_src = victim.src;
        mov(scratch, old_src, victim.type);
        for (Copy& c : pending) {
            if (c.src == old_src) c.src = scratch;
        }
    }
}

// ---------------------------------------------------------------------------
// Terminators
// ---------------------------------------------------------------------------

void PtxISel::lower_branch(const brass::Instruction& inst) {
    const BranchTarget& target = inst.branch_target();
    emit_parallel_copies(edge_copies(inst, target), nullptr);
    emit(Inst::make(Opcode::bra).src(label_of(target.block)));
}

void PtxISel::lower_branch_if(const brass::Instruction& inst) {
    if (!inst.operand(0)) malformed(inst, "missing condition");
    const BranchTarget& taken = inst.true_target();
    const BranchTarget& fallthrough = inst.false_target();
    if (!taken.block || !fallthrough.block) malformed(inst, "missing branch target");

    Reg p = materialize_pred(inst.operand(0));

    emit_parallel_copies(edge_copies(inst, taken), &p);
    emit(Inst::make(Opcode::bra).guard(p).src(label_of(taken.block)));

    emit_parallel_copies(edge_copies(inst, fallthrough), nullptr);
    emit(Inst::make(Opcode::bra).src(label_of(fallthrough.block)));
}

void PtxISel::lower_return(const brass::Instruction& inst) {
    // Non-void kernels were rejected in lower(); a stray operand here would be
    // a verifier error, so drop nothing silently.
    if (inst.operand_count() != 0 && inst.operand(0) != nullptr) {
        throw std::runtime_error("PtxISel: ret with an operand inside kernel '" + fn_->name +
                                 "'; .entry kernels cannot return values");
    }
    emit(Inst::make(Opcode::ret));
}

void PtxISel::lower_unreachable(const brass::Instruction&) {
    emit(Inst::make(Opcode::trap));
}

} // namespace brass::ptx
