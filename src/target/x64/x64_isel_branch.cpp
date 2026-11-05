// A br_if's condition as flags: the comparison it reduces to, tested by the
// branch itself.
//
// Generated code often branches on a comparison wrapped in boolean glue: a
// front end's `!(a === b)` is `eq (eq a b) 0`, its ToBoolean of a 0/1 is
// `and c, 1`, and a join that merged a comparison's answer, once a pass has
// copied the join's test into the edge that computed it, is the same glue
// around a comparison in the edge's block. Lowered value-wise each layer is
// a setcc, a zero extension and a test. Peeled here, after every MIR pass
// has run, the branch reads the innermost comparison's flags:
//
//   and c, 1    is c        (c is a comparison: 0 or 1)
//   ne c, 0     is c
//   eq c, 0     is not c    (the inverted condition code)
//
// Inverting a condition code is exactly the negation of its comparison,
// floats included: a fused float compare is one flag condition (A, AE, P,
// NP: ucomis's unordered result sets ZF, PF and CF), so its inverse is true
// at NaN as `!(a < b)` is.
//
// A float `eq` / `ne` needs two flags, ZF and PF (unordered compares equal
// in ZF), and is not fused anywhere else. As a branch with no edge
// arguments it is ucomis and two jumps to one side:
//
//   eq (ordered):    jp F; je T; jmp F
//   ne (unordered):  jp T; jne T; jmp F
//
// Only single-use instructions of the branch's own block are peeled, so no
// value anything else reads is lost, and the comparison's operands are
// read where the comparison was.

#include <brass/target/x64/x64_isel.hpp>

namespace brass::x64 {

using namespace brass::codegen;

namespace {

bool is_comparison_value(const Value* v) {
    const Instruction* def = v && v->is_instruction() ? v->defining_instruction() : nullptr;
    return def && is_comparison(def->opcode()) && def->operand_count() == 2;
}

}  // namespace

bool X64ISel::find_peeled_branch(const Instruction& br) {
    const BasicBlock* bb = br.parent();
    const Value* cond = br.operand_count() > 0 ? br.operand(0) : nullptr;
    PeeledBranch pb;
    // A handful of layers: `!!x` and an `and 1` around it is three.
    for (int depth = 0; depth < 6 && cond; ++depth) {
        const Instruction* def = cond->is_instruction() ? cond->defining_instruction() : nullptr;
        if (!def || def->parent() != bb || def->operand_count() != 2) return false;
        if (use_count_[cond] != 1) return false;
        const Value* a = def->operand(0);
        const Value* b = def->operand(1);
        const ImmIntInfo ia = get_imm_int_info(a);
        const ImmIntInfo ib = get_imm_int_info(b);
        const Value* inner = nullptr;
        bool flips = false;
        if (def->opcode() == Opcode::and_) {
            if (ib.is_imm && ib.val == 1 && is_comparison_value(a)) inner = a;
            else if (ia.is_imm && ia.val == 1 && is_comparison_value(b)) inner = b;
        } else if ((def->opcode() == Opcode::eq || def->opcode() == Opcode::ne) && !a->type().is_float()) {
            if (ib.is_imm && ib.val == 0 && is_comparison_value(a)) inner = a;
            else if (ia.is_imm && ia.val == 0 && is_comparison_value(b)) inner = b;
            flips = def->opcode() == Opcode::eq;
        }
        if (!inner) {
            if (is_comparison(def->opcode())) pb.cmp = def;
            break;
        }
        pb.peeled.push_back(def);
        pb.negate = pb.negate != flips;
        cond = inner;
    }
    if (!pb.cmp) return false;
    const Type t = pb.cmp->operand(0) ? pb.cmp->operand(0)->type() : Type::void_type();
    const bool float_eq = t.is_float() && !t.is_vector() &&
                          (pb.cmp->opcode() == Opcode::eq || pb.cmp->opcode() == Opcode::ne) &&
                          !is_self_nan_test(*pb.cmp);
    if (float_eq) {
        // The two jumps to one side need that side to take no copies.
        if (!br.true_target().args.empty() || !br.false_target().args.empty()) return false;
        pb.parity = true;
    } else if (pb.peeled.empty() || !comparison_fusible(*pb.cmp)) {
        return false;
    }
    for (const Instruction* p : pb.peeled) skipped_insts_.insert(p);
    skipped_insts_.insert(pb.cmp);
    peeled_branches_.emplace(&br, std::move(pb));
    return true;
}

void X64ISel::count_peeled_folds(const PeeledBranch& pb, std::unordered_map<const Value*, uint32_t>& folded_uses) const {
    for (const Instruction* p : pb.peeled) {
        for (size_t i = 0; i < 2; ++i) {
            if (get_imm_int_info(p->operand(i)).is_imm) folded_uses[p->operand(i)]++;
        }
    }
}

void X64ISel::lower_parity_branch(const Instruction& inst, const PeeledBranch& pb, LirBlock& lir_bb) {
    const Instruction& cmp = *pb.cmp;
    uint8_t sz = static_cast<uint8_t>(cmp.operand(0)->type().size_in_bytes());
    if (sz == 0) sz = 8;
    auto ucomi = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Ucomiss : LirOpcode::Ucomisd);
    ucomi->add_use(LirOperand::vreg(get_vreg(cmp.operand(0)), sz));
    ucomi->add_use(LirOperand::vreg(get_vreg(cmp.operand(1)), sz));
    lir_bb.append_inst(std::move(ucomi));

    const bool ordered_eq = (cmp.opcode() == Opcode::eq) != pb.negate;
    const uint32_t t = inst.true_target().block->id();
    const uint32_t f = inst.false_target().block->id();
    auto jump = [&](LirOpcode op, Condition c, uint32_t label) {
        auto j = std::make_unique<LirInst>(op);
        j->condition = c;
        j->add_use(LirOperand::label(label));
        j->mir_origin = &inst;
        lir_bb.append_inst(std::move(j));
    };
    // Unordered: false for eq, true for ne; then the zero flag decides.
    jump(LirOpcode::Jcc, Condition::P, ordered_eq ? f : t);
    jump(LirOpcode::Jcc, ordered_eq ? Condition::E : Condition::NE, t);
    jump(LirOpcode::Jmp, Condition::None, f);
}

void X64ISel::lower_branch_if(const Instruction& inst, LirBlock& lir_bb) {
    if (auto chain = and_chains_.find(&inst); chain != and_chains_.end()) {
        lower_and_chain_branch(inst, chain->second, lir_bb);
        return;
    }
    const Value* cond_val = inst.operand(0);
    const auto& t_target = inst.true_target();
    const auto& f_target = inst.false_target();

    const Instruction* cmp_inst = cond_val ? cond_val->defining_instruction() : nullptr;
    // Only a comparison the analysis folded into this branch is recomputed
    // here; any other is a value in a register like every other condition.
    bool is_fused_cmp = fused_into(cmp_inst, inst);

    Condition branch_cond = Condition::NE;

    const auto peeled = peeled_branches_.find(&inst);
    if (peeled != peeled_branches_.end() && peeled->second.parity) {
        lower_parity_branch(inst, peeled->second, lir_bb);
        return;
    }
    if (peeled != peeled_branches_.end()) {
        const Condition c = append_fused_compare(*peeled->second.cmp, inst, lir_bb);
        branch_cond = peeled->second.negate ? invert(c) : c;
    } else if (is_fused_cmp) {
        branch_cond = append_fused_compare(*cmp_inst, inst, lir_bb);
    } else {
        VReg cond = get_vreg(cond_val);
        uint8_t sz = cond.size;
        auto test_inst = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
        test_inst->add_use(LirOperand::vreg(cond, sz));
        test_inst->add_use(LirOperand::vreg(cond, sz));
        lir_bb.append_inst(std::move(test_inst));
        branch_cond = Condition::NE;
    }

    auto emit_target_args = [&](LirBlock& bb, const BranchTarget& target) {
        if (target.args.empty()) return;
        if (target.args.size() == 1) {
            VReg arg_v = get_vreg(target.args[0]);
            VReg param_v = get_vreg(target.block->param(0));
            uint8_t sz = param_v.size;
            if (arg_v != param_v) {
                auto mov_inst = std::make_unique<LirInst>(
                    param_v.is_xmm() ? ((sz == 16) ? LirOpcode::Movaps : LirOpcode::Movsd) : (sz == 4 ? LirOpcode::Mov32 : LirOpcode::Mov)
                );
                mov_inst->add_def(LirOperand::vreg(param_v, sz));
                mov_inst->add_use(LirOperand::vreg(arg_v, sz));
                bb.append_inst(std::move(mov_inst));
            }
        } else {
            auto pcopy = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
            for (size_t i = 0; i < target.args.size(); ++i) {
                VReg arg_v = get_vreg(target.args[i]);
                VReg param_v = get_vreg(target.block->param(i));
                uint8_t sz = param_v.size;
                pcopy->add_def(LirOperand::vreg(param_v, sz));
                pcopy->add_use(LirOperand::vreg(arg_v, sz));
            }
            bb.append_inst(std::move(pcopy));
        }
    };

    if (t_target.args.empty() && f_target.args.empty()) {
        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = branch_cond;
        jcc_inst->add_use(LirOperand::label(t_target.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_inst->add_use(LirOperand::label(f_target.block->id()));
        jmp_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_inst));
    } else if (t_target.args.empty() || f_target.args.empty()) {
        // The side with arguments gets a block of its own for their copies,
        // so that the block ends in a `jcc` + `jmp` pair the emitter can
        // turn either way: a loop's head whose exit carries a value falls
        // into its body (one branch an iteration, not two), and a copy
        // allocation leaves empty is a jump that branches thread through.
        const bool true_side = !t_target.args.empty();
        const auto& with_args = true_side ? t_target : f_target;
        const auto& plain = true_side ? f_target : t_target;
        auto* trampoline = lir_fn_->create_block(true_side ? "br_if_true" : "br_if_false");
        link_blocks(lir_bb, *trampoline);
        if (auto* target_lir = lir_fn_->get_block_by_id(with_args.block->id())) link_blocks(*trampoline, *target_lir);

        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = true_side ? invert(branch_cond) : branch_cond;
        jcc_inst->add_use(LirOperand::label(plain.block->id()));
        lir_bb.append_inst(std::move(jcc_inst));

        auto jmp_inst = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_inst->add_use(LirOperand::label(trampoline->id));
        jmp_inst->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_inst));

        emit_target_args(*trampoline, with_args);
        auto jmp_on = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_on->add_use(LirOperand::label(with_args.block->id()));
        trampoline->append_inst(std::move(jmp_on));
    } else {
        auto* false_trampoline = lir_fn_->create_block("br_if_false");
        link_blocks(lir_bb, *false_trampoline);
        if (auto* f_lir = lir_fn_->get_block_by_id(f_target.block->id())) link_blocks(*false_trampoline, *f_lir);

        auto jcc_inst = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc_inst->condition = invert(branch_cond);
        jcc_inst->add_use(LirOperand::label(false_trampoline->id));
        lir_bb.append_inst(std::move(jcc_inst));

        emit_target_args(lir_bb, t_target);

        auto jmp_t = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_t->add_use(LirOperand::label(t_target.block->id()));
        jmp_t->mir_origin = &inst;
        lir_bb.append_inst(std::move(jmp_t));

        emit_target_args(*false_trampoline, f_target);

        auto jmp_f = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp_f->add_use(LirOperand::label(f_target.block->id()));
        false_trampoline->append_inst(std::move(jmp_f));
    }
}


}  // namespace brass::x64
