// NaN-CANONICALIZING STORES. A NaN-boxing front end stores a double as its
// bits with every NaN replaced by one canonical pattern:
//
//     store [m], select(ne f, f, K, bitcast_i64_f64 f)
//
// Instead of moving f to a GPR, running csel on K, and storing from GPR, keep
// everything in the FP domain:
//
//     fcmp  d_f, d_f
//     fcsel d_out, d_k, d_f, vs   ; d_k cached per stretch (ldur [x28, #off] or literal)
//     str   d_out, [m]

#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/mir/instruction.hpp>

namespace brass::aarch64 {

using namespace brass::codegen;

namespace {

const Instruction* def_of(const Value* v) {
    return v && v->is_instruction() ? v->defining_instruction() : nullptr;
}

}  // namespace

void AArch64ISel::find_nan_canon_stores(const Function& mir_fn, std::unordered_map<const Value*, uint32_t>& folded_uses) {
    nan_canon_stores_.clear();
    xmm_constants_.clear();
    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            const Opcode op = inst->opcode();
            if ((op != Opcode::store && op != Opcode::store_indexed) || skipped_insts_.count(inst)) continue;
            const Value* src = inst->operand(op == Opcode::store ? 1 : 2);
            const Instruction* wrap = nullptr;
            if (const Instruction* d0 = def_of(src); d0 && d0->parent() == bb &&
                (d0->opcode() == Opcode::bitcast_i64_tagged || d0->opcode() == Opcode::bitcast_tagged_i64) &&
                use_count_[src] == 1 && !skipped_insts_.count(d0)) {
                wrap = d0;
                src = d0->operand(0);
            }
            const Instruction* sel = def_of(src);
            if (!sel || sel->opcode() != Opcode::select || sel->parent() != bb ||
                use_count_[src] != 1 || skipped_insts_.count(sel)) {
                continue;
            }
            if (src->type().is_float() || src->type().size_in_bytes() != 8) continue;
            const Instruction* cmp = def_of(sel->operand(0));
            if (!cmp || cmp->parent() != bb || !is_self_nan_test(*cmp) || !skipped_insts_.count(cmp)) continue;
            const Value* f = cmp->operand(0);
            if (f->type() != Type::f64()) continue;
            const bool nan_selects_first = cmp->opcode() == Opcode::ne;
            const Value* k = sel->operand(nan_selects_first ? 1 : 2);
            const Value* bits = sel->operand(nan_selects_first ? 2 : 1);
            const ImmIntInfo ki = get_imm_int_info(k);
            if (!ki.is_imm) continue;
            const Instruction* bc = def_of(bits);
            if (!bc || bc->opcode() != Opcode::bitcast_i64_f64 || bc->operand(0) != f) continue;

            nan_canon_stores_[inst] = NanCanonStore{f, ki.val};
            if (wrap) skipped_insts_.insert(wrap);
            skipped_insts_.insert(sel);
            if (use_count_[bits] == 1) skipped_insts_.insert(bc);
            folded_uses[k]++;
        }
    }
}

VReg AArch64ISel::xmm_constant(LirBlock& lir_bb, int64_t bits) {
    if (const auto kc = xmm_constants_.find(bits); kc != xmm_constants_.end() && kc->second.block == &lir_bb) {
        constexpr size_t kWindow = 64;
        const size_t from = kc->second.end;
        const size_t to = lir_bb.instructions.size();
        bool reusable = from <= to && to - from <= kWindow;
        for (size_t i = from; reusable && i < to; ++i) reusable = !lir_bb.instructions[i]->is_call();
        if (reusable) {
            kc->second.end = to;
            return kc->second.vreg;
        }
    }
    const VReg g = lir_fn_->allocate_vreg(RegClass::GPR, 8);
    auto mabs = std::make_unique<LirInst>(LirOpcode::Movabs);
    mabs->add_def(LirOperand::vreg(g, 8));
    mabs->add_use(LirOperand::imm(bits, 8));
    lir_bb.append_inst(std::move(mabs));

    const VReg k = lir_fn_->allocate_vreg(RegClass::XMM, 8);
    auto mq = std::make_unique<LirInst>(LirOpcode::Movq_xg);
    mq->add_def(LirOperand::vreg(k, 8));
    mq->add_use(LirOperand::vreg(g, 8));
    lir_bb.append_inst(std::move(mq));

    xmm_constants_[bits] = XmmConstant{&lir_bb, lir_bb.instructions.size(), k};
    return k;
}

bool AArch64ISel::lower_nan_canon_store(const Instruction& inst, const LirOperand& mem_op, LirBlock& lir_bb) {
    const auto it = nan_canon_stores_.find(&inst);
    if (it == nan_canon_stores_.end()) return false;
    const NanCanonStore& c = it->second;
    const VReg f = get_vreg(c.f);

    // Materialize `k` before `ucomi` so NZCV stays live straight into `Cmovcc`.
    const VReg k = xmm_constant(lir_bb, c.canon);

    auto ucomi = std::make_unique<LirInst>(LirOpcode::Ucomisd);
    ucomi->add_use(LirOperand::vreg(f, 8));
    ucomi->add_use(LirOperand::vreg(f, 8));
    ucomi->mir_origin = &inst;
    lir_bb.append_inst(std::move(ucomi));

    const VReg r = lir_fn_->allocate_vreg(RegClass::XMM, 8);
    auto cmov = std::make_unique<LirInst>(LirOpcode::Cmovcc);
    cmov->condition = x64::Condition::P;
    cmov->add_def(LirOperand::vreg(r, 8));
    cmov->add_use(LirOperand::vreg(f, 8));
    cmov->add_use(LirOperand::vreg(k, 8));
    cmov->mir_origin = &inst;
    lir_bb.append_inst(std::move(cmov));
    xmm_constants_[c.canon].end = lir_bb.instructions.size();

    auto st = std::make_unique<LirInst>(LirOpcode::Movsd);
    st->add_def(mem_op);
    st->add_use(LirOperand::vreg(r, 8));
    st->mir_origin = &inst;
    lir_bb.append_inst(std::move(st));
    return true;
}

} // namespace brass::aarch64
