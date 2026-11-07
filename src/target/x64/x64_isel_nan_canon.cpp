// NaN-CANONICALIZING STORES. A NaN-boxing front end stores a double as its
// bits with every NaN replaced by one canonical pattern (x86 arithmetic makes
// the negative default NaN, which such a front end reads as a tag):
//
//     store [m], select(ne f, f, K, bitcast_i64_f64 f)
//
// As a select, that is ucomisd f, f; movq r, f; cmovp r, K; mov [m], r: a
// move out of the FP domain and a GPR held for K. With the select's only use
// the store, it stays in the FP domain instead:
//
//     vcmpunordsd t, f, f      ; all ones where f is a NaN
//     vblendvpd   r, f, k, t   ; k: K in an xmm, made once per stretch
//     movsd       [m], r
//
// The same bits reach memory: f's own where f is not a NaN, K where it is.
// VEX forms, so only with AVX (set_avx).
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/mir/instruction.hpp>

namespace brass::x64 {

using namespace brass::codegen;

namespace {

const Instruction* def_of(const Value* v) {
    return v && v->is_instruction() ? v->defining_instruction() : nullptr;
}

constexpr int kUnordQ = 3;  // vcmppd predicate UNORD_Q

}  // namespace

void X64ISel::find_nan_canon_stores(const Function& mir_fn, std::unordered_map<const Value*, uint32_t>& folded_uses) {
    nan_canon_stores_.clear();
    xmm_constants_.clear();
    if (!avx_) return;
    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            const Opcode op = inst->opcode();
            if ((op != Opcode::store && op != Opcode::store_indexed) || skipped_insts_.count(inst)) continue;
            const Value* src = inst->operand(op == Opcode::store ? 1 : 2);
            const Instruction* sel = def_of(src);
            if (!sel || sel->opcode() != Opcode::select || sel->parent() != bb || use_count_[src] != 1) continue;
            if (src->type().is_float() || src->type().size_in_bytes() != 8) continue;
            const Instruction* cmp = def_of(sel->operand(0));
            // The NaN test fused into the select: single-use, this block.
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
            skipped_insts_.insert(sel);
            if (use_count_[bits] == 1) skipped_insts_.insert(bc);
            folded_uses[k]++;
        }
    }
}

bool X64ISel::lower_nan_canon_store(const Instruction& inst, const LirOperand& mem_op, LirBlock& lir_bb) {
    const auto it = nan_canon_stores_.find(&inst);
    if (it == nan_canon_stores_.end()) return false;
    const NanCanonStore& c = it->second;
    const VReg f = get_vreg(c.f);

    const VReg nan_mask = lir_fn_->allocate_vreg(RegClass::XMM, 8);
    auto cmp = std::make_unique<LirInst>(LirOpcode::Vcmppd);
    cmp->add_def(LirOperand::vreg(nan_mask, 8));
    cmp->add_use(LirOperand::vreg(f, 8));
    cmp->add_use(LirOperand::vreg(f, 8));
    cmp->add_use(LirOperand::imm(kUnordQ, 1));
    cmp->mir_origin = &inst;
    lir_bb.append_inst(std::move(cmp));

    // K in an xmm: the one an earlier store of this block made, unless a call
    // came between (it would be kept across the call for what two
    // instructions cost) or it is far back (it would hold a register).
    VReg k{};
    if (const auto kc = xmm_constants_.find(c.canon); kc != xmm_constants_.end() && kc->second.block == &lir_bb) {
        constexpr size_t kWindow = 64;
        const size_t from = kc->second.end;
        const size_t to = lir_bb.instructions.size();
        bool reusable = from <= to && to - from <= kWindow;
        for (size_t i = from; reusable && i < to; ++i) reusable = !lir_bb.instructions[i]->is_call();
        if (reusable) k = kc->second.vreg;
    }
    if (!k.is_valid()) {
        const VReg g = lir_fn_->allocate_vreg(RegClass::GPR, 8);
        auto mabs = std::make_unique<LirInst>(LirOpcode::Movabs);
        mabs->add_def(LirOperand::vreg(g, 8));
        mabs->add_use(LirOperand::imm(c.canon, 8));
        lir_bb.append_inst(std::move(mabs));
        k = lir_fn_->allocate_vreg(RegClass::XMM, 8);
        auto mq = std::make_unique<LirInst>(LirOpcode::Movq_xg);
        mq->add_def(LirOperand::vreg(k, 8));
        mq->add_use(LirOperand::vreg(g, 8));
        lir_bb.append_inst(std::move(mq));
    }

    const VReg r = lir_fn_->allocate_vreg(RegClass::XMM, 8);
    auto blend = std::make_unique<LirInst>(LirOpcode::Vblendvpd);
    blend->add_def(LirOperand::vreg(r, 8));
    blend->add_use(LirOperand::vreg(f, 8));         // lanes where the mask is clear
    blend->add_use(LirOperand::vreg(k, 8));         // lanes where it is set
    blend->add_use(LirOperand::vreg(nan_mask, 8));
    blend->mir_origin = &inst;
    lir_bb.append_inst(std::move(blend));
    xmm_constants_[c.canon] = XmmConstant{&lir_bb, lir_bb.instructions.size(), k};

    auto st = std::make_unique<LirInst>(LirOpcode::Movsd);
    st->add_def(mem_op);
    st->add_use(LirOperand::vreg(r, 8));
    st->mir_origin = &inst;
    lir_bb.append_inst(std::move(st));
    return true;
}

}  // namespace brass::x64
