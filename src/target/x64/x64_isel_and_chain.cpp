// A br_if on an AND of comparisons, as a chain of compare-and-branch.
//
// `br_if (and (and c0 c1) c2)` lowered value-wise is a setcc and a zero
// extension per comparison, an `and` per node and a final test: three
// instructions a test where a branch per test is one. Generated code builds
// exactly this shape on purpose (an inline cache's hit is "an object, a
// plain one, this shape, an own slot" as one MIR branch, so a function of
// hundreds of reads keeps few MIR blocks for the optimizer), so the branch is
// split here, after every MIR pass has run: each leaf's flags go straight
// into a Jcc to the false target, in a LIR block of its own, and the last
// leaf branches both ways.
//
// Only the condition is restructured. Every operand of every leaf was
// computed before the branch, in the block, exactly as before; a leaf is a
// pure test of values already in hand, so testing them one at a time and
// stopping at the first false one answers what their AND answers.

#include <brass/target/x64/x64_isel.hpp>

namespace brass::x64 {

using namespace brass::codegen;

namespace {

// The most leaves a chain takes; a wider AND keeps its value form.
constexpr size_t kMaxChainLeaves = 8;

bool is_compare_opcode(Opcode op) noexcept {
    switch (op) {
        case Opcode::eq: case Opcode::ne:
        case Opcode::slt: case Opcode::sle: case Opcode::sgt: case Opcode::sge:
        case Opcode::ult: case Opcode::ule: case Opcode::ugt: case Opcode::uge:
            return true;
        default:
            return false;
    }
}

// Whether `v` is 0 or 1: only then is "v != 0" the same test as v's bit in
// an AND. A comparison is; an and / or of such values is; nothing else is
// assumed (an `and` of two masks can be zero with both operands nonzero).
bool is_boolean(const Value* v, int depth = 0) {
    if (!v || !v->is_instruction() || depth > 8) return false;
    const Instruction* def = v->defining_instruction();
    if (!def) return false;
    if (is_compare_opcode(def->opcode())) return true;
    if (def->opcode() == Opcode::and_ || def->opcode() == Opcode::or_) {
        return is_boolean(def->operand(0), depth + 1) && is_boolean(def->operand(1), depth + 1);
    }
    return false;
}

}  // namespace

void X64ISel::find_and_chains(const Function& mir_fn) {
    and_chains_.clear();
    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (inst->opcode() != Opcode::br_if) continue;
            // Edge arguments need their copies on the edge; a chain has
            // several edges to the false target, so it takes none.
            if (!inst->true_target().args.empty() || !inst->false_target().args.empty()) continue;
            const Value* cond = inst->operand(0);
            const Instruction* root = cond && cond->is_instruction() ? cond->defining_instruction() : nullptr;
            if (!root || root->opcode() != Opcode::and_ || root->parent() != bb || use_count_[cond] != 1) continue;

            std::vector<const Value*> leaves;
            std::vector<const Instruction*> nodes;
            bool ok = true;
            // Left to right, so the leaves are tested in the order the
            // generated code wrote them (the cheap tag test before the shape
            // test it guards).
            auto collect = [&](auto&& self, const Value* v) -> void {
                if (!ok) return;
                const Instruction* def = v && v->is_instruction() ? v->defining_instruction() : nullptr;
                if (def && def->opcode() == Opcode::and_ && def->parent() == bb && use_count_[v] == 1 &&
                    is_boolean(v)) {
                    nodes.push_back(def);
                    self(self, def->operand(0));
                    self(self, def->operand(1));
                    return;
                }
                if (!is_boolean(v) || leaves.size() >= kMaxChainLeaves) {
                    ok = false;
                    return;
                }
                leaves.push_back(v);
            };
            collect(collect, cond);
            if (!ok || leaves.size() < 2) continue;

            for (const Instruction* n : nodes) skipped_insts_.insert(n);
            // A leaf compared nowhere else, in this block, is folded into
            // its Jcc; any other is a value in a register, tested.
            for (const Value* leaf : leaves) {
                const Instruction* def = leaf->defining_instruction();
                if (def->parent() == bb && use_count_[leaf] == 1 && is_compare_opcode(def->opcode()) &&
                    comparison_fusible(*def)) {
                    skipped_insts_.insert(def);
                }
            }
            and_chains_.emplace(inst, std::move(leaves));
        }
    }
}

void X64ISel::lower_and_chain_branch(const Instruction& inst, const std::vector<const Value*>& leaves,
                                     LirBlock& lir_bb) {
    const auto& t_target = inst.true_target();
    const auto& f_target = inst.false_target();
    LirBlock* t_lir = lir_fn_->get_block_by_id(t_target.block->id());
    LirBlock* f_lir = lir_fn_->get_block_by_id(f_target.block->id());

    auto leaf_condition = [&](const Value* leaf, LirBlock& bb) -> Condition {
        const Instruction* def = leaf->defining_instruction();
        if (def && skipped_insts_.count(def) && is_compare_opcode(def->opcode())) {
            return append_fused_compare(*def, inst, bb);
        }
        VReg v = get_vreg(leaf);
        const uint8_t sz = v.size;
        auto test = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Test32 : LirOpcode::Test);
        test->add_use(LirOperand::vreg(v, sz));
        test->add_use(LirOperand::vreg(v, sz));
        test->mir_origin = &inst;
        bb.append_inst(std::move(test));
        return Condition::NE;
    };
    auto jump = [&](LirBlock& bb, LirOpcode op, Condition c, uint32_t label) {
        auto j = std::make_unique<LirInst>(op);
        j->condition = c;
        j->add_use(LirOperand::label(label));
        j->mir_origin = &inst;
        bb.append_inst(std::move(j));
    };

    LirBlock* cur = &lir_bb;
    for (size_t i = 0; i < leaves.size(); ++i) {
        const Condition c = leaf_condition(leaves[i], *cur);
        if (i + 1 < leaves.size()) {
            LirBlock* next = lir_fn_->create_block("and_chain");
            if (f_lir) link_blocks(*cur, *f_lir);
            link_blocks(*cur, *next);
            jump(*cur, LirOpcode::Jcc, invert(c), f_target.block->id());
            jump(*cur, LirOpcode::Jmp, Condition::None, next->id);
            cur = next;
        } else {
            if (cur != &lir_bb) {
                if (t_lir) link_blocks(*cur, *t_lir);
                if (f_lir) link_blocks(*cur, *f_lir);
            }
            jump(*cur, LirOpcode::Jcc, c, t_target.block->id());
            jump(*cur, LirOpcode::Jmp, Condition::None, f_target.block->id());
        }
    }
}

}  // namespace brass::x64
