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
// An OR is the same with the sides exchanged (a leaf that holds leaves for
// the true target, the last one branches both ways), and the two nest: the
// condition is a tree of `and` / `or` nodes over leaves, each node's first
// operand deciding whether its second is tested at all. A front end's "both
// strings" test, `(t == S || t == R) && t == u`, is such a tree.
//
// A float `eq` / `ne` leaf is two flags (x64_isel_branch.cpp): ucomis and two
// jumps to one side, where its value form is two setcc, two zero extensions
// and an `and` / `or` before the chain's test of it.
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

// A float `eq` / `ne` of two values: ZF and PF both decide it.
bool is_parity_compare(const Instruction& cmp) {
    if (cmp.opcode() != Opcode::eq && cmp.opcode() != Opcode::ne) return false;
    if (cmp.operand_count() != 2 || !cmp.operand(0) || !cmp.operand(1) || cmp.operand(0) == cmp.operand(1)) return false;
    const Type t = cmp.operand(0)->type();
    return t.is_float() && !t.is_vector();
}

}  // namespace

void X64ISel::find_and_chains(const Function& mir_fn) {
    and_chains_.clear();
    chain_nodes_.clear();
    for (const auto* bb : mir_fn.blocks()) {
        for (const auto* inst : *bb) {
            if (inst->opcode() != Opcode::br_if) continue;
            // Edge arguments need their copies on the edge; a chain has
            // several edges to the false target, so it takes none.
            if (!inst->true_target().args.empty() || !inst->false_target().args.empty()) continue;
            const Value* cond = inst->operand(0);
            const Instruction* root = cond && cond->is_instruction() ? cond->defining_instruction() : nullptr;
            if (!root || (root->opcode() != Opcode::and_ && root->opcode() != Opcode::or_)) continue;
            if (root->parent() != bb || use_count_[cond] != 1) continue;

            std::vector<const Value*> leaves;
            std::vector<const Instruction*> nodes;
            bool ok = true;
            // Left to right, so the leaves are tested in the order the
            // generated code wrote them (the cheap tag test before the shape
            // test it guards).
            auto collect = [&](auto&& self, const Value* v) -> void {
                if (!ok) return;
                const Instruction* def = v && v->is_instruction() ? v->defining_instruction() : nullptr;
                if (def && (def->opcode() == Opcode::and_ || def->opcode() == Opcode::or_) && def->parent() == bb &&
                    use_count_[v] == 1 &&
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

            for (const Instruction* n : nodes) {
                skipped_insts_.insert(n);
                chain_nodes_.insert(n);
            }
            // A leaf compared nowhere else, in this block, is folded into
            // its Jcc; any other is a value in a register, tested.
            for (const Value* leaf : leaves) {
                const Instruction* def = leaf->defining_instruction();
                if (def->parent() == bb && use_count_[leaf] == 1 && is_compare_opcode(def->opcode()) &&
                    (comparison_fusible(*def) || is_parity_compare(*def))) {
                    skipped_insts_.insert(def);
                }
            }
            and_chains_.emplace(inst, std::move(leaves));
        }
    }
}

void X64ISel::lower_and_chain_branch(const Instruction& inst, LirBlock& lir_bb) {
    const auto& t_target = inst.true_target();
    const auto& f_target = inst.false_target();

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

    // `v` decided in `cur`, which then leaves for `on_true` or `on_false`. A
    // node's first operand goes on to a block of its own for the second
    // when it holds (an AND) or when it does not (an OR). `falls` is the
    // target laid out after `cur`, which the unconditional jump takes.
    auto lower = [&](auto&& self, const Value* v, LirBlock* cur, uint32_t on_true, uint32_t on_false,
                     uint32_t falls) -> void {
        const Instruction* def = v->defining_instruction();
        if (def && chain_nodes_.count(def)) {
            const bool is_or = def->opcode() == Opcode::or_;
            LirBlock* second = lir_fn_->create_block(is_or ? "or_chain" : "and_chain");
            self(self, def->operand(0), cur, is_or ? on_true : second->id, is_or ? second->id : on_false, second->id);
            self(self, def->operand(1), second, on_true, on_false, falls);
            return;
        }
        for (const uint32_t to : {on_true, on_false}) {
            if (LirBlock* b = lir_fn_->get_block_by_id(to)) link_blocks(*cur, *b);
        }
        if (def && skipped_insts_.count(def) && is_parity_compare(*def)) {
            uint8_t sz = static_cast<uint8_t>(def->operand(0)->type().size_in_bytes());
            if (sz == 0) sz = 8;
            auto ucomi = std::make_unique<LirInst>(sz == 4 ? LirOpcode::Ucomiss : LirOpcode::Ucomisd);
            ucomi->add_use(LirOperand::vreg(get_vreg(def->operand(0)), sz));
            ucomi->add_use(LirOperand::vreg(get_vreg(def->operand(1)), sz));
            ucomi->mir_origin = &inst;
            cur->append_inst(std::move(ucomi));
            // Unordered is unequal: `eq` fails on parity, `ne` holds on it.
            const bool eq = def->opcode() == Opcode::eq;
            const uint32_t on_parity = eq ? on_false : on_true;
            const uint32_t on_equal = eq ? on_true : on_false;
            const uint32_t on_unequal = eq ? on_false : on_true;
            jump(*cur, LirOpcode::Jcc, Condition::P, on_parity);
            if (on_equal == falls) {
                jump(*cur, LirOpcode::Jcc, Condition::NE, on_unequal);
                jump(*cur, LirOpcode::Jmp, Condition::None, on_equal);
            } else {
                jump(*cur, LirOpcode::Jcc, Condition::E, on_equal);
                jump(*cur, LirOpcode::Jmp, Condition::None, on_unequal);
            }
            return;
        }
        const Condition c = leaf_condition(v, *cur);
        if (on_true == falls) {
            jump(*cur, LirOpcode::Jcc, invert(c), on_false);
            jump(*cur, LirOpcode::Jmp, Condition::None, on_true);
        } else {
            jump(*cur, LirOpcode::Jcc, c, on_true);
            jump(*cur, LirOpcode::Jmp, Condition::None, on_false);
        }
    };
    lower(lower, inst.operand(0), &lir_bb, t_target.block->id(), f_target.block->id(), f_target.block->id());
}

}  // namespace brass::x64
