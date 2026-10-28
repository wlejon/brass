#include <brass/mir/branch_split.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/uses.hpp>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

namespace brass {

namespace {


bool is_and_or(Opcode op) noexcept {
    return op == Opcode::and_ || op == Opcode::or_;
}

// A condition made in `bb` and read only once: a comparison, or an and/or
// of such conditions.
bool is_split_tree(const Value* v, const BasicBlock* bb, const UseCounts& uses, int depth) {
    if (!v || !v->is_instruction() || depth > 4) return false;
    const Instruction* d = v->defining_instruction();
    if (!d || d->parent() != bb) return false;
    if (uses.value_or(v, 0) != 1) return false;
    if (is_comparison(d->opcode())) return true;
    return is_and_or(d->opcode()) && d->operand_count() == 2 &&
           is_split_tree(d->operand(0), bb, uses, depth + 1) &&
           is_split_tree(d->operand(1), bb, uses, depth + 1);
}

// Moves the instructions of a split tree from `from` to the end of `to`,
// operands before their readers.
void move_tree(Instruction* inst, BasicBlock* from, BasicBlock* to) {
    if (is_and_or(inst->opcode())) {
        move_tree(inst->operand(0)->defining_instruction(), from, to);
        move_tree(inst->operand(1)->defining_instruction(), from, to);
    }
    from->remove_instruction(inst);
    to->append_instruction(inst);
}

// Splits bb's branch once, the block split off placed right after bb.
bool split_block(Function& fn, BasicBlock* bb, Builder& b, const UseCounts& uses) {
    Instruction* term = bb->terminator();
    if (!term || term->opcode() != Opcode::br_if) return false;
    Value* cond = term->operand(0);
    if (!cond || !cond->is_instruction()) return false;
    Instruction* op = cond->defining_instruction();
    if (!op || !is_and_or(op->opcode()) || !is_split_tree(cond, bb, uses, 0)) return false;
    Value* first = op->operand(0);
    Value* second = op->operand(1);

    std::string name = (bb->name().empty() ? "b" + std::to_string(bb->id()) : std::string(bb->name())) + "_split";
    BasicBlock* mid = b.create_block(name);
    mid->set_parent(&fn);
    auto& blocks = fn.blocks();
    blocks.insert(std::find(blocks.begin(), blocks.end(), bb) + 1, mid);

    move_tree(second->defining_instruction(), bb, mid);
    const BranchTarget t = term->true_target();
    const BranchTarget f = term->false_target();
    b.position_at_end(mid);
    b.build_br_if(second, t.block, Span<Value* const>(t.args.data(), t.args.size()),
                  f.block, Span<Value* const>(f.args.data(), f.args.size()));
    term->set_operand(0, first);
    if (op->opcode() == Opcode::and_) {
        term->true_target() = BranchTarget(mid);
    } else {
        term->false_target() = BranchTarget(mid);
    }
    bb->remove_instruction(op);
    return true;
}

} // namespace

bool split_branch_conditions(Function& fn) {
    const UseCounts uses = compute_use_counts(fn);
    Builder b(fn);
    bool changed = false;
    // A block split is tried again (its first condition may be an and too);
    // the block split off comes next.
    size_t bi = 0;
    while (bi < fn.blocks().size()) {
        BasicBlock* bb = fn.blocks()[bi];
        if (bb && split_block(fn, bb, b, uses)) {
            changed = true;
        } else {
            ++bi;
        }
    }
    if (changed) fn.rebuild_cfg_predecessors();
    return changed;
}

} // namespace brass
