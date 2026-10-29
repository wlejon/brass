#include "test_framework.hpp"
#include <brass/codegen/lir.hpp>
#include <brass/codegen/block_layout.hpp>
#include <brass/target/x64/x64_registers.hpp>

#include <algorithm>

// Where cold blocks and the two sides of a branch go: the allocator's order
// puts a loop's cold block (its back edge's interrupt poll) after the hot
// latch, so the poll's call evicts nothing on the hot path, and with no
// profile the final layout lets the side of a branch that does work fall
// through.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

void jmp(LirBlock* from, LirBlock* to) {
    auto j = std::make_unique<LirInst>(LirOpcode::Jmp);
    j->add_use(LirOperand::label(to->id));
    from->append_inst(std::move(j));
    from->successors.push_back(to);
}

// `jcc taken; jmp other`.
void branch(LirBlock* from, LirBlock* taken, LirBlock* other) {
    auto j = std::make_unique<LirInst>(LirOpcode::Jcc);
    j->condition = Condition::NE;
    j->add_use(LirOperand::label(taken->id));
    from->append_inst(std::move(j));
    from->successors.push_back(taken);
    jmp(from, other);
}

void work(LirBlock* b) {
    auto add = std::make_unique<LirInst>(LirOpcode::Add);
    add->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    add->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
    add->add_use(LirOperand::imm(1, 8));
    b->append_inst(std::move(add));
}

size_t position(const LirFunction& fn, const LirBlock* b) {
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        if (fn.blocks[i].get() == b) return i;
    }
    return SIZE_MAX;
}

} // namespace

TEST_CASE("Block order - a loop's cold poll is ordered after its latch") {
    // entry -> head; head: body or exit; body, the latch, polls: the
    // continuation back to head, or (through its edge block) the cold
    // handler, which calls and goes back to head too. The continuation is
    // the branch's first successor, which plain DFS order visited first and
    // so placed after the handler.
    LirFunction fn;
    fn.name = "cold_poll";
    LirBlock* entry = fn.create_block("entry");
    LirBlock* head = fn.create_block("head");
    LirBlock* body = fn.create_block("body");
    LirBlock* cont = fn.create_block("cont");
    LirBlock* edge = fn.create_block("edge");
    LirBlock* poll = fn.create_block("poll");
    LirBlock* exit = fn.create_block("exit");
    poll->cold = true;

    jmp(entry, head);
    branch(head, body, exit);
    work(body);
    branch(body, cont, edge);
    work(cont);
    jmp(cont, head);
    jmp(edge, poll);
    jmp(poll, head);
    exit->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    fn.sort_blocks_rpo();

    CHECK(edge->cold);  // the edge block into the cold handler is cold with it
    CHECK_FALSE(cont->cold);
    CHECK_EQ(position(fn, entry), size_t(0));
    CHECK(position(fn, body) < position(fn, cont));
    CHECK(position(fn, cont) < position(fn, edge));
    CHECK(position(fn, cont) < position(fn, poll));
}

TEST_CASE("Block layout - an if's body falls through, not the join") {
    // cond: then or join; then -> join. Falling into the join left a taken
    // branch to the body and a jump back from it.
    LirFunction fn;
    fn.name = "triangle";
    LirBlock* cond = fn.create_block("cond");
    LirBlock* join = fn.create_block("join");
    LirBlock* then = fn.create_block("then");
    branch(cond, then, join);
    work(then);
    jmp(then, join);
    join->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    optimize_block_layout(fn);

    CHECK_EQ(position(fn, cond), size_t(0));
    CHECK_EQ(position(fn, then), size_t(1));
    CHECK_EQ(position(fn, join), size_t(2));
}

TEST_CASE("Block layout - the side with work falls through, not the copies") {
    // cond: work or copies; both go on to join. The copies side only moves
    // a default into the join's register (a typed array read's NaN out of
    // bounds), so the work side is the one to fall into.
    LirFunction fn;
    fn.name = "diamond";
    LirBlock* cond = fn.create_block("cond");
    LirBlock* copies = fn.create_block("copies");
    LirBlock* join = fn.create_block("join");
    LirBlock* body = fn.create_block("work");
    branch(cond, body, copies);
    work(body);
    jmp(body, join);
    auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
    mov->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
    mov->add_use(LirOperand::preg_gpr(GPR::RCX, 8));
    copies->append_inst(std::move(mov));
    jmp(copies, join);
    join->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    optimize_block_layout(fn);

    CHECK_EQ(position(fn, cond), size_t(0));
    CHECK_EQ(position(fn, body), size_t(1));
}
