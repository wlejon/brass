#include "test_framework.hpp"
#include <brass/codegen/lir_flags.hpp>
#include <brass/codegen/sched_dag.hpp>
#include <brass/codegen/instruction_scheduler.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/target/x64/x64_registers.hpp>
#include <brass/target/aarch64/aarch64_registers.hpp>

using namespace brass;
using namespace brass::codegen;

namespace {

bool has_edge(SchedDAG& dag, uint32_t from, uint32_t to) {
    for (const auto& s : dag.node(from).succs) {
        if (s.target_node == to) return true;
    }
    return false;
}

LirOperand x64r(x64::GPR g) { return LirOperand::preg_gpr(g, 8); }
LirOperand a64r(aarch64::GPR g) { return LirOperand::preg_aarch64_gpr(g, 8); }

// cmp a, b ; setcc c ; <op> d, e   -- the op must not move above the setcc
// when it writes flags.
void build_cmp_setcc_op(LirBlock& bb, LirOpcode op, LirOperand a, LirOperand b, LirOperand c,
                        LirOperand d, LirOperand e) {
    auto cmp = std::make_unique<LirInst>(LirOpcode::Cmp);
    cmp->add_use(a);
    cmp->add_use(b);
    bb.append_inst(std::move(cmp));
    auto set = std::make_unique<LirInst>(LirOpcode::Setcc);
    set->add_def(c);
    bb.append_inst(std::move(set));
    auto inst = std::make_unique<LirInst>(op);
    inst->add_def(d);
    inst->add_use(e);
    if (op == LirOpcode::Adds || op == LirOpcode::Subs) inst->add_use(LirOperand::imm(1, 8));
    bb.append_inst(std::move(inst));
}

} // namespace

TEST_CASE("LIR flags - x64 bit-scan and count instructions write the flags") {
    for (LirOpcode op : {LirOpcode::Bsr, LirOpcode::Bsr32, LirOpcode::Bsf, LirOpcode::Bsf32,
                         LirOpcode::Popcnt, LirOpcode::Popcnt32, LirOpcode::Lzcnt, LirOpcode::Lzcnt32,
                         LirOpcode::Tzcnt, LirOpcode::Tzcnt32, LirOpcode::Idiv, LirOpcode::Div,
                         LirOpcode::Smulh, LirOpcode::Umulh, LirOpcode::Imul, LirOpcode::Shl}) {
        LirInst inst(op);
        CHECK(lir_writes_flags(inst, Arch::x64));
        // Partial / undefined results: a liveness scan must not stop there.
        CHECK_FALSE(lir_kills_flags(inst, Arch::x64));
    }
    for (LirOpcode op : {LirOpcode::Cmp, LirOpcode::Test, LirOpcode::Add, LirOpcode::Sub,
                         LirOpcode::And, LirOpcode::Xor, LirOpcode::Neg, LirOpcode::Ucomisd}) {
        LirInst inst(op);
        CHECK(lir_kills_flags(inst, Arch::x64));
    }
    for (LirOpcode op : {LirOpcode::Mov, LirOpcode::Lea, LirOpcode::Not, LirOpcode::Setcc,
                         LirOpcode::Cmovcc, LirOpcode::Addsd, LirOpcode::Movabs}) {
        LirInst inst(op);
        CHECK_FALSE(lir_writes_flags(inst, Arch::x64));
    }
}

TEST_CASE("LIR flags - AArch64 only the S forms and compares write NZCV") {
    for (LirOpcode op : {LirOpcode::Adds, LirOpcode::Adds32, LirOpcode::Subs, LirOpcode::Subs32,
                         LirOpcode::Cmp, LirOpcode::Cmp32, LirOpcode::Test, LirOpcode::Ucomiss}) {
        LirInst inst(op);
        CHECK(lir_kills_flags(inst, Arch::aarch64));
    }
    for (LirOpcode op : {LirOpcode::Add, LirOpcode::Sub, LirOpcode::And, LirOpcode::Xor, LirOpcode::Neg,
                         LirOpcode::Imul, LirOpcode::Shl, LirOpcode::Popcnt, LirOpcode::Lzcnt,
                         LirOpcode::Bsr, LirOpcode::Idiv}) {
        LirInst inst(op);
        CHECK_FALSE(lir_writes_flags(inst, Arch::aarch64));
    }
    for (LirOpcode op : {LirOpcode::Call, LirOpcode::Safepoint, LirOpcode::WriteBarrier, LirOpcode::GuardExit}) {
        LirInst inst(op);
        CHECK(lir_writes_flags(inst, Arch::aarch64));
        CHECK(lir_writes_flags(inst, Arch::x64));
    }
    LirInst jcc(LirOpcode::Jcc);
    CHECK(lir_reads_flags(jcc, Arch::aarch64));
    CHECK(lir_reads_flags(jcc, Arch::x64));
}

TEST_CASE("LIR flags - scheduler keeps x64 bsr/popcnt/lzcnt/tzcnt after the flag reader") {
    for (LirOpcode op : {LirOpcode::Bsr, LirOpcode::Bsf, LirOpcode::Popcnt, LirOpcode::Lzcnt, LirOpcode::Tzcnt}) {
        LirFunction fn;
        LirBlock* bb = fn.create_block("entry");
        build_cmp_setcc_op(*bb, op, x64r(x64::GPR::RAX), x64r(x64::GPR::RBX), x64r(x64::GPR::RCX),
                           x64r(x64::GPR::RDX), x64r(x64::GPR::RSI));
        SchedDAG dag(*bb, Arch::x64, false);
        dag.build();
        // setcc -> op (the op overwrites what setcc reads)
        CHECK(has_edge(dag, 1, 2));
        CHECK(has_edge(dag, 0, 2));
    }
}

TEST_CASE("LIR flags - scheduler orders AArch64 adds/subs but not plain arithmetic") {
    for (LirOpcode op : {LirOpcode::Adds, LirOpcode::Subs}) {
        LirFunction fn;
        LirBlock* bb = fn.create_block("entry");
        build_cmp_setcc_op(*bb, op, a64r(aarch64::GPR::X0), a64r(aarch64::GPR::X1), a64r(aarch64::GPR::X2),
                           a64r(aarch64::GPR::X3), a64r(aarch64::GPR::X4));
        SchedDAG dag(*bb, Arch::aarch64, false);
        dag.build();
        CHECK(has_edge(dag, 1, 2));
    }
    for (LirOpcode op : {LirOpcode::Popcnt, LirOpcode::Lzcnt, LirOpcode::Neg}) {
        LirFunction fn;
        LirBlock* bb = fn.create_block("entry");
        build_cmp_setcc_op(*bb, op, a64r(aarch64::GPR::X0), a64r(aarch64::GPR::X1), a64r(aarch64::GPR::X2),
                           a64r(aarch64::GPR::X3), a64r(aarch64::GPR::X4));
        SchedDAG dag(*bb, Arch::aarch64, false);
        dag.build();
        // No register or flag relation: free to move.
        CHECK_FALSE(has_edge(dag, 1, 2));
        CHECK_FALSE(has_edge(dag, 0, 2));
    }
}

TEST_CASE("LIR flags - lir_arch follows the function's calling convention") {
    LirFunction fn;
    fn.calling_conv = CallingConvention::aapcs64();
    CHECK(lir_arch(fn) == Arch::aarch64);
    fn.calling_conv = CallingConvention::sysv64();
    CHECK(lir_arch(fn) == Arch::x64);
}
