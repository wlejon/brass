#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/kernel_jit.hpp>
#include <brass/target/ptx/ptx_cleanup.hpp>
#include <brass/target/ptx/ptx_schedule.hpp>
#include <brass/target/ptx/ptx_verifier.hpp>
#include <brass/target/ptx_target.hpp>
#include <brass/verify/gpu_equiv.hpp>

using namespace brass;
using namespace brass::codegen;

TEST_CASE("PtxSchedule - Formal SMT Equivalence Across All Modes and Remat") {
    Module mod("ptx_sched_equiv");
    Function* fn = mod.create_function(
        "wide_kernel", Type::void_type(),
        {Type::ptr(), Type::ptr(), Type::ptr(), Type::i32()});
    KernelBuilder kb(mod, fn);
    Builder& b = kb.builder();
    BasicBlock* entry = b.append_block("entry");
    Value* p_a = b.add_block_param(entry, Type::ptr());
    Value* p_b = b.add_block_param(entry, Type::ptr());
    Value* p_o = b.add_block_param(entry, Type::ptr());
    Value* n   = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);

    Value* gid = kb.global_tid_x();
    kb.if_then(b.build_slt(gid, n), [&] {
        Value* idx0 = gid;
        Value* idx1 = b.build_add(gid, b.build_iconst_i32(4));
        Value* x0 = kb.load_f32_indexed(p_a, idx0);
        Value* y0 = kb.load_f32_indexed(p_b, idx0);
        Value* x1 = kb.load_f32_indexed(p_a, idx1);
        Value* y1 = kb.load_f32_indexed(p_b, idx1);

        Value* s0 = kb.fma(x0, y0, x1);
        Value* s1 = kb.fma(x1, y1, y0);
        Value* fin = b.build_fadd(s0, s1);
        kb.store_f32_indexed(p_o, idx0, fin);
    });
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    for (ptx::PtxSchedMode mode : {
            ptx::PtxSchedMode::None,
            ptx::PtxSchedMode::LatencyHide,
            ptx::PtxSchedMode::MinRegPressure,
            ptx::PtxSchedMode::MmaInterleave,
            ptx::PtxSchedMode::Auto}) {
        target::PtxOptions popts;
        popts.sched_mode = mode;
        popts.enable_ptx_remat = true;
        popts.sched_reg_pressure_limit = 16;

        ptx::Function ptx_fn = target::PtxTarget::lower_function(*fn, popts);
        CHECK(ptx::verify(ptx_fn).empty());

        std::string equiv_err;
        auto eq = verify::verify_ptx_equivalence(*fn, ptx_fn, &equiv_err);
        CHECK_EQ(eq.status, verify::EquivStatus::ProvenEquivalent);
    }
}

TEST_CASE("PtxSchedule - Remat and MinRegPressure Reduce Peak Live Registers") {
    ptx::Function fn("synthetic_pressure");
    auto pa = fn.add_param(ptx::Type::u64, "pa");
    ptx::Block* blk = fn.add_block("$L0");

    ptx::Reg base = fn.new_b64();
    blk->append(ptx::Inst::make(ptx::Opcode::ld, ptx::Type::u64).space(ptx::StateSpace::param).dst(base).src(pa));

    std::vector<ptx::Reg> temps;
    for (int i = 0; i < 16; ++i) {
        ptx::Reg c = fn.new_b32();
        blk->append(ptx::Inst::make(ptx::Opcode::mov, ptx::Type::b32).dst(c).src(ptx::Operand::imm(i * 17 + 5)));
        temps.push_back(c);
    }
    ptx::Reg acc = fn.new_b32();
    blk->append(ptx::Inst::make(ptx::Opcode::mov, ptx::Type::b32).dst(acc).src(ptx::Operand::imm(1)));
    for (int i = 0; i < 16; ++i) {
        ptx::Reg nxt = fn.new_b32();
        blk->append(ptx::Inst::make(ptx::Opcode::add, ptx::Type::s32).dst(nxt).src(acc).src(temps[i]));
        acc = nxt;
    }
    blk->append(ptx::Inst::make(ptx::Opcode::st, ptx::Type::b32)
                    .space(ptx::StateSpace::global)
                    .src(ptx::Operand::addr(base))
                    .src(acc));
    blk->append(ptx::Inst::make(ptx::Opcode::ret));
    CHECK(ptx::verify(fn).empty());

    const uint32_t peak_before = ptx::estimate_peak_live_regs(fn);
    ptx::PtxScheduleOptions sopts;
    sopts.mode = ptx::PtxSchedMode::MinRegPressure;
    sopts.enable_remat = true;
    sopts.reg_pressure_limit = 8;
    const ptx::PtxScheduleStats st = ptx::schedule_and_remat(fn, sopts);
    ptx::renumber_registers(fn);
    CHECK(ptx::verify(fn).empty());
    CHECK(st.remats_applied > 0u);
    CHECK(st.peak_live_after < peak_before);
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
