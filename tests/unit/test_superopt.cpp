#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/superopt.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/verify/smt_equiv.hpp>

using namespace brass;

namespace {

size_t count_opcode(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == op) n++;
        }
    }
    return n;
}

} // namespace

TEST_CASE("Superopt - Solver-Proven Redundant Bitmask Elision (#71)") {
    // 1. Guarded 0 <= x < 256: (x & 0xFF) -> x
    {
        Module mod("superopt_mask");
        Builder b(mod);
        Function* fn = mod.create_function("mask_elide", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* zero = b.build_iconst_i32(0);
        Value* c256 = b.build_iconst_i32(256);
        Value* mask = b.build_iconst_i32(0xFF);
        b.build_guard(b.build_sge(x, zero), "deopt");
        b.build_guard(b.build_slt(x, c256), "deopt");

        Value* masked = b.build_and(x, mask);
        b.build_ret(masked);
        fn->rebuild_cfg_predecessors();

        auto before_mod = clone_module(mod);
        SuperoptStats stats;
        SuperoptOptions opts;
        opts.stats = &stats;
        CHECK(run_mir_superopt(*fn, mod, opts));
        CHECK_EQ(stats.masks_elided, 1u);
        CHECK_EQ(count_opcode(*fn, Opcode::and_), 0u);
        CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("mask_elide"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);
    }

    // 2. Unconditional shift-mask: (lshr(x, 24) & 0xFF) -> lshr(x, 24) on i32
    {
        Module mod("superopt_shift_mask");
        Builder b(mod);
        Function* fn = mod.create_function("shift_mask", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* sh = b.build_lshr(x, b.build_iconst_i32(24));
        Value* masked = b.build_and(sh, b.build_iconst_i32(0xFF));
        b.build_ret(masked);
        fn->rebuild_cfg_predecessors();

        auto before_mod = clone_module(mod);
        SuperoptStats stats;
        SuperoptOptions opts;
        opts.stats = &stats;
        CHECK(run_mir_superopt(*fn, mod, opts));
        CHECK_EQ(stats.masks_elided, 1u);
        CHECK_EQ(count_opcode(*fn, Opcode::and_), 0u);
        CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("shift_mask"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);
    }

    // 3. Soundness: 0 <= x <= 256 does NOT allow eliding (x & 0xFF) because x == 256 gives 0 != 256
    {
        Module mod("superopt_mask_keep");
        Builder b(mod);
        Function* fn = mod.create_function("mask_keep", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        b.build_guard(b.build_sge(x, b.build_iconst_i32(0)), "deopt");
        b.build_guard(b.build_sle(x, b.build_iconst_i32(256)), "deopt");
        Value* masked = b.build_and(x, b.build_iconst_i32(0xFF));
        b.build_ret(masked);
        fn->rebuild_cfg_predecessors();

        SuperoptStats stats;
        SuperoptOptions opts;
        opts.stats = &stats;
        run_mir_superopt(*fn, mod, opts);
        CHECK_EQ(stats.masks_elided, 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::and_), 1u);
    }
}

TEST_CASE("Superopt - Solver-Proven Redundant Extension & Truncation Elision (#71)") {
    // 1. sext_i64(trunc_i32(x)) -> x when -1000 <= x <= 1000
    {
        Module mod("superopt_sext_trunc");
        Builder b(mod);
        Function* fn = mod.create_function("sext_trunc", Type::i64(), {Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        b.position_at_end(entry);

        b.build_guard(b.build_sge(x, b.build_iconst_i64(-1000)), "deopt");
        b.build_guard(b.build_sle(x, b.build_iconst_i64(1000)), "deopt");
        Value* tr = b.build_trunc_i32(x);
        Value* ext = b.build_sext_i64(tr);
        b.build_ret(ext);
        fn->rebuild_cfg_predecessors();

        auto before_mod = clone_module(mod);
        SuperoptStats stats;
        SuperoptOptions opts;
        opts.stats = &stats;
        CHECK(run_mir_superopt(*fn, mod, opts));
        CHECK_EQ(stats.extensions_elided, 1u);
        CHECK_EQ(count_opcode(*fn, Opcode::sext_i64), 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::trunc_i32), 0u);
        CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("sext_trunc"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);
    }

    // 2. zext_i64(trunc_i32(x)) -> x when 0 <= x <= 100000
    {
        Module mod("superopt_zext_trunc");
        Builder b(mod);
        Function* fn = mod.create_function("zext_trunc", Type::i64(), {Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        b.position_at_end(entry);

        b.build_guard(b.build_sge(x, b.build_iconst_i64(0)), "deopt");
        b.build_guard(b.build_sle(x, b.build_iconst_i64(100000)), "deopt");
        Value* tr = b.build_trunc_i32(x);
        Value* ext = b.build_zext_i64(tr);
        b.build_ret(ext);
        fn->rebuild_cfg_predecessors();

        auto before_mod = clone_module(mod);
        SuperoptStats stats;
        SuperoptOptions opts;
        opts.stats = &stats;
        CHECK(run_mir_superopt(*fn, mod, opts));
        CHECK_EQ(stats.extensions_elided, 1u);
        CHECK_EQ(count_opcode(*fn, Opcode::zext_i64), 0u);
        CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("zext_trunc"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);
    }
}

TEST_CASE("Superopt - Solver-Proven Signed Division & Modulo Strength Reduction (#71)") {
    // 1. When x >= 0 is guarded: sdiv(x, 8) -> lshr(x, 3) and smod(x, 8) -> and(x, 7)
    {
        Module mod("superopt_sdiv_pos");
        Builder b(mod);
        Function* fn = mod.create_function("sdiv_smod_pos", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        b.build_guard(b.build_sge(x, b.build_iconst_i32(0)), "deopt");
        Value* eight = b.build_iconst_i32(8);
        Value* q = b.build_sdiv(x, eight);
        Value* r = b.build_smod(x, eight);
        Value* sum = b.build_add(q, r);
        b.build_ret(sum);
        fn->rebuild_cfg_predecessors();

        auto before_mod = clone_module(mod);
        SuperoptStats stats;
        SuperoptOptions opts;
        opts.stats = &stats;
        CHECK(run_mir_superopt(*fn, mod, opts));
        CHECK_EQ(stats.dags_synthesized, 2u);
        CHECK_EQ(count_opcode(*fn, Opcode::sdiv), 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::smod), 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::lshr), 1u);
        CHECK_EQ(count_opcode(*fn, Opcode::and_), 1u);
        CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("sdiv_smod_pos"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);
    }

    // 2. Soundness: When x can be negative, sdiv(x, 8) != lshr(x, 3) and smod(x, 8) != and(x, 7)
    {
        Module mod("superopt_sdiv_signed");
        Builder b(mod);
        Function* fn = mod.create_function("sdiv_smod_signed", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* eight = b.build_iconst_i32(8);
        Value* q = b.build_sdiv(x, eight);
        Value* r = b.build_smod(x, eight);
        b.build_ret(b.build_add(q, r));
        fn->rebuild_cfg_predecessors();

        SuperoptStats stats;
        SuperoptOptions opts;
        opts.stats = &stats;
        run_mir_superopt(*fn, mod, opts);
        CHECK_EQ(count_opcode(*fn, Opcode::sdiv), 1u);
        CHECK_EQ(count_opcode(*fn, Opcode::smod), 1u);
    }
}

TEST_CASE("Superopt - Multi-Instruction DAG Synthesis & Distributive Factoring (#71)") {
    Module mod("superopt_dag");
    Builder b(mod);
    Function* fn = mod.create_function("dag_synth", Type::i32(), {Type::i32(), Type::i32(), Type::i32()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i32());
    Value* y = b.add_block_param(entry, Type::i32());
    Value* z = b.add_block_param(entry, Type::i32());
    b.position_at_end(entry);

    // 1. (x ^ y) ^ y -> x
    Value* xy = b.build_xor(x, y);
    Value* x_back = b.build_xor(xy, y);

    // 2. Distributive factoring: (x_back & y) | (x_back & z) -> x & (y | z)
    Value* lhs = b.build_and(x_back, y);
    Value* rhs = b.build_and(x_back, z);
    Value* factored = b.build_or(lhs, rhs);
    b.build_ret(factored);
    fn->rebuild_cfg_predecessors();

    auto before_mod = clone_module(mod);
    SuperoptStats stats;
    SuperoptOptions opts;
    opts.stats = &stats;
    CHECK(run_mir_superopt(*fn, mod, opts));
    CHECK(stats.dags_synthesized >= 2u);
    CHECK_EQ(count_opcode(*fn, Opcode::xor_), 0u);
    CHECK_EQ(count_opcode(*fn, Opcode::and_), 1u);
    CHECK_EQ(count_opcode(*fn, Opcode::or_), 1u);
    CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("dag_synth"), *fn).status,
             verify::EquivStatus::ProvenEquivalent);
}

TEST_CASE("Superopt - x64 LIR Peephole Superoptimizer with Solver Verification (#71)") {
    using namespace brass::codegen;

    // 1. Collapses Mov(rax, 10); Add(rax, 22); Imul(rax, 1); Mov(rax, rax) -> Mov(rax, 32)
    {
        LirBlock blk(0, "lir_superopt_ok");
        auto m0 = std::make_unique<LirInst>(LirOpcode::Mov);
        m0->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        m0->add_use(LirOperand::imm(10, 8));
        blk.append_inst(std::move(m0));

        auto a1 = std::make_unique<LirInst>(LirOpcode::Add);
        a1->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        a1->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        a1->add_use(LirOperand::imm(22, 8));
        blk.append_inst(std::move(a1));

        auto mul1 = std::make_unique<LirInst>(LirOpcode::Imul);
        mul1->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        mul1->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        mul1->add_use(LirOperand::imm(1, 8));
        blk.append_inst(std::move(mul1));

        auto self_mov = std::make_unique<LirInst>(LirOpcode::Mov);
        self_mov->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        self_mov->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        blk.append_inst(std::move(self_mov));

        LirBlock orig(0, "orig");
        for (const auto& inst : blk.instructions) {
            orig.append_inst(std::make_unique<LirInst>(*inst));
        }

        SuperoptStats stats;
        SuperoptOptions opts;
        opts.stats = &stats;
        opts.lir_check_flags = false;
        CHECK(run_x64_lir_superopt(blk, opts));
        CHECK_EQ(blk.instructions.size(), 1u);
        CHECK(blk.instructions[0]->opcode == LirOpcode::Mov);
        CHECK_EQ(blk.instructions[0]->uses[0].imm_int, 32);
        CHECK_EQ(verify::verify_x64_lir_block_equivalence(orig, blk, 0xFFFFu, 0u, false).status,
                 verify::EquivStatus::ProvenEquivalent);
    }

    // 2. Soundness: Preserves Add(rax, 0) when RFLAGS are live-out (lir_check_flags = true)
    {
        LirBlock blk(0, "lir_superopt_flags");
        auto add0 = std::make_unique<LirInst>(LirOpcode::Add);
        add0->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        add0->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        add0->add_use(LirOperand::imm(0, 8));
        blk.append_inst(std::move(add0));

        SuperoptStats stats;
        SuperoptOptions opts;
        opts.stats = &stats;
        opts.lir_check_flags = true;
        CHECK(!run_x64_lir_superopt(blk, opts));
        CHECK_EQ(blk.instructions.size(), 1u);
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
