#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/smt_guard_elim.hpp>
#include <brass/mir/superopt.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/verify/smt_equiv.hpp>
#include <brass/verify/arm64_lir_equiv.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/fuzz/diff_fuzzer.hpp>
#include <brass/fuzz/program_generator.hpp>
#include <brass/fuzz/ir_mutator.hpp>
#include <fstream>
#include <sstream>

using namespace brass;
using namespace brass::verify;
using namespace brass::codegen;

namespace {

void run_all_mir_passes_with_solver(Module& mod) {
    fuzz::run_fuzz_pipeline(mod, fuzz::FuzzPipeline::AllPasses);
    run_smt_guard_elimination(mod);
    run_mir_superopt(mod);
    for (auto& fn_ptr : mod.functions()) {
        cfg_simplify_function(*fn_ptr);
    }
}

void verify_function_backends(const Function& fn) {
    // 1. x86-64 ISel + LinearScan + Peephole + Superopt equivalence
    {
        CallingConvention cc_x64 = CallingConvention::win64();
        x64::X64ISel isel_x64(Target::x64_windows(), cc_x64);
        auto lir_x64 = isel_x64.lower(fn);
        REQUIRE(lir_x64 != nullptr);
        LivenessAnalysis live_x64(*lir_x64);
        live_x64.run();
        run_linear_scan_regalloc(*lir_x64, live_x64, cc_x64);

        LirEquivResult peep_x64 = verify_x64_peephole(*lir_x64);
        if (peep_x64.is_counterexample()) {
            std::cerr << "[x64 peephole CEX in " << fn.name() << "]: " << peep_x64.diagnostic << "\n";
        }
        CHECK(!peep_x64.is_counterexample());

        run_lir_peephole_optimizations(*lir_x64);
        for (auto& blk : lir_x64->blocks) {
            LirBlock before_blk(blk->id, blk->name);
            for (const auto& inst : blk->instructions) {
                if (inst) before_blk.append_inst(std::make_unique<LirInst>(*inst));
            }
            size_t rw = run_x64_lir_superopt(*blk);
            if (rw > 0) {
                LirEquivResult sres = verify_x64_lir_block_equivalence(before_blk, *blk, 0xFFFF, 0xFFFF, false);
                CHECK_EQ(sres.status, EquivStatus::ProvenEquivalent);
            }
        }
    }

    // 2. AArch64 ISel + LinearScan + Peephole equivalence
    {
        CallingConvention cc_arm = CallingConvention::apple_aapcs64();
        auto lir_arm = aarch64::lower_to_aarch64_lir(fn, Target::aarch64_macos(), cc_arm);
        REQUIRE(lir_arm != nullptr);
        LivenessAnalysis live_arm(*lir_arm);
        live_arm.run();
        run_linear_scan_regalloc(*lir_arm, live_arm, cc_arm);

        LirEquivResult peep_arm = verify_arm64_peephole(*lir_arm);
        if (peep_arm.is_counterexample()) {
            std::cerr << "[arm64 peephole CEX in " << fn.name() << "]: " << peep_arm.diagnostic << "\n";
        }
        CHECK(!peep_arm.is_counterexample());
    }
}

} // namespace

TEST_CASE("TV Corpus - Peephole 32-Bit Load-After-Store Zero-Extension Regression (#83)") {
    // 1. x86-64: Mov32 [slot 0, 4], eax; Mov32 eax, [slot 0, 4]
    //    Deleting the 32-bit reload leaves dirty bits [63:32] in RAX;
    //    rewriting to `Mov32 eax, eax` preserves architectural 32-bit zero-extension!
    {
        LirFunction fn;
        fn.name = "x64_mov32_store_reload";
        fn.calling_conv = CallingConvention::win64();
        LirBlock* bb = fn.create_block("entry");
        bb->id = 0;

        auto st32 = std::make_unique<LirInst>(LirOpcode::Mov32);
        st32->add_def(LirOperand::slot(0, 4));
        st32->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        bb->append_inst(std::move(st32));

        auto ld32 = std::make_unique<LirInst>(LirOpcode::Mov32);
        ld32->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        ld32->add_use(LirOperand::slot(0, 4));
        bb->append_inst(std::move(ld32));

        auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
        ret->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        bb->append_inst(std::move(ret));

        // Prove that the old buggy behavior (deleting `Mov32 eax, [slot 0, 4]` completely)
        // is caught by `verify_x64_lir_block_equivalence` as a Counterexample on RAX:
        LirBlock buggy_deleted(0, "buggy_deleted");
        auto st32_b = std::make_unique<LirInst>(LirOpcode::Mov32);
        st32_b->add_def(LirOperand::slot(0, 4));
        st32_b->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 4));
        buggy_deleted.append_inst(std::move(st32_b));
        auto ret_b = std::make_unique<LirInst>(LirOpcode::Ret);
        ret_b->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 8));
        buggy_deleted.append_inst(std::move(ret_b));

        LirEquivResult bug_res = verify_x64_lir_block_equivalence(*bb, buggy_deleted);
        CHECK_EQ(bug_res.status, EquivStatus::Counterexample);
        CHECK(bug_res.diagnostic.find("rax mismatch") != std::string::npos);

        // Prove that fixed `PeepholeOptimizer` is ProvenEquivalent!
        LirEquivResult fixed_res = verify_x64_peephole(fn);
        CHECK_EQ(fixed_res.status, EquivStatus::ProvenEquivalent);

        PeepholeStats stats = run_lir_peephole_optimizations(fn);
        CHECK_EQ(stats.load_after_store_eliminated, size_t(1));
        REQUIRE_EQ(bb->instructions.size(), size_t(3));
        CHECK(bb->instructions[1]->opcode == LirOpcode::Mov32);
        CHECK(bb->instructions[1]->uses[0].is_preg());
        CHECK(bb->instructions[1]->uses[0].preg_val.as_gpr() == x64::GPR::RAX);
    }

    // 2. AArch64: Mov32 [slot 0, 4], w0; Mov32 w0, [slot 0, 4]
    {
        LirFunction fn;
        fn.name = "arm64_mov32_store_reload";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");
        bb->id = 0;

        auto st32 = std::make_unique<LirInst>(LirOpcode::Mov32);
        st32->add_def(LirOperand::slot(0, 4));
        st32->add_use(LirOperand::preg_aarch64_gpr(aarch64::GPR::X0, 4));
        bb->append_inst(std::move(st32));

        auto ld32 = std::make_unique<LirInst>(LirOpcode::Mov32);
        ld32->add_def(LirOperand::preg_aarch64_gpr(aarch64::GPR::X0, 4));
        ld32->add_use(LirOperand::slot(0, 4));
        bb->append_inst(std::move(ld32));

        auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
        ret->add_use(LirOperand::preg_aarch64_gpr(aarch64::GPR::X0, 8));
        bb->append_inst(std::move(ret));

        LirBlock buggy_deleted(0, "buggy_deleted_arm64");
        auto st32_b = std::make_unique<LirInst>(LirOpcode::Mov32);
        st32_b->add_def(LirOperand::slot(0, 4));
        st32_b->add_use(LirOperand::preg_aarch64_gpr(aarch64::GPR::X0, 4));
        buggy_deleted.append_inst(std::move(st32_b));
        auto ret_b = std::make_unique<LirInst>(LirOpcode::Ret);
        ret_b->add_use(LirOperand::preg_aarch64_gpr(aarch64::GPR::X0, 8));
        buggy_deleted.append_inst(std::move(ret_b));

        LirEquivResult bug_res = verify_arm64_lir_block_equivalence(*bb, buggy_deleted);
        CHECK_EQ(bug_res.status, EquivStatus::Counterexample);
        CHECK(bug_res.diagnostic.find("x0 mismatch") != std::string::npos);

        LirEquivResult fixed_res = verify_arm64_peephole(fn);
        CHECK_EQ(fixed_res.status, EquivStatus::ProvenEquivalent);
    }
}

TEST_CASE("TV Corpus - Real Lowered JS & Numeric MIR Kernels Across Full Pipeline") {
    const char* corpus_mir = R"(
module @tv_corpus

func @deopt_nanbox(%a: i64, %b: i64) -> i64 {
bb0:
    %z = iconst.i64 -1
    ret %z
}

func @js_nanbox_tag_dispatch(%box_a: i64, %box_b: i64) -> i64 {
bb0:
    %c47 = iconst.i64 47
    %tag_mask = iconst.i64 65535
    %int32_tag = iconst.i64 65529
    %payload_mask = iconst.i64 4294967295
    %box_prefix = iconst.i64 -1970324836974592

    %shr_a = lshr %box_a, %c47
    %tag_a = and %shr_a, %tag_mask
    %is_int_a = eq %tag_a, %int32_tag
    guard %is_int_a, @deopt_nanbox, [%box_a, %box_b]

    %shr_b = lshr %box_b, %c47
    %tag_b = and %shr_b, %tag_mask
    %is_int_b = eq %tag_b, %int32_tag
    guard %is_int_b, @deopt_nanbox, [%box_a, %box_b]

    %pa32 = trunc.i32 %box_a
    %pb32 = trunc.i32 %box_b
    %sum32 = add %pa32, %pb32
    %sum_zext = zext.i64 %sum32
    %masked = and %sum_zext, %payload_mask
    %reboxed = or %masked, %box_prefix
    ret %reboxed
}

func @deopt_ic(%obj: ptr, %expected_shape: i64) -> i64 {
bb0:
    %z = iconst.i64 0
    ret %z
}

func @js_shape_ic_guarded_add(%obj: ptr, %expected_shape: i64) -> i64 {
bb0:
    %shape = load.i64 %obj
    %ok = eq %shape, %expected_shape
    guard %ok, @deopt_ic, [%obj, %expected_shape]

    %v0 = load.i64 %obj, 8
    %v1 = load.i64 %obj, 16
    %res = add %v0, %v1
    store.i64 %obj, 8, %res
    ret %res
}

func @crypto_murmur_mix64(%k: i64, %seed: i64) -> i64 {
bb0:
    %c33 = iconst.i64 33
    %m1 = iconst.i64 -49064778989728563
    %m2 = iconst.i64 -4265267296055464877
    %x0 = xor %k, %seed
    %s1 = lshr %x0, %c33
    %x1 = xor %x0, %s1
    %x2 = mul %x1, %m1
    %s2 = lshr %x2, %c33
    %x3 = xor %x2, %s2
    %x4 = mul %x3, %m2
    %s3 = lshr %x4, %c33
    %x5 = xor %x4, %s3
    ret %x5
}

func @deopt_clamp(%x: i64, %lo: i64, %hi: i64) -> i64 {
bb0:
    ret %x
}

func @branch_diamond_clamp_strength_reduce(%x: i64, %lo: i64, %hi: i64) -> i64 {
bb0:
    %ok_range = sle %lo, %hi
    guard %ok_range, @deopt_clamp, [%x, %lo, %hi]
    %c8 = iconst.i64 8
    %q = udiv %x, %c8
    %r = umod %x, %c8
    %reconstructed = mul %q, %c8
    %val = add %reconstructed, %r
    %too_low = slt %val, %lo
    br_if %too_low, below, check_high

below:
    br merge(%lo)

check_high:
    %too_high = sgt %val, %hi
    %clamped_hi = select %too_high, %hi, %val
    br merge(%clamped_hi)

merge(%res: i64):
    %zero = sub %x, %x
    %out = add %res, %zero
    ret %out
}

func @deopt_bounds(%idx: i64, %len: i64) -> i64 {
bb0:
    %neg1 = iconst.i64 -1
    ret %neg1
}

func @redundant_guard_chain_kernel(%idx: i64, %len: i64) -> i64 {
bb0:
    %c0 = iconst.i64 0
    %c100 = iconst.i64 100
    %pos = sge %idx, %c0
    guard %pos, @deopt_bounds, [%idx, %len]
    %lt100 = slt %idx, %c100
    guard %lt100, @deopt_bounds, [%idx, %len]
    %len_ge100 = sge %len, %c100
    guard %len_ge100, @deopt_bounds, [%idx, %len]
    %in_bounds_signed = slt %idx, %len
    guard %in_bounds_signed, @deopt_bounds, [%idx, %len]
    %in_bounds_unsigned = ult %idx, %len
    guard %in_bounds_unsigned, @deopt_bounds, [%idx, %len]
    %scaled = mul %idx, %c100
    ret %scaled
}
)";

    DiagnosticReporter diag;
    auto before_mod = parse_module(corpus_mir, &diag, "tv_corpus");
    if (!before_mod) {
        std::cerr << "parse_module failed: " << diag.format_all() << "\n";
    }
    REQUIRE(before_mod != nullptr);
    if (!verify_module(*before_mod, &diag)) {
        std::cerr << "verify_module failed: " << diag.format_all() << "\n";
    }
    CHECK(verify_module(*before_mod, &diag));

    auto after_mod = clone_module(*before_mod);
    REQUIRE(after_mod != nullptr);
    run_all_mir_passes_with_solver(*after_mod);
    CHECK(verify_module(*after_mod, &diag));

    // 1. Verify module-wide MIR equivalence across all 5 kernels
    EquivResult mod_eq = verify_mir_module_equivalence(*before_mod, *after_mod);
    CHECK_EQ(mod_eq.status, EquivStatus::ProvenEquivalent);

    // 2. Verify each function individually + x64 & AArch64 backend peephole/superopt equivalence
    for (size_t i = 0; i < before_mod->functions().size(); ++i) {
        const Function& b_fn = *before_mod->functions()[i];
        const Function& a_fn = *after_mod->functions()[i];
        EquivResult fn_eq = verify_mir_equivalence(b_fn, a_fn);
        if (fn_eq.status != EquivStatus::ProvenEquivalent) {
            std::cerr << "[MIR equiv " << b_fn.name() << "]: " << fn_eq.status << " (" << fn_eq.diagnostic << ")\nAFTER:\n";
            print_function(a_fn, std::cerr);
        }
        CHECK_EQ(fn_eq.status, EquivStatus::ProvenEquivalent);
        verify_function_backends(a_fn);
    }

    // Check that redundant_guard_chain_kernel eliminated the implied #404 and #405 guards
    const Function* guard_fn = after_mod->get_function("redundant_guard_chain_kernel");
    REQUIRE(guard_fn != nullptr);
    size_t remaining_guards = 0;
    for (const auto& bb : guard_fn->blocks()) {
        for (const auto* inst : *bb) {
            if (inst->opcode() == Opcode::guard) remaining_guards++;
        }
    }
    CHECK_EQ(remaining_guards, size_t(3));
}

TEST_CASE("TV Corpus - CLI & AOT .mir File Translation Validation Sweep") {
    const std::vector<std::string> mir_files = {
        "tests/cli/div_zero.mir",
        "tests/cli/ret_i32.mir",
        "tests/cli/guard_param0_dispatch.mir",
        "tests/cli/guard_osr_stub.mir",
        "tests/cli/guard_stub_many.mir",
        "tests/cli/guard_stub_f64.mir"
    };

    size_t functions_checked = 0;
    size_t proven_count = 0;

    for (const std::string& rel_path : mir_files) {
        std::ifstream ifs(rel_path);
        if (!ifs.is_open()) ifs.open("/home/j/projects/brass/" + rel_path);
        if (!ifs.is_open()) ifs.open("../" + rel_path);
        if (!ifs.is_open()) continue;
        std::ostringstream ss;
        ss << ifs.rdbuf();

        DiagnosticReporter diag;
        auto before_mod = parse_module(ss.str(), &diag, rel_path);
        if (!before_mod || !verify_module(*before_mod, &diag)) continue;

        auto after_mod = clone_module(*before_mod);
        REQUIRE(after_mod != nullptr);
        run_all_mir_passes_with_solver(*after_mod);
        CHECK(verify_module(*after_mod, &diag));

        for (size_t i = 0; i < before_mod->functions().size(); ++i) {
            const Function& b_fn = *before_mod->functions()[i];
            const Function& a_fn = *after_mod->functions()[i];
            if (b_fn.blocks().empty()) continue;

            EquivResult eq = verify_mir_equivalence(b_fn, a_fn);
            CHECK(!eq.is_counterexample());
            if (eq.is_equivalent()) proven_count++;
            verify_function_backends(a_fn);
            functions_checked++;
        }
    }

    CHECK(functions_checked >= size_t(6));
    CHECK(proven_count >= size_t(4));
}

TEST_CASE("TV Corpus - DiffFuzzer smt_verify_passes & Solver Pass Ratchet") {
    // 1. DiffFuzzer AllPasses & Production with smt_verify_passes = true across ProgramGenerator seeds
    fuzz::ProgramGeneratorOptions gen_opts;
    gen_opts.max_depth = 1;
    gen_opts.max_statements = 10;
    gen_opts.max_iteration_product = 16;
    gen_opts.enable_gc = false;
    fuzz::ProgramGenerator gen(gen_opts);

    for (fuzz::FuzzPipeline pipe : {fuzz::FuzzPipeline::AllPasses, fuzz::FuzzPipeline::Production}) {
        fuzz::DiffFuzzerOptions fopts;
        fopts.pipeline = pipe;
        fopts.smt_verify_passes = true;
        fopts.tier6_baseline = true;
        fopts.save_reproducers = false;
        fuzz::DiffFuzzer fuzzer(fopts);

        for (uint64_t seed = 1; seed <= 2; ++seed) {
            Module mod("fuzz_mod_" + std::to_string(seed));
            Function* entry = gen.generate(mod, "fuzz_fn", seed * 1009 + 17);
            REQUIRE(entry != nullptr);

            fuzz::DiffResult res = fuzzer.run_test(
                mod, "fuzz_fn",
                {RuntimeValue::from_i64(static_cast<int64_t>(seed * 7 - 19)),
                 RuntimeValue::from_i64(static_cast<int64_t>(seed * 13 + 5))},
                seed);
            CHECK(res.passed);
        }
    }

    for (uint64_t seed = 1; seed <= 12; ++seed) {
        Module mod("tv_fuzz_" + std::to_string(seed));
        Function* entry = gen.generate(mod, "fuzz_fn", seed * 1009 + 17);
        REQUIRE(entry != nullptr);

        auto opt_mod = clone_module(mod);
        REQUIRE(opt_mod != nullptr);
        run_all_mir_passes_with_solver(*opt_mod);
        DiagnosticReporter diag;
        CHECK(verify_module(*opt_mod, &diag));

        for (size_t i = 0; i < mod.functions().size(); ++i) {
            const Function& b_fn = *mod.functions()[i];
            const Function& a_fn = *opt_mod->functions()[i];
            EquivResult eq = verify_mir_equivalence(b_fn, a_fn);
            CHECK(!eq.is_counterexample());
            verify_function_backends(a_fn);
        }
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
