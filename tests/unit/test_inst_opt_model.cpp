#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/egraph.hpp>
#include <brass/mir/inst_opt_model.hpp>
#include <brass/mir/superopt.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/verify/smt_equiv.hpp>
#include <cmath>
#include <filesystem>

using namespace brass;

namespace {

InstOptModel make_synthetic_test_model() {
    InstOptModel m;
    for (uint32_t i = 0; i < kInstOptHidden1; ++i) {
        m.b1[i] = 0.01f * static_cast<float>((i % 7) + 1);
        for (uint32_t j = 0; j < kInstOptInputDim; ++j) {
            m.w1[i * kInstOptInputDim + j] = 0.02f * std::sin(static_cast<float>(i * 3 + j));
        }
    }
    for (uint32_t i = 0; i < kInstOptHidden2; ++i) {
        m.b2[i] = 0.005f;
        for (uint32_t j = 0; j < kInstOptHidden1; ++j) {
            m.w2[i * kInstOptHidden1 + j] = 0.015f * std::cos(static_cast<float>(i + j * 5));
        }
    }
    for (uint32_t i = 0; i < kInstOptHidden3; ++i) {
        m.b3[i] = 0.02f;
        m.w_cost[i] = 0.1f * static_cast<float>((i % 5) + 1);
        for (uint32_t j = 0; j < kInstOptHidden2; ++j) {
            m.w3[i * kInstOptHidden2 + j] = 0.01f * std::sin(static_cast<float>(i * 7 + j));
        }
    }
    m.b_cost = 0.2f;
    for (uint32_t a = 0; a < kNumInstOptActions; ++a) {
        m.b_pol[a] = (a == 0) ? 2.5f : -1.0f;
        for (uint32_t c = 0; c < kInstOptHidden3; ++c) {
            m.w_pol[a * kInstOptHidden3 + c] = 0.01f * static_cast<float>((a == c % kNumInstOptActions) ? 1 : -1);
        }
    }
    m.loaded = true;
    return m;
}

uint32_t expr_depth(const ExtractedExpr& e) noexcept {
    uint32_t d = 0;
    for (const auto& ch : e.children) {
        d = std::max(d, expr_depth(ch));
    }
    return e.children.empty() ? 0u : (d + 1u);
}

} // namespace

TEST_CASE("InstOptModel - Features, Serialization, and Online Calibration") {
    EGraph eg;
    const EClassId a = eg.add_var(0, 64);
    const EClassId b = eg.add_var(1, 64);
    const EClassId c = eg.add_var(2, 64);
    const EClassId d = eg.add_var(3, 64);
    const EClassId s1 = eg.add_node(ENode{EOp::Add, 64, a, b});
    const EClassId s2 = eg.add_node(ENode{EOp::Add, 64, s1, c});
    const EClassId root = eg.add_node(ENode{EOp::Add, 64, s2, d});
    eg.saturate();

    std::vector<ExtractedExpr> cands;
    REQUIRE(extract_top_k_candidates(eg, root, 12, cands));
    CHECK(cands.size() >= 2u);

    bool has_depth2 = false;
    bool has_depth3 = false;
    for (const auto& cand : cands) {
        const uint32_t dp = expr_depth(cand);
        if (dp == 2u) has_depth2 = true;
        if (dp == 3u) has_depth3 = true;
        const InstOptFeatures feat = extract_inst_opt_features(cand, InstOptTarget::X64, false);
        CHECK(feat.x[54] > 0.0f);
        CHECK(feat.x[58] > 0.99f); // all 64-bit ops
    }
    CHECK(has_depth2);
    CHECK(has_depth3);

    // .brim INT8 and FP32 round-trip check
    const InstOptModel syn = make_synthetic_test_model();
    const std::filesystem::path p_int8 = std::filesystem::temp_directory_path() / "brass_inst_opt_test_int8.brim";
    const std::filesystem::path p_fp32 = std::filesystem::temp_directory_path() / "brass_inst_opt_test_fp32.brim";
    REQUIRE(syn.save_file(p_int8.string(), true));
    REQUIRE(syn.save_file(p_fp32.string(), false));

    InstOptModel loaded_int8, loaded_fp32;
    REQUIRE(InstOptModel::load_file(p_int8.string(), loaded_int8));
    REQUIRE(InstOptModel::load_file(p_fp32.string(), loaded_fp32));
    CHECK(loaded_int8.loaded);
    CHECK(loaded_fp32.loaded);

    const InstOptFeatures f0 = extract_inst_opt_features(cands[0], InstOptTarget::X64, false);
    const InstOptPrediction p_orig = syn.predict(f0);
    const InstOptPrediction p_fp = loaded_fp32.predict(f0);
    const InstOptPrediction p_i8 = loaded_int8.predict(f0);
    CHECK(std::fabs(p_orig.predicted_cost - p_fp.predicted_cost) < 1e-4f);
    CHECK(std::fabs(p_orig.predicted_cost - p_i8.predicted_cost) < 0.05f);

    float sum_p = 0.0f;
    for (float pr : p_i8.action_probs) {
        CHECK(pr >= 0.0f);
        sum_p += pr;
    }
    CHECK(std::fabs(sum_p - 1.0f) < 1e-4f);

    // OnlineCostCalibrator reduces residual against observed hardware timing
    OnlineCostCalibrator cal;
    const float target_cost = p_orig.predicted_cost + 3.5f;
    const float err_before = std::fabs(syn.predict_cost(f0, &cal) - target_cost);
    for (int step = 0; step < 30; ++step) {
        cal.observe(p_orig, target_cost, 0.1f);
    }
    const float err_after = std::fabs(syn.predict_cost(f0, &cal) - target_cost);
    CHECK(err_after < err_before * 0.25f);

    std::error_code ec;
    std::filesystem::remove(p_int8, ec);
    std::filesystem::remove(p_fp32, ec);
}

TEST_CASE("InstOptModel - Policy Filtering and Learned EGraph Superopt") {
    // 1. Policy filtering in run_mir_superopt skips low-probability solver queries safely
    {
        Module mod("inst_opt_policy_filter");
        Builder b(mod);
        Function* fn = mod.create_function("mask_check", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);
        b.build_guard(b.build_sge(x, b.build_iconst_i32(0)), "deopt");
        b.build_guard(b.build_slt(x, b.build_iconst_i32(256)), "deopt");
        b.build_ret(b.build_and(x, b.build_iconst_i32(0xFF)));
        fn->rebuild_cfg_predecessors();

        InstOptModel noop_mdl;
        noop_mdl.loaded = true;
        noop_mdl.b_pol.fill(-6.0f);
        noop_mdl.b_pol[static_cast<size_t>(InstOptAction::NoOp)] = 6.0f;

        SuperoptStats stats;
        SuperoptOptions opts;
        opts.enable_egraph = false;
        opts.use_inst_opt_model = true;
        opts.inst_opt_model = &noop_mdl;
        opts.model_policy_min_prob = 0.05f;
        opts.stats = &stats;

        run_mir_superopt(*fn, mod, opts);
        CHECK(stats.solver_queries_skipped_by_model >= 1u);
    }

    // 2. Learned builtin weights verify realistic latency & rewrite ordering
    const InstOptModel& builtin = InstOptModel::builtin();
    REQUIRE(builtin.loaded);
    {
        ExtractedExpr e_add{kInvalidEClass, 1, ENode{EOp::Add, 64}, nullptr, {
            ExtractedExpr{kInvalidEClass, 0, ENode{EOp::Var, 64, kInvalidEClass, kInvalidEClass, kInvalidEClass, 0}},
            ExtractedExpr{kInvalidEClass, 0, ENode{EOp::Var, 64, kInvalidEClass, kInvalidEClass, kInvalidEClass, 1}}
        }};
        ExtractedExpr e_mul = e_add;
        e_mul.node.op = EOp::Mul;
        e_mul.cost = 3;
        ExtractedExpr e_div = e_add;
        e_div.node.op = EOp::SDiv;
        e_div.cost = 8;

        const float c_add = builtin.predict_cost(extract_inst_opt_features(e_add, InstOptTarget::X64));
        const float c_mul = builtin.predict_cost(extract_inst_opt_features(e_mul, InstOptTarget::X64));
        const float c_div = builtin.predict_cost(extract_inst_opt_features(e_div, InstOptTarget::X64));
        CHECK(c_add < c_mul);
        CHECK(c_mul < c_div);

        // EGraph superopt with builtin model balances deep serial add chain and verifies equivalence
        Module mod("inst_opt_egraph_balance");
        Builder b(mod);
        Function* fn = mod.create_function("serial_sum4", Type::i64(), {Type::i64(), Type::i64(), Type::i64(), Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* va = b.add_block_param(entry, Type::i64());
        Value* vb = b.add_block_param(entry, Type::i64());
        Value* vc = b.add_block_param(entry, Type::i64());
        Value* vd = b.add_block_param(entry, Type::i64());
        b.position_at_end(entry);
        Value* s1 = b.build_add(va, vb);
        Value* s2 = b.build_add(s1, vc);
        Value* s3 = b.build_add(s2, vd);
        b.build_ret(s3);
        fn->rebuild_cfg_predecessors();

        auto before = clone_module(mod);
        SuperoptStats stats;
        SuperoptOptions opts;
        opts.enable_egraph = true;
        opts.use_inst_opt_model = true;
        opts.stats = &stats;

        const bool changed = run_mir_egraph_superopt(*fn, mod, opts);
        CHECK(changed);
        CHECK(stats.model_guided_rewrites >= 1u);
        CHECK_EQ(verify::verify_mir_equivalence(*before->get_function("serial_sum4"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);

        // Also verify run_mir_superopt with builtin trained model preserves mask elision
        Module mod2("inst_opt_superopt_builtin");
        Builder b2(mod2);
        Function* f2 = mod2.create_function("mask_builtin", Type::i32(), {Type::i32()});
        b2.set_function(f2);
        BasicBlock* e2 = b2.append_block("entry");
        Value* x2 = b2.add_block_param(e2, Type::i32());
        b2.position_at_end(e2);
        b2.build_guard(b2.build_sge(x2, b2.build_iconst_i32(0)), "deopt");
        b2.build_guard(b2.build_slt(x2, b2.build_iconst_i32(256)), "deopt");
        b2.build_ret(b2.build_and(x2, b2.build_iconst_i32(0xFF)));
        f2->rebuild_cfg_predecessors();

        auto before2 = clone_module(mod2);
        SuperoptStats st2;
        SuperoptOptions o2;
        o2.use_inst_opt_model = true;
        o2.stats = &st2;
        CHECK(run_mir_superopt(*f2, mod2, o2));
        CHECK_EQ(verify::verify_mir_equivalence(*before2->get_function("mask_builtin"), *f2).status,
                 verify::EquivStatus::ProvenEquivalent);
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
