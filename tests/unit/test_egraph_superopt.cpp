#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/egraph.hpp>
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

TEST_CASE("EGraph - Congruence Closure, Hashconsing, and Canonicalization") {
    EGraph eg;
    EClassId x = eg.add_var(0, 32);
    EClassId y = eg.add_var(1, 32);
    EClassId z = eg.add_var(2, 32);
    EClassId w = eg.add_var(3, 32);

    // Commutative hashconsing: x + y and y + x map to the same EClassId immediately
    EClassId add_xy = eg.add_node(ENode{EOp::Add, 32, x, y});
    EClassId add_yx = eg.add_node(ENode{EOp::Add, 32, y, x});
    CHECK_EQ(add_xy, add_yx);

    // Multi-level congruence closure:
    //   p1 = (x + z) * w
    //   p2 = (y + z) * w
    // Merging x and y must propagate upward to merge (x + z) with (y + z) and p1 with p2.
    EClassId xz = eg.add_node(ENode{EOp::Add, 32, x, z});
    EClassId yz = eg.add_node(ENode{EOp::Add, 32, y, z});
    EClassId p1 = eg.add_node(ENode{EOp::Mul, 32, xz, w});
    EClassId p2 = eg.add_node(ENode{EOp::Mul, 32, yz, w});
    CHECK(eg.find(p1) != eg.find(p2));

    CHECK(eg.merge(x, y));
    eg.rebuild();
    CHECK_EQ(eg.find(xz), eg.find(yz));
    CHECK_EQ(eg.find(p1), eg.find(p2));
}

TEST_CASE("EGraph - Equality Saturation and Min-Cost Extraction on Bitwise & Algebraic Identities") {
    // 1. (x | y) + (x & y) -> x + y
    {
        EGraph eg;
        EClassId x = eg.add_var(0, 32);
        EClassId y = eg.add_var(1, 32);
        EClassId or_xy = eg.add_node(ENode{EOp::Or, 32, x, y});
        EClassId and_xy = eg.add_node(ENode{EOp::And, 32, x, y});
        EClassId root = eg.add_node(ENode{EOp::Add, 32, or_xy, and_xy});

        EGraphStats stats;
        eg.saturate({}, &stats);
        CHECK(stats.rewrites_applied > 0u);

        ExtractedExpr best;
        REQUIRE(extract_min_cost(eg, root, best));
        CHECK_EQ(best.cost, 1u);
        CHECK(best.node.op == EOp::Add);
        REQUIRE_EQ(best.children.size(), 2u);
        CHECK(best.children[0].node.op == EOp::Var);
        CHECK(best.children[1].node.op == EOp::Var);
    }

    // 2. (x ^ y) + ((x & y) << 1) -> x + y
    {
        EGraph eg;
        EClassId x = eg.add_var(0, 64);
        EClassId y = eg.add_var(1, 64);
        EClassId c1 = eg.add_const(1, 64);
        EClassId xor_xy = eg.add_node(ENode{EOp::Xor, 64, x, y});
        EClassId and_xy = eg.add_node(ENode{EOp::And, 64, x, y});
        EClassId shl_and = eg.add_node(ENode{EOp::Shl, 64, and_xy, c1});
        EClassId root = eg.add_node(ENode{EOp::Add, 64, xor_xy, shl_and});

        eg.saturate();
        ExtractedExpr best;
        REQUIRE(extract_min_cost(eg, root, best));
        CHECK_EQ(best.cost, 1u);
        CHECK(best.node.op == EOp::Add);
    }

    // 3. (x & m) | (x & ~m) -> x and (x & m) ^ (x & ~m) -> x
    {
        EGraph eg;
        EClassId x = eg.add_var(0, 32);
        EClassId m = eg.add_var(1, 32);
        EClassId not_m = eg.add_node(ENode{EOp::Not, 32, m});
        EClassId xm = eg.add_node(ENode{EOp::And, 32, x, m});
        EClassId xnm = eg.add_node(ENode{EOp::And, 32, x, not_m});
        EClassId or_root = eg.add_node(ENode{EOp::Or, 32, xm, xnm});
        EClassId xor_root = eg.add_node(ENode{EOp::Xor, 32, xm, xnm});

        eg.saturate();
        CHECK_EQ(eg.find(or_root), eg.find(x));
        CHECK_EQ(eg.find(xor_root), eg.find(x));

        ExtractedExpr best_or, best_xor;
        REQUIRE(extract_min_cost(eg, or_root, best_or));
        REQUIRE(extract_min_cost(eg, xor_root, best_xor));
        CHECK_EQ(best_or.cost, 0u);
        CHECK(best_or.node.op == EOp::Var);
        CHECK_EQ(best_or.node.imm, 0u);
        CHECK_EQ(best_xor.cost, 0u);
        CHECK(best_xor.node.op == EOp::Var);
        CHECK_EQ(best_xor.node.imm, 0u);
    }

    // 4. Two's complement: ~x + 1 -> -x and -(x + 1) -> ~x
    {
        EGraph eg;
        EClassId x = eg.add_var(0, 32);
        EClassId c1 = eg.add_const(1, 32);
        EClassId not_x = eg.add_node(ENode{EOp::Not, 32, x});
        EClassId root_neg = eg.add_node(ENode{EOp::Add, 32, not_x, c1});

        EClassId y = eg.add_var(1, 32);
        EClassId yp1 = eg.add_node(ENode{EOp::Add, 32, y, c1});
        EClassId root_not = eg.add_node(ENode{EOp::Neg, 32, yp1});

        eg.saturate();

        ExtractedExpr best_neg, best_not;
        REQUIRE(extract_min_cost(eg, root_neg, best_neg));
        CHECK_EQ(best_neg.cost, 1u);
        CHECK(best_neg.node.op == EOp::Neg);

        REQUIRE(extract_min_cost(eg, root_not, best_not));
        CHECK_EQ(best_not.cost, 1u);
        CHECK(best_not.node.op == EOp::Not);
    }

    // 5. Multi-step distributive + shift strength reduction: (x * 3) + (x * 5) -> x * 8 -> x << 3
    {
        EGraph eg;
        EClassId x = eg.add_var(0, 32);
        EClassId c3 = eg.add_const(3, 32);
        EClassId c5 = eg.add_const(5, 32);
        EClassId m3 = eg.add_node(ENode{EOp::Mul, 32, x, c3});
        EClassId m5 = eg.add_node(ENode{EOp::Mul, 32, x, c5});
        EClassId root = eg.add_node(ENode{EOp::Add, 32, m3, m5});

        eg.saturate();
        ExtractedExpr best;
        REQUIRE(extract_min_cost(eg, root, best));
        CHECK_EQ(best.cost, 1u);
        CHECK(best.node.op == EOp::Shl);
        REQUIRE_EQ(best.children.size(), 2u);
        CHECK(best.children[0].node.op == EOp::Var);
        CHECK(best.children[1].node.op == EOp::Const);
        CHECK_EQ(best.children[1].node.imm, 3u);
    }

    // 6. Cancellations vs non-cancellations:
    //    (((x + y) - y) ^ y) ^ y -> x, whereas ((x + y) ^ y) - y does NOT cancel to x
    {
        EGraph eg;
        EClassId x = eg.add_var(0, 32);
        EClassId y = eg.add_var(1, 32);

        EClassId x_plus_y = eg.add_node(ENode{EOp::Add, 32, x, y});
        EClassId sub_y = eg.add_node(ENode{EOp::Sub, 32, x_plus_y, y});
        EClassId xor1 = eg.add_node(ENode{EOp::Xor, 32, sub_y, y});
        EClassId valid_cancel = eg.add_node(ENode{EOp::Xor, 32, xor1, y});

        EClassId bad_xor = eg.add_node(ENode{EOp::Xor, 32, x_plus_y, y});
        EClassId invalid_cancel = eg.add_node(ENode{EOp::Sub, 32, bad_xor, y});

        eg.saturate();
        CHECK_EQ(eg.find(valid_cancel), eg.find(x));
        CHECK(eg.find(invalid_cancel) != eg.find(x));
    }
}

TEST_CASE("EGraph - Strict Budget Enforcement (max_enodes, max_eclasses, max_iters)") {
    // 1. Zero iterations budget: saturate must do 0 iterations and 0 rewrites
    {
        EGraph eg;
        EClassId x = eg.add_var(0, 32);
        EClassId c0 = eg.add_const(0, 32);
        EClassId add0 = eg.add_node(ENode{EOp::Add, 32, x, c0});

        EGraphLimits lim;
        lim.max_iters = 0;
        EGraphStats stats;
        CHECK_EQ(eg.saturate(lim, &stats), 0u);
        CHECK_EQ(stats.saturation_iters, 0u);
        CHECK_EQ(stats.rewrites_applied, 0u);
        CHECK(eg.find(add0) != eg.find(x));
    }

    // 2. Hard max_enodes cap on an associative chain: must terminate immediately at cap
    {
        EGraph eg;
        EClassId cur = eg.add_var(0, 32);
        for (uint32_t i = 1; i <= 12; ++i) {
            EClassId v = eg.add_var(i, 32);
            cur = eg.add_node(ENode{EOp::Add, 32, cur, v});
        }
        const uint32_t initial_nodes = eg.num_enodes();
        EGraphLimits lim;
        lim.max_enodes = initial_nodes + 6;
        lim.max_eclasses = 512;
        lim.max_iters = 10;

        EGraphStats stats;
        eg.saturate(lim, &stats);
        CHECK(eg.num_enodes() <= lim.max_enodes);
        CHECK(stats.budget_hit >= 1u);
    }

    // 3. Hard max_eclasses cap: must terminate immediately at class cap
    {
        EGraph eg;
        EClassId cur = eg.add_var(0, 32);
        for (uint32_t i = 1; i <= 10; ++i) {
            EClassId v = eg.add_var(i, 32);
            cur = eg.add_node(ENode{EOp::Xor, 32, cur, v});
        }
        const uint32_t initial_classes = eg.num_eclasses();
        EGraphLimits lim;
        lim.max_enodes = 1024;
        lim.max_eclasses = initial_classes + 4;
        lim.max_iters = 10;

        EGraphStats stats;
        eg.saturate(lim, &stats);
        CHECK(eg.num_eclasses() <= lim.max_eclasses);
        CHECK(stats.budget_hit >= 1u);
    }
}

TEST_CASE("EGraph MIR Superopt - End-to-End Multi-Step Simplification & Solver Verification") {
    // 1. (x | y) + (x & y) and (x ^ y) + ((x & y) << 1) simplified to (x + y) + (x + y) -> (x + y) << 1
    {
        Module mod("egraph_mir_arith_bit");
        Builder b(mod);
        Function* fn = mod.create_function("bitwise_add_identities", Type::i32(), {Type::i32(), Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i32());
        Value* y = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* or_xy = b.build_or(x, y);
        Value* and_xy = b.build_and(x, y);
        Value* sum1 = b.build_add(or_xy, and_xy);

        Value* xor_xy = b.build_xor(x, y);
        Value* and_xy2 = b.build_and(x, y);
        Value* shl_and = b.build_shl(and_xy2, b.build_iconst_i32(1));
        Value* sum2 = b.build_add(xor_xy, shl_and);

        Value* total = b.build_add(sum1, sum2);
        b.build_ret(total);
        fn->rebuild_cfg_predecessors();

        auto before_mod = clone_module(mod);
        SuperoptStats stats;
        SuperoptOptions opts;
        opts.enable_egraph = true;
        opts.stats = &stats;

        CHECK(run_mir_superopt(*fn, mod, opts));
        CHECK(stats.egraph_rewrites >= 1u);
        CHECK(stats.solver_queries >= 1u);
        CHECK_EQ(count_opcode(*fn, Opcode::or_), 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::and_), 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::xor_), 0u);

        DiagnosticReporter diag;
        CHECK(verify_module(mod, &diag));
        CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("bitwise_add_identities"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);
    }

    // 2. Partitioned bitmask & two's complement: ((x & m) | (x & ~m)) + (~y + 1) -> x - y
    {
        Module mod("egraph_mir_mask_twos");
        Builder b(mod);
        Function* fn = mod.create_function("mask_twos", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        Value* m = b.add_block_param(entry, Type::i64());
        b.position_at_end(entry);

        Value* xm = b.build_and(x, m);
        Value* not_m = b.build_not(m);
        Value* xnm = b.build_and(x, not_m);
        Value* x_recon = b.build_or(xm, xnm);

        Value* not_y = b.build_not(y);
        Value* neg_y = b.build_add(not_y, b.build_iconst_i64(1));
        Value* diff = b.build_add(x_recon, neg_y);
        b.build_ret(diff);
        fn->rebuild_cfg_predecessors();

        auto before_mod = clone_module(mod);
        SuperoptStats stats;
        SuperoptOptions opts;
        opts.enable_egraph = true;
        opts.stats = &stats;

        CHECK(run_mir_superopt(*fn, mod, opts));
        CHECK(stats.egraph_rewrites >= 1u);
        CHECK_EQ(count_opcode(*fn, Opcode::and_), 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::or_), 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::not_), 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::sub), 1u);

        DiagnosticReporter diag;
        CHECK(verify_module(mod, &diag));
        CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("mask_twos"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);
    }

    // 3. Multi-step distributive factoring + strength reduction: (x * 3) + (x * 5) -> x << 3
    {
        Module mod("egraph_mir_mul_factor");
        Builder b(mod);
        Function* fn = mod.create_function("mul_factor_shift", Type::i32(), {Type::i32()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i32());
        b.position_at_end(entry);

        Value* m3 = b.build_mul(x, b.build_iconst_i32(3));
        Value* m5 = b.build_mul(x, b.build_iconst_i32(5));
        Value* sum = b.build_add(m3, m5);
        b.build_ret(sum);
        fn->rebuild_cfg_predecessors();

        auto before_mod = clone_module(mod);
        SuperoptStats stats;
        SuperoptOptions opts;
        opts.enable_egraph = true;
        opts.stats = &stats;

        CHECK(run_mir_superopt(*fn, mod, opts));
        CHECK(stats.egraph_rewrites >= 1u);
        CHECK_EQ(count_opcode(*fn, Opcode::mul), 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::add), 0u);
        CHECK_EQ(count_opcode(*fn, Opcode::shl), 1u);

        DiagnosticReporter diag;
        CHECK(verify_module(mod, &diag));
        CHECK_EQ(verify::verify_mir_equivalence(*before_mod->get_function("mul_factor_shift"), *fn).status,
                 verify::EquivStatus::ProvenEquivalent);
    }
}

#ifdef BRASS_STANDALONE_TEST
BRASS_TEST_MAIN()
#endif
