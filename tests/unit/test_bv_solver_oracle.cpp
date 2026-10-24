#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/verify/bv_solver.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>
#include <vector>

using namespace brass;
using namespace brass::verify;

TEST_CASE("BVSolver #153 - Soundness Regressions: Unsupported Op Unknown, Bit-Sliced MSB, and Interval Bounds") {
    // 1. Unsupported floating-point op whose uninterpreted SAT model fails eval_concrete
    //    must return SolveOutcome::Unknown (never SolveOutcome::Unsat).
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 32, true);
        BVNodeId y = ctx.make_var("y", 32, true);
        BVNodeId fsum = ctx.make_binary(BVOp::FAdd, x, y, 32);
        // Use 5 variables so nvars > 4 (bypassing Stage 1 pair search), while Stage 2 bit-blaster
        // treats FAdd as uninterpreted.
        BVNodeId z1 = ctx.make_var("z1", 32, false);
        BVNodeId z2 = ctx.make_var("z2", 32, false);
        BVNodeId z3 = ctx.make_var("z3", 32, false);
        BVNodeId dom = ctx.make_binary(BVOp::And,
            ctx.make_binary(BVOp::Eq, ctx.make_binary(BVOp::Add, z1, z2, 32), ctx.make_const(0x12345678u, 32), 1),
            ctx.make_binary(BVOp::Eq, ctx.make_binary(BVOp::Xor, z2, z3, 32), ctx.make_const(0x9ABCDEF0u, 32), 1), 1);
        BVNodeId f_cond = ctx.make_binary(BVOp::Eq, fsum, ctx.make_const(0x40400000u, 32), 1);
        BVNodeId bad = ctx.make_binary(BVOp::And, dom, f_cond, 1);
        SolveOutcome out = solve_nonzero(ctx, bad, 10000);
        // Satisfiable (e.g. x = 1.0f, y = 2.0f, z2 = 0, z1 = 0x12345678, z3 = 0x9ABCDEF0),
        // so solver must NEVER claim Unsat!
        CHECK(out.status != SolveOutcome::Unsat);
    }

    // 2. Pin bit-sliced `(a != b) && dom` path where `a` and `b` differ ONLY in the MSB under `dom`.
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 16, false);
        BVNodeId y = ctx.make_var("y", 16, false);
        // Under dom: (x ^ 0x5A3C) == 0x8000 && y == 0x5A3C -> x = 0xDA3C, y = 0x5A3C (differ ONLY in bit 15, the MSB!)
        BVNodeId d1 = ctx.make_binary(BVOp::Eq,
            ctx.make_binary(BVOp::Xor, x, ctx.make_const(0x5A3Cu, 16), 16),
            ctx.make_const(0x8000u, 16), 1);
        BVNodeId d2 = ctx.make_binary(BVOp::Eq,
            ctx.make_binary(BVOp::Add, y, ctx.make_const(0x1111u, 16), 16),
            ctx.make_const(0x5A3Cu + 0x1111u, 16), 1);
        BVNodeId dom = ctx.make_binary(BVOp::And, d1, d2, 1);
        BVNodeId neq = ctx.make_binary(BVOp::Ne, x, y, 1);
        BVNodeId bad = ctx.make_binary(BVOp::And, neq, dom, 1);

        SolveOutcome res = solve_nonzero(ctx, bad, 10000);
        CHECK_EQ(res.status, SolveOutcome::Sat);
        REQUIRE_EQ(res.model.size(), 2u);
        CHECK_EQ(ctx.eval_concrete(bad, res.model), 1u);
        CHECK_EQ(res.model[0], 0xDA3Cu);
        CHECK_EQ(res.model[1], 0x5A3Cu);
    }

    // 3. Pin `Or`-rooted query that must skip the top-level `And` bit-sliced shortcut.
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 16, false);
        BVNodeId y = ctx.make_var("y", 16, false);
        BVNodeId neq = ctx.make_binary(BVOp::Ne, x, y, 1);
        BVNodeId eq_magic = ctx.make_binary(BVOp::Eq, x, ctx.make_const(0xBEEFu, 16), 1);
        BVNodeId or_root = ctx.make_binary(BVOp::Or, neq, eq_magic, 1);

        SolveOutcome res = solve_nonzero(ctx, or_root, 10000);
        CHECK_EQ(res.status, SolveOutcome::Sat);
        CHECK_EQ(ctx.eval_concrete(or_root, res.model), 1u);
    }

    // 4. Interval soundness regressions for BVOp::And (umax) and BVOp::Or (umin)
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 8, false);
        BVNodeId c = ctx.make_extract(x, 1, 1);
        // u = ite(c, 4, 3) -> umin = 3, umax = 4, known_zeros = ~7 (0xF8)
        BVNodeId u = ctx.make_ite(c, ctx.make_const(4, 8), ctx.make_const(3, 8));
        BVNodeId u_and_3 = ctx.make_binary(BVOp::And, u, ctx.make_const(3, 8), 8);
        BVNodeId eq_3 = ctx.make_binary(BVOp::Eq, u_and_3, ctx.make_const(3, 8), 1);
        SolveOutcome r_and = solve_nonzero(ctx, eq_3, 1000);
        CHECK_EQ(r_and.status, SolveOutcome::Sat);
        CHECK_EQ(ctx.eval_concrete(eq_3, r_and.model), 1u);

        // BVOp::Or umin regression:
        // p = ite(c, 5, 6) -> umin = 5, umax = 6, known_ones = 4 (0b100)
        // q = 2 -> umin = 2, umax = 2, known_ones = 2 (0b010)
        // When c = 0, p = 6, so (p | 2) = 6 | 2 = 6!
        BVNodeId p = ctx.make_ite(c, ctx.make_const(5, 8), ctx.make_const(6, 8));
        BVNodeId p_or_2 = ctx.make_binary(BVOp::Or, p, ctx.make_const(2, 8), 8);
        BVNodeId eq_6 = ctx.make_binary(BVOp::Eq, p_or_2, ctx.make_const(6, 8), 1);
        SolveOutcome r_or = solve_nonzero(ctx, eq_6, 1000);
        CHECK_EQ(r_or.status, SolveOutcome::Sat);
        CHECK_EQ(ctx.eval_concrete(eq_6, r_or.model), 1u);
    }
}

TEST_CASE("BVSolver #153 - Per-Op Differential Test Across Widths 1, 7, 8, 32, 64") {
    const uint32_t widths[] = {1, 7, 8, 32, 64};
    const BVOp bin_ops[] = {
        BVOp::And, BVOp::Or, BVOp::Xor, BVOp::Add, BVOp::Sub, BVOp::Mul,
        BVOp::UDiv, BVOp::SDiv, BVOp::URem, BVOp::SRem,
        BVOp::Shl, BVOp::LShr, BVOp::AShr,
        BVOp::Eq, BVOp::Ne, BVOp::Ult, BVOp::Ule, BVOp::Ugt, BVOp::Uge,
        BVOp::Slt, BVOp::Sle, BVOp::Sgt, BVOp::Sge,
        BVOp::SAddOvf, BVOp::SSubOvf, BVOp::SMulOvf,
        BVOp::UAddOvf, BVOp::USubOvf, BVOp::UMulOvf
    };
    const BVOp un_ops[] = {
        BVOp::Not, BVOp::Neg, BVOp::Clz, BVOp::Ctz, BVOp::Popcnt
    };

    uint64_t rng = 0xCAFEBABE12345678ULL;
    auto next_rng = [&]() {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        return rng * 0x2545F4914F6CDD1DULL;
    };

    for (uint32_t w : widths) {
        const uint64_t m = width_mask(w);
        const uint64_t smin_u = (w >= 64) ? (1ULL << 63) : (1ULL << (w - 1));
        const uint64_t smax_u = smin_u - 1ULL;

        std::vector<uint64_t> edge_vals = {
            0ULL,
            1ULL & m,
            m,
            smin_u & m,
            smax_u & m,
            static_cast<uint64_t>(w - 1) & m,
            static_cast<uint64_t>(w) & m,
            static_cast<uint64_t>(w + 1) & m,
            next_rng() & m,
            next_rng() & m
        };

        // 1. Test all unary ops at width w
        for (BVOp uop : un_ops) {
            for (uint64_t cx : edge_vals) {
                BVContext ctx;
                BVNodeId x = ctx.make_var("x", w, false);
                BVNodeId z = ctx.make_var("z", w, false);
                BVNodeId op_node = ctx.make_unary(uop, x, w);
                uint64_t expected_z = ctx.eval_concrete(op_node, {cx, 0});

                BVNodeId cond = ctx.make_binary(BVOp::And,
                    ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                    ctx.make_binary(BVOp::Eq, z, op_node, 1), 1);
                SolveOutcome sat_res = solve_nonzero(ctx, cond, 10000);
                CHECK_EQ(sat_res.status, SolveOutcome::Sat);
                REQUIRE_EQ(sat_res.model.size(), 2u);
                CHECK_EQ(sat_res.model[1], expected_z);

                BVNodeId bad_neq = ctx.make_binary(BVOp::And,
                    ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                    ctx.make_binary(BVOp::Ne, op_node, ctx.make_const(expected_z, w), 1), 1);
                SolveOutcome unsat_res = solve_nonzero(ctx, bad_neq, 10000);
                CHECK_EQ(unsat_res.status, SolveOutcome::Unsat);
            }
        }

        // 2. Test Extract, ZExt, SExt, Concat, Ite at width w
        for (uint64_t cx : edge_vals) {
            uint64_t cy = edge_vals[(cx + 3) % edge_vals.size()];
            if (w <= 59) {
                for (bool is_sext : {false, true}) {
                    BVContext ctx;
                    BVNodeId x = ctx.make_var("x", w, false);
                    BVNodeId z = ctx.make_var("z", w + 5, false);
                    BVNodeId ext = is_sext ? ctx.make_sext(x, 5) : ctx.make_zext(x, 5);
                    uint64_t expected = ctx.eval_concrete(ext, {cx, 0});
                    BVNodeId bad_neq = ctx.make_binary(BVOp::And,
                        ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                        ctx.make_binary(BVOp::Ne, ext, ctx.make_const(expected, w + 5), 1), 1);
                    CHECK_EQ(solve_nonzero(ctx, bad_neq, 10000).status, SolveOutcome::Unsat);
                    (void)z;
                }
            }
            {
                uint32_t hi = (w > 1) ? (w - 1) : 0;
                uint32_t lo = (w > 2) ? 1 : 0;
                uint32_t ew = hi - lo + 1;
                BVContext ctx;
                BVNodeId x = ctx.make_var("x", w, false);
                BVNodeId ex = ctx.make_extract(x, hi, lo);
                uint64_t expected = ctx.eval_concrete(ex, {cx});
                BVNodeId bad_neq = ctx.make_binary(BVOp::And,
                    ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                    ctx.make_binary(BVOp::Ne, ex, ctx.make_const(expected, ew), 1), 1);
                CHECK_EQ(solve_nonzero(ctx, bad_neq, 10000).status, SolveOutcome::Unsat);
            }
            if (w <= 32) {
                BVContext ctx;
                BVNodeId x = ctx.make_var("x", w, false);
                BVNodeId y = ctx.make_var("y", w, false);
                BVNodeId c = ctx.make_var("c", 1, false);
                BVNodeId cat = ctx.make_concat(x, y);
                BVNodeId ite = ctx.make_ite(c, x, y);
                uint64_t exp_cat = ctx.eval_concrete(cat, {cx, cy, 1});
                uint64_t exp_ite = ctx.eval_concrete(ite, {cx, cy, 1});
                BVNodeId fix = ctx.make_binary(BVOp::And,
                    ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                    ctx.make_binary(BVOp::And,
                        ctx.make_binary(BVOp::Eq, y, ctx.make_const(cy, w), 1),
                        ctx.make_binary(BVOp::Eq, c, ctx.make_const(1, 1), 1), 1), 1);
                CHECK_EQ(solve_nonzero(ctx, ctx.make_binary(BVOp::And, fix,
                    ctx.make_binary(BVOp::Ne, cat, ctx.make_const(exp_cat, 2 * w), 1), 1), 10000).status,
                    SolveOutcome::Unsat);
                CHECK_EQ(solve_nonzero(ctx, ctx.make_binary(BVOp::And, fix,
                    ctx.make_binary(BVOp::Ne, ite, ctx.make_const(exp_ite, w), 1), 1), 10000).status,
                    SolveOutcome::Unsat);
            }
        }

        // 3. Test all binary ops at width w across critical pairs (including INT_MIN / -1, div-by-0, shift >= w)
        std::vector<std::pair<uint64_t, uint64_t>> pairs = {
            {smin_u & m, m},                       // INT_MIN, -1
            {smin_u & m, 0ULL},                    // INT_MIN, 0 (div/rem by 0)
            {m, 0ULL},                             // -1, 0 (div/rem by 0)
            {smax_u & m, 1ULL & m},                // INT_MAX, 1
            {m, static_cast<uint64_t>(w) & m},     // shift == width
            {m, static_cast<uint64_t>(w + 1) & m}, // shift > width
            {smin_u & m, static_cast<uint64_t>(w - 1) & m},
            {edge_vals[8], edge_vals[9]},
            {edge_vals[9], edge_vals[8]}
        };

        for (BVOp bop : bin_ops) {
            bool is_pred = (bop == BVOp::Eq || bop == BVOp::Ne ||
                            bop == BVOp::Ult || bop == BVOp::Ule || bop == BVOp::Ugt || bop == BVOp::Uge ||
                            bop == BVOp::Slt || bop == BVOp::Sle || bop == BVOp::Sgt || bop == BVOp::Sge ||
                            bop == BVOp::SAddOvf || bop == BVOp::SSubOvf || bop == BVOp::SMulOvf ||
                            bop == BVOp::UAddOvf || bop == BVOp::USubOvf || bop == BVOp::UMulOvf);
            uint32_t zw = is_pred ? 1u : w;

            for (const auto& [cx, cy] : pairs) {
                BVContext ctx;
                BVNodeId x = ctx.make_var("x", w, false);
                BVNodeId y = ctx.make_var("y", w, false);
                BVNodeId z = ctx.make_var("z", zw, false);
                BVNodeId op_node = ctx.make_binary(bop, x, y, zw);
                uint64_t expected_z = ctx.eval_concrete(op_node, {cx, cy, 0});

                BVNodeId fix_xy = ctx.make_binary(BVOp::And,
                    ctx.make_binary(BVOp::Eq, x, ctx.make_const(cx, w), 1),
                    ctx.make_binary(BVOp::Eq, y, ctx.make_const(cy, w), 1), 1);

                BVNodeId sat_q = ctx.make_binary(BVOp::And, fix_xy,
                    ctx.make_binary(BVOp::Eq, z, op_node, 1), 1);
                SolveOutcome sat_res = solve_nonzero(ctx, sat_q, 10000);
                CHECK_EQ(sat_res.status, SolveOutcome::Sat);
                REQUIRE_EQ(sat_res.model.size(), 3u);
                CHECK_EQ(sat_res.model[2], expected_z);

                BVNodeId unsat_q = ctx.make_binary(BVOp::And, fix_xy,
                    ctx.make_binary(BVOp::Ne, op_node, ctx.make_const(expected_z, zw), 1), 1);
                SolveOutcome unsat_res = solve_nonzero(ctx, unsat_q, 10000);
                CHECK_EQ(unsat_res.status, SolveOutcome::Unsat);
            }
        }
    }
}

TEST_CASE("BVSolver #153 - Exhaustive 4-Bit Truth-Table Cross-Check on Random DAGs") {
    uint64_t rng = 0x7F4A7C159E3779B9ULL;
    auto next_rng = [&]() {
        rng ^= rng >> 12;
        rng ^= rng << 25;
        rng ^= rng >> 27;
        return rng * 0x2545F4914F6CDD1DULL;
    };

    const BVOp ops[] = {
        BVOp::And, BVOp::Or, BVOp::Xor, BVOp::Add, BVOp::Sub, BVOp::Mul,
        BVOp::UDiv, BVOp::SDiv, BVOp::URem, BVOp::SRem,
        BVOp::Shl, BVOp::LShr, BVOp::AShr
    };
    const BVOp cmp_ops[] = {
        BVOp::Eq, BVOp::Ne, BVOp::Ult, BVOp::Ule, BVOp::Ugt, BVOp::Uge,
        BVOp::Slt, BVOp::Sle, BVOp::Sgt, BVOp::Sge,
        BVOp::SAddOvf, BVOp::SSubOvf, BVOp::SMulOvf,
        BVOp::UAddOvf, BVOp::USubOvf, BVOp::UMulOvf
    };

    uint32_t sat_count = 0;
    uint32_t unsat_count = 0;

    for (uint32_t trial = 0; trial < 200; ++trial) {
        BVContext ctx;
        BVNodeId v0 = ctx.make_var("x", 4, false);
        BVNodeId v1 = ctx.make_var("y", 4, false);
        BVNodeId v2 = ctx.make_var("z", 4, false);
        std::vector<BVNodeId> pool = {
            v0, v1, v2,
            ctx.make_const(next_rng() & 0xFULL, 4),
            ctx.make_const(next_rng() & 0xFULL, 4)
        };

        for (int step = 0; step < 6; ++step) {
            uint64_t r = next_rng();
            BVNodeId a = pool[static_cast<size_t>(r % pool.size())];
            BVNodeId b = pool[static_cast<size_t>((r >> 8) % pool.size())];
            uint32_t kind = static_cast<uint32_t>((r >> 16) % 10);
            if (kind == 0) {
                pool.push_back(ctx.make_unary(BVOp::Not, a, 4));
            } else if (kind == 1) {
                pool.push_back(ctx.make_unary(BVOp::Neg, a, 4));
            } else if (kind == 2) {
                BVOp uop = ((r >> 24) % 3 == 0) ? BVOp::Clz : (((r >> 24) % 3 == 1) ? BVOp::Ctz : BVOp::Popcnt);
                pool.push_back(ctx.make_unary(uop, a, 4));
            } else if (kind == 3) {
                BVNodeId c = ctx.make_binary(cmp_ops[(r >> 24) % 16], a, b, 1);
                BVNodeId d = pool[static_cast<size_t>((r >> 32) % pool.size())];
                pool.push_back(ctx.make_ite(c, a, d));
            } else {
                BVOp bop = ops[(r >> 24) % (sizeof(ops) / sizeof(ops[0]))];
                pool.push_back(ctx.make_binary(bop, a, b, 4));
            }
        }

        BVNodeId lhs = pool[pool.size() - 1];
        BVNodeId rhs = pool[pool.size() - 2];
        BVOp root_cmp = cmp_ops[next_rng() % 16];
        BVNodeId c1 = ctx.make_binary(root_cmp, lhs, rhs, 1);
        BVNodeId c2 = ctx.make_binary(cmp_ops[next_rng() % 16], pool[pool.size() - 3], v0, 1);
        BVNodeId bad_cond = ((trial & 1u) == 0)
            ? ctx.make_binary(BVOp::And, c1, c2, 1)
            : ctx.make_binary(BVOp::Ne, lhs, rhs, 1);

        SolveOutcome res = solve_nonzero(ctx, bad_cond, 10000);
        CHECK(res.status != SolveOutcome::Unknown);

        bool brute_force_sat = false;
        std::vector<uint64_t> vals(3, 0);
        std::vector<uint64_t> node_vals;
        for (uint64_t x = 0; x < 16 && !brute_force_sat; ++x) {
            vals[0] = x;
            for (uint64_t y = 0; y < 16 && !brute_force_sat; ++y) {
                vals[1] = y;
                for (uint64_t z = 0; z < 16; ++z) {
                    vals[2] = z;
                    ctx.evaluate_all(vals, node_vals);
                    if (node_vals[bad_cond] != 0) {
                        brute_force_sat = true;
                        break;
                    }
                }
            }
        }

        if (brute_force_sat) {
            ++sat_count;
            CHECK_EQ(res.status, SolveOutcome::Sat);
            CHECK_EQ(ctx.eval_concrete(bad_cond, res.model), 1u);
        } else {
            ++unsat_count;
            CHECK_EQ(res.status, SolveOutcome::Unsat);
        }
    }
    CHECK(sat_count > 20u);
    CHECK(unsat_count > 20u);
}

TEST_CASE("BVSolver #153 - Bitvector Identity & Edge-Case Regression Corpus (with SMT-LIB2 Export Check)") {
    struct CorpusCase {
        const char* name;
        SolveOutcome::Status expected;
        BVNodeId (*build)(BVContext&);
    };

    const CorpusCase corpus[] = {
        {"de_morgan_64", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 64), y = c.make_var("y", 64);
            BVNodeId lhs = c.make_unary(BVOp::Not, c.make_binary(BVOp::And, x, y, 64), 64);
            BVNodeId rhs = c.make_binary(BVOp::Or, c.make_unary(BVOp::Not, x, 64), c.make_unary(BVOp::Not, y, 64), 64);
            return c.make_binary(BVOp::Ne, lhs, rhs, 1);
        }},
        {"add_xor_and_identity_32", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 32), y = c.make_var("y", 32);
            BVNodeId lhs = c.make_binary(BVOp::Add, x, y, 32);
            BVNodeId rhs = c.make_binary(BVOp::Add,
                c.make_binary(BVOp::Xor, x, y, 32),
                c.make_binary(BVOp::Shl, c.make_binary(BVOp::And, x, y, 32), c.make_const(1, 32), 32), 32);
            return c.make_binary(BVOp::Ne, lhs, rhs, 1);
        }},
        {"sdiv_int_min_neg1_8", SolveOutcome::Sat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 8), y = c.make_var("y", 8);
            BVNodeId zero = c.make_const(0, 8);
            BVNodeId q = c.make_binary(BVOp::SDiv, x, y, 8);
            return c.make_binary(BVOp::And,
                c.make_binary(BVOp::Slt, x, zero, 1),
                c.make_binary(BVOp::And,
                    c.make_binary(BVOp::Slt, y, zero, 1),
                    c.make_binary(BVOp::Slt, q, zero, 1), 1), 1);
        }},
        {"udiv_rem_euclid_5", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 5), y = c.make_var("y", 5);
            BVNodeId q = c.make_binary(BVOp::UDiv, x, y, 5);
            BVNodeId r = c.make_binary(BVOp::URem, x, y, 5);
            BVNodeId recon = c.make_binary(BVOp::Add, c.make_binary(BVOp::Mul, q, y, 5), r, 5);
            BVNodeId y_nz = c.make_binary(BVOp::Ne, y, c.make_const(0, 5), 1);
            return c.make_binary(BVOp::And, y_nz, c.make_binary(BVOp::Ne, recon, x, 1), 1);
        }},
        {"sdiv_srem_euclid_5", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 5), y = c.make_var("y", 5);
            BVNodeId q = c.make_binary(BVOp::SDiv, x, y, 5);
            BVNodeId r = c.make_binary(BVOp::SRem, x, y, 5);
            BVNodeId recon = c.make_binary(BVOp::Add, c.make_binary(BVOp::Mul, q, y, 5), r, 5);
            BVNodeId y_nz = c.make_binary(BVOp::Ne, y, c.make_const(0, 5), 1);
            BVNodeId not_ovf = c.make_unary(BVOp::Not,
                c.make_binary(BVOp::And,
                    c.make_binary(BVOp::Eq, x, c.make_const(0x10, 5), 1),
                    c.make_binary(BVOp::Eq, y, c.make_const(0x1F, 5), 1), 1), 1);
            return c.make_binary(BVOp::And, c.make_binary(BVOp::And, y_nz, not_ovf, 1),
                                 c.make_binary(BVOp::Ne, recon, x, 1), 1);
        }},
        {"div_by_zero_is_zero_8", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 8), y = c.make_var("y", 8);
            BVNodeId y_zero = c.make_binary(BVOp::Eq, y, c.make_const(0, 8), 1);
            BVNodeId uq = c.make_binary(BVOp::UDiv, x, y, 8);
            BVNodeId sq = c.make_binary(BVOp::SDiv, x, y, 8);
            BVNodeId ur = c.make_binary(BVOp::URem, x, y, 8);
            BVNodeId sr = c.make_binary(BVOp::SRem, x, y, 8);
            BVNodeId any_nz = c.make_binary(BVOp::Ne,
                c.make_binary(BVOp::Or, c.make_binary(BVOp::Or, uq, sq, 8),
                              c.make_binary(BVOp::Or, ur, sr, 8), 8),
                c.make_const(0, 8), 1);
            return c.make_binary(BVOp::And, y_zero, any_nz, 1);
        }},
        {"non_power_of_two_width7_shift_add", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 7);
            BVNodeId sum = c.make_binary(BVOp::Add, x, x, 7);
            BVNodeId sh1 = c.make_binary(BVOp::Shl, x, c.make_const(1, 7), 7);
            return c.make_binary(BVOp::Ne, sum, sh1, 1);
        }},
        {"sext_zext_non_neg_equiv_16", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 8);
            BVNodeId non_neg = c.make_binary(BVOp::Sge, x, c.make_const(0, 8), 1);
            BVNodeId diff = c.make_binary(BVOp::Ne, c.make_sext(x, 8), c.make_zext(x, 8), 1);
            return c.make_binary(BVOp::And, non_neg, diff, 1);
        }},
        {"clz_ctz_single_bit_sum_8", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 8);
            BVNodeId nz = c.make_binary(BVOp::Ne, x, c.make_const(0, 8), 1);
            BVNodeId pow2 = c.make_binary(BVOp::Eq,
                c.make_binary(BVOp::And, x, c.make_binary(BVOp::Sub, x, c.make_const(1, 8), 8), 8),
                c.make_const(0, 8), 1);
            BVNodeId sum = c.make_binary(BVOp::Add,
                c.make_unary(BVOp::Clz, x, 8), c.make_unary(BVOp::Ctz, x, 8), 8);
            return c.make_binary(BVOp::And, c.make_binary(BVOp::And, nz, pow2, 1),
                                 c.make_binary(BVOp::Ne, sum, c.make_const(7, 8), 1), 1);
        }},
        {"overshift_mask_32", SolveOutcome::Sat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 32), sh = c.make_var("sh", 32);
            BVNodeId ge32 = c.make_binary(BVOp::Uge, sh, c.make_const(32, 32), 1);
            BVNodeId nz = c.make_binary(BVOp::Ne, c.make_binary(BVOp::Shl, x, sh, 32), c.make_const(0, 32), 1);
            return c.make_binary(BVOp::And, ge32, nz, 1);
        }},
        {"smul_ovf_vs_sext_5", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 5), y = c.make_var("y", 5);
            BVNodeId prod10 = c.make_binary(BVOp::Mul, c.make_sext(x, 5), c.make_sext(y, 5), 10);
            BVNodeId sext_lo = c.make_sext(c.make_extract(prod10, 4, 0), 5);
            BVNodeId ref_ovf = c.make_binary(BVOp::Ne, prod10, sext_lo, 1);
            BVNodeId ovf = c.make_binary(BVOp::SMulOvf, x, y, 1);
            return c.make_binary(BVOp::Ne, ovf, ref_ovf, 1);
        }},
        {"umul_ovf_vs_zext_5", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 5), y = c.make_var("y", 5);
            BVNodeId prod10 = c.make_binary(BVOp::Mul, c.make_zext(x, 5), c.make_zext(y, 5), 10);
            BVNodeId hi5 = c.make_extract(prod10, 9, 5);
            BVNodeId ref_ovf = c.make_binary(BVOp::Ne, hi5, c.make_const(0, 5), 1);
            BVNodeId ovf = c.make_binary(BVOp::UMulOvf, x, y, 1);
            return c.make_binary(BVOp::Ne, ovf, ref_ovf, 1);
        }},
        {"popcnt_clear_lsb_8", SolveOutcome::Unsat, [](BVContext& c) {
            BVNodeId x = c.make_var("x", 8);
            BVNodeId nz = c.make_binary(BVOp::Ne, x, c.make_const(0, 8), 1);
            BVNodeId x_clr = c.make_binary(BVOp::And, x, c.make_binary(BVOp::Sub, x, c.make_const(1, 8), 8), 8);
            BVNodeId p1 = c.make_unary(BVOp::Popcnt, x, 8);
            BVNodeId p2 = c.make_binary(BVOp::Add, c.make_unary(BVOp::Popcnt, x_clr, 8), c.make_const(1, 8), 8);
            return c.make_binary(BVOp::And, nz, c.make_binary(BVOp::Ne, p1, p2, 1), 1);
        }}
    };

    CHECK(strict_solver_check_enabled());
    const uint64_t checks_before = unsat_sample_checks_performed();
    uint64_t expected_unsat_count = 0;
#if defined(_MSC_VER)
    // MSVC deprecates getenv (C4996, an error under /WX); _dupenv_s is its
    // owned-copy equivalent.
    std::string dump_dir_owned;
    {
        char* owned = nullptr;
        size_t len = 0;
        if (_dupenv_s(&owned, &len, "BRASS_DUMP_SMT2_CORPUS_DIR") == 0 && owned) {
            dump_dir_owned = owned;
            std::free(owned);
        }
    }
    const char* dump_dir_env = dump_dir_owned.empty() ? nullptr : dump_dir_owned.c_str();
#else
    const char* dump_dir_env = std::getenv("BRASS_DUMP_SMT2_CORPUS_DIR");
#endif

    for (const auto& tc : corpus) {
        BVContext ctx;
        BVNodeId bad = tc.build(ctx);
        SolveOutcome res = solve_nonzero(ctx, bad, 10000);
        CHECK_EQ(res.status, tc.expected);
        if (res.status == SolveOutcome::Sat) {
            CHECK_EQ(ctx.eval_concrete(bad, res.model), 1u);
        } else if (res.status == SolveOutcome::Unsat) {
            ++expected_unsat_count;
        }
        std::string comment = std::string(tc.name) + " | EXPECTED: " +
                              (tc.expected == SolveOutcome::Unsat ? "unsat" : "sat");
        std::string smt2 = ctx.export_smt2(bad, comment);
        CHECK(smt2.find("(set-logic QF_BV)") != std::string::npos);
        CHECK(smt2.find("(check-sat)") != std::string::npos);
        if (dump_dir_env && dump_dir_env[0] != '\0') {
            std::filesystem::path out_path = std::filesystem::path(dump_dir_env) / (std::string(tc.name) + ".smt2");
            std::ofstream ofs(out_path);
            ofs << smt2;
        }
    }

    const uint64_t checks_after = unsat_sample_checks_performed();
    CHECK_EQ(checks_after - checks_before, expected_unsat_count);

    // Verify that disabling strict solver checks suppresses the 32-sample Unsat sweep
    // (the production Release default), and re-enabling restores it.
    set_strict_solver_check(false);
    CHECK(!strict_solver_check_enabled());
    {
        BVContext ctx;
        BVNodeId x = ctx.make_var("x", 16);
        BVNodeId bad = ctx.make_binary(BVOp::Ne, x, x, 1);
        CHECK_EQ(solve_nonzero(ctx, bad, 1000).status, SolveOutcome::Unsat);
        CHECK_EQ(unsat_sample_checks_performed(), checks_after);
    }
    set_strict_solver_check(true);
    CHECK(strict_solver_check_enabled());
}
