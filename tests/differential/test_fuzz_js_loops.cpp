#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_opt.hpp>
#include <random>
#include <vector>
#include <cmath>
#include <string>

using namespace brass;
using namespace brass::test;

// =============================================================================
// Helper: Build a single loop with f64 induction variable and accumulator
// =============================================================================
static void build_f64_induction_loop(
    Module& mod,
    const std::string& fn_name,
    double start_val,
    double step_val,
    double init_acc,
    double coeff_i,
    double coeff_acc,
    bool count_down = false
) {
    Builder b(mod);
    Function* fn = mod.create_function(fn_name, Type::f64(), {Type::f64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* limit = b.add_block_param(entry, Type::f64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* start_c = b.build_fconst_f64(start_val);
    Value* acc_c = b.build_fconst_f64(init_acc);
    Value* step_c = b.build_fconst_f64(step_val);
    Value* coeff_i_c = b.build_fconst_f64(coeff_i);
    Value* coeff_acc_c = b.build_fconst_f64(coeff_acc);

    b.build_br(loop_hdr, {start_c, acc_c});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::f64());
    Value* acc = b.add_block_param(loop_hdr, Type::f64());

    Value* cond = count_down ? b.build_sgt(i, limit) : b.build_slt(i, limit);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* term = b.build_mul(i, coeff_i_c);
    Value* acc_scaled = b.build_mul(acc, coeff_acc_c);
    Value* next_acc = b.build_add(acc_scaled, term);
    Value* next_i = count_down ? b.build_sub(i, step_c) : b.build_add(i, step_c);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::f64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
}

// =============================================================================
// Helper: Build a Bronze-lowered 2D nested loop with carried accumulator
// (Canonical JS lowered pattern, matching 11_matrix_recurrence)
// =============================================================================
static void build_bronze_2d_recurrence_loop(
    Module& mod,
    const std::string& fn_name,
    double c_i,
    double c_j,
    double c_const,
    double step_i_val = 1.0,
    double step_j_val = 1.0
) {
    Builder b(mod);
    Function* fn = mod.create_function(fn_name, Type::f64(), {Type::f64(), Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* m_bound = b.add_block_param(b0, Type::f64());
    Value* n_bound = b.add_block_param(b0, Type::f64());

    BasicBlock* b1_outer_hdr = b.create_block("b1_outer_hdr");
    BasicBlock* b2_outer_body = b.create_block("b2_outer_body");
    BasicBlock* b3_exit = b.create_block("b3_exit");
    BasicBlock* b4_inner_hdr = b.create_block("b4_inner_hdr");
    BasicBlock* b5_inner_body = b.create_block("b5_inner_body");
    BasicBlock* b6_inner_exit = b.create_block("b6_inner_exit");

    Value* zero = b.build_fconst_f64(0.0);
    Value* step_i = b.build_fconst_f64(step_i_val);
    Value* step_j = b.build_fconst_f64(step_j_val);
    Value* c1 = b.build_fconst_f64(c_i);
    Value* c2 = b.build_fconst_f64(c_j);
    Value* c3 = b.build_fconst_f64(c_const);

    b.position_at_end(b0);
    b.build_br(b1_outer_hdr, {zero, zero});

    // Outer Loop Header: params (sum, i)
    fn->append_block(b1_outer_hdr);
    b.position_at_end(b1_outer_hdr);
    Value* sum_out = b.add_block_param(b1_outer_hdr, Type::f64());
    Value* i_val = b.add_block_param(b1_outer_hdr, Type::f64());
    Value* cond_out = b.build_slt(i_val, m_bound);
    b.build_br_if(cond_out, b2_outer_body, {}, b3_exit, {sum_out, i_val});

    // Outer Loop Body: enter inner loop with j = 0.0 and current sum
    fn->append_block(b2_outer_body);
    b.position_at_end(b2_outer_body);
    b.build_br(b4_inner_hdr, {sum_out, zero});

    // Exit Block: returns accumulated sum
    fn->append_block(b3_exit);
    b.position_at_end(b3_exit);
    Value* ret_sum = b.add_block_param(b3_exit, Type::f64());
    [[maybe_unused]] Value* ret_i = b.add_block_param(b3_exit, Type::f64());
    b.build_ret(ret_sum);

    // Inner Loop Header: params (inner_sum, j)
    fn->append_block(b4_inner_hdr);
    b.position_at_end(b4_inner_hdr);
    Value* sum_in = b.add_block_param(b4_inner_hdr, Type::f64());
    Value* j_val = b.add_block_param(b4_inner_hdr, Type::f64());
    Value* cond_in = b.build_slt(j_val, n_bound);
    b.build_br_if(cond_in, b5_inner_body, {}, b6_inner_exit, {sum_in, j_val});

    // Inner Loop Body: evaluates (i * c1 + j * c2 + c3) + inner_sum
    fn->append_block(b5_inner_body);
    b.position_at_end(b5_inner_body);
    Value* t1 = b.build_mul(i_val, c1);
    Value* t2 = b.build_mul(j_val, c2);
    Value* t12 = b.build_add(t1, t2);
    Value* term = b.build_add(t12, c3);
    Value* next_inner_sum = b.build_add(sum_in, term);
    Value* next_j = b.build_add(j_val, step_j);
    b.build_br(b4_inner_hdr, {next_inner_sum, next_j});

    // Inner Loop Exit: advances outer index i and loops back to outer header
    fn->append_block(b6_inner_exit);
    b.position_at_end(b6_inner_exit);
    Value* exit_inner_sum = b.add_block_param(b6_inner_exit, Type::f64());
    [[maybe_unused]] Value* exit_j = b.add_block_param(b6_inner_exit, Type::f64());
    Value* next_i = b.build_add(i_val, step_i);
    b.build_br(b1_outer_hdr, {exit_inner_sum, next_i});

    fn->rebuild_cfg_predecessors();
}

// =============================================================================
// Helper: Build a 3D nested loop with f64 loop counters and carried accumulator
// =============================================================================
static void build_f64_3d_nested_loop(Module& mod, const std::string& fn_name) {
    Builder b(mod);
    Function* fn = mod.create_function(fn_name, Type::f64(), {Type::f64(), Type::f64(), Type::f64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* dim_x = b.add_block_param(entry, Type::f64());
    Value* dim_y = b.add_block_param(entry, Type::f64());
    Value* dim_z = b.add_block_param(entry, Type::f64());

    BasicBlock* loop_x_hdr = b.create_block("loop_x_hdr");
    BasicBlock* loop_x_body = b.create_block("loop_x_body");
    BasicBlock* loop_x_exit = b.create_block("loop_x_exit");

    BasicBlock* loop_y_hdr = b.create_block("loop_y_hdr");
    BasicBlock* loop_y_body = b.create_block("loop_y_body");
    BasicBlock* loop_y_exit = b.create_block("loop_y_exit");

    BasicBlock* loop_z_hdr = b.create_block("loop_z_hdr");
    BasicBlock* loop_z_body = b.create_block("loop_z_body");
    BasicBlock* loop_z_exit = b.create_block("loop_z_exit");

    Value* zero = b.build_fconst_f64(0.0);
    Value* one = b.build_fconst_f64(1.0);
    Value* half = b.build_fconst_f64(0.5);

    b.position_at_end(entry);
    b.build_br(loop_x_hdr, {zero, zero});

    // Level X
    fn->append_block(loop_x_hdr);
    b.position_at_end(loop_x_hdr);
    Value* x = b.add_block_param(loop_x_hdr, Type::f64());
    Value* acc_x = b.add_block_param(loop_x_hdr, Type::f64());
    Value* cond_x = b.build_slt(x, dim_x);
    b.build_br_if(cond_x, loop_x_body, {}, loop_x_exit, {acc_x});

    fn->append_block(loop_x_body);
    b.position_at_end(loop_x_body);
    b.build_br(loop_y_hdr, {zero, acc_x});

    // Level Y
    fn->append_block(loop_y_hdr);
    b.position_at_end(loop_y_hdr);
    Value* y = b.add_block_param(loop_y_hdr, Type::f64());
    Value* acc_y = b.add_block_param(loop_y_hdr, Type::f64());
    Value* cond_y = b.build_slt(y, dim_y);
    b.build_br_if(cond_y, loop_y_body, {}, loop_y_exit, {acc_y});

    fn->append_block(loop_y_body);
    b.position_at_end(loop_y_body);
    b.build_br(loop_z_hdr, {zero, acc_y});

    // Level Z (Innermost)
    fn->append_block(loop_z_hdr);
    b.position_at_end(loop_z_hdr);
    Value* z = b.add_block_param(loop_z_hdr, Type::f64());
    Value* acc_z = b.add_block_param(loop_z_hdr, Type::f64());
    Value* cond_z = b.build_slt(z, dim_z);
    b.build_br_if(cond_z, loop_z_body, {}, loop_z_exit, {acc_z});

    fn->append_block(loop_z_body);
    b.position_at_end(loop_z_body);
    Value* xy = b.build_mul(x, y);
    Value* xyz = b.build_add(xy, z);
    Value* next_acc_z = b.build_add(acc_z, xyz);
    Value* next_z = b.build_add(z, half);
    b.build_br(loop_z_hdr, {next_z, next_acc_z});

    fn->append_block(loop_z_exit);
    b.position_at_end(loop_z_exit);
    Value* z_res = b.add_block_param(loop_z_exit, Type::f64());
    Value* next_y = b.build_add(y, one);
    b.build_br(loop_y_hdr, {next_y, z_res});

    fn->append_block(loop_y_exit);
    b.position_at_end(loop_y_exit);
    Value* y_res = b.add_block_param(loop_y_exit, Type::f64());
    Value* next_x = b.build_add(x, one);
    b.build_br(loop_x_hdr, {next_x, y_res});

    fn->append_block(loop_x_exit);
    b.position_at_end(loop_x_exit);
    Value* final_res = b.add_block_param(loop_x_exit, Type::f64());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
}

// =============================================================================
// Test Case 1: f64 Induction Variables (Integer & Fractional Steps, Trip Counts)
// =============================================================================
TEST_CASE("Differential Fuzzer - f64 Induction Variables (Integer and Fractional Steps)") {
    struct StepConfig {
        double start;
        double step;
        double init_acc;
        double coeff_i;
        double coeff_acc;
        bool count_down;
    };

    std::vector<StepConfig> configs = {
        // Integer steps ascending
        {0.0, 1.0, 0.0, 1.0, 1.0, false},
        {0.0, 2.0, 10.0, 3.0, 1.0, false},
        {5.0, 3.0, 100.0, 2.5, 1.0, false},
        {1.0, 5.0, 0.0, 0.5, 1.0, false},

        // Fractional steps ascending (power-of-2 exact fractions)
        {0.0, 0.5, 0.0, 2.0, 1.0, false},
        {0.0, 0.25, 5.0, 4.0, 1.0, false},
        {0.0, 0.125, 0.0, 8.0, 1.0, false},
        {0.0, 0.0625, 1.0, 16.0, 1.0, false},
        {1.5, 0.75, 10.0, 1.5, 1.0, false},
        {0.25, 1.25, 0.0, 0.75, 1.0, false},

        // Countdown integer & fractional steps
        {100.0, 1.0, 0.0, 1.0, 1.0, true},
        {50.0, 2.0, 10.0, 1.5, 1.0, true},
        {20.0, 0.5, 0.0, 2.0, 1.0, true},
        {10.0, 0.25, 5.0, 4.0, 1.0, true},
        {8.0, 0.125, 0.0, 1.0, 1.0, true},

        // Geometric accumulator with f64 induction
        {0.0, 1.0, 1.0, 1.0, 1.0, false},
        {0.0, 0.5, 2.0, 0.5, 1.0, false}
    };

    std::vector<double> limits = {
        0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0,
        15.0, 16.0, 17.0, 31.0, 32.0, 33.0, 50.0, 64.0, 100.0
    };

    for (size_t cfg_idx = 0; cfg_idx < configs.size(); ++cfg_idx) {
        const auto& cfg = configs[cfg_idx];
        std::string mod_name = "f64_ind_mod_" + std::to_string(cfg_idx);
        Module mod(mod_name);
        std::string fn_name = "f64_loop_fn";

        build_f64_induction_loop(
            mod, fn_name,
            cfg.start, cfg.step, cfg.init_acc,
            cfg.coeff_i, cfg.coeff_acc, cfg.count_down
        );

        for (double limit_val : limits) {
            // For countdown loops, test various lower bounds
            double target_limit = cfg.count_down ? (cfg.start - limit_val) : (cfg.start + limit_val);
            if (cfg.count_down && target_limit < -50.0) target_limit = 0.0;

            assert_diff_3way(mod, fn_name, {RuntimeValue::from_f64(target_limit)});
        }
    }
}

// =============================================================================
// Test Case 2: Nested Loops with f64 Loop Counters
// =============================================================================
TEST_CASE("Differential Fuzzer - Nested Loops with f64 Loop Counters") {
    struct GridSize {
        double m;
        double n;
    };

    std::vector<GridSize> test_grids = {
        {0.0, 0.0},
        {0.0, 5.0},
        {5.0, 0.0},
        {1.0, 1.0},
        {1.0, 8.0},
        {8.0, 1.0},
        {2.0, 3.0},
        {3.0, 2.0},
        {4.0, 4.0},
        {5.0, 7.0},
        {7.0, 5.0},
        {8.0, 8.0},
        {9.0, 13.0},
        {16.0, 16.0},
        {20.0, 25.0}
    };

    // 2D nested loops with various coefficients
    std::vector<std::tuple<double, double, double, double, double>> nested_configs = {
        {1.0, 1.0, 0.0, 1.0, 1.0},
        {3.0, 7.0, 1.0, 1.0, 1.0},      // Canonical 11_matrix_recurrence coefficients
        {0.5, 0.25, 2.5, 0.5, 0.5},    // Fractional step counters
        {2.0, -1.5, 10.0, 1.0, 0.5},   // Negative coefficients
        {1.25, 0.75, -5.0, 0.25, 0.25} // Fractional step and fractional coefficients
    };

    for (size_t i = 0; i < nested_configs.size(); ++i) {
        auto [c_i, c_j, c_const, step_i, step_j] = nested_configs[i];
        std::string mod_name = "nested_f64_mod_" + std::to_string(i);
        Module mod(mod_name);
        std::string fn_name = "nested_f64_fn";

        build_bronze_2d_recurrence_loop(mod, fn_name, c_i, c_j, c_const, step_i, step_j);

        for (const auto& grid : test_grids) {
            assert_diff_3way(mod, fn_name, {
                RuntimeValue::from_f64(grid.m),
                RuntimeValue::from_f64(grid.n)
            });
        }
    }

    // 3D nested loop execution
    {
        Module mod_3d("f64_3d_mod");
        build_f64_3d_nested_loop(mod_3d, "eval_3d_grid");

        std::vector<std::tuple<double, double, double>> test_cubes = {
            {0.0, 0.0, 0.0},
            {1.0, 1.0, 1.0},
            {2.0, 3.0, 4.0},
            {4.0, 2.0, 3.0},
            {5.0, 5.0, 2.0},
            {3.0, 4.0, 5.0}
        };

        for (const auto& [dx, dy, dz] : test_cubes) {
            assert_diff_3way(mod_3d, "eval_3d_grid", {
                RuntimeValue::from_f64(dx),
                RuntimeValue::from_f64(dy),
                RuntimeValue::from_f64(dz)
            });
        }
    }
}

// =============================================================================
// Test Case 3: Accumulator Carried Across Both Loop Levels (Bronze JS Matrix Recurrence)
// =============================================================================
TEST_CASE("Differential Fuzzer - Bronze-lowered Matrix Recurrence 2D Loop Nest (11_matrix_recurrence parity)") {
    // Exact match for 11_matrix_recurrence logic:
    // sum = 0; i = 0; while (i < n) { j = 0; while (j < n) { sum += (i * 3 + j * 7 + 1); j++; } i++; }
    Module mod("bronze_matrix_recurrence_mod");
    build_bronze_2d_recurrence_loop(mod, "recurrence2D", 3.0, 7.0, 1.0, 1.0, 1.0);

    for (double n_val : {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0}) {
        assert_diff_3way(mod, "recurrence2D", {
            RuntimeValue::from_f64(n_val),
            RuntimeValue::from_f64(n_val)
        });
    }
}

// =============================================================================
// Test Case 4: Randomized Fuzzer for f64 Loop Shapes with Branch Diamonds
// =============================================================================
TEST_CASE("Differential Fuzzer - Randomized f64 Loop Shapes with Inner Conditionals") {
    std::mt19937_64 rng(9999);

    for (uint64_t seed = 0; seed < 25; ++seed) {
        std::string mod_name = "fuzz_f64_branch_mod_" + std::to_string(seed);
        Module mod(mod_name);
        Builder b(mod);

        Function* fn = mod.create_function("fuzz_branch_loop", Type::f64(), {Type::f64(), Type::f64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* limit = b.add_block_param(entry, Type::f64());
        Value* threshold = b.add_block_param(entry, Type::f64());

        BasicBlock* loop_hdr = b.create_block("loop_hdr");
        BasicBlock* loop_body = b.create_block("loop_body");
        BasicBlock* bb_then = b.create_block("bb_then");
        BasicBlock* bb_else = b.create_block("bb_else");
        BasicBlock* bb_merge = b.create_block("bb_merge");
        BasicBlock* exit_bb = b.create_block("exit");

        Value* zero = b.build_fconst_f64(0.0);
        Value* step = b.build_fconst_f64(static_cast<double>((seed % 3) + 1) * 0.5);
        Value* c_then = b.build_fconst_f64(2.5);
        Value* c_else = b.build_fconst_f64(0.75);

        b.position_at_end(entry);
        b.build_br(loop_hdr, {zero, zero});

        fn->append_block(loop_hdr);
        b.position_at_end(loop_hdr);
        Value* i = b.add_block_param(loop_hdr, Type::f64());
        Value* acc = b.add_block_param(loop_hdr, Type::f64());
        Value* cond = b.build_slt(i, limit);
        b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

        fn->append_block(loop_body);
        b.position_at_end(loop_body);
        Value* is_above = b.build_sgt(i, threshold);
        b.build_br_if(is_above, bb_then, {}, bb_else, {});

        fn->append_block(bb_then);
        b.position_at_end(bb_then);
        Value* v_then = b.build_mul(i, c_then);
        b.build_br(bb_merge, {v_then});

        fn->append_block(bb_else);
        b.position_at_end(bb_else);
        Value* v_else = b.build_mul(i, c_else);
        b.build_br(bb_merge, {v_else});

        fn->append_block(bb_merge);
        b.position_at_end(bb_merge);
        Value* delta = b.add_block_param(bb_merge, Type::f64());
        Value* next_acc = b.build_add(acc, delta);
        Value* next_i = b.build_add(i, step);
        b.build_br(loop_hdr, {next_i, next_acc});

        fn->append_block(exit_bb);
        b.position_at_end(exit_bb);
        Value* final_acc = b.add_block_param(exit_bb, Type::f64());
        b.build_ret(final_acc);

        fn->rebuild_cfg_predecessors();

        for (int t = 0; t < 5; ++t) {
            double limit_val = static_cast<double>(rng() % 30);
            double thresh_val = static_cast<double>(rng() % 20);

            assert_diff_3way(mod, "fuzz_branch_loop", {
                RuntimeValue::from_f64(limit_val),
                RuntimeValue::from_f64(thresh_val)
            });
        }
    }
}
