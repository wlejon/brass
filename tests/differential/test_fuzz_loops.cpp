#include "test_framework.hpp"
#include "diff_harness.hpp"
#include "fuzz_generator.hpp"
#include <random>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential Fuzzer - Complex Nested Loops with Block Arguments") {
    std::mt19937_64 rng(4242);

    for (uint64_t seed = 0; seed < 50; ++seed) {
        std::string mod_name = "fuzz_loop_mod_" + std::to_string(seed);
        Module mod(mod_name);
        std::string fn_name = "fuzz_loop_fn";

        generate_fuzz_loops(mod, fn_name, seed + 100);

        for (int t = 0; t < 3; ++t) {
            int64_t limit = static_cast<int64_t>(rng() % 50);
            int64_t init_val = static_cast<int64_t>(rng() % 1000);

            assert_diff(mod, fn_name, {
                RuntimeValue::from_i64(limit),
                RuntimeValue::from_i64(init_val)
            });
        }
    }
}

TEST_CASE("Differential Fuzzer - Collatz Arithmetic Fusion Loop") {
    for (int64_t max_n : {1, 5, 10, 27, 50, 100, 500, 1000}) {
        Module mod("collatz_diff_mod");
        Builder b(mod);

        Function* fn = mod.create_function("collatz_sum", Type::i64(), {Type::i64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* mn = b.add_block_param(entry, Type::i64());

        BasicBlock* outer_hdr = b.create_block("outer_hdr");
        BasicBlock* outer_body = b.create_block("outer_body");
        BasicBlock* inner_hdr = b.create_block("inner_hdr");
        BasicBlock* inner_check = b.create_block("inner_check");
        BasicBlock* inner_even = b.create_block("inner_even");
        BasicBlock* inner_odd = b.create_block("inner_odd");
        BasicBlock* outer_next = b.create_block("outer_next");
        BasicBlock* exit_bb = b.create_block("exit");

        Value* one = b.build_iconst_i64(1);
        Value* zero = b.build_iconst_i64(0);
        b.build_br(outer_hdr, {one, zero});

        fn->append_block(outer_hdr);
        b.position_at_end(outer_hdr);
        Value* i = b.add_block_param(outer_hdr, Type::i64());
        Value* total_steps = b.add_block_param(outer_hdr, Type::i64());
        Value* in_range = b.build_sle(i, mn);
        b.build_br_if(in_range, outer_body, {}, exit_bb, {total_steps});

        fn->append_block(outer_body);
        b.position_at_end(outer_body);
        b.build_br(inner_hdr, {i, zero});

        fn->append_block(inner_hdr);
        b.position_at_end(inner_hdr);
        Value* cur_n = b.add_block_param(inner_hdr, Type::i64());
        Value* cur_steps = b.add_block_param(inner_hdr, Type::i64());
        Value* is_done = b.build_sle(cur_n, one);
        b.build_br_if(is_done, outer_next, {}, inner_check, {});

        fn->append_block(inner_check);
        b.position_at_end(inner_check);
        Value* rem = b.build_and(cur_n, one);
        Value* is_even = b.build_eq(rem, zero);
        b.build_br_if(is_even, inner_even, {}, inner_odd, {});

        fn->append_block(inner_even);
        b.position_at_end(inner_even);
        Value* half = b.build_ashr(cur_n, one);
        Value* next_steps_even = b.build_add(cur_steps, one);
        b.build_br(inner_hdr, {half, next_steps_even});

        fn->append_block(inner_odd);
        b.position_at_end(inner_odd);
        Value* three = b.build_iconst_i64(3);
        Value* odd_next = b.build_add(b.build_mul(cur_n, three), one);
        Value* next_steps_odd = b.build_add(cur_steps, one);
        b.build_br(inner_hdr, {odd_next, next_steps_odd});

        fn->append_block(outer_next);
        b.position_at_end(outer_next);
        Value* next_tot = b.build_add(total_steps, cur_steps);
        Value* next_i = b.build_add(i, one);
        b.build_br(outer_hdr, {next_i, next_tot});

        fn->append_block(exit_bb);
        b.position_at_end(exit_bb);
        Value* final_steps = b.add_block_param(exit_bb, Type::i64());
        b.build_ret(final_steps);

        fn->rebuild_cfg_predecessors();

        assert_diff(mod, "collatz_sum", {RuntimeValue::from_i64(max_n)});
    }
}

