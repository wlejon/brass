#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/brass.hpp>
#include <random>
#include <cstring>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential Fuzzer - NaN-Boxing and Bitcast Conversions") {
    std::mt19937_64 rng(1337);

    for (uint64_t seed = 0; seed < 40; ++seed) {
        Module mod("diff_nanbox_mod_" + std::to_string(seed));
        Builder b(mod);

        Function* fn = mod.create_function("diff_nanbox_fn", Type::f64(), {Type::ptr(), Type::i64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* data_ptr = b.add_block_param(entry, Type::ptr());
        Value* len = b.add_block_param(entry, Type::i64());

        BasicBlock* loop_hdr = b.create_block("loop_hdr");
        BasicBlock* loop_body = b.create_block("loop_body");
        BasicBlock* bb_double = b.create_block("bb_double");
        BasicBlock* bb_tag_check = b.create_block("bb_tag_check");
        BasicBlock* bb_int = b.create_block("bb_int");
        BasicBlock* bb_fallback = b.create_block("bb_fallback");
        BasicBlock* exit_bb = b.create_block("exit");

        Value* zero_i = b.build_iconst_i64(0);
        Value* zero_f = b.build_fconst_f64(0.0);
        Value* one_i = b.build_iconst_i64(1);
        Value* one_f = b.build_fconst_f64(1.0);
        Value* tag_thresh = b.build_iconst_i64(static_cast<int64_t>(0xFFF8000000000000ULL));
        Value* shift_32 = b.build_iconst_i64(32);
        Value* int_tag = b.build_iconst_i64(0xFFF90000LL);

        b.build_br(loop_hdr, {zero_i, zero_f});

        fn->append_block(loop_hdr);
        b.position_at_end(loop_hdr);
        Value* i = b.add_block_param(loop_hdr, Type::i64());
        Value* acc = b.add_block_param(loop_hdr, Type::f64());
        Value* cond = b.build_slt(i, len);
        b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

        fn->append_block(loop_body);
        b.position_at_end(loop_body);
        Value* v = b.build_load_indexed(Type::i64(), data_ptr, i, 8, 0);
        Value* is_double = b.build_ult(v, tag_thresh);
        b.build_br_if(is_double, bb_double, {}, bb_tag_check, {});

        fn->append_block(bb_double);
        b.position_at_end(bb_double);
        Value* d = b.build_bitcast_f64_i64(v);
        Value* acc_d = b.build_add(acc, d);
        Value* next_i_d = b.build_add(i, one_i);
        b.build_br(loop_hdr, {next_i_d, acc_d});

        fn->append_block(bb_tag_check);
        b.position_at_end(bb_tag_check);
        Value* tag = b.build_lshr(v, shift_32);
        Value* is_int = b.build_eq(tag, int_tag);
        b.build_br_if(is_int, bb_int, {}, bb_fallback, {});

        fn->append_block(bb_int);
        b.position_at_end(bb_int);
        Value* iv = b.build_trunc_i32(v);
        Value* div = b.build_sitofp_f64_i32(iv);
        Value* acc_int = b.build_add(acc, div);
        Value* next_i_int = b.build_add(i, one_i);
        b.build_br(loop_hdr, {next_i_int, acc_int});

        fn->append_block(bb_fallback);
        b.position_at_end(bb_fallback);
        Value* acc_fb = b.build_add(acc, one_f);
        Value* next_i_fb = b.build_add(i, one_i);
        b.build_br(loop_hdr, {next_i_fb, acc_fb});

        fn->append_block(exit_bb);
        b.position_at_end(exit_bb);
        Value* final_res = b.add_block_param(exit_bb, Type::f64());
        b.build_ret(final_res);

        fn->rebuild_cfg_predecessors();

        size_t N = 100;
        std::vector<uint64_t> data(N);
        for (size_t k = 0; k < N; ++k) {
            uint64_t kind = rng() % 3;
            if (kind == 0) {
                double val_d = static_cast<double>(rng() % 100) + 0.25;
                std::memcpy(&data[k], &val_d, sizeof(double));
            } else if (kind == 1) {
                int32_t val_iv = static_cast<int32_t>(rng() % 100) - 50;
                data[k] = (0xFFF90000ULL << 32) | static_cast<uint32_t>(val_iv);
            } else {
                data[k] = 0xFFFA000000000000ULL | (rng() % 1000);
            }
        }

        assert_diff(mod, "diff_nanbox_fn", {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(data.data())),
            RuntimeValue::from_i64(static_cast<int64_t>(N))
        });
    }
}

TEST_CASE("Differential Fuzzer - Polymorphic Shape Guards with Out-of-Line Fallbacks") {
    std::mt19937_64 rng(777);

    struct ShapeObj {
        uint64_t shape;
        int64_t s0;
        int64_t s1;
        int64_t s2;
    };

    for (uint64_t seed = 0; seed < 30; ++seed) {
        Module mod("diff_shape_mod_" + std::to_string(seed));
        Builder b(mod);

        Function* fn = mod.create_function("diff_shape_fn", Type::i64(), {Type::ptr(), Type::i64()});
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* objs_ptr = b.add_block_param(entry, Type::ptr());
        Value* len = b.add_block_param(entry, Type::i64());

        BasicBlock* loop_body = b.create_block("loop_body");
        BasicBlock* bb_hit_a = b.create_block("bb_hit_a");
        BasicBlock* bb_slow_path = b.create_block("bb_slow_path");
        BasicBlock* bb_hit_b = b.create_block("bb_hit_b");
        BasicBlock* bb_hit_c = b.create_block("bb_hit_c");
        BasicBlock* exit_bb = b.create_block("exit");

        Value* zero = b.build_iconst_i64(0);
        Value* one = b.build_iconst_i64(1);

        Value* init_cond = b.build_slt(zero, len);
        b.build_br_if(init_cond, loop_body, {zero, zero}, exit_bb, {zero});

        fn->append_block(loop_body);
        b.position_at_end(loop_body);
        Value* i = b.add_block_param(loop_body, Type::i64());
        Value* sum = b.add_block_param(loop_body, Type::i64());

        Value* obj = b.build_load_indexed(Type::ptr(), objs_ptr, i, 8, 0);
        Value* shape = b.build_load(Type::i64(), obj, 0);
        Value* is_a = b.build_eq(shape, b.build_iconst_i64(0xAA01));
        b.build_br_if(is_a, bb_hit_a, {}, bb_slow_path, {});

        fn->append_block(bb_hit_a);
        b.position_at_end(bb_hit_a);
        Value* val_a = b.build_load(Type::i64(), obj, 8);
        Value* next_sum_a = b.build_add(sum, val_a);
        Value* next_i_a = b.build_add(i, one);
        Value* cond_a = b.build_slt(next_i_a, len);
        b.build_br_if(cond_a, loop_body, {next_i_a, next_sum_a}, exit_bb, {next_sum_a});

        fn->append_block(bb_slow_path);
        b.position_at_end(bb_slow_path);
        Value* is_b = b.build_eq(shape, b.build_iconst_i64(0xBB02));
        b.build_br_if(is_b, bb_hit_b, {}, bb_hit_c, {});

        fn->append_block(bb_hit_b);
        b.position_at_end(bb_hit_b);
        Value* val_b = b.build_load(Type::i64(), obj, 16);
        Value* next_sum_b = b.build_add(sum, val_b);
        Value* next_i_b = b.build_add(i, one);
        Value* cond_b = b.build_slt(next_i_b, len);
        b.build_br_if(cond_b, loop_body, {next_i_b, next_sum_b}, exit_bb, {next_sum_b});

        fn->append_block(bb_hit_c);
        b.position_at_end(bb_hit_c);
        Value* val_c = b.build_load(Type::i64(), obj, 24);
        Value* val_c2 = b.build_add(val_c, val_c);
        Value* next_sum_c = b.build_add(sum, val_c2);
        Value* next_i_c = b.build_add(i, one);
        Value* cond_c = b.build_slt(next_i_c, len);
        b.build_br_if(cond_c, loop_body, {next_i_c, next_sum_c}, exit_bb, {next_sum_c});

        fn->append_block(exit_bb);
        b.position_at_end(exit_bb);
        Value* final_res = b.add_block_param(exit_bb, Type::i64());
        b.build_ret(final_res);

        fn->rebuild_cfg_predecessors();

        size_t N = 60;
        std::vector<ShapeObj> storage(N);
        std::vector<uintptr_t> ptrs(N);
        for (size_t k = 0; k < N; ++k) {
            uint64_t roll = rng() % 10;
            if (roll < 7) {
                storage[k] = {0xAA01, static_cast<int64_t>(k + 1), 0, 0};
            } else if (roll < 9) {
                storage[k] = {0xBB02, 0, static_cast<int64_t>((k + 1) * 3), 0};
            } else {
                storage[k] = {0xCC03, 0, 0, static_cast<int64_t>(k + 1)};
            }
            ptrs[k] = reinterpret_cast<uintptr_t>(&storage[k]);
        }

        assert_diff(mod, "diff_shape_fn", {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(ptrs.data())),
            RuntimeValue::from_i64(static_cast<int64_t>(N))
        });
    }
}
