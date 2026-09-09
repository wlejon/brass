#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential SROA - Multi-point 3D Distance Vector Math") {
    Module mod("diff_3d_dist");
    Builder b(mod);

    // func @point_dist_sq(%x1: f64, %y1: f64, %z1: f64, %x2: f64, %y2: f64, %z2: f64) -> f64
    Function* fn = mod.create_function("point_dist_sq", Type::f64(),
        {Type::f64(), Type::f64(), Type::f64(), Type::f64(), Type::f64(), Type::f64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");

    Value* x1 = b.add_block_param(entry, Type::f64());
    Value* y1 = b.add_block_param(entry, Type::f64());
    Value* z1 = b.add_block_param(entry, Type::f64());
    Value* x2 = b.add_block_param(entry, Type::f64());
    Value* y2 = b.add_block_param(entry, Type::f64());
    Value* z2 = b.add_block_param(entry, Type::f64());

    Value* sz24 = b.build_iconst_i64(24);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);

    // Temp Point 1
    Value* p1 = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag});
    b.build_store(Type::f64(), p1, 0, x1);
    b.build_store(Type::f64(), p1, 8, y1);
    b.build_store(Type::f64(), p1, 16, z1);

    // Temp Point 2
    Value* p2 = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag});
    b.build_store(Type::f64(), p2, 0, x2);
    b.build_store(Type::f64(), p2, 8, y2);
    b.build_store(Type::f64(), p2, 16, z2);

    // Delta Point
    Value* p_delta = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag});
    Value* dx = b.build_sub(b.build_load(Type::f64(), p2, 0), b.build_load(Type::f64(), p1, 0));
    Value* dy = b.build_sub(b.build_load(Type::f64(), p2, 8), b.build_load(Type::f64(), p1, 8));
    Value* dz = b.build_sub(b.build_load(Type::f64(), p2, 16), b.build_load(Type::f64(), p1, 16));
    b.build_store(Type::f64(), p_delta, 0, dx);
    b.build_store(Type::f64(), p_delta, 8, dy);
    b.build_store(Type::f64(), p_delta, 16, dz);

    // Compute squared distance from p_delta
    Value* rdx = b.build_load(Type::f64(), p_delta, 0);
    Value* rdy = b.build_load(Type::f64(), p_delta, 8);
    Value* rdz = b.build_load(Type::f64(), p_delta, 16);
    Value* d2x = b.build_mul(rdx, rdx);
    Value* d2y = b.build_mul(rdy, rdy);
    Value* d2z = b.build_mul(rdz, rdz);
    Value* dist_sq = b.build_add(b.build_add(d2x, d2y), d2z);
    b.build_ret(dist_sq);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Optimize with SROA
    bool changed = sroa_module(mod);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    // Verify native JIT vs Reference Interpreter
    assert_diff(mod, "point_dist_sq", {
        RuntimeValue::from_f64(1.0), RuntimeValue::from_f64(2.0), RuntimeValue::from_f64(3.0),
        RuntimeValue::from_f64(4.0), RuntimeValue::from_f64(6.0), RuntimeValue::from_f64(15.0)
    });

    assert_diff(mod, "point_dist_sq", {
        RuntimeValue::from_f64(-5.5), RuntimeValue::from_f64(10.25), RuntimeValue::from_f64(0.0),
        RuntimeValue::from_f64(2.5), RuntimeValue::from_f64(-1.75), RuntimeValue::from_f64(7.0)
    });
}

TEST_CASE("Differential SROA - Multi-field Loop Accumulator (sum, sum_squares, count)") {
    Module mod("diff_multi_acc");
    Builder b(mod);

    // func @stat_acc(%n: i64) -> i64
    Function* fn = mod.create_function("stat_acc", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_header = b.append_block("loop_header");
    BasicBlock* loop_body = b.append_block("loop_body");
    BasicBlock* loop_exit = b.append_block("loop_exit");

    Value* n = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    // entry: allocate accumulator with sum (offset 0) and count (offset 8)
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* acc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});

    Value* zero = b.build_iconst_i64(0);
    b.build_store(Type::i64(), acc, 0, zero);
    b.build_store(Type::i64(), acc, 8, zero);

    Value* i0 = b.build_iconst_i64(0);
    b.build_br(loop_header, {i0});

    // loop_header(i): cond = i < n
    Value* i_param = b.add_block_param(loop_header, Type::i64());
    b.position_at_end(loop_header);
    Value* cond = b.build_slt(i_param, n);
    b.build_br_if(cond, loop_body, loop_exit);

    // loop_body: sum += i, count += 1
    b.position_at_end(loop_body);
    Value* cur_s = b.build_load(Type::i64(), acc, 0);
    Value* cur_c = b.build_load(Type::i64(), acc, 8);

    Value* new_s = b.build_add(cur_s, i_param);
    Value* one = b.build_iconst_i64(1);
    Value* new_c = b.build_add(cur_c, one);

    b.build_store(Type::i64(), acc, 0, new_s);
    b.build_store(Type::i64(), acc, 8, new_c);

    Value* next_i = b.build_add(i_param, one);
    b.build_br(loop_header, {next_i});

    // loop_exit: return sum + count
    b.position_at_end(loop_exit);
    Value* fs = b.build_load(Type::i64(), acc, 0);
    Value* fc = b.build_load(Type::i64(), acc, 8);
    Value* total = b.build_add(fs, fc);
    b.build_ret(total);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    bool changed = sroa_module(mod);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "stat_acc", {RuntimeValue::from_i64(5)});
    assert_diff(mod, "stat_acc", {RuntimeValue::from_i64(15)});
    assert_diff(mod, "stat_acc", {RuntimeValue::from_i64(30)});
    assert_diff(mod, "stat_acc", {RuntimeValue::from_i64(15)});
    assert_diff(mod, "stat_acc", {RuntimeValue::from_i64(30)});
}

TEST_CASE("Differential SROA - Diamond Conditional Updates with Nested Control Flow") {
    Module mod("diff_diamond_control");
    Builder b(mod);

    // func @diamond_worker(%mode: i32, %base_val: i64) -> i64
    Function* fn = mod.create_function("diamond_worker", Type::i64(), {Type::i32(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* b_then = b.append_block("b_then");
    BasicBlock* b_else = b.append_block("b_else");
    BasicBlock* b_merge = b.append_block("b_merge");

    Value* mode = b.add_block_param(entry, Type::i32());
    Value* base_val = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* st = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});

    Value* init10 = b.build_iconst_i64(10);
    b.build_store(Type::i64(), st, 0, init10);
    b.build_store(Type::i64(), st, 8, base_val);

    Value* zero32 = b.build_iconst_i32(0);
    Value* cond = b.build_sgt(mode, zero32);
    b.build_br_if(cond, b_then, b_else);

    // then: field 0 = base_val * 2
    b.position_at_end(b_then);
    Value* c2 = b.build_iconst_i64(2);
    Value* prod = b.build_mul(base_val, c2);
    b.build_store(Type::i64(), st, 0, prod);
    b.build_br(b_merge);

    // else: field 8 = base_val + 50
    b.position_at_end(b_else);
    Value* c50 = b.build_iconst_i64(50);
    Value* added = b.build_add(base_val, c50);
    b.build_store(Type::i64(), st, 8, added);
    b.build_br(b_merge);

    // merge: return field 0 + field 8
    b.position_at_end(b_merge);
    Value* r0 = b.build_load(Type::i64(), st, 0);
    Value* r8 = b.build_load(Type::i64(), st, 8);
    Value* total = b.build_add(r0, r8);
    b.build_ret(total);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    bool changed = sroa_module(mod);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    // Mode > 0: then path (prod = base_val*2, r8 = base_val) -> total = 3*base_val
    assert_diff(mod, "diamond_worker", {RuntimeValue::from_i32(1), RuntimeValue::from_i64(100)});
    assert_diff(mod, "diamond_worker", {RuntimeValue::from_i32(5), RuntimeValue::from_i64(42)});

    // Mode <= 0: else path (r0 = 10, r8 = base_val + 50) -> total = base_val + 60
    assert_diff(mod, "diamond_worker", {RuntimeValue::from_i32(0), RuntimeValue::from_i64(100)});
    assert_diff(mod, "diamond_worker", {RuntimeValue::from_i32(-3), RuntimeValue::from_i64(42)});
}
