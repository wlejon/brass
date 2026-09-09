#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential GVN - Integer Matrix Transform and Memory Pipeline") {
    Module mod("diff_int_transform");
    Builder b(mod);

    // func @int_transform(%buf: ptr, %scale: i64, %offset: i64) -> i64
    Function* fn = mod.create_function("int_transform", Type::i64(),
        {Type::ptr(), Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* buf = b.add_block_param(entry, Type::ptr());
    Value* scale = b.add_block_param(entry, Type::i64());
    Value* offset = b.add_block_param(entry, Type::i64());

    // Compute base values
    Value* v0 = b.build_mul(scale, b.build_iconst_i64(2));
    Value* v1 = b.build_add(v0, offset);

    // Stores to buffer
    b.build_store(Type::i64(), buf, 0, v0);
    b.build_store(Type::i64(), buf, 8, v1);
    b.build_store(Type::i64(), buf, 16, b.build_iconst_i64(100));

    // Overwritten dead store to field 24
    b.build_store(Type::i64(), buf, 24, b.build_iconst_i64(999));
    b.build_store(Type::i64(), buf, 24, b.build_iconst_i64(50));

    // Load back coordinates
    Value* lv0 = b.build_load(Type::i64(), buf, 0);
    Value* lv1 = b.build_load(Type::i64(), buf, 8);

    // Commutative redundant expressions: (lv0 + lv1) and (lv1 + lv0)
    Value* sum1 = b.build_add(lv0, lv1);
    Value* sum2 = b.build_add(lv1, lv0);
    Value* prod = b.build_mul(sum1, sum2);

    // Redundant load of lv0
    Value* lv0_dup = b.build_load(Type::i64(), buf, 0);
    Value* extra = b.build_add(prod, lv0_dup);

    b.build_ret(extra);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Run GVN optimization
    bool changed = gvn_module(mod);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    int64_t mem[4] = {0, 0, 0, 0};
    uintptr_t mem_ptr = reinterpret_cast<uintptr_t>(mem);

    assert_diff(mod, "int_transform", {
        RuntimeValue::from_ptr(mem_ptr), RuntimeValue::from_i64(10), RuntimeValue::from_i64(5)
    });

    assert_diff(mod, "int_transform", {
        RuntimeValue::from_ptr(mem_ptr), RuntimeValue::from_i64(-4), RuntimeValue::from_i64(20)
    });

    assert_diff(mod, "int_transform", {
        RuntimeValue::from_ptr(mem_ptr), RuntimeValue::from_i64(100), RuntimeValue::from_i64(0)
    });
}

TEST_CASE("Differential GVN - Multi-branch Decision Matrix Kernel") {
    Module mod("diff_branch_gvn");
    Builder b(mod);

    // func @branch_calc(%state: ptr, %a: i64, %b: i64, %flag: i32) -> i64
    Function* fn = mod.create_function("branch_calc", Type::i64(),
        {Type::ptr(), Type::i64(), Type::i64(), Type::i32()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* b_true = b.append_block("b_true");
    BasicBlock* b_false = b.append_block("b_false");
    BasicBlock* merge = b.append_block("merge");

    b.position_at_end(entry);
    Value* state = b.add_block_param(entry, Type::ptr());
    Value* a = b.add_block_param(entry, Type::i64());
    Value* b_val = b.add_block_param(entry, Type::i64());
    Value* flag = b.add_block_param(entry, Type::i32());

    // Compute common product in entry
    Value* prod = b.build_mul(a, b_val);

    b.build_store(Type::i64(), state, 0, prod);
    b.build_br_if(flag, b_true, b_false);

    // True branch: store overwritten with a + prod
    b.position_at_end(b_true);
    Value* t_val = b.build_add(a, prod);
    b.build_store(Type::i64(), state, 0, t_val);
    b.build_br(merge);

    // False branch: store overwritten with b + prod
    b.position_at_end(b_false);
    Value* f_val = b.build_add(b_val, prod);
    b.build_store(Type::i64(), state, 0, f_val);
    b.build_br(merge);

    // Merge block: reads state, uses commutative product again
    b.position_at_end(merge);
    Value* merged_prod = b.build_mul(b_val, a); // Commutative duplicate of prod in entry!
    Value* res_state = b.build_load(Type::i64(), state, 0);
    Value* final_res = b.build_add(merged_prod, res_state);
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    bool changed = gvn_module(mod);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    int64_t state_buf[2] = {0, 0};
    uintptr_t s_ptr = reinterpret_cast<uintptr_t>(state_buf);

    assert_diff(mod, "branch_calc", {
        RuntimeValue::from_ptr(s_ptr), RuntimeValue::from_i64(10),
        RuntimeValue::from_i64(20), RuntimeValue::from_i32(1)
    });

    assert_diff(mod, "branch_calc", {
        RuntimeValue::from_ptr(s_ptr), RuntimeValue::from_i64(10),
        RuntimeValue::from_i64(20), RuntimeValue::from_i32(0)
    });

    assert_diff(mod, "branch_calc", {
        RuntimeValue::from_ptr(s_ptr), RuntimeValue::from_i64(-5),
        RuntimeValue::from_i64(15), RuntimeValue::from_i32(1)
    });
}

TEST_CASE("Differential GVN - 3D Vector Math Pipeline with SROA and GVN") {
    Module mod("diff_3d_sroa_gvn");
    Builder b(mod);

    // func @vec_dist(%x1: f64, %y1: f64, %z1: f64, %x2: f64, %y2: f64, %z2: f64) -> f64
    Function* fn = mod.create_function("vec_dist", Type::f64(),
        {Type::f64(), Type::f64(), Type::f64(), Type::f64(), Type::f64(), Type::f64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* x1 = b.add_block_param(entry, Type::f64());
    Value* y1 = b.add_block_param(entry, Type::f64());
    Value* z1 = b.add_block_param(entry, Type::f64());
    Value* x2 = b.add_block_param(entry, Type::f64());
    Value* y2 = b.add_block_param(entry, Type::f64());
    Value* z2 = b.add_block_param(entry, Type::f64());

    Value* sz24 = b.build_iconst_i64(24);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);

    Value* v1 = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag});
    Value* v2 = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag});

    b.build_store(Type::f64(), v1, 0, x1);
    b.build_store(Type::f64(), v1, 8, y1);
    b.build_store(Type::f64(), v1, 16, z1);

    b.build_store(Type::f64(), v2, 0, x2);
    b.build_store(Type::f64(), v2, 8, y2);
    b.build_store(Type::f64(), v2, 16, z2);

    // Delta coordinates
    Value* dx = b.build_sub(b.build_load(Type::f64(), v2, 0), b.build_load(Type::f64(), v1, 0));
    Value* dy = b.build_sub(b.build_load(Type::f64(), v2, 8), b.build_load(Type::f64(), v1, 8));
    Value* dz = b.build_sub(b.build_load(Type::f64(), v2, 16), b.build_load(Type::f64(), v1, 16));

    // dx * dx, dy * dy, dz * dz
    Value* dx2 = b.build_mul(dx, dx);
    Value* dy2 = b.build_mul(dy, dy);
    Value* dz2 = b.build_mul(dz, dz);

    // Redundant sum operations
    Value* dxy1 = b.build_add(dx2, dy2);
    Value* dxy2 = b.build_add(dy2, dx2); // Commutative duplicate
    Value* dxy = b.build_mul(dxy1, b.build_fconst_f64(0.5));
    Value* dxy_other = b.build_mul(dxy2, b.build_fconst_f64(0.5));
    Value* avg_dxy = b.build_add(dxy, dxy_other);

    Value* total = b.build_add(avg_dxy, dz2);
    b.build_ret(total);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Full optimization pipeline: SROA dissolves allocations, GVN eliminates redundant expressions
    LoopOptOptions opt_opts;
    opt_opts.enable_sroa = true;
    opt_opts.enable_gvn = true;
    bool changed = optimize_function(*fn, opt_opts);
    CHECK(changed);
    REQUIRE(verify_function(*fn));

    assert_diff(mod, "vec_dist", {
        RuntimeValue::from_f64(1.0), RuntimeValue::from_f64(2.0), RuntimeValue::from_f64(3.0),
        RuntimeValue::from_f64(4.0), RuntimeValue::from_f64(6.0), RuntimeValue::from_f64(15.0)
    });

    assert_diff(mod, "vec_dist", {
        RuntimeValue::from_f64(-5.0), RuntimeValue::from_f64(10.0), RuntimeValue::from_f64(0.0),
        RuntimeValue::from_f64(5.0), RuntimeValue::from_f64(-10.0), RuntimeValue::from_f64(2.0)
    });

    assert_diff(mod, "vec_dist", {
        RuntimeValue::from_f64(0.0), RuntimeValue::from_f64(0.0), RuntimeValue::from_f64(0.0),
        RuntimeValue::from_f64(0.0), RuntimeValue::from_f64(0.0), RuntimeValue::from_f64(0.0)
    });
}
