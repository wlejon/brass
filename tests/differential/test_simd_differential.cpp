#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>
#include <iostream>
#include <cmath>

using namespace brass;
using namespace brass::test;

inline void build_binop_test(
    Module& mod,
    std::string_view name,
    Type vec_type,
    Opcode op
) {
    Function* fn = mod.create_function(name, vec_type, {vec_type, vec_type});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* p0 = b.add_block_param(entry, vec_type);
    Value* p1 = b.add_block_param(entry, vec_type);
    Value* res = nullptr;
    switch (op) {
        case Opcode::vadd: res = b.build_vadd(p0, p1); break;
        case Opcode::vsub: res = b.build_vsub(p0, p1); break;
        case Opcode::vmul: res = b.build_vmul(p0, p1); break;
        case Opcode::vdiv: res = b.build_vdiv(p0, p1); break;
        case Opcode::vmin: res = b.build_vmin(p0, p1); break;
        case Opcode::vmax: res = b.build_vmax(p0, p1); break;
        case Opcode::vand: res = b.build_vand(p0, p1); break;
        case Opcode::vor:  res = b.build_vor(p0, p1); break;
        case Opcode::vxor: res = b.build_vxor(p0, p1); break;
        default: break;
    }
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
}

inline void build_unop_test(
    Module& mod,
    std::string_view name,
    Type vec_type,
    Opcode op
) {
    Function* fn = mod.create_function(name, vec_type, {vec_type});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* p0 = b.add_block_param(entry, vec_type);
    Value* res = nullptr;
    switch (op) {
        case Opcode::vneg:  res = b.build_vneg(p0); break;
        case Opcode::vsqrt: res = b.build_vsqrt(p0); break;
        case Opcode::vnot:  res = b.build_vnot(p0); break;
        default: break;
    }
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
}

TEST_CASE("SIMD Differential - F32x4 Arithmetic") {
    Module mod("simd_diff_f32x4");
    build_binop_test(mod, "f32x4_add", Type::f32x4(), Opcode::vadd);
    build_binop_test(mod, "f32x4_sub", Type::f32x4(), Opcode::vsub);
    build_binop_test(mod, "f32x4_mul", Type::f32x4(), Opcode::vmul);
    build_binop_test(mod, "f32x4_div", Type::f32x4(), Opcode::vdiv);
    build_binop_test(mod, "f32x4_min", Type::f32x4(), Opcode::vmin);
    build_binop_test(mod, "f32x4_max", Type::f32x4(), Opcode::vmax);
    build_unop_test(mod, "f32x4_neg", Type::f32x4(), Opcode::vneg);
    build_unop_test(mod, "f32x4_sqrt", Type::f32x4(), Opcode::vsqrt);

    RuntimeValue a = RuntimeValue::from_f32x4(1.5f, -2.0f, 3.25f, 4.0f);
    RuntimeValue b = RuntimeValue::from_f32x4(0.5f, 4.0f, -1.0f, 2.0f);
    RuntimeValue pos = RuntimeValue::from_f32x4(4.0f, 9.0f, 16.0f, 25.0f);

    assert_diff(mod, "f32x4_add", {a, b});
    assert_diff(mod, "f32x4_sub", {a, b});
    assert_diff(mod, "f32x4_mul", {a, b});
    assert_diff(mod, "f32x4_div", {a, b});
    assert_diff(mod, "f32x4_min", {a, b});
    assert_diff(mod, "f32x4_max", {a, b});
    assert_diff(mod, "f32x4_neg", {a});
    assert_diff(mod, "f32x4_sqrt", {pos});
}

TEST_CASE("SIMD Differential - F64x2 Arithmetic") {
    Module mod("simd_diff_f64x2");
    build_binop_test(mod, "f64x2_add", Type::f64x2(), Opcode::vadd);
    build_binop_test(mod, "f64x2_sub", Type::f64x2(), Opcode::vsub);
    build_binop_test(mod, "f64x2_mul", Type::f64x2(), Opcode::vmul);
    build_binop_test(mod, "f64x2_div", Type::f64x2(), Opcode::vdiv);
    build_binop_test(mod, "f64x2_min", Type::f64x2(), Opcode::vmin);
    build_binop_test(mod, "f64x2_max", Type::f64x2(), Opcode::vmax);
    build_unop_test(mod, "f64x2_neg", Type::f64x2(), Opcode::vneg);
    build_unop_test(mod, "f64x2_sqrt", Type::f64x2(), Opcode::vsqrt);

    RuntimeValue a = RuntimeValue::from_f64x2(12.5, -45.75);
    RuntimeValue b = RuntimeValue::from_f64x2(2.5, 5.0);
    RuntimeValue pos = RuntimeValue::from_f64x2(144.0, 625.0);

    assert_diff(mod, "f64x2_add", {a, b});
    assert_diff(mod, "f64x2_sub", {a, b});
    assert_diff(mod, "f64x2_mul", {a, b});
    assert_diff(mod, "f64x2_div", {a, b});
    assert_diff(mod, "f64x2_min", {a, b});
    assert_diff(mod, "f64x2_max", {a, b});
    assert_diff(mod, "f64x2_neg", {a});
    assert_diff(mod, "f64x2_sqrt", {pos});
}

TEST_CASE("SIMD Differential - I32x4 Arithmetic") {
    Module mod("simd_diff_i32x4");
    build_binop_test(mod, "i32x4_add", Type::i32x4(), Opcode::vadd);
    build_binop_test(mod, "i32x4_sub", Type::i32x4(), Opcode::vsub);
    build_binop_test(mod, "i32x4_mul", Type::i32x4(), Opcode::vmul);
    build_binop_test(mod, "i32x4_min", Type::i32x4(), Opcode::vmin);
    build_binop_test(mod, "i32x4_max", Type::i32x4(), Opcode::vmax);
    build_unop_test(mod, "i32x4_neg", Type::i32x4(), Opcode::vneg);

    RuntimeValue a = RuntimeValue::from_i32x4(10, -25, 100, -500);
    RuntimeValue b = RuntimeValue::from_i32x4(3, 50, -20, 250);

    assert_diff(mod, "i32x4_add", {a, b});
    assert_diff(mod, "i32x4_sub", {a, b});
    assert_diff(mod, "i32x4_mul", {a, b});
    assert_diff(mod, "i32x4_min", {a, b});
    assert_diff(mod, "i32x4_max", {a, b});
    assert_diff(mod, "i32x4_neg", {a});
}

TEST_CASE("SIMD Differential - I64x2 Arithmetic") {
    Module mod("simd_diff_i64x2");
    build_binop_test(mod, "i64x2_add", Type::i64x2(), Opcode::vadd);
    build_binop_test(mod, "i64x2_sub", Type::i64x2(), Opcode::vsub);
    build_unop_test(mod, "i64x2_neg", Type::i64x2(), Opcode::vneg);

    RuntimeValue a = RuntimeValue::from_i64x2(10000000000LL, -5555555555LL);
    RuntimeValue b = RuntimeValue::from_i64x2(20000000000LL, 1111111111LL);

    assert_diff(mod, "i64x2_add", {a, b});
    assert_diff(mod, "i64x2_sub", {a, b});
    assert_diff(mod, "i64x2_neg", {a});
}

TEST_CASE("SIMD Differential - Bitwise Operations") {
    Module mod("simd_diff_bitwise");
    build_binop_test(mod, "vec_and", Type::i32x4(), Opcode::vand);
    build_binop_test(mod, "vec_or",  Type::i32x4(), Opcode::vor);
    build_binop_test(mod, "vec_xor", Type::i32x4(), Opcode::vxor);
    build_unop_test(mod,  "vec_not", Type::i32x4(), Opcode::vnot);

    RuntimeValue a = RuntimeValue::from_i32x4(0x00FF00FF, 0x12345678, static_cast<int32_t>(0xAAAAAAAA), 0);
    RuntimeValue b = RuntimeValue::from_i32x4(0x0F0F0F0F, static_cast<int32_t>(0x9ABCDEF0), 0x55555555, -1);

    assert_diff(mod, "vec_and", {a, b});
    assert_diff(mod, "vec_or",  {a, b});
    assert_diff(mod, "vec_xor", {a, b});
    assert_diff(mod, "vec_not", {a});
}

TEST_CASE("SIMD Differential - Lane & Swizzle") {
    Module mod("simd_diff_swizzle");

    // 1. vzero
    {
        Function* fn = mod.create_function("test_vzero", Type::f32x4(), {});
        Builder b(mod);
        b.set_function(fn);
        b.append_block("entry");
        Value* z = b.build_vzero(Type::f32x4());
        b.build_ret(z);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        assert_diff(mod, "test_vzero", {});
    }

    // 2. vbroadcast (i32 -> i32x4)
    {
        Function* fn = mod.create_function("test_bcast_i32", Type::i32x4(), {Type::i32()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* p0 = b.add_block_param(bb, Type::i32());
        Value* bc = b.build_vbroadcast(Type::i32x4(), p0);
        b.build_ret(bc);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        assert_diff(mod, "test_bcast_i32", {RuntimeValue::from_i32(42)});
        assert_diff(mod, "test_bcast_i32", {RuntimeValue::from_i32(-99)});
    }

    // 3. vextract_lane
    {
        Function* fn = mod.create_function("test_extract_lane2", Type::i32(), {Type::i32x4()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* p0 = b.add_block_param(bb, Type::i32x4());
        Value* ex = b.build_vextract_lane(p0, 2);
        b.build_ret(ex);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        assert_diff(mod, "test_extract_lane2", {RuntimeValue::from_i32x4(10, 20, 30, 40)});
    }

    // 4. vinsert_lane
    {
        Function* fn = mod.create_function("test_insert_lane1", Type::i32x4(), {Type::i32x4(), Type::i32()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* p0 = b.add_block_param(bb, Type::i32x4());
        Value* p1 = b.add_block_param(bb, Type::i32());
        Value* ins = b.build_vinsert_lane(p0, p1, 1);
        b.build_ret(ins);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        assert_diff(mod, "test_insert_lane1", {RuntimeValue::from_i32x4(1, 2, 3, 4), RuntimeValue::from_i32(999)});
    }

    // 5. vshuffle
    {
        Function* fn = mod.create_function("test_shuffle_reverse", Type::i32x4(), {Type::i32x4(), Type::i32x4()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* p0 = b.add_block_param(bb, Type::i32x4());
        Value* p1 = b.add_block_param(bb, Type::i32x4());
        Value* shuf = b.build_vshuffle(p0, p1, 0x1B);
        b.build_ret(shuf);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        assert_diff(mod, "test_shuffle_reverse", {
            RuntimeValue::from_i32x4(11, 22, 33, 44),
            RuntimeValue::from_i32x4(55, 66, 77, 88)
        });
    }
}

TEST_CASE("SIMD Differential - Memory Operations (VStore & VLoad)") {
    Module mod("simd_diff_memory");
    alignas(16) uint8_t buffer[64] = {0};
    uintptr_t buf_ptr = reinterpret_cast<uintptr_t>(buffer);

    // vstore then vload at offset 16
    Function* fn = mod.create_function("test_store_load", Type::i32x4(), {Type::ptr(), Type::i32x4()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* bb = b.append_block("entry");
    Value* ptr = b.add_block_param(bb, Type::ptr());
    Value* val = b.add_block_param(bb, Type::i32x4());
    b.build_vstore(Type::i32x4(), ptr, 16, val);
    Value* loaded = b.build_vload(Type::i32x4(), ptr, 16);
    b.build_ret(loaded);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    RuntimeValue vec_val = RuntimeValue::from_i32x4(100, 200, 300, 400);
    assert_diff(mod, "test_store_load", {RuntimeValue::from_ptr(buf_ptr), vec_val});
}

TEST_CASE("SIMD Differential - Math Kernels (Dot Product & Euclidean Norm)") {
    // 1. 4D Vector Dot Product (f32x4)
    {
        Module mod("simd_dot4");
        Function* fn = mod.create_function("dot4", Type::f32(), {Type::f32x4(), Type::f32x4()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* a = b.add_block_param(bb, Type::f32x4());
        Value* b_param = b.add_block_param(bb, Type::f32x4());

        Value* prod = b.build_vmul(a, b_param);
        Value* l0 = b.build_vextract_lane(prod, 0);
        Value* l1 = b.build_vextract_lane(prod, 1);
        Value* l2 = b.build_vextract_lane(prod, 2);
        Value* l3 = b.build_vextract_lane(prod, 3);

        Value* sum01 = b.build_add(l0, l1);
        Value* sum23 = b.build_add(l2, l3);
        Value* dot = b.build_add(sum01, sum23);
        b.build_ret(dot);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        RuntimeValue v1 = RuntimeValue::from_f32x4(1.0f, 2.0f, 3.0f, 4.0f);
        RuntimeValue v2 = RuntimeValue::from_f32x4(2.0f, 0.5f, -1.0f, 3.0f);
        // dot = 1*2 + 2*0.5 + 3*-1 + 4*3 = 2 + 1 - 3 + 12 = 12.0f
        assert_diff(mod, "dot4", {v1, v2});
    }

    // 2. 4D Vector Euclidean Norm
    {
        Module mod("simd_norm4");
        Function* fn = mod.create_function("norm4", Type::f32(), {Type::f32x4()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* vec = b.add_block_param(bb, Type::f32x4());

        Value* sq = b.build_vmul(vec, vec);
        Value* l0 = b.build_vextract_lane(sq, 0);
        Value* l1 = b.build_vextract_lane(sq, 1);
        Value* l2 = b.build_vextract_lane(sq, 2);
        Value* l3 = b.build_vextract_lane(sq, 3);

        Value* sum01 = b.build_add(l0, l1);
        Value* sum23 = b.build_add(l2, l3);
        Value* sum = b.build_add(sum01, sum23);
        Value* v_sum = b.build_vbroadcast(Type::f32x4(), sum);
        Value* v_norm = b.build_vsqrt(v_sum);
        Value* norm = b.build_vextract_lane(v_norm, 0);
        b.build_ret(norm);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        // Norm of (1, 2, 2, 4) -> 1 + 4 + 4 + 16 = 25 -> sqrt(25) = 5.0f
        RuntimeValue v = RuntimeValue::from_f32x4(1.0f, 2.0f, 2.0f, 4.0f);
        assert_diff(mod, "norm4", {v});
    }
}
