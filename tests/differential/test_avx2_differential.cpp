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

static void build_v256_binop_test(
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

static void build_v256_fma_test(
    Module& mod,
    std::string_view name,
    Type vec_type
) {
    Function* fn = mod.create_function(name, vec_type, {vec_type, vec_type, vec_type});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* p0 = b.add_block_param(entry, vec_type);
    Value* p1 = b.add_block_param(entry, vec_type);
    Value* p2 = b.add_block_param(entry, vec_type);
    Value* res = b.build_vfma(p0, p1, p2);
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
}

static void build_scalar_fma_test(
    Module& mod,
    std::string_view name,
    Type scalar_type
) {
    Function* fn = mod.create_function(name, scalar_type, {scalar_type, scalar_type, scalar_type});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* p0 = b.add_block_param(entry, scalar_type);
    Value* p1 = b.add_block_param(entry, scalar_type);
    Value* p2 = b.add_block_param(entry, scalar_type);
    Value* res = b.build_fma(p0, p1, p2);
    b.build_ret(res);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
}

TEST_CASE("AVX2 Differential - F32x8 Arithmetic") {
    Module mod("avx2_diff_f32x8");
    build_v256_binop_test(mod, "f32x8_add", Type::f32x8(), Opcode::vadd);
    build_v256_binop_test(mod, "f32x8_sub", Type::f32x8(), Opcode::vsub);
    build_v256_binop_test(mod, "f32x8_mul", Type::f32x8(), Opcode::vmul);
    build_v256_binop_test(mod, "f32x8_div", Type::f32x8(), Opcode::vdiv);
    build_v256_binop_test(mod, "f32x8_min", Type::f32x8(), Opcode::vmin);
    build_v256_binop_test(mod, "f32x8_max", Type::f32x8(), Opcode::vmax);
    build_v256_binop_test(mod, "f32x8_xor", Type::f32x8(), Opcode::vxor);

    RuntimeValue a = RuntimeValue::from_f32x8(1.0f, -2.0f, 3.5f, 4.0f, 5.0f, -6.5f, 7.0f, 8.25f);
    RuntimeValue b = RuntimeValue::from_f32x8(2.0f, 3.0f, -1.0f, 0.5f, -2.0f, 4.0f, 1.5f, 2.0f);

    assert_diff(mod, "f32x8_add", {a, b});
    assert_diff(mod, "f32x8_sub", {a, b});
    assert_diff(mod, "f32x8_mul", {a, b});
    assert_diff(mod, "f32x8_div", {a, b});
    assert_diff(mod, "f32x8_min", {a, b});
    assert_diff(mod, "f32x8_max", {a, b});
    assert_diff(mod, "f32x8_xor", {a, b});
}

TEST_CASE("AVX2 Differential - F64x4 Arithmetic") {
    Module mod("avx2_diff_f64x4");
    build_v256_binop_test(mod, "f64x4_add", Type::f64x4(), Opcode::vadd);
    build_v256_binop_test(mod, "f64x4_sub", Type::f64x4(), Opcode::vsub);
    build_v256_binop_test(mod, "f64x4_mul", Type::f64x4(), Opcode::vmul);
    build_v256_binop_test(mod, "f64x4_div", Type::f64x4(), Opcode::vdiv);

    RuntimeValue a = RuntimeValue::from_f64x4(10.5, -20.25, 30.0, 40.5);
    RuntimeValue b = RuntimeValue::from_f64x4(2.5, 4.0, -5.0, 0.5);

    assert_diff(mod, "f64x4_add", {a, b});
    assert_diff(mod, "f64x4_sub", {a, b});
    assert_diff(mod, "f64x4_mul", {a, b});
    assert_diff(mod, "f64x4_div", {a, b});
}

TEST_CASE("AVX2 Differential - I32x8 and I64x4 Arithmetic") {
    Module mod("avx2_diff_int");
    build_v256_binop_test(mod, "i32x8_add", Type::i32x8(), Opcode::vadd);
    build_v256_binop_test(mod, "i32x8_sub", Type::i32x8(), Opcode::vsub);
    build_v256_binop_test(mod, "i32x8_mul", Type::i32x8(), Opcode::vmul);
    build_v256_binop_test(mod, "i32x8_and", Type::i32x8(), Opcode::vand);
    build_v256_binop_test(mod, "i32x8_or",  Type::i32x8(), Opcode::vor);
    build_v256_binop_test(mod, "i32x8_xor", Type::i32x8(), Opcode::vxor);

    RuntimeValue a32 = RuntimeValue::from_i32x8(10, 20, 30, 40, 50, 60, 70, 80);
    RuntimeValue b32 = RuntimeValue::from_i32x8(1, 2, 3, 4, 5, 6, 7, 8);

    assert_diff(mod, "i32x8_add", {a32, b32});
    assert_diff(mod, "i32x8_sub", {a32, b32});
    assert_diff(mod, "i32x8_mul", {a32, b32});
    assert_diff(mod, "i32x8_and", {a32, b32});
    assert_diff(mod, "i32x8_or",  {a32, b32});
    assert_diff(mod, "i32x8_xor", {a32, b32});

    build_v256_binop_test(mod, "i64x4_add", Type::i64x4(), Opcode::vadd);
    build_v256_binop_test(mod, "i64x4_sub", Type::i64x4(), Opcode::vsub);
    build_v256_binop_test(mod, "i64x4_xor", Type::i64x4(), Opcode::vxor);

    RuntimeValue a64 = RuntimeValue::from_i64x4(100, 200, 300, 400);
    RuntimeValue b64 = RuntimeValue::from_i64x4(10, 20, 30, 40);

    assert_diff(mod, "i64x4_add", {a64, b64});
    assert_diff(mod, "i64x4_sub", {a64, b64});
    assert_diff(mod, "i64x4_xor", {a64, b64});
}

TEST_CASE("AVX2 Differential - FMA3 Operations") {
    Module mod("avx2_diff_fma");
    build_v256_fma_test(mod, "vfma_f32x8", Type::f32x8());
    build_v256_fma_test(mod, "vfma_f64x4", Type::f64x4());
    build_scalar_fma_test(mod, "fma_f32", Type::f32());
    build_scalar_fma_test(mod, "fma_f64", Type::f64());

    RuntimeValue a_v8 = RuntimeValue::from_f32x8(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f);
    RuntimeValue b_v8 = RuntimeValue::from_f32x8(0.5f, 1.5f, 2.0f, 2.5f, 3.0f, 0.5f, 1.0f, 2.0f);
    RuntimeValue c_v8 = RuntimeValue::from_f32x8(10.0f, 10.0f, 10.0f, 10.0f, 20.0f, 20.0f, 20.0f, 20.0f);

    assert_diff(mod, "vfma_f32x8", {a_v8, b_v8, c_v8});

    RuntimeValue a_v4 = RuntimeValue::from_f64x4(2.0, 3.0, 4.0, 5.0);
    RuntimeValue b_v4 = RuntimeValue::from_f64x4(1.5, 2.5, 3.5, 4.5);
    RuntimeValue c_v4 = RuntimeValue::from_f64x4(100.0, 200.0, 300.0, 400.0);

    assert_diff(mod, "vfma_f64x4", {a_v4, b_v4, c_v4});

    assert_diff(mod, "fma_f32", {RuntimeValue::from_f32(3.0f), RuntimeValue::from_f32(4.0f), RuntimeValue::from_f32(5.5f)});
    assert_diff(mod, "fma_f64", {RuntimeValue::from_f64(3.0), RuntimeValue::from_f64(4.0), RuntimeValue::from_f64(5.5)});
}
