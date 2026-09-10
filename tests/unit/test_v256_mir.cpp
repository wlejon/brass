#include "test_framework.hpp"
#include <brass/mir/types.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <cmath>
#include <sstream>

using namespace brass;

TEST_CASE("V256 Vector Types Properties") {
    // Check f32x8
    Type f32x8_ty = Type::f32x8();
    CHECK(f32x8_ty.is_vector());
    CHECK(f32x8_ty.is_v256());
    CHECK_EQ(f32x8_ty.size_in_bytes(), 32u);
    CHECK_EQ(f32x8_ty.vector_lanes(), 8u);
    CHECK(f32x8_ty.element_type() == Type::f32());
    CHECK_EQ(to_string(f32x8_ty), "f32x8");

    // Check f64x4
    Type f64x4_ty = Type::f64x4();
    CHECK(f64x4_ty.is_vector());
    CHECK(f64x4_ty.is_v256());
    CHECK_EQ(f64x4_ty.size_in_bytes(), 32u);
    CHECK_EQ(f64x4_ty.vector_lanes(), 4u);
    CHECK(f64x4_ty.element_type() == Type::f64());
    CHECK_EQ(to_string(f64x4_ty), "f64x4");

    // Check i32x8
    Type i32x8_ty = Type::i32x8();
    CHECK(i32x8_ty.is_vector());
    CHECK(i32x8_ty.is_v256());
    CHECK_EQ(i32x8_ty.size_in_bytes(), 32u);
    CHECK_EQ(i32x8_ty.vector_lanes(), 8u);
    CHECK(i32x8_ty.element_type() == Type::i32());
    CHECK_EQ(to_string(i32x8_ty), "i32x8");

    // Check i64x4
    Type i64x4_ty = Type::i64x4();
    CHECK(i64x4_ty.is_vector());
    CHECK(i64x4_ty.is_v256());
    CHECK_EQ(i64x4_ty.size_in_bytes(), 32u);
    CHECK_EQ(i64x4_ty.vector_lanes(), 4u);
    CHECK(i64x4_ty.element_type() == Type::i64());
    CHECK_EQ(to_string(i64x4_ty), "i64x4");

    // 128-bit types are NOT v256
    CHECK(!Type::f32x4().is_v256());
    CHECK(!Type::f64x2().is_v256());
    CHECK(!Type::i32x4().is_v256());
    CHECK(!Type::i64x2().is_v256());
}

TEST_CASE("V256 MIR Builder and Verifier") {
    Module mod("v256_builder_test");
    Function* fn = mod.create_function("v256_ops", Type::f32x8(), {Type::f32x8(), Type::f32x8(), Type::f32x8(), Type::f32()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);

    Value* a = b.add_block_param(entry, Type::f32x8());
    Value* c_val = b.add_block_param(entry, Type::f32x8());
    Value* d = b.add_block_param(entry, Type::f32x8());
    Value* s = b.add_block_param(entry, Type::f32());

    Value* v_add = b.build_vadd(a, c_val);
    Value* v_fma = b.build_vfma(v_add, d, a);
    Value* bcast = b.build_vbroadcast(Type::f32x8(), s);
    Value* res = b.build_vsub(v_fma, bcast);
    b.build_ret(res);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));

    // Verifier rejects scalar/vector type mismatch on vfma
    Function* bad_fn = mod.create_function("bad_vfma", Type::f32(), {Type::f32(), Type::f32x8(), Type::f32x8()});
    BasicBlock* bad_entry = b.append_block("bad_entry");
    bad_entry->set_parent(bad_fn);
    b.set_function(bad_fn);
    b.position_at_end(bad_entry);

    Value* bad_p0 = b.add_block_param(bad_entry, Type::f32());
    Value* bad_p1 = b.add_block_param(bad_entry, Type::f32x8());
    Value* bad_p2 = b.add_block_param(bad_entry, Type::f32x8());

    Instruction* bad_inst = mod.arena().make<Instruction>(Opcode::vfma, Type::f32x8());
    bad_inst->add_operand(bad_p0); // scalar!
    bad_inst->add_operand(bad_p1);
    bad_inst->add_operand(bad_p2);
    bad_entry->append_instruction(bad_inst);
    b.build_ret(bad_p0);

    DiagnosticReporter bad_diag;
    CHECK(!verify_function(*bad_fn, &bad_diag));
}

TEST_CASE("V256 MIR Printer and Parser Roundtrip") {
    std::string mir_src =
        "module @v256_rt\n\n"
        "func @vec_math(%0: f32x8, %1: f32x8, %2: f32) -> f32x8 {\n"
        "entry:\n"
        "  %3: f32x8 = vmul %0, %1\n"
        "  %4: f32x8 = vfma %3, %0, %1\n"
        "  %5: f32x8 = vbroadcast.f32x8 %2\n"
        "  %6: f32x8 = vadd %4, %5\n"
        "  ret %6\n"
        "}\n\n"
        "func @scalar_fma(%0: f32, %1: f32, %2: f32) -> f32 {\n"
        "entry:\n"
        "  %3 = fma.f32 %0, %1, %2\n"
        "  ret %3\n"
        "}\n";

    DiagnosticReporter parse_diag;
    auto parsed_mod = parse_module(mir_src, &parse_diag);
    REQUIRE(parsed_mod != nullptr);

    DiagnosticReporter vdiag;
    CHECK(verify_module(*parsed_mod, &vdiag));

    std::ostringstream ss;
    print_module(*parsed_mod, ss);
    std::string printed = ss.str();
    auto reparsed_mod = parse_module(printed, &parse_diag);
    REQUIRE(reparsed_mod != nullptr);
    CHECK(verify_module(*reparsed_mod, &vdiag));
}

TEST_CASE("V256 Interpreter Value Execution") {
    // 1. Scalar FMA
    RuntimeValue sf32 = val_fma_f32(RuntimeValue::from_f32(2.5f), RuntimeValue::from_f32(4.0f), RuntimeValue::from_f32(1.5f));
    CHECK(std::abs(sf32.as_f32() - 11.5f) < 1e-5f);

    RuntimeValue sf64 = val_fma_f64(RuntimeValue::from_f64(3.0), RuntimeValue::from_f64(5.0), RuntimeValue::from_f64(2.5));
    CHECK(std::abs(sf64.as_f64() - 17.5) < 1e-5);

    // 2. 256-bit f32x8 broadcast, lanes, arithmetic, vfma
    RuntimeValue v_bcast = val_vbroadcast(Type::f32x8(), RuntimeValue::from_f32(3.0f));
    for (uint32_t i = 0; i < 8; ++i) {
        RuntimeValue lane = val_vextract_lane(v_bcast, i);
        CHECK(std::abs(lane.as_f32() - 3.0f) < 1e-5f);
    }

    RuntimeValue v_ones = val_vbroadcast(Type::f32x8(), RuntimeValue::from_f32(1.0f));
    RuntimeValue v_twos = val_vbroadcast(Type::f32x8(), RuntimeValue::from_f32(2.0f));
    RuntimeValue v_tens = val_vbroadcast(Type::f32x8(), RuntimeValue::from_f32(10.0f));

    // vfma(v_ones * v_twos + v_tens) = 1.0 * 2.0 + 10.0 = 12.0
    RuntimeValue v_fma = val_vfma(v_ones, v_twos, v_tens);
    for (uint32_t i = 0; i < 8; ++i) {
        RuntimeValue lane = val_vextract_lane(v_fma, i);
        CHECK(std::abs(lane.as_f32() - 12.0f) < 1e-5f);
    }

    // 3. 256-bit i32x8 arithmetic and bitwise
    RuntimeValue i_four = val_vbroadcast(Type::i32x8(), RuntimeValue::from_i32(4));
    RuntimeValue i_six = val_vbroadcast(Type::i32x8(), RuntimeValue::from_i32(6));
    RuntimeValue i_add = val_vadd(i_four, i_six);
    RuntimeValue i_mul = val_vmul(i_four, i_six);
    RuntimeValue i_xor = val_vxor(i_four, i_six);

    for (uint32_t i = 0; i < 8; ++i) {
        CHECK_EQ(val_vextract_lane(i_add, i).as_i32(), 10);
        CHECK_EQ(val_vextract_lane(i_mul, i).as_i32(), 24);
        CHECK_EQ(val_vextract_lane(i_xor, i).as_i32(), 4 ^ 6);
    }
}
