#include "test_framework.hpp"
#include <brass/mir/types.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>

using namespace brass;

TEST_CASE("SIMD Vector Types Properties") {
    // Check scalar f32
    Type f32_ty = Type::f32();
    CHECK(f32_ty.is_float());
    CHECK(f32_ty.kind() == TypeKind::F32);
    CHECK(!f32_ty.is_vector());
    CHECK_EQ(f32_ty.size_in_bytes(), 4u);
    CHECK_EQ(to_string(f32_ty), "f32");

    // Check f32x4
    Type f32x4_ty = Type::f32x4();
    CHECK(f32x4_ty.is_vector());
    CHECK_EQ(f32x4_ty.size_in_bytes(), 16u);
    CHECK_EQ(f32x4_ty.vector_lanes(), 4u);
    CHECK(f32x4_ty.element_type() == Type::f32());
    CHECK_EQ(to_string(f32x4_ty), "f32x4");

    // Check f64x2
    Type f64x2_ty = Type::f64x2();
    CHECK(f64x2_ty.is_vector());
    CHECK_EQ(f64x2_ty.size_in_bytes(), 16u);
    CHECK_EQ(f64x2_ty.vector_lanes(), 2u);
    CHECK(f64x2_ty.element_type() == Type::f64());
    CHECK_EQ(to_string(f64x2_ty), "f64x2");

    // Check i32x4
    Type i32x4_ty = Type::i32x4();
    CHECK(i32x4_ty.is_vector());
    CHECK_EQ(i32x4_ty.size_in_bytes(), 16u);
    CHECK_EQ(i32x4_ty.vector_lanes(), 4u);
    CHECK(i32x4_ty.element_type() == Type::i32());
    CHECK_EQ(to_string(i32x4_ty), "i32x4");

    // Check i64x2
    Type i64x2_ty = Type::i64x2();
    CHECK(i64x2_ty.is_vector());
    CHECK_EQ(i64x2_ty.size_in_bytes(), 16u);
    CHECK_EQ(i64x2_ty.vector_lanes(), 2u);
    CHECK(i64x2_ty.element_type() == Type::i64());
    CHECK_EQ(to_string(i64x2_ty), "i64x2");
}

TEST_CASE("SIMD MIR Builder and Opcodes") {
    Module mod("simd_builder_test");
    Function* fn = mod.create_function("simd_ops", Type::f32x4(), {Type::f32x4(), Type::f32x4(), Type::ptr()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* v0 = b.add_block_param(entry, Type::f32x4());
    Value* v1 = b.add_block_param(entry, Type::f32x4());
    Value* ptr = b.add_block_param(entry, Type::ptr());

    // Arithmetic
    Value* r_add = b.build_vadd(v0, v1);
    CHECK(is_vector_op(Opcode::vadd));
    CHECK_EQ(r_add->type(), Type::f32x4());

    Value* r_sub = b.build_vsub(r_add, v1);
    Value* r_mul = b.build_vmul(r_sub, v0);
    Value* r_div = b.build_vdiv(r_mul, v1);
    Value* r_neg = b.build_vneg(r_div);
    Value* r_min = b.build_vmin(r_neg, v0);
    Value* r_max = b.build_vmax(r_min, v1);
    Value* r_sqrt = b.build_vsqrt(r_max);

    // Bitwise
    Value* r_and = b.build_vand(r_sqrt, v0);
    Value* r_or = b.build_vor(r_and, v1);
    Value* r_xor = b.build_vxor(r_or, v0);
    Value* r_not = b.build_vnot(r_xor);

    // Memory
    b.build_vstore(Type::f32x4(), ptr, r_not);
    Value* loaded = b.build_vload(Type::f32x4(), ptr);

    // Broadcast, lane extract/insert, shuffle, zero
    Value* lane0 = b.build_vextract_lane(loaded, 0);
    CHECK_EQ(lane0->type(), Type::f32());

    Value* bcast = b.build_vbroadcast(Type::f32x4(), lane0);
    CHECK_EQ(bcast->type(), Type::f32x4());

    Value* inserted = b.build_vinsert_lane(bcast, lane0, 2);
    CHECK_EQ(inserted->type(), Type::f32x4());

    Value* shuf = b.build_vshuffle(inserted, v0, 0x1B);
    CHECK_EQ(shuf->type(), Type::f32x4());

    Value* vzero = b.build_vzero(Type::f32x4());
    CHECK_EQ(vzero->type(), Type::f32x4());

    b.build_ret(vzero);
    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    if (!ok) {
        std::cerr << diag.format_all() << "\n";
    }
    REQUIRE(ok);
}

TEST_CASE("SIMD MIR Verifier Negative Rules") {
    // 1. Mismatched operand types in vadd
    {
        Module mod("bad_vadd");
        Function* fn = mod.create_function("bad_fn", Type::f32x4(), {Type::f32x4(), Type::i32x4()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* v0 = b.add_block_param(entry, Type::f32x4());
        Value* v1 = b.add_block_param(entry, Type::i32x4());

        Instruction* inst = mod.arena().make<Instruction>(Opcode::vadd, Type::f32x4());
        inst->add_operand(v0);
        inst->add_operand(v1);
        Value* res = b.create_value(Type::f32x4());
        res->set_defining_instruction(inst);
        inst->set_result(res);
        b.insert(inst);
        b.build_ret(res);
        fn->rebuild_cfg_predecessors();

        DiagnosticReporter diag;
        bool ok = verify_module(mod, &diag);
        CHECK(!ok);
    }

    // 2. Non-vector operand to vadd
    {
        Module mod("scalar_vadd");
        Function* fn = mod.create_function("bad_fn", Type::f32x4(), {Type::i32(), Type::i32()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* v0 = b.add_block_param(entry, Type::i32());
        Value* v1 = b.add_block_param(entry, Type::i32());

        Instruction* inst = mod.arena().make<Instruction>(Opcode::vadd, Type::i32());
        inst->add_operand(v0);
        inst->add_operand(v1);
        Value* res = b.create_value(Type::i32());
        res->set_defining_instruction(inst);
        inst->set_result(res);
        b.insert(inst);
        b.build_ret(res);
        fn->rebuild_cfg_predecessors();

        DiagnosticReporter diag;
        bool ok = verify_module(mod, &diag);
        CHECK(!ok);
    }

    // 3. Out-of-bounds lane index in extract_lane
    {
        Module mod("oob_extract");
        Function* fn = mod.create_function("bad_fn", Type::f32(), {Type::f32x4()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* v0 = b.add_block_param(entry, Type::f32x4());

        Instruction* inst = mod.arena().make<Instruction>(Opcode::vextract_lane, Type::f32());
        inst->add_operand(v0);
        inst->set_lane(4); // Only lanes 0..3 valid
        Value* res = b.create_value(Type::f32());
        res->set_defining_instruction(inst);
        inst->set_result(res);
        b.insert(inst);
        b.build_ret(res);
        fn->rebuild_cfg_predecessors();

        DiagnosticReporter diag;
        bool ok = verify_module(mod, &diag);
        CHECK(!ok);
    }

    // 4. Mismatched element type in vinsert_lane
    {
        Module mod("mismatch_insert");
        Function* fn = mod.create_function("bad_fn", Type::f32x4(), {Type::f32x4(), Type::i32()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* v0 = b.add_block_param(entry, Type::f32x4());
        Value* scalar_i32 = b.add_block_param(entry, Type::i32());

        Instruction* inst = mod.arena().make<Instruction>(Opcode::vinsert_lane, Type::f32x4());
        inst->add_operand(v0);
        inst->add_operand(scalar_i32); // Expects f32!
        inst->set_lane(1);
        Value* res = b.create_value(Type::f32x4());
        res->set_defining_instruction(inst);
        inst->set_result(res);
        b.insert(inst);
        b.build_ret(res);
        fn->rebuild_cfg_predecessors();

        DiagnosticReporter diag;
        bool ok = verify_module(mod, &diag);
        CHECK(!ok);
    }

    // 5. vload from non-pointer
    {
        Module mod("bad_vload");
        Function* fn = mod.create_function("bad_fn", Type::f32x4(), {Type::i64()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* not_ptr = b.add_block_param(entry, Type::i64());

        Instruction* inst = mod.arena().make<Instruction>(Opcode::vload, Type::f32x4());
        inst->add_operand(not_ptr);
        Value* res = b.create_value(Type::f32x4());
        res->set_defining_instruction(inst);
        inst->set_result(res);
        b.insert(inst);
        b.build_ret(res);
        fn->rebuild_cfg_predecessors();

        DiagnosticReporter diag;
        bool ok = verify_module(mod, &diag);
        CHECK(!ok);
    }
}

TEST_CASE("SIMD Printer and Parser Roundtrip") {
    Module mod("simd_roundtrip");
    Function* fn = mod.create_function("vector_pipeline", Type::f32x4(), {Type::f32x4(), Type::ptr()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* v0 = b.add_block_param(entry, Type::f32x4());
    Value* ptr = b.add_block_param(entry, Type::ptr());

    Value* vzero = b.build_vzero(Type::f32x4());
    Value* vloaded = b.build_vload(Type::f32x4(), ptr);
    Value* vsum = b.build_vadd(v0, vloaded);
    Value* lane1 = b.build_vextract_lane(vsum, 1);
    Value* vbcast = b.build_vbroadcast(Type::f32x4(), lane1);
    Value* vshuf = b.build_vshuffle(vsum, vbcast, 0xE4);
    Value* vres = b.build_vsub(vshuf, vzero);
    b.build_vstore(Type::f32x4(), ptr, vres);
    b.build_ret(vres);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    bool ok = verify_module(mod, &diag);
    if (!ok) {
        std::cerr << diag.format_all() << "\n";
    }
    REQUIRE(ok);

    std::string printed1 = to_string(mod);
    CHECK(!printed1.empty());

    auto parsed = parse_module(printed1);
    REQUIRE(parsed != nullptr);

    DiagnosticReporter parsed_diag;
    bool ok_parsed = verify_module(*parsed, &parsed_diag);
    if (!ok_parsed) {
        std::cerr << parsed_diag.format_all() << "\n";
    }
    REQUIRE(ok_parsed);

    std::string printed2 = to_string(*parsed);
    CHECK_EQ(printed1, printed2);
}
