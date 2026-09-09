#include "test_framework.hpp"
#include <brass/mir/types.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/slp_vectorize.hpp>
#include <brass/mir/loop_vectorize.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <vector>
#include <cmath>

using namespace brass;

TEST_CASE("SLP - 4D Coordinate Operation Packetization") {
    // Computes:
    //   out[0] = in[0] * 2.0f + 10.0f
    //   out[1] = in[1] * 2.0f + 10.0f
    //   out[2] = in[2] * 2.0f + 10.0f
    //   out[3] = in[3] * 2.0f + 10.0f
    Module mod("slp_coord4_test");
    Function* fn = mod.create_function("coord4_transform", Type::void_type(), {Type::ptr(), Type::ptr()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* pin = b.add_block_param(entry, Type::ptr());
    Value* pout = b.add_block_param(entry, Type::ptr());

    // Scalar loads
    Value* x = b.build_load(Type::f32(), pin, 0);
    Value* y = b.build_load(Type::f32(), pin, 4);
    Value* z = b.build_load(Type::f32(), pin, 8);
    Value* w = b.build_load(Type::f32(), pin, 12);



    // Multiply and Add in i32 for simplicity or float
    Value* x1 = b.build_add(x, y); // general arithmetic
    Value* y1 = b.build_add(y, z);
    Value* z1 = b.build_add(z, w);
    Value* w1 = b.build_add(w, x);

    b.build_store(Type::f32(), pout, 0, x1);
    b.build_store(Type::f32(), pout, 4, y1);
    b.build_store(Type::f32(), pout, 8, z1);
    b.build_store(Type::f32(), pout, 12, w1);
    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    // Run SLP vectorization
    SlpOptions options;
    bool changed = slp_vectorize_function(*fn, options);
    CHECK(changed);

    // Verify module after SLP
    DiagnosticReporter diag_after;
    if (!verify_module(mod, &diag_after)) {
        std::cerr << diag_after.format_all() << "\n";
        print_module(mod, std::cerr);
    }
    REQUIRE(verify_module(mod, &diag_after));

    // Verify vector store exists in entry block
    bool has_vstore = false;
    for (Instruction* inst = entry->head(); inst != nullptr; inst = inst->next()) {
        if (inst->opcode() == Opcode::vstore) {
            has_vstore = true;
            CHECK_EQ(inst->memory_type(), Type::f32x4());
        }
    }
    CHECK(has_vstore);

    // JIT Execution
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    alignas(16) float in_buf[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    alignas(16) float out_buf[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    jit.invoke("coord4_transform", {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_buf)),
                                   RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_buf))});

    CHECK_EQ(out_buf[0], 3.0f); // 1 + 2
    CHECK_EQ(out_buf[1], 5.0f); // 2 + 3
    CHECK_EQ(out_buf[2], 7.0f); // 3 + 4
    CHECK_EQ(out_buf[3], 5.0f); // 4 + 1
}

TEST_CASE("SLP - Contiguous Load to VLoad and Store to VStore Fusion") {
    // 1. Contiguous 4 loads fused into vload.f32x4
    {
        Module mod("slp_load_fusion");
        Function* fn = mod.create_function("load_fusion", Type::f32(), {Type::ptr()});
        Builder b(mod);
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* pin = b.add_block_param(entry, Type::ptr());

        Value* l0 = b.build_load(Type::f32(), pin, 0);
        Value* l1 = b.build_load(Type::f32(), pin, 4);
        Value* l2 = b.build_load(Type::f32(), pin, 8);
        Value* l3 = b.build_load(Type::f32(), pin, 12);

        Value* s01 = b.build_add(l0, l1);
        Value* s23 = b.build_add(l2, l3);
        Value* sum = b.build_add(s01, s23);
        b.build_ret(sum);
        fn->rebuild_cfg_predecessors();

        SlpOptions options;
        bool changed = slp_vectorize_function(*fn, options);
        CHECK(changed);

        DiagnosticReporter diag;
        REQUIRE(verify_module(mod, &diag));

        bool has_vload = false;
        for (Instruction* inst = entry->head(); inst != nullptr; inst = inst->next()) {
            if (inst->opcode() == Opcode::vload) {
                has_vload = true;
                CHECK_EQ(inst->memory_type(), Type::f32x4());
            }
        }
        CHECK(has_vload);

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(mod));

        alignas(16) float in_buf[4] = {10.0f, 20.0f, 30.0f, 40.0f};
        RuntimeValue res = jit.invoke("load_fusion", {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_buf))});
        CHECK_EQ(res.as_f32(), 100.0f);
    }

    // 2. Contiguous 2 loads and 2 stores of f64 fused into vload.f64x2 and vstore.f64x2
    {
        Module mod("slp_f64_fusion");
        Function* fn = mod.create_function("f64_copy2", Type::void_type(), {Type::ptr(), Type::ptr()});
        Builder b(mod);
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        Value* pin = b.add_block_param(entry, Type::ptr());
        Value* pout = b.add_block_param(entry, Type::ptr());

        Value* d0 = b.build_load(Type::f64(), pin, 0);
        Value* d1 = b.build_load(Type::f64(), pin, 8);
        b.build_store(Type::f64(), pout, 0, d0);
        b.build_store(Type::f64(), pout, 8, d1);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();

        SlpOptions options;
        bool changed = slp_vectorize_function(*fn, options);
        CHECK(changed);

        DiagnosticReporter diag;
        REQUIRE(verify_module(mod, &diag));

        bool has_vstore = false;
        for (Instruction* inst = entry->head(); inst != nullptr; inst = inst->next()) {
            if (inst->opcode() == Opcode::vstore) {
                has_vstore = true;
                CHECK_EQ(inst->memory_type(), Type::f64x2());
            }
        }
        CHECK(has_vstore);

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(mod));

        alignas(16) double in_buf[2] = {3.14159, 2.71828};
        alignas(16) double out_buf[2] = {0.0, 0.0};

        jit.invoke("f64_copy2", {RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_buf)),
                                 RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_buf))});

        CHECK_EQ(out_buf[0], 3.14159);
        CHECK_EQ(out_buf[1], 2.71828);
    }
}

TEST_CASE("Loop Vectorizer - Exact Multiple of 4 Trip Count") {
    // for (i = 0; i < count; ++i) c[i] = a[i] + b[i]
    Module mod("vec_loop_exact");
    Function* fn = mod.create_function("vec_add_array", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* pa = b.add_block_param(entry, Type::ptr());
    Value* pb = b.add_block_param(entry, Type::ptr());
    Value* pc = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    b.build_br(loop_hdr, {zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, count);
    b.build_br_if(cond, loop_body, exit_bb);

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* va = b.build_load_indexed(Type::f32(), pa, i, 4, 0);
    Value* vb = b.build_load_indexed(Type::f32(), pb, i, 4, 0);
    Value* vc = b.build_add(va, vb);
    b.build_store_indexed(Type::f32(), pc, i, 4, 0, vc);

    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    DominatorTree dom(*fn);
    LoopVectorizeOptions vec_opts;
    bool changed = loop_vectorize_pass(*fn, dom, vec_opts);
    CHECK(changed);

    DiagnosticReporter diag_after;
    if (!verify_module(mod, &diag_after)) {
        std::cerr << diag_after.format_all() << "\n";
        print_module(mod, std::cerr);
    }
    REQUIRE(verify_module(mod, &diag_after));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    constexpr int N = 16;
    alignas(16) float a[N];
    alignas(16) float c[N];
    alignas(16) float out[N];

    for (int idx = 0; idx < N; ++idx) {
        a[idx] = static_cast<float>(idx * 2);
        c[idx] = 1.5f;
        out[idx] = 0.0f;
    }

    jit.invoke("vec_add_array", {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a)),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(c)),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out)),
        RuntimeValue::from_i64(N)
    });

    for (int idx = 0; idx < N; ++idx) {
        CHECK_EQ(out[idx], a[idx] + c[idx]);
    }
}

TEST_CASE("Loop Vectorizer - Non-Multiple of 4 Trip Count with Remainder Loop") {
    // 19 elements: 4 vector iterations (16) + 3 remainder iterations (3)
    Module mod("vec_loop_rem");
    Function* fn = mod.create_function("vec_rem_test", Type::void_type(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* pa = b.add_block_param(entry, Type::ptr());
    Value* pb = b.add_block_param(entry, Type::ptr());
    Value* pc = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    b.build_br(loop_hdr, {zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, count);
    b.build_br_if(cond, loop_body, exit_bb);

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* va = b.build_load_indexed(Type::f32(), pa, i, 4, 0);
    Value* vb = b.build_load_indexed(Type::f32(), pb, i, 4, 0);
    Value* vc = b.build_sub(va, vb);
    b.build_store_indexed(Type::f32(), pc, i, 4, 0, vc);

    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    DominatorTree dom(*fn);
    LoopVectorizeOptions vec_opts;
    bool changed = loop_vectorize_pass(*fn, dom, vec_opts);
    CHECK(changed);

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    constexpr int N = 19;
    alignas(16) float a[N];
    alignas(16) float b_arr[N];
    alignas(16) float out[N];

    for (int idx = 0; idx < N; ++idx) {
        a[idx] = static_cast<float>(idx * 10);
        b_arr[idx] = static_cast<float>(idx + 1);
        out[idx] = 0.0f;
    }

    jit.invoke("vec_rem_test", {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a)),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b_arr)),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out)),
        RuntimeValue::from_i64(N)
    });

    for (int idx = 0; idx < N; ++idx) {
        CHECK_EQ(out[idx], a[idx] - b_arr[idx]);
    }
}

TEST_CASE("Loop Vectorizer - Vector Reduction with Horizontal Sum") {
    // Float reduction (dot product): sum += a[i] * b[i]
    // 23 elements (4*5 = 20 vector, 3 remainder)
    Module mod("vec_reduction_test");
    Function* fn = mod.create_function("vec_dot_product", Type::f32(), {Type::ptr(), Type::ptr(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* pa = b.add_block_param(entry, Type::ptr());
    Value* pb = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero_i = b.build_iconst_i64(0);
    Value* zero_f = b.build_fconst_f64(0.0);
    (void)zero_f;
    // We can load zero or pass 0.0f
    Value* zero_init = b.build_load_indexed(Type::f32(), pa, zero_i, 4, 0); // or initial value
    Value* zero_acc = b.build_sub(zero_init, zero_init); // 0.0f

    b.build_br(loop_hdr, {zero_i, zero_acc});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::f32());
    Value* cond = b.build_slt(i, count);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* va = b.build_load_indexed(Type::f32(), pa, i, 4, 0);
    Value* vb = b.build_load_indexed(Type::f32(), pb, i, 4, 0);
    Value* prod = b.build_mul(va, vb);
    Value* next_acc = b.build_add(acc, prod);

    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_acc = b.add_block_param(exit_bb, Type::f32());
    b.build_ret(final_acc);

    fn->rebuild_cfg_predecessors();
    DominatorTree dom(*fn);
    LoopVectorizeOptions vec_opts;
    vec_opts.allow_fp_reassociation = true;
    bool changed = loop_vectorize_pass(*fn, dom, vec_opts);
    CHECK(changed);

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    constexpr int N = 23;
    alignas(16) float a[N];
    alignas(16) float b_arr[N];
    float expected_sum = 0.0f;

    for (int idx = 0; idx < N; ++idx) {
        a[idx] = static_cast<float>(idx + 1);
        b_arr[idx] = 0.5f;
        expected_sum += a[idx] * b_arr[idx];
    }

    RuntimeValue res = jit.invoke("vec_dot_product", {
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a)),
        RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b_arr)),
        RuntimeValue::from_i64(N)
    });

    float diff = std::abs(res.as_f32() - expected_sum);
    CHECK(diff < 1e-4f);

    Interpreter interp;
    for (int test_n : {1, 2, 3, 4, 5, 8, 20, 23}) {
        RuntimeValue jit_val = jit.invoke("vec_dot_product", {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a)),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b_arr)),
            RuntimeValue::from_i64(test_n)
        });
        RuntimeValue interp_val = interp.run(*fn, {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a)),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b_arr)),
            RuntimeValue::from_i64(test_n)
        });
        CHECK(std::abs(jit_val.as_f32() - interp_val.as_f32()) < 1e-4f);
    }
}