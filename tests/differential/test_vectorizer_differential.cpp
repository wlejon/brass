#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/slp_vectorize.hpp>
#include <brass/mir/loop_vectorize.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <iostream>

using namespace brass;

namespace {

struct DifferentialHarness {
    static void run_differential_array_test(
        const Module& mod,
        std::string_view fn_name,
        float* a,
        float* b,
        float* out_scalar,
        float* out_vectorized,
        int64_t count
    ) {
        // 1. Run Non-vectorized Reference Interpreter Oracle
        Interpreter interp;
        std::vector<RuntimeValue> interp_args = {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a)),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b)),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_scalar)),
            RuntimeValue::from_i64(count)
        };
        interp.set_module(&mod);
        interp.run(*mod.get_function(fn_name), interp_args);

        // 2. Clone and Auto-Vectorize Module
        auto opt_mod = clone_module(mod);
        REQUIRE(opt_mod != nullptr);

        Function* opt_fn = opt_mod->get_function(fn_name);
        REQUIRE(opt_fn != nullptr);

        opt_fn->rebuild_cfg_predecessors();
        DominatorTree dom(*opt_fn);

        SlpOptions slp_opts;
        slp_opts.allow_fp_reassociation = true;
        slp_vectorize_function(*opt_fn, slp_opts);

        LoopVectorizeOptions vec_opts;
        vec_opts.allow_fp_reassociation = true;
        loop_vectorize_pass(*opt_fn, dom, vec_opts);

        opt_fn->rebuild_cfg_predecessors();
        DiagnosticReporter diag;
        REQUIRE(verify_module(*opt_mod, &diag));

        // 3. Compile and Run Native JIT on Vectorized Module
        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*opt_mod));

        std::vector<RuntimeValue> jit_args = {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a)),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b)),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_vectorized)),
            RuntimeValue::from_i64(count)
        };
        jit.invoke(fn_name, jit_args);

        // 4. Verify outputs match across all elements
        for (int64_t idx = 0; idx < count; ++idx) {
            float expected = out_scalar[idx];
            float actual = out_vectorized[idx];
            float diff = std::abs(expected - actual);
            CHECK(diff < 1e-4f);
        }
    }
};

std::unique_ptr<Module> build_array_math_module() {
    auto mod = std::make_unique<Module>("diff_array_math");
    Function* fn = mod->create_function("saxpy_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* pa = b.add_block_param(entry, Type::ptr());
    Value* pb = b.add_block_param(entry, Type::ptr());
    Value* pout = b.add_block_param(entry, Type::ptr());
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
    Value* vsum = b.build_add(va, vb);
    b.build_store_indexed(Type::f32(), pout, i, 4, 0, vsum);

    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_coord4_module() {
    auto mod = std::make_unique<Module>("diff_coord4");
    Function* fn = mod->create_function("coord4_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* pin = b.add_block_param(entry, Type::ptr());
    Value* pscale = b.add_block_param(entry, Type::ptr());
    Value* pout = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());
    (void)count;

    // Load 4 coordinates
    Value* x = b.build_load(Type::f32(), pin, 0);
    Value* y = b.build_load(Type::f32(), pin, 4);
    Value* z = b.build_load(Type::f32(), pin, 8);
    Value* w = b.build_load(Type::f32(), pin, 12);

    Value* sx = b.build_load(Type::f32(), pscale, 0);
    Value* sy = b.build_load(Type::f32(), pscale, 4);
    Value* sz = b.build_load(Type::f32(), pscale, 8);
    Value* sw = b.build_load(Type::f32(), pscale, 12);

    Value* rx = b.build_add(x, sx);
    Value* ry = b.build_add(y, sy);
    Value* rz = b.build_add(z, sz);
    Value* rw = b.build_add(w, sw);

    b.build_store(Type::f32(), pout, 0, rx);
    b.build_store(Type::f32(), pout, 4, ry);
    b.build_store(Type::f32(), pout, 8, rz);
    b.build_store(Type::f32(), pout, 12, rw);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

TEST_CASE("Vectorizer Differential - Array Addition Across Loop Bounds") {
    auto mod = build_array_math_module();
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

    const std::vector<int64_t> test_bounds = {0, 1, 3, 4, 7, 8, 15, 16, 23, 32, 47, 64, 100};

    for (int64_t n : test_bounds) {
        std::vector<float> a(static_cast<size_t>(n + 16));
        std::vector<float> b(static_cast<size_t>(n + 16));
        std::vector<float> out_scalar(static_cast<size_t>(n + 16), 0.0f);
        std::vector<float> out_vec(static_cast<size_t>(n + 16), 0.0f);

        for (int64_t idx = 0; idx < n; ++idx) {
            a[static_cast<size_t>(idx)] = dist(rng);
            b[static_cast<size_t>(idx)] = dist(rng);
        }

        DifferentialHarness::run_differential_array_test(
            *mod,
            "saxpy_kernel",
            a.data(),
            b.data(),
            out_scalar.data(),
            out_vec.data(),
            n
        );
    }
}

TEST_CASE("Vectorizer Differential - 4D Coordinate Transformations") {
    auto mod = build_coord4_module();
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> dist(-50.0f, 50.0f);

    for (int trial = 0; trial < 10; ++trial) {
        alignas(16) float pin[4] = {dist(rng), dist(rng), dist(rng), dist(rng)};
        alignas(16) float pscale[4] = {dist(rng), dist(rng), dist(rng), dist(rng)};
        alignas(16) float out_scalar[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        alignas(16) float out_vec[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        DifferentialHarness::run_differential_array_test(
            *mod,
            "coord4_kernel",
            pin,
            pscale,
            out_scalar,
            out_vec,
            4
        );
    }
}

TEST_CASE("Vectorizer Differential - Integer Loop Vectorization") {
    auto mod = std::make_unique<Module>("diff_int_loop");
    Function* fn = mod->create_function("int_add_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* pa = b.add_block_param(entry, Type::ptr());
    Value* pb = b.add_block_param(entry, Type::ptr());
    Value* pout = b.add_block_param(entry, Type::ptr());
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
    Value* va = b.build_load_indexed(Type::i32(), pa, i, 4, 0);
    Value* vb = b.build_load_indexed(Type::i32(), pb, i, 4, 0);
    Value* vsum = b.build_add(va, vb);
    b.build_store_indexed(Type::i32(), pout, i, 4, 0, vsum);

    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    const std::vector<int64_t> test_bounds = {0, 1, 4, 7, 16, 29, 64};
    for (int64_t n : test_bounds) {
        std::vector<int32_t> a(static_cast<size_t>(n + 16));
        std::vector<int32_t> b_arr(static_cast<size_t>(n + 16));
        std::vector<int32_t> out_scalar(static_cast<size_t>(n + 16), 0);
        std::vector<int32_t> out_vec(static_cast<size_t>(n + 16), 0);

        for (int64_t idx = 0; idx < n; ++idx) {
            a[static_cast<size_t>(idx)] = static_cast<int32_t>(idx * 17);
            b_arr[static_cast<size_t>(idx)] = static_cast<int32_t>(idx * 3 + 5);
        }

        // Run Interpreter
        Interpreter interp;
        interp.set_module(mod.get());
        interp.run(*mod->get_function("int_add_kernel"), {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b_arr.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_scalar.data())),
            RuntimeValue::from_i64(n)
        });

        // Run Vectorized JIT
        auto opt_mod = clone_module(*mod);
        Function* opt_fn = opt_mod->get_function("int_add_kernel");
        DominatorTree dom(*opt_fn);
        LoopVectorizeOptions vec_opts;
        loop_vectorize_pass(*opt_fn, dom, vec_opts);

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*opt_mod));
        jit.invoke("int_add_kernel", {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(a.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(b_arr.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_vec.data())),
            RuntimeValue::from_i64(n)
        });

        for (int64_t idx = 0; idx < n; ++idx) {
            CHECK_EQ(out_vec[static_cast<size_t>(idx)], out_scalar[static_cast<size_t>(idx)]);
        }
    }
}