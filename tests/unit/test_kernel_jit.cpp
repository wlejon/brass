#include "test_framework.hpp"
#include <brass/codegen/kernel_jit.hpp>
#include <brass/brass_c_api.h>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/target/target.hpp>

#include <vector>
#include <cmath>
#include <chrono>

using namespace brass;
using namespace brass::codegen;

#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a) - (b)) <= (eps))

TEST_CASE("Kernel JIT - AVX2 v256 Fused Multiply-Add Vector Loop") {
    if (!Target::host().is_x64()) { std::cout << "  [SKIP] AVX2 not supported on non-x86 host\n"; return; }
    Module mod("fma_vec_module");
    mod.set_allow_fp_reassociation(true);

    // Signature: fma_kernel(const float* a, const float* b, const float* c, float* out, int64_t n) -> void
    Function* fn = mod.create_function("fma_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_a = b.add_block_param(entry, Type::ptr());
    Value* in_b = b.add_block_param(entry, Type::ptr());
    Value* in_c = b.add_block_param(entry, Type::ptr());
    Value* out_p = b.add_block_param(entry, Type::ptr());
    Value* n_val = b.add_block_param(entry, Type::i64());

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("loop_exit");

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* eight = b.build_iconst_i64(8);
    Value* four = b.build_iconst_i64(4);
    b.build_br(hdr, {zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, n_val);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);

    // Byte offset = iv * 4 bytes
    Value* byte_off = b.build_mul(iv, four);
    Value* ptr_a = b.build_add(in_a, byte_off);
    Value* ptr_b = b.build_add(in_b, byte_off);
    Value* ptr_c = b.build_add(in_c, byte_off);
    Value* ptr_out = b.build_add(out_p, byte_off);

    KernelBuilder kb(b);
    Value* va = kb.vload_f32x8(ptr_a);
    Value* vb = kb.vload_f32x8(ptr_b);
    Value* vc = kb.vload_f32x8(ptr_c);
    Value* vres = kb.vfma(va, vb, vc);
    kb.vstore_f32x8(ptr_out, vres);

    Value* next_iv = b.build_add(iv, eight);
    b.build_br(hdr, {next_iv});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    KernelOptions opts;
    opts.enable_avx2 = true;
    opts.enable_fma = true;
    opts.enable_fp_reassociation = true;

    KernelJit jit(opts);

    auto t0 = std::chrono::high_resolution_clock::now();
    KernelFunction kfn = jit.compile(mod, "fma_kernel");
    auto t1 = std::chrono::high_resolution_clock::now();
    double compile_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Fast in-memory compilation check
    CHECK(compile_ms < 50.0);
    REQUIRE(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, const float*, float*, int64_t)>();
    REQUIRE(fn_ptr != nullptr);

    const int64_t N = 1024;
    std::vector<float> a_data(static_cast<size_t>(N));
    std::vector<float> b_data(static_cast<size_t>(N));
    std::vector<float> c_data(static_cast<size_t>(N));
    std::vector<float> out_data(static_cast<size_t>(N), 0.0f);

    for (int64_t i = 0; i < N; ++i) {
        a_data[static_cast<size_t>(i)] = static_cast<float>(i) * 0.25f;
        b_data[static_cast<size_t>(i)] = 2.0f;
        c_data[static_cast<size_t>(i)] = 1.5f;
    }

    fn_ptr(a_data.data(), b_data.data(), c_data.data(), out_data.data(), N);

    for (int64_t i = 0; i < N; ++i) {
        float expected = a_data[static_cast<size_t>(i)] * b_data[static_cast<size_t>(i)] + c_data[static_cast<size_t>(i)];
        CHECK_NEAR(out_data[static_cast<size_t>(i)], expected, 1e-4f);
    }
}

TEST_CASE("Kernel JIT - Fused Activation Loop (ReLU with Bias Add)") {
    if (!Target::host().is_x64()) { std::cout << "  [SKIP] AVX2 not supported on non-x86 host\n"; return; }
    Module mod("relu_bias_module");

    // Signature: relu_bias_kernel(const float* in, const float* bias, float* out, int64_t n) -> void
    Function* fn = mod.create_function("relu_bias_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_p = b.add_block_param(entry, Type::ptr());
    Value* bias_p = b.add_block_param(entry, Type::ptr());
    Value* out_p = b.add_block_param(entry, Type::ptr());
    Value* n_val = b.add_block_param(entry, Type::i64());

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("loop_exit");

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* eight = b.build_iconst_i64(8);
    Value* four = b.build_iconst_i64(4);

    // Load scalar bias once outside loop and broadcast to 8-lane AVX2 vector
    KernelBuilder kb(b);
    Value* scalar_bias = kb.load_f32(bias_p, 0);
    Value* vbias = kb.vbroadcast(Type::f32x8(), scalar_bias);

    b.build_br(hdr, {zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, n_val);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);

    Value* byte_off = b.build_mul(iv, four);
    Value* ptr_in = b.build_add(in_p, byte_off);
    Value* ptr_out = b.build_add(out_p, byte_off);

    Value* vin = kb.vload_f32x8(ptr_in);
    Value* vres = kb.relu_bias(vin, vbias);
    kb.vstore_f32x8(ptr_out, vres);

    Value* next_iv = b.build_add(iv, eight);
    b.build_br(hdr, {next_iv});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    KernelOptions opts;
    opts.enable_avx2 = true;
    KernelJit jit(opts);

    KernelFunction kfn = jit.compile(mod, "relu_bias_kernel");
    REQUIRE(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const float*, const float*, float*, int64_t)>();
    REQUIRE(fn_ptr != nullptr);

    const int64_t N = 512;
    std::vector<float> in_data(static_cast<size_t>(N));
    std::vector<float> out_data(static_cast<size_t>(N), 0.0f);
    float bias_val = 50.0f;

    for (int64_t i = 0; i < N; ++i) {
        // Range from -200 to +311
        in_data[static_cast<size_t>(i)] = static_cast<float>(i) - 200.0f;
    }

    fn_ptr(in_data.data(), &bias_val, out_data.data(), N);

    for (int64_t i = 0; i < N; ++i) {
        float x = in_data[static_cast<size_t>(i)] + bias_val;
        float expected = (x > 0.0f) ? x : 0.0f;
        CHECK_NEAR(out_data[static_cast<size_t>(i)], expected, 1e-5f);
    }
}

TEST_CASE("Kernel JIT - Fused SiLU Activation Loop") {
    Module mod("silu_module");

    // Signature: silu_kernel(const double* in, double* out, int64_t n) -> void
    Function* fn = mod.create_function("silu_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::i64()
    });

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_p = b.add_block_param(entry, Type::ptr());
    Value* out_p = b.add_block_param(entry, Type::ptr());
    Value* n_val = b.add_block_param(entry, Type::i64());

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("loop_exit");

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* one_i64 = b.build_iconst_i64(1);
    b.build_br(hdr, {zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, n_val);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);

    KernelBuilder kb(b);
    Value* x = kb.load_f64_indexed(in_p, iv, 8, 0);

    // silu(x) = x / (1.0 + exp(-x))
    Value* neg_x = b.build_neg(x);
    Value* exp_val = b.build_call("exp", Type::f64(), {neg_x});
    Value* one_f64 = b.build_fconst_f64(1.0);

    Value* denom = b.build_add(one_f64, exp_val);
    Value* res = b.build_sdiv(x, denom);

    kb.store_f64_indexed(out_p, iv, res, 8, 0);

    Value* next_iv = b.build_add(iv, one_i64);
    b.build_br(hdr, {next_iv});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    KernelOptions opts;
    KernelJit jit(opts);

    KernelFunction kfn = jit.compile(mod, "silu_kernel");
    REQUIRE(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const double*, double*, int64_t)>();
    REQUIRE(fn_ptr != nullptr);

    const int64_t N = 128;
    std::vector<double> in_data(static_cast<size_t>(N));
    std::vector<double> out_data(static_cast<size_t>(N), 0.0);

    for (int64_t i = 0; i < N; ++i) {
        in_data[static_cast<size_t>(i)] = (static_cast<double>(i) - 64.0) * 0.1;
    }

    fn_ptr(in_data.data(), out_data.data(), N);

    for (int64_t i = 0; i < N; ++i) {
        double x_val = in_data[static_cast<size_t>(i)];
        double expected = x_val / (1.0 + std::exp(-x_val));
        CHECK_NEAR(out_data[static_cast<size_t>(i)], expected, 1e-6);
    }
}

TEST_CASE("Kernel JIT - Parallel Chunk Execution") {
    Module mod("par_chunk_mod");

    // Signature: par_scale_kernel(const int64_t* in, int64_t* out, int64_t factor, int64_t n) -> void
    Function* fn = mod.create_function("par_scale_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::i64(), Type::i64()
    });

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_p = b.add_block_param(entry, Type::ptr());
    Value* out_p = b.add_block_param(entry, Type::ptr());
    Value* factor = b.add_block_param(entry, Type::i64());
    Value* n_val = b.add_block_param(entry, Type::i64());
    in_p->set_noalias(true);
    out_p->set_noalias(true);

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("loop_exit");

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(hdr, {zero});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, n_val);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);

    KernelBuilder kb(b);
    Value* elem = kb.load_i64_indexed(in_p, iv, 8, 0);
    Value* scaled = kb.mul(elem, factor);
    kb.store_i64_indexed(out_p, iv, scaled, 8, 0);

    Value* next_iv = b.build_add(iv, one);
    b.build_br(hdr, {next_iv});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    KernelOptions opts;
    opts.enable_parallel = true;
    opts.parallel_threshold = 100;
    opts.parallel_workers = 4;

    KernelJit jit(opts);
    CHECK_EQ(jit.get_parallel_workers(), 4u);

    // Verify polyhedral analysis detects DOALL
    std::vector<KernelLoopAnalysis> analyses = jit.analyze_loops(*fn);
    REQUIRE(!analyses.empty());
    CHECK(analyses[0].is_parallelizable);
    CHECK(analyses[0].is_doall);
    CHECK(!analyses[0].is_reduction);

    KernelFunction kfn = jit.compile(mod, "par_scale_kernel");
    REQUIRE(kfn.is_valid());

    auto fn_ptr = kfn.as<void(*)(const int64_t*, int64_t*, int64_t, int64_t)>();
    REQUIRE(fn_ptr != nullptr);

    const int64_t N = 20000;
    std::vector<int64_t> in_data(static_cast<size_t>(N));
    std::vector<int64_t> out_data(static_cast<size_t>(N), 0);

    for (int64_t i = 0; i < N; ++i) {
        in_data[static_cast<size_t>(i)] = i + 1;
    }

    fn_ptr(in_data.data(), out_data.data(), 7, N);

    for (int64_t i = 0; i < N; ++i) {
        CHECK_EQ(out_data[static_cast<size_t>(i)], (i + 1) * 7);
    }
}

TEST_CASE("Kernel JIT - Polyhedral Loop Dependence Analysis") {
    Module mod("dep_test_mod");

    // Loop with loop-carried RAW dependence: a[i] = a[i-1] + 1
    Function* fn = mod.create_function("carried_dep_loop", Type::void_type(), {
        Type::ptr(), Type::i64()
    });

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* arr_p = b.add_block_param(entry, Type::ptr());
    Value* n_val = b.add_block_param(entry, Type::i64());

    BasicBlock* hdr = b.create_block("loop_hdr");
    BasicBlock* body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("loop_exit");

    b.position_at_end(entry);
    Value* one = b.build_iconst_i64(1);
    b.build_br(hdr, {one});

    fn->append_block(hdr);
    b.position_at_end(hdr);
    Value* iv = b.add_block_param(hdr, Type::i64());
    Value* cond = b.build_slt(iv, n_val);
    b.build_br_if(cond, body, {}, exit, {});

    fn->append_block(body);
    b.position_at_end(body);

    KernelBuilder kb(b);
    Value* prev_iv = b.build_sub(iv, one);
    Value* prev_elem = kb.load_i64_indexed(arr_p, prev_iv, 8, 0);
    Value* new_elem = b.build_add(prev_elem, one);
    kb.store_i64_indexed(arr_p, iv, new_elem, 8, 0);

    Value* next_iv = b.build_add(iv, one);
    b.build_br(hdr, {next_iv});

    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();

    KernelJit jit;
    std::vector<KernelLoopAnalysis> analyses = jit.analyze_loops(*fn);
    REQUIRE(!analyses.empty());

    // Loop-carried dependence must reject parallelization
    CHECK(!analyses[0].is_parallelizable);
    CHECK(!analyses[0].is_doall);
    CHECK(!analyses[0].rejection_reason.empty());
}

TEST_CASE("Kernel JIT - C-API Bindings (Compile & Execute Pure-Compute Kernel)") {
    if (!Target::host().is_x64()) { std::cout << "  [SKIP] AVX2 not supported on non-x86 host\n"; return; }
    BrassContext ctx = brass_context_create();
    REQUIRE(ctx != nullptr);

    BrassKernelOptions opts = brass_kernel_options_create();
    REQUIRE(opts != nullptr);

    brass_kernel_options_set_optimize(opts, 1);
    brass_kernel_options_set_fma(opts, 1);
    brass_kernel_options_set_avx2(opts, 1);
    brass_kernel_options_set_fp_reassociation(opts, 1);
    brass_kernel_options_set_parallel_workers(opts, 2);

    BrassKernelJit kj = brass_kernel_jit_create(ctx, opts);
    REQUIRE(kj != nullptr);
    CHECK_EQ(brass_kernel_jit_get_parallel_workers(kj), 2u);

    BrassModule mod = brass_module_create(ctx, "c_api_kernel_mod");
    REQUIRE(mod != nullptr);

    BrassType ptr_t = brass_type_ptr();
    BrassType i64_t = brass_type_i64();
    BrassType params[5] = { ptr_t, ptr_t, ptr_t, ptr_t, i64_t };

    BrassFunction fn = brass_function_create(mod, "c_fma_kernel", brass_type_void(), params, 5);
    REQUIRE(fn != nullptr);
    // Without the promise that the four arrays are distinct, the store to
    // `out` could feed a later iteration's loads and the loop is not DOALL.
    for (size_t p = 0; p < 4; ++p) CHECK_EQ(brass_function_set_param_noalias(fn, p, 1), BRASS_OK);
    CHECK_EQ(brass_function_set_param_noalias(fn, 4, 1), BRASS_ERR_INVALID_ARGUMENT);

    BrassBlock entry = brass_function_append_block(fn, "entry");
    BrassValue in_a = brass_block_add_param(entry, ptr_t);
    BrassValue in_b = brass_block_add_param(entry, ptr_t);
    BrassValue in_c = brass_block_add_param(entry, ptr_t);
    BrassValue in_out = brass_block_add_param(entry, ptr_t);
    BrassValue n_val = brass_block_add_param(entry, i64_t);

    BrassBlock hdr = brass_function_append_block(fn, "hdr");
    BrassValue iv = brass_block_add_param(hdr, i64_t);

    BrassBlock body = brass_function_append_block(fn, "body");
    BrassBlock exit = brass_function_append_block(fn, "exit");

    BrassBuilder b = brass_builder_create(ctx, fn);
    REQUIRE(b != nullptr);

    // entry:
    brass_builder_position_at_end(b, entry);
    BrassValue zero = brass_build_iconst_i64(b, 0);
    BrassValue eight = brass_build_iconst_i64(b, 8);
    BrassValue four = brass_build_iconst_i64(b, 4);
    brass_build_br(b, hdr, &zero, 1);

    // hdr:
    brass_builder_position_at_end(b, hdr);
    BrassValue cond = brass_build_cmp(b, BRASS_CMP_SLT, iv, n_val);
    brass_build_br_if(b, cond, body, nullptr, 0, exit, nullptr, 0);

    // body:
    brass_builder_position_at_end(b, body);
    BrassValue byte_off = brass_build_mul(b, iv, four);
    BrassValue ptr_a = brass_build_add(b, in_a, byte_off);
    BrassValue ptr_b = brass_build_add(b, in_b, byte_off);
    BrassValue ptr_c = brass_build_add(b, in_c, byte_off);
    BrassValue ptr_out = brass_build_add(b, in_out, byte_off);

    BrassType v256_f32 = brass_type_v256(BRASS_LANE_F32);
    BrassValue va = brass_build_vload(b, v256_f32, ptr_a, 0);
    BrassValue vb = brass_build_vload(b, v256_f32, ptr_b, 0);
    BrassValue vc = brass_build_vload(b, v256_f32, ptr_c, 0);
    BrassValue vres = brass_build_vfma(b, va, vb, vc);
    brass_build_vstore(b, v256_f32, ptr_out, 0, vres);

    BrassValue next_iv = brass_build_add(b, iv, eight);
    brass_build_br(b, hdr, &next_iv, 1);

    // exit:
    brass_builder_position_at_end(b, exit);
    brass_build_ret(b, nullptr);

    // Test C-API Polyhedral Loop Analysis
    BrassLoopAnalysis loop_analyses[4];
    size_t loop_count = 0;
    BrassStatus status = brass_kernel_jit_analyze_loops(kj, fn, loop_analyses, 4, &loop_count);
    CHECK_EQ(status, BRASS_OK);
    CHECK_EQ(loop_count, 1u);
    CHECK(loop_analyses[0].is_parallelizable);
    CHECK(loop_analyses[0].is_doall);

    // Compile Kernel via C-API
    BrassKernelFunction kfn = brass_kernel_jit_compile(kj, mod, "c_fma_kernel");
    REQUIRE(kfn != nullptr);

    void* addr = brass_kernel_function_get_address(kfn);
    REQUIRE(addr != nullptr);

    typedef void (*KernelPtr)(const float*, const float*, const float*, float*, int64_t);
    KernelPtr c_kernel = reinterpret_cast<KernelPtr>(addr);

    const int64_t N = 256;
    std::vector<float> a_arr(static_cast<size_t>(N));
    std::vector<float> b_arr(static_cast<size_t>(N));
    std::vector<float> c_arr(static_cast<size_t>(N));
    std::vector<float> out_arr(static_cast<size_t>(N), 0.0f);

    for (int64_t i = 0; i < N; ++i) {
        a_arr[static_cast<size_t>(i)] = static_cast<float>(i + 1);
        b_arr[static_cast<size_t>(i)] = 3.0f;
        c_arr[static_cast<size_t>(i)] = 10.0f;
    }

    c_kernel(a_arr.data(), b_arr.data(), c_arr.data(), out_arr.data(), N);

    for (int64_t i = 0; i < N; ++i) {
        float expected = a_arr[static_cast<size_t>(i)] * b_arr[static_cast<size_t>(i)] + c_arr[static_cast<size_t>(i)];
        CHECK_NEAR(out_arr[static_cast<size_t>(i)], expected, 1e-4f);
    }

    // Clean up
    brass_builder_destroy(b);
    brass_kernel_function_destroy(kfn);
    brass_module_destroy(mod);
    brass_kernel_jit_destroy(kj);
    brass_kernel_options_destroy(opts);
    brass_context_destroy(ctx);
}
