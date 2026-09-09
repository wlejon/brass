#include "bench_simd_math.hpp"
#include "bench_utils.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <immintrin.h>

using namespace brass;
using namespace brass::bench;
using namespace brass::codegen;

namespace {

// ============================================================================
// Native Scalar Baselines (prevent auto-vectorization)
// ============================================================================

#if defined(__GNUC__) || defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
float native_dot4_scalar(const float* a, const float* b, int64_t count) {
    float sum = 0.0f;
    for (int64_t i = 0; i < count; ++i) {
        int64_t idx = i * 4;
        sum += a[idx + 0] * b[idx + 0]
             + a[idx + 1] * b[idx + 1]
             + a[idx + 2] * b[idx + 2]
             + a[idx + 3] * b[idx + 3];
    }
    return sum;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
void native_vec3_math_scalar(const float* a, const float* b, float* out, int64_t count, float half_val) {
    for (int64_t i = 0; i < count; ++i) {
        int64_t idx = i * 4;
        float x = a[idx + 0];
        float y = a[idx + 1];
        float z = a[idx + 2];
        float len_sq = x * x + y * y + z * z;
        float inv_len = (len_sq > 0.0f) ? (1.0f / std::sqrt(len_sq)) : 0.0f;
        out[idx + 0] = x * inv_len * half_val + b[idx + 0] * half_val;
        out[idx + 1] = y * inv_len * half_val + b[idx + 1] * half_val;
        out[idx + 2] = z * inv_len * half_val + b[idx + 2] * half_val;
        out[idx + 3] = 0.0f;
    }
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
void native_matmul4x4_scalar(const float* a, const float* b, float* c, int64_t count) {
    for (int64_t m = 0; m < count; ++m) {
        const float* ma = a + m * 16;
        const float* mb = b + m * 16;
        float* mc = c + m * 16;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                float s = 0.0f;
                for (int k = 0; k < 4; ++k) {
                    s += ma[i * 4 + k] * mb[k * 4 + j];
                }
                mc[i * 4 + j] = s;
            }
        }
    }
}

// ============================================================================
// Brass MIR SIMD Modules
// ============================================================================

std::unique_ptr<Module> build_simd_dot4_module() {
    auto mod = std::make_unique<Module>("bench_simd_dot4");
    Function* fn = mod->create_function("simd_dot4", Type::f32(), {Type::ptr(), Type::ptr(), Type::i64()});
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ptr_a = b.add_block_param(entry, Type::ptr());
    Value* ptr_b = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero_idx = b.build_iconst_i64(0);
    Value* zero_acc0 = b.build_vzero(Type::f32x4());
    Value* zero_acc1 = b.build_vzero(Type::f32x4());
    Value* zero_acc2 = b.build_vzero(Type::f32x4());
    Value* zero_acc3 = b.build_vzero(Type::f32x4());
    b.build_br(loop_hdr, {zero_idx, zero_acc0, zero_acc1, zero_acc2, zero_acc3});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc0 = b.add_block_param(loop_hdr, Type::f32x4());
    Value* acc1 = b.add_block_param(loop_hdr, Type::f32x4());
    Value* acc2 = b.add_block_param(loop_hdr, Type::f32x4());
    Value* acc3 = b.add_block_param(loop_hdr, Type::f32x4());
    Value* cond = b.build_slt(i, count);
    b.build_br_if(cond, loop_body, exit_bb);

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* four = b.build_iconst_i64(4);
    Value* offset = b.build_shl(i, four);
    Value* pa = b.build_add(ptr_a, offset);
    Value* pb = b.build_add(ptr_b, offset);

    Value* va0 = b.build_vload(Type::f32x4(), pa, 0);
    Value* vb0 = b.build_vload(Type::f32x4(), pb, 0);
    Value* p0 = b.build_vmul(va0, vb0);
    Value* next_acc0 = b.build_vadd(acc0, p0);

    Value* va1 = b.build_vload(Type::f32x4(), pa, 16);
    Value* vb1 = b.build_vload(Type::f32x4(), pb, 16);
    Value* p1 = b.build_vmul(va1, vb1);
    Value* next_acc1 = b.build_vadd(acc1, p1);

    Value* va2 = b.build_vload(Type::f32x4(), pa, 32);
    Value* vb2 = b.build_vload(Type::f32x4(), pb, 32);
    Value* p2 = b.build_vmul(va2, vb2);
    Value* next_acc2 = b.build_vadd(acc2, p2);

    Value* va3 = b.build_vload(Type::f32x4(), pa, 48);
    Value* vb3 = b.build_vload(Type::f32x4(), pb, 48);
    Value* p3 = b.build_vmul(va3, vb3);
    Value* next_acc3 = b.build_vadd(acc3, p3);

    Value* step = b.build_iconst_i64(4);
    Value* next_i = b.build_add(i, step);
    b.build_br(loop_hdr, {next_i, next_acc0, next_acc1, next_acc2, next_acc3});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* s01_v = b.build_vadd(acc0, acc1);
    Value* s23_v = b.build_vadd(acc2, acc3);
    Value* acc = b.build_vadd(s01_v, s23_v);
    Value* l0 = b.build_vextract_lane(acc, 0);
    Value* l1 = b.build_vextract_lane(acc, 1);
    Value* l2 = b.build_vextract_lane(acc, 2);
    Value* l3 = b.build_vextract_lane(acc, 3);
    Value* s01 = b.build_add(l0, l1);
    Value* s23 = b.build_add(l2, l3);
    Value* sum = b.build_add(s01, s23);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_simd_vec3_math_module() {
    auto mod = std::make_unique<Module>("bench_simd_vec3_math");
    Function* fn = mod->create_function("simd_vec3_math", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64(), Type::ptr()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ptr_a = b.add_block_param(entry, Type::ptr());
    Value* ptr_b = b.add_block_param(entry, Type::ptr());
    Value* ptr_out = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());
    Value* half_ptr = b.add_block_param(entry, Type::ptr());
    Value* half_vec = b.build_vload(Type::f32x4(), half_ptr, 0);

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero_idx = b.build_iconst_i64(0);
    b.build_br(loop_hdr, {zero_idx});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, count);
    b.build_br_if(cond, loop_body, exit_bb);

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* four = b.build_iconst_i64(4);
    Value* offset = b.build_shl(i, four);
    Value* pa = b.build_add(ptr_a, offset);
    Value* pb = b.build_add(ptr_b, offset);
    Value* po = b.build_add(ptr_out, offset);

    // Vector 0
    Value* va0 = b.build_vload(Type::f32x4(), pa, 0);
    Value* vb0 = b.build_vload(Type::f32x4(), pb, 0);
    Value* sq0 = b.build_vmul(va0, va0);
    Value* x2_0 = b.build_vextract_lane(sq0, 0);
    Value* y2_0 = b.build_vextract_lane(sq0, 1);
    Value* z2_0 = b.build_vextract_lane(sq0, 2);
    Value* xy2_0 = b.build_add(x2_0, y2_0);
    Value* len_sq0 = b.build_add(xy2_0, z2_0);
    Value* v_len_sq0 = b.build_vbroadcast(Type::f32x4(), len_sq0);
    Value* vlen0 = b.build_vsqrt(v_len_sq0);
    Value* norm0 = b.build_vdiv(va0, vlen0);
    Value* sum_v0 = b.build_vadd(norm0, vb0);
    Value* res0 = b.build_vmul(sum_v0, half_vec);
    b.build_vstore(Type::f32x4(), po, 0, res0);

    // Vector 1
    Value* va1 = b.build_vload(Type::f32x4(), pa, 16);
    Value* vb1 = b.build_vload(Type::f32x4(), pb, 16);
    Value* sq1 = b.build_vmul(va1, va1);
    Value* x2_1 = b.build_vextract_lane(sq1, 0);
    Value* y2_1 = b.build_vextract_lane(sq1, 1);
    Value* z2_1 = b.build_vextract_lane(sq1, 2);
    Value* xy2_1 = b.build_add(x2_1, y2_1);
    Value* len_sq1 = b.build_add(xy2_1, z2_1);
    Value* v_len_sq1 = b.build_vbroadcast(Type::f32x4(), len_sq1);
    Value* vlen1 = b.build_vsqrt(v_len_sq1);
    Value* norm1 = b.build_vdiv(va1, vlen1);
    Value* sum_v1 = b.build_vadd(norm1, vb1);
    Value* res1 = b.build_vmul(sum_v1, half_vec);
    b.build_vstore(Type::f32x4(), po, 16, res1);

    Value* two = b.build_iconst_i64(2);
    Value* next_i = b.build_add(i, two);
    b.build_br(loop_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_simd_matmul4x4_module() {
    auto mod = std::make_unique<Module>("bench_simd_matmul4x4");
    Function* fn = mod->create_function("simd_matmul4x4", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ptr_a = b.add_block_param(entry, Type::ptr());
    Value* ptr_b = b.add_block_param(entry, Type::ptr());
    Value* ptr_c = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero_idx = b.build_iconst_i64(0);
    b.build_br(loop_hdr, {zero_idx});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* m = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(m, count);
    b.build_br_if(cond, loop_body, exit_bb);

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* six = b.build_iconst_i64(6);
    Value* offset = b.build_shl(m, six); // m * 64 bytes
    Value* pa = b.build_add(ptr_a, offset);
    Value* pb = b.build_add(ptr_b, offset);
    Value* pc = b.build_add(ptr_c, offset);

    Value* b0 = b.build_vload(Type::f32x4(), pb, 0);
    Value* b1 = b.build_vload(Type::f32x4(), pb, 16);
    Value* b2 = b.build_vload(Type::f32x4(), pb, 32);
    Value* b3 = b.build_vload(Type::f32x4(), pb, 48);

    for (int row = 0; row < 4; ++row) {
        Value* a_row = b.build_vload(Type::f32x4(), pa, row * 16);
        Value* a0 = b.build_vshuffle(a_row, a_row, 0x00);
        Value* a1 = b.build_vshuffle(a_row, a_row, 0x55);
        Value* a2 = b.build_vshuffle(a_row, a_row, 0xAA);
        Value* a3 = b.build_vshuffle(a_row, a_row, 0xFF);
        Value* p0 = b.build_vmul(a0, b0);
        Value* p1 = b.build_vmul(a1, b1);
        Value* p2 = b.build_vmul(a2, b2);
        Value* p3 = b.build_vmul(a3, b3);
        Value* s01 = b.build_vadd(p0, p1);
        Value* s23 = b.build_vadd(p2, p3);
        Value* c_row = b.build_vadd(s01, s23);
        b.build_vstore(Type::f32x4(), pc, row * 16, c_row);
    }

    Value* one = b.build_iconst_i64(1);
    Value* next_m = b.build_add(m, one);
    b.build_br(loop_hdr, {next_m});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    return mod;
}

} // anonymous namespace

namespace brass::bench {

void run_simd_math_benchmarks(std::vector<BenchmarkResult>& results, const RatchetManager& ratchet) {
    // 1. 4D Vector Dot Product Benchmark
    {
        size_t iters = 500;
        int64_t count = 4096;
        std::vector<float> a_data(count * 4, 1.5f);
        std::vector<float> b_data(count * 4, 2.0f);

        auto run_scalar = [a = a_data.data(), b = b_data.data(), count, iters]() {
            float total = 0.0f;
            for (size_t iter = 0; iter < iters; ++iter) {
                total += native_dot4_scalar(a, b, count);
                DoNotOptimize(total);
            }
        };

        auto mod = build_simd_dot4_module();
        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->compile_and_load(*mod, padding);
            auto dot4_fn = jit->get_function_ptr<float(*)(const float*, const float*, int64_t)>("simd_dot4");
            if (!dot4_fn) {
                std::cerr << "FATAL: simd_dot4 function pointer is null!\n";
                std::abort();
            }
            return [jit, dot4_fn, a = a_data.data(), b = b_data.data(), count, iters]() {
                float total = 0.0f;
                for (size_t iter = 0; iter < iters; ++iter) {
                    total += dot4_fn(a, b, count);
                    DoNotOptimize(total);
                }
            };
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_scalar, make_jit_runner);
        results.push_back(make_paired_result("simd_dot4", "SIMD 4D Dot Product (f32x4 vs scalar)", iters, paired, ratchet.get_ratio("simd_dot4", 0.50)));
        BenchmarkReporter::print_row(results.back());
    }

    // 2. 3D Vector Math (Normalize + Lerp) Benchmark
    {
        size_t iters = 200;
        int64_t count = 2048;
        std::vector<float> a_data(count * 4, 2.0f);
        std::vector<float> b_data(count * 4, 4.0f);
        std::vector<float> out_scalar(count * 4, 0.0f);
        std::vector<float> out_jit(count * 4, 0.0f);
        float half_val = 0.5f;

        auto run_scalar = [a = a_data.data(), b = b_data.data(), out = out_scalar.data(), count, half_val, iters]() {
            for (size_t iter = 0; iter < iters; ++iter) {
                native_vec3_math_scalar(a, b, out, count, half_val);
                DoNotOptimize(out);
            }
        };

        auto mod = build_simd_vec3_math_module();
        alignas(16) float half_arr[4] = {0.5f, 0.5f, 0.5f, 0.0f};

        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->compile_and_load(*mod, padding);
            auto vec3_fn = jit->get_function_ptr<void(*)(const float*, const float*, float*, int64_t, const float*)>("simd_vec3_math");
            if (!vec3_fn) {
                std::cerr << "FATAL: simd_vec3_math function pointer is null!\n";
                std::abort();
            }
            return [jit, vec3_fn, a = a_data.data(), b = b_data.data(), out = out_jit.data(), count, half = half_arr, iters]() {
                for (size_t iter = 0; iter < iters; ++iter) {
                    vec3_fn(a, b, out, count, half);
                    DoNotOptimize(out);
                }
            };
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_scalar, make_jit_runner);
        results.push_back(make_paired_result("simd_vec3_math", "SIMD 3D Vector Math (Normalize+Lerp)", iters, paired, ratchet.get_ratio("simd_vec3_math", 0.85)));
        BenchmarkReporter::print_row(results.back());
    }

    // 3. 4x4 Matrix Multiplication Benchmark
    {
        size_t iters = 500;
        int64_t count = 512;
        std::vector<float> a_data(count * 16, 1.25f);
        std::vector<float> b_data(count * 16, 2.5f);
        std::vector<float> c_scalar(count * 16, 0.0f);
        std::vector<float> c_jit(count * 16, 0.0f);

        auto run_scalar = [a = a_data.data(), b = b_data.data(), c = c_scalar.data(), count, iters]() {
            for (size_t iter = 0; iter < iters; ++iter) {
                native_matmul4x4_scalar(a, b, c, count);
                DoNotOptimize(c);
            }
        };

        auto mod = build_simd_matmul4x4_module();
        auto make_jit_runner = [&](size_t padding) {
            auto jit = std::make_shared<JitExecutionEngine>();
            jit->compile_and_load(*mod, padding);
            auto matmul4x4_fn = jit->get_function_ptr<void(*)(const float*, const float*, float*, int64_t)>("simd_matmul4x4");
            if (!matmul4x4_fn) {
                std::cerr << "FATAL: simd_matmul4x4 function pointer is null!\n";
                std::abort();
            }
            return [jit, matmul4x4_fn, a = a_data.data(), b = b_data.data(), c = c_jit.data(), count, iters]() {
                for (size_t iter = 0; iter < iters; ++iter) {
                    matmul4x4_fn(a, b, c, count);
                    DoNotOptimize(c);
                }
            };
        };

        auto paired = measure_paired_multi_placement(DEFAULT_BENCH_REPETITIONS, run_scalar, make_jit_runner);
        results.push_back(make_paired_result("simd_matmul4x4", "SIMD 4x4 MatMul (f32x4 vs scalar loop)", iters, paired, ratchet.get_ratio("simd_matmul4x4", 0.50)));
        BenchmarkReporter::print_row(results.back());
    }
}

} // namespace brass::bench
