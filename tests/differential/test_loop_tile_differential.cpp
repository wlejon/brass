#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_tile.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <brass/mir/printer.hpp>

using namespace brass;

namespace {

std::unique_ptr<Module> build_diff_2d_scale(std::string_view mod_name, std::string_view fn_name) {
    auto mod = std::make_unique<Module>(mod_name);
    Function* fn = mod->create_function(fn_name, Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::i64(), Type::i64(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i64());
    Value* M = b.add_block_param(entry, Type::i64());
    Value* factor = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_i_hdr = b.create_block("loop_i_hdr");
    BasicBlock* loop_i_body = b.create_block("loop_i_body");
    BasicBlock* loop_j_hdr = b.create_block("loop_j_hdr");
    BasicBlock* loop_j_body = b.create_block("loop_j_body");
    BasicBlock* loop_i_next = b.create_block("loop_i_next");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_i_hdr, {zero});

    fn->append_block(loop_i_hdr);
    b.position_at_end(loop_i_hdr);
    Value* i = b.add_block_param(loop_i_hdr, Type::i64());
    Value* cond_i = b.build_slt(i, N);
    b.build_br_if(cond_i, loop_i_body, {}, exit_bb, {});

    fn->append_block(loop_i_body);
    b.position_at_end(loop_i_body);
    Value* row = b.build_mul(i, M);
    b.build_br(loop_j_hdr, {zero});

    fn->append_block(loop_j_hdr);
    b.position_at_end(loop_j_hdr);
    Value* j = b.add_block_param(loop_j_hdr, Type::i64());
    Value* cond_j = b.build_slt(j, M);
    b.build_br_if(cond_j, loop_j_body, {}, loop_i_next, {});

    fn->append_block(loop_j_body);
    b.position_at_end(loop_j_body);
    Value* idx = b.build_add(row, j);
    Value* val = b.build_load_indexed(Type::i64(), in_ptr, idx, 8, 0);
    Value* res = b.build_mul(val, factor);
    b.build_store_indexed(Type::i64(), out_ptr, idx, 8, 0, res);
    Value* next_j = b.build_add(j, one);
    b.build_br(loop_j_hdr, {next_j});

    fn->append_block(loop_i_next);
    b.position_at_end(loop_i_next);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_i_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_diff_2d_transpose(std::string_view mod_name, std::string_view fn_name) {
    auto mod = std::make_unique<Module>(mod_name);
    Function* fn = mod->create_function(fn_name, Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_i_hdr = b.create_block("loop_i_hdr");
    BasicBlock* loop_i_body = b.create_block("loop_i_body");
    BasicBlock* loop_j_hdr = b.create_block("loop_j_hdr");
    BasicBlock* loop_j_body = b.create_block("loop_j_body");
    BasicBlock* loop_i_next = b.create_block("loop_i_next");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_i_hdr, {zero});

    fn->append_block(loop_i_hdr);
    b.position_at_end(loop_i_hdr);
    Value* i = b.add_block_param(loop_i_hdr, Type::i64());
    Value* cond_i = b.build_slt(i, N);
    b.build_br_if(cond_i, loop_i_body, {}, exit_bb, {});

    fn->append_block(loop_i_body);
    b.position_at_end(loop_i_body);
    Value* row_i = b.build_mul(i, N);
    b.build_br(loop_j_hdr, {zero});

    fn->append_block(loop_j_hdr);
    b.position_at_end(loop_j_hdr);
    Value* j = b.add_block_param(loop_j_hdr, Type::i64());
    Value* cond_j = b.build_slt(j, N);
    b.build_br_if(cond_j, loop_j_body, {}, loop_i_next, {});

    fn->append_block(loop_j_body);
    b.position_at_end(loop_j_body);
    Value* src_idx = b.build_add(row_i, j);
    Value* val = b.build_load_indexed(Type::i64(), in_ptr, src_idx, 8, 0);

    Value* row_j = b.build_mul(j, N);
    Value* dst_idx = b.build_add(row_j, i);
    b.build_store_indexed(Type::i64(), out_ptr, dst_idx, 8, 0, val);

    Value* next_j = b.build_add(j, one);
    b.build_br(loop_j_hdr, {next_j});

    fn->append_block(loop_i_next);
    b.position_at_end(loop_i_next);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_i_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    return mod;
}

std::unique_ptr<Module> build_diff_3d_matmul(std::string_view mod_name, std::string_view fn_name) {
    auto mod = std::make_unique<Module>(mod_name);
    Function* fn = mod->create_function(fn_name, Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* A = b.add_block_param(entry, Type::ptr());
    Value* B = b.add_block_param(entry, Type::ptr());
    Value* C = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_i_hdr = b.create_block("loop_i_hdr");
    BasicBlock* loop_i_body = b.create_block("loop_i_body");
    BasicBlock* loop_j_hdr = b.create_block("loop_j_hdr");
    BasicBlock* loop_j_body = b.create_block("loop_j_body");
    BasicBlock* loop_k_hdr = b.create_block("loop_k_hdr");
    BasicBlock* loop_k_body = b.create_block("loop_k_body");
    BasicBlock* loop_j_next = b.create_block("loop_j_next");
    BasicBlock* loop_i_next = b.create_block("loop_i_next");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_i_hdr, {zero});

    fn->append_block(loop_i_hdr);
    b.position_at_end(loop_i_hdr);
    Value* i = b.add_block_param(loop_i_hdr, Type::i64());
    Value* cond_i = b.build_slt(i, N);
    b.build_br_if(cond_i, loop_i_body, {}, exit_bb, {});

    fn->append_block(loop_i_body);
    b.position_at_end(loop_i_body);
    Value* row_a = b.build_mul(i, N);
    b.build_br(loop_j_hdr, {zero});

    fn->append_block(loop_j_hdr);
    b.position_at_end(loop_j_hdr);
    Value* j = b.add_block_param(loop_j_hdr, Type::i64());
    Value* cond_j = b.build_slt(j, N);
    b.build_br_if(cond_j, loop_j_body, {}, loop_i_next, {});

    fn->append_block(loop_j_body);
    b.position_at_end(loop_j_body);
    b.build_br(loop_k_hdr, {zero, zero});

    fn->append_block(loop_k_hdr);
    b.position_at_end(loop_k_hdr);
    Value* k = b.add_block_param(loop_k_hdr, Type::i64());
    Value* sum = b.add_block_param(loop_k_hdr, Type::i64());
    Value* cond_k = b.build_slt(k, N);
    b.build_br_if(cond_k, loop_k_body, {}, loop_j_next, {sum});

    fn->append_block(loop_k_body);
    b.position_at_end(loop_k_body);
    Value* idx_a = b.build_add(row_a, k);
    Value* val_a = b.build_load_indexed(Type::i64(), A, idx_a, 8, 0);

    Value* row_b = b.build_mul(k, N);
    Value* idx_b = b.build_add(row_b, j);
    Value* val_b = b.build_load_indexed(Type::i64(), B, idx_b, 8, 0);

    Value* term = b.build_mul(val_a, val_b);
    Value* next_sum = b.build_add(sum, term);
    Value* next_k = b.build_add(k, one);
    b.build_br(loop_k_hdr, {next_k, next_sum});

    fn->append_block(loop_j_next);
    b.position_at_end(loop_j_next);
    Value* final_sum = b.add_block_param(loop_j_next, Type::i64());
    Value* idx_c = b.build_add(row_a, j);
    b.build_store_indexed(Type::i64(), C, idx_c, 8, 0, final_sum);
    Value* next_j = b.build_add(j, one);
    b.build_br(loop_j_hdr, {next_j});

    fn->append_block(loop_i_next);
    b.position_at_end(loop_i_next);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_i_hdr, {next_i});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

TEST_CASE("Differential - 2D Matrix Scale Tiling Across Diverse Trip Counts") {
    std::mt19937_64 rng(42);

    for (int64_t size_n : {1, 2, 3, 7, 13, 16, 23, 32, 35}) {
        int64_t N = size_n;
        int64_t M = size_n + 2;
        size_t total = static_cast<size_t>(N * M);

        std::vector<int64_t> in_arr(total);
        std::vector<int64_t> out_interp(total, 0);
        std::vector<int64_t> out_jit(total, 0);

        for (size_t idx = 0; idx < total; ++idx) {
            in_arr[idx] = static_cast<int64_t>((rng() % 200) - 100);
        }

        int64_t factor = 7;

        // 1. Reference Interpreter execution on un-tiled module
        auto ref_mod = build_diff_2d_scale("scale_ref", "mat_scale");
        Interpreter interp;
        interp.run(*ref_mod->get_function("mat_scale"), {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_arr.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_interp.data())),
            RuntimeValue::from_i64(N),
            RuntimeValue::from_i64(M),
            RuntimeValue::from_i64(factor)
        });

        // 2. Native JIT execution on tiled module
        auto opt_mod = build_diff_2d_scale("scale_opt", "mat_scale");
        Function* opt_fn = opt_mod->get_function("mat_scale");
        DominatorTree dom(*opt_fn);
        LoopTileOptions opts;
        opts.tile_size_i = 8;
        opts.tile_size_j = 8;
        bool changed = loop_tile_pass(*opt_fn, dom, opts);
        CHECK(changed);

        DiagnosticReporter diag;
        REQUIRE(verify_function(*opt_fn, &diag));

        std::vector<int64_t> out_tiled_interp(total, 0);
        Interpreter tiled_interp;
        tiled_interp.run(*opt_fn, {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_arr.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_tiled_interp.data())),
            RuntimeValue::from_i64(N),
            RuntimeValue::from_i64(M),
            RuntimeValue::from_i64(factor)
        });
        if (out_interp != out_tiled_interp) {
            std::cerr << "TILED INTERP MISMATCH for N=" << N << ", M=" << M << ":\n";
            std::cerr << "interp: ";
            for (size_t k = 0; k < std::min<size_t>(out_interp.size(), 10); ++k) std::cerr << out_interp[k] << " ";
            std::cerr << "\ntiled:  ";
            for (size_t k = 0; k < std::min<size_t>(out_tiled_interp.size(), 10); ++k) std::cerr << out_tiled_interp[k] << " ";
            std::cerr << "\n";
        }
        CHECK(out_interp == out_tiled_interp);

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*opt_mod));

        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t, int64_t, int64_t)>("mat_scale");
        REQUIRE(fn_ptr != nullptr);
        fn_ptr(in_arr.data(), out_jit.data(), N, M, factor);

        // 3. Assert exact match
        if (out_interp != out_jit) {
            std::cerr << "DIFF MISMATCH for N=" << N << ", M=" << M << ":\n";
            std::cerr << "interp: ";
            for (size_t k = 0; k < std::min<size_t>(out_interp.size(), 10); ++k) std::cerr << out_interp[k] << " ";
            std::cerr << "\njit:    ";
            for (size_t k = 0; k < std::min<size_t>(out_jit.size(), 10); ++k) std::cerr << out_jit[k] << " ";
            std::cerr << "\n";
        }
        CHECK(out_interp == out_jit);
    }
}

TEST_CASE("Differential - 2D Matrix Transpose Tiling Across Diverse Trip Counts") {
    std::mt19937_64 rng(101);

    for (int64_t size_n : {1, 2, 4, 7, 13, 16, 23, 32, 35}) {
        int64_t N = size_n;
        size_t total = static_cast<size_t>(N * N);

        std::vector<int64_t> in_mat(total);
        std::vector<int64_t> out_interp(total, 0);
        std::vector<int64_t> out_jit(total, 0);

        for (size_t idx = 0; idx < total; ++idx) {
            in_mat[idx] = static_cast<int64_t>((rng() % 500) - 250);
        }

        // 1. Reference Interpreter execution
        auto ref_mod = build_diff_2d_transpose("trans_ref", "mat_transpose");
        Interpreter interp;
        interp.run(*ref_mod->get_function("mat_transpose"), {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(in_mat.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(out_interp.data())),
            RuntimeValue::from_i64(N)
        });

        // 2. Native JIT execution on tiled module
        auto opt_mod = build_diff_2d_transpose("trans_opt", "mat_transpose");
        Function* opt_fn = opt_mod->get_function("mat_transpose");
        DominatorTree dom(*opt_fn);
        LoopTileOptions opts;
        opts.tile_size_i = 8;
        opts.tile_size_j = 8;
        bool changed = loop_tile_pass(*opt_fn, dom, opts);
        CHECK(changed);

        DiagnosticReporter diag;
        REQUIRE(verify_function(*opt_fn, &diag));

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*opt_mod));

        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t)>("mat_transpose");
        REQUIRE(fn_ptr != nullptr);
        fn_ptr(in_mat.data(), out_jit.data(), N);

        // 3. Assert exact match
        CHECK(out_interp == out_jit);
    }
}

TEST_CASE("Differential - 3D Matrix Multiplication Tiling Across Diverse Trip Counts") {
    std::mt19937_64 rng(303);

    for (int64_t size_n : {1, 2, 3, 5, 8, 13, 16, 23, 32}) {
        int64_t N = size_n;
        size_t total = static_cast<size_t>(N * N);

        std::vector<int64_t> A(total);
        std::vector<int64_t> B(total);
        std::vector<int64_t> C_interp(total, 0);
        std::vector<int64_t> C_jit(total, 0);

        for (size_t idx = 0; idx < total; ++idx) {
            A[idx] = static_cast<int64_t>((rng() % 20) - 10);
            B[idx] = static_cast<int64_t>((rng() % 20) - 10);
        }

        // 1. Reference Interpreter execution
        auto ref_mod = build_diff_3d_matmul("matmul_ref", "matmul");
        Interpreter interp;
        interp.run(*ref_mod->get_function("matmul"), {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(A.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(B.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(C_interp.data())),
            RuntimeValue::from_i64(N)
        });

        // 2. Native JIT execution on tiled module
        auto opt_mod = build_diff_3d_matmul("matmul_opt", "matmul");
        Function* opt_fn = opt_mod->get_function("matmul");
        DominatorTree dom(*opt_fn);
        LoopTileOptions opts;
        opts.tile_size_i = 8;
        opts.tile_size_j = 8;
        opts.tile_size_k = 8;
        opts.enable_loop_interchange = false; // Standard 3D tiling preserves exact order
        bool changed = loop_tile_pass(*opt_fn, dom, opts);
        CHECK(changed);

        DiagnosticReporter diag;
        REQUIRE(verify_function(*opt_fn, &diag));

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*opt_mod));

        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t*, int64_t)>("matmul");
        REQUIRE(fn_ptr != nullptr);
        fn_ptr(A.data(), B.data(), C_jit.data(), N);

        // 3. Assert exact match
        CHECK(C_interp == C_jit);
    }
}
