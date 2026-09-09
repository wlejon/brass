#include "test_framework.hpp"
#include <brass/mir/loop_nest.hpp>
#include <brass/mir/loop_tile.hpp>
#include <brass/mir/loop_vectorize.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <vector>
#include <cmath>

using namespace brass;

namespace {

std::unique_ptr<Module> build_2d_scale_module(std::string_view name) {
    auto mod = std::make_unique<Module>(name);
    Function* fn = mod->create_function("mat_scale", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::i64(), Type::i64(), Type::i64()
    });
    Builder b(*mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* A = b.add_block_param(entry, Type::ptr());
    Value* B = b.add_block_param(entry, Type::ptr());
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
    Value* row_off = b.build_mul(i, M);
    b.build_br(loop_j_hdr, {zero});

    fn->append_block(loop_j_hdr);
    b.position_at_end(loop_j_hdr);
    Value* j = b.add_block_param(loop_j_hdr, Type::i64());
    Value* cond_j = b.build_slt(j, M);
    b.build_br_if(cond_j, loop_j_body, {}, loop_i_next, {});

    fn->append_block(loop_j_body);
    b.position_at_end(loop_j_body);
    Value* idx = b.build_add(row_off, j);
    Value* val = b.build_load_indexed(Type::i64(), A, idx, 8, 0);
    Value* scaled = b.build_mul(val, factor);
    b.build_store_indexed(Type::i64(), B, idx, 8, 0, scaled);
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

std::unique_ptr<Module> build_2d_transpose_module(std::string_view name) {
    auto mod = std::make_unique<Module>(name);
    Function* fn = mod->create_function("mat_transpose", Type::void_type(), {
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

std::unique_ptr<Module> build_3d_matmul_module(std::string_view name) {
    auto mod = std::make_unique<Module>(name);
    Function* fn = mod->create_function("matmul", Type::void_type(), {
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

TEST_CASE("Loop Nest - 2D Loop Nest Detection & Legality Check") {
    auto mod = build_2d_scale_module("test_nest_detection");
    Function* fn = mod->get_function("mat_scale");
    REQUIRE(fn != nullptr);

    DominatorTree dom(*fn);
    LoopNestAnalysis nest_analysis(*fn, dom);

    CHECK_EQ(nest_analysis.nests().size(), 1);
    const LoopNest& nest = *nest_analysis.nests()[0];

    CHECK_EQ(nest.depth(), 2);
    CHECK(nest.is_tileable());
    CHECK(nest.is_interchange_legal(0, 1));
}

TEST_CASE("Loop Tile - 2D Matrix Copy, Transpose, and Scale") {
    // 1. Test 2D Scale
    {
        auto mod = build_2d_scale_module("test_scale_tiling");
        Function* fn = mod->get_function("mat_scale");
        REQUIRE(fn != nullptr);

        DominatorTree dom(*fn);
        LoopTileOptions opts;
        opts.tile_size_i = 8;
        opts.tile_size_j = 8;
        bool changed = loop_tile_pass(*fn, dom, opts);
        CHECK(changed);

        DiagnosticReporter diag;
        REQUIRE(verify_function(*fn, &diag));

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*mod));

        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t, int64_t, int64_t)>("mat_scale");
        REQUIRE(fn_ptr != nullptr);

        const int64_t N = 16;
        const int64_t M = 16;
        std::vector<int64_t> A(N * M, 3);
        std::vector<int64_t> B(N * M, 0);

        fn_ptr(A.data(), B.data(), N, M, 5);
        for (int64_t val : B) {
            CHECK_EQ(val, 15);
        }
    }

    // 2. Test 2D Transpose
    {
        auto mod = build_2d_transpose_module("test_transpose_tiling");
        Function* fn = mod->get_function("mat_transpose");
        REQUIRE(fn != nullptr);

        DominatorTree dom(*fn);
        LoopTileOptions opts;
        opts.tile_size_i = 8;
        opts.tile_size_j = 8;
        bool changed = loop_tile_pass(*fn, dom, opts);
        CHECK(changed);

        DiagnosticReporter diag;
        REQUIRE(verify_function(*fn, &diag));

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*mod));

        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t)>("mat_transpose");
        REQUIRE(fn_ptr != nullptr);

        const int64_t N = 16;
        std::vector<int64_t> in_mat(N * N);
        std::vector<int64_t> out_mat(N * N, 0);
        for (int64_t r = 0; r < N; ++r) {
            for (int64_t c = 0; c < N; ++c) {
                in_mat[static_cast<size_t>(r * N + c)] = r * 100 + c;
            }
        }

        fn_ptr(in_mat.data(), out_mat.data(), N);

        for (int64_t r = 0; r < N; ++r) {
            for (int64_t c = 0; c < N; ++c) {
                int64_t expected = c * 100 + r;
                CHECK_EQ(out_mat[static_cast<size_t>(r * N + c)], expected);
            }
        }
    }
}

TEST_CASE("Loop Tile - 3D Matrix Multiplication Loop Nest Tiling (i, j, k)") {
    // 1. Standard 3D Tiling (enable_loop_interchange = false)
    {
        auto mod = build_3d_matmul_module("test_matmul_standard");
        Function* fn = mod->get_function("matmul");
        REQUIRE(fn != nullptr);

        DominatorTree dom(*fn);
        LoopTileOptions opts;
        opts.tile_size_i = 4;
        opts.tile_size_j = 4;
        opts.tile_size_k = 4;
        opts.enable_loop_interchange = false;

        bool changed = loop_tile_pass(*fn, dom, opts);
        CHECK(changed);

        DiagnosticReporter diag;
        bool v = verify_function(*fn, &diag);
        if (!v) {
            for (const auto& msg : diag.diagnostics()) {
                std::cerr << "VERIFY ERROR (3D): " << msg.message << "\n";
            }
        }
        REQUIRE(v);

        const int64_t N = 8;
        std::vector<int64_t> A(N * N, 2);
        std::vector<int64_t> B(N * N, 3);
        std::vector<int64_t> C_interp(N * N, 0);

        Interpreter interp;
        interp.run(*fn, {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(A.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(B.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(C_interp.data())),
            RuntimeValue::from_i64(N)
        });

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*mod));

        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t*, int64_t)>("matmul");
        REQUIRE(fn_ptr != nullptr);

        std::vector<int64_t> C(N * N, 0);
        fn_ptr(A.data(), B.data(), C.data(), N);

        int64_t expected = 2 * 3 * N;
        for (int64_t val : C) {
            CHECK_EQ(val, expected);
        }
        for (int64_t val : C_interp) {
            CHECK_EQ(val, expected);
        }
    }

    // 2. Cache-Friendly Interchanged 3D Tiling (enable_loop_interchange = true)
    {
        auto mod = build_3d_matmul_module("test_matmul_interchanged");
        Function* fn = mod->get_function("matmul");
        REQUIRE(fn != nullptr);

        DominatorTree dom(*fn);
        LoopTileOptions opts;
        opts.tile_size_i = 4;
        opts.tile_size_j = 4;
        opts.tile_size_k = 4;
        opts.enable_loop_interchange = true;

        bool changed = loop_tile_pass(*fn, dom, opts);
        CHECK(changed);

        DiagnosticReporter diag;
        bool v = verify_function(*fn, &diag);
        REQUIRE(v);

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*mod));

        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t*, int64_t)>("matmul");
        REQUIRE(fn_ptr != nullptr);

        const int64_t N = 8;
        std::vector<int64_t> A(N * N, 2);
        std::vector<int64_t> B(N * N, 3);
        std::vector<int64_t> C(N * N, 0);
        std::vector<int64_t> C_interp2(N * N, 0);

        Interpreter interp2;
        interp2.run(*fn, {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(A.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(B.data())),
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(C_interp2.data())),
            RuntimeValue::from_i64(N)
        });

        fn_ptr(A.data(), B.data(), C.data(), N);

        int64_t expected = 2 * 3 * N;
        for (int64_t val : C) {
            CHECK_EQ(val, expected);
        }
        for (int64_t val : C_interp2) {
            CHECK_EQ(val, expected);
        }
    }
}

TEST_CASE("Loop Tile - Non-Multiple Tile Bounds with Remainder Peeling (23x23, 35x35)") {
    for (int64_t size_n : {23, 35}) {
        auto mod = build_3d_matmul_module("test_matmul_nonmultiple_" + std::to_string(size_n));
        Function* fn = mod->get_function("matmul");
        REQUIRE(fn != nullptr);

        DominatorTree dom(*fn);
        LoopTileOptions opts;
        opts.tile_size_i = 16;
        opts.tile_size_j = 16;
        opts.tile_size_k = 16;
        opts.enable_loop_interchange = false;

        bool changed = loop_tile_pass(*fn, dom, opts);
        CHECK(changed);

        DiagnosticReporter diag;
        REQUIRE(verify_function(*fn, &diag));

        codegen::JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*mod));

        auto fn_ptr = jit.get_function_ptr<void(*)(int64_t*, int64_t*, int64_t*, int64_t)>("matmul");
        REQUIRE(fn_ptr != nullptr);

        std::vector<int64_t> A(static_cast<size_t>(size_n * size_n), 2);
        std::vector<int64_t> B(static_cast<size_t>(size_n * size_n), 3);
        std::vector<int64_t> C(static_cast<size_t>(size_n * size_n), 0);

        fn_ptr(A.data(), B.data(), C.data(), size_n);

        int64_t expected = 2 * 3 * size_n;
        for (int64_t val : C) {
            CHECK_EQ(val, expected);
        }
    }
}

TEST_CASE("Loop Tile - Inner Point Loop Auto-Vectorization Integration") {
    // 2D kernel with elementwise float arithmetic
    Module mod("test_tile_vectorize");
    Function* fn = mod.create_function("tile_vec_kernel", Type::void_type(), {
        Type::ptr(), Type::ptr(), Type::i64(), Type::i64()
    });
    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* in_ptr = b.add_block_param(entry, Type::ptr());
    Value* out_ptr = b.add_block_param(entry, Type::ptr());
    Value* N = b.add_block_param(entry, Type::i64());
    Value* M = b.add_block_param(entry, Type::i64());

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
    Value* row_off = b.build_mul(i, M);
    Value* four = b.build_iconst_i64(4);
    Value* row_bytes = b.build_mul(row_off, four);
    Value* in_row = b.build_add(in_ptr, row_bytes);
    Value* out_row = b.build_add(out_ptr, row_bytes);
    b.build_br(loop_j_hdr, {zero});

    fn->append_block(loop_j_hdr);
    b.position_at_end(loop_j_hdr);
    Value* j = b.add_block_param(loop_j_hdr, Type::i64());
    Value* cond_j = b.build_slt(j, M);
    b.build_br_if(cond_j, loop_j_body, {}, loop_i_next, {});

    fn->append_block(loop_j_body);
    b.position_at_end(loop_j_body);
    Value* val = b.build_load_indexed(Type::f32(), in_row, j, 4, 0);
    Value* res = b.build_add(val, val);
    b.build_store_indexed(Type::f32(), out_row, j, 4, 0, res);

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

    // 1. Tile the 2D loop
    DominatorTree dom(*fn);
    LoopTileOptions tile_opts;
    tile_opts.tile_size_i = 16;
    tile_opts.tile_size_j = 16;
    bool tiled = loop_tile_pass(*fn, dom, tile_opts);
    CHECK(tiled);

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    // 2. Vectorize the inner point loop
    fn->rebuild_cfg_predecessors();
    DominatorTree dom_after(*fn);
    LoopVectorizeOptions vec_opts;
    bool vectorized = loop_vectorize_pass(*fn, dom_after, vec_opts);
    CHECK(vectorized);

    REQUIRE(verify_function(*fn, &diag));

    // Check that vector instructions were generated
    bool found_vload = false;
    bool found_vstore = false;
    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::vload) found_vload = true;
            if (inst->opcode() == Opcode::vstore) found_vstore = true;
        }
    }
    CHECK(found_vload);
    CHECK(found_vstore);

    // JIT execute
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    auto fn_ptr = jit.get_function_ptr<void(*)(float*, float*, int64_t, int64_t)>("tile_vec_kernel");
    REQUIRE(fn_ptr != nullptr);

    const int64_t N_dim = 32;
    const int64_t M_dim = 32;
    std::vector<float> in_arr(N_dim * M_dim, 1.5f);
    std::vector<float> out_arr(N_dim * M_dim, 0.0f);

    fn_ptr(in_arr.data(), out_arr.data(), N_dim, M_dim);

    for (float v : out_arr) {
        CHECK_EQ(v, 3.0f);
    }
}
