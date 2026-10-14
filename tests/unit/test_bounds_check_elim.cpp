#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/range_analysis.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/mir/cfg_simplify.hpp>
#include <brass/mir/pass_pipeline.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>
#include <numeric>
#include <cmath>

using namespace brass;

TEST_CASE("BCE - Local Constant Bounds Check Elimination (In Bounds)") {
    Module mod("test_bce_local_in_bounds");
    Builder b(mod);

    Function* fn = mod.create_function("test_in_bounds", Type::i32(), {});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* bb_then = b.create_block("bb_then");
    BasicBlock* bb_else = b.create_block("bb_else");

    // len = 10, idx = 3 => ult 3, 10 is true
    Value* len = b.build_iconst_i64(10);
    Value* idx = b.build_iconst_i64(3);
    Value* cond = b.build_ult(idx, len);
    b.build_br_if(cond, bb_then, {}, bb_else, {});

    fn->append_block(bb_then);
    b.position_at_end(bb_then);
    Value* ret_then = b.build_iconst_i32(42);
    b.build_ret(ret_then);

    fn->append_block(bb_else);
    b.position_at_end(bb_else);
    Value* ret_else = b.build_iconst_i32(0);
    b.build_ret(ret_else);

    fn->rebuild_cfg_predecessors();

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;

    bool changed = run_bounds_check_elimination(*fn, mod, opts);
    CHECK(changed);
    CHECK(stats.bounds_checks_eliminated > 0);
    CHECK(stats.branches_folded > 0);

    // After folding and CFG simplification, bb_then is merged into entry with ret 42
    Instruction* term = entry->terminator();
    REQUIRE(term != nullptr);
    CHECK_EQ(term->opcode(), Opcode::ret);
    REQUIRE(term->operand(0) != nullptr);
    REQUIRE(term->operand(0)->defining_instruction() != nullptr);
    CHECK_EQ(term->operand(0)->defining_instruction()->imm_i32(), 42);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
}

TEST_CASE("BCE - Local Constant Bounds Check Elimination (Out of Bounds)") {
    Module mod("test_bce_local_oob");
    Builder b(mod);

    Function* fn = mod.create_function("test_oob", Type::i32(), {});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* bb_then = b.create_block("bb_then");
    BasicBlock* bb_else = b.create_block("bb_else");

    // len = 5, idx = 10 => ult 10, 5 is false
    Value* len = b.build_iconst_i64(5);
    Value* idx = b.build_iconst_i64(10);
    Value* cond = b.build_ult(idx, len);
    b.build_br_if(cond, bb_then, {}, bb_else, {});

    fn->append_block(bb_then);
    b.position_at_end(bb_then);
    Value* ret_then = b.build_iconst_i32(1);
    b.build_ret(ret_then);

    fn->append_block(bb_else);
    b.position_at_end(bb_else);
    Value* ret_else = b.build_iconst_i32(99);
    b.build_ret(ret_else);

    fn->rebuild_cfg_predecessors();

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;

    bool changed = run_bounds_check_elimination(*fn, mod, opts);
    CHECK(changed);
    CHECK(stats.bounds_checks_eliminated > 0);
    CHECK(stats.branches_folded > 0);

    // After folding and CFG simplification, bb_else is merged into entry with ret 99
    Instruction* term = entry->terminator();
    REQUIRE(term != nullptr);
    CHECK_EQ(term->opcode(), Opcode::ret);
    REQUIRE(term->operand(0) != nullptr);
    REQUIRE(term->operand(0)->defining_instruction() != nullptr);
    CHECK_EQ(term->operand(0)->defining_instruction()->imm_i32(), 99);

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
}

TEST_CASE("BCE - Redundant Guard Elimination") {
    Module mod("test_bce_guard");
    Builder b(mod);

    Function* fn = mod.create_function("test_guard", Type::i32(), {});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* true_cond = b.build_iconst_i32(1);
    b.build_guard(true_cond, "@test_stub");
    Value* ret_val = b.build_iconst_i32(7);
    b.build_ret(ret_val);

    fn->rebuild_cfg_predecessors();

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;

    bool changed = run_bounds_check_elimination(*fn, mod, opts);
    CHECK(changed);
    CHECK(stats.guards_eliminated > 0);

    // Verify guard instruction was removed
    for (Instruction* inst : *entry) {
        CHECK_NE(inst->opcode(), Opcode::guard);
    }

    DiagnosticReporter diag;
    CHECK(verify_function(*fn, &diag));
}

namespace {

// int64 sum(int64* arr, int64 len, int64 n):
//   for (i = 0; i < n; ++i) acc += (i <u bound) ? arr[i] : 0
// where `bound` is either `n` itself (the check repeats the loop test) or the
// independent `len` (nothing in the function relates it to `n`).
struct SumLoop {
    Function* fn = nullptr;
    BasicBlock* loop_check = nullptr;
    Instruction* check = nullptr;
};

SumLoop build_sum_loop(Module& mod, bool check_against_n) {
    Builder b(mod);
    SumLoop out;
    Function* fn = mod.create_function("sum_array", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    out.fn = fn;
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* arr = b.add_block_param(entry, Type::i64());
    Value* len = b.add_block_param(entry, Type::i64());
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* preheader = b.create_block("preheader");
    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_check = b.create_block("loop_check");
    BasicBlock* loop_fast = b.create_block("loop_fast");
    BasicBlock* loop_fallback = b.create_block("loop_fallback");
    BasicBlock* latch = b.create_block("latch");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(preheader, {});

    fn->append_block(preheader);
    b.position_at_end(preheader);
    b.build_br(loop_hdr, {zero, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_check, {}, exit_bb, {acc});

    fn->append_block(loop_check);
    b.position_at_end(loop_check);
    Value* in_bounds = b.build_ult(i, check_against_n ? n : len);
    out.check = in_bounds->defining_instruction();
    out.loop_check = loop_check;
    b.build_br_if(in_bounds, loop_fast, {}, loop_fallback, {});

    fn->append_block(loop_fast);
    b.position_at_end(loop_fast);
    Value* elem = b.build_load_indexed(Type::i64(), arr, i, 8, 0);
    Value* next_acc_fast = b.build_add(acc, elem);
    b.build_br(latch, {next_acc_fast});

    fn->append_block(loop_fallback);
    b.position_at_end(loop_fallback);
    b.build_br(latch, {acc});

    fn->append_block(latch);
    b.position_at_end(latch);
    Value* cur_acc = b.add_block_param(latch, Type::i64());
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, cur_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_sum = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();
    return out;
}

size_t count_opcode(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == op) ++n;
        }
    }
    return n;
}

} // namespace

TEST_CASE("BCE - Loop check implied by the loop's own exit test is folded") {
    Module mod("test_bce_implied");
    SumLoop loop = build_sum_loop(mod, /*check_against_n=*/true);

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;
    CHECK(run_bounds_check_elimination(*loop.fn, mod, opts));
    CHECK(stats.checks_implied > 0);

    // `i <u n` inside a body entered on `i <s n` from i = 0 always holds: the
    // fallback path is gone, and nothing was speculated to get there.
    Instruction* term = loop.loop_check->terminator();
    REQUIRE(term != nullptr);
    CHECK_EQ(term->opcode(), Opcode::br);
    CHECK_EQ(count_opcode(*loop.fn, Opcode::guard), size_t(0));
    CHECK(verify_function(*loop.fn));
}

TEST_CASE("BCE - Check against an unrelated bound is not folded or guarded") {
    Module mod("test_bce_unrelated");
    SumLoop loop = build_sum_loop(mod, /*check_against_n=*/false);

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;
    run_bounds_check_elimination(*loop.fn, mod, opts);

    // Nothing relates `len` to `n`, so the per-iteration check has to stay,
    // and no deopt guard may stand in for it (AOT code has nowhere to go).
    CHECK_EQ(stats.checks_implied, size_t(0));
    CHECK_EQ(count_opcode(*loop.fn, Opcode::guard), size_t(0));
    CHECK_EQ(count_opcode(*loop.fn, Opcode::ult), size_t(1));
    Instruction* term = loop.loop_check->terminator();
    REQUIRE(term != nullptr);
    CHECK_EQ(term->opcode(), Opcode::br_if);
    CHECK(verify_function(*loop.fn));
}

TEST_CASE("BCE - JIT Execution of Bounds Checked Loops") {
    std::vector<int64_t> data(100);
    std::iota(data.begin(), data.end(), int64_t{1});
    using SumFn = int64_t(*)(const int64_t*, int64_t, int64_t);

    for (bool against_n : {true, false}) {
        Module mod("test_bce_jit");
        SumLoop loop = build_sum_loop(mod, against_n);

        LoopOptOptions loop_opts;
        loop_opts.enable_bce = true;
        loop_opts.enable_unroll = false;
        loop_opts.enable_vectorize = false;
        optimize_function_loops(*loop.fn, loop_opts);
        CHECK(verify_function(*loop.fn));

        codegen::JitExecutionEngine jit;
        REQUIRE(jit.compile_and_load(mod));
        auto sum_ptr = jit.get_function_ptr<SumFn>("sum_array");
        REQUIRE(sum_ptr != nullptr);

        CHECK_EQ(sum_ptr(data.data(), 100, 100), int64_t{5050});
        CHECK_EQ(sum_ptr(data.data(), 100, 50), int64_t{1275});
        if (!against_n) {
            // n beyond len: the elements past len fall back to 0 every time.
            CHECK_EQ(sum_ptr(data.data(), 10, 100), int64_t{55});
            CHECK_EQ(sum_ptr(data.data(), 0, 100), int64_t{0});
        }
    }
}

namespace {

struct MockTypedArrayHeader {
    uint64_t pad0 = 0;
    uint64_t pad1 = 0;
    uint64_t pad2 = 0;
    double* data_ptr = nullptr; // offset 24
    uint64_t byte_len = 0;      // offset 32
};

// Builds a bronze-style 5-block ElemGetTyped (f64) diamond at the end of `cur_bb`,
// returning {merge_bb, loaded_f64_val}.
std::pair<BasicBlock*, Value*> build_elem_get_f64_diamond(
    Builder& b, Function* fn, BasicBlock* cur_bb, Value* ta_ptr, Value* idx_i32, const char* tag
) {
    BasicBlock* bb_idx_ok = b.create_block(std::string(tag) + "_idx_ok");
    BasicBlock* bb_load = b.create_block(std::string(tag) + "_load");
    BasicBlock* bb_slow = b.create_block(std::string(tag) + "_slow");
    BasicBlock* bb_merge = b.create_block(std::string(tag) + "_merge");

    b.position_at_end(cur_bb);
    Value* idx_i64 = b.build_sext_i64(idx_i32);
    Value* zero64 = b.build_iconst_i64(0);
    Value* non_neg = b.build_sge(idx_i64, zero64);
    b.build_br_if(non_neg, bb_idx_ok, {}, bb_slow, {});

    fn->append_block(bb_idx_ok);
    b.position_at_end(bb_idx_ok);
    Value* data_ptr = b.build_load(Type::i64(), ta_ptr, 24);
    Value* not_detached = b.build_ne(data_ptr, zero64);
    Value* byte_len = b.build_load(Type::i64(), ta_ptr, 32);
    Value* byte_off = b.build_shl(idx_i64, b.build_iconst_i64(3));
    Value* in_bounds = b.build_ult(byte_off, byte_len);
    Value* fast_ok = b.build_and(not_detached, in_bounds);
    b.build_br_if(fast_ok, bb_load, {}, bb_slow, {});

    fn->append_block(bb_load);
    b.position_at_end(bb_load);
    Value* raw_val = b.build_load_indexed(Type::f64(), data_ptr, idx_i64, 8, 0);
    // Canonical NaN boxing pattern emitted by il_elem_lowering.cpp:
    // select(eq(raw, raw), bitcast_i64_f64(raw), kCanonicalNaNBits) -> bitcast_f64_i64
    Value* is_num = b.build_eq(raw_val, raw_val);
    Value* raw_bits = b.build_bitcast_i64_f64(raw_val);
    Value* canon_nan = b.build_iconst_i64(0x7FF8000000000000LL);
    Value* boxed = b.build_select(is_num, raw_bits, canon_nan);
    b.build_br(bb_merge, {boxed});

    fn->append_block(bb_slow);
    b.position_at_end(bb_slow);
    Value* slow_val = b.build_call("mock_ta_slow_get", Type::i64(), {ta_ptr, idx_i64});
    b.build_br(bb_merge, {slow_val});

    fn->append_block(bb_merge);
    b.position_at_end(bb_merge);
    Value* merged_bits = b.add_block_param(bb_merge, Type::i64());
    Value* merged_f64 = b.build_bitcast_f64_i64(merged_bits);
    return {bb_merge, merged_f64};
}

// Builds a bronze-style 5-block ElemSetTyped (f64) diamond at the end of `cur_bb`,
// returning merge_bb.
BasicBlock* build_elem_set_f64_diamond(
    Builder& b, Function* fn, BasicBlock* cur_bb, Value* ta_ptr, Value* idx_i32, Value* val_f64, const char* tag
) {
    BasicBlock* bb_idx_ok = b.create_block(std::string(tag) + "_idx_ok");
    BasicBlock* bb_store = b.create_block(std::string(tag) + "_store");
    BasicBlock* bb_slow = b.create_block(std::string(tag) + "_slow");
    BasicBlock* bb_merge = b.create_block(std::string(tag) + "_merge");

    b.position_at_end(cur_bb);
    Value* idx_i64 = b.build_sext_i64(idx_i32);
    Value* zero64 = b.build_iconst_i64(0);
    Value* non_neg = b.build_sge(idx_i64, zero64);
    b.build_br_if(non_neg, bb_idx_ok, {}, bb_slow, {});

    fn->append_block(bb_idx_ok);
    b.position_at_end(bb_idx_ok);
    Value* data_ptr = b.build_load(Type::i64(), ta_ptr, 24);
    Value* not_detached = b.build_ne(data_ptr, zero64);
    Value* byte_len = b.build_load(Type::i64(), ta_ptr, 32);
    Value* byte_off = b.build_shl(idx_i64, b.build_iconst_i64(3));
    Value* in_bounds = b.build_ult(byte_off, byte_len);
    Value* fast_ok = b.build_and(not_detached, in_bounds);
    b.build_br_if(fast_ok, bb_store, {}, bb_slow, {});

    fn->append_block(bb_store);
    b.position_at_end(bb_store);
    b.build_store_indexed(Type::f64(), data_ptr, idx_i64, 8, 0, val_f64);
    b.build_br(bb_merge, {});

    fn->append_block(bb_slow);
    b.position_at_end(bb_slow);
    Value* val_bits = b.build_bitcast_i64_f64(val_f64);
    b.build_call("mock_ta_slow_set", Type::i64(), {ta_ptr, idx_i64, val_bits});
    b.build_br(bb_merge, {});

    fn->append_block(bb_merge);
    return bb_merge;
}

} // namespace

TEST_CASE("BCE - TypedArray 5-Block Diamond Collapses When Bound Derived From ByteLength") {
    Module mod("test_ta_bce_bytelength");
    Builder b(mod);

    Function* fn = mod.create_function("ta_scale_derived", Type::i32(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ta = b.add_block_param(entry, Type::i64());
    BasicBlock* preheader = b.create_block("preheader");
    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* data0 = b.build_load(Type::i64(), ta, 24);
    Value* valid_data = b.build_ne(data0, b.build_iconst_i64(0));
    b.build_br_if(valid_data, preheader, {}, exit_bb, {});

    fn->append_block(preheader);
    b.position_at_end(preheader);
    Value* byte_len0 = b.build_load(Type::i64(), ta, 32);
    Value* len64 = b.build_lshr(byte_len0, b.build_iconst_i64(3));
    Value* len32 = b.build_trunc_i32(len64);
    b.build_br(loop_hdr, {b.build_iconst_i32(0)});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* iv = b.add_block_param(loop_hdr, Type::i32());
    Value* cond = b.build_slt(iv, len32);
    b.build_br_if(cond, loop_body, {}, exit_bb, {});

    fn->append_block(loop_body);
    auto [after_get, loaded] = build_elem_get_f64_diamond(b, fn, loop_body, ta, iv, "get_a");
    b.position_at_end(after_get);
    Value* scaled = b.build_mul(loaded, b.build_fconst_f64(2.0));
    BasicBlock* after_set = build_elem_set_f64_diamond(b, fn, after_get, ta, iv, scaled, "set_a");
    b.position_at_end(after_set);
    Value* next_iv = b.build_add(iv, b.build_iconst_i32(1));
    b.build_br(loop_hdr, {next_iv});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret(b.build_iconst_i32(0));

    fn->rebuild_cfg_predecessors();

    RangeAnalysisStats stats;
    RangeAnalysisOptions opts;
    opts.stats = &stats;
    CHECK(run_bounds_check_elimination(*fn, mod, opts));
    hoist_loop_invariants(*fn);
    fold_constants(*fn);
    cfg_simplify_function(*fn);

    // Every slow-path fallback call is eliminated and the loop body is a single block.
    CHECK_EQ(count_opcode(*fn, Opcode::call), size_t{0});
    DominatorTree dom(*fn);
    LoopAnalysis loops(*fn, dom);
    REQUIRE_EQ(loops.top_level_loops().size(), size_t{1});
    CHECK_EQ(loops.top_level_loops()[0]->blocks().size(), size_t{2}); // header + single latch block
    CHECK(verify_function(*fn));
}

TEST_CASE("BCE & Loop Unswitch - TypedArray 5-Block Diamond Unswitches, Fuses FMA, and Vectorizes") {
    Module mod("test_ta_unswitch_fma_vec");
    Builder b(mod);

    // void ta_fma(ta_a, ta_b, ta_c, int32 n):
    //   for (int32 i = 0; i < n; ++i) a[i] = a[i] * b[i] + c[i]
    Function* fn = mod.create_function(
        "ta_fma", Type::i32(), {Type::i64(), Type::i64(), Type::i64(), Type::i32()}
    );
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ta_a = b.add_block_param(entry, Type::i64());
    Value* ta_b = b.add_block_param(entry, Type::i64());
    Value* ta_c = b.add_block_param(entry, Type::i64());
    Value* n = b.add_block_param(entry, Type::i32());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    b.build_br(loop_hdr, {b.build_iconst_i32(0)});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* iv = b.add_block_param(loop_hdr, Type::i32());
    Value* cond = b.build_slt(iv, n);
    b.build_br_if(cond, loop_body, {}, exit_bb, {});

    fn->append_block(loop_body);
    auto [bb1, va] = build_elem_get_f64_diamond(b, fn, loop_body, ta_a, iv, "get_a");
    auto [bb2, vb] = build_elem_get_f64_diamond(b, fn, bb1, ta_b, iv, "get_b");
    auto [bb3, vc] = build_elem_get_f64_diamond(b, fn, bb2, ta_c, iv, "get_c");
    b.position_at_end(bb3);
    Value* prod = b.build_mul(va, vb);
    Value* res = b.build_fadd(prod, vc);
    BasicBlock* bb4 = build_elem_set_f64_diamond(b, fn, bb3, ta_a, iv, res, "set_a");
    b.position_at_end(bb4);
    Value* next_iv = b.build_add(iv, b.build_iconst_i32(1));
    b.build_br(loop_hdr, {next_iv});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret(b.build_iconst_i32(0));

    fn->rebuild_cfg_predecessors();

    PassPipelineOptions pipe_opts = production_pass_pipeline_options();
    pipe_opts.loop.enable_fma = true;
    pipe_opts.loop.enable_vectorize = true;
    pipe_opts.loop.vector_width = 16;
    REQUIRE(run_pass_pipeline(mod, pipe_opts));

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    // Verify the fast loop was SIMD-vectorized with vload, vfma, and vstore!
    CHECK(count_opcode(*fn, Opcode::vload) > 0);
    CHECK(count_opcode(*fn, Opcode::vfma) > 0);
    CHECK(count_opcode(*fn, Opcode::vstore) > 0);

    // Execute via JIT and verify numerical results
    constexpr size_t kLen = 64;
    std::vector<double> buf_a(kLen), buf_b(kLen), buf_c(kLen), expected(kLen);
    for (size_t i = 0; i < kLen; ++i) {
        buf_a[i] = static_cast<double>(i + 1) * 1.5;
        buf_b[i] = static_cast<double>(i + 2) * 0.25;
        buf_c[i] = static_cast<double>(i) * -0.5;
        expected[i] = std::fma(buf_a[i], buf_b[i], buf_c[i]);
    }
    MockTypedArrayHeader hdr_a{0, 0, 0, buf_a.data(), kLen * sizeof(double)};
    MockTypedArrayHeader hdr_b{0, 0, 0, buf_b.data(), kLen * sizeof(double)};
    MockTypedArrayHeader hdr_c{0, 0, 0, buf_c.data(), kLen * sizeof(double)};

    codegen::JitExecutionEngine jit;
    jit.register_external_symbol("mock_ta_slow_get", reinterpret_cast<void*>(+[](int64_t, int64_t) -> int64_t { return 0; }));
    jit.register_external_symbol("mock_ta_slow_set", reinterpret_cast<void*>(+[](int64_t, int64_t, int64_t) -> int64_t { return 0; }));
    REQUIRE(jit.compile_and_load(mod));
    using TaFmaFn = int32_t(*)(MockTypedArrayHeader*, MockTypedArrayHeader*, MockTypedArrayHeader*, int32_t);
    auto fn_ptr = jit.get_function_ptr<TaFmaFn>("ta_fma");
    REQUIRE(fn_ptr != nullptr);

    CHECK_EQ(fn_ptr(&hdr_a, &hdr_b, &hdr_c, static_cast<int32_t>(kLen)), 0);
    for (size_t i = 0; i < kLen; ++i) {
        CHECK_EQ(buf_a[i], expected[i]);
    }
}

TEST_CASE("BCE & Loop Unswitch - TypedArray Dot-Product Reduction Unswitches and Vectorizes") {
    Module mod("test_ta_dot_vec");
    Builder b(mod);

    // double ta_dot(ta_a, ta_b, int32 n):
    //   double sum = 0.0;
    //   for (int32 i = 0; i < n; ++i) sum += a[i] * b[i];
    //   return sum;
    Function* fn = mod.create_function(
        "ta_dot", Type::f64(), {Type::i64(), Type::i64(), Type::i32()}
    );
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ta_a = b.add_block_param(entry, Type::i64());
    Value* ta_b = b.add_block_param(entry, Type::i64());
    Value* n = b.add_block_param(entry, Type::i32());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    b.build_br(loop_hdr, {b.build_iconst_i32(0), b.build_fconst_f64(0.0)});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* iv = b.add_block_param(loop_hdr, Type::i32());
    Value* sum = b.add_block_param(loop_hdr, Type::f64());
    Value* cond = b.build_slt(iv, n);
    b.build_br_if(cond, loop_body, {}, exit_bb, {sum});

    fn->append_block(loop_body);
    auto [bb1, va] = build_elem_get_f64_diamond(b, fn, loop_body, ta_a, iv, "get_a");
    auto [bb2, vb] = build_elem_get_f64_diamond(b, fn, bb1, ta_b, iv, "get_b");
    b.position_at_end(bb2);
    Value* prod = b.build_mul(va, vb);
    Value* next_sum = b.build_fadd(sum, prod);
    Value* next_iv = b.build_add(iv, b.build_iconst_i32(1));
    b.build_br(loop_hdr, {next_iv, next_sum});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_sum = b.add_block_param(exit_bb, Type::f64());
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();

    PassPipelineOptions pipe_opts = production_pass_pipeline_options();
    pipe_opts.loop.enable_fma = true;
    pipe_opts.loop.enable_vectorize = true;
    pipe_opts.loop.enable_fp_reassociation = true;
    pipe_opts.loop.vector_width = 16;
    REQUIRE(run_pass_pipeline(mod, pipe_opts));

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));
    CHECK(count_opcode(*fn, Opcode::vload) > 0);
    CHECK(count_opcode(*fn, Opcode::vfma) > 0);

    constexpr size_t kLen = 64;
    std::vector<double> buf_a(kLen), buf_b(kLen);
    double expected = 0.0;
    for (size_t i = 0; i < kLen; ++i) {
        buf_a[i] = static_cast<double>(i + 1) * 0.5;
        buf_b[i] = static_cast<double>(i + 2) * 2.0;
        expected += buf_a[i] * buf_b[i];
    }
    MockTypedArrayHeader hdr_a{0, 0, 0, buf_a.data(), kLen * sizeof(double)};
    MockTypedArrayHeader hdr_b{0, 0, 0, buf_b.data(), kLen * sizeof(double)};

    codegen::JitExecutionEngine jit;
    jit.register_external_symbol("mock_ta_slow_get", reinterpret_cast<void*>(+[](int64_t, int64_t) -> int64_t { return 0; }));
    REQUIRE(jit.compile_and_load(mod));
    using TaDotFn = double(*)(MockTypedArrayHeader*, MockTypedArrayHeader*, int32_t);
    auto fn_ptr = jit.get_function_ptr<TaDotFn>("ta_dot");
    REQUIRE(fn_ptr != nullptr);

    CHECK_EQ(fn_ptr(&hdr_a, &hdr_b, static_cast<int32_t>(kLen)), expected);
}

namespace {

// Builds a bronze-style 5-block ElemGetTyped (f64) diamond taking an f64 index
// (emitting fptosi_i32 + sitofp_f64_i32 + eq exact-integer check), returning {merge_bb, loaded_f64_val}.
std::pair<BasicBlock*, Value*> build_elem_get_f64_idx_diamond(
    Builder& b, Function* fn, BasicBlock* cur_bb, Value* ta_ptr, Value* idx_f64, const char* tag
) {
    BasicBlock* bb_idx_ok = b.create_block(std::string(tag) + "_idx_ok");
    BasicBlock* bb_load = b.create_block(std::string(tag) + "_load");
    BasicBlock* bb_slow = b.create_block(std::string(tag) + "_slow");
    BasicBlock* bb_merge = b.create_block(std::string(tag) + "_merge");

    b.position_at_end(cur_bb);
    Value* idx_i32 = b.build_fptosi_i32(idx_f64);
    Value* back_f64 = b.build_sitofp_f64_i32(idx_i32);
    Value* is_int = b.build_eq(back_f64, idx_f64);
    Value* non_neg = b.build_sge(idx_i32, b.build_iconst_i32(0));
    Value* idx_valid = b.build_and(is_int, non_neg);
    b.build_br_if(idx_valid, bb_idx_ok, {}, bb_slow, {});

    fn->append_block(bb_idx_ok);
    b.position_at_end(bb_idx_ok);
    Value* zero64 = b.build_iconst_i64(0);
    Value* data_ptr = b.build_load(Type::i64(), ta_ptr, 24);
    Value* not_detached = b.build_ne(data_ptr, zero64);
    Value* byte_len = b.build_load(Type::i64(), ta_ptr, 32);
    Value* idx_i64 = b.build_sext_i64(idx_i32);
    Value* byte_off = b.build_shl(idx_i64, b.build_iconst_i64(3));
    Value* in_bounds = b.build_ult(byte_off, byte_len);
    Value* fast_ok = b.build_and(not_detached, in_bounds);
    b.build_br_if(fast_ok, bb_load, {}, bb_slow, {});

    fn->append_block(bb_load);
    b.position_at_end(bb_load);
    Value* raw_val = b.build_load_indexed(Type::f64(), data_ptr, idx_i64, 8, 0);
    Value* is_num = b.build_eq(raw_val, raw_val);
    Value* raw_bits = b.build_bitcast_i64_f64(raw_val);
    Value* canon_nan = b.build_iconst_i64(0x7FF8000000000000LL);
    Value* boxed = b.build_select(is_num, raw_bits, canon_nan);
    b.build_br(bb_merge, {boxed});

    fn->append_block(bb_slow);
    b.position_at_end(bb_slow);
    Value* slow_idx = b.build_bitcast_i64_f64(idx_f64);
    Value* slow_val = b.build_call("mock_ta_slow_get", Type::i64(), {ta_ptr, slow_idx});
    b.build_br(bb_merge, {slow_val});

    fn->append_block(bb_merge);
    b.position_at_end(bb_merge);
    Value* merged_bits = b.add_block_param(bb_merge, Type::i64());
    Value* merged_f64 = b.build_bitcast_f64_i64(merged_bits);
    return {bb_merge, merged_f64};
}

// Builds a bronze-style 5-block ElemSetTyped (f64) diamond taking an f64 index.
BasicBlock* build_elem_set_f64_idx_diamond(
    Builder& b, Function* fn, BasicBlock* cur_bb, Value* ta_ptr, Value* idx_f64, Value* val_f64, const char* tag
) {
    BasicBlock* bb_idx_ok = b.create_block(std::string(tag) + "_idx_ok");
    BasicBlock* bb_store = b.create_block(std::string(tag) + "_store");
    BasicBlock* bb_slow = b.create_block(std::string(tag) + "_slow");
    BasicBlock* bb_merge = b.create_block(std::string(tag) + "_merge");

    b.position_at_end(cur_bb);
    Value* idx_i32 = b.build_fptosi_i32(idx_f64);
    Value* back_f64 = b.build_sitofp_f64_i32(idx_i32);
    Value* is_int = b.build_eq(back_f64, idx_f64);
    Value* non_neg = b.build_sge(idx_i32, b.build_iconst_i32(0));
    Value* idx_valid = b.build_and(is_int, non_neg);
    b.build_br_if(idx_valid, bb_idx_ok, {}, bb_slow, {});

    fn->append_block(bb_idx_ok);
    b.position_at_end(bb_idx_ok);
    Value* zero64 = b.build_iconst_i64(0);
    Value* data_ptr = b.build_load(Type::i64(), ta_ptr, 24);
    Value* not_detached = b.build_ne(data_ptr, zero64);
    Value* byte_len = b.build_load(Type::i64(), ta_ptr, 32);
    Value* idx_i64 = b.build_sext_i64(idx_i32);
    Value* byte_off = b.build_shl(idx_i64, b.build_iconst_i64(3));
    Value* in_bounds = b.build_ult(byte_off, byte_len);
    Value* fast_ok = b.build_and(not_detached, in_bounds);
    b.build_br_if(fast_ok, bb_store, {}, bb_slow, {});

    fn->append_block(bb_store);
    b.position_at_end(bb_store);
    b.build_store_indexed(Type::f64(), data_ptr, idx_i64, 8, 0, val_f64);
    b.build_br(bb_merge, {});

    fn->append_block(bb_slow);
    b.position_at_end(bb_slow);
    Value* slow_idx = b.build_bitcast_i64_f64(idx_f64);
    Value* val_bits = b.build_bitcast_i64_f64(val_f64);
    b.build_call("mock_ta_slow_set", Type::i64(), {ta_ptr, slow_idx, val_bits});
    b.build_br(bb_merge, {});

    fn->append_block(bb_merge);
    return bb_merge;
}

} // namespace

TEST_CASE("BCE & F64 Demote - Counted f64 IV Loop Demotes, Unswitches, Fuses FMA, and Vectorizes") {
    Module mod("test_f64_iv_ta_fma_vec");
    Builder b(mod);

    // Mirrors bronze-lowered JS loop:
    //   for (let i = 0.0; i < n_f64; i += 1.0) a[i] = a[i] * b[i] + c[i];
    Function* fn = mod.create_function(
        "ta_f64_iv_fma", Type::i32(), {Type::i64(), Type::i64(), Type::i64(), Type::f64()}
    );
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ta_a = b.add_block_param(entry, Type::i64());
    Value* ta_b = b.add_block_param(entry, Type::i64());
    Value* ta_c = b.add_block_param(entry, Type::i64());
    Value* n_f64 = b.add_block_param(entry, Type::f64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    b.build_br(loop_hdr, {b.build_fconst_f64(0.0)});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* iv_f64 = b.add_block_param(loop_hdr, Type::f64());
    Value* cond = b.build_slt(iv_f64, n_f64);
    b.build_br_if(cond, loop_body, {}, exit_bb, {});

    fn->append_block(loop_body);
    auto [bb1, va] = build_elem_get_f64_idx_diamond(b, fn, loop_body, ta_a, iv_f64, "get_a");
    auto [bb2, vb] = build_elem_get_f64_idx_diamond(b, fn, bb1, ta_b, iv_f64, "get_b");
    auto [bb3, vc] = build_elem_get_f64_idx_diamond(b, fn, bb2, ta_c, iv_f64, "get_c");
    b.position_at_end(bb3);
    Value* prod = b.build_mul(va, vb);
    Value* res = b.build_fadd(prod, vc);
    BasicBlock* bb4 = build_elem_set_f64_idx_diamond(b, fn, bb3, ta_a, iv_f64, res, "set_a");
    b.position_at_end(bb4);
    Value* next_iv = b.build_add(iv_f64, b.build_fconst_f64(1.0));
    b.build_br(loop_hdr, {next_iv});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret(b.build_iconst_i32(0));

    fn->rebuild_cfg_predecessors();

    PassPipelineOptions pipe_opts = production_pass_pipeline_options();
    pipe_opts.loop.enable_fma = true;
    pipe_opts.loop.enable_vectorize = true;
    pipe_opts.loop.vector_width = 16;
    REQUIRE(run_pass_pipeline(mod, pipe_opts));

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    // Verify the f64 IV was demoted to i64 and the fast loop was SIMD-vectorized with vload, vfma, and vstore!
    CHECK_EQ(loop_hdr->param(0)->type(), Type::i64());
    CHECK(count_opcode(*fn, Opcode::vload) > 0);
    CHECK(count_opcode(*fn, Opcode::vfma) > 0);
    CHECK(count_opcode(*fn, Opcode::vstore) > 0);

    constexpr size_t kLen = 64;
    std::vector<double> buf_a(kLen), buf_b(kLen), buf_c(kLen), expected(kLen);
    for (size_t i = 0; i < kLen; ++i) {
        buf_a[i] = static_cast<double>(i + 1) * 1.5;
        buf_b[i] = static_cast<double>(i + 2) * 0.25;
        buf_c[i] = static_cast<double>(i) * -0.5;
        expected[i] = std::fma(buf_a[i], buf_b[i], buf_c[i]);
    }
    MockTypedArrayHeader hdr_a{0, 0, 0, buf_a.data(), kLen * sizeof(double)};
    MockTypedArrayHeader hdr_b{0, 0, 0, buf_b.data(), kLen * sizeof(double)};
    MockTypedArrayHeader hdr_c{0, 0, 0, buf_c.data(), kLen * sizeof(double)};

    codegen::JitExecutionEngine jit;
    jit.register_external_symbol("mock_ta_slow_get", reinterpret_cast<void*>(+[](int64_t, int64_t) -> int64_t { return 0; }));
    jit.register_external_symbol("mock_ta_slow_set", reinterpret_cast<void*>(+[](int64_t, int64_t, int64_t) -> int64_t { return 0; }));
    REQUIRE(jit.compile_and_load(mod));
    using TaF64FmaFn = int32_t(*)(MockTypedArrayHeader*, MockTypedArrayHeader*, MockTypedArrayHeader*, double);
    auto fn_ptr = jit.get_function_ptr<TaF64FmaFn>("ta_f64_iv_fma");
    REQUIRE(fn_ptr != nullptr);

    CHECK_EQ(fn_ptr(&hdr_a, &hdr_b, &hdr_c, static_cast<double>(kLen)), 0);
    for (size_t i = 0; i < kLen; ++i) {
        CHECK_EQ(buf_a[i], expected[i]);
    }
}

TEST_CASE("BCE & F64 Demote - Counted f64 IV Dot-Product Reduction Demotes, Unswitches, and Vectorizes") {
    Module mod("test_f64_iv_dot_vec");
    Builder b(mod);

    // Mirrors bronze-lowered JS reduction loop with f64 IV + f64 accumulator:
    //   let sum = 0.0;
    //   for (let i = 0.0; i < 64.0; i += 1.0) sum += a[i] * b[i];
    //   return sum;
    Function* fn = mod.create_function(
        "ta_f64_dot", Type::f64(), {Type::i64(), Type::i64()}
    );
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ta_a = b.add_block_param(entry, Type::i64());
    Value* ta_b = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    b.build_br(loop_hdr, {b.build_fconst_f64(0.0), b.build_fconst_f64(0.0)});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* iv_f64 = b.add_block_param(loop_hdr, Type::f64());
    Value* sum = b.add_block_param(loop_hdr, Type::f64());
    Value* cond = b.build_slt(iv_f64, b.build_fconst_f64(64.0));
    b.build_br_if(cond, loop_body, {}, exit_bb, {sum});

    fn->append_block(loop_body);
    auto [bb1, va] = build_elem_get_f64_idx_diamond(b, fn, loop_body, ta_a, iv_f64, "get_a");
    auto [bb2, vb] = build_elem_get_f64_idx_diamond(b, fn, bb1, ta_b, iv_f64, "get_b");
    b.position_at_end(bb2);
    Value* prod = b.build_mul(va, vb);
    Value* next_sum = b.build_fadd(sum, prod);
    Value* next_iv = b.build_add(iv_f64, b.build_fconst_f64(1.0));
    b.build_br(loop_hdr, {next_iv, next_sum});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_sum = b.add_block_param(exit_bb, Type::f64());
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();

    PassPipelineOptions pipe_opts = production_pass_pipeline_options();
    pipe_opts.loop.enable_fma = true;
    pipe_opts.loop.enable_vectorize = true;
    pipe_opts.loop.enable_fp_reassociation = true;
    pipe_opts.loop.vector_width = 16;
    REQUIRE(run_pass_pipeline(mod, pipe_opts));

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));
    CHECK(count_opcode(*fn, Opcode::vload) > 0);
    CHECK(count_opcode(*fn, Opcode::vfma) > 0);

    constexpr size_t kLen = 64;
    std::vector<double> buf_a(kLen), buf_b(kLen);
    double expected = 0.0;
    for (size_t i = 0; i < kLen; ++i) {
        buf_a[i] = static_cast<double>(i + 1) * 0.5;
        buf_b[i] = static_cast<double>(i + 2) * 2.0;
        expected += buf_a[i] * buf_b[i];
    }
    MockTypedArrayHeader hdr_a{0, 0, 0, buf_a.data(), kLen * sizeof(double)};
    MockTypedArrayHeader hdr_b{0, 0, 0, buf_b.data(), kLen * sizeof(double)};

    codegen::JitExecutionEngine jit;
    jit.register_external_symbol("mock_ta_slow_get", reinterpret_cast<void*>(+[](int64_t, int64_t) -> int64_t { return 0; }));
    REQUIRE(jit.compile_and_load(mod));
    using TaF64DotFn = double(*)(MockTypedArrayHeader*, MockTypedArrayHeader*);
    auto fn_ptr = jit.get_function_ptr<TaF64DotFn>("ta_f64_dot");
    REQUIRE(fn_ptr != nullptr);

    CHECK_EQ(fn_ptr(&hdr_a, &hdr_b), expected);
}



