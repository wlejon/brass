#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/f64_demote.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/pass_pipeline.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/builder.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <cmath>
#include <vector>

using namespace brass;
using namespace brass::test;

namespace {
// The host's float remainder (a SymbolRole::FloatRem helper).
double host_f64_mod(double a, double b) { return std::fmod(a, b); }
}

TEST_CASE("F64 Demote - Collatz Inner Loop Demotion & JIT Execution") {
    Module mod("test_collatz_demote");
    mod.add_symbol_role("host_f64_mod", SymbolRole::FloatRem);
    Builder b(mod);

    Function* fn = mod.create_function("collatz", Type::f64(), {Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* arg0 = b.add_block_param(b0, Type::f64());

    BasicBlock* b1 = b.create_block("b1");
    BasicBlock* b2 = b.create_block("b2");
    BasicBlock* b3 = b.create_block("b3");
    BasicBlock* b4 = b.create_block("b4");
    BasicBlock* b5 = b.create_block("b5");
    BasicBlock* b6 = b.create_block("b6");

    Value* zero_f = b.build_fconst_f64(0.0);
    b.build_br(b1, {arg0, zero_f});

    fn->append_block(b1);
    b.position_at_end(b1);
    Value* n = b.add_block_param(b1, Type::f64());
    Value* steps = b.add_block_param(b1, Type::f64());
    Value* one_f = b.build_fconst_f64(1.0);
    Value* cond = b.build_sgt(n, one_f);
    b.build_br_if(cond, b2, {}, b3, {n, steps});

    fn->append_block(b2);
    b.position_at_end(b2);
    Value* two_f = b.build_fconst_f64(2.0);
    Value* mod_val = b.build_call("host_f64_mod", Type::f64(), {n, two_f});
    Value* is_even = b.build_eq(mod_val, zero_f);
    b.build_br_if(is_even, b4, {}, b5, {});

    fn->append_block(b3);
    b.position_at_end(b3);
    Value* ret_n = b.add_block_param(b3, Type::f64());
    Value* ret_steps = b.add_block_param(b3, Type::f64());
    (void)ret_n;
    b.build_ret(ret_steps);

    fn->append_block(b4);
    b.position_at_end(b4);
    Value* two_f4 = b.build_fconst_f64(2.0);
    Value* half = b.build_sdiv(n, two_f4);
    b.build_br(b6, {half});

    fn->append_block(b5);
    b.position_at_end(b5);
    Value* three_f = b.build_fconst_f64(3.0);
    Value* mul3 = b.build_mul(n, three_f);
    Value* one_f5 = b.build_fconst_f64(1.0);
    Value* odd_next = b.build_add(mul3, one_f5);
    b.build_br(b6, {odd_next});

    fn->append_block(b6);
    b.position_at_end(b6);
    Value* next_n = b.add_block_param(b6, Type::f64());
    Value* one_f6 = b.build_fconst_f64(1.0);
    Value* next_steps = b.build_add(steps, one_f6);
    b.build_br(b1, {next_n, next_steps});

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    DemoteStats stats;
    F64DemoteOptions options;
    options.stats = &stats;

    bool demoted = f64_demote_pass(*fn, options);
    CHECK(!demoted);
    REQUIRE(verify_function(*fn, &diag));

    // Verify b1 loop was refused due to profitability model
    CHECK_EQ(b1->param(0)->type(), Type::f64());
    CHECK_EQ(b1->param(1)->type(), Type::f64());
    CHECK_EQ(stats.loops_refused, 1u);
    CHECK_EQ(stats.refusal_counts[DemoteRefusalReason::Unprofitable], 1u);

    // Optimize and run with JIT
    LoopOptOptions opt;
    optimize_function_loops(*fn, opt);
    REQUIRE(verify_function(*fn, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("host_f64_mod", reinterpret_cast<void*>(&host_f64_mod));
    REQUIRE(jit.compile_and_load(mod));

    auto collatz_fn = jit.get_function_ptr<double(*)(double)>("collatz");
    REQUIRE(collatz_fn != nullptr);
    CHECK_EQ(collatz_fn(27.0), 111.0);
    CHECK_EQ(collatz_fn(1.0), 0.0);
    CHECK_EQ(collatz_fn(6.0), 8.0);
    CHECK_EQ(collatz_fn(12.0), 9.0);
}

TEST_CASE("F64 Demote - Fib Iteration Demotion & Loop Unrolling") {
    Module mod("test_fib_demote");
    Builder b(mod);

    Function* fn = mod.create_function("fibIter", Type::f64(), {Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* n = b.add_block_param(b0, Type::f64());

    BasicBlock* b1 = b.create_block("b1");
    BasicBlock* b2 = b.create_block("b2");
    BasicBlock* b3 = b.create_block("b3");

    Value* zero = b.build_fconst_f64(0.0);
    Value* one = b.build_fconst_f64(1.0);
    b.build_br(b1, {zero, one, zero});

    fn->append_block(b1);
    b.position_at_end(b1);
    Value* a = b.add_block_param(b1, Type::f64());
    Value* cur_b = b.add_block_param(b1, Type::f64());
    Value* i = b.add_block_param(b1, Type::f64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, b2, {}, b3, {a, cur_b, i});

    fn->append_block(b2);
    b.position_at_end(b2);
    Value* next_b = b.build_add(a, cur_b);
    Value* one_b2 = b.build_fconst_f64(1.0);
    Value* next_i = b.build_add(i, one_b2);
    b.build_br(b1, {cur_b, next_b, next_i});

    fn->append_block(b3);
    b.position_at_end(b3);
    Value* ret_a = b.add_block_param(b3, Type::f64());
    Value* ret_b = b.add_block_param(b3, Type::f64());
    Value* ret_i = b.add_block_param(b3, Type::f64());
    (void)ret_b;
    (void)ret_i;
    b.build_ret(ret_a);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    LoopOptOptions opt;
    opt.enable_unroll = true;
    opt.unroll_factor = 4;
    bool changed = optimize_function_loops(*fn, opt);
    CHECK(changed);
    REQUIRE(verify_function(*fn, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    auto fib_fn = jit.get_function_ptr<double(*)(double)>("fibIter");
    REQUIRE(fib_fn != nullptr);
    CHECK_EQ(fib_fn(0.0), 0.0);
    CHECK_EQ(fib_fn(1.0), 1.0);
    CHECK_EQ(fib_fn(2.0), 1.0);
    CHECK_EQ(fib_fn(10.0), 55.0);
    CHECK_EQ(fib_fn(20.0), 6765.0);
    CHECK_EQ(fib_fn(30.0), 832040.0);
}

TEST_CASE("F64 Demote - Prime Sieve and Count Demotion & JIT Execution") {
    Module mod("test_prime_demote");
    mod.add_symbol_role("host_f64_mod", SymbolRole::FloatRem);
    Builder b(mod);

    Function* fn = mod.create_function("isPrime", Type::f64(), {Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* n = b.add_block_param(b0, Type::f64());

    BasicBlock* b_small = b.create_block("b_small");
    BasicBlock* b_init = b.create_block("b_init");
    BasicBlock* b_loop = b.create_block("b_loop");
    BasicBlock* b_body = b.create_block("b_body");
    BasicBlock* b_div = b.create_block("b_div");
    BasicBlock* b_next = b.create_block("b_next");
    BasicBlock* b_exit = b.create_block("b_exit");

    Value* two_f = b.build_fconst_f64(2.0);
    Value* is_small = b.build_slt(n, two_f);
    b.build_br_if(is_small, b_small, {}, b_init, {});

    fn->append_block(b_small);
    b.position_at_end(b_small);
    b.build_ret(b.build_fconst_f64(0.0));

    fn->append_block(b_init);
    b.position_at_end(b_init);
    b.build_br(b_loop, {two_f});

    fn->append_block(b_loop);
    b.position_at_end(b_loop);
    Value* p = b.add_block_param(b_loop, Type::f64());
    Value* p_sq = b.build_mul(p, p);
    Value* in_range = b.build_sle(p_sq, n);
    b.build_br_if(in_range, b_body, {}, b_exit, {});

    fn->append_block(b_body);
    b.position_at_end(b_body);
    Value* rem = b.build_call("host_f64_mod", Type::f64(), {n, p});
    Value* is_div = b.build_eq(rem, b.build_fconst_f64(0.0));
    b.build_br_if(is_div, b_div, {}, b_next, {});

    fn->append_block(b_div);
    b.position_at_end(b_div);
    b.build_ret(b.build_fconst_f64(0.0));

    fn->append_block(b_next);
    b.position_at_end(b_next);
    Value* next_p = b.build_add(p, b.build_fconst_f64(1.0));
    b.build_br(b_loop, {next_p});

    fn->append_block(b_exit);
    b.position_at_end(b_exit);
    b.build_ret(b.build_fconst_f64(1.0));

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    DemoteStats stats;
    F64DemoteOptions options;
    options.stats = &stats;

    bool demoted = f64_demote_pass(*fn, options);
    CHECK(!demoted);
    REQUIRE(verify_function(*fn, &diag));
    CHECK_EQ(stats.loops_refused, 1u);
    CHECK_EQ(stats.refusal_counts[DemoteRefusalReason::Unprofitable], 1u);

    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("host_f64_mod", reinterpret_cast<void*>(&host_f64_mod));
    REQUIRE(jit.compile_and_load(mod));

    auto is_prime_fn = jit.get_function_ptr<double(*)(double)>("isPrime");
    REQUIRE(is_prime_fn != nullptr);
    CHECK_EQ(is_prime_fn(1.0), 0.0);
    CHECK_EQ(is_prime_fn(2.0), 1.0);
    CHECK_EQ(is_prime_fn(3.0), 1.0);
    CHECK_EQ(is_prime_fn(4.0), 0.0);
    CHECK_EQ(is_prime_fn(29.0), 1.0);
    CHECK_EQ(is_prime_fn(100.0), 0.0);
    CHECK_EQ(is_prime_fn(101.0), 1.0);
}

TEST_CASE("F64 Demote - Matrix 2D Recurrence Nested Loop Demotion & Loop Unrolling") {
    Module mod("test_matrix_recurrence_demote");
    Builder b(mod);

    Function* fn = mod.create_function("recurrence2D", Type::f64(), {Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* n = b.add_block_param(b0, Type::f64());

    BasicBlock* b1 = b.create_block("b1");
    BasicBlock* b2 = b.create_block("b2");
    BasicBlock* b3 = b.create_block("b3");
    BasicBlock* b4 = b.create_block("b4");
    BasicBlock* b5 = b.create_block("b5");
    BasicBlock* b6 = b.create_block("b6");

    Value* zero = b.build_fconst_f64(0.0);
    Value* c1 = b.build_fconst_f64(1.0);
    Value* c3 = b.build_fconst_f64(3.0);
    Value* c7 = b.build_fconst_f64(7.0);
    b.build_br(b1, {zero, zero});

    fn->append_block(b1);
    b.position_at_end(b1);
    Value* acc_i = b.add_block_param(b1, Type::f64());
    Value* i = b.add_block_param(b1, Type::f64());
    Value* cond_i = b.build_slt(i, n);
    b.build_br_if(cond_i, b2, {}, b3, {acc_i, i});

    fn->append_block(b2);
    b.position_at_end(b2);
    b.build_br(b4, {acc_i, zero});

    fn->append_block(b3);
    b.position_at_end(b3);
    Value* ret_acc = b.add_block_param(b3, Type::f64());
    Value* ret_i = b.add_block_param(b3, Type::f64());
    (void)ret_i;
    b.build_ret(ret_acc);

    fn->append_block(b4);
    b.position_at_end(b4);
    Value* acc_j = b.add_block_param(b4, Type::f64());
    Value* j = b.add_block_param(b4, Type::f64());
    Value* cond_j = b.build_slt(j, n);
    b.build_br_if(cond_j, b5, {}, b6, {acc_j, j});

    fn->append_block(b5);
    b.position_at_end(b5);
    Value* t3 = b.build_mul(i, c3);
    Value* t7 = b.build_mul(j, c7);
    Value* sum_t = b.build_add(t3, t7);
    Value* term = b.build_add(sum_t, c1);
    Value* next_acc_j = b.build_add(acc_j, term);
    Value* next_j = b.build_add(j, c1);
    b.build_br(b4, {next_acc_j, next_j});

    fn->append_block(b6);
    b.position_at_end(b6);
    Value* final_acc_j = b.add_block_param(b6, Type::f64());
    Value* final_j = b.add_block_param(b6, Type::f64());
    (void)final_j;
    Value* next_i = b.build_add(i, c1);
    b.build_br(b1, {final_acc_j, next_i});

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    LoopOptOptions opt;
    opt.enable_unroll = true;
    opt.unroll_factor = 4;
    bool changed = optimize_function_loops(*fn, opt);
    CHECK(changed);
    REQUIRE(verify_function(*fn, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    auto rec_fn = jit.get_function_ptr<double(*)(double)>("recurrence2D");
    REQUIRE(rec_fn != nullptr);
    CHECK_EQ(rec_fn(5.0), 525.0);
    CHECK_EQ(rec_fn(10.0), 4600.0);
}

TEST_CASE("F64 Demote - Non-exact Floating Point Division Preserved") {
    Module mod("test_float_div_preserved");
    Builder b(mod);

    Function* fn = mod.create_function("floatDiv", Type::f64(), {Type::f64(), Type::f64()});
    b.set_function(fn);

    BasicBlock* b0 = b.append_block("b0");
    Value* arg0 = b.add_block_param(b0, Type::f64());
    Value* arg1 = b.add_block_param(b0, Type::f64());

    Value* div_res = b.build_sdiv(arg0, arg1);
    b.build_ret(div_res);

    fn->rebuild_cfg_predecessors();

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    // f64_demote must not demote unguarded non-exact division
    f64_demote_pass(*fn);
    REQUIRE(verify_function(*fn, &diag));

    CHECK_EQ(div_res->type(), Type::f64());

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    auto div_fn = jit.get_function_ptr<double(*)(double, double)>("floatDiv");
    REQUIRE(div_fn != nullptr);
    CHECK(std::abs(div_fn(7.0, 3.0) - (7.0 / 3.0)) < 1e-9);
}

namespace {

struct MockTaHeader {
    uint64_t pad0 = 0;
    uint64_t pad1 = 0;
    uint64_t pad2 = 0;
    double* data_ptr = nullptr; // offset 24
    uint64_t byte_len = 0;      // offset 32
};

size_t count_opcode_in_fn(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == op) ++n;
        }
    }
    return n;
}

std::pair<BasicBlock*, Value*> build_ta_get_f64_idx_diamond(
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

BasicBlock* build_ta_set_f64_idx_diamond(
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

TEST_CASE("F64 Demote - Affine i*3+k TypedArray Crunch Loop Demotes, Unswitches, and Fuses FMA") {
    Module mod("test_f64_iv_verlet_crunch");
    Builder b(mod);

    Function* fn = mod.create_function(
        "verlet_step", Type::i32(), {Type::i64(), Type::i64(), Type::i32(), Type::f64()}
    );
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ta_pos = b.add_block_param(entry, Type::i64());
    Value* ta_vel = b.add_block_param(entry, Type::i64());
    Value* n_i32 = b.add_block_param(entry, Type::i32());
    Value* dt = b.add_block_param(entry, Type::f64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* n_f64 = b.build_sitofp_f64_i32(n_i32);
    b.build_br(loop_hdr, {b.build_fconst_f64(0.0)});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* iv_f64 = b.add_block_param(loop_hdr, Type::f64());
    Value* cond = b.build_slt(iv_f64, n_f64);
    b.build_br_if(cond, loop_body, {}, exit_bb, {});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* ix = b.build_mul(iv_f64, b.build_fconst_f64(3.0));
    Value* iy = b.build_add(ix, b.build_fconst_f64(1.0));
    Value* iz = b.build_add(ix, b.build_fconst_f64(2.0));

    auto [bb1, px] = build_ta_get_f64_idx_diamond(b, fn, loop_body, ta_pos, ix, "get_px");
    auto [bb2, vx] = build_ta_get_f64_idx_diamond(b, fn, bb1, ta_vel, ix, "get_vx");
    b.position_at_end(bb2);
    Value* npx = b.build_fadd(b.build_mul(vx, dt), px);
    BasicBlock* bb3 = build_ta_set_f64_idx_diamond(b, fn, bb2, ta_pos, ix, npx, "set_px");

    auto [bb4, py] = build_ta_get_f64_idx_diamond(b, fn, bb3, ta_pos, iy, "get_py");
    auto [bb5, vy] = build_ta_get_f64_idx_diamond(b, fn, bb4, ta_vel, iy, "get_vy");
    b.position_at_end(bb5);
    Value* npy = b.build_fadd(b.build_mul(vy, dt), py);
    BasicBlock* bb6 = build_ta_set_f64_idx_diamond(b, fn, bb5, ta_pos, iy, npy, "set_py");

    auto [bb7, pz] = build_ta_get_f64_idx_diamond(b, fn, bb6, ta_pos, iz, "get_pz");
    auto [bb8, vz] = build_ta_get_f64_idx_diamond(b, fn, bb7, ta_vel, iz, "get_vz");
    b.position_at_end(bb8);
    Value* npz = b.build_fadd(b.build_mul(vz, dt), pz);
    BasicBlock* bb9 = build_ta_set_f64_idx_diamond(b, fn, bb8, ta_pos, iz, npz, "set_pz");

    b.position_at_end(bb9);
    Value* next_iv = b.build_add(iv_f64, b.build_fconst_f64(1.0));
    b.build_br(loop_hdr, {next_iv});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret(b.build_iconst_i32(0));

    fn->rebuild_cfg_predecessors();

    PassPipelineOptions pipe_opts = production_pass_pipeline_options();
    pipe_opts.loop.enable_fma = true;
    REQUIRE(run_pass_pipeline(mod, pipe_opts));

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    // Verify the f64 IV and affine indices were demoted, the fast loop collapsed into
    // header + 1 body block with zero slow-path calls, and scalar FMA fused all 3 updates!
    CHECK_EQ(loop_hdr->param(0)->type(), Type::i64());
    CHECK(count_opcode_in_fn(*fn, Opcode::fma_f64) >= 3);
    DominatorTree dom(*fn);
    LoopAnalysis loops(*fn, dom);
    bool found_fast_loop = false;
    for (LoopInfo* l : loops.post_order_loops()) {
        if (l->header() == loop_hdr) {
            found_fast_loop = true;
            CHECK_EQ(l->blocks().size(), size_t{2});
            for (BasicBlock* lbb : l->blocks()) {
                for (Instruction* inst : *lbb) {
                    CHECK(!is_call(inst->opcode()));
                }
            }
        }
    }
    CHECK(found_fast_loop);

    constexpr size_t kParticles = 32;
    constexpr size_t kLen = kParticles * 3;
    std::vector<double> pos(kLen), vel(kLen), expected(kLen);
    const double dt_val = 0.125;
    for (size_t i = 0; i < kLen; ++i) {
        pos[i] = static_cast<double>(i) * 1.25;
        vel[i] = static_cast<double>(i + 1) * -0.5;
        expected[i] = std::fma(vel[i], dt_val, pos[i]);
    }
    MockTaHeader hdr_pos{0, 0, 0, pos.data(), kLen * sizeof(double)};
    MockTaHeader hdr_vel{0, 0, 0, vel.data(), kLen * sizeof(double)};

    codegen::JitExecutionEngine jit;
    jit.register_external_symbol("mock_ta_slow_get", reinterpret_cast<void*>(+[](int64_t, int64_t) -> int64_t { return 0; }));
    jit.register_external_symbol("mock_ta_slow_set", reinterpret_cast<void*>(+[](int64_t, int64_t, int64_t) -> int64_t { return 0; }));
    REQUIRE(jit.compile_and_load(mod));
    using VerletFn = int32_t(*)(MockTaHeader*, MockTaHeader*, int32_t, double);
    auto fn_ptr = jit.get_function_ptr<VerletFn>("verlet_step");
    REQUIRE(fn_ptr != nullptr);

    CHECK_EQ(fn_ptr(&hdr_pos, &hdr_vel, static_cast<int32_t>(kParticles), dt_val), 0);
    for (size_t i = 0; i < kLen; ++i) {
        CHECK_EQ(pos[i], expected[i]);
    }
}

TEST_CASE("F64 Demote - Counted i32 IV With Lifted f64 Affine Index i*3+k Demotes, Unswitches, and Fuses FMA") {
    Module mod("test_i32_iv_lifted_f64_affine_ta");
    Builder b(mod);

    // Mirrors bronze LoopCounted lowering where i is already an i32 loop IV,
    // lifted via sitofp_f64_i32(i) for affine index arithmetic (i * 3.0 + k):
    Function* fn = mod.create_function(
        "verlet_i32_iv", Type::i32(), {Type::i64(), Type::i64(), Type::i32(), Type::f64()}
    );
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ta_pos = b.add_block_param(entry, Type::i64());
    Value* ta_vel = b.add_block_param(entry, Type::i64());
    Value* n_i32 = b.add_block_param(entry, Type::i32());
    Value* dt = b.add_block_param(entry, Type::f64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    b.build_br(loop_hdr, {b.build_iconst_i32(0)});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* iv_i32 = b.add_block_param(loop_hdr, Type::i32());
    Value* cond = b.build_slt(iv_i32, n_i32);
    b.build_br_if(cond, loop_body, {}, exit_bb, {});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* iv_f64 = b.build_sitofp_f64_i32(iv_i32);
    Value* ix = b.build_mul(iv_f64, b.build_fconst_f64(3.0));
    Value* iy = b.build_add(ix, b.build_fconst_f64(1.0));
    Value* iz = b.build_add(ix, b.build_fconst_f64(2.0));

    auto [bb1, px] = build_ta_get_f64_idx_diamond(b, fn, loop_body, ta_pos, ix, "get_px");
    auto [bb2, vx] = build_ta_get_f64_idx_diamond(b, fn, bb1, ta_vel, ix, "get_vx");
    b.position_at_end(bb2);
    Value* npx = b.build_fadd(b.build_mul(vx, dt), px);
    BasicBlock* bb3 = build_ta_set_f64_idx_diamond(b, fn, bb2, ta_pos, ix, npx, "set_px");

    auto [bb4, py] = build_ta_get_f64_idx_diamond(b, fn, bb3, ta_pos, iy, "get_py");
    auto [bb5, vy] = build_ta_get_f64_idx_diamond(b, fn, bb4, ta_vel, iy, "get_vy");
    b.position_at_end(bb5);
    Value* npy = b.build_fadd(b.build_mul(vy, dt), py);
    BasicBlock* bb6 = build_ta_set_f64_idx_diamond(b, fn, bb5, ta_pos, iy, npy, "set_py");

    auto [bb7, pz] = build_ta_get_f64_idx_diamond(b, fn, bb6, ta_pos, iz, "get_pz");
    auto [bb8, vz] = build_ta_get_f64_idx_diamond(b, fn, bb7, ta_vel, iz, "get_vz");
    b.position_at_end(bb8);
    Value* npz = b.build_fadd(b.build_mul(vz, dt), pz);
    BasicBlock* bb9 = build_ta_set_f64_idx_diamond(b, fn, bb8, ta_pos, iz, npz, "set_pz");

    b.position_at_end(bb9);
    Value* next_iv = b.build_add(iv_i32, b.build_iconst_i32(1));
    b.build_br(loop_hdr, {next_iv});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    b.build_ret(b.build_iconst_i32(0));

    fn->rebuild_cfg_predecessors();

    PassPipelineOptions pipe_opts = production_pass_pipeline_options();
    pipe_opts.loop.enable_fma = true;
    REQUIRE(run_pass_pipeline(mod, pipe_opts));

    DiagnosticReporter diag;
    REQUIRE(verify_function(*fn, &diag));

    CHECK(count_opcode_in_fn(*fn, Opcode::fma_f64) >= 3);
    DominatorTree dom(*fn);
    LoopAnalysis loops(*fn, dom);
    bool found_fast_loop = false;
    for (LoopInfo* l : loops.post_order_loops()) {
        if (l->header() == loop_hdr) {
            found_fast_loop = true;
            CHECK_EQ(l->blocks().size(), size_t{2});
            for (BasicBlock* lbb : l->blocks()) {
                for (Instruction* inst : *lbb) {
                    CHECK(!is_call(inst->opcode()));
                }
            }
        }
    }
    CHECK(found_fast_loop);

    constexpr size_t kParticles = 32;
    constexpr size_t kLen = kParticles * 3;
    std::vector<double> pos(kLen), vel(kLen), expected(kLen);
    const double dt_val = 0.25;
    for (size_t i = 0; i < kLen; ++i) {
        pos[i] = static_cast<double>(i) * 0.75;
        vel[i] = static_cast<double>(i + 1) * 1.5;
        expected[i] = std::fma(vel[i], dt_val, pos[i]);
    }
    MockTaHeader hdr_pos{0, 0, 0, pos.data(), kLen * sizeof(double)};
    MockTaHeader hdr_vel{0, 0, 0, vel.data(), kLen * sizeof(double)};

    codegen::JitExecutionEngine jit;
    jit.register_external_symbol("mock_ta_slow_get", reinterpret_cast<void*>(+[](int64_t, int64_t) -> int64_t { return 0; }));
    jit.register_external_symbol("mock_ta_slow_set", reinterpret_cast<void*>(+[](int64_t, int64_t, int64_t) -> int64_t { return 0; }));
    REQUIRE(jit.compile_and_load(mod));
    using VerletFn = int32_t(*)(MockTaHeader*, MockTaHeader*, int32_t, double);
    auto fn_ptr = jit.get_function_ptr<VerletFn>("verlet_i32_iv");
    REQUIRE(fn_ptr != nullptr);

    CHECK_EQ(fn_ptr(&hdr_pos, &hdr_vel, static_cast<int32_t>(kParticles), dt_val), 0);
    for (size_t i = 0; i < kLen; ++i) {
        CHECK_EQ(pos[i], expected[i]);
    }
}

