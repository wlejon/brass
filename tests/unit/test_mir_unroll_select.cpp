#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/select_opt.hpp>
#include <brass/mir/loop_unroll.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <cmath>
#include <vector>

using namespace brass;

TEST_CASE("Select - Verification and Roundtrip") {
    Module mod("test_select_roundtrip");
    Builder b(mod);

    Function* fn = mod.create_function("test_sel", Type::i64(), {Type::i32(), Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* c = b.add_block_param(entry, Type::i32());
    Value* t = b.add_block_param(entry, Type::i64());
    Value* f = b.add_block_param(entry, Type::i64());

    Value* sel = b.build_select(c, t, f);
    b.build_ret(sel);

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    std::string printed = to_string(mod);
    DiagnosticReporter p_diag;
    auto parsed_mod = parse_module(printed, &p_diag);
    if (!parsed_mod || p_diag.has_errors()) {
        std::cerr << "Roundtrip failed!\nPrinted:\n" << printed << "\nDiag:\n" << p_diag.format_all() << "\n";
    }
    REQUIRE(parsed_mod != nullptr);
    REQUIRE(verify_module(*parsed_mod, &p_diag));
}

TEST_CASE("Select - JIT Branchless Execution (i64 & f64)") {
    Module mod("test_select_exec");
    Builder b(mod);

    Function* fn_i64 = mod.create_function("sel_i64", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    b.set_function(fn_i64);
    BasicBlock* entry_i = b.append_block("entry");
    b.position_at_end(entry_i);
    Value* cond_val = b.add_block_param(entry_i, Type::i64());
    Value* t_val = b.add_block_param(entry_i, Type::i64());
    Value* f_val = b.add_block_param(entry_i, Type::i64());
    Value* cmp_i = b.build_slt(cond_val, b.build_iconst_i64(10));
    Value* res_i = b.build_select(cmp_i, t_val, f_val);
    b.build_ret(res_i);

    Function* fn_f64 = mod.create_function("sel_f64", Type::f64(), {Type::f64(), Type::f64(), Type::f64()});
    b.set_function(fn_f64);
    BasicBlock* entry_f = b.append_block("entry");
    b.position_at_end(entry_f);
    Value* fcond = b.add_block_param(entry_f, Type::f64());
    Value* ft = b.add_block_param(entry_f, Type::f64());
    Value* ff = b.add_block_param(entry_f, Type::f64());
    Value* cmp_f = b.build_slt(fcond, b.build_fconst_f64(0.0));
    Value* res_f = b.build_select(cmp_f, ft, ff);
    b.build_ret(res_f);

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));

    auto ptr_i = jit.get_function_ptr<int64_t(*)(int64_t, int64_t, int64_t)>("sel_i64");
    REQUIRE(ptr_i != nullptr);
    CHECK_EQ(ptr_i(5, 100, 200), 100);
    CHECK_EQ(ptr_i(15, 100, 200), 200);

    auto ptr_f = jit.get_function_ptr<double(*)(double, double, double)>("sel_f64");
    REQUIRE(ptr_f != nullptr);
    CHECK_EQ(ptr_f(-2.5, 3.14, 2.71), 3.14);
    CHECK_EQ(ptr_f(2.5, 3.14, 2.71), 2.71);
}

TEST_CASE("Diamond to Select - Optimization Pass Transformation") {
    Module mod("test_diamond_opt");
    Builder b(mod);

    Function* fn = mod.create_function("abs_diff", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* bb_then = b.create_block("bb_then");
    BasicBlock* bb_else = b.create_block("bb_else");
    BasicBlock* bb_join = b.create_block("bb_join");

    b.position_at_end(entry);
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());
    Value* is_gt = b.build_sgt(x, y);
    b.build_br_if(is_gt, bb_then, {}, bb_else, {});

    fn->append_block(bb_then);
    b.position_at_end(bb_then);
    Value* d_pos = b.build_sub(x, y);
    b.build_br(bb_join, {d_pos});

    fn->append_block(bb_else);
    b.position_at_end(bb_else);
    Value* d_neg = b.build_sub(y, x);
    b.build_br(bb_join, {d_neg});

    fn->append_block(bb_join);
    b.position_at_end(bb_join);
    Value* diff = b.add_block_param(bb_join, Type::i64());
    b.build_ret(diff);

    fn->rebuild_cfg_predecessors();

    CHECK_EQ(fn->block_count(), 4);
    bool changed = simplify_cfg_diamonds(*fn);
    CHECK(changed);
    CHECK_EQ(fn->block_count(), 2);

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    auto abs_diff_fn = jit.get_function_ptr<int64_t(*)(int64_t, int64_t)>("abs_diff");
    REQUIRE(abs_diff_fn != nullptr);

    CHECK_EQ(abs_diff_fn(10, 4), 6);
    CHECK_EQ(abs_diff_fn(4, 10), 6);
    CHECK_EQ(abs_diff_fn(7, 7), 0);
}

TEST_CASE("Loop Unrolling - Reduction Accumulator with Remainder Loops") {
    Module mod("test_unroll_reduction");
    Builder b(mod);

    Function* fn = mod.create_function("sum_array_f64", Type::f64(), {Type::ptr(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* arr = b.add_block_param(entry, Type::ptr());
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero_i = b.build_iconst_i64(0);
    Value* zero_f = b.build_fconst_f64(0.0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_hdr, {zero_i, zero_f});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::f64());
    Value* cond = b.build_slt(i, n);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* elem = b.build_load_indexed(Type::f64(), arr, i, 8, 0);
    Value* next_acc = b.build_add(acc, elem);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* res = b.add_block_param(exit_bb, Type::f64());
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();

    bool changed = optimize_function_loops(*fn);
    CHECK(changed);

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    auto sum_fn = jit.get_function_ptr<double(*)(const double*, int64_t)>("sum_array_f64");
    REQUIRE(sum_fn != nullptr);

    for (int64_t count : {0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 64, 100}) {
        std::vector<double> vals(static_cast<size_t>(count + 4));
        double expected = 0.0;
        for (size_t k = 0; k < static_cast<size_t>(count); ++k) {
            vals[k] = static_cast<double>(k + 1) * 1.5;
            expected += vals[k];
        }

        double got = sum_fn(vals.data(), count);
        CHECK(std::abs(got - expected) < 1e-9);
    }
}
