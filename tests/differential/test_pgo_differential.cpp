#include "test_framework.hpp"
#include "diff_harness.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/module.hpp>
#include <brass/pgo/profile_data.hpp>
#include <brass/pgo/instrument.hpp>
#include <brass/pgo/pgo_opt.hpp>
#include <brass/mir/branch_probability.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::test;

TEST_CASE("Differential PGO - Knuth-Stevenson Instrumentation and JIT Profiling") {
    // Build a function with a conditional branch:
    // classify(x):
    //   if x > 0 -> hot_bb: (x * 3) ^ 5
    //   else     -> cold_bb: x - 100
    //   exit(res): ret res

    auto build_module = [](const std::string& name) {
        Module m(name);
        Function* fn = m.create_function("classify", Type::i64(), {Type::i64()});
        Builder b(m);
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        BasicBlock* hot_bb = b.append_block("hot_bb");
        BasicBlock* cold_bb = b.append_block("cold_bb");
        BasicBlock* exit_bb = b.append_block("exit");

        Value* x = b.add_block_param(entry, Type::i64());
        Value* res_param = b.add_block_param(exit_bb, Type::i64());

        // entry
        b.position_at_end(entry);
        Value* zero = b.build_iconst_i64(0);
        Value* cond = b.build_sgt(x, zero);
        b.build_br_if(cond, hot_bb, {}, cold_bb, {});

        // hot_bb
        b.position_at_end(hot_bb);
        Value* c3 = b.build_iconst_i64(3);
        Value* mul = b.build_mul(x, c3);
        Value* c5 = b.build_iconst_i64(5);
        Value* hot_res = b.build_xor(mul, c5);
        b.build_br(exit_bb, {hot_res});

        // cold_bb
        b.position_at_end(cold_bb);
        Value* c100 = b.build_iconst_i64(100);
        Value* cold_res = b.build_sub(x, c100);
        b.build_br(exit_bb, {cold_res});

        // exit_bb
        b.position_at_end(exit_bb);
        b.build_ret(res_param);

        fn->rebuild_cfg_predecessors();
        return m;
    };

    Module mod_ref = build_module("mod_ref");
    Module mod_inst = build_module("mod_inst");

    REQUIRE(verify_module(mod_ref));
    REQUIRE(verify_module(mod_inst));

    // Instrument mod_inst
    auto inst_res = pgo::instrument_module(mod_inst);
    REQUIRE(verify_module(mod_inst));
    CHECK(inst_res.metadata.total_counters > 0);

    // Initialize runtime counters
    pgo::brass_pgo_init_counters(inst_res.metadata.total_counters);

    // Setup JIT with brass_pgo_inc symbol registered
    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));
    REQUIRE(jit.compile_and_load(mod_inst));

    Interpreter interp;

    // Run 50 positive values and 2 negative values
    for (int64_t v = 1; v <= 50; ++v) {
        RuntimeValue ref_res = interp.run(mod_ref, "classify", {RuntimeValue::from_i64(v)});
        RuntimeValue jit_res = jit.invoke("classify", {RuntimeValue::from_i64(v)});
        CHECK_EQ(ref_res.as_i64(), jit_res.as_i64());
    }
    for (int64_t v = -2; v <= -1; ++v) {
        RuntimeValue ref_res = interp.run(mod_ref, "classify", {RuntimeValue::from_i64(v)});
        RuntimeValue jit_res = jit.invoke("classify", {RuntimeValue::from_i64(v)});
        CHECK_EQ(ref_res.as_i64(), jit_res.as_i64());
    }

    // Inspect counters
    size_t num_counters = 0;
    const uint64_t* counters = pgo::brass_pgo_get_counters(&num_counters);
    REQUIRE(counters != nullptr);
    REQUIRE(num_counters >= inst_res.metadata.total_counters);

    // Entry count should be 52
    const auto* fn_meta = inst_res.metadata.find_function("classify");
    REQUIRE(fn_meta != nullptr);
    REQUIRE(fn_meta->entry_counter_index < num_counters);
    CHECK_EQ(counters[fn_meta->entry_counter_index], 52ULL);

    // Dump profile to file
    const std::string tmp_bprof = "test_pgo_diff1.bprof";
    bool dumped = pgo::brass_pgo_dump(tmp_bprof.c_str(), counters, num_counters, inst_res.metadata);
    CHECK(dumped);

    // Read back profile
    std::string err_msg;
    auto read_prof = pgo::ProfileData::read_from_file(tmp_bprof, &err_msg);
    CHECK(read_prof != nullptr);
    if (read_prof) {
        CHECK_EQ(read_prof->module_name(), "mod_inst");
        const auto* fp = read_prof->find_function("classify");
        REQUIRE(fp != nullptr);
        CHECK_EQ(fp->entry_count, 52ULL);

        // Branch probability analysis on the read profile
        Function* ref_fn = mod_ref.get_function("classify");
        REQUIRE(ref_fn != nullptr);
        mir::BranchProbabilityAnalysis bpa(*ref_fn, *fp);
        const auto& bfi = bpa.block_frequency_info();
        const auto& bpi = bpa.branch_probability_info();

        CHECK_EQ(bfi.entry_count(), 52ULL);
        BasicBlock* entry_bb = ref_fn->entry_block();
        REQUIRE(entry_bb != nullptr);
        CHECK_EQ(bfi.get_block_count(entry_bb), 52ULL);

        // Check that hot branch has high probability and cold branch has low probability
        BasicBlock* hot_succ = nullptr;
        BasicBlock* cold_succ = nullptr;
        for (BasicBlock* s : entry_bb->successors()) {
            if (s->name() == "hot_bb") hot_succ = s;
            if (s->name() == "cold_bb") cold_succ = s;
        }
        REQUIRE(hot_succ != nullptr);
        REQUIRE(cold_succ != nullptr);

        double hot_prob = bpi.get_edge_probability(entry_bb, hot_succ);
        double cold_prob = bpi.get_edge_probability(entry_bb, cold_succ);

        CHECK(hot_prob > 0.9);
        CHECK(cold_prob < 0.1);
        CHECK(std::abs((hot_prob + cold_prob) - 1.0) < 1e-4);
    }

    std::remove(tmp_bprof.c_str());
}

TEST_CASE("Differential PGO - PGO Loop Optimization & Trace Layout Equivalence") {
    // Loop sum_squares(n):
    // sum = 0, i = 0
    // while i < n:
    //   sum += i * i
    //   i++
    // return sum

    auto build_loop_mod = [](const std::string& name) {
        Module m(name);
        Function* fn = m.create_function("sum_squares", Type::i64(), {Type::i64()});
        Builder b(m);
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        BasicBlock* loop_header = b.append_block("loop_header");
        BasicBlock* loop_body = b.append_block("loop_body");
        BasicBlock* exit_bb = b.append_block("exit");

        Value* n = b.add_block_param(entry, Type::i64());
        Value* i_phi = b.add_block_param(loop_header, Type::i64());
        Value* sum_phi = b.add_block_param(loop_header, Type::i64());

        // entry
        b.position_at_end(entry);
        Value* zero = b.build_iconst_i64(0);
        b.build_br(loop_header, {zero, zero});

        // loop_header
        b.position_at_end(loop_header);
        Value* cmp = b.build_slt(i_phi, n);
        b.build_br_if(cmp, loop_body, {}, exit_bb, {});

        // loop_body
        b.position_at_end(loop_body);
        Value* sq = b.build_mul(i_phi, i_phi);
        Value* next_sum = b.build_add(sum_phi, sq);
        Value* one = b.build_iconst_i64(1);
        Value* next_i = b.build_add(i_phi, one);
        b.build_br(loop_header, {next_i, next_sum});

        // exit
        b.position_at_end(exit_bb);
        b.build_ret(sum_phi);

        fn->rebuild_cfg_predecessors();
        return m;
    };

    Module mod_ref = build_loop_mod("mod_ref");
    Module mod_opt = build_loop_mod("mod_opt");

    REQUIRE(verify_module(mod_ref));
    REQUIRE(verify_module(mod_opt));

    // Construct profile: function invoked 100 times with high trip count
    pgo::ProfileData profile;
    profile.set_module_name("mod_opt");
    pgo::FunctionProfile fp;
    fp.name = "sum_squares";
    fp.entry_count = 100;
    // Edge counters for chords: loop back-edge chord has high count
    fp.edge_counters = {10000, 100};
    profile.add_function(std::move(fp));

    // Optimize mod_opt with PGO pipeline
    bool opt_ok = pgo::optimize_module_pgo(mod_opt, profile);
    CHECK(opt_ok);
    REQUIRE(verify_module(mod_opt));

    Interpreter interp;
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod_opt));

    // Differential test across multiple trip counts
    for (int64_t n = 0; n <= 40; ++n) {
        RuntimeValue ref_res = interp.run(mod_ref, "sum_squares", {RuntimeValue::from_i64(n)});
        RuntimeValue jit_res = jit.invoke("sum_squares", {RuntimeValue::from_i64(n)});
        CHECK_EQ(ref_res.as_i64(), jit_res.as_i64());
    }
}

TEST_CASE("Differential PGO - Hot Inlining and Cold Suppression Differential") {
    // caller(x, flag):
    //   if flag > 0:
    //     return hot_callee(x)
    //   else:
    //     return cold_callee(x)
    //
    // hot_callee(x): return (x * 7) + 11
    // cold_callee(x): return (x ^ 13) - 17

    auto build_inlining_module = [](const std::string& name) {
        Module m(name);

        Function* hot_fn = m.create_function("hot_callee", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(hot_fn);
            BasicBlock* bb = b.append_block("entry");
            Value* x = b.add_block_param(bb, Type::i64());
            Value* c7 = b.build_iconst_i64(7);
            Value* mul = b.build_mul(x, c7);
            Value* c11 = b.build_iconst_i64(11);
            b.build_ret(b.build_add(mul, c11));
            hot_fn->rebuild_cfg_predecessors();
        }

        Function* cold_fn = m.create_function("cold_callee", Type::i64(), {Type::i64()});
        {
            Builder b(m);
            b.set_function(cold_fn);
            BasicBlock* bb = b.append_block("entry");
            Value* x = b.add_block_param(bb, Type::i64());
            Value* c13 = b.build_iconst_i64(13);
            Value* xor_v = b.build_xor(x, c13);
            Value* c17 = b.build_iconst_i64(17);
            b.build_ret(b.build_sub(xor_v, c17));
            cold_fn->rebuild_cfg_predecessors();
        }

        Function* caller_fn = m.create_function("caller", Type::i64(), {Type::i64(), Type::i64()});
        {
            Builder b(m);
            b.set_function(caller_fn);
            BasicBlock* entry = b.append_block("entry");
            BasicBlock* take_hot = b.append_block("take_hot");
            BasicBlock* take_cold = b.append_block("take_cold");

            Value* x = b.add_block_param(entry, Type::i64());
            Value* flag = b.add_block_param(entry, Type::i64());

            b.position_at_end(entry);
            Value* zero = b.build_iconst_i64(0);
            Value* cond = b.build_sgt(flag, zero);
            b.build_br_if(cond, take_hot, {}, take_cold, {});

            b.position_at_end(take_hot);
            Value* r_hot = b.build_call("hot_callee", Type::i64(), {x});
            b.build_ret(r_hot);

            b.position_at_end(take_cold);
            Value* r_cold = b.build_call("cold_callee", Type::i64(), {x});
            b.build_ret(r_cold);

            caller_fn->rebuild_cfg_predecessors();
        }

        return m;
    };

    Module mod_ref = build_inlining_module("mod_ref");
    Module mod_opt = build_inlining_module("mod_opt");

    REQUIRE(verify_module(mod_ref));
    REQUIRE(verify_module(mod_opt));

    // Profile: caller executed 10,000 times: hot_callee called 9,990 times, cold_callee called 10 times
    pgo::ProfileData profile;
    profile.set_module_name("mod_opt");

    pgo::FunctionProfile fp_caller;
    fp_caller.name = "caller";
    fp_caller.entry_count = 10000;
    fp_caller.edge_counters = {9990, 10};
    profile.add_function(std::move(fp_caller));

    pgo::FunctionProfile fp_hot;
    fp_hot.name = "hot_callee";
    fp_hot.entry_count = 9990;
    profile.add_function(std::move(fp_hot));

    pgo::FunctionProfile fp_cold;
    fp_cold.name = "cold_callee";
    fp_cold.entry_count = 10;
    profile.add_function(std::move(fp_cold));

    // Run PGO optimization
    bool opt_ok = pgo::optimize_module_pgo(mod_opt, profile);
    CHECK(opt_ok);
    REQUIRE(verify_module(mod_opt));

    // Verify hot_callee was inlined into caller (no call to hot_callee remains in caller)
    Function* opt_caller = mod_opt.get_function("caller");
    REQUIRE(opt_caller != nullptr);
    bool found_hot_call = false;
    for (const BasicBlock* bb : opt_caller->blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call && inst->symbol() == "hot_callee") {
                found_hot_call = true;
            }
        }
    }
    CHECK(!found_hot_call);

    // Differential test between reference interpreter and optimized JIT
    Interpreter interp;
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod_opt));

    for (int64_t x = -20; x <= 20; ++x) {
        for (int64_t flag : {-1, 0, 1, 5}) {
            RuntimeValue ref_res = interp.run(mod_ref, "caller", {RuntimeValue::from_i64(x), RuntimeValue::from_i64(flag)});
            RuntimeValue jit_res = jit.invoke("caller", {RuntimeValue::from_i64(x), RuntimeValue::from_i64(flag)});
            CHECK_EQ(ref_res.as_i64(), jit_res.as_i64());
        }
    }
}

TEST_CASE("Differential PGO - Nested Diamond Flow Conservation and Differential") {
    // nested_diamond(a, b):
    //   if a > 0:
    //     if b > 0: return a + b
    //     else:     return a - b
    //   else:
    //     if b > 0: return b - a
    //     else:     return -(a + b)

    auto build_nested_module = [](const std::string& name) {
        Module m(name);
        Function* fn = m.create_function("nested_diamond", Type::i64(), {Type::i64(), Type::i64()});
        Builder b(m);
        b.set_function(fn);

        BasicBlock* entry = b.append_block("entry");
        BasicBlock* a_pos = b.append_block("a_pos");
        BasicBlock* a_neg = b.append_block("a_neg");
        BasicBlock* pp = b.append_block("pp");
        BasicBlock* pn = b.append_block("pn");
        BasicBlock* np = b.append_block("np");
        BasicBlock* nn = b.append_block("nn");
        BasicBlock* exit_bb = b.append_block("exit");

        Value* a = b.add_block_param(entry, Type::i64());
        Value* b_val = b.add_block_param(entry, Type::i64());
        Value* res = b.add_block_param(exit_bb, Type::i64());

        // entry
        b.position_at_end(entry);
        Value* zero = b.build_iconst_i64(0);
        Value* cond_a = b.build_sgt(a, zero);
        b.build_br_if(cond_a, a_pos, {}, a_neg, {});

        // a_pos
        b.position_at_end(a_pos);
        Value* cond_b1 = b.build_sgt(b_val, zero);
        b.build_br_if(cond_b1, pp, {}, pn, {});

        // a_neg
        b.position_at_end(a_neg);
        Value* cond_b2 = b.build_sgt(b_val, zero);
        b.build_br_if(cond_b2, np, {}, nn, {});

        // pp: a + b
        b.position_at_end(pp);
        Value* v_pp = b.build_add(a, b_val);
        b.build_br(exit_bb, {v_pp});

        // pn: a - b
        b.position_at_end(pn);
        Value* v_pn = b.build_sub(a, b_val);
        b.build_br(exit_bb, {v_pn});

        // np: b - a
        b.position_at_end(np);
        Value* v_np = b.build_sub(b_val, a);
        b.build_br(exit_bb, {v_np});

        // nn: -(a + b)
        b.position_at_end(nn);
        Value* sum_nn = b.build_add(a, b_val);
        Value* v_nn = b.build_sub(zero, sum_nn);
        b.build_br(exit_bb, {v_nn});

        // exit
        b.position_at_end(exit_bb);
        b.build_ret(res);

        fn->rebuild_cfg_predecessors();
        return m;
    };

    Module mod_ref = build_nested_module("nested_ref");
    Module mod_inst = build_nested_module("nested_inst");
    Module mod_opt = build_nested_module("nested_opt");

    REQUIRE(verify_module(mod_ref));
    REQUIRE(verify_module(mod_inst));
    REQUIRE(verify_module(mod_opt));

    auto inst_res = pgo::instrument_module(mod_inst);
    REQUIRE(verify_module(mod_inst));
    pgo::brass_pgo_init_counters(inst_res.metadata.total_counters);

    codegen::JitExecutionEngine jit_inst(Target::host());
    jit_inst.register_external_symbol("brass_pgo_inc", reinterpret_cast<void*>(&brass_pgo_inc));
    REQUIRE(jit_inst.compile_and_load(mod_inst));

    // Profile training run: heavily bias towards quadrant 1 (a > 0, b > 0)
    for (int i = 0; i < 80; ++i) {
        jit_inst.invoke("nested_diamond", {RuntimeValue::from_i64(5), RuntimeValue::from_i64(10)});
    }
    for (int i = 0; i < 15; ++i) {
        jit_inst.invoke("nested_diamond", {RuntimeValue::from_i64(5), RuntimeValue::from_i64(-10)});
    }
    for (int i = 0; i < 4; ++i) {
        jit_inst.invoke("nested_diamond", {RuntimeValue::from_i64(-5), RuntimeValue::from_i64(10)});
    }
    for (int i = 0; i < 1; ++i) {
        jit_inst.invoke("nested_diamond", {RuntimeValue::from_i64(-5), RuntimeValue::from_i64(-10)});
    }

    size_t num_counters = 0;
    const uint64_t* counters = pgo::brass_pgo_get_counters(&num_counters);
    REQUIRE(counters != nullptr);

    const std::string bprof_nested = "test_pgo_nested.bprof";
    CHECK(pgo::brass_pgo_dump(bprof_nested.c_str(), counters, num_counters, inst_res.metadata));

    auto profile = pgo::ProfileData::read_from_file(bprof_nested);
    REQUIRE(profile != nullptr);

    const auto* fp = profile->find_function("nested_diamond");
    REQUIRE(fp != nullptr);
    CHECK_EQ(fp->entry_count, 100ULL);

    Function* ref_fn = mod_ref.get_function("nested_diamond");
    REQUIRE(ref_fn != nullptr);
    mir::BranchProbabilityAnalysis bpa(*ref_fn, *fp);
    const auto& bfi = bpa.block_frequency_info();
    const auto& bpi = bpa.branch_probability_info();

    // Verify Kirchhoff flow conservation on all internal blocks
    for (const BasicBlock* bb : ref_fn->blocks()) {
        if (!bb || bb->successors().empty()) continue;
        uint64_t out_flow = 0;
        for (const BasicBlock* s : bb->successors()) {
            out_flow += bpi.get_edge_count(bb, s);
        }
        CHECK_EQ(bfi.get_block_count(bb), out_flow);
    }

    // Optimize mod_opt with profile
    CHECK(pgo::optimize_module_pgo(mod_opt, *profile));
    REQUIRE(verify_module(mod_opt));

    // Differential test across 4 quadrants
    Interpreter interp;
    codegen::JitExecutionEngine jit_opt(Target::host());
    REQUIRE(jit_opt.compile_and_load(mod_opt));

    for (int64_t a_val : {-10, -1, 0, 1, 10}) {
        for (int64_t b_val : {-10, -1, 0, 1, 10}) {
            RuntimeValue ref_res = interp.run(mod_ref, "nested_diamond", {RuntimeValue::from_i64(a_val), RuntimeValue::from_i64(b_val)});
            RuntimeValue jit_res = jit_opt.invoke("nested_diamond", {RuntimeValue::from_i64(a_val), RuntimeValue::from_i64(b_val)});
            CHECK_EQ(ref_res.as_i64(), jit_res.as_i64());
        }
    }

    std::remove(bprof_nested.c_str());
}

