#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/escape_analysis.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <vector>

using namespace brass;

namespace {

size_t count_opcodes(const Function& fn, Opcode op) {
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == op) count++;
        }
    }
    return count;
}

size_t count_allocations(const Function& fn) {
    size_t count = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (inst && inst->opcode() == Opcode::call && is_allocation_callee(inst->symbol())) {
                count++;
            }
        }
    }
    return count;
}

} // namespace

TEST_CASE("SROA - 3D Vector temporary object allocation completely dissolved into scalars") {
    Module mod("test_3d_vector");
    Builder b(mod);

    // func @vector_length_sq(%x: f64, %y: f64, %z: f64) -> f64
    Function* fn = mod.create_function("vector_length_sq", Type::f64(), {Type::f64(), Type::f64(), Type::f64()});
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");

    Value* x = b.add_block_param(entry, Type::f64());
    Value* y = b.add_block_param(entry, Type::f64());
    Value* z = b.add_block_param(entry, Type::f64());

    Value* sz24 = b.build_iconst_i64(24);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);

    // Allocate 3D vector object
    Value* vec = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag});

    // Store x, y, z
    b.build_store(Type::f64(), vec, 0, x);
    b.build_store(Type::f64(), vec, 8, y);
    b.build_store(Type::f64(), vec, 16, z);

    // Load x, y, z
    Value* rx = b.build_load(Type::f64(), vec, 0);
    Value* ry = b.build_load(Type::f64(), vec, 8);
    Value* rz = b.build_load(Type::f64(), vec, 16);

    // x*x + y*y + z*z
    Value* xx = b.build_mul(rx, rx);
    Value* yy = b.build_mul(ry, ry);
    Value* zz = b.build_mul(rz, rz);
    Value* xxyy = b.build_add(xx, yy);
    Value* res = b.build_add(xxyy, zz);
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Escape Analysis check
    EscapeAnalysis ea(*fn);
    CHECK_EQ(ea.allocations().size(), 1ULL);
    CHECK_EQ(ea.non_escaping_allocations().size(), 1ULL);
    CHECK_EQ(ea.get_escape_state(vec), EscapeState::NoEscape);
    CHECK(!ea.does_escape(vec));

    // Run SROA
    SroaStats stats;
    SroaOptions sroa_opts;
    sroa_opts.stats = &stats;
    bool changed = sroa_function(*fn, sroa_opts);
    CHECK(changed);
    CHECK_EQ(stats.allocations_eliminated, 1ULL);
    CHECK_EQ(stats.loads_eliminated, 3ULL);
    CHECK_EQ(stats.stores_eliminated, 3ULL);

    // Verify all memory operations and allocations are completely gone
    CHECK_EQ(count_allocations(*fn), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::load), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::store), 0ULL);
    REQUIRE(verify_function(*fn));

    // Execute in Interpreter: 3.0^2 + 4.0^2 + 12.0^2 = 9 + 16 + 144 = 169.0
    Interpreter interp;
    RuntimeValue iv = interp.run(*fn, {RuntimeValue::from_f64(3.0), RuntimeValue::from_f64(4.0), RuntimeValue::from_f64(12.0)});
    CHECK_EQ(iv.as_f64(), 169.0);
}

TEST_CASE("SROA - Field write-then-read forwarding within single block") {
    Module mod("test_forwarding");
    Builder b(mod);

    Function* fn = mod.create_function("forwarding_test", Type::i64(), {});
    b.set_function(fn);
    b.append_block("entry");

    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);

    Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});

    Value* v10 = b.build_iconst_i64(10);
    b.build_store(Type::i64(), obj, 0, v10);
    Value* r1 = b.build_load(Type::i64(), obj, 0);

    Value* v20 = b.build_iconst_i64(20);
    b.build_store(Type::i64(), obj, 0, v20);
    Value* r2 = b.build_load(Type::i64(), obj, 0);

    Value* sum = b.build_add(r1, r2);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    SroaStats stats;
    SroaOptions sroa_opts;
    sroa_opts.stats = &stats;
    bool changed = sroa_function(*fn, sroa_opts);
    CHECK(changed);
    CHECK_EQ(stats.allocations_eliminated, 1ULL);
    CHECK_EQ(stats.loads_eliminated, 2ULL);
    CHECK_EQ(stats.stores_eliminated, 2ULL);
    CHECK_EQ(count_allocations(*fn), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::load), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::store), 0ULL);
    REQUIRE(verify_function(*fn));

    Interpreter interp;
    RuntimeValue iv = interp.run(*fn, {});
    CHECK_EQ(iv.as_i64(), 30);
}

TEST_CASE("SROA - Conditional field update across if-then-else with block parameter phi insertion") {
    Module mod("test_cond_sroa");
    Builder b(mod);

    // func @cond_update(%cond: i32, %v: i64) -> i64
    Function* fn = mod.create_function("cond_update", Type::i64(), {Type::i32(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* then_bb = b.append_block("then_bb");
    BasicBlock* else_bb = b.append_block("else_bb");
    BasicBlock* merge_bb = b.append_block("merge_bb");

    Value* cond = b.add_block_param(entry, Type::i32());
    Value* v = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});

    Value* c100 = b.build_iconst_i64(100);
    b.build_store(Type::i64(), obj, 0, c100);
    b.build_br_if(cond, then_bb, else_bb);

    // then_bb: store v + 50
    b.position_at_end(then_bb);
    Value* c50 = b.build_iconst_i64(50);
    Value* v_plus_50 = b.build_add(v, c50);
    b.build_store(Type::i64(), obj, 0, v_plus_50);
    b.build_br(merge_bb);

    // else_bb: leaves obj unchanged (100)
    b.position_at_end(else_bb);
    b.build_br(merge_bb);

    // merge_bb: load and return
    b.position_at_end(merge_bb);
    Value* loaded = b.build_load(Type::i64(), obj, 0);
    b.build_ret(loaded);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    SroaStats stats;
    SroaOptions sroa_opts;
    sroa_opts.stats = &stats;
    bool changed = sroa_function(*fn, sroa_opts);
    CHECK(changed);
    CHECK_EQ(stats.allocations_eliminated, 1ULL);
    CHECK(stats.block_params_created >= 1ULL);
    CHECK_EQ(count_allocations(*fn), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::load), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::store), 0ULL);
    REQUIRE(verify_function(*fn));

    // Verify merge_bb received a block parameter for the field
    CHECK(merge_bb->param_count() >= 1ULL);

    // Test then branch (cond = 1, v = 20): expected 20 + 50 = 70
    Interpreter interp1;
    RuntimeValue iv1 = interp1.run(*fn, {RuntimeValue::from_i32(1), RuntimeValue::from_i64(20)});
    CHECK_EQ(iv1.as_i64(), 70);

    // Test else branch (cond = 0, v = 20): expected 100
    Interpreter interp2;
    RuntimeValue iv2 = interp2.run(*fn, {RuntimeValue::from_i32(0), RuntimeValue::from_i64(20)});
    CHECK_EQ(iv2.as_i64(), 100);
}

TEST_CASE("SROA - Loop-carried struct accumulator completely dissolved into block parameters") {
    Module mod("test_loop_accumulator");
    Builder b(mod);

    // func @loop_acc(%n: i64) -> i64
    Function* fn = mod.create_function("loop_acc", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_header = b.append_block("loop_header");
    BasicBlock* loop_body = b.append_block("loop_body");
    BasicBlock* loop_exit = b.append_block("loop_exit");

    Value* n = b.add_block_param(entry, Type::i64());

    // entry: allocate accumulator with sum (offset 0) and count (offset 8)
    b.position_at_end(entry);
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* acc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});

    Value* zero = b.build_iconst_i64(0);
    b.build_store(Type::i64(), acc, 0, zero);
    b.build_store(Type::i64(), acc, 8, zero);

    Value* i0 = b.build_iconst_i64(0);
    b.build_br(loop_header, {i0});

    // loop_header(i): cond = i < n
    Value* i_param = b.add_block_param(loop_header, Type::i64());
    b.position_at_end(loop_header);
    Value* cond = b.build_slt(i_param, n);
    b.build_br_if(cond, loop_body, loop_exit);

    // loop_body: sum += i, count += 1, i++
    b.position_at_end(loop_body);
    Value* cur_sum = b.build_load(Type::i64(), acc, 0);
    Value* cur_count = b.build_load(Type::i64(), acc, 8);
    Value* new_sum = b.build_add(cur_sum, i_param);
    Value* one = b.build_iconst_i64(1);
    Value* new_count = b.build_add(cur_count, one);
    b.build_store(Type::i64(), acc, 0, new_sum);
    b.build_store(Type::i64(), acc, 8, new_count);
    Value* next_i = b.build_add(i_param, one);
    b.build_br(loop_header, {next_i});

    // loop_exit: return sum
    b.position_at_end(loop_exit);
    Value* final_sum = b.build_load(Type::i64(), acc, 0);
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    SroaStats stats;
    SroaOptions sroa_opts;
    sroa_opts.stats = &stats;
    bool changed = sroa_function(*fn, sroa_opts);
    CHECK(changed);
    CHECK_EQ(stats.allocations_eliminated, 1ULL);
    CHECK_EQ(count_allocations(*fn), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::load), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::store), 0ULL);
    REQUIRE(verify_function(*fn));

    // Verify loop_header received block parameters for sum and count
    CHECK(loop_header->param_count() >= 3ULL);

    // Evaluate sum of 0..9: 0+1+2+3+4+5+6+7+8+9 = 45
    Interpreter interp;
    RuntimeValue iv = interp.run(*fn, {RuntimeValue::from_i64(10)});
    CHECK_EQ(iv.as_i64(), 45);
}

TEST_CASE("SROA - Escape Negative Tests (returned or stored to escaping pointers NOT dissolved)") {
    // 1. Returned object must NOT be dissolved
    {
        Module mod("escape_ret");
        Builder b(mod);
        Function* fn = mod.create_function("ret_alloc", Type::gcref(), {});
        b.set_function(fn);
        b.append_block("entry");
        Value* sz = b.build_iconst_i64(16);
        Value* mask = b.build_iconst_i64(0);
        Value* tag = b.build_iconst_i32(1);
        Value* obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz, mask, tag});
        b.build_ret(obj);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        EscapeAnalysis ea(*fn);
        CHECK_EQ(ea.get_escape_state(obj), EscapeState::GlobalEscape);
        CHECK(ea.does_escape(obj));
        CHECK_EQ(ea.non_escaping_allocations().size(), 0ULL);

        bool changed = sroa_function(*fn);
        CHECK(!changed);
        CHECK_EQ(count_allocations(*fn), 1ULL);
    }

    // 2. Stored into escaping pointer (function argument) must NOT be dissolved
    {
        Module mod("escape_store");
        Builder b(mod);
        Function* fn = mod.create_function("store_to_arg", Type::void_type(), {Type::gcref()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* arg_ptr = b.add_block_param(entry, Type::gcref());

        Value* sz = b.build_iconst_i64(16);
        Value* mask = b.build_iconst_i64(0);
        Value* tag = b.build_iconst_i32(1);
        Value* temp_obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz, mask, tag});

        b.build_store(Type::gcref(), arg_ptr, 0, temp_obj);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        EscapeAnalysis ea(*fn);
        CHECK_NE(ea.get_escape_state(temp_obj), EscapeState::NoEscape);
        CHECK(ea.does_escape(temp_obj));
        CHECK_EQ(ea.non_escaping_allocations().size(), 0ULL);

        bool changed = sroa_function(*fn);
        CHECK(!changed);
        CHECK_EQ(count_allocations(*fn), 1ULL);
    }

    // 3. Passed to unhandled external call must NOT be dissolved
    {
        Module mod("escape_call");
        Builder b(mod);
        Function* fn = mod.create_function("pass_to_extern", Type::void_type(), {});
        b.set_function(fn);
        b.append_block("entry");

        Value* sz = b.build_iconst_i64(16);
        Value* mask = b.build_iconst_i64(0);
        Value* tag = b.build_iconst_i32(1);
        Value* temp_obj = b.build_call("brass_gc_alloc", Type::gcref(), {sz, mask, tag});

        b.build_call("unhandled_foreign_callee", Type::void_type(), {temp_obj});
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        EscapeAnalysis ea(*fn);
        CHECK_EQ(ea.get_escape_state(temp_obj), EscapeState::GlobalEscape);
        CHECK(ea.does_escape(temp_obj));

        bool changed = sroa_function(*fn);
        CHECK(!changed);
        CHECK_EQ(count_allocations(*fn), 1ULL);
    }
}

TEST_CASE("SROA - Zero GC Allocation Verification (Cheney GC total allocations remains 0)") {
    Module mod("test_zero_gc");
    Builder b(mod);

    // func @run_point_loop(%iters: i64) -> f64
    // Inside the loop: allocates a 3D point each iteration, writes components, computes dist_sq, adds to total
    Function* fn = mod.create_function("run_point_loop", Type::f64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_header = b.append_block("loop_header");
    BasicBlock* loop_body = b.append_block("loop_body");
    BasicBlock* loop_exit = b.append_block("loop_exit");

    Value* iters = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* i0 = b.build_iconst_i64(0);
    Value* total0 = b.build_fconst_f64(0.0);
    b.build_br(loop_header, {i0, total0});

    // loop_header(i, total): cond = i < iters
    Value* i_cur = b.add_block_param(loop_header, Type::i64());
    Value* total_cur = b.add_block_param(loop_header, Type::f64());
    b.position_at_end(loop_header);
    Value* cond = b.build_slt(i_cur, iters);
    b.build_br_if(cond, loop_body, loop_exit);

    // loop_body: allocate point, write fields, read fields, update total
    b.position_at_end(loop_body);
    Value* sz24 = b.build_iconst_i64(24);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* pt = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag});

    Value* fx = b.build_sitofp_f64_i64(i_cur);
    Value* c2 = b.build_fconst_f64(2.0);
    Value* fy = b.build_mul(fx, c2);
    Value* c3 = b.build_fconst_f64(3.0);
    Value* fz = b.build_mul(fx, c3);

    b.build_store(Type::f64(), pt, 0, fx);
    b.build_store(Type::f64(), pt, 8, fy);
    b.build_store(Type::f64(), pt, 16, fz);

    Value* rx = b.build_load(Type::f64(), pt, 0);
    Value* ry = b.build_load(Type::f64(), pt, 8);
    Value* rz = b.build_load(Type::f64(), pt, 16);

    Value* sum_xyz = b.build_add(rx, b.build_add(ry, rz));
    Value* new_total = b.build_add(total_cur, sum_xyz);

    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i_cur, one);
    b.build_br(loop_header, {next_i, new_total});

    // loop_exit: return total
    b.position_at_end(loop_exit);
    b.build_ret(total_cur);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Run unoptimized in Interpreter first: verify baseline works and performs allocations
    {
        Interpreter interp_baseline;
        RuntimeValue baseline_res = interp_baseline.run(*fn, {RuntimeValue::from_i64(100)});
        CHECK(interp_baseline.gc().total_allocations() >= 100ULL);
        CHECK_EQ(baseline_res.as_f64(), 29700.0); // sum of 6*i for i=0..99: 6 * (99*100/2) = 29700.0
    }

    // Now run SROA on the function
    SroaStats stats;
    SroaOptions sroa_opts;
    sroa_opts.stats = &stats;
    bool changed = sroa_function(*fn, sroa_opts);
    CHECK(changed);
    CHECK_EQ(stats.allocations_eliminated, 1ULL);
    CHECK_EQ(count_allocations(*fn), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::load), 0ULL);
    CHECK_EQ(count_opcodes(*fn, Opcode::store), 0ULL);
    REQUIRE(verify_function(*fn));

    // Run SROA-optimized in Interpreter with fresh GC:
    // Verify Cheney GC total allocations remains EXACTLY 0!
    {
        Interpreter interp_sroa;
        RuntimeValue sroa_res = interp_sroa.run(*fn, {RuntimeValue::from_i64(100)});
        CHECK_EQ(sroa_res.as_f64(), 29700.0);
        CHECK_EQ(interp_sroa.gc().total_allocations(), 0ULL);
    }
}
