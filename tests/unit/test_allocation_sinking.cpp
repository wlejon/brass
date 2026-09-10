#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/allocation_sinking.hpp>
#include <brass/interpreter/interpreter.hpp>

using namespace brass;

namespace {

size_t count_allocations_in_block(const BasicBlock* bb) {
    if (!bb) return 0;
    size_t cnt = 0;
    for (const Instruction* inst : *bb) {
        if (inst && inst->opcode() == Opcode::call && is_allocation_callee(inst->symbol())) {
            cnt++;
        }
    }
    return cnt;
}

size_t count_opcodes_in_block(const BasicBlock* bb, Opcode op) {
    if (!bb) return 0;
    size_t cnt = 0;
    for (const Instruction* inst : *bb) {
        if (inst && inst->opcode() == op) cnt++;
    }
    return cnt;
}

} // namespace

TEST_CASE("Allocation Sinking - Loop Allocation Sunk to Exit Edge") {
    Module mod("test_sink_loop");
    Builder b(mod);

    // func @loop_alloc(%n: i64) -> gcref
    Function* fn = mod.create_function("loop_alloc", Type::gcref(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* header = b.append_block("header");
    BasicBlock* body = b.append_block("body");
    BasicBlock* exit = b.append_block("exit");

    Value* n = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* zero = b.build_iconst_i64(0);
    b.build_br(header, {zero});

    // header(%i: i64)
    Value* i_param = b.add_block_param(header, Type::i64());
    b.position_at_end(header);

    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);

    // Temporary object allocated on every iteration
    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});
    Value* c42 = b.build_iconst_i64(42);
    b.build_store(Type::i64(), alloc, 0, i_param);
    b.build_store(Type::i64(), alloc, 8, c42);

    Value* cond = b.build_sge(i_param, n);
    b.build_br_if(cond, exit, body);

    // body
    b.position_at_end(body);
    Value* v0 = b.build_load(Type::i64(), alloc, 0);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(v0, one);
    b.build_br(header, {next_i});

    // exit: returns escaping object
    b.position_at_end(exit);
    b.build_ret(alloc);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Run Allocation Sinking Pass
    PartialEscapeStats stats;
    AllocationSinkingOptions opts;
    opts.stats = &stats;
    bool changed = sink_allocations(*fn, opts);

    CHECK(changed);
    CHECK(stats.sunk_allocations >= 1ULL);
    CHECK(stats.scalarized_loads >= 1ULL);
    CHECK(stats.scalarized_stores >= 1ULL);
    CHECK(stats.materialized_allocations >= 1ULL);

    fn->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    if (!verify_function(*fn, &diag)) {
        std::cerr << "VERIFIER ERROR in test 1:\n" << diag.format_all() << "\n";
    }
    REQUIRE(verify_function(*fn));

    // Verify: loop header and body no longer have allocations
    CHECK_EQ(count_allocations_in_block(header), 0ULL);
    CHECK_EQ(count_allocations_in_block(body), 0ULL);
    CHECK_EQ(count_opcodes_in_block(body, Opcode::load), 0ULL);
    CHECK_EQ(count_opcodes_in_block(header, Opcode::store), 0ULL);

    // Verify execution with Interpreter
    Interpreter interp;
    auto res = interp.run(*fn, {RuntimeValue::from_i64(5)});
    CHECK(res.is_gcref());
    uintptr_t obj_raw = static_cast<uintptr_t>(res.as_gcref());
    CHECK_NE(obj_raw, 0ULL);
    int64_t* fields_ptr = reinterpret_cast<int64_t*>(obj_raw);
    CHECK_EQ(fields_ptr[0], 5);
    CHECK_EQ(fields_ptr[1], 42);
}

TEST_CASE("Allocation Sinking - Hot Path Scalarization with Cold Exit") {
    Module mod("test_sink_hot_path");
    Builder b(mod);

    // func @compute(%cond: i32, %x: i64, %y: i64) -> i64
    Function* fn = mod.create_function("compute", Type::i64(), {Type::i32(), Type::i64(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* hot_path = b.append_block("hot_path");
    BasicBlock* cold_exit = b.append_block("cold_exit");

    Value* cond = b.add_block_param(entry, Type::i32());
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* sz16 = b.build_iconst_i64(16);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);

    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz16, mask0, tag});
    b.build_store(Type::i64(), alloc, 0, x);
    b.build_store(Type::i64(), alloc, 8, y);

    Value* zero = b.build_iconst_i32(0);
    Value* is_zero = b.build_eq(cond, zero);
    b.build_br_if(is_zero, hot_path, cold_exit);

    // Hot path: loads fields, adds, returns scalar
    b.position_at_end(hot_path);
    Value* rx = b.build_load(Type::i64(), alloc, 0);
    Value* ry = b.build_load(Type::i64(), alloc, 8);
    Value* sum = b.build_add(rx, ry);
    b.build_ret(sum);

    // Cold exit: escapes object to host call, returns -1
    b.position_at_end(cold_exit);
    b.build_call("escape_sink", Type::void_type(), {alloc});
    Value* minus_one = b.build_iconst_i64(-1);
    b.build_ret(minus_one);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    PartialEscapeStats stats;
    AllocationSinkingOptions opts;
    opts.stats = &stats;
    bool changed = sink_allocations(*fn, opts);

    CHECK(changed);
    CHECK(stats.sunk_allocations >= 1ULL);
    CHECK(stats.scalarized_loads >= 2ULL);
    CHECK(stats.scalarized_stores >= 2ULL);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Entry block must have no allocations or stores
    CHECK_EQ(count_allocations_in_block(entry), 0ULL);
    CHECK_EQ(count_opcodes_in_block(entry, Opcode::store), 0ULL);

    // Hot path must have no loads
    CHECK_EQ(count_opcodes_in_block(hot_path, Opcode::load), 0ULL);

    // Execute hot path (cond=0)
    Interpreter interp;
    int64_t escaped_val0 = 0;
    int64_t escaped_val8 = 0;
    interp.register_external_function("escape_sink", [&](Interpreter&, const std::vector<RuntimeValue>& args) {
        if (!args.empty() && args[0].is_gcref()) {
            int64_t* ptr = reinterpret_cast<int64_t*>(args[0].as_gcref());
            if (ptr) {
                escaped_val0 = ptr[0];
                escaped_val8 = ptr[1];
            }
        }
        return RuntimeValue::from_i32(0);
    });

    auto res_hot = interp.run(*fn, {RuntimeValue::from_i32(0), RuntimeValue::from_i64(10), RuntimeValue::from_i64(20)});
    CHECK_EQ(res_hot.as_i64(), 30);

    // Execute cold path (cond=1)
    auto res_cold = interp.run(*fn, {RuntimeValue::from_i32(1), RuntimeValue::from_i64(100), RuntimeValue::from_i64(200)});
    CHECK_EQ(res_cold.as_i64(), -1);
    CHECK_EQ(escaped_val0, 100);
    CHECK_EQ(escaped_val8, 200);
}

TEST_CASE("Allocation Sinking - Multi-Field Loop Accumulator") {
    Module mod("test_sink_multi_acc");
    Builder b(mod);

    // func @stat_acc(%n: i64) -> gcref
    Function* fn = mod.create_function("stat_acc", Type::gcref(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* header = b.append_block("header");
    BasicBlock* body = b.append_block("body");
    BasicBlock* exit = b.append_block("exit");

    Value* n = b.add_block_param(entry, Type::i64());

    b.position_at_end(entry);
    Value* sz24 = b.build_iconst_i64(24);
    Value* mask0 = b.build_iconst_i64(0);
    Value* tag = b.build_iconst_i32(1);
    Value* zero = b.build_iconst_i64(0);

    Value* alloc = b.build_call("brass_gc_alloc", Type::gcref(), {sz24, mask0, tag});
    b.build_store(Type::i64(), alloc, 0, zero); // sum
    b.build_store(Type::i64(), alloc, 8, zero); // sum of squares
    b.build_store(Type::i64(), alloc, 16, zero); // count

    b.build_br(header, {zero});

    // header(%i: i64)
    Value* i_param = b.add_block_param(header, Type::i64());
    b.position_at_end(header);

    Value* cond = b.build_slt(i_param, n);
    b.build_br_if(cond, body, exit);

    // body
    b.position_at_end(body);
    Value* cur_sum = b.build_load(Type::i64(), alloc, 0);
    Value* cur_sq = b.build_load(Type::i64(), alloc, 8);
    Value* cur_cnt = b.build_load(Type::i64(), alloc, 16);

    Value* new_sum = b.build_add(cur_sum, i_param);
    Value* i_sq = b.build_mul(i_param, i_param);
    Value* new_sq = b.build_add(cur_sq, i_sq);
    Value* one = b.build_iconst_i64(1);
    Value* new_cnt = b.build_add(cur_cnt, one);

    b.build_store(Type::i64(), alloc, 0, new_sum);
    b.build_store(Type::i64(), alloc, 8, new_sq);
    b.build_store(Type::i64(), alloc, 16, new_cnt);

    Value* next_i = b.build_add(i_param, one);
    b.build_br(header, {next_i});

    // exit
    b.position_at_end(exit);
    b.build_ret(alloc);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    PartialEscapeStats stats;
    AllocationSinkingOptions opts;
    opts.stats = &stats;
    bool changed = sink_allocations(*fn, opts);

    CHECK(changed);
    CHECK(stats.sunk_allocations >= 1ULL);
    CHECK(stats.materialized_allocations >= 1ULL);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    // Allocation must be removed from entry
    CHECK_EQ(count_allocations_in_block(entry), 0ULL);

    // Run with n = 4: sum=0+1+2+3=6, sq=0+1+4+9=14, cnt=4
    Interpreter interp_acc;
    auto res = interp_acc.run(*fn, {RuntimeValue::from_i64(4)});
    CHECK(res.is_gcref());
    uintptr_t res_raw = static_cast<uintptr_t>(res.as_gcref());
    int64_t* res_ptr = reinterpret_cast<int64_t*>(res_raw);
    CHECK_EQ(res_ptr[0], 6);
    CHECK_EQ(res_ptr[1], 14);
    CHECK_EQ(res_ptr[2], 4);
}
