#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/array_contraction.hpp>
#include <brass/mir/loop_fusion.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/gc/mini_cheney.hpp>
#include <brass/interpreter/interpreter.hpp>

using namespace brass;

TEST_CASE("Array Contraction - Eliminate Buffer and Verify 0 Cheney GC Allocations") {
    Module mod("test_contraction_gc");
    Builder b(mod);

    // fn(n: i64) -> i64
    // Allocate buffer: arr = brass_gc_alloc(n * 8)
    // Loop 1: arr[i] = i * 2 (i = 0..n)
    // Loop 2: sum += arr[i] (i = 0..n)
    // ret sum
    Function* fn = mod.create_function("map_reduce_buf", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* l1_hdr = b.create_block("l1_hdr");
    BasicBlock* l1_body = b.create_block("l1_body");
    BasicBlock* l1_exit = b.create_block("l1_exit");

    BasicBlock* l2_hdr = b.create_block("l2_hdr");
    BasicBlock* l2_body = b.create_block("l2_body");
    BasicBlock* l2_exit = b.create_block("l2_exit");

    b.position_at_end(entry);
    Value* eight = b.build_iconst_i64(8);
    Value* byte_size = b.build_mul(n, eight);
    Value* arr = b.build_call("brass_gc_alloc", Type::gcref(), {byte_size});

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* two = b.build_iconst_i64(2);
    b.build_br(l1_hdr, {zero});

    // Loop 1: arr[i1] = i1 * 2
    fn->append_block(l1_hdr);
    b.position_at_end(l1_hdr);
    Value* i1 = b.add_block_param(l1_hdr, Type::i64());
    Value* cond1 = b.build_slt(i1, n);
    b.build_br_if(cond1, l1_body, {}, l1_exit, {});

    fn->append_block(l1_body);
    b.position_at_end(l1_body);
    Value* produced_val = b.build_mul(i1, two);
    b.build_store_indexed(Type::i64(), arr, i1, 8, produced_val);
    Value* next_i1 = b.build_add(i1, one);
    b.build_br(l1_hdr, {next_i1});

    // l1_exit -> Loop 2
    fn->append_block(l1_exit);
    b.position_at_end(l1_exit);
    b.build_br(l2_hdr, {zero, zero});

    // Loop 2: sum += arr[i2]
    fn->append_block(l2_hdr);
    b.position_at_end(l2_hdr);
    Value* i2 = b.add_block_param(l2_hdr, Type::i64());
    Value* sum2 = b.add_block_param(l2_hdr, Type::i64());
    Value* cond2 = b.build_slt(i2, n);
    b.build_br_if(cond2, l2_body, {}, l2_exit, {sum2});

    fn->append_block(l2_body);
    b.position_at_end(l2_body);
    Value* loaded_val = b.build_load_indexed(Type::i64(), arr, i2, 8);
    Value* next_sum2 = b.build_add(sum2, loaded_val);
    Value* next_i2 = b.build_add(i2, one);
    b.build_br(l2_hdr, {next_i2, next_sum2});

    fn->append_block(l2_exit);
    b.position_at_end(l2_exit);
    Value* final_sum = b.add_block_param(l2_exit, Type::i64());
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();

    // 1. Verify before optimization: running performs 1 GC allocation
    {
        Interpreter interp_unopt;
        auto res = interp_unopt.run(*fn, {RuntimeValue::from_i64(10)});
        // sum_{i=0..9} (2*i) = 2 * 45 = 90
        CHECK_EQ(res.as_i64(), 90);
        CHECK_EQ(interp_unopt.gc().total_allocations(), 1);
    }

    // 2. Run Array Contraction pass (which fuses and contracts)
    DominatorTree dom(*fn);
    ArrayContractionOptions contract_opts;
    ArrayContractionStats stats;
    contract_opts.stats = &stats;

    bool contracted = array_contraction_pass(*fn, dom, contract_opts);
    CHECK(contracted);
    CHECK_EQ(stats.arrays_contracted, 1);
    CHECK_EQ(stats.allocations_eliminated, 1);
    CHECK_EQ(stats.loads_eliminated, 1);
    CHECK_EQ(stats.stores_eliminated, 1);

    CHECK(verify_function(*fn));

    // Check that NO call to brass_gc_alloc remains in the function!
    size_t alloc_calls = 0;
    size_t loads = 0;
    size_t stores = 0;
    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call && is_allocation_callee(inst->symbol())) alloc_calls++;
            if (inst->opcode() == Opcode::load_indexed) loads++;
            if (inst->opcode() == Opcode::store_indexed) stores++;
        }
    }
    CHECK_EQ(alloc_calls, 0);
    CHECK_EQ(loads, 0);
    CHECK_EQ(stores, 0);

    // 3. Verify with Cheney GC: Total heap allocations drops strictly to 0!
    {
        Interpreter interp_opt;
        auto res = interp_opt.run(*fn, {RuntimeValue::from_i64(10)});
        CHECK_EQ(res.as_i64(), 90);
        CHECK_EQ(interp_opt.gc().total_allocations(), 0);
        CHECK_EQ(interp_opt.gc().total_allocated_bytes(), 0);
    }
}

TEST_CASE("Array Contraction - Bronze Dynamic Object Arrays") {
    Module mod("test_contraction_bronze");
    Builder b(mod);

    Function* fn = mod.create_function("bronze_map_reduce", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* l1_hdr = b.create_block("l1_hdr");
    BasicBlock* l1_body = b.create_block("l1_body");
    BasicBlock* l1_exit = b.create_block("l1_exit");

    BasicBlock* l2_hdr = b.create_block("l2_hdr");
    BasicBlock* l2_body = b.create_block("l2_body");
    BasicBlock* l2_exit = b.create_block("l2_exit");

    b.position_at_end(entry);
    Value* arr = b.build_call("bronze_create_array", Type::i64(), {n});

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    Value* three = b.build_iconst_i64(3);
    Value* ic0 = b.build_iconst_i32(0);
    b.build_br(l1_hdr, {zero});

    // Loop 1: elem.set arr, i1, i1 * 3
    fn->append_block(l1_hdr);
    b.position_at_end(l1_hdr);
    Value* i1 = b.add_block_param(l1_hdr, Type::i64());
    Value* cond1 = b.build_slt(i1, n);
    b.build_br_if(cond1, l1_body, {}, l1_exit, {});

    fn->append_block(l1_body);
    b.position_at_end(l1_body);
    Value* val = b.build_mul(i1, three);
    b.build_call("bronze_elem_set", Type::void_type(), {arr, i1, val, ic0});
    Value* next_i1 = b.build_add(i1, one);
    b.build_br(l1_hdr, {next_i1});

    fn->append_block(l1_exit);
    b.position_at_end(l1_exit);
    b.build_br(l2_hdr, {zero, zero});

    // Loop 2: sum += elem.get arr, i2
    fn->append_block(l2_hdr);
    b.position_at_end(l2_hdr);
    Value* i2 = b.add_block_param(l2_hdr, Type::i64());
    Value* sum2 = b.add_block_param(l2_hdr, Type::i64());
    Value* cond2 = b.build_slt(i2, n);
    b.build_br_if(cond2, l2_body, {}, l2_exit, {sum2});

    fn->append_block(l2_body);
    b.position_at_end(l2_body);
    Value* loaded = b.build_call("bronze_elem_get", Type::i64(), {arr, i2});
    Value* next_sum2 = b.build_add(sum2, loaded);
    Value* next_i2 = b.build_add(i2, one);
    b.build_br(l2_hdr, {next_i2, next_sum2});

    fn->append_block(l2_exit);
    b.position_at_end(l2_exit);
    Value* final_sum = b.add_block_param(l2_exit, Type::i64());
    b.build_ret(final_sum);

    fn->rebuild_cfg_predecessors();

    DominatorTree dom(*fn);
    ArrayContractionOptions contract_opts;
    ArrayContractionStats stats;
    contract_opts.stats = &stats;

    bool contracted = array_contraction_pass(*fn, dom, contract_opts);
    CHECK(contracted);
    CHECK_EQ(stats.arrays_contracted, 1);
    CHECK_EQ(stats.allocations_eliminated, 1);
    CHECK_EQ(stats.loads_eliminated, 1);
    CHECK_EQ(stats.stores_eliminated, 1);

    CHECK(verify_function(*fn));

    // Verify bronze_create_array, bronze_elem_set, bronze_elem_get are completely eliminated
    for (BasicBlock* bb : fn->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call) {
                CHECK_NE(inst->symbol(), "bronze_create_array");
                CHECK_NE(inst->symbol(), "bronze_elem_set");
                CHECK_NE(inst->symbol(), "bronze_elem_get");
            }
        }
    }
}
