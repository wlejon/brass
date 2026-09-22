#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/loop_vectorize.hpp>
#include <brass/mir/loop_unroll.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <vector>
#include <cmath>

using namespace brass;

TEST_CASE("Alias Safety - Overlapping and Non-Overlapping Offset Queries") {
    Module mod("test_alias_offsets");
    Builder b(mod);
    Function* fn = mod.create_function("test_offsets", Type::void_type(), {Type::ptr()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* p = b.add_block_param(entry, Type::ptr());
    (void)p;

    // Allocate local object: alloca 32 bytes
    Value* base = b.build_alloca(32);

    Instruction* st_0_i64 = b.build_store(Type::i64(), base, 0, b.build_iconst_i64(123));
    Instruction* ld_4_i32 = b.build_load(Type::i32(), base, 4)->defining_instruction();
    Instruction* ld_8_i64 = b.build_load(Type::i64(), base, 8)->defining_instruction();
    Instruction* ld_0_i64 = b.build_load(Type::i64(), base, 0)->defining_instruction();

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    AliasAnalysis aa(*fn);

    // 1. Overlapping offset alias queries:
    // 8-byte write at offset 0 [0, 8), 4-byte read at offset 4 [4, 8) -> Overlap -> MayAlias
    AliasResult res_overlap = aa.alias(base, 0, Type::i64(), base, 4, Type::i32());
    CHECK_EQ(res_overlap, AliasResult::MayAlias);
    CHECK(aa.can_clobber(st_0_i64, ld_4_i32));

    // MustAlias: same offset and same type
    AliasResult res_must = aa.alias(base, 0, Type::i64(), base, 0, Type::i64());
    CHECK_EQ(res_must, AliasResult::MustAlias);
    CHECK(aa.can_clobber(st_0_i64, ld_0_i64));

    // 2. Non-overlapping offset alias queries:
    // 8-byte write at offset 0 [0, 8), 8-byte read at offset 8 [8, 16) -> Disjoint -> NoAlias
    AliasResult res_no_alias = aa.alias(base, 0, Type::i64(), base, 8, Type::i64());
    CHECK_EQ(res_no_alias, AliasResult::NoAlias);
    CHECK(!aa.can_clobber(st_0_i64, ld_8_i64));

    // Non-overlapping 4-byte accesses: [0, 4) vs [4, 8) -> NoAlias
    AliasResult res_no_alias_4 = aa.alias(base, 0, Type::i32(), base, 4, Type::i32());
    CHECK_EQ(res_no_alias_4, AliasResult::NoAlias);
}

TEST_CASE("Alias Safety - Function Parameter Disambiguation Safety") {
    // In Brass MIR, distinct entry block parameters are treated as non-aliasing buffers
    // (Fortran/restrict semantics) for loop kernels, while identical parameters MustAlias.
    Module mod("test_param_alias");
    Builder b(mod);
    Function* fn = mod.create_function("test_params", Type::void_type(), {Type::ptr(), Type::ptr()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* p1 = b.add_block_param(entry, Type::ptr());
    Value* p2 = b.add_block_param(entry, Type::ptr());
    p1->set_noalias(true);
    p2->set_noalias(true);

    // Local non-escaping allocation
    Value* local_alloc = b.build_alloca(32);

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    AliasAnalysis aa(*fn);

    // Two distinct entry parameters are distinct buffers
    CHECK(aa.is_distinct_allocation(p1, p2));
    AliasResult res_params = aa.alias(p1, p2);
    CHECK_EQ(res_params, AliasResult::NoAlias);

    // Same parameter MustAlias
    CHECK(!aa.is_distinct_allocation(p1, p1));
    CHECK_EQ(aa.alias(p1, p1), AliasResult::MustAlias);

    // A non-escaping local allocation does not alias external argument parameters
    AliasResult res_local_p1 = aa.alias(local_alloc, p1);
    CHECK_EQ(res_local_p1, AliasResult::NoAlias);
    AliasResult res_local_p2 = aa.alias(local_alloc, p2);
    CHECK_EQ(res_local_p2, AliasResult::NoAlias);
}

TEST_CASE("Vector Reduction - Float Types Emit fadd") {
    // Loop vectorizer horizontal reduction tree with float types must emit fadd
    Module mod("test_vec_reduction_f32");
    Builder b(mod);
    Function* fn = mod.create_function("dot_product_f32", Type::f32(), {Type::ptr(), Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* arr = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero_i = b.build_iconst_i64(0);
    Value* zero_f = b.build_fconst_f32(0.0f);
    b.build_br(loop_hdr, {zero_i, zero_f});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::f32());
    Value* cond = b.build_slt(i, count);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* elem = b.build_load_indexed(Type::f32(), arr, i, 4, 0);
    Value* next_acc = b.build_fadd(acc, elem);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* final_res = b.add_block_param(exit_bb, Type::f32());
    b.build_ret(final_res);

    fn->rebuild_cfg_predecessors();
    DominatorTree dom(*fn);

    LoopVectorizeOptions vec_opts;
    vec_opts.allow_fp_reassociation = true;
    bool changed = loop_vectorize_pass(*fn, dom, vec_opts);
    CHECK(changed);

    // Verify module integrity
    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    // Inspect the vec_exit block: horizontal reduction additions must have type f32
    BasicBlock* vec_exit = nullptr;
    for (BasicBlock* bb : fn->blocks()) {
        if (bb->name().find("_vec_exit") != std::string_view::npos) {
            vec_exit = bb;
            break;
        }
    }
    REQUIRE(vec_exit != nullptr);

    bool found_fp_add = false;
    for (Instruction* inst : *vec_exit) {
        if (inst->opcode() == Opcode::add) {
            CHECK_EQ(inst->type(), Type::f32());
            found_fp_add = true;
        }
    }
    CHECK(found_fp_add);

    // Verify JIT execution accuracy
    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    auto dot_fn = jit.get_function_ptr<float(*)(const float*, int64_t)>("dot_product_f32");
    REQUIRE(dot_fn != nullptr);

    constexpr int N = 23;
    alignas(16) float input[N];
    float expected = 0.0f;
    for (int k = 0; k < N; ++k) {
        input[k] = static_cast<float>(k + 1) * 0.5f;
        expected += input[k];
    }
    float actual = dot_fn(input, N);
    CHECK(std::abs(actual - expected) < 1e-4f);
}

TEST_CASE("Loop Unroll - RHS Comparison Normalization and Overflow Safety") {
    // Test loop with induction variable on RHS: e.g. 0 < i (equivalent to i > 0)
    // when counting down. The comparison opcode must be normalized to sgt.
    Module mod("test_unroll_rhs_cmp");
    Builder b(mod);
    Function* fn = mod.create_function("sum_countdown", Type::i64(), {Type::i64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* start_val = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero = b.build_iconst_i64(0);
    Value* one = b.build_iconst_i64(1);
    b.build_br(loop_hdr, {start_val, zero});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* i = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::i64());

    // Induction variable on RHS: 0 < i (equivalent to i > 0)
    Value* cond = b.build_slt(zero, i);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* next_acc = b.build_add(acc, i);
    Value* next_i = b.build_sub(i, one);
    b.build_br(loop_hdr, {next_i, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    DominatorTree dom(*fn);

    LoopUnrollOptions unroll_opts;
    unroll_opts.unroll_factor = 4;
    bool changed = loop_unroll_pass(*fn, dom, unroll_opts);
    CHECK(changed);

    DiagnosticReporter diag;
    REQUIRE(verify_module(mod, &diag));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    auto count_fn = jit.get_function_ptr<int64_t(*)(int64_t)>("sum_countdown");
    REQUIRE(count_fn != nullptr);

    // Sum of 1..10 is 55
    CHECK_EQ(count_fn(10), 55);
    // Sum of 1..23 is 276
    CHECK_EQ(count_fn(23), 276);
}

TEST_CASE("Write Barrier Elimination - Preserving Barrier for Opcode::or_ Tagged Pointers") {
    Module mod("test_wbe_tagged_ptr");
    Builder b(mod);
    Function* fn = mod.create_function("store_tagged_object", Type::void_type(), {Type::ptr(), Type::ptr()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    b.position_at_end(entry);
    Value* container = b.add_block_param(entry, Type::ptr());
    Value* raw_ptr = b.add_block_param(entry, Type::ptr());

    // Create tagged pointer via AND with payload mask, then OR with TAG_OBJECT
    Value* ptr_mask = b.build_iconst_i64(0x0000FFFFFFFFFFFFLL);
    Value* masked_ptr = b.build_and(raw_ptr, ptr_mask);
    Value* obj_tag = b.build_iconst_i64(static_cast<int64_t>(0xFFF1000000000000ULL));
    Value* tagged_obj = b.build_or(masked_ptr, obj_tag);

    // Store tagged pointer into container
    b.build_store(Type::i64(), container, 8, tagged_obj);
    Instruction* wb_tagged = b.build_write_barrier(container, tagged_obj);

    // Also store a plain integer into container
    Value* plain_int = b.build_iconst_i64(42);
    b.build_store(Type::i64(), container, 16, plain_int);
    Instruction* wb_int = b.build_write_barrier(container, plain_int);

    // Also store an OR of two non-pointers into container
    Value* int_or = b.build_or(b.build_iconst_i64(1), b.build_iconst_i64(2));
    b.build_store(Type::i64(), container, 24, int_or);
    Instruction* wb_int_or = b.build_write_barrier(container, int_or);

    b.build_ret_void();
    fn->rebuild_cfg_predecessors();

    WriteBarrierElimination wbe;
    CHECK(!wbe.is_non_pointer_value(tagged_obj));
    CHECK(wbe.is_non_pointer_value(plain_int));
    CHECK(wbe.is_non_pointer_value(int_or));

    bool changed = wbe.run_on_function(*fn);
    CHECK(changed);

    // The write barrier for plain_int and int_or should be eliminated,
    // but the write barrier for tagged_obj MUST be preserved!
    bool found_wb_tagged = false;
    bool found_wb_int = false;
    bool found_wb_int_or = false;

    for (Instruction* inst : *entry) {
        if (inst == wb_tagged) found_wb_tagged = true;
        if (inst == wb_int) found_wb_int = true;
        if (inst == wb_int_or) found_wb_int_or = true;
    }

    CHECK(found_wb_tagged);
    CHECK(!found_wb_int);
    CHECK(!found_wb_int_or);
}
