#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/gc/stack_walker.hpp>
#include <chrono>
#include <cmath>

using namespace brass;
using namespace brass::codegen;
using namespace brass::runtime;

TEST_CASE("Baseline JIT - Basic arithmetic, logic, and comparisons") {
    Module mod;
    Function* fn = mod.create_function("calc", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* d = b.add_block_param(entry, Type::i64());

    Value* sum = b.build_add(a, c);
    Value* prod = b.build_mul(sum, d);
    Value* diff = b.build_sub(prod, a);
    Value* div = b.build_sdiv(diff, c);
    Value* rem = b.build_smod(div, d);
    Value* shl = b.build_shl(rem, b.build_iconst_i64(2));
    Value* xor_val = b.build_xor(shl, b.build_iconst_i64(7));
    b.build_ret(xor_val);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler;
    BaselineCompiledFunction compiled = compiler.compile(*fn);
    REQUIRE(compiled.is_valid());
    CHECK(compiled.code_size() > 0);

    auto fn_ptr = compiled.get_function_ptr<int64_t(*)(int64_t, int64_t, int64_t)>();
    REQUIRE(fn_ptr != nullptr);

    // Compute expected in C++:
    // a=10, c=3, d=4
    // sum = 10 + 3 = 13
    // prod = 13 * 4 = 52
    // diff = 52 - 10 = 42
    // div = 42 / 3 = 14
    // rem = 14 % 4 = 2
    // shl = 2 << 2 = 8
    // xor = 8 ^ 7 = 15
    int64_t res = fn_ptr(10, 3, 4);
    CHECK_EQ(res, 15);

    // Test invoke wrapper
    RuntimeValue inv_res = compiled.invoke({
        RuntimeValue::from_i64(10),
        RuntimeValue::from_i64(3),
        RuntimeValue::from_i64(4)
    });
    CHECK_EQ(inv_res.as_i64(), 15);
}

TEST_CASE("Baseline JIT - Floating point computation") {
    Module mod;
    Function* fn = mod.create_function("fcalc", Type::f64(), {Type::f64(), Type::f64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::f64());
    Value* y = b.add_block_param(entry, Type::f64());

    Value* prod = b.build_mul(y, b.build_fconst_f64(2.5));
    Value* sum = b.build_add(x, prod);
    Value* diff = b.build_sub(sum, b.build_fconst_f64(1.0));
    Value* div = b.build_sdiv(diff, b.build_fconst_f64(2.0));
    b.build_ret(div);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler;
    BaselineCompiledFunction compiled = compiler.compile(*fn);
    REQUIRE(compiled.is_valid());

    auto fn_ptr = compiled.get_function_ptr<double(*)(double, double)>();
    REQUIRE(fn_ptr != nullptr);

    // x = 3.0, y = 4.0
    // prod = 4.0 * 2.5 = 10.0
    // sum = 3.0 + 10.0 = 13.0
    // diff = 13.0 - 1.0 = 12.0
    // div = 12.0 / 2.0 = 6.0
    double res = fn_ptr(3.0, 4.0);
    CHECK(std::abs(res - 6.0) < 1e-6);

    RuntimeValue inv_res = compiled.invoke({
        RuntimeValue::from_f64(3.0),
        RuntimeValue::from_f64(4.0)
    });
    CHECK(std::abs(inv_res.as_f64() - 6.0) < 1e-6);
}

TEST_CASE("Baseline JIT - Loop execution (Sum 1..N)") {
    Module mod;
    Function* fn = mod.create_function("sum_to_n", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* loop_header = b.create_block("loop_header");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    fn->append_block(loop_header);
    fn->append_block(loop_body);
    fn->append_block(exit_bb);

    Value* n = b.add_block_param(entry, Type::i64());
    Value* i_init = b.build_iconst_i64(1);
    Value* acc_init = b.build_iconst_i64(0);
    b.build_br(loop_header, {i_init, acc_init});

    b.position_at_end(loop_header);
    Value* i_val = b.add_block_param(loop_header, Type::i64());
    Value* acc_val = b.add_block_param(loop_header, Type::i64());
    Value* cond = b.build_sle(i_val, n);
    b.build_br_if(cond, loop_body, exit_bb);

    b.position_at_end(loop_body);
    Value* new_acc = b.build_add(acc_val, i_val);
    Value* one = b.build_iconst_i64(1);
    Value* new_i = b.build_add(i_val, one);
    b.build_br(loop_header, {new_i, new_acc});

    b.position_at_end(exit_bb);
    b.build_ret(acc_val);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler;
    BaselineCompiledFunction compiled = compiler.compile(*fn);
    REQUIRE(compiled.is_valid());

    auto fn_ptr = compiled.get_function_ptr<int64_t(*)(int64_t)>();
    REQUIRE(fn_ptr != nullptr);

    CHECK_EQ(fn_ptr(0), 0);
    CHECK_EQ(fn_ptr(1), 1);
    CHECK_EQ(fn_ptr(10), 55);
    CHECK_EQ(fn_ptr(100), 5050);
}

TEST_CASE("Baseline JIT - Direct recursion (Fibonacci)") {
    Module mod;
    Function* fn = mod.create_function("fib", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* base_case = b.create_block("base");
    BasicBlock* recurse = b.create_block("recurse");

    fn->append_block(base_case);
    fn->append_block(recurse);

    Value* n = b.add_block_param(entry, Type::i64());
    Value* cond = b.build_sle(n, b.build_iconst_i64(1));
    b.build_br_if(cond, base_case, recurse);

    b.position_at_end(base_case);
    b.build_ret(n);

    b.position_at_end(recurse);
    Value* n_minus_1 = b.build_sub(n, b.build_iconst_i64(1));
    Value* call1 = b.build_call("fib", Type::i64(), {n_minus_1});
    Value* n_minus_2 = b.build_sub(n, b.build_iconst_i64(2));
    Value* call2 = b.build_call("fib", Type::i64(), {n_minus_2});
    Value* sum = b.build_add(call1, call2);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler;
    BaselineCompiledFunction compiled = compiler.compile(*fn);
    REQUIRE(compiled.is_valid());

    auto fn_ptr = compiled.get_function_ptr<int64_t(*)(int64_t)>();
    REQUIRE(fn_ptr != nullptr);

    CHECK_EQ(fn_ptr(0), 0);
    CHECK_EQ(fn_ptr(1), 1);
    CHECK_EQ(fn_ptr(2), 1);
    CHECK_EQ(fn_ptr(3), 2);
    CHECK_EQ(fn_ptr(4), 3);
    CHECK_EQ(fn_ptr(5), 5);
    CHECK_EQ(fn_ptr(6), 8);
    CHECK_EQ(fn_ptr(7), 13);
    CHECK_EQ(fn_ptr(10), 55);
}

namespace {

// Leaves 0xFF in the stack region the next call's frame will occupy, so a
// read of a slot's unwritten upper bytes sees garbage instead of lucky zeros.
#if defined(_MSC_VER) && !defined(__clang__)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
void dirty_stack() {
    volatile uint8_t junk[4096];
    for (size_t i = 0; i < sizeof(junk); ++i) junk[i] = 0xFF;
}

} // namespace

TEST_CASE("Baseline JIT - Switch on a narrow value dispatches on its width only") {
    // An i32 load writes only the low 4 bytes of its 8-byte slot; the switch
    // used to compare all 8, so dispatch depended on stale stack contents
    // (it broke CoroTransformPass's state_id dispatch after the first resume).
    Module mod;
    Function* fn = mod.create_function("sel32", Type::i64(), {Type::ptr()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* one = b.create_block("one");
    BasicBlock* neg = b.create_block("neg");
    BasicBlock* big = b.create_block("big");
    BasicBlock* dflt = b.create_block("dflt");
    fn->append_block(one);
    fn->append_block(neg);
    fn->append_block(big);
    fn->append_block(dflt);

    Value* p = b.add_block_param(entry, Type::ptr());
    Value* x = b.build_load(Type::i32(), p, 0);
    b.build_switch(x, dflt, {SwitchCase(1, one), SwitchCase(-1, neg), SwitchCase(70000, big)});
    b.position_at_end(one);
    b.build_ret(b.build_iconst_i64(11));
    b.position_at_end(neg);
    b.build_ret(b.build_iconst_i64(22));
    b.position_at_end(big);
    b.build_ret(b.build_iconst_i64(33));
    b.position_at_end(dflt);
    b.build_ret(b.build_iconst_i64(99));

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler;
    BaselineCompiledFunction compiled = compiler.compile(*fn);
    REQUIRE(compiled.is_valid());
    auto fn_ptr = compiled.get_function_ptr<int64_t(*)(int32_t*)>();
    REQUIRE(fn_ptr != nullptr);

    auto run = [&](int32_t v) {
        dirty_stack();
        return fn_ptr(&v);
    };
    CHECK_EQ(run(1), 11);
    CHECK_EQ(run(-1), 22);
    CHECK_EQ(run(70000), 33);
    CHECK_EQ(run(0), 99);
    CHECK_EQ(run(2), 99);
}

TEST_CASE("Baseline JIT - Switch on i64 matches cases wider than imm32") {
    // cmp's imm32 is sign-extended, so a case outside int32 needs a register;
    // truncating it would match the wrong value.
    Module mod;
    Function* fn = mod.create_function("sel64", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    BasicBlock* wide = b.create_block("wide");
    BasicBlock* dflt = b.create_block("dflt");
    fn->append_block(wide);
    fn->append_block(dflt);

    Value* x = b.add_block_param(entry, Type::i64());
    const int64_t wide_case = (int64_t{1} << 32) + 5;
    b.build_switch(x, dflt, {SwitchCase(wide_case, wide)});
    b.position_at_end(wide);
    b.build_ret(b.build_iconst_i64(1));
    b.position_at_end(dflt);
    b.build_ret(b.build_iconst_i64(0));

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler;
    BaselineCompiledFunction compiled = compiler.compile(*fn);
    REQUIRE(compiled.is_valid());
    auto fn_ptr = compiled.get_function_ptr<int64_t(*)(int64_t)>();
    REQUIRE(fn_ptr != nullptr);

    CHECK_EQ(fn_ptr(wide_case), 1);
    CHECK_EQ(fn_ptr(5), 0);
}

TEST_CASE("Baseline JIT - Compilation throughput benchmark") {
    Module mod;
    Function* fn = mod.create_function("bench_fn", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());

    Value* v1 = b.build_add(x, y);
    Value* v2 = b.build_sub(v1, b.build_iconst_i64(10));
    Value* v3 = b.build_mul(v2, b.build_iconst_i64(3));
    Value* v4 = b.build_sdiv(v3, b.build_iconst_i64(2));
    Value* v5 = b.build_xor(v4, y);
    b.build_ret(v5);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler;

    const size_t iterations = 2000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iterations; ++i) {
        BaselineCompiledFunction compiled = compiler.compile(*fn);
        (void)compiled;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_sec = std::chrono::duration<double>(t1 - t0).count();
    double fns_per_sec = static_cast<double>(iterations) / elapsed_sec;

    // Verify baseline compiler achieves high throughput (> 15,000 fns/sec in debug/test, sub-microsecond in opt)
    CHECK(fns_per_sec > 15000.0);
}

TEST_CASE("Baseline JIT - Stack map generation for GC roots") {
    Module mod;
    Function* fn = mod.create_function("gc_safe_fn", Type::void_type(), {Type::gcref()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* obj = b.add_block_param(entry, Type::gcref());
    (void)obj;
    b.build_safepoint();
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler;
    BaselineCompiledFunction compiled = compiler.compile(*fn);
    REQUIRE(compiled.is_valid());

    const FunctionStackMap& sm = compiled.stack_map();
    CHECK_EQ(sm.function_name, "gc_safe_fn");
    CHECK(sm.code_size > 0);
    CHECK_EQ(sm.records.size(), 1);

    const StackMapRecord* rec = sm.find_record_by_offset(compiled.stack_map().records[0].instruction_offset);
    REQUIRE(rec != nullptr);
    CHECK_EQ(rec->roots.size(), 1);
    CHECK_EQ(rec->roots[0].kind, StackMapRootKind::FrameSlot);
    // Offset is negative (relative to RBP) on x86_64, positive (relative to FP) on AArch64
#if defined(__aarch64__) || defined(_M_ARM64)
    CHECK(rec->roots[0].offset_from_rbp > 0);
#else
    CHECK(rec->roots[0].offset_from_rbp < 0);
#endif
}

TEST_CASE("MultiTierPipeline - Tier 0 to Tier 1 to Tier 2 progression") {
    Module mod;
    Function* fn = mod.create_function("tiered_fn", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* doubled = b.build_add(x, x);
    b.build_ret(doubled);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    TieringConfig config;
    config.invocation_tier1_threshold = 3;
    config.invocation_tier2_threshold = 6;
    config.enable_background_compile = false;

    auto& pipeline = MultiTierPipeline::instance();
    pipeline.initialize(config);
    pipeline.reset_stats();

    TieringRegistry::instance().set_active_module(&mod);
    auto* handle = FunctionDispatchTable::instance().get_or_create("tiered_fn", fn);
    handle->set_tier(TierLevel::Tier0_Interpreter);

    CHECK(!handle->has_native_entry());
    CHECK_EQ(handle->tier(), TierLevel::Tier0_Interpreter);

    // Record 2 calls (below Tier 1 threshold)
    pipeline.on_invocation("tiered_fn");
    pipeline.on_invocation("tiered_fn");
    CHECK(!handle->has_native_entry());
    CHECK_EQ(handle->tier(), TierLevel::Tier0_Interpreter);

    // 3rd call triggers Tier 1 compilation
    pipeline.on_invocation("tiered_fn");
    CHECK(handle->has_native_entry());
    CHECK_EQ(handle->tier(), TierLevel::Tier1_Baseline);

    MultiTierStats stats = pipeline.stats();
    CHECK_EQ(stats.tier1_compilations.load(), 1u);

    // Execute via handle
    auto fn_ptr = handle->get_function_ptr<int64_t(*)(int64_t)>();
    REQUIRE(fn_ptr != nullptr);
    CHECK_EQ(fn_ptr(21), 42);

    pipeline.shutdown();
}
