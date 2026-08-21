#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/emit_context.hpp>

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

TEST_CASE("x64 Codegen - End-to-End Simple Add Function") {
    Module mod;
    Function* fn = mod.create_function("add_i64", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* sum = b.build_add(a, c);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    // Pipeline: ISEL -> Liveness -> RegAlloc -> Emit
    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    LivenessAnalysis liveness(*lir);
    liveness.run();

    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::win64());
    regalloc.allocate();

    EmitContext emit_ctx(*lir, Target::x64_windows());
    CompilationResult res = emit_ctx.compile();

    CHECK(!res.code_buffer.empty());
    CHECK(!res.code_buffer.has_unresolved_labels());

    // Machine code should end with RET (0xC3)
    const auto& bytes = res.code_buffer.bytes();
    CHECK_EQ(bytes.back(), 0xC3);
}

TEST_CASE("x64 Codegen - Loop with Block Parameters (Sum 1..N)") {
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

    // Entry: start with i = 1, acc = 0
    Value* n = b.add_block_param(entry, Type::i64());
    Value* i_init = b.build_iconst_i64(1);
    Value* acc_init = b.build_iconst_i64(0);
    b.build_br(loop_header, {i_init, acc_init});

    // Loop Header: params (i, acc)
    b.position_at_end(loop_header);
    Value* i_val = b.add_block_param(loop_header, Type::i64());
    Value* acc_val = b.add_block_param(loop_header, Type::i64());
    Value* cond = b.build_sle(i_val, n);
    b.build_br_if(cond, loop_body, exit_bb);

    // Loop Body: acc += i; i += 1; jmp loop_header(i, acc)
    b.position_at_end(loop_body);
    Value* new_acc = b.build_add(acc_val, i_val);
    Value* one = b.build_iconst_i64(1);
    Value* new_i = b.build_add(i_val, one);
    b.build_br(loop_header, {new_i, new_acc});

    // Exit: return acc
    b.position_at_end(exit_bb);
    b.build_ret(acc_val);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    LivenessAnalysis liveness(*lir);
    liveness.run();

    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::win64());
    regalloc.allocate();

    EmitContext emit_ctx(*lir, Target::x64_windows());
    CompilationResult res = emit_ctx.compile();

    CHECK(!res.code_buffer.empty());
    CHECK(!res.code_buffer.has_unresolved_labels());
    CHECK_EQ(res.block_offsets.size(), lir->blocks.size());
}

TEST_CASE("x64 Codegen - Recursive Fibonacci Function") {
    Module mod;
    Function* fn = mod.create_function("fib", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    BasicBlock* base_case = b.create_block("base_case");
    BasicBlock* rec_case = b.create_block("rec_case");

    fn->append_block(base_case);
    fn->append_block(rec_case);

    Value* n = b.add_block_param(entry, Type::i64());
    Value* two = b.build_iconst_i64(2);
    Value* cond = b.build_slt(n, two);
    b.build_br_if(cond, base_case, rec_case);

    b.position_at_end(base_case);
    b.build_ret(n);

    b.position_at_end(rec_case);
    Value* one = b.build_iconst_i64(1);
    Value* n_minus_1 = b.build_sub(n, one);
    Value* n_minus_2 = b.build_sub(n, two);
    Value* fib1 = b.build_call("fib", Type::i64(), {n_minus_1});
    Value* fib2 = b.build_call("fib", Type::i64(), {n_minus_2});
    Value* sum = b.build_add(fib1, fib2);
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    LivenessAnalysis liveness(*lir);
    liveness.run();

    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::win64());
    regalloc.allocate();

    EmitContext emit_ctx(*lir, Target::x64_windows());
    CompilationResult res = emit_ctx.compile();

    CHECK(!res.code_buffer.empty());
    CHECK(!res.code_buffer.has_unresolved_labels());

    size_t fib_relocs = 0;
    for (const auto& reloc : res.code_buffer.relocations()) {
        if (reloc.symbol_name == "fib") fib_relocs++;
    }
    CHECK_EQ(fib_relocs, size_t(2));
}

TEST_CASE("x64 Codegen - Multi-Argument Calls and Frame 16-byte Alignment") {
    Module mod;
    std::vector<Type> params(8, Type::i64());
    Function* fn = mod.create_function("multi_arg_func", Type::i64(), params);

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    std::vector<Value*> args;
    for (size_t i = 0; i < 8; ++i) {
        args.push_back(b.add_block_param(entry, Type::i64()));
    }

    Value* s = args[0];
    for (size_t i = 1; i < 8; ++i) {
        s = b.build_add(s, args[i]);
    }
    b.build_ret(s);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    // Win64
    {
        X64ISel isel(Target::x64_windows(), CallingConvention::win64());
        auto lir = isel.lower(*fn);
        LivenessAnalysis liveness(*lir);
        liveness.run();
        LinearScanAllocator regalloc(*lir, liveness, CallingConvention::win64());
        regalloc.allocate();

        EmitContext emit_ctx(*lir, Target::x64_windows());
        CompilationResult res = emit_ctx.compile();
        CHECK(!res.code_buffer.empty());
        CHECK(!res.code_buffer.has_unresolved_labels());
    }

    // SysV64
    {
        X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
        auto lir = isel.lower(*fn);
        LivenessAnalysis liveness(*lir);
        liveness.run();
        LinearScanAllocator regalloc(*lir, liveness, CallingConvention::sysv64());
        regalloc.allocate();

        EmitContext emit_ctx(*lir, Target::x64_linux());
        CompilationResult res = emit_ctx.compile();
        CHECK(!res.code_buffer.empty());
        CHECK(!res.code_buffer.has_unresolved_labels());
    }
}

TEST_CASE("x64 Codegen - Safepoint Emission and Live GC Stack Map") {
    Module mod;
    Function* fn = mod.create_function("fn_with_safepoint", Type::void_type(), {Type::gcref()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ref = b.add_block_param(entry, Type::gcref());
    b.build_safepoint();
    b.build_store(Type::gcref(), ref, 0, ref);
    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);

    LivenessAnalysis liveness(*lir);
    liveness.run();

    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::win64());
    regalloc.allocate();

    EmitContext emit_ctx(*lir, Target::x64_windows());
    CompilationResult res = emit_ctx.compile();

    CHECK(!res.code_buffer.empty());
    CHECK_EQ(res.safepoints.size(), size_t(1));
}
