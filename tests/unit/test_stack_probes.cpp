#include "test_framework.hpp"
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/target.hpp>
#include <thread>

// A frame larger than a page, entered on a thread whose stack is barely
// committed: Windows commits a stack a page at a time as its guard page is
// touched, so the prologue must touch each page the frame spans, nearest
// first (x64 emit_stack_probes). Without it the first store to the far end
// of the frame faults.

using namespace brass;

TEST_CASE("Stack probes - a 256 KB frame on a fresh thread") {
    constexpr uint32_t kBytes = 256 * 1024;
    Module mod("stack_probes");
    Function* fn = mod.create_function("big_frame", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* buf = b.build_alloca(kBytes, 16);
    Value* lo = b.build_iconst_i64(0);
    Value* hi = b.build_iconst_i64(kBytes / 8 - 1);
    b.build_store_indexed(Type::i64(), buf, lo, 8, 0, x);
    b.build_store_indexed(Type::i64(), buf, hi, 8, 0, b.build_add(x, b.build_iconst_i64(1)));
    b.build_ret(b.build_add(b.build_load_indexed(Type::i64(), buf, lo, 8, 0),
                            b.build_load_indexed(Type::i64(), buf, hi, 8, 0)));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(mod));
    int64_t got = 0;
    std::thread t([&] { got = jit.invoke("big_frame", {RuntimeValue::from_i64(20)}).as_i64(); });
    t.join();
    CHECK_EQ(got, int64_t(41));
}
