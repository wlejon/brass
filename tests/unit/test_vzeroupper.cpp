#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/emit_context.hpp>
#include <vector>

// A function that leaves the upper halves of the YMM registers dirty makes
// every SSE instruction after it, in its caller or in a function it calls,
// pay for the merge (a stall on older cores, a false dependency on newer
// ones). The ABI expects them clean at every call and return, unless a
// 256-bit value is passed or returned: so a function that writes YMM clears
// them (vzeroupper, C5 F8 77) before each.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

std::vector<uint8_t> compile_x64(Function* fn) {
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    const Target target = Target::x64_linux();
    X64ISel isel(target, CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    LivenessAnalysis liveness(*lir);
    liveness.run();
    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::sysv64());
    regalloc.allocate();
    EmitContext emit_ctx(*lir, target);
    CompilationResult res = emit_ctx.compile();
    const auto& bytes = res.code_buffer.bytes();
    return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

size_t count_vzeroupper(const std::vector<uint8_t>& code) {
    size_t n = 0;
    for (size_t i = 0; i + 2 < code.size(); ++i) {
        if (code[i] == 0xC5 && code[i + 1] == 0xF8 && code[i + 2] == 0x77) ++n;
    }
    return n;
}

// The position of the last vzeroupper, or SIZE_MAX.
size_t last_vzeroupper(const std::vector<uint8_t>& code) {
    size_t at = SIZE_MAX;
    for (size_t i = 0; i + 2 < code.size(); ++i) {
        if (code[i] == 0xC5 && code[i + 1] == 0xF8 && code[i + 2] == 0x77) at = i;
    }
    return at;
}

} // namespace

TEST_CASE("vzeroupper - a function that writes YMM clears the upper state before it returns") {
    Module mod;
    Function* fn = mod.create_function("double8", Type::void_type(), {Type::ptr()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* p = b.add_block_param(entry, Type::ptr());
    Value* v = b.build_load(Type::f32x8(), p);
    b.build_store(Type::f32x8(), p, b.build_vadd(v, v));
    b.build_ret_void();

    const auto code = compile_x64(fn);
    REQUIRE(!code.empty());
    CHECK_EQ(code.back(), 0xC3);
    CHECK_EQ(count_vzeroupper(code), 1u);
    // After the last 256-bit instruction: the epilogue alone follows it.
    const size_t at = last_vzeroupper(code);
    REQUIRE(at != SIZE_MAX);
    for (size_t i = at + 3; i + 1 < code.size(); ++i) {
        CHECK(!(code[i] == 0xC5 || code[i] == 0xC4));
    }
}

TEST_CASE("vzeroupper - a function that writes YMM clears the upper state before a call") {
    Module mod;
    Function* fn = mod.create_function("double8_then_call", Type::void_type(), {Type::ptr()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* p = b.add_block_param(entry, Type::ptr());
    Value* v = b.build_load(Type::f32x8(), p);
    b.build_store(Type::f32x8(), p, b.build_vadd(v, v));
    b.build_call("ext", Type::void_type(), std::initializer_list<Value*>{});
    b.build_ret_void();

    const auto code = compile_x64(fn);
    // One before the call, one before the return.
    CHECK_EQ(count_vzeroupper(code), 2u);
}

TEST_CASE("vzeroupper - not before returning a 256-bit value, and not in 128-bit code") {
    {
        Module mod;
        Function* fn = mod.create_function("sum8", Type::f32x8(), {Type::f32x8(), Type::f32x8()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::f32x8());
        Value* y = b.add_block_param(entry, Type::f32x8());
        b.build_ret(b.build_vadd(x, y));
        // The result is in YMM0: clearing the upper halves would drop it.
        CHECK_EQ(count_vzeroupper(compile_x64(fn)), 0u);
    }
    {
        Module mod;
        Function* fn = mod.create_function("double4", Type::void_type(), {Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* p = b.add_block_param(entry, Type::ptr());
        Value* v = b.build_load(Type::f32x4(), p);
        b.build_store(Type::f32x4(), p, b.build_vadd(v, v));
        b.build_call("ext", Type::void_type(), std::initializer_list<Value*>{});
        b.build_ret_void();
        CHECK_EQ(count_vzeroupper(compile_x64(fn)), 0u);
    }
}
