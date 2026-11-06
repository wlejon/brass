#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <cstring>
#include <memory>
#include <string>

// Module::pinned_constants: 64-bit constants the host keeps at fixed offsets
// from the pinned TLS register. x64 reads them there, `[r13 + d8]`, instead
// of from the function's rip-relative constant pool or a 10-byte movabs; a
// function that writes the register keeps the pool.

using namespace brass;
using namespace brass::codegen;

namespace {

constexpr uint64_t kTag = 0xfff1000000000000ull;
constexpr uint64_t kLimit = 0x0001000000000000ull;
constexpr uint64_t kMask = 0x0000ffffffffffffull;
constexpr uint64_t kOther = 0x9e3779b97f4a7c15ull;  // not pinned

// The host's block: the pinned register points at words[4]; the constants sit
// below it, at -32 (kTag), -24 (kLimit) and -16 (kMask).
struct alignas(16) Block {
    uint64_t words[8] = {};
};
thread_local Block t_block;

int64_t block_arg() { return static_cast<int64_t>(reinterpret_cast<uintptr_t>(&t_block.words[4])); }

// @entry writes the register and calls @f, which reads the constants: the
// tag and mask live across calls, more than the callee-saved registers keep,
// and the limit and the unpinned constant are compared and added. @f calls
// itself on one input no test passes, so it is not inlined into @entry.
const char* kModule = R"(module @m

attributes pinned_tls_register pinned_constant 0xfff1000000000000 -32 pinned_constant 0x0001000000000000 -24 pinned_constant 0x0000ffffffffffff -16

func @g(%a: i64) -> i64 {
entry:
  %one = iconst.i64 1
  %r = add.i64 %a, %one
  ret %r
}

func @f(%x: i64) -> i64 {
entry:
  %magic = iconst.i64 0x5a5a5a5a
  %again = eq.i64 %x, %magic
  br_if %again, deeper, body
deeper:
  %one = iconst.i64 1
  %x1 = add.i64 %x, %one
  %r1 = call.i64 @f(%x1)
  ret %r1
body:
  %k1 = iconst.i64 0xfff1000000000000
  %k2 = iconst.i64 0x0001000000000000
  %k3 = iconst.i64 0x0000ffffffffffff
  %k4 = iconst.i64 0x9e3779b97f4a7c15
  %k5 = iconst.i64 0x7ff8000000000000
  %k6 = iconst.i64 0xbf58476d1ce4e5b9
  %a = xor.i64 %x, %k1
  %b = call.i64 @g(%a)
  %c = call.i64 @g(%b)
  %d = xor.i64 %c, %k1
  %lt = ult.i64 %d, %k2
  %z = zext.i64 %lt
  %e = add.i64 %d, %z
  %f = and.i64 %e, %k3
  %h = xor.i64 %f, %k5
  %i = add.i64 %h, %k4
  %j = xor.i64 %i, %k6
  %m = and.i64 %j, %k3
  %t = and.i64 %x, %k1
  %out = xor.i64 %m, %t
  ret %out
}

func @entry(%tls: i64, %x: i64) -> i64 {
entry:
  pinned_tls_write %tls
  %k = iconst.i64 0xfff1000000000000
  %y = xor.i64 %x, %k
  %r = call.i64 @f(%y)
  ret %r
}
)";

uint64_t expected_f(uint64_t x) {
    uint64_t a = x ^ kTag;
    uint64_t c = a + 2;
    uint64_t d = c ^ kTag;
    uint64_t e = d + (d < kLimit ? 1 : 0);
    uint64_t f = e & kMask;
    uint64_t h = f ^ 0x7ff8000000000000ull;
    uint64_t i = h + kOther;
    uint64_t j = i ^ 0xbf58476d1ce4e5b9ull;
    return (j & kMask) ^ (x & kTag);
}

CompilationResult emit_function(Module& mod, const char* name) {
    Function* fn = mod.get_function(name);
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    brass::x64::X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    lir->sort_blocks_rpo();
    LivenessAnalysis liveness(*lir);
    liveness.run();
    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::sysv64());
    regalloc.allocate();
    EmitContext emit(*lir, Target::x64_linux());
    return emit.compile();
}

// Occurrences of a memory operand [r13 + d8]: REX with B set (and W), any
// opcode byte, then a ModRM with mod 01 and r/m 101 (r13), then `disp`.
size_t count_r13_reads(const std::vector<uint8_t>& code, int8_t disp) {
    size_t n = 0;
    for (size_t i = 0; i + 3 < code.size(); ++i) {
        if ((code[i] & 0xF9) != 0x49) continue;  // REX.W + REX.B, R either way
        const uint8_t modrm = code[i + 2];
        if ((modrm & 0xC7) == 0x45 && static_cast<int8_t>(code[i + 3]) == disp) ++n;
    }
    return n;
}

bool contains_u64(const std::vector<uint8_t>& code, uint64_t v) {
    for (size_t i = 0; i + 8 <= code.size(); ++i) {
        uint64_t w = 0;
        std::memcpy(&w, code.data() + i, 8);
        if (w == v) return true;
    }
    return false;
}

} // namespace

TEST_CASE("x64 pinned constants - the attribute prints and parses back") {
    auto mod = parse_module(kModule);
    REQUIRE(mod != nullptr);
    REQUIRE_EQ(mod->pinned_constants().size(), size_t(3));
    CHECK_EQ(mod->pinned_constants()[0].value, kTag);
    CHECK_EQ(mod->pinned_constants()[0].tls_offset, -32);
    CHECK_EQ(mod->pinned_constants()[2].value, kMask);
    std::string text = to_string(*mod);
    auto again = parse_module(text);
    REQUIRE(again != nullptr);
    REQUIRE_EQ(again->pinned_constants().size(), size_t(3));
    CHECK_EQ(again->pinned_constants()[1].value, kLimit);
    CHECK_EQ(again->pinned_constants()[1].tls_offset, -24);
}

TEST_CASE("x64 pinned constants - a function reads them at the pinned register, never from its pool") {
    auto mod = parse_module(kModule);
    REQUIRE(mod != nullptr);
    CompilationResult res = emit_function(*mod, "f");
    const auto& code = res.code_buffer.bytes();
    // Each pinned constant is read at its offset at least once, and neither
    // a movabs immediate nor a pool word holds it.
    CHECK(count_r13_reads(code, -32) >= 1u);
    CHECK(count_r13_reads(code, -16) >= 1u);
    CHECK(count_r13_reads(code, -24) >= 1u);
    CHECK_FALSE(contains_u64(code, kTag));
    CHECK_FALSE(contains_u64(code, kLimit));
    CHECK_FALSE(contains_u64(code, kMask));
    // An unpinned constant still has its 8 bytes somewhere (pool or movabs).
    CHECK(contains_u64(code, kOther));
}

TEST_CASE("x64 pinned constants - the function that writes the register keeps them in its code") {
    auto mod = parse_module(kModule);
    REQUIRE(mod != nullptr);
    CompilationResult res = emit_function(*mod, "entry");
    const auto& code = res.code_buffer.bytes();
    CHECK(contains_u64(code, kTag));
    CHECK_EQ(count_r13_reads(code, -32), size_t(0));
}

TEST_CASE("x64 pinned constants - code reading them computes what the pool would") {
    t_block.words[0] = kTag;
    t_block.words[1] = kLimit;
    t_block.words[2] = kMask;
    auto run = parse_module(kModule);
    REQUIRE(run != nullptr);
    REQUIRE(verify_module(*run));
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*run));
    auto entry = jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("entry");
    REQUIRE(entry != nullptr);
    for (uint64_t x : {uint64_t{0}, uint64_t{7}, ~uint64_t{0}, uint64_t{1} << 47, uint64_t{0x123456789abcdef},
                       kTag | 5u}) {
        CHECK_EQ(static_cast<uint64_t>(entry(block_arg(), static_cast<int64_t>(x))), expected_f(x ^ kTag));
    }
}


TEST_CASE("x64 pinned constants - foldable uses read the constant in place and it takes no register") {
    auto mod = parse_module(kModule);
    REQUIRE(mod != nullptr);
    Function* fn = mod->get_function("f");
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    brass::x64::X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    const PReg r13 = PReg::gpr(x64::GPR::R13);
    size_t movabs_pinned = 0, folded = 0;
    for (const auto& b : lir->blocks) {
        for (const auto& inst : b->instructions) {
            if ((inst->opcode == LirOpcode::Movabs || inst->opcode == LirOpcode::Mov) && inst->uses.size() == 1 &&
                inst->uses[0].is_imm_int()) {
                const uint64_t v = static_cast<uint64_t>(inst->uses[0].imm_int);
                movabs_pinned += v == kTag || v == kLimit || v == kMask;
            }
            for (const auto& u : inst->uses) {
                if (u.is_mem() && u.mem_val.base_preg == r13 && !u.mem_val.index_preg.is_valid() &&
                    (u.mem_val.disp == -32 || u.mem_val.disp == -24 || u.mem_val.disp == -16)) {
                    ++folded;
                    CHECK((inst->opcode == LirOpcode::Xor || inst->opcode == LirOpcode::And ||
                           inst->opcode == LirOpcode::Or || inst->opcode == LirOpcode::Cmp ||
                           inst->opcode == LirOpcode::Mov));
                }
            }
        }
    }
    CHECK_EQ(movabs_pinned, size_t(0));
    // xor with the tag (twice, plus the and of x), the compare with the limit,
    // the two ands with the mask.
    CHECK(folded >= 5u);
}
