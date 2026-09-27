#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/emit_context.hpp>
#include <string>

// Block order, and the branches the x64 emitter lays out: loops stay
// contiguous in the order register allocation sees, a loop's latch is one
// conditional branch back (not a branch out of the loop and a jump back),
// and a loop's head starts a 32-byte fetch window.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

// i (outer) and j (middle) are live through the inner loop and used only
// after it: in plain DFS order the inner loop's body landed after its exit.
const char* kNested = R"(module @nested

func @nested(%A: ptr, %N: i64) -> i64 {
entry:
  %zero = iconst.i64 0
  %one = iconst.i64 1
  br ih(%zero, %zero)
ih(%i: i64, %acc: i64):
  %ci = slt.i64 %i, %N
  br_if %ci, jh(%zero, %acc), exit(%acc)
jh(%j: i64, %a2: i64):
  %cj = slt.i64 %j, %N
  br_if %cj, kh(%zero, %a2), inx(%a2)
kh(%k: i64, %s: i64):
  %ck = slt.i64 %k, %N
  br_if %ck, kb, jn(%s)
kb:
  %v = load_indexed.i64 %A, %k, 8
  %ns = add.i64 %s, %v
  %nk = add.i64 %k, %one
  br kh(%nk, %ns)
jn(%fs: i64):
  %t = add.i64 %fs, %j
  %nj = add.i64 %j, %one
  br jh(%nj, %t)
inx(%a3: i64):
  %u = add.i64 %a3, %i
  %ni = add.i64 %i, %one
  br ih(%ni, %u)
exit(%r: i64):
  ret %r
}
)";

struct Lowered {
    std::unique_ptr<Module> mod;
    std::unique_ptr<LirFunction> lir;
};

Lowered lower(const char* text) {
    Lowered out;
    out.mod = parse_module(text);
    REQUIRE(out.mod != nullptr);
    Function* fn = out.mod->functions().front();
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    out.lir = isel.lower(*fn);
    REQUIRE(out.lir != nullptr);
    out.lir->sort_blocks_rpo();
    return out;
}

size_t position_of(const LirFunction& fn, const LirBlock* b) {
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        if (fn.blocks[i].get() == b) return i;
    }
    return SIZE_MAX;
}

} // namespace

TEST_CASE("Block order - every loop's blocks lie between its head and its last latch") {
    Lowered l = lower(kNested);
    LirFunction& fn = *l.lir;
    LivenessAnalysis liveness(fn);
    liveness.run();
    size_t loops = 0;
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        for (const LirBlock* s : fn.blocks[i]->successors) {
            const size_t h = position_of(fn, s);
            if (h > i) continue;
            ++loops;
            // Back edge i -> h: nothing between is shallower than the head.
            for (size_t k = h; k <= i; ++k) {
                CHECK(fn.blocks[k]->loop_depth >= fn.blocks[h]->loop_depth);
            }
        }
    }
    CHECK_EQ(loops, 3u);
}

TEST_CASE("Branches - a latch with block arguments is one branch back, and a loop's head is 32-byte aligned") {
    Lowered l = lower(kNested);
    LirFunction& fn = *l.lir;
    LivenessAnalysis liveness(fn);
    liveness.run();
    LinearScanAllocator regalloc(fn, liveness, CallingConvention::sysv64());
    regalloc.allocate();
    EmitContext emit_ctx(fn, Target::x64_linux());
    CompilationResult res = emit_ctx.compile();
    const auto& code = res.code_buffer.bytes();

    // A jcc rel32 (0F 8x) or rel8 (7x) straight followed by a jmp (E9/EB)
    // is a branch out and a jump back.
    size_t pairs = 0;
    for (size_t i = 0; i + 1 < code.size(); ++i) {
        if (code[i] == 0x0F && (code[i + 1] & 0xF0) == 0x80 && i + 6 < code.size() &&
            (code[i + 6] == 0xE9 || code[i + 6] == 0xEB)) {
            ++pairs;
        }
    }
    CHECK_EQ(pairs, 0u);

    // Each loop's head: the target of a branch from itself or later.
    size_t heads = 0;
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        for (const LirBlock* s : fn.blocks[i]->successors) {
            const size_t h = position_of(fn, s);
            if (h > i) continue;
            auto it = res.block_offsets.find(fn.blocks[h]->id);
            REQUIRE(it != res.block_offsets.end());
            CHECK_EQ(it->second % 32, 0u);
            ++heads;
        }
    }
    CHECK(heads >= 3u);
}

TEST_CASE("Branches - the nested loop computes the same sum through the JIT") {
    auto mod = parse_module(kNested);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(const int64_t*, int64_t)>("nested");
    REQUIRE(fn != nullptr);
    int64_t a[7] = {3, -1, 4, 1, -5, 9, 2};
    for (int64_t n = 0; n <= 7; ++n) {
        int64_t want = 0;
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t k = 0; k < n; ++k) want += a[k];
                want += j;
            }
            want += i;
        }
        CHECK_EQ(fn(a, n), want);
    }
}
