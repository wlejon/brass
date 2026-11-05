#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/mir/interrupt_poll.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <memory>
#include <string>

// A call whose callee reaches no GC point (SymbolRole::NoCollect or Pure: the
// LIR's no_gc_point) records no roots. A reference live across a call that
// can collect may keep a callee-saved register too, which the collector's
// stack scan pins (gc_contract.md 1.1); only one in a slot is a root.
// A function declared nocollect must itself reach no GC point, which the
// verifier checks, and it gets no interrupt polls.

using namespace brass;
using namespace brass::codegen;

namespace {

// %t (tagged) is live across the call to @g and read after it.
const char* kAcross = R"(module @m

func @g(%a: i64) -> i64 {
entry:
  %r = add.i64 %a, %a
  ret %r
}

func @f(%x: i64) -> i64 {
entry:
  %t = bitcast.tagged.i64 %x
  %c = call.i64 @g(%x)
  %u = bitcast.i64.tagged %t
  %r = xor.i64 %u, %c
  ret %r
}
)";

// %t is live across a call to @g and then across a call to @h.
const char* kAcrossTwo = R"(module @m

func @g(%a: i64) -> i64 {
entry:
  %r = add.i64 %a, %a
  ret %r
}

func @h(%a: i64) -> i64 {
entry:
  %r = mul.i64 %a, %a
  ret %r
}

func @f(%x: i64) -> i64 {
entry:
  %t = bitcast.tagged.i64 %x
  %c = call.i64 @g(%x)
  %d = call.i64 @h(%c)
  %u = bitcast.i64.tagged %t
  %r = xor.i64 %u, %d
  ret %r
}
)";

struct Allocated {
    std::unique_ptr<Module> mod;
    std::unique_ptr<LirFunction> lir;
};

// @f lowered for x64 SysV or aarch64 AAPCS64 and allocated, with the calls to
// the symbols in `quiet` marked as reaching no GC point (what the object
// writer's mark_no_gc_calls does for a nocollect callee).
Allocated allocate(const char* text, bool aarch64, std::initializer_list<const char*> quiet) {
    Allocated out;
    out.mod = parse_module(text);
    REQUIRE(out.mod != nullptr);
    Function* fn = out.mod->get_function("f");
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    if (aarch64) {
        brass::aarch64::AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
        out.lir = isel.lower(*fn);
    } else {
        brass::x64::X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
        out.lir = isel.lower(*fn);
    }
    REQUIRE(out.lir != nullptr);
    out.lir->sort_blocks_rpo();
    for (auto& b : out.lir->blocks) {
        for (auto& inst : b->instructions) {
            if (!inst->is_call()) continue;
            for (const char* q : quiet) {
                if (inst->callee_symbol == q) inst->no_gc_point = true;
            }
        }
    }
    LivenessAnalysis liveness(*out.lir);
    liveness.run();
    LinearScanAllocator regalloc(*out.lir, liveness, aarch64 ? CallingConvention::aapcs64() : CallingConvention::sysv64());
    regalloc.allocate();
    return out;
}

size_t slot_moves(const LirFunction& fn) {
    size_t n = 0;
    for (const auto& b : fn.blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->is_call()) continue;
            bool slot = false;
            for (const auto& d : inst->defs) slot = slot || d.is_spill_slot();
            for (const auto& u : inst->uses) slot = slot || u.is_spill_slot();
            if (slot) ++n;
        }
    }
    return n;
}

size_t call_roots(const LirFunction& fn) {
    size_t n = 0;
    for (const auto& b : fn.blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->is_call()) n += inst->live_gcrefs.size();
        }
    }
    return n;
}

} // namespace

TEST_CASE("nocollect - a tagged value live across a call keeps a register, and no-GC calls record no roots") {
    for (bool aarch64 : {false, true}) {
        // Across a collecting call: a callee-saved register, which no stack
        // map names.
        Allocated gc = allocate(kAcross, aarch64, {});
        CHECK_EQ(slot_moves(*gc.lir), size_t(0));
        CHECK_EQ(call_roots(*gc.lir), size_t(0));
        Allocated quiet = allocate(kAcross, aarch64, {"g"});
        CHECK_EQ(slot_moves(*quiet.lir), size_t(0));
        CHECK_EQ(call_roots(*quiet.lir), size_t(0));
    }
}

TEST_CASE("nocollect - a value that also spans a collecting call keeps its register") {
    for (bool aarch64 : {false, true}) {
        // @g reaches no GC point, @h may: %t spans both, in a callee-saved
        // register either way.
        Allocated one = allocate(kAcrossTwo, aarch64, {"g"});
        CHECK_EQ(slot_moves(*one.lir), size_t(0));
        CHECK_EQ(call_roots(*one.lir), size_t(0));
        Allocated both = allocate(kAcrossTwo, aarch64, {"g", "h"});
        CHECK_EQ(slot_moves(*both.lir), size_t(0));
    }
}

TEST_CASE("nocollect - a declared nocollect callee compiles and runs with the reference in a register") {
    auto mod = parse_module(kAcrossTwo);
    REQUIRE(mod != nullptr);
    mod->add_symbol_role("g", SymbolRole::NoCollect);
    mod->add_symbol_role("h", SymbolRole::NoCollect);
    REQUIRE(verify_module(*mod));
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(int64_t)>("f");
    REQUIRE(fn != nullptr);
    // Small-integer bits: no pointer a stack map could be asked to trace.
    for (int64_t x : {int64_t{3}, int64_t{-8}, int64_t{1} << 40}) CHECK_EQ(fn(x), x ^ ((x + x) * (x + x)));
}

namespace {

// @quiet is declared nocollect below; what its body may contain.
std::unique_ptr<Module> nocollect_body(const char* body) {
    std::string text = "module @m\n\nfunc @loud(%a: i64) -> i64 {\nentry:\n  ret %a\n}\n\n"
                       "func @calm(%a: i64) -> i64 {\nentry:\n  ret %a\n}\n\n"
                       "func @quiet(%a: i64, %p: ptr) -> i64 {\nentry:\n";
    text += body;
    text += "}\n";
    auto mod = parse_module(text);
    REQUIRE(mod != nullptr);
    mod->add_symbol_role("quiet", SymbolRole::NoCollect);
    mod->add_symbol_role("calm", SymbolRole::NoCollect);
    return mod;
}

} // namespace

TEST_CASE("nocollect - the verifier refuses a GC point in a nocollect body") {
    auto fine = nocollect_body("  %r = call.i64 @calm(%a)\n  %s = add.i64 %r, %a\n  ret %s\n");
    CHECK(verify_module(*fine));
    auto direct = nocollect_body("  %r = call.i64 @loud(%a)\n  ret %r\n");
    CHECK_FALSE(verify_module(*direct));
    auto indirect = nocollect_body("  %r = call_indirect.i64 %p(%a)\n  ret %r\n");
    CHECK_FALSE(verify_module(*indirect));
    auto safepoint = nocollect_body("  safepoint\n  ret %a\n");
    CHECK_FALSE(verify_module(*safepoint));
}

TEST_CASE("nocollect - a nocollect function's loops get no interrupt poll") {
    const char* text = R"(module @m

func @spin(%n: i64) -> i64 {
entry:
  br loop(%n)
loop(%i: i64):
  %one = iconst.i64 1
  %j = sub.i64 %i, %one
  %zero = iconst.i64 0
  %more = sgt.i64 %j, %zero
  br_if %more, loop(%j), done(%j)
done(%r: i64):
  ret %r
}
)";
    auto mod = parse_module(text);
    REQUIRE(mod != nullptr);
    mod->set_pinned_tls_register(true);
    mod->set_interrupt_poll(64, "handler");
    Function* fn = mod->get_function("spin");
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    CHECK(needs_interrupt_polls(*fn));
    mod->add_symbol_role("spin", SymbolRole::NoCollect);
    CHECK_FALSE(needs_interrupt_polls(*fn));
}
