// Values live across a coro_suspend. The lowered body runs once per resume,
// entering at its dispatch block, so a value defined before a suspend and
// used after it comes from an earlier invocation. CoroTransformPass once
// spilled such values only just before the suspend and reloaded them only in
// blocks the resume block dominates: a merge point such as a loop header,
// re-entered from a resume, read a register the entry path defined (a
// parameter or an entry constant), the spill stores themselves used values
// that did not dominate them, and values used only as branch arguments were
// never spilled at all. The lowered body failed verification; run
// unverified it yielded garbage or hung.
#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/native_frames.hpp>
#include <brass/runtime/coroutine.hpp>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

using namespace brass;

namespace {

std::unique_ptr<Module> lower(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::printf("%s", diag.format_all().c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    CoroTransformPass pass;
    REQUIRE(pass.run_on_module(*mod));
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::printf("%s", vdiag.format_all().c_str());
    REQUIRE(ok);
    return mod;
}

enum class Tier { Interp, Fast, Baseline, Jit };
constexpr Tier kTiers[] = {Tier::Interp, Tier::Fast, Tier::Baseline, Tier::Jit};

const char* tier_name(Tier t) {
    switch (t) {
        case Tier::Interp: return "interpreter";
        case Tier::Fast: return "fast interpreter";
        case Tier::Baseline: return "baseline jit";
        case Tier::Jit: return "optimizing jit";
    }
    return "?";
}

struct Run {
    std::vector<uint64_t> yields;  // every resume's result, the last the return value
    bool done = false;
};

// Creates a frame for @name with `args` in its argument slots and resumes
// it with 0, 1, 2, ... until it is done (at most 16 times). `between` runs
// after every resume that suspended, with the (rooted) frame.
Run drive(Module& mod, const char* name, Tier tier, const std::vector<uint64_t>& args,
          const std::function<void(uintptr_t&)>& between = {}) {
    Function* fn = mod.get_function(name);
    REQUIRE(fn != nullptr);
    const CoroFrameLayout layout = compute_coro_frame_layout(*fn);
    const uint32_t slots = std::max<uint32_t>(layout.slot_count, static_cast<uint32_t>(args.size()));

    std::unique_ptr<codegen::JitExecutionEngine> jit;
    codegen::BaselineCompiledFunction baseline;
    void* code = nullptr;
    if (tier == Tier::Jit) {
        jit = std::make_unique<codegen::JitExecutionEngine>();
        REQUIRE(jit->compile_and_load(mod));
        code = jit->get_symbol_address(name);
    } else if (tier == Tier::Baseline) {
        codegen::BaselineJitCompiler compiler;
        baseline = compiler.compile(*fn);
        REQUIRE(baseline.is_valid());
        code = reinterpret_cast<void*>(baseline.get_function_ptr<uint64_t (*)(uintptr_t)>());
    }
    const bool native = tier == Tier::Jit || tier == Tier::Baseline;
    if (native) REQUIRE(code != nullptr);

    uintptr_t frame = brass_coro_create_at(native ? code : nullptr, slots, layout.pointer_mask, 0, 0);
    REQUIRE(frame != 0);
    ThreadRootsScope keep([](void* ctx, std::vector<uintptr_t*>& roots) {
        roots.push_back(static_cast<uintptr_t*>(ctx));
    }, &frame);
    for (size_t i = 0; i < args.size(); ++i) {
        reinterpret_cast<runtime::BrassCoroFrame*>(frame)->slots[i] = args[i];
    }

    Interpreter interp;
    FastInterpreter fast;
    Run run;
    for (uint64_t step = 0; step < 16; ++step) {
        uint64_t y = 0;
        if (native) {
            y = brass_coro_resume(frame, step);
        } else {
            reinterpret_cast<runtime::BrassCoroFrame*>(frame)->resume_arg = step;
            const std::vector<RuntimeValue> a = {RuntimeValue::from_ptr(frame)};
            const RuntimeValue r = tier == Tier::Interp ? interp.run(*fn, a) : fast.run(*fn, a);
            y = static_cast<uint64_t>(r.as_i64());
        }
        run.yields.push_back(y);
        if (reinterpret_cast<runtime::BrassCoroFrame*>(frame)->is_done) {
            run.done = true;
            break;
        }
        if (between) between(frame);
    }
    brass_coro_destroy(frame);
    return run;
}

void expect(Module& mod, const char* name, const std::vector<uint64_t>& args,
            const std::vector<uint64_t>& want) {
    for (Tier t : kTiers) {
        const Run r = drive(mod, name, t, args);
        if (r.yields != want || !r.done) std::printf("  @%s on the %s\n", name, tier_name(t));
        CHECK(r.done);
        CHECK(r.yields == want);
    }
}

// A parameter and an entry constant used after a suspend in a later block.
const char* kParamConst = R"(module @pc
func @pc(%p: i64) -> i64 {
b0:
  %k = iconst.i64 100
  br mid
mid:
  %y = coro_suspend.i64 %p, 1
  %s = add.i64 %p, %k
  %t = add.i64 %s, %y
  ret %t
}
)";

// A suspend inside a loop: the header, re-entered from the resume block,
// reads the parameter and an entry constant; loop-carried block parameters;
// a value from before the suspend used after it; an entry constant used only
// as a branch argument.
const char* kLoop = R"(module @lp
func @lp(%p: i64) -> i64 {
b0:
  %k = iconst.i64 100
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %a1 = add.i64 %acc, %p
  %a2 = add.i64 %a1, %k
  %y = coro_suspend.i64 %a2, 1
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %lim = iconst.i64 3
  %more = slt.i64 %ni, %lim
  br_if %more, loop(%ni, %a2), done(%a2, %k)
done(%r: i64, %kk: i64):
  %s = add.i64 %r, %kk
  ret %s
}
)";

// Two suspends in one loop iteration, the second in the first's resume
// block, with the resume arguments carried around the loop.
const char* kTwoSuspends = R"(module @ts
func @ts(%p: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z, %p)
loop(%i: i64, %acc: i64):
  %y1 = coro_suspend.i64 %acc, 1
  %b = add.i64 %acc, %y1
  %y2 = coro_suspend.i64 %b, 2
  %c = add.i64 %b, %y2
  %c2 = add.i64 %c, %p
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %lim = iconst.i64 2
  %more = slt.i64 %ni, %lim
  br_if %more, loop(%ni, %c2), done
done:
  ret %c2
}
)";

// A gcref argument and a loop-carried gcref block parameter, both live
// across the suspend, plus a caller that passes the gcref to coro_create.
const char* kGc = R"(module @gc
func @gcb(%o: gcref, %n: i64) -> i64 {
b0:
  %z = iconst.i64 0
  br loop(%z, %o)
loop(%i: i64, %cur: gcref):
  %v = load.i64 %cur, 16
  %y = coro_suspend.i64 %v, 1
  %one = iconst.i64 1
  %v2 = add.i64 %v, %one
  store.i64 %cur, 16, %v2
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni, %cur), done
done:
  %w = load.i64 %o, 16
  ret %w
}
func @make(%o: gcref) -> i64 {
b0:
  %n = iconst.i64 3
  %c = coro_create @gcb(%o, %n)
  %z = iconst.i64 0
  %r0 = coro_resume.i64 %c, %z
  %r1 = coro_resume.i64 %c, %z
  %r2 = coro_resume.i64 %c, %z
  %r3 = coro_resume.i64 %c, %z
  coro_destroy %c
  %m = iconst.i64 100
  %t0 = mul.i64 %r0, %m
  %t1 = add.i64 %t0, %r1
  %t2 = mul.i64 %t1, %m
  %t3 = add.i64 %t2, %r2
  %t4 = mul.i64 %t3, %m
  %t5 = add.i64 %t4, %r3
  ret %t5
}
)";

struct GenHeap {
    GenerationalGC gc{32 * 1024, 16 * 1024, 1 << 20, 2};
    GenHeap() { brass_set_active_generational_gc(&gc); }
    ~GenHeap() { brass_set_active_generational_gc(nullptr); }
};

} // namespace

TEST_CASE("Coro live values - a parameter and an entry constant survive a suspend") {
    auto mod = lower(kParamConst);
    // yields p; resumed with 1: p + 100 + 1.
    expect(*mod, "pc", {5}, {5, 106});
}

TEST_CASE("Coro live values - a loop header re-entered from a resume keeps params and constants") {
    auto mod = lower(kLoop);
    expect(*mod, "lp", {5}, {105, 210, 315, 415});
}

TEST_CASE("Coro live values - two suspends per loop iteration carry resume arguments") {
    auto mod = lower(kTwoSuspends);
    // acc=5: yield 5; resumed 1 -> b=6, yield 6; resumed 2 -> c=8, c2=13;
    // yield 13; resumed 3 -> b=16, yield 16; resumed 4 -> c=20, c2=25, done.
    expect(*mod, "ts", {5}, {5, 6, 13, 16, 25});
}

TEST_CASE("Coro live values - a gcref argument and a loop-carried gcref survive suspends") {
    auto mod = lower(kGc);
    const CoroFrameLayout layout = compute_coro_frame_layout(*mod->get_function("gcb"));
    CHECK((layout.pointer_mask & 1u) != 0);  // the gcref argument's slot
    for (Tier t : kTiers) {
        GenHeap heap;
        uintptr_t obj = heap.gc.allocate(48, 0, 2);
        REQUIRE(obj != 0);
        *reinterpret_cast<int64_t*>(obj + 16) = 40;
        const Run r = drive(*mod, "gcb", t, {obj, 3});
        if (!r.done) std::printf("  @gcb on the %s\n", tier_name(t));
        CHECK(r.done);
        CHECK((r.yields == std::vector<uint64_t>{40, 41, 42, 43}));
    }
    // coro_create passing the gcref, from the interpreter and the JIT.
    {
        GenHeap heap;
        uintptr_t obj = heap.gc.allocate(48, 0, 2);
        *reinterpret_cast<int64_t*>(obj + 16) = 40;
        Interpreter interp;
        interp.set_module(mod.get());
        CHECK_EQ(interp.run(*mod->get_function("make"), {RuntimeValue::from_ptr(obj)}).as_i64(), 40414243);
    }
    {
        GenHeap heap;
        uintptr_t obj = heap.gc.allocate(48, 0, 2);
        *reinterpret_cast<int64_t*>(obj + 16) = 40;
        codegen::JitExecutionEngine jit;
        REQUIRE(jit.compile_and_load(*mod));
        auto make = reinterpret_cast<int64_t (*)(uintptr_t)>(jit.get_symbol_address("make"));
        REQUIRE(make != nullptr);
        CHECK_EQ(make(obj), 40414243);
    }
}

// Collections while the coroutine is suspended holding the gcref in two
// frame slots (the argument and the loop-carried copy): both are updated,
// so the body resumes on the moved object.
TEST_CASE("Coro live values - a collection while suspended moves the frame's gcrefs") {
    auto mod = lower(kGc);
    const CoroFrameLayout layout = compute_coro_frame_layout(*mod->get_function("gcb"));
    int pointer_slots = 0;
    for (uint32_t s = 0; s < layout.slot_count; ++s) pointer_slots += ((layout.pointer_mask >> s) & 1) ? 1 : 0;
    CHECK(pointer_slots >= 2);
    for (Tier t : kTiers) {
        GenHeap heap;
        uintptr_t obj = heap.gc.allocate(48, 0, 2);
        REQUIRE(obj != 0);
        *reinterpret_cast<int64_t*>(obj + 16) = 40;
        ThreadRootsScope keep_obj([](void* ctx, std::vector<uintptr_t*>& roots) {
            roots.push_back(static_cast<uintptr_t*>(ctx));
        }, &obj);
        int collections = 0;
        bool moved = false;
        bool slots_ok = true;
        const Run r = drive(*mod, "gcb", t, {obj, 3}, [&](uintptr_t& frame) {
            const uintptr_t before = obj;
            if (collections++ % 2 == 0) heap.gc.minor_collect(); else heap.gc.major_collect();
            if (obj != before) moved = true;
            // Scribble over the old copy: a stale slot reads garbage.
            if (obj != before) *reinterpret_cast<int64_t*>(before + 16) = -1;
            auto* f = reinterpret_cast<runtime::BrassCoroFrame*>(frame);
            for (uint32_t s = 0; s < layout.slot_count; ++s) {
                if (((layout.pointer_mask >> s) & 1) && f->slots[s] != 0 && f->slots[s] != obj) slots_ok = false;
            }
        });
        if (!r.done || !moved || !slots_ok) std::printf("  @gcb under collections on the %s\n", tier_name(t));
        CHECK(r.done);
        CHECK(moved);
        CHECK(slots_ok);
        CHECK((r.yields == std::vector<uint64_t>{40, 41, 42, 43}));
    }
}
