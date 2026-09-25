// Deoptimization with the fast interpreter as Tier 0, and handles rebound
// while code compiled from their old function still runs.
//  - Under a running FastInterpreter the deopt continuation re-enters it: a
//    gcref the exit stub allocates lives in its heap, and the outer gcrefs
//    the continuation holds are that GC's roots.
//  - Entered from the host, a fresh FastInterpreter allocates from the
//    thread's active GC and its frames are that GC's roots.
//  - A call that entered a function's native code finishes with that
//    code's signature, although the handle is rebound meanwhile.
//  - Tier-2 code a handle was rebound away from still deoptimizes through
//    its resumer, into the Function it was compiled from.
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/gc/runtime_gc.hpp>
#include "gc_test_heap.hpp"
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::runtime;

namespace {

std::unique_ptr<Module> parse_ok(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod || !verify_module(*mod, &diag)) {
        std::cerr << diag.format_all();
        return nullptr;
    }
    return mod;
}

TieringConfig no_tierup(bool fast) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(fast);
    return cfg;
}

const char* kDeopt = R"(module @dfr
func @dfr_churn(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %sz = iconst.i64 64
  %kind = iconst.i32 2
  %t = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %big = iconst.i64 777
  store.i64 %t, 16, %big
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  ret %n
}
func @dfr_exit(%o: gcref, %x: i64) -> i64 {
b0:
  %r = call.i64 @dfr_churn(%x)
  %v = load.i64 %o, 16
  ret %v
}
func @dfr_spec(%o: gcref, %x: i64, %t: i32) -> i64 {
b0:
  %one = iconst.i32 1
  %c = eq.i32 %t, %one
  guard %c, @dfr_exit, [%o, %x]
  %v = load.i64 %o, 16
  ret %v
}
func @dfr_mainr(%x: i64) -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %o, 16, %k
  %zt = iconst.i32 0
  %v = call.i64 @dfr_spec(%o, %x, %zt)
  ret %v
}
func @dfr_exitb(%x: i64) -> gcref {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %o, 16, %x
  ret %o
}
func @dfr_specb(%x: i64, %t: i32) -> gcref {
b0:
  %one = iconst.i32 1
  %c = eq.i32 %t, %one
  guard %c, @dfr_exitb, [%x]
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %o, 16, %x
  ret %o
}
)";

bool install_spec(FunctionDispatchTable& prog, Module& mod, const char* spec) {
    prog.tiering().get_feedback(spec).set_deopt_threshold(1000000);
    FunctionHandle* h = prog.get_or_create(spec, mod.get_function(spec));
    CodeInstaller installer(prog);
    CodeInstallResult res = installer.install_tier2(*h, mod, spec);
    if (!res.success) std::cerr << res.error_message << "\n";
    return res.success;
}

using ActiveGc = gc::HeapScope;

} // namespace

TEST_CASE("Fast deopt - a gcref the exit stub returns lives in the running fast interpreter's heap") {
    auto mod = parse_ok(kDeopt);
    REQUIRE(mod != nullptr);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(true));
    FastInterpreter interp(test::small_heap_config());
    ActiveGc active(interp.heap());
    interp.set_dispatch_table(&prog);
    REQUIRE(install_spec(prog, *mod, "dfr_specb"));

    const uint64_t deopts = prog.pipeline().tier2_deopts();
    RuntimeValue r = interp.run(*mod->get_function("dfr_specb"), {RuntimeValue::from_i64(7), RuntimeValue::from_i32(0)});
    CHECK(prog.pipeline().tier2_deopts() == deopts + 1);
    CHECK(interp.heap().is_valid_object(r.raw_bits()));
    CHECK_EQ(gc::Heap::load(r.raw_bits(), 2), 7ull);
}

TEST_CASE("Fast deopt - outer gcrefs the continuation holds are updated by a collection") {
    auto mod = parse_ok(kDeopt);
    REQUIRE(mod != nullptr);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(true));
    // A 64 KB eden: @dfr_churn(1000)'s objects collect it.
    FastInterpreter interp(test::small_heap_config());
    ActiveGc active(interp.heap());
    interp.set_dispatch_table(&prog);
    REQUIRE(install_spec(prog, *mod, "dfr_spec"));
    REQUIRE(prog.pipeline().compile_and_install_tier1("dfr_churn", mod->get_function("dfr_churn")));

    for (int64_t x : {1, 10, 100, 1000}) {
        CHECK_EQ(interp.run(*mod->get_function("dfr_mainr"), {RuntimeValue::from_i64(x)}).as_i64(), 42);
    }
    CHECK(interp.heap().collection_count() >= 1);
}

TEST_CASE("Fast deopt - entered from the host, the continuation allocates in the active GC") {
    auto mod = parse_ok(kDeopt);
    REQUIRE(mod != nullptr);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(true));
    gc::Heap heap(test::small_heap_config());
    ActiveGc active(heap);
    REQUIRE(install_spec(prog, *mod, "dfr_specb"));
    FunctionHandle* h = prog.find("dfr_specb");
    REQUIRE(h != nullptr && h->has_native_entry());

    ProgramScope scope(prog);
    RuntimeValue r = h->call_native({RuntimeValue::from_i64(9), RuntimeValue::from_i32(0)});
    CHECK(prog.pipeline().tier2_deopts() >= 1);
    CHECK(heap.is_valid_object(r.raw_bits()));
    CHECK_EQ(gc::Heap::load(r.raw_bits(), 2), 9ull);
}

TEST_CASE("Fast deopt - entered from the host, the continuation's frames are roots of the active GC") {
    auto mod = parse_ok(kDeopt);
    REQUIRE(mod != nullptr);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(true));
    gc::Heap heap(test::small_heap_config());
    ActiveGc active(heap);
    REQUIRE(install_spec(prog, *mod, "dfr_spec"));
    REQUIRE(prog.pipeline().compile_and_install_tier1("dfr_churn", mod->get_function("dfr_churn")));
    FunctionHandle* h = prog.find("dfr_spec");
    REQUIRE(h != nullptr && h->has_native_entry());

    ProgramScope scope(prog);
    const uintptr_t o = heap.allocate_masked(48, 0, 2);
    heap.store(o, 2, uint64_t{42});
    const size_t before = heap.collection_count();
    RuntimeValue r = h->call_native({RuntimeValue::from_gcref(o), RuntimeValue::from_i64(1000), RuntimeValue::from_i32(0)});
    CHECK(heap.collection_count() > before);
    CHECK_EQ(r.as_i64(), 42);
}

namespace {

// @rb_f of module A (-> i64) calls the host, which runs module B's @rb_f
// (-> f64) on the same program: that rebinds the handle while A's native
// code is still running.
const char* kRebindA = R"(module @rba
func @rb_f(%x: i64) -> i64 {
b0:
  %r = call.i64 @dfr_rebind_cb(%x)
  %one = iconst.i64 1
  %s = add.i64 %x, %one
  ret %s
}
)";
const char* kRebindB = R"(module @rbb
func @rb_f(%x: i64) -> f64 {
b0:
  %d = fconst.f64 2.5
  ret %d
}
)";

// @rg of module A is tier-2 code whose guard fails after the host rebinds
// the handle to module B's @rg; the frame must finish in A's exit stub.
const char* kGuardA = R"(module @rga
func @rg_exit(%x: i64) -> i64 {
b0:
  %k = iconst.i64 1000
  %r = add.i64 %x, %k
  ret %r
}
func @rg(%x: i64, %t: i32) -> i64 {
b0:
  %r = call.i64 @dfr_rebind_cb(%x)
  %one = iconst.i32 1
  %c = eq.i32 %t, %one
  guard %c, @rg_exit, [%x]
  ret %x
}
)";
const char* kGuardB = R"(module @rgb
func @rg(%x: i64, %t: i32) -> i64 {
b0:
  %k = iconst.i64 -5
  ret %k
}
)";

FunctionDispatchTable* g_prog = nullptr;
Module* g_other = nullptr;
std::string g_other_fn;

extern "C" int64_t dfr_rebind_cb(int64_t x) {
    Interpreter ib;
    ib.set_dispatch_table(g_prog);
    ib.set_module(g_other);
    const Function* fn = g_other->get_function(g_other_fn);
    std::vector<RuntimeValue> args{RuntimeValue::from_i64(x)};
    if (fn->param_count() == 2) args.push_back(RuntimeValue::from_i32(0));
    ib.run(*fn, args);
    return 0;
}

} // namespace

TEST_CASE("Rebind - a call finishes with the signature of the code it entered") {
    auto a = parse_ok(kRebindA);
    auto b = parse_ok(kRebindB);
    REQUIRE(a != nullptr && b != nullptr);
    FunctionDispatchTable prog;
    g_prog = &prog;
    g_other = b.get();
    g_other_fn = "rb_f";
    prog.pipeline().initialize(no_tierup(false));
    prog.pipeline().register_external_symbol("dfr_rebind_cb", reinterpret_cast<void*>(&dfr_rebind_cb));
    prog.get_or_create("rb_f", a->get_function("rb_f"));
    REQUIRE(prog.pipeline().compile_and_install_tier1("rb_f", a->get_function("rb_f")));

    Interpreter ia;
    ia.set_dispatch_table(&prog);
    ia.set_module(a.get());
    RuntimeValue r = ia.run(*a->get_function("rb_f"), {RuntimeValue::from_i64(41)});
    CHECK(r.type().kind() == TypeKind::I64);
    CHECK_EQ(r.raw_bits(), 42ull);
    // The handle now describes B's function.
    CHECK(prog.find("rb_f")->return_type().kind() == TypeKind::F64);
}

TEST_CASE("Rebind - retained tier-2 code deoptimizes into the Function it was compiled from") {
    auto a = parse_ok(kGuardA);
    auto b = parse_ok(kGuardB);
    REQUIRE(a != nullptr && b != nullptr);
    FunctionDispatchTable prog;
    g_prog = &prog;
    g_other = b.get();
    g_other_fn = "rg";
    prog.pipeline().initialize(no_tierup(false));
    prog.tiering().get_feedback("rg").set_deopt_threshold(1000000);
    FunctionHandle* h = prog.get_or_create("rg", a->get_function("rg"));
    CodeInstaller installer(prog);
    installer.register_external_symbol("dfr_rebind_cb", reinterpret_cast<void*>(&dfr_rebind_cb));
    CodeInstallResult res = installer.install_tier2(*h, *a, "rg");
    if (!res.success) std::cerr << res.error_message << "\n";
    REQUIRE(res.success);

    Interpreter ia;
    ia.set_dispatch_table(&prog);
    ia.set_module(a.get());
    const uint64_t deopts = prog.pipeline().tier2_deopts();
    RuntimeValue r = ia.run(*a->get_function("rg"), {RuntimeValue::from_i64(7), RuntimeValue::from_i32(0)});
    CHECK_EQ(r.as_i64(), 1007);
    // The retained code's resumer finished the call, in A's @rg.
    CHECK(prog.pipeline().tier2_deopts() == deopts + 1);
    CHECK(h->mir_function() == b->get_function("rg"));
    // B's function runs B's code.
    Interpreter ib;
    ib.set_dispatch_table(&prog);
    ib.set_module(b.get());
    CHECK_EQ(ib.run(*b->get_function("rg"), {RuntimeValue::from_i64(7), RuntimeValue::from_i32(0)}).as_i64(), -5);
}
