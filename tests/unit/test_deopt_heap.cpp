// A tier-2 deoptimization finishes the call in the heap of the code that
// called it, and one name's handle follows the function that owns it.
//  - Under a running interpreter the continuation re-enters it: a gcref the
//    exit stub allocates lives in that heap, and the outer gcrefs it holds
//    are that GC's roots.
//  - Entered from the host, a fresh interpreter allocates from the thread's
//    active GC.
//  - Two modules defining one name in the default program: the second
//    one's function never runs the first one's native code.
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/gc/runtime_gc.hpp>
#include "gc_test_heap.hpp"
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
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

TieringConfig no_tierup() {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(false);
    return cfg;
}

const char* kDeopt = R"(module @dh
func @dh_churn(%n: i64) -> i64 {
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
func @dh_exit(%o: gcref, %x: i64) -> i64 {
b0:
  %r = call.i64 @dh_churn(%x)
  %v = load.i64 %o, 16
  ret %v
}
func @dh_spec(%o: gcref, %x: i64, %t: i32) -> i64 {
b0:
  %one = iconst.i32 1
  %c = eq.i32 %t, %one
  guard %c, @dh_exit, [%o, %x]
  %v = load.i64 %o, 16
  ret %v
}
func @dh_mainr(%x: i64) -> i64 {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  %k = iconst.i64 42
  store.i64 %o, 16, %k
  %zt = iconst.i32 0
  %v = call.i64 @dh_spec(%o, %x, %zt)
  ret %v
}
func @dh_exitb(%x: i64) -> gcref {
b0:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %o, 16, %x
  ret %o
}
func @dh_specb(%x: i64, %t: i32) -> gcref {
b0:
  %one = iconst.i32 1
  %c = eq.i32 %t, %one
  guard %c, @dh_exitb, [%x]
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %o, 16, %x
  ret %o
}
func @dh_mainb(%x: i64) -> i64 {
b0:
  %zt = iconst.i32 0
  %o1 = call.gcref @dh_specb(%x, %zt)
  %y = iconst.i64 5555
  %o2 = call.gcref @dh_specb(%y, %zt)
  %v = load.i64 %o1, 16
  ret %v
}
)";

// Installs `spec` at tier 2 in `prog`; the caller owns the rest.
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

TEST_CASE("Deopt heap - a gcref the exit stub returns lives in the calling interpreter's heap") {
    auto mod = parse_ok(kDeopt);
    REQUIRE(mod != nullptr);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup());
    Interpreter interp(test::small_heap_config());
    ActiveGc active(interp.heap());
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    REQUIRE(install_spec(prog, *mod, "dh_specb"));

    const uint64_t deopts = prog.pipeline().tier2_deopts();
    RuntimeValue r = interp.run(*mod->get_function("dh_specb"), {RuntimeValue::from_i64(7), RuntimeValue::from_i32(0)});
    CHECK(prog.pipeline().tier2_deopts() == deopts + 1);
    CHECK(interp.heap().is_valid_object(r.raw_bits()));
    CHECK_EQ(gc::Heap::load(r.raw_bits(), 2), 7ull);
    for (int64_t x : {11, 22, 33}) {
        CHECK_EQ(interp.run(*mod->get_function("dh_mainb"), {RuntimeValue::from_i64(x)}).as_i64(), x);
    }
}

TEST_CASE("Deopt heap - outer gcrefs the deopt continuation holds are updated by a collection") {
    auto mod = parse_ok(kDeopt);
    REQUIRE(mod != nullptr);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup());
    // A 64 KB eden: @dh_churn(1000)'s objects collect it.
    Interpreter interp(test::small_heap_config());
    ActiveGc active(interp.heap());
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    REQUIRE(install_spec(prog, *mod, "dh_spec"));
    REQUIRE(prog.pipeline().compile_and_install_tier1("dh_churn", mod->get_function("dh_churn")));

    for (int64_t x : {1, 10, 100, 1000}) {
        CHECK_EQ(interp.run(*mod->get_function("dh_mainr"), {RuntimeValue::from_i64(x)}).as_i64(), 42);
    }
    CHECK(interp.heap().collection_count() >= 1);
}

TEST_CASE("Deopt heap - entered from the host, the continuation allocates in the active GC") {
    auto mod = parse_ok(kDeopt);
    REQUIRE(mod != nullptr);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup());
    gc::Heap heap(test::small_heap_config());
    ActiveGc active(heap);
    REQUIRE(install_spec(prog, *mod, "dh_specb"));
    FunctionHandle* h = prog.find("dh_specb");
    REQUIRE(h != nullptr && h->has_native_entry());

    ProgramScope scope(prog);
    RuntimeValue r = h->call_native({RuntimeValue::from_i64(9), RuntimeValue::from_i32(0)});
    CHECK(prog.pipeline().tier2_deopts() >= 1);
    CHECK(heap.is_valid_object(r.raw_bits()));
    CHECK_EQ(gc::Heap::load(r.raw_bits(), 2), 9ull);
}

namespace {

const char* kModA = R"(module @ma
func @twomod_f(%x: i64) -> i64 {
b0:
  %one = iconst.i64 1
  %r = add.i64 %x, %one
  ret %r
}
func @twomod_main(%x: i64) -> i64 {
b0:
  %r = call.i64 @twomod_f(%x)
  ret %r
}
)";
const char* kModB = R"(module @mb
func @twomod_f(%x: i64) -> i64 {
b0:
  %h = iconst.i64 100
  %r = mul.i64 %x, %h
  ret %r
}
func @twomod_main(%x: i64) -> i64 {
b0:
  %r = call.i64 @twomod_f(%x)
  ret %r
}
)";

} // namespace

TEST_CASE("Two modules - a same-named function never runs the other module's native code") {
    auto a = parse_ok(kModA);
    auto b = parse_ok(kModB);
    REQUIRE(a != nullptr && b != nullptr);
    FunctionDispatchTable& prog = FunctionDispatchTable::instance();
    MultiTierPipeline::instance().initialize(no_tierup());
    REQUIRE(prog.pipeline().compile_and_install_tier1("twomod_f", a->get_function("twomod_f")));

    Interpreter ia;
    ia.set_module(a.get());
    CHECK_EQ(ia.run(*a->get_function("twomod_main"), {RuntimeValue::from_i64(7)}).as_i64(), 8);
    Interpreter ib;
    ib.set_module(b.get());
    CHECK_EQ(ib.run(*b->get_function("twomod_main"), {RuntimeValue::from_i64(7)}).as_i64(), 700);
    Interpreter direct;
    CHECK_EQ(direct.run(*b->get_function("twomod_f"), {RuntimeValue::from_i64(7)}).as_i64(), 700);
    // A's code was retired, not freed, and A's function takes the name back.
    CHECK_EQ(ia.run(*a->get_function("twomod_main"), {RuntimeValue::from_i64(7)}).as_i64(), 8);
    FunctionDispatchTable::instance().forget_module(*a);
    FunctionDispatchTable::instance().forget_module(*b);
}

TEST_CASE("Two modules - a module replacing a destroyed one does not inherit its native code") {
    MultiTierPipeline::instance().initialize(no_tierup());
    {
        auto a = parse_ok(kModA);
        REQUIRE(a != nullptr);
        REQUIRE(FunctionDispatchTable::instance().pipeline().compile_and_install_tier1("twomod_f",
                                                                                      a->get_function("twomod_f")));
        Interpreter ia;
        CHECK_EQ(ia.run(*a->get_function("twomod_f"), {RuntimeValue::from_i64(7)}).as_i64(), 8);
    }
    auto b = parse_ok(kModB);
    REQUIRE(b != nullptr);
    Interpreter ib;
    CHECK_EQ(ib.run(*b->get_function("twomod_main"), {RuntimeValue::from_i64(7)}).as_i64(), 700);
    FunctionDispatchTable::instance().forget_module(*b);
}
