// A pipeline's executes run on the calling thread's heap: the fast
// interpreter kept across executes serves only its own heap's thread, and a
// thread with another current heap runs in an interpreter on that heap, whose
// frames are that heap's roots through the collections the call makes.
#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/gc/runtime_gc.hpp>
#include "gc_test_heap.hpp"
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <iostream>
#include <memory>
#include <thread>

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

TieringConfig fast_no_tierup() {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(true);
    return cfg;
}

// Allocates a box holding %x, churns enough garbage to collect several
// times while the box is live in the frame, and returns the box.
const char* kHeld = R"(module @pth
func @pth_main(%x: i64) -> gcref {
entry:
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %o, 16, %x
  br loop(%z)
loop(%i: i64):
  %csz = iconst.i64 256
  %t = call.gcref @brass_gc_alloc(%csz, %z, %kind)
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %n = iconst.i64 20000
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  ret %o
}
)";

} // namespace

TEST_CASE("Pipeline thread heap - an execute on another thread runs on that thread's heap") {
    auto mod = parse_ok(kHeld);
    REQUIRE(mod != nullptr);
    // Outlives the pipeline, whose kept interpreter is built on it.
    test::BoundHeap home;
    FunctionDispatchTable prog;
    prog.pipeline().initialize(fast_no_tierup());
    ProgramScope scope(prog);

    RuntimeValue first = prog.pipeline().execute(*mod, "pth_main", {RuntimeValue::from_i64(11)});
    CHECK(home->is_valid_object(first.raw_bits()));
    CHECK_EQ(gc::Heap::load(first.raw_bits(), 2), 11ull);
    CHECK(home.minor_collections() >= 1);

    bool valid_there = false;
    bool absent_home = false;
    uint64_t value_there = 0;
    uint64_t collections_there = 0;
    std::thread worker([&] {
        ProgramScope worker_scope(prog);
        test::BoundHeap there;
        RuntimeValue r = prog.pipeline().execute(*mod, "pth_main", {RuntimeValue::from_i64(23)});
        valid_there = there->is_valid_object(r.raw_bits());
        absent_home = !home->contains(r.raw_bits());
        value_there = gc::Heap::load(r.raw_bits(), 2);
        collections_there = there.minor_collections();
    });
    worker.join();
    CHECK(valid_there);
    CHECK(absent_home);
    CHECK_EQ(value_there, 23ull);
    CHECK(collections_there >= 1);

    // The kept interpreter still serves its own thread.
    RuntimeValue again = prog.pipeline().execute(*mod, "pth_main", {RuntimeValue::from_i64(31)});
    CHECK(home->is_valid_object(again.raw_bits()));
    CHECK_EQ(gc::Heap::load(again.raw_bits(), 2), 31ull);
}
