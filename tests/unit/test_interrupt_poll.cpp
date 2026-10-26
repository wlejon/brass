// Interrupt polls (Module::interrupt_poll): every tier polls at its loop
// backedges, a poll that fires runs the module's handler, which may suspend
// the fiber the loop runs on (a host's time slice) or terminate it with a
// TerminationException no landing pad sees, and a collection while loops
// are suspended at their polls keeps their references.
#include "test_framework.hpp"
#include "gc_test_heap.hpp"
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/fiber.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/mir/interrupt_poll.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/vm/fast_interpreter.hpp>

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace brass;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

// The block the pinned TLS register points at; the poll word is tls[1].
constexpr int32_t kPollOffset = 8;

struct PollState {
    alignas(16) uint64_t tls[4] = {};
    int calls = 0;
    int disarm_after = 1;    // the poll is disarmed once this many calls ran
    bool yield = false;      // suspend the running fiber (and stay armed)
    bool terminate = false;  // throw TerminationException
};
thread_local PollState t_poll;

void reset_poll() { t_poll = PollState{}; }
void arm() { t_poll.tls[1] = UINT64_MAX; }
int64_t tls_arg() { return static_cast<int64_t>(reinterpret_cast<uintptr_t>(&t_poll.tls[0])); }

extern "C" void ip_handler() {
    ++t_poll.calls;
    if (t_poll.terminate) throw runtime::TerminationException(&t_poll);
    if (t_poll.yield && gc::Fiber::current() != nullptr) {
        gc::Fiber::suspend();
        return;
    }
    if (t_poll.calls >= t_poll.disarm_after) t_poll.tls[1] = 0;
}

// The loops read their stop word through a call, which no optimizer hoists.
extern "C" int64_t ip_read(int64_t addr) {
    return *reinterpret_cast<volatile int64_t*>(static_cast<uintptr_t>(addr));
}

// Every function takes (tls, a, b).
const char* kModule = R"(module @ip
attributes pinned_tls_register interrupt_poll 8 @ip_handler

extern @brass_gc_alloc
extern @ip_read

func @ip_sum(%tls: i64, %n: i64, %u: i64) -> i64 {
entry:
  pinned_tls_write %tls
  %z = iconst.i64 0
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %acc2 = add.i64 %acc, %i
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  %more = slt.i64 %i2, %n
  br_if %more, loop(%i2, %acc2), done(%acc2)
done(%r: i64):
  ret %r
}

func @ip_spin(%tls: i64, %stop: i64, %u: i64) -> i64 {
entry:
  pinned_tls_write %tls
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  %s = call.i64 @ip_read(%stop)
  %zero = iconst.i64 0
  %go = eq.i64 %s, %zero
  br_if %go, loop(%i2), done(%i2)
done(%r: i64):
  ret %r
}

func @ip_keep(%tls: i64, %stop: i64, %val: i64) -> i64 {
entry:
  pinned_tls_write %tls
  %sz = iconst.i64 48
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %o = call.gcref @brass_gc_alloc(%sz, %z, %kind)
  store.i64 %o, 16, %val
  br loop(%o)
loop(%p: gcref):
  %s = call.i64 @ip_read(%stop)
  %zero = iconst.i64 0
  %go = eq.i64 %s, %zero
  br_if %go, loop(%p), done(%p)
done(%q: gcref):
  %v = load.i64 %q, 16
  ret %v
}

func @ip_guarded(%tls: i64, %stop: i64, %flag: i64) -> i64 {
entry:
  %r = invoke.i64 @ip_spin(%tls, %stop, %flag), ok, pad
ok:
  ret %r
pad:
  %e = landing_pad
  %one = iconst.i64 1
  store.i64 %flag, %one
  ret %e
}

func @ip_straight(%tls: i64, %a: i64, %b: i64) -> i64 {
entry:
  pinned_tls_write %tls
  %s = add.i64 %a, %b
  ret %s
}
)";

std::unique_ptr<Module> parse_ok(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << vdiag.format_all() << "\n";
    REQUIRE(ok);
    return mod;
}

using Fn3 = int64_t (*)(int64_t, int64_t, int64_t);
using Call = std::function<int64_t(const char*, int64_t, int64_t, int64_t)>;

// One tier's way to call the module's functions.
struct Tier {
    const char* name;
    Call call;
};

// Tier 0: a FastInterpreter per call, as each stack gets its own.
Call tier0(const Module& mod) {
    return [&mod](const char* fn, int64_t a, int64_t b, int64_t c) {
        FastInterpreter fi;
        fi.set_module(&mod);
        fi.register_external_symbol("ip_handler", reinterpret_cast<void*>(&ip_handler));
        fi.register_external_symbol("ip_read", reinterpret_cast<void*>(&ip_read));
        return fi.run(*mod.get_function(fn), {RuntimeValue::from_i64(a), RuntimeValue::from_i64(b),
                                               RuntimeValue::from_i64(c)}).as_i64();
    };
}

struct Tier1 {
    codegen::BaselineJitCompiler compiler;
    std::vector<codegen::BaselineCompiledFunction> fns;
    explicit Tier1(const Module& mod) {
        compiler.register_external_symbol("ip_handler", reinterpret_cast<void*>(&ip_handler));
        compiler.register_external_symbol("brass_gc_alloc", reinterpret_cast<void*>(&brass_gc_alloc));
        compiler.register_external_symbol("ip_read", reinterpret_cast<void*>(&ip_read));
        fns = compiler.compile_module(mod);
    }
    Fn3 find(const char* name) const {
        for (const auto& f : fns) {
            if (f.name() == name) return f.get_function_ptr<Fn3>();
        }
        return nullptr;
    }
};

struct Tier2 {
    codegen::JitExecutionEngine jit;
    explicit Tier2(const Module& mod) {
        jit.register_external_symbol("ip_handler", reinterpret_cast<void*>(&ip_handler));
        jit.register_external_symbol("ip_read", reinterpret_cast<void*>(&ip_read));
        REQUIRE(jit.compile_and_load(mod));
    }
};

// Runs fn on a fiber of its own; the host resumes it.
struct FiberRun {
    std::function<int64_t()> body;
    int64_t result = -1;
    bool terminated = false;
    void* token = nullptr;
    std::unique_ptr<gc::Fiber> fiber;

    explicit FiberRun(std::function<int64_t()> b) : body(std::move(b)) {
        fiber = gc::Fiber::create(&FiberRun::entry, this, 1024 * 1024);
        REQUIRE(fiber != nullptr);
    }
    static void entry(void* arg) {
        auto* self = static_cast<FiberRun*>(arg);
        try {
            self->result = self->body();
        } catch (const runtime::TerminationException& e) {
            self->terminated = true;
            self->token = e.token();
        }
    }
};

template <typename F>
void for_each_tier(const Module& mod, F&& f) {
    Tier1 t1(mod);
    Tier2 t2(mod);
    const Tier tiers[] = {
        {"tier 0", tier0(mod)},
        {"tier 1", [&t1](const char* fn, int64_t a, int64_t b, int64_t c) {
             Fn3 p = t1.find(fn);
             REQUIRE(p != nullptr);
             return p(a, b, c);
         }},
        {"tier 2", [&t2](const char* fn, int64_t a, int64_t b, int64_t c) {
             Fn3 p = t2.jit.get_function_ptr<Fn3>(fn);
             REQUIRE(p != nullptr);
             return p(a, b, c);
         }},
    };
    for (const Tier& t : tiers) {
        reset_poll();
        f(t);
    }
    runtime::FunctionDispatchTable::instance().forget_module(mod);
}

} // namespace

TEST_CASE("Interrupt poll - the attribute prints and parses back") {
    auto mod = parse_ok(kModule);
    REQUIRE(static_cast<bool>(mod->interrupt_poll()));
    CHECK_EQ(mod->interrupt_poll().tls_offset, kPollOffset);
    CHECK(mod->interrupt_poll().handler == "ip_handler");
    CHECK(mod->has_external_symbol("ip_handler"));
    std::ostringstream os;
    print_module(*mod, os);
    CHECK(os.str().find("interrupt_poll 8 @ip_handler") != std::string::npos);
    auto again = parse_ok(os.str().c_str());
    CHECK_EQ(again->interrupt_poll().tls_offset, kPollOffset);
    CHECK(again->interrupt_poll().handler == "ip_handler");
}

TEST_CASE("Interrupt poll - tier 2's pass polls each back edge once and leaves straight code alone") {
    auto mod = parse_ok(kModule);
    CHECK(!needs_interrupt_polls(*mod->get_function("ip_straight")));
    CHECK(!needs_interrupt_polls(*mod->get_function("ip_guarded")));
    for (const char* name : {"ip_sum", "ip_spin", "ip_keep"}) {
        Module copy("copy");
        copy.set_pinned_tls_register(true);
        copy.copy_declarations_from(*mod);
        Function* fn = clone_function(*mod->get_function(name), copy);
        REQUIRE(needs_interrupt_polls(*fn));
        const size_t before = fn->blocks().size();
        CHECK_EQ(insert_interrupt_polls(*fn), size_t{1});
        // A conditional latch goes through a polling block, then the header
        // or the header's cold handler block.
        CHECK_EQ(fn->blocks().size(), before + 2);
        DiagnosticReporter diag;
        const bool ok = verify_function(*fn, &diag);
        if (!ok) std::cerr << diag.format_all() << "\n";
        CHECK(ok);
    }
    // An unconditional latch keeps its block: its jump becomes the poll's
    // branch to the header or the cold block.
    auto latch = parse_ok(R"(module @lt
attributes pinned_tls_register interrupt_poll 8 @ip_handler
func @lt(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br head(%z)
head(%i: i64):
  %more = slt.i64 %i, %n
  br_if %more, body, done(%i)
body:
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  br head(%i2)
done(%r: i64):
  ret %r
}
)");
    Function* lt = latch->get_function("lt");
    const size_t before = lt->blocks().size();
    CHECK_EQ(insert_interrupt_polls(*lt), size_t{1});
    CHECK_EQ(lt->blocks().size(), before + 1);
    DiagnosticReporter diag;
    CHECK(verify_function(*lt, &diag));
}

TEST_CASE("Interrupt poll - every tier runs a loop unchanged while disarmed and calls the handler when armed") {
    auto mod = parse_ok(kModule);
    for_each_tier(*mod, [](const Tier& t) {
        const int64_t n = 100000;
        CHECK_EQ(t.call("ip_sum", tls_arg(), n, 0), n * (n - 1) / 2);
        CHECK_EQ(t_poll.calls, 0);
        t_poll.disarm_after = 5;
        arm();
        CHECK_EQ(t.call("ip_sum", tls_arg(), n, 0), n * (n - 1) / 2);
        if (t_poll.calls != 5) std::cerr << t.name << ": " << t_poll.calls << " handler calls\n";
        CHECK_EQ(t_poll.calls, 5);
        CHECK_EQ(t_poll.tls[1], uint64_t{0});
        // Straight-line code never polls.
        arm();
        t_poll.calls = 0;
        CHECK_EQ(t.call("ip_straight", tls_arg(), 2, 3), int64_t{5});
        CHECK_EQ(t_poll.calls, 0);
    });
}

TEST_CASE("Interrupt poll - a runaway loop in each tier yields its fiber at the poll and resumes where it was") {
    auto mod = parse_ok(kModule);
    for_each_tier(*mod, [](const Tier& t) {
        int64_t stop = 0;
        t_poll.yield = true;
        arm();
        FiberRun run([&] { return t.call("ip_spin", tls_arg(), static_cast<int64_t>(reinterpret_cast<uintptr_t>(&stop)), 0); });
        for (int slice = 1; slice <= 20; ++slice) {
            run.fiber->resume();
            REQUIRE(!run.fiber->finished());
            CHECK_EQ(t_poll.calls, slice);
        }
        stop = 1;
        t_poll.tls[1] = 0;
        run.fiber->resume();
        CHECK(run.fiber->finished());
        CHECK(!run.terminated);
        // The loop resumed after each slice: it counted at least one
        // iteration per slice and the one that saw `stop`.
        CHECK(run.result >= 21);
    });
}

TEST_CASE("Interrupt poll - termination unwinds a loop in each tier past its caller's landing pad") {
    auto mod = parse_ok(kModule);
    for_each_tier(*mod, [](const Tier& t) {
        int64_t stop = 0;
        int64_t flag = 0;
        t_poll.terminate = true;
        arm();
        FiberRun run([&] {
            return t.call("ip_guarded", tls_arg(), static_cast<int64_t>(reinterpret_cast<uintptr_t>(&stop)),
                          static_cast<int64_t>(reinterpret_cast<uintptr_t>(&flag)));
        });
        run.fiber->resume();
        CHECK(run.fiber->finished());
        if (!run.terminated) std::cerr << t.name << ": not terminated\n";
        CHECK(run.terminated);
        CHECK(run.token == static_cast<void*>(&t_poll));
        CHECK_EQ(flag, int64_t{0});
        CHECK_EQ(std::uncaught_exceptions(), 0);
        // The tier runs on afterwards.
        reset_poll();
        CHECK_EQ(t.call("ip_sum", tls_arg(), 10, 0), int64_t{45});
    });
}

TEST_CASE("Interrupt poll - a collection while many loops are suspended at their polls keeps their references") {
    test::BoundHeap heap;
    auto mod = parse_ok(kModule);
    for_each_tier(*mod, [](const Tier& t) {
        int64_t stop = 0;
        t_poll.yield = true;
        arm();
        constexpr int kFibers = 16;
        std::vector<std::unique_ptr<FiberRun>> runs;
        for (int i = 0; i < kFibers; ++i) {
            const int64_t val = 1000 + i;
            runs.push_back(std::make_unique<FiberRun>([&t, &stop, val] {
                return t.call("ip_keep", tls_arg(), static_cast<int64_t>(reinterpret_cast<uintptr_t>(&stop)), val);
            }));
        }
        for (int round = 0; round < 4; ++round) {
            for (auto& r : runs) {
                r->fiber->resume();
                REQUIRE(!r->fiber->finished());
            }
            // Garbage between the slices, then a full collection: every
            // suspended loop's object moves or stays, and must be found.
            gc::Heap* heap = gc::Heap::current();
            REQUIRE(heap != nullptr);
            for (int k = 0; k < 2000; ++k) (void)heap->allocate_masked(64, 0, 2);
            heap->collect();
        }
        stop = 1;
        t_poll.tls[1] = 0;
        for (int i = 0; i < kFibers; ++i) {
            runs[i]->fiber->resume();
            CHECK(runs[i]->fiber->finished());
            CHECK_EQ(runs[i]->result, int64_t{1000 + i});
        }
    });
}

#endif
