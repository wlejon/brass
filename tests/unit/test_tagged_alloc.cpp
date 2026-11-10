// The module's tagged allocator (Module::tagged_allocator, tagged_alloc.hpp):
// the declaration round-trips through the text form; the analysis finds an
// object escaping exactly where it can be seen from outside the function;
// scalar replacement removes the ones that do not escape and answers their
// tag tests and identity comparisons; write_barrier_elim drops a barrier on
// a fresh small object; the verifier refuses a GC point before the payload
// is written.
#include "test_framework.hpp"
#include <brass/mir/escape_analysis.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/sroa.hpp>
#include <brass/mir/tagged_alloc.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/write_barrier_elim.hpp>
#include <brass/vm/fast_interpreter.hpp>

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

using namespace brass;

namespace {

// -4222124650659840 is 0xFFF1 << 48, the allocator's tag, and
// 281474976710655 the 48-bit address mask.
const char* kHeader = R"(module @ta
attributes tagged_allocator @ta_alloc -1 -1 0xfff1000000000000
extern @ta_sink
)";

std::unique_ptr<Module> parse(const std::string& body, bool must_verify = true) {
    const std::string src = std::string(kHeader) + body;
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    if (must_verify) {
        DiagnosticReporter vdiag;
        const bool ok = verify_module(*mod, &vdiag);
        if (!ok) std::cerr << vdiag.format_all() << "\n";
        REQUIRE(ok);
    }
    return mod;
}

// The one site of `fn`.
const TaggedAllocSite& only_site(const TaggedAllocations& a) {
    REQUIRE_EQ(a.sites().size(), size_t{1});
    return a.sites()[0];
}

size_t calls_to(const Function& fn, std::string_view sym) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) n += inst->opcode() == Opcode::call && inst->symbol() == sym;
    }
    return n;
}

size_t count_op(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) n += inst->opcode() == op;
    }
    return n;
}

// A two-word object {x, y}: its prologue, `%p` its address.
const char* kMake = R"(
  %bytes = iconst.i64 16
  %hdr = iconst.i64 0
  %o = call.tagged @ta_alloc(%bytes, %hdr)
  %ob = bitcast.i64.tagged %o
  %mask = iconst.i64 281474976710655
  %p = and.i64 %ob, %mask
  store.i64 %p, 0, %x
  store.i64 %p, 8, %y
)";

std::string fn_with(const std::string& name, const std::string& params, const std::string& rest) {
    return "func @" + name + "(" + params + ") -> i64 {\nb0:" + kMake + rest + "}\n";
}

// A bump arena for the interpreter runs: the allocator's contract, nothing
// collects.
alignas(16) uint64_t g_arena[4096];
size_t g_arena_top = 0;
int g_alloc_calls = 0;

extern "C" uint64_t ta_alloc(uint64_t bytes, uint64_t header) {
    ++g_alloc_calls;
    g_arena[g_arena_top] = header | bytes;
    const uint64_t payload = reinterpret_cast<uintptr_t>(&g_arena[g_arena_top + 1]);
    g_arena_top = (g_arena_top + 1 + bytes / 8) % 4000;
    return payload | 0xFFF1000000000000ULL;
}

int64_t run(Module& mod, std::string_view fn, std::initializer_list<int64_t> args) {
    FastInterpreter fi;
    fi.set_module(&mod);
    fi.register_external_function(
        "ta_alloc", std::function<RuntimeValue(const std::vector<RuntimeValue>&)>(
                        [](const std::vector<RuntimeValue>& a) {
                            return RuntimeValue::from_tagged(ta_alloc(a[0].raw_bits(), a[1].raw_bits()));
                        }));
    std::vector<RuntimeValue> a;
    for (int64_t v : args) a.push_back(RuntimeValue::from_i64(v));
    return fi.run(*mod.get_function(fn), a).as_i64();
}

} // namespace

TEST_CASE("Tagged allocator - the declaration prints and parses back") {
    auto mod = parse("");
    const Module::TaggedAllocator& a = mod->tagged_allocator();
    REQUIRE(static_cast<bool>(a));
    CHECK_EQ(a.symbol, std::string_view("ta_alloc"));
    CHECK(!a.inline_window());
    CHECK_EQ(a.tag, uint64_t{0xFFF1000000000000ULL});
    mod->set_tagged_allocator("ta_alloc", 16, 24, 0xFFF1000000000000ULL);
    std::ostringstream os;
    print_module(*mod, os);
    CHECK(os.str().find("tagged_allocator @ta_alloc 16 24 0xfff1000000000000") != std::string::npos);
    DiagnosticReporter diag;
    auto again = parse_module(os.str(), &diag);
    REQUIRE(again != nullptr);
    CHECK_EQ(again->tagged_allocator().top_offset, 16);
    CHECK_EQ(again->tagged_allocator().end_offset, 24);
    CHECK(again->has_external_symbol("ta_alloc"));
}

TEST_CASE("Tagged allocator - what lets an object escape") {
    struct Case {
        const char* name;
        const char* rest;
        bool escapes;
    };
    const Case cases[] = {
        // Read back, tag-tested and compared: the object stays here.
        {"reads", R"(  %a = load.i64 %p, 0
  %b = load.i64 %p, 8
  %s = add.i64 %a, %b
  ret %s
)", false},
        {"tag_test", R"(  %tm = iconst.i64 -281474976710656
  %t = and.i64 %ob, %tm
  %n = iconst.i64 -4503599627370496
  %isnum = ule.i64 %ob, %n
  %z = zext.i64 %isnum
  %s = add.i64 %t, %z
  ret %s
)", false},
        {"identity", R"(  %ob2 = bitcast.i64.tagged %o
  %same = eq.i64 %ob, %ob2
  %other = eq.i64 %ob, %x
  %d = sub.i32 %same, %other
  %r = zext.i64 %d
  ret %r
)", false},
        {"barrier", R"(  write_barrier %ob, %x
  keep_alive %o
  ret %x
)", false},
        // Seen from outside.
        {"stored", R"(  store.i64 %x, 0, %ob
  ret %x
)", true},
        {"stored_in_itself", R"(  store.i64 %p, 8, %ob
  ret %x
)", true},
        {"passed", R"(  %r = call.i64 @ta_sink(%ob)
  ret %r
)", true},
        {"returned", R"(  ret %ob
)", true},
        {"barrier_value", R"(  write_barrier %x, %ob
  ret %x
)", true},
        {"deopt_state", R"(  %one = iconst.i32 1
  guard %one, @ta_exit, [%ob]
  ret %x
)", true},
        {"address_compared", R"(  %c = ult.i64 %p, %x
  %r = zext.i64 %c
  ret %r
)", true},
        {"constant_in_range", R"(  %k = iconst.i64 -4222124650659840
  %c = eq.i64 %ob, %k
  %r = zext.i64 %c
  ret %r
)", true},
    };
    for (const Case& c : cases) {
        std::string body = fn_with(c.name, "%x: i64, %y: i64", c.rest);
        body += "func @ta_exit(%v: i64) -> i64 {\nb0:\n  ret %v\n}\n";
        auto mod = parse(body);
        const Function& fn = *mod->get_function(c.name);
        TaggedAllocations a(fn);
        const TaggedAllocSite& s = only_site(a);
        if (s.escapes() != c.escapes) std::cerr << c.name << ": escape '" << s.escape << "'\n";
        CHECK_EQ(s.escapes(), c.escapes);
        CHECK_EQ(s.bytes, int64_t{16});
        EscapeAnalysis ea(fn);
        CHECK_EQ(ea.does_escape(s.call->result()), c.escapes);
    }
}

TEST_CASE("Tagged allocator - parameters carry the object or let it escape") {
    // Both arms pass the object's address: the join's parameter is it.
    auto joined = parse(fn_with("joined", "%x: i64, %y: i64, %c: i32", R"(  br_if %c, l(%p), r(%p)
l(%lp: i64):
  br j(%lp)
r(%rp: i64):
  br j(%rp)
j(%q: i64):
  %v = load.i64 %q, 8
  ret %v
)"));
    CHECK(!only_site(TaggedAllocations(*joined->get_function("joined"))).escapes());
    // One arm passes something else: the parameter is not the object.
    auto mixed = parse(fn_with("mixed", "%x: i64, %y: i64, %c: i32", R"(  br_if %c, j(%p), j(%x)
j(%q: i64):
  %v = load.i64 %q, 8
  ret %v
)"));
    CHECK(only_site(TaggedAllocations(*mixed->get_function("mixed"))).escapes());
    // Different deltas into one parameter.
    auto skew = parse(fn_with("skew", "%x: i64, %y: i64, %c: i32", R"(  %eight = iconst.i64 8
  %p8 = add.i64 %p, %eight
  br_if %c, j(%p), j(%p8)
j(%q: i64):
  %v = load.i64 %q, 0
  ret %v
)"));
    CHECK(only_site(TaggedAllocations(*skew->get_function("skew"))).escapes());
}

TEST_CASE("Tagged allocator - scalar replacement removes the object and keeps the answers") {
    const std::string body =
        fn_with("sum", "%x: i64, %y: i64", R"(  %a = load.i64 %p, 0
  %b = load.i64 %p, 8
  %s = add.i64 %a, %b
  ret %s
)") +
        fn_with("folds", "%x: i64, %y: i64", R"(  %tm = iconst.i64 -281474976710656
  %t = and.i64 %ob, %tm
  %sh = iconst.i64 48
  %hi = lshr.i64 %ob, %sh
  %n = iconst.i64 -4503599627370496
  %isnum = ule.i64 %ob, %n
  %ob2 = bitcast.i64.tagged %o
  %same = eq.i64 %ob, %ob2
  %other = ne.i64 %ob, %x
  %tag = iconst.i64 -4222124650659840
  %u = xor.i64 %ob, %tag
  %w = load.i64 %u, 8
  %lo = load.i32 %p, 12
  %b1 = add.i32 %same, %other
  %b2 = sub.i32 %b1, %isnum
  %lo64 = zext.i64 %lo
  %z = zext.i64 %b2
  %r0 = add.i64 %hi, %z
  %r1 = add.i64 %r0, %w
  %r2 = add.i64 %r1, %lo64
  %r3 = xor.i64 %r2, %t
  ret %r3
)") +
        // A loop rewrites a field; a join merges two writes.
        fn_with("loop", "%x: i64, %y: i64", R"(  %z = iconst.i64 0
  br h(%z)
h(%i: i64):
  %cur = load.i64 %p, 0
  %one = iconst.i64 1
  %nx = add.i64 %cur, %one
  store.i64 %p, 0, %nx
  %i2 = add.i64 %i, %one
  %more = slt.i64 %i2, %y
  br_if %more, h(%i2), done
done:
  %f = load.i64 %p, 0
  ret %f
)");
    auto mod = parse(body);
    const int64_t x = 1000, y = 0x123456789;
    const int64_t want_sum = run(*mod, "sum", {x, y});
    const int64_t want_folds = run(*mod, "folds", {x, y});
    const int64_t want_loop = run(*mod, "loop", {x, 5});
    CHECK_EQ(want_sum, x + y);
    CHECK_EQ(want_loop, x + 5);
    for (const char* name : {"sum", "folds", "loop"}) {
        Function& fn = *mod->get_function(name);
        SroaStats stats;
        SroaOptions o;
        o.stats = &stats;
        CHECK_EQ(scalar_replace_tagged_allocations(fn, o), size_t{1});
        CHECK_EQ(calls_to(fn, "ta_alloc"), size_t{0});
        CHECK_EQ(count_op(fn, Opcode::store), size_t{0});
        CHECK_EQ(count_op(fn, Opcode::load), size_t{0});
        DiagnosticReporter diag;
        const bool ok = verify_function(fn, &diag);
        if (!ok) std::cerr << name << ":\n" << diag.format_all() << "\n";
        CHECK(ok);
    }
    g_alloc_calls = 0;
    CHECK_EQ(run(*mod, "sum", {x, y}), want_sum);
    CHECK_EQ(run(*mod, "folds", {x, y}), want_folds);
    CHECK_EQ(run(*mod, "loop", {x, 5}), want_loop);
    CHECK_EQ(g_alloc_calls, 0);
}

TEST_CASE("Tagged allocator - scalar replacement leaves an escaping object alone") {
    auto mod = parse(fn_with("kept", "%x: i64, %y: i64", R"(  %r = call.i64 @ta_sink(%ob)
  %a = load.i64 %p, 0
  ret %a
)"));
    Function& fn = *mod->get_function("kept");
    CHECK_EQ(scalar_replace_tagged_allocations(fn, SroaOptions{}), size_t{0});
    CHECK_EQ(calls_to(fn, "ta_alloc"), size_t{1});
    CHECK(!sroa_function(fn));
}

TEST_CASE("Tagged allocator - scalar replacement leaves an object live into a landing pad alone") {
    // Two invokes unwind to one pad, between them a store: the word the pad
    // reads would need a parameter there, and no edge passes the pad values.
    auto mod = parse(fn_with("pad", "%x: i64, %y: i64", R"(  %r1 = invoke.i64 @ta_work(%x), n1, pad
n1:
  store.i64 %p, 0, %r1
  %r2 = invoke.i64 @ta_work(%r1), n2, pad
n2:
  ret %r2
pad:
  %e = landing_pad
  %v = load.i64 %p, 0
  ret %v
)") + "func @ta_work(%v: i64) -> i64 {\nb0:\n  ret %v\n}\n");
    Function& fn = *mod->get_function("pad");
    CHECK(!only_site(TaggedAllocations(fn)).escapes());
    CHECK_EQ(scalar_replace_tagged_allocations(fn, SroaOptions{}), size_t{0});
    CHECK_EQ(calls_to(fn, "ta_alloc"), size_t{1});
}

TEST_CASE("Tagged allocator - a fresh small object's barriers go, until a GC point") {
    auto mod = parse(fn_with("wb", "%x: i64, %y: i64", R"(  store.i64 %p, 8, %x
  write_barrier %ob, %x
  %u = xor.i64 %ob, %x
  %r = call.i64 @ta_sink(%ob)
  store.i64 %p, 0, %y
  write_barrier %ob, %y
  ret %r
)") + R"(func @big(%x: i64) -> i64 {
b0:
  %bytes = iconst.i64 1048576
  %hdr = iconst.i64 0
  %o = call.tagged @ta_alloc(%bytes, %hdr)
  %ob = bitcast.i64.tagged %o
  %mask = iconst.i64 281474976710655
  %p = and.i64 %ob, %mask
  store.i64 %p, 0, %x
  write_barrier %ob, %x
  ret %x
}
)", false);
    WriteBarrierElimination wbe;
    wbe.run_on_function(*mod->get_function("wb"));
    // The first follows the allocation; the second follows a call.
    CHECK_EQ(count_op(*mod->get_function("wb"), Opcode::write_barrier), size_t{1});
    CHECK_EQ(wbe.stats().eliminated_young_provenance, size_t{1});
    WriteBarrierElimination wbe2;
    wbe2.run_on_function(*mod->get_function("big"));
    CHECK_EQ(count_op(*mod->get_function("big"), Opcode::write_barrier), size_t{1});
}

TEST_CASE("Tagged allocator - the verifier wants the payload written before a GC point") {
    const char* partial = R"(func @partial(%x: i64) -> i64 {
b0:
  %bytes = iconst.i64 16
  %hdr = iconst.i64 0
  %o = call.tagged @ta_alloc(%bytes, %hdr)
  %ob = bitcast.i64.tagged %o
  %mask = iconst.i64 281474976710655
  %p = and.i64 %ob, %mask
  store.i64 %p, 0, %x
  %r = call.i64 @ta_sink(%ob)
  ret %r
}
)";
    auto bad = parse(partial, false);
    DiagnosticReporter diag;
    CHECK(!verify_module(*bad, &diag));
    CHECK(diag.format_all().find("tagged allocation") != std::string::npos);
    // Written on one arm only: the join's return sees it half written.
    auto arm = parse(R"(func @arm(%x: i64, %c: i32) -> i64 {
b0:
  %bytes = iconst.i64 8
  %hdr = iconst.i64 0
  %o = call.tagged @ta_alloc(%bytes, %hdr)
  %ob = bitcast.i64.tagged %o
  %mask = iconst.i64 281474976710655
  %p = and.i64 %ob, %mask
  br_if %c, w, j
w:
  store.i64 %p, 0, %x
  br j
j:
  ret %ob
}
)", false);
    DiagnosticReporter diag2;
    CHECK(!verify_module(*arm, &diag2));
    // Whole on every path: fine.
    parse(fn_with("whole", "%x: i64, %y: i64", R"(  %r = call.i64 @ta_sink(%ob)
  ret %r
)"));
}
