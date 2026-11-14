// LICM and a moving collector (licm.cpp, gc_remat.hpp): what is made of a
// tagged value's bits leaves a loop only when no GC point of the loop can
// move the object, a load through such an address only when no test before
// it in an iteration asks what the object is, and the interrupt poll the
// back edge gets later computes a hoisted address again after its handler.
// GVN's merge of tagged reads (gvn_gc_free.hpp) treats a polled loop's
// header as the GC point its poll will be.
#include "test_framework.hpp"
#include <brass/mir/gvn.hpp>
#include <brass/mir/interrupt_poll.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/verifier.hpp>

#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

using namespace brass;

namespace {

std::unique_ptr<Module> parse_ok(const std::string& src) {
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

void verify_ok(Function& fn) {
    DiagnosticReporter diag;
    const bool ok = verify_function(fn, &diag);
    if (!ok) std::cerr << diag.format_all() << "\n";
    CHECK(ok);
}

size_t count_in(const Function& fn, Opcode op, std::string_view block = {}) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb || (!block.empty() && bb->name() != block)) continue;
        for (const Instruction* inst : *bb) n += inst->opcode() == op;
    }
    return n;
}

const char* kPolled = "attributes pinned_tls_register interrupt_poll 8 @handler\n";

// A loop summing element i of the Array-like object `t`: the elements
// block's word (offset 24) and the length (offset 8) read through t's bits.
// `extra` goes in the body before the element load.
std::string sum_loop(const std::string& attrs, const std::string& extra) {
    return "module @m\n" + attrs + R"(extern @gc_point
extern @handler

func @f(%t: tagged, %n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br head(%z, %z)
head(%i: i64, %acc: i64):
  %more = slt.i64 %i, %n
  br_if %more, body, done(%acc)
body:
  %bits = bitcast.i64.tagged %t
  %len = load.i32 %bits, 8
  %blk = load.i64 %bits, 24
)" + extra + R"(  %e = load_indexed.i64 %blk, %i, 8, 8
  %l64 = zext.i64 %len
  %s = add.i64 %e, %l64
  %acc2 = add.i64 %acc, %s
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  br head(%i2, %acc2)
done(%r: i64):
  ret %r
}
)";
}

} // namespace

TEST_CASE("LICM - a tagged value's bits and loads through them leave a loop with no GC point") {
    auto mod = parse_ok(sum_loop("", ""));
    Function* fn = mod->get_function("f");
    CHECK(hoist_loop_invariants(*fn));
    verify_ok(*fn);
    // The loads move with a copy of the bits they read; the bits themselves
    // stay (no longer used, for DCE).
    CHECK_EQ(count_in(*fn, Opcode::load, "body"), size_t{0});
    CHECK_EQ(count_in(*fn, Opcode::load, "entry"), size_t{2});
    CHECK_EQ(count_in(*fn, Opcode::bitcast_i64_tagged, "entry"), size_t{1});
    CHECK_EQ(count_in(*fn, Opcode::load_indexed, "body"), size_t{1});
}

TEST_CASE("LICM - a GC point in the loop keeps what reads where the object is") {
    auto mod = parse_ok(sum_loop("", "  call @gc_point()\n"));
    Function* fn = mod->get_function("f");
    hoist_loop_invariants(*fn);
    verify_ok(*fn);
    CHECK_EQ(count_in(*fn, Opcode::bitcast_i64_tagged, "body"), size_t{1});
    // The block's word may be stale past the call; the length is no address
    // but the call may write it.
    CHECK_EQ(count_in(*fn, Opcode::load, "body"), size_t{2});
}

TEST_CASE("LICM - a store the load may read keeps it in the loop") {
    auto mod = parse_ok(sum_loop("", "  %w = iconst.i32 7\n  store.i32 %bits, 8, %w\n"));
    Function* fn = mod->get_function("f");
    hoist_loop_invariants(*fn);
    verify_ok(*fn);
    // The length load stays (the store writes it); the block's word moves.
    CHECK_EQ(count_in(*fn, Opcode::load, "body"), size_t{1});
}

TEST_CASE("LICM - a load after a test of its object stays, the test's condition moves") {
    auto mod = parse_ok(R"(module @m
func @f(%t: tagged, %n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br head(%z, %z)
head(%i: i64, %acc: i64):
  %more = slt.i64 %i, %n
  br_if %more, body, done(%acc)
body:
  %bits = bitcast.i64.tagged %t
  %k = iconst.i64 -4222124650659840
  %x = xor.i64 %bits, %k
  %lim = iconst.i64 281474976710656
  %isobj = ult.i64 %x, %lim
  br_if %isobj, read, bail
read:
  %len = load.i32 %x, 8
  %l64 = zext.i64 %len
  %acc2 = add.i64 %acc, %l64
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  br head(%i2, %acc2)
bail:
  ret %z
done(%r: i64):
  ret %r
}
)");
    Function* fn = mod->get_function("f");
    CHECK(hoist_loop_invariants(*fn));
    verify_ok(*fn);
    // Hoisted, the load would read a number's bits as an address.
    CHECK_EQ(count_in(*fn, Opcode::load, "read"), size_t{1});
    // The tag test is no address: it moves, with a copy of what it reads.
    CHECK_EQ(count_in(*fn, Opcode::ult, "body"), size_t{0});
    CHECK_EQ(count_in(*fn, Opcode::bitcast_i64_tagged), size_t{2});
}

// A loop over the view `%buf`, read out of the object `%o` before the loop,
// whose exit test compares the counter with `bound`.
std::string view_loop(const std::string& bound) {
    return R"(module @m
func @f(%o: tagged) -> i64 {
entry:
  %ob = bitcast.i64.tagged %o
  %bufbits = load.i64 %ob, 24
  %buf = bitcast.tagged.i64 %bufbits
  %n = load.i64 %ob, 16
  %z = iconst.i64 0
  br head(%z, %z)
head(%i: i64, %acc: i64):
  %vb = bitcast.i64.tagged %buf
  %more = ne.i64 %i, )" + bound + R"(
  br_if %more, body, done(%acc)
body:
  %bits = bitcast.i64.tagged %buf
  %len = load.i32 %bits, 8
  %l64 = zext.i64 %len
  %acc2 = add.i64 %acc, %l64
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  br head(%i2, %acc2)
done(%r: i64):
  ret %r
}
)";
}

TEST_CASE("LICM - a test of a number read from the object a view came out of is no test of the view") {
    // `for (i = 0; i !== this.n; ++i) this.buffer[...]`: the bound and the
    // view were both read from `this`, and the bound says nothing about what
    // the view is.
    auto mod = parse_ok(view_loop("%n"));
    Function* fn = mod->get_function("f");
    CHECK(hoist_loop_invariants(*fn));
    verify_ok(*fn);
    CHECK_EQ(count_in(*fn, Opcode::load, "body"), size_t{0});
}

TEST_CASE("LICM - a test of the view's own bits keeps a load through it") {
    auto mod = parse_ok(view_loop("%vb"));
    Function* fn = mod->get_function("f");
    hoist_loop_invariants(*fn);
    verify_ok(*fn);
    CHECK_EQ(count_in(*fn, Opcode::load, "body"), size_t{1});
}

// A loop whose header tests its counter, then what kind of object `%t` is,
// both by guards, and reads the object's length behind the second. `first`
// goes at the top of the header.
std::string guarded_loop(const std::string& first) {
    return R"(module @m
func @out(%0: tagged, %1: i64) -> i64 {
bb0:
  ret %1
}
func @f(%t: tagged, %n: i64, %p: ptr) -> i64 {
entry:
  %z = iconst.i64 0
  br head(%z, %z)
head(%i: i64, %acc: i64):
)" + first + R"(  %more = ne.i64 %i, %n
  guard %more, @out, [%t, %acc]
  %bits = bitcast.i64.tagged %t
  %hdr = load.i32 %bits, 0
  %want = iconst.i32 7
  %is = eq.i32 %hdr, %want
  guard %is, @out, [%t, %acc]
  %len = load.i32 %bits, 8
  %l64 = zext.i64 %len
  %acc2 = add.i64 %acc, %l64
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  %go = slt.i64 %i2, %n
  br_if %go, head(%i2, %acc2), done(%acc2)
done(%r: i64):
  ret %r
}
)";
}

TEST_CASE("LICM - a header guard of a value the loop does not compute is tested once, ahead of the loop") {
    auto mod = parse_ok(guarded_loop(""));
    Function* fn = mod->get_function("f");
    CHECK(hoist_loop_invariants(*fn));
    verify_ok(*fn);
    // The kind test left the loop; the counter's test above it was copied
    // for the first iteration, with the state that iteration leaves with.
    CHECK_EQ(count_in(*fn, Opcode::guard, "head"), size_t{1});
    CHECK_EQ(count_in(*fn, Opcode::guard, "entry"), size_t{2});
    const BasicBlock* entry = fn->entry_block();
    const Instruction* copy = nullptr;
    for (const Instruction* inst : *entry) {
        if (inst->opcode() == Opcode::guard && !copy) copy = inst;
    }
    REQUIRE(copy != nullptr);
    REQUIRE(copy->state_map().size() == size_t{2});
    const Instruction* acc0 = copy->state_map()[1]->defining_instruction();
    CHECK(acc0 != nullptr && acc0->opcode() == Opcode::iconst_i64);
    // With no test of the object left in the loop, the length follows.
    CHECK_EQ(count_in(*fn, Opcode::load, "head"), size_t{0});
}

TEST_CASE("LICM - a header guard below a write stays in the loop") {
    // What the write stored would be stored once more were the header's top
    // run ahead of the loop.
    auto mod = parse_ok(guarded_loop("  store.i64 %p, 0, %acc\n"));
    Function* fn = mod->get_function("f");
    hoist_loop_invariants(*fn);
    verify_ok(*fn);
    CHECK_EQ(count_in(*fn, Opcode::guard, "head"), size_t{2});
    CHECK_EQ(count_in(*fn, Opcode::guard, "entry"), size_t{0});
}

TEST_CASE("Constant folding - a counter passes the whole-number test of an index") {
    // `i` counts 0, 1, 2, ...: converted to an i64 and back it is itself. A
    // counter by two, or one that starts at a fraction, is not folded.
    const auto loop = [](const char* init, const char* step) {
        return std::string(R"(module @m
func @f(%n: f64) -> i32 {
entry:
  %z = fconst.f64 )") + init + R"(
  %a = iconst.i32 0
  br head(%z, %a)
head(%i: f64, %acc: i32):
  %more = ne.f64 %i, %n
  br_if %more, body, done(%acc)
body:
  %w = fptosi.i64 %i
  %back = sitofp.f64.i64 %w
  %same = eq.f64 %back, %i
  %acc2 = add.i32 %acc, %same
  %one = fconst.f64 )" + step + R"(
  %i2 = add.f64 %i, %one
  br head(%i2, %acc2)
done(%r: i32):
  ret %r
}
)";
    };
    {
        auto mod = parse_ok(loop("0.0", "1.0"));
        Function* fn = mod->get_function("f");
        CHECK(fold_constants(*fn));
        eliminate_dead_code(*fn);
        verify_ok(*fn);
        CHECK_EQ(count_in(*fn, Opcode::fptosi_i64), size_t{0});
    }
    for (const auto& [init, step] : {std::pair{"0.0", "2.0"}, std::pair{"0.5", "1.0"}, std::pair{"-1.0", "1.0"}}) {
        auto mod = parse_ok(loop(init, step));
        Function* fn = mod->get_function("f");
        fold_constants(*fn);
        eliminate_dead_code(*fn);
        verify_ok(*fn);
        CHECK_EQ(count_in(*fn, Opcode::fptosi_i64), size_t{1});
    }
}

TEST_CASE("Interrupt poll - a hoisted address is computed again after the handler") {
    auto mod = parse_ok(sum_loop(kPolled, ""));
    Function* fn = mod->get_function("f");
    CHECK(hoist_loop_invariants(*fn));
    REQUIRE(needs_interrupt_polls(*fn));
    CHECK_EQ(insert_interrupt_polls(*fn), size_t{1});
    verify_ok(*fn);
    // The handler's block reads the bits and the block's word again; the
    // length is no address and is not read again.
    const BasicBlock* handler = nullptr;
    for (const BasicBlock* bb : fn->blocks()) {
        if (bb && bb->head() && bb->head()->is_call()) handler = bb;
    }
    REQUIRE(handler != nullptr);
    auto in_handler = [&](Opcode op) {
        size_t n = 0;
        for (const Instruction* inst : *handler) n += inst->opcode() == op;
        return n;
    };
    CHECK_EQ(in_handler(Opcode::bitcast_i64_tagged), size_t{1});
    CHECK_EQ(in_handler(Opcode::load), size_t{1});
}

TEST_CASE("GVN - a polled loop's header is a GC point for tagged reads") {
    const std::string body = R"(extern @handler
func @f(%t: tagged, %n: i64) -> i64 {
entry:
  %a = bitcast.i64.tagged %t
  %z = iconst.i64 0
  br head(%z, %a)
head(%i: i64, %acc: i64):
  %more = slt.i64 %i, %n
  br_if %more, body, done(%acc)
body:
  %c = bitcast.i64.tagged %t
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  %acc2 = add.i64 %acc, %c
  br head(%i2, %acc2)
done(%r: i64):
  ret %r
}
)";
    auto plain = parse_ok("module @m\n" + body);
    gvn_function(*plain->get_function("f"));
    CHECK_EQ(count_in(*plain->get_function("f"), Opcode::bitcast_i64_tagged), size_t{1});
    auto polled = parse_ok(std::string("module @m\n") + kPolled + body);
    gvn_function(*polled->get_function("f"));
    CHECK_EQ(count_in(*polled->get_function("f"), Opcode::bitcast_i64_tagged), size_t{2});
}

TEST_CASE("LICM - an i64 induction cycle is not GC-sensitive past a GC point") {
    auto mod = parse_ok(R"(module @m
extern @gc_point

func @f(%n: i64, %k: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %one = iconst.i64 1
  br outer_head(%z, %z)
outer_head(%i: i64, %acc: i64):
  %more_i = slt.i64 %i, %n
  br_if %more_i, inner_head(%z, %acc), done(%acc)
inner_head(%j: i64, %inner_acc: i64):
  %more_j = slt.i64 %j, %n
  br_if %more_j, inner_body, outer_latch(%inner_acc)
inner_body:
  %inv = mul.i64 %i, %k
  call @gc_point()
  %sum = add.i64 %inner_acc, %inv
  %j2 = add.i64 %j, %one
  br inner_head(%j2, %sum)
outer_latch(%next_acc: i64):
  %i2 = add.i64 %i, %one
  br outer_head(%i2, %next_acc)
done(%r: i64):
  ret %r
}
)");
    Function* fn = mod->get_function("f");
    CHECK(hoist_loop_invariants(*fn));
    verify_ok(*fn);
    CHECK_EQ(count_in(*fn, Opcode::mul, "inner_body"), size_t{0});
}
