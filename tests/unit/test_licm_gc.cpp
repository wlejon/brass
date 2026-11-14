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
