// LICM and a moving collector (licm.cpp, gc_remat.hpp): what is made of a
// tagged value's bits leaves a loop only when no GC point of the loop can
// move the object, a load through such an address only when no test before
// it in an iteration asks what the object is, and the interrupt poll the
// back edge gets later computes a hoisted address again after its handler.
// GVN's merge of tagged reads (gvn_gc_free.hpp) treats a polled loop's
// header as the GC point its poll will be.
#include "test_framework.hpp"
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/interrupt_poll.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/host_symbols.hpp>

#include <cstdint>
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

TEST_CASE("LICM - what an inner loop put in front of itself is still inside the loop around it") {
    // The inner loop is entered from a block with two ways out, so it gets
    // a new block in front of it for its constant; that block runs once per
    // outer trip, and what reads the constant after the inner loop must not
    // go above the outer loop without it.
    auto mod = parse_ok(R"(module @m

func @f(%n: i64, %k: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br outer_head(%z)
outer_head(%i: i64):
  %more_i = slt.i64 %i, %n
  br_if %more_i, inner_head(%i), done
inner_head(%j: i64):
  %seven = iconst.i64 7
  %j2 = add.i64 %j, %seven
  %more_j = slt.i64 %j2, %k
  br_if %more_j, inner_head(%j2), tail
tail:
  %step = add.i64 %k, %seven
  %i2 = add.i64 %i, %step
  br outer_head(%i2)
done:
  ret %i
}
)");
    Function* fn = mod->get_function("f");
    CHECK(hoist_loop_invariants(*fn));
    verify_ok(*fn);
}

TEST_CASE("LICM - non_trapping_load view_data chain hoists past invariant-load bounds check") {
    // Models Bronze TypedArray lowering: %len is read in entry before the loop,
    // the loop checks %i < %len, and view_data's three header loads carry
    // non_trapping_load so the entire pointer-chasing chain hoists to entry.
    auto build_case = [](bool with_inloop_type_check) {
        auto mod = std::make_unique<Module>("non_trapping_view");
        mod->set_pinned_tls_register(true);
        mod->set_interrupt_poll(8, "handler");
        mod->add_external_symbol("handler");
        Builder b(*mod);
        Function* h = mod->create_function("handler", Type::void_type(), {});
        {
            b.set_function(h);
            b.append_block("entry");
            b.build_ret_void();
        }
        Function* fn = mod->create_function("fill", Type::i64(), {Type::tagged()});
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        BasicBlock* head = b.create_block("head");
        BasicBlock* body = b.create_block("body");
        BasicBlock* bail = b.create_block("bail");
        BasicBlock* done = b.create_block("done");
        fn->append_block(head);
        fn->append_block(body);
        fn->append_block(bail);
        fn->append_block(done);

        Value* ta = b.add_block_param(entry, Type::tagged());
        Value* mask = b.build_iconst_i64(0x0000FFFFFFFFFFFFLL);
        Value* ta_ptr = b.build_and(b.build_bitcast_i64_tagged(ta), mask);
        Value* len = b.build_zext_i64(b.build_load(Type::i32(), ta_ptr, 68));
        Value* zero = b.build_iconst_i64(0);
        b.build_br(head, {zero});

        b.position_at_end(head);
        Value* iv = b.add_block_param(head, Type::i64());
        Value* cond = b.build_ult(iv, len);
        b.build_br_if(cond, body, {}, done, {iv});

        b.position_at_end(body);
        if (with_inloop_type_check) {
            Value* inloop_bits = b.build_and(b.build_bitcast_i64_tagged(ta), mask);
            Value* kind = b.build_load(Type::i32(), inloop_bits, 32);
            Value* ok = b.build_eq(kind, b.build_iconst_i32(1));
            BasicBlock* after_check = b.create_block("after_check");
            fn->append_block(after_check);
            b.build_br_if(ok, after_check, bail);
            b.position_at_end(after_check);
        }
        Value* loop_ta_ptr = b.build_and(b.build_bitcast_i64_tagged(ta), mask);
        Value* buf_raw = b.build_load(Type::i64(), loop_ta_ptr, 56);
        buf_raw->defining_instruction()->set_alias_class(1);
        buf_raw->defining_instruction()->set_non_trapping_load(true);
        Value* buf = b.build_and(buf_raw, mask);

        Value* ext = b.build_load(Type::i64(), buf, 72);
        ext->defining_instruction()->set_alias_class(1);
        ext->defining_instruction()->set_non_trapping_load(true);

        Value* inline_ptr = b.build_add(buf, b.build_iconst_i64(80));
        Value* data_base = b.build_select(b.build_ne(ext, zero), ext, inline_ptr);

        Value* byte_off = b.build_load(Type::i32(), loop_ta_ptr, 64);
        byte_off->defining_instruction()->set_alias_class(1);
        byte_off->defining_instruction()->set_non_trapping_load(true);

        Value* data = b.build_add(data_base, b.build_zext_i64(byte_off));
        Instruction* st = b.build_store_indexed(Type::i32(), data, iv, 4, 0, b.build_trunc_i32(iv));
        st->set_alias_class(2);
        Value* next = b.build_add(iv, b.build_iconst_i64(1));
        b.build_br(head, {next});

        b.position_at_end(bail);
        b.build_ret(b.build_iconst_i64(-1));

        b.position_at_end(done);
        Value* retv = b.add_block_param(done, Type::i64());
        b.build_ret(retv);

        fn->rebuild_cfg_predecessors();
        verify_ok(*fn);
        return mod;
    };

    // Case 1: Only invariant-load bounds test in loop -> all view_data loads,
    // select, and byteOffset add hoist to entry, and interrupt_poll works.
    {
        auto mod = build_case(/*with_inloop_type_check=*/false);
        Function* fn = mod->get_function("fill");
        CHECK(hoist_loop_invariants(*fn));
        verify_ok(*fn);
        CHECK_EQ(count_in(*fn, Opcode::load, "body"), size_t{0});
        CHECK_EQ(count_in(*fn, Opcode::select, "body"), size_t{0});
        CHECK_EQ(count_in(*fn, Opcode::load, "entry"), size_t{4});
        CHECK_EQ(count_in(*fn, Opcode::select, "entry"), size_t{1});

        REQUIRE(needs_interrupt_polls(*fn));
        CHECK(insert_interrupt_polls(*fn) >= size_t{1});
        verify_ok(*fn);

        // Execute JIT-compiled fill against mock TypedArray + ArrayBuffer structs.
        struct MockBuf {
            uint8_t pad[72];
            uint64_t ext_ptr;
            uint32_t data[16];
        } buf{};
        buf.ext_ptr = 0;  // use inline data at offset 80
        struct MockTA {
            uint8_t pad0[32];
            uint32_t kind;
            uint8_t pad1[20];
            uint64_t buffer;
            uint32_t byte_offset;
            uint32_t length;
        } ta{};
        ta.kind = 1;
        ta.buffer = (0xFFF1ULL << 48) | reinterpret_cast<uintptr_t>(&buf);
        ta.byte_offset = 8;  // start at buf.data[2]
        ta.length = 6;

        uint64_t tls_block[4] = {0, 0x0FFFFFFFFFFFFFFFULL, 0, 0};
        struct HostProvider final : runtime::HostSymbolProvider {
            void* tls;
            explicit HostProvider(void* p) : tls(p) {}
            void* pinned_tls_block() override { return tls; }
        } provider(tls_block);
        runtime::HostSymbolProvider* prev_host = runtime::host_symbol_provider();
        runtime::set_host_symbol_provider(&provider);
        struct RestoreHost {
            runtime::HostSymbolProvider* prev;
            ~RestoreHost() { runtime::set_host_symbol_provider(prev); }
        } restore_host{prev_host};
        codegen::JitExecutionEngine jit;
        jit.register_external_symbol("handler", reinterpret_cast<void*>(+[]() {}));
        REQUIRE(jit.compile_and_load(*mod));
        const uint64_t ta_tagged = (0xFFF1ULL << 48) | reinterpret_cast<uintptr_t>(&ta);
        RuntimeValue rv = jit.invoke("fill", {RuntimeValue::from_tagged(ta_tagged)});
        CHECK_EQ(rv.as_i64(), 6);
        for (uint32_t i = 0; i < 6; ++i) {
            CHECK_EQ(buf.data[2 + i], i);
        }
    }

    // Case 2: An in-loop type check before view_data still blocks hoisting.
    {
        auto mod = build_case(/*with_inloop_type_check=*/true);
        Function* fn = mod->get_function("fill");
        hoist_loop_invariants(*fn);
        verify_ok(*fn);
        CHECK_EQ(count_in(*fn, Opcode::load, "after_check"), size_t{3});
        CHECK_EQ(count_in(*fn, Opcode::select, "after_check"), size_t{1});
    }
}
