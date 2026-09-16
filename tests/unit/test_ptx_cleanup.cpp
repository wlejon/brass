// The ptx::cleanup passes (ptx_cleanup.hpp) and the immediate /
// special-register handling in PtxISel.
//
//   (a) each pass on a small hand-built ptx::Function, checked on the printed
//       text: copy propagation (both-single-def and coalescing), dead
//       instruction elimination (and what it must keep), branch
//       simplification (fall-through, inversion, unreachable blocks),
//       register renumbering; every pass is a no-op on clean input,
//   (b) the ISel folds constants into the positions the rule table allows
//       and reads invariant special registers once,
//   (c) the whole pipeline on every intrinsic of the table and on the ten
//       MlFusionCompiler::build_ptx_* kernels: verify passes, ptxas assembles
//       (visible [SKIP] without ptxas), the instruction count drops, and a
//       second cleanup changes nothing. Before/after counts are printed.

#include "ptx_test_support.hpp"

#include <brass/codegen/ml_fusion.hpp>
#include <brass/target/ptx/ptx_cleanup.hpp>

#include <algorithm>
#include <cstdio>

// The unqualified Type/Opcode/Function of this file are the ptx IR ones; the
// MIR-building tests name brass::Type locally.
using namespace ptxtest;
namespace codegen = brass::codegen;
namespace ptx = brass::ptx;
namespace target = brass::target;
using brass::BasicBlock;
using brass::Builder;
using brass::Function;
using brass::Module;
using brass::Value;
using ptx::Block;
using ptx::CmpOp;
using ptx::Inst;
using ptx::Opcode;
using ptx::Operand;
using ptx::Reg;
using ptx::RegClass;
using ptx::SpecialReg;
using ptx::StateSpace;
using ptx::Type;

namespace {

// The block bodies of a function, printed one instruction per line with the
// same text the printer emits (labels included), for exact comparisons.
std::string body_text(const ptx::Function& fn) {
    std::string out;
    for (const auto& b : fn.blocks) {
        out += b->label + ":\n";
        for (const Inst& i : b->insts) out += "  " + ptx::to_string(i) + "\n";
    }
    return out;
}

void expect_body(const ptx::Function& fn, const std::string& expected) {
    std::string got = body_text(fn);
    if (got != expected) std::cerr << "expected:\n" << expected << "got:\n" << got;
    CHECK_EQ(got, expected);
}

void expect_verifies(const ptx::Function& fn) {
    auto diags = ptx::verify(fn);
    if (!diags.empty()) std::cerr << ptx::format_diagnostics(diags) << ptx::print_body(fn);
    CHECK(diags.empty());
}

// A kernel with one u64 param (loaded into %rd0) and an entry block.
struct Fixture {
    ptx::Function fn{"k"};
    Block* entry = nullptr;
    Reg rd0;

    Fixture() {
        Operand p0 = fn.add_param(Type::u64, "param_0");
        rd0 = fn.new_b64();
        entry = fn.add_block("$L_entry");
        entry->append(Inst::make(Opcode::ld, Type::u64).space(StateSpace::param).dst(rd0).src(p0));
    }
};

Inst st_u32(Reg base, int32_t disp, Reg v) {
    return Inst::make(Opcode::st, Type::u32).space(StateSpace::global).src(Operand::addr(base, disp)).src(v);
}

} // namespace

// ---------------------------------------------------------------------------
// (a) Passes on hand-built functions
// ---------------------------------------------------------------------------

TEST_CASE("PTX cleanup - copy propagation replaces a single-def copy and resolves chains") {
    Fixture fx;
    Reg r0 = fx.fn.new_b32(), r1 = fx.fn.new_b32(), r2 = fx.fn.new_b32(), r3 = fx.fn.new_b32();
    Reg f0 = fx.fn.new_f32(), f1 = fx.fn.new_f32();
    Block* b = fx.entry;
    b->append(Inst::make(Opcode::mov, Type::u32).dst(r0).src(Operand::special(SpecialReg::tid_x)));
    b->append(Inst::make(Opcode::mov, Type::b32).dst(r1).src(r0));          // r1 = r0
    b->append(Inst::make(Opcode::mov, Type::b32).dst(r2).src(r1));          // r2 = r1 (chain)
    b->append(Inst::make(Opcode::add, Type::s32).dst(r3).src(r2).src(Operand::imm(1)));
    b->append(Inst::make(Opcode::mov, Type::b32).dst(f0).src(r3));          // cross-class: not a copy
    b->append(Inst::make(Opcode::mov, Type::f32).dst(f1).src(f0));
    b->append(st_u32(fx.rd0, 0, r2));
    b->append(Inst::make(Opcode::st, Type::f32).space(StateSpace::global).src(Operand::addr(fx.rd0, 4)).src(f1));
    b->append(Inst::make(Opcode::ret));

    CHECK_EQ(ptx::propagate_copies(fx.fn), size_t(3));
    expect_body(fx.fn,
        "$L_entry:\n"
        "  ld.param.u64 %rd0, [param_0];\n"
        "  mov.u32 %r0, %tid.x;\n"
        "  add.s32 %r3, %r0, 1;\n"
        "  mov.b32 %f0, %r3;\n"
        "  st.global.u32 [%rd0], %r0;\n"
        "  st.global.f32 [%rd0 + 4], %f0;\n"
        "  ret;\n");
    expect_verifies(fx.fn);
    CHECK_EQ(ptx::propagate_copies(fx.fn), size_t(0)); // no-op on clean input
}

TEST_CASE("PTX cleanup - copy coalescing writes a block parameter directly, but not across a branch or a read") {
    // loop(i): i = 0; head: setp; @p bra exit; body: t = i + 1; mov i, t; bra head
    Fixture fx;
    Reg i = fx.fn.new_b32(), t = fx.fn.new_b32(), n = fx.fn.new_b32(), u = fx.fn.new_b32(), v = fx.fn.new_b32();
    Reg p = fx.fn.new_pred();
    Block* entry = fx.entry;
    Block* head = fx.fn.add_block("$L_head");
    Block* body = fx.fn.add_block("$L_body");
    Block* exit = fx.fn.add_block("$L_exit");
    entry->append(Inst::make(Opcode::ld, Type::u32).space(StateSpace::global).dst(n).src(Operand::addr(fx.rd0)));
    entry->append(Inst::make(Opcode::mov, Type::b32).dst(i).src(Operand::imm(0)));
    entry->append(Inst::make(Opcode::bra).src(Operand::label(head->label)));
    head->append(Inst::make(Opcode::setp, Type::s32).cmp(CmpOp::ge).dst(p).src(i).src(n));
    head->append(Inst::make(Opcode::bra).guard(p).src(Operand::label(exit->label)));
    // Coalescable: t is single-def, single-use, nothing touches i in between.
    body->append(Inst::make(Opcode::add, Type::s32).dst(t).src(i).src(Operand::imm(1)));
    body->append(st_u32(fx.rd0, 4, t));   // second use of t: blocks coalescing until removed below
    body->append(Inst::make(Opcode::mov, Type::b32).dst(i).src(t));
    body->append(Inst::make(Opcode::bra).src(Operand::label(head->label)));
    // Not coalescable: a guarded bra between the def of u and the mov (the
    // taken path would see the early write), and a read of v's target
    // between def and mov (the swap pattern).
    exit->append(Inst::make(Opcode::add, Type::s32).dst(u).src(n).src(Operand::imm(2)));
    exit->append(Inst::make(Opcode::bra).guard(p).src(Operand::label(head->label)));
    exit->append(Inst::make(Opcode::mov, Type::b32).dst(n).src(u));
    exit->append(Inst::make(Opcode::add, Type::s32).dst(v).src(n).src(Operand::imm(3)));
    exit->append(st_u32(fx.rd0, 8, n));
    exit->append(Inst::make(Opcode::mov, Type::b32).dst(n).src(v));
    exit->append(st_u32(fx.rd0, 12, n));
    exit->append(Inst::make(Opcode::ret));

    // t has two uses: nothing to coalesce in body; u and v never qualify.
    CHECK_EQ(ptx::propagate_copies(fx.fn), size_t(0));
    // Drop the extra use of t: the add now writes i directly.
    body->insts.erase(body->insts.begin() + 1);
    CHECK_EQ(ptx::propagate_copies(fx.fn), size_t(1));
    expect_body(fx.fn,
        "$L_entry:\n"
        "  ld.param.u64 %rd0, [param_0];\n"
        "  ld.global.u32 %r2, [%rd0];\n"
        "  mov.b32 %r0, 0;\n"
        "  bra $L_head;\n"
        "$L_head:\n"
        "  setp.ge.s32 %p0, %r0, %r2;\n"
        "  @%p0 bra $L_exit;\n"
        "$L_body:\n"
        "  add.s32 %r0, %r0, 1;\n"
        "  bra $L_head;\n"
        "$L_exit:\n"
        "  add.s32 %r3, %r2, 2;\n"
        "  @%p0 bra $L_head;\n"
        "  mov.b32 %r2, %r3;\n"
        "  add.s32 %r4, %r2, 3;\n"
        "  st.global.u32 [%rd0 + 8], %r2;\n"
        "  mov.b32 %r2, %r4;\n"
        "  st.global.u32 [%rd0 + 12], %r2;\n"
        "  ret;\n");
    expect_verifies(fx.fn);
}

TEST_CASE("PTX cleanup - copy propagation leaves guarded edge copies and multi-def sources alone") {
    Fixture fx;
    Reg a = fx.fn.new_b32(), b = fx.fn.new_b32(), c = fx.fn.new_b32();
    Reg p = fx.fn.new_pred();
    Block* e = fx.entry;
    e->append(Inst::make(Opcode::mov, Type::b32).dst(a).src(Operand::imm(1)));
    e->append(Inst::make(Opcode::mov, Type::b32).dst(a).src(Operand::imm(2)));   // a: two defs
    e->append(Inst::make(Opcode::mov, Type::b32).dst(b).src(a));                 // b = a: a is multi-def, keep
    e->append(Inst::make(Opcode::setp, Type::s32).cmp(CmpOp::eq).dst(p).src(b).src(Operand::imm(2)));
    e->append(Inst::make(Opcode::mov, Type::b32).guard(p).dst(c).src(b));        // guarded: keep
    e->append(st_u32(fx.rd0, 0, c));
    e->append(Inst::make(Opcode::ret));
    std::string before = body_text(fx.fn);
    CHECK_EQ(ptx::propagate_copies(fx.fn), size_t(0));
    CHECK_EQ(body_text(fx.fn), before);
}

TEST_CASE("PTX cleanup - dead instruction elimination removes chains and keeps side effects") {
    Fixture fx;
    Reg r0 = fx.fn.new_b32(), r1 = fx.fn.new_b32(), r2 = fx.fn.new_b32(), r3 = fx.fn.new_b32(), r4 = fx.fn.new_b32();
    Reg f0 = fx.fn.new_f32(), f1 = fx.fn.new_f32(), f2 = fx.fn.new_f32();
    Reg clk = fx.fn.new_b32();
    Reg p = fx.fn.new_pred();
    fx.fn.add_shared(Type::f32, "smem", 32, 16);
    Block* b = fx.entry;
    b->append(Inst::make(Opcode::mov, Type::b32).dst(r0).src(Operand::imm(2)));           // dead after r1 dies
    b->append(Inst::make(Opcode::shl, Type::b32).dst(r1).src(r0).src(Operand::imm(1)));   // dead
    b->append(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).dst(f0).src(Operand::addr(fx.rd0))); // dead load
    b->append(Inst::make(Opcode::mov, Type::u32).dst(clk).src(Operand::special(SpecialReg::clock)));           // volatile: keep
    b->append(Inst::make(Opcode::mov, Type::u32).dst(r2).src(Operand::special(SpecialReg::tid_x)));           // invariant, dead: remove
    b->append(Inst::make(Opcode::setp, Type::s32).cmp(CmpOp::eq).dst(p).src(r3).src(Operand::imm(0)));         // dead pred
    b->append(Inst::make(Opcode::mov, Type::u32).dst(r3).src(Operand::special(SpecialReg::ntid_x)));          // used
    b->append(Inst::make(Opcode::shfl, Type::b32).sync().shfl(ptx::ShflMode::down).dst(f1).src(f2)
                  .src(Operand::imm(1)).src(Operand::imm(31)).src(Operand::imm(0xffffffffLL)));                // collective: keep
    b->append(Inst::make(Opcode::atom, Type::u32).space(StateSpace::global).atom(ptx::AtomOp::add)
                  .dst(r4).src(Operand::addr(fx.rd0)).src(Operand::imm(1)));                                   // atom: keep
    b->append(Inst::make(Opcode::bar).sync().src(Operand::imm(0)));
    b->append(st_u32(fx.rd0, 4, r3));
    b->append(Inst::make(Opcode::ret));

    CHECK_EQ(ptx::eliminate_dead_instructions(fx.fn), size_t(5));
    expect_body(fx.fn,
        "$L_entry:\n"
        "  ld.param.u64 %rd0, [param_0];\n"
        "  mov.u32 %r5, %clock;\n"
        "  mov.u32 %r3, %ntid.x;\n"
        "  shfl.sync.down.b32 %f1, %f2, 1, 31, 4294967295;\n"
        "  atom.global.add.u32 %r4, [%rd0], 1;\n"
        "  bar.sync 0;\n"
        "  st.global.u32 [%rd0 + 4], %r3;\n"
        "  ret;\n");
    CHECK_EQ(ptx::eliminate_dead_instructions(fx.fn), size_t(0));

    // Renumbering after the deletions: the declarations shrink to what is used.
    ptx::renumber_registers(fx.fn);
    CHECK_EQ(fx.fn.reg_count(RegClass::B32), 3u);
    CHECK_EQ(fx.fn.reg_count(RegClass::F32), 2u);
    CHECK_EQ(fx.fn.reg_count(RegClass::B64), 1u);
    CHECK_EQ(fx.fn.reg_count(RegClass::Pred), 0u);
    expect_body(fx.fn,
        "$L_entry:\n"
        "  ld.param.u64 %rd0, [param_0];\n"
        "  mov.u32 %r2, %clock;\n"
        "  mov.u32 %r0, %ntid.x;\n"
        "  shfl.sync.down.b32 %f0, %f1, 1, 31, 4294967295;\n"
        "  atom.global.add.u32 %r1, [%rd0], 1;\n"
        "  bar.sync 0;\n"
        "  st.global.u32 [%rd0 + 4], %r0;\n"
        "  ret;\n");
    expect_verifies(fx.fn);
}

TEST_CASE("PTX cleanup - branch simplification: fall-through, inversion, unreachable blocks") {
    Fixture fx;
    Reg r = fx.fn.new_b32();
    Reg p = fx.fn.new_pred(), q = fx.fn.new_pred();
    Block* e = fx.entry;
    Block* a = fx.fn.add_block("$L_a");
    Block* dead = fx.fn.add_block("$L_dead");
    Block* b = fx.fn.add_block("$L_b");
    Block* c = fx.fn.add_block("$L_c");
    Block* d = fx.fn.add_block("$L_d");
    Block* x = fx.fn.add_block("$L_x");
    e->append(Inst::make(Opcode::ld, Type::u32).space(StateSpace::global).dst(r).src(Operand::addr(fx.rd0)));
    e->append(Inst::make(Opcode::bra).src(Operand::label(a->label)));            // bra next: removed
    a->append(Inst::make(Opcode::setp, Type::s32).cmp(CmpOp::eq).dst(p).src(r).src(Operand::imm(0)));
    a->append(Inst::make(Opcode::bra).guard(p).src(Operand::label(c->label)));
    a->append(Inst::make(Opcode::bra).src(Operand::label(b->label)));            // b becomes next once dead is gone: removed
    dead->append(Inst::make(Opcode::mov, Type::b32).dst(r).src(Operand::imm(7)));  // unreachable
    dead->append(Inst::make(Opcode::bra).src(Operand::label(x->label)));
    b->append(st_u32(fx.rd0, 4, r));
    b->append(Inst::make(Opcode::setp, Type::s32).cmp(CmpOp::lt).dst(q).src(r).src(Operand::imm(5)));
    b->append(Inst::make(Opcode::bra).guard(q).src(Operand::label(c->label)));   // @q bra next; bra d -> @!q bra d
    b->append(Inst::make(Opcode::bra).src(Operand::label(d->label)));
    c->append(st_u32(fx.rd0, 8, r));
    c->append(Inst::make(Opcode::bra).guard(p).src(Operand::label(d->label)));   // guarded bra to next: removed
    d->append(st_u32(fx.rd0, 12, r));
    d->append(Inst::make(Opcode::ret));
    x->append(Inst::make(Opcode::ret));                                          // only reachable from dead: removed

    CHECK_EQ(ptx::simplify_branches(fx.fn), size_t(6)); // 2 blocks + bra(e) + bra(a) + inversion(b) + bra(c)
    expect_body(fx.fn,
        "$L_entry:\n"
        "  ld.param.u64 %rd0, [param_0];\n"
        "  ld.global.u32 %r0, [%rd0];\n"
        "$L_a:\n"
        "  setp.eq.s32 %p0, %r0, 0;\n"
        "  @%p0 bra $L_c;\n"
        "$L_b:\n"
        "  st.global.u32 [%rd0 + 4], %r0;\n"
        "  setp.lt.s32 %p1, %r0, 5;\n"
        "  @!%p1 bra $L_d;\n"
        "$L_c:\n"
        "  st.global.u32 [%rd0 + 8], %r0;\n"
        "$L_d:\n"
        "  st.global.u32 [%rd0 + 12], %r0;\n"
        "  ret;\n");
    expect_verifies(fx.fn);
    CHECK_EQ(ptx::simplify_branches(fx.fn), size_t(0));

    // Fall-through block ends are valid PTX.
    std::string text = ptx::print(fx.fn);
    if (ptxas_available()) CHECK(ptxas_assembles(text));

    // The first block is never removed even when nothing references it.
    ptx::Function lone("lone");
    lone.add_block("$L_only")->append(Inst::make(Opcode::ret));
    CHECK_EQ(ptx::simplify_branches(lone), size_t(0));
    CHECK_EQ(lone.blocks.size(), size_t(1));
}

TEST_CASE("PTX cleanup - the pipeline is a no-op on clean input and keeps register order") {
    Fixture fx;
    Reg r0 = fx.fn.new_b32(), r1 = fx.fn.new_b32();
    Reg p = fx.fn.new_pred();
    Block* e = fx.entry;
    Block* t = fx.fn.add_block("$L_t");
    Block* j = fx.fn.add_block("$L_j");
    e->append(Inst::make(Opcode::mov, Type::u32).dst(r0).src(Operand::special(SpecialReg::tid_x)));
    e->append(Inst::make(Opcode::setp, Type::s32).cmp(CmpOp::eq).dst(p).src(r0).src(Operand::imm(0)));
    e->append(Inst::make(Opcode::bra).guard(p, true).src(Operand::label(j->label)));
    t->append(Inst::make(Opcode::add, Type::s32).dst(r1).src(r0).src(Operand::imm(1)));
    t->append(st_u32(fx.rd0, 0, r1));
    j->append(Inst::make(Opcode::ret));
    std::string before = ptx::print_body(fx.fn);
    ptx::CleanupStats stats = ptx::cleanup(fx.fn);
    CHECK_EQ(stats.copies_removed, size_t(0));
    CHECK_EQ(stats.dead_removed, size_t(0));
    CHECK_EQ(stats.branches_removed, size_t(0));
    CHECK_EQ(stats.insts_before, stats.insts_after);
    CHECK_EQ(ptx::print_body(fx.fn), before);
}

// ---------------------------------------------------------------------------
// (b) ISel: immediates and cached special registers
// ---------------------------------------------------------------------------

TEST_CASE("PTX cleanup - ISel folds constants into immediate positions and caches special registers") {
    Module mod("imm");
    using brass::Type;
    Function* f = mod.create_function("imm", Type::void_type(), {Type::ptr(), Type::i32(), Type::f32(), Type::i64()});
    codegen::KernelBuilder kb(mod, f);
    Builder& b = kb.builder();
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* out = b.add_block_param(e, Type::ptr());
    Value* i = b.add_block_param(e, Type::i32());
    Value* x = b.add_block_param(e, Type::f32());
    Value* l = b.add_block_param(e, Type::i64());
    Value* tid = kb.tid_x();
    Value* gid = kb.global_tid_x();                                  // re-reads tid/ctaid/ntid: cached
    Value* sum = b.build_add(i, kb.const_i32(5));                     // add.s32 %r, %r, 5
    Value* rsub = b.build_sub(kb.const_i32(100), i);                  // sub.s32 %r, 100, %r  (either source)
    Value* sh = b.build_shl(l, kb.const_i64(3));                      // shl.b64 %rd, %rd, 3  (64-bit count -> u32 imm)
    Value* lt = b.build_slt(i, kb.const_i32(7));                      // setp.lt.s32 %p, %r, 7
    Value* rlt = b.build_slt(kb.const_i32(7), i);                     // setp.lt.s32 %p, %rC, %r (constant lhs stays a register)
    Value* sel = b.build_select(lt, kb.const_i32(1), kb.const_i32(2)); // selp.b32 %r, 1, 2, %p
    Value* rsel = b.build_select(rlt, kb.const_i32(10), kb.const_i32(20));
    Value* fm = kb.fma(x, kb.const_f32(2.0f), kb.const_f32(0.5f));    // fma.rn.f32 %f, %f, 0f40000000, 0f3F000000
    Value* mx = kb.fmax(x, kb.const_f32(-1.0f));                      // max.f32 %f, %f, 0fBF800000
    Value* vb = kb.vbroadcast(Type::f32x4(), kb.const_f32(3.0f));     // mov.f32 lane, 0f40400000 x4
    Value* nib = b.build_and(b.build_lshr(i, kb.const_i32(4)), kb.const_i32(15));
    Value* wide = b.build_call("ptx_mul_wide_u32", Type::i64(), {i, kb.const_i32(144)}); // mul.wide.u32 %rd, %r, 144
    kb.store_i32(out, tid, 0);
    kb.store_i32(out, gid, 4);
    kb.store_i32(out, sum, 8);
    kb.store_i32(out, rsub, 12);
    b.build_store(Type::i64(), out, 16, sh);
    kb.store_i32(out, b.build_add(sel, rsel), 24);
    kb.store_f32(out, fm, 28);
    kb.store_f32(out, mx, 32);
    kb.vstore(Type::f32x4(), out, 48, vb);
    kb.store_i32(out, nib, 64);
    b.build_store(Type::i64(), out, 72, wide);
    b.build_store(Type::i32(), out, 80, kb.const_i32(42));            // st.global.u32 [..], 42
    b.build_ret_void();

    std::string ptx = emit_checked(*f);
    // Lines "    <prefix>...<suffix>" (register numbers are not part of the
    // contract), counted.
    auto lines_matching = [&](const char* prefix, const char* suffix) {
        size_t n = 0;
        std::string pre = std::string("    ") + prefix, suf = suffix;
        for (size_t pos = 0; pos < ptx.size();) {
            size_t eol = ptx.find('\n', pos);
            if (eol == std::string::npos) eol = ptx.size();
            std::string line = ptx.substr(pos, eol - pos);
            pos = eol + 1;
            if (line.size() >= pre.size() + suf.size() && line.compare(0, pre.size(), pre) == 0 &&
                line.compare(line.size() - suf.size(), suf.size(), suf) == 0) ++n;
        }
        return n;
    };
    struct Expect { const char* prefix; const char* suffix; size_t count; };
    for (const Expect& e : {Expect{"add.s32 %r", ", 5;", 1},
                            Expect{"sub.s32 %r", ";", 1},                       // sub.s32 %r, 100, %r (checked below)
                            Expect{"shl.b64 %rd", ", 3;", 1},                   // i64 count folded to a .u32 immediate
                            Expect{"setp.lt.s32 %p", ", 7;", 1},
                            Expect{"selp.b32 %r", ";", 2},                      // selp.b32 %r, 1, 2, %p / 10, 20
                            Expect{"fma.rn.f32 %f", ", 0f40000000, 0f3F000000;", 1},
                            Expect{"max.f32 %f", ", 0fBF800000;", 1},
                            Expect{"mov.f32 %f", ", 0f40400000;", 4},           // vbroadcast lanes
                            Expect{"shr.u32 %r", ", 4;", 1},
                            Expect{"and.b32 %r", ", 15;", 1},
                            Expect{"mul.wide.u32 %rd", ", 144;", 1},
                            Expect{"st.global.u32 [%rd", " + 80], 42;", 1},
                            Expect{"mad.lo.s32 %r", ";", 1},
                            // The constant 7 on the left of a comparison keeps its
                            // register; it is the only constant materialized.
                            Expect{"mov.b32 %r", ", 7;", 1},
                            Expect{"mov.b32 %r", ";", 1}}) {
        size_t n = lines_matching(e.prefix, e.suffix);
        if (n != e.count) std::cerr << "expected " << e.count << " of '" << e.prefix << "..." << e.suffix << "'\n" << ptx;
        CHECK_EQ(n, e.count);
    }
    // Immediates in the first source position of sub and selp.
    CHECK(ptx.find(", 100, %r") != std::string::npos);
    CHECK(ptx.find(", 1, 2, %p") != std::string::npos);
    CHECK(ptx.find(", 10, 20, %p") != std::string::npos);
    // Special registers: read once each, in the prologue, before the entry block.
    size_t params = ptx.find("$L_params:"), bb0 = ptx.find("$L_bb_");
    REQUIRE(params != std::string::npos && bb0 != std::string::npos);
    for (const char* sreg : {"%tid.x", "%ctaid.x", "%ntid.x"}) {
        size_t first = ptx.find(sreg);
        CHECK(first > params && first < bb0);
        CHECK_EQ(ptx.find(sreg, first + 1), std::string::npos);
    }
    if (!gpu_ready()) return;

    CudaBuffer dout = CudaBuffer::alloc(96);
    REQUIRE(dout.valid() && dout.zero());
    void* po = dout.device_ptr();
    int32_t iv = 9; float xv = 1.5f; int64_t lv = 5;
    launch(ptx, "imm", 1, 1, {&po, &iv, &xv, &lv});
    auto got = download<uint32_t>(dout, 24);
    CHECK_EQ(got[0], 0u); CHECK_EQ(got[1], 0u);
    CHECK_EQ(got[2], 14u); CHECK_EQ(got[3], 91u);
    CHECK_EQ(got[4], 40u); CHECK_EQ(got[5], 0u);
    CHECK_EQ(got[6], 2u + 10u);                      // (9 < 7 ? 1 : 2) + (7 < 9 ? 10 : 20)
    float fmv, mxv; std::memcpy(&fmv, &got[7], 4); std::memcpy(&mxv, &got[8], 4);
    CHECK_EQ(fmv, 3.5f); CHECK_EQ(mxv, 1.5f);
    for (int k = 12; k < 16; ++k) { float lane; std::memcpy(&lane, &got[k], 4); CHECK_EQ(lane, 3.0f); }
    CHECK_EQ(got[16], 0u);
    CHECK_EQ(got[18], 9u * 144u); CHECK_EQ(got[19], 0u);
    CHECK_EQ(got[20], 42u);
}

TEST_CASE("PTX cleanup - volatile special registers are read at every use") {
    Module mod("clk");
    using brass::Type;
    Function* f = mod.create_function("clk", Type::void_type(), {Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* out = b.add_block_param(e, Type::ptr());
    Value* c0 = b.build_call("ptx_clock", Type::i32());
    Value* c1 = b.build_call("ptx_clock", Type::i32());
    Value* w0 = b.build_call("ptx_warpid", Type::i32());
    Value* w1 = b.build_call("ptx_warpid", Type::i32());
    b.build_store(Type::i32(), out, 0, b.build_sub(c1, c0));
    b.build_store(Type::i32(), out, 4, b.build_sub(w1, w0));
    b.build_ret_void();
    std::string ptx = emit_checked(*f);
    auto count = [&](const char* needle) {
        size_t n = 0;
        for (size_t p = ptx.find(needle); p != std::string::npos; p = ptx.find(needle, p + 1)) ++n;
        return n;
    };
    CHECK_EQ(count("%clock;"), size_t(2));
    CHECK_EQ(count("%warpid;"), size_t(2));
    // Nothing invariant was read: no special register lands in the prologue.
    size_t params = ptx.find("$L_params:"), bb0 = ptx.find("$L_bb_");
    REQUIRE(params != std::string::npos && bb0 != std::string::npos);
    size_t first_special = std::min(ptx.find("%clock"), ptx.find("%warpid"));
    CHECK(first_special > bb0);
}

// ---------------------------------------------------------------------------
// (c) The whole pipeline over the intrinsic table and the fused kernels
// ---------------------------------------------------------------------------

namespace {

struct PipelineResult {
    size_t before = 0;
    size_t after = 0;
    std::string ptx;
};

// ISel -> count -> cleanup -> verify -> ptxas; then a second cleanup must
// change nothing.
PipelineResult run_pipeline(const Function& fn, const char* what) {
    ptx::PtxISel isel;
    ptx::Function raw = isel.lower(fn);
    auto raw_diags = ptx::verify(raw);
    if (!raw_diags.empty()) std::cerr << what << " (raw):\n" << ptx::format_diagnostics(raw_diags);
    CHECK(raw_diags.empty());
    PipelineResult r;
    r.before = ptx::instruction_count(raw);

    ptx::Function cleaned = isel.lower(fn);
    ptx::CleanupStats stats = ptx::cleanup(cleaned);
    r.after = ptx::instruction_count(cleaned);
    CHECK_EQ(stats.insts_before, r.before);
    CHECK_EQ(stats.insts_after, r.after);
    auto diags = ptx::verify(cleaned);
    if (!diags.empty()) std::cerr << what << ":\n" << ptx::format_diagnostics(diags) << ptx::print_body(cleaned);
    CHECK(diags.empty());
    r.ptx = ptx::print(cleaned);

    std::string once = ptx::print_body(cleaned);
    ptx::CleanupStats again = ptx::cleanup(cleaned);
    CHECK_EQ(again.copies_removed + again.dead_removed + again.branches_removed, size_t(0));
    CHECK_EQ(ptx::print_body(cleaned), once);

    // PtxTarget's default path is exactly this pipeline.
    CHECK_EQ(target::PtxTarget::emit_function(fn), r.ptx);
    return r;
}

} // namespace

TEST_CASE("PTX cleanup - every intrinsic verifies, assembles and gets shorter after cleanup") {
    bool have_ptxas = ptxas_available();
    size_t total_before = 0, total_after = 0, n = 0;
    for (std::string_view name : ptx::PtxISel::intrinsic_names()) {
        Module mod("intrin");
        Function* f = build_intrinsic_kernel(mod, std::string(name));
        PipelineResult r = run_pipeline(*f, std::string(name).c_str());
        if (r.after >= r.before) std::cerr << "intrinsic " << name << ": " << r.before << " -> " << r.after << "\n" << r.ptx;
        CHECK(r.after < r.before);
        if (have_ptxas) {
            bool ok = ptxas_assembles(r.ptx);
            if (!ok) std::cerr << "ptxas rejected " << name << ":\n" << r.ptx;
            CHECK(ok);
        }
        total_before += r.before;
        total_after += r.after;
        ++n;
    }
    std::printf("    %zu intrinsic kernels: %zu -> %zu instructions after cleanup\n", n, total_before, total_after);
}

TEST_CASE("PTX cleanup - the ten fused kernels verify, assemble and get shorter after cleanup") {
    using Build = Function* (*)(codegen::MlFusionCompiler&, Module&);
    struct K { const char* name; Build build; };
    const K kernels[] = {
        {"swiglu",             [](codegen::MlFusionCompiler& c, Module& m) { return c.build_ptx_swiglu(m); }},
        {"adaln_modulate",     [](codegen::MlFusionCompiler& c, Module& m) { return c.build_ptx_adaln_modulate(m, false); }},
        {"adaln_gated",        [](codegen::MlFusionCompiler& c, Module& m) { return c.build_ptx_adaln_modulate(m, true); }},
        {"residual_rms_norm",  [](codegen::MlFusionCompiler& c, Module& m) { return c.build_ptx_residual_rms_norm(m); }},
        {"layernorm_modulate", [](codegen::MlFusionCompiler& c, Module& m) { return c.build_ptx_layernorm_modulate(m); }},
        {"residual_layernorm", [](codegen::MlFusionCompiler& c, Module& m) { return c.build_ptx_residual_layernorm(m); }},
        {"gemv_swiglu",        [](codegen::MlFusionCompiler& c, Module& m) { return c.build_ptx_gemv_swiglu(m); }},
        {"gemv_residual",      [](codegen::MlFusionCompiler& c, Module& m) { return c.build_ptx_gemv_residual(m); }},
        {"gemv_q8_0",          [](codegen::MlFusionCompiler& c, Module& m) { return c.build_ptx_gemv_q8_0(m); }},
        {"gemv_q4_k",          [](codegen::MlFusionCompiler& c, Module& m) { return c.build_ptx_gemv_q4_k(m); }},
    };
    bool have_ptxas = ptxas_available();
    codegen::MlFusionCompiler c;
    std::printf("    %-20s %6s %6s\n", "kernel", "isel", "clean");
    for (const K& k : kernels) {
        Module mod(k.name);
        Function* f = k.build(c, mod);
        PipelineResult r = run_pipeline(*f, k.name);
        std::printf("    %-20s %6zu %6zu\n", k.name, r.before, r.after);
        CHECK(r.after < r.before);
        if (have_ptxas) {
            bool ok = ptxas_assembles(r.ptx, "sm_89");
            if (!ok) std::cerr << "ptxas rejected " << k.name << ":\n" << r.ptx;
            CHECK(ok);
        }
        // No `bra` to the next block and no block-argument copy chains remain.
        std::string prev_label;
        size_t pos = 0;
        while (pos < r.ptx.size()) {
            size_t eol = r.ptx.find('\n', pos);
            std::string line = r.ptx.substr(pos, eol - pos);
            pos = eol + 1;
            if (!line.empty() && line.back() == ':') {
                if (!prev_label.empty()) CHECK(prev_label != line.substr(0, line.size() - 1));
                prev_label.clear();
            } else if (line.rfind("    bra ", 0) == 0) {
                prev_label = line.substr(8, line.size() - 9);
            } else {
                prev_label.clear();
            }
        }
    }
}
