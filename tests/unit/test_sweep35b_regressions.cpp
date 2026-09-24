// Sweep 35b regressions.
//
// 1. A speculative-inliner guard resumed nowhere: its resume id matched no
//    guard or resume point, so a failing guard threw (Tier 0) or had no
//    state to resume with. The identity check is now a br_if to a slow
//    block of the same function that makes the original indirect call: the
//    CFG shows the slow path to every later pass (a resume_table block,
//    unreachable from the entry, let GVN make the code after the call read
//    a value only the fast path computes), and no deoptimization is needed.
//    Both interpreters' frameless resume from a guard also rebuilds the
//    guard's state-map values, not only the resume block's parameters.
// 2. Tier 2 refused clz / ctz and the overflow checks on i8 / i16 operands
//    (x64 and AArch64 isel). They are now lowered at the narrow width.
// 3. The AArch64 guard exit probed a large deopt record by moving SP one
//    page at a time, so SP sat at heights no unwind info describes. It now
//    probes below SP through X16 and moves SP once.
// 4. Tier 2 refused code with a speculative-inliner guard (no Tier-0 guard
//    matched it); with a branch there is nothing to refuse.
// 5. func_addr in tier-2 code yielded the engine's own copy of a function,
//    so a pointer made in Tier 0 or Tier 1 (the function's module stub)
//    never passed a tier-2 speculative identity check. Tier 2's func_addr
//    now yields the same stub.
#include "test_framework.hpp"
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/pass_catalog.hpp>
#include <brass/mir/pass_manager.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/speculative_inliner.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/runtime/type_feedback.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using namespace brass;
using namespace brass::codegen;
using namespace brass::runtime;

namespace {

// caller(fp, x) = fp(x) + 7x, the call site speculated on @expected_fn.
struct SpecCase {
    Module mod{"s35b_spec"};
    Function* expected_fn = nullptr;
    Function* unexpected_fn = nullptr;
    Function* caller = nullptr;
    Value* fp = nullptr;
    Value* x = nullptr;
    Value* k = nullptr;

    SpecCase() {
        expected_fn = mod.create_function("expected_fn", Type::i64(), {Type::i64()});
        {
            Builder b(mod);
            b.set_function(expected_fn);
            BasicBlock* entry = b.append_block("entry");
            Value* a = b.add_block_param(entry, Type::i64());
            b.build_ret(b.build_add(a, b.build_iconst_i64(10)));
            expected_fn->rebuild_cfg_predecessors();
        }
        unexpected_fn = mod.create_function("unexpected_fn", Type::i64(), {Type::i64()});
        {
            Builder b(mod);
            b.set_function(unexpected_fn);
            BasicBlock* entry = b.append_block("entry");
            Value* a = b.add_block_param(entry, Type::i64());
            b.build_ret(b.build_mul(a, b.build_iconst_i64(100)));
            unexpected_fn->rebuild_cfg_predecessors();
        }
        caller = mod.create_function("caller", Type::i64(), {Type::ptr(), Type::i64()});
        {
            Builder b(mod);
            b.set_function(caller);
            BasicBlock* entry = b.append_block("entry");
            fp = b.add_block_param(entry, Type::ptr());
            x = b.add_block_param(entry, Type::i64());
            // Live across the call: only the code after it reads %k.
            k = b.build_mul(x, b.build_iconst_i64(7));
            Value* r = b.build_call_indirect(fp, Type::i64(), {x});
            r->defining_instruction()->set_site_id(1);
            b.build_ret(b.build_add(r, k));
            caller->rebuild_cfg_predecessors();
        }
        REQUIRE(verify_module(mod));

        TypeFeedbackVector tfv("caller");
        tfv.record_call_target(1, 0, "expected_fn");
        SpeculativeInlinerOptions opts;
        opts.enable_inlining = true;
        REQUIRE(run_speculative_devirtualization(*caller, mod, &tfv, opts));
        DiagnosticReporter diag;
        const bool ok = verify_function(*caller, &diag);
        if (!ok) std::cerr << diag.format_all() << "\n";
        REQUIRE(ok);
    }

    // The only instruction of `op` in the caller.
    const Instruction* only(Opcode op) const {
        const Instruction* found = nullptr;
        for (const BasicBlock* bb : caller->blocks()) {
            for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
                if (inst && inst->opcode() == op) {
                    REQUIRE(found == nullptr);
                    found = inst;
                }
            }
        }
        REQUIRE(found != nullptr);
        return found;
    }
};

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

// Each operand is a wrapped narrow add, so bits above the width are garbage
// in a register that does not mask them.
const char* kNarrow = R"(module @s35b_narrow
func @clz8(%x: i64, %y: i64) -> i64 {
b0:
  %xt = trunc_i32 %x
  %a = trunc.i8 %xt
  %yt = trunc_i32 %y
  %b = trunc.i8 %yt
  %s = add.i8 %a, %b
  %c = clz.i8 %s
  %z = zext_i64 %c
  ret %z
}
func @ctz16(%x: i64, %y: i64) -> i64 {
b0:
  %p = alloca 32, 8
  %xt = trunc_i32 %x
  store.i32 %p, 0, %xt
  %a = load.i16 %p, 0
  %yt = trunc_i32 %y
  store.i32 %p, 8, %yt
  %b = load.i16 %p, 8
  %s = add.i16 %a, %b
  %c = ctz.i16 %s
  %zero = iconst.i64 0
  store.i64 %p, 16, %zero
  store.i16 %p, 16, %c
  %z = load.i64 %p, 16
  ret %z
}
func @smulo8(%x: i64, %y: i64) -> i64 {
b0:
  %xt = trunc_i32 %x
  %a = trunc.i8 %xt
  %yt = trunc_i32 %y
  %b = trunc.i8 %yt
  %s = add.i8 %a, %b
  %o = smul_overflow.i8 %s, %b
  %z = zext_i64 %o
  ret %z
}
func @usubo16(%x: i64, %y: i64) -> i64 {
b0:
  %p = alloca 16, 8
  %xt = trunc_i32 %x
  store.i32 %p, 0, %xt
  %a = load.i16 %p, 0
  %yt = trunc_i32 %y
  store.i32 %p, 8, %yt
  %b = load.i16 %p, 8
  %s = add.i16 %a, %b
  %o = usub_overflow.i16 %s, %b
  %z = zext_i64 %o
  ret %z
}
func @uaddo8(%x: i64, %y: i64) -> i64 {
b0:
  %xt = trunc_i32 %x
  %a = trunc.i8 %xt
  %yt = trunc_i32 %y
  %b = trunc.i8 %yt
  %o = uadd_overflow.i8 %a, %b
  %z = zext_i64 %o
  ret %z
}
)";

// References on the N-bit value of the wrapped sums above.
uint64_t zm(unsigned bits, int64_t v) { return static_cast<uint64_t>(v) & ((uint64_t{1} << bits) - 1); }
int64_t sm(unsigned bits, int64_t v) {
    const int64_t z = static_cast<int64_t>(zm(bits, v));
    const int64_t s = int64_t{1} << (bits - 1);
    return (z ^ s) - s;
}
int64_t ref(const std::string& f, int64_t x, int64_t y) {
    if (f == "clz8") {
        const uint64_t a = zm(8, x + y);
        int n = 0;
        for (int i = 7; i >= 0 && !((a >> i) & 1); --i) ++n;
        return n;
    }
    if (f == "ctz16") {
        const uint64_t a = zm(16, zm(16, x) + zm(16, y));
        int n = 0;
        while (n < 16 && !((a >> n) & 1)) ++n;
        return n;
    }
    if (f == "smulo8") {
        const int64_t p = sm(8, x + y) * sm(8, y);
        return p < -128 || p > 127;
    }
    if (f == "usubo16") {
        return zm(16, zm(16, x) + zm(16, y)) < zm(16, y);
    }
    if (f == "uaddo8") return zm(8, x) + zm(8, y) > 255;
    return -1;
}

uint32_t word_at(const std::vector<uint8_t>& code, size_t i) {
    return static_cast<uint32_t>(code[4 * i]) | (static_cast<uint32_t>(code[4 * i + 1]) << 8) |
           (static_cast<uint32_t>(code[4 * i + 2]) << 16) | (static_cast<uint32_t>(code[4 * i + 3]) << 24);
}

// ADD/SUB (immediate or extended register), no flags, destination SP.
bool writes_sp(uint32_t w) {
    const bool add_sub_imm = (w & 0x1F000000u) == 0x11000000u && ((w >> 29) & 1u) == 0;
    const bool add_sub_ext = (w & 0x1FE00000u) == 0x0B200000u && ((w >> 29) & 1u) == 0;
    return (add_sub_imm || add_sub_ext) && (w & 31u) == 31u;
}

size_t count_op(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst && inst->opcode() == op) ++n;
        }
    }
    return n;
}

// The site id of fn's only call_indirect.
uint32_t call_site(const Function& fn) {
    uint32_t site = UINT32_MAX;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst && inst->opcode() == Opcode::call_indirect) {
                REQUIRE(site == UINT32_MAX);
                site = inst->site_id();
            }
        }
    }
    REQUIRE(site != UINT32_MAX);
    return site;
}

// A guard whose resume block's successor reads %k, a state value that is
// not one of the block's parameters.
const char* kGuardResume = R"(module @s35b_resume
func @f(%x: i64) -> i64 {
b0:
  %seven = iconst.i64 7
  %k = mul %x, %seven
  %lim = iconst.i64 100
  %ok = slt %x, %lim
  guard %ok, @s35b_slow_path, [%x, %k], id 0
  br done(%x)
slow(%sx: i64):
  br done(%sx)
done(%v: i64):
  %s = add %v, %k
  ret %s
resume_table {
  entry 0 -> slow
}
}
)";

// caller(fp, x) = fp(x) + (x + 10); expected_fn computes the same x + 10.
const char* kSpecGvn = R"(module @s35b_gvn
func @expected_fn(%a: i64) -> i64 {
b0:
  %c = iconst.i64 10
  %r = add %a, %c
  ret %r
}
func @unexpected_fn(%a: i64) -> i64 {
b0:
  %c = iconst.i64 100
  %r = mul %a, %c
  ret %r
}
func @caller(%fp: ptr, %x: i64) -> i64 {
b0:
  %r = call_indirect.i64 %fp(%x)
  %ten = iconst.i64 10
  %t = add %x, %ten
  %s = add %r, %t
  ret %s
}
)";

} // namespace

TEST_CASE("Sweep35b - a speculative identity check branches to a slow path that makes the original call") {
    SpecCase c;
    // No guard and no resume point: nothing to deoptimize to.
    CHECK_EQ(count_op(*c.caller, Opcode::guard), 0u);
    CHECK(c.caller->resume_points().empty());
    // br_if (fp == func_addr @expected_fn): the slow target makes the
    // original call on the original callee and argument.
    const Instruction* eq = c.only(Opcode::eq);
    CHECK(eq->operand(0) == c.fp);
    REQUIRE(eq->operand(1)->defining_instruction() != nullptr);
    CHECK(eq->operand(1)->defining_instruction()->opcode() == Opcode::func_addr);
    CHECK_EQ(std::string(eq->operand(1)->defining_instruction()->symbol()), std::string("expected_fn"));
    const Instruction* br = c.only(Opcode::br_if);
    CHECK(br->operand(0) == eq->result());
    const Instruction* slow_call = c.only(Opcode::call_indirect);
    CHECK(slow_call->parent() == br->false_target().block);
    REQUIRE(slow_call->operand_count() == 2);
    CHECK(slow_call->operand(0) == c.fp);
    CHECK(slow_call->operand(1) == c.x);
    // Every block is reachable from the entry: the slow path is ordinary CFG.
    c.caller->rebuild_cfg_predecessors();
    for (const BasicBlock* bb : c.caller->blocks()) {
        if (bb != c.caller->entry_block()) CHECK(!bb->predecessors().empty());
    }
}

TEST_CASE("Sweep35b - a failing speculative identity check finishes the call in both interpreters") {
    SpecCase c;
    Interpreter interp;
    const uintptr_t exp_ptr = interp.function_address(*c.expected_fn);
    const uintptr_t unexp_ptr = interp.function_address(*c.unexpected_fn);

    RuntimeValue fast = interp.run(c.mod, "caller", {RuntimeValue::from_ptr(exp_ptr), RuntimeValue::from_i64(5)});
    CHECK_EQ(fast.as_i64(), 15 + 35);
    CHECK(!interp.last_deopt().deoptimized);

    // No deopt handler needed: the slow path makes the call (it used to throw).
    RuntimeValue slow = interp.run(c.mod, "caller", {RuntimeValue::from_ptr(unexp_ptr), RuntimeValue::from_i64(5)});
    CHECK(!interp.last_deopt().deoptimized);
    CHECK_EQ(slow.as_i64(), 500 + 35);

    FastInterpreter fi;
    const uintptr_t fexp = fi.function_address(*c.expected_fn);
    const uintptr_t funexp = fi.function_address(*c.unexpected_fn);
    CHECK_EQ(fi.run(c.mod, "caller", {RuntimeValue::from_ptr(fexp), RuntimeValue::from_i64(6)}).as_i64(), 16 + 42);
    CHECK_EQ(fi.run(c.mod, "caller", {RuntimeValue::from_ptr(funexp), RuntimeValue::from_i64(6)}).as_i64(),
             600 + 42);
    CHECK(!fi.last_deopt().deoptimized);
}

TEST_CASE("Sweep35b - a guard's frameless resume rebuilds its state-map values") {
    // What a native deopt does: a fresh frame built from the guard's state.
    // The code after the resume block reads %k, a state value that is not
    // one of the block's parameters.
    auto mod = parse_ok(kGuardResume);
    const Function* f = mod->get_function("f");
    REQUIRE(f != nullptr);
    const std::vector<RuntimeValue> state = {RuntimeValue::from_i64(500), RuntimeValue::from_i64(42)};
    Interpreter interp;
    interp.set_module(mod.get());
    CHECK_EQ(interp.resume_after_guard(*f, 0, state, nullptr).as_i64(), 500 + 42);
    // Sanity: run normally, the guard holds.
    CHECK_EQ(interp.run(*f, {RuntimeValue::from_i64(6)}).as_i64(), 6 + 42);

    // The FastInterpreter's resume rebuilds the same registers.
    FastInterpreter fi;
    CHECK_EQ(fi.resume_from_native(*f, 0, state).as_i64(), 500 + 42);
}

TEST_CASE("Sweep35b - GVN cannot make the code after a speculated call read a fast-path value") {
    // caller(fp, x) = fp(x) + (x + 10), speculated on expected_fn(a) = a + 10
    // and inlined. With the slow path hidden behind a guard's resume block
    // (unreachable from the entry), the inlined `x + 10` dominated the code
    // after the call, GVN replaced that code's `x + 10` with it, and a
    // failing guard resumed into code reading a value never computed.
    auto mod = parse_ok(kSpecGvn);
    Function* caller = mod->get_function("caller");
    TypeFeedbackVector tfv("caller");
    tfv.record_call_target(call_site(*caller), 0, "expected_fn");
    SpeculativeInlinerOptions opts;
    opts.enable_inlining = true;
    REQUIRE(run_speculative_devirtualization(*caller, *mod, &tfv, opts));
    REQUIRE_EQ(count_op(*caller, Opcode::call), 0u);  // inlined

    Pipeline p;
    p.add(passes::gvn(FunctionFilter::All));
    p.add(passes::gvn_pre(nullptr, FunctionFilter::All));
    p.add(passes::sccp(true, FunctionFilter::All));
    p.add(passes::cfg_simplify("cfg_simplify", FunctionFilter::All));
    run_pipeline(*mod, p);
    DiagnosticReporter diag;
    const bool ok = verify_module(*mod, &diag);
    if (!ok) std::cerr << diag.format_all() << "\n";
    REQUIRE(ok);
    CHECK_EQ(count_op(*caller, Opcode::call_indirect), 1u);

    Interpreter interp;
    const uintptr_t exp_ptr = interp.function_address(*mod->get_function("expected_fn"));
    const uintptr_t unexp_ptr = interp.function_address(*mod->get_function("unexpected_fn"));
    FastInterpreter fi;
    const uintptr_t fexp = fi.function_address(*mod->get_function("expected_fn"));
    const uintptr_t funexp = fi.function_address(*mod->get_function("unexpected_fn"));
    for (int64_t x : {int64_t{0}, int64_t{3}, int64_t{-7}}) {
        CHECK_EQ(interp.run(*mod, "caller", {RuntimeValue::from_ptr(exp_ptr), RuntimeValue::from_i64(x)}).as_i64(),
                 (x + 10) + (x + 10));
        CHECK_EQ(interp.run(*mod, "caller", {RuntimeValue::from_ptr(unexp_ptr), RuntimeValue::from_i64(x)}).as_i64(),
                 x * 100 + (x + 10));
        CHECK(!interp.last_deopt().deoptimized);
        CHECK_EQ(fi.run(*mod, "caller", {RuntimeValue::from_ptr(fexp), RuntimeValue::from_i64(x)}).as_i64(),
                 (x + 10) + (x + 10));
        CHECK_EQ(fi.run(*mod, "caller", {RuntimeValue::from_ptr(funexp), RuntimeValue::from_i64(x)}).as_i64(),
                 x * 100 + (x + 10));
    }
}

TEST_CASE("Sweep35b - tier 2 computes narrow clz, ctz and overflow checks at their width") {
    auto mod = parse_ok(kNarrow);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    using Fn2 = int64_t (*)(int64_t, int64_t);
    const int64_t inputs[] = {0, 1, -1, 5, 16, 100, 127, -128, 200, 0x7f, 0x80, 0xff00, 0x8000, 0x1234, -0x4000};
    for (const char* name : {"clz8", "ctz16", "smulo8", "usubo16", "uaddo8"}) {
        auto f = reinterpret_cast<Fn2>(jit.get_symbol_address(name));
        REQUIRE(f != nullptr);
        Interpreter interp;
        interp.set_module(mod.get());
        int bad = 0;
        for (int64_t x : inputs) {
            for (int64_t y : inputs) {
                const int64_t want = ref(name, x, y);
                const int64_t got = f(x, y);
                const int64_t t0 = interp.run(*mod->get_function(name),
                                              {RuntimeValue::from_i64(x), RuntimeValue::from_i64(y)}).as_i64();
                if (got != want || t0 != want) {
                    if (bad++ < 3) {
                        std::cerr << name << "(" << x << ", " << y << "): tier 2 " << got << ", tier 0 " << t0
                                  << ", want " << want << "\n";
                    }
                }
            }
        }
        CHECK_EQ(bad, 0);
    }
}

TEST_CASE("Sweep35b - AArch64 isel lowers narrow clz, ctz and overflow checks") {
    auto mod = parse_ok(kNarrow);
    for (const char* name : {"clz8", "ctz16", "smulo8", "usubo16", "uaddo8"}) {
        aarch64::AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
        std::unique_ptr<LirFunction> lir = isel.lower(*mod->get_function(name));
        REQUIRE(lir != nullptr);
        bool count_op = false;
        for (const auto& bb : lir->blocks) {
            for (const auto& inst : bb->instructions) {
                if (!inst) continue;
                if (inst->opcode == LirOpcode::Lzcnt32 || inst->opcode == LirOpcode::Tzcnt32) count_op = true;
                // Nothing counts at 64 bits.
                CHECK(inst->opcode != LirOpcode::Lzcnt);
                CHECK(inst->opcode != LirOpcode::Tzcnt);
            }
        }
        const std::string n = name;
        CHECK_EQ(count_op, n == "clz8" || n == "ctz16");
    }
}

TEST_CASE("Sweep35b - the AArch64 guard exit probes a large record without moving SP, then moves it once") {
    // 1100 state values: a 9936-byte record, three pages.
    LirFunction fn;
    fn.name = "big_exit";
    fn.frame.total_frame_size = 16;
    auto bb = std::make_unique<LirBlock>(0, "entry");
    auto exit = std::make_unique<LirInst>(LirOpcode::GuardExit);
    for (int i = 0; i < 1100; ++i) {
        exit->add_use(LirOperand::imm(i, 8));
        exit->deopt_kinds.push_back(static_cast<uint8_t>(DeoptValueKind::Int64));
    }
    exit->resume_id = 3;
    exit->deopt_reason = 1;
    bb->append_inst(std::move(exit));
    bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
    fn.blocks.push_back(std::move(bb));

    aarch64::AArch64EmitContext emitter(fn, Target::aarch64_linux());
    aarch64::AArch64CompilationResult res = emitter.compile();
    const size_t n = res.code_buffer.size() / 4;
    const std::vector<uint8_t> code(res.code_buffer.data(), res.code_buffer.data() + res.code_buffer.size());

    constexpr uint32_t kStrXzrX16 = 0xF900021Fu;  // str xzr, [x16]
    constexpr uint32_t kStrXzrSp = 0xF90003FFu;   // str xzr, [sp]
    constexpr uint32_t kSubX16Sp1Page = 0xD14007F0u;  // sub x16, sp, #1, lsl #12
    constexpr uint32_t kSubX16Sp2Page = 0xD1400BF0u;  // sub x16, sp, #2, lsl #12
    std::vector<size_t> probes;
    size_t first_bl = n;
    for (size_t i = 0; i < n; ++i) {
        const uint32_t w = word_at(code, i);
        CHECK(w != kStrXzrSp);
        if (w == kStrXzrX16) probes.push_back(i);
        if ((w & 0xFC000000u) == 0x94000000u && first_bl == n) first_bl = i;
    }
    // One probe per page below the first, each through X16 at SP - page.
    REQUIRE_EQ(probes.size(), 2u);
    CHECK(probes[0] >= 1);
    CHECK_EQ(word_at(code, probes[0] - 1), kSubX16Sp1Page);
    CHECK_EQ(word_at(code, probes[1] - 1), kSubX16Sp2Page);
    REQUIRE(first_bl < n);
    REQUIRE(probes[1] < first_bl);
    // From the first probe to the call into the runtime, one instruction
    // writes SP, after every probe.
    size_t sp_writes = 0, sp_write_at = 0;
    for (size_t i = probes[0] - 1; i < first_bl; ++i) {
        if (writes_sp(word_at(code, i))) {
            ++sp_writes;
            sp_write_at = i;
        }
    }
    CHECK_EQ(sp_writes, 1u);
    CHECK(sp_write_at > probes[1]);
}

namespace {

const char* kSpecTier2 = R"(module @s35b_t2spec
func @expected_fn(%a: i64) -> i64 {
b0:
  %c = iconst.i64 10
  %r = add %a, %c
  ret %r
}
func @unexpected_fn(%a: i64) -> i64 {
b0:
  %c = iconst.i64 100
  %r = mul %a, %c
  ret %r
}
func @caller(%fp: ptr, %x: i64) -> i64 {
b0:
  %seven = iconst.i64 7
  %k = mul %x, %seven
  %r = call_indirect.i64 %fp(%x)
  %s = add %r, %k
  ret %s
}
func @get_exp() -> ptr {
b0:
  %p = func_addr @expected_fn
  ret %p
}
)";

TieringConfig no_tierup() {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(false);
    return cfg;
}

} // namespace

TEST_CASE("Sweep35b - tier 2 compiles a speculated call with no guard exit") {
    SpecCase c;
    // Tier 2 compiles it with no guard exit, on either target.
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(c.mod));
    using CallerFn = int64_t (*)(void*, int64_t);
    auto native = reinterpret_cast<CallerFn>(jit.get_symbol_address("caller"));
    REQUIRE(native != nullptr);
    CHECK_EQ(native(jit.get_symbol_address("expected_fn"), 6), 16 + 42);
    CHECK_EQ(native(jit.get_symbol_address("unexpected_fn"), 6), 600 + 42);

    aarch64::AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
    std::unique_ptr<LirFunction> lir = isel.lower(*c.caller);
    REQUIRE(lir != nullptr);
    for (const auto& bb : lir->blocks) {
        for (const auto& inst : bb->instructions) {
            if (inst) CHECK(inst->opcode != LirOpcode::GuardExit);
        }
    }
    CHECK(lir->resume_entries.empty());
}

TEST_CASE("Sweep35b - tier 2 installs speculatively inlined code whose identity check passes and fails natively") {
    auto mod = parse_ok(kSpecTier2);
    FunctionDispatchTable prog;
    prog.tiering().type_feedback().get_or_create("caller").record_call_target(1, 0, "expected_fn");

    // The tier-2 pipeline's speculation fires on this feedback: an identity
    // check branching to a slow call, no guard.
    {
        auto copy = parse_ok(kSpecTier2);
        SpeculativeInlinerOptions opts;
        opts.enable_inlining = true;
        REQUIRE(run_speculative_devirtualization(*copy, prog.tiering().type_feedback(), opts));
        const Function* spec = copy->get_function("caller");
        REQUIRE_EQ(count_op(*spec, Opcode::func_addr), 1u);
        CHECK_EQ(count_op(*spec, Opcode::call_indirect), 1u);
        CHECK_EQ(count_op(*spec, Opcode::guard), 0u);
    }

    FunctionHandle* h = prog.get_or_create("caller", mod->get_function("caller"));
    CodeInstaller installer(prog);
    CodeInstallResult res = installer.install_tier2(*h, *mod, "caller");
    if (!res.success) std::cerr << res.error_message << "\n";
    REQUIRE(res.success);
    CHECK(!h->tier2_rejected());
    CHECK(h->tier() == TierLevel::Tier2_Optimized);
    REQUIRE(h->jit_engine() != nullptr);

    // The pointers the tier-2 code's own func_addr yields in a program whose
    // pipeline is not initialized (the engine's copies): @expected_fn's
    // passes the check, @unexpected_fn's fails it and makes the call.
    void* exp_native = h->jit_engine()->get_symbol_address("expected_fn");
    void* unexp_native = h->jit_engine()->get_symbol_address("unexpected_fn");
    REQUIRE(exp_native != nullptr);
    REQUIRE(unexp_native != nullptr);
    auto native = h->get_function_ptr<int64_t (*)(void*, int64_t)>();
    REQUIRE(native != nullptr);

    Interpreter interp;
    interp.set_module(mod.get());
    const uintptr_t exp_ptr = interp.function_address(*mod->get_function("expected_fn"));
    const uintptr_t unexp_ptr = interp.function_address(*mod->get_function("unexpected_fn"));
    const uint64_t deopts0 = prog.pipeline().tier2_deopts();
    for (int64_t x : {int64_t{0}, int64_t{5}, int64_t{-3}, int64_t{1} << 40}) {
        const int64_t want_pass =
            interp.run(*mod->get_function("caller"), {RuntimeValue::from_ptr(exp_ptr), RuntimeValue::from_i64(x)})
                .as_i64();
        const int64_t want_fail =
            interp.run(*mod->get_function("caller"), {RuntimeValue::from_ptr(unexp_ptr), RuntimeValue::from_i64(x)})
                .as_i64();
        CHECK_EQ(want_pass, x + 10 + 7 * x);
        CHECK_EQ(want_fail, x * 100 + 7 * x);
        CHECK_EQ(native(exp_native, x), want_pass);
        CHECK_EQ(native(unexp_native, x), want_fail);
    }
    // A failing check branched inside the tier-2 code: nothing deoptimized,
    // and the code is still installed.
    CHECK_EQ(prog.pipeline().tier2_deopts(), deopts0);
    CHECK(h->tier() == TierLevel::Tier2_Optimized);
}

TEST_CASE("Sweep35b - a Tier-0 function pointer takes the tier-2 speculative fast path") {
    auto mod = parse_ok(kSpecTier2);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup());
    Interpreter interp;
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    const Function* expected_fn = mod->get_function("expected_fn");
    const Function* unexpected_fn = mod->get_function("unexpected_fn");
    const Function* caller = mod->get_function("caller");

    // The pointers Tier 0 makes (the functions' module stubs).
    const uintptr_t exp_ptr = interp.function_address(*expected_fn);
    const uintptr_t unexp_ptr = interp.function_address(*unexpected_fn);
    // The callees run in Tier 1, which counts their invocations: a call
    // through a pointer reaches them, the inlined fast path does not.
    REQUIRE(prog.pipeline().compile_and_install_tier1("expected_fn", expected_fn));
    REQUIRE(prog.pipeline().compile_and_install_tier1("unexpected_fn", unexpected_fn));
    FunctionHandle* get = prog.get_or_create("get_exp", mod->get_function("get_exp"));

    prog.tiering().type_feedback().get_or_create("caller").record_call_target(call_site(*caller), 0, "expected_fn");
    FunctionHandle* h = prog.get_or_create("caller", caller);
    CodeInstaller installer(prog);
    CodeInstallResult res = installer.install_tier2(*h, *mod, "caller");
    if (!res.success) std::cerr << res.error_message << "\n";
    REQUIRE(res.success);
    REQUIRE(h->tier() == TierLevel::Tier2_Optimized);
    auto native = h->get_function_ptr<int64_t (*)(void*, int64_t)>();
    REQUIRE(native != nullptr);

    // func_addr in tier-2 code yields the pointer Tier 0 made.
    REQUIRE(get->native_entry() != nullptr);
    CHECK_EQ(reinterpret_cast<uintptr_t>(get->get_function_ptr<void* (*)()>()()), exp_ptr);

    TieringFeedback& exp_fb = prog.tiering().get_feedback("expected_fn");
    TieringFeedback& unexp_fb = prog.tiering().get_feedback("unexpected_fn");
    const uint64_t exp0 = exp_fb.invocation_count();
    const uint64_t unexp0 = unexp_fb.invocation_count();
    for (int64_t x : {int64_t{0}, int64_t{5}, int64_t{-3}, int64_t{1} << 40}) {
        CHECK_EQ(native(reinterpret_cast<void*>(exp_ptr), x), x + 10 + 7 * x);
        CHECK_EQ(native(reinterpret_cast<void*>(unexp_ptr), x), x * 100 + 7 * x);
    }
    // The Tier-0 pointer to @expected_fn passed the check every time (the
    // inlined body ran, the function was never called); the one to
    // @unexpected_fn failed it and was called through its stub.
    CHECK_EQ(exp_fb.invocation_count(), exp0);
    CHECK_EQ(unexp_fb.invocation_count(), unexp0 + 4);
}
