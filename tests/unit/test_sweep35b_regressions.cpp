// Sweep 35b regressions.
//
// 1. A speculative-inliner guard resumed nowhere: its resume id matched no
//    guard or resume point, so a failing guard threw (Tier 0) or had no
//    state to resume with. It now resumes at a resume_table block of its
//    own that makes the original indirect call, and its state holds the
//    callee, the arguments and every value live after the call.
// 2. Tier 2 refused clz / ctz and the overflow checks on i8 / i16 operands
//    (x64 and AArch64 isel). They are now lowered at the narrow width.
// 3. The AArch64 guard exit probed a large deopt record by moving SP one
//    page at a time, so SP sat at heights no unwind info describes. It now
//    probes below SP through X16 and moves SP once.
// 4. Tier 2 refused code with a speculative-inliner guard: no Tier-0 guard
//    matched it. Such a guard resumes in the optimized function itself, so
//    the installer now lowers it to a branch to its resume block.
#include "test_framework.hpp"
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/speculative_inliner.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
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

    const Instruction* guard() const {
        const Instruction* g = nullptr;
        for (const BasicBlock* bb : caller->blocks()) {
            for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
                if (inst && inst->opcode() == Opcode::guard) {
                    REQUIRE(g == nullptr);
                    g = inst;
                }
            }
        }
        REQUIRE(g != nullptr);
        return g;
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

} // namespace

TEST_CASE("Sweep35b - a speculative guard has its own id and a resume target that makes the original call") {
    SpecCase c;
    const Instruction* g = c.guard();
    // Its id names this guard and a block of the function.
    CHECK(c.caller->find_guard(g->resume_id()) == g);
    const BasicBlock* slow = c.caller->get_resume_target(g->resume_id());
    REQUIRE(slow != nullptr);
    CHECK(c.caller->guard_exit_stub(*g) == nullptr);
    // The block's parameters take the callee and the argument; the state
    // also carries %k, which the code after the call reads.
    REQUIRE(slow->param_count() == 2);
    REQUIRE(g->state_map().size() >= 3);
    CHECK(g->state_map()[0] == c.fp);
    CHECK(g->state_map()[1] == c.x);
    bool has_k = false;
    for (const Value* v : g->state_map()) has_k = has_k || v == c.k;
    CHECK(has_k);
}

TEST_CASE("Sweep35b - a failing speculative guard finishes the call in Tier 0") {
    SpecCase c;
    Interpreter interp;
    const uintptr_t exp_ptr = interp.function_address(*c.expected_fn);
    const uintptr_t unexp_ptr = interp.function_address(*c.unexpected_fn);

    RuntimeValue fast = interp.run(c.mod, "caller", {RuntimeValue::from_ptr(exp_ptr), RuntimeValue::from_i64(5)});
    CHECK_EQ(fast.as_i64(), 15 + 35);
    CHECK(!interp.last_deopt().deoptimized);

    // No deopt handler: the guard resumes at its block (it used to throw).
    RuntimeValue slow = interp.run(c.mod, "caller", {RuntimeValue::from_ptr(unexp_ptr), RuntimeValue::from_i64(5)});
    CHECK(interp.last_deopt().deoptimized);
    CHECK_EQ(slow.as_i64(), 500 + 35);
}

TEST_CASE("Sweep35b - a speculative guard's deopt state alone resumes the call") {
    // What a native deopt does: a fresh frame built from the guard's state.
    SpecCase c;
    const Instruction* g = c.guard();
    Interpreter interp;
    const uintptr_t unexp_ptr = interp.function_address(*c.unexpected_fn);
    std::unordered_map<const Value*, RuntimeValue> known = {
        {c.fp, RuntimeValue::from_ptr(unexp_ptr)},
        {c.x, RuntimeValue::from_i64(6)},
        {c.k, RuntimeValue::from_i64(42)},
    };
    std::vector<RuntimeValue> state;
    for (const Value* v : g->state_map()) {
        auto it = known.find(v);
        REQUIRE(it != known.end());
        state.push_back(it->second);
    }
    RuntimeValue r = interp.resume_after_guard(*c.caller, g->resume_id(), state, nullptr);
    CHECK_EQ(r.as_i64(), 600 + 42);

    // The FastInterpreter's resume rebuilds the same registers.
    FastInterpreter fi;
    state[0] = RuntimeValue::from_ptr(fi.function_address(*c.unexpected_fn));
    RuntimeValue rf = fi.resume_from_native(*c.caller, g->resume_id(), state);
    CHECK_EQ(rf.as_i64(), 600 + 42);
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
)";

size_t count_op(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst && inst->opcode() == op) ++n;
        }
    }
    return n;
}

} // namespace

TEST_CASE("Sweep35b - a speculative guard lowers to a branch to its resume block") {
    SpecCase c;
    const uint32_t id = c.guard()->resume_id();
    REQUIRE(lower_guards_to_local_branch(*c.caller, c.mod, id));
    DiagnosticReporter diag;
    const bool ok = verify_function(*c.caller, &diag);
    if (!ok) std::cerr << diag.format_all() << "\n";
    REQUIRE(ok);
    CHECK_EQ(count_op(*c.caller, Opcode::guard), 0u);
    CHECK(c.caller->get_resume_target(id) == nullptr);
    CHECK_EQ(count_op(*c.caller, Opcode::call_indirect), 1u);

    // The interpreter takes the branch as it took the guard.
    Interpreter interp;
    const uintptr_t exp_ptr = interp.function_address(*c.expected_fn);
    const uintptr_t unexp_ptr = interp.function_address(*c.unexpected_fn);
    CHECK_EQ(interp.run(c.mod, "caller", {RuntimeValue::from_ptr(exp_ptr), RuntimeValue::from_i64(5)}).as_i64(),
             15 + 35);
    CHECK_EQ(interp.run(c.mod, "caller", {RuntimeValue::from_ptr(unexp_ptr), RuntimeValue::from_i64(5)}).as_i64(),
             500 + 35);

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

TEST_CASE("Sweep35b - tier 2 installs speculatively inlined code whose guard passes and fails natively") {
    auto mod = parse_ok(kSpecTier2);
    FunctionDispatchTable prog;
    prog.tiering().type_feedback().get_or_create("caller").record_call_target(1, 0, "expected_fn");

    // The tier-2 pipeline's speculation fires on this feedback: a guard
    // Tier 0 does not have.
    {
        auto copy = parse_ok(kSpecTier2);
        SpeculativeInlinerOptions opts;
        opts.enable_inlining = true;
        REQUIRE(run_speculative_devirtualization(*copy, prog.tiering().type_feedback(), opts));
        const Function* spec = copy->get_function("caller");
        REQUIRE_EQ(count_op(*spec, Opcode::guard), 1u);
        CHECK_EQ(count_op(*mod->get_function("caller"), Opcode::guard), 0u);
    }

    FunctionHandle* h = prog.get_or_create("caller", mod->get_function("caller"));
    CodeInstaller installer(prog);
    CodeInstallResult res = installer.install_tier2(*h, *mod, "caller");
    if (!res.success) std::cerr << res.error_message << "\n";
    REQUIRE(res.success);
    CHECK(!h->tier2_rejected());
    CHECK(h->tier() == TierLevel::Tier2_Optimized);
    REQUIRE(h->jit_engine() != nullptr);

    // The pointers the tier-2 code's own func_addr yields: @expected_fn's
    // passes the guard, @unexpected_fn's fails it and makes the call.
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
    // A failing guard branched inside the tier-2 code: nothing deoptimized,
    // and the code is still installed.
    CHECK_EQ(prog.pipeline().tier2_deopts(), deopts0);
    CHECK(h->tier() == TierLevel::Tier2_Optimized);
}
