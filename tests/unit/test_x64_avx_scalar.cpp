#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/emit_context.hpp>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>

// Scalar float arithmetic in the three-operand VEX form (X64ISel::set_avx):
// an operand read again needs no copy, the result is what the SSE form
// computes (the first source's NaN when both are NaN), and a function not
// compiled for AVX refuses to emit one. And the allocator's stores into a
// slot that already holds the value: a value with several definitions (a
// block parameter, a two-address chain) is stored back after its reload
// only when it changed.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

struct Lowered {
    std::unique_ptr<Module> mod;
    std::unique_ptr<LirFunction> lir;
};

Lowered lower_and_allocate(const std::string& text, const char* name, bool avx) {
    Lowered out;
    out.mod = parse_module(text);
    REQUIRE(out.mod != nullptr);
    Function* fn = out.mod->get_function(name);
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    isel.set_avx(avx);
    out.lir = isel.lower(*fn);
    REQUIRE(out.lir != nullptr);
    out.lir->sort_blocks_rpo();
    LivenessAnalysis liveness(*out.lir);
    liveness.run();
    LinearScanAllocator regalloc(*out.lir, liveness, CallingConvention::sysv64());
    regalloc.allocate();
    return out;
}

bool is_scalar_arith(LirOpcode op) {
    return op == LirOpcode::Addsd || op == LirOpcode::Subsd || op == LirOpcode::Mulsd || op == LirOpcode::Divsd;
}

bool is_xmm_copy(const LirInst& inst) {
    return (inst.opcode == LirOpcode::Movaps || inst.opcode == LirOpcode::Movsd) && inst.defs.size() == 1 &&
           inst.uses.size() == 1 && inst.defs[0].is_preg() && inst.uses[0].is_preg() &&
           inst.defs[0].preg_val.is_xmm();
}

// Each input read twice: in the two-address form every product copies its
// left operand first, since that operand is read again.
const char* kReused = R"(module @m

func @reused(%a: f64, %b: f64, %c: f64) -> f64 {
entry:
  %ab = mul.f64 %a, %b
  %ac = mul.f64 %a, %c
  %bc = mul.f64 %b, %c
  %s = add.f64 %ab, %ac
  %t = sub.f64 %s, %bc
  %u = sdiv.f64 %t, %a
  ret %u
}
)";

double reused_expected(double a, double b, double c) {
    volatile double ab = a * b, ac = a * c, bc = b * c;
    volatile double s = ab + ac;
    volatile double t = s - bc;
    return t / a;
}

} // namespace

TEST_CASE("x64 AVX - scalar arithmetic is three-operand and copies no operand") {
    Lowered sse = lower_and_allocate(kReused, "reused", false);
    Lowered avx = lower_and_allocate(kReused, "reused", true);
    CHECK_FALSE(sse.lir->x64_avx);
    CHECK(avx.lir->x64_avx);
    size_t sse_copies = 0, avx_copies = 0, untied = 0;
    for (const auto& b : sse.lir->blocks) {
        for (const auto& inst : b->instructions) {
            if (is_xmm_copy(*inst)) ++sse_copies;
            if (is_scalar_arith(inst->opcode)) CHECK(inst->uses[0].preg_val == inst->defs[0].preg_val);
        }
    }
    for (const auto& b : avx.lir->blocks) {
        for (const auto& inst : b->instructions) {
            if (is_xmm_copy(*inst)) ++avx_copies;
            if (is_scalar_arith(inst->opcode) && inst->uses.size() == 2 && inst->uses[0].is_preg() &&
                inst->uses[0].preg_val != inst->defs[0].preg_val) {
                ++untied;
            }
        }
    }
    CHECK(sse_copies >= 2u);
    CHECK_EQ(avx_copies, 0u);
    CHECK(untied >= 2u);

    // Both emit; the AVX one with VEX prefixes (C4/C5) the SSE one never has.
    EmitContext sse_emit(*sse.lir, Target::x64_linux());
    CompilationResult sse_res = sse_emit.compile();
    EmitContext avx_emit(*avx.lir, Target::x64_linux());
    CompilationResult avx_res = avx_emit.compile();
    CHECK(sse_res.code_buffer.size() > avx_res.code_buffer.size());
}

TEST_CASE("x64 AVX - a first source other than the destination without AVX is a hard error") {
    Lowered avx = lower_and_allocate(kReused, "reused", true);
    avx.lir->x64_avx = false;
    EmitContext emit(*avx.lir, Target::x64_linux());
    bool threw = false;
    try {
        (void)emit.compile();
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
}

TEST_CASE("x64 AVX - the JIT computes what the SSE form does, NaN operand order included") {
    JitExecutionEngine jit;
    auto mod = parse_module(kReused);
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<double (*)(double, double, double)>("reused");
    REQUIRE(fn != nullptr);
    const double cases[][3] = {{1.5, 2.25, -3.0}, {1e300, 1e10, -1e300}, {-0.0, 7.0, 3.0}, {3.0, INFINITY, -INFINITY}};
    for (const auto& c : cases) {
        const double want = reused_expected(c[0], c[1], c[2]);
        const double got = fn(c[0], c[1], c[2]);
        CHECK(std::bit_cast<uint64_t>(got) == std::bit_cast<uint64_t>(want) || (std::isnan(got) && std::isnan(want)));
    }
    // Two NaNs with different payloads: x86 returns the first source's (quieted),
    // which is the MIR lhs rule; a commuted product would return %b's.
    const double na = std::bit_cast<double>(uint64_t{0x7ff8000000000a0a});
    const double nb = std::bit_cast<double>(uint64_t{0x7ff8000000000b0b});
    const double got = fn(na, nb, nb);
    CHECK_EQ(std::bit_cast<uint64_t>(got), uint64_t{0x7ff8000000000a0a});
}

namespace {

// %v reaches the join from two blocks (two definitions); twelve values live
// through the join and read at its end keep every register busy, so %v
// waits in its slot and is reloaded for each of its reads, each followed by
// new values that evict it again.
std::string param_text() {
    std::string s = "module @m\n\nfunc @param(%p: i64, %x: f64, %y: f64) -> f64 {\nentry:\n";
    s += "  %z = iconst.i64 0\n  %c = sgt.i64 %p, %z\n  br_if %c, a, b\n";
    s += "a:\n  %ma = mul.f64 %x, %y\n  br join(%ma)\n";
    s += "b:\n  %mb = add.f64 %x, %y\n  br join(%mb)\n";
    s += "join(%v: f64):\n";
    for (int i = 0; i < 12; ++i) {
        const std::string n = std::to_string(i);
        s += "  %k" + n + " = fconst.f64 " + std::to_string(i) + ".5\n";
        s += "  %w" + n + " = add.f64 %x, %k" + n + "\n";
    }
    std::string acc = "%v";
    for (int r = 0; r < 4; ++r) {
        const std::string n = std::to_string(r);
        s += "  %u" + n + " = mul.f64 %v, %w" + std::to_string(r) + "\n";
        for (int i = 0; i < 12; ++i) {
            const std::string m = n + "_" + std::to_string(i);
            s += "  %t" + m + " = mul.f64 %u" + n + ", %w" + std::to_string(i) + "\n";
            s += "  %s" + m + " = add.f64 " + acc + ", %t" + m + "\n";
            acc = "%s" + m;
        }
    }
    for (int i = 0; i < 12; ++i) {
        const std::string n = std::to_string(i);
        s += "  %f" + n + " = add.f64 " + acc + ", %w" + n + "\n";
        acc = "%f" + n;
    }
    s += "  ret " + acc + "\n}\n";
    return s;
}

double param_expected(int64_t p, double x, double y) {
    volatile double v = p > 0 ? x * y : x + y;
    double w[12];
    for (int i = 0; i < 12; ++i) w[i] = x + (i + 0.5);
    double acc = v;
    for (int r = 0; r < 4; ++r) {
        volatile double u = v * w[r];
        for (int i = 0; i < 12; ++i) {
            volatile double t = u * w[i];
            acc = acc + t;
        }
    }
    for (int i = 0; i < 12; ++i) acc = acc + w[i];
    return acc;
}

// Stores of a register to the slot whose value it holds: loaded from it, or
// stored to it, with neither written since. Per block, from its start.
size_t stores_back(const LirFunction& fn) {
    size_t n = 0;
    for (const auto& b : fn.blocks) {
        std::unordered_map<uint32_t, int32_t> mirror;  // preg code -> slot
        for (const auto& inst : b->instructions) {
            const bool copy = (inst->opcode == LirOpcode::Movsd || inst->opcode == LirOpcode::Mov) &&
                              inst->defs.size() == 1 && inst->uses.size() == 1;
            if (copy && inst->defs[0].is_spill_slot() && inst->uses[0].is_preg()) {
                const uint32_t r = inst->uses[0].preg_val.code | (inst->uses[0].preg_val.is_xmm() ? 64u : 0u);
                const int32_t slot = inst->defs[0].spill_slot;
                if (auto it = mirror.find(r); it != mirror.end() && it->second == slot) ++n;
                for (auto it = mirror.begin(); it != mirror.end();) it = it->second == slot ? mirror.erase(it) : std::next(it);
                mirror[r] = slot;
                continue;
            }
            if (inst->is_call()) mirror.clear();
            for (const auto& d : inst->defs) {
                if (d.is_preg()) mirror.erase(d.preg_val.code | (d.preg_val.is_xmm() ? 64u : 0u));
                if (d.is_spill_slot()) {
                    for (auto it = mirror.begin(); it != mirror.end();) it = it->second == d.spill_slot ? mirror.erase(it) : std::next(it);
                }
            }
            if (copy && inst->defs[0].is_preg() && inst->uses[0].is_spill_slot()) {
                mirror[inst->defs[0].preg_val.code | (inst->defs[0].preg_val.is_xmm() ? 64u : 0u)] = inst->uses[0].spill_slot;
            }
        }
    }
    return n;
}

} // namespace

TEST_CASE("Linear Scan - a value defined twice is not stored back to the slot it was reloaded from") {
    const std::string text = param_text();
    for (bool avx : {false, true}) {
        Lowered l = lower_and_allocate(text, "param", avx);
        size_t reloads = 0;
        for (const auto& b : l.lir->blocks) {
            for (const auto& inst : b->instructions) {
                if (inst->opcode == LirOpcode::Movsd && inst->defs[0].is_preg() && inst->uses[0].is_spill_slot()) ++reloads;
            }
        }
        CHECK(reloads >= 4u);  // the pressure this is about is there
        CHECK_EQ(stores_back(*l.lir), 0u);
    }

    JitExecutionEngine jit;
    auto mod = parse_module(text);
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<double (*)(int64_t, double, double)>("param");
    REQUIRE(fn != nullptr);
    for (int64_t p : {int64_t{1}, int64_t{-1}}) {
        for (double x : {0.75, -3.25}) {
            const double want = param_expected(p, x, 1.125);
            CHECK_EQ(std::bit_cast<uint64_t>(fn(p, x, 1.125)), std::bit_cast<uint64_t>(want));
        }
    }
}
