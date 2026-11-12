#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <bit>
#include <cmath>
#include <cstdint>

// A store of a NaN-canonicalized double, `select(ne f, f, K, bitcast f)`
// (a NaN-boxing front end's boxed number), stays in the FP domain with AVX
// (x64_isel_nan_canon.cpp): vcmpunordsd, vblendvpd against K in an xmm,
// movsd. The bits stored are the select's: f's own unless f is a NaN of any
// sign or payload, then K. A select with another use keeps the GPR form.

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

namespace {

const char* kStore = R"(module @m

func @canon(%p: i64, %x: f64, %y: f64) -> i64 {
entry:
  %s = mul.f64 %x, %y
  %b = bitcast.i64.f64 %s
  %n = ne.f64 %s, %s
  %k = iconst.i64 9221120237041090560
  %v = select.i64 %n, %k, %b
  store.i64 %p, 8, %v
  %b2 = bitcast.i64.f64 %x
  %e = eq.f64 %x, %x
  %v2 = select.i64 %e, %b2, %k
  store.i64 %p, 16, %v2
  %z = iconst.i64 0
  ret %z
}

func @shared(%p: i64, %x: f64) -> i64 {
entry:
  %b = bitcast.i64.f64 %x
  %n = ne.f64 %x, %x
  %k = iconst.i64 9221120237041090560
  %v = select.i64 %n, %k, %b
  store.i64 %p, 8, %v
  ret %v
}
)";

size_t count(const LirFunction& fn, LirOpcode op) {
    size_t n = 0;
    for (const auto& b : fn.blocks) {
        for (const auto& inst : b->instructions) n += inst->opcode == op;
    }
    return n;
}

std::unique_ptr<LirFunction> lower(Module& mod, const char* name, bool avx) {
    Function* fn = mod.get_function(name);
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    isel.set_avx(avx);
    return isel.lower(*fn);
}

constexpr uint64_t kCanon = 0x7ff8000000000000ull;

} // namespace

TEST_CASE("x64 NaN-canonicalizing store - blends in the FP domain with AVX, a cmov without") {
    auto mod = parse_module(kStore);
    REQUIRE(mod != nullptr);
    auto avx = lower(*mod, "canon", true);
    CHECK_EQ(count(*avx, LirOpcode::Vblendvpd), 2u);
    CHECK_EQ(count(*avx, LirOpcode::Vcmppd), 2u);
    CHECK_EQ(count(*avx, LirOpcode::Cmovcc), 0u);
    CHECK_EQ(count(*avx, LirOpcode::Movq_xg), 1u);  // K made once for both stores
    auto sse = lower(*mod, "canon", false);
    CHECK_EQ(count(*sse, LirOpcode::Vblendvpd), 0u);
    CHECK_EQ(count(*sse, LirOpcode::Cmovcc), 2u);
    // The select is also returned: it stays a select.
    auto shared = lower(*mod, "shared", true);
    CHECK_EQ(count(*shared, LirOpcode::Vblendvpd), 0u);
    CHECK_EQ(count(*shared, LirOpcode::Cmovcc), 1u);
}

TEST_CASE("x64 NaN-canonicalizing store - the JIT stores the select's bits for every NaN and signed zero") {
    JitExecutionEngine jit;
    auto mod = parse_module(kStore);
    REQUIRE(mod != nullptr);
    REQUIRE(jit.compile_and_load(*mod));
    auto canon = jit.get_function_ptr<int64_t (*)(uint64_t*, double, double)>("canon");
    auto shared = jit.get_function_ptr<int64_t (*)(uint64_t*, double)>("shared");
    REQUIRE(canon != nullptr);
    REQUIRE(shared != nullptr);
    const uint64_t nans[] = {0x7ff8000000000000ull, 0xfff8000000000000ull, 0x7ff0000000000001ull,
                             0xfff123456789abcdull, 0x7fffffffffffffffull, 0xfff7ffffffffffffull};
    for (uint64_t bits : nans) {
        const double n = std::bit_cast<double>(bits);
        uint64_t mem[3] = {1, 2, 3};
        canon(mem, n, 2.0);
        CHECK_EQ(mem[0], 1u);
        CHECK_EQ(mem[1], kCanon);
        CHECK_EQ(mem[2], kCanon);
        uint64_t mem2[2] = {0, 0};
        CHECK_EQ(static_cast<uint64_t>(shared(mem2, n)), kCanon);
        CHECK_EQ(mem2[1], kCanon);
    }
    // inf * 0: the negative default NaN x86 makes, canonicalized.
    uint64_t mem[3] = {0, 0, 0};
    canon(mem, INFINITY, 0.0);
    CHECK_EQ(mem[1], kCanon);
    CHECK_EQ(mem[2], std::bit_cast<uint64_t>(double(INFINITY)));
    const double plain[][2] = {{-0.0, 1.0}, {0.0, -1.0}, {1.5, -2.25}, {1e308, 10.0}, {-INFINITY, 3.0}};
    for (const auto& c : plain) {
        canon(mem, c[0], c[1]);
        const double prod = c[0] * c[1];
        CHECK_EQ(mem[1], std::bit_cast<uint64_t>(prod));
        CHECK_EQ(mem[2], std::bit_cast<uint64_t>(c[0]));
    }
}

TEST_CASE("x64 NaN-canonicalizing store - a pinned K is loaded into its xmm from the pinned word") {
    // K kept at the pinned register: `movsd k, [r13 - 16]`, where the movabs
    // into a GPR and the movq out of it were two instructions.
    const std::string text = std::string("module @m\n\nattributes pinned_tls_register pinned_constant "
                                         "0x7ff8000000000000 -16\n") +
                             (std::string(kStore).substr(std::string("module @m\n").size()));
    auto mod = parse_module(text);
    REQUIRE(mod != nullptr);
    auto lir = lower(*mod, "canon", true);
    CHECK_EQ(count(*lir, LirOpcode::Vblendvpd), 2u);
    CHECK_EQ(count(*lir, LirOpcode::Movq_xg), 0u);
    CHECK_EQ(count(*lir, LirOpcode::Movabs), 0u);
    size_t loads = 0;
    for (const auto& b : lir->blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->opcode != LirOpcode::Movsd || inst->uses.size() != 1 || !inst->uses[0].is_mem()) continue;
            const auto& m = inst->uses[0].mem_val;
            loads += m.base_preg == PReg::gpr(GPR::R13) && m.disp == -16 && inst->defs[0].is_vreg() &&
                     inst->defs[0].vreg_val.is_xmm();
        }
    }
    CHECK_EQ(loads, 1u);
}
