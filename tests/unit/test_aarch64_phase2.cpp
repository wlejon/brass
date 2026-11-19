#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>

using namespace brass;
using namespace brass::codegen;
using namespace brass::aarch64;

namespace {

size_t count_opcode(const LirFunction& fn, LirOpcode op) {
    size_t n = 0;
    for (const auto& b : fn.blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->opcode == op) ++n;
        }
    }
    return n;
}

std::unique_ptr<LirFunction> lower_a64(Module& mod, const char* name) {
    Function* fn = mod.get_function(name);
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    AArch64ISel isel(Target::aarch64_macos(), CallingConvention::apple_aapcs64());
    return isel.lower(*fn);
}

std::unique_ptr<LirFunction> allocate_a64(Module& mod, const char* name) {
    auto lir = lower_a64(mod, name);
    REQUIRE(lir != nullptr);
    lir->sort_blocks_rpo();
    LivenessAnalysis liveness(*lir);
    liveness.run();
    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::apple_aapcs64());
    regalloc.allocate();
    return lir;
}

bool contains_u64(const std::vector<uint8_t>& code, uint64_t v) {
    for (size_t i = 0; i + 8 <= code.size(); ++i) {
        uint64_t w = 0;
        std::memcpy(&w, code.data() + i, 8);
        if (w == v) return true;
    }
    return false;
}

// Mirrors bronze's pinned TLS block layout for JIT runtime tests.
struct alignas(16) PinnedTlsBlock {
    uint64_t words[8] = {};
};
thread_local PinnedTlsBlock t_a64_tls;

} // namespace

// 1. Encoder: single-instruction mov imm, movn/movk, fcsel, and ldr_literal
TEST_CASE("AArch64 Phase 2 - Encoder single-inst mov, fcsel, and ldr_literal") {
    auto mov_bytes = [](uint64_t imm) {
        CodeBuffer buf;
        AArch64Encoder enc(buf);
        enc.mov(GPR::X0, imm);
        return buf.size();
    };

    CHECK(AArch64Encoder::single_inst_mov_imm(0));
    CHECK(AArch64Encoder::single_inst_mov_imm(0x1234ull));
    CHECK(AArch64Encoder::single_inst_mov_imm(~0ull));
    CHECK(AArch64Encoder::single_inst_mov_imm(0x0000ffffffffffffull));
    CHECK(AArch64Encoder::single_inst_mov_imm(0xfff0ffffffffffffull));
    CHECK(AArch64Encoder::single_inst_mov_imm(0xfff1000000000000ull));
    CHECK(!AArch64Encoder::single_inst_mov_imm(0x0001000200030004ull));

    CHECK_EQ(mov_bytes(0), size_t(4));
    CHECK_EQ(mov_bytes(~0ull), size_t(4));
    CHECK_EQ(mov_bytes(0x0000ffffffffffffull), size_t(4));  // orr x0, xzr, #mask
    CHECK_EQ(mov_bytes(0xfff0ffffffffffffull), size_t(4));  // movn x0, #0xf, lsl #48
    CHECK_EQ(mov_bytes(0xfff1000000000000ull), size_t(4));  // movz x0, #0xfff1, lsl #48
    CHECK_EQ(mov_bytes(0xfffe1234ffffffffull), size_t(8));  // movn + movk instead of 4 insns
    CHECK_EQ(mov_bytes(0x0001000200030004ull), size_t(16));

    // fcsel / fcsel_s / ldr_literal encodings
    CodeBuffer buf;
    AArch64Encoder enc(buf);
    Label lit = buf.create_label();
    enc.ldr_literal(GPR::X0, lit, 0x9e3779b97f4a7c15ull);
    enc.ldr_literal(FPR::V1, lit, 0x7ff8000000000000ull);
    enc.fcsel(FPR::V0, FPR::V1, FPR::V2, Condition::VS);
    enc.fcsel_s(FPR::V3, FPR::V4, FPR::V5, Condition::EQ);
    enc.ret();
    buf.align(8);
    buf.bind(lit);
    buf.emit64(0x9e3779b97f4a7c15ull);

    auto read32 = [&](size_t off) {
        uint32_t w = 0;
        std::memcpy(&w, buf.data() + off, 4);
        return w;
    };
    CHECK_EQ(read32(0), 0x580000c0u);  // ldr x0, .+24
    CHECK_EQ(read32(4), 0x5c0000a1u);  // ldr d1, .+20
    CHECK_EQ(read32(8), 0x1e626c20u);  // fcsel d0, d1, d2, vs
    CHECK_EQ(read32(12), 0x1e250c83u); // fcsel s3, s4, s5, eq
}

// 2. Pinned TLS Constants (Movq_xg fold + [x28, #off] loads + pinned_tls_write guard)
TEST_CASE("AArch64 Phase 2 - Pinned TLS constants fold FPR loads and read from X28") {
    constexpr uint64_t kOther = 0x9e3779b97f4a7c15ull;
    const char* kPinned = R"(module @m

attributes pinned_tls_register pinned_constant 0xfff1000000000000 -32 pinned_constant 0x7ff8000000000000 -24 pinned_constant 0xbf58476d1ce4e5b9 -16

func @pin_fp(%x: i64) -> f64 {
entry:
  %k_nan = iconst.i64 0x7ff8000000000000
  %d_nan = bitcast.f64.i64 %k_nan
  %k_neg = iconst.i64 0xbf58476d1ce4e5b9
  %a = xor.i64 %x, %k_neg
  %da = bitcast.f64.i64 %a
  %r = add.f64 %da, %d_nan
  ret %r
}

func @entry(%tls: i64, %x: i64) -> f64 {
entry:
  pinned_tls_write %tls
  %k = iconst.i64 0xbf58476d1ce4e5b9
  %y = xor.i64 %x, %k
  %r = call.f64 @pin_fp(%y)
  ret %r
}
)";
    t_a64_tls.words[0] = 0xfff1000000000000ull; // -32 from words[4]
    t_a64_tls.words[1] = 0x7ff8000000000000ull; // -24
    t_a64_tls.words[2] = 0xbf58476d1ce4e5b9ull; // -16
    t_a64_tls.words[3] = kOther;                // -8 (not pinned)

    auto mod = parse_module(kPinned);
    REQUIRE(mod != nullptr);

    // Movq_xg of pinned 0x7ff8... folds to Movsd [X28, #-24]
    auto lir_pin = lower_a64(*mod, "pin_fp");
    REQUIRE(lir_pin != nullptr);
    bool saw_x28_movsd = false;
    for (const auto& b : lir_pin->blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->opcode == LirOpcode::Movsd && inst->uses.size() == 1 && inst->uses[0].is_mem()) {
                const auto& m = inst->uses[0].mem_val;
                if (m.base_preg == PReg::aarch64_gpr(GPR::X28) && m.disp == -24) saw_x28_movsd = true;
            }
        }
    }
    CHECK(saw_x28_movsd);

    // @entry writes pinned_tls_write, so 0xbf58476d1ce4e5b9 stays unpinned there
    auto lir_entry = allocate_a64(*mod, "entry");
    AArch64EmitContext em_entry(*lir_entry, Target::aarch64_macos());
    auto res_entry = em_entry.compile();
    CHECK((lir_entry->forced_saved_gprs & reg_mask(GPR::X28)) != 0);
    (void)res_entry;

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto entry_fn = jit.get_function_ptr<double (*)(uint64_t, uint64_t)>("entry");
    REQUIRE(entry_fn != nullptr);
    double out = entry_fn(reinterpret_cast<uintptr_t>(&t_a64_tls.words[4]), 0xbf58476d1ce4e5b9ull ^ 0x3ff0000000000000ull);
    // xor with k twice -> 0x3ff0000000000000 (1.0), plus NaN -> NaN
    CHECK(std::isnan(out));
}

// 3. Exit-Only Guard State Folding (CanonF64Bits, I32ToF64Bits, BoolTag) + stack arg alignment
TEST_CASE("AArch64 Phase 2 - Exit-only guard state folds convert only on guard exit") {
    const char* kGuardMod = R"(module @m

func @exit_stub(%c: tagged, %d: i64, %b: i64, %s0: i64, %s1: i64, %s2: i64, %s3: i64, %s4: i64, %s5: i64, %s_canon: tagged, %s_i32f64: i64, %s_bool: i64) -> i64 {
entry:
  %c0 = bitcast.i64.tagged %c
  %sc = bitcast.i64.tagged %s_canon
  %h1 = xor.i64 %sc, %s_i32f64
  %h2 = xor.i64 %h1, %s_bool
  %h3 = xor.i64 %h2, %c0
  %h4 = xor.i64 %h3, %d
  %h5 = xor.i64 %h4, %b
  ret %h5
}

func @hot(%flag: i64, %f: f64, %x: i32, %bv: i32) -> i64 {
entry:
  %fbits = bitcast.i64.f64 %f
  %isnan = ne.f64 %f, %f
  %k_nan = iconst.i64 0x7ff8000000000000
  %canon = select.i64 %isnan, %k_nan, %fbits
  %c_tag = bitcast.tagged.i64 %canon
  %fd = sitofp.f64.i32 %x
  %fb = bitcast.i64.f64 %fd
  %one32 = iconst.i32 1
  %b1 = and.i32 %bv, %one32
  %bz = zext.i64 %b1
  %k_bool = iconst.i64 0xfff4000000000000
  %btag = or.i64 %bz, %k_bool
  %z = iconst.i64 0
  guard %flag, @exit_stub, [%c_tag, %fb, %btag, %z, %z, %z, %z, %z, %z, %c_tag, %fb, %btag]
  %ok = iconst.i64 42
  ret %ok
}
)";
    auto mod = parse_module(kGuardMod);
    REQUIRE(mod != nullptr);
    auto lir = lower_a64(*mod, "hot");
    REQUIRE(lir != nullptr);

    // Hot path skipped sitofp.f64.i32 and select for NaN canon
    CHECK_EQ(count_opcode(*lir, LirOpcode::Cvtsi2sd32), 0u);
    CHECK_EQ(count_opcode(*lir, LirOpcode::Cmovcc), 0u);

    const LirInst* guard_exit = nullptr;
    for (const auto& b : lir->blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->opcode == LirOpcode::GuardExit) guard_exit = inst.get();
        }
    }
    REQUIRE(guard_exit != nullptr);
    REQUIRE_EQ(guard_exit->state_convs.size(), size_t(12));
    CHECK_EQ(guard_exit->state_convs[0], static_cast<uint8_t>(GuardStateConv::CanonF64Bits));
    CHECK_EQ(guard_exit->state_convs[1], static_cast<uint8_t>(GuardStateConv::I32ToF64Bits));
    CHECK_EQ(guard_exit->state_convs[2], static_cast<uint8_t>(GuardStateConv::BoolTag));
    CHECK_EQ(guard_exit->state_convs[9], static_cast<uint8_t>(GuardStateConv::CanonF64Bits));
    CHECK_EQ(guard_exit->state_convs[10], static_cast<uint8_t>(GuardStateConv::I32ToF64Bits));
    CHECK_EQ(guard_exit->state_convs[11], static_cast<uint8_t>(GuardStateConv::BoolTag));

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto hot = jit.get_function_ptr<uint64_t (*)(int64_t, double, int32_t, int32_t)>("hot");
    REQUIRE(hot != nullptr);

    CHECK_EQ(hot(1, 1.5, 7, 1), 42ull);

    const uint64_t raw_nan = 0xfff123456789abcdull;
    const uint64_t exp_nan = 0x7ff8000000000000ull;
    const uint64_t exp_d = std::bit_cast<uint64_t>(-42.0);
    const uint64_t exp_b = 0xfff4000000000001ull;
    // Stack args mirror reg args in exit_stub, so xor cancels and wait: let's return non-zero check!
    // exit_stub xors canon^d^b (stack) ^ canon^d^b (reg) -> wait! That would be 0 if identical!
    // Even better: check stack == reg (0 implies both register AND stack slots match identically!)
    CHECK_EQ(hot(0, std::bit_cast<double>(raw_nan), -42, 3), 0ull);
    (void)exp_nan;
    (void)exp_d;
    (void)exp_b;
    CHECK_EQ(hot(0, -3.25, 123456, 2), 0ull);
}

// Separate asymmetrical check verifying the exact converted values themselves (not just stack == reg)
TEST_CASE("AArch64 Phase 2 - Guard state folding produces exact bits") {
    const char* kExact = R"(module @m
func @stub(%c: i64, %d: i64, %b: i64) -> i64 {
entry:
  %three = iconst.i64 3
  %k1 = mul.i64 %c, %three
  %k2 = xor.i64 %k1, %d
  %r = add.i64 %k2, %b
  ret %r
}
func @hot(%flag: i64, %f: f64, %x: i32, %bv: i32) -> i64 {
entry:
  %fbits = bitcast.i64.f64 %f
  %isnan = ne.f64 %f, %f
  %k_nan = iconst.i64 0x7ff8000000000000
  %canon = select.i64 %isnan, %k_nan, %fbits
  %fd = sitofp.f64.i32 %x
  %fb = bitcast.i64.f64 %fd
  %one32 = iconst.i32 1
  %b1 = and.i32 %bv, %one32
  %bz = zext.i64 %b1
  %k_bool = iconst.i64 0xfff4000000000000
  %btag = or.i64 %bz, %k_bool
  guard %flag, @stub, [%canon, %fb, %btag]
  %ok = iconst.i64 11
  ret %ok
}
)";
    auto mod = parse_module(kExact);
    REQUIRE(mod != nullptr);
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto hot = jit.get_function_ptr<uint64_t (*)(int64_t, double, int32_t, int32_t)>("hot");
    REQUIRE(hot != nullptr);

    CHECK_EQ(hot(0, std::bit_cast<double>(0xfff123456789abcdull), -42, 3),
             ((0x7ff8000000000000ull * 3ull) ^ std::bit_cast<uint64_t>(-42.0)) + 0xfff4000000000001ull);
    CHECK_EQ(hot(0, -3.25, 19, 2),
             ((std::bit_cast<uint64_t>(-3.25) * 3ull) ^ std::bit_cast<uint64_t>(19.0)) + 0xfff4000000000000ull);
}

// 4. Native fcsel float select + in-FP-domain NaN-canonicalizing stores
TEST_CASE("AArch64 Phase 2 - Float select fcsel and NaN-canon store stay in FP domain") {
    const char* kFpMod = R"(module @m
func @canon(%p: ptr, %x: f64, %y: f64) -> i64 {
entry:
  %s = mul.f64 %x, %y
  %b = bitcast.i64.f64 %s
  %n = ne.f64 %s, %s
  %k = iconst.i64 0x7ff8000000000000
  %v = select.i64 %n, %k, %b
  store.i64 %p, 8, %v
  %b2 = bitcast.i64.f64 %x
  %e = eq.f64 %x, %x
  %v2 = select.i64 %e, %b2, %k
  store.i64 %p, 16, %v2
  %z = iconst.i64 0
  ret %z
}
func @sel_f64(%a: i64, %x: f64, %y: f64) -> f64 {
entry:
  %z = iconst.i64 0
  %c = ne.i64 %a, %z
  %r = select.f64 %c, %x, %y
  ret %r
}
func @sel_f32(%a: i64, %x: f32, %y: f32) -> f32 {
entry:
  %z = iconst.i64 0
  %c = ne.i64 %a, %z
  %r = select.f32 %c, %x, %y
  ret %r
}
)";
    auto mod = parse_module(kFpMod);
    REQUIRE(mod != nullptr);

    auto lir_canon = lower_a64(*mod, "canon");
    CHECK_EQ(count_opcode(*lir_canon, LirOpcode::Ucomisd), 2u);
    CHECK_EQ(count_opcode(*lir_canon, LirOpcode::Cmovcc), 2u);
    CHECK_EQ(count_opcode(*lir_canon, LirOpcode::Movq_xg), 1u); // cached K across both stores
    CHECK_EQ(count_opcode(*lir_canon, LirOpcode::Movq_gx), 0u);

    auto lir_sel64 = lower_a64(*mod, "sel_f64");
    CHECK_EQ(count_opcode(*lir_sel64, LirOpcode::Cmovcc), 1u);
    CHECK_EQ(count_opcode(*lir_sel64, LirOpcode::Movq_gx), 0u);
    CHECK_EQ(count_opcode(*lir_sel64, LirOpcode::Movq_xg), 0u);

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto canon = jit.get_function_ptr<int64_t (*)(uint64_t*, double, double)>("canon");
    auto sel64 = jit.get_function_ptr<double (*)(int64_t, double, double)>("sel_f64");
    auto sel32 = jit.get_function_ptr<float (*)(int64_t, float, float)>("sel_f32");
    REQUIRE(canon && sel64 && sel32);

    const uint64_t nans[] = {0x7ff8000000000000ull, 0xfff8000000000000ull,
                             0x7ff0000000000001ull, 0xfff123456789abcdull};
    for (uint64_t bits : nans) {
        uint64_t mem[3] = {1, 2, 3};
        canon(mem, std::bit_cast<double>(bits), 2.0);
        CHECK_EQ(mem[0], 1ull);
        CHECK_EQ(mem[1], 0x7ff8000000000000ull);
        CHECK_EQ(mem[2], 0x7ff8000000000000ull);
    }
    uint64_t mem[3] = {0, 0, 0};
    canon(mem, -0.0, 1.0);
    CHECK_EQ(mem[1], std::bit_cast<uint64_t>(-0.0));
    CHECK_EQ(mem[2], std::bit_cast<uint64_t>(-0.0));

    CHECK_EQ(sel64(1, 3.5, -2.25), 3.5);
    CHECK_EQ(sel64(0, 3.5, -2.25), -2.25);
    CHECK_EQ(sel32(1, 1.25f, -4.5f), 1.25f);
    CHECK_EQ(sel32(0, 1.25f, -4.5f), -4.5f);
}

// 5. Constant-slot rematerialization & literal pool on AArch64
TEST_CASE("AArch64 Phase 2 - Spilled constants pool cleanly and execute accurately") {
    const char* kConstMod = R"(module @m
func @sink(%x: i64) -> i64 {
entry:
  %one = iconst.i64 1
  %r = add.i64 %x, %one
  ret %r
}
func @many(%x: i64) -> i64 {
entry:
  %c0 = iconst.i64 0x9e3779b97f4a7c15
  %c1 = iconst.i64 0xbf58476d1ce4e5b9
  %c2 = iconst.i64 0x94d049bb133111eb
  %c3 = iconst.i64 0x517cc1b727220a95
  %c4 = iconst.i64 0x2545f4914f6cdd1d
  %c5 = iconst.i64 0x106689d45497fdb5
  %c6 = iconst.i64 0xd1b54a32d192ed03
  %c7 = iconst.i64 0x6c078965c3a8d111
  %c8 = iconst.i64 0x123456789abcdef1
  %c9 = iconst.i64 0x23456789abcdef03
  %c10 = iconst.i64 0x3456789abcdef015
  %c11 = iconst.i64 0x456789abcdef0127
  %c12 = iconst.i64 0x56789abcdef01239
  %c13 = iconst.i64 0x6789abcdef01234b
  %r0 = call.i64 @sink(%x)
  %a0 = xor.i64 %r0, %c0
  %a1 = add.i64 %a0, %c1
  %a2 = xor.i64 %a1, %c2
  %a3 = add.i64 %a2, %c3
  %a4 = xor.i64 %a3, %c4
  %a5 = add.i64 %a4, %c5
  %a6 = xor.i64 %a5, %c6
  %a7 = add.i64 %a6, %c7
  %a8 = xor.i64 %a7, %c8
  %a9 = add.i64 %a8, %c9
  %a10 = xor.i64 %a9, %c10
  %a11 = add.i64 %a10, %c11
  %a12 = xor.i64 %a11, %c12
  %a13 = add.i64 %a12, %c13
  %r1 = call.i64 @sink(%a13)
  %b0 = xor.i64 %r1, %c0
  %b1 = xor.i64 %b0, %c1
  %b2 = xor.i64 %b1, %c2
  %b3 = xor.i64 %b2, %c3
  %b4 = xor.i64 %b3, %c4
  %b5 = xor.i64 %b4, %c5
  %b6 = xor.i64 %b5, %c6
  %b7 = xor.i64 %b6, %c7
  %b8 = xor.i64 %b7, %c8
  %b9 = xor.i64 %b8, %c9
  %b10 = xor.i64 %b9, %c10
  %b11 = xor.i64 %b10, %c11
  %b12 = xor.i64 %b11, %c12
  %out = xor.i64 %b12, %c13
  ret %out
}
)";
    auto mod = parse_module(kConstMod);
    REQUIRE(mod != nullptr);
    auto lir = allocate_a64(*mod, "many");
    REQUIRE(lir != nullptr);
    CHECK(!lir->frame.constant_slots.empty());
    std::unordered_set<int32_t> pooled;
    for (const auto& [slot, val] : lir->frame.constant_slots) pooled.insert(slot);
    for (const auto& b : lir->blocks) {
        for (const auto& inst : b->instructions) {
            for (const auto& d : inst->defs) CHECK_FALSE(d.is_spill_slot() && pooled.count(d.spill_slot));
        }
    }

    AArch64EmitContext em(*lir, Target::aarch64_macos());
    auto res = em.compile();
    for (const auto& [slot, val] : lir->frame.constant_slots) {
        (void)slot;
        CHECK(contains_u64(res.code_buffer.bytes(), static_cast<uint64_t>(val)));
    }

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto many = jit.get_function_ptr<uint64_t (*)(uint64_t)>("many");
    REQUIRE(many != nullptr);
    uint64_t x = 123456789ull;
    uint64_t r = (((((((((((((((x + 1) ^ 0x9e3779b97f4a7c15ull) + 0xbf58476d1ce4e5b9ull) ^
                              0x94d049bb133111ebull) + 0x517cc1b727220a95ull) ^
                            0x2545f4914f6cdd1dull) + 0x106689d45497fdb5ull) ^
                          0xd1b54a32d192ed03ull) + 0x6c078965c3a8d111ull) ^
                        0x123456789abcdef1ull) + 0x23456789abcdef03ull) ^
                      0x3456789abcdef015ull) + 0x456789abcdef0127ull) ^
                    0x56789abcdef01239ull) + 0x6789abcdef01234bull) + 1;
    uint64_t exp = r ^ 0x9e3779b97f4a7c15ull ^ 0xbf58476d1ce4e5b9ull ^ 0x94d049bb133111ebull ^
                   0x517cc1b727220a95ull ^ 0x2545f4914f6cdd1dull ^ 0x106689d45497fdb5ull ^
                   0xd1b54a32d192ed03ull ^ 0x6c078965c3a8d111ull ^ 0x123456789abcdef1ull ^
                   0x23456789abcdef03ull ^ 0x3456789abcdef015ull ^ 0x456789abcdef0127ull ^
                   0x56789abcdef01239ull ^ 0x6789abcdef01234bull;
    CHECK_EQ(many(x), exp);
}

// 6. bitcast_through_slots peephole enabled on AArch64
TEST_CASE("AArch64 Phase 2 - bitcast_through_slots peephole folds GPR spill round-trips") {
    LirFunction fn;
    fn.calling_conv = CallingConvention::apple_aapcs64();
    LirBlock* bb = fn.create_block("entry");

    auto load = std::make_unique<LirInst>(LirOpcode::Mov);
    load->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    load->add_use(LirOperand::slot(0, 8));
    bb->append_inst(std::move(load));

    auto to_fpr = std::make_unique<LirInst>(LirOpcode::Movq_xg);
    to_fpr->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
    to_fpr->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    bb->append_inst(std::move(to_fpr));

    auto to_gpr = std::make_unique<LirInst>(LirOpcode::Movq_gx);
    to_gpr->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
    to_gpr->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 8));
    bb->append_inst(std::move(to_gpr));

    auto store = std::make_unique<LirInst>(LirOpcode::Mov);
    store->add_def(LirOperand::slot(1, 8));
    store->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
    bb->append_inst(std::move(store));

    for (GPR g : {GPR::X0, GPR::X1}) {
        auto kill = std::make_unique<LirInst>(LirOpcode::Mov);
        kill->add_def(LirOperand::preg_aarch64_gpr(g, 8));
        kill->add_use(LirOperand::imm(7, 8));
        bb->append_inst(std::move(kill));
    }
    auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
    ret->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    bb->append_inst(std::move(ret));

    PeepholeStats stats = run_lir_peephole_optimizations(fn);
    CHECK_EQ(stats.bitcasts_through_slots, size_t(2));
    CHECK(bb->instructions[0]->opcode == LirOpcode::Movsd);
    CHECK(bb->instructions[0]->defs[0].preg_val == PReg::aarch64_fpr(FPR::V0));
    CHECK(bb->instructions[0]->uses[0].spill_slot == 0);
    CHECK(bb->instructions[1]->opcode == LirOpcode::Movsd);
    CHECK(bb->instructions[1]->defs[0].spill_slot == 1);
    CHECK(bb->instructions[1]->uses[0].preg_val == PReg::aarch64_fpr(FPR::V1));
}
