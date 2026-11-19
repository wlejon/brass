#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/branch_threading.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

using namespace brass;
using namespace brass::codegen;
using namespace brass::aarch64;

namespace {

uint32_t read32_at(const std::vector<uint8_t>& bytes, size_t off) {
    REQUIRE(off + 4 <= bytes.size());
    uint32_t w = 0;
    std::memcpy(&w, bytes.data() + off, 4);
    return w;
}

uint32_t word_at(const CodeBuffer& buf, size_t off) {
    return read32_at(buf.bytes(), off);
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

size_t count_opcode(const LirFunction& fn, LirOpcode op) {
    size_t n = 0;
    for (const auto& b : fn.blocks) {
        for (const auto& inst : b->instructions) {
            if (inst->opcode == op) ++n;
        }
    }
    return n;
}

bool has_u32_word(const std::vector<uint8_t>& bytes, uint32_t needle) {
    for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
        if (read32_at(bytes, i) == needle) return true;
    }
    return false;
}

} // namespace

// 1. Encoder bitwords verified against Apple clang + entry_cell load_symbol_u64
TEST_CASE("AArch64 Phase 3 - Encoder csinv/csneg/SP/shifted/madd/stp/load_symbol_u64") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);
    enc.csinv(GPR::X0, GPR::X1, GPR::X2, Condition::EQ);
    enc.csinv32(GPR::X0, GPR::X1, GPR::X2, Condition::NE);
    enc.csneg(GPR::X0, GPR::X1, GPR::X2, Condition::LT);
    enc.csneg32(GPR::X0, GPR::X1, GPR::X2, Condition::GE);
    enc.csetm(GPR::X0, Condition::EQ);
    enc.csetm32(GPR::X0, Condition::NE);
    enc.cmp(GPR::SP, GPR::X0);
    enc.adds(GPR::X1, GPR::SP, GPR::X2);
    enc.subs32(GPR::X3, GPR::X4, GPR::X5);
    enc.add_shifted(GPR::X0, GPR::X1, GPR::X2, ShiftType::LSL, 3);
    enc.add32_shifted(GPR::X0, GPR::X1, GPR::X2, ShiftType::LSR, 4);
    enc.sub_shifted(GPR::X0, GPR::X1, GPR::X2, ShiftType::ASR, 5);
    enc.sub32_shifted(GPR::X0, GPR::X1, GPR::X2, ShiftType::LSL, 2);
    enc.madd(GPR::X0, GPR::X1, GPR::X2, GPR::X3);
    enc.msub32(GPR::X0, GPR::X1, GPR::X2, GPR::X3);
    enc.stp(GPR::XZR, GPR::XZR, ptr(GPR::FP, 16));
    enc.add_lsl(GPR::X0, GPR::X1, GPR::X2, 3);

    CHECK_EQ(word_at(buf, 0), 0xDA820020u);   // csinv x0, x1, x2, eq
    CHECK_EQ(word_at(buf, 4), 0x5A821020u);   // csinv w0, w1, w2, ne
    CHECK_EQ(word_at(buf, 8), 0xDA82B420u);   // csneg x0, x1, x2, lt
    CHECK_EQ(word_at(buf, 12), 0x5A82A420u);  // csneg w0, w1, w2, ge
    CHECK_EQ(word_at(buf, 16), 0xDA9F13E0u);  // csetm x0, eq
    CHECK_EQ(word_at(buf, 20), 0x5A9F03E0u);  // csetm w0, ne
    CHECK_EQ(word_at(buf, 24), 0xEB2063FFu);  // cmp sp, x0
    CHECK_EQ(word_at(buf, 28), 0xAB2263E1u);  // adds x1, sp, x2
    CHECK_EQ(word_at(buf, 32), 0x6B050083u);  // subs w3, w4, w5
    CHECK_EQ(word_at(buf, 36), 0x8B020C20u);  // add x0, x1, x2, lsl #3
    CHECK_EQ(word_at(buf, 40), 0x0B421020u);  // add w0, w1, w2, lsr #4
    CHECK_EQ(word_at(buf, 44), 0xCB821420u);  // sub x0, x1, x2, asr #5
    CHECK_EQ(word_at(buf, 48), 0x4B020820u);  // sub w0, w1, w2, lsl #2
    CHECK_EQ(word_at(buf, 52), 0x9B020C20u);  // madd x0, x1, x2, x3
    CHECK_EQ(word_at(buf, 56), 0x1B028C20u);  // msub w0, w1, w2, w3
    CHECK_EQ(word_at(buf, 60), 0xA9017FBFu);  // stp xzr, xzr, [x29, #16]
    CHECK_EQ(word_at(buf, 64), 0x8B020C20u);  // add x0, x1, x2, lsl #3

    CodeBuffer cbuf;
    AArch64Encoder cenc(cbuf);
    cenc.load_symbol_u64(GPR::X16, "cell_sym");
    REQUIRE_EQ(cbuf.size(), size_t(8));
    CHECK_EQ(word_at(cbuf, 0), 0x90000010u);  // adrp x16, #0
    CHECK_EQ(word_at(cbuf, 4), 0xF9400210u);  // ldr x16, [x16, #0]
    REQUIRE_EQ(cbuf.relocations().size(), size_t(2));
    CHECK(cbuf.relocations()[0].kind == RelocationKind::Page21);
    CHECK(cbuf.relocations()[1].kind == RelocationKind::LdSt64Lo12);
    CHECK_EQ(cbuf.relocations()[0].symbol_name, "cell_sym");

    // AArch64EmitContext Call with entry_cell uses 2 insns + blr x16 (3 total before return)
    LirFunction fn;
    fn.calling_conv = CallingConvention::apple_aapcs64();
    LirBlock* bb = fn.create_block("entry");
    auto call = std::make_unique<LirInst>(LirOpcode::Call);
    call->callee_symbol = "cell_target";
    call->add_use(LirOperand::symbol("cell_target"));
    call->entry_cell = "cell_sym";
    call->clobbered_gprs = 0x1FFu;
    bb->append_inst(std::move(call));
    bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
    AArch64EmitContext em(fn, Target::aarch64_macos());
    auto res = em.compile();
    bool saw_ldr_x16_blr = false;
    const auto& bytes = res.code_buffer.bytes();
    for (size_t i = 0; i + 12 <= bytes.size(); i += 4) {
        if (read32_at(bytes, i) == 0x90000010u &&
            read32_at(bytes, i + 4) == 0xF9400210u &&
            read32_at(bytes, i + 8) == 0xD63F0200u) {
            saw_ldr_x16_blr = true;
        }
    }
    CHECK(saw_ldr_x16_blr);
    CHECK(res.code_buffer.relocations().size() >= 2);
}

// 2. Emit-time cbz/cbnz/tbz/tbnz fusion + non-adjacent scheduling + relaxation + live JIT
TEST_CASE("AArch64 Phase 3 - Emit-time cbz/cbnz/tbz/tbnz fusion and relaxation") {
    // Direct LIR test: non-adjacent independent instruction between cmp and jcc + >32 KiB tbz relax
    {
        LirFunction fn;
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* entry = fn.create_block("entry");
        LirBlock* bit_blk = fn.create_block("bit_blk");
        LirBlock* pad_blk = fn.create_block("pad_blk");
        LirBlock* yes_blk = fn.create_block("yes_blk");
        LirBlock* far_blk = fn.create_block("far_blk");

        // Cmp x0, #0 ; Add x1, x1, #5 (does not touch flags or x0) ; Jcc.NE bit_blk ; Jmp yes_blk
        // With bit_blk immediately following entry, falls_to(true_target) inverts cbnz -> cbz x0, yes_blk!
        auto cmp0 = std::make_unique<LirInst>(LirOpcode::Cmp);
        cmp0->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        cmp0->add_use(LirOperand::imm(0, 8));
        entry->append_inst(std::move(cmp0));

        auto add1 = std::make_unique<LirInst>(LirOpcode::Add);
        add1->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        add1->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        add1->add_use(LirOperand::imm(5, 8));
        entry->append_inst(std::move(add1));

        auto j1 = std::make_unique<LirInst>(LirOpcode::Jcc);
        j1->condition = x64::Condition::NE;
        j1->add_use(LirOperand::label(bit_blk->id));
        entry->append_inst(std::move(j1));
        auto jmp1 = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp1->add_use(LirOperand::label(yes_blk->id));
        entry->append_inst(std::move(jmp1));

        // Test bit 40 against far_blk (> 32 KiB away -> relaxed tbnz + b far_blk).
        auto tst = std::make_unique<LirInst>(LirOpcode::Test);
        tst->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        tst->add_use(LirOperand::imm(1ll << 40, 8));
        bit_blk->append_inst(std::move(tst));

        auto j2 = std::make_unique<LirInst>(LirOpcode::Jcc);
        j2->condition = x64::Condition::E;
        j2->add_use(LirOperand::label(far_blk->id));
        bit_blk->append_inst(std::move(j2));
        auto jmp2 = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp2->add_use(LirOperand::label(pad_blk->id));
        bit_blk->append_inst(std::move(jmp2));

        // Fill > 32 KiB (9000 adds) so tbz x0, #40 to far_blk must relax.
        for (int i = 0; i < 9000; ++i) {
            auto n = std::make_unique<LirInst>(LirOpcode::Add);
            n->add_def(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
            n->add_use(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
            n->add_use(LirOperand::imm(1, 8));
            pad_blk->append_inst(std::move(n));
        }
        auto jmp3 = std::make_unique<LirInst>(LirOpcode::Jmp);
        jmp3->add_use(LirOperand::label(far_blk->id));
        pad_blk->append_inst(std::move(jmp3));

        auto ret_y = std::make_unique<LirInst>(LirOpcode::Ret);
        ret_y->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        yes_blk->append_inst(std::move(ret_y));

        auto ret_f = std::make_unique<LirInst>(LirOpcode::Ret);
        ret_f->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        far_blk->append_inst(std::move(ret_f));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto res = em.compile();
        const auto& bytes = res.code_buffer.bytes();
        // Leaf function: entry starts at 0. cmp x0, #0 elided; add x1, x1, #5 first; cbz x0, yes_blk second.
        // cbz x0: 0xB4000000 | (imm19 << 5) | 0
        bool saw_cbz_x0 = false;
        bool saw_relaxed_tbnz40 = false;
        bool saw_cmp_x0_0 = has_u32_word(bytes, 0xF100001Fu);
        bool saw_tst_x0 = false;
        for (size_t i = 0; i + 8 <= bytes.size(); i += 4) {
            uint32_t w = read32_at(bytes, i);
            if ((w & 0xFF00001Fu) == 0xB4000000u) saw_cbz_x0 = true;
            // Relaxed tbz x0, #40 (b5=1, b40=8) -> tbnz x0, #40, +8 followed by b far_blk
            if (w == (0xB7000000u | (8u << 19) | (2u << 5) | 0u) &&
                (read32_at(bytes, i + 4) & 0xFC000000u) == 0x14000000u) {
                saw_relaxed_tbnz40 = true;
            }
            // tst x0, #imm is ands xzr, x0, #imm: 0xF200001F mask (SF=1, opc=11, Rn=0, Rd=31)
            if ((w & 0xFF8003FFu) == 0xF200001Fu) saw_tst_x0 = true;
        }
        CHECK_FALSE(saw_cmp_x0_0);
        CHECK_FALSE(saw_tst_x0);
        CHECK(saw_cbz_x0);
        CHECK(saw_relaxed_tbnz40);
    }

    // Live JIT MIR tests for cbz, cbnz, sign-bit tbz/tbnz (64 & 32-bit), bit-mask tbz/tbnz
    const char* kBranches = R"(module @m

func @br_i64_eq0(%x: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %c = eq.i64 %x, %z
  br_if %c, yes, no
yes:
  %a = iconst.i64 11
  ret %a
no:
  %b = iconst.i64 22
  ret %b
}

func @br_i32_ne0(%x: i32) -> i64 {
entry:
  %z = iconst.i32 0
  %c = ne.i32 %x, %z
  br_if %c, yes, no
yes:
  %a = iconst.i64 33
  ret %a
no:
  %b = iconst.i64 44
  ret %b
}

func @br_i64_lt0(%x: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %c = slt.i64 %x, %z
  br_if %c, yes, no
yes:
  %a = iconst.i64 55
  ret %a
no:
  %b = iconst.i64 66
  ret %b
}

func @br_i32_ge0(%x: i32) -> i64 {
entry:
  %z = iconst.i32 0
  %c = sge.i32 %x, %z
  br_if %c, yes, no
yes:
  %a = iconst.i64 77
  ret %a
no:
  %b = iconst.i64 88
  ret %b
}

func @br_bit39_nz(%x: i64) -> i64 {
entry:
  %m = iconst.i64 549755813888
  %a = and.i64 %x, %m
  %z = iconst.i64 0
  %c = ne.i64 %a, %z
  br_if %c, yes, no
yes:
  %r1 = iconst.i64 99
  ret %r1
no:
  %r2 = iconst.i64 100
  ret %r2
}
)";
    auto mod = parse_module(kBranches);
    REQUIRE(mod != nullptr);

    // Check fused encodings in each function
    auto check_word_mask = [&](const char* fname, uint32_t mask, uint32_t match) {
        auto lir = allocate_a64(*mod, fname);
        AArch64EmitContext em(*lir, Target::aarch64_macos());
        auto res = em.compile();
        const auto& bytes = res.code_buffer.bytes();
        bool found = false;
        for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
            if ((read32_at(bytes, i) & mask) == match) found = true;
        }
        CHECK(found);
    };
    check_word_mask("br_i64_eq0", 0xFE000000u, 0xB4000000u);   // cbz or cbnz 64-bit
    check_word_mask("br_i32_ne0", 0xFE000000u, 0x34000000u);   // cbz32 or cbnz32
    check_word_mask("br_i64_lt0", 0x7E000000u, 0x36000000u);   // tbz/tbnz
    check_word_mask("br_i32_ge0", 0x7E000000u, 0x36000000u);   // tbz/tbnz #31
    check_word_mask("br_bit39_nz", 0xFE000000u, 0xB6000000u);  // tbz/tbnz with b5=1

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto eq0 = jit.get_function_ptr<int64_t (*)(int64_t)>("br_i64_eq0");
    auto ne0 = jit.get_function_ptr<int64_t (*)(int32_t)>("br_i32_ne0");
    auto lt0 = jit.get_function_ptr<int64_t (*)(int64_t)>("br_i64_lt0");
    auto ge0 = jit.get_function_ptr<int64_t (*)(int32_t)>("br_i32_ge0");
    auto b39 = jit.get_function_ptr<int64_t (*)(int64_t)>("br_bit39_nz");
    REQUIRE(eq0 && ne0 && lt0 && ge0 && b39);

    CHECK_EQ(eq0(0), int64_t(11));
    CHECK_EQ(eq0(1), int64_t(22));
    CHECK_EQ(eq0(std::numeric_limits<int64_t>::min()), int64_t(22));
    CHECK_EQ(ne0(0), int64_t(44));
    CHECK_EQ(ne0(-1), int64_t(33));
    CHECK_EQ(ne0(std::numeric_limits<int32_t>::min()), int64_t(33));
    CHECK_EQ(lt0(0), int64_t(66));
    CHECK_EQ(lt0(-1), int64_t(55));
    CHECK_EQ(lt0(std::numeric_limits<int64_t>::min()), int64_t(55));
    CHECK_EQ(lt0(std::numeric_limits<int64_t>::max()), int64_t(66));
    CHECK_EQ(ge0(0), int64_t(77));
    CHECK_EQ(ge0(-1), int64_t(88));
    CHECK_EQ(ge0(std::numeric_limits<int32_t>::min()), int64_t(88));
    CHECK_EQ(b39(0), int64_t(100));
    CHECK_EQ(b39(int64_t(1) << 39), int64_t(99));
    CHECK_EQ(b39(int64_t(1) << 38), int64_t(100));
}

// 3. Madd/Msub, Shifted ALU (AddShifted/SubShifted), and single-insn Lea (add_lsl)
TEST_CASE("AArch64 Phase 3 - Madd/Msub, shifted ALU, and single-insn Lea") {
    const char* kAluMod = R"(module @m

func @madd64(%a: i64, %b: i64, %c: i64) -> i64 {
entry:
  %p = mul.i64 %a, %b
  %r = add.i64 %c, %p
  ret %r
}

func @msub32(%a: i32, %b: i32, %c: i32) -> i32 {
entry:
  %p = mul.i32 %a, %b
  %r = sub.i32 %c, %p
  ret %r
}

func @smod64(%a: i64, %b: i64) -> i64 {
entry:
  %r = smod.i64 %a, %b
  ret %r
}

func @sh_add64(%x: i64, %y: i64) -> i64 {
entry:
  %three = iconst.i64 3
  %s = shl.i64 %y, %three
  %r = add.i64 %x, %s
  ret %r
}

func @sh_sub32(%x: i32, %y: i32) -> i32 {
entry:
  %two = iconst.i32 2
  %s = ashr.i32 %y, %two
  %r = sub.i32 %x, %s
  ret %r
}

func @mul_multi_use(%a: i64, %b: i64, %c: i64) -> i64 {
entry:
  %p = mul.i64 %a, %b
  %r1 = add.i64 %c, %p
  %r2 = xor.i64 %r1, %p
  ret %r2
}
)";
    auto mod = parse_module(kAluMod);
    REQUIRE(mod != nullptr);

    // Build load_idx_i8_s8 programmatically for Lea -> add_lsl check
    {
        Function* lf = mod->create_function("load_idx_i8_s8", Type::i64(), {Type::ptr(), Type::i64()});
        Builder lb(*mod);
        lb.set_function(lf);
        BasicBlock* entry = lb.append_block("entry");
        Value* base = lb.add_block_param(entry, Type::ptr());
        Value* idx = lb.add_block_param(entry, Type::i64());
        lb.build_ret(lb.build_zext_i64(lb.build_load_indexed(Type::i8(), base, idx, 8, 5)));
        lf->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*lf));
    }

    auto lir_madd = lower_a64(*mod, "madd64");
    auto lir_msub = lower_a64(*mod, "msub32");
    auto lir_smod = lower_a64(*mod, "smod64");
    auto lir_sh_add = lower_a64(*mod, "sh_add64");
    auto lir_sh_sub = lower_a64(*mod, "sh_sub32");
    auto lir_multi = lower_a64(*mod, "mul_multi_use");
    CHECK_EQ(count_opcode(*lir_madd, LirOpcode::Madd), size_t(1));
    CHECK_EQ(count_opcode(*lir_madd, LirOpcode::Imul), size_t(0));
    CHECK_EQ(count_opcode(*lir_msub, LirOpcode::Msub32), size_t(1));
    CHECK_EQ(count_opcode(*lir_msub, LirOpcode::Imul32), size_t(0));
    CHECK_EQ(count_opcode(*lir_smod, LirOpcode::Msub), size_t(1));
    CHECK_EQ(count_opcode(*lir_sh_add, LirOpcode::AddShifted), size_t(1));
    CHECK_EQ(count_opcode(*lir_sh_add, LirOpcode::Shl), size_t(0));
    CHECK_EQ(count_opcode(*lir_sh_sub, LirOpcode::Sub32Shifted), size_t(1));
    CHECK_EQ(count_opcode(*lir_sh_sub, LirOpcode::Sar32), size_t(0));
    CHECK_EQ(count_opcode(*lir_multi, LirOpcode::Madd), size_t(0));

    // Check Lea emits single-insn add ..., lsl #3 (0x8B..0C..) without lsl x16
    auto lir_lea = allocate_a64(*mod, "load_idx_i8_s8");
    AArch64EmitContext em_lea(*lir_lea, Target::aarch64_macos());
    auto res_lea = em_lea.compile();
    bool saw_add_lsl3 = false;
    for (size_t i = 0; i + 4 <= res_lea.code_buffer.size(); i += 4) {
        uint32_t w = read32_at(res_lea.code_buffer.bytes(), i);
        if ((w & 0xFFE0FC00u) == 0x8B000C00u) saw_add_lsl3 = true;
    }
    CHECK(saw_add_lsl3);

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto madd64 = jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t)>("madd64");
    auto msub32 = jit.get_function_ptr<int32_t (*)(int32_t, int32_t, int32_t)>("msub32");
    auto smod64 = jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("smod64");
    auto sh_add64 = jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("sh_add64");
    auto sh_sub32 = jit.get_function_ptr<int32_t (*)(int32_t, int32_t)>("sh_sub32");
    auto multi = jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t)>("mul_multi_use");
    auto load_i8 = jit.get_function_ptr<uint64_t (*)(const uint8_t*, int64_t)>("load_idx_i8_s8");
    REQUIRE(madd64 && msub32 && smod64 && sh_add64 && sh_sub32 && multi && load_i8);

    CHECK_EQ(madd64(7, -9, 100), int64_t(37));
    CHECK_EQ(msub32(5, 11, 20), int32_t(-35));
    CHECK_EQ(smod64(-107, 13), int64_t(-107 % 13));
    CHECK_EQ(sh_add64(10, 9), int64_t(10 + (9 << 3)));
    CHECK_EQ(sh_sub32(100, -17), int32_t(100 - (-17 >> 2)));
    CHECK_EQ(multi(6, 7, 10), int64_t(52 ^ 42));

    uint8_t buf[64]{};
    buf[3 * 8 + 5] = 0xAB;
    CHECK_EQ(load_i8(buf, 3), uint64_t(0xAB));
}

// 4. Native integer select (Setcc/Csinv/Csinc/Csneg/Cmovcc imm(0)) + zero-uxtb + all 6 float cmps
TEST_CASE("AArch64 Phase 3 - Native integer select, zero-uxtb cmps, and single-cset float cmps") {
    const char* kSelMod = R"(module @m

func @sel_01(%a: i64, %b: i64) -> i64 {
entry:
  %c = slt.i64 %a, %b
  %o = iconst.i64 1
  %z = iconst.i64 0
  %r = select.i64 %c, %o, %z
  ret %r
}

func @sel_m1_0(%a: i64, %b: i64) -> i64 {
entry:
  %c = eq.i64 %a, %b
  %m = iconst.i64 -1
  %z = iconst.i64 0
  %r = select.i64 %c, %m, %z
  ret %r
}

func @sel_inc32(%a: i32, %b: i32, %x: i32, %t: i32) -> i32 {
entry:
  %c = eq.i32 %a, %b
  %one = iconst.i32 1
  %inc = add.i32 %x, %one
  %r = select.i32 %c, %t, %inc
  ret %r
}

func @sel_neg64(%a: i64, %b: i64, %x: i64, %t: i64) -> i64 {
entry:
  %c = sgt.i64 %a, %b
  %nx = neg.i64 %x
  %r = select.i64 %c, %t, %nx
  ret %r
}

func @sel_inv64(%a: i64, %b: i64, %x: i64, %t: i64) -> i64 {
entry:
  %c = ult.i64 %a, %b
  %ix = not.i64 %x
  %r = select.i64 %c, %t, %ix
  ret %r
}

func @sel_zero_x(%a: i64, %b: i64, %x: i64) -> i64 {
entry:
  %c = slt.i64 %a, %b
  %z = iconst.i64 0
  %r = select.i64 %c, %x, %z
  ret %r
}

func @cmp_f64_eq(%a: f64, %b: f64) -> i64 { entry: %c = eq.f64 %a, %b %r = zext.i64 %c ret %r }
func @cmp_f64_ne(%a: f64, %b: f64) -> i64 { entry: %c = ne.f64 %a, %b %r = zext.i64 %c ret %r }
func @cmp_f64_lt(%a: f64, %b: f64) -> i64 { entry: %c = slt.f64 %a, %b %r = zext.i64 %c ret %r }
func @cmp_f64_le(%a: f64, %b: f64) -> i64 { entry: %c = sle.f64 %a, %b %r = zext.i64 %c ret %r }
func @cmp_f64_gt(%a: f64, %b: f64) -> i64 { entry: %c = sgt.f64 %a, %b %r = zext.i64 %c ret %r }
func @cmp_f64_ge(%a: f64, %b: f64) -> i64 { entry: %c = sge.f64 %a, %b %r = zext.i64 %c ret %r }

func @cmp_f32_eq(%a: f32, %b: f32) -> i64 { entry: %c = eq.f32 %a, %b %r = zext.i64 %c ret %r }
func @cmp_f32_ne(%a: f32, %b: f32) -> i64 { entry: %c = ne.f32 %a, %b %r = zext.i64 %c ret %r }
func @cmp_f32_lt(%a: f32, %b: f32) -> i64 { entry: %c = slt.f32 %a, %b %r = zext.i64 %c ret %r }
func @cmp_f32_le(%a: f32, %b: f32) -> i64 { entry: %c = sle.f32 %a, %b %r = zext.i64 %c ret %r }
func @cmp_f32_gt(%a: f32, %b: f32) -> i64 { entry: %c = sgt.f32 %a, %b %r = zext.i64 %c ret %r }
func @cmp_f32_ge(%a: f32, %b: f32) -> i64 { entry: %c = sge.f32 %a, %b %r = zext.i64 %c ret %r }

func @sel_f64_lt(%a: f64, %b: f64, %t: i64, %f: i64) -> i64 {
entry:
  %c = slt.f64 %a, %b
  %r = select.i64 %c, %t, %f
  ret %r
}
)";
    auto mod = parse_module(kSelMod);
    REQUIRE(mod != nullptr);

    CHECK_EQ(count_opcode(*lower_a64(*mod, "sel_01"), LirOpcode::Setcc), size_t(1));
    CHECK_EQ(count_opcode(*lower_a64(*mod, "sel_01"), LirOpcode::Cmovcc), size_t(0));
    CHECK_EQ(count_opcode(*lower_a64(*mod, "sel_01"), LirOpcode::Movzx8), size_t(0));
    CHECK_EQ(count_opcode(*lower_a64(*mod, "sel_m1_0"), LirOpcode::Csinv), size_t(1));
    CHECK_EQ(count_opcode(*lower_a64(*mod, "sel_inc32"), LirOpcode::Csinc), size_t(1));
    CHECK_EQ(count_opcode(*lower_a64(*mod, "sel_neg64"), LirOpcode::Csneg), size_t(1));
    CHECK_EQ(count_opcode(*lower_a64(*mod, "sel_inv64"), LirOpcode::Csinv), size_t(1));

    auto lir_zx = lower_a64(*mod, "sel_zero_x");
    CHECK_EQ(count_opcode(*lir_zx, LirOpcode::Xor), size_t(0));
    CHECK_EQ(count_opcode(*lir_zx, LirOpcode::Cmovcc), size_t(1));

    for (const char* f : {"cmp_f64_eq", "cmp_f64_ne", "cmp_f64_lt", "cmp_f64_le", "cmp_f64_gt", "cmp_f64_ge"}) {
        auto lir = lower_a64(*mod, f);
        CHECK_EQ(count_opcode(*lir, LirOpcode::Setcc), size_t(1));
        CHECK_EQ(count_opcode(*lir, LirOpcode::Movzx8), size_t(0));
    }

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    CHECK_EQ(jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("sel_01")(3, 5), int64_t(1));
    CHECK_EQ(jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("sel_01")(5, 3), int64_t(0));
    CHECK_EQ(jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("sel_m1_0")(7, 7), int64_t(-1));
    CHECK_EQ(jit.get_function_ptr<int64_t (*)(int64_t, int64_t)>("sel_m1_0")(7, 8), int64_t(0));
    CHECK_EQ(jit.get_function_ptr<int32_t (*)(int32_t, int32_t, int32_t, int32_t)>("sel_inc32")(1, 1, 10, 99), int32_t(99));
    CHECK_EQ(jit.get_function_ptr<int32_t (*)(int32_t, int32_t, int32_t, int32_t)>("sel_inc32")(1, 2, 10, 99), int32_t(11));
    CHECK_EQ(jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t, int64_t)>("sel_neg64")(9, 2, 15, 42), int64_t(42));
    CHECK_EQ(jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t, int64_t)>("sel_neg64")(2, 9, 15, 42), int64_t(-15));
    CHECK_EQ(jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t, int64_t)>("sel_inv64")(1, 2, 0xF0, 7), int64_t(7));
    CHECK_EQ(jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t, int64_t)>("sel_inv64")(5, 2, 0xF0, 7), ~int64_t(0xF0));
    CHECK_EQ(jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t)>("sel_zero_x")(1, 2, 55), int64_t(55));
    CHECK_EQ(jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t)>("sel_zero_x")(2, 1, 55), int64_t(0));

    auto d_eq = jit.get_function_ptr<int64_t (*)(double, double)>("cmp_f64_eq");
    auto d_ne = jit.get_function_ptr<int64_t (*)(double, double)>("cmp_f64_ne");
    auto d_lt = jit.get_function_ptr<int64_t (*)(double, double)>("cmp_f64_lt");
    auto d_le = jit.get_function_ptr<int64_t (*)(double, double)>("cmp_f64_le");
    auto d_gt = jit.get_function_ptr<int64_t (*)(double, double)>("cmp_f64_gt");
    auto d_ge = jit.get_function_ptr<int64_t (*)(double, double)>("cmp_f64_ge");
    auto f_eq = jit.get_function_ptr<int64_t (*)(float, float)>("cmp_f32_eq");
    auto f_ne = jit.get_function_ptr<int64_t (*)(float, float)>("cmp_f32_ne");
    auto f_lt = jit.get_function_ptr<int64_t (*)(float, float)>("cmp_f32_lt");
    auto f_le = jit.get_function_ptr<int64_t (*)(float, float)>("cmp_f32_le");
    auto f_gt = jit.get_function_ptr<int64_t (*)(float, float)>("cmp_f32_gt");
    auto f_ge = jit.get_function_ptr<int64_t (*)(float, float)>("cmp_f32_ge");
    auto sel_lt = jit.get_function_ptr<int64_t (*)(double, double, int64_t, int64_t)>("sel_f64_lt");

    const double dvals[] = {-1.0, -0.0, 0.0, 1.0,
                            std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(),
                            std::numeric_limits<double>::quiet_NaN()};
    for (double a : dvals) {
        for (double b : dvals) {
            CHECK_EQ(d_eq(a, b), int64_t(a == b));
            CHECK_EQ(d_ne(a, b), int64_t(a != b));
            CHECK_EQ(d_lt(a, b), int64_t(a < b));
            CHECK_EQ(d_le(a, b), int64_t(a <= b));
            CHECK_EQ(d_gt(a, b), int64_t(a > b));
            CHECK_EQ(d_ge(a, b), int64_t(a >= b));
            CHECK_EQ(sel_lt(a, b, 111, 222), (a < b) ? int64_t(111) : int64_t(222));
            float fa = static_cast<float>(a), fb = static_cast<float>(b);
            CHECK_EQ(f_eq(fa, fb), int64_t(fa == fb));
            CHECK_EQ(f_ne(fa, fb), int64_t(fa != fb));
            CHECK_EQ(f_lt(fa, fb), int64_t(fa < fb));
            CHECK_EQ(f_le(fa, fb), int64_t(fa <= fb));
            CHECK_EQ(f_gt(fa, fb), int64_t(fa > fb));
            CHECK_EQ(f_ge(fa, fb), int64_t(fa >= fb));
        }
    }
}

// 5. Indexed EA cache, fold_reserved_register_copies (X28/SP), name_tagged_sources, stp xzr, xzr
TEST_CASE("AArch64 Phase 3 - EA cache, X28/SP copy folding, tagged-source aliasing, stp xzr") {
    // 5a. Indexed EA cache across 4 loads with adjacent displacements + JIT check
    {
        Module mod("ea_cache");
        Function* fn = mod.create_function("sum4", Type::i64(), {Type::ptr(), Type::i64()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* base = b.add_block_param(entry, Type::ptr());
        Value* idx = b.add_block_param(entry, Type::i64());
        Value* zero = b.build_iconst_i64(0);
        Value* one = b.build_iconst_i64(1);
        Value* two = b.build_iconst_i64(2);
        Value* three = b.build_iconst_i64(3);
        Value* l0 = b.build_load_indexed(Type::i64(), base, b.build_add(idx, zero), 8, 8);
        Value* l1 = b.build_load_indexed(Type::i64(), base, b.build_add(idx, one), 8, 8);
        Value* l2 = b.build_load_indexed(Type::i64(), base, b.build_add(idx, two), 8, 8);
        Value* l3 = b.build_load_indexed(Type::i64(), base, b.build_add(idx, three), 8, 8);
        b.build_ret(b.build_add(b.build_add(l0, l1), b.build_add(l2, l3)));
        fn->rebuild_cfg_predecessors();

        auto lir = lower_a64(mod, "sum4");
        // All 4 share a single Lea [base + idx*8] with disp 8/16/24/32 folded into Mov!
        CHECK_EQ(count_opcode(*lir, LirOpcode::Lea), size_t(1));

        JitExecutionEngine jit;
        REQUIRE(jit.compile_and_load(mod));
        auto sum4 = jit.get_function_ptr<int64_t (*)(const int64_t*, int64_t)>("sum4");
        REQUIRE(sum4 != nullptr);
        int64_t arr[10] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100};
        // Offset 8 bytes + (idx + k)*8 -> elements at arr[1 + idx + k]
        CHECK_EQ(sum4(arr, 2), int64_t(40 + 50 + 60 + 70));
    }

    // 5b. fold_reserved_register_copies for pinned X28 + SP stack check
    {
        Module mod("sp_x28_fold");
        mod.set_pinned_tls_register(true);
        Function* fn = mod.create_function("sp_check", Type::i64(), {});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        b.position_at_end(entry);
        Value* tls = b.build_pinned_tls_read();
        Value* limit = b.build_load(Type::i64(), tls, 0x38);
        Value* sp = b.build_read_sp();
        Value* below = b.build_ult(sp, limit);
        BasicBlock* yes = b.append_block("yes");
        BasicBlock* no = b.append_block("no");
        b.position_at_end(entry);
        b.build_br_if(below, yes, no);
        b.position_at_end(yes);
        b.build_ret(b.build_iconst_i64(1));
        b.position_at_end(no);
        b.build_ret(b.build_iconst_i64(0));
        fn->rebuild_cfg_predecessors();

        auto lir = lower_a64(mod, "sp_check");
        const PReg sp_reg = PReg::aarch64_gpr(GPR::SP);
        const PReg x28_reg = PReg::aarch64_gpr(GPR::X28);
        size_t reserved_copies = 0;
        bool saw_x28_mem = false;
        bool saw_cmp_sp = false;
        for (const auto& blk : lir->blocks) {
            for (const auto& inst : blk->instructions) {
                if (inst->opcode == LirOpcode::Mov && inst->uses.size() == 1 && inst->uses[0].is_preg() &&
                    (inst->uses[0].preg_val == sp_reg || inst->uses[0].preg_val == x28_reg)) {
                    ++reserved_copies;
                }
                if (inst->opcode == LirOpcode::Mov && inst->uses.size() == 1 && inst->uses[0].is_mem() &&
                    inst->uses[0].mem_val.base_preg == x28_reg && inst->uses[0].mem_val.disp == 0x38) {
                    saw_x28_mem = true;
                }
                if (inst->opcode == LirOpcode::Cmp && !inst->uses.empty() &&
                    inst->uses[0].is_preg() && inst->uses[0].preg_val == sp_reg) {
                    saw_cmp_sp = true;
                }
            }
        }
        CHECK_EQ(reserved_copies, size_t(0));
        CHECK(saw_x28_mem);
        CHECK(saw_cmp_sp);
    }

    // 5c. name_tagged_sources eliminates tagged Mov copy when reading raw join bits + stp xzr, xzr
    {
        const char* kTaggedZero = R"(module @m
func @tag_join(%c: i64, %a: i64, %b: i64) -> i64 {
entry:
  %z = iconst.i64 0
  %t = ne.i64 %c, %z
  br_if %t, left, right
left:
  br join(%a)
right:
  br join(%b)
join(%x: i64):
  %tv = bitcast.tagged.i64 %x
  %k = iconst.i64 42
  %lt = ult.i64 %x, %k
  br_if %lt, fast, slow
fast:
  %bits = bitcast.i64.tagged %tv
  ret %bits
slow:
  %one = iconst.i64 1
  %bits2 = bitcast.i64.tagged %tv
  %r = add.i64 %bits2, %one
  ret %r
}

func @zero_allocas(%v: tagged) -> tagged {
entry:
  %a0 = alloca.tagged 8, 8
  %a1 = alloca.tagged 8, 8
  %a2 = alloca.tagged 8, 8
  %a3 = alloca.tagged 8, 8
  store.tagged %a0, 0, %v
  store.tagged %a1, 0, %v
  store.tagged %a2, 0, %v
  store.tagged %a3, 0, %v
  %r0 = load.tagged %a3, 0
  ret %r0
}
)";
        auto mod = parse_module(kTaggedZero);
        REQUIRE(mod != nullptr);
        auto lir_tag = lower_a64(*mod, "tag_join");
        size_t join_tagged_copies = 0;
        for (const auto& blk : lir_tag->blocks) {
            if (blk->name != "join") continue;
            for (const auto& inst : blk->instructions) {
                if (inst->opcode == LirOpcode::Mov && inst->defs.size() == 1 && inst->defs[0].is_vreg() &&
                    inst->defs[0].vreg_val.is_tagged) {
                    ++join_tagged_copies;
                }
            }
        }
        CHECK_EQ(join_tagged_copies, size_t(0));

        auto lir_zero = allocate_a64(*mod, "zero_allocas");
        AArch64EmitContext em(*lir_zero, Target::aarch64_macos());
        auto res = em.compile();
        size_t stp_xzr_count = 0;
        const auto& bytes = res.code_buffer.bytes();
        for (size_t i = 0; i + 4 <= bytes.size(); i += 4) {
            if ((read32_at(bytes, i) & 0xFFC07FFFu) == 0xA9007FBFu) ++stp_xzr_count;
        }
        CHECK_EQ(stp_xzr_count, size_t(2));
    }
}

// 6. BranchThreading, loop-only alignment, & callee-saved V8..V15 across calls
TEST_CASE("AArch64 Phase 3 - BranchThreading, loop alignment, and scalar V8..V15 call spanning") {
    // 6a. Straight fallthrough emits 0 NOPs; empty trampoline is skipped; loop backedge head is 16-byte aligned
    {
        LirFunction fn;
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* entry = fn.create_block("entry");
        LirBlock* mid = fn.create_block("mid_trampoline");
        LirBlock* loop = fn.create_block("loop");
        LirBlock* exit_b = fn.create_block("exit_b");

        // Odd number of insns in entry + fallthrough to skipped trampoline, then loop head.
        auto m0 = std::make_unique<LirInst>(LirOpcode::Mov);
        m0->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        m0->add_use(LirOperand::imm(1, 8));
        entry->append_inst(std::move(m0));
        auto je = std::make_unique<LirInst>(LirOpcode::Jmp);
        je->add_use(LirOperand::label(mid->id));
        entry->append_inst(std::move(je));

        auto jm = std::make_unique<LirInst>(LirOpcode::Jmp);
        jm->add_use(LirOperand::label(loop->id));
        mid->append_inst(std::move(jm));

        auto cmp = std::make_unique<LirInst>(LirOpcode::Cmp);
        cmp->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        cmp->add_use(LirOperand::imm(0, 8));
        loop->append_inst(std::move(cmp));
        auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc->condition = x64::Condition::E;
        jcc->add_use(LirOperand::label(exit_b->id));
        loop->append_inst(std::move(jcc));
        auto jback = std::make_unique<LirInst>(LirOpcode::Jmp);
        jback->add_use(LirOperand::label(loop->id));
        loop->append_inst(std::move(jback));

        auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
        ret->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        exit_b->append_inst(std::move(ret));

        BranchThreading bt(fn);
        CHECK(bt.skipped(1));
        CHECK_EQ(bt.target(mid->id), loop->id);
        CHECK(bt.loop_head(2));
        CHECK_FALSE(bt.loop_head(3));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto res = em.compile();
        CHECK_EQ(res.block_offsets[loop->id] % 16, size_t(0));

        // Ensure straight non-loop fallthrough exit_b after a non-falling block has no extra nop before it
        // Specifically verify straight fallthrough blocks produce 0 NOPs in a pure chain:
        LirFunction straight;
        straight.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* b0 = straight.create_block("b0");
        LirBlock* b1 = straight.create_block("b1");
        LirBlock* b2 = straight.create_block("b2");
        for (LirBlock* b : {b0, b1}) {
            auto a = std::make_unique<LirInst>(LirOpcode::Add);
            a->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
            a->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
            a->add_use(LirOperand::imm(1, 8));
            b->append_inst(std::move(a));
            auto j = std::make_unique<LirInst>(LirOpcode::Jmp);
            j->add_use(LirOperand::label(b == b0 ? b1->id : b2->id));
            b->append_inst(std::move(j));
        }
        auto r = std::make_unique<LirInst>(LirOpcode::Ret);
        r->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        b2->append_inst(std::move(r));
        AArch64EmitContext em_s(straight, Target::aarch64_macos());
        auto res_s = em_s.compile();
        CHECK_FALSE(has_u32_word(res_s.code_buffer.bytes(), 0xD503201Fu));
        CHECK_EQ(res_s.block_offsets[b1->id], size_t(4));
        CHECK_EQ(res_s.block_offsets[b2->id], size_t(8));
    }

    // 6b. Scalar f64 call-spanning value in loop uses callee-saved V8..V15 (0 spills);
    //     128-bit vector call-spanning value is NOT assigned to V8..V15.
    {
        Module mod("fp_callee_saved");
        Function* ping = mod.create_function("ping", Type::i64(), {Type::i64()});
        {
            Builder b(mod);
            b.set_function(ping);
            BasicBlock* e = b.append_block("entry");
            Value* x = b.add_block_param(e, Type::i64());
            b.build_ret(b.build_add(x, b.build_iconst_i64(1)));
        }

        // Scalar f64 live in a loop across calls
        Function* f_loop = mod.create_function("f64_call_loop", Type::f64(), {Type::i64(), Type::f64()});
        {
            Builder b(mod);
            b.set_function(f_loop);
            BasicBlock* entry = b.append_block("entry");
            Value* n = b.add_block_param(entry, Type::i64());
            Value* s0 = b.add_block_param(entry, Type::f64());
            BasicBlock* loop = b.append_block("loop");
            Value* i = b.add_block_param(loop, Type::i64());
            Value* acc = b.add_block_param(loop, Type::f64());
            BasicBlock* exit_b = b.append_block("exit");
            Value* res_p = b.add_block_param(exit_b, Type::f64());

            b.position_at_end(entry);
            b.build_br(loop, {b.build_iconst_i64(0), b.build_fconst_f64(0.0)});

            b.position_at_end(loop);
            Value* i_next = b.build_call("ping", Type::i64(), {i});
            Value* acc2 = b.build_fadd(acc, s0);  // s0 and acc both live across call in loop!
            b.build_br_if(b.build_slt(i_next, n), loop, {i_next, acc2}, exit_b, {acc2});

            b.position_at_end(exit_b);
            b.build_ret(res_p);
        }

        // 128-bit vector live across a call must NOT stay in V8..V15
        Function* v_call = mod.create_function("v128_call", Type::f64(), {Type::f64(), Type::i64()});
        {
            Builder b(mod);
            b.set_function(v_call);
            BasicBlock* entry = b.append_block("entry");
            Value* s = b.add_block_param(entry, Type::f64());
            Value* k = b.add_block_param(entry, Type::i64());
            Value* vec = b.build_vbroadcast(Type::f64x2(), s);
            (void)b.build_call("ping", Type::i64(), {k});
            Value* vec2 = b.build_vadd(vec, vec);
            Value* l0 = b.build_vextract_lane(vec2, 0);
            Value* l1 = b.build_vextract_lane(vec2, 1);
            b.build_ret(b.build_fadd(l0, l1));
        }

        auto lir_f = allocate_a64(mod, "f64_call_loop");
        CHECK_EQ(lir_f->frame.num_spill_slots, uint32_t(0));
        bool saw_v8_to_v15 = false;
        for (const auto& blk : lir_f->blocks) {
            for (const auto& inst : blk->instructions) {
                for (const auto& d : inst->defs) {
                    if (d.is_preg() && d.preg_val.is_xmm()) {
                        uint8_t c = static_cast<uint8_t>(reg_code(d.preg_val.as_aarch64_fpr()));
                        if (c >= 8 && c <= 15) saw_v8_to_v15 = true;
                    }
                }
            }
        }
        CHECK(saw_v8_to_v15);

        auto lir_v = allocate_a64(mod, "v128_call");
        bool saw_q_in_v8_to_v15 = false;
        for (const auto& blk : lir_v->blocks) {
            for (const auto& inst : blk->instructions) {
                auto chk = [&](const LirOperand& op) {
                    if (op.is_preg() && op.preg_val.is_xmm() && op.size == 16) {
                        uint8_t c = static_cast<uint8_t>(reg_code(op.preg_val.as_aarch64_fpr()));
                        if (c >= 8 && c <= 15) saw_q_in_v8_to_v15 = true;
                    }
                };
                for (const auto& d : inst->defs) chk(d);
                for (const auto& u : inst->uses) chk(u);
            }
        }
        CHECK_FALSE(saw_q_in_v8_to_v15);

        JitExecutionEngine jit;
        REQUIRE(jit.compile_and_load(mod));
        auto fl = jit.get_function_ptr<double (*)(int64_t, double)>("f64_call_loop");
        auto vc = jit.get_function_ptr<double (*)(double, int64_t)>("v128_call");
        REQUIRE(fl && vc);
        CHECK_EQ(fl(5, 2.5), 12.5);
        CHECK_EQ(vc(3.25, 7), 13.0);
    }
}
