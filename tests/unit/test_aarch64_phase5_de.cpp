#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/aarch64_frame.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

using namespace brass;
using namespace brass::codegen;
using namespace brass::aarch64;

namespace {

std::vector<uint32_t> extract_words(const std::vector<uint8_t>& bytes) {
    REQUIRE(bytes.size() % 4 == 0);
    std::vector<uint32_t> words(bytes.size() / 4);
    std::memcpy(words.data(), bytes.data(), bytes.size());
    return words;
}

std::unique_ptr<LirInst> make_mem_load(LirOpcode op, PReg reg, PReg base, int32_t disp, uint8_t sz) {
    auto inst = std::make_unique<LirInst>(op);
    inst->add_def(LirOperand::preg(reg, sz));
    inst->add_use(LirOperand::mem(base, disp, sz));
    return inst;
}

std::unique_ptr<LirInst> make_mem_store(LirOpcode op, PReg base, int32_t disp, LirOperand src, uint8_t sz) {
    auto inst = std::make_unique<LirInst>(op);
    inst->add_def(LirOperand::mem(base, disp, sz));
    inst->add_use(std::move(src));
    return inst;
}

} // namespace

// 5D.1: Encoder ldp_s/stp_s/ldp_q/stp_q exact bitwords verified against Apple clang
TEST_CASE("AArch64 Phase 5D - Encoder ldp_s/stp_s/ldp_q/stp_q encodings") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);

    enc.ldp_s(FPR::V0, FPR::V1, ptr(GPR::FP, -16));
    enc.stp_s(FPR::V2, FPR::V3, ptr(GPR::FP, 24));
    enc.ldp_q(FPR::V0, FPR::V1, ptr(GPR::FP, -32));
    enc.stp_q(FPR::V2, FPR::V3, ptr(GPR::FP, 32));

    auto words = extract_words(buf.bytes());
    REQUIRE(words.size() == 4u);
    CHECK_EQ(words[0], 0x2D7E07A0u); // ldp s0, s1, [x29, #-16]
    CHECK_EQ(words[1], 0x2D030FA2u); // stp s2, s3, [x29, #24]
    CHECK_EQ(words[2], 0xAD7F07A0u); // ldp q0, q1, [x29, #-32]
    CHECK_EQ(words[3], 0xAD010FA2u); // stp q2, q3, [x29, #32]
}

// 5D.2: Emit-time ldp/stp pairing positive cases and strict safety exclusions
TEST_CASE("AArch64 Phase 5D - Emit-time ldp/stp pairing invariants and JIT correctness") {
    // 1. Positive raw LIR pairing across GPR64, GPR32, zero-stores, FPR64, FPR32, and small-window reordering
    {
        LirFunction fn;
        fn.name = "p5d_positive";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");
        const PReg x0 = PReg::aarch64_gpr(GPR::X0);
        const PReg x1 = PReg::aarch64_gpr(GPR::X1);
        const PReg x2 = PReg::aarch64_gpr(GPR::X2);
        const PReg x3 = PReg::aarch64_gpr(GPR::X3);
        const PReg x4 = PReg::aarch64_gpr(GPR::X4);
        const PReg x5 = PReg::aarch64_gpr(GPR::X5);
        const PReg d0 = PReg::aarch64_fpr(FPR::V0);
        const PReg d1 = PReg::aarch64_fpr(FPR::V1);
        const PReg s2 = PReg::aarch64_fpr(FPR::V2);
        const PReg s3 = PReg::aarch64_fpr(FPR::V3);

        // GPR 64-bit swapped-offset load, paired as ldp x2, x1, [x0, #0]
        bb->append_inst(make_mem_load(LirOpcode::Mov, x1, x0, 8, 8));
        bb->append_inst(make_mem_load(LirOpcode::Mov, x2, x0, 0, 8));
        // GPR 32-bit adjacent load + store
        bb->append_inst(make_mem_load(LirOpcode::Mov32, x3, x0, 16, 4));
        bb->append_inst(make_mem_load(LirOpcode::Mov32, x4, x0, 20, 4));
        bb->append_inst(make_mem_store(LirOpcode::Mov32, x0, 24, LirOperand::preg(x3, 4), 4));
        bb->append_inst(make_mem_store(LirOpcode::Mov32, x0, 28, LirOperand::imm(0, 4), 4));
        // GPR 64-bit immediate-zero + reg store (swapped order)
        bb->append_inst(make_mem_store(LirOpcode::Mov, x0, 40, LirOperand::imm(0, 8), 8));
        bb->append_inst(make_mem_store(LirOpcode::Mov, x0, 32, LirOperand::preg(x1, 8), 8));
        // Small-window non-adjacent FPR 64-bit load (d0 at 48, d1 at 56 separated by unrelated x5 at 64)
        bb->append_inst(make_mem_load(LirOpcode::Movsd, d0, x0, 48, 8));
        bb->append_inst(make_mem_load(LirOpcode::Mov, x5, x0, 64, 8));
        bb->append_inst(make_mem_load(LirOpcode::Movsd, d1, x0, 56, 8));
        // Adjacent FPR 32-bit stores
        bb->append_inst(make_mem_store(LirOpcode::Movss, x0, 72, LirOperand::preg(s2, 4), 4));
        bb->append_inst(make_mem_store(LirOpcode::Movss, x0, 76, LirOperand::preg(s3, 4), 4));
        // Second load overwrites base x0 -> legal! ldp x2, x0, [x0, #0]
        bb->append_inst(make_mem_load(LirOpcode::Mov, x2, x0, 8, 8));
        bb->append_inst(make_mem_load(LirOpcode::Mov, x0, x0, 0, 8));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto words = extract_words(em.compile().code_buffer.bytes());
        int ldp64 = 0, ldp32 = 0, stp32 = 0, stp64 = 0, ldp_d = 0, stp_s = 0;
        for (uint32_t w : words) {
            if ((w & 0xFFC00000u) == 0xA9400000u) ++ldp64;
            if ((w & 0xFFC00000u) == 0x29400000u) ++ldp32;
            if ((w & 0xFFC00000u) == 0x29000000u) ++stp32;
            if ((w & 0xFFC00000u) == 0xA9000000u) ++stp64;
            if ((w & 0xFFC00000u) == 0x6D400000u) ++ldp_d;
            if ((w & 0xFFC00000u) == 0x2D000000u) ++stp_s;
        }
        CHECK_EQ(ldp64, 2);
        CHECK_EQ(ldp32, 1);
        CHECK_EQ(stp32, 1);
        CHECK_EQ(stp64, 1);
        CHECK_EQ(ldp_d, 1);
        CHECK_EQ(stp_s, 1);
    }

    // 2. Negative safety invariants: MUST NOT emit any signed-offset ldp/stp
    {
        LirFunction fn;
        fn.name = "p5d_negative";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        fn.frame.num_spill_slots = 4;
        fn.frame.constant_slots.emplace_back(0, 0x123456789ABCDEF0LL);
        LirBlock* bb = fn.create_block("entry");
        const PReg x0 = PReg::aarch64_gpr(GPR::X0);
        const PReg x1 = PReg::aarch64_gpr(GPR::X1);
        const PReg x2 = PReg::aarch64_gpr(GPR::X2);
        const PReg x3 = PReg::aarch64_gpr(GPR::X3);

        // rt1 == rt2 (UNPREDICTABLE / SIGILL on ARM64)
        bb->append_inst(make_mem_load(LirOpcode::Mov, x2, x1, 0, 8));
        bb->append_inst(make_mem_load(LirOpcode::Mov, x2, x1, 8, 8));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Nop));
        // First load overwrites base x1
        bb->append_inst(make_mem_load(LirOpcode::Mov, x1, x1, 0, 8));
        bb->append_inst(make_mem_load(LirOpcode::Mov, x2, x1, 8, 8));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Nop));
        // Constant-pooled spill slot adjacent to normal spill slot
        auto cs = std::make_unique<LirInst>(LirOpcode::Mov);
        cs->add_def(LirOperand::preg(x2, 8));
        cs->add_use(LirOperand::slot(0, 8));
        bb->append_inst(std::move(cs));
        auto ns = std::make_unique<LirInst>(LirOpcode::Mov);
        ns->add_def(LirOperand::preg(x3, 8));
        ns->add_use(LirOperand::slot(1, 8));
        bb->append_inst(std::move(ns));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Nop));
        // Unaligned offset (4 for 8B) and out-of-range offset (512 for 8B)
        bb->append_inst(make_mem_load(LirOpcode::Mov, x2, x0, 4, 8));
        bb->append_inst(make_mem_load(LirOpcode::Mov, x3, x0, 12, 8));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Nop));
        bb->append_inst(make_mem_load(LirOpcode::Mov, x2, x0, 512, 8));
        bb->append_inst(make_mem_load(LirOpcode::Mov, x3, x0, 520, 8));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Nop));
        // Intervening store to different base x1 blocks store reordering on x0
        bb->append_inst(make_mem_store(LirOpcode::Mov, x0, 0, LirOperand::preg(x2, 8), 8));
        bb->append_inst(make_mem_store(LirOpcode::Mov, x1, 8, LirOperand::preg(x2, 8), 8));
        bb->append_inst(make_mem_store(LirOpcode::Mov, x0, 8, LirOperand::preg(x3, 8), 8));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto words = extract_words(em.compile().code_buffer.bytes());
        for (uint32_t w : words) {
            CHECK((w & 0xFFC00000u) != 0xA9400000u); // no ldp x
            CHECK((w & 0xFFC00000u) != 0xA9000000u); // no stp x
        }
    }

    // 3. GuardExit consecutive state values pair into stp
    {
        LirFunction fn;
        fn.name = "p5d_guard_stp";
        fn.return_type = Type::i64();
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");
        auto g = std::make_unique<LirInst>(LirOpcode::GuardExit);
        g->condition = x64::Condition::NE;
        g->resume_id = 7;
        g->add_use(LirOperand::preg(PReg::aarch64_gpr(GPR::X0), 8));
        g->add_use(LirOperand::imm(0, 8));
        g->add_use(LirOperand::preg(PReg::aarch64_fpr(FPR::V0), 8));
        g->add_use(LirOperand::preg(PReg::aarch64_fpr(FPR::V1), 8));
        g->deopt_kinds.assign(4, uint8_t{0});
        bb->append_inst(std::move(g));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto words = extract_words(em.compile().code_buffer.bytes());
        bool saw_stp_gpr = false, saw_stp_fpr = false;
        for (uint32_t w : words) {
            if ((w & 0xFFC00000u) == 0xA9000000u) saw_stp_gpr = true;
            if ((w & 0xFFC00000u) == 0x6D000000u) saw_stp_fpr = true;
        }
        CHECK(saw_stp_gpr);
        CHECK(saw_stp_fpr);
    }

    // 4. MIR-to-JIT end-to-end adjacent field read/write and f64 pair swap
    auto mod = std::make_unique<Module>("p5d_jit");
    {
        Function* fn = mod->create_function("pair_fields", Type::i64(), {Type::ptr(), Type::ptr()});
        Builder b(*mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* src = b.add_block_param(bb, Type::ptr());
        Value* dst = b.add_block_param(bb, Type::ptr());
        Value* a0 = b.build_load(Type::i64(), src, 0);
        Value* a1 = b.build_load(Type::i64(), src, 8);
        Value* f0 = b.build_load(Type::f64(), src, 16);
        Value* f1 = b.build_load(Type::f64(), src, 24);
        b.build_store(Type::i64(), dst, 0, b.build_add(a0, b.build_iconst_i64(1)));
        b.build_store(Type::i64(), dst, 8, b.build_add(a1, b.build_iconst_i64(2)));
        b.build_store(Type::f64(), dst, 16, b.build_fadd(f0, b.build_fconst_f64(0.5)));
        b.build_store(Type::f64(), dst, 24, b.build_fadd(f1, b.build_fconst_f64(1.5)));
        b.build_store(Type::i64(), dst, 32, b.build_iconst_i64(0));
        b.build_store(Type::i64(), dst, 40, b.build_iconst_i64(0));
        b.build_ret(b.build_add(a0, a1));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto j_pair = jit.get_function_ptr<int64_t (*)(const void*, void*)>("pair_fields");
    struct Buf { int64_t i0, i1; double d0, d1; int64_t z0, z1; };
    const Buf src{10, 25, 2.0, -4.0, 99, 88};
    Buf dst{0, 0, 0.0, 0.0, 123, 456};
    CHECK_EQ(j_pair(&src, &dst), int64_t(35));
    CHECK_EQ(dst.i0, int64_t(11));
    CHECK_EQ(dst.i1, int64_t(27));
    CHECK_EQ(dst.d0, 2.5);
    CHECK_EQ(dst.d1, -2.5);
    CHECK_EQ(dst.z0, int64_t(0));
    CHECK_EQ(dst.z1, int64_t(0));
}

// 5E.1: 24-bit and shifted 12-bit immediates in Add/Sub/Adds/Subs/Cmp/ensure_accessible_mem/Lea
TEST_CASE("AArch64 Phase 5E - 24-bit and shifted 12-bit immediates") {
    // 1. Verify raw-LIR emission encodes without movz/movn into X16 for 24-bit Add/Sub, shifted Cmp/Adds/Subs, and +0x8290/-0x1230 mem
    {
        LirFunction fn;
        fn.name = "p5e_imm24";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");
        const PReg x0 = PReg::aarch64_gpr(GPR::X0);
        const PReg x1 = PReg::aarch64_gpr(GPR::X1);

        auto add_imm = [&](LirOpcode op, uint8_t sz, int64_t imm, bool has_def = true) {
            auto ins = std::make_unique<LirInst>(op);
            if (has_def) ins->add_def(LirOperand::preg(x0, sz));
            ins->add_use(LirOperand::preg(x0, sz));
            ins->add_use(LirOperand::imm(imm, sz));
            bb->append_inst(std::move(ins));
        };
        add_imm(LirOpcode::Add, 8, 0x123456);
        add_imm(LirOpcode::Sub, 8, -0x123456);
        add_imm(LirOpcode::Add32, 4, 0xFFFFFF);
        add_imm(LirOpcode::Sub32, 4, -0x00FFF000);
        add_imm(LirOpcode::Adds, 8, 0x123000);
        add_imm(LirOpcode::Subs, 8, -0x123000);
        add_imm(LirOpcode::Cmp, 8, 0x123000, false);
        add_imm(LirOpcode::Cmp, 8, -0x123000, false);
        add_imm(LirOpcode::Cmp32, 4, 0x123000, false);
        add_imm(LirOpcode::Cmp32, 4, -0x123000, false);

        // Large-offset loads: +0x8290 (8-byte aligned) and -0x1230 (16-byte aligned) use 1 shifted add/sub + folded ldr
        bb->append_inst(make_mem_load(LirOpcode::Mov, x0, x1, 0x8290, 8));
        bb->append_inst(make_mem_load(LirOpcode::Mov, x0, x1, -0x1230, 8));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto words = extract_words(em.compile().code_buffer.bytes());
        bool saw_cmp_sh = false, saw_cmn_sh = false;
        bool saw_mem_add_sh = false, saw_mem_sub_sh = false;
        int movz_x16 = 0;
        for (uint32_t w : words) {
            if (w == 0xF1448C1Fu) saw_cmp_sh = true;     // cmp x0, #0x123, lsl #12
            if (w == 0xB1448C1Fu) saw_cmn_sh = true;     // cmn x0, #0x123, lsl #12
            if (w == 0x91402030u) saw_mem_add_sh = true; // add x16, x1, #0x8, lsl #12
            if (w == 0xD1400830u) saw_mem_sub_sh = true; // sub x16, x1, #0x2, lsl #12
            if (((w >> 23) & 0x3Fu) == 0b100101u && (w & 0x1Fu) == 16u) ++movz_x16;
        }
        CHECK(saw_cmp_sh);
        CHECK(saw_cmn_sh);
        CHECK(saw_mem_add_sh);
        CHECK(saw_mem_sub_sh);
        CHECK_EQ(movz_x16, 0);
    }

    // 1b. Out-of-range edge: INT64_MIN / INT32_MIN safely fall back without signed negation UB
    {
        LirFunction fn;
        fn.name = "p5e_imm_edge";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");
        const PReg x0 = PReg::aarch64_gpr(GPR::X0);
        auto add_imm = [&](LirOpcode op, uint8_t sz, int64_t imm, bool has_def) {
            auto ins = std::make_unique<LirInst>(op);
            if (has_def) ins->add_def(LirOperand::preg(x0, sz));
            ins->add_use(LirOperand::preg(x0, sz));
            ins->add_use(LirOperand::imm(imm, sz));
            bb->append_inst(std::move(ins));
        };
        add_imm(LirOpcode::Add, 8, std::numeric_limits<int64_t>::min(), true);
        add_imm(LirOpcode::Sub, 8, std::numeric_limits<int64_t>::min(), true);
        add_imm(LirOpcode::Cmp32, 4, std::numeric_limits<int32_t>::min(), false);
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
        AArch64EmitContext em(fn, Target::aarch64_macos());
        CHECK(!em.compile().code_buffer.bytes().empty());
    }

    // 2. JIT end-to-end verification of 24-bit arithmetic & shifted-12-bit NZCV comparisons
    auto mod = std::make_unique<Module>("p5e_jit_imm");
    {
        Function* fn = mod->create_function("imm24_ops", Type::i64(), {Type::i64()});
        Builder b(*mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::i64());
        Value* a = b.build_add(x, b.build_iconst_i64(0x123456));
        Value* s = b.build_sub(a, b.build_iconst_i64(0x23456));
        Value* lt = b.build_select(b.build_slt(x, b.build_iconst_i64(-0x123000)),
                                   b.build_iconst_i64(1), b.build_iconst_i64(0));
        Value* ult = b.build_select(b.build_ult(x, b.build_iconst_i64(0x123000)),
                                    b.build_iconst_i64(2), b.build_iconst_i64(0));
        Value* eq = b.build_select(b.build_eq(x, b.build_iconst_i64(0x123000)),
                                   b.build_iconst_i64(4), b.build_iconst_i64(0));
        b.build_ret(b.build_add(s, b.build_add(lt, b.build_add(ult, eq))));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
    }

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto j_ops = jit.get_function_ptr<int64_t (*)(int64_t)>("imm24_ops");
    CHECK_EQ(j_ops(0x123000), int64_t(0x123000 + 0x100000 + 4));
    CHECK_EQ(j_ops(0x122FFF), int64_t(0x122FFF + 0x100000 + 2));
    CHECK_EQ(j_ops(-0x123001), int64_t(-0x123001 + 0x100000 + 1));
    CHECK_EQ(j_ops(-0x123000), int64_t(-0x123000 + 0x100000));
}

// 5E.2: Extending-load fusion (ldrsb/ldrsh/ldrsw and single-inst ldrb/ldrh/ldr w) with memory-order safety
TEST_CASE("AArch64 Phase 5E - Extending-load fusion and store hazard blocking") {
    auto mod = std::make_unique<Module>("p5e_ext_loads");

    auto make_load_ext = [&](const char* name, Type load_ty, bool is_signed, bool indexed) {
        std::vector<Type> params = indexed ? std::vector<Type>{Type::ptr(), Type::i64()}
                                           : std::vector<Type>{Type::ptr()};
        Function* fn = mod->create_function(name, Type::i64(), params);
        Builder b(*mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* p = b.add_block_param(bb, Type::ptr());
        Value* ld = nullptr;
        if (indexed) {
            Value* idx = b.add_block_param(bb, Type::i64());
            uint8_t sc = static_cast<uint8_t>(load_ty.size_in_bytes());
            ld = b.build_load_indexed(load_ty, p, idx, sc, 0);
        } else {
            ld = b.build_load(load_ty, p, 8);
        }
        Value* ext = is_signed ? b.build_sext_i64(ld) : b.build_zext_i64(ld);
        b.build_ret(ext);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        return fn;
    };

    Function* fn_sw = make_load_ext("sw", Type::i32(), true, false);
    Function* fn_sw_idx = make_load_ext("sw_idx", Type::i32(), true, true);
    Function* fn_ub = make_load_ext("ub", Type::i8(), false, true);
    Function* fn_ub2 = make_load_ext("ub2", Type::i8(), false, false);
    Function* fn_uw = make_load_ext("uw", Type::i32(), false, true);

    // Intervening store to same address must BLOCK fusion so pre-store byte is returned
    Function* fn_hazard = mod->create_function("ext_load_store_hazard", Type::i64(), {Type::ptr()});
    {
        Builder b(*mod);
        b.set_function(fn_hazard);
        BasicBlock* bb = b.append_block("entry");
        Value* p = b.add_block_param(bb, Type::ptr());
        Value* pre = b.build_load(Type::i8(), p, 0);
        b.build_store(Type::i8(), p, 0, b.build_trunc_i8(b.build_iconst_i64(0x11)));
        b.build_ret(b.build_zext_i64(pre));
        fn_hazard->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn_hazard));
    }

    // LIR verification: fused cases emit single Movsxd/Movzx8/Mov32 reading Mem without separate extend
    AArch64ISel isel(Target::aarch64_macos(), CallingConvention::apple_aapcs64());
    auto check_single_fused = [&](Function* f, LirOpcode expected_op) {
        auto lir = isel.lower(*f);
        REQUIRE(lir != nullptr);
        int ext_count = 0;
        bool saw_mem_use = false;
        for (const auto& blk : lir->blocks) {
            for (const auto& inst : blk->instructions) {
                if (inst->opcode == expected_op) {
                    ++ext_count;
                    if (!inst->uses.empty() && inst->uses[0].is_mem()) saw_mem_use = true;
                }
            }
        }
        CHECK_EQ(ext_count, 1);
        CHECK(saw_mem_use);
    };
    check_single_fused(fn_sw, LirOpcode::Movsxd);
    check_single_fused(fn_sw_idx, LirOpcode::Movsxd);
    check_single_fused(fn_ub, LirOpcode::Movzx8);
    check_single_fused(fn_ub2, LirOpcode::Movzx8);
    check_single_fused(fn_uw, LirOpcode::Mov32);
    {
        auto lir = isel.lower(*fn_hazard);
        REQUIRE(lir != nullptr);
        int movzx8_count = 0;
        for (const auto& blk : lir->blocks) {
            for (const auto& inst : blk->instructions) {
                if (inst->opcode == LirOpcode::Movzx8) ++movzx8_count;
            }
        }
        // Unfused: 1 Movzx8 for the load (before the store) + 1 Movzx8 for zext_i64
        CHECK_EQ(movzx8_count, 2);
    }

    // Verify no sxtw or uxtb is emitted in machine code for the fused loads
    auto has_uxtb_sxtw = [&](Function* f) {
        auto lir = isel.lower(*f);
        LivenessAnalysis liv(*lir);
        liv.run();
        LinearScanAllocator ra(*lir, liv, CallingConvention::apple_aapcs64());
        ra.allocate();
        AArch64EmitContext em(*lir, Target::aarch64_macos());
        for (uint32_t w : extract_words(em.compile().code_buffer.bytes())) {
            // uxtb = 0x53001C00 mask 0xFFFFFC00, sxtw = 0x93407C00 mask 0xFFFFFC00
            if ((w & 0xFFFFFC00u) == 0x53001C00u || (w & 0xFFFFFC00u) == 0x93407C00u) return true;
        }
        return false;
    };
    CHECK(!has_uxtb_sxtw(fn_sw));
    CHECK(!has_uxtb_sxtw(fn_ub));
    CHECK(!has_uxtb_sxtw(fn_ub2));

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));

    alignas(16) std::array<uint8_t, 32> buf{};
    const int32_t s32 = std::numeric_limits<int32_t>::min();
    std::memcpy(buf.data() + 8, &s32, sizeof(s32));
    auto j_sw = jit.get_function_ptr<int64_t (*)(const void*)>("sw");
    CHECK_EQ(j_sw(buf.data()), int64_t(std::numeric_limits<int32_t>::min()));

    const int32_t s32_neg1 = -1;
    std::memcpy(buf.data() + 12, &s32_neg1, sizeof(s32_neg1));
    auto j_sw_idx = jit.get_function_ptr<int64_t (*)(const void*, int64_t)>("sw_idx");
    CHECK_EQ(j_sw_idx(buf.data(), 3), int64_t(-1));

    buf[5] = 0xABu;
    auto j_ub = jit.get_function_ptr<int64_t (*)(const void*, int64_t)>("ub");
    CHECK_EQ(j_ub(buf.data(), 5), int64_t(0xAB));

    buf[8] = 0xFEu;
    auto j_ub2 = jit.get_function_ptr<int64_t (*)(const void*)>("ub2");
    CHECK_EQ(j_ub2(buf.data()), int64_t(0xFE));

    const uint32_t u32 = 0xFEDCBA98u;
    std::memcpy(buf.data() + 12, &u32, sizeof(u32));
    auto j_uw = jit.get_function_ptr<int64_t (*)(const void*, int64_t)>("uw");
    CHECK_EQ(j_uw(buf.data(), 3), int64_t(0xFEDCBA98ULL));

    // Store hazard test: cell initially 0xFB (251), overwritten by store(0x11)
    uint8_t cell = 0xFBu;
    auto j_haz = jit.get_function_ptr<int64_t (*)(void*)>("ext_load_store_hazard");
    CHECK_EQ(j_haz(&cell), int64_t(0xFB));
    CHECK_EQ(cell, uint8_t(0x11));
}

