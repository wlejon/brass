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
