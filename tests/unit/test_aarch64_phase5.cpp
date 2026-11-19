#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <array>
#include <bit>
#include <cmath>
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

bool writes_x15_alu_or_mov(uint32_t w) noexcept {
    // MOVZ/MOVN/MOVK (64/32-bit): bits [28:23] == 0b100101, Rd in bits [4:0]
    if (((w >> 23) & 0x3Fu) == 0b100101u && (w & 0x1Fu) == 15u) return true;
    // ADD/SUB imm (64/32-bit): bits [28:24] == 0b10001, Rd in bits [4:0]
    if (((w >> 24) & 0x1Fu) == 0b10001u && (w & 0x1Fu) == 15u) return true;
    // LDR (reg/unsigned imm/unscaled) overwriting X15/W15
    if (((w >> 22) & 0x3FFu) == 0b1111100101u && (w & 0x1Fu) == 15u) return true;
    if (((w >> 22) & 0x3FFu) == 0b1011100101u && (w & 0x1Fu) == 15u) return true;
    return false;
}

} // namespace

// 5A.1: Raw-LIR indexed stores with X15 index + disp must not clobber X15 in data_scratch/src_scratch
TEST_CASE("AArch64 Phase 5A - X15 indexed store with disp preserves X15 in Mov, Mov32, and ParallelCopy") {
    // Case 1: Mov [X0 + X15*8 + 16], #0x123456
    {
        LirFunction fn;
        fn.name = "p5a_mov64_x15";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");
        auto st = std::make_unique<LirInst>(LirOpcode::Mov);
        LirMem dst_mem;
        dst_mem.base_preg = PReg::aarch64_gpr(GPR::X0);
        dst_mem.index_preg = PReg::aarch64_gpr(GPR::X15);
        dst_mem.scale = x64::Scale::Eight;
        dst_mem.disp = 16;
        st->add_def(LirOperand::mem_custom(dst_mem, 8));
        st->add_use(LirOperand::imm(0x123456, 8));
        bb->append_inst(std::move(st));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto res = em.compile();
        auto words = extract_words(res.code_buffer.bytes());
        bool saw_str_x16_x17_x15 = false;
        for (uint32_t w : words) {
            CHECK(!writes_x15_alu_or_mov(w));
            // STR X16, [X17, X15, lsl #3] -> 0xF8207800 | (15 << 16) | (17 << 5) | 16 = 0xF82F7A30
            if (w == 0xF82F7A30u) saw_str_x16_x17_x15 = true;
        }
        CHECK(saw_str_x16_x17_x15);
    }

    // Case 2: Mov32 [X0 + X15*4 + 16], #0x12345678
    {
        LirFunction fn;
        fn.name = "p5a_mov32_x15";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");
        auto st = std::make_unique<LirInst>(LirOpcode::Mov32);
        LirMem dst_mem;
        dst_mem.base_preg = PReg::aarch64_gpr(GPR::X0);
        dst_mem.index_preg = PReg::aarch64_gpr(GPR::X15);
        dst_mem.scale = x64::Scale::Four;
        dst_mem.disp = 16;
        st->add_def(LirOperand::mem_custom(dst_mem, 4));
        st->add_use(LirOperand::imm(0x12345678, 4));
        bb->append_inst(std::move(st));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto res = em.compile();
        auto words = extract_words(res.code_buffer.bytes());
        bool saw_str_w16_x17_x15 = false;
        for (uint32_t w : words) {
            CHECK(!writes_x15_alu_or_mov(w));
            // STR W16, [X17, X15, lsl #2] -> 0xB8207800 | (15 << 16) | (17 << 5) | 16 = 0xB82F7A30
            if (w == 0xB82F7A30u) saw_str_w16_x17_x15 = true;
        }
        CHECK(saw_str_w16_x17_x15);
    }

    // Case 3: Mem-to-mem Mov [X0 + X15*8 + 16], [X1 + 4097] (src requires src_scratch; must not touch X15/X17)
    {
        LirFunction fn;
        fn.name = "p5a_mem2mem_x15";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");
        auto st = std::make_unique<LirInst>(LirOpcode::Mov);
        LirMem dst_mem;
        dst_mem.base_preg = PReg::aarch64_gpr(GPR::X0);
        dst_mem.index_preg = PReg::aarch64_gpr(GPR::X15);
        dst_mem.scale = x64::Scale::Eight;
        dst_mem.disp = 16;
        LirMem src_mem;
        src_mem.base_preg = PReg::aarch64_gpr(GPR::X1);
        src_mem.disp = 4097;
        st->add_def(LirOperand::mem_custom(dst_mem, 8));
        st->add_use(LirOperand::mem_custom(src_mem, 8));
        bb->append_inst(std::move(st));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto res = em.compile();
        auto words = extract_words(res.code_buffer.bytes());
        bool saw_str_x16_x17_x15 = false;
        for (uint32_t w : words) {
            CHECK(!writes_x15_alu_or_mov(w));
            if (w == 0xF82F7A30u) saw_str_x16_x17_x15 = true;
        }
        CHECK(saw_str_x16_x17_x15);
    }

    // Case 4: ParallelCopy [X0 + X15*8 + 16] <- [X1 + 4097] also avoids X15 and preserves X17 base
    {
        LirFunction fn;
        fn.name = "p5a_pcopy_x15";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");
        auto pc = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
        LirMem dst_mem;
        dst_mem.base_preg = PReg::aarch64_gpr(GPR::X0);
        dst_mem.index_preg = PReg::aarch64_gpr(GPR::X15);
        dst_mem.scale = x64::Scale::Eight;
        dst_mem.disp = 16;
        LirMem src_mem;
        src_mem.base_preg = PReg::aarch64_gpr(GPR::X1);
        src_mem.disp = 4097;
        pc->add_def(LirOperand::mem_custom(dst_mem, 8));
        pc->add_use(LirOperand::mem_custom(src_mem, 8));
        bb->append_inst(std::move(pc));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto res = em.compile();
        auto words = extract_words(res.code_buffer.bytes());
        bool saw_str_x16_x17_x15 = false;
        for (uint32_t w : words) {
            CHECK(!writes_x15_alu_or_mov(w));
            if (w == 0xF82F7A30u) saw_str_x16_x17_x15 = true;
        }
        CHECK(saw_str_x16_x17_x15);
    }
}

// 5A.2: Do not fuse mul -> add into Madd across an intervening call; guard bitcast_through_slots store size
TEST_CASE("AArch64 Phase 5A - No madd fusion across call and peephole bitcast size guard") {
    auto mod = std::make_unique<Module>("p5a_isel");
    Function* helper = mod->create_function("helper", Type::i64(), {Type::i64()});
    {
        Builder b(*mod);
        b.set_function(helper);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::i64());
        b.build_ret(b.build_add(x, b.build_iconst_i64(1)));
        helper->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*helper));
    }

    Function* across_call = mod->create_function("across_call", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    {
        Builder b(*mod);
        b.set_function(across_call);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::i64());
        Value* y = b.add_block_param(bb, Type::i64());
        Value* z = b.add_block_param(bb, Type::i64());
        Value* prod = b.build_mul(x, y);
        Value* c = b.build_call("helper", Type::i64(), {z});
        b.build_ret(b.build_add(c, prod));
        across_call->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*across_call));
    }

    Function* same_region = mod->create_function("same_region", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
    {
        Builder b(*mod);
        b.set_function(same_region);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::i64());
        Value* y = b.add_block_param(bb, Type::i64());
        Value* z = b.add_block_param(bb, Type::i64());
        Value* prod = b.build_mul(x, y);
        b.build_ret(b.build_add(z, prod));
        same_region->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*same_region));
    }

    AArch64ISel isel(Target::aarch64_macos(), CallingConvention::apple_aapcs64());
    auto count_madd = [](const LirFunction& lir) {
        size_t n = 0;
        for (const auto& b : lir.blocks) {
            for (const auto& inst : b->instructions) {
                if (inst && inst->opcode == LirOpcode::Madd) ++n;
            }
        }
        return n;
    };
    auto lir_across = isel.lower(*across_call);
    REQUIRE(lir_across != nullptr);
    CHECK_EQ(count_madd(*lir_across), size_t(0));

    auto lir_same = isel.lower(*same_region);
    REQUIRE(lir_same != nullptr);
    CHECK_EQ(count_madd(*lir_same), size_t(1));

    // Verify JIT execution accuracy as well
    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(int64_t, int64_t, int64_t)>("across_call");
    REQUIRE(fn != nullptr);
    CHECK_EQ(fn(6, 7, 10), int64_t(53));

    // Peephole bitcast_through_slots store path must ignore Movq_gx with uses[0].size != 8
    auto run_bitcast = [](uint8_t src_size) {
        LirFunction fn;
        fn.name = "p5a_bitcast";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        fn.frame.num_spill_slots = 1;
        LirBlock* bb = fn.create_block("entry");
        auto mq = std::make_unique<LirInst>(LirOpcode::Movq_gx);
        mq->add_def(LirOperand::preg(PReg::aarch64_gpr(GPR::X9), 8));
        mq->add_use(LirOperand::preg(PReg::aarch64_fpr(FPR::V0), src_size));
        bb->append_inst(std::move(mq));
        auto st = std::make_unique<LirInst>(LirOpcode::Mov);
        st->add_def(LirOperand::slot(0, 8));
        st->add_use(LirOperand::preg(PReg::aarch64_gpr(GPR::X9), 8));
        bb->append_inst(std::move(st));
        auto overwrite = std::make_unique<LirInst>(LirOpcode::Mov);
        overwrite->add_def(LirOperand::preg(PReg::aarch64_gpr(GPR::X9), 8));
        overwrite->add_use(LirOperand::imm(0, 8));
        bb->append_inst(std::move(overwrite));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
        PeepholeOptimizer opt(fn);
        return opt.run().bitcasts_through_slots;
    };
    CHECK_EQ(run_bitcast(4), size_t(0));
    CHECK_EQ(run_bitcast(8), size_t(1));
}

// 5B: 64-bit ARMv8 logical immediates fold directly into And/Or/Xor without Movabs
TEST_CASE("AArch64 Phase 5B - 64-bit ARMv8 logical immediates in And/Or/Xor") {
    constexpr uint64_t kPayloadMask = 0x0000FFFFFFFFFFFFULL;
    constexpr uint64_t kTagMask = 0xFFFF000000000000ULL;
    constexpr uint64_t kIntTag = 0xFFFC000000000000ULL;
    constexpr uint64_t kSignBit = 0x8000000000000000ULL;
    constexpr uint64_t kNonLogical = 0x123456789ABCDEF0ULL;

    auto mod = std::make_unique<Module>("p5b_logical");
    auto make_mask_fn = [&](const char* name, Opcode op, uint64_t mask, bool lhs_const) -> Function* {
        Function* fn = mod->create_function(name, Type::i64(), {Type::i64()});
        Builder b(*mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::i64());
        Value* k = b.build_iconst_i64(static_cast<int64_t>(mask));
        Value* lhs = lhs_const ? k : x;
        Value* rhs = lhs_const ? x : k;
        Value* res = (op == Opcode::and_) ? b.build_and(lhs, rhs)
                   : (op == Opcode::or_)  ? b.build_or(lhs, rhs)
                                          : b.build_xor(lhs, rhs);
        b.build_ret(res);
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        return fn;
    };

    Function* fn_and = make_mask_fn("and_payload", Opcode::and_, kPayloadMask, false);
    Function* fn_and_lhs = make_mask_fn("and_tag_lhs", Opcode::and_, kTagMask, true);
    Function* fn_or = make_mask_fn("or_tag", Opcode::or_, kIntTag, false);
    Function* fn_xor = make_mask_fn("xor_sign", Opcode::xor_, kSignBit, false);
    Function* fn_non_log = make_mask_fn("and_non_log", Opcode::and_, kNonLogical, false);

    AArch64ISel isel(Target::aarch64_macos(), CallingConvention::apple_aapcs64());
    auto count_movabs = [](const LirFunction& lir) {
        size_t n = 0;
        for (const auto& b : lir.blocks) {
            for (const auto& inst : b->instructions) {
                if (inst && inst->opcode == LirOpcode::Movabs) ++n;
            }
        }
        return n;
    };
    auto has_bitwise_imm = [](const LirFunction& lir, LirOpcode op, int64_t imm) {
        for (const auto& b : lir.blocks) {
            for (const auto& inst : b->instructions) {
                if (inst && inst->opcode == op && inst->uses.size() >= 2 &&
                    inst->uses.back().is_imm_int() && inst->uses.back().imm_int == imm) {
                    return true;
                }
            }
        }
        return false;
    };

    auto lir_and = isel.lower(*fn_and);
    auto lir_and_lhs = isel.lower(*fn_and_lhs);
    auto lir_or = isel.lower(*fn_or);
    auto lir_xor = isel.lower(*fn_xor);
    auto lir_non_log = isel.lower(*fn_non_log);

    CHECK_EQ(count_movabs(*lir_and), size_t(0));
    CHECK_EQ(count_movabs(*lir_and_lhs), size_t(0));
    CHECK_EQ(count_movabs(*lir_or), size_t(0));
    CHECK_EQ(count_movabs(*lir_xor), size_t(0));
    CHECK(has_bitwise_imm(*lir_and, LirOpcode::And, static_cast<int64_t>(kPayloadMask)));
    CHECK(has_bitwise_imm(*lir_and_lhs, LirOpcode::And, static_cast<int64_t>(kTagMask)));
    CHECK(has_bitwise_imm(*lir_or, LirOpcode::Or, static_cast<int64_t>(kIntTag)));
    CHECK(has_bitwise_imm(*lir_xor, LirOpcode::Xor, static_cast<int64_t>(kSignBit)));
    CHECK_EQ(count_movabs(*lir_non_log), size_t(1));

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    using Fn = int64_t (*)(int64_t);
    auto j_and = jit.get_function_ptr<Fn>("and_payload");
    auto j_and_lhs = jit.get_function_ptr<Fn>("and_tag_lhs");
    auto j_or = jit.get_function_ptr<Fn>("or_tag");
    auto j_xor = jit.get_function_ptr<Fn>("xor_sign");
    auto j_non_log = jit.get_function_ptr<Fn>("and_non_log");
    REQUIRE(j_and && j_and_lhs && j_or && j_xor && j_non_log);

    const uint64_t inputs[] = {0ULL, ~0ULL, 0xFFF8123456789ABCULL, 0x8000000000000001ULL, 0x0000DEAD1234BEEFULL};
    for (uint64_t u : inputs) {
        int64_t s = static_cast<int64_t>(u);
        CHECK_EQ(static_cast<uint64_t>(j_and(s)), u & kPayloadMask);
        CHECK_EQ(static_cast<uint64_t>(j_and_lhs(s)), kTagMask & u);
        CHECK_EQ(static_cast<uint64_t>(j_or(s)), u | kIntTag);
        CHECK_EQ(static_cast<uint64_t>(j_xor(s)), u ^ kSignBit);
        CHECK_EQ(static_cast<uint64_t>(j_non_log(s)), u & kNonLogical);
    }
}

// 5C.1: Encoder bitwords verified against Apple clang (fmov #imm8, movi #0, ldr_literal_s, zero-cycle mov xzr)
TEST_CASE("AArch64 Phase 5C - Encoder fmov #imm8, movi #0, ldr_literal_s, and zero-cycle Xor") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);

    uint8_t imm8 = 0;
    REQUIRE(AArch64Encoder::encode_fp64_imm8(std::bit_cast<uint64_t>(1.0), imm8));
    enc.fmov_imm(FPR::V0, imm8);
    REQUIRE(AArch64Encoder::encode_fp64_imm8(std::bit_cast<uint64_t>(-1.0), imm8));
    enc.fmov_imm(FPR::V3, imm8);
    REQUIRE(AArch64Encoder::encode_fp64_imm8(std::bit_cast<uint64_t>(0.5), imm8));
    enc.fmov_imm(FPR::V5, imm8);
    REQUIRE(AArch64Encoder::encode_fp64_imm8(std::bit_cast<uint64_t>(0.125), imm8));
    enc.fmov_imm(FPR::V7, imm8);
    REQUIRE(AArch64Encoder::encode_fp64_imm8(std::bit_cast<uint64_t>(31.0), imm8));
    enc.fmov_imm(FPR::V9, imm8);
    REQUIRE(AArch64Encoder::encode_fp32_imm8(std::bit_cast<uint32_t>(1.0f), imm8));
    enc.fmov_imm_s(FPR::V0, imm8);
    REQUIRE(AArch64Encoder::encode_fp32_imm8(std::bit_cast<uint32_t>(-1.0f), imm8));
    enc.fmov_imm_s(FPR::V3, imm8);

    // Non-FP8 values (+0.0, -0.0, 0.0625, 32.0, NaN, pi) must be rejected
    CHECK(!AArch64Encoder::encode_fp64_imm8(0ULL, imm8));
    CHECK(!AArch64Encoder::encode_fp64_imm8(0x8000000000000000ULL, imm8));
    CHECK(!AArch64Encoder::encode_fp64_imm8(std::bit_cast<uint64_t>(0.0625), imm8));
    CHECK(!AArch64Encoder::encode_fp64_imm8(std::bit_cast<uint64_t>(32.0), imm8));
    CHECK(!AArch64Encoder::encode_fp64_imm8(0x7FF8000000000000ULL, imm8));
    CHECK(!AArch64Encoder::encode_fp32_imm8(0u, imm8));
    CHECK(!AArch64Encoder::encode_fp32_imm8(0x80000000u, imm8));

    enc.movi_zero(FPR::V0);
    Label lit = buf.create_label();
    enc.ldr_literal_s(FPR::V4, lit, 0x3F800000u);
    buf.bind(lit);
    buf.emit32(0x3F800000u);

    auto words = extract_words(buf.bytes());
    REQUIRE(words.size() == 10u);
    CHECK_EQ(words[0], 0x1E6E1000u); // fmov d0, #1.0
    CHECK_EQ(words[1], 0x1E7E1003u); // fmov d3, #-1.0
    CHECK_EQ(words[2], 0x1E6C1005u); // fmov d5, #0.5
    CHECK_EQ(words[3], 0x1E681007u); // fmov d7, #0.125
    CHECK_EQ(words[4], 0x1E67F009u); // fmov d9, #31.0
    CHECK_EQ(words[5], 0x1E2E1000u); // fmov s0, #1.0
    CHECK_EQ(words[6], 0x1E3E1003u); // fmov s3, #-1.0
    CHECK_EQ(words[7], 0x2F00E400u); // movi d0, #0
    CHECK_EQ(words[8], 0x1C000024u); // ldr s4, +4

    // Self-Xor / self-Xor32 emit mov xd, xzr / mov wd, wzr instead of eor
    LirFunction fn;
    fn.name = "p5c_zero_idiom";
    fn.calling_conv = CallingConvention::apple_aapcs64();
    LirBlock* bb = fn.create_block("entry");
    auto x64_z = std::make_unique<LirInst>(LirOpcode::Xor);
    x64_z->add_def(LirOperand::preg(PReg::aarch64_gpr(GPR::X0), 8));
    x64_z->add_use(LirOperand::preg(PReg::aarch64_gpr(GPR::X0), 8));
    x64_z->add_use(LirOperand::preg(PReg::aarch64_gpr(GPR::X0), 8));
    bb->append_inst(std::move(x64_z));
    auto x32_z = std::make_unique<LirInst>(LirOpcode::Xor32);
    x32_z->add_def(LirOperand::preg(PReg::aarch64_gpr(GPR::X1), 4));
    x32_z->add_use(LirOperand::preg(PReg::aarch64_gpr(GPR::X1), 4));
    x32_z->add_use(LirOperand::preg(PReg::aarch64_gpr(GPR::X1), 4));
    bb->append_inst(std::move(x32_z));
    bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    AArch64EmitContext em(fn, Target::aarch64_macos());
    auto res = em.compile();
    auto zwords = extract_words(res.code_buffer.bytes());
    bool saw_mov_x0_xzr = false;
    bool saw_mov_w1_wzr = false;
    for (uint32_t w : zwords) {
        if (w == 0xAA1F03E0u) saw_mov_x0_xzr = true; // mov x0, xzr
        if (w == 0x2A1F03E1u) saw_mov_w1_wzr = true; // mov w1, wzr
        CHECK((w & 0xFFE0FC00u) != 0xCA000000u);     // no eor
        CHECK((w & 0xFFE0FC00u) != 0x4A000000u);     // no eor32
    }
    CHECK(saw_mov_x0_xzr);
    CHECK(saw_mov_w1_wzr);
}

// 5C.2: Native FP constants (+0.0, -0.0, #imm8, pool) and fcmp #0.0 with exact NaN/signed-zero semantics
TEST_CASE("AArch64 Phase 5C - Native FP constants and fcmp #0.0 NaN/signed-zero semantics") {
    auto mod = std::make_unique<Module>("p5c_fp");

    auto make_f64_bits_fn = [&](const char* name, double val) {
        Function* fn = mod->create_function(name, Type::i64(), {});
        Builder b(*mod);
        b.set_function(fn);
        b.append_block("entry");
        Value* c = b.build_fconst_f64(val);
        b.build_ret(b.build_bitcast_i64_f64(c));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        return fn;
    };
    auto make_f32_bits_fn = [&](const char* name, float val) {
        Function* fn = mod->create_function(name, Type::i32(), {});
        Builder b(*mod);
        b.set_function(fn);
        b.append_block("entry");
        Value* c = b.build_fconst_f32(val);
        b.build_ret(b.build_bitcast_i32_f32(c));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        return fn;
    };

    Function* fn_p0 = make_f64_bits_fn("k_p0", 0.0);
    Function* fn_n0 = make_f64_bits_fn("k_n0", -0.0);
    Function* fn_imm8 = make_f64_bits_fn("k_half", -0.5);
    Function* fn_pi = make_f64_bits_fn("k_pi", 3.141592653589793);
    Function* fn_s_p0 = make_f32_bits_fn("k_s_p0", 0.0f);
    Function* fn_s_n0 = make_f32_bits_fn("k_s_n0", -0.0f);
    Function* fn_s_imm8 = make_f32_bits_fn("k_s_imm8", 1.5f);
    Function* fn_s_pi = make_f32_bits_fn("k_s_pi", 3.1415927f);

    // Build fcmp #0.0 test functions: fused select + br_if, both RHS 0.0 and LHS -0.0, f64 & f32
    auto make_fcmp0_select = [&](const char* name, Opcode cmp_op, bool zero_lhs, double zero_val) {
        Function* fn = mod->create_function(name, Type::i64(), {Type::f64()});
        Builder b(*mod);
        b.set_function(fn);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::f64());
        Value* z = b.build_fconst_f64(zero_val);
        Value* l = zero_lhs ? z : x;
        Value* r = zero_lhs ? x : z;
        Value* cond = nullptr;
        switch (cmp_op) {
            case Opcode::eq:  cond = b.build_eq(l, r); break;
            case Opcode::ne:  cond = b.build_ne(l, r); break;
            case Opcode::slt: cond = b.build_slt(l, r); break;
            case Opcode::sle: cond = b.build_sle(l, r); break;
            case Opcode::sgt: cond = b.build_sgt(l, r); break;
            case Opcode::sge: cond = b.build_sge(l, r); break;
            default: break;
        }
        b.build_ret(b.build_select(cond, b.build_iconst_i64(1), b.build_iconst_i64(0)));
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));
        return fn;
    };

    Function* fn_eq0 = make_fcmp0_select("cmp_eq_0", Opcode::eq, false, 0.0);
    Function* fn_ne0 = make_fcmp0_select("cmp_ne_0", Opcode::ne, false, -0.0);
    Function* fn_lt0 = make_fcmp0_select("cmp_lt_0", Opcode::slt, false, 0.0);
    Function* fn_le0 = make_fcmp0_select("cmp_le_0", Opcode::sle, false, 0.0);
    Function* fn_gt0 = make_fcmp0_select("cmp_gt_0", Opcode::sgt, false, 0.0);
    Function* fn_ge0 = make_fcmp0_select("cmp_ge_0", Opcode::sge, false, 0.0);
    Function* fn_0lt = make_fcmp0_select("cmp_0_lt", Opcode::slt, true, -0.0);
    Function* fn_0le = make_fcmp0_select("cmp_0_le", Opcode::sle, true, 0.0);
    Function* fn_0gt = make_fcmp0_select("cmp_0_gt", Opcode::sgt, true, 0.0);
    Function* fn_0ge = make_fcmp0_select("cmp_0_ge", Opcode::sge, true, -0.0);

    Function* fn_f32_lt0 = mod->create_function("f32_lt0", Type::i64(), {Type::f32()});
    {
        Builder b(*mod);
        b.set_function(fn_f32_lt0);
        BasicBlock* bb = b.append_block("entry");
        Value* x = b.add_block_param(bb, Type::f32());
        Value* c = b.build_slt(b.build_fconst_f32(-0.0f), x);
        b.build_ret(b.build_zext_i64(c));
        fn_f32_lt0->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn_f32_lt0));
    }

    // Check LIR: dead 0.0 constants in fcmp #0.0 are eliminated and zero cross-domain fmov d, x is emitted
    AArch64ISel isel(Target::aarch64_macos(), CallingConvention::apple_aapcs64());
    for (Function* cf : {fn_eq0, fn_ne0, fn_lt0, fn_le0, fn_gt0, fn_ge0,
                          fn_0lt, fn_0le, fn_0gt, fn_0ge, fn_f32_lt0}) {
        auto lir = isel.lower(*cf);
        REQUIRE(lir != nullptr);
        bool saw_ucomi_imm0 = false;
        for (const auto& b : lir->blocks) {
            for (const auto& inst : b->instructions) {
                if ((inst->opcode == LirOpcode::Ucomisd || inst->opcode == LirOpcode::Ucomiss) &&
                    inst->uses.size() == 2 && inst->uses[1].is_imm_float() && inst->uses[1].imm_float == 0.0) {
                    saw_ucomi_imm0 = true;
                }
                CHECK(inst->opcode != LirOpcode::Movsd);
                CHECK(inst->opcode != LirOpcode::Movss);
                CHECK(inst->opcode != LirOpcode::Xorpd);
                CHECK(inst->opcode != LirOpcode::Xorps);
            }
        }
        CHECK(saw_ucomi_imm0);
    }

    // Verify k_p0 emits movi #0 (2f00e400 mask), k_half emits fmov #imm8, k_n0 & k_pi emit ldr_literal without fmov d, x
    auto emit_words = [&](Function* f) {
        auto lir = isel.lower(*f);
        REQUIRE(lir != nullptr);
        LivenessAnalysis liv(*lir);
        liv.run();
        LinearScanAllocator ra(*lir, liv, CallingConvention::apple_aapcs64());
        ra.allocate();
        AArch64EmitContext cem(*lir, Target::aarch64_macos());
        return extract_words(cem.compile().code_buffer.bytes());
    };
    {
        auto w_p0 = emit_words(fn_p0);
        auto w_n0 = emit_words(fn_n0);
        auto w_imm8 = emit_words(fn_imm8);
        auto w_pi = emit_words(fn_pi);
        bool p0_movi = false, imm8_fmov = false, n0_lit = false, pi_lit = false;
        for (uint32_t w : w_p0) {
            if ((w & ~0x1Fu) == 0x2F00E400u) p0_movi = true;
        }
        for (uint32_t w : w_imm8) {
            if ((w & 0xFFE0FC00u) == 0x1E601000u) imm8_fmov = true;
        }
        for (uint32_t w : w_n0) {
            CHECK((w & ~0x1Fu) != 0x2F00E400u); // -0.0 must NOT use movi #0
            if ((w & 0xFF000000u) == 0x58000000u || (w & 0xFF000000u) == 0x5C000000u) n0_lit = true;
        }
        for (uint32_t w : w_pi) {
            CHECK((w & 0xFFFFFC00u) != 0x9E670000u); // no fmov d, x cross-domain move
            if ((w & 0xFF000000u) == 0x5C000000u) pi_lit = true;
        }
        CHECK(p0_movi);
        CHECK(imm8_fmov);
        CHECK(n0_lit);
        CHECK(pi_lit);

        auto w_s_p0 = emit_words(fn_s_p0);
        auto w_s_n0 = emit_words(fn_s_n0);
        auto w_s_imm8 = emit_words(fn_s_imm8);
        auto w_s_pi = emit_words(fn_s_pi);
        bool s_p0_movi = false, s_imm8_fmov = false, s_n0_lit = false, s_pi_lit = false;
        for (uint32_t w : w_s_p0) {
            if ((w & ~0x1Fu) == 0x2F00E400u) s_p0_movi = true;
        }
        for (uint32_t w : w_s_imm8) {
            if ((w & 0xFFE0FC00u) == 0x1E201000u) s_imm8_fmov = true;
        }
        for (uint32_t w : w_s_n0) {
            CHECK((w & ~0x1Fu) != 0x2F00E400u);
            if ((w & 0xFF000000u) == 0x1C000000u) s_n0_lit = true;
        }
        for (uint32_t w : w_s_pi) {
            CHECK((w & 0xFFFFFC00u) != 0x1E270000u); // no fmov s, w cross-domain move
            if ((w & 0xFF000000u) == 0x1C000000u) s_pi_lit = true;
        }
        CHECK(s_p0_movi);
        CHECK(s_imm8_fmov);
        CHECK(s_n0_lit);
        CHECK(s_pi_lit);
    }

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));

    CHECK_EQ(static_cast<uint64_t>(jit.get_function_ptr<int64_t (*)()>("k_p0")()), 0x0000000000000000ULL);
    CHECK_EQ(static_cast<uint64_t>(jit.get_function_ptr<int64_t (*)()>("k_n0")()), 0x8000000000000000ULL);
    CHECK_EQ(static_cast<uint64_t>(jit.get_function_ptr<int64_t (*)()>("k_half")()),
             std::bit_cast<uint64_t>(-0.5));
    CHECK_EQ(static_cast<uint64_t>(jit.get_function_ptr<int64_t (*)()>("k_pi")()),
             std::bit_cast<uint64_t>(3.141592653589793));

    CHECK_EQ(static_cast<uint32_t>(jit.get_function_ptr<int32_t (*)()>("k_s_p0")()), 0x00000000u);
    CHECK_EQ(static_cast<uint32_t>(jit.get_function_ptr<int32_t (*)()>("k_s_n0")()), 0x80000000u);
    CHECK_EQ(static_cast<uint32_t>(jit.get_function_ptr<int32_t (*)()>("k_s_imm8")()),
             std::bit_cast<uint32_t>(1.5f));
    CHECK_EQ(static_cast<uint32_t>(jit.get_function_ptr<int32_t (*)()>("k_s_pi")()),
             std::bit_cast<uint32_t>(3.1415927f));

    auto j_eq0 = jit.get_function_ptr<int64_t (*)(double)>("cmp_eq_0");
    auto j_ne0 = jit.get_function_ptr<int64_t (*)(double)>("cmp_ne_0");
    auto j_lt0 = jit.get_function_ptr<int64_t (*)(double)>("cmp_lt_0");
    auto j_le0 = jit.get_function_ptr<int64_t (*)(double)>("cmp_le_0");
    auto j_gt0 = jit.get_function_ptr<int64_t (*)(double)>("cmp_gt_0");
    auto j_ge0 = jit.get_function_ptr<int64_t (*)(double)>("cmp_ge_0");
    auto j_0lt = jit.get_function_ptr<int64_t (*)(double)>("cmp_0_lt");
    auto j_0le = jit.get_function_ptr<int64_t (*)(double)>("cmp_0_le");
    auto j_0gt = jit.get_function_ptr<int64_t (*)(double)>("cmp_0_gt");
    auto j_0ge = jit.get_function_ptr<int64_t (*)(double)>("cmp_0_ge");
    auto j_f32_0lt = jit.get_function_ptr<int64_t (*)(float)>("f32_lt0");

    const std::array<double, 5> vals = {-2.5, -0.0, +0.0, 2.5, std::numeric_limits<double>::quiet_NaN()};
    for (double v : vals) {
        CHECK_EQ(j_eq0(v), int64_t(v == 0.0 ? 1 : 0));
        CHECK_EQ(j_ne0(v), int64_t(v != 0.0 ? 1 : 0));
        CHECK_EQ(j_lt0(v), int64_t(v < 0.0 ? 1 : 0));
        CHECK_EQ(j_le0(v), int64_t(v <= 0.0 ? 1 : 0));
        CHECK_EQ(j_gt0(v), int64_t(v > 0.0 ? 1 : 0));
        CHECK_EQ(j_ge0(v), int64_t(v >= 0.0 ? 1 : 0));
        CHECK_EQ(j_0lt(v), int64_t(0.0 < v ? 1 : 0));
        CHECK_EQ(j_0le(v), int64_t(0.0 <= v ? 1 : 0));
        CHECK_EQ(j_0gt(v), int64_t(0.0 > v ? 1 : 0));
        CHECK_EQ(j_0ge(v), int64_t(0.0 >= v ? 1 : 0));
    }
    const std::array<float, 5> fvals = {-2.5f, -0.0f, +0.0f, 2.5f, std::numeric_limits<float>::quiet_NaN()};
    for (float v : fvals) {
        CHECK_EQ(j_f32_0lt(v), int64_t(0.0f < v ? 1 : 0));
    }
}
