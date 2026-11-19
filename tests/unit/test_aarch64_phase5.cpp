#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
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
