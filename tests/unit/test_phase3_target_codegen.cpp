#include "test_framework.hpp"
#include <brass/target/x64/x64_frame.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/aarch64/aarch64_frame.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/target.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/codegen/peephole.hpp>
#include <vector>
#include <stdexcept>

using namespace brass;
using namespace brass::x64;
using namespace brass::codegen;
using namespace brass::object;

// =============================================================================
// Helper: read ULEB128 from buffer
// =============================================================================
static uint64_t decode_uleb128(const uint8_t*& p, const uint8_t* end) {
    uint64_t result = 0;
    int shift = 0;
    while (p < end) {
        uint8_t byte = *p++;
        result |= static_cast<uint64_t>(byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) break;
        shift += 7;
    }
    return result;
}

// =============================================================================
// Test 1: Win64 XMM stack layout alignment and disp_from_rsp % 16 == 0
// =============================================================================
TEST_CASE("Phase 3 - Win64 SEH XMM Stack Layout Alignment and disp_from_rsp % 16 == 0") {
    CallingConvention cc = CallingConvention::win64();

    // Case 1: Odd number of callee-saved GPRs (1 GPR: RBX)
    {
        FrameInfo frame;
        frame.saved_callee_gprs = (1u << static_cast<int>(GPR::RBX));
        frame.saved_callee_xmms = (1u << 6) | (1u << 7) | (1u << 8) | (1u << 15);
        frame.num_spill_slots = 3;
        frame.local_frame_bytes = 24;
        frame.has_calls = true;
        frame.outgoing_arg_space = 32;

        X64FrameLayout::compute_layout(frame, cc);

        CHECK(frame.total_frame_size > 0);
        CHECK_EQ(frame.total_frame_size % 16, size_t(0));

        auto saved_xmms = X64FrameLayout::get_saved_callee_xmms(frame);
        REQUIRE_EQ(saved_xmms.size(), size_t(4));

        for (XMM x : saved_xmms) {
            MemAddress addr = X64FrameLayout::callee_xmm_address(x, frame);
            int32_t disp_from_rbp = -addr.disp;
            // Offsets from RBP must be 16-byte aligned because gpr_bytes is padded to 16
            CHECK_EQ(disp_from_rbp % 16, 0);

            int32_t disp_from_rsp = static_cast<int32_t>(frame.total_frame_size - disp_from_rbp);
            // Win64 SEH UWOP_SAVE_XMM128 requires disp_from_rsp % 16 == 0 so integer division does not truncate
            CHECK_EQ(disp_from_rsp % 16, 0);

            uint16_t extra_slot1 = static_cast<uint16_t>(disp_from_rsp / 16);
            CHECK_EQ(static_cast<int32_t>(extra_slot1 * 16), disp_from_rsp);
        }

        // Check spill slot addresses start after XMM save slots
        for (int32_t s = 0; s < static_cast<int32_t>(frame.num_spill_slots); ++s) {
            MemAddress spill_addr = X64FrameLayout::spill_slot_address(s, frame);
            int32_t expected_disp = frame.spill_slot_offset(s);
            CHECK_EQ(spill_addr.disp, expected_disp);
        }
    }

    // Case 2: Even number of callee-saved GPRs (2 GPRs: RBX, R12)
    {
        FrameInfo frame;
        frame.saved_callee_gprs = (1u << static_cast<int>(GPR::RBX)) | (1u << static_cast<int>(GPR::R12));
        frame.saved_callee_xmms = (1u << 6) | (1u << 8);
        frame.num_spill_slots = 2;
        frame.has_calls = true;
        frame.outgoing_arg_space = 32;

        X64FrameLayout::compute_layout(frame, cc);

        CHECK_EQ(frame.total_frame_size % 16, size_t(0));

        auto saved_xmms = X64FrameLayout::get_saved_callee_xmms(frame);
        for (XMM x : saved_xmms) {
            MemAddress addr = X64FrameLayout::callee_xmm_address(x, frame);
            int32_t disp_from_rbp = -addr.disp;
            CHECK_EQ(disp_from_rbp % 16, 0);

            int32_t disp_from_rsp = static_cast<int32_t>(frame.total_frame_size - disp_from_rbp);
            CHECK_EQ(disp_from_rsp % 16, 0);
            uint16_t extra_slot1 = static_cast<uint16_t>(disp_from_rsp / 16);
            CHECK_EQ(static_cast<int32_t>(extra_slot1 * 16), disp_from_rsp);
        }
    }
}

// =============================================================================
// Test 2: DWARF CFI Factored Offset Calculation on x86_64
// =============================================================================
TEST_CASE("Phase 3 - DWARF CFI Factored Offset Calculation on x86_64") {
    ObjectFile obj;
    obj.target = Target::x64_linux();

    CompiledFunctionInfo fn;
    fn.name = "test_cfi_func";
    fn.text_offset = 0;
    fn.text_size = 128;

    fn.frame_info.is_leaf = false;
    fn.frame_info.total_frame_size = 64;
    // 3 callee-saved GPRs: RBX, R12, R14
    fn.frame_info.saved_callee_gprs = (1u << static_cast<int>(GPR::RBX)) |
                                     (1u << static_cast<int>(GPR::R12)) |
                                     (1u << static_cast<int>(GPR::R14));
    obj.functions.push_back(fn);

    Section eh_frame_sec;
    eh_frame_sec.name = ".eh_frame";
    ElfCfiBuilder::build_eh_frame(obj, eh_frame_sec);

    REQUIRE(eh_frame_sec.data.size() > 32);

    // Read CIE length at 0
    uint32_t cie_len = *reinterpret_cast<const uint32_t*>(eh_frame_sec.data.data());
    size_t fde_offset = 4 + cie_len;
    // Align to 8
    fde_offset = (fde_offset + 7) & ~size_t(7);
    REQUIRE(fde_offset < eh_frame_sec.data.size());

    // In FDE:
    // fde_offset + 0: Length (4 bytes)
    // fde_offset + 4: CIE pointer (4 bytes)
    // fde_offset + 8: PC Begin (4 bytes)
    // fde_offset + 12: PC Range (4 bytes)
    // fde_offset + 16: Augmentation Data Length (1 byte = 0)
    // fde_offset + 17: Call Frame Instructions
    size_t inst_ptr = fde_offset + 17;
    const uint8_t* p = eh_frame_sec.data.data() + inst_ptr;

    // Skip prologue instructions:
    // advance_loc 1, def_cfa_offset 16, offset 6 (RBP) 2
    REQUIRE(*p == (elf::DW_CFA_advance_loc | 1)); p++;
    REQUIRE(*p == elf::DW_CFA_def_cfa_offset); p++;
    REQUIRE(*p == 16); p++;
    REQUIRE(*p == (elf::DW_CFA_offset | 6)); p++;
    REQUIRE(*p == 2); p++;

    // advance_loc 3, def_cfa_register 6 (RBP)
    REQUIRE(*p == (elf::DW_CFA_advance_loc | 3)); p++;
    REQUIRE(*p == elf::DW_CFA_def_cfa_register); p++;
    REQUIRE(*p == 6); p++;

    // Callee-saved GPRs: RBX (dreg 3), R12 (dreg 12), R14 (dreg 14)
    auto saved_gprs = X64FrameLayout::get_saved_callee_gprs(fn.frame_info);
    REQUIRE_EQ(saved_gprs.size(), size_t(3));

    for (size_t i = 0; i < saved_gprs.size(); ++i) {
        uint8_t op = *p++;
        (void)op;
        uint8_t factored = *p++;

        // Factored offset MUST be i + 3 to produce (RBP + 16) - 8 * (i + 3) = RBP - 8 * (i + 1)
        CHECK_EQ(factored, static_cast<uint8_t>(i + 3));

        int32_t cfa_disp = 16 - 8 * static_cast<int32_t>(factored);
        int32_t expected_disp = X64FrameLayout::callee_gpr_address(saved_gprs[i], fn.frame_info).disp;
        CHECK_EQ(cfa_disp, expected_disp);
    }
}

// =============================================================================
// Test 3: Peephole Optimizer Preserving add/sub reg, 0 When Flags Live
// =============================================================================
TEST_CASE("Phase 3 - Peephole Optimizer Preserving add/sub reg, 0 When Flags Live") {
    // Subtest A: add rax, 0 followed by Jcc (flags ARE live) -> MUST NOT be erased
    {
        LirFunction fn;
        LirBlock* bb = fn.create_block("entry");

        auto add0 = std::make_unique<LirInst>(LirOpcode::Add);
        add0->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
        add0->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
        add0->add_use(LirOperand::imm(0, 8));
        bb->append_inst(std::move(add0));

        auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
        jcc->condition = x64::Condition::E;
        jcc->add_use(LirOperand::label(1));
        bb->append_inst(std::move(jcc));

        PeepholeStats stats = run_lir_peephole_optimizations(fn);
        CHECK_EQ(stats.arithmetic_simplified, size_t(0));
        REQUIRE_EQ(bb->instructions.size(), size_t(2));
        CHECK(bb->instructions[0]->opcode == LirOpcode::Add);
    }

    // Subtest B: sub rbx, 0 followed by Setcc (flags ARE live) -> MUST NOT be erased
    {
        LirFunction fn;
        LirBlock* bb = fn.create_block("entry");

        auto sub0 = std::make_unique<LirInst>(LirOpcode::Sub);
        sub0->add_def(LirOperand::preg_gpr(GPR::RBX, 8));
        sub0->add_use(LirOperand::preg_gpr(GPR::RBX, 8));
        sub0->add_use(LirOperand::imm(0, 8));
        bb->append_inst(std::move(sub0));

        auto setcc = std::make_unique<LirInst>(LirOpcode::Setcc);
        setcc->condition = x64::Condition::NE;
        setcc->add_def(LirOperand::preg_gpr(GPR::RAX, 1));
        bb->append_inst(std::move(setcc));

        PeepholeStats stats = run_lir_peephole_optimizations(fn);
        CHECK_EQ(stats.arithmetic_simplified, size_t(0));
        REQUIRE_EQ(bb->instructions.size(), size_t(2));
        CHECK(bb->instructions[0]->opcode == LirOpcode::Sub);
    }

    // Subtest C: add rax, 0 followed by Mov and Ret (flags are NOT live) -> IS erased
    {
        LirFunction fn;
        LirBlock* bb = fn.create_block("entry");

        auto add0 = std::make_unique<LirInst>(LirOpcode::Add);
        add0->add_def(LirOperand::preg_gpr(GPR::RAX, 8));
        add0->add_use(LirOperand::preg_gpr(GPR::RAX, 8));
        add0->add_use(LirOperand::imm(0, 8));
        bb->append_inst(std::move(add0));

        auto mov = std::make_unique<LirInst>(LirOpcode::Mov);
        mov->add_def(LirOperand::preg_gpr(GPR::RDX, 8));
        mov->add_use(LirOperand::imm(42, 8));
        bb->append_inst(std::move(mov));

        auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
        ret->add_use(LirOperand::preg_gpr(GPR::RDX, 8));
        bb->append_inst(std::move(ret));

        PeepholeStats stats = run_lir_peephole_optimizations(fn);
        CHECK_EQ(stats.arithmetic_simplified, size_t(1));
        REQUIRE_EQ(bb->instructions.size(), size_t(2));
        CHECK(bb->instructions[0]->opcode == LirOpcode::Mov);
    }
}

// =============================================================================
// Test 4: AArch64 DWARF CFI CFA Offset Adjustment with Outgoing Arguments
// =============================================================================
TEST_CASE("Phase 3 - AArch64 DWARF CFI CFA Offset Adjustment and Callee FPRs") {
    ObjectFile obj;
    obj.target = Target::aarch64_linux();

    CompiledFunctionInfo fn;
    fn.name = "test_aarch64_cfi";
    fn.text_offset = 0;
    fn.text_size = 256;

    fn.frame_info.is_leaf = false;
    fn.frame_info.total_frame_size = 96;
    fn.frame_info.outgoing_arg_space = 32; // outgoing_bytes = 32
    // CFA must be total_frame_size - outgoing_bytes = 96 - 32 = 64

    // Callee-saved GPR: X19
    fn.frame_info.saved_callee_gprs = (1u << 19);
    // Callee-saved FPRs: V8 and V9
    fn.frame_info.saved_callee_xmms = (1u << 8) | (1u << 9);

    obj.functions.push_back(fn);

    Section eh_frame_sec;
    eh_frame_sec.name = ".eh_frame";
    ElfCfiBuilder::build_eh_frame(obj, eh_frame_sec);

    REQUIRE(eh_frame_sec.data.size() > 32);

    uint32_t cie_len = *reinterpret_cast<const uint32_t*>(eh_frame_sec.data.data());
    size_t fde_offset = 4 + cie_len;
    fde_offset = (fde_offset + 7) & ~size_t(7);
    REQUIRE(fde_offset < eh_frame_sec.data.size());

    // Skip FDE header (4 len + 4 cie_ptr + 4 pc_begin + 4 pc_range + 1 aug_len = 17 bytes)
    const uint8_t* p = eh_frame_sec.data.data() + fde_offset + 17;
    const uint8_t* end = eh_frame_sec.data.data() + eh_frame_sec.data.size();

    // advance_loc (3 instructions because outgoing_arg_space > 0)
    REQUIRE(*p == (elf::DW_CFA_advance_loc | 3)); p++;

    // DW_CFA_def_cfa FP (29), cfa_offset (64)
    REQUIRE(*p == elf::DW_CFA_def_cfa); p++;
    REQUIRE(*p == 29); p++;
    uint64_t cfa_offset = decode_uleb128(p, end);
    // Verify CFA offset is 64 (adjusted by outgoing_arg_space 32), NOT 96!
    CHECK_EQ(cfa_offset, uint64_t(64));

    // DW_CFA_offset FP (29), fp_factored
    REQUIRE(*p == (elf::DW_CFA_offset | 29)); p++;
    uint64_t fp_factored = decode_uleb128(p, end);
    CHECK_EQ(fp_factored, uint64_t(8)); // 64 / 8 = 8

    // DW_CFA_offset LR (30), lr_factored
    REQUIRE(*p == (elf::DW_CFA_offset | 30)); p++;
    uint64_t lr_factored = decode_uleb128(p, end);
    CHECK_EQ(lr_factored, uint64_t(7)); // 8 - 1 = 7

    // Callee-saved GPR X19
    REQUIRE(*p == (elf::DW_CFA_offset | 19)); p++;
    uint64_t gpr_factored = decode_uleb128(p, end);
    // X19 is saved at FP + 16. CFA is FP + 64. Offset from CFA = 16 - 64 = -48. Factored = 48/8 = 6.
    CHECK_EQ(gpr_factored, uint64_t(6));

    // Callee-saved FPRs V8 and V9 via DW_CFA_offset_extended. The AArch64
    // DWARF register numbers of V0..V31 are 64..95, so V8 is 72 (64 would
    // be V0, a register the unwinder does not restore).
    REQUIRE(*p == elf::DW_CFA_offset_extended); p++;
    uint64_t dreg_v8 = decode_uleb128(p, end);
    CHECK_EQ(dreg_v8, uint64_t(72));
    uint64_t v8_factored = decode_uleb128(p, end);
    // V8 is saved at FP + 24 (16 header + 8 for X19). Offset from CFA = 24 - 64 = -40. Factored = 40/8 = 5.
    CHECK_EQ(v8_factored, uint64_t(5));

    REQUIRE(*p == elf::DW_CFA_offset_extended); p++;
    uint64_t dreg_v9 = decode_uleb128(p, end);
    CHECK_EQ(dreg_v9, uint64_t(73));
    uint64_t v9_factored = decode_uleb128(p, end);
    // V9 is saved at FP + 32. Offset from CFA = 32 - 64 = -32. Factored = 32/8 = 4.
    CHECK_EQ(v9_factored, uint64_t(4));
}

// =============================================================================
// Test 5: AArch64 Immediate Add/Sub Fallback Flag Splits Validation
// =============================================================================
TEST_CASE("Phase 3 - AArch64 Immediate Add/Sub Fallback Flag Splits Validation") {
    using aarch64::AArch64Encoder;
    using aarch64::CodeBuffer;
    using aarch64::GPR;

    CodeBuffer buf;
    AArch64Encoder enc(buf);

    // 1. Immediate <= 4095 with CMP: valid, does not throw
    enc.cmp(GPR::X0, 4095);
    CHECK_EQ(buf.size(), size_t(4));

    // 2. Immediate > 4095 shifted (e.g. 4096 = 1 << 12): valid, does not throw
    enc.cmp(GPR::X0, 4096);
    CHECK_EQ(buf.size(), size_t(8));

    // 3. Immediate > 4095 requiring split with CMP (XZR dst): throws invalid_argument
    bool threw_cmp = false;
    try {
        enc.cmp(GPR::X0, 5000);
    } catch (const std::invalid_argument&) {
        threw_cmp = true;
    }
    CHECK(threw_cmp);

    bool threw_cmn = false;
    try {
        enc.cmn(GPR::X0, 5000);
    } catch (const std::invalid_argument&) {
        threw_cmn = true;
    }
    CHECK(threw_cmn);

    // 4. Non-flag-setting add with immediate > 4095: valid, splits properly
    size_t prev_sz = buf.size();
    enc.add(GPR::X0, GPR::X1, 5000);
    CHECK_EQ(buf.size() - prev_sz, size_t(8)); // splits into 2 instructions (8 bytes)
}
