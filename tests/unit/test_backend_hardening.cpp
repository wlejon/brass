#include "test_framework.hpp"
#include <brass/target/x64/x64_frame.hpp>
#include <brass/target/x64/x64_encoder.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/target/target.hpp>
#include <brass/codegen/sched_dag.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/debug/dwarf_emitter.hpp>
#include <brass/debug/source_loc.hpp>
#include <vector>
#include <cstring>

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;
using namespace brass::object;
using namespace brass::debug;

// =============================================================================
// Task 1: 16-byte stack alignment & SEH compliant epilogue in X64FrameLayout
// =============================================================================

TEST_CASE("Backend Hardening - X64FrameLayout stack alignment with 1 saved GPR and vector spill slots") {
    CallingConvention cc = CallingConvention::win64();
    FrameInfo frame;
    // 1 saved GPR (RBX) - odd count (8 bytes)
    frame.saved_callee_gprs = (1u << static_cast<int>(GPR::RBX));
    frame.saved_callee_xmms = 0; // No saved XMMs
    frame.num_spill_slots = 4;
    frame.local_frame_bytes = 16;
    frame.has_calls = true;

    X64FrameLayout::compute_layout(frame, cc);

    // Total frame size must be 16-byte aligned
    CHECK_EQ(frame.total_frame_size % 16, size_t(0));

    // Because num_spill_slots > 0 and local_frame_bytes > 0, gpr_offset must be aligned to 16 bytes.
    // The disp from RBP for slot 1 (offset 16 from base) must be a multiple of 16.
    MemAddress slot_addr1 = X64FrameLayout::spill_slot_address(1, frame);
    CHECK_EQ((-slot_addr1.disp) % 16, 0);

    // Also check local frame address starts on a 16-byte boundary:
    MemAddress local_addr = X64FrameLayout::local_frame_address(0, frame);
    CHECK_EQ((-local_addr.disp) % 16, 0);

    // Verify epilogue emits add rsp, total_frame_size instead of mov rsp, rbp
    CodeBuffer buf;
    X64Encoder enc(buf);
    X64FrameLayout::emit_epilogue(enc, frame, cc);
    const auto& code = buf.bytes();
    REQUIRE(!code.empty());
    CHECK_EQ(code.back(), uint8_t(0xc3)); // ret

    // Verify mov rsp, rbp (0x48, 0x89, 0xec) is NOT present in non-leaf frame epilogue
    bool has_mov_rsp_rbp = false;
    for (size_t i = 0; i + 2 < code.size(); ++i) {
        if (code[i] == 0x48 && code[i+1] == 0x89 && code[i+2] == 0xec) {
            has_mov_rsp_rbp = true;
            break;
        }
    }
    CHECK(!has_mov_rsp_rbp);
}

// =============================================================================
// Task 2: Sched DAG flag dependency safety for NOT instructions
// =============================================================================

TEST_CASE("Backend Hardening - sched_dag flag safety for NOT instructions") {
    LirInst inst_not(LirOpcode::Not);
    LirInst inst_not32(LirOpcode::Not32);
    LirInst inst_neg(LirOpcode::Neg);
    LirInst inst_add(LirOpcode::Add);

    // NOT in x86/x64 does not define flags
    CHECK_FALSE(instruction_defines_flags(inst_not));
    CHECK_FALSE(instruction_defines_flags(inst_not32));

    // NEG and ADD do define flags
    CHECK(instruction_defines_flags(inst_neg));
    CHECK(instruction_defines_flags(inst_add));
}

// =============================================================================
// Task 3: Linear scan coalesce hints between PReg and VReg
// =============================================================================

TEST_CASE("Backend Hardening - Linear scan coalesce hint between PReg use and VReg def") {
    CallingConvention cc = CallingConvention::sysv64();

    // 1. Def is VReg, Use is PReg (mov v0, RSI)
    {
        LirFunction fn;
        auto block = std::make_unique<LirBlock>(0, "entry");

        VReg v0 = fn.allocate_vreg(RegClass::GPR, 8);
        // mov v0, RSI
        auto inst0 = std::make_unique<LirInst>(LirOpcode::Mov);
        inst0->add_def(LirOperand::vreg(v0, 8));
        inst0->add_use(LirOperand::preg(PReg::gpr(GPR::RSI), 8));
        block->append_inst(std::move(inst0));

        // add v0, 42
        auto inst1 = std::make_unique<LirInst>(LirOpcode::Add);
        inst1->add_def(LirOperand::vreg(v0, 8));
        inst1->add_use(LirOperand::vreg(v0, 8));
        inst1->add_use(LirOperand::imm(42, 8));
        block->append_inst(std::move(inst1));

        // ret v0
        auto inst2 = std::make_unique<LirInst>(LirOpcode::Ret);
        inst2->add_use(LirOperand::vreg(v0, 8));
        block->append_inst(std::move(inst2));

        fn.blocks.push_back(std::move(block));

        LivenessAnalysis liveness(fn);
        liveness.run();

        LinearScanAllocator allocator(fn, liveness, cc);
        allocator.allocate();

        const auto* int_v0 = liveness.get_interval(v0);
        REQUIRE(int_v0 != nullptr);
        CHECK(int_v0->assigned_preg.is_valid());
        CHECK_EQ(int_v0->assigned_preg.as_gpr(), GPR::RSI);
    }

    // 2. Def is PReg, Use is VReg (mov RDI, v1)
    {
        LirFunction fn;
        auto block = std::make_unique<LirBlock>(0, "entry");

        VReg v1 = fn.allocate_vreg(RegClass::GPR, 8);
        // mov v1, 10
        auto r0 = std::make_unique<LirInst>(LirOpcode::Mov);
        r0->add_def(LirOperand::vreg(v1, 8));
        r0->add_use(LirOperand::imm(10, 8));
        block->append_inst(std::move(r0));

        // mov RDI, v1
        auto r1 = std::make_unique<LirInst>(LirOpcode::Mov);
        r1->add_def(LirOperand::preg(PReg::gpr(GPR::RDI), 8));
        r1->add_use(LirOperand::vreg(v1, 8));
        block->append_inst(std::move(r1));

        // ret
        auto r2 = std::make_unique<LirInst>(LirOpcode::Ret);
        block->append_inst(std::move(r2));

        fn.blocks.push_back(std::move(block));

        LivenessAnalysis liveness(fn);
        liveness.run();

        LinearScanAllocator allocator(fn, liveness, cc);
        allocator.allocate();

        const auto* int_v1 = liveness.get_interval(v1);
        REQUIRE(int_v1 != nullptr);
        CHECK(int_v1->assigned_preg.is_valid());
        CHECK_EQ(int_v1->assigned_preg.as_gpr(), GPR::RDI);
    }
}

// =============================================================================
// Task 4: SysV call lowering emits %al setup
// =============================================================================

TEST_CASE("Backend Hardening - SysV call lowering emits %al vector count setup") {
    Module mod;
    mod.create_function("vector_callee", Type::f64(), {Type::f64(), Type::f64()});
    Function* caller = mod.create_function("caller_fn", Type::f64(), {Type::f64(), Type::f64()});
    Builder b(mod);
    b.set_function(caller);
    BasicBlock* entry = b.append_block("entry");
    Value* arg0 = b.add_block_param(entry, Type::f64());
    Value* arg1 = b.add_block_param(entry, Type::f64());
    Value* ret = b.build_call("vector_callee", Type::f64(), {arg0, arg1});
    b.build_ret(ret);
    caller->rebuild_cfg_predecessors();
    CHECK(verify_function(*caller));

    // Lower with SysV calling convention
    X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
    auto lir = isel.lower(*caller);
    REQUIRE(lir != nullptr);
    REQUIRE(!lir->blocks.empty());

    bool found_al_setup = false;
    bool call_has_rax_use = false;
    for (const auto& inst : lir->blocks[0]->instructions) {
        if (inst->opcode == LirOpcode::Mov32 && !inst->defs.empty() && inst->defs[0].is_preg() &&
            inst->defs[0].preg_val.as_gpr() == GPR::RAX && !inst->uses.empty() && inst->uses[0].is_imm_int()) {
            if (inst->uses[0].imm_int == 2) {
                found_al_setup = true;
            }
        }
        if (inst->opcode == LirOpcode::Call) {
            for (const auto& u : inst->uses) {
                if (u.is_preg() && u.preg_val.as_gpr() == GPR::RAX) {
                    call_has_rax_use = true;
                }
            }
        }
    }
    CHECK(found_al_setup);
    CHECK(call_has_rax_use);
}

// =============================================================================
// Task 5: COFF relocation inline addends
// =============================================================================

TEST_CASE("Backend Hardening - COFF writer patches inline addends") {
    ObjectFile obj;
    obj.target = Target::x64_windows();

    Section sec;
    sec.name = ".text";
    sec.kind = SectionKind::Text;
    sec.flags = SectionFlags::Read | SectionFlags::Execute;
    sec.data.resize(32, 0);

    // 32-bit relocation with addend 0x12345678 at offset 4
    ObjectRelocation r32;
    r32.offset = 4;
    r32.symbol_name = "sym32";
    r32.kind = RelocKind::SecRel32;
    r32.addend = 0x12345678;
    sec.relocations.push_back(r32);

    // 64-bit relocation with addend 0x0123456789ABCDEFLL at offset 16
    ObjectRelocation r64;
    r64.offset = 16;
    r64.symbol_name = "sym64";
    r64.kind = RelocKind::Abs64;
    r64.addend = 0x0123456789ABCDEFLL;
    sec.relocations.push_back(r64);

    ObjectSymbol s32;
    s32.name = "sym32";
    s32.section_index = 0;
    s32.value = 0;
    obj.symbols.push_back(s32);

    ObjectSymbol s64;
    s64.name = "sym64";
    s64.section_index = 0;
    s64.value = 0;
    obj.symbols.push_back(s64);

    obj.sections.push_back(std::move(sec));

    CoffWriter writer(obj);
    std::vector<uint8_t> output = writer.write();
    REQUIRE(!output.empty());

    // Section header is at offset 20 (IMAGE_FILE_HEADER = 20 bytes)
    // PointerToRawData is at section header offset 20 (file offset 40)
    REQUIRE(output.size() >= 44);
    uint32_t raw_data_ptr = *reinterpret_cast<const uint32_t*>(&output[40]);
    REQUIRE(output.size() >= raw_data_ptr + 32);

    uint32_t patched_val32 = *reinterpret_cast<const uint32_t*>(&output[raw_data_ptr + 4]);
    CHECK_EQ(patched_val32, uint32_t(0x12345678));

    uint64_t patched_val64 = *reinterpret_cast<const uint64_t*>(&output[raw_data_ptr + 16]);
    CHECK_EQ(patched_val64, uint64_t(0x0123456789ABCDEFULL));
}

// =============================================================================
// Task 6: DWARF Frame Base Register on AArch64
// =============================================================================

TEST_CASE("Backend Hardening - DWARF frame base emits DW_OP_reg29 on AArch64") {
    // 1. AArch64 target emits DW_OP_reg29 (0x6d)
    {
        ObjectFile obj;
        obj.target = Target::aarch64_linux();

        Section text;
        text.name = ".text";
        text.kind = SectionKind::Text;
        text.flags = SectionFlags::Read | SectionFlags::Execute;
        text.data.resize(64, 0);
        obj.sections.push_back(std::move(text));

        CompiledFunctionInfo fn_info;
        fn_info.name = "aarch64_fn";
        fn_info.text_offset = 0;
        fn_info.text_size = 64;
        obj.functions.push_back(fn_info);

        FunctionDebugTable tbl("aarch64_fn", 64);
        tbl.set_decl_file(1);
        tbl.set_decl_line(10);
        obj.debug_tables.push_back(tbl);
        obj.debug_context.get_or_add_file("test.brass");

        DwarfOptions opts;
        DwarfEmitter::emit(obj, opts);

        const Section* info_sec = obj.get_section(".debug_info");
        REQUIRE(info_sec != nullptr);
        REQUIRE(!info_sec->data.empty());

        bool found_reg29 = false;
        bool found_reg6 = false;
        for (uint8_t byte : info_sec->data) {
            if (byte == dwarf::DW_OP_reg29) found_reg29 = true;
            if (byte == dwarf::DW_OP_reg6) found_reg6 = true;
        }
        CHECK(found_reg29);
        CHECK(!found_reg6);
    }

    // 2. x64 target emits DW_OP_reg6 (0x56)
    {
        ObjectFile obj;
        obj.target = Target::x64_linux();

        Section text;
        text.name = ".text";
        text.kind = SectionKind::Text;
        text.flags = SectionFlags::Read | SectionFlags::Execute;
        text.data.resize(64, 0);
        obj.sections.push_back(std::move(text));

        CompiledFunctionInfo fn_info;
        fn_info.name = "x64_fn";
        fn_info.text_offset = 0;
        fn_info.text_size = 64;
        obj.functions.push_back(fn_info);

        FunctionDebugTable tbl("x64_fn", 64);
        tbl.set_decl_file(1);
        tbl.set_decl_line(10);
        obj.debug_tables.push_back(tbl);
        obj.debug_context.get_or_add_file("test.brass");

        DwarfOptions opts;
        DwarfEmitter::emit(obj, opts);

        const Section* info_sec = obj.get_section(".debug_info");
        REQUIRE(info_sec != nullptr);
        REQUIRE(!info_sec->data.empty());

        bool found_reg6 = false;
        for (uint8_t byte : info_sec->data) {
            if (byte == dwarf::DW_OP_reg6) found_reg6 = true;
        }
        CHECK(found_reg6);
    }
}
