#include "test_framework.hpp"
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/aarch64_logical_imm.hpp>
#include <brass/target/aarch64/aarch64_baseline_emit.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/code_buffer.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include <brass/target/elf_so_writer.hpp>
#include <vector>
#include <cstring>
#include <stdexcept>

using namespace brass;
using namespace brass::codegen;
using namespace brass::aarch64;
using namespace brass::object;
using namespace brass::target;

#define CHECK_THROWS_AS(expr, exc_type) \
    do { \
        bool _threw = false; \
        try { \
            (expr); \
        } catch (const exc_type&) { \
            _threw = true; \
        } \
        CHECK(_threw); \
    } while ((void)0, 0)

namespace {

uint16_t read_u16_le(const uint8_t* p) {
    return static_cast<uint16_t>(p[0] | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_u64_le(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= (static_cast<uint64_t>(p[i]) << (i * 8));
    }
    return v;
}

} // namespace

// =============================================================================
// Deliverable 1: AArch64 SP Manipulation Extended Register Encoding
// =============================================================================

TEST_CASE("AArch64 Backend Hardening - SP Manipulation Extended Register Encoding") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);

    // 1. ADD SP, SP, X16
    enc.add(GPR::SP, GPR::SP, GPR::X16);
    // 2. SUB SP, SP, X16
    enc.sub(GPR::SP, GPR::SP, GPR::X16);
    // 3. ADD X0, SP, X1
    enc.add(GPR::X0, GPR::SP, GPR::X1);
    // 4. SUB X0, SP, X1
    enc.sub(GPR::X0, GPR::SP, GPR::X1);
    // 5. ADD32 SP, SP, X1 (WSP, WSP, W1)
    enc.add32(GPR::SP, GPR::SP, GPR::X1);
    // 6. SUB32 SP, SP, X1 (WSP, WSP, W1)
    enc.sub32(GPR::SP, GPR::SP, GPR::X1);
    // 7. Standard ADD X0, X1, X2 (non-SP, shifted register format)
    enc.add(GPR::X0, GPR::X1, GPR::X2);
    // 8. Standard SUB X0, X1, X2 (non-SP, shifted register format)
    enc.sub(GPR::X0, GPR::X1, GPR::X2);

    REQUIRE_EQ(buf.size(), size_t(32));
    const uint8_t* p = buf.data();

    // 1. ADD SP, SP, X16 -> extended register format: 0x8B3063FF
    CHECK_EQ(read_u32_le(p + 0), 0x8B3063FFu);
    // 2. SUB SP, SP, X16 -> extended register format: 0xCB3063FF
    CHECK_EQ(read_u32_le(p + 4), 0xCB3063FFu);
    // 3. ADD X0, SP, X1 -> extended register format: 0x8B2163E0
    CHECK_EQ(read_u32_le(p + 8), 0x8B2163E0u);
    // 4. SUB X0, SP, X1 -> extended register format: 0xCB2163E0
    CHECK_EQ(read_u32_le(p + 12), 0xCB2163E0u);
    // 5. ADD32 WSP, WSP, W1 -> 32-bit extended format: 0x0B2143FF
    CHECK_EQ(read_u32_le(p + 16), 0x0B2143FFu);
    // 6. SUB32 WSP, WSP, W1 -> 32-bit extended format: 0x4B2143FF
    CHECK_EQ(read_u32_le(p + 20), 0x4B2143FFu);
    // 7. Standard ADD X0, X1, X2 -> shifted register format: 0x8B020020
    CHECK_EQ(read_u32_le(p + 24), 0x8B020020u);
    // 8. Standard SUB X0, X1, X2 -> shifted register format: 0xCB020020
    CHECK_EQ(read_u32_le(p + 28), 0xCB020020u);
}

// =============================================================================
// Deliverable 2: Baseline JIT Frame Size Clamping and Large Frame Emission
// =============================================================================

TEST_CASE("AArch64 Backend Hardening - Baseline JIT Frame Size Clamping and Large Frame Emission") {
    // Case 1: Frame size <= 504 bytes (frame_size = 496 bytes: 16 header + 472 alloca + 8 result = 496).
    // Valid for pre-indexed STP and post-indexed LDP within 7-bit signed scaled imm [-512, 504].
    {
        Module mod("test_clamp_496");
        Function* fn = mod.create_function("fn_496", Type::void_type(), {});
        Builder b(mod);
        b.set_function(fn);
        b.append_block("entry");
        b.build_alloca(472, 16);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        auto compiled = compile_baseline_aarch64(*fn, Target::aarch64_linux());
        REQUIRE(compiled.is_valid());
        const uint8_t* code = static_cast<const uint8_t*>(compiled.entry_point());
        // Prologue: stp fp, lr, [sp, #-496]! -> 0xA9A17BFD
        uint32_t first_inst = read_u32_le(code);
        CHECK_EQ(first_inst, 0xA9A17BFDu);
    }

    // Case 2: Exact boundary frame_size = 512 bytes (16 header + 488 alloca + 8 result = 512 bytes).
    // Clamped to frame_size <= 504, avoiding LDP post-index immediate +512 overflow.
    // Emits separate SUB/ADD SP instead.
    {
        Module mod("test_clamp_512");
        Function* fn = mod.create_function("fn_512", Type::void_type(), {});
        Builder b(mod);
        b.set_function(fn);
        b.append_block("entry");
        b.build_alloca(488, 16);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        auto compiled = compile_baseline_aarch64(*fn, Target::aarch64_linux());
        REQUIRE(compiled.is_valid());
        const uint8_t* code = static_cast<const uint8_t*>(compiled.entry_point());

        // Prologue
        uint32_t inst0 = read_u32_le(code);
        uint32_t inst1 = read_u32_le(code + 4);
        CHECK_EQ(inst0, 0xD10803FFu); // sub sp, sp, #512
        CHECK_EQ(inst1, 0xA9007BFDu); // stp fp, lr, [sp, #0]

        // Epilogue before ret (0xD65F03C0)
        size_t ret_idx = 0;
        for (size_t i = 0; i + 4 <= compiled.code_size(); i += 4) {
            if (read_u32_le(code + i) == 0xD65F03C0u) {
                ret_idx = i;
                break;
            }
        }
        REQUIRE(ret_idx >= 8);
        uint32_t ldp_inst = read_u32_le(code + ret_idx - 8);
        uint32_t add_inst = read_u32_le(code + ret_idx - 4);
        CHECK_EQ(ldp_inst, 0xA9407BFDu); // ldp fp, lr, [sp, #0]
        CHECK_EQ(add_inst, 0x910803FFu); // add sp, sp, #512
    }

    // Case 3: Large frame > 4095 bytes (frame_size = 5008 bytes).
    // Materializes size into X16 and uses extended register format SUB/ADD SP.
    {
        Module mod("test_large_frame");
        Function* fn = mod.create_function("fn_large", Type::void_type(), {});
        Builder b(mod);
        b.set_function(fn);
        b.append_block("entry");
        b.build_alloca(4984, 16);
        b.build_ret_void();
        fn->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*fn));

        auto compiled = compile_baseline_aarch64(*fn, Target::aarch64_linux());
        REQUIRE(compiled.is_valid());
        const uint8_t* code = static_cast<const uint8_t*>(compiled.entry_point());

        // Prologue must use sub sp, sp, x16 (0xCB3063FF)
        bool found_sub_sp = false;
        for (size_t i = 0; i < 24 && i + 4 <= compiled.code_size(); i += 4) {
            if (read_u32_le(code + i) == 0xCB3063FFu) {
                found_sub_sp = true;
                break;
            }
        }
        CHECK(found_sub_sp);

        // Epilogue must use add sp, sp, x16 (0x8B3063FF)
        bool found_add_sp = false;
        for (size_t i = 0; i + 4 <= compiled.code_size(); i += 4) {
            if (read_u32_le(code + i) == 0x8B3063FFu) {
                found_add_sp = true;
                break;
            }
        }
        CHECK(found_add_sp);
    }
}

// =============================================================================
// Deliverable 3: ARMv8 Logical Immediate Algorithm & Exhaustive Properties
// =============================================================================

TEST_CASE("AArch64 Backend Hardening - ARMv8 Logical Immediate Algorithm and Exhaustive Properties") {
    uint32_t n = 0, immr = 0, imms = 0;

    // 1. Invalid masks (all zeros, all ones, irregular bit patterns)
    CHECK_FALSE(AArch64Encoder::encode_logical_immediate(0, true, n, immr, imms));
    CHECK_FALSE(AArch64Encoder::encode_logical_immediate(~0ULL, true, n, immr, imms));
    CHECK_FALSE(AArch64Encoder::encode_logical_immediate(0, false, n, immr, imms));
    CHECK_FALSE(AArch64Encoder::encode_logical_immediate(0xFFFFFFFFULL, false, n, immr, imms));
    CHECK_FALSE(AArch64Encoder::encode_logical_immediate(0x101ULL, true, n, immr, imms));
    CHECK_FALSE(AArch64Encoder::encode_logical_immediate(0x12345678ULL, true, n, immr, imms));
    CHECK_FALSE(AArch64Encoder::encode_logical_immediate(0x5555555555555554ULL, true, n, immr, imms));

    // 2. Repeating power-of-two patterns (64-bit)
    // 0xFF (width 64, run 8, R=0) -> N=1, immr=0, imms=7
    CHECK(AArch64Encoder::encode_logical_immediate(0xFFULL, true, n, immr, imms));
    CHECK_EQ(n, 1u);
    CHECK_EQ(immr, 0u);
    CHECK_EQ(imms, 7u);

    // 0xFFFF (width 64, run 16, R=0) -> N=1, immr=0, imms=15
    CHECK(AArch64Encoder::encode_logical_immediate(0xFFFFULL, true, n, immr, imms));
    CHECK_EQ(n, 1u);
    CHECK_EQ(immr, 0u);
    CHECK_EQ(imms, 15u);

    // 0x0000FFFF0000FFFFULL (esize 32, run 16) -> N=0
    CHECK(AArch64Encoder::encode_logical_immediate(0x0000FFFF0000FFFFULL, true, n, immr, imms));
    CHECK_EQ(n, 0u);

    // 0x00FF00FF00FF00FFULL (esize 16, run 8) -> N=0
    CHECK(AArch64Encoder::encode_logical_immediate(0x00FF00FF00FF00FFULL, true, n, immr, imms));
    CHECK_EQ(n, 0u);

    // 0x0F0F0F0F0F0F0F0FULL (esize 8, run 4) -> N=0
    CHECK(AArch64Encoder::encode_logical_immediate(0x0F0F0F0F0F0F0F0FULL, true, n, immr, imms));
    CHECK_EQ(n, 0u);

    // 0x3333333333333333ULL (esize 4, run 2) -> N=0
    CHECK(AArch64Encoder::encode_logical_immediate(0x3333333333333333ULL, true, n, immr, imms));
    CHECK_EQ(n, 0u);

    // 0x5555555555555555ULL (esize 2, run 1) -> N=0
    CHECK(AArch64Encoder::encode_logical_immediate(0x5555555555555555ULL, true, n, immr, imms));
    CHECK_EQ(n, 0u);

    // Rotated bitmask (bit 63 and bit 0 set)
    CHECK(AArch64Encoder::encode_logical_immediate(0x8000000000000001ULL, true, n, immr, imms));
    CHECK_EQ(n, 1u);

    // Single bit set and single bit clear across 64 bits
    for (int i = 0; i < 64; ++i) {
        CHECK(AArch64Encoder::encode_logical_immediate(1ULL << i, true, n, immr, imms));
        CHECK(AArch64Encoder::encode_logical_immediate(~(1ULL << i), true, n, immr, imms));
    }

    // 32-bit mode
    CHECK(AArch64Encoder::encode_logical_immediate(0xFFULL, false, n, immr, imms));
    CHECK_EQ(n, 0u);
    CHECK(AArch64Encoder::encode_logical_immediate(0x00FF00FFULL, false, n, immr, imms));
    CHECK_EQ(n, 0u);
    CHECK(AArch64Encoder::encode_logical_immediate(0x55555555ULL, false, n, immr, imms));
    CHECK_EQ(n, 0u);
}

// =============================================================================
// Deliverable 3 (cont): Logical Immediate Instruction Emission and ALU Integration
// =============================================================================

TEST_CASE("AArch64 Backend Hardening - Logical Immediate Instruction Emission and ALU Integration") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);

    // 1. Single-instruction immediate emission
    enc.and_imm(GPR::X0, GPR::X1, 0xFF);
    enc.orr_imm(GPR::X2, GPR::X3, 0xFFFF);
    enc.eor_imm(GPR::X4, GPR::X5, 0x0F0F0F0F0F0F0F0FULL);
    enc.ands_imm(GPR::X6, GPR::X7, 0x5555555555555555ULL);
    enc.tst_imm(GPR::X8, 0x3333333333333333ULL);

    // 32-bit versions
    enc.and32_imm(GPR::X0, GPR::X1, 0xFF);
    enc.orr32_imm(GPR::X2, GPR::X3, 0xFF);
    enc.eor32_imm(GPR::X4, GPR::X5, 0xFF);
    enc.ands32_imm(GPR::X6, GPR::X7, 0xFF);
    enc.tst32_imm(GPR::X8, 0xFF);

    CHECK_EQ(buf.size(), size_t(40)); // 10 instructions * 4 bytes
    const uint8_t* p = buf.data();

    // AND X0, X1, #0xFF -> 0x92401C20
    CHECK_EQ(read_u32_le(p + 0), 0x92401C20u);

    // Invalid immediate throws std::invalid_argument
    CHECK_THROWS_AS(enc.and_imm(GPR::X0, GPR::X1, 0x12345678ULL), std::invalid_argument);
    CHECK_THROWS_AS(enc.orr_imm(GPR::X0, GPR::X1, 0), std::invalid_argument);

    // 2. ALU Emission test via AArch64EmitContext:
    // When imm is valid bitmask (0xFF), And emits 1 instruction (0x92401C20).
    {
        LirFunction fn;
        fn.name = "alu_valid_imm";
        fn.frame.total_frame_size = 16;
        fn.frame.is_leaf = true;
        auto bb = std::make_unique<LirBlock>(0, "entry");

        auto inst = std::make_unique<LirInst>(LirOpcode::And);
        inst->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        inst->add_use(LirOperand::imm(0xFF, 8));
        bb->append_inst(std::move(inst));
        fn.blocks.push_back(std::move(bb));

        AArch64EmitContext emitter(fn, Target::aarch64_linux());
        AArch64CompilationResult res = emitter.compile();
        bool found_and_imm = false;
        for (size_t i = 0; i + 4 <= res.code_buffer.size(); i += 4) {
            if (read_u32_le(res.code_buffer.data() + i) == 0x92401C20u) {
                found_and_imm = true;
                break;
            }
        }
        CHECK(found_and_imm);
    }

    // When imm is NOT a valid bitmask (0x12345678), falls back to mov + and
    {
        LirFunction fn;
        fn.name = "alu_fallback_imm";
        fn.frame.total_frame_size = 16;
        fn.frame.is_leaf = true;
        auto bb = std::make_unique<LirBlock>(0, "entry");

        auto inst = std::make_unique<LirInst>(LirOpcode::And);
        inst->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
        inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
        inst->add_use(LirOperand::imm(0x12345678, 8));
        bb->append_inst(std::move(inst));
        fn.blocks.push_back(std::move(bb));

        AArch64EmitContext emitter(fn, Target::aarch64_linux());
        AArch64CompilationResult res = emitter.compile();
        bool found_reg_and = false;
        for (size_t i = 0; i + 4 <= res.code_buffer.size(); i += 4) {
            if (read_u32_le(res.code_buffer.data() + i) == 0x8A100020u) {
                found_reg_and = true;
                break;
            }
        }
        CHECK(found_reg_and);
    }
}

// =============================================================================
// Deliverable 4: emit_add_sub_imm Range and Flag Safety
// =============================================================================

TEST_CASE("AArch64 Backend Hardening - emit_add_sub_imm Range and Flag Safety") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);

    // 1. Small imm <= 4095: single instruction
    enc.add(GPR::X0, GPR::X1, 100);
    CHECK_EQ(buf.size(), size_t(4));

    // 2. Shifted imm (val <= 4095 << 12 with low 12 bits zero): single instruction
    buf.clear();
    enc.add(GPR::X0, GPR::X1, 0x1000);
    CHECK_EQ(buf.size(), size_t(4));

    // 3. 24-bit imm (non-zero low and shifted parts): two instructions
    buf.clear();
    enc.add(GPR::X0, GPR::X1, 0x123456);
    CHECK_EQ(buf.size(), size_t(8));

    // 4. Large imm with non-zero bits in 24..31: must not truncate bits 24..31.
    // Materializes into X16 and emits register ADD.
    buf.clear();
    enc.add(GPR::X0, GPR::X1, 0x01234567ULL);
    CHECK(buf.size() >= 8);
    uint32_t last_inst = read_u32_le(buf.data() + buf.size() - 4);
    CHECK_EQ(last_inst, 0x8B100020u); // add x0, x1, x16

    // 5. Flag safety with CMP/CMN and ADDS
    // When dst == XZR and imm requires multiple parts: throws std::invalid_argument (Phase 3 invariant)
    CHECK_THROWS_AS(enc.cmp(GPR::X0, 5000), std::invalid_argument);
    CHECK_THROWS_AS(enc.cmn(GPR::X0, 5000), std::invalid_argument);

    // When dst != XZR and set_flags == true with multi-part imm:
    // Materializes into X16 and uses single ADDS X0, X1, X16 to prevent flag clobbering
    buf.clear();
    enc.adds(GPR::X0, GPR::X1, 5000);
    CHECK(buf.size() >= 8);
    uint32_t adds_inst = read_u32_le(buf.data() + buf.size() - 4);
    CHECK_EQ(adds_inst, 0xAB100020u); // adds x0, x1, x16
}

// =============================================================================
// Deliverable 5: Backward Branch Bounds Checking
// =============================================================================

TEST_CASE("AArch64 Backend Hardening - Backward Branch Bounds Checking") {
    // 1. Within 19-bit conditional branch range
    {
        CodeBuffer buf;
        AArch64Encoder enc(buf);
        Label target = buf.create_label();
        buf.bind(target);
        enc.nop();
        enc.nop();
        enc.b(Condition::EQ, target);
        enc.cbz(GPR::X0, target);
        enc.cbnz(GPR::X0, target);
        CHECK_EQ(buf.size(), size_t(20));
    }

    // 2. Exceeding 19-bit conditional branch range (> 1MB backward)
    {
        CodeBuffer buf;
        AArch64Encoder enc(buf);
        Label target = buf.create_label();
        buf.bind(target);

        // Advance buffer by 1,048,576 + 4 bytes
        std::vector<uint8_t> padding(1048576 + 4, 0);
        buf.emit_bytes(padding);

        // Must throw std::runtime_error due to exceeding 19-bit branch range
        CHECK_THROWS_AS(enc.b(Condition::EQ, target), std::runtime_error);
        CHECK_THROWS_AS(enc.cbz(GPR::X0, target), std::runtime_error);
        CHECK_THROWS_AS(enc.cbnz(GPR::X0, target), std::runtime_error);
    }

    // 3. Within 26-bit unconditional branch range
    {
        CodeBuffer buf;
        AArch64Encoder enc(buf);
        Label target = buf.create_label();
        buf.bind(target);
        enc.nop();
        enc.b(target);
        enc.bl(target);
        CHECK_EQ(buf.size(), size_t(12));
    }
}

// =============================================================================
// Deliverable 6: ADRP Relocation Mapping (R_AARCH64_ADR_PREL_PG_HI21)
// =============================================================================

TEST_CASE("AArch64 Backend Hardening - ADRP Relocation Mapping (R_AARCH64_ADR_PREL_PG_HI21)") {
    ObjectFile obj;
    obj.target = Target::aarch64_linux();
    obj.get_or_create_section(".text", SectionKind::Text, SectionFlags::Read | SectionFlags::Execute | SectionFlags::Alloc, 16);
    Section* sec = obj.get_section(".text");
    REQUIRE(sec != nullptr);

    // ADRP X0, 0 -> 0x90000000
    sec->emit32(0x90000000u);
    ObjectRelocation r;
    r.offset = 0;
    r.kind = RelocKind::AdrPage21;
    r.symbol_name = "target_sym";
    r.addend = 0;
    sec->relocations.push_back(r);

    ObjectSymbol sym;
    sym.name = "target_sym";
    sym.binding = SymbolBinding::Global;
    sym.type = SymbolType::Function;
    sym.section_index = SECTION_UNDEF;
    obj.symbols.push_back(sym);

    std::vector<uint8_t> elf_bytes = emit_elf_object(obj);
    REQUIRE(!elf_bytes.empty());

    const uint8_t* ehdr = elf_bytes.data();
    uint64_t e_shoff = read_u64_le(ehdr + 40);
    uint16_t e_shnum = read_u16_le(ehdr + 60);
    uint16_t e_shstrndx = read_u16_le(ehdr + 62);

    const uint8_t* shstr_shdr = ehdr + e_shoff + e_shstrndx * 64;
    uint64_t shstr_offset = read_u64_le(shstr_shdr + 24);
    const char* shstrtab = reinterpret_cast<const char*>(ehdr + shstr_offset);

    bool found_rela_text = false;
    for (uint16_t i = 0; i < e_shnum; ++i) {
        const uint8_t* shdr = ehdr + e_shoff + i * 64;
        uint32_t sh_name = read_u32_le(shdr + 0);
        uint32_t sh_type = read_u32_le(shdr + 4);
        const char* name = shstrtab + sh_name;

        if (std::strcmp(name, ".rela.text") == 0) {
            found_rela_text = true;
            CHECK_EQ(sh_type, elf::SHT_RELA);
            uint64_t rela_off = read_u64_le(shdr + 24);
            const uint8_t* rela = ehdr + rela_off;
            uint64_t r_info = read_u64_le(rela + 8);
            uint32_t r_type = static_cast<uint32_t>(r_info & 0xFFFFFFFF);
            // Must map to R_AARCH64_ADR_PREL_PG_HI21 (275)
            CHECK_EQ(r_type, 275u);
        }
    }
    CHECK(found_rela_text);

    // Verify Mach-O and COFF writers handle RelocKind::AdrPage21 gracefully
    std::vector<uint8_t> macho_bytes = emit_macho_object(obj);
    CHECK(!macho_bytes.empty());
    std::vector<uint8_t> coff_bytes = emit_coff_object(obj);
    CHECK(!coff_bytes.empty());
}

// =============================================================================
// Deliverable 6 (cont): ElfSoWriter 64KB Page Alignment
// =============================================================================

TEST_CASE("AArch64 Backend Hardening - ElfSoWriter 64KB Page Alignment") {
    Module mod("test_so_page_align");
    Function* fn = mod.create_function("so_func", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    b.build_ret(x);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    ObjectFile obj = compile_module_to_object(mod, Target::aarch64_linux());
    ElfSoOptions opts;
    opts.soname = "libtest_so.so";

    // 1. Default page alignment for AArch64 should be 64KB (0x10000)
    std::vector<uint8_t> so_bytes = ElfSoWriter::emit(obj, opts);
    REQUIRE(!so_bytes.empty());

    const uint8_t* ehdr = so_bytes.data();
    uint64_t e_phoff = read_u64_le(ehdr + 32);
    uint16_t e_phnum = read_u16_le(ehdr + 56);
    REQUIRE(e_phnum >= 2);

    for (uint16_t i = 0; i < e_phnum; ++i) {
        const uint8_t* phdr = ehdr + e_phoff + i * 56;
        uint32_t p_type = read_u32_le(phdr + 0);
        if (p_type == elf64::PT_LOAD) {
            uint64_t p_vaddr = read_u64_le(phdr + 16);
            uint64_t p_align = read_u64_le(phdr + 48);
            CHECK_EQ(p_align, 0x10000u);
            CHECK_EQ(p_vaddr % 0x10000u, 0u);
        }
    }

    // 2. Explicit custom page_size (e.g. 128KB = 0x20000)
    opts.page_size = 0x20000;
    std::vector<uint8_t> so_bytes_custom = ElfSoWriter::emit(obj, opts);
    REQUIRE(!so_bytes_custom.empty());
    const uint8_t* ehdr_c = so_bytes_custom.data();
    uint64_t e_phoff_c = read_u64_le(ehdr_c + 32);
    uint16_t e_phnum_c = read_u16_le(ehdr_c + 56);
    for (uint16_t i = 0; i < e_phnum_c; ++i) {
        const uint8_t* phdr = ehdr_c + e_phoff_c + i * 56;
        uint32_t p_type = read_u32_le(phdr + 0);
        if (p_type == elf64::PT_LOAD) {
            uint64_t p_vaddr = read_u64_le(phdr + 16);
            uint64_t p_align = read_u64_le(phdr + 48);
            CHECK_EQ(p_align, 0x20000u);
            CHECK_EQ(p_vaddr % 0x20000u, 0u);
        }
    }
}
