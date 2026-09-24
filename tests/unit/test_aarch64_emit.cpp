#include "test_framework.hpp"
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_baseline_emit.hpp>
#include <brass/target/aarch64/aarch64_registers.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <brass/object/macho_writer.hpp>
#include <cstring>
#include <vector>

using namespace brass;
using namespace brass::codegen;
using namespace brass::aarch64;
using namespace brass::object;

namespace {

uint32_t read_u32_le(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

} // namespace

// =============================================================================
// Test 1: Direct LirFunction Emission (Arithmetic, Logic, Moves)
// =============================================================================
TEST_CASE("AArch64 Emit - Direct LirFunction Emission (Arithmetic & Logic)") {
    LirFunction fn;
    fn.name = "calc_direct";
    fn.frame.total_frame_size = 32;
    fn.frame.num_spill_slots = 2;
    fn.frame.has_calls = false;
    fn.frame.is_leaf = false;

    auto bb = std::make_unique<LirBlock>(0, "entry");

    // add x0, x1, x2
    auto add_inst = std::make_unique<LirInst>(LirOpcode::Add);
    add_inst->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    add_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
    add_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X2, 8));
    bb->append_inst(std::move(add_inst));

    // sub x3, x4, x5
    auto sub_inst = std::make_unique<LirInst>(LirOpcode::Sub);
    sub_inst->add_def(LirOperand::preg_aarch64_gpr(GPR::X3, 8));
    sub_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X4, 8));
    sub_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X5, 8));
    bb->append_inst(std::move(sub_inst));

    // imul x0, x3
    auto mul_inst = std::make_unique<LirInst>(LirOpcode::Imul);
    mul_inst->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    mul_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X3, 8));
    bb->append_inst(std::move(mul_inst));

    // and x0, x1
    auto and_inst = std::make_unique<LirInst>(LirOpcode::And);
    and_inst->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    and_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
    bb->append_inst(std::move(and_inst));

    // ret
    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    bb->append_inst(std::move(ret_inst));

    fn.blocks.push_back(std::move(bb));

    AArch64EmitContext emitter(fn, Target::aarch64_linux());
    AArch64CompilationResult res = emitter.compile();

    CHECK(res.code_buffer.size() > 0);
    CHECK_EQ(res.code_buffer.size() % 4, size_t(0));

    // Check first instruction is STP FP, LR, [SP, #-32]! -> 0xA9BE7BFD
    const uint8_t* code = res.code_buffer.data();
    uint32_t first_inst = read_u32_le(code);
    CHECK_EQ(first_inst, 0xA9BE7BFDu); // stp fp, lr, [sp, #-32]!

    // Check second instruction is MOV FP, SP -> 0x910003FD (add fp, sp, #0)
    uint32_t second_inst = read_u32_le(code + 4);
    CHECK_EQ(second_inst, 0x910003FDu); // mov fp, sp

    // Check last instruction is RET -> 0xD65F03C0
    uint32_t last_inst = read_u32_le(code + res.code_buffer.size() - 4);
    CHECK_EQ(last_inst, 0xD65F03C0u); // ret
}

namespace {

// True when `seq` appears as consecutive instruction words in `code`.
bool contains_words(const uint8_t* code, size_t size, const std::vector<uint32_t>& seq) {
    for (size_t at = 0; at + seq.size() * 4 <= size; at += 4) {
        bool hit = true;
        for (size_t k = 0; k < seq.size() && hit; ++k) hit = read_u32_le(code + at + 4 * k) == seq[k];
        if (hit) return true;
    }
    return false;
}

constexpr uint32_t kBrkDivZero = 0xD4200000u | (static_cast<uint32_t>(kBrkIntegerDivideByZero) << 5);

} // namespace

// MIR defines division by zero as a program error (x64 faults with #DE);
// AArch64's sdiv/udiv would quietly return 0, so every division is guarded:
// cbnz divisor, +8; brk #0xd0; div. (Signed MIN / -1 needs no guard: sdiv
// already wraps to MIN, and msub turns that into MIN % -1 == 0.)
TEST_CASE("AArch64 Emit - Integer division traps on a zero divisor") {
    for (LirOpcode op : {LirOpcode::Idiv, LirOpcode::Idiv32, LirOpcode::Div, LirOpcode::Div32}) {
        LirFunction fn;
        fn.name = "div_guard";
        fn.frame.total_frame_size = 16;
        auto bb = std::make_unique<LirBlock>(0, "entry");
        const uint8_t width = (op == LirOpcode::Idiv || op == LirOpcode::Div) ? 8 : 4;
        auto div = std::make_unique<LirInst>(op);
        div->add_def(LirOperand::preg_aarch64_gpr(GPR::X0, width));
        div->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, width));
        div->add_use(LirOperand::preg_aarch64_gpr(GPR::X2, width));
        bb->append_inst(std::move(div));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
        fn.blocks.push_back(std::move(bb));

        AArch64EmitContext emitter(fn, Target::aarch64_linux());
        AArch64CompilationResult res = emitter.compile();
        const bool wide = width == 8;
        const bool is_signed = op == LirOpcode::Idiv || op == LirOpcode::Idiv32;
        const uint32_t cbnz_x2 = (wide ? 0xB5000000u : 0x35000000u) | (2u << 5) | 2u;
        const uint32_t div_x0_x1_x2 = (wide ? 0x9AC00800u : 0x1AC00800u) | (is_signed ? 0x400u : 0u) |
                                      (2u << 16) | (1u << 5);
        CHECK(contains_words(res.code_buffer.data(), res.code_buffer.size(), {cbnz_x2, kBrkDivZero, div_x0_x1_x2}));
    }
}

TEST_CASE("AArch64 Baseline JIT - Integer division and remainder trap on a zero divisor") {
    Module mod;
    Function* fn = mod.create_function("div_rem", Type::i64(), {Type::i64(), Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    Value* y = b.add_block_param(entry, Type::i64());
    Value* q = b.build_sdiv(x, y);
    Value* r = b.build_umod(q, y);
    b.build_ret(r);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler(Target::aarch64_linux());
    BaselineCompiledFunction compiled = compiler.compile(*fn);
    REQUIRE(compiled.is_valid());
    const auto* code = reinterpret_cast<const uint8_t*>(compiled.entry_point());
    const uint32_t cbnz_x1 = 0xB5000000u | (2u << 5) | 1u;
    // sdiv x0, x0, x1 and udiv x2, x0, x1 (the remainder's quotient).
    CHECK(contains_words(code, compiled.code_size(), {cbnz_x1, kBrkDivZero, 0x9AC10C00u}));
    CHECK(contains_words(code, compiled.code_size(), {cbnz_x1, kBrkDivZero, 0x9AC10802u}));
}

// =============================================================================
// Test 2: Control Flow, Branches, and Relocations
// =============================================================================
TEST_CASE("AArch64 Emit - Control Flow, Branches, and Relocations") {
    LirFunction fn;
    fn.name = "cfg_branch_test";
    fn.frame.total_frame_size = 16;
    fn.frame.num_spill_slots = 0;
    fn.frame.has_calls = true;
    fn.frame.is_leaf = false;

    // Block 0: entry
    auto b0 = std::make_unique<LirBlock>(0, "entry");
    auto cmp_inst = std::make_unique<LirInst>(LirOpcode::Cmp32);
    cmp_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 4));
    cmp_inst->add_use(LirOperand::imm(0, 4));
    b0->append_inst(std::move(cmp_inst));

    auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc->condition = brass::x64::Condition::E;
    jcc->add_use(LirOperand::label(2)); // target block 2
    b0->append_inst(std::move(jcc));

    auto jmp = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp->add_use(LirOperand::label(1)); // fallthrough to block 1
    b0->append_inst(std::move(jmp));
    fn.blocks.push_back(std::move(b0));

    // Block 1: call external function
    auto b1 = std::make_unique<LirBlock>(1, "call_blk");
    auto call_inst = std::make_unique<LirInst>(LirOpcode::Call);
    call_inst->add_use(LirOperand::symbol("external_target"));
    call_inst->safepoint_id = 42;
    b1->append_inst(std::move(call_inst));

    auto sp_inst = std::make_unique<LirInst>(LirOpcode::Safepoint);
    sp_inst->safepoint_id = 43;
    b1->append_inst(std::move(sp_inst));

    auto ret1 = std::make_unique<LirInst>(LirOpcode::Ret);
    b1->append_inst(std::move(ret1));
    fn.blocks.push_back(std::move(b1));

    // Block 2: exit
    auto b2 = std::make_unique<LirBlock>(2, "exit_blk");
    auto ret2 = std::make_unique<LirInst>(LirOpcode::Ret);
    b2->append_inst(std::move(ret2));
    fn.blocks.push_back(std::move(b2));

    AArch64EmitContext emitter(fn, Target::aarch64_linux());
    AArch64CompilationResult res = emitter.compile();

    CHECK(res.code_buffer.size() > 0);
    CHECK_EQ(res.block_offsets.size(), size_t(3));
    CHECK_EQ(res.safepoints.size(), size_t(1));
    CHECK(res.stack_map.records.size() >= 2);

    // Relocations should contain Call26 for external_target
    const auto& relocs = res.code_buffer.relocations();
    CHECK(!relocs.empty());
    bool found_reloc = false;
    for (const auto& r : relocs) {
        if (r.symbol_name == "external_target" && r.kind == RelocationKind::Call26) {
            found_reloc = true;
            break;
        }
    }
    CHECK(found_reloc);
}

// =============================================================================
// Test 3: Floating Point and SIMD Vector Operations
// =============================================================================
TEST_CASE("AArch64 Emit - Floating Point and SIMD Vector Ops") {
    LirFunction fn;
    fn.name = "fp_vector_test";
    fn.frame.total_frame_size = 32;
    fn.frame.is_leaf = true;

    auto bb = std::make_unique<LirBlock>(0, "entry");

    // Addsd v0, v1
    auto addsd = std::make_unique<LirInst>(LirOpcode::Addsd);
    addsd->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
    addsd->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 8));
    bb->append_inst(std::move(addsd));

    // Subsd v2, v3
    auto subsd = std::make_unique<LirInst>(LirOpcode::Subsd);
    subsd->add_def(LirOperand::preg_aarch64_fpr(FPR::V2, 8));
    subsd->add_use(LirOperand::preg_aarch64_fpr(FPR::V3, 8));
    bb->append_inst(std::move(subsd));

    // Mulsd v0, v2
    auto mulsd = std::make_unique<LirInst>(LirOpcode::Mulsd);
    mulsd->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
    mulsd->add_use(LirOperand::preg_aarch64_fpr(FPR::V2, 8));
    bb->append_inst(std::move(mulsd));

    // Divsd v0, v1
    auto divsd = std::make_unique<LirInst>(LirOpcode::Divsd);
    divsd->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 8));
    divsd->add_use(LirOperand::preg_aarch64_fpr(FPR::V1, 8));
    bb->append_inst(std::move(divsd));

    // Addps v4, v5 (128-bit vector 4x float)
    auto addps = std::make_unique<LirInst>(LirOpcode::Addps);
    addps->add_def(LirOperand::preg_aarch64_fpr(FPR::V4, 16));
    addps->add_use(LirOperand::preg_aarch64_fpr(FPR::V5, 16));
    bb->append_inst(std::move(addps));

    // Subps v6, v7
    auto subps = std::make_unique<LirInst>(LirOpcode::Subps);
    subps->add_def(LirOperand::preg_aarch64_fpr(FPR::V6, 16));
    subps->add_use(LirOperand::preg_aarch64_fpr(FPR::V7, 16));
    bb->append_inst(std::move(subps));

    // Fabs32 v0, [x0]
    auto fabs32_mem = std::make_unique<LirInst>(LirOpcode::Fabs32);
    fabs32_mem->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 4));
    fabs32_mem->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::X0), 0, 4));
    bb->append_inst(std::move(fabs32_mem));

    // Sqrtss v1, [x0 + 8]
    auto sqrtss_mem = std::make_unique<LirInst>(LirOpcode::Sqrtss);
    sqrtss_mem->add_def(LirOperand::preg_aarch64_fpr(FPR::V1, 4));
    sqrtss_mem->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::X0), 8, 4));
    bb->append_inst(std::move(sqrtss_mem));

    // Floor32 v2, [x0 + 16]
    auto floor32_mem = std::make_unique<LirInst>(LirOpcode::Floor32);
    floor32_mem->add_def(LirOperand::preg_aarch64_fpr(FPR::V2, 4));
    floor32_mem->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::X0), 16, 4));
    bb->append_inst(std::move(floor32_mem));

    // Cvtsi2sd v3, [x0 + 24]
    auto cvtsi2sd_mem = std::make_unique<LirInst>(LirOpcode::Cvtsi2sd);
    cvtsi2sd_mem->add_def(LirOperand::preg_aarch64_fpr(FPR::V3, 8));
    cvtsi2sd_mem->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::X0), 24, 8));
    bb->append_inst(std::move(cvtsi2sd_mem));

    // Cvttsd2si x1, [x0 + 32]
    auto cvttsd2si_mem = std::make_unique<LirInst>(LirOpcode::Cvttsd2si);
    cvttsd2si_mem->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 8));
    cvttsd2si_mem->add_use(LirOperand::mem(PReg::aarch64_gpr(GPR::X0), 32, 8));
    bb->append_inst(std::move(cvttsd2si_mem));

    // Xorps v0, v0 (zero float reg)
    auto xorps_inst = std::make_unique<LirInst>(LirOpcode::Xorps);
    xorps_inst->add_def(LirOperand::preg_aarch64_fpr(FPR::V0, 4));
    xorps_inst->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 4));
    xorps_inst->add_use(LirOperand::preg_aarch64_fpr(FPR::V0, 4));
    bb->append_inst(std::move(xorps_inst));

    // Not32 w1, w1
    auto not32_inst = std::make_unique<LirInst>(LirOpcode::Not32);
    not32_inst->add_def(LirOperand::preg_aarch64_gpr(GPR::X1, 4));
    not32_inst->add_use(LirOperand::preg_aarch64_gpr(GPR::X1, 4));
    bb->append_inst(std::move(not32_inst));

    // Ret
    auto ret = std::make_unique<LirInst>(LirOpcode::Ret);
    bb->append_inst(std::move(ret));

    fn.blocks.push_back(std::move(bb));

    AArch64EmitContext emitter(fn, Target::aarch64_macos());
    AArch64CompilationResult res = emitter.compile();

    CHECK(res.code_buffer.size() > 0);
    CHECK_EQ(res.code_buffer.size() % 4, size_t(0));
}

// =============================================================================
// Test 4: ObjectWriter Integration - Full Pipeline (ELF AArch64)
// =============================================================================
TEST_CASE("AArch64 ObjectWriter - Full Pipeline (ELF AArch64)") {
    Module mod("test_aarch64_elf");
    Function* fn = mod.create_function("calc_elf", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());

    Value* sum = b.build_add(a, c);
    Value* prod = b.build_mul(sum, a);
    b.build_ret(prod);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    Target target = Target::aarch64_linux();
    ObjectFile obj = compile_module_to_object(mod, target);

    CHECK_EQ(obj.target.arch(), Arch::aarch64);

    const auto* text_sec = obj.get_section(".text");
    REQUIRE(text_sec != nullptr);
    CHECK(text_sec->data.size() > 0);
    CHECK_EQ(text_sec->data.size() % 4, size_t(0));

    // Verify ELF object serialization
    std::vector<uint8_t> elf_bytes = emit_elf_object(obj);
    CHECK(elf_bytes.size() >= 64);

    // ELF Magic
    CHECK_EQ(elf_bytes[0], 0x7F);
    CHECK_EQ(elf_bytes[1], 'E');
    CHECK_EQ(elf_bytes[2], 'L');
    CHECK_EQ(elf_bytes[3], 'F');
    // Class 2 = 64-bit
    CHECK_EQ(elf_bytes[4], 2);
    // Data 1 = 2's complement, little-endian
    CHECK_EQ(elf_bytes[5], 1);

    // e_machine at offset 18: EM_AARCH64 = 183 = 0xB7
    uint16_t machine = static_cast<uint16_t>(elf_bytes[18] | (static_cast<uint16_t>(elf_bytes[19]) << 8));
    CHECK_EQ(machine, static_cast<uint16_t>(183)); // EM_AARCH64
}

// =============================================================================
// Test 5: ObjectWriter Integration - Full Pipeline (Mach-O AArch64)
// =============================================================================
TEST_CASE("AArch64 ObjectWriter - Full Pipeline (Mach-O AArch64)") {
    Module mod("test_aarch64_macho");
    Function* fn = mod.create_function("calc_macho", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());

    Value* sum = b.build_add(a, c);
    Value* diff = b.build_sub(sum, c);
    b.build_ret(diff);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    Target target = Target::aarch64_macos();
    ObjectFile obj = compile_module_to_object(mod, target);

    CHECK_EQ(obj.target.arch(), Arch::aarch64);

    const auto* text_sec = obj.get_section(".text");
    REQUIRE(text_sec != nullptr);
    CHECK(text_sec->data.size() > 0);
    CHECK_EQ(text_sec->data.size() % 4, size_t(0));

    // Verify Mach-O object serialization
    std::vector<uint8_t> macho_bytes = emit_macho_object(obj);
    REQUIRE(macho_bytes.size() >= 32);

    // Mach-O 64-bit magic: 0xFEEDFACF
    uint32_t magic = read_u32_le(macho_bytes.data());
    CHECK_EQ(magic, 0xFEEDFACFu);

    // cputype at offset 4: CPU_TYPE_ARM64 = 0x0100000C
    uint32_t cputype = read_u32_le(macho_bytes.data() + 4);
    CHECK_EQ(cputype, 0x0100000Cu);
}

// =============================================================================
// Test 6: AArch64 Baseline JIT - Compilation, Code Structure, and ABI
// =============================================================================
TEST_CASE("AArch64 Baseline JIT - Compilation and Code Structure") {
    Module mod;
    Function* fn = mod.create_function("calc_baseline", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* d = b.add_block_param(entry, Type::i64());

    Value* sum = b.build_add(a, c);
    Value* prod = b.build_mul(sum, d);
    Value* diff = b.build_sub(prod, a);
    Value* div = b.build_sdiv(diff, c);
    Value* rem = b.build_smod(div, d);
    b.build_ret(rem);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler(Target::aarch64_linux());
    BaselineCompiledFunction compiled = compiler.compile(*fn);

    CHECK(compiled.is_valid());
    CHECK(compiled.code_size() > 0);
    CHECK_EQ(compiled.code_size() % 4, size_t(0));
    CHECK_EQ(compiled.name(), "calc_baseline");
    CHECK_EQ(compiled.param_types().size(), size_t(3));

    // Verify prologue is stp fp, lr, [sp, #-frame_size]!
    const uint8_t* code = reinterpret_cast<const uint8_t*>(compiled.entry_point());
    uint32_t p_inst = read_u32_le(code);
    // Opcode bits for stp (64-bit pre-index): [31..23] == 0b101010011 -> 0xA9800000
    CHECK_EQ(p_inst & 0xFFC00000u, 0xA9800000u);
    // mov fp, sp -> 0x910003FD (add fp, sp, #0)
    uint32_t fp_mov = read_u32_le(code + 4);
    CHECK_EQ(fp_mov, 0x910003FDu);
}

// =============================================================================
// read_sp: the stack-limit check every bronze prologue makes (sp below the
// thread's limit -> RangeError)
// =============================================================================
namespace {

// f(limit) = read_sp < limit ? 1 : 0
Function* build_sp_below(Module& mod, const char* name) {
    Function* fn = mod.create_function(name, Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* limit = b.add_block_param(entry, Type::i64());
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
    return fn;
}

bool any_word(const uint8_t* code, size_t size, uint32_t mask, uint32_t value) {
    for (size_t i = 0; i + 4 <= size; i += 4) {
        if ((read_u32_le(code + i) & mask) == value) return true;
    }
    return false;
}

// add xN, sp, #0 (`mov xN, sp`) and the shifted-register `cmp xzr, xM`
// (subs xzr, xzr, xM): register 31 is XZR there, never SP.
constexpr uint32_t kMovFromSpMask = 0xFFFFFFE0u, kMovFromSp = 0x910003E0u;
constexpr uint32_t kCmpXzrMask = 0xFFE0FFFFu, kCmpXzr = 0xEB0003FFu;

} // namespace

TEST_CASE("AArch64 Emit - read_sp is copied out of SP, never allocated in register 31") {
    Module mod("sp_below");
    Function* fn = build_sp_below(mod, "sp_below");
    REQUIRE(verify_function(*fn));
    ObjectFile obj = compile_module_to_object(mod, Target::aarch64_macos());
    const auto* text = obj.get_section(".text");
    REQUIRE(text != nullptr);
    CHECK(any_word(text->data.data(), text->data.size(), kMovFromSpMask, kMovFromSp));
    CHECK(!any_word(text->data.data(), text->data.size(), kCmpXzrMask, kCmpXzr));
}

// =============================================================================
// Test 7: AArch64 Baseline JIT - Loop Execution (CFG with Parameters)
// =============================================================================
TEST_CASE("AArch64 Baseline JIT - Loop Structure and Block Args") {
    Module mod;
    Function* fn = mod.create_function("sum_loop", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* n = b.add_block_param(entry, Type::i64());

    BasicBlock* loop_header = b.create_block("loop_header");
    Value* i_val = b.add_block_param(loop_header, Type::i64());
    Value* acc_val = b.add_block_param(loop_header, Type::i64());

    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit = b.create_block("exit");

    // Entry branches to loop_header with i=0, acc=0
    b.position_at_end(entry);
    b.build_br(loop_header, {b.build_iconst_i64(0), b.build_iconst_i64(0)});

    // Header checks if i < n
    fn->append_block(loop_header);
    b.position_at_end(loop_header);
    Value* cond = b.build_slt(i_val, n);
    b.build_br_if(cond, loop_body, exit);

    // Body adds i to acc, increments i, jumps to header
    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* new_acc = b.build_add(acc_val, i_val);
    Value* new_i = b.build_add(i_val, b.build_iconst_i64(1));
    b.build_br(loop_header, {new_i, new_acc});

    // Exit returns acc
    fn->append_block(exit);
    b.position_at_end(exit);
    b.build_ret(acc_val);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler(Target::aarch64_macos());
    BaselineCompiledFunction compiled = compiler.compile(*fn);

    CHECK(compiled.is_valid());
    CHECK(compiled.code_size() > 0);
    CHECK_EQ(compiled.code_size() % 4, size_t(0));
}

// =============================================================================
// Test 8: AArch64 Baseline JIT - Switch, Select, and Memory Operations
// =============================================================================
TEST_CASE("AArch64 Baseline JIT - Switch, Select, and Memory") {
    Module mod;
    Function* fn = mod.create_function("mem_and_select", Type::i64(), {Type::i64(), Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ptr_val = b.add_block_param(entry, Type::i64());
    Value* idx_val = b.add_block_param(entry, Type::i64());
    Value* val = b.add_block_param(entry, Type::i64());

    // Store indexed
    b.build_store_indexed(Type::i64(), ptr_val, idx_val, 8, 0, val);

    // Load indexed
    Value* loaded = b.build_load_indexed(Type::i64(), ptr_val, idx_val, 8, 0);

    // Select
    Value* cond = b.build_eq(loaded, val);
    Value* res = b.build_select(cond, b.build_iconst_i64(100), b.build_iconst_i64(200));

    b.build_ret(res);

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler(Target::aarch64_linux());
    BaselineCompiledFunction compiled = compiler.compile(*fn);

    CHECK(compiled.is_valid());
    CHECK(compiled.code_size() > 0);
    CHECK_EQ(compiled.code_size() % 4, size_t(0));
}

// =============================================================================
// Test 9: AArch64 Baseline JIT - GC Roots and Stack Map Generation
// =============================================================================
TEST_CASE("AArch64 Baseline JIT - GC Roots and Stack Map Generation") {
    Module mod;
    Function* fn = mod.create_function("gc_safepoint_test", Type::void_type(), {Type::gcref(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* ref_param = b.add_block_param(entry, Type::gcref());
    (void)b.add_block_param(entry, Type::i64());

    // Safepoint site
    auto* sp1 = b.build_safepoint();
    sp1->set_site_id(101);

    // Write barrier
    b.build_write_barrier(ref_param, ref_param);

    // Second safepoint
    auto* sp2 = b.build_safepoint();
    sp2->set_site_id(102);

    b.build_ret_void();

    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    BaselineJitCompiler compiler(Target::aarch64_linux());
    BaselineCompiledFunction compiled = compiler.compile(*fn);

    CHECK(compiled.is_valid());
    const auto& sm = compiled.stack_map();
    CHECK_EQ(sm.function_name, "gc_safepoint_test");
    CHECK_EQ(sm.records.size(), size_t(2));

    for (size_t i = 0; i < sm.records.size(); ++i) {
        const auto& rec = sm.records[i];
        CHECK(rec.instruction_offset > 0);
        CHECK(rec.frame_size >= 16);
        // At least one GC root from ref_param
        CHECK_EQ(rec.roots.size(), size_t(1));
        const auto& root = rec.roots[0];
        // Location should be a positive frame slot relative to FP
        CHECK(root.kind == StackMapRootKind::FrameSlot);
        CHECK(root.offset_from_rbp > 0);
    }
}
