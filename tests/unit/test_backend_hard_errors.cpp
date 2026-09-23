#include "test_framework.hpp"
#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/codegen/unsupported_operation.hpp>
#include <brass/runtime/patcher.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/target/target.hpp>
#include <brass/target/x64/x64_registers.hpp>
#include <cstring>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace brass;
using namespace brass::codegen;

namespace {

// A one-block x64 LIR function holding `op` (with a register def and use)
// followed by ret.
LirFunction x64_function_with(LirOpcode op) {
    LirFunction fn;
    fn.name = "gap";
    fn.calling_conv = CallingConvention::sysv64();
    LirBlock* bb = fn.create_block("entry");
    auto inst = std::make_unique<LirInst>(op);
    inst->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 8));
    inst->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 8));
    inst->add_use(LirOperand::preg_gpr(x64::GPR::RCX, 8));
    bb->append_inst(std::move(inst));
    bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
    return fn;
}

// Emits `op`; returns the UnsupportedOperation's stage, or "" if none was
// thrown.
std::string x64_emit_error_stage(LirOpcode op) {
    LirFunction fn = x64_function_with(op);
    try {
        EmitContext ctx(fn, Target::x64_linux());
        (void)ctx.compile();
    } catch (const UnsupportedOperation& e) {
        CHECK(std::string(e.what()).find(std::string(to_string(op))) != std::string::npos);
        CHECK_EQ(e.operation(), std::string(to_string(op)));
        return e.stage();
    }
    return {};
}

} // namespace

TEST_CASE("Hard errors - x64 emitter rejects LIR it has no encoding for") {
    // AArch64-only LIR: before, these were an assert in debug and a ud2 in
    // the output in release.
    CHECK_EQ(x64_emit_error_stage(LirOpcode::Adds), std::string("x64 emit"));
    CHECK_EQ(x64_emit_error_stage(LirOpcode::Subs32), std::string("x64 emit"));
    CHECK_EQ(x64_emit_error_stage(LirOpcode::Smulh), std::string("x64 emit"));
    // A supported opcode still compiles.
    CHECK_EQ(x64_emit_error_stage(LirOpcode::Add), std::string());
}

TEST_CASE("Hard errors - UnsupportedOperation is a runtime_error naming stage and operation") {
    bool caught = false;
    try {
        throw_unsupported("aarch64 isel", "fancy_op");
    } catch (const std::runtime_error& e) {
        CHECK_EQ(std::string(e.what()), std::string("aarch64 isel: unsupported operation 'fancy_op'"));
        caught = true;
    }
    CHECK(caught);
}

// =============================================================================
// Code patcher: the architecture is stated, not guessed from the bytes
// =============================================================================

TEST_CASE("Patcher - x64 call site must be CALL/JMP rel32") {
    alignas(64) uint8_t buf[128] = {};
    uint8_t* target_area = buf + 64;  // near: within rel32 reach
    // Not a call: nop
    buf[0] = 0x90;
    CHECK_FALSE(runtime::patch_call_site(runtime::CodeArch::X64, buf, target_area));
    // CALL rel32
    buf[0] = 0xE8;
    REQUIRE(runtime::patch_call_site(runtime::CodeArch::X64, buf, target_area));
    int32_t disp = 0;
    std::memcpy(&disp, buf + 1, 4);
    CHECK_EQ(static_cast<intptr_t>(disp), reinterpret_cast<intptr_t>(target_area) - reinterpret_cast<intptr_t>(buf + 5));
    // JMP rel32
    buf[0] = 0xE9;
    CHECK(runtime::patch_call_site(runtime::CodeArch::X64, buf, target_area + 16));
    std::memcpy(&disp, buf + 1, 4);
    CHECK_EQ(static_cast<intptr_t>(disp), reinterpret_cast<intptr_t>(target_area + 16) - reinterpret_cast<intptr_t>(buf + 5));
}

TEST_CASE("Patcher - AArch64 call site must be B/BL, target word aligned and in range") {
    alignas(64) uint32_t words[16] = {};
    words[0] = 0xD503201Fu;  // nop
    CHECK_FALSE(runtime::patch_call_site(runtime::CodeArch::AArch64, &words[0], &words[8]));

    words[0] = 0x94000000u;  // bl .
    REQUIRE(runtime::patch_call_site(runtime::CodeArch::AArch64, &words[0], &words[8]));
    CHECK_EQ(words[0], 0x94000008u);  // bl +32 (8 words)

    words[1] = 0x14000000u;  // b .
    REQUIRE(runtime::patch_call_site(runtime::CodeArch::AArch64, &words[1], &words[0]));
    CHECK_EQ(words[1], 0x17FFFFFFu);  // b -4

    // Misaligned target
    CHECK_FALSE(runtime::patch_call_site(runtime::CodeArch::AArch64, &words[0],
                                         reinterpret_cast<const uint8_t*>(&words[8]) + 2));
}

TEST_CASE("Patcher - the same bytes patch differently per stated architecture") {
    // E8 00 00 94 is both an x64 CALL rel32 (opcode byte E8) and, read as a
    // little-endian word, an AArch64 BL (0x940000E8). Only the stated
    // architecture decides which instruction is rewritten.
    alignas(64) uint8_t a[64] = {0xE8, 0x00, 0x00, 0x94, 0x00};
    alignas(64) uint8_t b[64] = {0xE8, 0x00, 0x00, 0x94, 0x00};

    REQUIRE(runtime::patch_call_site(runtime::CodeArch::X64, a, a + 32));
    CHECK_EQ(a[0], uint8_t(0xE8));              // opcode byte kept
    int32_t disp = 0;
    std::memcpy(&disp, a + 1, 4);
    CHECK_EQ(disp, int32_t(32 - 5));

    REQUIRE(runtime::patch_call_site(runtime::CodeArch::AArch64, b, b + 32));
    uint32_t word = 0;
    std::memcpy(&word, b, 4);
    CHECK_EQ(word, 0x94000008u);                // bl +32
}

TEST_CASE("Patcher - registry call patch takes the architecture") {
    alignas(64) uint32_t words[16] = {};
    words[2] = 0x94000000u;
    runtime::PatchRegistry reg;
    reg.register_site(runtime::PatchSite("callee", runtime::PatchKind::Call, 8, 0, 4));
    CHECK(reg.patch_call(runtime::CodeArch::AArch64, words, "callee", &words[6]));
    CHECK_EQ(words[2], 0x94000004u);
    // As x64 the first byte (0x04) is not E8/E9: refused, nothing written.
    CHECK_FALSE(reg.patch_call(runtime::CodeArch::X64, words, "callee", &words[6]));
    CHECK_EQ(words[2], 0x94000004u);
}

// =============================================================================
// W^X JIT memory
// =============================================================================

TEST_CASE("W^X - sealed JIT code is read-execute, never writable and executable") {
    const size_t page = jit_system_page_size();
    JitMemoryBlock block(page, page);
    REQUIRE(block.is_valid());
    uint8_t* code = block.data();
    CHECK_FALSE(is_jit_code_address(code));
    code[0] = 0xC3;  // writable before sealing

    REQUIRE(block.make_executable_read_only(page));
    CHECK(is_jit_code_address(code));

#if defined(_WIN32)
    MEMORY_BASIC_INFORMATION info{};
    REQUIRE(VirtualQuery(code, &info, sizeof(info)) == sizeof(info));
    CHECK_EQ(info.Protect, static_cast<DWORD>(PAGE_EXECUTE_READ));
    if (block.size() > page) {
        // The data pages after the code stay read-write, not executable.
        REQUIRE(VirtualQuery(code + page, &info, sizeof(info)) == sizeof(info));
        CHECK_EQ(info.Protect, static_cast<DWORD>(PAGE_READWRITE));
    }
#endif

    REQUIRE(block.make_read_write());
    CHECK_FALSE(is_jit_code_address(code));
#if defined(_WIN32)
    REQUIRE(VirtualQuery(code, &info, sizeof(info)) == sizeof(info));
    CHECK_EQ(info.Protect, static_cast<DWORD>(PAGE_READWRITE));
#endif
    code[0] = 0xCC;  // writable again
    CHECK_EQ(code[0], uint8_t(0xCC));
}
