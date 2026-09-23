#include "test_framework.hpp"
#include <brass/runtime/patcher.hpp>
#include <brass/runtime/exception.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/verifier.hpp>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace brass;
using namespace brass::runtime;
using namespace brass::codegen;

// =============================================================================
// 1. Dynamic Call and Instruction Patching (ARM64 BL & B, Const Patching)
// =============================================================================
TEST_CASE("AArch64 Runtime - Call Patching BL Instruction") {
    // ARM64 BL instruction: opcode 0x94000000u | imm26
    alignas(4) uint32_t code[1024];
    std::memset(code, 0, sizeof(code));

    // Place an unlinked BL at code[0]
    code[0] = 0x94000000u;

    // Target is code[100] (offset +400 bytes, displacement = +100 words)
    void* call_site = &code[0];
    const void* target = &code[100];

    bool ok = patch_call_site(CodeArch::AArch64, call_site, target);
    CHECK(ok);

    uint32_t expected = 0x94000000u | 100u;
    CHECK_EQ(code[0], expected);

    // Negative displacement: target is code[0], call_site is code[50] (offset -200 bytes = -50 words)
    code[50] = 0x94000000u;
    void* call_site_back = &code[50];
    const void* target_back = &code[0];

    bool ok_back = patch_call_site(CodeArch::AArch64, call_site_back, target_back);
    CHECK(ok_back);

    uint32_t expected_back = 0x94000000u | (static_cast<uint32_t>(-50) & 0x03FFFFFFu);
    CHECK_EQ(code[50], expected_back);
}

TEST_CASE("AArch64 Runtime - Call Patching B Instruction") {
    // ARM64 B instruction: opcode 0x14000000u | imm26
    alignas(4) uint32_t code[512];
    std::memset(code, 0, sizeof(code));

    code[10] = 0x14000000u;
    void* call_site = &code[10];
    const void* target = &code[266]; // +256 words = +1024 bytes

    bool ok = patch_call_site(CodeArch::AArch64, call_site, target);
    CHECK(ok);

    uint32_t expected = 0x14000000u | 256u;
    CHECK_EQ(code[10], expected);
}

TEST_CASE("AArch64 Runtime - Call Patching Bounds and Alignment") {
    alignas(4) uint32_t code[16];
    code[0] = 0x94000000u;

    // 1. Unaligned target (disp not multiple of 4)
    const void* unaligned_target = reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(&code[0]) + 13);
    CHECK(!patch_call_site(CodeArch::AArch64, &code[0], unaligned_target));

    // 2. Out of range displacement (> +128MB or < -128MB)
    // 33554432 words = 134217728 bytes = 128MB
    intptr_t base_int = reinterpret_cast<intptr_t>(&code[0]);
    const void* too_far_forward = reinterpret_cast<const void*>(base_int + (33554432LL << 2));
    CHECK(!patch_call_site(CodeArch::AArch64, &code[0], too_far_forward));

    const void* too_far_backward = reinterpret_cast<const void*>(base_int + (-33554433LL << 2));
    CHECK(!patch_call_site(CodeArch::AArch64, &code[0], too_far_backward));
}

TEST_CASE("AArch64 Runtime - Const Patching With Cache Flush") {
    alignas(8) int32_t c32 = 42;
    alignas(8) int64_t c64 = 1000;

    CHECK(brass_patch_const32(&c32, 0x12345678));
    CHECK_EQ(c32, 0x12345678);

    CHECK(brass_patch_const64(&c64, 0x0123456789ABCDEFLL));
    CHECK_EQ(c64, 0x0123456789ABCDEFLL);

    // PatchRegistry integration
    PatchRegistry registry;
    PatchSite s32;
    s32.name = "site_32";
    s32.kind = PatchKind::Const32;
    s32.code_offset = 0;
    s32.imm_offset = 0;
    registry.register_site(s32);

    CHECK(registry.patch_const32(&c32, "site_32", 99999));
    CHECK_EQ(c32, 99999);

    PatchSite s_call;
    s_call.name = "site_bl";
    s_call.kind = PatchKind::Call;
    s_call.code_offset = 0;
    registry.register_site(s_call);

    alignas(4) uint32_t bl_inst = 0x94000000u;
    const void* target = reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(&bl_inst) + 80);
    CHECK(registry.patch_call(CodeArch::AArch64, &bl_inst, "site_bl", target));
    CHECK_EQ(bl_inst, 0x94000000u | 20u);
}

// =============================================================================
// 2. AAPCS64 Invocation Argument Partitioning & Stack Overflow
// =============================================================================
TEST_CASE("AArch64 Runtime - AAPCS64 Argument Partitioning (5 Ints, 5 Floats, 2 Vectors)") {
    std::vector<RuntimeValue> args;
    std::vector<Type> param_types;

    // 5 Ints
    for (int i = 0; i < 5; ++i) {
        args.push_back(RuntimeValue::from_i64(100 + i));
        param_types.push_back(Type::i64());
    }
    // 5 Floats
    for (int i = 0; i < 5; ++i) {
        args.push_back(RuntimeValue::from_f64(1.5 * (i + 1)));
        param_types.push_back(Type::f64());
    }
    // 2 Vectors (128-bit)
    for (int i = 0; i < 2; ++i) {
        alignas(16) uint8_t vbytes[16] = {0};
        vbytes[0] = static_cast<uint8_t>(i + 1);
        args.push_back(RuntimeValue::from_v128(Type::i32x4(), vbytes));
        param_types.push_back(Type::i32x4());
    }

    AArch64InvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    void* dummy_fn = reinterpret_cast<void*>(0x1000);

    partition_aarch64_invoke_args(args, &param_types, dummy_fn, invoke_args, stack_words);

    CHECK_EQ(invoke_args.target_fn, dummy_fn);

    // First 5 ints go into X0..X4
    for (int i = 0; i < 5; ++i) {
        CHECK_EQ(invoke_args.x[i], static_cast<uint64_t>(100 + i));
    }
    // X5..X7 should be 0
    for (int i = 5; i < 8; ++i) {
        CHECK_EQ(invoke_args.x[i], 0ULL);
    }

    // First 5 floats go into V0..V4
    for (int i = 0; i < 5; ++i) {
        double d = 0.0;
        std::memcpy(&d, invoke_args.v[i], sizeof(double));
        CHECK_EQ(d, 1.5 * (i + 1));
    }

    // 2 vectors go into V5 and V6
    CHECK_EQ(invoke_args.v[5][0], 1);
    CHECK_EQ(invoke_args.v[6][0], 2);

    // Total FPR count is 5 + 2 = 7 <= 8, so NO stack overflow
    CHECK_EQ(stack_words.size(), size_t(0));
    CHECK_EQ(invoke_args.stack_word_count, 0ULL);
}

TEST_CASE("AArch64 Runtime - AAPCS64 Argument Partitioning (Stack Overflow)") {
    std::vector<RuntimeValue> args;
    std::vector<Type> param_types;

    // 10 Ints (8 in registers X0..X7, 2 on stack)
    for (int i = 0; i < 10; ++i) {
        args.push_back(RuntimeValue::from_i64(i * 10));
        param_types.push_back(Type::i64());
    }
    // 10 Doubles (8 in registers V0..V7, 2 on stack)
    for (int i = 0; i < 10; ++i) {
        args.push_back(RuntimeValue::from_f64(i * 0.5));
        param_types.push_back(Type::f64());
    }

    AArch64InvokeArgs invoke_args;
    std::vector<uint64_t> stack_words;
    partition_aarch64_invoke_args(args, &param_types, nullptr, invoke_args, stack_words);

    // Verify registers
    for (int i = 0; i < 8; ++i) {
        CHECK_EQ(invoke_args.x[i], static_cast<uint64_t>(i * 10));
    }
    for (int i = 0; i < 8; ++i) {
        double d = 0.0;
        std::memcpy(&d, invoke_args.v[i], sizeof(double));
        CHECK_EQ(d, i * 0.5);
    }

    // Overflow: 2 ints (i=8, 9) + 2 doubles (i=8, 9) = 4 stack words
    CHECK_EQ(stack_words.size(), size_t(4));
    CHECK_EQ(stack_words.size() % 2, size_t(0)); // 16-byte alignment
    CHECK_EQ(stack_words[0], 80ULL);
    CHECK_EQ(stack_words[1], 90ULL);

    double d8 = 0.0, d9 = 0.0;
    std::memcpy(&d8, &stack_words[2], sizeof(double));
    std::memcpy(&d9, &stack_words[3], sizeof(double));
    CHECK_EQ(d8, 4.0);
    CHECK_EQ(d9, 4.5);
}

// Native AArch64 dynamic invocation execution test
#if defined(__aarch64__) || defined(_M_ARM64)
extern "C" double aarch64_native_sum_func(
    int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t i4,
    double d0, double d1, double d2, double d3, double d4,
    int64_t i5, int64_t i6, int64_t i7, int64_t i8, int64_t i9,
    double d5, double d6, double d7, double d8, double d9
) {
    return static_cast<double>(i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9) +
           (d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8 + d9);
}

TEST_CASE("AArch64 Runtime - Native Dynamic Invocation with Stack Overflow") {
    JitExecutionEngine jit;
    jit.register_external_symbol("native_sum", reinterpret_cast<void*>(&aarch64_native_sum_func));
    jit.register_function_signature("native_sum", Type::f64());

    std::vector<RuntimeValue> args;
    int64_t int_sum = 0;
    double dbl_sum = 0.0;

    for (int i = 0; i < 5; ++i) {
        args.push_back(RuntimeValue::from_i64(i + 1));
        int_sum += (i + 1);
    }
    for (int i = 0; i < 5; ++i) {
        args.push_back(RuntimeValue::from_f64((i + 1) * 0.25));
        dbl_sum += (i + 1) * 0.25;
    }
    for (int i = 5; i < 10; ++i) {
        args.push_back(RuntimeValue::from_i64(i + 1));
        int_sum += (i + 1);
    }
    for (int i = 5; i < 10; ++i) {
        args.push_back(RuntimeValue::from_f64((i + 1) * 0.25));
        dbl_sum += (i + 1) * 0.25;
    }

    RuntimeValue res = jit.invoke("native_sum", args);
    CHECK(res.is_f64());
    CHECK_EQ(res.as_f64(), static_cast<double>(int_sum) + dbl_sum);
}
#endif

// =============================================================================
// 3. Exception Unwinding, Throw, and Landing Pad Dispatch on AArch64
// =============================================================================
TEST_CASE("AArch64 Runtime - SavedRegisters ABI Structure Layout") {
    CHECK_EQ(sizeof(SavedRegisters), size_t(216));

    // x86_64 callee-saved (0..55)
    CHECK_EQ(offsetof(SavedRegisters, r15), size_t(0));
    CHECK_EQ(offsetof(SavedRegisters, r14), size_t(8));
    CHECK_EQ(offsetof(SavedRegisters, r13), size_t(16));
    CHECK_EQ(offsetof(SavedRegisters, r12), size_t(24));
    CHECK_EQ(offsetof(SavedRegisters, rdi), size_t(32));
    CHECK_EQ(offsetof(SavedRegisters, rsi), size_t(40));
    CHECK_EQ(offsetof(SavedRegisters, rbx), size_t(48));

    // AArch64 GPR callee-saved (56..151)
    CHECK_EQ(offsetof(SavedRegisters, x19), size_t(56));
    CHECK_EQ(offsetof(SavedRegisters, x20), size_t(64));
    CHECK_EQ(offsetof(SavedRegisters, x21), size_t(72));
    CHECK_EQ(offsetof(SavedRegisters, x22), size_t(80));
    CHECK_EQ(offsetof(SavedRegisters, x23), size_t(88));
    CHECK_EQ(offsetof(SavedRegisters, x24), size_t(96));
    CHECK_EQ(offsetof(SavedRegisters, x25), size_t(104));
    CHECK_EQ(offsetof(SavedRegisters, x26), size_t(112));
    CHECK_EQ(offsetof(SavedRegisters, x27), size_t(120));
    CHECK_EQ(offsetof(SavedRegisters, x28), size_t(128));
    CHECK_EQ(offsetof(SavedRegisters, fp),  size_t(136));
    CHECK_EQ(offsetof(SavedRegisters, lr),  size_t(144));

    // AArch64 FPR callee-saved D8..D15 (152..215)
    CHECK_EQ(offsetof(SavedRegisters, d8),  size_t(152));
    CHECK_EQ(offsetof(SavedRegisters, d9),  size_t(160));
    CHECK_EQ(offsetof(SavedRegisters, d10), size_t(168));
    CHECK_EQ(offsetof(SavedRegisters, d11), size_t(176));
    CHECK_EQ(offsetof(SavedRegisters, d12), size_t(184));
    CHECK_EQ(offsetof(SavedRegisters, d13), size_t(192));
    CHECK_EQ(offsetof(SavedRegisters, d14), size_t(200));
    CHECK_EQ(offsetof(SavedRegisters, d15), size_t(208));
}

TEST_CASE("AArch64 Runtime - Exception Table Registration and Scope Lookup") {
    ExceptionTableRegistry registry;

    FunctionExceptionTable table("test_func", 0x1000, 0x200);
    table.set_frame_size(48);
    table.add_scope(10, 30, 40);
    table.add_scope(50, 70, 80);

    registry.register_function_mapping(0x1000, 0x200, table);

    // PC 0x1015 is inside [10, 30) (offset 15) -> landing pad 40
    uintptr_t fn_start = 0;
    const FunctionExceptionTable* out_table = nullptr;
    const ExceptionScopeEntry* scope = registry.find_scope_by_pc(0x1015, &fn_start, &out_table);

    REQUIRE(scope != nullptr);
    CHECK_EQ(fn_start, 0x1000ULL);
    CHECK_EQ(scope->landing_pad_offset, 40u);

    // PC outside scope: offset 35 (decimal 35, outside [10, 30) and [50, 70))
    const ExceptionScopeEntry* no_scope = registry.find_scope_by_pc(0x1000 + 35);
    CHECK(no_scope == nullptr);

    // PC outside function
    CHECK(registry.find_function_by_pc(0x2000) == nullptr);
}

TEST_CASE("AArch64 Runtime - Unhandled Throw Envelope Catch") {
    HostValue val = HostValue::from_i32(777);
    bool caught = false;
    try {
        brass_throw(val);
    } catch (const BrassException& e) {
        caught = true;
        CHECK_EQ(e.value().as_i32(), 777);
    }
    CHECK(caught);
}

// =============================================================================
// 4. OSR Coordinator Vector Support
// =============================================================================
TEST_CASE("AArch64 Runtime - OSR Migration Frame Vector Packing") {
    OsrMigrationFrame frame;
    frame.loop_header_id = 42;

    alignas(16) uint8_t vec[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    uint64_t lo = 0, hi = 0;
    std::memcpy(&lo, vec, 8);
    std::memcpy(&hi, vec + 8, 8);

    frame.add_slot(0, lo);
    frame.add_slot(1, hi);

    CHECK_EQ(frame.count, 2u);
    CHECK_EQ(frame.get_raw_value(0), lo);
    CHECK_EQ(frame.get_raw_value(1), hi);
}
