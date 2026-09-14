#include "test_framework.hpp"
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <vector>
#include <iomanip>
#include <sstream>

using namespace brass::aarch64;

static void check_match(const CodeBuffer& buf, std::initializer_list<uint8_t> expected, const char* file, int line) {
    const auto& bytes = buf.bytes();
    std::vector<uint8_t> exp(expected);
    if (bytes.size() != exp.size()) {
        std::ostringstream oss;
        oss << "Size mismatch: got " << bytes.size() << " bytes, expected " << exp.size() << " bytes\nGot:      ";
        for (uint8_t b : bytes) oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b) << " ";
        oss << "\nExpected: ";
        for (uint8_t b : exp) oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b) << " ";
        ::brass::test::report_failure(file, line, "check_match", oss.str(), true);
    }
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (bytes[i] != exp[i]) {
            std::ostringstream oss;
            oss << "Byte mismatch at offset " << i << ": got 0x"
                << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i])
                << ", expected 0x" << static_cast<int>(exp[i]) << "\nGot:      ";
            for (uint8_t b : bytes) oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b) << " ";
            oss << "\nExpected: ";
            for (uint8_t b : exp) oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b) << " ";
            ::brass::test::report_failure(file, line, "check_match", oss.str(), true);
        }
    }
}

#define CHECK_BYTES(buf, ...) check_match((buf), { __VA_ARGS__ }, __FILE__, __LINE__)

TEST_CASE("AArch64 Golden - Basic Instructions and Encodings") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);

    // NOP and RET
    enc.nop();
    enc.ret();

    CHECK_BYTES(buf,
        0x1F, 0x20, 0x03, 0xD5,  // nop
        0xC0, 0x03, 0x5F, 0xD6   // ret
    );

    buf.clear();
    // ALU 64-bit and 32-bit register
    enc.add(GPR::X0, GPR::X1, GPR::X2);
    enc.sub(GPR::X3, GPR::X4, GPR::X5);
    enc.add32(GPR::X0, GPR::X1, GPR::X2);
    enc.sub32(GPR::X3, GPR::X4, GPR::X5);

    CHECK_BYTES(buf,
        0x20, 0x00, 0x02, 0x8B,  // add x0, x1, x2
        0x83, 0x00, 0x05, 0xCB,  // sub x3, x4, x5
        0x20, 0x00, 0x02, 0x0B,  // add w0, w1, w2
        0x83, 0x00, 0x05, 0x4B   // sub w3, w4, w5
    );

    buf.clear();
    // Immediate ALU
    enc.add(GPR::X0, GPR::X1, 1);
    enc.sub(GPR::SP, GPR::SP, 32);

    CHECK_BYTES(buf,
        0x20, 0x04, 0x00, 0x91,  // add x0, x1, #1
        0xFF, 0x83, 0x00, 0xD1   // sub sp, sp, #32
    );

    buf.clear();
    // Moves and Compare
    enc.mov(GPR::X0, GPR::X1);
    enc.mov32(GPR::X0, GPR::X1);
    enc.cmp(GPR::X0, GPR::X1);
    enc.cset(GPR::X0, Condition::EQ);

    CHECK_BYTES(buf,
        0xE0, 0x03, 0x01, 0xAA,  // mov x0, x1  (orr x0, xzr, x1)
        0xE0, 0x03, 0x01, 0x2A,  // mov w0, w1  (orr w0, wzr, w1)
        0x1F, 0x00, 0x01, 0xEB,  // cmp x0, x1  (subs xzr, x0, x1)
        0xE0, 0x17, 0x9F, 0x9A   // cset x0, eq (csinc x0, xzr, xzr, ne)
    );

    buf.clear();
    // Prologue/Epilogue pairs
    enc.stp(GPR::FP, GPR::LR, pre_idx(GPR::SP, -16));
    enc.ldp(GPR::FP, GPR::LR, post_idx(GPR::SP, 16));

    CHECK_BYTES(buf,
        0xFD, 0x7B, 0xBF, 0xA9,  // stp x29, x30, [sp, #-16]!
        0xFD, 0x7B, 0xC1, 0xA8   // ldp x29, x30, [sp], #16
    );

    buf.clear();
    // Memory loads & stores
    enc.ldr(GPR::X0, ptr(GPR::SP, 8));
    enc.str(GPR::X0, ptr(GPR::SP, 8));

    CHECK_BYTES(buf,
        0xE0, 0x07, 0x40, 0xF9,  // ldr x0, [sp, #8]
        0xE0, 0x07, 0x00, 0xF9   // str x0, [sp, #8]
    );

    buf.clear();
    // Multiply and Divide
    enc.mul(GPR::X0, GPR::X1, GPR::X2);
    enc.sdiv(GPR::X0, GPR::X1, GPR::X2);

    CHECK_BYTES(buf,
        0x20, 0x7C, 0x02, 0x9B,  // mul x0, x1, x2
        0x20, 0x0C, 0xC2, 0x9A   // sdiv x0, x1, x2
    );

    buf.clear();
    // Floating-point
    enc.fadd(FPR::V0, FPR::V1, FPR::V2);
    enc.fmov(FPR::V0, FPR::V1);

    CHECK_BYTES(buf,
        0x20, 0x28, 0x62, 0x1E,  // fadd d0, d1, d2
        0x20, 0x40, 0x60, 0x1E   // fmov d0, d1
    );
}

TEST_CASE("AArch64 Golden - Branch and Label Fixup Resolution") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);

    Label loop = buf.create_label();
    Label end = buf.create_label();

    enc.cmp(GPR::X0, 0);
    enc.b(Condition::EQ, end); // forward cond branch

    buf.bind(loop);
    enc.sub(GPR::X0, GPR::X0, 1);
    enc.cbnz(GPR::X0, loop);   // backward cond branch

    buf.bind(end);
    enc.ret();

    // Verify instructions emitted correctly with resolved displacements
    CHECK_EQ(buf.size(), 5 * 4); // 5 instructions = 20 bytes

    // Instruction 1: cmp x0, #0 -> subs xzr, x0, #0 -> 0xF100001F -> 1F 00 00 F1
    // Instruction 2: b.eq end (+12 bytes, disp = 3 words = 3) -> 0x54000060 -> 60 00 00 54
    // Instruction 3: sub x0, x0, #1 -> 0xD1000400 -> 00 04 00 D1
    // Instruction 4: cbnz x0, loop (-4 bytes, disp = -1 word = 0x7FFFF) -> 0xB5FFFFE0 -> E0 FF FF B5
    // Instruction 5: ret -> 0xD65F03C0 -> C0 03 5F D6
    CHECK_BYTES(buf,
        0x1F, 0x00, 0x00, 0xF1,
        0x60, 0x00, 0x00, 0x54,
        0x00, 0x04, 0x00, 0xD1,
        0xE0, 0xFF, 0xFF, 0xB5,
        0xC0, 0x03, 0x5F, 0xD6
    );
}
