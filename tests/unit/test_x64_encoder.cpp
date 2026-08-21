#include "test_framework.hpp"
#include <brass/target/x64/x64_encoder.hpp>
#include <vector>
#include <iomanip>
#include <sstream>

using namespace brass::x64;

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

TEST_CASE("x64 Golden - Stack and Frame Instructions") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // PUSH / POP GPRs
    enc.push(GPR::RAX); // 50
    enc.push(GPR::RBP); // 55
    enc.push(GPR::R8);  // 41 50
    enc.push(GPR::R15); // 41 57

    enc.pop(GPR::R15);  // 41 5F
    enc.pop(GPR::R8);   // 41 58
    enc.pop(GPR::RBP);  // 5D
    enc.pop(GPR::RAX);  // 58

    CHECK_BYTES(buf,
        0x50, 0x55, 0x41, 0x50, 0x41, 0x57,
        0x41, 0x5F, 0x41, 0x58, 0x5D, 0x58
    );

    buf.clear();
    // PUSH Immediates
    enc.push(0);           // 6A 00
    enc.push(-1);          // 6A FF
    enc.push(42);          // 6A 2A
    enc.push(0x12345678);  // 68 78 56 34 12

    CHECK_BYTES(buf,
        0x6A, 0x00,
        0x6A, 0xFF,
        0x6A, 0x2A,
        0x68, 0x78, 0x56, 0x34, 0x12
    );

    buf.clear();
    // PUSH / POP Memory
    enc.push(ptr(GPR::RCX));         // FF 31
    enc.push(ptr(GPR::R12));         // 41 FF 34 24
    enc.pop(ptr(GPR::RCX));          // 8F 01
    enc.pop(ptr(GPR::R12));          // 41 8F 04 24

    CHECK_BYTES(buf,
        0xFF, 0x31,
        0x41, 0xFF, 0x34, 0x24,
        0x8F, 0x01,
        0x41, 0x8F, 0x04, 0x24
    );

    buf.clear();
    // Frame & Misc
    enc.ret();        // C3
    enc.ret(8);       // C2 08 00
    enc.ret(32);      // C2 20 00
    enc.int3();       // CC
    enc.ud2();        // 0F 0B
    enc.cqo();        // 48 99
    enc.cdq();        // 99
    enc.nop();        // 90

    CHECK_BYTES(buf,
        0xC3,
        0xC2, 0x08, 0x00,
        0xC2, 0x20, 0x00,
        0xCC,
        0x0F, 0x0B,
        0x48, 0x99,
        0x99,
        0x90
    );
}

TEST_CASE("x64 Golden - Moves and REX Prefix Combinations") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // 64-bit Reg-Reg moves
    enc.mov(GPR::RAX, GPR::RBX); // 48 89 D8
    enc.mov(GPR::RAX, GPR::R8);  // 4C 89 C0 (src in reg=r8, dst in rm=rax)
    enc.mov(GPR::R8, GPR::RAX);  // 49 89 C0 (src in reg=rax, dst in rm=r8)
    enc.mov(GPR::R8, GPR::R9);   // 4D 89 C8
    enc.mov(GPR::R12, GPR::R15); // 4D 89 FC

    CHECK_BYTES(buf,
        0x48, 0x89, 0xD8,
        0x4C, 0x89, 0xC0,
        0x49, 0x89, 0xC0,
        0x4D, 0x89, 0xC8,
        0x4D, 0x89, 0xFC
    );

    buf.clear();
    // 32-bit Reg-Reg moves
    enc.mov32(GPR::RAX, GPR::RBX); // 89 D8
    enc.mov32(GPR::RAX, GPR::R8);  // 44 89 C0
    enc.mov32(GPR::R8, GPR::RAX);  // 41 89 C0
    enc.mov32(GPR::R8, GPR::R9);   // 45 89 C8

    CHECK_BYTES(buf,
        0x89, 0xD8,
        0x44, 0x89, 0xC0,
        0x41, 0x89, 0xC0,
        0x45, 0x89, 0xC8
    );

    buf.clear();
    // Immediate moves
    enc.mov32(GPR::RAX, 42); // B8 2A 00 00 00
    enc.mov32(GPR::R8, 42);  // 41 B8 2A 00 00 00
    enc.mov(GPR::RAX, 42);   // 48 C7 C0 2A 00 00 00 (sign-extended imm32)
    enc.movabs(GPR::RAX, 0x1122334455667788ULL); // 48 B8 88 77 66 55 44 33 22 11
    enc.movabs(GPR::R12, 0x1122334455667788ULL); // 49 BC 88 77 66 55 44 33 22 11

    CHECK_BYTES(buf,
        0xB8, 0x2A, 0x00, 0x00, 0x00,
        0x41, 0xB8, 0x2A, 0x00, 0x00, 0x00,
        0x48, 0xC7, 0xC0, 0x2A, 0x00, 0x00, 0x00,
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
        0x49, 0xBC, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11
    );
}

TEST_CASE("x64 Golden - 8-bit, 16-bit Moves and Extensions") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    enc.mov8(GPR::RAX, GPR::RBX); // 88 D8 (al, bl)
    enc.mov8(GPR::RAX, GPR::RSI); // 40 88 F0 (al, sil - needs REX)
    enc.mov8(GPR::R8, GPR::RAX);  // 41 88 C0 (r8b, al)
    enc.mov8(GPR::RAX, GPR::R8);  // 44 88 C0 (al, r8b)
    enc.mov8(GPR::RAX, 42);       // B0 2A (al, 42)
    enc.mov8(GPR::R8, 42);        // 41 B0 2A (r8b, 42)

    enc.mov16(GPR::RAX, GPR::RBX); // 66 89 D8 (ax, bx)
    enc.mov16(GPR::R8, GPR::R9);   // 66 45 89 C8 (r8w, r9w)

    CHECK_BYTES(buf,
        0x88, 0xD8,
        0x40, 0x88, 0xF0,
        0x41, 0x88, 0xC0,
        0x44, 0x88, 0xC0,
        0xB0, 0x2A,
        0x41, 0xB0, 0x2A,
        0x66, 0x89, 0xD8,
        0x66, 0x45, 0x89, 0xC8
    );

    buf.clear();
    // Extensions
    enc.movsxd(GPR::RAX, GPR::RBX); // 48 63 C3
    enc.movsxd(GPR::RAX, GPR::R8);  // 49 63 C0
    enc.movsxd(GPR::R8, GPR::RBX);  // 4C 63 C3

    enc.movzx8(GPR::RAX, GPR::RBX); // 0F B6 C3
    enc.movzx8(GPR::RAX, GPR::RSI); // 40 0F B6 C6 (sil)
    enc.movzx8(GPR::R8, GPR::RBX);  // 44 0F B6 C3

    enc.movzx16(GPR::RAX, GPR::RBX); // 0F B7 C3
    enc.movzx16(GPR::R8, GPR::R9);   // 45 0F B7 C1

    enc.movsx8(GPR::RAX, GPR::RBX);  // 48 0F BE C3
    enc.movsx8(GPR::RAX, GPR::RSI);  // 48 0F BE C6
    enc.movsx16(GPR::RAX, GPR::RBX); // 48 0F BF C3

    CHECK_BYTES(buf,
        0x48, 0x63, 0xC3,
        0x49, 0x63, 0xC0,
        0x4C, 0x63, 0xC3,
        0x0F, 0xB6, 0xC3,
        0x40, 0x0F, 0xB6, 0xC6,
        0x44, 0x0F, 0xB6, 0xC3,
        0x0F, 0xB7, 0xC3,
        0x45, 0x0F, 0xB7, 0xC1,
        0x48, 0x0F, 0xBE, 0xC3,
        0x48, 0x0F, 0xBE, 0xC6,
        0x48, 0x0F, 0xBF, 0xC3
    );
}

TEST_CASE("x64 Golden - Memory Addressing Modes") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // Base only & Base + Disp
    enc.mov(GPR::RAX, ptr(GPR::RCX));            // 48 8B 01
    enc.mov(ptr(GPR::RCX), GPR::RAX);            // 48 89 01
    enc.mov(GPR::RAX, ptr(GPR::RCX, 16));        // 48 8B 41 10
    enc.mov(GPR::RAX, ptr(GPR::RCX, 0x1000));    // 48 8B 81 00 10 00 00

    // RSP base (requires SIB)
    enc.mov(GPR::RAX, ptr(GPR::RSP));            // 48 8B 04 24
    enc.mov(GPR::RAX, ptr(GPR::RSP, 8));         // 48 8B 44 24 08
    enc.mov(GPR::RAX, ptr(GPR::RSP, 0x1000));    // 48 8B 84 24 00 10 00 00

    // RBP base (mod=01 with disp8=0 for disp=0)
    enc.mov(GPR::RAX, ptr(GPR::RBP));            // 48 8B 45 00
    enc.mov(GPR::RAX, ptr(GPR::RBP, 16));        // 48 8B 45 10
    enc.mov(GPR::RAX, ptr(GPR::RBP, 0x1000));    // 48 8B 85 00 10 00 00

    // R12 base (requires SIB + REX.B)
    enc.mov(GPR::RAX, ptr(GPR::R12));            // 49 8B 04 24
    enc.mov(GPR::RAX, ptr(GPR::R12, 8));         // 49 8B 44 24 08

    // R13 base (mod=01 with disp8=0 + REX.B)
    enc.mov(GPR::RAX, ptr(GPR::R13));            // 49 8B 45 00
    enc.mov(GPR::RAX, ptr(GPR::R13, 16));        // 49 8B 45 10

    CHECK_BYTES(buf,
        0x48, 0x8B, 0x01,
        0x48, 0x89, 0x01,
        0x48, 0x8B, 0x41, 0x10,
        0x48, 0x8B, 0x81, 0x00, 0x10, 0x00, 0x00,
        0x48, 0x8B, 0x04, 0x24,
        0x48, 0x8B, 0x44, 0x24, 0x08,
        0x48, 0x8B, 0x84, 0x24, 0x00, 0x10, 0x00, 0x00,
        0x48, 0x8B, 0x45, 0x00,
        0x48, 0x8B, 0x45, 0x10,
        0x48, 0x8B, 0x85, 0x00, 0x10, 0x00, 0x00,
        0x49, 0x8B, 0x04, 0x24,
        0x49, 0x8B, 0x44, 0x24, 0x08,
        0x49, 0x8B, 0x45, 0x00,
        0x49, 0x8B, 0x45, 0x10
    );

    buf.clear();
    // Scaled Index Addressing: [base + index*scale + disp]
    enc.mov(GPR::RAX, ptr(GPR::RCX, GPR::RDX, Scale::One));          // 48 8B 04 11
    enc.mov(GPR::RAX, ptr(GPR::RCX, GPR::RDX, Scale::Two));          // 48 8B 04 51
    enc.mov(GPR::RAX, ptr(GPR::RCX, GPR::RDX, Scale::Four));         // 48 8B 04 91
    enc.mov(GPR::RAX, ptr(GPR::RCX, GPR::RDX, Scale::Eight));        // 48 8B 04 D1
    enc.mov(GPR::RAX, ptr(GPR::RCX, GPR::RDX, Scale::Eight, 16));    // 48 8B 44 D1 10
    enc.mov(GPR::RAX, ptr(GPR::RCX, GPR::RDX, Scale::Eight, 0x1000));// 48 8B 84 D1 00 10 00 00
    enc.mov(GPR::RAX, ptr(GPR::R8,  GPR::R9,  Scale::Four, 32));     // 4B 8B 44 88 20 (REX.W + REX.X + REX.B = 4B)

    // LEA & RIP-relative
    enc.lea(GPR::RAX, ptr(GPR::RCX, GPR::RDX, Scale::Four, 8));      // 48 8D 44 91 08
    enc.lea(GPR::RAX, rip_rel(0x100));                               // 48 8D 05 00 01 00 00
    enc.mov(GPR::RAX, rip_rel(0x200));                               // 48 8B 05 00 02 00 00

    CHECK_BYTES(buf,
        0x48, 0x8B, 0x04, 0x11,
        0x48, 0x8B, 0x04, 0x51,
        0x48, 0x8B, 0x04, 0x91,
        0x48, 0x8B, 0x04, 0xD1,
        0x48, 0x8B, 0x44, 0xD1, 0x10,
        0x48, 0x8B, 0x84, 0xD1, 0x00, 0x10, 0x00, 0x00,
        0x4B, 0x8B, 0x44, 0x88, 0x20,
        0x48, 0x8D, 0x44, 0x91, 0x08,
        0x48, 0x8D, 0x05, 0x00, 0x01, 0x00, 0x00,
        0x48, 0x8B, 0x05, 0x00, 0x02, 0x00, 0x00
    );
}

TEST_CASE("x64 Golden - ALU 32-bit and 64-bit Operations") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // ADD / SUB
    enc.add(GPR::RAX, GPR::RBX); // 48 01 D8
    enc.add32(GPR::RAX, GPR::RBX); // 01 D8
    enc.add(GPR::R8, GPR::R9);   // 4D 01 C8
    enc.add(GPR::RAX, 1);        // 48 83 C0 01
    enc.add(GPR::RAX, 0x100);    // 48 81 C0 00 01 00 00
    enc.add32(GPR::RAX, 1);      // 83 C0 01
    enc.add32(GPR::RAX, 0x100);  // 81 C0 00 01 00 00
    enc.add(GPR::R8, 1);         // 49 83 C0 01
    enc.add(ptr(GPR::RCX), GPR::RAX); // 48 01 01
    enc.add(GPR::RAX, ptr(GPR::RCX)); // 48 03 01
    enc.add(ptr(GPR::RCX), 1);   // 48 83 01 01

    enc.sub(GPR::RSP, 40);       // 48 83 EC 28
    enc.sub(GPR::RSP, 0x1000);   // 48 81 EC 00 10 00 00

    CHECK_BYTES(buf,
        0x48, 0x01, 0xD8,
        0x01, 0xD8,
        0x4D, 0x01, 0xC8,
        0x48, 0x83, 0xC0, 0x01,
        0x48, 0x81, 0xC0, 0x00, 0x01, 0x00, 0x00,
        0x83, 0xC0, 0x01,
        0x81, 0xC0, 0x00, 0x01, 0x00, 0x00,
        0x49, 0x83, 0xC0, 0x01,
        0x48, 0x01, 0x01,
        0x48, 0x03, 0x01,
        0x48, 0x83, 0x01, 0x01,
        0x48, 0x83, 0xEC, 0x28,
        0x48, 0x81, 0xEC, 0x00, 0x10, 0x00, 0x00
    );

    buf.clear();
    // AND, OR, XOR, CMP
    enc.and_(GPR::RAX, GPR::RBX); // 48 21 D8
    enc.or_(GPR::RAX, GPR::RBX);  // 48 09 D8
    enc.xor_(GPR::RAX, GPR::RAX); // 48 31 C0
    enc.xor32(GPR::RAX, GPR::RAX);// 31 C0
    enc.xor_(GPR::R8, GPR::R8);   // 4D 31 C0
    enc.cmp(GPR::RAX, GPR::RBX);  // 48 39 D8
    enc.cmp(GPR::RAX, 0);         // 48 83 F8 00

    // TEST, NOT, NEG
    enc.test(GPR::RAX, GPR::RAX); // 48 85 C0
    enc.test32(GPR::RAX, GPR::RAX);// 85 C0
    enc.test(GPR::R8, GPR::R9);   // 4D 85 C8
    enc.test(GPR::RAX, 1);        // 48 F7 C0 01 00 00 00
    enc.not_(GPR::RAX);           // 48 F7 D0
    enc.neg(GPR::RAX);            // 48 F7 D8

    CHECK_BYTES(buf,
        0x48, 0x21, 0xD8,
        0x48, 0x09, 0xD8,
        0x48, 0x31, 0xC0,
        0x31, 0xC0,
        0x4D, 0x31, 0xC0,
        0x48, 0x39, 0xD8,
        0x48, 0x83, 0xF8, 0x00,
        0x48, 0x85, 0xC0,
        0x85, 0xC0,
        0x4D, 0x85, 0xC8,
        0x48, 0xF7, 0xC0, 0x01, 0x00, 0x00, 0x00,
        0x48, 0xF7, 0xD0,
        0x48, 0xF7, 0xD8
    );

    buf.clear();
    // MUL, DIV, IDIV, IMUL
    enc.mul(GPR::RBX);                // 48 F7 F3 (mul r/m64) -> wait, mul ext is 4 -> 11 100 011 = 0xE3!
    enc.div(GPR::RBX);                // 48 F7 F3 (div ext is 6 -> 11 110 011 = 0xF3)
    enc.idiv(GPR::RBX);               // 48 F7 FB (idiv ext is 7 -> 11 111 011 = 0xFB)
    enc.imul(GPR::RBX);               // 48 F7 EB (imul ext is 5 -> 11 101 011 = 0xEB)
    enc.imul(GPR::RAX, GPR::RBX);     // 48 0F AF C3
    enc.imul(GPR::R8, GPR::R9);       // 4D 0F AF C1
    enc.imul(GPR::RAX, GPR::RBX, 10); // 48 6B C3 0A
    enc.imul(GPR::RAX, GPR::RBX, 1000); // 48 69 C3 E8 03 00 00

    CHECK_BYTES(buf,
        0x48, 0xF7, 0xE3,
        0x48, 0xF7, 0xF3,
        0x48, 0xF7, 0xFB,
        0x48, 0xF7, 0xEB,
        0x48, 0x0F, 0xAF, 0xC3,
        0x4D, 0x0F, 0xAF, 0xC1,
        0x48, 0x6B, 0xC3, 0x0A,
        0x48, 0x69, 0xC3, 0xE8, 0x03, 0x00, 0x00
    );
}

TEST_CASE("x64 Golden - Shift and Rotate Instructions") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    enc.shl(GPR::RAX, 1);  // 48 D1 E0
    enc.shl(GPR::RAX, 4);  // 48 C1 E0 04
    enc.shl(GPR::RAX);     // 48 D3 E0 (by CL)

    enc.shr(GPR::RAX, 1);  // 48 D1 E8
    enc.shr(GPR::RAX, 4);  // 48 C1 E8 04
    enc.shr(GPR::RAX);     // 48 D3 E8 (by CL)

    enc.sar(GPR::RAX, 1);  // 48 D1 F8
    enc.sar(GPR::RAX, 4);  // 48 C1 F8 04
    enc.sar(GPR::RAX);     // 48 D3 F8 (by CL)

    enc.rol(GPR::RAX, 1);  // 48 D1 C0
    enc.ror(GPR::RAX, 1);  // 48 D1 C8

    CHECK_BYTES(buf,
        0x48, 0xD1, 0xE0,
        0x48, 0xC1, 0xE0, 0x04,
        0x48, 0xD3, 0xE0,
        0x48, 0xD1, 0xE8,
        0x48, 0xC1, 0xE8, 0x04,
        0x48, 0xD3, 0xE8,
        0x48, 0xD1, 0xF8,
        0x48, 0xC1, 0xF8, 0x04,
        0x48, 0xD3, 0xF8,
        0x48, 0xD1, 0xC0,
        0x48, 0xD1, 0xC8
    );
}

TEST_CASE("x64 Golden - SSE FP Arithmetic and Conversions") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // Moves
    enc.movss(XMM::XMM0, XMM::XMM1); // F3 0F 10 C1
    enc.movsd(XMM::XMM0, XMM::XMM1); // F2 0F 10 C1
    enc.movsd(XMM::XMM8, XMM::XMM9); // F2 45 0F 10 C1
    enc.movsd(XMM::XMM0, ptr(GPR::RCX)); // F2 0F 10 01
    enc.movsd(ptr(GPR::RCX), XMM::XMM0); // F2 0F 11 01

    enc.movq(XMM::XMM0, GPR::RAX);   // 66 48 0F 6E C0
    enc.movq(GPR::RAX, XMM::XMM0);   // 66 48 0F 7E C0
    enc.movq(XMM::XMM8, GPR::R8);    // 66 4D 0F 6E C0

    // Double FP Arithmetic
    enc.addsd(XMM::XMM0, XMM::XMM1); // F2 0F 58 C1
    enc.subsd(XMM::XMM0, XMM::XMM1); // F2 0F 5C C1
    enc.mulsd(XMM::XMM0, XMM::XMM1); // F2 0F 59 C1
    enc.divsd(XMM::XMM0, XMM::XMM1); // F2 0F 5E C1
    enc.sqrtsd(XMM::XMM0, XMM::XMM1);// F2 0F 51 C1
    enc.ucomisd(XMM::XMM0, XMM::XMM1);// 66 0F 2E C1
    enc.xorpd(XMM::XMM0, XMM::XMM0); // 66 0F 57 C0

    // Single FP Arithmetic
    enc.addss(XMM::XMM0, XMM::XMM1); // F3 0F 58 C1
    enc.subss(XMM::XMM0, XMM::XMM1); // F3 0F 5C C1
    enc.mulss(XMM::XMM0, XMM::XMM1); // F3 0F 59 C1
    enc.divss(XMM::XMM0, XMM::XMM1); // F3 0F 5E C1
    enc.ucomiss(XMM::XMM0, XMM::XMM1);// 0F 2E C1
    enc.xorps(XMM::XMM0, XMM::XMM0); // 0F 57 C0

    CHECK_BYTES(buf,
        0xF3, 0x0F, 0x10, 0xC1,
        0xF2, 0x0F, 0x10, 0xC1,
        0xF2, 0x45, 0x0F, 0x10, 0xC1,
        0xF2, 0x0F, 0x10, 0x01,
        0xF2, 0x0F, 0x11, 0x01,
        0x66, 0x48, 0x0F, 0x6E, 0xC0,
        0x66, 0x48, 0x0F, 0x7E, 0xC0,
        0x66, 0x4D, 0x0F, 0x6E, 0xC0,
        0xF2, 0x0F, 0x58, 0xC1,
        0xF2, 0x0F, 0x5C, 0xC1,
        0xF2, 0x0F, 0x59, 0xC1,
        0xF2, 0x0F, 0x5E, 0xC1,
        0xF2, 0x0F, 0x51, 0xC1,
        0x66, 0x0F, 0x2E, 0xC1,
        0x66, 0x0F, 0x57, 0xC0,
        0xF3, 0x0F, 0x58, 0xC1,
        0xF3, 0x0F, 0x5C, 0xC1,
        0xF3, 0x0F, 0x59, 0xC1,
        0xF3, 0x0F, 0x5E, 0xC1,
        0x0F, 0x2E, 0xC1,
        0x0F, 0x57, 0xC0
    );

    buf.clear();
    // Conversions
    enc.cvtsi2sd(XMM::XMM0, GPR::RAX);   // F2 48 0F 2A C0
    enc.cvtsi2sd32(XMM::XMM0, GPR::RAX); // F2 0F 2A C0
    enc.cvttsd2si(GPR::RAX, XMM::XMM0);  // F2 48 0F 2C C0
    enc.cvttsd2si32(GPR::RAX, XMM::XMM0);// F2 0F 2C C0
    enc.cvtsi2ss(XMM::XMM0, GPR::RAX);   // F3 48 0F 2A C0
    enc.cvttss2si(GPR::RAX, XMM::XMM0);  // F3 48 0F 2C C0
    enc.cvtsd2ss(XMM::XMM0, XMM::XMM1);  // F2 0F 5A C1
    enc.cvtss2sd(XMM::XMM0, XMM::XMM1);  // F3 0F 5A C1

    CHECK_BYTES(buf,
        0xF2, 0x48, 0x0F, 0x2A, 0xC0,
        0xF2, 0x0F, 0x2A, 0xC0,
        0xF2, 0x48, 0x0F, 0x2C, 0xC0,
        0xF2, 0x0F, 0x2C, 0xC0,
        0xF3, 0x48, 0x0F, 0x2A, 0xC0,
        0xF3, 0x48, 0x0F, 0x2C, 0xC0,
        0xF2, 0x0F, 0x5A, 0xC1,
        0xF3, 0x0F, 0x5A, 0xC1
    );
}

TEST_CASE("x64 Golden - Bit Operations and Conditionals") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // Bit manipulation
    enc.popcnt(GPR::RAX, GPR::RCX);   // F3 48 0F B8 C1
    enc.popcnt32(GPR::RAX, GPR::RCX); // F3 0F B8 C1
    enc.popcnt(GPR::R8, GPR::R9);     // F3 4D 0F B8 C1
    enc.lzcnt(GPR::RAX, GPR::RCX);    // F3 48 0F BD C1
    enc.tzcnt(GPR::RAX, GPR::RCX);    // F3 48 0F BC C1
    enc.bsf(GPR::RAX, GPR::RCX);      // 48 0F BC C1
    enc.bsr(GPR::RAX, GPR::RCX);      // 48 0F BD C1

    CHECK_BYTES(buf,
        0xF3, 0x48, 0x0F, 0xB8, 0xC1,
        0xF3, 0x0F, 0xB8, 0xC1,
        0xF3, 0x4D, 0x0F, 0xB8, 0xC1,
        0xF3, 0x48, 0x0F, 0xBD, 0xC1,
        0xF3, 0x48, 0x0F, 0xBC, 0xC1,
        0x48, 0x0F, 0xBC, 0xC1,
        0x48, 0x0F, 0xBD, 0xC1
    );

    buf.clear();
    // SETcc
    enc.sete(GPR::RAX); // 0F 94 C0 (al)
    enc.sete(GPR::R8);  // 41 0F 94 C0 (r8b)
    enc.setne(GPR::RCX);// 0F 95 C1 (cl)
    enc.setl(GPR::RDX); // 0F 9C C2 (dl)

    // CMOVcc
    enc.cmove(GPR::RAX, GPR::RBX); // 48 0F 44 C3
    enc.cmovne(GPR::RAX, GPR::RBX);// 48 0F 45 C3
    enc.cmovl(GPR::RAX, GPR::RBX); // 48 0F 4C C3

    // Indirect Calls and Jumps
    enc.call(GPR::RAX);            // FF D0
    enc.call(GPR::R8);             // 41 FF D0
    enc.call(ptr(GPR::RCX));       // FF 11
    enc.jmp(GPR::RAX);             // FF E0
    enc.jmp(GPR::R8);              // 41 FF E0
    enc.jmp(ptr(GPR::RCX));        // FF 21

    CHECK_BYTES(buf,
        0x0F, 0x94, 0xC0,
        0x41, 0x0F, 0x94, 0xC0,
        0x0F, 0x95, 0xC1,
        0x0F, 0x9C, 0xC2,
        0x48, 0x0F, 0x44, 0xC3,
        0x48, 0x0F, 0x45, 0xC3,
        0x48, 0x0F, 0x4C, 0xC3,
        0xFF, 0xD0,
        0x41, 0xFF, 0xD0,
        0xFF, 0x11,
        0xFF, 0xE0,
        0x41, 0xFF, 0xE0,
        0xFF, 0x21
    );
}

TEST_CASE("x64 Golden - Labels, Fixups, Jcc and Relocations") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // Forward Rel32 jump fixup
    Label target = buf.create_label();
    enc.jmp(target); // E9 05 00 00 00
    enc.nop(5);      // 5-byte NOP: 0F 1F 44 00 00
    buf.bind(target);
    enc.ret();       // C3

    // jmp offset is 0, next_off is 5, target is 5 + 5 = 10, disp = 10 - 5 = 5
    CHECK_BYTES(buf,
        0xE9, 0x05, 0x00, 0x00, 0x00,
        0x0F, 0x1F, 0x44, 0x00, 0x00,
        0xC3
    );

    buf.clear();
    // Forward Rel8 short jump fixup
    Label short_target = buf.create_label();
    enc.jmp_short(short_target); // EB 03
    enc.nop(3);                  // 0F 1F 00
    buf.bind(short_target);
    enc.ret();                   // C3

    CHECK_BYTES(buf,
        0xEB, 0x03,
        0x0F, 0x1F, 0x00,
        0xC3
    );

    buf.clear();
    // Backward Rel8 and Rel32 jumps
    Label loop_head = buf.create_label();
    buf.bind(loop_head);
    enc.nop(2); // 66 90 (2 bytes)
    enc.jmp_short(loop_head); // EB FC (disp = 0 - 4 = -4 = 0xFC)
    enc.jmp(loop_head);       // E9 F7 FF FF FF (disp = 0 - 9 = -9 = 0xFFFFFFF7)

    CHECK_BYTES(buf,
        0x66, 0x90,
        0xEB, 0xFC,
        0xE9, 0xF7, 0xFF, 0xFF, 0xFF
    );

    buf.clear();
    // Conditional branches forward fixups
    Label lbl_then = buf.create_label();
    Label lbl_else = buf.create_label();

    enc.cmp(GPR::RAX, 0);
    enc.je(lbl_then); // 0F 84 02 00 00 00 (disp = +2 over mov)
    enc.mov(GPR::RAX, 1);
    buf.bind(lbl_then);
    enc.jl_short(lbl_else); // 7C 05 (disp = +5 over nop(5))
    enc.nop(5);
    buf.bind(lbl_else);
    enc.ret();

    CHECK(!buf.has_unresolved_labels());

    // Symbol Relocation
    buf.clear();
    enc.call("my_external_func");
    CHECK_EQ(buf.relocations().size(), size_t(1));
    CHECK_EQ(buf.relocations()[0].symbol_name, "my_external_func");
    CHECK_EQ(buf.relocations()[0].kind, RelocationKind::PCRel32);
    CHECK_EQ(buf.relocations()[0].offset, size_t(1));
}

TEST_CASE("x64 Golden - Multi-byte NOPs and Alignment") {
    CodeBuffer buf;

    // 1 to 9 byte NOPs
    buf.emit_nops(1);
    CHECK_BYTES(buf, 0x90);
    buf.clear();

    buf.emit_nops(2);
    CHECK_BYTES(buf, 0x66, 0x90);
    buf.clear();

    buf.emit_nops(3);
    CHECK_BYTES(buf, 0x0F, 0x1F, 0x00);
    buf.clear();

    buf.emit_nops(4);
    CHECK_BYTES(buf, 0x0F, 0x1F, 0x40, 0x00);
    buf.clear();

    buf.emit_nops(5);
    CHECK_BYTES(buf, 0x0F, 0x1F, 0x44, 0x00, 0x00);
    buf.clear();

    buf.emit_nops(6);
    CHECK_BYTES(buf, 0x66, 0x0F, 0x1F, 0x44, 0x00, 0x00);
    buf.clear();

    buf.emit_nops(7);
    CHECK_BYTES(buf, 0x0F, 0x1F, 0x80, 0x00, 0x00, 0x00, 0x00);
    buf.clear();

    buf.emit_nops(8);
    CHECK_BYTES(buf, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00);
    buf.clear();

    buf.emit_nops(9);
    CHECK_BYTES(buf, 0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00);
    buf.clear();

    // 14 byte NOP (9 bytes + 5 bytes)
    buf.emit_nops(14);
    CHECK_EQ(buf.size(), size_t(14));
    CHECK_BYTES(buf,
        0x66, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00, // 9
        0x0F, 0x1F, 0x44, 0x00, 0x00                          // 5
    );

    // Alignment test
    buf.clear();
    buf.emit8(0xCC); // offset 1
    buf.align(16);   // pads 15 bytes to reach offset 16
    CHECK_EQ(buf.size(), size_t(16));
    CHECK_EQ(buf.data()[0], 0xCC);
}
