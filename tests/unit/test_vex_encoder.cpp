#include "test_framework.hpp"
#include <brass/target/x64/x64_encoder.hpp>
#include <vector>
#include <iomanip>
#include <sstream>

using namespace brass::x64;

static void check_vex_match(const CodeBuffer& buf, std::initializer_list<uint8_t> expected, const char* file, int line) {
    const auto& bytes = buf.bytes();
    std::vector<uint8_t> exp(expected);
    if (bytes.size() != exp.size()) {
        std::ostringstream oss;
        oss << "Size mismatch: got " << bytes.size() << " bytes, expected " << exp.size() << " bytes\nGot:      ";
        for (uint8_t b : bytes) oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b) << " ";
        oss << "\nExpected: ";
        for (uint8_t b : exp) oss << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b) << " ";
        ::brass::test::report_failure(file, line, "check_vex_match", oss.str(), true);
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
            ::brass::test::report_failure(file, line, "check_vex_match", oss.str(), true);
        }
    }
}

#define CHECK_VEX_BYTES(buf, ...) check_vex_match((buf), { __VA_ARGS__ }, __FILE__, __LINE__)

TEST_CASE("x64 VEX - 256-Bit Moves and Memory") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // vmovaps XMM0, XMM1 (256-bit) -> C5 FC 28 C1
    enc.vmovaps(XMM::XMM0, XMM::XMM1);
    CHECK_VEX_BYTES(buf, 0xC5, 0xFC, 0x28, 0xC1);

    buf.clear();
    // vmovups XMM0, [RAX] (256-bit load) -> C5 FC 10 00
    enc.vmovups(XMM::XMM0, MemAddress::base_only(GPR::RAX));
    CHECK_VEX_BYTES(buf, 0xC5, 0xFC, 0x10, 0x00);

    buf.clear();
    // vmovups [RAX], XMM1 (256-bit store) -> C5 FC 11 08
    enc.vmovups(MemAddress::base_only(GPR::RAX), XMM::XMM1);
    CHECK_VEX_BYTES(buf, 0xC5, 0xFC, 0x11, 0x08);
}

TEST_CASE("x64 VEX - 256-Bit Floating Point Arithmetic") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // vaddps XMM0, XMM1, XMM2 -> C5 F4 58 C2
    enc.vaddps(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF4, 0x58, 0xC2);

    buf.clear();
    // vaddpd XMM0, XMM1, XMM2 -> C5 F5 58 C2
    enc.vaddpd(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0x58, 0xC2);

    buf.clear();
    // vsubps XMM3, XMM4, XMM5 -> C5 DC 5C DD
    enc.vsubps(XMM::XMM3, XMM::XMM4, XMM::XMM5);
    CHECK_VEX_BYTES(buf, 0xC5, 0xDC, 0x5C, 0xDD);

    buf.clear();
    // vsubpd XMM0, XMM1, XMM2 -> C5 F5 5C C2
    enc.vsubpd(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0x5C, 0xC2);

    buf.clear();
    // vmulps XMM0, XMM1, XMM2 -> C5 F4 59 C2
    enc.vmulps(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF4, 0x59, 0xC2);

    buf.clear();
    // vmulpd XMM0, XMM1, XMM2 -> C5 F5 59 C2
    enc.vmulpd(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0x59, 0xC2);

    buf.clear();
    // vdivps XMM0, XMM1, XMM2 -> C5 F4 5E C2
    enc.vdivps(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF4, 0x5E, 0xC2);

    buf.clear();
    // vdivpd XMM0, XMM1, XMM2 -> C5 F5 5E C2
    enc.vdivpd(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0x5E, 0xC2);
}

TEST_CASE("x64 VEX - 256-Bit Integer Arithmetic") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // vpaddd XMM0, XMM1, XMM2 -> C5 F5 FE C2
    enc.vpaddd(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0xFE, 0xC2);

    buf.clear();
    // vpaddq XMM0, XMM1, XMM2 -> C5 F5 D4 C2
    enc.vpaddq(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0xD4, 0xC2);

    buf.clear();
    // vpsubd XMM0, XMM1, XMM2 -> C5 F5 FA C2
    enc.vpsubd(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0xFA, 0xC2);

    buf.clear();
    // vpsubq XMM0, XMM1, XMM2 -> C5 F5 FB C2
    enc.vpsubq(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0xFB, 0xC2);

    buf.clear();
    // vpmulld XMM0, XMM1, XMM2 (0x0F38 Map) -> C4 E2 75 40 C2
    enc.vpmulld(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0x75, 0x40, 0xC2);
}

TEST_CASE("x64 VEX - 256-Bit Bitwise and Broadcast") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // vandps XMM0, XMM1, XMM2 -> C5 F4 54 C2
    enc.vandps(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF4, 0x54, 0xC2);

    buf.clear();
    // vorps XMM0, XMM1, XMM2 -> C5 F4 56 C2
    enc.vorps(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF4, 0x56, 0xC2);

    buf.clear();
    // vxorps XMM0, XMM1, XMM2 -> C5 F4 57 C2
    enc.vxorps(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF4, 0x57, 0xC2);

    buf.clear();
    // vpand XMM0, XMM1, XMM2 -> C5 F5 DB C2
    enc.vpand(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0xDB, 0xC2);

    buf.clear();
    // vpor XMM0, XMM1, XMM2 -> C5 F5 EB C2
    enc.vpor(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0xEB, 0xC2);

    buf.clear();
    // vpxor XMM0, XMM1, XMM2 -> C5 F5 EF C2
    enc.vpxor(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC5, 0xF5, 0xEF, 0xC2);

    buf.clear();
    // vbroadcastss XMM0, XMM1 -> C4 E2 7D 18 C1
    enc.vbroadcastss(XMM::XMM0, XMM::XMM1);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0x7D, 0x18, 0xC1);

    buf.clear();
    // vbroadcastsd XMM0, XMM1 -> C4 E2 7D 19 C1
    enc.vbroadcastsd(XMM::XMM0, XMM::XMM1);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0x7D, 0x19, 0xC1);

    buf.clear();
    // vpbroadcastd XMM0, XMM1 -> C4 E2 7D 58 C1
    enc.vpbroadcastd(XMM::XMM0, XMM::XMM1);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0x7D, 0x58, 0xC1);

    buf.clear();
    // vpbroadcastq XMM0, XMM1 -> C4 E2 7D 59 C1
    enc.vpbroadcastq(XMM::XMM0, XMM::XMM1);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0x7D, 0x59, 0xC1);
}

TEST_CASE("x64 VEX - FMA3 Instructions") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // Vector 256-bit FMA
    // vfmadd213ps XMM0, XMM1, XMM2 -> C4 E2 75 A8 C2
    enc.vfmadd213ps(XMM::XMM0, XMM::XMM1, XMM::XMM2, true);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0x75, 0xA8, 0xC2);

    buf.clear();
    // vfmadd213pd XMM0, XMM1, XMM2 -> C4 E2 F5 A8 C2
    enc.vfmadd213pd(XMM::XMM0, XMM::XMM1, XMM::XMM2, true);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0xF5, 0xA8, 0xC2);

    buf.clear();
    // vfmadd231ps XMM0, XMM1, XMM2 -> C4 E2 75 B8 C2
    enc.vfmadd231ps(XMM::XMM0, XMM::XMM1, XMM::XMM2, true);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0x75, 0xB8, 0xC2);

    buf.clear();
    // vfmadd231pd XMM0, XMM1, XMM2 -> C4 E2 F5 B8 C2
    enc.vfmadd231pd(XMM::XMM0, XMM::XMM1, XMM::XMM2, true);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0xF5, 0xB8, 0xC2);

    // Scalar FMA
    buf.clear();
    // vfmadd213ss XMM0, XMM1, XMM2 (L=0) -> C4 E2 71 A9 C2
    enc.vfmadd213ss(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0x71, 0xA9, 0xC2);

    buf.clear();
    // vfmadd213sd XMM0, XMM1, XMM2 (L=0, W=1) -> C4 E2 F1 A9 C2
    enc.vfmadd213sd(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0xF1, 0xA9, 0xC2);

    buf.clear();
    // vfmadd231ss XMM0, XMM1, XMM2 -> C4 E2 71 B9 C2
    enc.vfmadd231ss(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0x71, 0xB9, 0xC2);

    buf.clear();
    // vfmadd231sd XMM0, XMM1, XMM2 -> C4 E2 F1 B9 C2
    enc.vfmadd231sd(XMM::XMM0, XMM::XMM1, XMM::XMM2);
    CHECK_VEX_BYTES(buf, 0xC4, 0xE2, 0xF1, 0xB9, 0xC2);
}

TEST_CASE("x64 VEX - Extended Registers (XMM8-15)") {
    CodeBuffer buf;
    X64Encoder enc(buf);

    // Destination is XMM8 (R bit set, can use 2-byte VEX)
    // vaddps XMM8, XMM0, XMM1 -> C5 7C 58 C1
    enc.vaddps(XMM::XMM8, XMM::XMM0, XMM::XMM1);
    CHECK_VEX_BYTES(buf, 0xC5, 0x7C, 0x58, 0xC1);

    buf.clear();
    // Source 2 is XMM8 (B bit set, requires 3-byte VEX)
    // vaddps XMM0, XMM1, XMM8 -> C4 C1 74 58 C0
    enc.vaddps(XMM::XMM0, XMM::XMM1, XMM::XMM8);
    CHECK_VEX_BYTES(buf, 0xC4, 0xC1, 0x74, 0x58, 0xC0);
}
