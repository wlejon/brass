#include "test_framework.hpp"
#include <brass/codegen/lir.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/aarch64/aarch64_encoder.hpp>
#include <brass/target/aarch64/aarch64_registers.hpp>
#include <brass/target/aarch64/code_buffer.hpp>
#include <cstring>
#include <vector>

// AArch64 short-branch relaxation: B.cond / CBZ / CBNZ reach +-1 MB and
// TBZ / TBNZ +-32 KB. A branch whose target is out of reach becomes the
// inverted short branch over an unconditional B (+-128 MB): immediately for a
// bound (backward) target, and for a forward one through a relax request and
// a re-emission with that site in long form.

using namespace brass;
using namespace brass::aarch64;
using namespace brass::codegen;

namespace {

uint32_t word_at(const CodeBuffer& buf, size_t off) {
    uint32_t w = 0;
    std::memcpy(&w, buf.data() + off, 4);
    return w;
}

int64_t b26_target(uint32_t inst, size_t at) {
    int64_t imm = static_cast<int64_t>(inst & 0x03FFFFFFu);
    if (imm & (int64_t(1) << 25)) imm -= int64_t(1) << 26;
    return static_cast<int64_t>(at) + imm * 4;
}

constexpr size_t kOneMB = size_t(1) << 20;

} // namespace

TEST_CASE("AArch64 branch relax - in-range short branches keep their one-word form") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);
    Label l = buf.create_label();
    enc.tbz(GPR::X3, 37, l);    // bit 37: b5 = 1, b40 = 5
    enc.tbnz(GPR::X4, 2, l);
    enc.cbz(GPR::X5, l);
    enc.b(Condition::NE, l);
    buf.bind(l);
    REQUIRE_EQ(buf.size(), size_t(16));
    // tbz x3, #37, +16
    CHECK_EQ(word_at(buf, 0), 0x36000000u | (1u << 31) | (5u << 19) | (4u << 5) | 3u);
    // tbnz x4, #2, +12
    CHECK_EQ(word_at(buf, 4), 0x37000000u | (2u << 19) | (3u << 5) | 4u);
    // cbz x5, +8
    CHECK_EQ(word_at(buf, 8), 0xB4000000u | (2u << 5) | 5u);
    // b.ne +4
    CHECK_EQ(word_at(buf, 12), 0x54000000u | (1u << 5) | 1u);
    CHECK(buf.relax_requests().empty());
}

TEST_CASE("AArch64 branch relax - backward target out of range: long form at once") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);
    Label top = buf.create_label();
    buf.bind(top);
    buf.emit_nops(kOneMB + 64);

    const size_t at = buf.size();
    enc.b(Condition::EQ, top);
    // b.ne +8 ; b top
    CHECK_EQ(word_at(buf, at), 0x54000000u | (2u << 5) | 1u);
    CHECK_EQ(word_at(buf, at + 4) & 0xFC000000u, 0x14000000u);
    CHECK_EQ(b26_target(word_at(buf, at + 4), at + 4), int64_t(0));

    const size_t at2 = buf.size();
    enc.cbnz32(GPR::X7, top);
    // cbz w7, +8 ; b top
    CHECK_EQ(word_at(buf, at2), 0x34000000u | (2u << 5) | 7u);
    CHECK_EQ(b26_target(word_at(buf, at2 + 4), at2 + 4), int64_t(0));
}

TEST_CASE("AArch64 branch relax - TBZ reaches only 32 KB") {
    CodeBuffer buf;
    AArch64Encoder enc(buf);
    Label top = buf.create_label();
    buf.bind(top);
    buf.emit_nops(40 * 1024);   // beyond TBZ, well within B.cond
    const size_t at = buf.size();
    enc.tbz(GPR::X1, 0, top);
    // tbnz x1, #0, +8 ; b top
    CHECK_EQ(word_at(buf, at), 0x37000000u | (2u << 5) | 1u);
    CHECK_EQ(b26_target(word_at(buf, at + 4), at + 4), int64_t(0));

    const size_t at2 = buf.size();
    enc.b(Condition::LT, top);  // same distance: B.cond still reaches
    CHECK_EQ(word_at(buf, at2) & 0xFF00001Fu, 0x54000000u | static_cast<uint32_t>(Condition::LT));
}

TEST_CASE("AArch64 branch relax - forward target out of range: request, then re-emit long") {
    auto emit = [](CodeBuffer& buf) {
        AArch64Encoder enc(buf);
        Label far = buf.create_label();
        Label near = buf.create_label();
        enc.cbz(GPR::X2, far);        // site 1: out of reach
        enc.tbnz(GPR::X2, 9, near);   // site 2: in reach
        buf.bind(near);
        buf.emit_nops(kOneMB + 128);
        buf.bind(far);
    };

    CodeBuffer pass1;
    emit(pass1);   // no throw: the out-of-range site is recorded
    REQUIRE_EQ(pass1.relax_requests().size(), size_t(1));
    CHECK_EQ(pass1.relax_requests()[0], 1u);

    CodeBuffer pass2;
    pass2.add_long_branch_sites(pass1.relax_requests());
    emit(pass2);
    CHECK(pass2.relax_requests().empty());
    // cbnz x2, +8 ; b far ; tbnz x2, #9, +4
    CHECK_EQ(word_at(pass2, 0), 0xB5000000u | (2u << 5) | 2u);
    CHECK_EQ(b26_target(word_at(pass2, 4), 4), static_cast<int64_t>(pass2.size()));
    CHECK_EQ(word_at(pass2, 8), 0x37000000u | (9u << 19) | (1u << 5) | 2u);
}

TEST_CASE("AArch64 branch relax - the emitter re-emits a function whose Jcc outgrew B.cond") {
    LirFunction fn;
    fn.name = "far_branch";
    fn.frame.total_frame_size = 16;
    fn.frame.is_leaf = false;

    auto b0 = std::make_unique<LirBlock>(0, "entry");
    auto cmp = std::make_unique<LirInst>(LirOpcode::Cmp);
    cmp->add_use(LirOperand::preg_aarch64_gpr(GPR::X0, 8));
    cmp->add_use(LirOperand::imm(0, 8));
    b0->append_inst(std::move(cmp));
    auto jcc = std::make_unique<LirInst>(LirOpcode::Jcc);
    jcc->condition = brass::x64::Condition::E;
    jcc->add_use(LirOperand::label(2));
    b0->append_inst(std::move(jcc));
    auto jmp = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp->add_use(LirOperand::label(1));
    b0->append_inst(std::move(jmp));
    fn.blocks.push_back(std::move(b0));

    // Over 1 MB of code between the branch and its target.
    auto b1 = std::make_unique<LirBlock>(1, "filler");
    for (int i = 0; i < 80000; ++i) {
        auto mov = std::make_unique<LirInst>(LirOpcode::Movabs);
        mov->add_def(LirOperand::preg_aarch64_gpr(GPR::X9, 8));
        mov->add_use(LirOperand::imm(static_cast<int64_t>(0x1111222233334444ull), 8));
        b1->append_inst(std::move(mov));
    }
    b1->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
    fn.blocks.push_back(std::move(b1));

    auto b2 = std::make_unique<LirBlock>(2, "far");
    b2->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
    fn.blocks.push_back(std::move(b2));

    AArch64EmitContext emitter(fn, Target::aarch64_linux());
    AArch64CompilationResult res = emitter.compile();
    const CodeBuffer& buf = res.code_buffer;
    REQUIRE(res.block_offsets.count(2) == 1);
    const size_t far = res.block_offsets.at(2);
    const size_t filler = res.block_offsets.at(1);
    REQUIRE(far - filler > kOneMB);

    // In block 0: b.ne +8 followed by b far.
    bool found = false;
    for (size_t off = res.block_offsets.at(0); off + 8 <= filler; off += 4) {
        if (word_at(buf, off) == (0x54000000u | (2u << 5) | 1u) &&
            b26_target(word_at(buf, off + 4), off + 4) == static_cast<int64_t>(far)) {
            found = true;
        }
    }
    CHECK(found);
}
