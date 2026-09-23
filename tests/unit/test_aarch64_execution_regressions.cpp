// Regressions found by running brass-generated AArch64 code under qemu-user
// (scripts/linux-tests.sh). The selector-level tests here run on any host;
// the programs in test_jit_fuzz_regressions.cpp execute on the host JIT, so
// the same cases run as AArch64 code under the harness.

#include "test_framework.hpp"
#include <brass/codegen/emit_context.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/lir.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <brass/target/calling_conv.hpp>
#include <brass/target/x64/x64_registers.hpp>
#include <initializer_list>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/target/elf_so_writer.hpp>
#include <cstring>
#include <string>
#include <vector>

using namespace brass;
using namespace brass::codegen;

namespace {

uint32_t rd32(const std::vector<uint8_t>& b, size_t off) {
    uint32_t v = 0;
    std::memcpy(&v, b.data() + off, 4);
    return v;
}

uint64_t rd64(const std::vector<uint8_t>& b, size_t off) {
    uint64_t v = 0;
    std::memcpy(&v, b.data() + off, 8);
    return v;
}

struct Phdr {
    uint32_t type = 0;
    uint64_t vaddr = 0;
    uint64_t filesz = 0;
    uint64_t memsz = 0;
};

std::vector<Phdr> program_headers(const std::vector<uint8_t>& elf) {
    std::vector<Phdr> out;
    const uint64_t phoff = rd64(elf, 32);
    uint16_t phnum = 0;
    std::memcpy(&phnum, elf.data() + 56, 2);
    for (uint16_t i = 0; i < phnum; ++i) {
        const size_t p = static_cast<size_t>(phoff) + i * 56u;
        out.push_back({rd32(elf, p), rd64(elf, p + 16), rd64(elf, p + 32), rd64(elf, p + 40)});
    }
    return out;
}

std::unique_ptr<LirFunction> select_aarch64(Module& mod, const char* fn_name) {
    Function* fn = mod.get_function(fn_name);
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));
    aarch64::AArch64ISel isel(Target::aarch64_linux());
    return isel.lower(*fn);
}

size_t count_opcode(const LirFunction& lir, LirOpcode op) {
    size_t n = 0;
    for (const auto& bb : lir.blocks) {
        for (const auto& li : bb->instructions) {
            if (li && li->opcode == op) ++n;
        }
    }
    return n;
}

// Every register operand of every instruction names a virtual register the
// selector allocated.
bool all_vreg_operands_valid(const LirFunction& lir) {
    auto ok = [&](const LirOperand& op) {
        if (op.is_vreg() && (!op.vreg_val.is_valid() || op.vreg_val.id >= lir.vreg_table.size())) return false;
        return true;
    };
    for (const auto& bb : lir.blocks) {
        for (const auto& li : bb->instructions) {
            for (const auto& d : li->defs) if (!ok(d)) return false;
            for (const auto& u : li->uses) if (!ok(u)) return false;
        }
    }
    return true;
}

} // namespace

TEST_CASE("AArch64 .so - PT_GNU_RELRO lies inside the mapped writable segment") {
    // No .data: before, RELRO ran to the next 64K boundary while the RW
    // segment stopped after .dynamic, and glibc on 4K pages refused the
    // library ("cannot apply additional memory protection after relocation").
    Module mod("relro");
    Function* fn = mod.create_function("cube", Type::i64(), {Type::i64()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    b.build_ret(b.build_mul(b.build_mul(x, x), x));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    object::ObjectFile obj = object::compile_module_to_object(mod, Target::aarch64_linux());
    target::ElfSoOptions opts;
    opts.soname = "librelro.so";
    const std::vector<uint8_t> so = target::ElfSoWriter::emit(obj, opts);

    const std::vector<Phdr> phdrs = program_headers(so);
    const Phdr* relro = nullptr;
    for (const Phdr& p : phdrs) {
        if (p.type == target::elf64::PT_GNU_RELRO) relro = &p;
    }
    REQUIRE(relro != nullptr);
    bool covered = false;
    for (const Phdr& p : phdrs) {
        if (p.type == target::elf64::PT_LOAD && p.vaddr <= relro->vaddr &&
            relro->vaddr + relro->memsz <= p.vaddr + p.memsz) {
            covered = true;
        }
    }
    CHECK(covered);
}

TEST_CASE("AArch64 ISel - an operand the consumer does not fold still has a register") {
    // Analysis used to predict which constant / load operands the lowering
    // would fold and skip them; udiv by a non-power-of-two constant, a mov
    // of a constant and a shift by a loaded amount were all left with no
    // virtual register ("operand has no virtual register").
    const char* src = R"(
module @m
func @f(%0: i64, %1: i64) -> i64 {
entry:
  %2 = iconst.i64 7
  %3 = udiv.i64 %0, %2
  %4 = umod.i64 %1, %2
  %5 = alloca 16, 8
  store.i64 %5, %1
  %6 = load.i64 %5
  %7 = shl.i64 %3, %6
  %8 = lshr.i64 %4, %6
  %9 = add.i64 %7, %8
  ret %9
}
)";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    auto lir = select_aarch64(*mod, "f");
    REQUIRE(lir != nullptr);
    CHECK(all_vreg_operands_valid(*lir));
}

TEST_CASE("AArch64 ISel - unsigned compares with zero are real compares, not tst") {
    // ANDS (tst) clears C; cmp #0 sets it. `ult x, 0` fused into a branch
    // was tst + b.lo, which is always taken.
    for (const char* op : {"ult", "uge", "ule", "ugt"}) {
        std::string src = std::string(R"(
module @m
func @f(%0: i64) -> i64 {
entry:
  %1 = iconst.i64 0
  %2 = )") + op + R"(.i64 %0, %1
  br_if %2, yes, no
yes:
  %3 = iconst.i64 1
  ret %3
no:
  %4 = iconst.i64 2
  ret %4
}
)";
        DiagnosticReporter diag;
        auto mod = parse_module(src, &diag);
        REQUIRE(mod != nullptr);
        auto lir = select_aarch64(*mod, "f");
        REQUIRE(lir != nullptr);
        CHECK_EQ(count_opcode(*lir, LirOpcode::Test), size_t{0});
        CHECK(count_opcode(*lir, LirOpcode::Cmp) >= size_t{1});
    }
}

TEST_CASE("AArch64 ISel - a fused f32 compare compares singles") {
    // The fused branch compared every float as a double (Ucomisd on 8-byte
    // operands), reading garbage above an f32.
    const char* src = R"(
module @m
func @f(%0: f32, %1: f32) -> i64 {
entry:
  %2 = slt.f32 %0, %1
  br_if %2, yes, no
yes:
  %3 = iconst.i64 1
  ret %3
no:
  %4 = iconst.i64 2
  ret %4
}
)";
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    auto lir = select_aarch64(*mod, "f");
    REQUIRE(lir != nullptr);
    CHECK_EQ(count_opcode(*lir, LirOpcode::Ucomisd), size_t{0});
    CHECK_EQ(count_opcode(*lir, LirOpcode::Ucomiss), size_t{1});
}

namespace {

bool has_word(const uint8_t* code, size_t size, uint32_t word) {
    for (size_t off = 0; off + 4 <= size; off += 4) {
        uint32_t w = 0;
        std::memcpy(&w, code + off, 4);
        if (w == word) return true;
    }
    return false;
}

bool has_bytes(const uint8_t* code, size_t size, std::initializer_list<uint8_t> seq) {
    const std::vector<uint8_t> want(seq);
    for (size_t off = 0; off + want.size() <= size; ++off) {
        if (std::memcmp(code + off, want.data(), want.size()) == 0) return true;
    }
    return false;
}

} // namespace

TEST_CASE("AArch64 parallel copy - an i64 parked across a register cycle keeps its upper half") {
    // Edge arguments (i32, i64) whose registers swap: x0 holds the i32 bound
    // for x1, x1 the i64 bound for x0. Breaking the cycle parks x1 in the
    // scratch register at the width of the i32 move, `mov w14, w1`, which
    // zeroed the i64's upper half (fuzz seeds 200889, 201044 and 202352 at
    // 70 statements, and six at 120, all on AArch64 only).
    LirFunction fn;
    fn.name = "swap";
    fn.frame.is_leaf = true;
    auto bb = std::make_unique<LirBlock>(0, "entry");
    auto pc = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
    pc->add_def(LirOperand::preg_aarch64_gpr(aarch64::GPR::X1, 4));
    pc->add_use(LirOperand::preg_aarch64_gpr(aarch64::GPR::X0, 4));
    pc->add_def(LirOperand::preg_aarch64_gpr(aarch64::GPR::X0, 8));
    pc->add_use(LirOperand::preg_aarch64_gpr(aarch64::GPR::X1, 8));
    bb->append_inst(std::move(pc));
    bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));
    fn.blocks.push_back(std::move(bb));

    aarch64::AArch64EmitContext emitter(fn, Target::aarch64_linux());
    auto res = emitter.compile();
    const uint8_t* code = res.code_buffer.data();
    const size_t size = res.code_buffer.size();
    CHECK(has_word(code, size, 0xAA0103EEu));   // mov x14, x1
    CHECK(!has_word(code, size, 0x2A0103EEu));  // mov w14, w1
}

TEST_CASE("x64 parallel copy - an i64 parked across a register cycle keeps its upper half") {
    // The same break in the shared x64 emitter, on a three-register cycle
    // (a two-register one is an xchg): ecx <- eax (i32), rax <- rdx,
    // rdx <- rcx. rcx's i64 is parked in R10 and must be copied whole.
    LirFunction fn;
    fn.name = "rotate";
    fn.calling_conv = CallingConvention::sysv64();
    LirBlock* bb = fn.create_block("entry");
    auto pc = std::make_unique<LirInst>(LirOpcode::ParallelCopy);
    pc->add_def(LirOperand::preg_gpr(x64::GPR::RCX, 4));
    pc->add_use(LirOperand::preg_gpr(x64::GPR::RAX, 4));
    pc->add_def(LirOperand::preg_gpr(x64::GPR::RAX, 8));
    pc->add_use(LirOperand::preg_gpr(x64::GPR::RDX, 8));
    pc->add_def(LirOperand::preg_gpr(x64::GPR::RDX, 8));
    pc->add_use(LirOperand::preg_gpr(x64::GPR::RCX, 8));
    bb->append_inst(std::move(pc));
    bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    EmitContext ctx(fn, Target::x64_linux());
    auto res = ctx.compile();
    const uint8_t* code = res.code_buffer.data();
    const size_t size = res.code_buffer.size();
    CHECK(has_bytes(code, size, {0x49, 0x89, 0xCA}));   // mov r10, rcx
    CHECK(!has_bytes(code, size, {0x41, 0x89, 0xCA}));  // mov r10d, ecx
}

TEST_CASE("x64 fabs - a result spilled through XMM15 is not overwritten by the mask") {
    // XMM15 is the spilled-def scratch; fabs built its sign mask there too,
    // so a spilled fabs result became `andpd xmm15, xmm15` = the mask
    // (a NaN), found by the native Linux x64 fuzz run.
    for (LirOpcode op : {LirOpcode::Fabs64, LirOpcode::Fabs32}) {
        LirFunction fn;
        fn.name = "fabs";
        fn.calling_conv = CallingConvention::sysv64();
        LirBlock* bb = fn.create_block("entry");
        const uint8_t sz = op == LirOpcode::Fabs64 ? 8 : 4;
        auto inst = std::make_unique<LirInst>(op);
        inst->add_def(LirOperand::preg_xmm(x64::XMM::XMM15, sz));
        inst->add_use(LirOperand::preg_xmm(x64::XMM::XMM15, sz));
        bb->append_inst(std::move(inst));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        EmitContext ctx(fn, Target::x64_linux());
        auto res = ctx.compile();
        const uint8_t* code = res.code_buffer.data();
        const size_t size = res.code_buffer.size();
        CHECK(!has_bytes(code, size, {0x66, 0x45, 0x0F, 0x54, 0xFF}));  // andpd xmm15, xmm15
        CHECK(has_bytes(code, size, {0x66, 0x45, 0x0F, 0x54, 0xFE}));   // andpd xmm15, xmm14
    }
}

TEST_CASE("Peephole - a reload is not forwarded from a scratch register an emitter reuses") {
    // movsd [slot0], xmm15 (a spilled def); fabs xmm0 (builds its mask in
    // xmm15 without a LIR def); movsd xmm1, [slot0]. Forwarding the reload
    // to xmm15 read the mask.
    for (bool a64 : {false, true}) {
        LirFunction fn;
        fn.name = "fwd";
        fn.calling_conv = a64 ? CallingConvention::aapcs64() : CallingConvention::sysv64();
        const PReg scratch = a64 ? PReg::aarch64_fpr(aarch64::FPR::V27) : PReg::xmm(x64::XMM::XMM15);
        const PReg r0 = a64 ? PReg::aarch64_fpr(aarch64::FPR::V0) : PReg::xmm(x64::XMM::XMM0);
        const PReg r1 = a64 ? PReg::aarch64_fpr(aarch64::FPR::V1) : PReg::xmm(x64::XMM::XMM1);
        LirBlock* bb = fn.create_block("entry");
        auto store = std::make_unique<LirInst>(LirOpcode::Movsd);
        store->add_def(LirOperand::slot(0, 8));
        store->add_use(LirOperand::preg(scratch, 8));
        bb->append_inst(std::move(store));
        auto fabs = std::make_unique<LirInst>(LirOpcode::Fabs64);
        fabs->add_def(LirOperand::preg(r0, 8));
        fabs->add_use(LirOperand::preg(r0, 8));
        bb->append_inst(std::move(fabs));
        auto reload = std::make_unique<LirInst>(LirOpcode::Movsd);
        reload->add_def(LirOperand::preg(r1, 8));
        reload->add_use(LirOperand::slot(0, 8));
        LirInst* reload_ptr = reload.get();
        bb->append_inst(std::move(reload));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        run_lir_peephole_optimizations(fn);
        bool reload_kept = false;
        for (const auto& inst : fn.blocks[0]->instructions) {
            if (inst.get() == reload_ptr) reload_kept = inst->uses[0].is_spill_slot();
        }
        CHECK(reload_kept);
    }
}

TEST_CASE("AArch64 regalloc - a vector live across a call is not kept in v8-v15") {
    // AAPCS64 preserves only the low 64 bits of v8-v15 (and the prologue
    // saves only d8-d15), so a 128-bit value kept there over a call comes
    // back with its upper lanes clobbered. It has to be spilled.
    Module mod;
    Function* fn = mod.create_function("f", Type::i32(), {Type::i32()});
    Builder b(mod);
    b.set_function(fn);
    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i32());
    Value* v = b.build_vbroadcast(Type::i32x4(), x);
    b.build_call("ext", Type::i64());
    Value* w = b.build_vadd(v, v);
    b.build_ret(b.build_vextract_lane(w, 3));
    fn->rebuild_cfg_predecessors();
    REQUIRE(verify_function(*fn));

    aarch64::AArch64ISel isel(Target::aarch64_linux());
    auto lir = isel.lower(*fn);
    REQUIRE(lir != nullptr);
    LivenessAnalysis liveness(*lir);
    liveness.run();
    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::aapcs64());
    regalloc.allocate();

    size_t spilled_vectors = 0;
    for (const VRegInfo& info : lir->vreg_table) {
        if (info.vreg.is_gpr() || info.vreg.size <= 8) continue;
        if (info.is_spilled) {
            ++spilled_vectors;
        } else if (info.assigned_preg.is_valid()) {
            const auto code = static_cast<int>(info.assigned_preg.as_aarch64_fpr());
            CHECK(code < 8 || code > 15);
        }
    }
    CHECK(spilled_vectors >= 1);
    CHECK_EQ(lir->frame.saved_callee_xmms, 0u);
}
