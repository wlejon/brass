// Unit tests for PtxISel (MIR -> ptx::Function).
//
//   (a) block-argument edges go through a real parallel-copy resolver
//       (swap, three-cycle, br_if with args on both edges), asserted on the
//       lowered IR by replaying the emitted moves, and on device when CUDA
//       is available,
//   (b) predicate materialization, 64-bit shift counts and unsigned ops
//       pick the right suffixes,
//   (c) every entry of the intrinsic table lowers, verifies and assembles
//       with ptxas (skipped with a visible [SKIP] when ptxas is missing).

#include "test_framework.hpp"

#include <brass/gpu/cuda_driver.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/target/ptx/ptx_isel.hpp>
#include <brass/target/ptx/ptx_printer.hpp>
#include <brass/target/ptx/ptx_verifier.hpp>
#include <brass/target/ptx_target.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace brass;
using brass::gpu::CudaBuffer;
using brass::gpu::CudaModule;

// MIR and PTX IR both have Opcode/Type; MIR's are unqualified here, the PTX
// IR's are spelled ptx::Opcode / ptx::Type.
using ptx::Block;
using ptx::CmpOp;
using ptx::Inst;
using ptx::PtxISel;
using ptx::Reg;
using ptx::RegClass;

namespace {

// ---------------------------------------------------------------------------
// ptxas / device helpers (same pattern as test_gpu_execution.cpp)
// ---------------------------------------------------------------------------

void report_skip(const char* what) {
    std::cout << "  [SKIP] " << what << "\n" << std::flush;
}

const char* null_device() {
#if defined(_WIN32)
    return "NUL";
#else
    return "/dev/null";
#endif
}

std::string quiet(const std::string& cmd) {
    return cmd + " >" + null_device() + " 2>&1";
}

std::filesystem::path scratch_path(const char* name) {
    return brass::test::scratch_dir() / name;
}

bool ptxas_available() {
    static const bool ok = (std::system(quiet("ptxas --version").c_str()) == 0);
    if (!ok) report_skip("ptxas not on PATH: assembly not validated");
    return ok;
}

bool ptxas_assembles(const std::string& ptx, const char* arch) {
    std::filesystem::path in = scratch_path("brass_ptx_isel_check.ptx");
    std::filesystem::path out = scratch_path("brass_ptx_isel_check.cubin");
    {
        std::ofstream f(in, std::ios::binary);
        if (!f) return false;
        f << ptx;
    }
    std::string cmd = std::string("ptxas -arch=") + arch + " \"" + in.string() + "\" -o \"" + out.string() + "\"";
    bool ok = std::system(quiet(cmd).c_str()) == 0;
    std::error_code ec;
    std::filesystem::remove(out, ec);
    return ok;
}

bool gpu_ready() {
    static const bool ok = brass::gpu::cuda_available();
    if (!ok) report_skip(("CUDA unavailable (" + brass::gpu::cuda_last_error() + "): test not executed on device").c_str());
    return ok;
}

// ---------------------------------------------------------------------------
// IR helpers
// ---------------------------------------------------------------------------

// Lower, verify (dumping diagnostics on failure) and return the IR.
ptx::Function lower_ok(const brass::Function& fn) {
    PtxISel isel;
    ptx::Function out = isel.lower(fn);
    auto diags = ptx::verify(out);
    if (!diags.empty()) std::cerr << ptx::format_diagnostics(diags) << ptx::print_body(out);
    REQUIRE(diags.empty());
    return out;
}

const Block* block_of(const ptx::Function& fn, const BasicBlock* bb) {
    const Block* b = fn.find_block("$L_bb_" + std::to_string(bb->id()));
    REQUIRE(b != nullptr);
    return b;
}

std::string reg_name(Reg r) { return ptx::to_string(r); }

// Replay the `mov` instructions of a block (optionally only the guarded or
// only the unguarded ones) over a symbolic register file. Registers that are
// read before being written take their own name as value, so after a correct
// parallel copy each destination holds the name of its original source.
std::map<std::string, std::string> replay_moves(const Block& bb, int guarded /* 1 = guarded, 0 = unguarded, -1 = all */) {
    std::map<std::string, std::string> file;
    auto read = [&](Reg r) {
        auto it = file.find(reg_name(r));
        return it == file.end() ? reg_name(r) : it->second;
    };
    for (const Inst& inst : bb.insts) {
        if (inst.op != ptx::Opcode::mov) continue;
        if (guarded == 1 && !inst.has_guard) continue;
        if (guarded == 0 && inst.has_guard) continue;
        REQUIRE(inst.dsts.size() == 1 && inst.srcs.size() == 1);
        if (!inst.srcs[0].is_reg()) continue; // constants
        std::string v = read(inst.srcs[0].reg_val);
        file[reg_name(inst.dsts[0].reg_val)] = v;
    }
    return file;
}

size_t count_ops(const Block& bb, ptx::Opcode op, int guarded = -1) {
    size_t n = 0;
    for (const Inst& inst : bb.insts) {
        if (inst.op != op) continue;
        if (guarded == 1 && !inst.has_guard) continue;
        if (guarded == 0 && inst.has_guard) continue;
        ++n;
    }
    return n;
}

const Inst* find_op(const Block& bb, ptx::Opcode op) {
    for (const Inst& inst : bb.insts) if (inst.op == op) return &inst;
    return nullptr;
}

// Launch a 1-thread kernel with the given scalar/pointer args and read `out`.
template <typename T>
std::vector<T> run_kernel(const std::string& ptx, const char* entry, std::vector<void*> args,
                          CudaBuffer& out, size_t n_out) {
    std::string err;
    CudaModule mod = CudaModule::load(ptx, &err);
    if (!mod.valid()) std::cerr << "driver JIT: " << err << "\n" << ptx;
    REQUIRE(mod.valid());
    bool launched = mod.launch_1d(entry, 1, 1, args.data(), 0, &err);
    if (!launched) std::cerr << "launch: " << err << "\n";
    REQUIRE(launched);
    std::vector<T> got(n_out);
    REQUIRE(out.download(got.data(), n_out * sizeof(T)));
    return got;
}

// ---------------------------------------------------------------------------
// Kernels under test
// ---------------------------------------------------------------------------

// swap(a, b, out): loop(x, y, i) stores (x, y) at out[2i], then
// br loop(y, x, i + 1) while i + 1 < iters. Exercises the 2-cycle.
struct SwapKernel {
    Module mod{"swap"};
    brass::Function* fn = nullptr;
    BasicBlock* loop = nullptr;

    explicit SwapKernel(int iters) {
        fn = mod.create_function("swap_kernel", Type::void_type(), {Type::i32(), Type::i32(), Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        loop = b.append_block("loop");
        BasicBlock* exit = b.append_block("exit");

        b.position_at_end(entry);
        b.add_block_param(entry, Type::i32());
        b.add_block_param(entry, Type::i32());
        b.add_block_param(entry, Type::ptr());
        Value* zero = b.build_iconst_i64(0);
        b.build_br(loop, {entry->param(0), entry->param(1), zero});

        b.position_at_end(loop);
        Value* x = b.add_block_param(loop, Type::i32());
        Value* y = b.add_block_param(loop, Type::i32());
        Value* i = b.add_block_param(loop, Type::i64());
        Value* two = b.build_iconst_i64(2);
        Value* slot = b.build_mul(i, two);
        b.build_store_indexed(Type::i32(), entry->param(2), slot, 4, 0, x);
        b.build_store_indexed(Type::i32(), entry->param(2), slot, 4, 4, y);
        Value* next = b.build_add(i, b.build_iconst_i64(1));
        Value* more = b.build_slt(next, b.build_iconst_i64(iters));
        b.build_br_if(more, loop, {y, x, next}, exit, {});

        b.position_at_end(exit);
        b.build_ret_void();
    }
};

// rotate3(a, b, c, out): loop(x, y, z, i) stores (x, y, z) at out[3i], then
// br loop(y, z, x, i + 1). Exercises a 3-cycle.
struct Rotate3Kernel {
    Module mod{"rot3"};
    brass::Function* fn = nullptr;
    BasicBlock* loop = nullptr;

    explicit Rotate3Kernel(int iters) {
        fn = mod.create_function("rot3_kernel", Type::void_type(),
                                 {Type::i32(), Type::i32(), Type::i32(), Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        loop = b.append_block("loop");
        BasicBlock* exit = b.append_block("exit");

        b.position_at_end(entry);
        for (int k = 0; k < 3; ++k) b.add_block_param(entry, Type::i32());
        b.add_block_param(entry, Type::ptr());
        Value* zero = b.build_iconst_i64(0);
        b.build_br(loop, {entry->param(0), entry->param(1), entry->param(2), zero});

        b.position_at_end(loop);
        Value* x = b.add_block_param(loop, Type::i32());
        Value* y = b.add_block_param(loop, Type::i32());
        Value* z = b.add_block_param(loop, Type::i32());
        Value* i = b.add_block_param(loop, Type::i64());
        Value* slot = b.build_mul(i, b.build_iconst_i64(3));
        Value* out = entry->param(3);
        b.build_store_indexed(Type::i32(), out, slot, 4, 0, x);
        b.build_store_indexed(Type::i32(), out, slot, 4, 4, y);
        b.build_store_indexed(Type::i32(), out, slot, 4, 8, z);
        Value* next = b.build_add(i, b.build_iconst_i64(1));
        Value* more = b.build_slt(next, b.build_iconst_i64(iters));
        b.build_br_if(more, loop, {y, z, x, next}, exit, {});

        b.position_at_end(exit);
        b.build_ret_void();
    }
};

// both_edges(cond, a, b, out): br_if cond, t(a, b), f(b, a); t stores its
// params at out[0..1], f stores its params at out[0..1] plus 100.
struct BothEdgesKernel {
    Module mod{"both"};
    brass::Function* fn = nullptr;
    BasicBlock* entry = nullptr;

    BothEdgesKernel() {
        fn = mod.create_function("both_edges", Type::void_type(),
                                 {Type::i32(), Type::i32(), Type::i32(), Type::ptr()});
        Builder b(mod);
        b.set_function(fn);
        entry = b.append_block("entry");
        BasicBlock* t = b.append_block("t");
        BasicBlock* f = b.append_block("f");

        b.position_at_end(entry);
        Value* cond = b.add_block_param(entry, Type::i32());
        Value* a = b.add_block_param(entry, Type::i32());
        Value* bb = b.add_block_param(entry, Type::i32());
        Value* out = b.add_block_param(entry, Type::ptr());
        b.build_br_if(cond, t, {a, bb}, f, {bb, a});

        b.position_at_end(t);
        Value* p = b.add_block_param(t, Type::i32());
        Value* q = b.add_block_param(t, Type::i32());
        b.build_store(Type::i32(), out, 0, p);
        b.build_store(Type::i32(), out, 4, q);
        b.build_ret_void();

        b.position_at_end(f);
        Value* r = b.add_block_param(f, Type::i32());
        Value* s = b.add_block_param(f, Type::i32());
        Value* hundred = b.build_iconst_i32(100);
        b.build_store(Type::i32(), out, 0, b.build_add(r, hundred));
        b.build_store(Type::i32(), out, 4, b.build_add(s, hundred));
        b.build_ret_void();
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Parallel copies
// ---------------------------------------------------------------------------

TEST_CASE("PTX ISel - parallel copy swap uses a scratch register") {
    SwapKernel k(2);
    ptx::Function fn = lower_ok(*k.fn);
    const Block* loop = block_of(fn, k.loop);

    // Guarded (taken-edge) moves: x <- y, y <- x, i <- next, plus one scratch.
    CHECK_EQ(count_ops(*loop, ptx::Opcode::mov, 1), size_t(4));
    // The guarded bra comes after every guarded mov.
    bool seen_bra = false;
    for (const Inst& inst : loop->insts) {
        if (inst.op == ptx::Opcode::bra && inst.has_guard) seen_bra = true;
        if (inst.op == ptx::Opcode::mov && inst.has_guard) CHECK(!seen_bra);
    }

    // Registers: entry params a=%r0 b=%r1; loop params x=%r2 y=%r3.
    Reg lx = Reg(RegClass::B32, 2), ly = Reg(RegClass::B32, 3);
    auto file = replay_moves(*loop, 1);
    CHECK_EQ(file[reg_name(lx)], reg_name(ly));
    CHECK_EQ(file[reg_name(ly)], reg_name(lx));

    std::string ptx = target::PtxTarget::emit_function(*k.fn);
    if (ptxas_available()) CHECK(ptxas_assembles(ptx, "sm_70"));
    if (!gpu_ready()) return;

    CudaBuffer out = CudaBuffer::alloc(4 * 4);
    REQUIRE(out.valid() && out.zero());
    int32_t a = 1, b = 2;
    void* po = out.device_ptr();
    auto got = run_kernel<int32_t>(ptx, "swap_kernel", {&a, &b, &po}, out, 4);
    CHECK_EQ(got[0], 1); CHECK_EQ(got[1], 2);   // iteration 0: (a, b)
    CHECK_EQ(got[2], 2); CHECK_EQ(got[3], 1);   // iteration 1: swapped
}

TEST_CASE("PTX ISel - parallel copy resolves a cycle of three") {
    Rotate3Kernel k(3);
    ptx::Function fn = lower_ok(*k.fn);
    const Block* loop = block_of(fn, k.loop);

    // x <- y, y <- z, z <- x, i <- next, plus one scratch: 5 guarded moves.
    CHECK_EQ(count_ops(*loop, ptx::Opcode::mov, 1), size_t(5));
    // Registers: entry params a,b,c=%r0..%r2; loop params x,y,z=%r3..%r5.
    Reg lx = Reg(RegClass::B32, 3), ly = Reg(RegClass::B32, 4), lz = Reg(RegClass::B32, 5);
    auto file = replay_moves(*loop, 1);
    CHECK_EQ(file[reg_name(lx)], reg_name(ly));
    CHECK_EQ(file[reg_name(ly)], reg_name(lz));
    CHECK_EQ(file[reg_name(lz)], reg_name(lx));

    std::string ptx = target::PtxTarget::emit_function(*k.fn);
    if (ptxas_available()) CHECK(ptxas_assembles(ptx, "sm_70"));
    if (!gpu_ready()) return;

    CudaBuffer out = CudaBuffer::alloc(9 * 4);
    REQUIRE(out.valid() && out.zero());
    int32_t a = 1, b = 2, c = 3;
    void* po = out.device_ptr();
    auto got = run_kernel<int32_t>(ptx, "rot3_kernel", {&a, &b, &c, &po}, out, 9);
    int32_t expect[9] = {1, 2, 3, 2, 3, 1, 3, 1, 2};
    for (int i = 0; i < 9; ++i) CHECK_EQ(got[i], expect[i]);
}

TEST_CASE("PTX ISel - br_if with arguments on both edges") {
    BothEdgesKernel k;
    ptx::Function fn = lower_ok(*k.fn);
    const Block* entry = block_of(fn, k.entry);

    // Layout: setp (materialized from the i32 cond), guarded moves, guarded
    // bra, unguarded moves, bra.
    const Inst* setp = find_op(*entry, ptx::Opcode::setp);
    REQUIRE(setp != nullptr);
    CHECK(setp->cmp_op == CmpOp::ne);
    CHECK(setp->type == ptx::Type::u32);
    CHECK_EQ(count_ops(*entry, ptx::Opcode::mov, 1), size_t(2));
    CHECK_EQ(count_ops(*entry, ptx::Opcode::mov, 0), size_t(2));
    int phase = 0; // 0: guarded moves, 1: after guarded bra, 2: after final bra
    for (const Inst& inst : entry->insts) {
        if (inst.op == ptx::Opcode::mov) {
            if (inst.has_guard) CHECK_EQ(phase, 0);
            else CHECK_EQ(phase, 1);
        } else if (inst.op == ptx::Opcode::bra) {
            if (inst.has_guard) { CHECK_EQ(phase, 0); phase = 1; }
            else { CHECK_EQ(phase, 1); phase = 2; }
        }
    }
    CHECK_EQ(phase, 2);

    // Registers: entry params cond=%r0 a=%r1 b=%r2; t params %r3 %r4; f params %r5 %r6.
    auto taken = replay_moves(*entry, 1);
    CHECK_EQ(taken["%r3"], std::string("%r1"));
    CHECK_EQ(taken["%r4"], std::string("%r2"));
    auto fall = replay_moves(*entry, 0);
    CHECK_EQ(fall["%r5"], std::string("%r2"));
    CHECK_EQ(fall["%r6"], std::string("%r1"));

    std::string ptx = target::PtxTarget::emit_function(*k.fn);
    if (ptxas_available()) CHECK(ptxas_assembles(ptx, "sm_70"));
    if (!gpu_ready()) return;

    int32_t a = 7, b = 9;
    for (int32_t cond : {1, 0}) {
        CudaBuffer out = CudaBuffer::alloc(8);
        REQUIRE(out.valid() && out.zero());
        void* po = out.device_ptr();
        int32_t c = cond;
        auto got = run_kernel<int32_t>(ptx, "both_edges", {&c, &a, &b, &po}, out, 2);
        if (cond) { CHECK_EQ(got[0], 7); CHECK_EQ(got[1], 9); }
        else      { CHECK_EQ(got[0], 109); CHECK_EQ(got[1], 107); }
    }
}

// ---------------------------------------------------------------------------
// Suffix selection
// ---------------------------------------------------------------------------

TEST_CASE("PTX ISel - select on an i32 condition materializes setp.ne") {
    Module mod("sel");
    brass::Function* f = mod.create_function("sel", Type::void_type(), {Type::i32(), Type::f32(), Type::f32(), Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* c = b.add_block_param(e, Type::i32());
    Value* t = b.add_block_param(e, Type::f32());
    Value* u = b.add_block_param(e, Type::f32());
    Value* out = b.add_block_param(e, Type::ptr());
    b.build_store(Type::f32(), out, 0, b.build_select(c, t, u));
    b.build_ret_void();

    ptx::Function fn = lower_ok(*f);
    const Block* bb = block_of(fn, e);
    const Inst* setp = find_op(*bb, ptx::Opcode::setp);
    const Inst* selp = find_op(*bb, ptx::Opcode::selp);
    REQUIRE(setp != nullptr);
    REQUIRE(selp != nullptr);
    CHECK(setp->type == ptx::Type::u32);
    CHECK(setp->cmp_op == CmpOp::ne);
    CHECK(setp->srcs[1].is_imm_int() && setp->srcs[1].imm_int == 0);
    CHECK(selp->type == ptx::Type::f32);
    CHECK(selp->srcs[2].is_reg() && selp->srcs[2].reg_val == setp->dsts[0].reg_val);

    std::string ptx = target::PtxTarget::emit_function(*f);
    CHECK(ptx.find("setp.ne.u32") != std::string::npos);
    CHECK(ptx.find("selp.f32") != std::string::npos);
}

TEST_CASE("PTX ISel - comparisons used only as conditions do not materialize an integer") {
    Module mod("cmp");
    brass::Function* f = mod.create_function("cmp", Type::void_type(), {Type::i32(), Type::i32(), Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    BasicBlock* t = b.append_block("t");
    BasicBlock* x = b.append_block("x");
    b.position_at_end(e);
    Value* p = b.add_block_param(e, Type::i32());
    Value* q = b.add_block_param(e, Type::i32());
    Value* out = b.add_block_param(e, Type::ptr());
    Value* lt = b.build_slt(p, q);        // condition only
    Value* eq = b.build_eq(p, q);         // stored as a value
    b.build_store(Type::i32(), out, 0, eq);
    b.build_br_if(lt, t, x);
    b.position_at_end(t);
    b.build_store(Type::i32(), out, 4, p);
    b.build_ret_void();
    b.position_at_end(x);
    b.build_ret_void();

    ptx::Function fn = lower_ok(*f);
    const Block* bb = block_of(fn, e);
    CHECK_EQ(count_ops(*bb, ptx::Opcode::setp), size_t(2));
    CHECK_EQ(count_ops(*bb, ptx::Opcode::selp), size_t(1)); // only for `eq`
    const Inst* bra = find_op(*bb, ptx::Opcode::bra);
    REQUIRE(bra != nullptr);
    CHECK(bra->has_guard);
    std::string ptx = target::PtxTarget::emit_function(*f);
    CHECK(ptx.find("setp.lt.s32") != std::string::npos);
    CHECK(ptx.find("setp.eq.s32") != std::string::npos);
}

TEST_CASE("PTX ISel - 64-bit shift by a 64-bit count narrows the count") {
    Module mod("shift");
    brass::Function* f = mod.create_function("shift", Type::void_type(), {Type::i64(), Type::i64(), Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* v = b.add_block_param(e, Type::i64());
    Value* n = b.add_block_param(e, Type::i64());
    Value* out = b.add_block_param(e, Type::ptr());
    b.build_store(Type::i64(), out, 0, b.build_shl(v, n));
    b.build_store(Type::i64(), out, 8, b.build_lshr(v, n));
    b.build_store(Type::i64(), out, 16, b.build_ashr(v, n));
    b.build_ret_void();

    ptx::Function fn = lower_ok(*f);
    const Block* bb = block_of(fn, e);
    size_t cvts = 0;
    for (const Inst& inst : bb->insts) {
        if (inst.op == ptx::Opcode::cvt) {
            ++cvts;
            CHECK(inst.type == ptx::Type::u32);
            CHECK(inst.src_type == ptx::Type::u64);
            CHECK(inst.dsts[0].reg_val.cls == RegClass::B32);
        }
        if (inst.op == ptx::Opcode::shl || inst.op == ptx::Opcode::shr) {
            CHECK(inst.srcs[1].is_reg() && inst.srcs[1].reg_val.cls == RegClass::B32);
        }
    }
    CHECK_EQ(cvts, size_t(3));
    std::string ptx = target::PtxTarget::emit_function(*f);
    CHECK(ptx.find("cvt.u32.u64") != std::string::npos);
    CHECK(ptx.find("shl.b64") != std::string::npos);
    CHECK(ptx.find("shr.u64") != std::string::npos);
    CHECK(ptx.find("shr.s64") != std::string::npos);
}

TEST_CASE("PTX ISel - unsigned div/rem/compare on i32 and i64") {
    Module mod("uops");
    brass::Function* f = mod.create_function("uops", Type::void_type(),
                                             {Type::i32(), Type::i32(), Type::i64(), Type::i64(), Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* a32 = b.add_block_param(e, Type::i32());
    Value* b32 = b.add_block_param(e, Type::i32());
    Value* a64 = b.add_block_param(e, Type::i64());
    Value* b64 = b.add_block_param(e, Type::i64());
    Value* out = b.add_block_param(e, Type::ptr());
    b.build_store(Type::i32(), out, 0, b.build_udiv(a32, b32));
    b.build_store(Type::i32(), out, 4, b.build_umod(a32, b32));
    b.build_store(Type::i32(), out, 8, b.build_ult(a32, b32));
    b.build_store(Type::i64(), out, 16, b.build_udiv(a64, b64));
    b.build_store(Type::i64(), out, 24, b.build_umod(a64, b64));
    b.build_store(Type::i32(), out, 32, b.build_ult(a64, b64));
    b.build_ret_void();

    ptx::Function fn = lower_ok(*f);
    std::string ptx = target::PtxTarget::emit_function(*f);
    for (const char* needle : {"div.u32", "rem.u32", "setp.lt.u32", "div.u64", "rem.u64", "setp.lt.u64"}) {
        if (ptx.find(needle) == std::string::npos) std::cerr << "missing " << needle << "\n" << ptx;
        CHECK(ptx.find(needle) != std::string::npos);
    }
    CHECK(ptx.find("div.s") == std::string::npos);
    CHECK(ptx.find("rem.s") == std::string::npos);
    if (ptxas_available()) CHECK(ptxas_assembles(ptx, "sm_70"));
    if (!gpu_ready()) return;

    uint32_t x32 = 0xFFFFFFF0u, y32 = 7u;
    uint64_t x64 = 0xFFFFFFFFFFFFFFF0ull, y64 = 9ull;
    CudaBuffer out_buf = CudaBuffer::alloc(40);
    REQUIRE(out_buf.valid() && out_buf.zero());
    void* po = out_buf.device_ptr();
    auto got = run_kernel<uint32_t>(ptx, "uops", {&x32, &y32, &x64, &y64, &po}, out_buf, 10);
    CHECK_EQ(got[0], x32 / y32);
    CHECK_EQ(got[1], x32 % y32);
    CHECK_EQ(got[2], 0u);
    uint64_t q64 = uint64_t(got[4]) | (uint64_t(got[5]) << 32);
    uint64_t r64 = uint64_t(got[6]) | (uint64_t(got[7]) << 32);
    CHECK_EQ(q64, x64 / y64);
    CHECK_EQ(r64, x64 % y64);
    CHECK_EQ(got[8], 0u);
}

TEST_CASE("PTX ISel - integer division follows MIR for zero and MIN / -1 divisors") {
    Module mod("sdivs");
    brass::Function* f = mod.create_function("sdivs", Type::void_type(), {Type::i32(), Type::i32(), Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* x = b.add_block_param(e, Type::i32());
    Value* y = b.add_block_param(e, Type::i32());
    Value* out = b.add_block_param(e, Type::ptr());
    b.build_store(Type::i32(), out, 0, b.build_sdiv(x, y));
    b.build_store(Type::i32(), out, 4, b.build_smod(x, y));
    b.build_store(Type::i32(), out, 8, b.build_sdiv(x, b.build_iconst_i32(7)));
    b.build_ret_void();

    lower_ok(*f);
    std::string ptx = target::PtxTarget::emit_function(*f);
    auto count = [&ptx](const std::string& needle) {
        size_t n = 0;
        for (size_t at = ptx.find(needle); at != std::string::npos; at = ptx.find(needle, at + 1)) ++n;
        return n;
    };
    // A variable divisor is tested for zero (trap) and for -1 (the wrapped
    // result is selected); the constant divisor 7 needs neither.
    if (count("trap;") != 2) std::cerr << ptx;
    CHECK_EQ(count("trap;"), size_t{2});
    CHECK_EQ(count("setp.eq.s32"), size_t{4});
    CHECK_EQ(count("neg.s32"), size_t{1});
    CHECK_EQ(count("selp.b32"), size_t{2});
    CHECK_EQ(count("div.s32"), size_t{2});
    CHECK_EQ(count("rem.s32"), size_t{1});
    if (ptxas_available()) CHECK(ptxas_assembles(ptx, "sm_70"));
    if (!gpu_ready()) return;

    int32_t xs = INT32_MIN, ys = -1;
    CudaBuffer out_buf = CudaBuffer::alloc(12);
    REQUIRE(out_buf.valid() && out_buf.zero());
    void* po = out_buf.device_ptr();
    auto got = run_kernel<int32_t>(ptx, "sdivs", {&xs, &ys, &po}, out_buf, 3);
    CHECK_EQ(got[0], INT32_MIN);
    CHECK_EQ(got[1], 0);
    CHECK_EQ(got[2], INT32_MIN / 7);
}

// ---------------------------------------------------------------------------
// Intrinsic table
// ---------------------------------------------------------------------------

namespace {

// Builds a kernel that calls `name` with arguments matching its signature and
// stores any result through the pointer param.
std::string intrinsic_kernel_ptx(std::string_view name) {
    Module mod("intrin");
    brass::Function* f = mod.create_function("intrin", Type::void_type(),
                                             {Type::i32(), Type::f32(), Type::ptr()});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Value* i = b.add_block_param(e, Type::i32());
    Value* x = b.add_block_param(e, Type::f32());
    Value* out = b.add_block_param(e, Type::ptr());

    std::string n(name);
    Value* r = nullptr;
    if (n.find("shfl_down_sync") != std::string::npos) {
        r = b.build_call(name, Type::f32(), {i, x, i});
    } else if (n.find("shfl_down") != std::string::npos) {
        r = b.build_call(name, Type::f32(), {x, i});
    } else if (n.find("sync") != std::string::npos) {
        b.build_call(name, Type::void_type());
    } else if (n == "i32_to_f32") {
        r = b.build_call(name, Type::f32(), {i});
    } else if (n.find("rsqrt") != std::string::npos || n.find("sqrt") != std::string::npos ||
               n.find("sin") != std::string::npos || n.find("cos") != std::string::npos ||
               n.find("ex2") != std::string::npos || n.find("exp") != std::string::npos) {
        r = b.build_call(name, Type::f32(), {x});
    } else {
        r = b.build_call(name, Type::i32());
    }
    if (r) b.build_store(r->type(), out, 0, r);
    b.build_ret_void();
    return target::PtxTarget::emit_function(*f);
}

} // namespace

// The intrinsic names and aliases the original string-matching emitter
// accepted; the full table is covered signature-by-signature in
// test_ptx_intrinsics.cpp.
TEST_CASE("PTX ISel - every original intrinsic alias lowers, verifies and assembles") {
    std::vector<std::string_view> names = {
             "ptx_tid_x", "ptx_tid_y", "ptx_tid_z", "ptx_ctaid_x", "ptx_ctaid_y", "ptx_ctaid_z",
             "ptx_ntid_x", "ptx_ntid_y", "ptx_ntid_z", "ptx_global_tid_x", "ptx_global_id_x",
             "rsqrtf", "rsqrt", "ptx_rsqrt", "sqrtf", "sqrt", "ptx_sqrt", "sinf", "sin", "ptx_sin",
             "cosf", "cos", "ptx_cos", "ex2f", "ex2", "ptx_ex2", "expf", "exp", "ptx_exp",
             "i32_to_f32", "bar.sync", "ptx_sync", "ptx_shfl_down_sync_f32", "shfl_down_sync_f32",
             "ptx_shfl_down_f32", "ptx_laneid", "ptx_lane_id", "ptx_warpid", "ptx_warp_id" };
    // Every name the string-matching emitter accepted must still be present.
    for (std::string_view required : names) {
        bool present = PtxISel::is_intrinsic(required);
        if (!present) std::cerr << "intrinsic table is missing " << required << "\n";
        CHECK(present);
    }
    CHECK(!PtxISel::is_intrinsic("custom_device_helper"));
    CHECK(PtxISel::intrinsic_names().size() >= names.size());

    bool have_ptxas = ptxas_available();
    for (std::string_view name : names) {
        std::string ptx;
        try {
            ptx = intrinsic_kernel_ptx(name); // lowers + verifies inside PtxTarget
        } catch (const std::exception& ex) {
            std::cerr << "intrinsic " << name << ": " << ex.what() << "\n";
            CHECK(false);
            continue;
        }
        CHECK(ptx.find("call") == std::string::npos); // lowered inline, not a call
        if (have_ptxas) {
            bool ok = ptxas_assembles(ptx, "sm_70");
            if (!ok) std::cerr << "ptxas rejected intrinsic " << name << ":\n" << ptx;
            CHECK(ok);
        }
    }
}

TEST_CASE("PTX ISel - verifier failures surface as loud errors, unknown opcodes name the opcode") {
    Module mod("bad");
    brass::Function* f = mod.create_function("bad", Type::void_type(), {});
    Builder b(mod);
    b.set_function(f);
    BasicBlock* e = b.append_block("entry");
    b.position_at_end(e);
    Instruction* bad = mod.arena().make<Instruction>(Opcode::popcnt, Type::i32());
    Value* v = mod.arena().make<Value>(f->next_value_id(), Type::i32(), ValueKind::InstructionResult);
    v->set_defining_instruction(bad);
    bad->set_result(v);
    e->append_instruction(bad);
    b.build_ret_void();

    bool threw = false;
    try {
        target::PtxTarget::emit_function(*f);
    } catch (const std::runtime_error& err) {
        threw = true;
        std::string msg = err.what();
        CHECK(msg.find("unsupported opcode") != std::string::npos);
        CHECK(msg.find("popcnt") != std::string::npos);
    }
    CHECK(threw);
}
