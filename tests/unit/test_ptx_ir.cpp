// Unit tests for the typed PTX IR (ptx_ir.hpp), its printer and its verifier.
//
//   (a) exact printed text for every operand kind and modifier combination,
//   (b) the verifier rejects each malformed construct with a located message,
//   (c) a real kernel built by hand in the IR verifies, assembles with ptxas
//       (skipped when ptxas is missing) and runs on a device (skipped when no
//       CUDA device is available).

#include "test_framework.hpp"

#include <brass/gpu/cuda_driver.hpp>
#include <brass/target/ptx/ptx_ir.hpp>
#include <brass/target/ptx/ptx_printer.hpp>
#include <brass/target/ptx/ptx_verifier.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace brass::ptx;
using brass::gpu::CudaBuffer;
using brass::gpu::CudaModule;

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
    std::error_code ec;
    std::filesystem::path dir = std::filesystem::temp_directory_path(ec);
    if (ec) dir = std::filesystem::current_path();
    return dir / name;
}

bool ptxas_available() {
    static const bool ok = (std::system(quiet("ptxas --version").c_str()) == 0);
    if (!ok) report_skip("ptxas not on PATH: assembly not validated");
    return ok;
}

bool ptxas_assembles(const std::string& ptx, const char* arch) {
    std::filesystem::path in = scratch_path("brass_ptx_ir_check.ptx");
    std::filesystem::path out = scratch_path("brass_ptx_ir_check.cubin");
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

bool near(float a, float b, float eps) {
    return std::fabs(a - b) <= eps * (1.0f + std::fabs(b));
}

// ---------------------------------------------------------------------------
// Verifier helpers
// ---------------------------------------------------------------------------

bool mentions(const std::vector<Diagnostic>& diags, const std::string& needle) {
    for (const auto& d : diags) {
        if (d.message.find(needle) != std::string::npos) return true;
    }
    return false;
}

void dump(const std::vector<Diagnostic>& diags) {
    if (!diags.empty()) std::cerr << format_diagnostics(diags);
}

// A function with one of each register class and a param, ready for a single
// instruction under test.
struct Fixture {
    Function fn{"k"};
    Block* bb = nullptr;
    Reg p, r, r1, rd, rd1, f, f1, fd;

    Fixture() {
        fn.add_param(Type::u64, "param_a");
        fn.add_shared(Type::f32, "smem", 32, 4);
        p = fn.new_pred();
        r = fn.new_b32(); r1 = fn.new_b32();
        rd = fn.new_b64(); rd1 = fn.new_b64();
        f = fn.new_f32(); f1 = fn.new_f32();
        fd = fn.new_f64();
        bb = fn.add_block("$L_entry");
    }

    std::vector<Diagnostic> reject(Inst i, const std::string& needle) {
        bb->append(std::move(i));
        bb->append(Inst::make(Opcode::ret));
        auto diags = verify(fn);
        if (diags.empty()) std::cerr << "expected a diagnostic mentioning: " << needle << "\n";
        else if (!mentions(diags, needle)) std::cerr << "diagnostics did not mention '" << needle << "':\n" << format_diagnostics(diags);
        CHECK(!diags.empty());
        CHECK(mentions(diags, needle));
        // Every diagnostic must locate the instruction.
        for (const auto& d : diags) {
            CHECK(d.function == "k");
            CHECK(d.block == "$L_entry");
            CHECK(d.inst_index == 0);
            CHECK(!d.inst_text.empty());
        }
        return diags;
    }
};

// ---------------------------------------------------------------------------
// The end-to-end kernel: out[i] = a[i] + b[i] (v4 fast path + scalar tail),
// sum[0] += reduce(a + b) via shfl warp reduction, .shared, bar.sync, atom.
// ---------------------------------------------------------------------------

Function build_vec_add_reduce() {
    Function fn("vec_add_reduce");
    Operand pa = fn.add_param(Type::u64, "param_a");
    Operand pb = fn.add_param(Type::u64, "param_b");
    Operand pout = fn.add_param(Type::u64, "param_out");
    Operand psum = fn.add_param(Type::u64, "param_sum");
    Operand pn = fn.add_param(Type::u32, "param_n");
    fn.add_shared(Type::f32, "smem", 32, 4);

    Block* entry = fn.add_block("$L_entry");
    Block* vec = fn.add_block("$L_vec");
    Block* tail = fn.add_block("$L_tail");
    Block* tail_loop = fn.add_block("$L_tail_loop");
    Block* tail_body = fn.add_block("$L_tail_body");
    Block* reduce = fn.add_block("$L_reduce");

    auto ld_param = [&](Type t, Reg dst, const Operand& src) {
        return Inst::make(Opcode::ld, t).space(StateSpace::param).dst(dst).src(src);
    };
    auto mov_special = [&](Reg dst, SpecialReg s) {
        return Inst::make(Opcode::mov, Type::u32).dst(dst).src(Operand::special(s));
    };
    auto warp_reduce = [&](Block* b, Reg acc) {
        for (int delta : {16, 8, 4, 2, 1}) {
            Reg t = fn.new_f32();
            b->append(Inst::make(Opcode::shfl, Type::b32).sync().shfl(ShflMode::down)
                          .dst(t).src(acc).src(Operand::imm(delta)).src(Operand::imm(31)).src(Operand::imm(0xffffffff)));
            b->append(Inst::make(Opcode::add, Type::f32).dst(acc).src(acc).src(t));
        }
    };

    Reg a = fn.new_b64(), b = fn.new_b64(), out = fn.new_b64(), sum = fn.new_b64();
    Reg n = fn.new_b32();
    entry->append(ld_param(Type::u64, a, pa));
    entry->append(ld_param(Type::u64, b, pb));
    entry->append(ld_param(Type::u64, out, pout));
    entry->append(ld_param(Type::u64, sum, psum));
    entry->append(ld_param(Type::u32, n, pn));

    Reg tid = fn.new_b32(), ctaid = fn.new_b32(), ntid = fn.new_b32(), gid = fn.new_b32();
    entry->append(mov_special(tid, SpecialReg::tid_x));
    entry->append(mov_special(ctaid, SpecialReg::ctaid_x));
    entry->append(mov_special(ntid, SpecialReg::ntid_x));
    entry->append(Inst::make(Opcode::mad, Type::u32).lo().dst(gid).src(ctaid).src(ntid).src(tid));

    Reg idx4 = fn.new_b32(), end4 = fn.new_b32(), acc = fn.new_f32(), p_skip = fn.new_pred();
    entry->append(Inst::make(Opcode::shl, Type::b32).dst(idx4).src(gid).src(Operand::imm(2)));
    entry->append(Inst::make(Opcode::add, Type::u32).dst(end4).src(idx4).src(Operand::imm(4)));
    entry->append(Inst::make(Opcode::mov, Type::f32).dst(acc).src(Operand::imm_f32(0.0f)));
    entry->append(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::gt).dst(p_skip).src(end4).src(n));
    entry->append(Inst::make(Opcode::bra).guard(p_skip).src(Operand::label(tail->label)));

    // Vector fast path: four floats per thread.
    Reg off = fn.new_b64(), pa_ = fn.new_b64(), pb_ = fn.new_b64(), po_ = fn.new_b64();
    vec->append(Inst::make(Opcode::cvt, Type::u64).from(Type::u32).dst(off).src(idx4));
    vec->append(Inst::make(Opcode::shl, Type::b64).dst(off).src(off).src(Operand::imm(2)));
    vec->append(Inst::make(Opcode::add, Type::u64).dst(pa_).src(a).src(off));
    vec->append(Inst::make(Opcode::add, Type::u64).dst(pb_).src(b).src(off));
    vec->append(Inst::make(Opcode::add, Type::u64).dst(po_).src(out).src(off));
    auto va = fn.new_regs(RegClass::F32, 4), vb = fn.new_regs(RegClass::F32, 4), vs = fn.new_regs(RegClass::F32, 4);
    vec->append(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).vec(VecWidth::v4).dst(Operand::vec(va)).src(Operand::addr(pa_)));
    vec->append(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).vec(VecWidth::v4).dst(Operand::vec(vb)).src(Operand::addr(pb_)));
    for (int i = 0; i < 4; ++i)
        vec->append(Inst::make(Opcode::add, Type::f32).dst(vs[i]).src(va[i]).src(vb[i]));
    vec->append(Inst::make(Opcode::st, Type::f32).space(StateSpace::global).vec(VecWidth::v4).src(Operand::addr(po_)).src(Operand::vec(vs)));
    vec->append(Inst::make(Opcode::add, Type::f32).dst(acc).src(vs[0]).src(vs[1]));
    vec->append(Inst::make(Opcode::add, Type::f32).dst(acc).src(acc).src(vs[2]));
    vec->append(Inst::make(Opcode::add, Type::f32).dst(acc).src(acc).src(vs[3]));

    // Scalar tail: elements [n & ~3, n), handled by global thread 0.
    Reg p_not0 = fn.new_pred(), i = fn.new_b32(), p_done = fn.new_pred();
    tail->append(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::ne).dst(p_not0).src(gid).src(Operand::imm(0)));
    tail->append(Inst::make(Opcode::bra).guard(p_not0).src(Operand::label(reduce->label)));
    tail->append(Inst::make(Opcode::shr, Type::u32).dst(i).src(n).src(Operand::imm(2)));
    tail->append(Inst::make(Opcode::shl, Type::b32).dst(i).src(i).src(Operand::imm(2)));

    tail_loop->append(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::ge).dst(p_done).src(i).src(n));
    tail_loop->append(Inst::make(Opcode::bra).guard(p_done).src(Operand::label(reduce->label)));

    Reg toff = fn.new_b64(), ta = fn.new_b64(), tb = fn.new_b64(), to = fn.new_b64();
    Reg x = fn.new_f32(), y = fn.new_f32(), z = fn.new_f32();
    tail_body->append(Inst::make(Opcode::cvt, Type::u64).from(Type::u32).dst(toff).src(i));
    tail_body->append(Inst::make(Opcode::shl, Type::b64).dst(toff).src(toff).src(Operand::imm(2)));
    tail_body->append(Inst::make(Opcode::add, Type::u64).dst(ta).src(a).src(toff));
    tail_body->append(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).dst(x).src(Operand::addr(ta)));
    tail_body->append(Inst::make(Opcode::add, Type::u64).dst(tb).src(b).src(toff));
    tail_body->append(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).dst(y).src(Operand::addr(tb)));
    tail_body->append(Inst::make(Opcode::add, Type::f32).dst(z).src(x).src(y));
    tail_body->append(Inst::make(Opcode::add, Type::u64).dst(to).src(out).src(toff));
    tail_body->append(Inst::make(Opcode::st, Type::f32).space(StateSpace::global).src(Operand::addr(to)).src(z));
    tail_body->append(Inst::make(Opcode::add, Type::f32).dst(acc).src(acc).src(z));
    tail_body->append(Inst::make(Opcode::add, Type::u32).dst(i).src(i).src(Operand::imm(1)));
    tail_body->append(Inst::make(Opcode::bra).src(Operand::label(tail_loop->label)));

    // Block reduction: warp shuffle, lane 0 -> smem[warp], bar.sync, warp 0 folds.
    warp_reduce(reduce, acc);
    Reg lane = fn.new_b32(), warp = fn.new_b32(), p_lane0 = fn.new_pred();
    Reg sbase = fn.new_b32(), woff = fn.new_b32(), saddr = fn.new_b32();
    reduce->append(Inst::make(Opcode::and_, Type::b32).dst(lane).src(tid).src(Operand::imm(31)));
    reduce->append(Inst::make(Opcode::shr, Type::u32).dst(warp).src(tid).src(Operand::imm(5)));
    reduce->append(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::eq).dst(p_lane0).src(lane).src(Operand::imm(0)));
    reduce->append(Inst::make(Opcode::mov, Type::u32).dst(sbase).src(Operand::symbol("smem")));
    reduce->append(Inst::make(Opcode::shl, Type::b32).dst(woff).src(warp).src(Operand::imm(2)));
    reduce->append(Inst::make(Opcode::add, Type::u32).dst(saddr).src(sbase).src(woff));
    reduce->append(Inst::make(Opcode::st, Type::f32).space(StateSpace::shared).guard(p_lane0).src(Operand::addr(saddr)).src(acc));
    reduce->append(Inst::make(Opcode::bar).sync().src(Operand::imm(0)));

    Reg p_notw0 = fn.new_pred(), nwarps = fn.new_b32(), p_valid = fn.new_pred();
    Reg wsum = fn.new_f32(), loff = fn.new_b32(), laddr = fn.new_b32();
    reduce->append(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::ne).dst(p_notw0).src(warp).src(Operand::imm(0)));
    reduce->append(Inst::make(Opcode::ret).guard(p_notw0));
    reduce->append(Inst::make(Opcode::shr, Type::u32).dst(nwarps).src(ntid).src(Operand::imm(5)));
    reduce->append(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::lt).dst(p_valid).src(lane).src(nwarps));
    reduce->append(Inst::make(Opcode::mov, Type::f32).dst(wsum).src(Operand::imm_f32(0.0f)));
    reduce->append(Inst::make(Opcode::shl, Type::b32).dst(loff).src(lane).src(Operand::imm(2)));
    reduce->append(Inst::make(Opcode::add, Type::u32).dst(laddr).src(sbase).src(loff));
    reduce->append(Inst::make(Opcode::ld, Type::f32).space(StateSpace::shared).guard(p_valid).dst(wsum).src(Operand::addr(laddr)));
    warp_reduce(reduce, wsum);

    Reg p_notlane0 = fn.new_pred(), old = fn.new_f32();
    reduce->append(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::ne).dst(p_notlane0).src(lane).src(Operand::imm(0)));
    reduce->append(Inst::make(Opcode::ret).guard(p_notlane0));
    reduce->append(Inst::make(Opcode::atom, Type::f32).space(StateSpace::global).atom(AtomOp::add)
                       .dst(old).src(Operand::addr(sum)).src(wsum));
    reduce->append(Inst::make(Opcode::ret));
    return fn;
}

} // namespace

// ---------------------------------------------------------------------------
// Type tables
// ---------------------------------------------------------------------------

TEST_CASE("PTX IR - type tables map MIR types to suffixes and register classes") {
    using MirType = brass::Type;
    CHECK(type_for(MirType::i32()) == Type::u32);
    CHECK(type_for(MirType::i64()) == Type::u64);
    CHECK(type_for(MirType::ptr()) == Type::u64);
    CHECK(type_for(MirType::gcref()) == Type::u64);
    CHECK(type_for(MirType::f32()) == Type::f32);
    CHECK(type_for(MirType::f64()) == Type::f64);
    CHECK(type_for(MirType::f32x4()) == Type::f32);
    CHECK(type_for(MirType::f64x2()) == Type::f64);
    CHECK(type_for(MirType::i32x4()) == Type::u32);
    CHECK(type_for(MirType::void_type()) == Type::none);

    CHECK(signed_type_for(MirType::i32()) == Type::s32);
    CHECK(signed_type_for(MirType::i64()) == Type::s64);
    CHECK(signed_type_for(MirType::f32()) == Type::f32);
    CHECK(bit_type_for(MirType::i32()) == Type::b32);
    CHECK(bit_type_for(MirType::ptr()) == Type::b64);
    CHECK(bit_type_for(MirType::f64()) == Type::f64);
    CHECK(wide_type_for(Type::u32) == Type::u64);
    CHECK(wide_type_for(Type::s32) == Type::s64);
    CHECK(wide_type_for(Type::u64) == Type::none);

    CHECK(reg_class_for(Type::pred) == RegClass::Pred);
    CHECK(reg_class_for(Type::u8) == RegClass::B32);
    CHECK(reg_class_for(Type::u16) == RegClass::B32);
    CHECK(reg_class_for(Type::f16) == RegClass::B32);
    CHECK(reg_class_for(Type::b32) == RegClass::B32);
    CHECK(reg_class_for(Type::s32) == RegClass::B32);
    CHECK(reg_class_for(Type::u64) == RegClass::B64);
    CHECK(reg_class_for(Type::b64) == RegClass::B64);
    CHECK(reg_class_for(Type::f32) == RegClass::F32);
    CHECK(reg_class_for(Type::f64) == RegClass::F64);
    CHECK(reg_class_for(MirType::ptr()) == RegClass::B64);
    CHECK(reg_class_for(MirType::f32x4()) == RegClass::F32);
    CHECK(reg_class_for(SpecialReg::tid_x) == RegClass::B32);
    CHECK(reg_class_for(SpecialReg::clock64) == RegClass::B64);

    CHECK(bit_width(Type::u16) == 16u);
    CHECK(bit_width(Type::f64) == 64u);
    CHECK(std::string(to_string(Type::s64)) == "s64");
    CHECK(std::string(to_string(Opcode::and_)) == "and");
}

TEST_CASE("PTX IR - register allocation and function helpers") {
    Function fn("k");
    Reg p0 = fn.new_pred(), p1 = fn.new_pred();
    Reg r0 = fn.new_b32();
    auto fs = fn.new_regs(RegClass::F32, 4);
    CHECK(p0.index == 0 && p1.index == 1 && p0.cls == RegClass::Pred);
    CHECK(r0.index == 0 && r0.cls == RegClass::B32);
    CHECK(fs.size() == 4 && fs[0].index == 0 && fs[3].index == 3 && fs[3].cls == RegClass::F32);
    CHECK(fn.reg_count(RegClass::Pred) == 2u);
    CHECK(fn.reg_count(RegClass::B32) == 1u);
    CHECK(fn.reg_count(RegClass::B64) == 0u);
    CHECK(fn.reg_count(RegClass::F32) == 4u);
    CHECK(to_string(fs[2]) == "%f2");
    CHECK(to_string(Reg(RegClass::B64, 7)) == "%rd7");
    CHECK(to_string(Reg(RegClass::F64, 1)) == "%fd1");

    Operand pa = fn.add_param(Type::u64, "param_a");
    CHECK(pa.is_param() && pa.name == "param_a");
    CHECK(fn.find_param("param_a") != nullptr && fn.find_param("param_a")->type == Type::u64);
    CHECK(fn.find_param("nope") == nullptr);
    fn.add_shared(Type::f32, "smem", 32, 16);
    CHECK(fn.find_shared("smem") != nullptr && fn.find_shared("smem")->align == 16u);

    Block* b0 = fn.add_block(fn.unique_label("loop"));
    Block* b1 = fn.add_block(fn.unique_label("loop"));
    CHECK(b0->label == "$L_loop_0");
    CHECK(b1->label == "$L_loop_1");
    CHECK(fn.entry_block() == b0);
    CHECK(fn.find_block("$L_loop_1") == b1);
    CHECK(fn.find_block("$L_none") == nullptr);
}

// ---------------------------------------------------------------------------
// (a) Printer: exact text
// ---------------------------------------------------------------------------

TEST_CASE("PTX IR - printer formats operands, guards and modifier suffixes exactly") {
    Fixture fx;
    Reg p = fx.p, r = fx.r, r1 = fx.r1, rd = fx.rd, rd1 = fx.rd1, f = fx.f, f1 = fx.f1, fd = fx.fd;
    Reg f2 = fx.fn.new_f32(), f3 = fx.fn.new_f32(), fd1 = fx.fn.new_f64();
    auto I = [](Opcode o, Type t = Type::none) { return Inst::make(o, t); };

    // Operand kinds
    CHECK_EQ(to_string(I(Opcode::ld, Type::u64).space(StateSpace::param).dst(rd).src(Operand::param("param_a"))),
             "ld.param.u64 %rd0, [param_a];");
    CHECK_EQ(to_string(I(Opcode::ld, Type::u64).space(StateSpace::param).dst(rd).src(Operand::addr("param_a", 8))),
             "ld.param.u64 %rd0, [param_a + 8];");
    CHECK_EQ(to_string(I(Opcode::ld, Type::f32).space(StateSpace::global).vec(VecWidth::v4)
                           .dst(Operand::vec({f, f1, f2, f3})).src(Operand::addr(rd, 16))),
             "ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd0 + 16];");
    CHECK_EQ(to_string(I(Opcode::ld, Type::f64).space(StateSpace::global).vec(VecWidth::v2)
                           .dst(Operand::vec({fd, fd1})).src(Operand::addr(rd))),
             "ld.global.v2.f64 {%fd0, %fd1}, [%rd0];");
    CHECK_EQ(to_string(I(Opcode::ld, Type::u16).space(StateSpace::global).dst(r).src(Operand::addr(rd, -2))),
             "ld.global.u16 %r0, [%rd0 + -2];");
    CHECK_EQ(to_string(I(Opcode::ld, Type::f32).space(StateSpace::shared).dst(f).src(Operand::addr("smem", 4))),
             "ld.shared.f32 %f0, [smem + 4];");
    CHECK_EQ(to_string(I(Opcode::st, Type::f32).space(StateSpace::global).vec(VecWidth::v4)
                           .src(Operand::addr(rd)).src(Operand::vec({f, f1, f2, f3}))),
             "st.global.v4.f32 [%rd0], {%f0, %f1, %f2, %f3};");
    CHECK_EQ(to_string(I(Opcode::st, Type::f32).space(StateSpace::shared).guard(p).src(Operand::addr(r1)).src(f)),
             "@%p0 st.shared.f32 [%r1], %f0;");
    CHECK_EQ(to_string(I(Opcode::ret).guard(p, true)), "@!%p0 ret;");
    CHECK_EQ(to_string(I(Opcode::mov, Type::u32).dst(r).src(Operand::special(SpecialReg::tid_x))), "mov.u32 %r0, %tid.x;");
    CHECK_EQ(to_string(I(Opcode::mov, Type::u32).dst(r).src(Operand::special(SpecialReg::nctaid_y))), "mov.u32 %r0, %nctaid.y;");
    CHECK_EQ(to_string(I(Opcode::mov, Type::u32).dst(r).src(Operand::special(SpecialReg::laneid))), "mov.u32 %r0, %laneid;");
    CHECK_EQ(to_string(I(Opcode::mov, Type::u32).dst(r).src(Operand::symbol("smem"))), "mov.u32 %r0, smem;");
    CHECK_EQ(to_string(I(Opcode::mov, Type::f32).dst(f).src(Operand::imm_f32(1.0f))), "mov.f32 %f0, 0f3F800000;");
    CHECK_EQ(to_string(I(Opcode::mov, Type::f32).dst(f).src(Operand::imm_f32(-1.44269504f))), "mov.f32 %f0, 0fBFB8AA3B;");
    CHECK_EQ(to_string(I(Opcode::mov, Type::f64).dst(fd).src(Operand::imm_f64(1.0))), "mov.f64 %fd0, 0d3FF0000000000000;");
    CHECK_EQ(to_string(I(Opcode::mov, Type::b64).dst(rd).src(Operand::imm(-42))), "mov.b64 %rd0, -42;");
    CHECK_EQ(to_string(I(Opcode::bra).src(Operand::label("$L_exit"))), "bra $L_exit;");
    CHECK_EQ(to_string(I(Opcode::bra).uni().guard(p).src(Operand::label("$L_exit"))), "@%p0 bra.uni $L_exit;");
    CHECK_EQ(to_string(I(Opcode::call).dst(r).src(Operand::symbol("callee")).src(r1).src(rd)),
             "call (%r0), callee, (%r1, %rd0);");
    CHECK_EQ(to_string(I(Opcode::call).src(Operand::symbol("callee")).src(r1)), "call callee, (%r1);");
    CHECK_EQ(to_string(I(Opcode::trap)), "trap;");

    // Modifier combinations, in PTX order
    CHECK_EQ(to_string(I(Opcode::cvt, Type::f32).from(Type::u32).rnd(Rounding::rn).dst(f).src(r)), "cvt.rn.f32.u32 %f0, %r0;");
    CHECK_EQ(to_string(I(Opcode::cvt, Type::s32).from(Type::f32).rnd(Rounding::rzi).dst(r).src(f)), "cvt.rzi.s32.f32 %r0, %f0;");
    CHECK_EQ(to_string(I(Opcode::cvt, Type::f32).from(Type::f16).dst(f).src(r)), "cvt.f32.f16 %f0, %r0;");
    CHECK_EQ(to_string(I(Opcode::cvt, Type::u64).from(Type::u32).dst(rd).src(r)), "cvt.u64.u32 %rd0, %r0;");
    CHECK_EQ(to_string(I(Opcode::cvt, Type::f32).from(Type::f64).rnd(Rounding::rn).ftz().dst(f).src(fd)), "cvt.rn.ftz.f32.f64 %f0, %fd0;");
    CHECK_EQ(to_string(I(Opcode::add, Type::f32).rnd(Rounding::rn).ftz().sat().dst(f).src(f1).src(f2)), "add.rn.ftz.sat.f32 %f0, %f1, %f2;");
    CHECK_EQ(to_string(I(Opcode::add, Type::s64).dst(rd).src(rd).src(rd1)), "add.s64 %rd0, %rd0, %rd1;");
    CHECK_EQ(to_string(I(Opcode::mul, Type::u64).lo().dst(rd).src(rd).src(rd1)), "mul.lo.u64 %rd0, %rd0, %rd1;");
    CHECK_EQ(to_string(I(Opcode::mul, Type::u32).wide().dst(rd).src(r).src(r1)), "mul.wide.u32 %rd0, %r0, %r1;");
    CHECK_EQ(to_string(I(Opcode::mul, Type::s32).hi().dst(r).src(r).src(r1)), "mul.hi.s32 %r0, %r0, %r1;");
    CHECK_EQ(to_string(I(Opcode::mad, Type::u32).lo().dst(r).src(r).src(r1).src(r)), "mad.lo.u32 %r0, %r0, %r1, %r0;");
    CHECK_EQ(to_string(I(Opcode::fma, Type::f32).rnd(Rounding::rn).dst(f).src(f1).src(f2).src(f3)), "fma.rn.f32 %f0, %f1, %f2, %f3;");
    CHECK_EQ(to_string(I(Opcode::div, Type::f32).approx().dst(f).src(f1).src(f2)), "div.approx.f32 %f0, %f1, %f2;");
    CHECK_EQ(to_string(I(Opcode::div, Type::f64).rnd(Rounding::rn).dst(fd).src(fd).src(fd1)), "div.rn.f64 %fd0, %fd0, %fd1;");
    CHECK_EQ(to_string(I(Opcode::div, Type::u32).dst(r).src(r).src(r1)), "div.u32 %r0, %r0, %r1;");
    CHECK_EQ(to_string(I(Opcode::rem, Type::s32).dst(r).src(r).src(r1)), "rem.s32 %r0, %r0, %r1;");
    CHECK_EQ(to_string(I(Opcode::neg, Type::f32).dst(f).src(f1)), "neg.f32 %f0, %f1;");
    CHECK_EQ(to_string(I(Opcode::abs, Type::s32).dst(r).src(r1)), "abs.s32 %r0, %r1;");
    CHECK_EQ(to_string(I(Opcode::min, Type::f32).dst(f).src(f1).src(f2)), "min.f32 %f0, %f1, %f2;");
    CHECK_EQ(to_string(I(Opcode::max, Type::f32).ftz().dst(f).src(f1).src(f2)), "max.ftz.f32 %f0, %f1, %f2;");
    CHECK_EQ(to_string(I(Opcode::and_, Type::b32).dst(r).src(r).src(Operand::imm(31))), "and.b32 %r0, %r0, 31;");
    CHECK_EQ(to_string(I(Opcode::or_, Type::b64).dst(rd).src(rd).src(rd1)), "or.b64 %rd0, %rd0, %rd1;");
    CHECK_EQ(to_string(I(Opcode::xor_, Type::b32).dst(r).src(r).src(r1)), "xor.b32 %r0, %r0, %r1;");
    CHECK_EQ(to_string(I(Opcode::not_, Type::b32).dst(r).src(r1)), "not.b32 %r0, %r1;");
    CHECK_EQ(to_string(I(Opcode::shl, Type::b64).dst(rd).src(rd).src(Operand::imm(2))), "shl.b64 %rd0, %rd0, 2;");
    CHECK_EQ(to_string(I(Opcode::shr, Type::u32).dst(r).src(r).src(Operand::imm(5))), "shr.u32 %r0, %r0, 5;");
    CHECK_EQ(to_string(I(Opcode::shr, Type::s32).dst(r).src(r).src(r1)), "shr.s32 %r0, %r0, %r1;");
    CHECK_EQ(to_string(I(Opcode::setp, Type::u32).cmp(CmpOp::ge).dst(p).src(r).src(r1)), "setp.ge.u32 %p0, %r0, %r1;");
    CHECK_EQ(to_string(I(Opcode::setp, Type::f32).cmp(CmpOp::neu).dst(p).src(f).src(Operand::imm_f32(0.0f))), "setp.neu.f32 %p0, %f0, 0f00000000;");
    CHECK_EQ(to_string(I(Opcode::setp, Type::s64).cmp(CmpOp::lt).dst(p).src(rd).src(Operand::imm(0))), "setp.lt.s64 %p0, %rd0, 0;");
    CHECK_EQ(to_string(I(Opcode::selp, Type::u32).dst(r).src(Operand::imm(1)).src(Operand::imm(0)).src(p)), "selp.u32 %r0, 1, 0, %p0;");
    CHECK_EQ(to_string(I(Opcode::selp, Type::f32).dst(f).src(f1).src(f2).src(p)), "selp.f32 %f0, %f1, %f2, %p0;");
    CHECK_EQ(to_string(I(Opcode::shfl, Type::b32).sync().shfl(ShflMode::down).dst(f).src(f1)
                           .src(Operand::imm(16)).src(Operand::imm(31)).src(Operand::imm(0xffffffff))),
             "shfl.sync.down.b32 %f0, %f1, 16, 31, 4294967295;");
    CHECK_EQ(to_string(I(Opcode::shfl, Type::b32).sync().shfl(ShflMode::bfly).dst(r).src(r1)
                           .src(r1).src(Operand::imm(31)).src(Operand::imm(-1))),
             "shfl.sync.bfly.b32 %r0, %r1, %r1, 31, -1;");
    CHECK_EQ(to_string(I(Opcode::bar).sync().src(Operand::imm(0))), "bar.sync 0;");
    CHECK_EQ(to_string(I(Opcode::bar).sync().src(Operand::imm(1)).src(Operand::imm(64))), "bar.sync 1, 64;");
    CHECK_EQ(to_string(I(Opcode::rsqrt, Type::f32).approx().dst(f).src(f1)), "rsqrt.approx.f32 %f0, %f1;");
    CHECK_EQ(to_string(I(Opcode::sqrt, Type::f32).rnd(Rounding::rn).dst(f).src(f1)), "sqrt.rn.f32 %f0, %f1;");
    CHECK_EQ(to_string(I(Opcode::sqrt, Type::f32).approx().ftz().dst(f).src(f1)), "sqrt.approx.ftz.f32 %f0, %f1;");
    CHECK_EQ(to_string(I(Opcode::ex2, Type::f32).approx().dst(f).src(f1)), "ex2.approx.f32 %f0, %f1;");
    CHECK_EQ(to_string(I(Opcode::lg2, Type::f32).approx().dst(f).src(f1)), "lg2.approx.f32 %f0, %f1;");
    CHECK_EQ(to_string(I(Opcode::rcp, Type::f32).approx().dst(f).src(f1)), "rcp.approx.f32 %f0, %f1;");
    CHECK_EQ(to_string(I(Opcode::sin, Type::f32).approx().dst(f).src(f1)), "sin.approx.f32 %f0, %f1;");
    CHECK_EQ(to_string(I(Opcode::cos, Type::f32).approx().dst(f).src(f1)), "cos.approx.f32 %f0, %f1;");
    CHECK_EQ(to_string(I(Opcode::atom, Type::f32).space(StateSpace::global).atom(AtomOp::add).dst(f).src(Operand::addr(rd)).src(f1)),
             "atom.global.add.f32 %f0, [%rd0], %f1;");
    CHECK_EQ(to_string(I(Opcode::atom, Type::b32).space(StateSpace::shared).atom(AtomOp::cas).dst(r).src(Operand::addr(r1)).src(r1).src(Operand::imm(7))),
             "atom.shared.cas.b32 %r0, [%r1], %r1, 7;");

    CHECK_EQ(format_f32_hex(0.0f), "0f00000000");
    CHECK_EQ(format_f32_hex(1.44269504f), "0f3FB8AA3B");
    CHECK_EQ(format_f64_hex(-2.0), "0dC000000000000000");
}

TEST_CASE("PTX IR - printer emits header, signature, declarations and blocks") {
    Function fn("k");
    fn.add_param(Type::u64, "param_a");
    fn.add_param(Type::u32, "param_n");
    fn.add_shared(Type::f32, "smem", 32, 4);
    Reg rd = fn.new_b64(), r = fn.new_b32(), p = fn.new_pred();
    Block* e = fn.add_block("$L_entry");
    Block* x = fn.add_block("$L_exit");
    e->append(Inst::make(Opcode::ld, Type::u64).space(StateSpace::param).dst(rd).src(Operand::param("param_a")));
    e->append(Inst::make(Opcode::ld, Type::u32).space(StateSpace::param).dst(r).src(Operand::param("param_n")));
    e->append(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::eq).dst(p).src(r).src(Operand::imm(0)));
    e->append(Inst::make(Opcode::bra).guard(p).src(Operand::label("$L_exit")));
    x->append(Inst::make(Opcode::ret));

    CHECK(verify(fn).empty());

    const std::string expected =
        "// Generated by Brass PTX Target Emitter\n"
        ".version 7.0\n"
        ".target sm_70\n"
        ".address_size 64\n"
        "\n"
        ".visible .entry k(\n"
        "    .param .u64 param_a,\n"
        "    .param .u32 param_n\n"
        ")\n"
        "{\n"
        "    .reg .pred %p<1>;\n"
        "    .reg .b32 %r<1>;\n"
        "    .reg .b64 %rd<1>;\n"
        "    .shared .align 4 .f32 smem[32];\n"
        "\n"
        "$L_entry:\n"
        "    ld.param.u64 %rd0, [param_a];\n"
        "    ld.param.u32 %r0, [param_n];\n"
        "    setp.eq.u32 %p0, %r0, 0;\n"
        "    @%p0 bra $L_exit;\n"
        "$L_exit:\n"
        "    ret;\n"
        "}\n";
    CHECK_EQ(print(fn), expected);

    brass::target::PtxOptions ada;
    ada.sm_arch = "sm_89";
    std::string hdr = print_header(ada);
    CHECK(hdr.find(".version 7.8\n") != std::string::npos);
    CHECK(hdr.find(".target sm_89\n") != std::string::npos);

    // A function with no params, no shared and no f64 registers omits those lines.
    Function empty("noop");
    empty.add_block("$L_entry")->append(Inst::make(Opcode::ret));
    CHECK_EQ(print_body(empty), ".visible .entry noop()\n{\n\n$L_entry:\n    ret;\n}\n");

    std::string mod = print_module({&fn, &empty});
    CHECK(mod.find(".visible .entry k(") != std::string::npos);
    CHECK(mod.find(".visible .entry noop()") != std::string::npos);
    CHECK(mod.find(".version") == mod.rfind(".version"));
}

// ---------------------------------------------------------------------------
// (b) Verifier: malformed cases
// ---------------------------------------------------------------------------

TEST_CASE("PTX IR - verifier rejects register class and range errors") {
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::f32).dst(x.f).src(x.r).src(x.f1), "requires a register of class f32"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::u64).dst(x.rd).src(x.r).src(x.rd1), "requires a register of class b64"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::s32).dst(x.f).src(x.r).src(x.r1), "destination is %f0"); }
    { Fixture x; x.reject(Inst::make(Opcode::mov, Type::f64).dst(x.fd).src(x.f), "requires a register of class f64"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::f32).dst(x.f).src(x.f1).src(Reg(RegClass::F32, 7)), "%f7 is out of range"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).dst(Operand::vec({x.f, Reg(RegClass::F32, 9)}))
                              .vec(VecWidth::v2).src(Operand::addr(x.rd)), "%f9 is out of range"); }
    { Fixture x; x.reject(Inst::make(Opcode::ret).guard(Reg(RegClass::Pred, 3)), "%p3 is out of range"); }
    { Fixture x; x.reject(Inst::make(Opcode::ret).guard(x.r), "guard %r0 is not a pred register"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::f32).dst(x.f).src(x.f1).src(Reg(RegClass::F32, Reg::kInvalid)), "invalid f32 register"); }
    // Bit types accept either register file of the same width (shfl of f32, mov.b64 bitcast).
    { Fixture x; x.bb->append(Inst::make(Opcode::mov, Type::b64).dst(x.rd).src(x.fd)); x.bb->append(Inst::make(Opcode::ret));
      auto d = verify(x.fn); dump(d); CHECK(d.empty()); }
}

TEST_CASE("PTX IR - verifier rejects control-flow and structural errors") {
    { Fixture x; x.reject(Inst::make(Opcode::ret).src(x.r), "ret inside .entry takes no operand"); }
    { Fixture x; x.reject(Inst::make(Opcode::bra).src(Operand::label("$L_nowhere")), "label '$L_nowhere' does not exist"); }
    { Fixture x; x.reject(Inst::make(Opcode::bra).src(x.r), "bra target must be a label"); }
    { Fixture x; x.reject(Inst::make(Opcode::bra, Type::u32).src(Operand::label("$L_entry")), "does not take a type suffix"); }
    { Fixture x; x.reject(Inst::make(Opcode::add).dst(x.r).src(x.r).src(x.r1), "missing type suffix"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::u64).space(StateSpace::param).dst(x.rd).src(Operand::param("param_zzz")),
                          "'param_zzz' which is not a declared param"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::u64).space(StateSpace::param).dst(x.rd).src(Operand::addr("param_zzz")),
                          "'param_zzz' which is not a declared param"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::u32).space(StateSpace::param).dst(x.r).src(Operand::param("param_a")),
                          "does not match param 'param_a'"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::u64).space(StateSpace::global).dst(x.rd).src(Operand::param("param_a")),
                          "only valid with ld.param"); }
    { Fixture x; x.reject(Inst::make(Opcode::mov, Type::u32).dst(x.r).src(Operand::symbol("nosuch")), "not a declared .shared array"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).space(StateSpace::shared).dst(x.f).src(Operand::addr("nosuch")), "not a declared .shared array"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).dst(x.f).src(Operand::addr("smem")), "only .param and .shared addressing may use symbols"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).dst(x.f).src(Operand::addr(x.rd)), "ld requires a state space"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).dst(x.f).src(Operand::addr(x.r)), "must be a b64 register"); }
    { Fixture x; x.reject(Inst::make(Opcode::st, Type::f32).space(StateSpace::shared).src(Operand::addr(x.f)).src(x.f1), "must be a b32 or b64 register"); }
    { Fixture x; x.reject(Inst::make(Opcode::mov, Type::u32).dst(x.r).src(Operand::special(SpecialReg::clock64)), "must be moved into a b64 register"); }

    // Function-level diagnostics carry no instruction location.
    Function dup("dup");
    dup.add_block("$L_a")->append(Inst::make(Opcode::ret));
    dup.add_block("$L_a")->append(Inst::make(Opcode::ret));
    auto d = verify(dup);
    CHECK(mentions(d, "duplicate block label '$L_a'"));
    CHECK(!d.empty() && d[0].block.empty() && d[0].inst_index == Diagnostic::kNoInst);
    CHECK(to_string(d[0]).rfind("dup: duplicate", 0) == 0);

    Function none("none");
    CHECK(mentions(verify(none), "no blocks"));
    Function badp("badp");
    badp.add_param(Type::pred, "p");
    badp.add_shared(Type::f32, "s", 0, 3);
    badp.add_block("$L_entry")->append(Inst::make(Opcode::ret));
    d = verify(badp);
    CHECK(mentions(d, "invalid type .pred"));
    CHECK(mentions(d, "zero elements"));
    CHECK(mentions(d, "power of two"));
}

TEST_CASE("PTX IR - verifier rejects vector, shift, immediate and modifier misuse") {
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).vec(VecWidth::v4)
                              .dst(Operand::vec({x.f, x.f1})).src(Operand::addr(x.rd)), "has 2 elements but .v4 requires 4"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).vec(VecWidth::v2)
                              .dst(x.f).src(Operand::addr(x.rd)), "must be a {..} vector tuple for .v2"); }
    { Fixture x; x.reject(Inst::make(Opcode::st, Type::f32).space(StateSpace::global)
                              .src(Operand::addr(x.rd)).src(Operand::vec({x.f, x.f1})), "is a vector tuple but the instruction has no .v2/.v4"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).vec(VecWidth::v2)
                              .dst(Operand::vec({x.f, x.r})).src(Operand::addr(x.rd)), "element 1 is %r0"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::f32).vec(VecWidth::v2).dst(x.f).src(x.f1).src(x.f1), "only ld/st take a .v2/.v4"); }
    { Fixture x; x.reject(Inst::make(Opcode::shl, Type::b64).dst(x.rd).src(x.rd1).src(x.rd1), "shift amount is %rd1 but must be a b32 register"); }
    { Fixture x; x.reject(Inst::make(Opcode::shr, Type::u32).dst(x.r).src(x.r1).src(x.f), "shift amount is %f0"); }
    { Fixture x; x.reject(Inst::make(Opcode::shl, Type::u32).dst(x.r).src(x.r1).src(Operand::imm(1)), "shl requires a bit type"); }
    { Fixture x; x.reject(Inst::make(Opcode::mov, Type::f32).dst(x.f).src(Operand::imm_f64(1.0)), "f64 immediate on an .f32 instruction"); }
    { Fixture x; x.reject(Inst::make(Opcode::mov, Type::f64).dst(x.fd).src(Operand::imm_f32(1.0f)), "f32 immediate on an .f64 instruction"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::f32).dst(x.f).src(x.f1).src(Operand::imm(1)), "requires a float immediate"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::u32).dst(x.r).src(x.r1).src(Operand::imm_f32(1.0f)), "requires an integer immediate"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::u32).dst(Operand::imm(1)).src(x.r1).src(x.r), "is not a register"); }
    // Immediate positions follow the allows_immediate table: setp
    // takes one only as its second source, cvt and the shfl value never do,
    // and an integer immediate must fit the instruction width.
    { Fixture x; x.reject(Inst::make(Opcode::setp, Type::s32).cmp(CmpOp::lt).dst(x.p).src(Operand::imm(5)).src(x.r), "source 0 of setp may not be an immediate"); }
    { Fixture x; x.reject(Inst::make(Opcode::cvt, Type::f32).from(Type::s32).rnd(Rounding::rn).dst(x.f).src(Operand::imm(5)), "source 0 of cvt may not be an immediate"); }
    { Fixture x; x.reject(Inst::make(Opcode::shfl, Type::b32).sync().shfl(ShflMode::down).dst(x.f).src(Operand::imm_f32(1.0f))
                              .src(Operand::imm(1)).src(Operand::imm(31)).src(Operand::imm(-1)), "source 0 of shfl may not be an immediate"); }
    { Fixture x; x.reject(Inst::make(Opcode::selp, Type::b32).dst(x.r).src(x.r1).src(x.r1).src(Operand::imm(1)), "source 2 of selp may not be an immediate"); }
    { Fixture x; x.reject(Inst::make(Opcode::mov, Type::b32).dst(x.r).src(Operand::imm(5000000000LL)), "immediate 5000000000 does not fit .b32"); }
    { Fixture x; x.reject(Inst::make(Opcode::mul, Type::u32).wide().dst(x.rd).src(x.r1).src(Operand::imm(-3000000000LL)), "does not fit .u32"); }
    { Fixture x; x.reject(Inst::make(Opcode::shl, Type::b64).dst(x.rd).src(x.rd1).src(Operand::imm(4294967296LL)), "shift amount immediate 4294967296 does not fit .u32"); }
    { Fixture x; x.bb->append(Inst::make(Opcode::sub, Type::s32).dst(x.r).src(Operand::imm(5)).src(x.r1)); // either source of an ALU op
      x.bb->append(Inst::make(Opcode::setp, Type::s32).cmp(CmpOp::lt).dst(x.p).src(x.r).src(Operand::imm(-5)));
      x.bb->append(Inst::make(Opcode::mov, Type::u32).dst(x.r).src(Operand::imm(4294967295LL)));            // unsigned reading of a 32-bit slot
      x.bb->append(Inst::make(Opcode::st, Type::f32).space(StateSpace::global).src(Operand::addr(x.rd)).src(Operand::imm_f32(2.0f)));
      x.bb->append(Inst::make(Opcode::ret));
      auto d = verify(x.fn); dump(d); CHECK(d.empty()); }
    { CHECK(allows_immediate(Opcode::add, 0)); CHECK(allows_immediate(Opcode::fma, 2)); CHECK(allows_immediate(Opcode::bar, 1));
      CHECK(!allows_immediate(Opcode::setp, 0)); CHECK(!allows_immediate(Opcode::ld, 0)); CHECK(!allows_immediate(Opcode::call, 0));
      CHECK(imm_fits(Type::u8, 255)); CHECK(!imm_fits(Type::u8, 256)); CHECK(imm_fits(Type::s32, -2147483648LL)); CHECK(imm_fits(Type::u64, -1)); }
    { Fixture x; x.reject(Inst::make(Opcode::mul, Type::u32).dst(x.r).src(x.r1).src(x.r), "integer mul requires .lo, .hi or .wide"); }
    { Fixture x; x.reject(Inst::make(Opcode::mul, Type::u32).wide().dst(x.r).src(x.r1).src(x.r), "destination is %r0 (b32) but .u64"); }
    { Fixture x; x.reject(Inst::make(Opcode::mul, Type::f32).lo().dst(x.f).src(x.f1).src(x.f), "only valid for integer mul"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::u32).lo().dst(x.r).src(x.r1).src(x.r), "only mul/mad take .lo/.hi/.wide"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::u32).cmp(CmpOp::eq).dst(x.r).src(x.r1).src(x.r), "only setp takes a comparison"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::u32).space(StateSpace::global).dst(x.r).src(x.r1).src(x.r), "only ld/st/atom take a state space"); }
    { Fixture x; x.reject(Inst::make(Opcode::add, Type::u32).from(Type::u64).dst(x.r).src(x.r1).src(x.r), "only cvt takes a second"); }
    { Fixture x; x.reject(Inst::make(Opcode::cvt, Type::f32).dst(x.f).src(x.r), "cvt requires a source type"); }
    { Fixture x; x.reject(Inst::make(Opcode::cvt, Type::f32).from(Type::u32).dst(x.f).src(x.r), "integer -> float cvt requires a rounding"); }
    { Fixture x; x.reject(Inst::make(Opcode::cvt, Type::s32).from(Type::f32).dst(x.r).src(x.f), "float -> integer cvt requires an integer rounding"); }
    { Fixture x; x.reject(Inst::make(Opcode::cvt, Type::u64).from(Type::u32).dst(x.rd).src(x.f), "source 0 is %f0"); }
    { Fixture x; x.reject(Inst::make(Opcode::div, Type::f32).dst(x.f).src(x.f1).src(x.f), "float div requires .approx or a rounding"); }
    { Fixture x; x.reject(Inst::make(Opcode::fma, Type::f32).dst(x.f).src(x.f1).src(x.f).src(x.f), "fma requires a rounding modifier"); }
    { Fixture x; x.reject(Inst::make(Opcode::and_, Type::u32).dst(x.r).src(x.r1).src(x.r), "and requires a bit type"); }
    { Fixture x; x.reject(Inst::make(Opcode::rsqrt, Type::f32).dst(x.f).src(x.f1), "rsqrt only exists in the .approx form"); }
    { Fixture x; x.reject(Inst::make(Opcode::sqrt, Type::f32).dst(x.f).src(x.f1), "sqrt requires .approx or a rounding"); }
    { Fixture x; x.reject(Inst::make(Opcode::ex2, Type::u32).approx().dst(x.r).src(x.r1), "ex2 requires a float type"); }
    { Fixture x; x.reject(Inst::make(Opcode::shfl, Type::b32).shfl(ShflMode::down).dst(x.f).src(x.f1)
                              .src(Operand::imm(1)).src(Operand::imm(31)).src(Operand::imm(-1)), "shfl requires .sync"); }
    { Fixture x; x.reject(Inst::make(Opcode::shfl, Type::b32).sync().dst(x.f).src(x.f1)
                              .src(Operand::imm(1)).src(Operand::imm(31)).src(Operand::imm(-1)), "shfl requires a mode"); }
    { Fixture x; x.reject(Inst::make(Opcode::shfl, Type::f32).sync().shfl(ShflMode::down).dst(x.f).src(x.f1)
                              .src(Operand::imm(1)).src(Operand::imm(31)).src(Operand::imm(-1)), "shfl requires .b32"); }
    { Fixture x; x.reject(Inst::make(Opcode::shfl, Type::b32).sync().shfl(ShflMode::down).dst(x.f).src(x.f1)
                              .src(x.rd).src(Operand::imm(31)).src(Operand::imm(-1)), "lane delta (source 1) is %rd0"); }
    { Fixture x; x.reject(Inst::make(Opcode::bar).src(Operand::imm(0)), "bar requires .sync"); }
    { Fixture x; x.reject(Inst::make(Opcode::atom, Type::f32).atom(AtomOp::add).dst(x.f).src(Operand::addr(x.rd)).src(x.f1), "atom requires .global or .shared"); }
    { Fixture x; x.reject(Inst::make(Opcode::atom, Type::f32).space(StateSpace::global).dst(x.f).src(Operand::addr(x.rd)).src(x.f1), "atom requires an operation"); }
}

TEST_CASE("PTX IR - verifier enforces opcode arity and predicate operands") {
    { Fixture x; x.reject(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::eq).dst(x.p).src(x.r), "setp takes 2 source(s), got 1"); }
    { Fixture x; x.reject(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::eq).dst(x.r).src(x.r).src(x.r1), "setp destination is %r0 but must be a pred register"); }
    { Fixture x; x.reject(Inst::make(Opcode::setp, Type::u32).dst(x.p).src(x.r).src(x.r1), "setp requires a comparison operator"); }
    { Fixture x; x.reject(Inst::make(Opcode::selp, Type::b32).dst(x.r).src(x.r1).src(x.r1).src(x.r), "selp condition is %r0 but must be a pred register"); }
    { Fixture x; x.reject(Inst::make(Opcode::selp, Type::b32).dst(x.r).src(x.r1).src(x.p), "selp takes 3 source(s), got 2"); }
    { Fixture x; x.reject(Inst::make(Opcode::fma, Type::f32).rnd(Rounding::rn).dst(x.f).src(x.f1).src(x.f1), "fma takes 3 source(s), got 2"); }
    { Fixture x; x.reject(Inst::make(Opcode::mad, Type::u32).lo().dst(x.r).src(x.r1).src(x.r1), "mad takes 3 source(s), got 2"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).src(Operand::addr(x.rd)), "ld takes 1 destination(s), got 0"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).dst(x.f).src(Operand::addr(x.rd)).src(x.f1), "ld takes 1 source(s), got 2"); }
    { Fixture x; x.reject(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).dst(x.f).src(x.rd), "address must be an address"); }
    { Fixture x; x.reject(Inst::make(Opcode::st, Type::f32).space(StateSpace::global).src(Operand::addr(x.rd)), "st takes 2 source(s), got 1"); }
    { Fixture x; x.reject(Inst::make(Opcode::st, Type::f32).space(StateSpace::global).dst(x.f).src(Operand::addr(x.rd)).src(x.f1), "st takes 0 destination(s), got 1"); }
    { Fixture x; x.reject(Inst::make(Opcode::mov, Type::u32).dst(x.r), "mov takes 1 source(s), got 0"); }
    { Fixture x; x.reject(Inst::make(Opcode::neg, Type::f32).dst(x.f).src(x.f1).src(x.f1), "neg takes 1 source(s), got 2"); }
    { Fixture x; x.reject(Inst::make(Opcode::bra).src(Operand::label("$L_entry")).src(Operand::label("$L_entry")), "bra takes 1 source(s), got 2"); }
    { Fixture x; x.reject(Inst::make(Opcode::trap).src(x.r), "trap takes 0 source(s), got 1"); }
    { Fixture x; x.reject(Inst::make(Opcode::bar).sync(), "bar takes 1..2 source(s), got 0"); }
    { Fixture x; x.reject(Inst::make(Opcode::shfl, Type::b32).sync().shfl(ShflMode::down).dst(x.f).src(x.f1).src(Operand::imm(1)), "shfl takes 4 source(s), got 2"); }
    { Fixture x; x.reject(Inst::make(Opcode::call).src(x.r), "call requires a callee symbol"); }
    { Fixture x; x.reject(Inst::make(Opcode::atom, Type::b32).space(StateSpace::shared).atom(AtomOp::cas).dst(x.r).src(Operand::addr(x.r1)).src(x.r1), "atom takes 3 source(s), got 2"); }
}

// ---------------------------------------------------------------------------
// (c) End to end
// ---------------------------------------------------------------------------

TEST_CASE("PTX IR - representative instructions verify and assemble with ptxas") {
    Fixture fx;
    Function& fn = fx.fn;
    Block* b = fx.bb;
    Reg p = fx.p, r = fx.r, r1 = fx.r1, rd = fx.rd, rd1 = fx.rd1, f = fx.f, f1 = fx.f1, fd = fx.fd;
    Reg f2 = fn.new_f32(), f3 = fn.new_f32(), fd1 = fn.new_f64();
    fn.add_param(Type::u32, "param_n");
    fn.add_param(Type::f32, "param_eps");
    Block* exit = fn.add_block("$L_exit");

    b->append(Inst::make(Opcode::ld, Type::u64).space(StateSpace::param).dst(rd).src(Operand::param("param_a")));
    b->append(Inst::make(Opcode::ld, Type::u32).space(StateSpace::param).dst(r).src(Operand::param("param_n")));
    b->append(Inst::make(Opcode::ld, Type::f32).space(StateSpace::param).dst(f).src(Operand::param("param_eps")));
    b->append(Inst::make(Opcode::mov, Type::u32).dst(r1).src(Operand::special(SpecialReg::tid_x)));
    b->append(Inst::make(Opcode::mov, Type::u32).dst(r1).src(Operand::special(SpecialReg::nctaid_x)));
    b->append(Inst::make(Opcode::mov, Type::u32).dst(r1).src(Operand::special(SpecialReg::laneid)));
    b->append(Inst::make(Opcode::mov, Type::u32).dst(r1).src(Operand::special(SpecialReg::warpid)));
    b->append(Inst::make(Opcode::ld, Type::f32).space(StateSpace::global).vec(VecWidth::v4).dst(Operand::vec({f, f1, f2, f3})).src(Operand::addr(rd, 16)));
    b->append(Inst::make(Opcode::ld, Type::f64).space(StateSpace::global).vec(VecWidth::v2).dst(Operand::vec({fd, fd1})).src(Operand::addr(rd)));
    b->append(Inst::make(Opcode::ld, Type::u16).space(StateSpace::global).dst(r).src(Operand::addr(rd, -2)));
    b->append(Inst::make(Opcode::ld, Type::u32).space(StateSpace::global).vec(VecWidth::v4).dst(Operand::vec(fn.new_regs(RegClass::B32, 4))).src(Operand::addr(rd)));
    b->append(Inst::make(Opcode::cvt, Type::f32).from(Type::f16).dst(f).src(r));
    b->append(Inst::make(Opcode::cvt, Type::f32).from(Type::u32).rnd(Rounding::rn).dst(f).src(r));
    b->append(Inst::make(Opcode::cvt, Type::f32).from(Type::s32).rnd(Rounding::rn).dst(f).src(r));
    b->append(Inst::make(Opcode::cvt, Type::s32).from(Type::f32).rnd(Rounding::rzi).dst(r).src(f));
    b->append(Inst::make(Opcode::cvt, Type::u64).from(Type::u32).dst(rd1).src(r));
    b->append(Inst::make(Opcode::cvt, Type::s64).from(Type::s32).dst(rd1).src(r));
    b->append(Inst::make(Opcode::cvt, Type::u32).from(Type::u64).dst(r).src(rd1));
    b->append(Inst::make(Opcode::cvt, Type::f32).from(Type::f64).rnd(Rounding::rn).dst(f).src(fd));
    b->append(Inst::make(Opcode::cvt, Type::f64).from(Type::f32).dst(fd).src(f));
    b->append(Inst::make(Opcode::mov, Type::f32).dst(f1).src(Operand::imm_f32(1.0f)));
    b->append(Inst::make(Opcode::mov, Type::f64).dst(fd1).src(Operand::imm_f64(1.0)));
    b->append(Inst::make(Opcode::mov, Type::b64).dst(rd1).src(Operand::imm(-42)));
    b->append(Inst::make(Opcode::mov, Type::b64).dst(rd1).src(fd));
    b->append(Inst::make(Opcode::mov, Type::b32).dst(f2).src(r));
    b->append(Inst::make(Opcode::add, Type::f32).rnd(Rounding::rn).ftz().sat().dst(f).src(f1).src(f2));
    b->append(Inst::make(Opcode::sub, Type::f32).dst(f).src(f1).src(f2));
    b->append(Inst::make(Opcode::add, Type::s64).dst(rd1).src(rd1).src(rd1));
    b->append(Inst::make(Opcode::sub, Type::u32).dst(r).src(r).src(Operand::imm(3)));
    b->append(Inst::make(Opcode::mul, Type::u64).lo().dst(rd1).src(rd1).src(rd1));
    b->append(Inst::make(Opcode::mul, Type::u32).wide().dst(rd1).src(r).src(r1));
    b->append(Inst::make(Opcode::mul, Type::s32).hi().dst(r).src(r).src(r1));
    b->append(Inst::make(Opcode::mul, Type::f32).dst(f).src(f1).src(Operand::imm_f32(1.44269504f)));
    b->append(Inst::make(Opcode::mad, Type::u32).lo().dst(r).src(r).src(r1).src(r));
    b->append(Inst::make(Opcode::mad, Type::s32).lo().dst(r).src(r).src(r1).src(r));
    b->append(Inst::make(Opcode::mad, Type::u32).wide().dst(rd1).src(r).src(r1).src(rd1));
    b->append(Inst::make(Opcode::fma, Type::f32).rnd(Rounding::rn).dst(f).src(f1).src(f2).src(f3));
    b->append(Inst::make(Opcode::fma, Type::f64).rnd(Rounding::rn).dst(fd).src(fd).src(fd1).src(fd));
    b->append(Inst::make(Opcode::div, Type::f32).approx().dst(f).src(f1).src(f2));
    b->append(Inst::make(Opcode::div, Type::f32).rnd(Rounding::rn).dst(f).src(f1).src(f2));
    b->append(Inst::make(Opcode::div, Type::f64).rnd(Rounding::rn).dst(fd).src(fd).src(fd1));
    b->append(Inst::make(Opcode::div, Type::u32).dst(r).src(r).src(r1));
    b->append(Inst::make(Opcode::div, Type::s64).dst(rd1).src(rd1).src(rd1));
    b->append(Inst::make(Opcode::rem, Type::s32).dst(r).src(r).src(r1));
    b->append(Inst::make(Opcode::rem, Type::u64).dst(rd1).src(rd1).src(rd1));
    b->append(Inst::make(Opcode::neg, Type::f32).dst(f).src(f1));
    b->append(Inst::make(Opcode::neg, Type::s32).dst(r).src(r1));
    b->append(Inst::make(Opcode::abs, Type::f32).dst(f).src(f1));
    b->append(Inst::make(Opcode::min, Type::f32).dst(f).src(f1).src(f2));
    b->append(Inst::make(Opcode::max, Type::f32).ftz().dst(f).src(f1).src(f2));
    b->append(Inst::make(Opcode::min, Type::u32).dst(r).src(r).src(r1));
    b->append(Inst::make(Opcode::and_, Type::b32).dst(r).src(r).src(Operand::imm(31)));
    b->append(Inst::make(Opcode::and_, Type::b32).dst(r).src(r).src(Operand::imm(0xFFFFFFFC)));
    b->append(Inst::make(Opcode::or_, Type::b64).dst(rd1).src(rd1).src(rd1));
    b->append(Inst::make(Opcode::xor_, Type::b32).dst(r).src(r).src(r1));
    b->append(Inst::make(Opcode::not_, Type::b32).dst(r).src(r1));
    b->append(Inst::make(Opcode::shl, Type::b64).dst(rd1).src(rd1).src(Operand::imm(2)));
    b->append(Inst::make(Opcode::shl, Type::b32).dst(r).src(r).src(r1));
    b->append(Inst::make(Opcode::shr, Type::u32).dst(r).src(r).src(Operand::imm(5)));
    b->append(Inst::make(Opcode::shr, Type::s32).dst(r).src(r).src(r1));
    b->append(Inst::make(Opcode::shr, Type::u64).dst(rd1).src(rd1).src(r1));
    b->append(Inst::make(Opcode::setp, Type::u32).cmp(CmpOp::ge).dst(p).src(r).src(r1));
    b->append(Inst::make(Opcode::setp, Type::f32).cmp(CmpOp::neu).dst(p).src(f).src(Operand::imm_f32(0.0f)));
    b->append(Inst::make(Opcode::setp, Type::s64).cmp(CmpOp::lt).dst(p).src(rd1).src(Operand::imm(0)));
    b->append(Inst::make(Opcode::setp, Type::f64).cmp(CmpOp::gt).dst(p).src(fd).src(fd1));
    b->append(Inst::make(Opcode::selp, Type::u32).dst(r).src(Operand::imm(1)).src(Operand::imm(0)).src(p));
    b->append(Inst::make(Opcode::selp, Type::f32).dst(f).src(f1).src(f2).src(p));
    b->append(Inst::make(Opcode::selp, Type::b64).dst(rd1).src(rd1).src(rd1).src(p));
    b->append(Inst::make(Opcode::shfl, Type::b32).sync().shfl(ShflMode::down).dst(f).src(f1).src(Operand::imm(16)).src(Operand::imm(31)).src(Operand::imm(0xffffffff)));
    b->append(Inst::make(Opcode::shfl, Type::b32).sync().shfl(ShflMode::bfly).dst(r).src(r1).src(r1).src(Operand::imm(31)).src(Operand::imm(0xffffffff)));
    b->append(Inst::make(Opcode::bar).sync().src(Operand::imm(0)));
    b->append(Inst::make(Opcode::rsqrt, Type::f32).approx().dst(f).src(f1));
    b->append(Inst::make(Opcode::sqrt, Type::f32).rnd(Rounding::rn).dst(f).src(f1));
    b->append(Inst::make(Opcode::sqrt, Type::f32).approx().ftz().dst(f).src(f1));
    b->append(Inst::make(Opcode::ex2, Type::f32).approx().dst(f).src(f1));
    b->append(Inst::make(Opcode::lg2, Type::f32).approx().dst(f).src(f1));
    b->append(Inst::make(Opcode::rcp, Type::f32).approx().dst(f).src(f1));
    b->append(Inst::make(Opcode::rcp, Type::f64).rnd(Rounding::rn).dst(fd).src(fd1));
    b->append(Inst::make(Opcode::sin, Type::f32).approx().dst(f).src(f1));
    b->append(Inst::make(Opcode::cos, Type::f32).approx().dst(f).src(f1));
    b->append(Inst::make(Opcode::mov, Type::u32).dst(r1).src(Operand::symbol("smem")));
    b->append(Inst::make(Opcode::st, Type::f32).space(StateSpace::shared).guard(p).src(Operand::addr(r1)).src(f));
    b->append(Inst::make(Opcode::ld, Type::f32).space(StateSpace::shared).guard(p, true).dst(f).src(Operand::addr(r1, 4)));
    b->append(Inst::make(Opcode::ld, Type::f32).space(StateSpace::shared).dst(f).src(Operand::addr("smem", 8)));
    b->append(Inst::make(Opcode::st, Type::f32).space(StateSpace::global).vec(VecWidth::v4).src(Operand::addr(rd)).src(Operand::vec({f, f1, f2, f3})));
    b->append(Inst::make(Opcode::st, Type::u32).space(StateSpace::global).src(Operand::addr(rd, 4)).src(Operand::imm(7)));
    b->append(Inst::make(Opcode::atom, Type::f32).space(StateSpace::global).atom(AtomOp::add).dst(f).src(Operand::addr(rd)).src(f1));
    b->append(Inst::make(Opcode::atom, Type::b32).space(StateSpace::shared).atom(AtomOp::cas).dst(r).src(Operand::addr(r1)).src(r1).src(Operand::imm(7)));
    b->append(Inst::make(Opcode::atom, Type::u32).space(StateSpace::global).atom(AtomOp::max).dst(r).src(Operand::addr(rd)).src(r1));
    b->append(Inst::make(Opcode::bra).uni().src(Operand::label("$L_exit")));
    exit->append(Inst::make(Opcode::ret));

    auto diags = verify(fn);
    dump(diags);
    REQUIRE(diags.empty());

    std::string ptx = print(fn);
    if (!ptxas_available()) return;
    bool ok = ptxas_assembles(ptx, "sm_70");
    if (!ok) std::cerr << ptx;
    CHECK(ok);
}

TEST_CASE("PTX IR - hand-built vector add + reduction kernel verifies, assembles and runs") {
    Function fn = build_vec_add_reduce();
    auto diags = verify(fn);
    dump(diags);
    REQUIRE(diags.empty());

    std::string ptx = print(fn);
    CHECK(ptx.find("ld.global.v4.f32 {") != std::string::npos);
    CHECK(ptx.find("shfl.sync.down.b32") != std::string::npos);
    CHECK(ptx.find("bar.sync 0;") != std::string::npos);
    CHECK(ptx.find(".shared .align 4 .f32 smem[32];") != std::string::npos);
    CHECK(ptx.find("mad.lo.u32") != std::string::npos);
    CHECK(ptx.find("atom.global.add.f32") != std::string::npos);

    if (ptxas_available()) {
        bool ok = ptxas_assembles(ptx, "sm_70");
        if (!ok) std::cerr << ptx;
        CHECK(ok);
    }
    if (!gpu_ready()) return;

    std::string err;
    CudaModule mod = CudaModule::load(ptx, &err);
    if (!mod.valid()) std::cerr << "driver JIT: " << err << "\n" << ptx;
    REQUIRE(mod.valid());

    const uint32_t grid = 2, block = 64;
    const uint32_t n = grid * block * 4 + 3; // 3-element scalar tail
    std::vector<float> a(n), b(n), ref(n);
    double ref_sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        a[i] = float(i % 13) * 0.25f - 1.0f;
        b[i] = float(i % 7) * 0.5f;
        ref[i] = a[i] + b[i];
        ref_sum += ref[i];
    }
    CudaBuffer da = CudaBuffer::alloc(n * 4), db = CudaBuffer::alloc(n * 4);
    CudaBuffer dout = CudaBuffer::alloc(n * 4), dsum = CudaBuffer::alloc(4);
    REQUIRE(da.valid() && db.valid() && dout.valid() && dsum.valid());
    REQUIRE(da.upload(a.data(), n * 4));
    REQUIRE(db.upload(b.data(), n * 4));
    REQUIRE(dout.zero());
    REQUIRE(dsum.zero());

    void* pa = da.device_ptr(); void* pb = db.device_ptr();
    void* po = dout.device_ptr(); void* ps = dsum.device_ptr();
    uint32_t nn = n;
    std::vector<void*> args = { &pa, &pb, &po, &ps, &nn };
    bool launched = mod.launch_1d("vec_add_reduce", grid, block, args.data(), 0, &err);
    if (!launched) std::cerr << "launch: " << err << "\n";
    REQUIRE(launched);

    std::vector<float> got(n);
    float got_sum = 0.0f;
    REQUIRE(dout.download(got.data(), n * 4));
    REQUIRE(dsum.download(&got_sum, 4));
    size_t bad = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (!near(got[i], ref[i], 1e-6f)) ++bad;
    }
    CHECK_EQ(bad, size_t(0));
    CHECK(near(got_sum, float(ref_sum), 1e-4f));
}
