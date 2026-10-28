#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/fma_opt.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/pass_catalog.hpp>
#include <brass/mir/pass_pipeline.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/codegen/baseline_jit.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/runtime/deterministic_math.hpp>
#include <brass/object/object_writer.hpp>
#include <brass/object/coff_writer.hpp>
#include <brass/object/elf_writer.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>
#include <random>

using namespace brass;
using namespace brass::object;

namespace {

void scramble_heap() {
    static std::mt19937_64 rng(0x123456789ABCDEF0ULL);
    std::vector<std::vector<uint8_t>> garbage;
    for (int i = 0; i < 50; ++i) {
        size_t sz = static_cast<size_t>(rng() % 4096) + 16;
        garbage.emplace_back(sz, static_cast<uint8_t>(rng() & 0xFF));
    }
}

std::unique_ptr<Module> build_complex_numeric_module() {
    auto mod = std::make_unique<Module>("numeric_determinism");
    Builder b(*mod);

    // 1. Math and Bitwise Function
    Function* fn1 = mod->create_function("math_kernel", Type::i64(), {Type::i64(), Type::i64(), Type::f64()});
    b.set_function(fn1);
    BasicBlock* entry1 = b.append_block("entry");
    Value* x = b.add_block_param(entry1, Type::i64());
    Value* y = b.add_block_param(entry1, Type::i64());
    Value* z = b.add_block_param(entry1, Type::f64());

    Value* sum = b.build_add(x, y);
    Value* diff = b.build_sub(x, y);
    Value* prod = b.build_mul(sum, diff);
    Value* xor_v = b.build_xor(prod, x);
    Value* popc = b.build_popcnt(xor_v);

    Value* fprod = b.build_mul(z, z);
    Value* f_as_i = b.build_fptosi_i64(fprod);
    Value* final_v = b.build_add(popc, f_as_i);
    b.build_ret(final_v);
    fn1->rebuild_cfg_predecessors();

    // 2. Loop Function with Block Parameters
    Function* fn2 = mod->create_function("loop_kernel", Type::i64(), {Type::i64()});
    b.set_function(fn2);
    BasicBlock* entry2 = b.append_block("entry");
    Value* n = b.add_block_param(entry2, Type::i64());

    BasicBlock* loop_header = b.create_block("loop_header");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* init_i = b.build_iconst_i64(0);
    Value* init_acc = b.build_iconst_i64(1);
    b.build_br(loop_header, {init_i, init_acc});

    fn2->append_block(loop_header);
    b.position_at_end(loop_header);
    Value* cur_i = b.add_block_param(loop_header, Type::i64());
    Value* cur_acc = b.add_block_param(loop_header, Type::i64());
    Value* cmp = b.build_slt(cur_i, n);
    b.build_br_if(cmp, loop_body, {}, exit_bb, {cur_acc});

    fn2->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* one = b.build_iconst_i64(1);
    Value* next_i = b.build_add(cur_i, one);
    Value* three = b.build_iconst_i64(3);
    Value* next_acc = b.build_add(b.build_mul(cur_acc, three), cur_i);
    b.build_br(loop_header, {next_i, next_acc});

    fn2->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* exit_res = b.add_block_param(exit_bb, Type::i64());
    b.build_ret(exit_res);
    fn2->rebuild_cfg_predecessors();

    return mod;
}

std::unique_ptr<Module> build_gc_and_speculation_module() {
    auto mod = std::make_unique<Module>("gc_spec_determinism");
    mod->add_external_symbol("brass_gc_alloc");
    mod->add_external_symbol("brass_gc_safepoint");
    mod->add_external_symbol("fallback_twin");
    Builder b(*mod);

    // 1. GC Safepoint and Allocation Function
    Function* fn_gc = mod->create_function("gc_worker", Type::i64(), {Type::gcref(), Type::i64()});
    b.set_function(fn_gc);
    BasicBlock* entry = b.append_block("entry");
    Value* ref = b.add_block_param(entry, Type::gcref());
    Value* count = b.add_block_param(entry, Type::i64());

    Value* loaded = b.build_load(Type::i64(), ref, 8);
    b.build_safepoint();
    Value* stored_val = b.build_add(loaded, count);
    b.build_store(Type::i64(), ref, 8, stored_val);
    b.build_ret(stored_val);
    fn_gc->rebuild_cfg_predecessors();

    // 2. Speculative Function with Guard, Patchable Const, and Resume Point
    Function* fn_spec = mod->create_function("spec_kernel", Type::i64(), {Type::i64(), Type::i64()});
    b.set_function(fn_spec);
    BasicBlock* spec_entry = b.append_block("entry");
    Value* a = b.add_block_param(spec_entry, Type::i64());
    Value* c = b.add_block_param(spec_entry, Type::i64());

    Value* pconst = b.build_patchable_const_i32("ic_type_tag", 100);
    Value* pconst64 = b.build_zext_i64(pconst);
    Value* cond = b.build_eq(a, pconst64);

    b.build_guard(cond, "deopt_stub_1", {a, c, pconst64});
    Value* fast_res = b.build_mul(a, c);
    b.build_ret(fast_res);

    BasicBlock* resume_bb = b.append_block("resume_target");
    b.position_at_end(resume_bb);
    b.build_resume_point(10);
    b.build_ret(c);

    fn_spec->add_resume_point(10, resume_bb);
    fn_spec->rebuild_cfg_predecessors();

    return mod;
}

} // namespace

TEST_CASE("Determinism Ratchet - Numeric & CFG Module (COFF & ELF64)") {
    std::vector<uint8_t> first_coff;
    std::vector<uint8_t> first_elf;

    for (int iter = 0; iter < 5; ++iter) {
        scramble_heap();

        auto mod = build_complex_numeric_module();
        CHECK(verify_module(*mod));

        // COFF Win64 Emission
        ObjectFile coff_obj = compile_module_to_object(*mod, Target::x64_windows());
        std::vector<uint8_t> coff_bytes = emit_coff_object(coff_obj);

        // ELF64 Linux Emission
        ObjectFile elf_obj = compile_module_to_object(*mod, Target::x64_linux());
        std::vector<uint8_t> elf_bytes = emit_elf_object(elf_obj);

        if (iter == 0) {
            first_coff = std::move(coff_bytes);
            first_elf = std::move(elf_bytes);
            CHECK(!first_coff.empty());
            CHECK(!first_elf.empty());
        } else {
            REQUIRE_EQ(coff_bytes.size(), first_coff.size());
            CHECK(std::memcmp(coff_bytes.data(), first_coff.data(), first_coff.size()) == 0);

            REQUIRE_EQ(elf_bytes.size(), first_elf.size());
            CHECK(std::memcmp(elf_bytes.data(), first_elf.data(), first_elf.size()) == 0);
        }
    }
}

TEST_CASE("Determinism Ratchet - GC & Speculation Module (COFF & ELF64)") {
    std::vector<uint8_t> first_coff;
    std::vector<uint8_t> first_elf;

    for (int iter = 0; iter < 5; ++iter) {
        scramble_heap();

        auto mod = build_gc_and_speculation_module();
        CHECK(verify_module(*mod));

        // COFF Win64 Emission
        ObjectFile coff_obj = compile_module_to_object(*mod, Target::x64_windows());
        std::vector<uint8_t> coff_bytes = emit_coff_object(coff_obj);

        // ELF64 Linux Emission
        ObjectFile elf_obj = compile_module_to_object(*mod, Target::x64_linux());
        std::vector<uint8_t> elf_bytes = emit_elf_object(elf_obj);

        if (iter == 0) {
            first_coff = std::move(coff_bytes);
            first_elf = std::move(elf_bytes);
            CHECK(!first_coff.empty());
            CHECK(!first_elf.empty());
        } else {
            REQUIRE_EQ(coff_bytes.size(), first_coff.size());
            CHECK(std::memcmp(coff_bytes.data(), first_coff.data(), first_coff.size()) == 0);

            REQUIRE_EQ(elf_bytes.size(), first_elf.size());
            CHECK(std::memcmp(elf_bytes.data(), first_elf.data(), first_elf.size()) == 0);
        }
    }
}

TEST_CASE("Determinism Ratchet - Text Parser Roundtrip Compilation") {
    const char* kMirText = R"(module @parsed_determinism

func @fibonacci(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 2
  %2 = slt.i64 %0, %1
  br_if %2, bb_base, bb_rec

bb_base:
  ret %0

bb_rec:
  %3 = iconst.i64 1
  %4 = sub.i64 %0, %3
  %5 = call.i64 @fibonacci(%4)
  %6 = iconst.i64 2
  %7 = sub.i64 %0, %6
  %8 = call.i64 @fibonacci(%7)
  %9 = add.i64 %5, %8
  ret %9
}
)";

    std::vector<uint8_t> base_coff;
    std::vector<uint8_t> base_elf;

    for (int iter = 0; iter < 5; ++iter) {
        scramble_heap();

        auto mod = parse_module(kMirText);
        REQUIRE(mod != nullptr);
        CHECK(verify_module(*mod));

        ObjectFile coff_obj = compile_module_to_object(*mod, Target::x64_windows());
        std::vector<uint8_t> coff_bytes = emit_coff_object(coff_obj);

        ObjectFile elf_obj = compile_module_to_object(*mod, Target::x64_linux());
        std::vector<uint8_t> elf_bytes = emit_elf_object(elf_obj);

        if (iter == 0) {
            base_coff = std::move(coff_bytes);
            base_elf = std::move(elf_bytes);
        } else {
            REQUIRE_EQ(coff_bytes.size(), base_coff.size());
            CHECK(std::memcmp(coff_bytes.data(), base_coff.data(), base_coff.size()) == 0);

            REQUIRE_EQ(elf_bytes.size(), base_elf.size());
            CHECK(std::memcmp(elf_bytes.data(), base_elf.data(), base_elf.size()) == 0);
        }
    }
}

namespace {

uint64_t f64_raw_bits(double x) noexcept {
    uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

uint64_t ulp_distance(double a, double b) {
    if (std::isnan(a) && std::isnan(b)) return 0;
    if (std::isinf(a) && std::isinf(b) && std::signbit(a) == std::signbit(b)) return 0;
    if (a == 0.0 && b == 0.0) return 0;
    uint64_t ua = f64_raw_bits(a);
    uint64_t ub = f64_raw_bits(b);
    auto to_ordered = [](uint64_t u) -> int64_t {
        if (u & 0x8000000000000000ULL) {
            return -static_cast<int64_t>(u & 0x7FFFFFFFFFFFFFFFULL);
        }
        return static_cast<int64_t>(u);
    };
    int64_t oa = to_ordered(ua);
    int64_t ob = to_ordered(ub);
    return static_cast<uint64_t>(oa >= ob ? (oa - ob) : (ob - oa));
}

// Build a 2D physics + FMA-sensitive + loop-reduction + deterministic-transcendental simulation tick
std::unique_ptr<Module> build_deterministic_sim_module(bool deterministic) {
    auto mod = std::make_unique<Module>("det_sim_mod");
    mod->set_deterministic_fp(deterministic);
    mod->add_external_symbol("brass_det_sin");
    mod->add_external_symbol("brass_det_cos");
    mod->add_external_symbol("brass_det_atan2");
    mod->add_external_symbol("brass_det_hypot");
    mod->add_external_symbol("brass_det_exp");
    mod->add_external_symbol("brass_det_log");

    Builder b(*mod);
    // void step_frame(ptr state, i64 count, f64 dt)
    // state layout: 16 doubles:
    //   [0..7]: particle positions/velocities updated in loop + reduction
    //   [8..15]: summary accumulators (FMA-sensitive a*b+c, reduction sum, trig/exp/log state)
    Function* fn = mod->create_function("step_frame", Type::f64(),
                                        {Type::ptr(), Type::i64(), Type::f64()});
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* state_ptr = b.add_block_param(entry, Type::ptr());
    Value* count = b.add_block_param(entry, Type::i64());
    Value* dt = b.add_block_param(entry, Type::f64());

    BasicBlock* loop_hdr = b.create_block("loop_hdr");
    BasicBlock* loop_body = b.create_block("loop_body");
    BasicBlock* exit_bb = b.create_block("exit");

    Value* zero_i64 = b.build_iconst_i64(0);
    Value* init_acc = b.build_load(Type::f64(), state_ptr, 8 * 8);
    b.build_br(loop_hdr, {zero_i64, init_acc});

    fn->append_block(loop_hdr);
    b.position_at_end(loop_hdr);
    Value* idx = b.add_block_param(loop_hdr, Type::i64());
    Value* acc = b.add_block_param(loop_hdr, Type::f64());
    Value* cond = b.build_slt(idx, count);
    b.build_br_if(cond, loop_body, {}, exit_bb, {acc});

    fn->append_block(loop_body);
    b.position_at_end(loop_body);
    Value* elem = b.build_load_indexed(Type::f64(), state_ptr, idx, 8, 0);
    // FMA-sensitive mul + add: elem * dt + acc
    Value* prod = b.build_mul(elem, dt);
    Value* next_acc = b.build_add(acc, prod);
    // Elementwise update: elem_new = elem * 0.9990234375 + prod * 0.125
    Value* damp = b.build_fconst_f64(0.9990234375);
    Value* k = b.build_fconst_f64(0.125);
    Value* elem_new = b.build_add(b.build_mul(elem, damp), b.build_mul(prod, k));
    b.build_store_indexed(Type::f64(), state_ptr, idx, 8, 0, elem_new);
    Value* next_idx = b.build_add(idx, b.build_iconst_i64(1));
    b.build_br(loop_hdr, {next_idx, next_acc});

    fn->append_block(exit_bb);
    b.position_at_end(exit_bb);
    Value* red_sum = b.add_block_param(exit_bb, Type::f64());

    // Now invoke deterministic math runtime helpers on the accumulated state
    Value* x0 = b.build_load(Type::f64(), state_ptr, 0);
    Value* y0 = b.build_load(Type::f64(), state_ptr, 8);
    Value* s_val = b.build_call("brass_det_sin", Type::f64(), {red_sum});
    Value* c_val = b.build_call("brass_det_cos", Type::f64(), {x0});
    Value* at_val = b.build_call("brass_det_atan2", Type::f64(), {y0, x0});
    Value* hyp_val = b.build_call("brass_det_hypot", Type::f64(), {s_val, c_val});
    Value* exp_val = b.build_call("brass_det_exp", Type::f64(), {b.build_mul(s_val, dt)});
    Value* log_val = b.build_call("brass_det_log", Type::f64(),
                                  {b.build_add(hyp_val, b.build_fconst_f64(1.0))});

    // FMA-contraction sensitive expression: (1 + 2^-27) * (1 + 2^-27) - (1 + 2^-26)
    Value* fma_a = b.build_add(b.build_fconst_f64(1.0000000074505806),
                               b.build_mul(dt, b.build_fconst_f64(1e-9)));
    Value* fma_c = b.build_neg(b.build_fconst_f64(1.0000000149011612));
    Value* fma_expr = b.build_add(b.build_mul(fma_a, fma_a), fma_c);

    Value* combined = b.build_add(
        b.build_add(red_sum, b.build_add(s_val, c_val)),
        b.build_add(b.build_add(at_val, exp_val), b.build_add(log_val, fma_expr)));

    b.build_store(Type::f64(), state_ptr, 8 * 8, combined);
    b.build_store(Type::f64(), state_ptr, 9 * 8, s_val);
    b.build_store(Type::f64(), state_ptr, 10 * 8, c_val);
    b.build_store(Type::f64(), state_ptr, 11 * 8, at_val);
    b.build_store(Type::f64(), state_ptr, 12 * 8, hyp_val);
    b.build_store(Type::f64(), state_ptr, 13 * 8, exp_val);
    b.build_store(Type::f64(), state_ptr, 14 * 8, log_val);
    b.build_store(Type::f64(), state_ptr, 15 * 8, fma_expr);
    b.build_ret(combined);

    fn->rebuild_cfg_predecessors();
    return mod;
}

} // namespace

TEST_CASE("Determinism #156 - Bit-Exact Double-Double Transcendentals (Accuracy & Special Cases)") {
    // 1. Special IEEE-754 inputs (NaN, +-0, +-Inf, quadrant boundaries)
    const double qnan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    CHECK_EQ(f64_raw_bits(math::det_sin(qnan)), math::kCanonicalNaNBits);
    CHECK_EQ(f64_raw_bits(math::det_cos(qnan)), math::kCanonicalNaNBits);
    CHECK_EQ(f64_raw_bits(math::det_sin(inf)), math::kCanonicalNaNBits);
    CHECK_EQ(f64_raw_bits(math::det_cos(-inf)), math::kCanonicalNaNBits);
    CHECK_EQ(f64_raw_bits(math::det_sin(0.0)), f64_raw_bits(0.0));
    CHECK_EQ(f64_raw_bits(math::det_sin(-0.0)), f64_raw_bits(-0.0));
    CHECK_EQ(f64_raw_bits(math::det_cos(0.0)), f64_raw_bits(1.0));
    CHECK_EQ(f64_raw_bits(math::det_exp(0.0)), f64_raw_bits(1.0));
    CHECK_EQ(f64_raw_bits(math::det_exp(-inf)), f64_raw_bits(0.0));
    CHECK_EQ(f64_raw_bits(math::det_exp(inf)), f64_raw_bits(inf));
    CHECK_EQ(f64_raw_bits(math::det_log(1.0)), f64_raw_bits(0.0));
    CHECK_EQ(f64_raw_bits(math::det_log(0.0)), f64_raw_bits(-inf));
    CHECK_EQ(f64_raw_bits(math::det_log(-1.0)), math::kCanonicalNaNBits);
    CHECK_EQ(f64_raw_bits(math::det_pow(0.0, 0.0)), f64_raw_bits(1.0));
    CHECK_EQ(f64_raw_bits(math::det_pow(-2.0, 3.0)), f64_raw_bits(-8.0));
    CHECK_EQ(f64_raw_bits(math::det_hypot(3.0, 4.0)), f64_raw_bits(5.0));
    CHECK_EQ(f64_raw_bits(math::det_hypot(inf, qnan)), f64_raw_bits(inf));
    CHECK_EQ(f64_raw_bits(math::det_cbrt(-27.0)), f64_raw_bits(-3.0));

    // 2. Accuracy across representative domain (< 2 ULPs vs host libm, while being 100% pure integer/IEEE arithmetic)
    const double test_angles[] = {
        -100.0, -3.141592653589793, -1.5707963267948966, -0.7853981633974483,
        -1e-9, -0.0, 0.0, 1e-9, 0.1234567890123456, 0.7853981633974483,
        1.5707963267948966, 3.141592653589793, 42.5, 1000.0
    };
    for (double x : test_angles) {
        CHECK(ulp_distance(math::det_sin(x), std::sin(x)) <= 2);
        CHECK(ulp_distance(math::det_cos(x), std::cos(x)) <= 2);
        CHECK(ulp_distance(math::det_tan(x), std::tan(x)) <= 4);
        CHECK(ulp_distance(math::det_atan(x), std::atan(x)) <= 2);
        if (x > -20.0 && x < 20.0) {
            CHECK(ulp_distance(math::det_exp(x), std::exp(x)) <= 2);
        }
        if (x > 0.0) {
            CHECK(ulp_distance(math::det_log(x), std::log(x)) <= 2);
            CHECK(ulp_distance(math::det_cbrt(x), std::cbrt(x)) <= 2);
        }
    }
    for (double u : {-0.999, -0.5, -0.125, 0.0, 0.125, 0.5, 0.999}) {
        CHECK(ulp_distance(math::det_asin(u), std::asin(u)) <= 2);
        CHECK(ulp_distance(math::det_acos(u), std::acos(u)) <= 2);
    }

    // 3. Cross-platform bit-exact IEEE-754 golden table for all 12 brass_det_* functions.
    //    Every architecture (Linux x64, Windows x64, macOS arm64, Linux arm64) must produce
    //    these exact 64-bit IEEE-754 bit patterns with zero ULP drift.
    struct UnaryGolden {
        double x;
        uint64_t sin_bits;
        uint64_t cos_bits;
        uint64_t tan_bits;
        uint64_t atan_bits;
        uint64_t exp_bits;
        uint64_t cbrt_bits;
    };
    static constexpr UnaryGolden kUnaryGoldens[] = {
        {-100.0,
         0x3fe03425b78c4db8ULL, 0x3feb981dbf665fdfULL, 0x3fe2ca74d62b5d38ULL,
         0xbff8f905eb2def22ULL, 0x36ea8c1f14e2af5dULL, 0xc01290fca9c761f8ULL},
        {-3.141592653589793,
         0xbca1a62633145c07ULL, 0xbff0000000000000ULL, 0x3ca1a62633145c07ULL,
         0xbff433b8a322ddd3ULL, 0x3fa620227b598efaULL, 0xbff76ef7e73104b7ULL},
        {-0.7853981633974483,
         0xbfe6a09e667f3bccULL, 0x3fe6a09e667f3bcdULL, 0xbfefffffffffffffULL,
         0xbfe54e04c05d06a0ULL, 0x3fdd2e171cf04879ULL, 0xbfed8639fdcb60eaULL},
        {-1e-7,
         0xbe7ad7f29abcaf3bULL, 0x3fefffffffffffd3ULL, 0xbe7ad7f29abcaf61ULL,
         0xbe7ad7f29abcaf2fULL, 0x3fefffffca501af8ULL, 0xbf73030f03de9991ULL},
        {0.1234567890123456,
         0x3fbf8653acfa781fULL, 0x3fefc1a655cf1085ULL, 0x3fbfc438ddd978aaULL,
         0x3fbf722151138c9aULL, 0x3ff21a380e18c48dULL, 0x3fdfde25fa3f00a4ULL},
        {0.7853981633974483,
         0x3fe6a09e667f3bccULL, 0x3fe6a09e667f3bcdULL, 0x3fefffffffffffffULL,
         0x3fe54e04c05d06a0ULL, 0x40018bd669471caaULL, 0x3fed8639fdcb60eaULL},
        {1.5707963267948966,
         0x3ff0000000000000ULL, 0x3c91a62633145c07ULL, 0x434d02967c31cdb5ULL,
         0x3ff00fe987ed02ffULL, 0x40133dedc855935fULL, 0x3ff2996264e0e3feULL},
        {42.5,
         0xbfefdff0cd5c5750ULL, 0x3fb6a0506a04973eULL, 0xc0268a40b864aa65ULL,
         0x3ff8c19f811be511ULL, 0x43c3e5d83907a3dcULL, 0x400beb0a72eb6e31ULL},
    };
    for (const auto& g : kUnaryGoldens) {
        CHECK_EQ(f64_raw_bits(brass_det_sin(g.x)), g.sin_bits);
        CHECK_EQ(f64_raw_bits(brass_det_cos(g.x)), g.cos_bits);
        CHECK_EQ(f64_raw_bits(brass_det_tan(g.x)), g.tan_bits);
        CHECK_EQ(f64_raw_bits(brass_det_atan(g.x)), g.atan_bits);
        CHECK_EQ(f64_raw_bits(brass_det_exp(g.x)), g.exp_bits);
        CHECK_EQ(f64_raw_bits(brass_det_cbrt(g.x)), g.cbrt_bits);
    }

    // Arguments past fdlibm's medium reduction take the Payne-Hanek path,
    // which needs 2/pi to beyond a thousand bits (sin(1e10) once came out
    // 1e-9). The bits are V8's (fdlibm), which these agree with.
    struct TrigGolden {
        double x;
        uint64_t sin_bits;
        uint64_t cos_bits;
    };
    static constexpr TrigGolden kLargeTrigGoldens[] = {
        {1e10, 0xbfdf334c7896a4e3ULL, 0x3febf098901c931aULL},
        {1e22, 0xbfeb453ab76bf397ULL, 0x3fe0be2cef01c8f4ULL},
        {1e300, 0xbfea2c16b010e385ULL, 0xbfe2699022adc4c1ULL},
        {-823551.0, 0xbfef1e1d322aef27ULL, 0x3fcdd9e00a413200ULL},
    };
    for (const auto& g : kLargeTrigGoldens) {
        CHECK_EQ(f64_raw_bits(brass_det_sin(g.x)), g.sin_bits);
        CHECK_EQ(f64_raw_bits(brass_det_cos(g.x)), g.cos_bits);
    }

    // One argument down each of fdlibm's paths, which the fast path takes
    // without its branches (deterministic_trig.cpp): kernel_cos's three qx,
    // the +-1 step through the medium one, both signs of every quadrant, x by
    // pi/2's own high word and the cancellation steps (fdlibm_rem_pio2 whole),
    // and the medium range's top. The bits are V8's (fdlibm).
    static constexpr TrigGolden kPathTrigGoldens[] = {
        {0.2, 0x3fc96dff233dd2bcULL, 0x3fef5cb49577627aULL},
        {0.5, 0x3fdeaee8744b05f0ULL, 0x3fec1528065b7d50ULL},
        {0.785, 0x3fe69e4fd79ac743ULL, 0x3fe6a2ecb934b59aULL},
        {1.0, 0x3feaed548f090ceeULL, 0x3fe14a280fb5068cULL},
        {-1.0, 0xbfeaed548f090ceeULL, 0x3fe14a280fb5068cULL},
        {2.0, 0x3fed18f6ead1b446ULL, 0xbfdaa22657537205ULL},
        {-2.3, 0xbfe7dcd12d582f24ULL, 0xbfe5522217302fe0ULL},
        {2.356, 0x3fe6a1bec8ab043dULL, 0xbfe69f7df5f75366ULL},
        {1.5707963, 0x3feffffffffffffdULL, 0x3e5cc553391a6262ULL},
        {-1.5707964, 0xbfefffffffffffe8ULL, 0xbe73a69dccb96762ULL},
        {3.0, 0x3fc210386db6d55bULL, 0xbfefae04be85e5d2ULL},
        {-3.0, 0xbfc210386db6d55bULL, 0xbfefae04be85e5d2ULL},
        {4.0, 0xbfe837b9dddc1eaeULL, 0xbfe4eaa606db24c1ULL},
        {-4.0, 0x3fe837b9dddc1eaeULL, 0xbfe4eaa606db24c1ULL},
        {5.0, 0xbfeeaf81f5e09933ULL, 0x3fd22785706b4ad9ULL},
        {-5.0, 0x3feeaf81f5e09933ULL, 0x3fd22785706b4ad9ULL},
        {6.0, 0xbfd1e1f18ab0a2c0ULL, 0x3feeb9b7097822f6ULL},
        {-6.0, 0x3fd1e1f18ab0a2c0ULL, 0x3feeb9b7097822f6ULL},
        {7.0, 0x3fe50608c26d0a08ULL, 0x3fe81ff79ed92017ULL},
        {250.41, 0xbfe968b443c61358ULL, 0x3fe373ba497791d0ULL},
        {4.71238898038469, 0xbff0000000000000ULL, 0xbcaa79394c9e8a0aULL},
        {-7.853981633974483, 0xbff0000000000000ULL, 0x3cb60fafbfd97309ULL},
        {6.283185307179586, 0xbcb1a62633145c07ULL, 0x3ff0000000000000ULL},
        {157079.63267948966, 0x3daaa8ecfb27878aULL, 0x3ff0000000000000ULL},
        {823549.0, 0xbfe3bc41ec71c06dULL, 0x3fe930880dddf826ULL},
    };
    for (const auto& g : kPathTrigGoldens) {
        CHECK_EQ(f64_raw_bits(brass_det_sin(g.x)), g.sin_bits);
        CHECK_EQ(f64_raw_bits(brass_det_cos(g.x)), g.cos_bits);
    }

    struct UnitDomainGolden {
        double u;
        uint64_t asin_bits;
        uint64_t acos_bits;
        uint64_t log_u_plus_1p5_bits;
    };
    static constexpr UnitDomainGolden kUnitGoldens[] = {
        {-0.999, 0xbff86ac9ad18f803ULL, 0x4008c66280ae928eULL, 0xbfe61de4e1773bf8ULL},
        {-0.5,   0xbfe0c152382d7366ULL, 0x4000c152382d7366ULL, 0x0000000000000000ULL},
        {-0.125, 0xbfc00abe0c129e1eULL, 0x3ffb235315c680dcULL, 0x3fd4618bc21c5ec2ULL},
        {0.125,  0x3fc00abe0c129e1eULL, 0x3ff720a392c1d955ULL, 0x3fdf128f5faf06edULL},
        {0.5,    0x3fe0c152382d7366ULL, 0x3ff0c152382d7365ULL, 0x3fe62e42fefa39efULL},
        {0.999,  0x3ff86ac9ad18f803ULL, 0x3fa6e634e566a29eULL, 0x3fed4ef9e98d7058ULL},
    };
    for (const auto& g : kUnitGoldens) {
        CHECK_EQ(f64_raw_bits(brass_det_asin(g.u)), g.asin_bits);
        CHECK_EQ(f64_raw_bits(brass_det_acos(g.u)), g.acos_bits);
        CHECK_EQ(f64_raw_bits(brass_det_log(g.u + 1.5)), g.log_u_plus_1p5_bits);
    }

    struct BinaryGolden {
        double a;
        double b;
        uint64_t atan2_bits;
        uint64_t hypot_bits;
        uint64_t pow_abs_a_b_bits;
    };
    static constexpr BinaryGolden kBinaryGoldens[] = {
        {0.375, 1.25, 0x3fd2a73a661eaf06ULL, 0x3ff4e16fdacff937ULL, 0x3fd2c7f098b2a9f8ULL},
        {-2.5,  0.75, 0xbff4782cbabc8157ULL, 0x4004e16fdacff937ULL, 0x3fffcf927fccd2a0ULL},
        {3.141592653589793, -1.5707963267948966, 0x4000468a8ace4df6ULL, 0x400c196908691da6ULL, 0x3fc5329c93a33fb0ULL},
        {1e-6,  1e6,  0x3d719799812dea11ULL, 0x412e848000000000ULL, 0x0000000000000000ULL},
        {2.25,  1.5,  0x3fef730bd281f69bULL, 0x4005a22073490377ULL, 0x400b000000000000ULL},
        {7.5,  -0.25, 0x3ff9aa76ecfd4070ULL, 0x401e0443f6a332a8ULL, 0x3fe35638b103c0a2ULL},
    };
    for (const auto& g : kBinaryGoldens) {
        CHECK_EQ(f64_raw_bits(brass_det_atan2(g.a, g.b)), g.atan2_bits);
        CHECK_EQ(f64_raw_bits(brass_det_hypot(g.a, g.b)), g.hypot_bits);
        CHECK_EQ(f64_raw_bits(brass_det_pow(std::fabs(g.a), g.b)), g.pow_abs_a_b_bits);
    }
}

TEST_CASE("Determinism #156 - deterministic_fp MIR Attribute Disables FMA Contraction and Horizontal FP Reduction") {
    // 1. Verify printer/parser roundtrip and C API for deterministic_fp
    {
        Module mod("det_attr_test");
        Function* fn = mod.create_function("det_kernel", Type::f64(),
                                           {Type::f64(), Type::f64(), Type::f64()});
        fn->set_deterministic_fp(true);
        CHECK(fn->is_deterministic_fp());

        Builder b(mod);
        b.set_function(fn);
        BasicBlock* entry = b.append_block("entry");
        Value* a = b.add_block_param(entry, Type::f64());
        Value* x = b.add_block_param(entry, Type::f64());
        Value* c = b.add_block_param(entry, Type::f64());
        Value* prod = b.build_mul(a, x);
        Value* sum = b.build_add(prod, c);
        b.build_ret(sum);
        fn->rebuild_cfg_predecessors();

        std::string printed = to_string(mod);
        CHECK(printed.find("deterministic_fp") != std::string::npos);

        auto parsed = parse_module(printed);
        REQUIRE(parsed != nullptr);
        Function* parsed_fn = parsed->get_function("det_kernel");
        REQUIRE(parsed_fn != nullptr);
        CHECK(parsed_fn->is_deterministic_fp());

        // fma_opt_pass must be a no-op when deterministic_fp is enabled
        CHECK(!fma_opt_pass(*parsed_fn));
        for (const Instruction* inst : *parsed_fn->entry_block()) {
            CHECK(inst->opcode() != Opcode::fma_f64);
        }

        // Disabling deterministic_fp allows fma_opt_pass to fuse mul+add -> fma.f64
        parsed_fn->set_deterministic_fp(false);
        CHECK(fma_opt_pass(*parsed_fn));
    }

    // 2. Verify horizontal FP reduction is NOT vectorized/reassociated when deterministic_fp is true
    {
        auto mod_det = build_deterministic_sim_module(true);
        Function* fn_det = mod_det->get_function("step_frame");
        REQUIRE(fn_det != nullptr);
        CHECK(fn_det->is_deterministic_fp());

        PassPipelineOptions opts = production_pass_pipeline_options();
        (void)run_pass_pipeline(*mod_det, opts);

        for (const BasicBlock* bb : fn_det->blocks()) {
            for (const Value* param : bb->params()) {
                CHECK(!param->type().is_vector());
            }
            for (const Instruction* inst : *bb) {
                CHECK(inst->opcode() != Opcode::fma_f64);
                CHECK(inst->opcode() != Opcode::vfma);
            }
        }

        // Verify that lowering fn_det through BOTH x64 and AArch64 instruction selectors
        // emits zero fused multiply-add (Vfmadd*) LIR instructions on either architecture.
        auto is_lir_fma = [](codegen::LirOpcode op) {
            return op == codegen::LirOpcode::Vfmadd213ss || op == codegen::LirOpcode::Vfmadd231ss ||
                   op == codegen::LirOpcode::Vfmadd213sd || op == codegen::LirOpcode::Vfmadd231sd ||
                   op == codegen::LirOpcode::Vfmadd213ps || op == codegen::LirOpcode::Vfmadd231ps ||
                   op == codegen::LirOpcode::Vfmadd213pd || op == codegen::LirOpcode::Vfmadd231pd;
        };

        x64::X64ISel x64_isel(Target::x64_linux());
        auto x64_lir = x64_isel.lower(*fn_det);
        REQUIRE(x64_lir != nullptr);
        for (const auto& lbb : x64_lir->blocks) {
            for (const auto& linst : lbb->instructions) {
                CHECK(!is_lir_fma(linst->opcode));
            }
        }

        aarch64::AArch64ISel arm64_isel(Target::aarch64_linux());
        auto arm64_lir = arm64_isel.lower(*fn_det);
        REQUIRE(arm64_lir != nullptr);
        for (const auto& lbb : arm64_lir->blocks) {
            for (const auto& linst : lbb->instructions) {
                CHECK(!is_lir_fma(linst->opcode));
            }
        }
    }
}

TEST_CASE("Determinism #156 - Tier-0 vs Tier-1 vs Tier-2 1,000-Frame Lockstep State-Hash Differential Verification") {
    constexpr size_t kStateDoubles = 16;
    constexpr size_t kLoopCount = 8;
    constexpr int kNumFrames = 1000;
    const double kDt = 0.016666666666666666;

    auto init_state = [](std::vector<double>& s) {
        s.assign(kStateDoubles, 0.0);
        for (size_t i = 0; i < kLoopCount; ++i) {
            s[i] = 0.125 * static_cast<double>(i + 1) + 1e-7 * static_cast<double>((i * 7) % 5);
        }
        s[8] = 0.5;
    };

    // 1. With deterministic_fp = true:
    //    Tier-0 (Interpreter), Tier-0b (FastInterpreter), Tier-1 (BaselineJitCompiler),
    //    and Tier-2 (production_pass_pipeline + JitExecutionEngine) must produce
    //    bit-identical 64-bit state hashes on every single frame across all 1,000 frames!
    auto mod_tier0 = build_deterministic_sim_module(true);
    auto mod_tier0_fast = build_deterministic_sim_module(true);
    auto mod_tier1 = build_deterministic_sim_module(true);
    auto mod_tier2 = build_deterministic_sim_module(true);

    PassPipelineOptions opt_opts = production_pass_pipeline_options();
    (void)run_pass_pipeline(*mod_tier2, opt_opts);
    CHECK(verify_module(*mod_tier2));

    Interpreter interp;
    interp.set_module(mod_tier0.get());

    FastInterpreter fast_interp;
    fast_interp.set_module(mod_tier0_fast.get());

    codegen::BaselineJitCompiler baseline_jit;
    auto t1_compiled = baseline_jit.compile(*mod_tier1->get_function("step_frame"));
    REQUIRE(t1_compiled.is_valid());
    using StepFnPtr = double (*)(double*, int64_t, double);
    auto tier1_fn = t1_compiled.get_function_ptr<StepFnPtr>();
    REQUIRE(tier1_fn != nullptr);

    codegen::JitExecutionEngine tier2_jit;
    REQUIRE(tier2_jit.compile_and_load(*mod_tier2));
    auto tier2_fn = tier2_jit.get_function_ptr<StepFnPtr>("step_frame");
    REQUIRE(tier2_fn != nullptr);

    std::vector<double> state_t0, state_t0_fast, state_t1, state_t2;
    init_state(state_t0);
    init_state(state_t0_fast);
    init_state(state_t1);
    init_state(state_t2);

    uint64_t rolling_hash_t0 = 0;
    uint64_t rolling_hash_t2 = 0;

    for (int frame = 0; frame < kNumFrames; ++frame) {
        std::vector<RuntimeValue> args_t0 = {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(state_t0.data())),
            RuntimeValue::from_i64(static_cast<int64_t>(kLoopCount)),
            RuntimeValue::from_f64(kDt)
        };
        std::vector<RuntimeValue> args_t0_fast = {
            RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(state_t0_fast.data())),
            RuntimeValue::from_i64(static_cast<int64_t>(kLoopCount)),
            RuntimeValue::from_f64(kDt)
        };

        RuntimeValue r0 = interp.run("step_frame", args_t0);
        RuntimeValue r0_fast = fast_interp.run("step_frame", args_t0_fast);
        double r1 = tier1_fn(state_t1.data(), static_cast<int64_t>(kLoopCount), kDt);
        double r2 = tier2_fn(state_t2.data(), static_cast<int64_t>(kLoopCount), kDt);

        CHECK_EQ(f64_raw_bits(r0.as_f64()), f64_raw_bits(r2));
        CHECK_EQ(f64_raw_bits(r0_fast.as_f64()), f64_raw_bits(r2));
        CHECK_EQ(f64_raw_bits(r1), f64_raw_bits(r2));

        uint64_t h0 = math::hash_f64_state(state_t0.data(), state_t0.size());
        uint64_t h0_fast = math::hash_f64_state(state_t0_fast.data(), state_t0_fast.size());
        uint64_t h1 = math::hash_f64_state(state_t1.data(), state_t1.size());
        uint64_t h2 = math::hash_f64_state(state_t2.data(), state_t2.size());

        REQUIRE_EQ(h0, h2);
        REQUIRE_EQ(h0_fast, h2);
        REQUIRE_EQ(h1, h2);

        rolling_hash_t0 = math::hash_f64_state(state_t0.data(), state_t0.size(), rolling_hash_t0);
        rolling_hash_t2 = math::hash_f64_state(state_t2.data(), state_t2.size(), rolling_hash_t2);
        REQUIRE_EQ(rolling_hash_t0, rolling_hash_t2);

        // Cross-machine golden state-hash checkpoints at frames 1, 10, 100, 500, and 1000:
        if (frame == 0) {
            CHECK_EQ(h2, 0xac76118ffe36f8eeULL);
            CHECK_EQ(f64_raw_bits(r2), 0x4013efb07fcd96c2ULL);
        } else if (frame == 9) {
            CHECK_EQ(h2, 0xc4f18a340467cb31ULL);
            CHECK_EQ(rolling_hash_t2, 0x6bf263ead634651cULL);
            CHECK_EQ(f64_raw_bits(r2), 0x40456ab60524b086ULL);
        } else if (frame == 99) {
            CHECK_EQ(h2, 0x87160d4b376384a9ULL);
            CHECK_EQ(rolling_hash_t2, 0x347b7c99a4272397ULL);
            CHECK_EQ(f64_raw_bits(r2), 0x407a4109fdc79123ULL);
        } else if (frame == 499) {
            CHECK_EQ(h2, 0xf22834e15b5b0759ULL);
            CHECK_EQ(rolling_hash_t2, 0xa7063a6e0a85014aULL);
            CHECK_EQ(f64_raw_bits(r2), 0x40a05d77c45cc2d9ULL);
        } else if (frame == 999) {
            CHECK_EQ(h2, 0xceda671d02fd55d8ULL);
            CHECK_EQ(rolling_hash_t2, 0xbb0addf237e24647ULL);
            CHECK_EQ(f64_raw_bits(r2), 0x40b05dfafd7e4684ULL);
        }
    }

    static constexpr uint64_t kExpected1000FrameRollingHash = 0xbb0addf237e24647ULL;
    CHECK_EQ(rolling_hash_t0, kExpected1000FrameRollingHash);
    CHECK_EQ(rolling_hash_t2, kExpected1000FrameRollingHash);

    // 2. Contrast: When deterministic_fp is FALSE, Tier-2 FMA contraction & vector reduction
    //    reassociation produce single-rounding / reordered-sum drift relative to Tier-0 Interpreter.
    auto mod_nondet_t2 = build_deterministic_sim_module(false);
    (void)run_pass_pipeline(*mod_nondet_t2, opt_opts);

    codegen::JitExecutionEngine nondet_jit;
    REQUIRE(nondet_jit.compile_and_load(*mod_nondet_t2));
    auto nondet_t2_fn = nondet_jit.get_function_ptr<StepFnPtr>("step_frame");
    REQUIRE(nondet_t2_fn != nullptr);

    std::vector<double> state_nondet_t2;
    init_state(state_nondet_t2);
    uint64_t rolling_hash_nondet_t2 = 0;
    for (int frame = 0; frame < kNumFrames; ++frame) {
        (void)nondet_t2_fn(state_nondet_t2.data(), static_cast<int64_t>(kLoopCount), kDt);
        rolling_hash_nondet_t2 = math::hash_f64_state(
            state_nondet_t2.data(), state_nondet_t2.size(), rolling_hash_nondet_t2);
    }
    // Without deterministic_fp, FMA single-rounding causes the 1,000-frame state hash to diverge from Tier 0!
    CHECK(rolling_hash_nondet_t2 != rolling_hash_t0);
}
