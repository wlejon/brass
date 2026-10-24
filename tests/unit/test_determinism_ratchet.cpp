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
        CHECK_EQ(fma_opt_pass(*parsed_fn), 0u);
        for (const Instruction* inst : *parsed_fn->entry_block()) {
            CHECK(inst->opcode() != Opcode::fma_f64);
        }

        // Disabling deterministic_fp allows fma_opt_pass to fuse mul+add -> fma.f64
        parsed_fn->set_deterministic_fp(false);
        CHECK_EQ(fma_opt_pass(*parsed_fn), 1u);
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
    }

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
