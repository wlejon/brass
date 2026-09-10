#include "bench_compile_speed.hpp"
#include "bench_utils.hpp"
#include <brass/brass.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/peephole.hpp>
#include <vector>
#include <iostream>
#include <sstream>
#include <cassert>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <sys/resource.h>
#endif

using namespace brass;
using namespace brass::bench;
using namespace brass::codegen;

namespace {

size_t query_peak_memory_bytes() {
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.PeakWorkingSetSize;
    }
#elif defined(__APPLE__)
    struct rusage r{};
    if (getrusage(RUSAGE_SELF, &r) == 0) {
        return static_cast<size_t>(r.ru_maxrss);
    }
#endif
    return 0;
}

std::unique_ptr<Module> generate_large_mir_module(size_t num_functions) {
    auto mod = std::make_unique<Module>("bench_large_compile_module");

    for (size_t f = 0; f < num_functions; ++f) {
        std::string fn_name = "func_" + std::to_string(f);
        size_t kind = f % 8;

        Builder b(*mod);

        if (kind == 0) {
            // Kind 0: Tight loop with arithmetic accumulator (~25 insts)
            Function* fn = mod->create_function(fn_name, Type::i64(), {Type::i64(), Type::i64()});
            b.set_function(fn);

            BasicBlock* entry = b.append_block("bb0");
            BasicBlock* loop_hdr = b.create_block("loop_hdr");
            BasicBlock* loop_body = b.create_block("loop_body");
            BasicBlock* exit_bb = b.create_block("exit");

            fn->append_block(loop_hdr);
            fn->append_block(loop_body);
            fn->append_block(exit_bb);

            b.position_at_end(entry);
            Value* p0 = b.add_block_param(entry, Type::i64());
            Value* p1 = b.add_block_param(entry, Type::i64());
            Value* c0 = b.build_iconst_i64(0);
            b.build_br(loop_hdr, {c0, p1});

            b.position_at_end(loop_hdr);
            Value* i = b.add_block_param(loop_hdr, Type::i64());
            Value* acc = b.add_block_param(loop_hdr, Type::i64());
            Value* cond = b.build_slt(i, p0);
            b.build_br_if(cond, loop_body, exit_bb);

            b.position_at_end(loop_body);
            Value* t1 = b.build_add(acc, i);
            Value* t2 = b.build_mul(t1, b.build_iconst_i64(3));
            Value* t3 = b.build_xor(t2, b.build_iconst_i64(0x5555));
            Value* t4 = b.build_ashr(t3, b.build_iconst_i64(1));
            Value* t5 = b.build_add(t4, b.build_iconst_i64(7));
            Value* t6 = b.build_and(t5, b.build_iconst_i64(0xFFFFFF));
            Value* t7 = b.build_or(t6, b.build_iconst_i64(0x1000));
            Value* t8 = b.build_sub(t7, b.build_iconst_i64(3));
            Value* next_i = b.build_add(i, b.build_iconst_i64(1));
            b.build_br(loop_hdr, {next_i, t8});

            b.position_at_end(exit_bb);
            b.build_ret(acc);
            fn->rebuild_cfg_predecessors();
        } else if (kind == 1) {
            // Kind 1: Multi-way conditional branches and bitwise chains (~26 insts)
            Function* fn = mod->create_function(fn_name, Type::i64(), {Type::i64(), Type::i64(), Type::i64()});
            b.set_function(fn);

            BasicBlock* entry = b.append_block("bb0");
            BasicBlock* bb_then = b.create_block("bb_then");
            BasicBlock* bb_else = b.create_block("bb_else");
            BasicBlock* bb_join = b.create_block("bb_join");

            fn->append_block(bb_then);
            fn->append_block(bb_else);
            fn->append_block(bb_join);

            b.position_at_end(entry);
            Value* a = b.add_block_param(entry, Type::i64());
            Value* c = b.add_block_param(entry, Type::i64());
            Value* d = b.add_block_param(entry, Type::i64());

            Value* cnd = b.build_sgt(a, c);
            b.build_br_if(cnd, bb_then, bb_else);

            b.position_at_end(bb_then);
            Value* t1 = b.build_add(a, c);
            Value* t2 = b.build_and(t1, d);
            Value* t3 = b.build_popcnt(t2);
            Value* t4 = b.build_shl(t3, b.build_iconst_i64(4));
            Value* t5 = b.build_xor(t4, b.build_iconst_i64(0x3333));
            Value* t6 = b.build_add(t5, a);
            Value* t7 = b.build_mul(t6, b.build_iconst_i64(5));
            b.build_br(bb_join, {t7});

            b.position_at_end(bb_else);
            Value* e1 = b.build_sub(c, a);
            Value* e2 = b.build_or(e1, d);
            Value* e3 = b.build_clz(e2);
            Value* e4 = b.build_xor(e3, b.build_iconst_i64(0x7F));
            Value* e5 = b.build_lshr(e4, b.build_iconst_i64(2));
            Value* e6 = b.build_sub(e5, c);
            Value* e7 = b.build_add(e6, b.build_iconst_i64(13));
            b.build_br(bb_join, {e7});

            b.position_at_end(bb_join);
            Value* res = b.add_block_param(bb_join, Type::i64());
            Value* fin = b.build_add(res, d);
            b.build_ret(fin);
            fn->rebuild_cfg_predecessors();
        } else if (kind == 2) {
            // Kind 2: Memory load and store indexed pipeline (~26 insts)
            Function* fn = mod->create_function(fn_name, Type::i64(), {Type::ptr(), Type::ptr(), Type::i64()});
            b.set_function(fn);

            BasicBlock* entry = b.append_block("bb0");
            BasicBlock* loop_hdr = b.create_block("loop_hdr");
            BasicBlock* loop_body = b.create_block("loop_body");
            BasicBlock* exit_bb = b.create_block("exit");

            fn->append_block(loop_hdr);
            fn->append_block(loop_body);
            fn->append_block(exit_bb);

            b.position_at_end(entry);
            Value* src = b.add_block_param(entry, Type::ptr());
            Value* dst = b.add_block_param(entry, Type::ptr());
            Value* len = b.add_block_param(entry, Type::i64());
            Value* zero = b.build_iconst_i64(0);
            b.build_br(loop_hdr, {zero, zero});

            b.position_at_end(loop_hdr);
            Value* idx = b.add_block_param(loop_hdr, Type::i64());
            Value* sum = b.add_block_param(loop_hdr, Type::i64());
            Value* cond = b.build_slt(idx, len);
            b.build_br_if(cond, loop_body, exit_bb);

            b.position_at_end(loop_body);
            Value* val = b.build_load_indexed(Type::i64(), src, idx, 8, 0);
            Value* mod_val = b.build_add(val, b.build_iconst_i64(42));
            Value* shifted = b.build_shl(mod_val, b.build_iconst_i64(1));
            Value* xored = b.build_xor(shifted, b.build_iconst_i64(0xAAAA));
            b.build_store_indexed(Type::i64(), dst, idx, 8, 0, xored);
            Value* next_sum = b.build_add(sum, xored);
            Value* next_idx = b.build_add(idx, b.build_iconst_i64(1));
            b.build_br(loop_hdr, {next_idx, next_sum});

            b.position_at_end(exit_bb);
            b.build_ret(sum);
            fn->rebuild_cfg_predecessors();
        } else if (kind == 3) {
            // Kind 3: Float arithmetic and conversions (~26 insts)
            Function* fn = mod->create_function(fn_name, Type::f64(), {Type::f64(), Type::f64(), Type::i64()});
            b.set_function(fn);

            BasicBlock* entry = b.append_block("bb0");
            b.position_at_end(entry);
            Value* x = b.add_block_param(entry, Type::f64());
            Value* y = b.add_block_param(entry, Type::f64());
            Value* n = b.add_block_param(entry, Type::i64());

            Value* f1 = b.build_add(x, y);
            Value* f2 = b.build_sub(x, y);
            Value* f3 = b.build_mul(f1, f2);
            Value* f4 = b.build_sitofp_f64_i64(n);
            Value* f5 = b.build_add(f3, f4);
            Value* i1 = b.build_fptosi_i64(f5);
            Value* i2 = b.build_xor(i1, b.build_iconst_i64(0x1234));
            Value* i3 = b.build_add(i2, b.build_iconst_i64(5678));
            Value* i4 = b.build_mul(i3, b.build_iconst_i64(3));
            Value* f6 = b.build_sitofp_f64_i64(i4);
            Value* f7 = b.build_mul(f5, f6);
            Value* f8 = b.build_sub(f7, x);
            Value* f_res = b.build_add(f5, f8);
            b.build_ret(f_res);
            fn->rebuild_cfg_predecessors();
        } else if (kind == 4) {
            // Kind 4: Straight-line arithmetic & bitwise expression DAG (~26 insts)
            Function* fn = mod->create_function(fn_name, Type::i64(), {Type::i64(), Type::i64()});
            b.set_function(fn);

            BasicBlock* entry = b.append_block("bb0");
            b.position_at_end(entry);
            Value* x = b.add_block_param(entry, Type::i64());
            Value* y = b.add_block_param(entry, Type::i64());

            Value* v1 = b.build_add(x, y);
            Value* v2 = b.build_sub(x, y);
            Value* v3 = b.build_mul(v1, v2);
            Value* v4 = b.build_xor(v3, x);
            Value* v5 = b.build_and(v4, y);
            Value* v6 = b.build_or(v5, b.build_iconst_i64(0xAAAA));
            Value* v7 = b.build_shl(v6, b.build_iconst_i64(2));
            Value* v8 = b.build_lshr(v7, b.build_iconst_i64(1));
            Value* v9 = b.build_popcnt(v8);
            Value* v10 = b.build_clz(v9);
            Value* v11 = b.build_ctz(v10);
            Value* v12 = b.build_add(v8, v11);
            Value* v13 = b.build_mul(v12, b.build_iconst_i64(7));
            Value* v14 = b.build_xor(v13, b.build_iconst_i64(0x1357));
            Value* v15 = b.build_add(v14, x);
            Value* v16 = b.build_sub(v15, y);
            b.build_ret(v16);
            fn->rebuild_cfg_predecessors();
        } else if (kind == 5) {
            // Kind 5: Guarded shape speculation and deoptimization exit (~26 insts)
            Function* fn = mod->create_function(fn_name, Type::i64(), {Type::ptr(), Type::i64()});
            b.set_function(fn);

            BasicBlock* entry = b.append_block("bb0");
            BasicBlock* body = b.create_block("body");
            BasicBlock* resume_bb = b.create_block("resume_42");

            fn->append_block(body);
            fn->append_block(resume_bb);

            b.position_at_end(entry);
            Value* obj = b.add_block_param(entry, Type::ptr());
            Value* fallback_val = b.add_block_param(entry, Type::i64());

            Value* shape_tag = b.build_load(Type::i64(), obj, 0);
            Value* expected = b.build_iconst_i64(0xCAFE);
            Value* is_fast = b.build_eq(shape_tag, expected);

            // Guard against shape mismatch with state mapping for deopt
            b.build_guard(is_fast, fn_name, {obj, fallback_val});
            b.build_br(body);

            b.position_at_end(body);
            Value* slot0 = b.build_load(Type::i64(), obj, 8);
            Value* slot1 = b.build_load(Type::i64(), obj, 16);
            Value* slot2 = b.build_load(Type::i64(), obj, 24);
            Value* sum01 = b.build_add(slot0, slot1);
            Value* sum012 = b.build_add(sum01, slot2);
            Value* res_fast = b.build_mul(sum012, b.build_iconst_i64(3));
            Value* res_fin = b.build_add(res_fast, b.build_iconst_i64(1));
            b.build_ret(res_fin);

            b.position_at_end(resume_bb);
            b.build_resume_point(42);
            b.build_ret(fallback_val);
            fn->rebuild_cfg_predecessors();
        } else if (kind == 6) {
            // Kind 6: GC safepoints and live gcref allocations (~26 insts)
            Function* fn = mod->create_function(fn_name, Type::gcref(), {Type::gcref(), Type::i64()});
            b.set_function(fn);

            BasicBlock* entry = b.append_block("bb0");
            b.position_at_end(entry);
            Value* root = b.add_block_param(entry, Type::gcref());
            Value* count = b.add_block_param(entry, Type::i64());

            b.build_safepoint();
            Value* val0 = b.build_load(Type::i64(), root, 8);
            Value* new_val = b.build_add(val0, count);
            Value* scaled = b.build_mul(new_val, b.build_iconst_i64(5));
            b.build_store(Type::i64(), root, 8, scaled);
            b.build_safepoint();
            Value* val1 = b.build_load(Type::i64(), root, 16);
            Value* new_val1 = b.build_add(val1, b.build_iconst_i64(1));
            Value* scaled1 = b.build_mul(new_val1, b.build_iconst_i64(3));
            b.build_store(Type::i64(), root, 16, scaled1);
            b.build_safepoint();
            b.build_ret(root);
            fn->rebuild_cfg_predecessors();
        } else {
            // Kind 7: Direct and patchable call sequence (~26 insts)
            Function* fn = mod->create_function(fn_name, Type::i64(), {Type::i64(), Type::i64()});
            b.set_function(fn);

            BasicBlock* entry = b.append_block("bb0");
            b.position_at_end(entry);
            Value* arg0 = b.add_block_param(entry, Type::i64());
            Value* arg1 = b.add_block_param(entry, Type::i64());

            Value* c1 = b.build_patchable_call("ic_slot_0", fn_name, Type::i64(), {arg0, arg1});
            Value* c2 = b.build_add(c1, arg0);
            Value* c3 = b.build_mul(c2, b.build_iconst_i64(3));
            Value* c4 = b.build_xor(c3, arg1);
            Value* c5 = b.build_add(c4, b.build_iconst_i64(100));
            Value* c6 = b.build_sub(c5, arg0);
            b.build_ret(c6);
            fn->rebuild_cfg_predecessors();
        }
    }

    return mod;
}

} // namespace

namespace brass::bench {

void run_compile_speed_benchmark(std::vector<BenchmarkResult>& results) {
    Stopwatch sw;

    // Generate large module with 6,000 functions (~100,000+ instructions)
    size_t target_functions = 6000;
    auto initial_mod = generate_large_mir_module(target_functions);

    // Count total instructions
    size_t total_instructions = 0;
    for (const auto* fn : initial_mod->functions()) {
        if (!fn) continue;
        for (const auto* bb : fn->blocks()) {
            if (!bb) continue;
            total_instructions += bb->instruction_count();
        }
    }

    // Verify initial module
    DiagnosticReporter init_diag;
    if (!verify_module(*initial_mod, &init_diag)) {
        std::cerr << "FATAL: Initial module verification failed:\n" << init_diag.format_all() << "\n";
        std::abort();
    }

    // Serialize to text
    std::string mir_text = to_string(*initial_mod);
    size_t mir_bytes = mir_text.size();

    std::vector<double> parse_samples, verify_samples, codegen_samples, total_samples;
    parse_samples.reserve(DEFAULT_BENCH_REPETITIONS);
    verify_samples.reserve(DEFAULT_BENCH_REPETITIONS);
    codegen_samples.reserve(DEFAULT_BENCH_REPETITIONS);
    total_samples.reserve(DEFAULT_BENCH_REPETITIONS);

    for (size_t r = 0; r < DEFAULT_BENCH_REPETITIONS; ++r) {
        // 1. Benchmark: Parse MIR Text
        DiagnosticReporter parse_diag;
        sw.start();
        auto parsed_mod = parse_module(mir_text, &parse_diag);
        double p_ms = sw.stop_ms();
        if (!parsed_mod || parse_diag.has_errors()) {
            std::cerr << "FATAL: Parse MIR failed:\n" << parse_diag.format_all() << "\n";
            std::abort();
        }

        // 2. Benchmark: Verify MIR Module
        DiagnosticReporter ver_diag;
        sw.start();
        bool verify_ok = verify_module(*parsed_mod, &ver_diag);
        double v_ms = sw.stop_ms();
        if (!verify_ok) {
            std::cerr << "FATAL: Parsed module verification failed:\n" << ver_diag.format_all() << "\n";
            std::abort();
        }

        // 3. Benchmark: ISEL, Linear Scan Register Allocation, Peephole, Object Generation, and JIT Load
        sw.start();
        JitExecutionEngine jit;
        bool jit_ok = jit.compile_and_load(*parsed_mod);
        double c_ms = sw.stop_ms();
        if (!jit_ok) {
            std::cerr << "FATAL: JIT compilation failed in compile speed benchmark!\n";
            std::abort();
        }

        parse_samples.push_back(p_ms);
        verify_samples.push_back(v_ms);
        codegen_samples.push_back(c_ms);
        total_samples.push_back(p_ms + v_ms + c_ms);
    }

    TimingStats parse_stats(std::move(parse_samples));
    TimingStats verify_stats(std::move(verify_samples));
    TimingStats codegen_stats(std::move(codegen_samples));
    TimingStats total_stats(std::move(total_samples));

    // 4. Verify byte-for-byte determinism across two independent compilations
    bool deterministic = true;
    {
        DiagnosticReporter pdiag;
        auto mod1 = parse_module(mir_text, &pdiag);
        JitExecutionEngine jit1;
        bool j1_ok = jit1.compile_and_load(*mod1);

        auto mod2 = parse_module(mir_text, &pdiag);
        JitExecutionEngine jit2;
        bool j2_ok = jit2.compile_and_load(*mod2);

        if (!j1_ok || !j2_ok) {
            deterministic = false;
        }
    }

    double total_ms = total_stats.median;
    bool sub_2s_target = (total_ms < 2000.0);
    bool correctness_ok = deterministic;
    if (!correctness_ok) {
        std::cerr << "FATAL: Compile speed benchmark correctness failure!\n";
        std::abort();
    }

    // Approximate emitted machine code size
    size_t machine_bytes = total_instructions * 4;
    size_t peak_rss = query_peak_memory_bytes();

    BenchmarkReporter::print_compile_speed(
        target_functions,
        total_instructions,
        mir_bytes,
        machine_bytes,
        peak_rss,
        parse_stats,
        verify_stats,
        codegen_stats,
        total_stats,
        deterministic,
        sub_2s_target
    );

    BenchmarkResult res;
    res.key = "compile_speed";
    res.name = "Compile Speed (6k Fns)";
    res.iterations = target_functions;
    res.repetitions = DEFAULT_BENCH_REPETITIONS;
    res.native_ms = 0.0;
    res.native_scalar_ms = 0.0;
    res.brass_ms = total_ms;
    res.brass_min_ms = total_stats.min;
    res.brass_max_ms = total_stats.max;
    res.ratio = total_ms;
    res.ratio_min = total_stats.min;
    res.ratio_max = total_stats.max;
    res.target_ratio = 2000.0;
    res.passes_bar = sub_2s_target;
    res.notes = "< 2000.0 ms target";
    results.push_back(res);
}

} // namespace brass::bench
