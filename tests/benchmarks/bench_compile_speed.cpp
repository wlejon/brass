#include "bench_compile_speed.hpp"
#include "bench_utils.hpp"
#include <brass/brass.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <vector>
#include <iostream>
#include <sstream>
#include <cassert>

using namespace brass;
using namespace brass::bench;
using namespace brass::codegen;

namespace {

std::unique_ptr<Module> generate_large_mir_module(size_t num_functions) {
    auto mod = std::make_unique<Module>("bench_large_compile_module");

    for (size_t f = 0; f < num_functions; ++f) {
        std::string fn_name = "func_" + std::to_string(f);
        size_t kind = f % 5;

        Builder b(*mod);

        if (kind == 0) {
            // Kind 0: Iterative loop with arithmetic and block parameters (~40 insts)
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
            Value* next_i = b.build_add(i, b.build_iconst_i64(1));
            b.build_br(loop_hdr, {next_i, t4});

            b.position_at_end(exit_bb);
            b.build_ret(acc);
            fn->rebuild_cfg_predecessors();
        } else if (kind == 1) {
            // Kind 1: Multi-way conditional branches and bitwise chains (~45 insts)
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
            b.build_br(bb_join, {t4});

            b.position_at_end(bb_else);
            Value* e1 = b.build_sub(c, a);
            Value* e2 = b.build_or(e1, d);
            Value* e3 = b.build_clz(e2);
            Value* e4 = b.build_xor(e3, b.build_iconst_i64(0x7F));
            b.build_br(bb_join, {e4});

            b.position_at_end(bb_join);
            Value* res = b.add_block_param(bb_join, Type::i64());
            Value* fin = b.build_add(res, d);
            b.build_ret(fin);
            fn->rebuild_cfg_predecessors();
        } else if (kind == 2) {
            // Kind 2: Memory load and store indexed pipeline (~40 insts)
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
            b.build_store_indexed(Type::i64(), dst, idx, 8, 0, mod_val);
            Value* next_sum = b.build_add(sum, mod_val);
            Value* next_idx = b.build_add(idx, b.build_iconst_i64(1));
            b.build_br(loop_hdr, {next_idx, next_sum});

            b.position_at_end(exit_bb);
            b.build_ret(sum);
            fn->rebuild_cfg_predecessors();
        } else if (kind == 3) {
            // Kind 3: Float arithmetic and conversions (~40 insts)
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
            Value* f6 = b.build_sitofp_f64_i64(i2);
            Value* f_res = b.build_add(f5, f6);
            b.build_ret(f_res);
            fn->rebuild_cfg_predecessors();
        } else {
            // Kind 4: Straight-line arithmetic & bitwise expression DAG (~50 insts)
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
            b.build_ret(v12);
            fn->rebuild_cfg_predecessors();
        }
    }

    return mod;
}

} // namespace

namespace brass::bench {

void run_compile_speed_benchmark(std::vector<BenchmarkResult>& results) {
    (void)results;
    Stopwatch sw;

    // Generate large module with 500 functions (~10,000+ instructions)
    size_t target_functions = 500;
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
        std::cerr << "Initial module verification failed:\n" << init_diag.format_all() << "\n";
        assert(false);
    }

    // Serialize to text
    std::string mir_text = to_string(*initial_mod);
    size_t mir_bytes = mir_text.size();

    // 1. Benchmark: Parse MIR Text
    DiagnosticReporter parse_diag;
    sw.start();
    auto parsed_mod = parse_module(mir_text, &parse_diag);
    double parse_ms = sw.stop_ms();
    if (!parsed_mod || parse_diag.has_errors()) {
        std::cerr << "Parse MIR failed:\n" << parse_diag.format_all() << "\n";
        assert(false);
    }

    // 2. Benchmark: Verify MIR Module
    DiagnosticReporter ver_diag;
    sw.start();
    bool verify_ok = verify_module(*parsed_mod, &ver_diag);
    double verify_ms = sw.stop_ms();
    if (!verify_ok) {
        std::cerr << "FATAL: Parsed module verification failed:\n" << ver_diag.format_all() << "\n";
        std::abort();
    }

    // 3. Benchmark: ISEL, Linear Scan Register Allocation, Peephole, Object Generation, and JIT Load
    sw.start();
    JitExecutionEngine jit;
    bool jit_ok = jit.compile_and_load(*parsed_mod);
    double codegen_ms = sw.stop_ms();
    if (!jit_ok) {
        std::cerr << "FATAL: JIT compilation failed in compile speed benchmark!\n";
        std::abort();
    }

    double total_ms = parse_ms + verify_ms + codegen_ms;
    bool passes_compile_bar = (total_ms < 2000.0) && verify_ok && jit_ok;

    // Approximate emitted machine code size
    size_t machine_bytes = total_instructions * 4; // ~4-5 bytes per x86_64 inst average

    BenchmarkReporter::print_compile_speed(
        target_functions,
        total_instructions,
        mir_bytes,
        machine_bytes,
        parse_ms,
        verify_ms,
        codegen_ms,
        total_ms,
        passes_compile_bar
    );

    if (!passes_compile_bar) {
        std::cerr << "FATAL: Compile speed benchmark failed to meet bar!\n";
        std::abort();
    }
}

} // namespace brass::bench
