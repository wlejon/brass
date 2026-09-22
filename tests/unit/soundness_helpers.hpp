#pragma once

// Shared helpers for the optimizer soundness tests: parse a module, count
// what a pass left behind, and run code before and after a transform on the
// interpreter and the JIT to compare answers.

#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/bounds_check_elim.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace soundness {

using namespace brass;

inline std::unique_ptr<Module> parse(std::string_view text) {
    DiagnosticReporter diag;
    auto mod = parse_module(text, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    return mod;
}

inline Function* find_fn(Module& mod, std::string_view name) {
    Function* fn = mod.get_function(name);
    REQUIRE(fn != nullptr);
    return fn;
}

inline size_t count_opcode(const Function& fn, Opcode op) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == op) ++n;
        }
    }
    return n;
}

inline size_t count_calls(const Function& fn, std::string_view symbol) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call && inst->symbol() == symbol) ++n;
        }
    }
    return n;
}

inline bool verifies(const Module& mod) {
    DiagnosticReporter diag;
    const bool ok = verify_module(mod, &diag);
    if (!ok) {
        std::cerr << diag.format_all() << "\n";
        print_module(mod, std::cerr);
    }
    return ok;
}

// Runs `fn_name` on the interpreter, applies `transform` to a fresh copy of
// the module, then runs the transformed code on the interpreter and the JIT.
// All three must agree on every argument set.
inline void check_transform_preserves(std::string_view mir, std::string_view fn_name,
                                      const std::function<void(Module&)>& transform,
                                      const std::vector<std::vector<RuntimeValue>>& arg_sets) {
    auto original = parse(mir);
    auto optimized = parse(mir);
    transform(*optimized);
    REQUIRE(verifies(*optimized));

    codegen::JitExecutionEngine jit(Target::host());
    REQUIRE(jit.compile_and_load(*optimized));

    for (const auto& args : arg_sets) {
        Interpreter before_interp;
        const uint64_t before = before_interp.run(*original, fn_name, args).raw_bits();
        Interpreter after_interp;
        const uint64_t after = after_interp.run(*optimized, fn_name, args).raw_bits();
        const uint64_t after_jit = jit.invoke(fn_name, args).raw_bits();
        if (before != after || before != after_jit) {
            std::cerr << "MISMATCH in " << fn_name << ": before=" << before << " after=" << after
                      << " jit=" << after_jit << "\n";
            print_module(*optimized, std::cerr);
        }
        CHECK_EQ(before, after);
        CHECK_EQ(before, after_jit);
    }
}

inline RuntimeValue i64(int64_t v) { return RuntimeValue::from_i64(v); }
inline RuntimeValue i32(int32_t v) { return RuntimeValue::from_i32(v); }

inline void run_loop_pipeline(Module& mod) {
    LoopOptOptions opts;
    opts.enable_bce = true;
    optimize_module_loops(mod, opts);
}

inline void run_bce(Module& mod) {
    run_bounds_check_elimination(mod);
}

} // namespace soundness
