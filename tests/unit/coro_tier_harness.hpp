#pragma once

// Runs a coroutine program with its body and its driver each placed in a
// chosen tier of one owned program: Tier 0 (Interpreter or FastInterpreter),
// Tier 1 (baseline) or Tier 2 (optimized). Tiering is manual (no automatic
// tier-up) unless a test sets its own config.

#include "test_framework.hpp"
#include "gc_test_heap.hpp"
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/coro_transform.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace brass::coro_test {

enum class Tier { Interp, Fast, Base, Opt };
inline constexpr Tier kTiers[] = {Tier::Interp, Tier::Fast, Tier::Base, Tier::Opt};

inline const char* tier_name(Tier t) {
    switch (t) {
        case Tier::Interp: return "interp";
        case Tier::Fast: return "fast";
        case Tier::Base: return "tier1";
        case Tier::Opt: return "tier2";
    }
    return "?";
}

// Parsed, verified, NOT lowered: every entry point lowers it (the pipeline
// lowers what it executes; here `lower_coroutines` stands in for it).
inline std::unique_ptr<Module> parse_program(std::string_view src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::printf("%s", diag.format_all().c_str());
    REQUIRE(mod != nullptr);
    REQUIRE(verify_module(*mod, &diag));
    lower_coroutines(*mod);
    DiagnosticReporter post;
    const bool ok = verify_module(*mod, &post);
    if (!ok) std::printf("%s", post.format_all().c_str());
    REQUIRE(ok);
    return mod;
}

inline runtime::TieringConfig manual_tiering() {
    runtime::TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(false);
    return cfg;
}

// One program over `mod`: every function has its handle.
struct Program {
    runtime::FunctionDispatchTable table;
    Module& mod;

    explicit Program(Module& m, const runtime::TieringConfig& cfg = manual_tiering()) : mod(m) {
        table.pipeline().initialize(cfg);
        table.tiering().set_active_module(&mod);
        for (const Function* fn : mod.functions()) {
            if (fn) table.get_or_create(fn->name(), fn);
        }
    }
    ~Program() { table.tiering().set_active_module(nullptr); }

    // Installs `fn` natively at `tier` (nothing for a Tier-0 placement).
    void place(std::string_view fn, Tier tier) {
        const Function* f = mod.get_function(fn);
        REQUIRE(f != nullptr);
        runtime::FunctionHandle* h = table.get_or_create(fn, f);
        if (tier == Tier::Base) {
            const bool ok = table.pipeline().compile_and_install_tier1(fn, f);
            if (!ok) std::printf("tier 1 rejected %s\n", std::string(fn).c_str());
            REQUIRE(ok);
            REQUIRE(h->native_entry() != nullptr);
        } else if (tier == Tier::Opt) {
            table.tiering().get_feedback(fn).set_deopt_threshold(1000000);
            runtime::CodeInstaller installer(table);
            const runtime::CodeInstallResult res = installer.install_tier2(*h, mod, fn);
            if (!res.success) std::printf("install_tier2 %s: %s\n", std::string(fn).c_str(), res.error_message.c_str());
            REQUIRE(res.success);
        }
    }

    // Calls `fn` in `tier` (Tier 0: an interpreter of this program; native:
    // its installed entry).
    RuntimeValue call(std::string_view fn, Tier tier, const std::vector<RuntimeValue>& args) {
        runtime::ProgramScope scope(table);
        const Function* f = mod.get_function(fn);
        REQUIRE(f != nullptr);
        switch (tier) {
            case Tier::Interp: {
                Interpreter in;
                in.set_dispatch_table(&table);
                in.set_module(&mod);
                return in.run(*f, args);
            }
            case Tier::Fast: {
                FastInterpreter fi;
                fi.set_dispatch_table(&table);
                fi.set_module(&mod);
                return fi.run(*f, args);
            }
            default:
                return table.find(fn)->call_native(args);
        }
    }
};

} // namespace brass::coro_test
