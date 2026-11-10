// The write barrier's inline young-target filter (Module::young_range,
// write_barrier_lowering.hpp): the attribute round-trips through the text
// form, Tier 2's expansion gives each i64 barrier a test and a cold call, and
// in every tier a host's brass_gc_write_barrier is called exactly for the
// targets outside the young range (Tier 0 calls it for every barrier).
#include "test_framework.hpp"
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/write_barrier_lowering.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/vm/fast_interpreter.hpp>

#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>

using namespace brass;

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

// The pinned TLS block: the young range's base at 16, its span at 24.
struct WbState {
    alignas(16) uint64_t tls[4] = {};
    uint64_t young[4] = {};
    uint64_t old[4] = {};
    int calls = 0;
    uint64_t last_obj = 0;
    uint64_t last_val = 0;
};
thread_local WbState t_wb;

extern "C" void wbl_barrier(uint64_t obj, uint64_t val) {
    ++t_wb.calls;
    t_wb.last_obj = obj;
    t_wb.last_val = val;
}

int64_t addr(const void* p) { return static_cast<int64_t>(reinterpret_cast<uintptr_t>(p)); }

// wbl_store(tls, obj, val): stores val at obj and barriers it; obj may carry
// a tag above its 48 address bits, which the store strips.
const char* kModule = R"(module @wbl
attributes pinned_tls_register young_range 16 24

func @wbl_store(%tls: i64, %obj: i64, %val: i64) -> i64 {
entry:
  pinned_tls_write %tls
  %mask = iconst.i64 281474976710655
  %a = and.i64 %obj, %mask
  store.i64 %a, %val
  write_barrier %obj, %val
  %one = iconst.i64 1
  %r = add.i64 %val, %one
  ret %r
}
)";

std::unique_ptr<Module> parse_ok(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << vdiag.format_all() << "\n";
    REQUIRE(ok);
    return mod;
}

using Fn3 = int64_t (*)(int64_t, int64_t, int64_t);

void reset() {
    t_wb = WbState{};
    t_wb.tls[2] = static_cast<uint64_t>(addr(&t_wb.young[0]));
    t_wb.tls[3] = sizeof(t_wb.young);
}

// Runs the three targets through `call` and checks which reached the barrier:
// a young one (none), an old one, and a young one under a tag (none).
template <typename Call>
void check_tier(const char* name, bool filters, Call&& call) {
    reset();
    const int64_t tls = addr(&t_wb.tls[0]);
    CHECK_EQ(call(tls, addr(&t_wb.young[1]), 41), int64_t{42});
    CHECK_EQ(t_wb.young[1], uint64_t{41});
    const int young_calls = t_wb.calls;
    CHECK_EQ(call(tls, addr(&t_wb.old[1]), 7), int64_t{8});
    CHECK_EQ(t_wb.old[1], uint64_t{7});
    const int old_calls = t_wb.calls - young_calls;
    CHECK_EQ(t_wb.last_obj, static_cast<uint64_t>(addr(&t_wb.old[1])));
    CHECK_EQ(t_wb.last_val, uint64_t{7});
    const int64_t tagged = static_cast<int64_t>(static_cast<uint64_t>(addr(&t_wb.young[2])) | (uint64_t{0xFFF1} << 48));
    CHECK_EQ(call(tls, tagged, 9), int64_t{10});
    CHECK_EQ(t_wb.young[2], uint64_t{9});
    const int tagged_calls = t_wb.calls - young_calls - old_calls;
    if (old_calls != 1) std::cerr << name << ": " << old_calls << " barrier calls for an old target\n";
    CHECK_EQ(old_calls, 1);
    CHECK_EQ(young_calls, filters ? 0 : 1);
    CHECK_EQ(tagged_calls, filters ? 0 : 1);
}

} // namespace

TEST_CASE("Write barrier filter - the attribute prints and parses back") {
    auto mod = parse_ok(kModule);
    REQUIRE(static_cast<bool>(mod->young_range()));
    CHECK_EQ(mod->young_range().base_offset, 16);
    CHECK_EQ(mod->young_range().span_offset, 24);
    std::ostringstream os;
    print_module(*mod, os);
    CHECK(os.str().find("young_range 16 24") != std::string::npos);
    auto again = parse_ok(os.str().c_str());
    CHECK_EQ(again->young_range().base_offset, 16);
    CHECK_EQ(again->young_range().span_offset, 24);
}

TEST_CASE("Write barrier filter - tier 2's expansion tests the target and calls from a cold block") {
    auto mod = parse_ok(kModule);
    Module copy("copy");
    copy.set_pinned_tls_register(true);
    copy.copy_declarations_from(*mod);
    Function* fn = clone_function(*mod->get_function("wbl_store"), copy);
    REQUIRE(needs_write_barrier_expansion(*fn));
    const size_t before = fn->blocks().size();
    CHECK_EQ(expand_write_barriers(*fn), size_t{1});
    CHECK_EQ(fn->blocks().size(), before + 2);
    CHECK(!needs_write_barrier_expansion(*fn));
    size_t cold = 0, calls = 0;
    for (const BasicBlock* bb : fn->blocks()) {
        if (bb->is_cold()) ++cold;
        for (const Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::call && inst->symbol() == "brass_gc_write_barrier") ++calls;
        }
    }
    CHECK_EQ(cold, size_t{1});
    CHECK_EQ(calls, size_t{1});
    DiagnosticReporter diag;
    const bool ok = verify_function(*fn, &diag);
    if (!ok) std::cerr << diag.format_all() << "\n";
    CHECK(ok);

    // Without the attribute nothing is expanded.
    Module plain("plain");
    plain.set_pinned_tls_register(true);
    Function* fn2 = clone_function(*mod->get_function("wbl_store"), plain);
    CHECK(!needs_write_barrier_expansion(*fn2));
    CHECK_EQ(expand_write_barriers(*fn2), size_t{0});
}

TEST_CASE("Write barrier filter - every tier calls the host's barrier for an old target only") {
    auto mod = parse_ok(kModule);
    {
        // Tier 0 has no filter: the host's barrier sees every barrier.
        FastInterpreter fi;
        fi.set_module(mod.get());
        fi.register_external_symbol("brass_gc_write_barrier", reinterpret_cast<void*>(&wbl_barrier));
        check_tier("tier 0", false, [&](int64_t a, int64_t b, int64_t c) {
            return fi.run(*mod->get_function("wbl_store"),
                          {RuntimeValue::from_i64(a), RuntimeValue::from_i64(b), RuntimeValue::from_i64(c)})
                .as_i64();
        });
    }
    {
        codegen::BaselineJitCompiler compiler;
        compiler.register_external_symbol("brass_gc_write_barrier", reinterpret_cast<void*>(&wbl_barrier));
        auto fns = compiler.compile_module(*mod);
        Fn3 p = nullptr;
        for (const auto& f : fns) {
            if (f.name() == "wbl_store") p = f.get_function_ptr<Fn3>();
        }
        REQUIRE(p != nullptr);
        check_tier("tier 1", true, p);
    }
    {
        codegen::JitExecutionEngine jit;
        jit.register_external_symbol("brass_gc_write_barrier", reinterpret_cast<void*>(&wbl_barrier));
        REQUIRE(jit.compile_and_load(*mod));
        Fn3 p = jit.get_function_ptr<Fn3>("wbl_store");
        REQUIRE(p != nullptr);
        check_tier("tier 2", true, p);
    }
    runtime::FunctionDispatchTable::instance().forget_module(*mod);
}

#endif
