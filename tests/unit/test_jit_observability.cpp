// JIT code as named functions: the registry follows code lifetime at every
// tier, the symbolizer and native stack walks name JIT frames by function and
// tier, and the perf map, jitdump and GDB JIT interface carry what the knobs
// ask for.

#include "test_framework.hpp"
#include <brass/codegen/baseline_jit.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/debug/jit_code_registry.hpp>
#include <brass/debug/symbolicator.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace brass;

namespace {

const char* kObsModule = R"(
module @jit_obs

extern @obs_probe

func @obs_inner(%0: i64) -> i64 {
bb0:
  %1 = call.i64 @obs_probe(%0)
  %2 = iconst.i64 1
  %3 = add.i64 %1, %2
  ret %3
}

func @obs_outer(%0: i64) -> i64 {
bb0:
  %1 = call.i64 @obs_inner(%0)
  %2 = iconst.i64 2
  %3 = mul.i64 %1, %2
  ret %3
}
)";

const char* kBaselineModule = R"(
module @jit_obs_bl

extern @obs_tier2_inner

func @obs_bl_outer(%0: i64) -> i64 {
bb0:
  %1 = call.i64 @obs_tier2_inner(%0)
  %2 = iconst.i64 3
  %3 = add.i64 %1, %2
  ret %3
}
)";

std::vector<std::string> g_stack;       // the symbolizer's frames at the probe
std::string g_native;                   // format_native_stack at the probe

int64_t obs_probe(int64_t v) {
    g_stack.clear();
    for (const auto& f : Symbolicator().symbolize_current_stack(32).frames) g_stack.push_back(f.function_name);
    g_native = debug::format_native_stack(debug::capture_native_stack(32), false);
    return v + 10;
}

std::unique_ptr<Module> parse_or_fail(const char* src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    REQUIRE(mod != nullptr);
    return mod;
}

std::unique_ptr<codegen::JitExecutionEngine> load_obs(debug::JitTier tier = debug::JitTier::Optimized) {
    static std::unique_ptr<Module> mod = parse_or_fail(kObsModule);
    auto jit = std::make_unique<codegen::JitExecutionEngine>(Target::host());
    jit->set_code_tier(tier);
    jit->register_external_symbol("obs_probe", reinterpret_cast<void*>(&obs_probe));
    REQUIRE(jit->compile_and_load(*mod));
    return jit;
}

[[maybe_unused]] size_t index_of(const std::vector<std::string>& v, const std::string& s) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == s) return i;
    }
    return SIZE_MAX;
}

void set_env(const char* name, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    if (value.empty()) unsetenv(name);
    else setenv(name, value.c_str(), 1);
#endif
}

// Sets the knobs for one test's scope, and restores the defaults after.
struct KnobScope {
    explicit KnobScope(std::initializer_list<const char*> on, const std::string& dir) {
        for (const char* k : on) set_env(k, "1");
        set_env("BRASS_PERF_DIR", dir);
        debug::reload_jit_profiler_config();
    }
    ~KnobScope() {
        for (const char* k : {"BRASS_PERF_MAP", "BRASS_JITDUMP", "BRASS_GDB_JIT", "BRASS_PERF_DIR"}) set_env(k, "");
        debug::reload_jit_profiler_config();
    }
};

std::string temp_dir(const char* leaf) {
    auto dir = std::filesystem::temp_directory_path() / leaf;
    std::filesystem::create_directories(dir);
    return dir.string();
}

std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

template <typename T>
T rd(const std::vector<uint8_t>& b, size_t off) {
    T v{};
    std::memcpy(&v, b.data() + off, sizeof v);
    return v;
}

} // namespace

TEST_CASE("JIT observability - loaded code is registered by name and tier, and unregistered with its engine") {
    const size_t before = debug::jit_code_count();
    uintptr_t inner = 0, outer = 0;
    {
        auto jit = load_obs();
        inner = reinterpret_cast<uintptr_t>(jit->get_symbol_address("obs_inner"));
        outer = reinterpret_cast<uintptr_t>(jit->get_symbol_address("obs_outer"));
        CHECK_EQ(debug::jit_code_count(), before + 2);
        debug::JitCodeInfo info;
        REQUIRE(debug::find_jit_code(inner + 1, &info));
        CHECK(info.name == "obs_inner");
        CHECK(info.tier == debug::JitTier::Optimized);
        CHECK_EQ(info.start, inner);
        REQUIRE(debug::find_jit_code(outer, &info));
        CHECK(info.name == "obs_outer");
        CHECK(debug::describe_jit_address(outer + 4) == "obs_outer [tier 2]+0x4");

        // Reloading retires the old code's names before it is freed.
        auto mod = parse_or_fail(kObsModule);
        REQUIRE(jit->compile_and_load(*mod));
        CHECK_EQ(debug::jit_code_count(), before + 2);
        const auto inner2 = reinterpret_cast<uintptr_t>(jit->get_symbol_address("obs_inner"));
        REQUIRE(debug::find_jit_code(inner2, &info));
        CHECK(info.name == "obs_inner");

        // A moved engine keeps its registrations.
        codegen::JitExecutionEngine moved(std::move(*jit));
        CHECK_EQ(debug::jit_code_count(), before + 2);
        CHECK(debug::find_jit_code(inner2, nullptr));
    }
    CHECK_EQ(debug::jit_code_count(), before);
    CHECK(!debug::find_jit_code(inner, nullptr));
    CHECK(debug::describe_jit_address(outer).empty());
}

TEST_CASE("JIT observability - baseline code is tier 1 until its last copy goes, OSR code is tier 2 osr") {
    auto mod = parse_or_fail(kBaselineModule);
    const size_t before = debug::jit_code_count();
    uintptr_t entry = 0;
    {
        codegen::BaselineJitCompiler compiler;
        compiler.register_external_symbol("obs_tier2_inner", reinterpret_cast<void*>(&obs_probe));
        auto compiled = compiler.compile(*mod->get_function("obs_bl_outer"));
        REQUIRE(compiled.is_valid());
        entry = reinterpret_cast<uintptr_t>(compiled.entry_point());
        debug::JitCodeInfo info;
        REQUIRE(debug::find_jit_code(entry, &info));
        CHECK(info.name == "obs_bl_outer");
        CHECK(info.tier == debug::JitTier::Baseline);
        CHECK_EQ(info.size, compiled.code_size());
        {
            auto copy = compiled;
            (void)copy;
        }
        CHECK(debug::find_jit_code(entry, nullptr));
    }
    CHECK(!debug::find_jit_code(entry, nullptr));
    CHECK_EQ(debug::jit_code_count(), before);

    auto osr = load_obs(debug::JitTier::Osr);
    const auto a = reinterpret_cast<uintptr_t>(osr->get_symbol_address("obs_outer"));
    CHECK(debug::describe_jit_address(a) == "obs_outer [tier 2 osr]+0x0");
}

TEST_CASE("JIT observability - with every tool off, a registration is small beside the compile it follows") {
    using clock = std::chrono::steady_clock;
    static uint8_t fake_code[64 * 1024];
    constexpr int kRegs = 20000;
    const auto r0 = clock::now();
    for (int i = 0; i < kRegs; ++i) {
        auto h = debug::register_jit_code(debug::JitTier::Optimized, "obs_cost", fake_code + (i % 4096) * 16, 16);
        (void)h;
    }
    const double reg_us = std::chrono::duration<double, std::micro>(clock::now() - r0).count() / kRegs;

    auto mod = parse_or_fail(kObsModule);
    constexpr int kCompiles = 40;
    const auto c0 = clock::now();
    for (int i = 0; i < kCompiles; ++i) {
        codegen::JitExecutionEngine jit(Target::host());
        jit.register_external_symbol("obs_probe", reinterpret_cast<void*>(&obs_probe));
        REQUIRE(jit.compile_and_load(*mod));
    }
    const double compile_us = std::chrono::duration<double, std::micro>(clock::now() - c0).count() / kCompiles;
    std::printf("  registration + unregistration: %.3f us; a two-function compile_and_load: %.1f us\n", reg_us,
                compile_us);
    CHECK(reg_us < 50.0);
}

#if defined(_WIN32) && (defined(_M_X64) || defined(__x86_64__))
// The unwinder walks JIT frames through registered unwind data on Windows
// (elsewhere _Unwind_Backtrace through .eh_frame, not run here).
TEST_CASE("JIT observability - the symbolizer names JIT frames across tiers on a live stack") {
    auto t2 = load_obs();
    using Fn = int64_t (*)(int64_t);
    g_stack.clear();
    CHECK_EQ(t2->get_function_ptr<Fn>("obs_outer")(5), (5 + 10 + 1) * 2);
    // probe <- obs_inner [tier 2] <- obs_outer [tier 2] <- this test
    const size_t in = index_of(g_stack, "obs_inner [tier 2]");
    const size_t out = index_of(g_stack, "obs_outer [tier 2]");
    REQUIRE(in != SIZE_MAX);
    REQUIRE(out != SIZE_MAX);
    CHECK_EQ(out, in + 1);
    CHECK(g_native.find("obs_inner [tier 2]+0x") != std::string::npos);

    // A baseline function calling tier-2 code: both tiers on one stack.
    auto mod = parse_or_fail(kBaselineModule);
    codegen::BaselineJitCompiler compiler;
    compiler.register_external_symbol("obs_tier2_inner", t2->get_symbol_address("obs_inner"));
    auto bl = compiler.compile(*mod->get_function("obs_bl_outer"));
    REQUIRE(bl.is_valid());
    g_stack.clear();
    CHECK_EQ(bl.get_function_ptr<Fn>()(1), (1 + 10 + 1) + 3);
    const size_t t2_in = index_of(g_stack, "obs_inner [tier 2]");
    const size_t t1_out = index_of(g_stack, "obs_bl_outer [tier 1]");
    REQUIRE(t2_in != SIZE_MAX);
    REQUIRE(t1_out != SIZE_MAX);
    CHECK_EQ(t1_out, t2_in + 1);
    CHECK(debug::jit_crash_report_installed());
}
#endif

TEST_CASE("JIT observability - the perf map and jitdump carry each function by name and tier") {
    const std::string dir = temp_dir("brass_jit_obs_perf");
    std::string map_path, dump_path;
    uintptr_t inner = 0;
    size_t inner_size = 0;
    std::vector<uint8_t> inner_code;
    {
        KnobScope knobs({"BRASS_PERF_MAP", "BRASS_JITDUMP"}, dir);
        map_path = debug::perf_map_path();
        dump_path = debug::jitdump_path();
        std::filesystem::remove(map_path);
        auto jit = load_obs();
        for (const auto& f : jit->loaded_functions()) {
            if (f.name == "obs_inner") {
                inner = reinterpret_cast<uintptr_t>(f.code);
                inner_size = f.size;
                inner_code.assign(static_cast<const uint8_t*>(f.code), static_cast<const uint8_t*>(f.code) + f.size);
            }
        }
    }
    REQUIRE(inner != 0);

    std::ifstream map(map_path);
    REQUIRE(map.good());
    std::ostringstream want;
    want << std::hex << inner << " " << inner_size << " obs_inner [tier 2]";
    bool found = false;
    for (std::string line; std::getline(map, line);) found = found || line == want.str();
    CHECK(found);

    const std::vector<uint8_t> dump = read_file(dump_path);
    REQUIRE(dump.size() >= 40);
    CHECK_EQ(rd<uint32_t>(dump, 0), 0x4A695444u);
    CHECK_EQ(rd<uint32_t>(dump, 4), 1u);
    CHECK_EQ(rd<uint32_t>(dump, 8), 40u);
    bool record = false;
    for (size_t p = 40; p + 56 <= dump.size();) {
        const uint32_t id = rd<uint32_t>(dump, p);
        const uint32_t total = rd<uint32_t>(dump, p + 4);
        REQUIRE(total >= 56);
        if (id == 0 && rd<uint64_t>(dump, p + 32) == inner) {
            const char* name = reinterpret_cast<const char*>(dump.data() + p + 56);
            CHECK(std::string(name) == "obs_inner [tier 2]");
            CHECK_EQ(rd<uint64_t>(dump, p + 40), uint64_t{inner_size});
            const size_t code_at = p + 56 + std::strlen(name) + 1;
            CHECK(std::memcmp(dump.data() + code_at, inner_code.data(), inner_size) == 0);
            record = true;
        }
        p += total;
    }
    CHECK(record);
}

TEST_CASE("JIT observability - the GDB JIT interface gets an ELF symfile per load, and loses it with the code") {
    jit_code_entry* const first_before = __jit_debug_descriptor.first_entry;
    uintptr_t inner = 0;
    {
        KnobScope knobs({"BRASS_GDB_JIT"}, temp_dir("brass_jit_obs_gdb"));
        auto jit = load_obs();
        inner = reinterpret_cast<uintptr_t>(jit->get_symbol_address("obs_inner"));
        jit_code_entry* e = __jit_debug_descriptor.first_entry;
        REQUIRE(e != nullptr);
        CHECK(e != first_before);
        CHECK(__jit_debug_descriptor.relevant_entry == e);
        CHECK_EQ(__jit_debug_descriptor.action_flag, uint32_t{BRASS_JIT_REGISTER_FN});
        std::vector<uint8_t> elf(e->symfile_addr, e->symfile_addr + e->symfile_size);
        REQUIRE(elf.size() > 64);
        CHECK(elf[0] == 0x7f && elf[1] == 'E' && elf[2] == 'L' && elf[3] == 'F');
        // Walk the symtab for obs_inner's address.
        const uint64_t shoff = rd<uint64_t>(elf, 40);
        const size_t symtab = static_cast<size_t>(shoff + 2 * 64);
        const size_t strtab = static_cast<size_t>(shoff + 3 * 64);
        const uint64_t sym_off = rd<uint64_t>(elf, symtab + 24), sym_size = rd<uint64_t>(elf, symtab + 32);
        const uint64_t str_off = rd<uint64_t>(elf, strtab + 24);
        bool named = false;
        for (uint64_t s = sym_off + 24; s < sym_off + sym_size; s += 24) {
            const char* nm = reinterpret_cast<const char*>(elf.data() + str_off + rd<uint32_t>(elf, s));
            if (rd<uint64_t>(elf, s + 8) == inner) named = std::string(nm) == "obs_inner [tier 2]";
        }
        CHECK(named);
    }
    CHECK(__jit_debug_descriptor.first_entry == first_before);
    CHECK_EQ(__jit_debug_descriptor.action_flag, uint32_t{BRASS_JIT_NOACTION});
}
