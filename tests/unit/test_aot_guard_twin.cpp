// Guards in AOT images: a speculated function whose failed guards finish in
// its generic twin, compiled into the same image (docs/speculation_and_deopt.md,
// "Guards in AOT images").

#include "test_framework.hpp"
#include <sstream>
#include "aot_guard_test_util.hpp"
#include "diff_harness.hpp"
#include <brass/gc/code_stack_maps.hpp>
#include <brass/gc/heap.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/inliner.hpp>
#include <brass/mir/loop_opt.hpp>
#include <brass/mir/printer.hpp>
#include <brass/mir/resume_twin.hpp>
#include <brass/mir/speculation_sites.hpp>

using namespace brass;
using namespace brass::test_aot_guard;

namespace {

bool contains(const std::vector<std::string>& v, std::string_view s) {
    for (const auto& e : v) {
        if (e == s) return true;
    }
    return false;
}

} // namespace

// A guard with an ordinary exit stub asks the runtime for a lower tier
// first (brass_deopt_exit_record), so its code is not self-contained: an
// image holding one needs the runtime at load.
TEST_CASE("AOT guards - an exit stub guard needs the runtime") {
    const char* text = R"(module @aot_guard_stub
func @ag_stub(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 2
  %2 = mul.i64 %0, %1
  ret %2
}
func @ag_f(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 100
  %2 = slt.i64 %0, %1
  guard %2, @ag_stub, [%0]
  %3 = iconst.i64 1
  %4 = add.i64 %0, %3
  ret %4
}
)";
    auto mod = parse_checked(text);
    REQUIRE(mod != nullptr);
    object::ModuleCompiler compiler(Target::host());
    object::ObjectFile obj = compiler.compile(*mod);
    bool asks_runtime = false;
    for (const auto& sym : obj.symbols) {
        asks_runtime |= sym.name == "brass_deopt_exit_record" && sym.section_index == object::SECTION_UNDEF;
    }
    CHECK(asks_runtime);
}

// A guard naming a resume twin exits into it directly: the image imports
// nothing, and the twin's result is the function's.
TEST_CASE("AOT guards - a twin exit is direct") {
    const char* text = R"(module @aot_guard_direct
func @ag2_twin(%0: i32, %1: ptr) -> i64 resume_twin {
bb0:
  %2 = load.i64 %1, 0
  %3 = sext.i64 %0
  %4 = mul.i64 %2, %3
  ret %4
}
func @ag2_f(%0: i64) -> i64 {
bb0:
  %1 = iconst.i64 100
  %2 = slt.i64 %0, %1
  guard %2, @ag2_twin, [%0], id 7
  %3 = iconst.i64 1
  %4 = add.i64 %0, %3
  ret %4
}
)";
    auto mod = parse_checked(text);
    REQUIRE(mod != nullptr);
    LoadedImage img = link_and_load(*mod, "ag_direct");
    REQUIRE(img.lib != nullptr);
    CHECK(img.undefined.empty());
    CHECK(!contains(img.undefined, "brass_deopt_exit_record"));
    auto f = img.lib->get_function<int64_t (*)(int64_t)>("ag2_f");
    REQUIRE(f != nullptr);
    CHECK_EQ(f(5), 6);
    CHECK_EQ(f(500), 3500);
    CHECK_EQ(f(99), 100);
    CHECK_EQ(f(100), 700);
}

namespace {

// sum(a, n, misses): the sum of a[0..n). The site speculates that an element
// is not negative; its generic code counts itself in *misses, adds the
// element and goes on with the loop.
const char* kSumSite = R"(module @aot_sum
func @sum(%0: ptr, %1: i64, %2: ptr) -> i64 {
bb0:
  %z = iconst.i64 0
  br bb_head(%z, %z)

bb_head(%i: i64, %acc: i64):
  %c = slt.i64 %i, %1
  br_if %c, bb_body, bb_exit

bb_body:
  %x = load_indexed.i64 %0, %i, 8
  %zero = iconst.i64 0
  %ok = sge.i64 %x, %zero
  guard %ok, @deopt, [%x, %0, %1, %2, %i, %acc], id 3
  %acc2 = add.i64 %acc, %x
  br bb_latch(%acc2)

bb_slow(%sx: i64):
  %m = load.i64 %2, 0
  %one = iconst.i64 1
  %m2 = add.i64 %m, %one
  store.i64 %2, 0, %m2
  %acc3 = add.i64 %sx, %acc
  br bb_latch(%acc3)

bb_latch(%a: i64):
  %one2 = iconst.i64 1
  %i2 = add.i64 %i, %one2
  br bb_head(%i2, %a)

bb_exit:
  ret %acc

resume_table {
  entry 3 -> bb_slow
}
}
)";

using SumFn = int64_t (*)(const int64_t*, int64_t, int64_t*);

} // namespace

TEST_CASE("AOT guards - a loop's guard exits into its twin and the twin finishes the loop") {
    auto site = parse_checked(kSumSite);
    REQUIRE(site != nullptr);

    std::vector<int64_t> clean{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> dirty{1, 2, -3, 4, -5, 6, 7, 8};
    auto oracle = [&](const std::vector<int64_t>& a, int64_t* misses) {
        Interpreter in;
        return in
            .run(*site, "sum",
                 {RuntimeValue::from_ptr(static_cast<const void*>(a.data())),
                  RuntimeValue::from_i64(static_cast<int64_t>(a.size())),
                  RuntimeValue::from_ptr(static_cast<const void*>(misses))})
            .as_i64();
    };
    int64_t m = 0;
    CHECK_EQ(oracle(clean, &m), 36);
    CHECK_EQ(m, 0);
    CHECK_EQ(oracle(dirty, &m), 20);
    CHECK_EQ(m, 2);

    std::string why;
    const bool ok = speculate_module_against_twins(*site, &why);
    if (!ok) std::cerr << why << "\n";
    REQUIRE(ok);
    const Function* twin = site->get_function(resume_twin_name("sum"));
    REQUIRE(twin != nullptr);
    CHECK(twin->is_resume_twin());
    CHECK(twin->resume_points().empty());
    const Function* fast = site->get_function("sum");
    CHECK(fast->resume_points().empty());
    CHECK(fast->get_block_by_name("bb_slow") == nullptr);
    DiagnosticReporter diag;
    const bool verified = verify_module(*site, &diag);
    if (!verified) std::cerr << diag.format_all() << "\n";
    REQUIRE(verified);

    // The speculated pair is still a program every interpreter runs.
    m = 0;
    CHECK_EQ(oracle(dirty, &m), 20);
    CHECK_EQ(m, 2);

    LoadedImage img = link_and_load(*site, "ag_sum");
    REQUIRE(img.lib != nullptr);
    CHECK(img.undefined.empty());
    auto sum = img.lib->get_function<SumFn>("sum");
    REQUIRE(sum != nullptr);
    m = 0;
    CHECK_EQ(sum(clean.data(), 8, &m), 36);
    CHECK_EQ(m, 0);
    // The first negative element leaves the speculated code; the twin runs
    // the rest of the loop, taking its own generic path for the second.
    CHECK_EQ(sum(dirty.data(), 8, &m), 20);
    CHECK_EQ(m, 2);
    // A later call starts in the speculated code again.
    m = 0;
    CHECK_EQ(sum(clean.data(), 8, &m), 36);
    CHECK_EQ(m, 0);
    CHECK_EQ(sum(dirty.data(), 0, &m), 0);
}

// State values of every scalar representation, two of them as the resume
// block's parameters and the rest under their own names.
TEST_CASE("AOT guards - state values of every scalar type reach the twin") {
    const char* text = R"(module @aot_mix
func @mix(%0: i32, %1: i64, %2: f64, %3: ptr, %4: f32) -> f64 {
bb0:
  %c0 = iconst.i32 0
  %ok = sge.i32 %0, %c0
  guard %ok, @deopt, [%0, %1, %2, %3, %4], id 0
  %a = sitofp.f64 %0
  %b = sitofp.f64 %1
  %s1 = add.f64 %a, %b
  %s2 = add.f64 %s1, %2
  %m = load.f64 %3, 0
  %s3 = add.f64 %s2, %m
  %fb = bitcast.i32 %4
  %fc = sitofp.f64 %fb
  %s4 = add.f64 %s3, %fc
  ret %s4

bb_slow(%p0: i32, %p1: i64):
  %a2 = sitofp.f64 %p0
  %b2 = sitofp.f64 %p1
  %t1 = add.f64 %a2, %b2
  %t2 = add.f64 %t1, %2
  %m2 = load.f64 %3, 0
  %t3 = add.f64 %t2, %m2
  %fb2 = bitcast.i32 %4
  %fc2 = sitofp.f64 %fb2
  %t4 = add.f64 %t3, %fc2
  %k = iconst.i64 1000
  %kf = sitofp.f64 %k
  %t5 = add.f64 %t4, %kf
  ret %t5

resume_table {
  entry 0 -> bb_slow
}
}
)";
    auto mod = parse_checked(text);
    REQUIRE(mod != nullptr);
    std::string why;
    const bool ok = speculate_module_against_twins(*mod, &why);
    if (!ok) std::cerr << why << "\n";
    REQUIRE(ok);
    LoadedImage img = link_and_load(*mod, "ag_mix");
    REQUIRE(img.lib != nullptr);
    CHECK(img.undefined.empty());
    auto mix = img.lib->get_function<double (*)(int32_t, int64_t, double, const double*, float)>("mix");
    REQUIRE(mix != nullptr);
    const double mem = 0.25;
    float f = 0;
    const int32_t fbits = 77;
    std::memcpy(&f, &fbits, 4);
    const int64_t big = (int64_t{1} << 40) + 5;
    CHECK_EQ(mix(3, big, 1.5, &mem, f), 3.0 + static_cast<double>(big) + 1.5 + 0.25 + 77.0);
    CHECK_EQ(mix(-3, big, 1.5, &mem, f), -3.0 + static_cast<double>(big) + 1.5 + 0.25 + 77.0 + 1000.0);
    CHECK_EQ(mix(-2147483647, -big, -0.0, &mem, f),
             -2147483647.0 - static_cast<double>(big) + 0.25 + 77.0 + 1000.0);
}

namespace {

gc::Heap* g_twin_heap = nullptr;
int g_twin_collections = 0;

// Called by generated code: a young collection that walks the generated
// frames above this one through their stack maps.
int64_t twin_collect_now() {
    void* frame = __builtin_frame_address(0);
    const uintptr_t caller_fp = *reinterpret_cast<uintptr_t*>(frame);
    const uintptr_t caller_ip = *reinterpret_cast<uintptr_t*>(reinterpret_cast<uintptr_t>(frame) + 8);
    g_twin_heap->collect_at(gc::CollectionKind::Minor, caller_fp, caller_ip);
    ++g_twin_collections;
    return 0;
}

// The image's stack maps, registered at the addresses it loaded at.
std::shared_ptr<const void> register_image_maps(const LoadedImage& img) {
    ModuleStackMap maps = img.obj.stack_maps;
    for (const auto& fn : img.obj.functions) {
        if (void* addr = img.lib->get_symbol(fn.name)) {
            maps.register_function_address(fn.name, reinterpret_cast<uintptr_t>(addr),
                                           static_cast<uint32_t>(fn.text_size));
        }
    }
    return register_code_stack_maps(maps);
}

} // namespace

// References in a guard's state: the twin holds them as roots from its first
// instruction, so a collection right after the exit keeps (or moves) the
// objects and the twin reads them where they are then.
TEST_CASE("AOT guards - gcref state survives a collection in the twin") {
    const char* text = R"(module @aot_gcx
func @gcx(%0: gcref, %1: ptr, %2: i64, %3: gcref) -> i64 {
bb0:
  %v = load.i64 %0, 0
  %z = iconst.i64 0
  %ok = eq.i64 %2, %z
  guard %ok, @deopt, [%3, %0, %1, %v], id 5
  %w = load.i64 %3, 0
  %r = add.i64 %v, %w
  ret %r

bb_slow(%q: gcref):
  %j = call_indirect.i64 %1()
  %v2 = load.i64 %0, 0
  %w2 = load.i64 %q, 0
  %s = add.i64 %v2, %w2
  %s2 = add.i64 %s, %v
  %j2 = call_indirect.i64 %1()
  %w3 = load.i64 %q, 0
  %s3 = add.i64 %s2, %w3
  ret %s3

resume_table {
  entry 5 -> bb_slow
}
}
)";
    auto mod = parse_checked(text);
    REQUIRE(mod != nullptr);
    std::string why;
    const bool ok = speculate_module_against_twins(*mod, &why);
    if (!ok) std::cerr << why << "\n";
    REQUIRE(ok);
    LoadedImage img = link_and_load(*mod, "ag_gcx");
    REQUIRE(img.lib != nullptr);
    auto gcx = img.lib->get_function<int64_t (*)(uintptr_t, void*, int64_t, uintptr_t)>("gcx");
    REQUIRE(gcx != nullptr);

    gc::HeapConfig config;
    config.read_environment = false;
    config.poison = true;
    gc::Heap heap(config);
    g_twin_heap = &heap;
    g_twin_collections = 0;
    const auto maps = register_image_maps(img);

    auto fresh = [&](int64_t field) {
        const uintptr_t o = heap.allocate_masked(16, 0, 1);
        *reinterpret_cast<int64_t*>(o) = field;
        return o;
    };
    void* collect = reinterpret_cast<void*>(&twin_collect_now);
    CHECK_EQ(gcx(fresh(11), collect, 0, fresh(31)), 42);
    CHECK_EQ(g_twin_collections, 0);
    // The exit: two collections in the twin with both objects live across them.
    CHECK_EQ(gcx(fresh(11), collect, 1, fresh(31)), 11 + 31 + 11 + 31);
    CHECK_EQ(g_twin_collections, 2);
    CHECK_EQ(heap.collection_count(), 2ULL);
    g_twin_heap = nullptr;
}

// Every tier and the image give one answer, with every guard forced to
// fail, each alone, and at several periods.
TEST_CASE("AOT guards - the image agrees with every tier under deopt stress") {
    using brass::test::assert_diff_tiers;
    brass::test::TierDiffOptions opts;
    opts.exits_equivalent = true;
    opts.require_aot = true;
    // The pipeline's tier 2 optimizes the site form with its resume block
    // still an entry, and its passes leave that block reading a constant of
    // the entry block that the guard's state lacks (the verifier rejects the
    // result). A front end's tier-2 pass drops the resume table first; this
    // module has none, so that leg is off.
    opts.tier2 = false;
    // Standalone tier-2 code cannot continue a resume-only guard at all.
    opts.jit = false;
    std::vector<int64_t> clean{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> dirty{1, 2, -3, 4, -5, 6, 7, -8};
    for (const auto* a : {&clean, &dirty}) {
        for (int64_t n : {int64_t{0}, int64_t{1}, int64_t{8}}) {
            auto site = parse_checked(kSumSite);
            REQUIRE(site != nullptr);
            int64_t misses = 0;
            assert_diff_tiers(*site, "sum",
                              {RuntimeValue::from_ptr(static_cast<const void*>(a->data())), RuntimeValue::from_i64(n),
                               RuntimeValue::from_ptr(static_cast<const void*>(&misses))},
                              opts);
        }
    }
}

namespace {

// twice(x, misses) speculates that x is not negative; total(a, n, misses)
// sums twice(a[i]).
const char* kInlineSite = R"(module @aot_inline
func @twice(%0: i64, %1: ptr) -> i64 {
bb0:
  %z = iconst.i64 0
  %ok = sge.i64 %0, %z
  guard %ok, @deopt, [%0, %1], id 0
  %two = iconst.i64 2
  %r = mul.i64 %0, %two
  ret %r

bb_slow:
  %m = load.i64 %1, 0
  %one = iconst.i64 1
  %m2 = add.i64 %m, %one
  store.i64 %1, 0, %m2
  %r2 = add.i64 %0, %0
  ret %r2

resume_table {
  entry 0 -> bb_slow
}
}
func @total(%0: ptr, %1: i64, %2: ptr) -> i64 {
bb0:
  %z = iconst.i64 0
  br bb_head(%z, %z)

bb_head(%i: i64, %acc: i64):
  %c = slt.i64 %i, %1
  br_if %c, bb_body, bb_exit

bb_body:
  %x = load_indexed.i64 %0, %i, 8
  %y = call.i64 @twice(%x, %2)
  %acc2 = add.i64 %acc, %y
  %one = iconst.i64 1
  %i2 = add.i64 %i, %one
  br bb_head(%i2, %acc2)

bb_exit:
  ret %acc
}
)";

size_t count_calls(const Function& fn, std::string_view callee) {
    size_t n = 0;
    for (const BasicBlock* bb : fn.blocks()) {
        for (const Instruction* inst : *const_cast<BasicBlock*>(bb)) {
            if (inst && inst->opcode() == Opcode::call && inst->symbol() == callee) ++n;
        }
    }
    return n;
}

} // namespace

// A speculated callee inlined: its guard becomes a branch to a call of the
// callee's twin, which finishes the callee alone; the caller goes on in its
// own speculated code with the twin's result.
TEST_CASE("AOT guards - an inlined callee's guard finishes the callee in its twin") {
    auto mod = parse_checked(kInlineSite);
    REQUIRE(mod != nullptr);
    std::string why;
    const bool ok = speculate_module_against_twins(*mod, &why);
    if (!ok) std::cerr << why << "\n";
    REQUIRE(ok);
    Function* total = mod->get_function("total");
    REQUIRE(total != nullptr);
    REQUIRE(inline_function(*total, *mod));
    total->rebuild_cfg_predecessors();
    DiagnosticReporter diag;
    const bool verified = verify_module(*mod, &diag);
    if (!verified) std::cerr << diag.format_all() << "\n";
    REQUIRE(verified);
    CHECK_EQ(count_calls(*total, "twice"), size_t{0});
    CHECK_EQ(count_calls(*total, resume_twin_name("twice")), size_t{1});

    std::vector<int64_t> dirty{1, 2, -3, 4, -5, 6, 7, -8};
    int64_t m = 0;
    {
        Interpreter in;
        CHECK_EQ(in.run(*mod, "total",
                        {RuntimeValue::from_ptr(static_cast<const void*>(dirty.data())), RuntimeValue::from_i64(8),
                         RuntimeValue::from_ptr(static_cast<const void*>(&m))})
                     .as_i64(),
                 8);
        CHECK_EQ(m, 3);
    }
    LoadedImage img = link_and_load(*mod, "ag_inline");
    REQUIRE(img.lib != nullptr);
    CHECK(img.undefined.empty());
    auto fn = img.lib->get_function<SumFn>("total");
    REQUIRE(fn != nullptr);
    m = 0;
    CHECK_EQ(fn(dirty.data(), 8, &m), 8);
    // Each negative element left the inlined body and came back.
    CHECK_EQ(m, 3);
}

// A tagged state value, and the address of a tagged frame block: the block
// stays the speculated frame's (which is still on the stack under the twin),
// and its words are that frame's roots at the exit's call.
TEST_CASE("AOT guards - a tagged alloca in the state stays rooted in the speculated frame") {
    const char* text = R"(module @aot_tb
func @tb(%0: tagged, %1: ptr, %2: i64, %3: i64) -> i64 {
bb0:
  %buf = alloca.tagged 16, 8
  store.i64 %buf, 8, %3
  %z = iconst.i64 0
  %ok = eq.i64 %2, %z
  guard %ok, @deopt, [%0, %buf, %1], id 1
  %r = iconst.i64 -1
  ret %r

bb_slow(%q: tagged):
  %j = call_indirect.i64 %1()
  %mask = iconst.i64 281474976710655
  %ua = bitcast.i64.tagged %q
  %pa = and.i64 %ua, %mask
  %va = load.i64 %pa, 0
  %ub = load.i64 %buf, 8
  %pb = and.i64 %ub, %mask
  %vb = load.i64 %pb, 0
  %s = add.i64 %va, %vb
  ret %s

resume_table {
  entry 1 -> bb_slow
}
}
)";
    auto mod = parse_checked(text);
    REQUIRE(mod != nullptr);
    std::string why;
    const bool ok = speculate_module_against_twins(*mod, &why);
    if (!ok) std::cerr << why << "\n";
    REQUIRE(ok);
    LoadedImage img = link_and_load(*mod, "ag_tb");
    REQUIRE(img.lib != nullptr);
    auto tb = img.lib->get_function<int64_t (*)(uint64_t, void*, int64_t, uint64_t)>("tb");
    REQUIRE(tb != nullptr);

    constexpr uint16_t kRefTag = 0xFFF1;
    constexpr uint64_t kTag = uint64_t{kRefTag} << 48;
    gc::HeapConfig config;
    config.read_environment = false;
    config.poison = true;
    config.reference_tags = {kRefTag};
    gc::Heap heap(config);
    g_twin_heap = &heap;
    g_twin_collections = 0;
    const auto maps = register_image_maps(img);
    auto fresh = [&](int64_t field) {
        const uintptr_t o = heap.allocate_masked(16, 0, 1);
        *reinterpret_cast<int64_t*>(o) = field;
        return static_cast<uint64_t>(o) | kTag;
    };
    void* collect = reinterpret_cast<void*>(&twin_collect_now);
    CHECK_EQ(tb(fresh(40), collect, 0, fresh(2)), -1);
    CHECK_EQ(tb(fresh(40), collect, 1, fresh(2)), 42);
    CHECK_EQ(g_twin_collections, 1);
    g_twin_heap = nullptr;
}

namespace {
int64_t g_sum_exits = 0;
}

// Counting failures costs the speculated code nothing: the count is in the
// twin's entries (ResumeTwinOptions::on_exit). A caller that sees it pass
// its limit calls the twin's own entry from then on, which runs the same
// code with no speculation and no count.
TEST_CASE("AOT guards - failures are counted in the twin and a caller pins to the generic entry") {
    auto mod = parse_checked(kSumSite);
    REQUIRE(mod != nullptr);
    Function* sum = mod->get_function("sum");
    REQUIRE(sum != nullptr);
    ResumeTwinOptions options;
    options.entry_id = 1000;
    options.on_exit = [](Builder& b, uint32_t) {
        Value* addr = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(&g_sum_exits)));
        b.build_store(Type::i64(), addr, 0, b.build_add(b.build_load(Type::i64(), addr, 0), b.build_iconst_i64(1)));
    };
    std::string why;
    const Function* twin = build_resume_twin(*sum, *mod, "sum_twin", options, &why);
    if (!twin) std::cerr << why << "\n";
    REQUIRE(twin != nullptr);
    REQUIRE(build_twin_entry(*sum, *twin, 1000, "sum_generic", &why) != nullptr);
    REQUIRE(speculate_against_twin(*sum, *twin, &why));
    DiagnosticReporter diag;
    const bool verified = verify_module(*mod, &diag);
    if (!verified) std::cerr << diag.format_all() << "\n";
    REQUIRE(verified);

    LoadedImage img = link_and_load(*mod, "ag_pin");
    REQUIRE(img.lib != nullptr);
    CHECK(img.undefined.empty());
    auto fast = img.lib->get_function<SumFn>("sum");
    auto generic = img.lib->get_function<SumFn>("sum_generic");
    REQUIRE(fast != nullptr);
    REQUIRE(generic != nullptr);

    std::vector<int64_t> clean{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> dirty{1, 2, -3, 4, -5, 6, 7, 8};
    g_sum_exits = 0;
    int64_t m = 0;
    SumFn entry = fast;
    int calls_speculated = 0;
    for (int call = 0; call < 10; ++call) {
        calls_speculated += entry == fast;
        CHECK_EQ(entry(dirty.data(), 8, &m), 20);
        CHECK_EQ(entry(clean.data(), 8, &m), 36);
        if (g_sum_exits >= 3) entry = generic;
    }
    // One exit a call of the speculated code on the dirty array (the twin
    // runs the rest of that loop), three calls until the limit.
    CHECK_EQ(g_sum_exits, 3);
    CHECK_EQ(calls_speculated, 3);
    // Both negative elements took the generic path in every call.
    CHECK_EQ(m, 20);
}

// The remedy without a caller's help: a call names the function and the
// word it goes through (`call @sum via @sum_entry`). The twin's entries count
// in the image's own data and, at the limit, store the generic entry into
// that word; the calls then reach the generic code, and the speculated
// function's code holds nothing of it.
TEST_CASE("AOT guards - a call through an entry cell is routed to the generic entry by the twin") {
    auto mod = parse_checked(std::string(kSumSite) + R"(
func @twice(%0: ptr, %1: i64, %2: ptr) -> i64 {
bb0:
  %a = call.i64 @sum via @sum_entry(%0, %1, %2)
  %b = call.i64 @sum via @sum_entry(%0, %1, %2)
  %r = add.i64 %a, %b
  ret %r
}
func @sum_entry_addr() -> ptr {
bb0:
  %a = func_addr @sum_entry
  ret %a
}
)");
    REQUIRE(mod != nullptr);
    Function* sum = mod->get_function("sum");
    REQUIRE(sum != nullptr);
    ResumeTwinOptions options;
    options.entry_id = 1000;
    // sum_entry: the entry word, then the count of exits.
    options.on_exit = [](Builder& b, uint32_t) {
        Value* cell = b.build_func_addr("sum_entry");
        Value* n = b.build_add(b.build_load(Type::i64(), cell, 8), b.build_iconst_i64(1));
        b.build_store(Type::i64(), cell, 8, n);
        Value* pinned = b.build_sge(n, b.build_iconst_i64(3));
        b.build_store(Type::ptr(), cell, 0,
                      b.build_select(pinned, b.build_func_addr("sum_generic"), b.build_func_addr("sum")));
    };
    std::string why;
    const Function* twin = build_resume_twin(*sum, *mod, "sum_twin", options, &why);
    if (!twin) std::cerr << why << "\n";
    REQUIRE(twin != nullptr);
    REQUIRE(build_twin_entry(*sum, *twin, 1000, "sum_generic", &why) != nullptr);
    REQUIRE(speculate_against_twin(*sum, *twin, &why));
    REQUIRE(verify_module(*mod));

    // The text form carries the cell.
    std::ostringstream os;
    print_module(*mod, os);
    CHECK(os.str().find("call.i64 @sum via @sum_entry(") != std::string::npos);
    auto again = parse_checked(os.str());
    REQUIRE(again != nullptr);
    CHECK_EQ(again->get_function("twice")->entry_block()->head()->extra_symbol(), "sum_entry");

    LoadedImage img = link_and_load(*mod, "ag_cell", nullptr, {}, [](object::ObjectFile& obj) {
        object::Section& data = obj.get_or_create_section(
            ".data", object::SectionKind::Data,
            object::SectionFlags::Read | object::SectionFlags::Write | object::SectionFlags::Alloc, 8);
        data.align_to(8);
        const size_t at = data.data.size();
        data.relocations.push_back({at, object::RelocKind::Abs64, "sum", 0});
        data.emit64(0);
        data.emit64(0);
        obj.add_symbol({"sum_entry", obj.get_section_index(".data"), at, 16, object::SymbolBinding::Global,
                        object::SymbolType::Object});
    });
    REQUIRE(img.lib != nullptr);
    CHECK(img.undefined.empty());
    auto twice = img.lib->get_function<SumFn>("twice");
    auto cell_addr = img.lib->get_function<int64_t* (*)()>("sum_entry_addr");
    REQUIRE(twice != nullptr);
    REQUIRE(cell_addr != nullptr);
    int64_t* cell = cell_addr();
    const int64_t speculated = cell[0];

    std::vector<int64_t> clean{1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int64_t> dirty{1, 2, -3, 4, -5, 6, 7, 8};
    int64_t m = 0;
    // Passing calls count nothing and leave the word alone.
    CHECK_EQ(twice(clean.data(), 8, &m), 72);
    CHECK_EQ(cell[1], 0);
    CHECK_EQ(cell[0], speculated);
    // Two exits: still the speculated function.
    CHECK_EQ(twice(dirty.data(), 8, &m), 40);
    CHECK_EQ(cell[1], 2);
    CHECK_EQ(cell[0], speculated);
    // The third exit stores the generic entry; the calls after it count no
    // exit, whatever the array.
    CHECK_EQ(twice(dirty.data(), 8, &m), 40);
    CHECK_EQ(cell[1], 3);
    CHECK(cell[0] != speculated);
    CHECK_EQ(twice(dirty.data(), 8, &m), 40);
    CHECK_EQ(twice(clean.data(), 8, &m), 72);
    CHECK_EQ(cell[1], 3);
    // Every negative element was added by generic code: 2 a call of sum.
    CHECK_EQ(m, 12);
}

// What the verifier holds a twin exit to.
TEST_CASE("AOT guards - the verifier checks a guard against its twin") {
    auto mod = parse_checked(kSumSite);
    REQUIRE(mod != nullptr);
    Function* sum = mod->get_function("sum");
    std::string why;
    Function* twin = build_resume_twin(*sum, *mod, "sum_twin", {}, &why);
    REQUIRE(twin != nullptr);
    REQUIRE(speculate_against_twin(*sum, *twin, &why));
    REQUIRE(verify_module(*mod));

    // The text form carries the attribute.
    std::ostringstream os;
    print_module(*mod, os);
    CHECK(os.str().find("-> i64 resume_twin {") != std::string::npos);
    auto again = parse_checked(os.str());
    REQUIRE(again != nullptr);
    CHECK(again->get_function("sum_twin")->is_resume_twin());

    Instruction* guard = nullptr;
    for (BasicBlock* bb : sum->blocks()) {
        for (Instruction* inst : *bb) {
            if (inst->opcode() == Opcode::guard) guard = inst;
        }
    }
    REQUIRE(guard != nullptr);
    // A state value dropped: the twin would read past what the guard wrote.
    Value* last = guard->state_map().back();
    guard->state_map().pop_back();
    CHECK(!verify_module(*mod));
    guard->state_map().push_back(last);
    CHECK(verify_module(*mod));
    // An id the twin has no entry for.
    guard->set_resume_id(4);
    CHECK(!verify_module(*mod));
    guard->set_resume_id(3);
    // A twin of another shape.
    twin->set_return_type(Type::i32());
    CHECK(!verify_module(*mod));
}

// A function with nothing to resume has no twin, and a coroutine body is
// refused by name.
TEST_CASE("AOT guards - functions without a twin") {
    auto mod = parse_checked(R"(module @aot_none
func @plain(%0: i64) -> i64 {
bb0:
  ret %0
}
)");
    REQUIRE(mod != nullptr);
    std::string why;
    CHECK(build_resume_twin(*mod->get_function("plain"), *mod, "plain_twin", {}, &why) == nullptr);
    CHECK(why.find("no guard") != std::string::npos);
    CHECK(speculate_module_against_twins(*mod, &why));
    CHECK(mod->get_function(resume_twin_name("plain")) == nullptr);
}

// The front end's flow with brass's site emitter: the site form is built
// once, run in the interpreter to collect counts, the twin is built from it,
// and then the same function is armed from the counts and pointed at the
// twin.
TEST_CASE("AOT guards - a site emitted by SpecSiteEmitter, armed from its counts") {
    SpecFeedback feedback([](uint8_t, uint32_t misses, bool) { return misses == 0; });
    Module mod("aot_site");
    Function* fn = mod.create_function("addsmall", Type::i64(), {Type::i64(), Type::i64()});
    {
        Builder b(mod);
        b.set_function(fn);
        SpecSiteEmitter sites;
        sites.set_feedback(&feedback);
        sites.begin_function(true, "addsmall");
        BasicBlock* entry = b.append_block("entry");
        Value* x = b.add_block_param(entry, Type::i64());
        Value* y = b.add_block_param(entry, Type::i64());
        BasicBlock* fast = b.append_block("fast");
        BasicBlock* slow = b.append_block("slow");
        BasicBlock* join = b.append_block("join");
        Value* r = b.add_block_param(join, Type::i64());
        b.position_at_end(entry);
        Value* hit = b.build_slt(x, b.build_iconst_i64(100));
        sites.emit_branch(b, hit, fast, slow, 0);
        // The generic code: the same sum, plus a mark that it ran.
        Value* generic = b.build_add(b.build_add(y, x), b.build_iconst_i64(1000000));
        b.build_br(join, {generic});
        b.position_at_end(fast);
        b.build_br(join, {b.build_add(x, y)});
        b.position_at_end(join);
        b.build_ret(r);
        fn->rebuild_cfg_predecessors();
        sites.finish_function(*fn);
    }
    DiagnosticReporter diag;
    const bool verified = verify_module(mod, &diag);
    if (!verified) std::cerr << diag.format_all() << "\n";
    REQUIRE(verified);
    {
        Interpreter in;
        CHECK_EQ(in.run(mod, "addsmall", {RuntimeValue::from_i64(5), RuntimeValue::from_i64(7)}).as_i64(), 12);
    }
    REQUIRE_EQ(feedback.size(), size_t{1});
    CHECK(feedback.should_speculate(0));

    std::string why;
    const Function* twin = build_resume_twin(*fn, mod, "addsmall_twin", {}, &why);
    if (!twin) std::cerr << why << "\n";
    REQUIRE(twin != nullptr);
    arm_speculation_sites(mod, feedback);
    const bool pointed = speculate_against_twin(*fn, *twin, &why);
    if (!pointed) std::cerr << why << "\n";
    REQUIRE(pointed);
    REQUIRE(verify_module(mod));
    CHECK(fn->get_block_by_name("slow") == nullptr);
    CHECK(fn->resume_points().empty());

    LoadedImage img = link_and_load(mod, "ag_site");
    REQUIRE(img.lib != nullptr);
    CHECK(img.undefined.empty());
    auto f = img.lib->get_function<int64_t (*)(int64_t, int64_t)>("addsmall");
    REQUIRE(f != nullptr);
    CHECK_EQ(f(5, 7), 12);
    CHECK_EQ(feedback.site(0).misses, 0u);
    // The armed guard fails: the twin's slow block runs and counts itself.
    CHECK_EQ(f(500, 7), 1000507);
    CHECK_EQ(feedback.site(0).misses, 1u);
    CHECK_EQ(f(99, 1), 100);
}

#if defined(BRASS_TEST_SHARED_LIB)
// An image built for another process has no counters to bake in: its stress
// guards ask the runtime it runs against, here a second copy of brass (the
// shared library), whose configuration decides.
TEST_CASE("AOT guards - an image's stress guards poll the runtime it runs against") {
    auto site = parse_checked(kSumSite);
    REQUIRE(site != nullptr);
    std::string why;
    REQUIRE(speculate_module_against_twins(*site, &why));
    DeoptStressPlan plan;
    plan.period = 1;
    plan.runtime_poll = true;
    LoadedImage img = link_and_load(*site, "ag_poll", &plan,
                                    {{BRASS_TEST_SHARED_LIB, {"brass_deopt_stress_poll_site"}}});
    REQUIRE(img.lib != nullptr);
    CHECK(contains(img.undefined, "brass_deopt_stress_poll_site"));
    CHECK_EQ(img.undefined.size(), size_t{1});
    auto runtime = target::DynamicLibrary::open(BRASS_TEST_SHARED_LIB, &why);
    REQUIRE(runtime != nullptr);
    auto set_stress = runtime->get_function<void (*)(uint64_t, int64_t)>("brass_set_deopt_stress");
    REQUIRE(set_stress != nullptr);
    auto sum = img.lib->get_function<SumFn>("sum");
    REQUIRE(sum != nullptr);

    std::vector<int64_t> clean{1, 2, 3, 4, 5, 6, 7, 8};
    int64_t m = 0;
    set_stress(0, -1);
    CHECK_EQ(sum(clean.data(), 8, &m), 36);
    CHECK_EQ(m, 0);
    // Every guard: the first evaluation leaves for the twin, which counts it.
    set_stress(1, -1);
    CHECK_EQ(sum(clean.data(), 8, &m), 36);
    CHECK_EQ(m, 1);
    // Another guard's id: nothing is forced.
    m = 0;
    set_stress(1, 99);
    CHECK_EQ(sum(clean.data(), 8, &m), 36);
    CHECK_EQ(m, 0);
    // Every third evaluation of guard 3.
    set_stress(3, 3);
    CHECK_EQ(sum(clean.data(), 8, &m), 36);
    CHECK_EQ(m, 1);
    set_stress(0, -1);
}
#endif
