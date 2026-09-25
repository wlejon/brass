// Tagged values are precise roots at every tier. A tagged word carries a
// reference in its low 48 bits under a reference tag, or something else under
// any other tag; a collection updates the address of the first kind and keeps
// its tag, and leaves the second alone. Each case holds a tagged SSA value and
// an alloca.tagged word across a callee that allocates until the young
// generation collects, then reads through both. The heap poisons freed
// memory, so a root the tier failed to report reads poison instead of the
// stored payload.
#include "test_framework.hpp"
#include "gc_test_heap.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/parser.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/vm/fast_interpreter.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/stack_map.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/runtime/compile_pool.hpp>
#include <brass/runtime/osr_coordinator.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/runtime/tiering.hpp>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

using namespace brass;
using namespace brass::runtime;

#if defined(_M_X64) || defined(__x86_64__) || defined(__aarch64__) || defined(_M_ARM64)

namespace {

constexpr uint16_t kRefTag = 0xFFF1;

// @tg_hold's result when every root was reported and updated: the payloads
// 42 and 7, plus 100 for each reference that kept its tag and 1000 for the
// non-reference word that was left exactly as it was.
constexpr int64_t kHoldExpected = 42 + 7 + 100 + 100 + 1000;

// -4222124650659840 is 0xFFF1 << 48, 281474976710655 the 48-bit address
// mask, and 4611686018427387904 (0x4000 << 48) a tag the heap does not take
// for a reference.
const char* kSrc = R"(module @tg
func @tg_churn(%n: i64) -> i64 {
entry:
  %z = iconst.i64 0
  br loop(%z)
loop(%i: i64):
  %sz = iconst.i64 64
  %kind = iconst.i32 2
  %t = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %one = iconst.i64 1
  %ni = add.i64 %i, %one
  %more = slt.i64 %ni, %n
  br_if %more, loop(%ni), done
done:
  ret %n
}
func @tg_check(%t: tagged, %buf: ptr, %tf: tagged, %fbits: i64) -> i64 {
b0:
  %tag = iconst.i64 -4222124650659840
  %mask = iconst.i64 281474976710655
  %zero = iconst.i64 0
  %hundred = iconst.i64 100
  %thousand = iconst.i64 1000
  %ua = bitcast.i64.tagged %t
  %pa = and.i64 %ua, %mask
  %va = load.i64 %pa, 16
  %hia = xor.i64 %ua, %pa
  %oka = eq.i64 %hia, %tag
  %sa = select.i64 %oka, %hundred, %zero
  %ub = load.i64 %buf, 8
  %pb = and.i64 %ub, %mask
  %vb = load.i64 %pb, 16
  %hib = xor.i64 %ub, %pb
  %okb = eq.i64 %hib, %tag
  %sb = select.i64 %okb, %hundred, %zero
  %uf = bitcast.i64.tagged %tf
  %okf = eq.i64 %uf, %fbits
  %sf = select.i64 %okf, %thousand, %zero
  %r0 = add.i64 %va, %vb
  %r1 = add.i64 %r0, %sa
  %r2 = add.i64 %r1, %sb
  %r3 = add.i64 %r2, %sf
  ret %r3
}
func @tg_hold(%n: i64) -> i64 {
b0:
  %sz = iconst.i64 32
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %tag = iconst.i64 -4222124650659840
  %dbl = iconst.i64 4611686018427387904
  %a = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %k42 = iconst.i64 42
  store.i64 %a, 16, %k42
  %abits = or.i64 %a, %tag
  %ta = bitcast.tagged.i64 %abits
  %fbits = or.i64 %a, %dbl
  %tf = bitcast.tagged.i64 %fbits
  %buf = alloca.tagged 16, 8
  %b = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %k7 = iconst.i64 7
  store.i64 %b, 16, %k7
  %bbits = or.i64 %b, %tag
  store.i64 %buf, 8, %bbits
  %r = call.i64 @tg_churn(%n)
  %v = call.i64 @tg_check(%ta, %buf, %tf, %fbits)
  ret %v
}
func @tg_host(%n: i64) -> i64 {
b0:
  %sz = iconst.i64 32
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %tag = iconst.i64 -4222124650659840
  %dbl = iconst.i64 4611686018427387904
  %a = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %k42 = iconst.i64 42
  store.i64 %a, 16, %k42
  %abits = or.i64 %a, %tag
  %ta = bitcast.tagged.i64 %abits
  %fbits = or.i64 %a, %dbl
  %tf = bitcast.tagged.i64 %fbits
  %buf = alloca.tagged 16, 8
  %b = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %k7 = iconst.i64 7
  store.i64 %b, 16, %k7
  %bbits = or.i64 %b, %tag
  store.i64 %buf, 8, %bbits
  %r = call.i64 @tg_host_collect(%n)
  %v = call.i64 @tg_check(%ta, %buf, %tf, %fbits)
  ret %v
}
func @tg_exit(%t: tagged, %n: i64) -> i64 {
b0:
  %r = call.i64 @tg_churn(%n)
  %mask = iconst.i64 281474976710655
  %u = bitcast.i64.tagged %t
  %p = and.i64 %u, %mask
  %v = load.i64 %p, 16
  ret %v
}
func @tg_spec(%n: i64, %g: i32) -> i64 {
b0:
  %sz = iconst.i64 32
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %tag = iconst.i64 -4222124650659840
  %a = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %k42 = iconst.i64 42
  store.i64 %a, 16, %k42
  %abits = or.i64 %a, %tag
  %ta = bitcast.tagged.i64 %abits
  %one = iconst.i32 1
  %c = eq.i32 %g, %one
  guard %c, @tg_exit, [%ta, %n]
  %mask = iconst.i64 281474976710655
  %u = bitcast.i64.tagged %ta
  %p = and.i64 %u, %mask
  %v = load.i64 %p, 16
  ret %v
}
func @tg_keep(%t: tagged, %n: i64) -> i64 {
b0:
  %r = call.i64 @tg_churn(%n)
  keep_alive %t
  ret %r
}
func @tg_nokeep(%t: tagged, %n: i64) -> i64 {
b0:
  %r = call.i64 @tg_churn(%n)
  ret %r
}
func @tg_keep_run(%n: i64) -> i64 {
b0:
  %sz = iconst.i64 32
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %tag = iconst.i64 -4222124650659840
  %a = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %abits = or.i64 %a, %tag
  %ta = bitcast.tagged.i64 %abits
  %r = call.i64 @tg_keep(%ta, %n)
  ret %r
}
func @tg_osr(%n: i64) -> i64 {
entry:
  %sz = iconst.i64 32
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %tag = iconst.i64 -4222124650659840
  %mask = iconst.i64 281474976710655
  %a = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %k42 = iconst.i64 42
  store.i64 %a, 16, %k42
  %abits = or.i64 %a, %tag
  %ta = bitcast.tagged.i64 %abits
  br loop(%z, %z)
loop(%i: i64, %acc: i64):
  %sz64 = iconst.i64 64
  %t = call.i64 @brass_gc_alloc(%sz64, %z, %kind)
  %u = bitcast.i64.tagged %ta
  %p = and.i64 %u, %mask
  %v = load.i64 %p, 16
  %a2 = add.i64 %acc, %v
  %one = iconst.i64 1
  %j = add.i64 %i, %one
  %more = slt.i64 %j, %n
  br_if %more, loop(%j, %a2), done(%a2)
done(%r: i64):
  ret %r
}
)";

std::unique_ptr<Module> parse_or_fail(const std::string& src) {
    DiagnosticReporter diag;
    auto mod = parse_module(src, &diag);
    if (!mod) std::cerr << "parse failed:\n" << diag.format_all() << "\n";
    REQUIRE(mod != nullptr);
    DiagnosticReporter vdiag;
    const bool ok = verify_module(*mod, &vdiag);
    if (!ok) std::cerr << "verify failed:\n" << vdiag.format_all() << "\n";
    REQUIRE(ok);
    return mod;
}

gc::HeapConfig tagged_heap_config() {
    gc::HeapConfig config = test::small_heap_config();
    config.reference_tags = {kRefTag};
    config.poison = true;
    return config;
}

TieringConfig no_tierup(bool fast) {
    TieringConfig cfg;
    cfg.invocation_tier1_threshold = 1000000;
    cfg.invocation_tier2_threshold = 1000000000;
    cfg.enable_background_compile = false;
    cfg.set_use_fast_interpreter(fast);
    return cfg;
}

// The sizes @tg_hold runs @tg_churn with: 1000 64-byte objects fill the
// small eden more than once.
constexpr int64_t kChurnSizes[] = {1, 100, 1000};

template <typename Interp>
void interpreter_case(gc::StressMode stress) {
    auto mod = parse_or_fail(kSrc);
    Interp interp(tagged_heap_config());
    interp.heap().set_stress(stress);
    gc::HeapScope bind(interp.heap());
    interp.set_module(mod.get());
    for (int64_t n : kChurnSizes) {
        const size_t before = interp.heap().collection_count();
        CHECK_EQ(interp.run(*mod->get_function("tg_hold"), {RuntimeValue::from_i64(n)}).as_i64(), kHoldExpected);
        if (n == 1000 || stress != gc::StressMode::None) CHECK(interp.heap().collection_count() > before);
        CHECK_EQ(interp.run(*mod->get_function("tg_keep_run"), {RuntimeValue::from_i64(n)}).as_i64(), n);
    }
}

void baseline_case(bool churn_in_baseline, gc::StressMode stress) {
    auto mod = parse_or_fail(kSrc);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(false));
    Interpreter interp(tagged_heap_config());
    interp.heap().set_stress(stress);
    gc::HeapScope bind(interp.heap());
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    REQUIRE(prog.pipeline().compile_and_install_tier1("tg_hold", mod->get_function("tg_hold")));
    REQUIRE(prog.pipeline().compile_and_install_tier1("tg_check", mod->get_function("tg_check")));
    REQUIRE(prog.pipeline().compile_and_install_tier1("tg_keep", mod->get_function("tg_keep")));
    if (churn_in_baseline) {
        REQUIRE(prog.pipeline().compile_and_install_tier1("tg_churn", mod->get_function("tg_churn")));
    }
    for (int64_t n : kChurnSizes) {
        const size_t before = interp.heap().collection_count();
        CHECK_EQ(interp.run(*mod->get_function("tg_hold"), {RuntimeValue::from_i64(n)}).as_i64(), kHoldExpected);
        if (n == 1000 || stress != gc::StressMode::None) CHECK(interp.heap().collection_count() > before);
        CHECK_EQ(interp.run(*mod->get_function("tg_keep_run"), {RuntimeValue::from_i64(n)}).as_i64(), n);
    }
}

// Called from generated code: a collection started from C++, with no
// generated caller frame handed to the heap.
int64_t tg_host_collect(int64_t n) {
    gc::Heap* heap = gc::Heap::current();
    if (heap == nullptr) return -1;
    heap->collect(gc::CollectionKind::Minor);
    if (n > 1) heap->collect(gc::CollectionKind::Full);
    return n;
}

size_t tagged_roots_in(const ModuleStackMap& maps, std::string_view fn) {
    const FunctionStackMap* m = maps.find_function_by_name(fn);
    if (m == nullptr) return 0;
    size_t n = 0;
    for (const StackMapRecord& rec : m->records) {
        for (const StackMapRootLocation& root : rec.roots) n += root.value == StackMapValueKind::Tagged;
    }
    return n;
}

void tier2_case(gc::StressMode stress) {
    auto mod = parse_or_fail(kSrc);
    gc::Heap heap(tagged_heap_config());
    heap.set_stress(stress);
    gc::HeapScope bind(heap);
    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("tg_host_collect", reinterpret_cast<void*>(&tg_host_collect));
    REQUIRE(jit.compile_and_load(*mod));
    CHECK(tagged_roots_in(jit.stack_maps(), "tg_hold") > 0);
    // A tagged value no later instruction reads is dead at the call, and so
    // no root there, until keep_alive reads it after the call.
    CHECK(tagged_roots_in(jit.stack_maps(), "tg_keep") > 0);
    CHECK_EQ(tagged_roots_in(jit.stack_maps(), "tg_nokeep"), 0u);
    for (int64_t n : kChurnSizes) {
        const size_t before = heap.collection_count();
        CHECK_EQ(jit.invoke("tg_hold", {RuntimeValue::from_i64(n)}).as_i64(), kHoldExpected);
        if (n == 1000 || stress != gc::StressMode::None) CHECK(heap.collection_count() > before);
        CHECK_EQ(jit.invoke("tg_keep_run", {RuntimeValue::from_i64(n)}).as_i64(), n);
    }
}

FunctionHandle* install_tier2(FunctionDispatchTable& prog, Module& mod, const char* name) {
    prog.tiering().get_feedback(name).set_deopt_threshold(1000000);
    FunctionHandle* h = prog.get_or_create(name, mod.get_function(name));
    CodeInstaller installer(prog);
    CodeInstallResult res = installer.install_tier2(*h, mod, name);
    REQUIRE(res.success);
    return h;
}

void deopt_case(bool fast) {
    auto mod = parse_or_fail(kSrc);
    FunctionDispatchTable prog;
    prog.pipeline().initialize(no_tierup(fast));
    gc::Heap heap(tagged_heap_config());
    gc::HeapScope bind(heap);
    FunctionHandle* h = install_tier2(prog, *mod, "tg_spec");
    const size_t before = heap.collection_count();
    // The guard holds (g == 1), then fails: the tagged state value crosses
    // into the Tier-0 exit, which collects before reading through it.
    for (int32_t g : {1, 0}) {
        RuntimeValue r = h->call_native({RuntimeValue::from_i64(1000), RuntimeValue::from_i32(g)});
        CHECK_EQ(r.as_i64(), 42);
    }
    CHECK_EQ(prog.pipeline().tier2_deopts(), 1u);
    CHECK(heap.collection_count() > before);
}

} // namespace

TEST_CASE("Tagged roots - the reference interpreter updates tagged values and alloca.tagged words") {
    interpreter_case<Interpreter>(gc::StressMode::None);
    interpreter_case<Interpreter>(gc::StressMode::Minor);
    interpreter_case<Interpreter>(gc::StressMode::Alternate);
}

TEST_CASE("Tagged roots - the fast interpreter updates tagged registers and alloca.tagged words") {
    interpreter_case<FastInterpreter>(gc::StressMode::None);
    interpreter_case<FastInterpreter>(gc::StressMode::Minor);
    interpreter_case<FastInterpreter>(gc::StressMode::Alternate);
}

TEST_CASE("Tagged roots - a baseline frame reports tagged slots to a collection in a Tier-0 callee") {
    baseline_case(false, gc::StressMode::None);
    baseline_case(false, gc::StressMode::Minor);
}

TEST_CASE("Tagged roots - a baseline frame reports tagged slots to a collection in a baseline callee") {
    baseline_case(true, gc::StressMode::None);
    baseline_case(true, gc::StressMode::Minor);
    baseline_case(true, gc::StressMode::Alternate);
}

TEST_CASE("Tagged roots - tier-2 stack maps carry tagged roots and a collection updates them") {
    tier2_case(gc::StressMode::None);
    tier2_case(gc::StressMode::Minor);
    tier2_case(gc::StressMode::Alternate);
}

TEST_CASE("Tagged roots - a collection a host function starts walks the generated frames beneath it") {
    auto mod = parse_or_fail(kSrc);
    gc::HeapConfig config = tagged_heap_config();
    config.walk_stack_on_host_collection = true;
    gc::Heap heap(config);
    gc::HeapScope bind(heap);
    codegen::JitExecutionEngine jit(Target::host());
    jit.register_external_symbol("tg_host_collect", reinterpret_cast<void*>(&tg_host_collect));
    REQUIRE(jit.compile_and_load(*mod));
    for (int64_t n : {1, 2}) {
        const size_t before = heap.collection_count();
        CHECK_EQ(jit.invoke("tg_host", {RuntimeValue::from_i64(n)}).as_i64(), kHoldExpected);
        CHECK(heap.collection_count() > before);
    }
}

TEST_CASE("Tagged roots - a tagged deopt state value survives a collection in the Tier-0 exit (Interpreter)") {
    deopt_case(false);
}

TEST_CASE("Tagged roots - a tagged deopt state value survives a collection in the Tier-0 exit (FastInterpreter)") {
    deopt_case(true);
}

TEST_CASE("Tagged roots - a tagged loop live-in moves into OSR code and stays a root there") {
    auto mod = parse_or_fail(kSrc);
    const Function& fn = *mod->get_function("tg_osr");
    FunctionDispatchTable prog;
    TieringConfig cfg = no_tierup(true);
    cfg.invocation_tier2_threshold = 1000000;
    prog.pipeline().initialize(cfg);
    prog.osr().set_enabled(true);
    prog.osr().set_threshold(100);
    FastInterpreter interp(tagged_heap_config());
    gc::HeapScope bind(interp.heap());
    interp.set_dispatch_table(&prog);
    interp.set_module(mod.get());
    // The first run asks for the loop's OSR code; the second enters it and
    // collects there many times.
    CHECK_EQ(interp.run(fn, {RuntimeValue::from_i64(2000)}).as_i64(), 42 * 2000);
    CompilePool::shared().wait_owner(&prog.osr());
    const uint64_t migrations = prog.osr().total_osr_migrations();
    const size_t collections = interp.heap().collection_count();
    CHECK_EQ(interp.run(fn, {RuntimeValue::from_i64(20000)}).as_i64(), 42 * 20000);
    CHECK(prog.osr().total_osr_migrations() > migrations);
    CHECK(interp.heap().collection_count() > collections);
}

#endif

TEST_CASE("Tagged roots - stack maps encode and decode the value kind of each root") {
    ModuleStackMap maps;
    FunctionStackMap fn;
    fn.function_name = "f";
    fn.code_size = 64;
    StackMapRecord rec;
    rec.instruction_offset = 12;
    rec.frame_size = 48;
    rec.add_root(StackMapRootLocation::frame_slot(-16));
    rec.add_root(StackMapRootLocation::frame_slot(-24, StackMapValueKind::Tagged));
    // A slot holds one value: the same slot again, of either kind, is the
    // root already listed, and the collector visits it once.
    rec.add_root(StackMapRootLocation::frame_slot(-16, StackMapValueKind::Tagged));
    fn.add_record(rec);
    maps.add_function(std::move(fn));
    REQUIRE_EQ(maps.functions()[0].records[0].roots.size(), 2ULL);

    const std::vector<uint8_t> bytes = encode_stack_maps(maps);
    ModuleStackMap decoded;
    REQUIRE(decode_stack_maps_into(bytes, decoded));
    const auto& roots = decoded.functions()[0].records[0].roots;
    REQUIRE_EQ(roots.size(), 2ULL);
    CHECK(roots[0].value == StackMapValueKind::GcRef);
    CHECK_EQ(roots[0].offset_from_rbp, -16);
    CHECK(roots[1].value == StackMapValueKind::Tagged);
    CHECK_EQ(roots[1].offset_from_rbp, -24);

    // A value kind the format does not define is corrupt input.
    std::vector<uint8_t> corrupt = bytes;
    bool patched = false;
    for (size_t i = 0; i + 8 <= corrupt.size(); ++i) {
        int32_t off = 0;
        std::memcpy(&off, &corrupt[i], 4);
        if (off == -24 && corrupt[i + 7] == 1) {
            corrupt[i + 7] = 9;
            patched = true;
            break;
        }
    }
    REQUIRE(patched);
    ModuleStackMap rejected;
    CHECK(!decode_stack_maps_into(corrupt, rejected));
}

TEST_CASE("Tagged roots - the verifier keeps tagged values out of arithmetic and memory addressing") {
    const char* bad_arith = R"(module @v
func @f(%t: tagged) -> i64 {
b0:
  %one = iconst.i64 1
  %r = add.i64 %t, %one
  ret %r
}
)";
    const char* bad_load = R"(module @v
func @f(%t: tagged) -> i64 {
b0:
  %r = load.i64 %t, 0
  ret %r
}
)";
    const char* good = R"(module @v
func @f(%t: tagged) -> tagged {
b0:
  %u = bitcast.i64.tagged %t
  %one = iconst.i64 1
  %r = add.i64 %u, %one
  %b = bitcast.tagged.i64 %r
  ret %b
}
)";
    for (const char* src : {bad_arith, bad_load}) {
        DiagnosticReporter diag;
        auto mod = parse_module(src, &diag);
        if (mod) CHECK(!verify_module(*mod, &diag));
    }
    DiagnosticReporter diag;
    auto mod = parse_module(good, &diag);
    REQUIRE(mod != nullptr);
    CHECK(verify_module(*mod, &diag));
}
