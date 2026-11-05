// GC references are split like any other value: across a GC point each one is
// in a callee-saved register or in its slot, piece by piece, and the stack map
// of the point names exactly the live ones in their slots there. A slot whose
// value is in a register at the point, or dead there, is no root of it: it may
// still hold an earlier value (a previous loop turn's, or another value that
// shared the slot). A register at the point is no root either: the
// collector's stack scan pins what it names (gc_contract.md 1.1, 2.5).
//
// The allocation cases check that rule at every GC point, against liveness
// computed here from the LIR before allocation, on x64 and aarch64. The run
// cases execute functions whose tagged values are split across collecting
// calls, in a loop that redefines them every turn and carries one round the
// back edge, under every stress mode: a stale root reads the heap's poison.
#include "test_framework.hpp"
#include "gc_test_heap.hpp"
#include <brass/brass.hpp>
#include <brass/mir/parser.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/target/aarch64/aarch64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/gc/heap.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/stack_map.hpp>
#include <map>
#include <set>
#include <string>

using namespace brass;
using namespace brass::codegen;

namespace {

constexpr int kRefs = 8;

// @sp_loop: each turn allocates kRefs objects (payload word i + k + 1) and
// holds them as tagged values across three collecting calls, reading the
// first half between the first two, all of them between the second and the
// third, and the second half after the third, so a value is in a register at
// one call and in its slot at another. %carry is a tagged block parameter:
// the turn's first object, read after the next turn's first call. Each read
// adds the payload, plus 100 where the reference kept its tag in phase B.
std::string loop_source() {
    std::string s = R"(module @sp
func @sp_churn(%n: i64) -> i64 {
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
func @sp_loop(%n: i64, %turns: i64) -> i64 {
entry:
  %sz = iconst.i64 32
  %z = iconst.i64 0
  %kind = iconst.i32 2
  %tag = iconst.i64 -4222124650659840
  %c0 = call.i64 @brass_gc_alloc(%sz, %z, %kind)
  %cb = or.i64 %c0, %tag
  %ct = bitcast.tagged.i64 %cb
  br loop(%z, %z, %ct)
loop(%i: i64, %acc: i64, %carry: tagged):
  %mask = iconst.i64 281474976710655
  %hundred = iconst.i64 100
  %tag2 = iconst.i64 -4222124650659840
  %sz2 = iconst.i64 32
  %kind2 = iconst.i32 2
)";
    for (int k = 0; k < kRefs; ++k) {
        const std::string n = std::to_string(k);
        s += "  %a" + n + " = call.i64 @brass_gc_alloc(%sz2, %z, %kind2)\n";
        s += "  %kk" + n + " = iconst.i64 " + std::to_string(k + 1) + "\n";
        s += "  %kv" + n + " = add.i64 %i, %kk" + n + "\n";
        s += "  store.i64 %a" + n + ", 16, %kv" + n + "\n";
        s += "  %b" + n + " = or.i64 %a" + n + ", %tag2\n";
        s += "  %t" + n + " = bitcast.tagged.i64 %b" + n + "\n";
    }
    std::string acc = "%acc";
    int seq = 0;
    // A read of %<v>: its payload added to the running sum, and 100 more
    // when `tag_check` and its tag survived.
    auto read = [&](const std::string& v, bool tag_check) {
        const std::string n = std::to_string(seq++);
        s += "  %u" + n + " = bitcast.i64.tagged " + v + "\n";
        s += "  %p" + n + " = and.i64 %u" + n + ", %mask\n";
        s += "  %v" + n + " = load.i64 %p" + n + ", 16\n";
        s += "  %x" + n + " = add.i64 " + acc + ", %v" + n + "\n";
        acc = "%x" + n;
        if (!tag_check) return;
        s += "  %h" + n + " = xor.i64 %u" + n + ", %p" + n + "\n";
        s += "  %ok" + n + " = eq.i64 %h" + n + ", %tag2\n";
        s += "  %c" + n + " = select.i64 %ok" + n + ", %hundred, %z\n";
        s += "  %y" + n + " = add.i64 " + acc + ", %c" + n + "\n";
        acc = "%y" + n;
    };
    s += "  %r1 = call.i64 @sp_churn(%n)\n";
    read("%carry", false);
    for (int k = 0; k < kRefs / 2; ++k) read("%t" + std::to_string(k), false);
    s += "  %r2 = call.i64 @sp_churn(%n)\n";
    for (int k = 0; k < kRefs; ++k) read("%t" + std::to_string(k), true);
    s += "  %r3 = call.i64 @sp_churn(%n)\n";
    for (int k = kRefs / 2; k < kRefs; ++k) read("%t" + std::to_string(k), false);
    s += "  %one = iconst.i64 1\n";
    s += "  %ni = add.i64 %i, %one\n";
    s += "  %more = slt.i64 %ni, %turns\n";
    s += "  br_if %more, loop(%ni, " + acc + ", %t0), done\n";
    s += "done:\n  ret " + acc + "\n}\n";
    return s;
}

int64_t loop_expected(int64_t turns) {
    int64_t acc = 0;
    for (int64_t i = 0; i < turns; ++i) {
        acc += i;  // %carry: the previous turn's first object, i - 1 + 1 (0 at entry)
        for (int k = 0; k < kRefs / 2; ++k) acc += i + k + 1;
        for (int k = 0; k < kRefs; ++k) acc += i + k + 1 + 100;
        for (int k = kRefs / 2; k < kRefs; ++k) acc += i + k + 1;
    }
    return acc;
}

// @sp_pad: two tagged values live into a landing pad across a collecting
// invoke, one more across it on the normal path only.
const char* kPad = R"(module @sp
func @sp_work(%n: i64) -> i64 {
entry:
  ret %n
}
func @sp_pad(%x: i64) -> i64 {
entry:
  %t = bitcast.tagged.i64 %x
  %one = iconst.i64 1
  %x1 = add.i64 %x, %one
  %t1 = bitcast.tagged.i64 %x1
  %x2 = add.i64 %x1, %one
  %t2 = bitcast.tagged.i64 %x2
  %r = invoke.i64 @sp_work(%x), ok, pad
ok:
  %u = bitcast.i64.tagged %t
  %u2 = bitcast.i64.tagged %t2
  %s = add.i64 %u, %r
  %s2 = add.i64 %s, %u2
  %c = call.i64 @sp_work(%s2)
  %u1 = bitcast.i64.tagged %t1
  %s3 = add.i64 %c, %u1
  ret %s3
pad:
  %e = landing_pad
  %v = bitcast.i64.tagged %t
  %v1 = bitcast.i64.tagged %t1
  %w = add.i64 %v, %v1
  %w2 = add.i64 %w, %e
  ret %w2
}
)";

struct Checked {
    size_t points = 0;         // GC points
    size_t slot_roots = 0;     // (point, value) pairs in a slot, and so roots
    size_t reg_held = 0;       // (point, value) pairs in a callee-saved register
    size_t split_values = 0;   // values in a slot at one point and a register at another
};

bool callee_saved(const PReg& r, bool aarch64) {
    if (!r.is_valid() || !r.is_gpr()) return false;
    if (aarch64) return CallingConvention::aapcs64().is_callee_saved(r.as_aarch64_gpr());
    return CallingConvention::sysv64().is_callee_saved(r.as_gpr());
}

// `name` lowered and allocated, then every GC point checked: each GC
// reference live across it (computed here, before allocation) is a root of
// it exactly when its location there is its slot, and otherwise is in a
// callee-saved register; nothing else is a root.
Checked check_roots(const std::string& text, const char* name, bool aarch64) {
    Checked out;
    auto mod = parse_module(text.c_str());
    REQUIRE(mod != nullptr);
    Function* fn = mod->get_function(name);
    REQUIRE(fn != nullptr);
    fn->rebuild_cfg_predecessors();
    std::unique_ptr<LirFunction> lir;
    if (aarch64) {
        brass::aarch64::AArch64ISel isel(Target::aarch64_linux(), CallingConvention::aapcs64());
        lir = isel.lower(*fn);
    } else {
        brass::x64::X64ISel isel(Target::x64_linux(), CallingConvention::sysv64());
        lir = isel.lower(*fn);
    }
    REQUIRE(lir != nullptr);
    lir->sort_blocks_rpo();
    LivenessAnalysis liveness(*lir);
    liveness.run();

    // Per GC point (instruction id), the GC references live across it.
    std::map<uint32_t, std::set<uint32_t>> live_at;
    auto is_ref = [&](VReg v) { return v.is_valid() && v.id < lir->vreg_table.size() && lir->vreg_table[v.id].vreg.is_gcref; };
    for (const auto& b : lir->blocks) {
        std::set<uint32_t> live;
        for (VReg v : liveness.block_liveness(b.get()).live_out) {
            if (is_ref(v)) live.insert(v.id);
        }
        for (auto it = b->instructions.rbegin(); it != b->instructions.rend(); ++it) {
            const LirInst& inst = **it;
            if ((inst.is_call() || inst.opcode == LirOpcode::Safepoint) && !inst.no_gc_point) {
                std::set<uint32_t> across = live;
                for (const auto& d : inst.defs) {
                    if (d.is_vreg()) across.erase(d.vreg_val.id);
                }
                live_at[inst.id] = across;
            }
            for (const auto& d : inst.defs) {
                if (d.is_vreg()) live.erase(d.vreg_val.id);
            }
            auto use = [&](VReg v) { if (is_ref(v)) live.insert(v.id); };
            for (const auto& d : inst.defs) {
                if (d.is_mem()) { use(d.mem_val.base_vreg); use(d.mem_val.index_vreg); }
            }
            for (const auto& u : inst.uses) {
                if (u.is_vreg()) use(u.vreg_val);
                else if (u.is_mem()) { use(u.mem_val.base_vreg); use(u.mem_val.index_vreg); }
            }
        }
    }

    LinearScanAllocator regalloc(*lir, liveness, aarch64 ? CallingConvention::aapcs64() : CallingConvention::sysv64());
    regalloc.allocate();

    std::map<uint32_t, int> kinds;  // per value: 1 slot seen, 2 register seen
    for (const auto& b : lir->blocks) {
        for (const auto& inst : b->instructions) {
            if (!(inst->is_call() || inst->opcode == LirOpcode::Safepoint) || inst->no_gc_point) continue;
            auto found = live_at.find(inst->id);
            REQUIRE(found != live_at.end());
            ++out.points;
            std::set<uint32_t> roots;
            for (VReg v : inst->live_gcrefs) {
                CHECK(roots.insert(v.id).second);
                CHECK(found->second.count(v.id) == 1u);
            }
            for (uint32_t id : found->second) {
                const VReg v = lir->vreg_table[id].vreg;
                const LirOperand at = regalloc.location_at(v, inst->id, 8);
                if (roots.count(id)) {
                    REQUIRE(at.is_spill_slot());
                    CHECK_EQ(at.spill_slot, lir->vreg_table[id].assigned_spill_slot);
                    CHECK(lir->vreg_table[id].is_spilled);
                    ++out.slot_roots;
                    kinds[id] |= 1;
                } else {
                    REQUIRE(at.is_preg());
                    CHECK(callee_saved(at.preg_val, aarch64));
                    ++out.reg_held;
                    kinds[id] |= 2;
                }
            }
        }
    }
    for (const auto& [id, k] : kinds) out.split_values += k == 3;
    return out;
}

gc::HeapConfig split_heap_config() {
    gc::HeapConfig config = test::small_heap_config();
    config.reference_tags = {0xFFF1};
    config.poison = true;
    return config;
}

} // namespace

TEST_CASE("Split roots - a GC point's roots are the live references in their slots there") {
    const std::string src = loop_source();
    for (bool aarch64 : {false, true}) {
        const Checked c = check_roots(src, "sp_loop", aarch64);
        CHECK(c.points >= static_cast<size_t>(kRefs + 4));
        CHECK(c.slot_roots >= 1u);
        CHECK(c.reg_held >= 1u);
        // More references across the calls than callee-saved registers, read
        // in different halves between them: some value is in its slot at one
        // call and in a register at another.
        CHECK(c.split_values >= 1u);
    }
}

TEST_CASE("Split roots - references live into a landing pad keep one location, and it is a root only as a slot") {
    for (bool aarch64 : {false, true}) {
        const Checked c = check_roots(kPad, "sp_pad", aarch64);
        CHECK(c.points >= 2u);
        CHECK(c.slot_roots + c.reg_held >= 4u);
    }
}

#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)

TEST_CASE("Split roots - references split across collecting calls in a loop survive every stress mode") {
    const std::string src = loop_source();
    for (gc::StressMode stress : {gc::StressMode::None, gc::StressMode::Minor, gc::StressMode::Alternate,
                                  gc::StressMode::Incremental}) {
        auto mod = parse_module(src.c_str());
        REQUIRE(mod != nullptr);
        gc::Heap heap(split_heap_config());
        heap.set_stress(stress);
        gc::HeapScope bind(heap);
        JitExecutionEngine jit(Target::host());
        REQUIRE(jit.compile_and_load(*mod));
        for (int64_t n : {int64_t{1}, int64_t{100}, int64_t{1000}}) {
            for (int64_t turns : {int64_t{1}, int64_t{3}, int64_t{17}}) {
                const size_t before = heap.collection_count();
                CHECK_EQ(jit.invoke("sp_loop", {RuntimeValue::from_i64(n), RuntimeValue::from_i64(turns)}).as_i64(),
                         loop_expected(turns));
                if (n == 1000 || stress != gc::StressMode::None) CHECK(heap.collection_count() > before);
            }
        }
    }
}

#endif
