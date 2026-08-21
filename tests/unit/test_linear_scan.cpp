#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/linear_scan.hpp>

using namespace brass;
using namespace brass::codegen;
using namespace brass::x64;

TEST_CASE("Linear Scan - Liveness and Live Intervals") {
    Module mod;
    Function* fn = mod.create_function("test_liveness", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* t1 = b.build_add(a, c);
    Value* t2 = b.build_sub(t1, a);
    Value* t3 = b.build_mul(t2, c);
    b.build_ret(t3);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);
    CHECK(lir != nullptr);

    LivenessAnalysis liveness(*lir);
    liveness.run();

    // Check monotonic instruction IDs (step by 2)
    uint32_t expected_id = 0;
    for (const auto& inst : lir->blocks[0]->instructions) {
        CHECK_EQ(inst->id, expected_id);
        expected_id += 2;
    }

    // Check intervals exist
    const auto& intervals = liveness.intervals();
    CHECK(!intervals.empty());

    for (const auto& interval : intervals) {
        if (interval.vreg.is_valid() && !interval.use_positions.empty()) {
            CHECK(interval.start_id <= interval.end_id);
            CHECK(!interval.segments.empty());
        }
    }
}

TEST_CASE("Linear Scan - Register Allocation without Spills") {
    Module mod;
    Function* fn = mod.create_function("simple_arith", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* a = b.add_block_param(entry, Type::i64());
    Value* c = b.add_block_param(entry, Type::i64());
    Value* t1 = b.build_add(a, c);
    Value* t2 = b.build_mul(t1, c);
    b.build_ret(t2);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);

    LivenessAnalysis liveness(*lir);
    liveness.run();

    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::win64());
    regalloc.allocate();

    // No spill slots needed for simple function
    CHECK_EQ(regalloc.num_spill_slots(), size_t(0));

    // Verify all rewritten operands have valid physical registers
    for (const auto& inst : lir->blocks[0]->instructions) {
        for (const auto& def_op : inst->defs) {
            if (def_op.is_reg()) {
                CHECK(def_op.is_preg());
                CHECK(def_op.preg_val.is_valid());
            }
        }
        for (const auto& use_op : inst->uses) {
            if (use_op.is_reg()) {
                CHECK(use_op.is_preg());
                CHECK(use_op.preg_val.is_valid());
            }
        }
    }
}

TEST_CASE("Linear Scan - High Register Pressure with Spills") {
    Module mod;
    Function* fn = mod.create_function("high_pressure", Type::i64(), {Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* x = b.add_block_param(entry, Type::i64());
    std::vector<Value*> vars;
    for (int i = 0; i < 25; ++i) {
        Value* c = b.build_iconst_i64(i + 1);
        Value* v = b.build_add(x, c);
        vars.push_back(v);
    }

    // Now combine all 25 values so all are simultaneously live
    Value* sum = vars[0];
    for (size_t i = 1; i < vars.size(); ++i) {
        sum = b.build_add(sum, vars[i]);
    }
    b.build_ret(sum);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);

    LivenessAnalysis liveness(*lir);
    liveness.run();

    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::win64());
    regalloc.allocate();

    // With 25 simultaneously live variables and ~14 allocatable GPRs, spills MUST occur
    CHECK(regalloc.num_spill_slots() > 0);
    CHECK(lir->frame.num_spill_slots > 0);
}

TEST_CASE("Linear Scan - GC Reference Tracking on Spill Slots") {
    Module mod;
    Function* fn = mod.create_function("gc_pressure", Type::gcref(), {Type::gcref()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* root = b.add_block_param(entry, Type::gcref());
    std::vector<Value*> gcrefs;
    for (int i = 0; i < 20; ++i) {
        Value* ref = b.build_load(Type::gcref(), root, i * 8);
        gcrefs.push_back(ref);
    }

    // Store all 20 gcrefs in reverse order so all are simultaneously live
    for (int i = 19; i >= 0; --i) {
        b.build_store(Type::gcref(), root, i * 8, gcrefs[i]);
    }
    b.build_ret(gcrefs[0]);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);

    LivenessAnalysis liveness(*lir);
    liveness.run();

    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::win64());
    regalloc.allocate();

    CHECK(lir->frame.num_spill_slots > 0);
    bool found_gcref_spill = false;
    for (bool is_gc : lir->frame.spill_slot_is_gcref) {
        if (is_gc) found_gcref_spill = true;
    }
    CHECK(found_gcref_spill);
}

TEST_CASE("Linear Scan - Callee-Saved Register Selection Across Calls") {
    Module mod;
    Function* fn = mod.create_function("call_cross", Type::i64(), {Type::i64(), Type::i64()});

    Builder b(mod);
    b.set_function(fn);

    BasicBlock* entry = b.append_block("entry");
    Value* val1 = b.add_block_param(entry, Type::i64());
    Value* val2 = b.add_block_param(entry, Type::i64());

    Value* res = b.build_call("external_func", Type::i64());

    Value* sum1 = b.build_add(val1, res);
    Value* sum2 = b.build_add(sum1, val2);
    b.build_ret(sum2);

    fn->rebuild_cfg_predecessors();
    CHECK(verify_function(*fn));

    X64ISel isel(Target::x64_windows(), CallingConvention::win64());
    auto lir = isel.lower(*fn);

    LivenessAnalysis liveness(*lir);
    liveness.run();

    for (const auto& interval : liveness.intervals()) {
        if (interval.vreg.is_valid() && (interval.vreg.id == 0 || interval.vreg.id == 1)) {
            CHECK(interval.spans_call);
        }
    }

    LinearScanAllocator regalloc(*lir, liveness, CallingConvention::win64());
    regalloc.allocate();

    CHECK_NE(regalloc.used_callee_saved_gprs(), 0);
}
