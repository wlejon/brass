#include "test_framework.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/target/x64/x64_isel.hpp>
#include <brass/target/target.hpp>
#include <brass/target/aarch64/aarch64_registers.hpp>
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

TEST_CASE("Linear Scan - AArch64 32 Register Pool Allocation") {
    LirFunction fn;
    fn.name = "test_aarch64_pool";
    fn.calling_conv = CallingConvention::for_target(Target::aarch64_linux());

    LirBlock* entry = fn.create_block("entry");

    // Allocate 22 virtual registers
    std::vector<VReg> vars;
    for (int i = 0; i < 22; ++i) {
        vars.push_back(fn.allocate_vreg(RegClass::GPR, 8));
    }

    // Initialize all 22 registers
    for (size_t i = 0; i < 22; ++i) {
        auto inst = std::make_unique<LirInst>(LirOpcode::Mov);
        inst->add_def(LirOperand::vreg(vars[i]));
        inst->add_use(LirOperand::imm(static_cast<int64_t>(i + 1)));
        entry->append_inst(std::move(inst));
    }

    // Accumulate all 22 registers so they have overlapping live ranges
    VReg acc = fn.allocate_vreg(RegClass::GPR, 8);
    auto init_acc = std::make_unique<LirInst>(LirOpcode::Mov);
    init_acc->add_def(LirOperand::vreg(acc));
    init_acc->add_use(LirOperand::vreg(vars[0]));
    entry->append_inst(std::move(init_acc));

    for (size_t i = 1; i < 22; ++i) {
        auto add_inst = std::make_unique<LirInst>(LirOpcode::Add);
        add_inst->add_def(LirOperand::vreg(acc));
        add_inst->add_use(LirOperand::vreg(acc));
        add_inst->add_use(LirOperand::vreg(vars[i]));
        entry->append_inst(std::move(add_inst));
    }

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    ret_inst->add_use(LirOperand::vreg(acc));
    entry->append_inst(std::move(ret_inst));

    LivenessAnalysis liveness(fn);
    liveness.run();

    LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
    regalloc.allocate();

    // Verify > 16 registers are successfully allocated without spilling (demonstrating 32-reg pool works)
    CHECK_EQ(regalloc.num_spill_slots(), size_t(0));
    CHECK_EQ(fn.frame.num_spill_slots, size_t(0));

    size_t allocated_pregs = 0;
    for (const auto& vinfo : fn.vreg_table) {
        if (!vinfo.is_spilled && vinfo.assigned_preg.is_valid()) {
            allocated_pregs++;
        }
    }
    CHECK(allocated_pregs > 16);
    CHECK_EQ(allocated_pregs, size_t(23));

    // Verify all rewritten operands have valid physical registers
    for (const auto& inst : fn.blocks[0]->instructions) {
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

    // Verify callee-saved registers (X19..X28) are tracked in fn.frame.saved_callee_gprs
    CHECK_NE(fn.frame.saved_callee_gprs, 0u);
    CHECK_NE(regalloc.used_callee_saved_gprs(), 0u);
    CHECK_EQ(fn.frame.saved_callee_gprs, regalloc.used_callee_saved_gprs());

    // Check that callee-saved registers X19..X28 were used (since 26 regs > 17 caller-saved)
    uint32_t callee_saved_range_mask = 0;
    for (int i = 19; i <= 28; ++i) {
        callee_saved_range_mask |= (1u << i);
    }
    CHECK_NE(fn.frame.saved_callee_gprs & callee_saved_range_mask, 0u);
}

TEST_CASE("Linear Scan - AArch64 FPR Pool Allocation") {
    LirFunction fn;
    fn.name = "test_aarch64_fpr_pool";
    fn.calling_conv = CallingConvention::for_target(Target::aarch64_linux());

    LirBlock* entry = fn.create_block("entry");

    // Allocate 25 virtual floating-point registers
    std::vector<VReg> vars;
    for (int i = 0; i < 25; ++i) {
        vars.push_back(fn.allocate_vreg(RegClass::XMM, 8));
    }

    // Initialize all 25 registers
    for (size_t i = 0; i < 25; ++i) {
        auto inst = std::make_unique<LirInst>(LirOpcode::Movsd);
        inst->add_def(LirOperand::vreg(vars[i]));
        inst->add_use(LirOperand::imm_f64(static_cast<double>(i + 1)));
        entry->append_inst(std::move(inst));
    }

    // Accumulate all 25 registers so they have overlapping live ranges
    VReg acc = fn.allocate_vreg(RegClass::XMM, 8);
    auto init_acc = std::make_unique<LirInst>(LirOpcode::Movsd);
    init_acc->add_def(LirOperand::vreg(acc));
    init_acc->add_use(LirOperand::vreg(vars[0]));
    entry->append_inst(std::move(init_acc));

    for (size_t i = 1; i < 25; ++i) {
        auto add_inst = std::make_unique<LirInst>(LirOpcode::Addsd);
        add_inst->add_def(LirOperand::vreg(acc));
        add_inst->add_use(LirOperand::vreg(acc));
        add_inst->add_use(LirOperand::vreg(vars[i]));
        entry->append_inst(std::move(add_inst));
    }

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    ret_inst->add_use(LirOperand::vreg(acc));
    entry->append_inst(std::move(ret_inst));

    LivenessAnalysis liveness(fn);
    liveness.run();

    LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
    regalloc.allocate();

    // Verify > 16 registers are successfully allocated without spilling
    CHECK_EQ(regalloc.num_spill_slots(), size_t(0));
    CHECK_EQ(fn.frame.num_spill_slots, size_t(0));

    size_t allocated_pregs = 0;
    for (const auto& vinfo : fn.vreg_table) {
        if (!vinfo.is_spilled && vinfo.assigned_preg.is_valid()) {
            allocated_pregs++;
        }
    }
    CHECK(allocated_pregs > 16);
    CHECK_EQ(allocated_pregs, size_t(26));

    // Caller-saved FPRs are V0..V7 (8) + V16..V29 (14) = 22.
    // With 26 live registers, at least 4 callee-saved FPRs (V8..V15) must be used.
    CHECK_NE(fn.frame.saved_callee_xmms, 0u);
    CHECK_NE(regalloc.used_callee_saved_xmms(), 0u);
    CHECK_EQ(fn.frame.saved_callee_xmms, regalloc.used_callee_saved_xmms());

    uint32_t callee_saved_fpr_mask = 0;
    for (int i = 8; i <= 15; ++i) {
        callee_saved_fpr_mask |= (1u << i);
    }
    CHECK_NE(fn.frame.saved_callee_xmms & callee_saved_fpr_mask, 0u);
}

TEST_CASE("Linear Scan - AArch64 Callee-Saved Register Across Calls") {
    LirFunction fn;
    fn.name = "test_aarch64_call_cross";
    fn.calling_conv = CallingConvention::for_target(Target::aarch64_linux());

    LirBlock* entry = fn.create_block("entry");

    VReg live_across = fn.allocate_vreg(RegClass::GPR, 8);
    auto init_live = std::make_unique<LirInst>(LirOpcode::Mov);
    init_live->add_def(LirOperand::vreg(live_across));
    init_live->add_use(LirOperand::imm(42));
    entry->append_inst(std::move(init_live));

    // Call instruction clobbering caller-saved registers
    auto call_inst = std::make_unique<LirInst>(LirOpcode::Call);
    call_inst->clobbered_gprs = fn.calling_conv.aarch64_caller_saved_gpr_mask();
    call_inst->clobbered_xmms = fn.calling_conv.aarch64_caller_saved_fpr_mask();
    entry->append_inst(std::move(call_inst));

    // Use live_across after the call
    VReg res = fn.allocate_vreg(RegClass::GPR, 8);
    auto use_inst = std::make_unique<LirInst>(LirOpcode::Add);
    use_inst->add_def(LirOperand::vreg(res));
    use_inst->add_use(LirOperand::vreg(live_across));
    use_inst->add_use(LirOperand::imm(10));
    entry->append_inst(std::move(use_inst));

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    ret_inst->add_use(LirOperand::vreg(res));
    entry->append_inst(std::move(ret_inst));

    LivenessAnalysis liveness(fn);
    liveness.run();

    LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
    regalloc.allocate();

    // live_across must be allocated to a callee-saved register without spilling
    const auto& info = fn.get_vreg_info(live_across);
    CHECK(!info.is_spilled);
    CHECK(info.assigned_preg.is_valid());
    CHECK(fn.calling_conv.is_callee_saved(info.assigned_preg.as_aarch64_gpr()));
    CHECK_NE(fn.frame.saved_callee_gprs & aarch64::reg_mask(info.assigned_preg.as_aarch64_gpr()), 0u);
}
