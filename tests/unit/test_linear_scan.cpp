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

TEST_CASE("Linear Scan - GPR 3 Spilled Operands Distinct Scratch Registers") {
    LirFunction fn;
    fn.name = "test_gpr_ternary_spill_reload";
    fn.calling_conv = CallingConvention::for_target(Target::x64_linux());

    LirBlock* entry = fn.create_block("entry");

    // Force register pressure with 20 live GPR vregs so spills are guaranteed
    std::vector<VReg> vars;
    for (int i = 0; i < 20; ++i) {
        vars.push_back(fn.allocate_vreg(RegClass::GPR, 8));
        auto inst = std::make_unique<LirInst>(LirOpcode::Mov);
        inst->add_def(LirOperand::vreg(vars.back()));
        inst->add_use(LirOperand::imm(i + 1));
        entry->append_inst(std::move(inst));
    }

    VReg res = fn.allocate_vreg(RegClass::GPR, 8);
    auto ternary_inst = std::make_unique<LirInst>(LirOpcode::Add);
    ternary_inst->add_def(LirOperand::vreg(res));
    ternary_inst->add_use(LirOperand::vreg(vars[0]));
    ternary_inst->add_use(LirOperand::vreg(vars[1]));
    ternary_inst->add_use(LirOperand::vreg(vars[2]));
    entry->append_inst(std::move(ternary_inst));

    for (size_t i = 3; i < vars.size(); ++i) {
        auto dummy = std::make_unique<LirInst>(LirOpcode::Add);
        dummy->add_def(LirOperand::vreg(res));
        dummy->add_use(LirOperand::vreg(res));
        dummy->add_use(LirOperand::vreg(vars[i]));
        entry->append_inst(std::move(dummy));
    }

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    ret_inst->add_use(LirOperand::vreg(res));
    entry->append_inst(std::move(ret_inst));

    LivenessAnalysis liveness(fn);
    liveness.run();

    LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
    regalloc.allocate();
    CHECK(regalloc.num_spill_slots() > 0);

    // Verify all 3 uses on the ternary instruction have distinct physical registers
    LirInst* rewritten_ternary = nullptr;
    for (const auto& inst : entry->instructions) {
        if (inst->opcode == LirOpcode::Add && inst->uses.size() == 3) {
            rewritten_ternary = inst.get();
            break;
        }
    }
    REQUIRE(rewritten_ternary != nullptr);
    REQUIRE_EQ(rewritten_ternary->uses.size(), 3u);

    std::vector<PReg> used_pregs;
    for (const auto& use : rewritten_ternary->uses) {
        REQUIRE(use.is_preg());
        used_pregs.push_back(use.preg_val);
    }
    CHECK(used_pregs[0] != used_pregs[1]);
    CHECK(used_pregs[1] != used_pregs[2]);
    CHECK(used_pregs[0] != used_pregs[2]);
}

TEST_CASE("Linear Scan - XMM 3 Spilled Operands Distinct Scratch Registers") {
    LirFunction fn;
    fn.name = "test_xmm_ternary_spill_reload";
    fn.calling_conv = CallingConvention::for_target(Target::x64_linux());

    LirBlock* entry = fn.create_block("entry");

    // Force register pressure with 20 live XMM vregs (pool only has 12)
    std::vector<VReg> vars;
    for (int i = 0; i < 20; ++i) {
        vars.push_back(fn.allocate_vreg(RegClass::XMM, 8));
        auto inst = std::make_unique<LirInst>(LirOpcode::Movsd);
        inst->add_def(LirOperand::vreg(vars.back()));
        inst->add_use(LirOperand::slot(static_cast<int32_t>(i), 8));
        entry->append_inst(std::move(inst));
    }

    VReg res = fn.allocate_vreg(RegClass::XMM, 8);
    auto fma_inst = std::make_unique<LirInst>(LirOpcode::Vfmadd213sd);
    fma_inst->add_def(LirOperand::vreg(res));
    fma_inst->add_use(LirOperand::vreg(vars[0]));
    fma_inst->add_use(LirOperand::vreg(vars[1]));
    fma_inst->add_use(LirOperand::vreg(vars[2]));
    entry->append_inst(std::move(fma_inst));

    for (size_t i = 3; i < vars.size(); ++i) {
        auto dummy = std::make_unique<LirInst>(LirOpcode::Addsd);
        dummy->add_def(LirOperand::vreg(res));
        dummy->add_use(LirOperand::vreg(res));
        dummy->add_use(LirOperand::vreg(vars[i]));
        entry->append_inst(std::move(dummy));
    }

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    ret_inst->add_use(LirOperand::vreg(res));
    entry->append_inst(std::move(ret_inst));

    LivenessAnalysis liveness(fn);
    liveness.run();

    LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
    regalloc.allocate();
    CHECK(regalloc.num_spill_slots() > 0);

    LirInst* rewritten_fma = nullptr;
    for (const auto& inst : entry->instructions) {
        if (inst->opcode == LirOpcode::Vfmadd213sd) {
            rewritten_fma = inst.get();
            break;
        }
    }
    REQUIRE(rewritten_fma != nullptr);
    REQUIRE_EQ(rewritten_fma->uses.size(), 3u);

    std::vector<PReg> used_pregs;
    for (const auto& use : rewritten_fma->uses) {
        REQUIRE(use.is_preg());
        used_pregs.push_back(use.preg_val);
    }
    CHECK(used_pregs[0] != used_pregs[1]);
    CHECK(used_pregs[1] != used_pregs[2]);
    CHECK(used_pregs[0] != used_pregs[2]);
}

// A physical register written by one instruction and read by a later one
// holds its value across the instructions between them (the pre-RA
// scheduler puts independent work there: a call result in RAX before the
// copy out of it, an argument in RCX before the call). No interval covering
// that gap may take the register.
TEST_CASE("Linear Scan - physical register held between its write and read") {
    for (GPR held : {GPR::RAX, GPR::RCX, GPR::RDX}) {
        LirFunction fn;
        fn.name = "test_held_preg";
        fn.calling_conv = CallingConvention::win64();
        LirBlock* entry = fn.create_block("entry");

        auto def_held = std::make_unique<LirInst>(LirOpcode::Mov);
        def_held->add_def(LirOperand::preg_gpr(held, 8), FixedConstraint::gpr(held));
        def_held->add_use(LirOperand::imm(1));
        entry->append_inst(std::move(def_held));

        VReg t = fn.allocate_vreg(RegClass::GPR, 8);
        auto mabs = std::make_unique<LirInst>(LirOpcode::Movabs);
        mabs->add_def(LirOperand::vreg(t));
        mabs->add_use(LirOperand::imm(int64_t{4617315517961601024}, 8));
        entry->append_inst(std::move(mabs));

        VReg u = fn.allocate_vreg(RegClass::GPR, 8);
        auto add = std::make_unique<LirInst>(LirOpcode::Add);
        add->add_def(LirOperand::vreg(u));
        add->add_use(LirOperand::vreg(t));
        add->add_use(LirOperand::imm(3));
        entry->append_inst(std::move(add));

        VReg v = fn.allocate_vreg(RegClass::GPR, 8);
        auto copy_out = std::make_unique<LirInst>(LirOpcode::Mov);
        copy_out->add_def(LirOperand::vreg(v));
        copy_out->add_use(LirOperand::preg_gpr(held, 8), FixedConstraint::gpr(held));
        entry->append_inst(std::move(copy_out));

        VReg w = fn.allocate_vreg(RegClass::GPR, 8);
        auto sum = std::make_unique<LirInst>(LirOpcode::Add);
        sum->add_def(LirOperand::vreg(w));
        sum->add_use(LirOperand::vreg(v));
        sum->add_use(LirOperand::vreg(u));
        entry->append_inst(std::move(sum));

        auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
        ret_inst->add_use(LirOperand::vreg(w));
        entry->append_inst(std::move(ret_inst));

        LivenessAnalysis liveness(fn);
        liveness.run();
        LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
        regalloc.allocate();

        for (VReg gap : {t, u}) {
            const PReg p = fn.get_vreg_info(gap).assigned_preg;
            REQUIRE(p.is_valid());
            CHECK(p != PReg::gpr(held));
        }
    }
}

// The emitter builds Fabs's mask in R11 and XMM15/XMM14. Unless the rewritten
// instruction records them as clobbers, the post-RA scheduler hoists a spill
// reload into R11 above it and the reload is lost (fuzz seed 2914).
TEST_CASE("Linear Scan - Fabs records the emitter's scratch registers as clobbers") {
    LirFunction fn;
    fn.name = "test_fabs_scratch";
    fn.calling_conv = CallingConvention::win64();
    LirBlock* entry = fn.create_block("entry");

    VReg x = fn.allocate_vreg(RegClass::XMM, 8);
    auto def = std::make_unique<LirInst>(LirOpcode::Xorpd);
    def->add_def(LirOperand::vreg(x));
    def->add_use(LirOperand::vreg(x));
    def->add_use(LirOperand::vreg(x));
    entry->append_inst(std::move(def));

    auto fabs = std::make_unique<LirInst>(LirOpcode::Fabs64);
    fabs->add_def(LirOperand::vreg(x));
    fabs->add_use(LirOperand::vreg(x));
    entry->append_inst(std::move(fabs));

    auto ret_inst = std::make_unique<LirInst>(LirOpcode::Ret);
    ret_inst->add_use(LirOperand::vreg(x));
    entry->append_inst(std::move(ret_inst));

    LivenessAnalysis liveness(fn);
    liveness.run();
    LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
    regalloc.allocate();

    const LirInst* rewritten = nullptr;
    for (const auto& inst : entry->instructions) {
        if (inst->opcode == LirOpcode::Fabs64) rewritten = inst.get();
    }
    REQUIRE(rewritten != nullptr);
    CHECK_NE(rewritten->clobbered_gprs & reg_mask(GPR::R11), 0u);
    CHECK_NE(rewritten->clobbered_xmms & reg_mask(XMM::XMM15), 0u);
    CHECK_NE(rewritten->clobbered_xmms & reg_mask(XMM::XMM14), 0u);
}
