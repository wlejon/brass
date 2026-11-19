#include "test_framework.hpp"
#include <brass/brass.hpp>
#include <brass/codegen/linear_scan.hpp>
#include <brass/codegen/live_range.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

using namespace brass;
using namespace brass::codegen;
using namespace brass::aarch64;

namespace {

void run_a64_regalloc(LirFunction& fn) {
    fn.calling_conv = CallingConvention::apple_aapcs64();
    fn.sort_blocks_rpo();
    LivenessAnalysis liveness(fn);
    liveness.run();
    LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
    regalloc.allocate();
}

LirInst* find_first_opcode(LirBlock& bb, LirOpcode op) {
    for (const auto& inst : bb.instructions) {
        if (inst && inst->opcode == op) return inst.get();
    }
    return nullptr;
}

size_t count_opcode(const LirBlock& bb, LirOpcode op) {
    size_t n = 0;
    for (const auto& inst : bb.instructions) {
        if (inst && inst->opcode == op) ++n;
    }
    return n;
}

bool has_scratch_save_pair(const LirBlock& bb, PReg reg, LirOpcode expected_op, uint8_t expected_size) {
    std::unordered_set<int32_t> saved_slots;
    for (const auto& inst : bb.instructions) {
        if (!inst || inst->opcode != expected_op || inst->defs.size() != 1 || inst->uses.size() != 1) continue;
        if (inst->defs[0].is_spill_slot() && inst->uses[0].is_preg() && inst->uses[0].preg_val == reg &&
            inst->defs[0].size == expected_size) {
            saved_slots.insert(inst->defs[0].spill_slot);
        } else if (inst->defs[0].is_preg() && inst->defs[0].preg_val == reg && inst->uses[0].is_spill_slot() &&
                   inst->defs[0].size == expected_size && saved_slots.count(inst->uses[0].spill_slot)) {
            return true;
        }
    }
    return false;
}

std::vector<VReg> add_call_blockers_gpr(LirFunction& fn, LirBlock* bb, size_t count) {
    std::vector<VReg> blockers;
    blockers.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        VReg c = fn.allocate_vreg(RegClass::GPR, 8);
        auto def = std::make_unique<LirInst>(LirOpcode::Mov);
        def->add_def(LirOperand::vreg(c, 8));
        def->add_use(LirOperand::imm(static_cast<int64_t>(100 + i), 8));
        bb->append_inst(std::move(def));
        blockers.push_back(c);
    }
    return blockers;
}

void emit_ext_call(LirFunction& fn, LirBlock* bb, const char* sym) {
    auto call = std::make_unique<LirInst>(LirOpcode::Call);
    call->add_use(LirOperand::symbol(sym));
    call->clobbered_gprs = fn.calling_conv.caller_saved_gpr_mask();
    call->clobbered_xmms = fn.calling_conv.caller_saved_xmm_mask();
    bb->append_inst(std::move(call));
}

void consume_gprs_twice(LirFunction& fn, LirBlock* bb, const std::vector<VReg>& vs) {
    VReg acc = fn.allocate_vreg(RegClass::GPR, 8);
    auto init = std::make_unique<LirInst>(LirOpcode::Mov);
    init->add_def(LirOperand::vreg(acc, 8));
    init->add_use(LirOperand::imm(0, 8));
    bb->append_inst(std::move(init));
    for (int pass = 0; pass < 2; ++pass) {
        for (VReg v : vs) {
            auto add = std::make_unique<LirInst>(LirOpcode::Add);
            add->add_def(LirOperand::vreg(acc, 8));
            add->add_use(LirOperand::vreg(acc, 8));
            add->add_use(LirOperand::vreg(v, 8));
            bb->append_inst(std::move(add));
        }
    }
    auto sink = std::make_unique<LirInst>(LirOpcode::Mov);
    sink->add_def(LirOperand::preg(PReg::aarch64_gpr(GPR::X0), 8), FixedConstraint::aarch64_gpr(GPR::X0));
    sink->add_use(LirOperand::vreg(acc, 8));
    bb->append_inst(std::move(sink));
}

} // namespace

// 1. Reclaimed caller-saved pools: X0..X13 (14 GPR) and V0..V7, V16..V27 (20 FPR)
TEST_CASE("AArch64 Phase 4 - Reclaimed X12/X13 and V26/V27 caller-saved pools") {
    // 14 simultaneous caller-saved GPRs use X0..X13 with 0 spills and 0 callee-saved
    {
        LirFunction fn;
        fn.name = "a64_14_caller_gpr";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");

        std::vector<VReg> xs(14);
        for (size_t i = 0; i < xs.size(); ++i) {
            xs[i] = fn.allocate_vreg(RegClass::GPR, 8);
            auto def = std::make_unique<LirInst>(LirOpcode::Mov);
            def->add_def(LirOperand::vreg(xs[i], 8));
            def->add_use(LirOperand::imm(static_cast<int64_t>(i + 1), 8));
            bb->append_inst(std::move(def));
        }
        for (size_t i = 1; i < xs.size(); ++i) {
            auto add = std::make_unique<LirInst>(LirOpcode::Add);
            add->add_def(LirOperand::vreg(xs[0], 8));
            add->add_use(LirOperand::vreg(xs[0], 8));
            add->add_use(LirOperand::vreg(xs[i], 8));
            bb->append_inst(std::move(add));
        }
        auto out = std::make_unique<LirInst>(LirOpcode::Mov);
        out->add_def(LirOperand::preg(PReg::aarch64_gpr(GPR::X0), 8), FixedConstraint::aarch64_gpr(GPR::X0));
        out->add_use(LirOperand::vreg(xs[0], 8));
        bb->append_inst(std::move(out));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        LivenessAnalysis liveness(fn);
        liveness.run();
        LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
        regalloc.allocate();

        CHECK_EQ(regalloc.num_spill_slots(), size_t(0));
        CHECK_EQ(regalloc.used_callee_saved_gprs(), uint32_t(0));
        bool saw_x12 = false;
        bool saw_x13 = false;
        for (VReg v : xs) {
            PReg p = fn.get_vreg_info(v).assigned_preg;
            REQUIRE(p.is_valid());
            CHECK(p.code <= 13);
            if (p == PReg::aarch64_gpr(GPR::X12)) saw_x12 = true;
            if (p == PReg::aarch64_gpr(GPR::X13)) saw_x13 = true;
        }
        CHECK(saw_x12);
        CHECK(saw_x13);
    }

    // 20 simultaneous caller-saved FPRs use V0..V7, V16..V27 with 0 spills and 0 callee-saved
    {
        LirFunction fn;
        fn.name = "a64_20_caller_fpr";
        fn.calling_conv = CallingConvention::apple_aapcs64();
        LirBlock* bb = fn.create_block("entry");

        std::vector<VReg> fs(20);
        for (size_t i = 0; i < fs.size(); ++i) {
            fs[i] = fn.allocate_vreg(RegClass::XMM, 8);
            auto def = std::make_unique<LirInst>(LirOpcode::Movsd);
            def->add_def(LirOperand::vreg(fs[i], 8));
            LirMem mem;
            mem.base_preg = PReg::aarch64_gpr(GPR::X0);
            mem.disp = static_cast<int32_t>(i * 8);
            def->add_use(LirOperand::mem_custom(mem, 8));
            bb->append_inst(std::move(def));
        }
        for (size_t i = 1; i < fs.size(); ++i) {
            auto add = std::make_unique<LirInst>(LirOpcode::Addsd);
            add->add_def(LirOperand::vreg(fs[0], 8));
            add->add_use(LirOperand::vreg(fs[0], 8));
            add->add_use(LirOperand::vreg(fs[i], 8));
            bb->append_inst(std::move(add));
        }
        auto out = std::make_unique<LirInst>(LirOpcode::Movsd);
        out->add_def(LirOperand::preg(PReg::aarch64_fpr(FPR::V0), 8), FixedConstraint::aarch64_fpr(FPR::V0));
        out->add_use(LirOperand::vreg(fs[0], 8));
        bb->append_inst(std::move(out));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        LivenessAnalysis liveness(fn);
        liveness.run();
        LinearScanAllocator regalloc(fn, liveness, fn.calling_conv);
        regalloc.allocate();

        CHECK_EQ(regalloc.num_spill_slots(), size_t(0));
        CHECK_EQ(regalloc.used_callee_saved_xmms(), uint32_t(0));
        bool saw_v26 = false;
        bool saw_v27 = false;
        for (VReg v : fs) {
            PReg p = fn.get_vreg_info(v).assigned_preg;
            REQUIRE(p.is_valid());
            if (p == PReg::aarch64_fpr(FPR::V26)) saw_v26 = true;
            if (p == PReg::aarch64_fpr(FPR::V27)) saw_v27 = true;
        }
        CHECK(saw_v26);
        CHECK(saw_v27);
    }
}

// 2. 4-operand all-spilled Madd & Vfmadd213sd with live X12/X13 and V26/V27 (f64 + v128)
TEST_CASE("AArch64 Phase 4 - 4-operand all-spilled Madd and FMA with live X12/X13 and V26/V27") {
    // GPR: 4-operand Madd (1 spilled def + 3 spilled uses) while X12 and X13 hold live values
    {
        LirFunction fn;
        fn.name = "a64_4op_madd_save";
        LirBlock* bb = fn.create_block("entry");

        auto blockers = add_call_blockers_gpr(fn, bb, 10);
        std::array<VReg, 3> ops{};
        for (size_t i = 0; i < ops.size(); ++i) {
            ops[i] = fn.allocate_vreg(RegClass::GPR, 8);
            auto def = std::make_unique<LirInst>(LirOpcode::Mov);
            def->add_def(LirOperand::vreg(ops[i], 8));
            def->add_use(LirOperand::imm(static_cast<int64_t>(3 + i), 8));
            bb->append_inst(std::move(def));
        }
        emit_ext_call(fn, bb, "ext_fn");

        // Occupy all 14 caller-saved GPRs X0..X13 across Madd (blockers occupy X19..X28)
        std::vector<VReg> caller_live(14);
        for (size_t i = 0; i < caller_live.size(); ++i) {
            caller_live[i] = fn.allocate_vreg(RegClass::GPR, 8);
            auto def = std::make_unique<LirInst>(LirOpcode::Mov);
            def->add_def(LirOperand::vreg(caller_live[i], 8));
            def->add_use(LirOperand::imm(static_cast<int64_t>(200 + i), 8));
            bb->append_inst(std::move(def));
        }

        VReg dst = fn.allocate_vreg(RegClass::GPR, 8);
        auto madd = std::make_unique<LirInst>(LirOpcode::Madd);
        madd->add_def(LirOperand::vreg(dst, 8));
        madd->add_use(LirOperand::vreg(ops[0], 8));
        madd->add_use(LirOperand::vreg(ops[1], 8));
        madd->add_use(LirOperand::vreg(ops[2], 8));
        bb->append_inst(std::move(madd));

        // Consume all 14 caller-saved GPRs in 2 passes before call2 so none spill
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t i = 1; i < caller_live.size(); ++i) {
                auto a = std::make_unique<LirInst>(LirOpcode::Add);
                a->add_def(LirOperand::vreg(caller_live[0], 8));
                a->add_use(LirOperand::vreg(caller_live[0], 8));
                a->add_use(LirOperand::vreg(caller_live[i], 8));
                bb->append_inst(std::move(a));
            }
        }

        // Sink caller_live[0] to memory before call2 so none of caller_live[0..13] span call2
        LirMem sink_mem;
        sink_mem.base_vreg = blockers[0];
        auto pre_call_st = std::make_unique<LirInst>(LirOpcode::Mov);
        pre_call_st->add_def(LirOperand::mem_custom(sink_mem, 8));
        pre_call_st->add_use(LirOperand::vreg(caller_live[0], 8));
        bb->append_inst(std::move(pre_call_st));

        emit_ext_call(fn, bb, "ext_fn");
        emit_ext_call(fn, bb, "ext_fn");
        // Keep blockers in X19..X28 (weight 2) while dst has single use and spills
        VReg acc = fn.allocate_vreg(RegClass::GPR, 8);
        auto init = std::make_unique<LirInst>(LirOpcode::Mov);
        init->add_def(LirOperand::vreg(acc, 8));
        init->add_use(LirOperand::imm(0, 8));
        bb->append_inst(std::move(init));
        for (int pass = 0; pass < 2; ++pass) {
            for (VReg v : blockers) {
                auto add = std::make_unique<LirInst>(LirOpcode::Add);
                add->add_def(LirOperand::vreg(acc, 8));
                add->add_use(LirOperand::vreg(acc, 8));
                add->add_use(LirOperand::vreg(v, 8));
                bb->append_inst(std::move(add));
            }
        }
        auto add_dst = std::make_unique<LirInst>(LirOpcode::Add);
        add_dst->add_def(LirOperand::vreg(acc, 8));
        add_dst->add_use(LirOperand::vreg(acc, 8));
        add_dst->add_use(LirOperand::vreg(dst, 8));
        bb->append_inst(std::move(add_dst));
        auto sink = std::make_unique<LirInst>(LirOpcode::Mov);
        sink->add_def(LirOperand::preg(PReg::aarch64_gpr(GPR::X0), 8), FixedConstraint::aarch64_gpr(GPR::X0));
        sink->add_use(LirOperand::vreg(acc, 8));
        bb->append_inst(std::move(sink));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        run_a64_regalloc(fn);

        const LirInst* rw = find_first_opcode(*bb, LirOpcode::Madd);
        REQUIRE(rw != nullptr);
        CHECK(rw->defs[0].is_preg());
        std::unordered_set<uint8_t> regs;
        regs.insert(rw->defs[0].preg_val.code);
        for (const auto& u : rw->uses) {
            REQUIRE(u.is_preg());
            regs.insert(u.preg_val.code);
        }
        CHECK_EQ(regs.size(), size_t(4));
        CHECK(regs.count(static_cast<uint8_t>(GPR::X14)));
        CHECK(regs.count(static_cast<uint8_t>(GPR::X15)));
        CHECK(regs.count(static_cast<uint8_t>(GPR::X12)));
        CHECK(regs.count(static_cast<uint8_t>(GPR::X13)));
        CHECK(has_scratch_save_pair(*bb, PReg::aarch64_gpr(GPR::X12), LirOpcode::Mov, 8));
        CHECK(has_scratch_save_pair(*bb, PReg::aarch64_gpr(GPR::X13), LirOpcode::Mov, 8));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto res = em.compile();
        CHECK(!res.code_buffer.bytes().empty());
    }

    // FPR: 4-operand Vfmadd213sd with live v128 vectors in V26/V27 -> 16-byte Movaps save/restore
    {
        LirFunction fn;
        fn.name = "a64_4op_fma_v128_save";
        LirBlock* bb = fn.create_block("entry");

        // Occupy callee-saved V8..V15 across calls
        std::vector<VReg> fblockers;
        for (size_t i = 0; i < 8; ++i) {
            VReg c = fn.allocate_vreg(RegClass::XMM, 8);
            auto def = std::make_unique<LirInst>(LirOpcode::Xorpd);
            def->add_def(LirOperand::vreg(c, 8));
            def->add_use(LirOperand::vreg(c, 8));
            def->add_use(LirOperand::vreg(c, 8));
            bb->append_inst(std::move(def));
            fblockers.push_back(c);
        }
        std::array<VReg, 3> u{};
        for (size_t i = 0; i < u.size(); ++i) {
            u[i] = fn.allocate_vreg(RegClass::XMM, 8);
            auto def = std::make_unique<LirInst>(LirOpcode::Xorpd);
            def->add_def(LirOperand::vreg(u[i], 8));
            def->add_use(LirOperand::vreg(u[i], 8));
            def->add_use(LirOperand::vreg(u[i], 8));
            bb->append_inst(std::move(def));
        }
        emit_ext_call(fn, bb, "ext_fn");

        // Occupy all 20 caller-saved FPRs V0..V7, V16..V27 with 16-byte v128 values
        std::vector<VReg> vec_live(20);
        for (size_t i = 0; i < vec_live.size(); ++i) {
            vec_live[i] = fn.allocate_vreg(RegClass::XMM, 16);
            auto def = std::make_unique<LirInst>(LirOpcode::Movaps);
            def->add_def(LirOperand::vreg(vec_live[i], 16));
            LirMem mem;
            mem.base_preg = PReg::aarch64_gpr(GPR::X0);
            mem.disp = static_cast<int32_t>(i * 16);
            def->add_use(LirOperand::mem_custom(mem, 16));
            bb->append_inst(std::move(def));
        }

        VReg fdst = fn.allocate_vreg(RegClass::XMM, 8);
        auto fma = std::make_unique<LirInst>(LirOpcode::Vfmadd213sd);
        fma->add_def(LirOperand::vreg(fdst, 8));
        fma->add_use(LirOperand::vreg(u[0], 8));
        fma->add_use(LirOperand::vreg(u[1], 8));
        fma->add_use(LirOperand::vreg(u[2], 8));
        bb->append_inst(std::move(fma));

        // Consume all 20 caller-saved vectors in 2 passes before call2 so none spill
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t i = 1; i < vec_live.size(); ++i) {
                auto addp = std::make_unique<LirInst>(LirOpcode::Paddd);
                addp->add_def(LirOperand::vreg(vec_live[0], 16));
                addp->add_use(LirOperand::vreg(vec_live[0], 16));
                addp->add_use(LirOperand::vreg(vec_live[i], 16));
                bb->append_inst(std::move(addp));
            }
        }

        emit_ext_call(fn, bb, "ext_fn");
        emit_ext_call(fn, bb, "ext_fn");
        VReg facc = fn.allocate_vreg(RegClass::XMM, 8);
        auto finit = std::make_unique<LirInst>(LirOpcode::Xorpd);
        finit->add_def(LirOperand::vreg(facc, 8));
        finit->add_use(LirOperand::vreg(facc, 8));
        finit->add_use(LirOperand::vreg(facc, 8));
        bb->append_inst(std::move(finit));
        for (int pass = 0; pass < 2; ++pass) {
            for (VReg v : fblockers) {
                auto add = std::make_unique<LirInst>(LirOpcode::Addsd);
                add->add_def(LirOperand::vreg(facc, 8));
                add->add_use(LirOperand::vreg(facc, 8));
                add->add_use(LirOperand::vreg(v, 8));
                bb->append_inst(std::move(add));
            }
        }
        auto fadd_dst = std::make_unique<LirInst>(LirOpcode::Addsd);
        fadd_dst->add_def(LirOperand::vreg(facc, 8));
        fadd_dst->add_use(LirOperand::vreg(facc, 8));
        fadd_dst->add_use(LirOperand::vreg(fdst, 8));
        bb->append_inst(std::move(fadd_dst));
        auto fout = std::make_unique<LirInst>(LirOpcode::Movsd);
        fout->add_def(LirOperand::preg(PReg::aarch64_fpr(FPR::V0), 8), FixedConstraint::aarch64_fpr(FPR::V0));
        fout->add_use(LirOperand::vreg(facc, 8));
        bb->append_inst(std::move(fout));
        bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

        run_a64_regalloc(fn);

        const LirInst* rw = find_first_opcode(*bb, LirOpcode::Vfmadd213sd);
        REQUIRE(rw != nullptr);
        std::unordered_set<uint8_t> fregs;
        fregs.insert(rw->defs[0].preg_val.code);
        for (const auto& op : rw->uses) {
            REQUIRE(op.is_preg());
            fregs.insert(op.preg_val.code);
        }
        CHECK_EQ(fregs.size(), size_t(4));
        CHECK(fregs.count(static_cast<uint8_t>(FPR::V28)));
        CHECK(fregs.count(static_cast<uint8_t>(FPR::V29)));
        CHECK(fregs.count(static_cast<uint8_t>(FPR::V26)));
        CHECK(fregs.count(static_cast<uint8_t>(FPR::V27)));
        // V26 & V27 hold 16-byte vectors -> ScratchSave must save/restore 16-byte Movaps!
        CHECK(has_scratch_save_pair(*bb, PReg::aarch64_fpr(FPR::V26), LirOpcode::Movaps, 16));
        CHECK(has_scratch_save_pair(*bb, PReg::aarch64_fpr(FPR::V27), LirOpcode::Movaps, 16));

        AArch64EmitContext em(fn, Target::aarch64_macos());
        auto res = em.compile();
        CHECK(!res.code_buffer.bytes().empty());
    }
}

// 3. Spilled indexed store [base_spilled + idx_spilled] <- val_spilled with live X12
TEST_CASE("AArch64 Phase 4 - Spilled indexed store with live X12") {
    LirFunction fn;
    fn.name = "a64_spilled_idx_store";
    LirBlock* bb = fn.create_block("entry");

    auto blockers = add_call_blockers_gpr(fn, bb, 10);
    VReg base_v = fn.allocate_vreg(RegClass::GPR, 8);
    VReg idx_v = fn.allocate_vreg(RegClass::GPR, 8);
    VReg val_v = fn.allocate_vreg(RegClass::GPR, 8);
    for (VReg v : {base_v, idx_v, val_v}) {
        auto d = std::make_unique<LirInst>(LirOpcode::Mov);
        d->add_def(LirOperand::vreg(v, 8));
        d->add_use(LirOperand::imm(8, 8));
        bb->append_inst(std::move(d));
    }
    emit_ext_call(fn, bb, "ext_fn");
    // Read blockers right after call1 so their next-use precedes base_v/idx_v/val_v (staying in X19..X28)
    for (size_t i = 1; i < blockers.size(); ++i) {
        auto u = std::make_unique<LirInst>(LirOpcode::Add);
        u->add_def(LirOperand::vreg(blockers[0], 8));
        u->add_use(LirOperand::vreg(blockers[0], 8));
        u->add_use(LirOperand::vreg(blockers[i], 8));
        bb->append_inst(std::move(u));
    }

    std::vector<VReg> caller_live(14);
    for (size_t i = 0; i < caller_live.size(); ++i) {
        caller_live[i] = fn.allocate_vreg(RegClass::GPR, 8);
        auto d = std::make_unique<LirInst>(LirOpcode::Mov);
        d->add_def(LirOperand::vreg(caller_live[i], 8));
        d->add_use(LirOperand::imm(static_cast<int64_t>(50 + i), 8));
        bb->append_inst(std::move(d));
    }

    auto st = std::make_unique<LirInst>(LirOpcode::Mov);
    LirMem mem;
    mem.base_vreg = base_v;
    mem.index_vreg = idx_v;
    mem.scale = x64::Scale::Eight;
    st->add_def(LirOperand::mem_custom(mem, 8));
    st->add_use(LirOperand::vreg(val_v, 8));
    bb->append_inst(std::move(st));

    for (size_t i = 1; i < caller_live.size(); ++i) {
        auto a = std::make_unique<LirInst>(LirOpcode::Add);
        a->add_def(LirOperand::vreg(caller_live[0], 8));
        a->add_use(LirOperand::vreg(caller_live[0], 8));
        a->add_use(LirOperand::vreg(caller_live[i], 8));
        bb->append_inst(std::move(a));
    }
    blockers.push_back(caller_live[0]);
    consume_gprs_twice(fn, bb, blockers);
    bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    run_a64_regalloc(fn);

    // Base uses X14, index uses X15, value uses X12, and X12 gets ScratchSave pair
    CHECK(has_scratch_save_pair(*bb, PReg::aarch64_gpr(GPR::X12), LirOpcode::Mov, 8));
    bool found_indexed_store = false;
    for (const auto& inst : bb->instructions) {
        if (!inst || inst->opcode != LirOpcode::Mov || inst->defs.size() != 1 || inst->uses.size() != 1) continue;
        if (inst->defs[0].is_mem() && inst->uses[0].is_preg()) {
            const LirMem& m = inst->defs[0].mem_val;
            CHECK(m.base_preg == PReg::aarch64_gpr(GPR::X14));
            CHECK(m.index_preg == PReg::aarch64_gpr(GPR::X15));
            CHECK(inst->uses[0].preg_val == PReg::aarch64_gpr(GPR::X12));
            found_indexed_store = true;
        }
    }
    CHECK(found_indexed_store);

    AArch64EmitContext em(fn, Target::aarch64_macos());
    auto res = em.compile();
    CHECK(!res.code_buffer.bytes().empty());
}

// 4. End-to-end JIT runtime test: 26 live i64 + 28 live f64 + madd + indexed store + call
TEST_CASE("AArch64 Phase 4 - High-pressure JIT correctness (26 i64 + 28 f64 + madd + call)") {
    auto mod = std::make_unique<Module>("p4_stress");
    Function* helper = mod->create_function("helper", Type::i64(), {Type::i64()});
    {
        Builder hb(*mod);
        hb.set_function(helper);
        BasicBlock* entry = hb.append_block("entry");
        Value* x = hb.add_block_param(entry, Type::i64());
        hb.build_ret(hb.build_add(x, hb.build_iconst_i64(3)));
        helper->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*helper));
    }

    Function* stress = mod->create_function(
        "stress", Type::i64(), {Type::ptr(), Type::ptr(), Type::ptr(), Type::ptr(), Type::i64()});
    {
        Builder sb(*mod);
        sb.set_function(stress);
        BasicBlock* entry = sb.append_block("entry");
        Value* pi = sb.add_block_param(entry, Type::ptr());
        Value* pf = sb.add_block_param(entry, Type::ptr());
        Value* out_i = sb.add_block_param(entry, Type::ptr());
        Value* out_f = sb.add_block_param(entry, Type::ptr());
        Value* idx = sb.add_block_param(entry, Type::i64());

        std::array<Value*, 26> vi{};
        for (size_t i = 0; i < 26; ++i) {
            vi[i] = sb.build_load(Type::i64(), pi, static_cast<int32_t>(i * 8));
        }
        std::array<Value*, 28> vf{};
        for (size_t i = 0; i < 28; ++i) {
            vf[i] = sb.build_load(Type::f64(), pf, static_cast<int32_t>(i * 8));
        }

        Value* mid = sb.build_call("helper", Type::i64(), {idx});
        Value* ai = sb.build_add(mid, sb.build_mul(vi[0], vi[1]));
        Value* isum = nullptr;
        for (size_t i = 2; i < 26; i += 2) {
            Value* term = sb.build_add(ai, sb.build_mul(vi[i], vi[i + 1]));
            isum = (isum == nullptr) ? term : sb.build_add(isum, term);
        }
        Value* fsum = vf[0];
        for (size_t i = 1; i < 28; ++i) {
            fsum = sb.build_fadd(fsum, vf[i]);
        }

        sb.build_store_indexed(Type::i64(), out_i, idx, 8, isum);
        sb.build_store(Type::f64(), out_f, 0, fsum);
        sb.build_ret(isum);
        stress->rebuild_cfg_predecessors();
        REQUIRE(verify_function(*stress));
    }

    JitExecutionEngine jit;
    REQUIRE(jit.compile_and_load(*mod));
    auto fn = jit.get_function_ptr<int64_t (*)(const int64_t*, const double*, int64_t*, double*, int64_t)>("stress");
    REQUIRE(fn != nullptr);

    std::array<int64_t, 26> in_i{};
    std::array<double, 28> in_f{};
    for (size_t i = 0; i < 26; ++i) in_i[i] = static_cast<int64_t>(i) + 2;
    for (size_t i = 0; i < 28; ++i) in_f[i] = 1.5 * static_cast<double>(i + 1);
    std::array<int64_t, 8> out_i{};
    double out_f = 0.0;
    const int64_t idx = 3;

    int64_t got_i = fn(in_i.data(), in_f.data(), out_i.data(), &out_f, idx);
    int64_t ai = (idx + 3) + in_i[0] * in_i[1];
    int64_t expected_i = 0;
    for (size_t i = 2; i < 26; i += 2) {
        expected_i += ai + in_i[i] * in_i[i + 1];
    }
    double expected_f = 0.0;
    for (double v : in_f) expected_f += v;

    CHECK_EQ(got_i, expected_i);
    CHECK_EQ(out_i[3], expected_i);
    CHECK(std::abs(out_f - expected_f) < 1e-9);
}

// 5. Peephole reload reuse tracks X14, X12, V28, V29, V26 (leaving X15 untracked)
TEST_CASE("AArch64 Phase 4 - Peephole reload reuse on X14/X12/V28/V29/V26 (X15 untracked)") {
    LirFunction fn;
    fn.name = "a64_reload_reuse_p4";
    fn.calling_conv = CallingConvention::apple_aapcs64();
    fn.frame.num_spill_slots = 6;
    LirBlock* bb = fn.create_block("entry");

    auto emit_pair = [&](PReg r, PReg acc, LirOpcode use_op, int32_t slot, LirOpcode op, uint8_t sz) {
        for (int pass = 0; pass < 2; ++pass) {
            auto load = std::make_unique<LirInst>(op);
            load->add_def(LirOperand::preg(r, sz));
            load->add_use(LirOperand::slot(slot, sz));
            bb->append_inst(std::move(load));

            auto use = std::make_unique<LirInst>(use_op);
            use->add_def(LirOperand::preg(acc, sz));
            use->add_use(LirOperand::preg(acc, sz));
            use->add_use(LirOperand::preg(r, sz));
            bb->append_inst(std::move(use));
        }
    };

    const PReg g_acc = PReg::aarch64_gpr(GPR::X0);
    const PReg f_acc = PReg::aarch64_fpr(FPR::V0);
    emit_pair(PReg::aarch64_gpr(GPR::X14), g_acc, LirOpcode::Add, 0, LirOpcode::Mov, 8);
    emit_pair(PReg::aarch64_gpr(GPR::X12), g_acc, LirOpcode::Add, 1, LirOpcode::Mov, 8);
    emit_pair(PReg::aarch64_fpr(FPR::V28), f_acc, LirOpcode::Addsd, 2, LirOpcode::Movsd, 8);
    emit_pair(PReg::aarch64_fpr(FPR::V29), f_acc, LirOpcode::Addsd, 3, LirOpcode::Movsd, 8);
    emit_pair(PReg::aarch64_fpr(FPR::V26), f_acc, LirOpcode::Addsd, 4, LirOpcode::Movsd, 8);
    emit_pair(PReg::aarch64_gpr(GPR::X15), g_acc, LirOpcode::Add, 5, LirOpcode::Mov, 8);  // untracked
    bb->append_inst(std::make_unique<LirInst>(LirOpcode::Ret));

    PeepholeOptimizer opt(fn);
    PeepholeStats stats = opt.run();
    CHECK_EQ(stats.reloads_reused, size_t(5));

    // Only X15 keeps both of its loads; each tracked register drops to 1
    size_t remaining_loads = count_opcode(*bb, LirOpcode::Mov) + count_opcode(*bb, LirOpcode::Movsd);
    CHECK_EQ(remaining_loads, size_t(7));
}
