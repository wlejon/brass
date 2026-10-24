#include <brass/verify/arm64_lir_equiv.hpp>
#include "lir_sym_exec_arm64.hpp"
#include <brass/codegen/lir_flags.hpp>
#include <brass/codegen/peephole.hpp>
#include <brass/target/aarch64/aarch64_emit.hpp>
#include <algorithm>
#include <sstream>
#include <unordered_map>

namespace brass::verify {

namespace {

using namespace brass::codegen;

BVNodeId compute_block_target_label(BVContext& ctx, const Arm64ExecState& st, uint32_t fallthrough_id) {
    BVNodeId def_node = ctx.make_const(st.has_uncond_branch ? st.uncond_target_label : fallthrough_id, 32);
    return st.has_cond_branch
        ? ctx.make_ite(st.cond_branch_taken, ctx.make_const(st.cond_target_label, 32), def_node)
        : def_node;
}

bool are_flags_live_out_of_block(const LirFunction& fn, size_t block_idx) noexcept {
    if (block_idx >= fn.blocks.size()) return false;
    for (const LirBlock* succ : fn.blocks[block_idx]->successors) {
        if (!succ) continue;
        for (const auto& inst : succ->instructions) {
            if (!inst) continue;
            if (lir_reads_flags(*inst, Arch::aarch64)) return true;
            if (lir_kills_flags(*inst, Arch::aarch64)) break;
        }
    }
    return false;
}

LirBlock clone_lir_block(const LirBlock& src) {
    LirBlock dst(src.id, src.name);
    dst.loop_depth = src.loop_depth;
    dst.is_landing_pad = src.is_landing_pad;
    for (const auto& inst : src.instructions) if (inst) dst.append_inst(std::make_unique<LirInst>(*inst));
    return dst;
}

} // namespace

LirEquivResult verify_arm64_lir_block_equivalence(
    const codegen::LirBlock& before, const codegen::LirBlock& after,
    uint32_t live_out_gpr_mask, uint32_t live_out_vec_mask,
    bool check_flags, uint32_t fallthrough_block_id) {

    BVContext ctx;
    Arm64InitEnv env;
    env.init_defaults(ctx);

    Arm64ExecState st_b = execute_arm64_block(ctx, before, env);
    if (!st_b.supported) return {EquivStatus::Unsupported, st_b.error_msg};
    Arm64ExecState st_a = execute_arm64_block(ctx, after, env);
    if (!st_a.supported) return {EquivStatus::Unsupported, st_a.error_msg};

    struct NamedCheck { std::string name; BVNodeId b_val; BVNodeId a_val; BVNodeId diff; };
    std::vector<NamedCheck> checks;

    for (uint8_t i = 0; i < 32; ++i) {
        if (live_out_gpr_mask & (1u << i)) {
            auto g = static_cast<aarch64::GPR>(i);
            std::string gname = (i == 31) ? "sp" : std::string(aarch64::to_string(g, aarch64::OperandSize::Xword));
            checks.push_back({std::move(gname), st_b.gpr[i], st_a.gpr[i], ctx.make_binary(BVOp::Ne, st_b.gpr[i], st_a.gpr[i], 1)});
        }
        if (live_out_vec_mask & (1u << i)) {
            std::string vn = "v" + std::to_string(i);
            checks.push_back({vn + ".lo", st_b.vec_lo[i], st_a.vec_lo[i], ctx.make_binary(BVOp::Ne, st_b.vec_lo[i], st_a.vec_lo[i], 1)});
            if (st_b.uses_vec128 || st_a.uses_vec128) {
                checks.push_back({vn + ".hi", st_b.vec_hi[i], st_a.vec_hi[i], ctx.make_binary(BVOp::Ne, st_b.vec_hi[i], st_a.vec_hi[i], 1)});
            }
        }
    }
    for (const auto& [vid, b_v] : st_b.vregs) {
        if (auto it = st_a.vregs.find(vid); it != st_a.vregs.end()) {
            checks.push_back({"vreg" + std::to_string(vid), b_v, it->second, ctx.make_binary(BVOp::Ne, b_v, it->second, 1)});
        }
    }

    auto check_slot_map = [&](const auto& map_b, const auto& map_a, const char* prefix, auto get_init) {
        std::vector<int32_t> keys;
        for (const auto& [k, _] : map_b) keys.push_back(k);
        for (const auto& [k, _] : map_a) if (std::find(keys.begin(), keys.end(), k) == keys.end()) keys.push_back(k);
        for (int32_t k : keys) {
            BVNodeId bv = map_b.count(k) ? map_b.at(k) : get_init(k), av = map_a.count(k) ? map_a.at(k) : get_init(k);
            checks.push_back({std::string(prefix) + "[" + std::to_string(k) + "]", bv, av, ctx.make_binary(BVOp::Ne, bv, av, 1)});
        }
    };
    check_slot_map(st_b.spill_slots, st_a.spill_slots, "spill", [&](int32_t s) { return env.get_spill_init(ctx, s); });
    check_slot_map(st_b.local_slots, st_a.local_slots, "local", [&](int32_t o) { return env.get_local_init(ctx, o); });

    auto eval_final_mem = [&](const std::vector<Arm64SymMemStore>& stores, BVNodeId addr, uint32_t sz) {
        BVNodeId v = env.read_init_mem(ctx, addr, sz);
        for (const auto& s : stores) {
            if (s.size == sz) v = ctx.make_ite(ctx.make_binary(BVOp::Eq, addr, s.addr, 1), s.val, v);
        }
        return v;
    };
    for (const auto* side : {&st_b.mem_stores, &st_a.mem_stores}) {
        for (const auto& ms : *side) {
            BVNodeId bv = eval_final_mem(st_b.mem_stores, ms.addr, ms.size);
            BVNodeId av = eval_final_mem(st_a.mem_stores, ms.addr, ms.size);
            checks.push_back({"mem_store", bv, av, ctx.make_binary(BVOp::Ne, bv, av, 1)});
        }
    }

    if (check_flags) {
        checks.push_back({"N", st_b.nzcv.n, st_a.nzcv.n, ctx.make_binary(BVOp::Xor, st_b.nzcv.n, st_a.nzcv.n, 1)});
        checks.push_back({"Z", st_b.nzcv.z, st_a.nzcv.z, ctx.make_binary(BVOp::Xor, st_b.nzcv.z, st_a.nzcv.z, 1)});
        checks.push_back({"C", st_b.nzcv.c, st_a.nzcv.c, ctx.make_binary(BVOp::Xor, st_b.nzcv.c, st_a.nzcv.c, 1)});
        checks.push_back({"V", st_b.nzcv.v, st_a.nzcv.v, ctx.make_binary(BVOp::Xor, st_b.nzcv.v, st_a.nzcv.v, 1)});
    }

    if (st_b.has_cond_branch || st_b.has_uncond_branch || st_a.has_cond_branch ||
        st_a.has_uncond_branch || fallthrough_block_id != UINT32_MAX) {
        BVNodeId tb = compute_block_target_label(ctx, st_b, fallthrough_block_id);
        BVNodeId ta = compute_block_target_label(ctx, st_a, fallthrough_block_id);
        checks.push_back({"branch_target", tb, ta, ctx.make_binary(BVOp::Ne, tb, ta, 1)});
    }

    BVNodeId total_diff = ctx.make_const(0, 1);
    for (const auto& c : checks) total_diff = ctx.make_binary(BVOp::Or, total_diff, c.diff, 1);

    SolveOutcome solved = solve_nonzero(ctx, total_diff, 10000);
    if (solved.status == SolveOutcome::Unsat) return {EquivStatus::ProvenEquivalent, {}};
    if (solved.status == SolveOutcome::Unknown) {
        return {EquivStatus::Unknown, "Solver budget exhausted in AArch64 LIR equivalence check"};
    }

    std::ostringstream diag;
    diag << "AArch64 LIR equivalence counterexample in block '" << before.name << "': ";
    for (const auto& c : checks) {
        if (ctx.eval_concrete(c.diff, solved.model) != 0) {
            diag << c.name << " mismatch (before=0x" << std::hex << ctx.eval_concrete(c.b_val, solved.model)
                 << ", after=0x" << ctx.eval_concrete(c.a_val, solved.model) << std::dec << ")";
            break;
        }
    }
    return {EquivStatus::Counterexample, diag.str()};
}

LirEquivResult verify_arm64_lir_function_equivalence(
    const codegen::LirFunction& before, const codegen::LirFunction& after) {

    if (before.blocks.size() != after.blocks.size()) {
        return {EquivStatus::Counterexample, "Block count mismatch between before and after LirFunction"};
    }
    for (size_t i = 0; i < before.blocks.size(); ++i) {
        uint32_t fallthrough_id = (i + 1 < before.blocks.size()) ? before.blocks[i + 1]->id : UINT32_MAX;
        bool check_flags = are_flags_live_out_of_block(before, i) || are_flags_live_out_of_block(after, i);
        LirEquivResult blk_res = verify_arm64_lir_block_equivalence(
            *before.blocks[i], *after.blocks[i],
            kArm64AllGprsMask, kArm64AllVecMask, check_flags, fallthrough_id);
        if (blk_res.status == EquivStatus::Counterexample || blk_res.status == EquivStatus::Unknown) return blk_res;
    }
    return {EquivStatus::ProvenEquivalent, {}};
}

LirEquivResult verify_arm64_peephole(const codegen::LirFunction& fn) {
    LirFunction copy;
    copy.name = fn.name;
    copy.return_type = fn.return_type;
    copy.calling_conv = fn.calling_conv;
    copy.vreg_table = fn.vreg_table;
    copy.frame = fn.frame;
    copy.resume_entries = fn.resume_entries;
    copy.reserved_gprs = fn.reserved_gprs;
    copy.forced_saved_gprs = fn.forced_saved_gprs;

    std::unordered_map<uint32_t, LirBlock*> id_map;
    for (const auto& blk : fn.blocks) {
        auto cloned = std::make_unique<LirBlock>(clone_lir_block(*blk));
        id_map[cloned->id] = cloned.get();
        copy.blocks.push_back(std::move(cloned));
    }
    for (size_t i = 0; i < fn.blocks.size(); ++i) {
        for (const LirBlock* succ : fn.blocks[i]->successors) {
            if (succ && id_map.count(succ->id)) link_blocks(*copy.blocks[i], *id_map[succ->id]);
        }
    }
    run_lir_peephole_optimizations(copy);
    return verify_arm64_lir_function_equivalence(fn, copy);
}

LirEquivResult verify_arm64_ccmp_chain(
    bool first_is_64, bool first_is_cmn,
    const codegen::LirOperand& first_lhs, const codegen::LirOperand& first_rhs,
    const Arm64CcmpStep& ccmp_step, aarch64::Condition final_cond,
    bool expect_and_conjunction, aarch64::Condition first_cond, aarch64::Condition second_cond) {

    BVContext ctx;
    Arm64InitEnv env;
    env.init_defaults(ctx);

    auto eval_op = [&](const LirOperand& op, uint32_t w) -> BVNodeId {
        if (op.is_preg() && op.preg_val.is_gpr()) {
            if (op.preg_val.code == static_cast<uint8_t>(aarch64::GPR::XZR)) return ctx.make_const(0, w);
            BVNodeId g = env.gpr_init[op.preg_val.code & 31u];
            return (w == 64) ? g : ctx.make_extract(g, 31, 0);
        }
        if (op.is_vreg()) {
            BVNodeId v = env.get_vreg_init(ctx, op.vreg_val.id);
            return (w == 64) ? v : ctx.make_extract(v, 31, 0);
        }
        if (op.is_imm_int()) return ctx.make_const(static_cast<uint64_t>(op.imm_int), w);
        return ctx.make_const(0, w);
    };

    uint32_t w1 = first_is_64 ? 64 : 32, w2 = ccmp_step.is_64 ? 64 : 32;
    BVNodeId a1 = eval_op(first_lhs, w1), b1 = eval_op(first_rhs, w1);
    BVNodeId a2 = eval_op(ccmp_step.lhs, w2), b2 = eval_op(ccmp_step.rhs, w2);
    Arm64NzcvState flags1{}, cmp2_flags{};
    if (first_is_cmn) set_arm64_adds_flags(ctx, flags1, a1, b1, ctx.make_binary(BVOp::Add, a1, b1, w1), w1);
    else              set_arm64_subs_flags(ctx, flags1, a1, b1, ctx.make_binary(BVOp::Sub, a1, b1, w1), w1);
    if (ccmp_step.is_cmn) set_arm64_adds_flags(ctx, cmp2_flags, a2, b2, ctx.make_binary(BVOp::Add, a2, b2, w2), w2);
    else                  set_arm64_subs_flags(ctx, cmp2_flags, a2, b2, ctx.make_binary(BVOp::Sub, a2, b2, w2), w2);

    BVNodeId c1_expected = eval_arm64_cond(ctx, flags1, first_cond);
    BVNodeId c2_expected = eval_arm64_cond(ctx, cmp2_flags, second_cond);
    BVNodeId spec_bool = expect_and_conjunction
        ? ctx.make_binary(BVOp::And, c1_expected, c2_expected, 1)
        : ctx.make_binary(BVOp::Or, c1_expected, c2_expected, 1);

    BVNodeId gate = eval_arm64_cond(ctx, flags1, ccmp_step.cond);
    Arm64NzcvState ccmp_out{};
    ccmp_out.n = ctx.make_ite(gate, cmp2_flags.n, ctx.make_const((ccmp_step.nzcv_imm >> 3) & 1u, 1));
    ccmp_out.z = ctx.make_ite(gate, cmp2_flags.z, ctx.make_const((ccmp_step.nzcv_imm >> 2) & 1u, 1));
    ccmp_out.c = ctx.make_ite(gate, cmp2_flags.c, ctx.make_const((ccmp_step.nzcv_imm >> 1) & 1u, 1));
    ccmp_out.v = ctx.make_ite(gate, cmp2_flags.v, ctx.make_const(ccmp_step.nzcv_imm & 1u, 1));

    BVNodeId ccmp_bool = eval_arm64_cond(ctx, ccmp_out, final_cond);
    BVNodeId diff = ctx.make_binary(BVOp::Xor, spec_bool, ccmp_bool, 1);

    SolveOutcome solved = solve_nonzero(ctx, diff, 10000);
    if (solved.status == SolveOutcome::Unsat) return {EquivStatus::ProvenEquivalent, {}};
    if (solved.status == SolveOutcome::Unknown) {
        return {EquivStatus::Unknown, "Solver budget exhausted in AArch64 CCMP equivalence check"};
    }

    std::ostringstream diag;
    diag << "AArch64 CCMP equivalence counterexample: expected="
         << ctx.eval_concrete(spec_bool, solved.model)
         << ", ccmp=" << ctx.eval_concrete(ccmp_bool, solved.model);
    return {EquivStatus::Counterexample, diag.str()};
}

} // namespace brass::verify
