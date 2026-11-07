#include <brass/verify/smt_equiv.hpp>
#include "mir_sym_exec.hpp"
#include <brass/mir/function.hpp>
#include <brass/mir/block.hpp>
#include <brass/mir/instruction.hpp>
#include <brass/mir/opcodes.hpp>
#include <algorithm>
#include <cstring>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace brass::verify {

namespace {

RuntimeValue bits_to_rv(Type t, uint64_t bits) {
    switch (t.kind()) {
        case TypeKind::I8:     return RuntimeValue::from_i32(static_cast<int32_t>(bits & 0xFFu));
        case TypeKind::I16:    return RuntimeValue::from_i32(static_cast<int32_t>(bits & 0xFFFFu));
        case TypeKind::I32:    return RuntimeValue::from_i32(static_cast<int32_t>(static_cast<uint32_t>(bits)));
        case TypeKind::I64:    return RuntimeValue::from_i64(static_cast<int64_t>(bits));
        case TypeKind::F32: {
            float f = 0.0f;
            uint32_t b32 = static_cast<uint32_t>(bits);
            std::memcpy(&f, &b32, 4);
            return RuntimeValue::from_f32(f);
        }
        case TypeKind::F64: {
            double d = 0.0;
            std::memcpy(&d, &bits, 8);
            return RuntimeValue::from_f64(d);
        }
        case TypeKind::Ptr:    return RuntimeValue::from_ptr(static_cast<uintptr_t>(bits));
        case TypeKind::GCRef:  return RuntimeValue::from_gcref(static_cast<uintptr_t>(bits));
        case TypeKind::Tagged: return RuntimeValue::from_tagged(bits);
        default:               return RuntimeValue::from_void();
    }
}

void collect_rpo(const BasicBlock* bb,
                 std::unordered_set<const BasicBlock*>& visited,
                 std::vector<const BasicBlock*>& postorder) {
    if (!bb || !visited.insert(bb).second) return;
    if (const Instruction* term = bb->terminator()) {
        switch (term->opcode()) {
            case Opcode::br:
                collect_rpo(term->branch_target().block, visited, postorder);
                break;
            case Opcode::br_if:
                collect_rpo(term->false_target().block, visited, postorder);
                collect_rpo(term->true_target().block, visited, postorder);
                break;
            case Opcode::switch_:
                collect_rpo(term->default_target().block, visited, postorder);
                for (const auto& sc : term->switch_cases()) collect_rpo(sc.target.block, visited, postorder);
                break;
            default: break;
        }
    }
    postorder.push_back(bb);
}

const BasicBlock* find_single_loop_header(const Function& fn) {
    if (!fn.entry_block()) return nullptr;
    std::unordered_set<const BasicBlock*> visited;
    std::vector<const BasicBlock*> rpo;
    collect_rpo(fn.entry_block(), visited, rpo);
    std::reverse(rpo.begin(), rpo.end());

    std::unordered_map<const BasicBlock*, size_t> rpo_pos;
    for (size_t i = 0; i < rpo.size(); ++i) rpo_pos[rpo[i]] = i;

    const BasicBlock* header = nullptr;
    for (size_t i = 0; i < rpo.size(); ++i) {
        const BasicBlock* bb = rpo[i];
        for (const BasicBlock* succ : bb->successors()) {
            auto it = rpo_pos.find(succ);
            if (it != rpo_pos.end() && it->second <= i) {
                if (header == nullptr) {
                    header = succ;
                } else if (header != succ) {
                    return nullptr; // Multiple loop headers
                }
            }
        }
    }
    return header;
}

struct MergedCapture {
    BVNodeId cond = 0;
    std::vector<BVNodeId> args;
    std::unordered_map<const Value*, BVNodeId> vals;
};

MergedCapture merge_captured_edges(
    BVContext& ctx,
    const BasicBlock* header,
    const std::vector<SymCapturedEdge>& edges) {

    MergedCapture mc;
    mc.cond = ctx.make_const(0, 1);
    size_t np = header->param_count();
    mc.args.resize(np, 0);
    for (size_t k = 0; k < np; ++k) {
        uint32_t w = type_bv_width(header->param(k)->type());
        mc.args[k] = ctx.make_const(0, w ? w : 64);
    }
    for (size_t ei = 0; ei < edges.size(); ++ei) {
        BVNodeId ec = edges[ei].cond;
        mc.cond = ctx.make_binary(BVOp::Or, mc.cond, ec, 1);
        for (size_t k = 0; k < np && k < edges[ei].args.size(); ++k) {
            BVNodeId v = edges[ei].args[k];
            if (v == 0) continue;
            mc.args[k] = (ei == 0) ? v : ctx.make_ite(ec, v, mc.args[k]);
        }
        for (const auto& [val_ptr, node_id] : edges[ei].vals) {
            auto vit = mc.vals.find(val_ptr);
            if (vit == mc.vals.end()) {
                mc.vals[val_ptr] = node_id;
            } else if (vit->second != node_id) {
                vit->second = ctx.make_ite(ec, node_id, vit->second);
            }
        }
    }
    return mc;
}

BVNodeId build_region_obs_diff(
    BVContext& ctx,
    BVNodeId domain_cond,
    InitMemEnv& mem_env,
    const SymExecOutput& out_b,
    const SymExecOutput& out_a,
    Type ret_type,
    const EquivOptions& opts) {

    BVNodeId diff = ctx.make_const(0, 1);
    if (opts.check_guards) {
        BVNodeId g_ne = ctx.make_binary(BVOp::Xor, out_b.guard_failed, out_a.guard_failed, 1);
        diff = ctx.make_binary(BVOp::Or, diff, ctx.make_binary(BVOp::And, domain_cond, g_ne, 1), 1);

        for (const auto& gb : out_b.guard_exits) {
            for (const auto& ga : out_a.guard_exits) {
                BVNodeId pair_cond = ctx.make_binary(
                    BVOp::And, domain_cond,
                    ctx.make_binary(BVOp::And, gb.fail_cond, ga.fail_cond, 1), 1);
                if (ctx.nodes()[pair_cond].op == BVOp::Const && ctx.nodes()[pair_cond].imm == 0) continue;
                BVNodeId pair_bad = ctx.make_const(0, 1);
                if (gb.resume_id != ga.resume_id || gb.exit_symbol != ga.exit_symbol ||
                    gb.state_vals.size() != ga.state_vals.size()) {
                    pair_bad = pair_cond;
                } else {
                    for (size_t k = 0; k < gb.state_vals.size(); ++k) {
                        if (gb.state_types[k] != ga.state_types[k] ||
                            ctx.width(gb.state_vals[k]) != ctx.width(ga.state_vals[k])) {
                            pair_bad = pair_cond;
                            break;
                        }
                        BVNodeId v_ne = ctx.make_binary(BVOp::Ne, gb.state_vals[k], ga.state_vals[k], 1);
                        pair_bad = ctx.make_binary(BVOp::Or, pair_bad,
                                                   ctx.make_binary(BVOp::And, pair_cond, v_ne, 1), 1);
                    }
                }
                diff = ctx.make_binary(BVOp::Or, diff, pair_bad, 1);
            }
        }
    }

    BVNodeId no_guard = ctx.make_binary(
        BVOp::And, domain_cond,
        ctx.make_binary(BVOp::And,
                        ctx.make_unary(BVOp::Not, out_b.guard_failed, 1),
                        ctx.make_unary(BVOp::Not, out_a.guard_failed, 1), 1), 1);

    BVNodeId t_ne = ctx.make_binary(BVOp::Xor, out_b.trapped, out_a.trapped, 1);
    diff = ctx.make_binary(BVOp::Or, diff, ctx.make_binary(BVOp::And, no_guard, t_ne, 1), 1);

    BVNodeId no_trap = ctx.make_binary(
        BVOp::And, no_guard,
        ctx.make_binary(BVOp::And,
                        ctx.make_unary(BVOp::Not, out_b.trapped, 1),
                        ctx.make_unary(BVOp::Not, out_a.trapped, 1), 1), 1);

    BVNodeId ret_cond_ne = ctx.make_binary(BVOp::Xor, out_b.returned, out_a.returned, 1);
    diff = ctx.make_binary(BVOp::Or, diff, ctx.make_binary(BVOp::And, no_trap, ret_cond_ne, 1), 1);

    BVNodeId both_ret = ctx.make_binary(
        BVOp::And, no_trap,
        ctx.make_binary(BVOp::And, out_b.returned, out_a.returned, 1), 1);

    if (!ret_type.is_void()) {
        BVNodeId r_ne = ctx.make_binary(BVOp::Ne, out_b.ret_val, out_a.ret_val, 1);
        diff = ctx.make_binary(BVOp::Or, diff, ctx.make_binary(BVOp::And, both_ret, r_ne, 1), 1);
    }

    if (opts.check_memory && (!out_b.stores.empty() || !out_a.stores.empty())) {
        std::vector<BVNodeId> check_byte_addrs;
        auto collect_addrs = [&](const std::vector<SymStore>& stores) {
            for (const auto& st : stores) {
                if (st.is_local_alloca) continue;
                for (uint32_t b = 0; b < st.num_bytes; ++b) {
                    BVNodeId ba = (b == 0) ? st.addr : ctx.make_binary(BVOp::Add, st.addr, ctx.make_const(b, 64), 64);
                    if (std::find(check_byte_addrs.begin(), check_byte_addrs.end(), ba) == check_byte_addrs.end())
                        check_byte_addrs.push_back(ba);
                }
            }
        };
        collect_addrs(out_b.stores);
        collect_addrs(out_a.stores);

        for (BVNodeId ba : check_byte_addrs) {
            BVNodeId mb = read_byte_at(ctx, mem_env, out_b.stores, ba);
            BVNodeId ma = read_byte_at(ctx, mem_env, out_a.stores, ba);
            BVNodeId m_ne = ctx.make_binary(BVOp::Ne, mb, ma, 1);
            diff = ctx.make_binary(BVOp::Or, diff, ctx.make_binary(BVOp::And, no_trap, m_ne, 1), 1);
        }
    }

    return diff;
}

} // namespace

EquivResult verify_mir_loop_equivalence(
    const Function& before,
    const Function& after,
    const EquivOptions& opts) {

    EquivResult res;
    if (before.param_count() != after.param_count() || before.return_type() != after.return_type()) {
        res.status = EquivStatus::Counterexample;
        res.diagnostic = "Function signature mismatch between unoptimized and optimized MIR";
        return res;
    }
    for (size_t i = 0; i < before.param_count(); ++i) {
        if (before.param_type(i) != after.param_type(i)) {
            res.status = EquivStatus::Counterexample;
            res.diagnostic = "Parameter " + std::to_string(i) + " type mismatch";
            return res;
        }
        if (type_bv_width(before.param_type(i)) == 0) {
            res.status = EquivStatus::Unsupported;
            res.diagnostic = "Unsupported parameter type";
            return res;
        }
    }

    const BasicBlock* hb = find_single_loop_header(before);
    const BasicBlock* ha = find_single_loop_header(after);
    if (!hb || !ha) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = "Functions do not both contain a single natural loop header";
        return res;
    }
    if (hb->param_count() == 0 || hb->param_count() != ha->param_count()) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = "Loop header block parameter count mismatch";
        return res;
    }

    BVContext ctx;
    std::vector<BVNodeId> param_vars;
    for (size_t i = 0; i < before.param_count(); ++i) {
        uint32_t w = type_bv_width(before.param_type(i));
        param_vars.push_back(ctx.make_var("arg" + std::to_string(i), w, before.param_type(i).is_float()));
    }

    EquivOptions reg_opts = opts;
    reg_opts.loop_unroll_bound = 0;

    // -----------------------------------------------------------------------
    // 1. Base Case: Entry / Preheader -> Loop Header (and zero-trip exits)
    // -----------------------------------------------------------------------
    InitMemEnv pre_mem;
    SymExecRegionConfig pre_cfg_b;
    pre_cfg_b.start_block = before.entry_block();
    pre_cfg_b.capture_block = hb;
    pre_cfg_b.capture_on_first_visit = true;
    SymExecOutput pre_b = symbolically_execute_mir_region(ctx, before, param_vars, pre_mem, reg_opts, pre_cfg_b);
    if (!pre_b.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = pre_b.error_msg;
        return res;
    }

    SymExecRegionConfig pre_cfg_a;
    pre_cfg_a.start_block = after.entry_block();
    pre_cfg_a.capture_block = ha;
    pre_cfg_a.capture_on_first_visit = true;
    SymExecOutput pre_a = symbolically_execute_mir_region(ctx, after, param_vars, pre_mem, reg_opts, pre_cfg_a);
    if (!pre_a.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = pre_a.error_msg;
        return res;
    }

    MergedCapture init_b = merge_captured_edges(ctx, hb, pre_b.captured_edges);
    MergedCapture init_a = merge_captured_edges(ctx, ha, pre_a.captured_edges);

    const size_t num_hp = hb->param_count();
    std::vector<size_t> perm(num_hp);
    for (size_t k = 0; k < num_hp; ++k) perm[k] = k;

    bool types_match = true;
    for (size_t k = 0; k < num_hp; ++k) {
        if (hb->param(k)->type() != ha->param(k)->type()) {
            types_match = false;
            break;
        }
    }
    if (!types_match) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = "Loop header parameter types do not match";
        return res;
    }

    BVNodeId base_diff = build_region_obs_diff(ctx, ctx.make_const(1, 1), pre_mem, pre_b, pre_a,
                                               before.return_type(), opts);
    BVNodeId pre_no_abort = ctx.make_binary(
        BVOp::And,
        ctx.make_binary(BVOp::And,
                        ctx.make_unary(BVOp::Not, pre_b.guard_failed, 1),
                        ctx.make_unary(BVOp::Not, pre_a.guard_failed, 1), 1),
        ctx.make_binary(BVOp::And,
                        ctx.make_unary(BVOp::Not, pre_b.trapped, 1),
                        ctx.make_unary(BVOp::Not, pre_a.trapped, 1), 1), 1);

    BVNodeId enter_ne = ctx.make_binary(BVOp::Xor, init_b.cond, init_a.cond, 1);
    base_diff = ctx.make_binary(BVOp::Or, base_diff, ctx.make_binary(BVOp::And, pre_no_abort, enter_ne, 1), 1);

    BVNodeId enter_ok = ctx.make_binary(
        BVOp::And, pre_no_abort,
        ctx.make_binary(BVOp::And, init_b.cond, init_a.cond, 1), 1);

    for (size_t k = 0; k < num_hp; ++k) {
        BVNodeId init_k_ne = ctx.make_binary(BVOp::Ne, init_b.args[k], init_a.args[perm[k]], 1);
        base_diff = ctx.make_binary(BVOp::Or, base_diff,
                                    ctx.make_binary(BVOp::And, enter_ok, init_k_ne, 1), 1);
    }

    // -----------------------------------------------------------------------
    // 2. Inductive Step: One Symbolic Iteration from Loop Header
    // -----------------------------------------------------------------------
    std::vector<BVNodeId> sym_h(num_hp);
    for (size_t k = 0; k < num_hp; ++k) {
        Type pt = hb->param(k)->type();
        sym_h[k] = ctx.make_var("loop_h" + std::to_string(k), type_bv_width(pt), pt.is_float());
    }

    InitMemEnv iter_mem;
    SymExecRegionConfig iter_cfg_b;
    iter_cfg_b.start_block = hb;
    iter_cfg_b.capture_block = hb;
    iter_cfg_b.capture_on_first_visit = false;
    iter_cfg_b.initial_vals = init_b.vals;
    for (size_t k = 0; k < num_hp; ++k) {
        iter_cfg_b.initial_vals[hb->param(k)] = sym_h[k];
    }
    SymExecOutput iter_b = symbolically_execute_mir_region(ctx, before, param_vars, iter_mem, reg_opts, iter_cfg_b);
    if (!iter_b.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = iter_b.error_msg;
        return res;
    }

    SymExecRegionConfig iter_cfg_a;
    iter_cfg_a.start_block = ha;
    iter_cfg_a.capture_block = ha;
    iter_cfg_a.capture_on_first_visit = false;
    iter_cfg_a.initial_vals = init_a.vals;
    for (size_t k = 0; k < num_hp; ++k) {
        iter_cfg_a.initial_vals[ha->param(perm[k])] = sym_h[k];
    }
    SymExecOutput iter_a = symbolically_execute_mir_region(ctx, after, param_vars, iter_mem, reg_opts, iter_cfg_a);
    if (!iter_a.supported) {
        res.status = EquivStatus::Unsupported;
        res.diagnostic = iter_a.error_msg;
        return res;
    }

    MergedCapture next_b = merge_captured_edges(ctx, hb, iter_b.captured_edges);
    MergedCapture next_a = merge_captured_edges(ctx, ha, iter_a.captured_edges);

    // Check sound inductive lower-bound invariants on integer induction variables (e.g. i >= 0)
    BVNodeId ind_hyp = enter_ok;
    for (size_t k = 0; k < num_hp; ++k) {
        if (!hb->param(k)->type().is_integer()) continue;
        BVNodeId init_k = init_b.args[k];
        if (ctx.interval(init_k).smin < 0) continue;
        BVNodeId cand_base = ctx.make_binary(BVOp::Sge, init_k, init_k, 1);
        BVNodeId cand_hyp = ctx.make_binary(BVOp::Sge, sym_h[k], init_k, 1);
        BVNodeId cand_step = ctx.make_binary(BVOp::Sge, next_b.args[k], init_k, 1);
        BVNodeId step_viol = ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::And, enter_ok, cand_hyp, 1),
            ctx.make_binary(BVOp::And, next_b.cond, ctx.make_unary(BVOp::Not, cand_step, 1), 1), 1);
        (void)cand_base;
        if (solve_nonzero(ctx, step_viol, 2000).status == SolveOutcome::Unsat) {
            ind_hyp = ctx.make_binary(BVOp::And, ind_hyp, cand_hyp, 1);
        }
    }

    BVNodeId step_diff = build_region_obs_diff(ctx, ind_hyp, iter_mem, iter_b, iter_a,
                                               before.return_type(), opts);
    BVNodeId iter_no_abort = ctx.make_binary(
        BVOp::And, ind_hyp,
        ctx.make_binary(
            BVOp::And,
            ctx.make_binary(BVOp::And,
                            ctx.make_unary(BVOp::Not, iter_b.guard_failed, 1),
                            ctx.make_unary(BVOp::Not, iter_a.guard_failed, 1), 1),
            ctx.make_binary(BVOp::And,
                            ctx.make_unary(BVOp::Not, iter_b.trapped, 1),
                            ctx.make_unary(BVOp::Not, iter_a.trapped, 1), 1), 1), 1);

    BVNodeId cont_ne = ctx.make_binary(BVOp::Xor, next_b.cond, next_a.cond, 1);
    step_diff = ctx.make_binary(BVOp::Or, step_diff, ctx.make_binary(BVOp::And, iter_no_abort, cont_ne, 1), 1);

    BVNodeId both_cont = ctx.make_binary(
        BVOp::And, iter_no_abort,
        ctx.make_binary(BVOp::And, next_b.cond, next_a.cond, 1), 1);

    for (size_t k = 0; k < num_hp; ++k) {
        BVNodeId next_k_ne = ctx.make_binary(BVOp::Ne, next_b.args[k], next_a.args[perm[k]], 1);
        step_diff = ctx.make_binary(BVOp::Or, step_diff,
                                    ctx.make_binary(BVOp::And, both_cont, next_k_ne, 1), 1);
    }

    BVNodeId total_diff = ctx.make_binary(BVOp::Or, base_diff, step_diff, 1);
    SolveOutcome ind_solved = solve_nonzero(ctx, total_diff, opts.max_sat_conflicts);
    if (ind_solved.status == SolveOutcome::Unsat) {
        res.status = EquivStatus::ProvenEquivalent;
        return res;
    }

    // -----------------------------------------------------------------------
    // 3. Counterexample Validation via Concrete / Bounded Unrolling
    // -----------------------------------------------------------------------
    // First check bounded unrolling from entry (1..4 trips) to obtain a guaranteed
    // reachable counterexample over the function parameters.
    {
        BVContext bctx;
        std::vector<BVNodeId> bparams;
        for (size_t i = 0; i < before.param_count(); ++i) {
            uint32_t w = type_bv_width(before.param_type(i));
            bparams.push_back(bctx.make_var("arg" + std::to_string(i), w, before.param_type(i).is_float()));
        }
        EquivOptions bopts = opts;
        bopts.loop_unroll_bound = std::max(opts.loop_unroll_bound, 4u);
        InitMemEnv bmem;
        SymExecOutput ob = symbolically_execute_mir(bctx, before, bparams, bmem, bopts);
        SymExecOutput oa = symbolically_execute_mir(bctx, after, bparams, bmem, bopts);
        if (ob.supported && oa.supported) {
            BVNodeId both_comp = bctx.make_binary(BVOp::And, ob.completed, oa.completed, 1);
            BVNodeId bdiff = build_region_obs_diff(bctx, both_comp, bmem, ob, oa, before.return_type(), bopts);
            SolveOutcome bsolved = solve_nonzero(bctx, bdiff, bopts.max_sat_conflicts);
            if (bsolved.status == SolveOutcome::Sat) {
                res.status = EquivStatus::Counterexample;
                for (size_t i = 0; i < before.param_count(); ++i) {
                    uint64_t bits = (i < bsolved.model.size()) ? bsolved.model[i] : 0;
                    res.counterexample.push_back(bits_to_rv(before.param_type(i), bits));
                }
                std::ostringstream diag;
                diag << "MIR loop equivalence counterexample in @" << before.name() << "(";
                for (size_t i = 0; i < res.counterexample.size(); ++i) {
                    if (i > 0) diag << ", ";
                    diag << to_string(res.counterexample[i]);
                }
                diag << ")";
                res.diagnostic = diag.str();
                return res;
            }
        }
    }

    if (ind_solved.status == SolveOutcome::Sat) {
        // Also check if substituting init_args into the 1-step inductive transition diverges
        // (catches cases where the loop condition continues beyond 4 iterations, e.g. n = 100,
        //  while iteration 0 already corrupts the latch state or continuation condition).
        std::unordered_map<const Value*, BVNodeId> iter0_vals_b = init_b.vals;
        std::unordered_map<const Value*, BVNodeId> iter0_vals_a = init_a.vals;
        for (size_t k = 0; k < num_hp; ++k) {
            iter0_vals_b[hb->param(k)] = init_b.args[k];
            iter0_vals_a[ha->param(perm[k])] = init_a.args[perm[k]];
        }
        InitMemEnv iter0_mem;
        SymExecRegionConfig cfg0_b = iter_cfg_b;
        cfg0_b.initial_vals = std::move(iter0_vals_b);
        SymExecRegionConfig cfg0_a = iter_cfg_a;
        cfg0_a.initial_vals = std::move(iter0_vals_a);
        SymExecOutput out0_b = symbolically_execute_mir_region(ctx, before, param_vars, iter0_mem, reg_opts, cfg0_b);
        SymExecOutput out0_a = symbolically_execute_mir_region(ctx, after, param_vars, iter0_mem, reg_opts, cfg0_a);
        MergedCapture nxt0_b = merge_captured_edges(ctx, hb, out0_b.captured_edges);
        MergedCapture nxt0_a = merge_captured_edges(ctx, ha, out0_a.captured_edges);

        BVNodeId diff0 = build_region_obs_diff(ctx, enter_ok, iter0_mem, out0_b, out0_a, before.return_type(), opts);
        BVNodeId c0_ne = ctx.make_binary(BVOp::Xor, nxt0_b.cond, nxt0_a.cond, 1);
        diff0 = ctx.make_binary(BVOp::Or, diff0, ctx.make_binary(BVOp::And, enter_ok, c0_ne, 1), 1);
        BVNodeId both_c0 = ctx.make_binary(BVOp::And, enter_ok,
                                           ctx.make_binary(BVOp::And, nxt0_b.cond, nxt0_a.cond, 1), 1);
        for (size_t k = 0; k < num_hp; ++k) {
            BVNodeId nk_ne = ctx.make_binary(BVOp::Ne, nxt0_b.args[k], nxt0_a.args[perm[k]], 1);
            diff0 = ctx.make_binary(BVOp::Or, diff0, ctx.make_binary(BVOp::And, both_c0, nk_ne, 1), 1);
        }
        BVNodeId base_or_iter0 = ctx.make_binary(BVOp::Or, base_diff, diff0, 1);
        SolveOutcome s0 = solve_nonzero(ctx, base_or_iter0, opts.max_sat_conflicts);
        if (s0.status == SolveOutcome::Sat) {
            res.status = EquivStatus::Counterexample;
            for (size_t i = 0; i < before.param_count(); ++i) {
                uint64_t bits = (i < s0.model.size()) ? s0.model[i] : 0;
                res.counterexample.push_back(bits_to_rv(before.param_type(i), bits));
            }
            res.diagnostic = "MIR loop equivalence counterexample on initial loop transition";
            return res;
        }
    }

    res.status = EquivStatus::Unknown;
    res.diagnostic = "Inductive loop equivalence check inconclusive";
    return res;
}

} // namespace brass::verify
