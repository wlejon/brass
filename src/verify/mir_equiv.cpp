#include <brass/verify/smt_equiv.hpp>
#include "mir_sym_exec.hpp"
#include <brass/mir/module.hpp>
#include <brass/mir/function.hpp>
#include <algorithm>
#include <cstring>
#include <sstream>

namespace brass::verify {
namespace {

RuntimeValue model_bits_to_runtime_value(Type t, uint64_t bits) {
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

struct BuiltEquivQuery {
    BVContext ctx;
    std::vector<BVNodeId> param_vars;
    SymExecOutput out_before;
    SymExecOutput out_after;
    BVNodeId mismatch = 0;
    BVNodeId guard_diff = 0;
    BVNodeId deopt_diff = 0;
    BVNodeId trap_diff = 0;
    BVNodeId ret_diff = 0;
    BVNodeId mem_diff = 0;
    EquivStatus early_status = EquivStatus::ProvenEquivalent;
    std::string early_diag;
};

BuiltEquivQuery build_mir_equiv_query(
    const Function& before,
    const Function& after,
    const EquivOptions& opts) {

    BuiltEquivQuery q;
    if (before.param_count() != after.param_count() || before.return_type() != after.return_type()) {
        q.early_status = EquivStatus::Counterexample;
        q.early_diag = "Function signature mismatch between unoptimized and optimized MIR";
        return q;
    }
    for (size_t i = 0; i < before.param_count(); ++i) {
        if (before.param_type(i) != after.param_type(i)) {
            q.early_status = EquivStatus::Counterexample;
            q.early_diag = "Parameter " + std::to_string(i) + " type mismatch";
            return q;
        }
        uint32_t w = type_bv_width(before.param_type(i));
        if (w == 0) {
            q.early_status = EquivStatus::Unsupported;
            q.early_diag = "Unsupported parameter type";
            return q;
        }
        q.param_vars.push_back(q.ctx.make_var("arg" + std::to_string(i), w, before.param_type(i).is_float()));
    }

    InitMemEnv init_mem;
    q.out_before = symbolically_execute_mir(q.ctx, before, q.param_vars, init_mem, opts);
    if (!q.out_before.supported) {
        q.early_status = EquivStatus::Unsupported;
        q.early_diag = q.out_before.error_msg;
        return q;
    }
    q.out_after = symbolically_execute_mir(q.ctx, after, q.param_vars, init_mem, opts);
    if (!q.out_after.supported) {
        q.early_status = EquivStatus::Unsupported;
        q.early_diag = q.out_after.error_msg;
        return q;
    }
    if (!q.out_before.has_type_error && q.out_after.has_type_error) {
        q.early_diag = "Optimized MIR has type/SSA error: " + q.out_after.error_msg;
    }

    BVNodeId both_completed = q.ctx.make_binary(BVOp::And, q.out_before.completed, q.out_after.completed, 1);

    q.guard_diff = q.ctx.make_const(0, 1);
    q.deopt_diff = q.ctx.make_const(0, 1);
    if (opts.check_guards) {
        BVNodeId g_ne = q.ctx.make_binary(BVOp::Xor, q.out_before.guard_failed, q.out_after.guard_failed, 1);
        q.guard_diff = q.ctx.make_binary(BVOp::And, both_completed, g_ne, 1);

        // Verify deopt state_map(), resume_id(), and exit_symbol when both trigger a guard on the same input (#75)
        for (const auto& gb : q.out_before.guard_exits) {
            for (const auto& ga : q.out_after.guard_exits) {
                BVNodeId pair_cond = q.ctx.make_binary(
                    BVOp::And, both_completed,
                    q.ctx.make_binary(BVOp::And, gb.fail_cond, ga.fail_cond, 1), 1);
                if (q.ctx.nodes()[pair_cond].op == BVOp::Const && q.ctx.nodes()[pair_cond].imm == 0) continue;

                BVNodeId pair_bad = q.ctx.make_const(0, 1);
                if (gb.resume_id != ga.resume_id || gb.exit_symbol != ga.exit_symbol ||
                    gb.state_vals.size() != ga.state_vals.size()) {
                    pair_bad = pair_cond;
                } else {
                    for (size_t k = 0; k < gb.state_vals.size(); ++k) {
                        if (gb.state_types[k] != ga.state_types[k] ||
                            q.ctx.width(gb.state_vals[k]) != q.ctx.width(ga.state_vals[k])) {
                            pair_bad = pair_cond;
                            break;
                        }
                        BVNodeId v_ne = q.ctx.make_binary(BVOp::Ne, gb.state_vals[k], ga.state_vals[k], 1);
                        pair_bad = q.ctx.make_binary(BVOp::Or, pair_bad,
                                                     q.ctx.make_binary(BVOp::And, pair_cond, v_ne, 1), 1);
                    }
                }
                q.deopt_diff = q.ctx.make_binary(BVOp::Or, q.deopt_diff, pair_bad, 1);
            }
        }
    }

    BVNodeId no_guard_fail = q.ctx.make_binary(
        BVOp::And, both_completed,
        q.ctx.make_binary(BVOp::And,
                          q.ctx.make_unary(BVOp::Not, q.out_before.guard_failed, 1),
                          q.ctx.make_unary(BVOp::Not, q.out_after.guard_failed, 1), 1), 1);

    BVNodeId t_ne = q.ctx.make_binary(BVOp::Xor, q.out_before.trapped, q.out_after.trapped, 1);
    q.trap_diff = q.ctx.make_binary(BVOp::And, no_guard_fail, t_ne, 1);

    BVNodeId both_returned = q.ctx.make_binary(
        BVOp::And, no_guard_fail,
        q.ctx.make_binary(BVOp::And, q.out_before.returned, q.out_after.returned, 1), 1);

    q.ret_diff = q.ctx.make_const(0, 1);
    if (!before.return_type().is_void()) {
        BVNodeId r_ne = q.ctx.make_binary(BVOp::Ne, q.out_before.ret_val, q.out_after.ret_val, 1);
        q.ret_diff = q.ctx.make_binary(BVOp::And, both_returned, r_ne, 1);
    }

    q.mem_diff = q.ctx.make_const(0, 1);
    if (opts.check_memory) {
        std::vector<BVNodeId> check_byte_addrs;
        auto collect_addrs = [&](const std::vector<SymStore>& stores) {
            for (const auto& st : stores) {
                if (st.is_local_alloca) continue;
                for (uint32_t b = 0; b < st.num_bytes; ++b) {
                    BVNodeId ba = (b == 0) ? st.addr : q.ctx.make_binary(BVOp::Add, st.addr, q.ctx.make_const(b, 64), 64);
                    if (std::find(check_byte_addrs.begin(), check_byte_addrs.end(), ba) == check_byte_addrs.end())
                        check_byte_addrs.push_back(ba);
                }
            }
        };
        collect_addrs(q.out_before.stores);
        collect_addrs(q.out_after.stores);

        for (BVNodeId ba : check_byte_addrs) {
            BVNodeId mb = read_byte_at(q.ctx, init_mem, q.out_before.stores, ba);
            BVNodeId ma = read_byte_at(q.ctx, init_mem, q.out_after.stores, ba);
            BVNodeId m_ne = q.ctx.make_binary(BVOp::Ne, mb, ma, 1);
            q.mem_diff = q.ctx.make_binary(BVOp::Or, q.mem_diff,
                                           q.ctx.make_binary(BVOp::And, both_returned, m_ne, 1), 1);
        }
    }

    BVNodeId any_guard_or_deopt = q.ctx.make_binary(BVOp::Or, q.guard_diff, q.deopt_diff, 1);
    q.mismatch = q.ctx.make_binary(
        BVOp::Or,
        q.ctx.make_binary(BVOp::Or, any_guard_or_deopt, q.trap_diff, 1),
        q.ctx.make_binary(BVOp::Or, q.ret_diff, q.mem_diff, 1), 1);

    if (!q.out_before.has_type_error && q.out_after.has_type_error) {
        q.mismatch = q.ctx.make_const(1, 1);
    }

    return q;
}

} // namespace

EquivResult verify_mir_equivalence(
    const Function& before,
    const Function& after,
    const EquivOptions& opts) {

    EquivResult res;
    BuiltEquivQuery q = build_mir_equiv_query(before, after, opts);
    if (q.early_status != EquivStatus::ProvenEquivalent) {
        res.status = q.early_status;
        res.diagnostic = q.early_diag;
        return res;
    }

    SolveOutcome solved = solve_nonzero(q.ctx, q.mismatch, opts.max_sat_conflicts);
    if (solved.status == SolveOutcome::Unsat) {
        res.status = EquivStatus::ProvenEquivalent;
        return res;
    }
    if (solved.status == SolveOutcome::Unknown) {
        res.status = EquivStatus::Unknown;
        res.diagnostic = "Solver budget exhausted before proving equivalence";
        return res;
    }

    res.status = EquivStatus::Counterexample;
    for (size_t i = 0; i < before.param_count(); ++i) {
        uint64_t bits = (i < solved.model.size()) ? solved.model[i] : 0;
        res.counterexample.push_back(model_bits_to_runtime_value(before.param_type(i), bits));
    }

    std::ostringstream diag;
    diag << "MIR equivalence counterexample in @" << before.name() << "(";
    for (size_t i = 0; i < res.counterexample.size(); ++i) {
        if (i > 0) diag << ", ";
        diag << to_string(res.counterexample[i]);
    }
    diag << "): ";
    if (!q.early_diag.empty()) {
        diag << q.early_diag;
    } else if (q.ctx.eval_concrete(q.guard_diff, solved.model) != 0) {
        diag << "guard failure mismatch (before guard_failed="
             << q.ctx.eval_concrete(q.out_before.guard_failed, solved.model)
             << ", after guard_failed="
             << q.ctx.eval_concrete(q.out_after.guard_failed, solved.model) << ")";
    } else if (q.ctx.eval_concrete(q.deopt_diff, solved.model) != 0) {
        diag << "deopt state_map/resume mismatch on triggered guard";
    } else if (q.ctx.eval_concrete(q.trap_diff, solved.model) != 0) {
        diag << "trap mismatch (before trapped="
             << q.ctx.eval_concrete(q.out_before.trapped, solved.model)
             << ", after trapped="
             << q.ctx.eval_concrete(q.out_after.trapped, solved.model) << ")";
    } else if (q.ctx.eval_concrete(q.ret_diff, solved.model) != 0) {
        diag << "return value mismatch (before="
             << to_string(model_bits_to_runtime_value(before.return_type(),
                                                      q.ctx.eval_concrete(q.out_before.ret_val, solved.model)))
             << ", after="
             << to_string(model_bits_to_runtime_value(after.return_type(),
                                                      q.ctx.eval_concrete(q.out_after.ret_val, solved.model)))
             << ")";
    } else {
        diag << "observable memory store mismatch";
    }
    res.diagnostic = diag.str();
    return res;
}

EquivResult verify_mir_module_equivalence(
    const Module& before,
    const Module& after,
    const EquivOptions& opts) {

    EquivResult overall;
    overall.status = EquivStatus::ProvenEquivalent;
    for (const Function* fn_before : before.functions()) {
        if (!fn_before || fn_before->blocks().empty()) continue;
        const Function* fn_after = after.get_function(fn_before->name());
        if (!fn_after || fn_after->blocks().empty()) continue;
        EquivResult r = verify_mir_equivalence(*fn_before, *fn_after, opts);
        if (r.status == EquivStatus::Counterexample) return r;
        if (r.status == EquivStatus::Unknown && overall.status == EquivStatus::ProvenEquivalent) overall = r;
    }
    return overall;
}

std::string export_mir_equivalence_smt2(
    const Function& before,
    const Function& after,
    const EquivOptions& opts) {

    BuiltEquivQuery q = build_mir_equiv_query(before, after, opts);
    if (q.early_status != EquivStatus::ProvenEquivalent && q.mismatch == 0) {
        return "; Unable to build SMT-LIB2 equivalence query: " + q.early_diag + "\n";
    }
    return q.ctx.export_smt2(q.mismatch, "MIR equivalence query for @" + std::string(before.name()));
}

} // namespace brass::verify
