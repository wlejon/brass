#include <brass/mir/gpu_opt.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/fdiv_pow2.hpp>
#include <brass/mir/fma_opt.hpp>
#include <brass/mir/gvn.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/scalar_opt.hpp>
#include <brass/mir/smt_guard_elim.hpp>
#include <brass/mir/superopt.hpp>

#include <unordered_set>
#include <utility>
#include <vector>

namespace brass {

namespace {

// Pure, non-trapping MIR opcodes that are safe to speculatively hoist out of
// GPU loops into a dominating preheader without creating register pressure from
// re-materialized constants or risking faults/barrier/memory side effects.
bool is_pure_hoistable_gpu_op(Opcode op) noexcept {
    switch (op) {
        case Opcode::sext_i64:
        case Opcode::zext_i64:
        case Opcode::trunc_i32:
        case Opcode::trunc_i8:
        case Opcode::fptosi_i32:
        case Opcode::fptosi_i64:
        case Opcode::fptosi_i32_f32:
        case Opcode::fptosi_i64_f32:
        case Opcode::sitofp_f64_i32:
        case Opcode::sitofp_f64_i64:
        case Opcode::sitofp_f32_i32:
        case Opcode::sitofp_f32_i64:
        case Opcode::fptrunc_f32_f64:
        case Opcode::fpext_f64_f32:
        case Opcode::bitcast_i64_f64:
        case Opcode::bitcast_f64_i64:
        case Opcode::bitcast_i32_f32:
        case Opcode::bitcast_f32_i32:
        case Opcode::add:
        case Opcode::sub:
        case Opcode::mul:
        case Opcode::fma_f32:
        case Opcode::fma_f64:
        case Opcode::floor_f32:
        case Opcode::floor_f64:
        case Opcode::ceil_f32:
        case Opcode::ceil_f64:
        case Opcode::round_f32:
        case Opcode::round_f64:
        case Opcode::fabs_f32:
        case Opcode::fabs_f64:
        case Opcode::fmin_f32:
        case Opcode::fmin_f64:
        case Opcode::fmax_f32:
        case Opcode::fmax_f64:
        case Opcode::neg:
        case Opcode::and_:
        case Opcode::or_:
        case Opcode::xor_:
        case Opcode::shl:
        case Opcode::lshr:
        case Opcode::ashr:
        case Opcode::not_:
        case Opcode::clz:
        case Opcode::ctz:
        case Opcode::popcnt:
        case Opcode::eq:
        case Opcode::ne:
        case Opcode::slt:
        case Opcode::ult:
        case Opcode::sle:
        case Opcode::ule:
        case Opcode::sgt:
        case Opcode::ugt:
        case Opcode::sge:
        case Opcode::uge:
        case Opcode::select:
        case Opcode::vadd:
        case Opcode::vsub:
        case Opcode::vmul:
        case Opcode::vfma:
        case Opcode::vneg:
        case Opcode::vmin:
        case Opcode::vmax:
        case Opcode::vand:
        case Opcode::vor:
        case Opcode::vxor:
        case Opcode::vnot:
        case Opcode::vbroadcast:
        case Opcode::vextract_lane:
        case Opcode::vinsert_lane:
        case Opcode::vcmp_eq:
        case Opcode::vcmp_ne:
        case Opcode::vcmp_lt:
        case Opcode::vcmp_le:
        case Opcode::vcmp_gt:
        case Opcode::vcmp_ge:
        case Opcode::vselect:
        case Opcode::vshl:
        case Opcode::vlshr:
        case Opcode::vashr:
        case Opcode::vconvert:
        case Opcode::vbitcast:
        case Opcode::vfloor:
        case Opcode::vceil:
        case Opcode::vtrunc:
        case Opcode::vround:
            return true;
        default:
            return false;
    }
}

bool is_defined_outside_loop(const LoopInfo& loop, const Value* op) noexcept {
    if (!op) return true;
    if (op->is_block_param()) return !loop.contains(op->defining_block());
    if (op->is_instruction()) {
        const Instruction* def = op->defining_instruction();
        return !def || !loop.contains(def->parent());
    }
    return true;
}

bool hoist_pure_gpu_loop_invariants(Function& fn) {
    if (!fn.entry_block() || fn.blocks().size() < 2) return false;
    fn.rebuild_cfg_predecessors();
    DominatorTree dom(fn);
    LoopAnalysis la(fn, dom);
    std::vector<LoopInfo*> loops = la.post_order_loops();
    if (loops.empty()) return false;

    bool changed = false;
    for (LoopInfo* loop : loops) {
        if (!loop) continue;
        std::unordered_set<const Value*> invariant_defs;
        std::vector<std::pair<BasicBlock*, Instruction*>> to_hoist;

        bool progress = true;
        while (progress) {
            progress = false;
            for (BasicBlock* bb : loop->blocks()) {
                if (!bb || bb == loop->preheader()) continue;
                for (Instruction* inst : *bb) {
                    if (!inst || !inst->produces_value() || !is_pure_hoistable_gpu_op(inst->opcode())) continue;
                    if (invariant_defs.count(inst->result())) continue;
                    bool ops_invariant = true;
                    std::vector<Instruction*> needed_consts;
                    for (size_t i = 0; i < inst->operand_count(); ++i) {
                        const Value* op = inst->operand(i);
                        if (!op || is_defined_outside_loop(*loop, op) || invariant_defs.count(op)) {
                            continue;
                        }
                        if (op->is_instruction()) {
                            Instruction* def = op->defining_instruction();
                            if (def && is_constant(def->opcode()) && def->parent() &&
                                loop->contains(def->parent())) {
                                needed_consts.push_back(def);
                                continue;
                            }
                        }
                        ops_invariant = false;
                        break;
                    }
                    if (ops_invariant) {
                        for (Instruction* cinst : needed_consts) {
                            if (cinst->result() && invariant_defs.insert(cinst->result()).second) {
                                to_hoist.push_back({cinst->parent(), cinst});
                            }
                        }
                        invariant_defs.insert(inst->result());
                        to_hoist.push_back({bb, inst});
                        progress = true;
                    }
                }
            }
        }

        if (to_hoist.empty()) continue;
        BasicBlock* preheader = LoopAnalysis::ensure_preheader(fn, *loop);
        if (!preheader || !preheader->terminator()) continue;

        Instruction* ph_term = preheader->terminator();
        for (auto& [bb, inst] : to_hoist) {
            bb->remove_instruction(inst);
            preheader->insert_before(inst, ph_term);
            changed = true;
        }
    }

    if (changed) fn.rebuild_cfg_predecessors();
    return changed;
}

} // namespace

bool optimize_gpu_function(Function& fn, Module& mod, const GpuOptOptions& opts) {
    if (opts.opt_level <= 0 || fn.blocks().empty()) return false;

    bool changed = false;
    fn.rebuild_cfg_predecessors();

    if (opts.enable_fold_constants) {
        changed |= fold_constants(fn);
    }
    if (opts.enable_cse) {
        changed |= dominator_cse(fn);
    }
    if (opts.enable_gvn) {
        GvnOptions gvn_opts;
        gvn_opts.enable_cse = true;
        gvn_opts.enable_rle = opts.enable_gvn_rle;
        gvn_opts.enable_dse = opts.enable_gvn_dse;
        changed |= gvn_function(fn, gvn_opts);
    }
    if (opts.enable_licm) {
        changed |= hoist_pure_gpu_loop_invariants(fn);
    }
    if (opts.enable_fdiv_pow2) {
        changed |= (rewrite_fdiv_pow2(fn) > 0);
    }
    if (opts.enable_small_loop_unroll) {
        if (unroll_small_gpu_loops(fn, opts.max_unroll_trip_count)) {
            changed = true;
            changed |= scalar_cleanup(fn);
        }
    }
    if (opts.enable_load_coalesce) {
        changed |= coalesce_gpu_v4_accesses(fn, opts.assume_disjoint_buffers);
    }
    if (opts.enable_fma) {
        FmaOptOptions fma_opts;
        fma_opts.enable_scalar = true;
        fma_opts.enable_vector = true;
        changed |= fma_opt_pass(fn, fma_opts);
    }
    if (opts.opt_level >= 2) {
        if (opts.enable_smt_guard_elim) {
            SmtGuardElimOptions ge_opts;
            ge_opts.fold_branches = opts.fold_branches;
            ge_opts.max_sat_conflicts = opts.max_sat_conflicts;
            changed |= run_smt_guard_elimination(fn, mod, ge_opts);
        }
        if (opts.enable_superopt || opts.enable_gpu_superopt) {
            SuperoptOptions sopt_opts;
            sopt_opts.max_sat_conflicts = opts.max_sat_conflicts;
            sopt_opts.enable_gpu_superopt = opts.enable_gpu_superopt;
            sopt_opts.gpu_max_ctaid = opts.gpu_max_ctaid;
            sopt_opts.gpu_max_global_tid = opts.gpu_max_global_tid;
            sopt_opts.gpu_subgroup_size = opts.gpu_subgroup_size;
            if (opts.enable_superopt) {
                changed |= run_mir_superopt(fn, mod, sopt_opts);
            } else {
                changed |= run_gpu_mir_superopt(fn, mod, sopt_opts);
            }
        }
    }
    if (opts.enable_scalar_cleanup) {
        changed |= scalar_cleanup(fn);
    }

    fn.rebuild_cfg_predecessors();
    return changed;
}

} // namespace brass