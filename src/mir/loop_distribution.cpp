#include <brass/mir/loop_distribution.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/opcodes.hpp>
#include "int_fold.hpp"
#include "ir_clone.hpp"
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Loop distribution splits one loop into two over the same iteration space:
// the first runs the vectorizable part of every iteration, the second the
// rest. Every operation of the second part that used to run before an
// operation of the first part of a later iteration now runs after it, so the
// two parts must commute across iterations. The analysis accepts only what
// makes that provable:
//   - the loop is a header holding just its exit test plus one body block,
//     counted by its only header parameter with a positive constant step;
//   - nothing in the body may trap, deoptimize, or call anything except a
//     module function whose single block computes a value from its
//     arguments alone;
//   - every memory access of one part is disjoint from every access of the
//     other unless both only read;
//   - a value the second part needs from the first is either recomputed from
//     the induction variable and loop invariants, or reloaded from the slot
//     of an induction-indexed store no other store can overwrite.
// Values the second part needs are never kept in a temporary buffer: sizing
// one needs a trip count that may be negative, overflow, or be unbounded.
namespace brass {

std::string LoopDistributionStats::format_report() const {
    std::ostringstream ss;
    ss << "=== Loop Distribution Statistics ===\n"
       << "  Loops distributed: " << loops_distributed << "\n"
       << "  Candidates checked: " << candidates_checked << "\n"
       << "  Rejected (pure vectorizable): " << rejected_pure_vectorizable << "\n"
       << "  Rejected (pure scalar): " << rejected_pure_scalar << "\n"
       << "  Rejected (dependency cycles): " << rejected_cycles << "\n";
    return ss.str();
}

namespace {

bool get_const_int(const Value* val, int64_t& out_val) {
    if (!val || !val->is_instruction()) return false;
    const Instruction* def = val->defining_instruction();
    if (!def) return false;
    if (def->opcode() == Opcode::iconst_i32) {
        out_val = static_cast<int64_t>(def->imm_i32());
        return true;
    }
    if (def->opcode() == Opcode::iconst_i64) {
        out_val = def->imm_i64();
        return true;
    }
    return false;
}

bool is_supported_vector_arithmetic(Opcode op) {
    switch (op) {
        case Opcode::add: case Opcode::sub: case Opcode::mul:
        case Opcode::sdiv: case Opcode::udiv: case Opcode::neg:
        case Opcode::vmin: case Opcode::vmax: case Opcode::vsqrt:
        case Opcode::and_: case Opcode::or_: case Opcode::xor_: case Opcode::not_:
        case Opcode::select:
        case Opcode::sext_i64: case Opcode::zext_i64: case Opcode::trunc_i32:
        case Opcode::sitofp_f64_i32: case Opcode::sitofp_f64_i64:
        case Opcode::fptosi_i32: case Opcode::fptosi_i64:
        case Opcode::eq: case Opcode::ne:
        case Opcode::slt: case Opcode::ult: case Opcode::sle: case Opcode::ule:
        case Opcode::sgt: case Opcode::ugt: case Opcode::sge: case Opcode::uge:
        case Opcode::iconst_i32: case Opcode::iconst_i64: case Opcode::fconst_f64:
            return true;
        default:
            return false;
    }
}

bool is_memory_access(Opcode op) {
    return op == Opcode::load || op == Opcode::load_indexed || op == Opcode::store || op == Opcode::store_indexed;
}

bool is_store(Opcode op) {
    return op == Opcode::store || op == Opcode::store_indexed;
}

// Computes its result from its operands alone: no memory, no effects, no trap.
bool is_pure_computation(const Instruction& inst) {
    if (inst.is_terminator() || inst.is_call() || inst.has_side_effects()) return false;
    switch (inst.opcode()) {
        case Opcode::load:
        case Opcode::load_indexed:
        case Opcode::vload:
        case Opcode::alloca_:
        case Opcode::landing_pad:
        case Opcode::safepoint:
        case Opcode::guard:
        case Opcode::resume_point:
        case Opcode::osr_entry:
            return false;
        case Opcode::sdiv:
        case Opcode::udiv:
        case Opcode::smod:
        case Opcode::umod: {
            if (inst.type().is_float()) return true;
            int64_t divisor = 0;
            const unsigned width = int_fold::width_of(inst.type());
            return get_const_int(inst.operand(1), divisor) && width != 0 &&
                   !int_fold::division_may_trap(inst.opcode(), width, divisor);
        }
        default:
            return true;
    }
}

// A call to a module function of one block that only computes from its
// arguments: it can run at any point relative to memory operations.
bool is_pure_leaf_call(const Function& fn, const Instruction& inst) {
    if (inst.opcode() != Opcode::call || !fn.parent()) return false;
    const Function* callee = fn.parent()->get_function(inst.symbol());
    if (!callee || callee == &fn || callee->blocks().size() != 1) return false;
    const BasicBlock* only = callee->blocks().front();
    if (!only) return false;
    for (const Instruction* ci : *only) {
        if (!ci) return false;
        if (ci->is_terminator()) {
            if (ci->opcode() != Opcode::ret) return false;
            continue;
        }
        if (!is_pure_computation(*ci)) return false;
    }
    return true;
}

struct DistributableLoopInfo {
    LoopInfo* loop = nullptr;
    BasicBlock* preheader = nullptr;
    BasicBlock* header = nullptr;
    BasicBlock* body = nullptr;
    BasicBlock* exit_bb = nullptr;
    Value* iv_param = nullptr;
    Value* init_val = nullptr;
    Value* limit_val = nullptr;
    Instruction* cmp_inst = nullptr;
    Instruction* iv_inc_inst = nullptr;
    int64_t step = 1;
    bool exit_on_false = true;
};

bool extract_distributable_loop(LoopInfo& loop, DistributableLoopInfo& info) {
    BasicBlock* header = loop.header();
    if (!header || loop.latches().size() != 1 || loop.blocks().size() != 2 || header->param_count() != 1) return false;
    BasicBlock* body = loop.latches()[0];
    if (!body || body == header) return false;

    BasicBlock* preheader = nullptr;
    for (BasicBlock* pred : header->predecessors()) {
        if (!pred || loop.contains(pred)) continue;
        if (preheader && preheader != pred) return false;
        preheader = pred;
    }
    if (!preheader) return false;
    const Instruction* ph_term = preheader->terminator();
    const Instruction* body_term = body->terminator();
    Instruction* hdr_term = header->terminator();
    if (!ph_term || ph_term->opcode() != Opcode::br || ph_term->branch_target().block != header) return false;
    if (!body_term || body_term->opcode() != Opcode::br || body_term->branch_target().block != header) return false;
    if (!hdr_term || hdr_term->opcode() != Opcode::br_if) return false;
    if (ph_term->branch_target().args.size() != 1 || body_term->branch_target().args.size() != 1) return false;

    bool exit_on_false = true;
    BasicBlock* exit_bb = nullptr;
    if (hdr_term->true_target().block == body && !loop.contains(hdr_term->false_target().block)) {
        exit_bb = hdr_term->false_target().block;
    } else if (hdr_term->false_target().block == body && !loop.contains(hdr_term->true_target().block)) {
        exit_bb = hdr_term->true_target().block;
        exit_on_false = false;
    } else {
        return false;
    }
    if (!exit_bb || !hdr_term->true_target().args.empty() || !hdr_term->false_target().args.empty()) return false;

    Value* iv = header->param(0);
    Value* cond = hdr_term->operand(0);
    Instruction* cmp = (cond && cond->is_instruction()) ? cond->defining_instruction() : nullptr;
    if (!cmp || cmp->parent() != header || header->head() != cmp || cmp->next() != hdr_term) return false;
    const Opcode op = cmp->opcode();
    if (op != Opcode::slt && op != Opcode::ult && op != Opcode::sle && op != Opcode::ule) return false;
    if (cmp->operand(0) != iv || !cmp->operand(1) || !loop.is_loop_invariant(cmp->operand(1))) return false;

    Value* next = body_term->branch_target().args[0];
    Instruction* inc = (next && next->is_instruction()) ? next->defining_instruction() : nullptr;
    if (!inc || inc->opcode() != Opcode::add || inc->parent() != body) return false;
    Value* step_op = inc->operand(0) == iv ? inc->operand(1) : (inc->operand(1) == iv ? inc->operand(0) : nullptr);
    int64_t step = 0;
    if (!get_const_int(step_op, step) || step <= 0) return false;

    info.loop = &loop;
    info.preheader = preheader;
    info.header = header;
    info.body = body;
    info.exit_bb = exit_bb;
    info.iv_param = iv;
    info.init_val = ph_term->branch_target().args[0];
    info.limit_val = cmp->operand(1);
    info.cmp_inst = cmp;
    info.iv_inc_inst = inc;
    info.step = step;
    info.exit_on_false = exit_on_false;
    return true;
}

enum class Reject { None, PureVectorizable, PureScalar, Illegal };

struct Plan {
    DistributableLoopInfo info;
    std::vector<Instruction*> part2;                              // body order
    std::vector<Instruction*> remat;                              // body order
    std::unordered_map<const Value*, const Instruction*> reload;  // value -> the store that holds it
};

class Planner {
public:
    Planner(Function& fn, Plan& plan) : fn_(fn), plan_(plan), info_(plan.info) {}

    Reject run() {
        std::unordered_set<const Instruction*> part2;
        size_t vectorizable = 0;
        size_t scalar = 0;
        for (Instruction* inst : *info_.body) {
            if (!inst || inst->is_terminator() || inst == info_.iv_inc_inst) continue;
            const Opcode op = inst->opcode();
            if (op == Opcode::call) {
                if (!is_pure_leaf_call(fn_, *inst)) return Reject::Illegal;
                part2.insert(inst);
                ++scalar;
            } else if (is_memory_access(op)) {
                if (is_vectorizable_access(*inst)) {
                    ++vectorizable;
                } else {
                    part2.insert(inst);
                    ++scalar;
                }
            } else if (is_pure_computation(*inst)) {
                if (is_supported_vector_arithmetic(op)) {
                    ++vectorizable;
                } else {
                    part2.insert(inst);
                    ++scalar;
                }
            } else {
                return Reject::Illegal;
            }
        }
        if (scalar == 0) return Reject::PureVectorizable;
        if (vectorizable == 0) return Reject::PureScalar;

        // Whatever consumes a second-part value joins the second part.
        for (bool grew = true; grew;) {
            grew = false;
            for (Instruction* inst : *info_.body) {
                if (!inst || inst->is_terminator() || part2.count(inst)) continue;
                bool uses_part2 = false;
                for_each_use(*inst, [&](Value* v) {
                    if (v && v->is_instruction() && part2.count(v->defining_instruction())) uses_part2 = true;
                });
                if (!uses_part2) continue;
                if (inst == info_.iv_inc_inst) return Reject::Illegal;
                part2.insert(inst);
                grew = true;
            }
        }

        std::vector<Instruction*> part1_memory;
        std::vector<Instruction*> part2_memory;
        bool part1_stores = false;
        for (Instruction* inst : *info_.body) {
            if (!inst || inst->is_terminator() || inst == info_.iv_inc_inst) continue;
            const bool second = part2.count(inst) != 0;
            if (second) plan_.part2.push_back(inst);
            if (!is_memory_access(inst->opcode())) continue;
            (second ? part2_memory : part1_memory).push_back(inst);
            if (!second && is_store(inst->opcode())) part1_stores = true;
        }
        if (plan_.part2.empty() || !part1_stores) return Reject::PureScalar;

        AliasAnalysis aa(fn_);
        for (const Instruction* a : part1_memory) {
            for (const Instruction* b : part2_memory) {
                if (!is_store(a->opcode()) && !is_store(b->opcode())) continue;
                if (aa.alias(a->operand(0), b->operand(0)) != AliasResult::NoAlias) return Reject::Illegal;
            }
        }

        // Every value the second part reads must be available in the second loop.
        std::unordered_set<const Instruction*> remat;
        for (const Instruction* inst : plan_.part2) {
            bool ok = true;
            for_each_use(*inst, [&](Value* v) {
                if (ok && !provide(v, part2, part1_memory, aa, remat, 0)) ok = false;
            });
            if (!ok) return Reject::Illegal;
        }
        // The second loop steps its own copy of the induction variable.
        bool step_ok = true;
        for_each_use(*info_.iv_inc_inst, [&](Value* v) {
            if (step_ok && !provide(v, part2, part1_memory, aa, remat, 0)) step_ok = false;
        });
        if (!step_ok) return Reject::Illegal;
        for (Instruction* inst : *info_.body) {
            if (remat.count(inst)) plan_.remat.push_back(inst);
        }
        return Reject::None;
    }

private:
    bool is_vectorizable_access(const Instruction& inst) const {
        const Opcode op = inst.opcode();
        if (op != Opcode::load_indexed && op != Opcode::store_indexed) return false;
        return info_.loop->is_loop_invariant(inst.operand(0)) && inst.operand(1) == info_.iv_param;
    }

    bool defined_in_loop(const Value* v) const {
        if (v->is_block_param()) return info_.loop->contains(v->defining_block());
        const Instruction* def = v->defining_instruction();
        return def && info_.loop->contains(def->parent());
    }

    // Whether the second loop can have `v`: it is defined outside the loop,
    // is the induction variable, comes from the second part, or can be
    // recomputed or reloaded there.
    bool provide(const Value* v, const std::unordered_set<const Instruction*>& part2,
                 const std::vector<Instruction*>& part1_memory, const AliasAnalysis& aa,
                 std::unordered_set<const Instruction*>& remat, int depth) {
        if (!v) return true;
        if (v == info_.iv_param) return true;
        if (!defined_in_loop(v)) return true;
        if (v->is_block_param() || depth > 8) return false;
        const Instruction* def = v->defining_instruction();
        if (def->parent() != info_.body || def == info_.iv_inc_inst) return false;
        if (part2.count(def) || remat.count(def) || plan_.reload.count(v)) return true;
        if (is_pure_computation(*def)) {
            const auto saved_remat = remat;
            const auto saved_reload = plan_.reload;
            bool ok = true;
            for_each_use(*def, [&](Value* op) {
                if (ok && !provide(op, part2, part1_memory, aa, remat, depth + 1)) ok = false;
            });
            if (ok) {
                remat.insert(def);
                return true;
            }
            remat = saved_remat;
            plan_.reload = saved_reload;
        }
        if (const Instruction* store = reloadable_store(v, part1_memory, aa)) {
            plan_.reload[v] = store;
            return true;
        }
        return false;
    }

    // A first-part store of `v` to base[iv] whose slot nothing else in the
    // first loop overwrites: the store's own later iterations hit other slots
    // because the elements do not overlap, and every other store is to a
    // disjoint object. Second-part stores are already known to be disjoint.
    const Instruction* reloadable_store(const Value* v, const std::vector<Instruction*>& part1_memory,
                                        const AliasAnalysis& aa) const {
        const Instruction* found = nullptr;
        for (const Instruction* s : part1_memory) {
            if (s->opcode() != Opcode::store_indexed || s->operand(2) != v || !is_vectorizable_access(*s)) continue;
            if (s->memory_type() != v->type() || s->scale() < v->type().size_in_bytes()) continue;
            found = s;
            break;
        }
        if (!found) return nullptr;
        for (const Instruction* s : part1_memory) {
            if (s == found || !is_store(s->opcode())) continue;
            if (aa.alias(s->operand(0), found->operand(0)) != AliasResult::NoAlias) return nullptr;
        }
        return found;
    }

    Function& fn_;
    Plan& plan_;
    DistributableLoopInfo& info_;
};

Reject plan_distribution(Function& fn, LoopInfo& loop, Plan& plan) {
    if (!extract_distributable_loop(loop, plan.info)) return Reject::Illegal;
    return Planner(fn, plan).run();
}

void record_rejection(Reject r, const LoopDistributionOptions& options) {
    if (!options.stats) return;
    switch (r) {
        case Reject::PureVectorizable: options.stats->rejected_pure_vectorizable++; break;
        case Reject::PureScalar: options.stats->rejected_pure_scalar++; break;
        case Reject::Illegal: options.stats->rejected_cycles++; break;
        case Reject::None: break;
    }
}

} // namespace

bool can_distribute_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopDistributionOptions& options
) {
    (void)dom;
    if (options.stats) options.stats->candidates_checked++;
    Plan plan;
    const Reject r = plan_distribution(fn, loop, plan);
    record_rejection(r, options);
    return r == Reject::None;
}

bool distribute_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    const LoopDistributionOptions& options
) {
    (void)dom;
    if (!fn.parent()) return false;
    Plan plan;
    if (plan_distribution(fn, loop, plan) != Reject::None) return false;
    const DistributableLoopInfo& info = plan.info;

    // ph -> hdr <-> body, hdr -> exit1 -> hdr2 <-> body2, hdr2 -> original exit
    BasicBlock* exit1 = ir::new_block(fn, "dist_exit1");
    BasicBlock* hdr2 = ir::new_block(fn, "dist_hdr2");
    BasicBlock* body2 = ir::new_block(fn, "dist_body2");
    Value* iv2 = ir::new_block_param(fn, hdr2, info.iv_param->type());

    ir::ValueMap values;
    values[info.iv_param] = iv2;
    const ir::BlockMap no_blocks;
    std::unordered_set<const Instruction*> remat(plan.remat.begin(), plan.remat.end());
    std::unordered_set<const Instruction*> part2(plan.part2.begin(), plan.part2.end());

    Builder b(*fn.parent());
    b.set_function(&fn);
    b.position_at_end(body2);

    // Walk the body in order so every value exists in body2 before its use.
    std::vector<Instruction*> order;
    for (Instruction* inst : *info.body) {
        if (inst && !inst->is_terminator()) order.push_back(inst);
    }
    for (Instruction* inst : order) {
        if (remat.count(inst)) {
            body2->append_instruction(ir::clone_instruction(fn, *inst, values, no_blocks));
        } else if (part2.count(inst)) {
            info.body->remove_instruction(inst);
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                auto it = values.find(inst->operand(i));
                if (it != values.end()) inst->set_operand(i, it->second);
            }
            body2->append_instruction(inst);
        }
        if (inst->result()) {
            auto rl = plan.reload.find(inst->result());
            if (rl != plan.reload.end()) {
                const Instruction* store = rl->second;
                Value* loaded = b.build_load_indexed(inst->result()->type(), store->operand(0), iv2,
                                                     store->scale(), store->offset());
                values[inst->result()] = loaded;
            }
        }
    }

    ir::ValueMap tail_values = values;
    Instruction* inc2 = ir::clone_instruction(fn, *info.iv_inc_inst, tail_values, no_blocks);
    body2->append_instruction(inc2);
    const ir::BlockMap to_hdr2{{info.header, hdr2}};
    body2->append_instruction(ir::clone_instruction(fn, *info.body->terminator(), tail_values, to_hdr2));

    ir::ValueMap hdr_values{{info.iv_param, iv2}};
    hdr2->append_instruction(ir::clone_instruction(fn, *info.cmp_inst, hdr_values, no_blocks));
    const ir::BlockMap to_body2{{info.body, body2}};
    hdr2->append_instruction(ir::clone_instruction(fn, *info.header->terminator(), hdr_values, to_body2));

    b.position_at_end(exit1);
    b.build_br(hdr2, {info.init_val});

    Instruction* hdr_term = info.header->terminator();
    BranchTarget& exit_edge = info.exit_on_false ? hdr_term->false_target() : hdr_term->true_target();
    exit_edge.block = exit1;

    fn.rebuild_cfg_predecessors();
    if (options.stats) options.stats->loops_distributed++;
    return true;
}

bool loop_distribution_pass(
    Function& fn,
    const DominatorTree& dom,
    const LoopDistributionOptions& options
) {
    bool any_distributed = false;
    for (bool changed = true; changed;) {
        changed = false;
        fn.rebuild_cfg_predecessors();
        DominatorTree current_dom(fn);
        LoopAnalysis la(fn, current_dom);
        for (LoopInfo* loop : la.post_order_loops()) {
            if (loop && distribute_loop(fn, *loop, current_dom, options)) {
                changed = true;
                any_distributed = true;
                break;
            }
        }
    }
    return any_distributed;
}

} // namespace brass
