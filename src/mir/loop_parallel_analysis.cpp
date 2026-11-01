// Legality of running a loop's iterations in parallel.
//
// The accepted shape is exactly what the outliner in
// loop_parallel_transform.cpp reproduces faithfully:
//   - one entry, from a preheader ending in `br header(...)`, one latch
//     ending in `br header(...)`, and one exit, the header's false edge;
//   - the header tests `iv slt|ult limit` with `limit` fixed before the loop,
//     and the latch advances `iv` by a positive constant that cannot wrap;
//   - every other header parameter is passed back unchanged, or is the one
//     integer (or opted-in floating-point) sum/product reduction;
//   - the header computes nothing that is used after the loop, and nothing
//     in the loop has an effect other than plain loads and stores (no calls,
//     no traps, no allocation);
//   - no two iterations can touch the same bytes where one of them writes.

#include "loop_affine.hpp"
#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/alias_analysis.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/opcodes.hpp>
#include <cstdlib>
#include <limits>
#include <unordered_set>

namespace brass {

namespace {

bool captured_type_ok(Type t) noexcept {
    return !t.is_vector() && !t.is_void() && t.size_in_bytes() <= 8;
}

// Visits every value the loop reads during an iteration: every use slot in
// the loop except the arguments of the header's exit edge, which are read
// only on leaving it.
void for_each_iteration_use(const LoopInfo& loop, const std::function<void(Instruction&, Value*&)>& f) {
    Instruction* hdr_term = loop.header()->terminator();
    for (BasicBlock* bb : loop.blocks()) {
        for (Instruction* inst : *bb) {
            if (inst == hdr_term) {
                for (Value*& v : inst->operands()) f(*inst, v);
                for (Value*& v : inst->true_target().args) f(*inst, v);
                continue;
            }
            for_each_use_slot(*inst, [&](Value*& u) { f(*inst, u); });
        }
    }
}

size_t uses_in_loop(const LoopInfo& loop, const Value* v) {
    size_t n = 0;
    for_each_iteration_use(loop, [&](Instruction&, Value*& u) { n += (u == v); });
    return n;
}

bool in_sub_loop(const LoopInfo& loop, const BasicBlock* bb) {
    for (const auto& sub : loop.sub_loops()) {
        if (sub && sub->contains(bb)) return true;
    }
    return false;
}

// The largest limit (signed for slt, unsigned for ult, as a bit pattern of
// the iv's width) for which the step cannot wrap: the largest iv value that
// passes the test is limit - 1, so limit - 1 + step must still fit.
int64_t max_wrap_free_limit(Opcode cmp, Type iv_type, int64_t step) {
    const bool wide = iv_type == Type::i64();
    if (cmp == Opcode::slt) {
        const int64_t max = wide ? std::numeric_limits<int64_t>::max() : std::numeric_limits<int32_t>::max();
        return max - (step - 1);
    }
    const uint64_t umax = wide ? std::numeric_limits<uint64_t>::max() : std::numeric_limits<uint32_t>::max();
    const uint64_t lim = umax - static_cast<uint64_t>(step - 1);
    return wide ? static_cast<int64_t>(lim) : static_cast<int64_t>(static_cast<int32_t>(static_cast<uint32_t>(lim)));
}

bool const_limit_wrap_free(Opcode cmp, Type iv_type, int64_t lim, int64_t step) {
    const int64_t max = max_wrap_free_limit(cmp, iv_type, step);
    const bool wide = iv_type == Type::i64();
    if (cmp == Opcode::slt) return (wide ? lim : static_cast<int32_t>(lim)) <= max;
    const uint64_t ul = wide ? static_cast<uint64_t>(lim) : static_cast<uint32_t>(lim);
    const uint64_t um = wide ? static_cast<uint64_t>(max) : static_cast<uint32_t>(max);
    return ul <= um;
}

bool header_param_used_after_loop(const Function& fn, const LoopInfo& loop) {
    for (Value* p : loop.header()->params()) {
        if (affine::used_outside(fn, loop, p)) return true;
    }
    return false;
}

bool const_trip_count(const ParallelLoopInfo& pli, uint64_t& out) {
    int64_t init = 0, lim = 0;
    if (!affine::is_int_constant(pli.init_iv, init) || !affine::is_int_constant(pli.limit_val, lim)) return false;
    const bool wide = pli.iv_type == Type::i64();
    uint64_t diff = 0;
    if (pli.cmp_opcode == Opcode::slt) {
        if (!wide) { init = static_cast<int32_t>(init); lim = static_cast<int32_t>(lim); }
        if (init >= lim) { out = 0; return true; }
        diff = static_cast<uint64_t>(lim) - static_cast<uint64_t>(init);
    } else {
        const uint64_t ui = wide ? static_cast<uint64_t>(init) : static_cast<uint32_t>(init);
        const uint64_t ul = wide ? static_cast<uint64_t>(lim) : static_cast<uint32_t>(lim);
        if (ui >= ul) { out = 0; return true; }
        diff = ul - ui;
    }
    const uint64_t step = static_cast<uint64_t>(pli.step);
    out = diff / step + (diff % step != 0 ? 1 : 0);
    return true;
}

// Every instruction the outlined kernel may contain.
bool body_instruction_ok(const Instruction& inst) {
    if (affine::is_pure_nontrapping(inst)) return true;
    const Opcode op = inst.opcode();
    return affine::is_plain_memory_access(op) || op == Opcode::br || op == Opcode::br_if || op == Opcode::switch_;
}

// A write-involving pair of accesses that different iterations can make to
// overlapping bytes.
bool may_conflict(const affine::Form& x, const affine::Form& y, const Value* iv, int64_t step,
                  const AliasAnalysis& aa) {
    int64_t period = 0;
    if (x.ok && y.ok && x.invariants == y.invariants && x.terms == y.terms && x.terms.size() == 1) {
        const auto& [key, coeff] = *x.terms.begin();
        // Consecutive iterations are `step` apart in the iv.
        if (key.first == iv && key.second == nullptr && affine::checked_mul(coeff, step, period) &&
            period != std::numeric_limits<int64_t>::min()) {
            period = std::llabs(period);
            // Iterations are at least one stride apart. With equal constant
            // parts two different iterations never meet when the stride
            // covers the access; otherwise the second access must sit in the
            // gap the first leaves within every stride period.
            if (x.constant == y.constant && x.size == y.size) return period < static_cast<int64_t>(x.size);
            int64_t delta = 0;
            if (!affine::checked_sub(y.constant, x.constant, delta)) return true;
            const int64_t r = ((delta % period) + period) % period;
            return !(r >= static_cast<int64_t>(x.size) && static_cast<int64_t>(y.size) <= period - r);
        }
    }
    return !affine::distinct_objects(aa, x.pointer, y.pointer);
}

} // namespace

bool analyze_parallel_loop(
    Function& fn,
    LoopInfo& loop,
    const DominatorTree& dom,
    ParallelLoopInfo& pli,
    const ParallelLoopOptions& options
) {
    (void)dom;
    pli = ParallelLoopInfo();
    pli.loop = &loop;
    pli.kind = LoopParallelKind::Sequential;
    auto reject = [&pli](const char* why) {
        pli.rejection_reason = why;
        return false;
    };

    BasicBlock* header = loop.header();
    if (!header || loop.latches().size() != 1) return reject("loop without single header/latch");
    if (fn.name().find("_par_k_") != std::string_view::npos ||
        header->name().find("_par_seq") != std::string_view::npos) {
        return reject("already parallelized");
    }

    BasicBlock* latch = loop.latches()[0];
    Instruction* latch_term = latch ? latch->terminator() : nullptr;
    if (!latch_term || latch_term->opcode() != Opcode::br || latch_term->branch_target().block != header ||
        latch_term->branch_target().args.size() != header->param_count()) {
        return reject("invalid latch branch");
    }

    BasicBlock* preheader = nullptr;
    for (BasicBlock* pred : header->predecessors()) {
        if (!pred || loop.contains(pred)) continue;
        if (preheader && preheader != pred) return reject("loop with more than one entry");
        preheader = pred;
    }
    Instruction* ph_term = preheader ? preheader->terminator() : nullptr;
    if (!ph_term || ph_term->opcode() != Opcode::br || ph_term->branch_target().block != header ||
        ph_term->branch_target().args.size() != header->param_count()) {
        return reject("invalid preheader branch");
    }

    Instruction* hdr_term = header->terminator();
    if (!hdr_term || hdr_term->opcode() != Opcode::br_if) return reject("header terminator not br_if");
    BasicBlock* body = hdr_term->true_target().block;
    BasicBlock* exit = hdr_term->false_target().block;
    if (!body || !exit || !loop.contains(body) || loop.contains(exit)) {
        return reject("header must continue on true and exit on false");
    }
    for (BasicBlock* bb : loop.blocks()) {
        for (BasicBlock* succ : bb->successors()) {
            if (!loop.contains(succ) && !(bb == header && succ == exit)) return reject("multi-exit loop");
        }
    }

    Value* cond = hdr_term->operand(0);
    Instruction* cmp = cond && cond->is_instruction() ? cond->defining_instruction() : nullptr;
    if (!cmp || cmp->parent() != header || (cmp->opcode() != Opcode::slt && cmp->opcode() != Opcode::ult)) {
        return reject("unsupported loop condition comparison");
    }
    Value* iv = cmp->operand(0);
    Value* limit = cmp->operand(1);
    if (!iv || !iv->is_block_param() || iv->defining_block() != header) {
        return reject("could not identify countable induction variable");
    }
    int64_t ignored = 0;
    if (!affine::defined_outside(loop, limit) && !affine::is_int_constant(limit, ignored)) {
        return reject("loop limit is not fixed before the loop");
    }
    const size_t iv_idx = iv->param_index();
    if (iv->type() != Type::i32() && iv->type() != Type::i64()) return reject("unsupported induction variable type");

    Value* iv_next = latch_term->branch_target().args[iv_idx];
    Instruction* inc = iv_next && iv_next->is_instruction() ? iv_next->defining_instruction() : nullptr;
    int64_t step = 0;
    if (!inc || inc->opcode() != Opcode::add) return reject("could not identify countable induction variable");
    if (!((inc->operand(0) == iv && affine::is_int_constant(inc->operand(1), step)) ||
          (inc->operand(1) == iv && affine::is_int_constant(inc->operand(0), step))) || step <= 0 ||
        step > std::numeric_limits<int32_t>::max()) {
        return reject("could not identify countable induction variable");
    }
    int64_t const_limit = 0;
    if (step != 1) {
        pli.wrap_free_limit_max = max_wrap_free_limit(cmp->opcode(), iv->type(), step);
        if (affine::is_int_constant(limit, const_limit)) {
            if (!const_limit_wrap_free(cmp->opcode(), iv->type(), const_limit, step)) {
                return reject("induction variable step wraps");
            }
        } else {
            // The loop keeps its sequential form for limits where the step
            // would wrap; after it, the header's values reach the code that
            // follows only through the exit edge.
            pli.needs_wrap_guard = true;
            if (header_param_used_after_loop(fn, loop)) return reject("induction variable step may wrap");
        }
    }

    pli.header = header;
    pli.preheader = preheader;
    pli.body = body;
    pli.latch = latch;
    pli.exit_bb = exit;
    pli.exit_on_false = true;
    pli.iv_param_index = iv_idx;
    pli.iv_param = iv;
    pli.iv_type = iv->type();
    pli.init_iv = ph_term->branch_target().args[iv_idx];
    pli.limit_val = limit;
    pli.step = step;
    pli.cmp_opcode = cmp->opcode();
    pli.has_const_trip_count = const_trip_count(pli, pli.const_trip_count);

    // Instructions: header work must be repeatable and stay inside the loop;
    // the body may only compute and access memory.
    pli.instruction_count = 0;
    for (BasicBlock* bb : loop.blocks()) {
        for (Instruction* inst : *bb) {
            ++pli.instruction_count;
            const bool ok = bb == header ? (inst == hdr_term || affine::is_pure_nontrapping(*inst))
                                         : body_instruction_ok(*inst);
            if (!ok) {
                if (options.stats) options.stats->loops_rejected_uncontrolled_effects++;
                return reject("loop contains uncontrolled side effects or exceptions");
            }
            if (bb == header && inst->result() && affine::used_outside(fn, loop, inst->result())) {
                return reject("header value used after the loop");
            }
        }
    }

    // Header parameters other than the induction variable.
    const bool allow_fp = options.allow_fp_reassociation || fn.allow_fp_reassociation() ||
                          (fn.parent() && fn.parent()->allow_fp_reassociation());
    for (size_t i = 0; i < header->param_count(); ++i) {
        if (i == iv_idx) continue;
        Value* param = header->param(i);
        Value* next = latch_term->branch_target().args[i];
        if (next == param) {
            pli.invariant_param_indices.push_back(i);
            continue;
        }
        Instruction* def = next && next->is_instruction() ? next->defining_instruction() : nullptr;
        const Type t = param->type();
        const bool int_red = t == Type::i32() || t == Type::i64();
        const bool fp_red = t == Type::f64();
        if (pli.has_reduction || !def || !def->parent() || !loop.contains(def->parent()) ||
            def->parent() == header || in_sub_loop(loop, def->parent()) ||
            (def->opcode() != Opcode::add && def->opcode() != Opcode::mul) || (!int_red && !fp_red) ||
            (def->operand(0) == param) == (def->operand(1) == param) ||
            uses_in_loop(loop, param) != 1 || uses_in_loop(loop, next) != 1) {
            return reject("unsupported loop-carried value");
        }
        if (fp_red && !allow_fp) return reject("FP reduction requires opt-in FP reassociation");
        const bool sum = def->opcode() == Opcode::add;
        pli.has_reduction = true;
        pli.reduction_kind = fp_red ? (sum ? runtime::ReductionKind::SumF64 : runtime::ReductionKind::ProdF64)
                                    : (sum ? runtime::ReductionKind::SumI64 : runtime::ReductionKind::ProdI64);
        pli.reduction_param_index = i;
        pli.reduction_param = param;
        pli.reduction_init_val = ph_term->branch_target().args[i];
        pli.reduction_op_inst = def;
        pli.reduction_type = t;
    }

    // Values the kernel must receive.
    std::unordered_set<Value*> seen;
    bool bad_capture = false;
    auto capture = [&](Value* v) {
        if (!v || !affine::defined_outside(loop, v) || affine::is_plain_constant(v)) return;
        if (!captured_type_ok(v->type())) bad_capture = true;
        if (seen.insert(v).second) pli.captured.push_back(v);
    };
    for_each_iteration_use(loop, [&](Instruction&, Value*& v) { capture(v); });
    // The kernel rebuilds the induction variable and the unchanged header
    // parameters from their preheader values.
    capture(pli.init_iv);
    for (size_t i : pli.invariant_param_indices) capture(ph_term->branch_target().args[i]);
    if (bad_capture) return reject("loop reads a value the kernel context cannot carry");

    // Memory: no two iterations may touch overlapping bytes where one writes.
    const std::vector<const Value*> ivs = {iv};
    const affine::InvariantFn invariant = [&loop](const Value* v) { return affine::defined_outside(loop, v); };
    std::vector<affine::Form> forms;
    for (BasicBlock* bb : loop.blocks()) {
        for (Instruction* inst : *bb) {
            if (affine::is_plain_memory_access(inst->opcode())) forms.push_back(affine::address_form(*inst, ivs, invariant));
        }
    }
    AliasAnalysis aa(fn);
    for (size_t a = 0; a < forms.size(); ++a) {
        for (size_t b = a; b < forms.size(); ++b) {
            if (!forms[a].is_store && !forms[b].is_store) continue;
            if (may_conflict(forms[a], forms[b], iv, step, aa)) {
                if (options.stats) options.stats->loops_rejected_carried_dependence++;
                return reject("loop-carried memory dependence detected");
            }
        }
    }

    if (pli.has_reduction) {
        pli.kind = LoopParallelKind::Reduction;
        if (options.stats) options.stats->reduction_loops_found++;
    } else {
        pli.kind = LoopParallelKind::DOALL;
        if (options.stats) options.stats->doall_loops_found++;
    }
    return true;
}

} // namespace brass
