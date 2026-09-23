// Outlines a loop accepted by analyze_parallel_loop into a kernel
// `void kernel(i64 start, i64 end, ptr ctx)` that runs iterations
// [start, end) of it, and replaces the loop with one brass_parallel_for call.
//
// The kernel is a faithful copy of the loop: every block except the header
// is cloned with its control flow, the header's own (pure) computations are
// recomputed per iteration from the rebuilt induction variable, and values
// from before the loop arrive through a context block of 8-byte slots.

#include "ir_clone.hpp"
#include "loop_affine.hpp"
#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/opcodes.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace brass {

namespace {

using runtime::ReductionKind;

bool is_fp_kind(ReductionKind k) { return k == ReductionKind::SumF64 || k == ReductionKind::ProdF64; }
bool is_sum_kind(ReductionKind k) { return k == ReductionKind::SumF64 || k == ReductionKind::SumI64; }

// The value that leaves a partial result unchanged. -0.0 for a
// floating-point sum, so that init + identity == init even for init == -0.0.
Value* build_identity(Builder& b, const ParallelLoopInfo& pli, bool wide_int) {
    if (is_fp_kind(pli.reduction_kind)) return b.build_fconst_f64(is_sum_kind(pli.reduction_kind) ? -0.0 : 1.0);
    const int64_t id = is_sum_kind(pli.reduction_kind) ? 0 : 1;
    if (!wide_int && pli.reduction_type == Type::i32()) return b.build_iconst_i32(static_cast<int32_t>(id));
    return b.build_iconst_i64(id);
}

Value* build_int_const(Builder& b, Type t, int64_t v) {
    return t == Type::i32() ? b.build_iconst_i32(static_cast<int32_t>(v)) : b.build_iconst_i64(v);
}

// A copy of an iconst/fconst in the builder's current block.
Value* rematerialize(Builder& b, const Value* v) {
    const Instruction* def = v->defining_instruction();
    switch (def->opcode()) {
        case Opcode::iconst_i32: return b.build_iconst_i32(def->imm_i32());
        case Opcode::iconst_i64: return b.build_iconst_i64(def->imm_i64());
        default: return b.build_fconst_f64(def->imm_f64());
    }
}

// Maps a value from before the loop into the kernel: constants are copied,
// everything else was loaded from the context.
Value* outside_value(Builder& b, ir::ValueMap& vm, Value* v) {
    auto it = vm.find(v);
    if (it != vm.end()) return it->second;
    Value* out = affine::is_plain_constant(v) ? rematerialize(b, v) : v;
    vm[v] = out;
    return out;
}

Function* outline_kernel(Function& fn, const ParallelLoopInfo& pli, const std::string& name) {
    Module* mod = fn.parent();
    Function* kfn = mod->create_function(name, Type::void_type(), {Type::i64(), Type::i64(), Type::ptr()});
    if (!kfn) return nullptr;
    Builder b(*mod);
    b.set_function(kfn);

    BasicBlock* k_entry = ir::new_block(*kfn, "k_entry");
    Value* p_start = ir::new_block_param(*kfn, k_entry, Type::i64());
    Value* p_end = ir::new_block_param(*kfn, k_entry, Type::i64());
    Value* p_ctx = ir::new_block_param(*kfn, k_entry, Type::ptr());
    BasicBlock* k_hdr = ir::new_block(*kfn, "k_hdr");
    BasicBlock* k_iter = ir::new_block(*kfn, "k_iter");
    BasicBlock* k_exit = ir::new_block(*kfn, "k_exit");

    ir::ValueMap vm;
    ir::BlockMap bm;

    // Entry: unpack the context and seed the partial result.
    b.position_at_end(k_entry);
    for (size_t j = 0; j < pli.captured.size(); ++j) {
        Value* v = pli.captured[j];
        vm[v] = b.build_load(v->type(), p_ctx, static_cast<int32_t>(j * 8));
    }
    Value* init_iv = outside_value(b, vm, pli.init_iv);
    std::vector<Value*> inv_values;
    const BranchTarget& ph_bt = pli.preheader->terminator()->branch_target();
    for (size_t i : pli.invariant_param_indices) inv_values.push_back(outside_value(b, vm, ph_bt.args[i]));
    std::vector<Value*> entry_args = {p_start};
    if (pli.has_reduction) entry_args.push_back(build_identity(b, pli, false));
    b.build_br(k_hdr, entry_args);

    // Header: idx < end.
    Value* idx = ir::new_block_param(*kfn, k_hdr, Type::i64());
    Value* acc = pli.has_reduction ? ir::new_block_param(*kfn, k_hdr, pli.reduction_type) : nullptr;
    b.position_at_end(k_hdr);
    Value* more = b.build_ult(idx, p_end);
    if (acc) b.build_br_if(more, k_iter, {}, k_exit, {acc});
    else b.build_br_if(more, k_iter, {}, k_exit, {});

    // Iteration: iv = init + idx * step, then the header's own work.
    b.position_at_end(k_iter);
    Value* n = pli.iv_type == Type::i32() ? b.build_trunc_i32(idx) : idx;
    if (pli.step != 1) n = b.build_mul(n, build_int_const(b, pli.iv_type, pli.step));
    vm[pli.iv_param] = b.build_add(init_iv, n);
    if (acc) vm[pli.reduction_param] = acc;
    for (size_t k = 0; k < pli.invariant_param_indices.size(); ++k) {
        vm[pli.header->param(pli.invariant_param_indices[k])] = inv_values[k];
    }

    // Constants from before the loop are copied into the entry block.
    b.position_before(k_entry->terminator());
    for (BasicBlock* bb : pli.loop->blocks()) {
        for (Instruction* inst : *bb) {
            for_each_use_slot(*inst, [&](Value*& v) {
                if (v && affine::defined_outside(*pli.loop, v) && affine::is_plain_constant(v)) outside_value(b, vm, v);
            });
        }
    }

    // Shells first so every use can be mapped, then the uses.
    Instruction* hdr_term = pli.header->terminator();
    Instruction* latch_term = pli.latch->terminator();
    std::vector<std::pair<const Instruction*, Instruction*>> clones;
    for (Instruction* inst : *pli.header) {
        if (inst == hdr_term) continue;
        Instruction* c = ir::clone_shell(*kfn, *inst, vm);
        k_iter->append_instruction(c);
        clones.emplace_back(inst, c);
    }
    for (BasicBlock* bb : pli.loop->blocks()) {
        if (bb == pli.header) continue;
        BasicBlock* kb = ir::new_block(*kfn, std::string(bb->name()));
        bm[bb] = kb;
        for (Value* p : bb->params()) vm[p] = ir::new_block_param(*kfn, kb, p->type());
    }
    for (BasicBlock* bb : pli.loop->blocks()) {
        if (bb == pli.header) continue;
        for (Instruction* inst : *bb) {
            if (inst == latch_term) continue;
            Instruction* c = ir::clone_shell(*kfn, *inst, vm);
            bm[bb]->append_instruction(c);
            clones.emplace_back(inst, c);
        }
    }
    for (const auto& [src, dst] : clones) ir::clone_uses(*src, *dst, vm, bm);

    // k_iter enters the body the way the header did.
    b.position_at_end(k_iter);
    std::vector<Value*> body_args;
    for (Value* a : hdr_term->true_target().args) body_args.push_back(vm.count(a) ? vm[a] : a);
    b.build_br(bm[pli.body], body_args);

    // The latch goes on to the next index.
    b.position_at_end(bm[pli.latch]);
    Value* next_idx = b.build_add(idx, b.build_iconst_i64(1));
    if (acc) b.build_br(k_hdr, {next_idx, vm.at(pli.reduction_op_inst->result())});
    else b.build_br(k_hdr, {next_idx});

    // Exit: hand the partial result to the runtime.
    b.position_at_end(k_exit);
    if (pli.has_reduction) {
        Value* part = ir::new_block_param(*kfn, k_exit, pli.reduction_type);
        if (is_fp_kind(pli.reduction_kind)) {
            b.build_call("brass_parallel_reduce_f64", Type::void_type(), {part});
        } else {
            Value* wide = pli.reduction_type == Type::i32() ? b.build_sext_i64(part) : part;
            b.build_call("brass_parallel_reduce_i64", Type::void_type(), {wide});
        }
    }
    b.build_ret_void();
    kfn->rebuild_cfg_predecessors();
    return kfn;
}

// Iterations of the loop: cmp(init, limit) ? ceil((limit - init) / step) : 0,
// with the difference taken in the induction variable's own width (exact as
// an unsigned value whenever the loop runs at all).
Value* build_trip_count(Builder& b, const ParallelLoopInfo& pli, Value* limit) {
    Value* init = pli.init_iv;
    Value* runs = pli.cmp_opcode == Opcode::slt ? b.build_slt(init, limit) : b.build_ult(init, limit);
    Value* diff = b.build_sub(limit, init);
    Value* wide = pli.iv_type == Type::i32() ? b.build_zext_i64(diff) : diff;
    Value* trips = wide;
    if (pli.step != 1) {
        Value* step = b.build_iconst_i64(pli.step);
        Value* q = b.build_udiv(wide, step);
        Value* r = b.build_umod(wide, step);
        Value* partial = b.build_ne(r, b.build_iconst_i64(0));
        trips = b.build_select(partial, b.build_add(q, b.build_iconst_i64(1)), q);
    }
    return b.build_select(runs, trips, b.build_iconst_i64(0));
}

bool transform_parallel_loop(Function& fn, ParallelLoopInfo& pli, const ParallelLoopOptions& options) {
    if (pli.has_const_trip_count) {
        const uint64_t work_units = pli.const_trip_count * pli.instruction_count;
        if (work_units < options.parallel_threshold) {
            if (options.stats) options.stats->loops_rejected_cost++;
            return false;
        }
    }

    Module* mod = fn.parent();
    if (!mod) return false;
    const std::string kernel_name = std::string(fn.name()) + "_par_k_" + std::to_string(pli.header->id());
    if (mod->get_function(kernel_name)) return false;
    if (!outline_kernel(fn, pli, kernel_name)) return false;

    Instruction* ph_term = pli.preheader->terminator();
    const BranchTarget ph_bt = ph_term->branch_target();
    Builder b(*mod);
    b.set_function(&fn);
    b.position_before(ph_term);

    Value* limit = affine::defined_outside(*pli.loop, pli.limit_val) ? pli.limit_val : rematerialize(b, pli.limit_val);
    if (pli.needs_wrap_guard) {
        // Limits where the step would wrap keep the sequential loop, which is
        // marked so it is not parallelized again.
        BasicBlock* dispatch = ir::new_block(fn, std::string(pli.header->name()) + "_par_dispatch");
        Value* max = build_int_const(b, pli.iv_type, pli.wrap_free_limit_max);
        Value* ok = pli.cmp_opcode == Opcode::slt ? b.build_sle(limit, max) : b.build_ule(limit, max);
        b.build_br_if(ok, dispatch, {}, pli.header, ph_bt.args);
        pli.preheader->remove_instruction(ph_term);
        pli.header->set_name(mod->string_pool().intern(std::string(pli.header->name()) + "_par_seq"));
        b.position_at_end(dispatch);
    }
    Value* trips = build_trip_count(b, pli, limit);

    const size_t slots = pli.captured.empty() ? 1 : pli.captured.size();
    Value* ctx = b.build_call("brass_parallel_alloc_context", Type::ptr(), {b.build_iconst_i64(static_cast<int64_t>(slots * 8))});
    for (size_t j = 0; j < pli.captured.size(); ++j) {
        Value* v = pli.captured[j];
        b.build_store(v->type(), ctx, static_cast<int32_t>(j * 8), v);
    }
    Value* red = nullptr;
    if (pli.has_reduction) {
        red = b.build_call("brass_parallel_alloc_context", Type::ptr(), {b.build_iconst_i64(8)});
        Value* id = build_identity(b, pli, true);
        b.build_store(id->type(), red, 0, id);
    }
    Value* kernel = b.build_func_addr(kernel_name);
    Value* kind = b.build_iconst_i32(static_cast<int32_t>(pli.reduction_kind));
    b.build_call("brass_parallel_for", Type::void_type(),
                 {trips, b.build_iconst_i64(0), kernel, ctx, kind, red ? red : b.build_iconst_i64(0)});
    b.build_call("brass_parallel_free_context", Type::void_type(), {ctx});

    // Values of the header parameters once the loop is done.
    ir::ValueMap final_values;
    Value* tn = pli.iv_type == Type::i32() ? b.build_trunc_i32(trips) : trips;
    if (pli.step != 1) tn = b.build_mul(tn, build_int_const(b, pli.iv_type, pli.step));
    final_values[pli.iv_param] = b.build_add(pli.init_iv, tn);
    if (pli.has_reduction) {
        const bool fp = is_fp_kind(pli.reduction_kind);
        Value* part = b.build_load(fp ? Type::f64() : Type::i64(), red, 0);
        b.build_call("brass_parallel_free_context", Type::void_type(), {red});
        if (pli.reduction_type == Type::i32()) part = b.build_trunc_i32(part);
        Value* init = pli.reduction_init_val;
        final_values[pli.reduction_param] = is_sum_kind(pli.reduction_kind) ? b.build_add(init, part) : b.build_mul(init, part);
    }
    for (size_t i : pli.invariant_param_indices) final_values[pli.header->param(i)] = ph_bt.args[i];

    // The preheader now continues straight to the exit.
    const BranchTarget& exit_bt = pli.header->terminator()->false_target();
    BranchTarget to_exit;
    to_exit.block = pli.exit_bb;
    for (Value* a : exit_bt.args) {
        auto it = final_values.find(a);
        to_exit.args.push_back(it != final_values.end() ? it->second : a);
    }
    if (pli.needs_wrap_guard) {
        b.build_br(to_exit.block, to_exit.args);
        fn.rebuild_cfg_predecessors();
        if (options.stats) options.stats->parallel_loops_transformed++;
        return true;
    }
    ph_term->set_branch_target(to_exit);

    // Code after the loop that read a header parameter directly reads its
    // final value (the preheader dominates everything the header did).
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb || pli.loop->contains(bb)) continue;
        for (Instruction* inst : *bb) {
            for_each_use_slot(*inst, [&](Value*& v) {
                auto it = final_values.find(v);
                if (it != final_values.end()) v = it->second;
            });
        }
    }

    const std::vector<BasicBlock*> dead = pli.loop->blocks();
    for (BasicBlock* bb : dead) fn.remove_block(bb);
    fn.rebuild_cfg_predecessors();
    if (options.stats) options.stats->parallel_loops_transformed++;
    return true;
}

} // namespace

bool auto_parallelize_function(
    Function& fn,
    const DominatorTree& dom,
    const ParallelLoopOptions& options
) {
    if (fn.name().find("_par_k_") != std::string_view::npos) return false;
    fn.rebuild_cfg_predecessors();
    LoopAnalysis loop_analysis(fn, dom);
    for (LoopInfo* loop : loop_analysis.post_order_loops()) {
        if (!loop) continue;
        if (options.stats) options.stats->loops_analyzed++;

        ParallelLoopInfo pli;
        if (analyze_parallel_loop(fn, *loop, dom, pli, options) && pli.is_parallelizable() &&
            transform_parallel_loop(fn, pli, options)) {
            return true; // The CFG changed; the caller re-analyzes.
        }
    }
    return false;
}

} // namespace brass
