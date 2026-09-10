#include <brass/mir/loop_parallel.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/verifier.hpp>
#include <brass/mir/opcodes.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>

namespace brass {

namespace {

bool is_constant_or_pure_value(const Value* val) {
    if (!val || !val->is_instruction()) return false;
    const Instruction* inst = val->defining_instruction();
    if (!inst) return false;
    Opcode op = inst->opcode();
    return is_constant(op);
}

void collect_captured_invariants(
    const Function& fn,
    const LoopInfo& loop,
    const ParallelLoopInfo& pli,
    std::vector<Value*>& out_invariants
) {
    (void)fn;
    std::unordered_set<Value*> seen;

    for (BasicBlock* bb : loop.blocks()) {
        if (!bb) continue;
        for (Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
            for (size_t i = 0; i < inst->operand_count(); ++i) {
                Value* op = inst->operand(i);
                if (!op) continue;

                // If defined inside loop, not an invariant
                if (op->is_instruction() && loop.contains(op->defining_instruction())) continue;
                if (op->is_block_param() && loop.contains(op->defining_block())) continue;

                // Skip primary IV and reduction initial/accumulator
                if (op == pli.iv_param || op == pli.init_iv) continue;
                if (pli.has_reduction && (op == pli.reduction_param || op == pli.reduction_init_val)) continue;

                if (seen.insert(op).second) {
                    out_invariants.push_back(op);
                }
            }
        }
    }
}

Function* outline_parallel_kernel(
    Function& fn,
    const ParallelLoopInfo& pli,
    const std::vector<Value*>& captured_invariants,
    const std::string& kernel_name
) {
    Module* mod = fn.parent();
    if (!mod) return nullptr;

    Arena& arena = mod->arena();
    StringPool& sp = mod->string_pool();

    Function* kernel_fn = mod->create_function(
        kernel_name, Type::void_type(), {Type::i64(), Type::i64(), Type::ptr()}
    );
    if (!kernel_fn) return nullptr;

    Builder b(*kernel_fn);

    BasicBlock* k_entry = b.append_block("k_entry");
    BasicBlock* k_hdr = b.create_block("k_hdr");
    BasicBlock* k_body = b.create_block("k_body");
    BasicBlock* k_exit = b.create_block("k_exit");

    // 1. Unpack context in k_entry
    Value* p_start = b.add_block_param(k_entry, Type::i64());
    Value* p_end = b.add_block_param(k_entry, Type::i64());
    Value* p_ctx = b.add_block_param(k_entry, Type::ptr());
    b.position_at_end(k_entry);

    std::unordered_map<const Value*, Value*> val_map;

    for (size_t j = 0; j < captured_invariants.size(); ++j) {
        Value* orig_inv = captured_invariants[j];
        Type inv_t = orig_inv->type();
        Value* unpacked = b.build_load(inv_t, p_ctx, static_cast<int32_t>(j * 8));
        val_map[orig_inv] = unpacked;
    }

    Value* k_init_acc = nullptr;
    if (pli.has_reduction) {
        if (pli.reduction_type.is_float()) {
            double d_id = (pli.reduction_kind == runtime::ReductionKind::ProdF64) ? 1.0 :
                          (pli.reduction_kind == runtime::ReductionKind::MinF64) ? std::numeric_limits<double>::infinity() :
                          (pli.reduction_kind == runtime::ReductionKind::MaxF64) ? -std::numeric_limits<double>::infinity() : 0.0;
            k_init_acc = b.build_fconst_f64(d_id);
        } else {
            int64_t i_id = (pli.reduction_kind == runtime::ReductionKind::ProdI64) ? 1 :
                           (pli.reduction_kind == runtime::ReductionKind::MinI64) ? std::numeric_limits<int64_t>::max() :
                           (pli.reduction_kind == runtime::ReductionKind::MaxI64) ? std::numeric_limits<int64_t>::min() : 0;
            k_init_acc = (pli.reduction_type == Type::i32())
                ? b.build_iconst_i32(static_cast<int32_t>(i_id))
                : b.build_iconst_i64(i_id);
        }
    }

    if (pli.has_reduction) {
        b.build_br(k_hdr, {p_start, k_init_acc});
    } else {
        b.build_br(k_hdr, {p_start});
    }

    // 2. Kernel Header
    kernel_fn->append_block(k_hdr);
    b.position_at_end(k_hdr);
    Value* k_iv = b.add_block_param(k_hdr, pli.iv_type);
    val_map[pli.iv_param] = k_iv;

    Value* k_acc_param = nullptr;
    if (pli.has_reduction) {
        k_acc_param = b.add_block_param(k_hdr, pli.reduction_type);
        val_map[pli.reduction_param] = k_acc_param;
    }

    Value* k_cond = b.build_slt(k_iv, p_end);
    if (pli.has_reduction) {
        b.build_br_if(k_cond, k_body, {}, k_exit, {k_acc_param});
    } else {
        b.build_br_if(k_cond, k_body, {}, k_exit, {});
    }

    // 3. Kernel Body
    kernel_fn->append_block(k_body);
    b.position_at_end(k_body);

    for (BasicBlock* orig_bb : pli.loop->blocks()) {
        if (!orig_bb) continue;
        for (Instruction* inst = orig_bb->head(); inst != nullptr; inst = inst->next()) {
            if (inst->is_terminator()) continue;

            // Skip primary IV step instruction in latch
            if (inst->result() && inst->result() == pli.latch->terminator()->branch_target().args[pli.iv_param_index]) {
                continue;
            }

            Opcode op = inst->opcode();
            Type t = inst->type();
            std::vector<Value*> cloned_args;
            cloned_args.reserve(inst->operand_count());

            for (size_t i = 0; i < inst->operand_count(); ++i) {
                Value* opd = inst->operand(i);
                auto it = val_map.find(opd);
                if (it != val_map.end()) {
                    cloned_args.push_back(it->second);
                } else if (is_constant_or_pure_value(opd)) {
                    // Clone invariant constant
                    Instruction* def = opd->defining_instruction();
                    Instruction* c_inst = arena.make<Instruction>(def->opcode(), def->type());
                    c_inst->set_imm_i64(def->imm_i64());
                    c_inst->set_imm_f64(def->imm_f64());
                    Value* c_res = b.create_value(def->type());
                    c_res->set_defining_instruction(c_inst);
                    c_inst->set_result(c_res);
                    b.insert(c_inst);
                    val_map[opd] = c_res;
                    cloned_args.push_back(c_res);
                } else {
                    cloned_args.push_back(opd);
                }
            }

            Instruction* new_inst = arena.make<Instruction>(op, t);
            for (Value* arg : cloned_args) {
                new_inst->add_operand(arg);
            }
            new_inst->set_imm_i64(inst->imm_i64());
            new_inst->set_imm_f64(inst->imm_f64());
            new_inst->set_scale(inst->scale());
            new_inst->set_offset(inst->offset());
            new_inst->set_memory_type(inst->memory_type());
            if (!inst->symbol().empty()) {
                new_inst->set_symbol(sp.intern(inst->symbol()));
            }

            if (inst->produces_value()) {
                Value* new_res = b.create_value(t);
                new_res->set_defining_instruction(new_inst);
                new_inst->set_result(new_res);
                val_map[inst->result()] = new_res;
            }

            b.insert(new_inst);
        }
    }

    Value* step_val = (pli.iv_type == Type::i32())
        ? b.build_iconst_i32(static_cast<int32_t>(pli.step))
        : b.build_iconst_i64(pli.step);
    Value* k_next_iv = b.build_add(k_iv, step_val);

    if (pli.has_reduction) {
        Value* k_next_acc = val_map[pli.reduction_op_inst->result()];
        b.build_br(k_hdr, {k_next_iv, k_next_acc});
    } else {
        b.build_br(k_hdr, {k_next_iv});
    }

    // 4. Kernel Exit
    kernel_fn->append_block(k_exit);
    b.position_at_end(k_exit);

    if (pli.has_reduction) {
        Value* chunk_acc = b.add_block_param(k_exit, pli.reduction_type);
        if (pli.reduction_type.is_float()) {
            b.build_call("brass_parallel_reduce_f64", Type::void_type(), {chunk_acc});
        } else {
            Value* acc_i64 = (pli.reduction_type == Type::i32())
                ? b.build_sext_i64(chunk_acc)
                : chunk_acc;
            b.build_call("brass_parallel_reduce_i64", Type::void_type(), {acc_i64});
        }
    }

    b.build_ret(nullptr);

    kernel_fn->rebuild_cfg_predecessors();
    return kernel_fn;
}

bool transform_parallel_loop(
    Function& fn,
    ParallelLoopInfo& pli,
    const ParallelLoopOptions& options
) {
    // 1. Cost model threshold check
    if (pli.has_const_trip_count) {
        uint64_t work_units = pli.const_trip_count * pli.instruction_count;
        if (work_units < options.parallel_threshold) {
            if (options.stats) options.stats->loops_rejected_cost++;
            return false;
        }
    }

    // 2. Collect captured loop invariants
    std::vector<Value*> captured_invariants;
    collect_captured_invariants(fn, *pli.loop, pli, captured_invariants);

    // 3. Outline kernel function
    std::string kernel_name = std::string(fn.name()) + "_par_k_" + std::to_string(pli.header->id());
    Function* kernel_fn = outline_parallel_kernel(fn, pli, captured_invariants, kernel_name);
    if (!kernel_fn) return false;

    // 4. Transform original loop preheader & dispatch
    Builder b(fn);
    b.position_before(pli.preheader->terminator());

    Value* ctx_ptr = nullptr;
    if (!captured_invariants.empty()) {
        size_t ctx_bytes = captured_invariants.size() * 8;
        Value* sz_val = b.build_iconst_i64(static_cast<int64_t>(ctx_bytes));
        ctx_ptr = b.build_call("brass_parallel_alloc_context", Type::ptr(), {sz_val});

        for (size_t j = 0; j < captured_invariants.size(); ++j) {
            Value* inv_val = captured_invariants[j];
            b.build_store(inv_val->type(), ctx_ptr, static_cast<int32_t>(j * 8), inv_val);
        }
    } else {
        ctx_ptr = b.build_iconst_i64(0);
    }

    // Calculate trip count
    Value* trip_count_val = nullptr;
    if (pli.has_const_trip_count) {
        trip_count_val = b.build_iconst_i64(static_cast<int64_t>(pli.const_trip_count));
    } else {
        Value* init_ext = (pli.init_iv->type() == Type::i32()) ? b.build_sext_i64(pli.init_iv) : pli.init_iv;
        Value* limit_ext = (pli.limit_val->type() == Type::i32()) ? b.build_sext_i64(pli.limit_val) : pli.limit_val;
        Value* diff = b.build_sub(limit_ext, init_ext);
        if (pli.step > 1) {
            Value* s = b.build_iconst_i64(pli.step);
            Value* added = b.build_add(diff, b.build_iconst_i64(pli.step - 1));
            trip_count_val = b.build_sdiv(added, s);
        } else {
            trip_count_val = diff;
        }
    }

    Value* grain_val = b.build_iconst_i64(0);
    Value* kernel_addr = b.build_func_addr(kernel_name);
    Value* red_kind_val = b.build_iconst_i32(static_cast<int32_t>(pli.reduction_kind));

    Value* red_target = nullptr;
    if (pli.has_reduction) {
        Value* sz8 = b.build_iconst_i64(8);
        red_target = b.build_call("brass_parallel_alloc_context", Type::ptr(), {sz8});
        b.build_store(pli.reduction_type, red_target, 0, pli.reduction_init_val);
    } else {
        red_target = b.build_iconst_i64(0);
    }

    // Emit runtime call to brass_parallel_for
    b.build_call(
        "brass_parallel_for",
        Type::void_type(),
        {trip_count_val, grain_val, kernel_addr, ctx_ptr, red_kind_val, red_target}
    );

    if (!captured_invariants.empty()) {
        b.build_call("brass_parallel_free_context", Type::void_type(), {ctx_ptr});
    }

    Value* final_red_val = nullptr;
    if (pli.has_reduction) {
        final_red_val = b.build_load(pli.reduction_type, red_target, 0);
        b.build_call("brass_parallel_free_context", Type::void_type(), {red_target});
    }

    // 5. Connect preheader to exit block
    const Instruction* hdr_term = pli.header->terminator();
    const BranchTarget& exit_bt = pli.exit_on_false ? hdr_term->false_target() : hdr_term->true_target();

    std::vector<Value*> exit_args;
    exit_args.reserve(exit_bt.args.size());
    for (Value* arg : exit_bt.args) {
        if (pli.has_reduction && (arg == pli.reduction_param || arg == pli.reduction_op_inst->result())) {
            exit_args.push_back(final_red_val);
        } else if (arg == pli.iv_param) {
            exit_args.push_back(pli.limit_val);
        } else {
            exit_args.push_back(arg);
        }
    }

    pli.preheader->remove_instruction(pli.preheader->terminator());
    b.position_at_end(pli.preheader);
    b.build_br(pli.exit_bb, exit_args);

    if (pli.has_reduction && final_red_val) {
        // Replace uses of reduction param outside loop with final_red_val
        for (BasicBlock* bb : fn.blocks()) {
            if (pli.loop->contains(bb)) continue;
            for (Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
                for (size_t i = 0; i < inst->operand_count(); ++i) {
                    if (inst->operand(i) == pli.reduction_param) {
                        inst->set_operand(i, final_red_val);
                    }
                }
            }
        }
    }

    // 6. Delete old loop blocks
    for (BasicBlock* bb : pli.loop->blocks()) {
        if (bb != pli.preheader && bb != pli.exit_bb) {
            fn.remove_block(bb);
        }
    }

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
    LoopAnalysis loop_analysis(fn, dom);
    std::vector<LoopInfo*> loops = loop_analysis.post_order_loops();
    bool changed = false;

    for (LoopInfo* loop : loops) {
        if (!loop) continue;
        if (options.stats) options.stats->loops_analyzed++;

        ParallelLoopInfo pli;
        if (analyze_parallel_loop(fn, *loop, dom, pli, options)) {
            if (pli.is_parallelizable()) {
                if (transform_parallel_loop(fn, pli, options)) {
                    changed = true;
                    break; // CFG modified, return and re-analyze
                }
            }
        }
    }

    return changed;
}

} // namespace brass