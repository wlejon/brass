#include <brass/codegen/software_pipeline.hpp>
#include <brass/codegen/sched_dag.hpp>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace brass::codegen {

namespace {

struct LoopParts {
    LirInst* load_inst = nullptr;
    size_t load_idx = 0;
    std::vector<size_t> compute_indices;
    std::vector<size_t> store_indices;
    size_t iv_idx = 0;
    size_t cmp_idx = 0;
    size_t branch_idx = 0;
    VReg load_dst;
    VReg iv_reg;
    int64_t trip_count = -1; // -1 if dynamic, or constant
};

static bool analyze_loop_block(const LirBlock& block, LoopParts& parts, size_t max_body_instructions) {
    size_t n = block.instructions.size();
    if (n < 4 || n > max_body_instructions) {
        return false;
    }

    // Must end with a branch back to this block
    bool has_self_branch = false;
    size_t branch_idx = n - 1;
    for (size_t i = n; i > 0; --i) {
        const auto& inst = *block.instructions[i - 1];
        if (inst.opcode == LirOpcode::Jcc) {
            if (!inst.uses.empty() && inst.uses[0].is_label() && inst.uses[0].label_id == block.id) {
                has_self_branch = true;
                branch_idx = i - 1;
                break;
            }
        }
    }
    if (!has_self_branch) return false;
    parts.branch_idx = branch_idx;

    // Check for barrier instructions
    for (size_t i = 0; i <= branch_idx; ++i) {
        const auto& inst = *block.instructions[i];
        if (inst.is_call() || inst.opcode == LirOpcode::Safepoint || inst.opcode == LirOpcode::GuardExit) {
            return false;
        }
    }

    // Find load instruction(s) and induction variable update
    bool found_load = false;
    bool found_iv = false;

    for (size_t i = 0; i < branch_idx; ++i) {
        const auto& inst = *block.instructions[i];

        // Check if instruction is a load
        bool is_load = false;
        for (const auto& u : inst.uses) {
            if (u.is_mem() || u.is_spill_slot()) {
                is_load = true;
                break;
            }
        }

        if (is_load && !found_load && !inst.defs.empty() && inst.defs[0].is_vreg()) {
            parts.load_inst = block.instructions[i].get();
            parts.load_idx = i;
            parts.load_dst = inst.defs[0].vreg_val;
            found_load = true;
            continue;
        }

        // Check if instruction is an induction variable update (e.g. Add iv, step)
        if ((inst.opcode == LirOpcode::Add || inst.opcode == LirOpcode::Add32 ||
             inst.opcode == LirOpcode::Sub || inst.opcode == LirOpcode::Sub32) &&
            !inst.defs.empty() && inst.defs[0].is_vreg() &&
            !inst.uses.empty() && inst.uses[0].is_vreg() &&
            inst.defs[0].vreg_val.id == inst.uses[0].vreg_val.id) {
            parts.iv_idx = i;
            parts.iv_reg = inst.defs[0].vreg_val;
            found_iv = true;
            continue;
        }

        // Check if instruction is a Cmp before the branch
        if (inst.opcode == LirOpcode::Cmp || inst.opcode == LirOpcode::Cmp32) {
            parts.cmp_idx = i;
            if (inst.uses.size() >= 2 && inst.uses[1].is_imm_int()) {
                parts.trip_count = inst.uses[1].imm_int;
            }
            continue;
        }

        // Check if instruction is a store
        bool is_store = false;
        for (const auto& d : inst.defs) {
            if (d.is_mem() || d.is_spill_slot()) {
                is_store = true;
                break;
            }
        }
        if (is_store) {
            parts.store_indices.push_back(i);
            continue;
        }

        // Otherwise consider it a compute instruction
        parts.compute_indices.push_back(i);
    }

    if (!found_load || !found_iv || parts.compute_indices.empty()) {
        return false;
    }

    // Verify at least one compute instruction uses the loaded value
    bool compute_uses_load = false;
    for (size_t c_idx : parts.compute_indices) {
        const auto& c_inst = *block.instructions[c_idx];
        for (const auto& u : c_inst.uses) {
            if (u.is_vreg() && u.vreg_val.id == parts.load_dst.id) {
                compute_uses_load = true;
                break;
            }
        }
    }

    return compute_uses_load;
}

static std::unique_ptr<LirInst> clone_instruction(
    const LirInst& src,
    const std::unordered_map<uint32_t, VReg>& vreg_subst
) {
    auto dst = std::make_unique<LirInst>(src.opcode);
    dst->condition = src.condition;
    dst->clobbered_gprs = src.clobbered_gprs;
    dst->clobbered_xmms = src.clobbered_xmms;
    dst->mir_origin = src.mir_origin;
    dst->safepoint_id = src.safepoint_id;
    dst->resume_id = src.resume_id;
    dst->deopt_reason = src.deopt_reason;
    dst->exit_symbol = src.exit_symbol;
    dst->is_patchable = src.is_patchable;
    dst->patch_symbol = src.patch_symbol;
    dst->callee_symbol = src.callee_symbol;
    dst->def_constraints = src.def_constraints;
    dst->use_constraints = src.use_constraints;

    auto map_vreg = [&](VReg v) -> VReg {
        auto it = vreg_subst.find(v.id);
        return (it != vreg_subst.end()) ? it->second : v;
    };

    auto map_op = [&](const LirOperand& op) -> LirOperand {
        LirOperand res = op;
        if (res.is_vreg()) {
            res.vreg_val = map_vreg(res.vreg_val);
        } else if (res.is_mem()) {
            if (res.mem_val.base_vreg.is_valid()) {
                res.mem_val.base_vreg = map_vreg(res.mem_val.base_vreg);
            }
            if (res.mem_val.index_vreg.is_valid()) {
                res.mem_val.index_vreg = map_vreg(res.mem_val.index_vreg);
            }
        }
        return res;
    };

    for (const auto& d : src.defs) dst->defs.push_back(map_op(d));
    for (const auto& u : src.uses) dst->uses.push_back(map_op(u));

    return dst;
}

} // namespace

bool is_pipelinable_loop(const LirFunction& fn, const LirBlock& block, const PipelineOptions& opts) {
    if (!opts.enable_software_pipelining) return false;

    // Must have a self-backedge
    bool has_self_edge = false;
    for (const auto* succ : block.successors) {
        if (succ == &block) {
            has_self_edge = true;
            break;
        }
    }
    if (!has_self_edge) return false;

    LoopParts parts;
    if (!analyze_loop_block(block, parts, opts.max_body_instructions)) {
        return false;
    }

    if (parts.trip_count >= 0 && static_cast<size_t>(parts.trip_count) < opts.min_trip_count) {
        return false;
    }

    return true;
}

bool is_pipelinable_loop(const LirFunction& fn, const LirBlock& block) {
    PipelineOptions default_opts;
    return is_pipelinable_loop(fn, block, default_opts);
}

bool pipeline_loop(LirFunction& fn, LirBlock* loop_body, const PipelineOptions& opts) {
    if (!loop_body || !is_pipelinable_loop(fn, *loop_body, opts)) {
        return false;
    }

    LoopParts parts;
    if (!analyze_loop_block(*loop_body, parts, opts.max_body_instructions)) {
        return false;
    }

    // Identify preheader and exit block
    LirBlock* preheader = nullptr;
    for (auto* pred : loop_body->predecessors) {
        if (pred != loop_body) {
            preheader = pred;
            break;
        }
    }
    if (!preheader) return false;

    LirBlock* exit_bb = nullptr;
    for (auto* succ : loop_body->successors) {
        if (succ != loop_body) {
            exit_bb = succ;
            break;
        }
    }
    if (!exit_bb) return false;

    // Allocate new virtual registers for stage decoupling
    VReg prime_reg = fn.allocate_vreg(parts.load_dst.reg_class, parts.load_dst.size);
    VReg next_reg = fn.allocate_vreg(parts.load_dst.reg_class, parts.load_dst.size);

    // Create Prologue block
    LirBlock* prologue = fn.create_block(loop_body->name + "_prologue");

    // Create Epilogue block
    LirBlock* epilogue = fn.create_block(loop_body->name + "_epilogue");

    // --- 1. Populate Prologue ---
    // Stage 0 load for iteration 0: load into prime_reg
    std::unordered_map<uint32_t, VReg> prologue_subst;
    prologue_subst[parts.load_dst.id] = prime_reg;
    prologue->append_inst(clone_instruction(*parts.load_inst, prologue_subst));

    // Jump from prologue to kernel (loop_body)
    auto jmp_to_kernel = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp_to_kernel->add_use(LirOperand::label(loop_body->id));
    prologue->append_inst(std::move(jmp_to_kernel));

    // --- 2. Populate Epilogue ---
    // Final drain: compute for final iteration N-1 using last loaded prime_reg
    std::unordered_map<uint32_t, VReg> epilogue_subst;
    epilogue_subst[parts.load_dst.id] = prime_reg;

    for (size_t c_idx : parts.compute_indices) {
        epilogue->append_inst(clone_instruction(*loop_body->instructions[c_idx], epilogue_subst));
    }
    for (size_t s_idx : parts.store_indices) {
        epilogue->append_inst(clone_instruction(*loop_body->instructions[s_idx], epilogue_subst));
    }
    // Jump from epilogue to original exit_bb
    auto jmp_to_exit = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp_to_exit->add_use(LirOperand::label(exit_bb->id));
    epilogue->append_inst(std::move(jmp_to_exit));

    // --- 3. Rebuild Kernel (loop_body) ---
    // The kernel interleaves:
    //  - Compute(i) using prime_reg
    //  - Store(i) (if any)
    //  - Load(i+1) into next_reg
    //  - Induction variable update
    //  - prime_reg = next_reg (move/copy)
    //  - Cmp & Jcc
    std::vector<std::unique_ptr<LirInst>> kernel_insts;

    // Compute(i) using prime_reg
    std::unordered_map<uint32_t, VReg> kernel_compute_subst;
    kernel_compute_subst[parts.load_dst.id] = prime_reg;
    for (size_t c_idx : parts.compute_indices) {
        kernel_insts.push_back(clone_instruction(*loop_body->instructions[c_idx], kernel_compute_subst));
    }

    // Store(i)
    for (size_t s_idx : parts.store_indices) {
        kernel_insts.push_back(clone_instruction(*loop_body->instructions[s_idx], kernel_compute_subst));
    }

    // Load(i+1) into next_reg
    std::unordered_map<uint32_t, VReg> kernel_load_subst;
    kernel_load_subst[parts.load_dst.id] = next_reg;
    kernel_insts.push_back(clone_instruction(*parts.load_inst, kernel_load_subst));

    // Induction variable update
    if (parts.iv_idx < loop_body->instructions.size()) {
        std::unordered_map<uint32_t, VReg> iv_subst;
        kernel_insts.push_back(clone_instruction(*loop_body->instructions[parts.iv_idx], iv_subst));
    }

    // prime_reg = next_reg (transfer next loaded data to prime register)
    LirOpcode mov_op = parts.load_dst.is_xmm() ? LirOpcode::Movaps : LirOpcode::Mov;
    auto copy_inst = std::make_unique<LirInst>(mov_op);
    copy_inst->add_def(LirOperand::vreg(prime_reg, prime_reg.size));
    copy_inst->add_use(LirOperand::vreg(next_reg, next_reg.size));
    kernel_insts.push_back(std::move(copy_inst));

    // Cmp & Jcc (test if we continue loop or exit to epilogue)
    if (parts.cmp_idx < loop_body->instructions.size()) {
        std::unordered_map<uint32_t, VReg> cmp_subst;
        kernel_insts.push_back(clone_instruction(*loop_body->instructions[parts.cmp_idx], cmp_subst));
    }

    // Branch: Jcc targeting loop_body
    std::unordered_map<uint32_t, VReg> branch_subst;
    kernel_insts.push_back(clone_instruction(*loop_body->instructions[parts.branch_idx], branch_subst));

    // Fallthrough or Jmp to epilogue
    auto jmp_to_epilogue = std::make_unique<LirInst>(LirOpcode::Jmp);
    jmp_to_epilogue->add_use(LirOperand::label(epilogue->id));
    kernel_insts.push_back(std::move(jmp_to_epilogue));

    loop_body->instructions = std::move(kernel_insts);

    // --- 4. Rewire CFG ---
    // Update preheader to branch to prologue instead of loop_body
    for (auto& inst : preheader->instructions) {
        if (inst->is_branch() || inst->is_terminator()) {
            for (auto& u : inst->uses) {
                if (u.is_label() && u.label_id == loop_body->id) {
                    u.label_id = prologue->id;
                }
            }
        }
    }
    for (auto& s : preheader->successors) {
        if (s == loop_body) s = prologue;
    }

    // Prologue CFG
    prologue->predecessors.push_back(preheader);
    prologue->successors.push_back(loop_body);

    // Kernel CFG: preheader is replaced by prologue
    for (auto& p : loop_body->predecessors) {
        if (p == preheader) p = prologue;
    }
    // Successors: loop_body (backedge) and epilogue
    for (auto& s : loop_body->successors) {
        if (s == exit_bb) s = epilogue;
    }

    // Epilogue CFG
    epilogue->predecessors.push_back(loop_body);
    epilogue->successors.push_back(exit_bb);

    // Exit CFG: loop_body predecessor is replaced by epilogue
    for (auto& p : exit_bb->predecessors) {
        if (p == loop_body) p = epilogue;
    }

    return true;
}

bool pipeline_loop(LirFunction& fn, LirBlock* loop_body) {
    PipelineOptions default_opts;
    return pipeline_loop(fn, loop_body, default_opts);
}

PipelineStats run_software_pipelining(LirFunction& fn, const PipelineOptions& opts) {
    PipelineStats stats;
    if (!opts.enable_software_pipelining) return stats;

    // Snapshot blocks to avoid iterating over newly added prologue/epilogue blocks
    std::vector<LirBlock*> candidate_blocks;
    for (const auto& block : fn.blocks) {
        if (block) candidate_blocks.push_back(block.get());
    }

    for (auto* block : candidate_blocks) {
        stats.loops_analyzed++;
        if (is_pipelinable_loop(fn, *block, opts)) {
            if (pipeline_loop(fn, block, opts)) {
                stats.loops_pipelined++;
                stats.prologues_created++;
                stats.epilogues_created++;
            }
        }
    }

    return stats;
}

PipelineStats run_software_pipelining(LirFunction& fn) {
    PipelineOptions default_opts;
    return run_software_pipelining(fn, default_opts);
}

} // namespace brass::codegen
