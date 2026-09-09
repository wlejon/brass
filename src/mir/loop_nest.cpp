#include <brass/mir/loop_nest.hpp>
#include "loop_dependence.hpp"
#include <brass/mir/opcodes.hpp>

namespace brass {

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

bool extract_loop_level(Function& fn, LoopInfo& loop, LoopNestLevel& level) {
    BasicBlock* header = loop.header();
    if (!header || loop.latches().size() != 1) return false;

    // Do not tile already tiled loops
    if (header->name().find("_2dt") != std::string_view::npos ||
        header->name().find("_3dt") != std::string_view::npos ||
        header->name().find("_t_") != std::string_view::npos ||
        header->name().find("_p_") != std::string_view::npos ||
        header->name().find("_tile") != std::string_view::npos ||
        header->name().find("_point") != std::string_view::npos ||
        header->name().find("_vec_hdr") != std::string_view::npos) {
        return false;
    }

    BasicBlock* latch = loop.latches()[0];
    if (!latch) return false;

    BasicBlock* preheader = loop.preheader();
    if (!preheader) preheader = LoopAnalysis::ensure_preheader(fn, loop);
    if (!preheader) return false;

    Instruction* ph_term = preheader->terminator();
    Instruction* latch_term = latch->terminator();
    Instruction* hdr_term = header->terminator();
    if (!ph_term || !latch_term || !hdr_term) return false;

    // Check preheader branch target
    BranchTarget* ph_bt = nullptr;
    if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == header) {
        ph_bt = &ph_term->branch_target();
    } else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == header) ph_bt = &ph_term->true_target();
        else if (ph_term->false_target().block == header) ph_bt = &ph_term->false_target();
    }
    if (!ph_bt || ph_bt->args.size() != header->param_count()) return false;

    // Check latch branch target
    BranchTarget* latch_bt = nullptr;
    if (latch_term->opcode() == Opcode::br && latch_term->branch_target().block == header) {
        latch_bt = &latch_term->branch_target();
    }
    if (!latch_bt || latch_bt->args.size() != header->param_count()) return false;

    // Header terminator must be br_if
    if (hdr_term->opcode() != Opcode::br_if) return false;

    Value* cond_val = hdr_term->operand(0);
    if (!cond_val || !cond_val->is_instruction()) return false;
    Instruction* cmp_inst = cond_val->defining_instruction();
    if (!cmp_inst || !is_comparison(cmp_inst->opcode()) || cmp_inst->parent() != header) return false;

    BasicBlock* body_bb = nullptr;
    BasicBlock* exit_bb = nullptr;
    bool exit_on_false = true;

    if (loop.contains(hdr_term->true_target().block) && !loop.contains(hdr_term->false_target().block)) {
        body_bb = hdr_term->true_target().block;
        exit_bb = hdr_term->false_target().block;
        exit_on_false = true;
    } else if (!loop.contains(hdr_term->true_target().block) && loop.contains(hdr_term->false_target().block)) {
        body_bb = hdr_term->false_target().block;
        exit_bb = hdr_term->true_target().block;
        exit_on_false = false;
    } else {
        return false;
    }

    if (!body_bb || !exit_bb) return false;

    Opcode cmp_op = cmp_inst->opcode();
    if (cmp_op != Opcode::slt && cmp_op != Opcode::ult &&
        cmp_op != Opcode::sle && cmp_op != Opcode::ule) {
        return false;
    }

    Value* cmp_lhs = cmp_inst->operand(0);
    Value* cmp_rhs = cmp_inst->operand(1);

    bool found_iv = false;
    size_t iv_idx = 0;
    int64_t step_val = 1;
    Value* limit = nullptr;

    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* param = header->param(i);
        if (param == cmp_lhs && loop.is_loop_invariant(cmp_rhs)) {
            Value* latch_next = latch_bt->args[i];
            if (latch_next && latch_next->is_instruction()) {
                Instruction* def = latch_next->defining_instruction();
                if (def && def->opcode() == Opcode::add) {
                    Value* step_op = (def->operand(0) == param) ? def->operand(1) : ((def->operand(1) == param) ? def->operand(0) : nullptr);
                    int64_t c = 0;
                    if (step_op && get_const_int(step_op, c) && c > 0) {
                        found_iv = true;
                        iv_idx = i;
                        step_val = c;
                        limit = cmp_rhs;
                        break;
                    }
                }
            }
        }
    }

    if (!found_iv || !limit) return false;

    level.loop = &loop;
    level.header = header;
    level.preheader = preheader;
    level.body = body_bb;
    level.latch = latch;
    level.exit_bb = exit_bb;
    level.blocks = loop.blocks();
    level.iv_param_index = iv_idx;
    level.iv_param = header->param(iv_idx);
    level.iv_type = level.iv_param->type();
    level.init_val = ph_bt->args[iv_idx];
    level.limit_val = limit;
    level.step = step_val;
    level.cmp_opcode = cmp_op;
    level.exit_on_false = exit_on_false;
    return true;
}

} // namespace

bool LoopNest::is_tileable() const noexcept {
    return check_nest_tiling_legality(*this);
}

bool LoopNest::is_interchange_legal(size_t level_a, size_t level_b) const noexcept {
    return check_nest_interchange_legality(*this, level_a, level_b);
}

bool LoopNest::is_matrix_multiply() const noexcept {
    return detect_matrix_multiply_pattern(*this);
}

LoopNestAnalysis::LoopNestAnalysis(Function& fn, const DominatorTree& dom) {
    discover_nests(fn, dom);
}

std::unique_ptr<LoopNest> LoopNestAnalysis::analyze_nest(Function& fn, LoopInfo& outer_loop, const DominatorTree& dom) {
    (void)dom;
    auto nest = std::make_unique<LoopNest>();

    LoopInfo* cur = &outer_loop;
    while (cur != nullptr) {
        LoopNestLevel lvl;
        if (!extract_loop_level(fn, *cur, lvl)) {
            break;
        }
        nest->levels().push_back(lvl);

        if (cur->sub_loops().size() == 1) {
            cur = cur->sub_loops()[0].get();
        } else {
            break;
        }

        if (nest->depth() >= 3) {
            break;
        }
    }

    if (nest->depth() < 2) {
        return nullptr;
    }

    analyze_nest_memory_accesses(fn, *nest);
    compute_nest_dependences(fn, *nest);

    // Look for reduction in innermost loop (e.g. 3D matmul accumulator)
    if (nest->depth() == 3) {
        const auto& inner_lvl = nest->level(2);
        Instruction* latch_term = inner_lvl.latch->terminator();
        if (latch_term && latch_term->opcode() == Opcode::br) {
            for (size_t p = 0; p < inner_lvl.header->param_count(); ++p) {
                if (p == inner_lvl.iv_param_index) continue;
                Value* param = inner_lvl.header->param(p);
                Value* next_val = latch_term->branch_target().args[p];
                if (next_val && next_val->is_instruction()) {
                    Instruction* def = next_val->defining_instruction();
                    if (def && def->opcode() == Opcode::add) {
                        Value* op0 = def->operand(0);
                        Value* op1 = def->operand(1);
                        if (op0 == param || op1 == param) {
                            // Found reduction parameter
                            // Check exit of inner loop
                            Instruction* hdr_term = inner_lvl.header->terminator();
                            const BranchTarget& exit_bt = inner_lvl.exit_on_false ? hdr_term->false_target() : hdr_term->true_target();
                            BasicBlock* exit_bb = exit_bt.block;

                            // Check if exit_bb stores the reduction result
                            Instruction* store_inst = nullptr;
                            if (exit_bb) {
                                for (Instruction* inst = exit_bb->head(); inst != nullptr; inst = inst->next()) {
                                    if (inst->opcode() == Opcode::store_indexed || inst->opcode() == Opcode::store) {
                                        store_inst = inst;
                                        break;
                                    }
                                }
                            }

                            Instruction* ph_term = inner_lvl.loop->preheader() ? inner_lvl.loop->preheader()->terminator() : nullptr;
                            Value* init_val = nullptr;
                            if (ph_term && ph_term->opcode() == Opcode::br && p < ph_term->branch_target().args.size()) {
                                init_val = ph_term->branch_target().args[p];
                            }

                            nest->set_reduction(init_val, def, store_inst, exit_bb);
                            break;
                        }
                    }
                }
            }
        }
    }

    return nest;
}

void LoopNestAnalysis::discover_nests(Function& fn, const DominatorTree& dom) {
    loop_analysis_ = std::make_unique<LoopAnalysis>(fn, dom);

    for (const auto& top_loop : loop_analysis_->top_level_loops()) {
        if (!top_loop) continue;
        auto nest = analyze_nest(fn, *top_loop, dom);
        if (nest) {
            nests_.push_back(std::move(nest));
        }
    }
}

} // namespace brass
