#include "loop_dependence.hpp"
#include <brass/mir/opcodes.hpp>
#include <algorithm>

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

bool find_iv_level(const Value* val, const LoopNest& nest, size_t& out_lvl) {
    if (!val) return false;
    for (size_t l = 0; l < nest.depth(); ++l) {
        if (nest.level(l).iv_param == val) {
            out_lvl = l;
            return true;
        }
    }
    return false;
}

bool find_derived_iv_stride(const Value* val, const LoopNest& nest, size_t& level_idx, Value*& stride_val, int64_t& const_stride) {
    if (!val || !val->is_block_param()) return false;
    BasicBlock* parent_bb = val->defining_block();
    if (!parent_bb) return false;

    for (size_t l = 0; l < nest.depth(); ++l) {
        const auto& lvl = nest.level(l);
        if (lvl.header == parent_bb && lvl.latch) {
            Instruction* term = lvl.latch->terminator();
            if (term && term->opcode() == Opcode::br && term->branch_target().block == lvl.header) {
                uint32_t p_idx = val->param_index();
                if (p_idx < term->branch_target().args.size()) {
                    Value* latch_arg = term->branch_target().args[p_idx];
                    if (latch_arg && latch_arg->is_instruction()) {
                        Instruction* step_inst = latch_arg->defining_instruction();
                        if (step_inst && step_inst->opcode() == Opcode::add) {
                            Value* op0 = step_inst->operand(0);
                            Value* op1 = step_inst->operand(1);
                            Value* step = (op0 == val) ? op1 : ((op1 == val) ? op0 : nullptr);
                            if (step && lvl.loop && lvl.loop->is_loop_invariant(step)) {
                                level_idx = l;
                                if (get_const_int(step, const_stride)) {
                                    stride_val = nullptr;
                                } else {
                                    const_stride = 0;
                                    stride_val = step;
                                }
                                return true;
                            }
                        }
                    }
                }
            }
        }
    }
    return false;
}

void parse_index_expr(const Value* val, const LoopNest& nest, std::vector<AccessIndexTerm>& terms, int64_t& const_offset) {
    if (!val) return;

    size_t iv_lvl = 0;
    if (find_iv_level(val, nest, iv_lvl)) {
        AccessIndexTerm term;
        term.level_index = iv_lvl;
        term.const_stride = 1;
        term.symbolic_stride = nullptr;
        terms.push_back(term);
        return;
    }

    size_t derived_lvl = 0;
    Value* derived_stride_val = nullptr;
    int64_t derived_const_stride = 0;
    if (find_derived_iv_stride(val, nest, derived_lvl, derived_stride_val, derived_const_stride)) {
        AccessIndexTerm term;
        term.level_index = derived_lvl;
        term.const_stride = derived_const_stride;
        term.symbolic_stride = derived_stride_val;
        terms.push_back(term);
        return;
    }

    int64_t c = 0;
    if (get_const_int(val, c)) {
        const_offset += c;
        return;
    }

    if (val->is_instruction()) {
        const Instruction* inst = val->defining_instruction();
        if (!inst) return;

        if (inst->opcode() == Opcode::add) {
            parse_index_expr(inst->operand(0), nest, terms, const_offset);
            parse_index_expr(inst->operand(1), nest, terms, const_offset);
            return;
        }

        if (inst->opcode() == Opcode::sub) {
            parse_index_expr(inst->operand(0), nest, terms, const_offset);
            int64_t sub_c = 0;
            if (get_const_int(inst->operand(1), sub_c)) {
                const_offset -= sub_c;
            }
            return;
        }

        if (inst->opcode() == Opcode::mul) {
            const Value* op0 = inst->operand(0);
            const Value* op1 = inst->operand(1);
            size_t mul_lvl = 0;
            const Value* mul_other = nullptr;
            if (find_iv_level(op0, nest, mul_lvl)) {
                mul_other = op1;
            } else if (find_iv_level(op1, nest, mul_lvl)) {
                mul_other = op0;
            }

            if (mul_other) {
                int64_t factor = 0;
                AccessIndexTerm term;
                term.level_index = mul_lvl;
                if (get_const_int(mul_other, factor)) {
                    term.const_stride = factor;
                    term.symbolic_stride = nullptr;
                } else {
                    term.const_stride = 0;
                    term.symbolic_stride = const_cast<Value*>(mul_other);
                }
                terms.push_back(term);
                return;
            }
        }

        if (inst->opcode() == Opcode::shl) {
            const Value* op0 = inst->operand(0);
            const Value* op1 = inst->operand(1);
            int64_t shift = 0;
            size_t shl_lvl = 0;
            if (get_const_int(op1, shift) && find_iv_level(op0, nest, shl_lvl)) {
                AccessIndexTerm term;
                term.level_index = shl_lvl;
                term.const_stride = 1LL << shift;
                term.symbolic_stride = nullptr;
                terms.push_back(term);
                return;
            }
        }
    }
}

} // namespace

void analyze_nest_memory_accesses(Function& fn, LoopNest& nest) {
    (void)fn;
    nest.memory_accesses().clear();

    for (size_t l = 0; l < nest.depth(); ++l) {
        const auto& lvl = nest.level(l);
        if (!lvl.loop) continue;

        for (BasicBlock* bb : lvl.loop->blocks()) {
            if (!bb) continue;
            for (Instruction* inst = bb->head(); inst != nullptr; inst = inst->next()) {
                if (inst->opcode() == Opcode::load_indexed || inst->opcode() == Opcode::store_indexed) {
                    NestMemoryAccess acc;
                    acc.inst = inst;
                    acc.is_store = (inst->opcode() == Opcode::store_indexed);
                    acc.elem_type = inst->memory_type();
                    acc.base = inst->operand(0);
                    acc.scale = inst->scale();

                    int64_t extra_offset = 0;
                    parse_index_expr(inst->operand(1), nest, acc.terms, extra_offset);
                    acc.const_offset = inst->offset() + static_cast<int32_t>(extra_offset);

                    nest.memory_accesses().push_back(acc);
                }
            }
        }
    }
}

void compute_nest_dependences(Function& fn, LoopNest& nest) {
    (void)fn;
    nest.dependences().clear();
    const auto& accesses = nest.memory_accesses();

    for (size_t i = 0; i < accesses.size(); ++i) {
        for (size_t j = i + 1; j < accesses.size(); ++j) {
            const auto& a1 = accesses[i];
            const auto& a2 = accesses[j];

            // Only analyze read-write, write-read, write-write
            if (!a1.is_store && !a2.is_store) continue;

            // If base pointers are distinct and invariant/function args, assume no alias
            if (a1.base != a2.base) continue;

            DependenceVector dep;
            dep.directions.assign(nest.depth(), DependenceDirection::Equal);
            dep.distances.assign(nest.depth(), 0);
            dep.has_distance = true;
            dep.is_loop_independent = true;

            for (size_t l = 0; l < nest.depth(); ++l) {
                const AccessIndexTerm* t1 = nullptr;
                const AccessIndexTerm* t2 = nullptr;
                for (const auto& term : a1.terms) {
                    if (term.level_index == l) { t1 = &term; break; }
                }
                for (const auto& term : a2.terms) {
                    if (term.level_index == l) { t2 = &term; break; }
                }

                if (t1 && t2) {
                    bool match_stride = (t1->const_stride == t2->const_stride) &&
                                        (t1->symbolic_stride == t2->symbolic_stride);
                    if (match_stride) {
                        int32_t diff = a2.const_offset - a1.const_offset;
                        int32_t elem_sz = static_cast<int32_t>(a1.scale > 0 ? a1.scale : 1);
                        int32_t dist = diff / elem_sz;
                        dep.distances[l] = dist;
                        if (dist > 0) {
                            dep.directions[l] = DependenceDirection::Forward;
                            dep.is_loop_independent = false;
                        } else if (dist < 0) {
                            dep.directions[l] = DependenceDirection::Backward;
                            dep.is_loop_independent = false;
                        } else {
                            dep.directions[l] = DependenceDirection::Equal;
                        }
                    } else {
                        dep.directions[l] = DependenceDirection::Any;
                        dep.has_distance = false;
                        dep.is_loop_independent = false;
                    }
                } else if (!t1 && !t2) {
                    dep.directions[l] = DependenceDirection::Equal;
                } else {
                    // Loop-invariant with respect to this dimension
                    dep.directions[l] = DependenceDirection::Equal;
                }
            }

            nest.dependences().push_back(dep);
        }
    }
}

bool check_nest_tiling_legality(const LoopNest& nest) {
    for (const auto& dep : nest.dependences()) {
        if (dep.is_loop_independent) continue;

        for (size_t l = 0; l < dep.directions.size(); ++l) {
            DependenceDirection dir = dep.directions[l];
            if (dir == DependenceDirection::Equal) continue;
            if (dir == DependenceDirection::Backward) {
                // Negative leading dependence distance violates causality
                return false;
            }
            if (dir == DependenceDirection::Any) {
                return false;
            }
            if (dir == DependenceDirection::Forward) {
                // Leading forward dependence satisfies causality
                break;
            }
        }
    }
    return true;
}

bool check_nest_interchange_legality(const LoopNest& nest, size_t level_a, size_t level_b) {
    if (level_a == level_b) return true;
    if (level_a >= nest.depth() || level_b >= nest.depth()) return false;

    for (const auto& dep : nest.dependences()) {
        if (dep.is_loop_independent) continue;

        auto dirs = dep.directions;
        std::swap(dirs[level_a], dirs[level_b]);

        for (DependenceDirection dir : dirs) {
            if (dir == DependenceDirection::Equal) continue;
            if (dir == DependenceDirection::Backward || dir == DependenceDirection::Any) {
                return false;
            }
            if (dir == DependenceDirection::Forward) {
                break;
            }
        }
    }
    return true;
}

bool detect_matrix_multiply_pattern(const LoopNest& nest) {
    if (nest.depth() != 3) return false;
    if (!nest.has_reduction()) return false;

    bool has_a = false;
    bool has_b = false;
    bool has_c = false;

    for (const auto& acc : nest.memory_accesses()) {
        bool has_i = false;
        bool has_j = false;
        bool has_k = false;

        for (const auto& t : acc.terms) {
            if (t.level_index == 0) has_i = true;
            if (t.level_index == 1) has_j = true;
            if (t.level_index == 2) has_k = true;
        }

        if (!acc.is_store && has_i && has_k && !has_j) has_a = true;
        if (!acc.is_store && has_k && has_j && !has_i) has_b = true;
        if (acc.is_store && has_i && has_j && !has_k) has_c = true;
    }

    return has_a && has_b && has_c;
}

} // namespace brass
