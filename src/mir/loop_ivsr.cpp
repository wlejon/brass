#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/builder.hpp>
#include <unordered_map>
#include <vector>

namespace brass {

void replace_all_uses(Function& fn, Value* old_val, Value* new_val);

namespace {

struct BasicIV {
    Value* param = nullptr;
    size_t param_idx = 0;
    Value* init = nullptr;
    Value* step = nullptr;
    Instruction* step_inst = nullptr;
    bool is_sub = false;
};

struct PairHash {
    size_t operator()(const std::pair<const Value*, uint8_t>& p) const noexcept {
        return std::hash<const void*>()(p.first) ^ (static_cast<size_t>(p.second) << 3);
    }
};

static bool get_const_int(const Value* val, int64_t& out_val) {
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

static Value* build_smart_const_i64(Builder& b, int64_t val) {
    return b.build_iconst_i64(val);
}

static Value* build_smart_mul(Builder& b, Value* lhs, Value* rhs) {
    int64_t c0, c1;
    bool has_c0 = get_const_int(lhs, c0);
    bool has_c1 = get_const_int(rhs, c1);
    if (has_c0 && has_c1) return build_smart_const_i64(b, c0 * c1);
    if (has_c0) {
        if (c0 == 0) return build_smart_const_i64(b, 0);
        if (c0 == 1) return rhs;
    }
    if (has_c1) {
        if (c1 == 0) return build_smart_const_i64(b, 0);
        if (c1 == 1) return lhs;
    }
    return b.build_mul(lhs, rhs);
}

static Value* build_smart_add(Builder& b, Value* lhs, Value* rhs) {
    int64_t c0, c1;
    bool has_c0 = get_const_int(lhs, c0);
    bool has_c1 = get_const_int(rhs, c1);
    if (has_c0 && has_c1) return build_smart_const_i64(b, c0 + c1);
    if (has_c0 && c0 == 0) return rhs;
    if (has_c1 && c1 == 0) return lhs;
    return b.build_add(lhs, rhs);
}

} // namespace

bool ivsr_pass(Function& fn, LoopInfo& loop, DominatorTree& dom) {
    (void)dom;
    BasicBlock* header = loop.header();
    BasicBlock* preheader = loop.preheader();
    if (!header || !preheader || loop.latches().size() != 1) return false;

    BasicBlock* latch = loop.latches()[0];
    if (!latch) return false;

    Instruction* ph_term = preheader->terminator();
    Instruction* latch_term = latch->terminator();
    if (!ph_term || !latch_term) return false;

    BranchTarget* ph_bt = nullptr;
    if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == header) ph_bt = &ph_term->branch_target();
    else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == header) ph_bt = &ph_term->true_target();
        else if (ph_term->false_target().block == header) ph_bt = &ph_term->false_target();
    }
    if (!ph_bt || ph_bt->args.size() != header->param_count()) return false;

    BranchTarget* latch_bt = nullptr;
    if (latch_term->opcode() == Opcode::br && latch_term->branch_target().block == header) latch_bt = &latch_term->branch_target();
    else if (latch_term->opcode() == Opcode::br_if) {
        if (latch_term->true_target().block == header) latch_bt = &latch_term->true_target();
        else if (latch_term->false_target().block == header) latch_bt = &latch_term->false_target();
    }
    if (!latch_bt || latch_bt->args.size() != header->param_count()) return false;

    std::vector<BasicIV> bivs;
    std::unordered_map<const Value*, size_t> param_to_biv;

    for (size_t i = 0; i < header->param_count(); ++i) {
        Value* param = header->param(i);
        if (!param || !param->type().is_integer()) continue;
        Value* latch_arg = latch_bt->args[i];
        if (!latch_arg || !latch_arg->is_instruction()) continue;
        Instruction* def = latch_arg->defining_instruction();
        if (!def) continue;

        if (def->opcode() == Opcode::add) {
            Value* op0 = def->operand(0);
            Value* op1 = def->operand(1);
            if (op0 == param && loop.is_loop_invariant(op1)) {
                param_to_biv[param] = bivs.size();
                bivs.push_back({param, i, ph_bt->args[i], op1, def, false});
            } else if (op1 == param && loop.is_loop_invariant(op0)) {
                param_to_biv[param] = bivs.size();
                bivs.push_back({param, i, ph_bt->args[i], op0, def, false});
            }
        } else if (def->opcode() == Opcode::sub && def->operand(0) == param && loop.is_loop_invariant(def->operand(1))) {
            param_to_biv[param] = bivs.size();
            bivs.push_back({param, i, ph_bt->args[i], def->operand(1), def, true});
        }
    }

    if (bivs.empty()) return false;

    Builder b_ph(*fn.parent());
    b_ph.set_function(&fn);
    b_ph.position_before(ph_term);

    Builder b_latch(*fn.parent());
    b_latch.set_function(&fn);
    b_latch.position_before(latch_term);

    bool changed = false;
    std::unordered_map<std::pair<const Value*, uint8_t>, Value*, PairHash> biv_scaled_map;

    auto get_or_create_scaled_biv = [&](const BasicIV& biv, uint8_t scale) -> Value* {
        if (scale == 1) return biv.param;
        auto key = std::make_pair(biv.param, scale);
        auto it = biv_scaled_map.find(key);
        if (it != biv_scaled_map.end()) return it->second;

        Value* scale_val = build_smart_const_i64(b_ph, scale);
        Value* init_bytes = build_smart_mul(b_ph, biv.init, scale_val);
        Value* step_bytes = build_smart_mul(b_ph, biv.step, scale_val);

        Value* biv_bytes_p = fn.parent()->arena().make<Value>(fn.next_value_id(), Type::i64(), ValueKind::BlockParam);
        header->add_param(biv_bytes_p);
        ph_bt->args.push_back(init_bytes);

        Value* next_biv_bytes = b_latch.build_add(biv_bytes_p, step_bytes);
        latch_bt->args.push_back(next_biv_bytes);

        biv_scaled_map[key] = biv_bytes_p;
        return biv_bytes_p;
    };

    for (BasicBlock* bb : loop.blocks()) {
        if (!bb || bb == preheader) continue;
        for (Instruction* inst : *bb) {
            if (!inst) continue;
            if (inst->opcode() == Opcode::load_indexed || inst->opcode() == Opcode::store_indexed) {
                Value* base = inst->operand(0);
                Value* index = inst->operand(1);
                uint8_t scale = inst->scale();
                int32_t offset = inst->offset();

                if (loop.is_loop_invariant(base)) {
                    if (index && index->is_instruction()) {
                        Instruction* idx_def = index->defining_instruction();
                        if (idx_def && idx_def->opcode() == Opcode::add) {
                            Value* a = idx_def->operand(0);
                            Value* b = idx_def->operand(1);
                            const BasicIV* matched_biv = nullptr;
                            Value* inv_offset_val = nullptr;

                            if (param_to_biv.count(a) && loop.is_loop_invariant(b)) {
                                matched_biv = &bivs[param_to_biv[a]];
                                inv_offset_val = b;
                            } else if (param_to_biv.count(b) && loop.is_loop_invariant(a)) {
                                matched_biv = &bivs[param_to_biv[b]];
                                inv_offset_val = a;
                            }

                            if (matched_biv && inv_offset_val) {
                                Value* scale_val = build_smart_const_i64(b_ph, scale);
                                Value* row_bytes = build_smart_mul(b_ph, inv_offset_val, scale_val);
                                Value* new_base = build_smart_add(b_ph, base, row_bytes);
                                if (offset != 0) new_base = build_smart_add(b_ph, new_base, build_smart_const_i64(b_ph, offset));

                                Value* biv_bytes_p = get_or_create_scaled_biv(*matched_biv, scale);
                                inst->set_operand(0, new_base);
                                inst->set_operand(1, biv_bytes_p);
                                inst->set_scale(1);
                                inst->set_offset(0);
                                changed = true;
                            }
                        }
                    }

                    if (index && param_to_biv.count(index) && scale > 1) {
                        const BasicIV& matched_biv = bivs[param_to_biv[index]];
                        Value* new_base = base;
                        if (offset != 0) new_base = build_smart_add(b_ph, base, build_smart_const_i64(b_ph, offset));

                        Value* biv_bytes_p = get_or_create_scaled_biv(matched_biv, scale);
                        inst->set_operand(0, new_base);
                        inst->set_operand(1, biv_bytes_p);
                        inst->set_scale(1);
                        inst->set_offset(0);
                        changed = true;
                    }
                }
            }
        }
    }

    for (const auto& [key, scaled_p] : biv_scaled_map) {
        const Value* biv_param = key.first;
        uint8_t sc = key.second;
        int64_t sc_int = static_cast<int64_t>(sc);
        int shl_k = (sc == 8 ? 3 : (sc == 4 ? 2 : (sc == 2 ? 1 : 0)));

        for (BasicBlock* bb : loop.blocks()) {
            if (!bb || bb == preheader) continue;
            Instruction* cur = bb->head();
            while (cur) {
                Instruction* next = cur->next();
                if (cur->opcode() == Opcode::mul && cur->produces_value()) {
                    Value* op0 = cur->operand(0);
                    Value* op1 = cur->operand(1);
                    int64_t c0, c1;
                    bool has_c0 = get_const_int(op0, c0);
                    bool has_c1 = get_const_int(op1, c1);
                    if ((op0 == biv_param && has_c1 && c1 == sc_int) || (op1 == biv_param && has_c0 && c0 == sc_int)) {
                        replace_all_uses(fn, cur->result(), scaled_p);
                        bb->remove_instruction(cur);
                        changed = true;
                    }
                } else if (cur->opcode() == Opcode::shl && cur->produces_value() && shl_k > 0) {
                    Value* op0 = cur->operand(0);
                    Value* op1 = cur->operand(1);
                    int64_t c1;
                    if (op0 == biv_param && get_const_int(op1, c1) && c1 == shl_k) {
                        replace_all_uses(fn, cur->result(), scaled_p);
                        bb->remove_instruction(cur);
                        changed = true;
                    }
                }
                cur = next;
            }
        }
    }

    Instruction* hdr_term = header->terminator();
    if (hdr_term && hdr_term->opcode() == Opcode::br_if) {
        Value* cond_val = hdr_term->operand(0);
        if (cond_val && cond_val->is_instruction()) {
            Instruction* cond_inst = cond_val->defining_instruction();
            if (cond_inst && cond_inst->parent() == header && is_comparison(cond_inst->opcode())) {
                Value* lhs = cond_inst->operand(0);
                Value* rhs = cond_inst->operand(1);
                if (param_to_biv.count(lhs) && loop.is_loop_invariant(rhs)) {
                    const BasicIV& matched_biv = bivs[param_to_biv[lhs]];
                    for (uint8_t sc : {uint8_t(8), uint8_t(4), uint8_t(2)}) {
                        auto key = std::make_pair(matched_biv.param, sc);
                        if (biv_scaled_map.count(key)) {
                            Value* biv_bytes_p = biv_scaled_map[key];
                            Value* scale_val = build_smart_const_i64(b_ph, sc);
                            Value* limit_bytes = build_smart_mul(b_ph, rhs, scale_val);
                            cond_inst->set_operand(0, biv_bytes_p);
                            cond_inst->set_operand(1, limit_bytes);
                            changed = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    return changed;
}

} // namespace brass