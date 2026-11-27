#include <brass/mir/gpu_opt.hpp>

#include <brass/mir/block.hpp>
#include <brass/mir/builder.hpp>
#include <brass/mir/dominators.hpp>
#include <brass/mir/loop_analysis.hpp>
#include <brass/mir/module.hpp>
#include <brass/mir/opcodes.hpp>
#include <brass/mir/uses.hpp>
#include <brass/mir/verifier.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace brass {

namespace {

bool get_const_i64(const Value* v, int64_t& out) noexcept {
    if (!v || !v->is_instruction()) return false;
    const Instruction* inst = v->defining_instruction();
    if (!inst) return false;
    if (inst->opcode() == Opcode::iconst_i32) {
        out = inst->imm_i32();
        return true;
    }
    if (inst->opcode() == Opcode::iconst_i64) {
        out = inst->imm_i64();
        return true;
    }
    return false;
}

bool is_multiple_of_4_index(const Value* v) noexcept {
    if (!v) return false;
    int64_t c = 0;
    if (get_const_i64(v, c)) return (c & 3) == 0;
    if (!v->is_instruction()) return false;
    const Instruction* inst = v->defining_instruction();
    if (!inst || inst->operand_count() != 2) return false;
    if (inst->opcode() == Opcode::shl) {
        int64_t sh = 0;
        return get_const_i64(inst->operand(1), sh) && sh >= 2;
    }
    if (inst->opcode() == Opcode::mul) {
        int64_t m = 0;
        if (get_const_i64(inst->operand(1), m) && (m & 3) == 0) return true;
        if (get_const_i64(inst->operand(0), m) && (m & 3) == 0) return true;
    }
    if (inst->opcode() == Opcode::sext_i64 || inst->opcode() == Opcode::zext_i64) {
        return is_multiple_of_4_index(inst->operand(0));
    }
    if (inst->opcode() == Opcode::add) {
        return is_multiple_of_4_index(inst->operand(0)) && is_multiple_of_4_index(inst->operand(1));
    }
    return false;
}

void replace_uses_globally(Function& fn, const Value* old_v, Value* new_v) {
    if (!old_v || !new_v || old_v == new_v) return;
    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (Instruction* inst : *bb) {
            if (inst) replace_uses_in(*inst, old_v, new_v);
        }
    }
}

struct SmallCountedLoop {
    BasicBlock* header = nullptr;
    BasicBlock* body = nullptr;
    BasicBlock* exit = nullptr;
    uint32_t trip_count = 0;
    std::vector<Value*> body_entry_args;
    std::vector<Value*> latch_back_args;
    std::vector<Value*> exit_args;
};

std::optional<SmallCountedLoop> match_small_counted_loop(const Function& fn, const LoopInfo& loop, uint32_t max_trip) {
    if (!loop.sub_loops().empty() || loop.blocks().size() != 2 || loop.latches().size() != 1) {
        return std::nullopt;
    }
    BasicBlock* header = loop.header();
    BasicBlock* body = loop.latches()[0];
    if (!header || !body || header == body || !header->terminator() || !body->terminator()) {
        return std::nullopt;
    }
    if (body->param_count() > 0) return std::nullopt; // keep simple uniform body-paramless form first
    // Ensure header has no side effects before its conditional exit branch.
    for (const Instruction* inst : *header) {
        if (!inst) continue;
        if (inst == header->terminator()) break;
        if (inst->has_side_effects()) return std::nullopt;
    }
    const Instruction* hterm = header->terminator();
    const Instruction* bterm = body->terminator();
    if (hterm->opcode() != Opcode::br_if || hterm->operand_count() != 1) return std::nullopt;
    if (bterm->opcode() != Opcode::br || bterm->branch_target().block != header) return std::nullopt;

    const BranchTarget& t_tgt = hterm->true_target();
    const BranchTarget& f_tgt = hterm->false_target();
    if (t_tgt.block != body || loop.contains(f_tgt.block) || !f_tgt.block) return std::nullopt;

    const Instruction* cmp = hterm->operand(0) && hterm->operand(0)->is_instruction()
        ? hterm->operand(0)->defining_instruction() : nullptr;
    if (!cmp || cmp->operand_count() != 2) return std::nullopt;
    const Value* iv = cmp->operand(0);
    int64_t bound = 0;
    if (!iv || !iv->is_block_param() || iv->defining_block() != header || !get_const_i64(cmp->operand(1), bound)) {
        return std::nullopt;
    }
    const size_t p_idx = iv->param_index();

    // Require single outside predecessor or identical constant init on all outside edges.
    bool has_init = false;
    int64_t init_val = 0;
    for (BasicBlock* pred : header->predecessors()) {
        if (!pred || !pred->terminator() || loop.contains(pred)) continue;
        for_each_edge(*pred->terminator(), [&](const BranchTarget& bt) {
            if (bt.block != header || p_idx >= bt.args.size()) return;
            int64_t c = 0;
            if (!get_const_i64(bt.args[p_idx], c) || (has_init && c != init_val)) {
                has_init = false;
                init_val = INT64_MIN;
                return;
            }
            init_val = c;
            has_init = true;
        });
        if (!has_init) return std::nullopt;
    }
    if (!has_init) return std::nullopt;

    const BranchTarget& back_bt = bterm->branch_target();
    if (p_idx >= back_bt.args.size()) return std::nullopt;
    const Value* step_v = back_bt.args[p_idx];
    const Instruction* step_inst = step_v && step_v->is_instruction() ? step_v->defining_instruction() : nullptr;
    if (!step_inst || step_inst->operand_count() != 2 || step_inst->operand(0) != iv) return std::nullopt;
    if (step_inst->parent() != body) return std::nullopt;
    int64_t step_c = 0;
    if (!get_const_i64(step_inst->operand(1), step_c) || step_c <= 0 || step_inst->opcode() != Opcode::add) {
        return std::nullopt;
    }

    const Opcode cop = cmp->opcode();
    if (cop != Opcode::slt && cop != Opcode::ult && cop != Opcode::sle && cop != Opcode::ule) {
        return std::nullopt;
    }
    if ((cop == Opcode::ult || cop == Opcode::ule) && init_val < 0) return std::nullopt;
    const int64_t inc = (cop == Opcode::sle || cop == Opcode::ule) ? 1 : 0;
    if (init_val > bound || (init_val == bound && inc == 0)) return std::nullopt;
    const int64_t span = (bound - init_val) + inc;
    const int64_t trips = (span + step_c - 1) / step_c;
    if (trips < 1 || trips > static_cast<int64_t>(max_trip)) return std::nullopt;

    // Require header instructions only feed header terminator/cmp, and no instruction defined in
    // header (used outside header) or body (used outside loop) escapes unrolling.
    for (const Value* ea : f_tgt.args) {
        if (!ea) return std::nullopt;
        if (ea->is_instruction() && loop.contains(ea->defining_instruction()->parent())) return std::nullopt;
    }
    bool escapes = false;
    for (const BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;
        for (const Instruction* inst : *bb) {
            if (!inst) continue;
            for_each_use(*inst, [&](const Value* v) {
                if (!v || !v->is_instruction() || !v->defining_instruction()) return;
                const BasicBlock* def_bb = v->defining_instruction()->parent();
                if (def_bb == header && bb != header) escapes = true;
                if (def_bb == body && !loop.contains(bb)) escapes = true;
            });
            if (escapes) return std::nullopt;
        }
    }
    (void)fn;

    SmallCountedLoop info;
    info.header = header;
    info.body = body;
    info.exit = f_tgt.block;
    info.trip_count = static_cast<uint32_t>(trips);
    info.latch_back_args = back_bt.args;
    info.exit_args = f_tgt.args;
    return info;
}

struct CoalesceKey {
    Value* base = nullptr;
    Value* root_idx = nullptr;
    Type elem_type = Type::void_type();

    bool operator==(const CoalesceKey& o) const noexcept {
        return base == o.base && root_idx == o.root_idx && elem_type == o.elem_type;
    }
};

struct CoalesceSlot {
    Instruction* inst = nullptr;
    size_t pos = 0;
    CoalesceKey key;
    int64_t byte_off = 0;
    bool is_load = false;
    bool is_indexed = false;
    Value* store_val = nullptr;
};

void unwrap_indexed_addr(Value* idx_v, Value*& out_root, int64_t& out_elem_off) {
    out_root = nullptr;
    out_elem_off = 0;
    if (!idx_v) return;
    int64_t c = 0;
    if (get_const_i64(idx_v, c)) {
        out_elem_off = c;
        return;
    }
    if (idx_v->is_instruction()) {
        const Instruction* def = idx_v->defining_instruction();
        if (def && def->opcode() == Opcode::add && def->operand_count() == 2) {
            if (get_const_i64(def->operand(1), c)) {
                out_root = def->operand(0);
                out_elem_off = c;
                return;
            }
            if (get_const_i64(def->operand(0), c)) {
                out_root = def->operand(1);
                out_elem_off = c;
                return;
            }
        }
    }
    out_root = idx_v;
    out_elem_off = 0;
}

bool inst_may_alias_access(const Instruction* between, Value* base, bool assume_disjoint_buffers) noexcept {
    if (!between) return false;
    const Opcode op = between->opcode();
    if (op == Opcode::call || op == Opcode::call_indirect || op == Opcode::patchable_call) {
        const std::string_view sym = between->symbol();
        if (sym.rfind("gpu_tid", 0) == 0 || sym.rfind("gpu_ctaid", 0) == 0 ||
            sym.rfind("gpu_ntid", 0) == 0 || sym.rfind("gpu_nctaid", 0) == 0 ||
            sym.rfind("gpu_global_tid", 0) == 0 || sym.rfind("gpu_global_id", 0) == 0 ||
            sym.rfind("ptx_tid", 0) == 0 || sym.rfind("ptx_ctaid", 0) == 0 ||
            sym.rfind("ptx_ntid", 0) == 0 || sym.rfind("ptx_nctaid", 0) == 0 ||
            sym.rfind("ptx_global_tid", 0) == 0 || sym.rfind("ptx_global_id", 0) == 0) {
            return false;
        }
        return true;
    }
    if (op == Opcode::store || op == Opcode::vstore || op == Opcode::store_indexed) {
        Value* store_base = between->operand(0);
        if (store_base == base) return true;
        if (!assume_disjoint_buffers) return true;
        if (!(store_base && base && store_base->is_block_param() && base->is_block_param())) return true;
        return false;
    }
    if (op == Opcode::load || op == Opcode::vload || op == Opcode::load_indexed) {
        return false; // load-vs-load only blocks store coalescing
    }
    return false;
}

bool inst_blocks_store_coalesce(const Instruction* between, Value* base, bool assume_disjoint_buffers) noexcept {
    if (!between) return false;
    const Opcode op = between->opcode();
    if (op == Opcode::load || op == Opcode::vload || op == Opcode::load_indexed) {
        Value* ld_base = between->operand(0);
        if (ld_base == base) return true;
        if (!assume_disjoint_buffers) return true;
        if (!(ld_base && base && ld_base->is_block_param() && base->is_block_param())) return true;
        return false;
    }
    return inst_may_alias_access(between, base, assume_disjoint_buffers);
}

Value* build_coalesced_base_ptr(Builder& b, Value* base_ptr, Value* root_idx) {
    if (!root_idx) return base_ptr;
    Value* idx64 = root_idx;
    if (root_idx->type() == Type::i32()) idx64 = b.build_sext_i64(root_idx);
    else if (root_idx->type() == Type::i8() || root_idx->type() == Type::i16()) idx64 = b.build_zext_i64(root_idx);
    Value* byte_off = b.build_shl(idx64, b.build_iconst_i64(2));
    return b.build_add(base_ptr, byte_off);
}

} // namespace

bool unroll_small_gpu_loops(Function& fn, uint32_t max_trip_count) {
    if (!fn.parent() || !fn.entry_block() || max_trip_count == 0) return false;
    Module& mod = *fn.parent();
    Builder b(mod);
    b.set_function(&fn);

    bool any_changed = false;
    bool progress = true;
    while (progress) {
        progress = false;
        fn.rebuild_cfg_predecessors();
        DominatorTree dom(fn);
        LoopAnalysis la(fn, dom);
        for (LoopInfo* loop : la.post_order_loops()) {
            if (!loop) continue;
            auto match = match_small_counted_loop(fn, *loop, max_trip_count);
            if (!match) continue;

            BasicBlock* preheader = LoopAnalysis::ensure_preheader(fn, *loop);
            if (!preheader || !preheader->terminator()) continue;
            Instruction* ph_term = preheader->terminator();
            if (ph_term->opcode() != Opcode::br || ph_term->branch_target().block != match->header) continue;

            std::vector<Value*> cur_params = ph_term->branch_target().args;
            if (cur_params.size() != match->header->param_count()) continue;

            b.position_before(ph_term);
            std::unordered_map<const Value*, Value*> val_map;
            auto map_v = [&](const Value* v) -> Value* {
                if (!v) return nullptr;
                auto it = val_map.find(v);
                return (it != val_map.end()) ? it->second : const_cast<Value*>(v);
            };

            Instruction* bterm = match->body->terminator();
            for (uint32_t iter = 0; iter < match->trip_count; ++iter) {
                val_map.clear();
                for (size_t k = 0; k < match->header->param_count(); ++k) {
                    val_map[match->header->param(k)] = cur_params[k];
                }
                for (Instruction* src_inst : *match->body) {
                    if (!src_inst || src_inst == bterm) continue;
                    Instruction* dst_inst = mod.arena().make<Instruction>(src_inst->opcode(), src_inst->type());
                    dst_inst->copy_attributes_from(*src_inst);
                    dst_inst->set_loc(src_inst->loc());
                    if (!src_inst->symbol().empty()) dst_inst->set_symbol(mod.string_pool().intern(src_inst->symbol()));
                    if (!src_inst->extra_symbol().empty()) {
                        dst_inst->set_extra_symbol(mod.string_pool().intern(src_inst->extra_symbol()));
                    }
                    for (const Value* op : src_inst->operands()) dst_inst->add_operand(map_v(op));
                    if (src_inst->produces_value()) {
                        Value* res_v = mod.arena().make<Value>(fn.next_value_id(), src_inst->type(), ValueKind::InstructionResult);
                        res_v->set_defining_instruction(dst_inst);
                        dst_inst->set_result(res_v);
                        val_map[src_inst->result()] = res_v;
                    }
                    preheader->insert_before(dst_inst, ph_term);
                }
                std::vector<Value*> next_params;
                next_params.reserve(match->latch_back_args.size());
                for (const Value* ba : match->latch_back_args) next_params.push_back(map_v(ba));
                cur_params = std::move(next_params);
            }

            // Final header exit argument mapping after N iterations.
            val_map.clear();
            for (size_t k = 0; k < match->header->param_count(); ++k) {
                val_map[match->header->param(k)] = cur_params[k];
            }
            std::vector<Value*> mapped_exit_args;
            mapped_exit_args.reserve(match->exit_args.size());
            for (const Value* ea : match->exit_args) mapped_exit_args.push_back(map_v(ea));

            ph_term->set_branch_target(BranchTarget(match->exit, std::move(mapped_exit_args)));
            for (size_t k = 0; k < match->header->param_count(); ++k) {
                replace_uses_globally(fn, match->header->param(k), cur_params[k]);
            }
            fn.remove_block(match->body);
            fn.remove_block(match->header);
            fn.rebuild_cfg_predecessors();

            any_changed = true;
            progress = true;
            break;
        }
    }

    if (any_changed) {
        fn.rebuild_cfg_predecessors();
        verify_function(fn);
    }
    return any_changed;
}

bool coalesce_gpu_v4_accesses(Function& fn, bool assume_disjoint_buffers) {
    if (!fn.parent() || fn.blocks().empty()) return false;
    Module& mod = *fn.parent();
    Builder b(mod);
    b.set_function(&fn);

    bool any_changed = false;

    for (BasicBlock* bb : fn.blocks()) {
        if (!bb) continue;

        bool block_changed = true;
        while (block_changed) {
            block_changed = false;
            std::vector<Instruction*> insts;
            for (Instruction* inst : *bb) {
                if (inst) insts.push_back(inst);
            }

            std::vector<CoalesceSlot> slots;
            for (size_t pos = 0; pos < insts.size(); ++pos) {
                Instruction* inst = insts[pos];
                const Opcode op = inst->opcode();
                if (op == Opcode::load && (inst->type() == Type::f32() || inst->type() == Type::i32())) {
                    slots.push_back({inst, pos, {inst->operand(0), nullptr, inst->type()},
                                     inst->offset(), true, false, nullptr});
                } else if (op == Opcode::store && inst->operand_count() >= 2 && inst->operand(1) &&
                           (inst->operand(1)->type() == Type::f32() || inst->operand(1)->type() == Type::i32())) {
                    slots.push_back({inst, pos, {inst->operand(0), nullptr, inst->operand(1)->type()},
                                     inst->offset(), false, false, inst->operand(1)});
                } else if (op == Opcode::load_indexed && inst->scale() == 4 &&
                           (inst->type() == Type::f32() || inst->type() == Type::i32())) {
                    Value* root_idx = nullptr;
                    int64_t elem_off = 0;
                    unwrap_indexed_addr(inst->operand(1), root_idx, elem_off);
                    if (!root_idx || is_multiple_of_4_index(root_idx)) {
                        slots.push_back({inst, pos, {inst->operand(0), root_idx, inst->type()},
                                         static_cast<int64_t>(inst->offset()) + elem_off * 4, true, true, nullptr});
                    }
                } else if (op == Opcode::store_indexed && inst->scale() == 4 && inst->operand_count() >= 3 &&
                           inst->operand(2) &&
                           (inst->operand(2)->type() == Type::f32() || inst->operand(2)->type() == Type::i32())) {
                    Value* root_idx = nullptr;
                    int64_t elem_off = 0;
                    unwrap_indexed_addr(inst->operand(1), root_idx, elem_off);
                    if (!root_idx || is_multiple_of_4_index(root_idx)) {
                        slots.push_back({inst, pos, {inst->operand(0), root_idx, inst->operand(2)->type()},
                                         static_cast<int64_t>(inst->offset()) + elem_off * 4, false, true, inst->operand(2)});
                    }
                }
            }

            // Look for a valid 4-lane group [base_off, base_off+4, base_off+8, base_off+12].
            for (size_t i = 0; i < slots.size() && !block_changed; ++i) {
                const CoalesceSlot& s0 = slots[i];
                if ((s0.byte_off & 15) != 0) continue;

                const CoalesceSlot* lanes[4] = {&s0, nullptr, nullptr, nullptr};
                for (size_t j = 0; j < slots.size(); ++j) {
                    if (j == i) continue;
                    const CoalesceSlot& sj = slots[j];
                    if (sj.is_load != s0.is_load || !(sj.key == s0.key)) continue;
                    const int64_t diff = sj.byte_off - s0.byte_off;
                    if (diff == 4 && !lanes[1]) lanes[1] = &sj;
                    else if (diff == 8 && !lanes[2]) lanes[2] = &sj;
                    else if (diff == 12 && !lanes[3]) lanes[3] = &sj;
                }
                if (!lanes[1] || !lanes[2] || !lanes[3]) continue;

                size_t min_pos = lanes[0]->pos;
                size_t max_pos = lanes[0]->pos;
                for (int k = 1; k < 4; ++k) {
                    min_pos = std::min(min_pos, lanes[k]->pos);
                    max_pos = std::max(max_pos, lanes[k]->pos);
                }

                bool safe = true;
                for (size_t p = min_pos + 1; p < max_pos; ++p) {
                    const Instruction* mid = insts[p];
                    bool is_member = false;
                    for (int k = 0; k < 4; ++k) {
                        if (lanes[k]->inst == mid) { is_member = true; break; }
                    }
                    if (is_member) continue;
                    if (s0.is_load ? inst_may_alias_access(mid, s0.key.base, assume_disjoint_buffers)
                                   : inst_blocks_store_coalesce(mid, s0.key.base, assume_disjoint_buffers)) {
                        safe = false;
                        break;
                    }
                }
                if (!safe) continue;

                const Type vec_t = (s0.key.elem_type == Type::f32()) ? Type::f32x4() : Type::i32x4();
                const int32_t off32 = static_cast<int32_t>(s0.byte_off);

                if (s0.is_load) {
                    Instruction* anchor = insts[min_pos];
                    b.position_before(anchor);
                    b.set_current_loc(anchor->loc());
                    Value* ptr = build_coalesced_base_ptr(b, s0.key.base, s0.key.root_idx);
                    Value* vec_v = b.build_vload(vec_t, ptr, off32);
                    Value* ex[4] = {nullptr, nullptr, nullptr, nullptr};
                    for (uint32_t k = 0; k < 4; ++k) {
                        ex[k] = b.build_vextract_lane(vec_v, k);
                    }
                    for (int k = 0; k < 4; ++k) {
                        replace_uses_globally(fn, lanes[k]->inst->result(), ex[k]);
                        bb->remove_instruction(lanes[k]->inst);
                    }
                } else {
                    Instruction* anchor = insts[max_pos];
                    b.position_before(anchor);
                    b.set_current_loc(anchor->loc());
                    Value* ptr = build_coalesced_base_ptr(b, s0.key.base, s0.key.root_idx);
                    Value* vec_v = b.build_vzero(vec_t);
                    for (uint32_t k = 0; k < 4; ++k) {
                        vec_v = b.build_vinsert_lane(vec_v, lanes[k]->store_val, k);
                    }
                    b.build_vstore(vec_t, ptr, off32, vec_v);
                    for (int k = 0; k < 4; ++k) {
                        bb->remove_instruction(lanes[k]->inst);
                    }
                }

                b.clear_current_loc();
                block_changed = true;
                any_changed = true;
            }
        }
    }

    if (any_changed) verify_function(fn);
    return any_changed;
}

} // namespace brass
