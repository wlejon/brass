#include "loop_tile_transform.hpp"
#include <brass/mir/opcodes.hpp>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <algorithm>

namespace brass {

namespace {

Value* build_const_int(Builder& b, Type t, int64_t val) {
    if (t == Type::i32()) {
        return b.build_iconst_i32(static_cast<int32_t>(val));
    }
    return b.build_iconst_i64(val);
}

Value* build_min_bound(Builder& b, Value* a, Value* limit, Opcode cmp_op) {
    Value* cmp = nullptr;
    switch (cmp_op) {
        case Opcode::slt: cmp = b.build_slt(a, limit); break;
        case Opcode::ult: cmp = b.build_ult(a, limit); break;
        case Opcode::sle: cmp = b.build_sle(a, limit); break;
        case Opcode::ule: cmp = b.build_ule(a, limit); break;
        default: cmp = b.build_slt(a, limit); break;
    }
    return b.build_select(cmp, a, limit);
}

void clone_block_instructions(
    Function& fn,
    BasicBlock* src_bb,
    BasicBlock* dst_bb,
    std::unordered_map<const Value*, Value*>& val_map
) {
    if (!src_bb || !dst_bb) return;

    for (Instruction* inst = src_bb->head(); inst != nullptr; inst = inst->next()) {
        if (inst->is_terminator()) break;

        Instruction* cloned = fn.parent()->arena().make<Instruction>(inst->opcode(), inst->type());
        cloned->set_imm_i64(inst->imm_i64());
        cloned->set_imm_f64(inst->imm_f64());
        cloned->set_scale(inst->scale());
        cloned->set_offset(inst->offset());
        cloned->set_memory_type(inst->memory_type());
        if (!inst->symbol().empty()) {
            cloned->set_symbol(fn.parent()->string_pool().intern(inst->symbol()));
        }

        for (Value* op : inst->operands()) {
            if (!op) continue;
            auto it = val_map.find(op);
            cloned->add_operand(it != val_map.end() ? it->second : op);
        }

        if (inst->produces_value()) {
            Value* res = fn.parent()->arena().make<Value>(
                fn.next_value_id(), inst->type(), ValueKind::InstructionResult);
            res->set_defining_instruction(cloned);
            cloned->set_result(res);
            val_map[inst->result()] = res;
        }

        dst_bb->append_instruction(cloned);
    }
}

static void remove_nest_blocks(Function& fn, const LoopNest& nest, BasicBlock* preheader, BasicBlock* outer_exit_bb) {
    std::unordered_set<BasicBlock*> to_remove;
    for (size_t l = 0; l < nest.depth(); ++l) {
        const auto& lvl = nest.level(l);
        for (BasicBlock* bb : lvl.blocks) {
            if (bb && bb != outer_exit_bb && bb != preheader) {
                to_remove.insert(bb);
            }
        }
        for (BasicBlock* bb : {lvl.header, lvl.body, lvl.latch, lvl.exit_bb}) {
            if (bb && bb != outer_exit_bb && bb != preheader) {
                to_remove.insert(bb);
            }
        }
    }
    if (nest.reduction_store_block() && nest.reduction_store_block() != outer_exit_bb && nest.reduction_store_block() != preheader) {
        to_remove.insert(nest.reduction_store_block());
    }

    for (BasicBlock* bb : to_remove) {
        fn.remove_block(bb);
    }
    fn.rebuild_cfg_predecessors();
}

} // namespace

bool transform_2d_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
) {
    (void)dom;
    if (nest.depth() < 2) return false;

    const auto& lvl0 = nest.level(0);
    const auto& lvl1 = nest.level(1);
    if (!lvl0.header || !lvl1.header) return false;

    BasicBlock* preheader = lvl0.preheader;
    if (!preheader && lvl0.loop) preheader = lvl0.loop->preheader();
    if (!preheader && lvl0.loop) preheader = LoopAnalysis::ensure_preheader(fn, *lvl0.loop);
    if (!preheader || !preheader->terminator()) return false;

    Builder b(*fn.parent());
    b.set_function(&fn);

    std::string pfx = std::string(lvl0.header->name()) + "_2dt";
    BasicBlock* tile_i_hdr = b.create_block(pfx + "_i_hdr");
    BasicBlock* tile_i_body = b.create_block(pfx + "_i_body");
    BasicBlock* tile_j_hdr = b.create_block(pfx + "_j_hdr");
    BasicBlock* tile_j_body = b.create_block(pfx + "_j_body");
    BasicBlock* point_i_hdr = b.create_block(pfx + "_pi_hdr");
    BasicBlock* point_i_body = b.create_block(pfx + "_pi_body");
    BasicBlock* point_j_hdr = b.create_block(pfx + "_pj_hdr");
    BasicBlock* point_j_body = b.create_block(pfx + "_pj_body");
    BasicBlock* point_i_latch = b.create_block(pfx + "_pi_latch");
    BasicBlock* tile_j_latch = b.create_block(pfx + "_j_latch");
    BasicBlock* tile_i_latch = b.create_block(pfx + "_i_latch");

    auto& fn_blocks = fn.blocks();
    auto it = std::find(fn_blocks.begin(), fn_blocks.end(), lvl0.header);
    std::vector<BasicBlock*> new_blocks = {
        tile_i_hdr, tile_i_body, tile_j_hdr, tile_j_body,
        point_i_hdr, point_i_body, point_j_hdr, point_j_body,
        point_i_latch, tile_j_latch, tile_i_latch
    };
    fn_blocks.insert(it, new_blocks.begin(), new_blocks.end());

    tile_i_hdr->set_parent(&fn);
    tile_i_body->set_parent(&fn);
    tile_j_hdr->set_parent(&fn);
    tile_j_body->set_parent(&fn);
    point_i_hdr->set_parent(&fn);
    point_i_body->set_parent(&fn);
    point_j_hdr->set_parent(&fn);
    point_j_body->set_parent(&fn);
    point_i_latch->set_parent(&fn);
    tile_j_latch->set_parent(&fn);
    tile_i_latch->set_parent(&fn);

    // 1. Preheader redirect to tile_i_hdr
    Instruction* ph_term = preheader->terminator();
    if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == lvl0.header) {
        ph_term->branch_target().block = tile_i_hdr;
    } else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == lvl0.header) ph_term->true_target().block = tile_i_hdr;
        if (ph_term->false_target().block == lvl0.header) ph_term->false_target().block = tile_i_hdr;
    }

    // 2. tile_i_hdr
    std::vector<Value*> tile_i_params;
    Value* i_tile = nullptr;
    for (size_t p = 0; p < lvl0.header->param_count(); ++p) {
        Value* param = b.add_block_param(tile_i_hdr, lvl0.header->param(p)->type());
        if (p == lvl0.iv_param_index) {
            i_tile = param;
        }
        tile_i_params.push_back(param);
    }

    b.position_at_end(tile_i_hdr);
    Value* cond_i = nullptr;
    switch (lvl0.cmp_opcode) {
        case Opcode::slt: cond_i = b.build_slt(i_tile, lvl0.limit_val); break;
        case Opcode::ult: cond_i = b.build_ult(i_tile, lvl0.limit_val); break;
        case Opcode::sle: cond_i = b.build_sle(i_tile, lvl0.limit_val); break;
        case Opcode::ule: cond_i = b.build_ule(i_tile, lvl0.limit_val); break;
        default: cond_i = b.build_slt(i_tile, lvl0.limit_val); break;
    }

    Instruction* old_lvl0_term = lvl0.header->terminator();
    const BranchTarget& old_exit_target = lvl0.exit_on_false ? old_lvl0_term->false_target() : old_lvl0_term->true_target();
    std::vector<Value*> final_exit_args;
    for (Value* arg : old_exit_target.args) {
        if (arg && arg->is_block_param() && arg->defining_block() == lvl0.header) {
            final_exit_args.push_back(tile_i_hdr->param(arg->param_index()));
        } else {
            final_exit_args.push_back(arg);
        }
    }
    b.build_br_if(cond_i, tile_i_body, {}, lvl0.exit_bb, final_exit_args);

    // 3. tile_i_body: compute i_limit = min(i_tile + Ti, limit)
    b.position_at_end(tile_i_body);
    Value* ti_val = build_const_int(b, lvl0.iv_type, static_cast<int64_t>(options.tile_size_i));
    Value* i_plus_ti = b.build_add(i_tile, ti_val);
    Value* i_limit = build_min_bound(b, i_plus_ti, lvl0.limit_val, lvl0.cmp_opcode);

    Value* zero_j = build_const_int(b, lvl1.iv_type, 0);
    b.build_br(tile_j_hdr, {zero_j});

    // 4. tile_j_hdr
    b.position_at_end(tile_j_hdr);
    Value* j_tile = b.add_block_param(tile_j_hdr, lvl1.iv_type);
    Value* cond_j = nullptr;
    switch (lvl1.cmp_opcode) {
        case Opcode::slt: cond_j = b.build_slt(j_tile, lvl1.limit_val); break;
        case Opcode::ult: cond_j = b.build_ult(j_tile, lvl1.limit_val); break;
        case Opcode::sle: cond_j = b.build_sle(j_tile, lvl1.limit_val); break;
        case Opcode::ule: cond_j = b.build_ule(j_tile, lvl1.limit_val); break;
        default: cond_j = b.build_slt(j_tile, lvl1.limit_val); break;
    }
    b.build_br_if(cond_j, tile_j_body, {}, tile_i_latch, {});

    // 5. tile_j_body: compute j_limit = min(j_tile + Tj, limit)
    b.position_at_end(tile_j_body);
    Value* tj_val = build_const_int(b, lvl1.iv_type, static_cast<int64_t>(options.tile_size_j));
    Value* j_plus_tj = b.build_add(j_tile, tj_val);
    Value* j_limit = build_min_bound(b, j_plus_tj, lvl1.limit_val, lvl1.cmp_opcode);
    b.build_br(point_i_hdr, {i_tile});

    // 6. point_i_hdr
    b.position_at_end(point_i_hdr);
    Value* i_point = b.add_block_param(point_i_hdr, lvl0.iv_type);
    Value* cond_pi = b.build_slt(i_point, i_limit);
    b.build_br_if(cond_pi, point_i_body, {}, tile_j_latch, {});

    // 7. point_i_body
    b.position_at_end(point_i_body);
    std::unordered_map<const Value*, Value*> val_map;
    val_map[lvl0.iv_param] = i_point;
    clone_block_instructions(fn, lvl0.body, point_i_body, val_map);
    b.build_br(point_j_hdr, {j_tile});

    // 8. point_j_hdr
    b.position_at_end(point_j_hdr);
    Value* j_point = b.add_block_param(point_j_hdr, lvl1.iv_type);
    Value* cond_pj = b.build_slt(j_point, j_limit);
    b.build_br_if(cond_pj, point_j_body, {}, point_i_latch, {});

    // 9. point_j_body
    b.position_at_end(point_j_body);
    val_map[lvl1.iv_param] = j_point;
    clone_block_instructions(fn, lvl1.body, point_j_body, val_map);

    Value* step_one_j = build_const_int(b, lvl1.iv_type, 1);
    Value* next_jp = b.build_add(j_point, step_one_j);
    b.build_br(point_j_hdr, {next_jp});

    // 10. point_i_latch
    b.position_at_end(point_i_latch);
    Value* step_one_i = build_const_int(b, lvl0.iv_type, 1);
    Value* next_ip = b.build_add(i_point, step_one_i);
    b.build_br(point_i_hdr, {next_ip});

    // 11. tile_j_latch
    b.position_at_end(tile_j_latch);
    Value* next_jt = b.build_add(j_tile, tj_val);
    b.build_br(tile_j_hdr, {next_jt});

    // 12. tile_i_latch
    b.position_at_end(tile_i_latch);
    Value* next_it = b.build_add(i_tile, ti_val);
    b.build_br(tile_i_hdr, {next_it});

    remove_nest_blocks(fn, nest, preheader, lvl0.exit_bb);
    return true;
}

static bool extract_matmul_arrays(
    const LoopNest& nest,
    Value*& base_a,
    Value*& base_b,
    Value*& base_c,
    Type& elem_type,
    uint8_t& scale
) {
    base_a = nullptr;
    base_b = nullptr;
    base_c = nullptr;
    for (const auto& acc : nest.memory_accesses()) {
        bool has_i = false;
        bool has_j = false;
        bool has_k = false;
        for (const auto& t : acc.terms) {
            if (t.level_index == 0) has_i = true;
            if (t.level_index == 1) has_j = true;
            if (t.level_index == 2) has_k = true;
        }
        if (!acc.is_store && has_i && has_k && !has_j) {
            base_a = acc.base;
            elem_type = acc.elem_type;
            scale = acc.scale;
        } else if (!acc.is_store && has_k && has_j && !has_i) {
            base_b = acc.base;
        } else if (acc.is_store && has_i && has_j && !has_k) {
            base_c = acc.base;
        }
    }
    return base_a != nullptr && base_b != nullptr && base_c != nullptr;
}

static bool transform_3d_matmul_interchanged(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
) {
    (void)dom;
    Value* base_a = nullptr;
    Value* base_b = nullptr;
    Value* base_c = nullptr;
    Type elem_type = Type::i64();
    uint8_t scale = 8;
    if (!extract_matmul_arrays(nest, base_a, base_b, base_c, elem_type, scale)) {
        return false;
    }

    const auto& lvl0 = nest.level(0);
    const auto& lvl1 = nest.level(1);
    const auto& lvl2 = nest.level(2);
    if (!lvl0.header || !lvl1.header || !lvl2.header) return false;

    BasicBlock* preheader = lvl0.preheader;
    if (!preheader && lvl0.loop) preheader = lvl0.loop->preheader();
    if (!preheader && lvl0.loop) preheader = LoopAnalysis::ensure_preheader(fn, *lvl0.loop);
    if (!preheader || !preheader->terminator()) return false;

    Builder b(*fn.parent());
    b.set_function(&fn);

    std::string pfx = std::string(lvl0.header->name()) + "_3dti";

    BasicBlock* tile_i_hdr = b.create_block(pfx + "_i_hdr");
    BasicBlock* tile_i_body = b.create_block(pfx + "_i_body");
    BasicBlock* tile_k_hdr = b.create_block(pfx + "_k_hdr");
    BasicBlock* tile_k_body = b.create_block(pfx + "_k_body");
    BasicBlock* tile_j_hdr = b.create_block(pfx + "_j_hdr");
    BasicBlock* tile_j_body = b.create_block(pfx + "_j_body");
    BasicBlock* point_i_hdr = b.create_block(pfx + "_pi_hdr");
    BasicBlock* point_i_body = b.create_block(pfx + "_pi_body");
    BasicBlock* point_k_hdr = b.create_block(pfx + "_pk_hdr");
    BasicBlock* point_k_body = b.create_block(pfx + "_pk_body");
    BasicBlock* point_j_hdr = b.create_block(pfx + "_pj_hdr");
    BasicBlock* point_j_body = b.create_block(pfx + "_pj_body");
    BasicBlock* point_k_latch = b.create_block(pfx + "_pk_latch");
    BasicBlock* point_i_latch = b.create_block(pfx + "_pi_latch");
    BasicBlock* tile_j_latch = b.create_block(pfx + "_j_latch");
    BasicBlock* tile_k_latch = b.create_block(pfx + "_k_latch");
    BasicBlock* tile_i_latch = b.create_block(pfx + "_i_latch");

    auto& fn_blocks = fn.blocks();
    auto it = std::find(fn_blocks.begin(), fn_blocks.end(), lvl0.header);
    std::vector<BasicBlock*> new_blocks = {
        tile_i_hdr, tile_i_body, tile_k_hdr, tile_k_body,
        tile_j_hdr, tile_j_body, point_i_hdr, point_i_body,
        point_k_hdr, point_k_body, point_j_hdr, point_j_body,
        point_k_latch, point_i_latch, tile_j_latch, tile_k_latch,
        tile_i_latch
    };
    fn_blocks.insert(it, new_blocks.begin(), new_blocks.end());

    for (BasicBlock* nb : {tile_i_hdr, tile_i_body, tile_k_hdr, tile_k_body,
                           tile_j_hdr, tile_j_body, point_i_hdr, point_i_body,
                           point_k_hdr, point_k_body, point_j_hdr, point_j_body,
                           point_k_latch, point_i_latch, tile_j_latch, tile_k_latch, tile_i_latch}) {
        nb->set_parent(&fn);
    }

    // Preheader redirect
    Instruction* ph_term = preheader->terminator();
    if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == lvl0.header) {
        ph_term->branch_target().block = tile_i_hdr;
    } else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == lvl0.header) ph_term->true_target().block = tile_i_hdr;
        if (ph_term->false_target().block == lvl0.header) ph_term->false_target().block = tile_i_hdr;
    }

    // tile_i_hdr
    Value* i_tile = nullptr;
    for (size_t p = 0; p < lvl0.header->param_count(); ++p) {
        Value* param = b.add_block_param(tile_i_hdr, lvl0.header->param(p)->type());
        if (p == lvl0.iv_param_index) i_tile = param;
    }

    b.position_at_end(tile_i_hdr);
    Value* cond_i = b.build_slt(i_tile, lvl0.limit_val);
    Instruction* old_lvl0_term = lvl0.header->terminator();
    const BranchTarget& old_exit_target = lvl0.exit_on_false ? old_lvl0_term->false_target() : old_lvl0_term->true_target();
    std::vector<Value*> exit_args;
    for (Value* arg : old_exit_target.args) {
        if (arg && arg->is_block_param() && arg->defining_block() == lvl0.header) {
            exit_args.push_back(tile_i_hdr->param(arg->param_index()));
        } else {
            exit_args.push_back(arg);
        }
    }
    b.build_br_if(cond_i, tile_i_body, {}, lvl0.exit_bb, exit_args);

    // tile_i_body
    b.position_at_end(tile_i_body);
    Value* ti_val = build_const_int(b, lvl0.iv_type, static_cast<int64_t>(options.tile_size_i));
    Value* i_plus_ti = b.build_add(i_tile, ti_val);
    Value* i_limit = build_min_bound(b, i_plus_ti, lvl0.limit_val, lvl0.cmp_opcode);
    Value* zero_k = build_const_int(b, lvl2.iv_type, 0);
    b.build_br(tile_k_hdr, {zero_k});

    // tile_k_hdr
    b.position_at_end(tile_k_hdr);
    Value* k_tile = b.add_block_param(tile_k_hdr, lvl2.iv_type);
    Value* cond_k = b.build_slt(k_tile, lvl2.limit_val);
    b.build_br_if(cond_k, tile_k_body, {}, tile_i_latch, {});

    // tile_k_body
    b.position_at_end(tile_k_body);
    Value* tk_val = build_const_int(b, lvl2.iv_type, static_cast<int64_t>(options.tile_size_k));
    Value* k_plus_tk = b.build_add(k_tile, tk_val);
    Value* k_limit = build_min_bound(b, k_plus_tk, lvl2.limit_val, lvl2.cmp_opcode);
    Value* zero_j = build_const_int(b, lvl1.iv_type, 0);
    b.build_br(tile_j_hdr, {zero_j});

    // tile_j_hdr
    b.position_at_end(tile_j_hdr);
    Value* j_tile = b.add_block_param(tile_j_hdr, lvl1.iv_type);
    Value* cond_j = b.build_slt(j_tile, lvl1.limit_val);
    b.build_br_if(cond_j, tile_j_body, {}, tile_k_latch, {});

    // tile_j_body
    b.position_at_end(tile_j_body);
    Value* tj_val = build_const_int(b, lvl1.iv_type, static_cast<int64_t>(options.tile_size_j));
    Value* j_plus_tj = b.build_add(j_tile, tj_val);
    Value* j_limit = build_min_bound(b, j_plus_tj, lvl1.limit_val, lvl1.cmp_opcode);
    b.build_br(point_i_hdr, {i_tile});

    // point_i_hdr
    b.position_at_end(point_i_hdr);
    Value* i_point = b.add_block_param(point_i_hdr, lvl0.iv_type);
    Value* cond_pi = b.build_slt(i_point, i_limit);
    b.build_br_if(cond_pi, point_i_body, {}, tile_j_latch, {});

    // point_i_body
    b.position_at_end(point_i_body);
    Value* row_a = b.build_mul(i_point, lvl2.limit_val);
    Value* row_c = b.build_mul(i_point, lvl1.limit_val);
    b.build_br(point_k_hdr, {k_tile});

    // point_k_hdr
    b.position_at_end(point_k_hdr);
    Value* k_point = b.add_block_param(point_k_hdr, lvl2.iv_type);
    Value* cond_pk = b.build_slt(k_point, k_limit);
    b.build_br_if(cond_pk, point_k_body, {}, point_i_latch, {});

    // point_k_body
    b.position_at_end(point_k_body);
    Value* idx_a = b.build_add(row_a, k_point);
    Value* val_a = b.build_load_indexed(elem_type, base_a, idx_a, scale, 0);
    Value* row_b = b.build_mul(k_point, lvl1.limit_val);
    b.build_br(point_j_hdr, {j_tile});

    // point_j_hdr
    b.position_at_end(point_j_hdr);
    Value* j_point = b.add_block_param(point_j_hdr, lvl1.iv_type);
    Value* cond_pj = b.build_slt(j_point, j_limit);
    b.build_br_if(cond_pj, point_j_body, {}, point_k_latch, {});

    // point_j_body
    b.position_at_end(point_j_body);
    Value* idx_b = b.build_add(row_b, j_point);
    Value* val_b = b.build_load_indexed(elem_type, base_b, idx_b, scale, 0);
    Value* idx_c = b.build_add(row_c, j_point);
    Value* val_c = b.build_load_indexed(elem_type, base_c, idx_c, scale, 0);
    Value* term = b.build_mul(val_a, val_b);
    Value* next_c = b.build_add(val_c, term);
    b.build_store_indexed(elem_type, base_c, idx_c, scale, 0, next_c);

    Value* step_one_j = build_const_int(b, lvl1.iv_type, 1);
    Value* next_jp = b.build_add(j_point, step_one_j);
    b.build_br(point_j_hdr, {next_jp});

    // point_k_latch
    b.position_at_end(point_k_latch);
    Value* step_one_k = build_const_int(b, lvl2.iv_type, 1);
    Value* next_kp = b.build_add(k_point, step_one_k);
    b.build_br(point_k_hdr, {next_kp});

    // point_i_latch
    b.position_at_end(point_i_latch);
    Value* step_one_i = build_const_int(b, lvl0.iv_type, 1);
    Value* next_ip = b.build_add(i_point, step_one_i);
    b.build_br(point_i_hdr, {next_ip});

    // tile_j_latch
    b.position_at_end(tile_j_latch);
    Value* next_jt = b.build_add(j_tile, tj_val);
    b.build_br(tile_j_hdr, {next_jt});

    // tile_k_latch
    b.position_at_end(tile_k_latch);
    Value* next_kt = b.build_add(k_tile, tk_val);
    b.build_br(tile_k_hdr, {next_kt});

    // tile_i_latch
    b.position_at_end(tile_i_latch);
    Value* next_it = b.build_add(i_tile, ti_val);
    b.build_br(tile_i_hdr, {next_it});

    remove_nest_blocks(fn, nest, preheader, lvl0.exit_bb);
    return true;
}

static bool transform_3d_matmul_standard(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
) {
    (void)dom;
    const auto& lvl0 = nest.level(0);
    const auto& lvl1 = nest.level(1);
    const auto& lvl2 = nest.level(2);

    BasicBlock* preheader = lvl0.preheader;
    if (!preheader && lvl0.loop) preheader = lvl0.loop->preheader();
    if (!preheader && lvl0.loop) preheader = LoopAnalysis::ensure_preheader(fn, *lvl0.loop);
    if (!preheader || !preheader->terminator()) return false;

    Builder b(*fn.parent());
    b.set_function(&fn);

    std::string pfx = std::string(lvl0.header->name()) + "_3dt";

    BasicBlock* tile_i_hdr = b.create_block(pfx + "_i_hdr");
    BasicBlock* tile_i_body = b.create_block(pfx + "_i_body");
    BasicBlock* tile_j_hdr = b.create_block(pfx + "_j_hdr");
    BasicBlock* tile_j_body = b.create_block(pfx + "_j_body");
    BasicBlock* point_i_hdr = b.create_block(pfx + "_pi_hdr");
    BasicBlock* point_i_body = b.create_block(pfx + "_pi_body");
    BasicBlock* point_j_hdr = b.create_block(pfx + "_pj_hdr");
    BasicBlock* point_j_body = b.create_block(pfx + "_pj_body");
    BasicBlock* tile_k_hdr = b.create_block(pfx + "_k_hdr");
    BasicBlock* tile_k_body = b.create_block(pfx + "_k_body");
    BasicBlock* point_k_hdr = b.create_block(pfx + "_pk_hdr");
    BasicBlock* point_k_body = b.create_block(pfx + "_pk_body");
    BasicBlock* tile_k_latch = b.create_block(pfx + "_k_latch");
    BasicBlock* point_j_store = b.create_block(pfx + "_pj_store");
    BasicBlock* point_i_latch = b.create_block(pfx + "_pi_latch");
    BasicBlock* tile_j_latch = b.create_block(pfx + "_j_latch");
    BasicBlock* tile_i_latch = b.create_block(pfx + "_i_latch");

    auto& fn_blocks = fn.blocks();
    auto it = std::find(fn_blocks.begin(), fn_blocks.end(), lvl0.header);
    std::vector<BasicBlock*> new_blocks = {
        tile_i_hdr, tile_i_body, tile_j_hdr, tile_j_body,
        point_i_hdr, point_i_body, point_j_hdr, point_j_body,
        tile_k_hdr, tile_k_body, point_k_hdr, point_k_body,
        tile_k_latch, point_j_store, point_i_latch, tile_j_latch,
        tile_i_latch
    };
    fn_blocks.insert(it, new_blocks.begin(), new_blocks.end());

    for (BasicBlock* nb : {tile_i_hdr, tile_i_body, tile_j_hdr, tile_j_body,
                           point_i_hdr, point_i_body, point_j_hdr, point_j_body,
                           tile_k_hdr, tile_k_body, point_k_hdr, point_k_body,
                           tile_k_latch, point_j_store, point_i_latch, tile_j_latch, tile_i_latch}) {
        nb->set_parent(&fn);
    }

    // 1. Preheader redirect
    Instruction* ph_term = preheader->terminator();
    if (ph_term->opcode() == Opcode::br && ph_term->branch_target().block == lvl0.header) {
        ph_term->branch_target().block = tile_i_hdr;
    } else if (ph_term->opcode() == Opcode::br_if) {
        if (ph_term->true_target().block == lvl0.header) ph_term->true_target().block = tile_i_hdr;
        if (ph_term->false_target().block == lvl0.header) ph_term->false_target().block = tile_i_hdr;
    }

    // 2. tile_i_hdr
    Value* i_tile = nullptr;
    for (size_t p = 0; p < lvl0.header->param_count(); ++p) {
        Value* param = b.add_block_param(tile_i_hdr, lvl0.header->param(p)->type());
        if (p == lvl0.iv_param_index) i_tile = param;
    }

    b.position_at_end(tile_i_hdr);
    Value* cond_i = b.build_slt(i_tile, lvl0.limit_val);
    Instruction* old_lvl0_term = lvl0.header->terminator();
    const BranchTarget& old_exit_target = lvl0.exit_on_false ? old_lvl0_term->false_target() : old_lvl0_term->true_target();
    std::vector<Value*> exit_args;
    for (Value* arg : old_exit_target.args) {
        if (arg && arg->is_block_param() && arg->defining_block() == lvl0.header) {
            exit_args.push_back(tile_i_hdr->param(arg->param_index()));
        } else {
            exit_args.push_back(arg);
        }
    }
    b.build_br_if(cond_i, tile_i_body, {}, lvl0.exit_bb, exit_args);

    // 3. tile_i_body
    b.position_at_end(tile_i_body);
    Value* ti_val = build_const_int(b, lvl0.iv_type, static_cast<int64_t>(options.tile_size_i));
    Value* i_plus_ti = b.build_add(i_tile, ti_val);
    Value* i_limit = build_min_bound(b, i_plus_ti, lvl0.limit_val, lvl0.cmp_opcode);

    Value* zero_j = build_const_int(b, lvl1.iv_type, 0);
    b.build_br(tile_j_hdr, {zero_j});

    // 4. tile_j_hdr
    b.position_at_end(tile_j_hdr);
    Value* j_tile = b.add_block_param(tile_j_hdr, lvl1.iv_type);
    Value* cond_j = b.build_slt(j_tile, lvl1.limit_val);
    b.build_br_if(cond_j, tile_j_body, {}, tile_i_latch, {});

    // 5. tile_j_body
    b.position_at_end(tile_j_body);
    Value* tj_val = build_const_int(b, lvl1.iv_type, static_cast<int64_t>(options.tile_size_j));
    Value* j_plus_tj = b.build_add(j_tile, tj_val);
    Value* j_limit = build_min_bound(b, j_plus_tj, lvl1.limit_val, lvl1.cmp_opcode);
    b.build_br(point_i_hdr, {i_tile});

    // 6. point_i_hdr
    b.position_at_end(point_i_hdr);
    Value* i_point = b.add_block_param(point_i_hdr, lvl0.iv_type);
    Value* cond_pi = b.build_slt(i_point, i_limit);
    b.build_br_if(cond_pi, point_i_body, {}, tile_j_latch, {});

    // 7. point_i_body
    b.position_at_end(point_i_body);
    std::unordered_map<const Value*, Value*> val_map;
    val_map[lvl0.iv_param] = i_point;
    clone_block_instructions(fn, lvl0.body, point_i_body, val_map);
    b.build_br(point_j_hdr, {j_tile});

    // 8. point_j_hdr
    b.position_at_end(point_j_hdr);
    Value* j_point = b.add_block_param(point_j_hdr, lvl1.iv_type);
    Value* cond_pj = b.build_slt(j_point, j_limit);
    b.build_br_if(cond_pj, point_j_body, {}, point_i_latch, {});

    // 9. point_j_body
    b.position_at_end(point_j_body);
    val_map[lvl1.iv_param] = j_point;
    clone_block_instructions(fn, lvl1.body, point_j_body, val_map);

    Value* init_sum = nest.reduction_init();
    if (!init_sum) {
        init_sum = lvl2.header->param(lvl2.header->param_count() - 1)->type().is_float()
            ? b.build_fconst_f64(0.0) : b.build_iconst_i64(0);
    }
    Value* zero_k = build_const_int(b, lvl2.iv_type, 0);
    b.build_br(tile_k_hdr, {zero_k, init_sum});

    // 10. tile_k_hdr(k_tile, sum_k)
    b.position_at_end(tile_k_hdr);
    Value* k_tile = b.add_block_param(tile_k_hdr, lvl2.iv_type);
    Value* sum_k = b.add_block_param(tile_k_hdr, init_sum->type());
    Value* cond_k = b.build_slt(k_tile, lvl2.limit_val);
    b.build_br_if(cond_k, tile_k_body, {}, point_j_store, {sum_k});

    // 11. tile_k_body
    b.position_at_end(tile_k_body);
    Value* tk_val = build_const_int(b, lvl2.iv_type, static_cast<int64_t>(options.tile_size_k));
    Value* k_plus_tk = b.build_add(k_tile, tk_val);
    Value* k_limit = build_min_bound(b, k_plus_tk, lvl2.limit_val, lvl2.cmp_opcode);
    b.build_br(point_k_hdr, {k_tile, sum_k});

    // 12. point_k_hdr(k_point, sum_p)
    b.position_at_end(point_k_hdr);
    Value* k_point = b.add_block_param(point_k_hdr, lvl2.iv_type);
    Value* sum_p = b.add_block_param(point_k_hdr, init_sum->type());
    Value* cond_pk = b.build_slt(k_point, k_limit);
    b.build_br_if(cond_pk, point_k_body, {}, tile_k_latch, {sum_p});

    // 13. point_k_body
    b.position_at_end(point_k_body);
    val_map[lvl2.iv_param] = k_point;

    // In matmul, the reduction accumulator parameter in lvl2.header is mapped to sum_p
    for (size_t p = 0; p < lvl2.header->param_count(); ++p) {
        if (p != lvl2.iv_param_index) {
            val_map[lvl2.header->param(p)] = sum_p;
        }
    }

    clone_block_instructions(fn, lvl2.body, point_k_body, val_map);

    Value* next_sum = nullptr;
    if (nest.reduction_inst() && val_map.count(nest.reduction_inst()->result())) {
        next_sum = val_map[nest.reduction_inst()->result()];
    } else {
        next_sum = sum_p;
    }

    Value* step_one_k = build_const_int(b, lvl2.iv_type, 1);
    Value* next_kp = b.build_add(k_point, step_one_k);
    b.build_br(point_k_hdr, {next_kp, next_sum});

    // 14. tile_k_latch(sum_after_k)
    b.position_at_end(tile_k_latch);
    Value* sum_after_k = b.add_block_param(tile_k_latch, init_sum->type());
    Value* next_kt = b.build_add(k_tile, tk_val);
    b.build_br(tile_k_hdr, {next_kt, sum_after_k});

    // 15. point_j_store(final_sum)
    b.position_at_end(point_j_store);
    Value* final_sum = b.add_block_param(point_j_store, init_sum->type());

    if (nest.reduction_store_block()) {
        if (nest.reduction_store_block()->param_count() > 0) {
            val_map[nest.reduction_store_block()->param(0)] = final_sum;
        }
        clone_block_instructions(fn, nest.reduction_store_block(), point_j_store, val_map);
    }

    Value* step_one_j = build_const_int(b, lvl1.iv_type, 1);
    Value* next_jp = b.build_add(j_point, step_one_j);
    b.build_br(point_j_hdr, {next_jp});

    // 16. point_i_latch
    b.position_at_end(point_i_latch);
    Value* step_one_i = build_const_int(b, lvl0.iv_type, 1);
    Value* next_ip = b.build_add(i_point, step_one_i);
    b.build_br(point_i_hdr, {next_ip});

    // 17. tile_j_latch
    b.position_at_end(tile_j_latch);
    Value* next_jt = b.build_add(j_tile, tj_val);
    b.build_br(tile_j_hdr, {next_jt});

    // 18. tile_i_latch
    b.position_at_end(tile_i_latch);
    Value* next_it = b.build_add(i_tile, ti_val);
    b.build_br(tile_i_hdr, {next_it});

    remove_nest_blocks(fn, nest, preheader, lvl0.exit_bb);
    return true;
}

bool transform_3d_matmul_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
) {
    if (options.enable_loop_interchange) {
        if (transform_3d_matmul_interchanged(fn, nest, dom, options)) {
            return true;
        }
    }
    return transform_3d_matmul_standard(fn, nest, dom, options);
}

bool transform_3d_generic_loop_nest(
    Function& fn,
    LoopNest& nest,
    const DominatorTree& dom,
    const LoopTileOptions& options
) {
    if (nest.is_matrix_multiply()) {
        return transform_3d_matmul_loop_nest(fn, nest, dom, options);
    }
    // For general 3D without reduction, tile the outer 2 dimensions
    return transform_2d_loop_nest(fn, nest, dom, options);
}

} // namespace brass
