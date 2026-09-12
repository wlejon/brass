#include "il_property_lowering.hpp"
#include "il_lowering.hpp"
#include <brass/il_translator/il_property.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/core/string_pool.hpp>
#include <brass/runtime/object.hpp>

#include <atomic>

namespace brass::il {

static std::atomic<uint32_t> g_global_auto_site_id{1};

Value* PropertyLoweringHelper::lower_prop_get(
    Builder& b,
    Value* obj,
    std::string_view prop_name,
    uint32_t symbol_id,
    uint32_t site_id
) {
    if (prop_name.empty()) {
        Value* sym_val = nullptr;
        if (symbol_id == 0xFFFFFFFFu) {
            sym_val = b.build_iconst_i32(static_cast<int32_t>(symbol_id));
        } else {
            Value* map_addr = b.build_func_addr("__bronze_key_map");
            sym_val = b.build_load(Type::i32(), map_addr, static_cast<int32_t>(symbol_id * sizeof(uint32_t)));
        }
        Value* null_entry = b.build_iconst_i64(0);
        return b.build_call("bronze_prop_get", Type::i64(), {obj, sym_val, null_entry});
    }

    Module* mod = b.current_block()->parent()->parent();
    const char* interned = mod->string_pool().intern(prop_name).data();
    Value* name_val = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(interned)));
    Value* sym_val = b.build_iconst_i32(static_cast<int32_t>(symbol_id));

    if (enable_pic_) {
        uint32_t sid = (site_id != 0) ? site_id : g_global_auto_site_id.fetch_add(1);
        Value* sid_val = b.build_iconst_i32(static_cast<int32_t>(sid));
        return b.build_call("brass_ic_get_prop", Type::i64(), {sid_val, obj, name_val, sym_val});
    }

    return b.build_call("brass_dynamic_object_get_prop_str", Type::i64(), {obj, name_val});
}

void PropertyLoweringHelper::lower_prop_set(
    Builder& b,
    Value* obj,
    std::string_view prop_name,
    uint32_t symbol_id,
    Value* val,
    uint32_t slot_idx,
    uint32_t imm,
    uint32_t site_id
) {
    if (prop_name.empty()) {
        Value* sym_val = nullptr;
        if (symbol_id == 0xFFFFFFFFu) {
            sym_val = b.build_iconst_i32(static_cast<int32_t>(symbol_id));
        } else {
            Value* map_addr = b.build_func_addr("__bronze_key_map");
            sym_val = b.build_load(Type::i32(), map_addr, static_cast<int32_t>(symbol_id * sizeof(uint32_t)));
        }
        Value* slot_val = b.build_iconst_i64(static_cast<int64_t>(slot_idx));
        Value* strict_val = b.build_iconst_i32(imm != 0 ? 1 : 0);
        b.build_call("bronze_prop_set", Type::void_type(), {obj, sym_val, val, slot_val, strict_val});
        return;
    }

    Module* mod = b.current_block()->parent()->parent();
    const char* interned = mod->string_pool().intern(prop_name).data();
    Value* name_val = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(interned)));
    Value* sym_val = b.build_iconst_i32(static_cast<int32_t>(symbol_id));

    if (enable_pic_) {
        uint32_t sid = (site_id != 0) ? site_id : g_global_auto_site_id.fetch_add(1);
        Value* sid_val = b.build_iconst_i32(static_cast<int32_t>(sid));
        b.build_call("brass_ic_set_prop", Type::void_type(), {sid_val, obj, name_val, sym_val, val});
        return;
    }

    b.build_call("brass_dynamic_object_set_prop_str", Type::void_type(), {obj, name_val, val});
}

static void extract_object_pointer(Builder& b, Value* obj, Value*& obj_ptr, Value*& is_valid_obj) {
    Value* tag = b.build_and(obj, b.build_iconst_i64(static_cast<int64_t>(0xFFFF000000000000ULL)));
    Value* is_gcref = b.build_eq(tag, b.build_iconst_i64(static_cast<int64_t>(0x7FFD000000000000ULL)));
    Value* is_zero_tag = b.build_eq(tag, b.build_iconst_i64(0));
    Value* is_non_null = b.build_ne(obj, b.build_iconst_i64(0));
    Value* is_raw = b.build_and(is_zero_tag, is_non_null);
    is_valid_obj = b.build_or(is_gcref, is_raw);
    obj_ptr = b.build_and(obj, b.build_iconst_i64(static_cast<int64_t>(0x0000FFFFFFFFFFFFULL)));
}

static Value* extract_integer_index(Builder& b, Value* index, Value*& fallback_idx, Value*& is_valid_idx) {
    Type idx_type = index->type();
    if (idx_type == Type::i32()) {
        Value* idx_i64 = b.build_sext_i64(index);
        fallback_idx = idx_i64;
        is_valid_idx = b.build_iconst_i32(1);
        return idx_i64;
    }
    if (idx_type == Type::f64()) {
        fallback_idx = b.build_bitcast_i64_f64(index);
        Value* idx_i64 = b.build_fptosi_i64(index);
        Value* back_f = b.build_sitofp_f64_i64(idx_i64);
        is_valid_idx = b.build_eq(back_f, index);
        return idx_i64;
    }

    fallback_idx = index;

    Value* f64_source = nullptr;
    if (index->defining_instruction()) {
        Instruction* def = index->defining_instruction();
        if (def->opcode() == Opcode::bitcast_i64_f64 && def->operand(0)->type() == Type::f64()) {
            f64_source = def->operand(0);
        } else if (def->opcode() == Opcode::select && def->operand_count() >= 3) {
            Value* false_val = def->operand(2);
            if (false_val->defining_instruction() &&
                false_val->defining_instruction()->opcode() == Opcode::bitcast_i64_f64 &&
                false_val->defining_instruction()->operand(0)->type() == Type::f64()) {
                f64_source = false_val->defining_instruction()->operand(0);
            }
        }
    }

    if (f64_source) {
        Value* idx_i64 = b.build_fptosi_i64(f64_source);
        Value* back_f = b.build_sitofp_f64_i64(idx_i64);
        is_valid_idx = b.build_eq(back_f, f64_source);
        return idx_i64;
    }

    Value* u_tag = b.build_lshr(index, b.build_iconst_i64(48));
    Value* is_tag_fff3 = b.build_eq(u_tag, b.build_iconst_i64(static_cast<int64_t>(0xFFF3LL)));
    Value* is_tag_7ff9 = b.build_eq(u_tag, b.build_iconst_i64(static_cast<int64_t>(0x7FF9LL)));
    Value* is_b_i32 = b.build_or(is_tag_fff3, is_tag_7ff9);

    Value* ge_min_f64 = b.build_uge(index, b.build_iconst_i64(static_cast<int64_t>(0x3FF0000000000000ULL)));
    Value* le_max_f64 = b.build_ule(index, b.build_iconst_i64(static_cast<int64_t>(0xFFF0000000000000ULL)));
    Value* is_b_f64 = b.build_and(ge_min_f64, le_max_f64);

    Value* trunc_val = b.build_trunc_i32(index);
    Value* i32_ext = b.build_sext_i64(trunc_val);

    Value* f_from_bits = b.build_bitcast_f64_i64(index);
    Value* f_to_i64 = b.build_fptosi_i64(f_from_bits);
    Value* f_back = b.build_sitofp_f64_i64(f_to_i64);
    Value* f_exact = b.build_eq(f_back, f_from_bits);

    Value* boxed_val = b.build_select(is_b_f64, f_to_i64, i32_ext);
    Value* is_boxed = b.build_or(is_b_f64, is_b_i32);
    Value* final_idx = b.build_select(is_boxed, boxed_val, index);

    Value* boxed_ok = b.build_or(is_b_i32, f_exact);
    is_valid_idx = b.build_select(is_b_f64, boxed_ok, b.build_iconst_i32(1));
    return final_idx;
}

Value* PropertyLoweringHelper::lower_prop_get(
    Builder& b,
    Value* obj,
    uint32_t slot_idx
) {
    return lower_prop_get_slot(b, obj, slot_idx);
}

Value* PropertyLoweringHelper::lower_prop_get_slot(
    Builder& b,
    Value* obj,
    uint32_t slot_idx
) {
    Value* obj_ptr = b.build_and(obj, b.build_iconst_i64(static_cast<int64_t>(0x0000FFFFFFFFFFFFULL)));
    return b.build_load(Type::i64(), obj_ptr, static_cast<int32_t>(24 + slot_idx * 8));
}

Value* PropertyLoweringHelper::lower_prop_get_guarded(
    Builder& b,
    Value* obj,
    const void* expected_shape,
    uint32_t slot_idx,
    std::string_view prop_name,
    uint32_t symbol_id,
    uint32_t site_id
) {
    if (!enable_inlined_fastpaths_ || slot_idx >= runtime::DynamicObject::DEFAULT_INLINE_SLOTS || !expected_shape) {
        return lower_prop_get(b, obj, prop_name, symbol_id, site_id);
    }

    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "prop_get_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    Value* obj_ptr = nullptr;
    Value* is_valid_obj = nullptr;
    extract_object_pointer(b, obj, obj_ptr, is_valid_obj);

    Value* shape_ptr = b.build_load(Type::i64(), obj_ptr, 0);
    Value* shape_match = b.build_eq(shape_ptr, b.build_iconst_i64(reinterpret_cast<int64_t>(expected_shape)));
    Value* guard = b.build_and(is_valid_obj, shape_match);
    b.build_br_if(guard, bb_fast, bb_fallback);

    b.position_at_end(bb_fast);
    Value* fast_val = lower_prop_get_slot(b, obj_ptr, slot_idx);
    b.build_br(bb_merge, {fast_val});

    b.position_at_end(bb_fallback);
    Value* fallback_val = lower_prop_get(b, obj, prop_name, symbol_id, site_id);
    b.build_br(bb_merge, {fallback_val});

    b.position_at_end(bb_merge);
    return merge_val;
}

void PropertyLoweringHelper::lower_prop_set(
    Builder& b,
    Value* obj,
    uint32_t slot_idx,
    Value* val
) {
    lower_prop_set_slot(b, obj, slot_idx, val);
}

void PropertyLoweringHelper::lower_prop_set_slot(
    Builder& b,
    Value* obj,
    uint32_t slot_idx,
    Value* val
) {
    Value* obj_ptr = b.build_and(obj, b.build_iconst_i64(static_cast<int64_t>(0x0000FFFFFFFFFFFFULL)));
    b.build_store(Type::i64(), obj_ptr, static_cast<int32_t>(24 + slot_idx * 8), val);
}

void PropertyLoweringHelper::lower_prop_set_guarded(
    Builder& b,
    Value* obj,
    const void* expected_shape,
    uint32_t slot_idx,
    Value* val,
    std::string_view prop_name,
    uint32_t symbol_id,
    uint32_t site_id
) {
    if (!enable_inlined_fastpaths_ || slot_idx >= runtime::DynamicObject::DEFAULT_INLINE_SLOTS || !expected_shape) {
        lower_prop_set(b, obj, prop_name, symbol_id, val, slot_idx, 0, site_id);
        return;
    }

    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "prop_set_" + std::to_string(bid);

    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");

    b.position_at_end(bb_current);

    Value* obj_ptr = nullptr;
    Value* is_valid_obj = nullptr;
    extract_object_pointer(b, obj, obj_ptr, is_valid_obj);

    Value* shape_ptr = b.build_load(Type::i64(), obj_ptr, 0);
    Value* shape_match = b.build_eq(shape_ptr, b.build_iconst_i64(reinterpret_cast<int64_t>(expected_shape)));
    Value* guard = b.build_and(is_valid_obj, shape_match);
    b.build_br_if(guard, bb_fast, bb_fallback);

    b.position_at_end(bb_fast);
    lower_prop_set_slot(b, obj_ptr, slot_idx, val);
    b.build_br(bb_merge);

    b.position_at_end(bb_fallback);
    lower_prop_set(b, obj, prop_name, symbol_id, val, slot_idx, 0, site_id);
    b.build_br(bb_merge);

    b.position_at_end(bb_merge);
}

Value* PropertyLoweringHelper::lower_elem_get(
    Builder& b,
    Value* obj,
    Value* index
) {
    if (!enable_inlined_fastpaths_) {
        Value* idx_i64 = index;
        if (index->type() == Type::i32()) {
            idx_i64 = b.build_sext_i64(index);
        } else if (index->type() == Type::f64()) {
            idx_i64 = b.build_bitcast_i64_f64(index);
        }
        return b.build_call("bronze_elem_get", Type::i64(), {obj, idx_i64});
    }

    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "elem_get_" + std::to_string(bid);

    BasicBlock* bb_check = b.append_block(prefix + "_check");
    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");
    Value* merge_val = b.add_block_param(bb_merge, Type::i64());

    b.position_at_end(bb_current);

    Value* obj_ptr = nullptr;
    Value* is_valid_obj = nullptr;
    extract_object_pointer(b, obj, obj_ptr, is_valid_obj);

    Value* fallback_idx = nullptr;
    Value* is_valid_idx = nullptr;
    Value* idx_i64 = extract_integer_index(b, index, fallback_idx, is_valid_idx);

    Value* initial_guard = b.build_and(is_valid_obj, is_valid_idx);
    b.build_br_if(initial_guard, bb_check, bb_fallback);

    b.position_at_end(bb_check);
    Value* count_i32 = b.build_load(Type::i32(), obj_ptr, 88);
    Value* count_i64 = b.build_zext_i64(count_i32);
    Value* in_bounds = b.build_ult(idx_i64, count_i64);

    Value* elements = b.build_load(Type::i64(), obj_ptr, 96);
    Value* elem_not_null = b.build_ne(elements, b.build_iconst_i64(0));
    Value* fast_ok = b.build_and(in_bounds, elem_not_null);
    b.build_br_if(fast_ok, bb_fast, bb_fallback);

    b.position_at_end(bb_fast);
    Value* fast_val = b.build_load_indexed(Type::i64(), elements, idx_i64, 8, 8);
    b.build_br(bb_merge, {fast_val});

    b.position_at_end(bb_fallback);
    Value* fallback_val = b.build_call("bronze_elem_get", Type::i64(), {obj, fallback_idx});
    b.build_br(bb_merge, {fallback_val});

    b.position_at_end(bb_merge);
    return merge_val;
}

void PropertyLoweringHelper::lower_elem_set(
    Builder& b,
    Value* obj,
    Value* index,
    Value* val,
    uint32_t ic_slot
) {
    if (!enable_inlined_fastpaths_) {
        Value* idx_i64 = index;
        if (index->type() == Type::i32()) {
            idx_i64 = b.build_sext_i64(index);
        } else if (index->type() == Type::f64()) {
            idx_i64 = b.build_bitcast_i64_f64(index);
        }
        Value* ic_val = b.build_iconst_i32(static_cast<int32_t>(ic_slot));
        b.build_call("bronze_elem_set", Type::void_type(), {obj, idx_i64, val, ic_val});
        return;
    }

    BasicBlock* bb_current = b.current_block();
    Function* fn = bb_current->parent();
    uint32_t bid = fn->next_block_id();
    std::string prefix = "elem_set_" + std::to_string(bid);

    BasicBlock* bb_check = b.append_block(prefix + "_check");
    BasicBlock* bb_fast = b.append_block(prefix + "_fast");
    BasicBlock* bb_fallback = b.append_block(prefix + "_fallback");
    BasicBlock* bb_merge = b.append_block(prefix + "_merge");

    b.position_at_end(bb_current);

    Value* obj_ptr = nullptr;
    Value* is_valid_obj = nullptr;
    extract_object_pointer(b, obj, obj_ptr, is_valid_obj);

    Value* fallback_idx = nullptr;
    Value* is_valid_idx = nullptr;
    Value* idx_i64 = extract_integer_index(b, index, fallback_idx, is_valid_idx);

    Value* initial_guard = b.build_and(is_valid_obj, is_valid_idx);
    b.build_br_if(initial_guard, bb_check, bb_fallback);

    b.position_at_end(bb_check);
    Value* count_i32 = b.build_load(Type::i32(), obj_ptr, 88);
    Value* count_i64 = b.build_zext_i64(count_i32);
    Value* in_bounds = b.build_ult(idx_i64, count_i64);

    Value* elements = b.build_load(Type::i64(), obj_ptr, 96);
    Value* elem_not_null = b.build_ne(elements, b.build_iconst_i64(0));
    Value* fast_ok = b.build_and(in_bounds, elem_not_null);
    b.build_br_if(fast_ok, bb_fast, bb_fallback);

    b.position_at_end(bb_fast);
    b.build_store_indexed(Type::i64(), elements, idx_i64, 8, 8, val);
    b.build_br(bb_merge);

    b.position_at_end(bb_fallback);
    Value* ic_val = b.build_iconst_i32(static_cast<int32_t>(ic_slot));
    b.build_call("bronze_elem_set", Type::void_type(), {obj, fallback_idx, val, ic_val});
    b.build_br(bb_merge);

    b.position_at_end(bb_merge);
}

void PropertyLoweringHelper::lower_method_def(
    Builder& b,
    Value* obj,
    std::string_view prop_name,
    uint32_t symbol_id,
    Value* closure
) {
    if (prop_name.empty()) {
        Value* sym_val = nullptr;
        if (symbol_id == 0xFFFFFFFFu) {
            sym_val = b.build_iconst_i32(static_cast<int32_t>(symbol_id));
        } else {
            Value* map_addr = b.build_func_addr("__bronze_key_map");
            sym_val = b.build_load(Type::i32(), map_addr, static_cast<int32_t>(symbol_id * sizeof(uint32_t)));
        }
        b.build_call("bronze_method_def", Type::void_type(), {obj, sym_val, closure});
    } else if (symbol_id != 0) {
        Value* sym_val = b.build_iconst_i32(static_cast<int32_t>(symbol_id));
        b.build_call("bronze_method_def", Type::void_type(), {obj, sym_val, closure});
    } else {
        Module* mod = b.current_block()->parent()->parent();
        const char* interned = mod->string_pool().intern(prop_name).data();
        Value* name_val = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(interned)));
        b.build_call("brass_dynamic_object_set_prop_str", Type::void_type(), {obj, name_val, closure});
    }
}

bool is_property_il_op(BronzeOp op) {
    switch (op) {
        case BronzeOp::PropGet:
        case BronzeOp::PropSet:
        case BronzeOp::PropDelete:
        case BronzeOp::MethodDef:
        case BronzeOp::MethodDefComputed:
        case BronzeOp::DefineOwnAttr:
        case BronzeOp::AccessorDef:
        case BronzeOp::AccessorDefComputed:
        case BronzeOp::ElemGet:
        case BronzeOp::ElemGetTyped:
        case BronzeOp::ElemSet:
        case BronzeOp::ElemSetTyped:
        case BronzeOp::ElemDelete:
            return true;
        default:
            return false;
    }
}

bool lower_property_instruction(
    IlLowering* lowering,
    const BronzeInstruction& inst_ast,
    Builder& b,
    Function* fn,
    std::unordered_map<uint32_t, Value*>& val_map,
    Value*& res_val,
    const std::function<void()>& emit_exception_check
) {
    auto get_opd = [&](size_t idx) -> Value* {
        if (idx < inst_ast.operands.size()) {
            uint32_t id = inst_ast.operands[idx];
            if (lowering) {
                return lowering->get_val_by_id(id, b, val_map);
            }
            if (val_map.count(id)) return val_map[id];
        }
        return nullptr;
    };

    auto ensure_type = [&](Value* val, Type target_type) -> Value* {
        if (lowering) {
            return lowering->ensure_type(val, target_type, b);
        }
        return val;
    };

    auto get_key_id = [&](uint32_t key_idx) -> Value* {
        if (lowering) {
            return lowering->get_key_id(b, key_idx);
        }
        return b.build_iconst_i32(static_cast<int32_t>(key_idx));
    };

    PropertyLoweringHelper& prop_lowering = lowering->prop_lowering();

    switch (inst_ast.op) {
        case BronzeOp::PropGet: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64());
            res_val = prop_lowering.lower_prop_get(
                b, obj_val, inst_ast.string_literal, inst_ast.index, inst_ast.depth
            );
            emit_exception_check();
            return true;
        }

        case BronzeOp::PropSet: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64());
            Value* val = ensure_type(get_opd(1), Type::i64());
            prop_lowering.lower_prop_set(
                b, obj_val, inst_ast.string_literal, inst_ast.index, val,
                inst_ast.depth, static_cast<uint32_t>(inst_ast.imm_i64), 0
            );
            b.build_write_barrier(obj_val, val);
            emit_exception_check();
            return true;
        }

        case BronzeOp::PropDelete: {
            Value* target = ensure_type(get_opd(0), Type::i64());
            Value* key_id = get_key_id(inst_ast.index);
            Value* strict = b.build_iconst_i32(inst_ast.imm_i64 != 0 ? 1 : 0);
            res_val = b.build_and(b.build_call("bronze_prop_delete", Type::i32(), {target, key_id, strict}), b.build_iconst_i32(1));
            emit_exception_check();
            return true;
        }

        case BronzeOp::MethodDef: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64());
            Value* closure_val = ensure_type(get_opd(1), Type::i64());
            prop_lowering.lower_method_def(
                b, obj_val, inst_ast.string_literal, inst_ast.index, closure_val
            );
            b.build_write_barrier(obj_val, closure_val);
            return true;
        }

        case BronzeOp::MethodDefComputed: {
            Value* target = ensure_type(get_opd(0), Type::i64());
            Value* key = ensure_type(get_opd(1), Type::i64());
            Value* closure_val = ensure_type(get_opd(2), Type::i64());
            b.build_call("bronze_method_def_computed", Type::void_type(), {target, key, closure_val});
            b.build_write_barrier(target, closure_val);
            return true;
        }

        case BronzeOp::DefineOwnAttr: {
            Value* target = ensure_type(get_opd(0), Type::i64());
            Value* value = ensure_type(get_opd(1), Type::i64());
            Value* key_id = get_key_id(inst_ast.index);
            Value* mask = b.build_iconst_i32(static_cast<int32_t>(inst_ast.imm_i64));
            b.build_call("bronze_define_own_attr", Type::void_type(), {target, key_id, value, mask});
            emit_exception_check();
            return true;
        }

        case BronzeOp::AccessorDef: {
            Value* target = ensure_type(get_opd(0), Type::i64());
            Value* key_id = get_key_id(inst_ast.index);
            Value* getter = ensure_type(get_opd(1), Type::i64());
            Value* setter = ensure_type(get_opd(2), Type::i64());
            Value* enum_val = b.build_iconst_i32(inst_ast.imm_bool ? 1 : 0);
            b.build_call("bronze_accessor_def", Type::void_type(), {target, key_id, getter, setter, enum_val});
            emit_exception_check();
            return true;
        }

        case BronzeOp::AccessorDefComputed: {
            Value* target = ensure_type(get_opd(0), Type::i64());
            Value* key = ensure_type(get_opd(1), Type::i64());
            Value* getter = ensure_type(get_opd(2), Type::i64());
            Value* setter = ensure_type(get_opd(3), Type::i64());
            Value* enum_val = b.build_iconst_i32(inst_ast.imm_bool ? 1 : 0);
            b.build_call("bronze_accessor_def_computed", Type::void_type(), {target, key, getter, setter, enum_val});
            emit_exception_check();
            return true;
        }

        case BronzeOp::ElemGet:
        case BronzeOp::ElemGetTyped: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64());
            Value* idx_val = get_opd(1);
            if (!idx_val) return false;
            res_val = prop_lowering.lower_elem_get(b, obj_val, idx_val);
            if (inst_ast.op == BronzeOp::ElemGet) {
                emit_exception_check();
            } else if (inst_ast.op == BronzeOp::ElemGetTyped && inst_ast.result_type == BronzeType::F64) {
                res_val = ensure_type(res_val, Type::f64());
            }
            return true;
        }

        case BronzeOp::ElemSet:
        case BronzeOp::ElemSetTyped: {
            Value* obj_val = ensure_type(get_opd(0), Type::i64());
            Value* idx_val = get_opd(1);
            if (!idx_val) return false;
            Value* val = ensure_type(get_opd(2), Type::i64());
            prop_lowering.lower_elem_set(b, obj_val, idx_val, val, inst_ast.index);
            b.build_write_barrier(obj_val, val);
            if (inst_ast.op == BronzeOp::ElemSet) {
                emit_exception_check();
            }
            return true;
        }

        case BronzeOp::ElemDelete: {
            Value* target = ensure_type(get_opd(0), Type::i64());
            Value* index = ensure_type(get_opd(1), Type::i64());
            Value* strict = b.build_iconst_i32(inst_ast.imm_i64 != 0 ? 1 : 0);
            res_val = b.build_and(b.build_call("bronze_elem_delete", Type::i32(), {target, index, strict}), b.build_iconst_i32(1));
            emit_exception_check();
            return true;
        }

        default:
            return false;
    }
}

} // namespace brass::il
