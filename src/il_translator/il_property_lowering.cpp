#include "il_property_lowering.hpp"
#include "il_lowering.hpp"
#include <brass/il_translator/il_property.hpp>
#include <brass/mir/function.hpp>
#include <brass/mir/module.hpp>
#include <brass/core/string_pool.hpp>

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

Value* PropertyLoweringHelper::lower_elem_get(
    Builder& b,
    Value* obj,
    Value* index
) {
    return b.build_call("bronze_elem_get", Type::i64(), {obj, index});
}

void PropertyLoweringHelper::lower_elem_set(
    Builder& b,
    Value* obj,
    Value* index,
    Value* val,
    uint32_t ic_slot
) {
    Value* ic_val = b.build_iconst_i32(static_cast<int32_t>(ic_slot));
    b.build_call("bronze_elem_set", Type::void_type(), {obj, index, val, ic_val});
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
            Value* idx_val = ensure_type(get_opd(1), Type::i64());
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
            Value* idx_val = ensure_type(get_opd(1), Type::i64());
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
