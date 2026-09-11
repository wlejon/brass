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
    Module* mod = b.current_block()->parent()->parent();

    Value* name_val = nullptr;
    if (!prop_name.empty()) {
        const char* interned = mod->string_pool().intern(prop_name).data();
        name_val = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(interned)));
    } else {
        name_val = b.build_iconst_i64(0);
    }

    Value* sym_val = b.build_iconst_i32(static_cast<int32_t>(symbol_id));

    if (symbol_id != 0 && prop_name.empty()) {
        Value* null_entry = b.build_iconst_i64(0);
        return b.build_call("bronze_prop_get", Type::i64(), {obj, sym_val, null_entry});
    }

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
    Module* mod = b.current_block()->parent()->parent();

    Value* name_val = nullptr;
    if (!prop_name.empty()) {
        const char* interned = mod->string_pool().intern(prop_name).data();
        name_val = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(interned)));
    } else {
        name_val = b.build_iconst_i64(0);
    }

    Value* sym_val = b.build_iconst_i32(static_cast<int32_t>(symbol_id));

    if (symbol_id != 0 && prop_name.empty()) {
        Value* slot_val = b.build_iconst_i64(static_cast<int64_t>(slot_idx));
        Value* strict_val = b.build_iconst_i32(1);
        b.build_call("bronze_prop_set", Type::void_type(), {obj, sym_val, val, slot_val, strict_val});
        return;
    }

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
    if (prop_name.empty() || symbol_id != 0) {
        Value* sym_val = b.build_iconst_i32(static_cast<int32_t>(symbol_id));
        b.build_call("bronze_method_def", Type::void_type(), {obj, sym_val, closure});
    } else {
        Module* mod = b.current_block()->parent()->parent();
        const char* interned = mod->string_pool().intern(prop_name).data();
        Value* name_val = b.build_iconst_i64(static_cast<int64_t>(reinterpret_cast<uintptr_t>(interned)));
        b.build_call("brass_dynamic_object_set_prop_str", Type::void_type(), {obj, name_val, closure});
    }
}

} // namespace brass::il
