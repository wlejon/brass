#include <brass/runtime/inline_cache.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/patcher.hpp>
#include <brass/embedding/host_gc.hpp>

namespace brass::runtime {

bool patch_monomorphic_ic(
    InlineCache& ic,
    const Shape* shape,
    uint32_t slot,
    PatchRegistry* patch_registry
) {
    if (!shape) {
        return false;
    }

    std::optional<uint32_t> prop_slot;
    if (!ic.prop_name().empty()) {
        prop_slot = shape->find_slot(ic.prop_name());
    } else if (ic.symbol_id() != 0) {
        prop_slot = shape->find_slot(ic.symbol_id());
    }
    if (prop_slot.has_value()) {
        slot = *prop_slot;
    } else if (!ic.prop_name().empty() || ic.symbol_id() != 0) {
        return false;
    }

    bool success = true;

    // 1. If direct patch point address is set for cached shape comparison
    if (ic.patch_point() != nullptr && shape != nullptr) {
        success = brass_patch_const64(ic.patch_point(), reinterpret_cast<int64_t>(shape)) && success;
    }

    // 2. If direct patch point address is set for property slot displacement
    if (ic.slot_patch_point() != nullptr) {
        int32_t disp = static_cast<int32_t>(offsetof(DynamicObject, inline_slots) + slot * sizeof(uint64_t));
        success = brass_patch_const32(ic.slot_patch_point(), disp) && success;
    }

    // 3. If registered in a PatchRegistry by name
    if (patch_registry != nullptr && !ic.patch_site_name().empty()) {
        if (shape != nullptr) {
            std::string shape_site = std::string(ic.patch_site_name()) + "_shape";
            if (patch_registry->has_site(shape_site)) {
                success = patch_registry->patch_const64(nullptr, shape_site, reinterpret_cast<int64_t>(shape)) && success;
            }
        }
        std::string slot_site = std::string(ic.patch_site_name()) + "_slot";
        if (patch_registry->has_site(slot_site)) {
            int32_t disp = static_cast<int32_t>(offsetof(DynamicObject, inline_slots) + slot * sizeof(uint64_t));
            success = patch_registry->patch_const32(nullptr, slot_site, disp) && success;
        }
    }

    return success;
}

bool patch_ic_call_target(InlineCache& ic, const void* new_stub_target) {
    if (ic.call_site_address() != nullptr && new_stub_target != nullptr) {
        return brass_patch_call(ic.call_site_address(), new_stub_target);
    }
    return false;
}

} // namespace brass::runtime

extern "C" {

using namespace brass;
using namespace brass::runtime;

static inline DynamicObject* unpack_obj(uint64_t obj_raw) {
    if (obj_raw == 0) return nullptr;
    HostValue hv(obj_raw);
    if (hv.is_gcref()) {
        return hv.as_gcref_ptr<DynamicObject>();
    }
    if (obj_raw < 0x0000800000000000ULL && obj_raw >= 0x1000ULL) {
        return reinterpret_cast<DynamicObject*>(obj_raw);
    }
    return nullptr;
}

uint64_t brass_ic_get_prop(uint32_t site_id, uint64_t obj_raw, const char* name, uint32_t symbol_id) {
    auto* obj = unpack_obj(obj_raw);
    if (!obj) {
        return HostValue::undefined_val().raw();
    }

    std::string_view prop_name = (name != nullptr) ? name : "";
    InlineCache* ic = ICRegistry::global().get_or_create_ic(site_id, prop_name, symbol_id, /*is_load=*/true);
    return ic->execute_get(obj).raw();
}

void brass_ic_set_prop(uint32_t site_id, uint64_t obj_raw, const char* name, uint32_t symbol_id, uint64_t val_raw) {
    auto* obj = unpack_obj(obj_raw);
    if (!obj) return;

    std::string_view prop_name = (name != nullptr) ? name : "";
    InlineCache* ic = ICRegistry::global().get_or_create_ic(site_id, prop_name, symbol_id, /*is_load=*/false);
    HostGC* gc = get_active_host_gc();
    ic->execute_set(obj, HostValue(val_raw), ShapeRegistry::global(), gc);
}

uint64_t brass_ic_miss_handler_get(uint32_t site_id, uint64_t obj_raw) {
    auto* obj = unpack_obj(obj_raw);
    if (!obj) return HostValue::undefined_val().raw();

    InlineCache* ic = ICRegistry::global().find_ic(site_id);
    if (!ic) return HostValue::undefined_val().raw();
    return ic->miss_handler_get(obj).raw();
}

void brass_ic_miss_handler_set(uint32_t site_id, uint64_t obj_raw, uint64_t val_raw) {
    auto* obj = unpack_obj(obj_raw);
    if (!obj) return;

    InlineCache* ic = ICRegistry::global().find_ic(site_id);
    if (!ic) return;
    HostGC* gc = get_active_host_gc();
    ic->miss_handler_set(obj, HostValue(val_raw), ShapeRegistry::global(), gc);
}

} // extern "C"
