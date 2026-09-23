#include "il_runtime.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/gc/tlab.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/inline_cache.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/runtime/code_installer.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <charconv>
#include <system_error>
#include <vector>
#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace brass::il {

using namespace brass::runtime;

#ifndef _WIN32
#define BRONZE_WEAK __attribute__((weak))
#define BRONZE_WEAK_DATA __attribute__((weak))
#else
#define BRONZE_WEAK
#define BRONZE_WEAK_DATA __declspec(selectany)
#endif

struct BronzeEnv {
    int64_t parent_box;
    uint32_t size;
    int64_t slots[1];
};

static codegen::JitExecutionEngine* g_active_jit = nullptr;
static void* (*g_custom_fn_resolver)(const char*) = nullptr;

void set_bronze_function_resolver(void* (*resolver)(const char*)) {
    g_custom_fn_resolver = resolver;
}

void* bronze_resolve_function(const char* name) {
    if (!name) return nullptr;
    if (g_custom_fn_resolver) {
        void* ptr = g_custom_fn_resolver(name);
        if (ptr) return ptr;
    }
    if (g_active_jit) {
        void* ptr = g_active_jit->get_symbol_address(name);
        if (ptr) return ptr;
    }
    // The running program's handles (the default program's outside any
    // ProgramScope).
    auto* handle = runtime::current_program().find(name);
    if (handle) {
        return handle->native_entry();
    }
    return nullptr;
}

#ifndef _WIN32
__attribute__((weak))
#endif
int64_t bronze_dynamic_add(int64_t a, int64_t b) {
    if ((static_cast<uint64_t>(a) >> 48) <= 0xFFF8 && (static_cast<uint64_t>(b) >> 48) <= 0xFFF8) {
        double da = std::bit_cast<double>(a);
        double db = std::bit_cast<double>(b);
        return std::bit_cast<int64_t>(da + db);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

double bronze_f64_mod(double a, double b) {
    if (b == 0.0) return std::numeric_limits<double>::quiet_NaN();
    if (std::trunc(a) == a && std::trunc(b) == b && std::abs(a) < 9007199254740992.0 && std::abs(b) < 9007199254740992.0) {
        int64_t ia = static_cast<int64_t>(a);
        int64_t ib = static_cast<int64_t>(b);
        if (ib != 0) {
            return static_cast<double>(ia % ib);
        }
    }
    return std::fmod(a, b);
}

int64_t bronze_name_resolve(const char* name) {
    if (!name) return static_cast<int64_t>(kUndefinedTag);
    if (std::strcmp(name, "print") == 0 || std::strcmp(name, "console.log") == 0) {
        return static_cast<int64_t>(kPrintTag);
    }
    if (std::strcmp(name, "print.err") == 0) {
        return static_cast<int64_t>(kPrintErrTag);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_env_create(int64_t parent_box, int32_t size) {
    size_t alloc_size = sizeof(BronzeEnv) + (size > 1 ? sizeof(int64_t) * (size - 1) : 0);
    uint64_t pointer_mask = 1ULL;
    for (int32_t i = 0; i < size; ++i) {
        if (2 + i < 64) {
            pointer_mask |= (1ULL << (2 + i));
        }
    }
    auto* tlab = brass::get_active_tlab();
    if (tlab && tlab->owner_gc && !tlab->owner_gc->stress_mode()) {
        tlab->refill(alloc_size + sizeof(HostGcHeader));
        uintptr_t addr = tlab->allocate_fast(alloc_size, pointer_mask, 1 /* type_tag */);
        if (addr != 0) {
            auto* env = reinterpret_cast<BronzeEnv*>(addr);
            env->parent_box = parent_box;
            env->size = size > 0 ? static_cast<uint32_t>(size) : 0;
            for (int32_t i = 0; i < size; ++i) {
                env->slots[i] = static_cast<int64_t>(kUndefinedTag);
            }
            return reinterpret_cast<int64_t>(env);
        }
    }
    BronzeEnv* env = nullptr;
    HostGC* host_gc = brass::get_active_host_gc();
    if (host_gc) {
        // Bit 0 is parent_box; bits 2+i are slots
        uintptr_t addr = host_gc->allocate(alloc_size, pointer_mask, 1 /* type_tag */);
        env = reinterpret_cast<BronzeEnv*>(addr);
    } else {
        env = reinterpret_cast<BronzeEnv*>(std::malloc(alloc_size));
    }
    if (!env) return static_cast<int64_t>(kUndefinedTag);
    env->parent_box = parent_box;
    env->size = size > 0 ? static_cast<uint32_t>(size) : 0;
    for (int32_t i = 0; i < size; ++i) {
        env->slots[i] = static_cast<int64_t>(kUndefinedTag);
    }
    return reinterpret_cast<int64_t>(env);
}

static inline BronzeEnv* unpack_env(int64_t env_box) {
    if (!env_box) return nullptr;
    uint64_t u = static_cast<uint64_t>(env_box);
    if ((u & HostValue::TAG_MASK) == HostValue::TAG_GCREF) {
        return reinterpret_cast<BronzeEnv*>(u & HostValue::PAYLOAD_MASK);
    }
    if (u < 0x0000800000000000ULL && u >= 0x1000ULL) {
        return reinterpret_cast<BronzeEnv*>(env_box);
    }
    return nullptr;
}

int64_t bronze_env_get(int64_t env_box, int32_t depth, int32_t index) {
    if (!env_box || env_box == static_cast<int64_t>(kUndefinedTag) || env_box == static_cast<int64_t>(kNullTag)) {
        return static_cast<int64_t>(kUndefinedTag);
    }
    if (is_active_coro_frame(static_cast<uintptr_t>(env_box))) {
        auto* frame = reinterpret_cast<const BrassCoroFrame*>(env_box);
        if (static_cast<uint32_t>(index) < frame->slot_count) {
            return static_cast<int64_t>(frame->slots[index]);
        }
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* cur = unpack_env(env_box);
    for (int32_t d = 0; d < depth && cur; ++d) {
        cur = unpack_env(cur->parent_box);
    }
    if (!cur || static_cast<uint32_t>(index) >= cur->size) {
        return static_cast<int64_t>(kUndefinedTag);
    }
    return cur->slots[index];
}

BRONZE_WEAK uint64_t bronze_env_get_tdz(uint64_t env_bits, uint32_t depth, uint32_t index, uint32_t key_index) {
    return static_cast<uint64_t>(bronze_env_get(static_cast<int64_t>(env_bits), static_cast<int32_t>(depth), static_cast<int32_t>(index)));
}

void bronze_env_set(int64_t env_box, int32_t depth, int32_t index, int64_t val) {
    if (!env_box || env_box == static_cast<int64_t>(kUndefinedTag) || env_box == static_cast<int64_t>(kNullTag)) return;
    auto* cur = unpack_env(env_box);
    for (int32_t d = 0; d < depth && cur; ++d) {
        cur = unpack_env(cur->parent_box);
    }
    if (cur && static_cast<uint32_t>(index) < cur->size) {
        cur->slots[index] = val;
    }
}

BRONZE_WEAK uint64_t bronze_env_ancestor(uint64_t env_bits, uint32_t depth) {
    if (depth == 0) return env_bits;
    int64_t cur_box = static_cast<int64_t>(env_bits);
    for (uint32_t d = 0; d < depth; ++d) {
        auto* cur = unpack_env(cur_box);
        if (!cur) break;
        cur_box = cur->parent_box;
    }
    return static_cast<uint64_t>(cur_box);
}

int64_t bronze_create_func(void* code_ptr, int32_t param_count, int64_t env_box) {
    BronzeClosure* closure = nullptr;
    HostGC* host_gc = brass::get_active_host_gc();
    if (host_gc) {
        // Field 0: fn_name/code_ptr, Field 1: env_box (pointer_mask bit 1)
        uint64_t pointer_mask = (1ULL << 1);
        uintptr_t addr = host_gc->allocate(sizeof(BronzeClosure), pointer_mask, 2 /* type_tag */);
        closure = reinterpret_cast<BronzeClosure*>(addr);
    } else {
        closure = reinterpret_cast<BronzeClosure*>(std::malloc(sizeof(BronzeClosure)));
    }
    if (!closure) return static_cast<int64_t>(kUndefinedTag);
    std::memset(closure->fn_name, 0, sizeof(closure->fn_name));
    closure->code_ptr = code_ptr;
    closure->env_box = env_box;
    closure->param_count = param_count >= 0 ? static_cast<uint32_t>(param_count) : 0;
    return reinterpret_cast<int64_t>(closure);
}

static inline DynamicObject* unpack_dynamic_object(int64_t obj_box) {
    if (!obj_box) return nullptr;
    uint64_t u = static_cast<uint64_t>(obj_box);
    if ((u & HostValue::TAG_MASK) == HostValue::TAG_GCREF) {
        return reinterpret_cast<DynamicObject*>(u & HostValue::PAYLOAD_MASK);
    }
    if (u < 0x0000800000000000ULL) {
        return reinterpret_cast<DynamicObject*>(obj_box);
    }
    return nullptr;
}

BRONZE_WEAK uint64_t bronze_create_function(void* code, uint32_t arity, uint32_t /*length*/, uint32_t /*name_key*/, uint32_t /*fn_flags*/, uint64_t env_bits) {
    return static_cast<uint64_t>(bronze_create_func(code, static_cast<int32_t>(arity), static_cast<int64_t>(env_bits)));
}

BRONZE_WEAK int64_t bronze_function_singleton(void* code, uint32_t arity, uint32_t /*length*/, uint32_t /*name_key*/, uint32_t /*fn_flags*/, uint64_t* /*slot_cell*/) {
    return bronze_create_func(code, static_cast<int32_t>(arity), static_cast<int64_t>(kUndefinedTag));
}

BRONZE_WEAK int32_t bronze_to_int32_f64(double d) {
    if (!std::isfinite(d) || d == 0.0) return 0;
    const double truncated = std::trunc(d);
    const double residue = std::fmod(truncated, 4294967296.0);
    return static_cast<int32_t>(static_cast<uint32_t>(
        static_cast<int64_t>(residue < 0 ? residue + 4294967296.0 : residue)));
}

BRONZE_WEAK int32_t bronze_to_int32(uint64_t bits) {
    if (bits <= 0xFFF0000000000000ULL) {
        double d;
        std::memcpy(&d, &bits, sizeof(double));
        return bronze_to_int32_f64(d);
    } else if ((bits >> 48) == 0xFFF9 || (bits >> 48) == 0xFFF3) {
        return static_cast<int32_t>(bits & 0xFFFFFFFFULL);
    }
    return 0;
}
BRONZE_WEAK uint64_t bronze_private_new(void) { return 0; }
BRONZE_WEAK bool bronze_private_has(uint64_t /*tableBits*/, uint64_t /*objBits*/, uint32_t /*nameKeyIndex*/) { return false; }
BRONZE_WEAK uint64_t bronze_private_get(uint64_t /*tableBits*/, uint64_t /*objBits*/, uint32_t /*nameKeyIndex*/) { return 0; }
BRONZE_WEAK void bronze_private_add(uint64_t /*tableBits*/, uint64_t /*objBits*/, uint64_t /*valueBits*/) {}
BRONZE_WEAK void bronze_private_set(uint64_t /*tableBits*/, uint64_t /*objBits*/, uint64_t /*valueBits*/, uint32_t /*nameKeyIndex*/) {}
BRONZE_WEAK uint64_t bronze_private_misuse(uint32_t /*nameKeyIndex*/, uint32_t /*code*/) { return 0; }

static inline int64_t unbox_to_int64(int64_t v) {
    uint64_t u = static_cast<uint64_t>(v);
    if ((u >> 48) == 0xFFF9 || (u >> 48) == 0xFFF3) {
        return static_cast<int64_t>(static_cast<int32_t>(u & 0xFFFFFFFFULL));
    }
    if ((u >> 48) == 0xFFFF) {
        return v;
    }
    if (u <= 0xFFF0000000000000ULL) {
        if (u < 0x0010000000000000ULL) {
            return static_cast<int64_t>(u);
        }
        double d = 0.0;
        std::memcpy(&d, &v, sizeof(double));
        return static_cast<int64_t>(d);
    }
    return -1;
}

int64_t bronze_create_array(int32_t size) {
    HostGC* host_gc = brass::get_active_host_gc();
    size_t initial_cap = size > 0 ? static_cast<size_t>(size) : 8;
    DynamicObject* arr = DynamicObject::create_array(host_gc, initial_cap);
    if (arr && size > 0) {
        arr->set_length(static_cast<size_t>(size));
    }
    return reinterpret_cast<int64_t>(arr);
}

int64_t bronze_create_object() {
    auto* tlab = brass::get_active_tlab();
    if (tlab && tlab->owner_gc && !tlab->owner_gc->stress_mode()) {
        tlab->refill(sizeof(DynamicObject) + sizeof(HostGcHeader));
        uintptr_t addr = tlab->allocate_fast(sizeof(DynamicObject), DynamicObject::POINTER_MASK, DynamicObject::TYPE_TAG_DYNAMIC_OBJECT);
        if (addr != 0) {
            auto* obj = reinterpret_cast<DynamicObject*>(addr);
            obj->shape = ShapeRegistry::global().get_root_shape();
            obj->inline_capacity = static_cast<uint32_t>(DynamicObject::DEFAULT_INLINE_SLOTS);
            obj->out_of_line_capacity = 0;
            obj->out_of_line_slots = 0;
            for (size_t i = 0; i < DynamicObject::DEFAULT_INLINE_SLOTS; ++i) {
                obj->inline_slots[i] = HostValue::undefined_val();
            }
            obj->element_count = 0;
            obj->element_capacity = 0;
            obj->elements = 0;
            return reinterpret_cast<int64_t>(obj);
        }
    }
    HostGC* host_gc = brass::get_active_host_gc();
    DynamicObject* obj = DynamicObject::create(host_gc, ShapeRegistry::global().get_root_shape());
    return reinterpret_cast<int64_t>(obj);
}

int64_t bronze_prop_get(int64_t obj_box, int32_t key_index, uint64_t* /*ic_entry*/) {
    auto* obj = unpack_dynamic_object(obj_box);
    if (!obj) return static_cast<int64_t>(kUndefinedTag);
    uint32_t sym = static_cast<uint32_t>(key_index);
    if (obj->shape != nullptr && sym < Shape::FAST_SYMBOL_CAP) {
        int16_t s = obj->shape->fast_symbol_to_slot(sym);
        if (s >= 0) {
            uint32_t slot = static_cast<uint32_t>(s);
            if (slot < obj->inline_capacity) {
                return static_cast<int64_t>(obj->inline_slots[slot].raw());
            }
        }
    }
    HostValue val = obj->get_property(sym);
    if (val.is_undefined()) return static_cast<int64_t>(kUndefinedTag);
    return static_cast<int64_t>(val.raw());
}

// The fourth operand is the site's inline-cache entry, or null for a site
// with none; this stand-in caches nothing and never reads it. A key here is
// an opaque symbol — the textual IL carries no key strings — so an array
// literal's element stores (`create.array` then `prop.set` with the key of
// "0", "1", ...) cannot be told apart from named properties by their keys.
// The stand-in's model: the k-th distinct key stored into an element-capable
// object is also its element k, which is exactly the order a literal's
// stores arrive in and what `elem.get` reads back.
void bronze_prop_set(int64_t obj_box, int32_t key_index, int64_t val, uint64_t* /*ic_entry*/, int32_t /*strict*/) {
    auto* obj = unpack_dynamic_object(obj_box);
    if (!obj) return;
    uint32_t sym = static_cast<uint32_t>(key_index);
    HostGC* gc = brass::get_active_host_gc();
    const HostValue hv(static_cast<uint64_t>(val));
    std::optional<uint32_t> slot = obj->shape ? obj->shape->find_slot(sym) : std::nullopt;
    if (slot && *slot < obj->inline_capacity) {
        obj->inline_slots[*slot] = hv;
    } else {
        obj->set_property(sym, hv, ShapeRegistry::global(), gc);
        slot = obj->shape ? obj->shape->find_slot(sym) : std::nullopt;
    }
    if (obj->element_capacity > 0 && slot) {
        obj->set_element(static_cast<int64_t>(*slot), hv, gc);
    }
}

int64_t bronze_elem_get(int64_t arr_box, int64_t index_box) {
    auto* arr = unpack_dynamic_object(arr_box);
    if (!arr) return static_cast<int64_t>(kUndefinedTag);
    int64_t idx = unbox_to_int64(index_box);
    if (idx >= 0 && static_cast<uint64_t>(idx) < arr->element_count && arr->elements != 0) {
        const auto* slots = reinterpret_cast<const HostValue*>(arr->elements + sizeof(DynamicObjectBuffer));
        return static_cast<int64_t>(slots[idx].raw());
    }
    HostValue val = arr->get_element(idx);
    if (val.is_undefined()) return static_cast<int64_t>(kUndefinedTag);
    return static_cast<int64_t>(val.raw());
}

void bronze_elem_set(int64_t arr_box, int64_t index_box, int64_t val, int32_t /*ic_slot*/) {
    auto* arr = unpack_dynamic_object(arr_box);
    if (!arr) return;
    int64_t idx = unbox_to_int64(index_box);
    if (idx >= 0 && static_cast<uint64_t>(idx) < arr->element_count && arr->elements != 0) {
        auto* slots = reinterpret_cast<HostValue*>(arr->elements + sizeof(DynamicObjectBuffer));
        slots[idx] = HostValue(static_cast<uint64_t>(val));
        return;
    }
    HostGC* gc = brass::get_active_host_gc();
    arr->set_element(idx, HostValue(static_cast<uint64_t>(val)), gc);
}

void bronze_method_def(int64_t obj_box, int32_t key_index, int64_t closure_box) {
    auto* obj = unpack_dynamic_object(obj_box);
    if (!obj) return;
    HostGC* gc = brass::get_active_host_gc();
    HostValue val(static_cast<uint64_t>(closure_box));
    obj->set_property(static_cast<uint32_t>(key_index), val, ShapeRegistry::global(), gc);
}

void bronze_method_def_computed(int64_t obj_box, int64_t key_box, int64_t closure_box) {
    auto* obj = unpack_dynamic_object(obj_box);
    if (!obj) return;
    HostGC* gc = brass::get_active_host_gc();
    HostValue val(static_cast<uint64_t>(closure_box));
    obj->set_element(static_cast<uint32_t>(key_box), val, gc);
}

int64_t bronze_ic_get(uint32_t site_id, int64_t obj_box, const char* name, int32_t symbol_id) {
    auto* obj = unpack_dynamic_object(obj_box);
    if (!obj) return static_cast<int64_t>(kUndefinedTag);
    std::string_view prop_name = (name != nullptr) ? name : "";
    InlineCache* ic = ICRegistry::global().get_or_create_ic(site_id, prop_name, static_cast<uint32_t>(symbol_id), true);
    HostValue val = ic->execute_get(obj);
    if (val.is_undefined()) return static_cast<int64_t>(kUndefinedTag);
    return static_cast<int64_t>(val.raw());
}

void bronze_ic_set(uint32_t site_id, int64_t obj_box, const char* name, int32_t symbol_id, int64_t val_box) {
    auto* obj = unpack_dynamic_object(obj_box);
    if (!obj) return;
    std::string_view prop_name = (name != nullptr) ? name : "";
    InlineCache* ic = ICRegistry::global().get_or_create_ic(site_id, prop_name, static_cast<uint32_t>(symbol_id), false);
    HostGC* gc = brass::get_active_host_gc();
    ic->execute_set(obj, HostValue(static_cast<uint64_t>(val_box)), ShapeRegistry::global(), gc);
}

#ifndef _WIN32
__attribute__((weak))
#endif
void bronze_register_value_cells(uint64_t* cells, uint64_t count) {
    (void)cells;
    (void)count;
}

BRONZE_WEAK void bronze_register_method_ic_cells(uint64_t* ic_table, const uint64_t* site_indexes, uint64_t count) {
    (void)ic_table;
    (void)site_indexes;
    (void)count;
}

BRONZE_WEAK void bronze_register_fn_sources(const char* text, uint32_t text_len, const uint64_t* entries, uint32_t count) {
    (void)text;
    (void)text_len;
    (void)entries;
    (void)count;
}

BRONZE_WEAK uint64_t bronze_import_meta(uint32_t url_key_index) {
    (void)url_key_index;
    return static_cast<uint64_t>(kUndefinedTag);
}

#ifndef _WIN32
__attribute__((weak))
#endif
int64_t bronze_concat_begin(int64_t a, int64_t /*b*/, uint32_t /*remaining*/) {
    return a;
}

#ifndef _WIN32
__attribute__((weak))
#endif
int64_t bronze_concat_append(int64_t a, int64_t /*b*/) {
    return a;
}

#ifndef _WIN32
__attribute__((weak))
#endif
int64_t bronze_concat_end(int64_t a) {
    return a;
}

#ifndef _WIN32
__attribute__((weak))
#endif
int64_t bronze_global_get_name(const char* name) {
    if (!name) return static_cast<int64_t>(kUndefinedTag);
    if (std::strcmp(name, "globalThis") == 0) {
        static int64_t g_glob = 0;
        if (!g_glob) g_glob = bronze_create_object();
        return g_glob;
    }
    return bronze_name_resolve(name);
}

BRONZE_WEAK uint64_t bronze_global_get(uint32_t key_index, uint64_t* cache_cell) {
    (void)key_index;
    (void)cache_cell;
    return static_cast<uint64_t>(kUndefinedTag);
}

BRONZE_WEAK uint64_t bronze_typeof(uint64_t bits) {
    (void)bits;
    return static_cast<uint64_t>(kUndefinedTag);
}

void set_active_jit(codegen::JitExecutionEngine* jit) {
    g_active_jit = jit;
}

codegen::JitExecutionEngine* get_active_jit() {
    return g_active_jit;
}

} // namespace brass::il
