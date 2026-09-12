#include "il_runtime.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/inline_cache.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/parallel_runtime.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <charconv>
#include <system_error>
#include <vector>

namespace brass::il {

using namespace brass::runtime;

#ifndef _WIN32
#define BRONZE_WEAK __attribute__((weak))
#else
#define BRONZE_WEAK
#endif


std::string format_js_number(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    if (v == 0.0) return "0";

    // Check if integer within representable JS integer range without exponent (< 1e21)
    if (std::trunc(v) == v && std::abs(v) < 1e21) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%.0f", v);
        return std::string(buf);
    }

    char buf[64];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v);
    if (static_cast<int>(ec) == 0) {
        return std::string(buf, ptr - buf);
    }
    snprintf(buf, sizeof(buf), "%.16g", v);
    return std::string(buf);
}

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
        return g_active_jit->get_symbol_address(name);
    }
    return nullptr;
}

static bool g_bronze_print_enabled = true;

void bronze_set_print_enabled(bool enabled) {
    g_bronze_print_enabled = enabled;
}

#ifndef _WIN32
__attribute__((weak))
#endif
void bronze_print_f64(double v) {
    if (!g_bronze_print_enabled) return;
    std::cout << format_js_number(v);
}

#ifndef _WIN32
__attribute__((weak))
#endif
void bronze_print_i32(int32_t v) {
    if (!g_bronze_print_enabled) return;
    std::cout << v;
}

#ifndef _WIN32
__attribute__((weak))
#endif
void bronze_print_dynamic(int64_t v) {
    if (!g_bronze_print_enabled) return;
    uint64_t u = static_cast<uint64_t>(v);
    if (u <= 0xFFF0000000000000ULL) {
        double d;
        std::memcpy(&d, &v, sizeof(double));
        std::cout << format_js_number(d);
    } else if ((u >> 48) == 0xFFF9 || (u >> 48) == 0xFFF3) {
        int32_t iv = static_cast<int32_t>(u & 0xFFFFFFFFULL);
        std::cout << iv;
    } else if ((u >> 48) == 0xFFFA || (u >> 48) == 0xFFF5) {
        std::cout << "null";
    } else if ((u >> 48) == 0xFFFB || (u >> 48) == 0xFFF4) {
        bool b = (u & 1) != 0;
        std::cout << (b ? "true" : "false");
    } else if ((u >> 48) == 0xFFFC || (u >> 48) == 0xFFF6) {
        std::cout << "undefined";
    } else if (u == kPrintTag) {
        std::cout << "[Function: print]";
    } else {
        std::cout << "undefined";
    }
}

#ifndef _WIN32
__attribute__((weak))
#endif
void bronze_print_space() {
    if (!g_bronze_print_enabled) return;
    std::cout << " ";
}

#ifndef _WIN32
__attribute__((weak))
#endif
void bronze_print_newline() {
    if (!g_bronze_print_enabled) return;
    std::cout << "\n";
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

BRONZE_WEAK uint64_t bronze_resolve_name(uint32_t key_index, int32_t soft) {
    return static_cast<uint64_t>(kUndefinedTag);
}

int64_t bronze_env_create(int64_t parent_box, int32_t size) {
    size_t alloc_size = sizeof(BronzeEnv) + (size > 1 ? sizeof(int64_t) * (size - 1) : 0);
    BronzeEnv* env = nullptr;
    HostGC* host_gc = brass::get_active_host_gc();
    if (host_gc) {
        // Bit 0 of pointer mask is parent_box link (which is a GC pointer)
        uint64_t pointer_mask = 1ULL;
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

void bronze_prop_set(int64_t obj_box, int32_t key_index, int64_t val, uint64_t* ic_entry, int32_t /*strict*/) {
    auto* obj = unpack_dynamic_object(obj_box);
    if (!obj) return;
    int32_t slot_idx = static_cast<int32_t>(reinterpret_cast<uintptr_t>(ic_entry));
    if (obj->element_capacity > 0 && slot_idx >= 0) {
        HostGC* gc = brass::get_active_host_gc();
        obj->set_element(slot_idx, HostValue(static_cast<uint64_t>(val)), gc);
        return;
    }
    uint32_t sym = static_cast<uint32_t>(key_index);
    if (obj->shape != nullptr && sym < Shape::FAST_SYMBOL_CAP) {
        int16_t s = obj->shape->fast_symbol_to_slot(sym);
        if (s >= 0) {
            uint32_t slot = static_cast<uint32_t>(s);
            if (slot < obj->inline_capacity) {
                obj->inline_slots[slot] = HostValue(static_cast<uint64_t>(val));
                return;
            }
        }
    }
    HostGC* gc = brass::get_active_host_gc();
    obj->set_property(sym, HostValue(static_cast<uint64_t>(val)), ShapeRegistry::global(), gc);
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

BRONZE_WEAK uint64_t bronze_create_generator_object(uint64_t resumeBits) {
    return resumeBits;
}
BRONZE_WEAK uint64_t bronze_create_async_generator_object(uint64_t resumeBits) {
    return resumeBits;
}
BRONZE_WEAK uint64_t bronze_dynamic_import(uint64_t, uint32_t) {
    return static_cast<uint64_t>(kUndefinedTag);
}
BRONZE_WEAK uint64_t bronze_iter_value(uint64_t) { return static_cast<uint64_t>(kUndefinedTag); }
BRONZE_WEAK void bronze_iter_close(uint64_t, int32_t) {}
BRONZE_WEAK uint64_t bronze_iter_rest(uint64_t) { return bronze_create_array(0); }
BRONZE_WEAK uint64_t bronze_iter_delegate(uint64_t, uint64_t, uint64_t) { return static_cast<uint64_t>(kUndefinedTag); }
BRONZE_WEAK uint64_t bronze_async_iter_open(uint64_t o) { return o; }
BRONZE_WEAK uint64_t bronze_async_iter_next(uint64_t) { return static_cast<uint64_t>(kUndefinedTag); }
BRONZE_WEAK void bronze_async_iter_close(uint64_t, int32_t) {}
BRONZE_WEAK uint64_t bronze_pattern_check(uint64_t src, uint32_t) { return src; }
BRONZE_WEAK void bronze_array_append(uint64_t, uint64_t) {}
BRONZE_WEAK void bronze_array_append_hole(uint64_t) {}
BRONZE_WEAK void bronze_array_spread(uint64_t, uint64_t) {}
BRONZE_WEAK void bronze_object_spread(uint64_t, uint64_t) {}
BRONZE_WEAK uint64_t bronze_object_rest(uint64_t, uint64_t) { return bronze_create_object(); }
BRONZE_WEAK uint64_t bronze_dynamic_call_spread(uint64_t callee, uint64_t this_val, uint64_t) {
    return bronze_call_dynamic_0(callee, this_val);
}
BRONZE_WEAK uint64_t bronze_call_method_spread(uint64_t, uint32_t, uint64_t, void*) {
    return static_cast<uint64_t>(kUndefinedTag);
}
BRONZE_WEAK uint64_t bronze_construct_spread(uint64_t callee, uint64_t) {
    return bronze_construct_0(callee);
}
BRONZE_WEAK uint64_t bronze_super_call_spread(uint64_t, uint64_t, uint64_t) {
    return static_cast<uint64_t>(kUndefinedTag);
}
BRONZE_WEAK int64_t bronze_arg_at(uint32_t argc, const int64_t* argv, uint32_t index) {
    if (index < argc && argv) return argv[index];
    return static_cast<int64_t>(kUndefinedTag);
}
BRONZE_WEAK int64_t bronze_arguments_object(uint32_t argc, const int64_t* /*argv*/, int64_t /*callee*/, int32_t /*is_strict*/) {
    return bronze_create_array(static_cast<int32_t>(argc));
}
BRONZE_WEAK int64_t bronze_rest_args(uint32_t argc, const int64_t* /*argv*/, uint32_t first_index) {
    uint32_t count = argc > first_index ? (argc - first_index) : 0;
    return bronze_create_array(static_cast<int32_t>(count));
}
BRONZE_WEAK int32_t bronze_instanceof(int64_t, int64_t) { return 1; }
BRONZE_WEAK int64_t bronze_super_get(int64_t proto_box, uint32_t key_index, int64_t) {
    return bronze_prop_get(proto_box, static_cast<int32_t>(key_index));
}
BRONZE_WEAK void bronze_super_set(uint64_t, uint32_t, uint64_t, uint64_t, int32_t) {}
BRONZE_WEAK int64_t bronze_object_keys(int64_t /*obj_box*/) {
    return bronze_create_array(0);
}
BRONZE_WEAK int64_t bronze_for_in_keys(int64_t /*obj_box*/) {
    return bronze_create_array(0);
}
BRONZE_WEAK int32_t bronze_has_property(int64_t, int64_t) { return 0; }
BRONZE_WEAK int32_t bronze_is_nullish(int64_t val_box) {
    return (val_box == static_cast<int64_t>(kUndefinedTag) || val_box == static_cast<int64_t>(kNullTag)) ? 1 : 0;
}

BRONZE_WEAK void bronze_define_own_attr(uint64_t /*obj_bits*/, uint32_t /*key_index*/, uint64_t /*val_bits*/, uint32_t /*mask*/) {}
BRONZE_WEAK void bronze_accessor_def(uint64_t, uint32_t, uint64_t, uint64_t, int32_t) {}
BRONZE_WEAK void bronze_accessor_def_computed(uint64_t, uint64_t, uint64_t, uint64_t, int32_t) {}
BRONZE_WEAK uint64_t bronze_module_namespace(uint64_t src) { return src; }
BRONZE_WEAK void bronze_pin_guard(int64_t /*val_box*/, int32_t /*shape*/, const char* /*name*/) {}
BRONZE_WEAK void bronze_census_record(uint32_t /*key_id*/, uint32_t /*site_info*/, uint64_t /*value_bits*/) {}
BRONZE_WEAK double bronze_unbox_f64(uint64_t bits) { return std::bit_cast<double>(bits); }
BRONZE_WEAK uint64_t bronze_box_f64(double v) { return std::bit_cast<uint64_t>(v); }
BRONZE_WEAK int32_t bronze_unbox_i32(uint64_t bits) { return static_cast<int32_t>(bits & 0xFFFFFFFFLL); }
BRONZE_WEAK uint64_t bronze_box_i32(int32_t v) {
    return (static_cast<uint64_t>(static_cast<int64_t>(v)) & 0xFFFFFFFFLL) | kInt32Tag;
}
BRONZE_WEAK int32_t bronze_unbox_bool(uint64_t bits) {
    if (bits <= 0xFFF0000000000000ULL) {
        double d = 0.0;
        std::memcpy(&d, &bits, sizeof(double));
        return (d != 0.0 && !std::isnan(d)) ? 1 : 0;
    }
    uint32_t tag = static_cast<uint32_t>(bits >> 48);
    if (tag == 0xFFF4) return (bits & 1) ? 1 : 0;
    if (tag == 0xFFF5 || tag == 0xFFF6) return 0;
    if (tag == 0xFFF1) return 1;
    return (bits & 1) ? 1 : 0;
}
BRONZE_WEAK uint64_t bronze_box_bool(int32_t v) { return static_cast<uint64_t>(v ? 1 : 0) | kBoolTag; }
uint32_t g_bronze_dummy_key_map[4096];
static struct DummyKeyMapInitializer {
    DummyKeyMapInitializer() {
        for (uint32_t i = 0; i < 4096; ++i) {
            g_bronze_dummy_key_map[i] = i;
        }
    }
} g_dummy_key_map_initializer;

BRONZE_WEAK uint64_t bronze_box_str_key(uint32_t /*key_index*/) { return kUndefinedTag; }
BRONZE_WEAK uint64_t bronze_box_str(const char* /*s*/) { return kUndefinedTag; }
BRONZE_WEAK const char* bronze_unbox_str(uint64_t /*bits*/) { return ""; }
BRONZE_WEAK void bronze_register_key_manifest(const uint8_t* /*data*/, uint32_t* /*key_map*/) {}

static thread_local uint64_t g_bronze_dummy_exception_cell = 0xFFFA000000000000ULL;
BRONZE_WEAK uint64_t bronze_exception_get() { return g_bronze_dummy_exception_cell; }
BRONZE_WEAK void bronze_exception_set(uint64_t bits) { g_bronze_dummy_exception_cell = bits; }
BRONZE_WEAK uint64_t bronze_exception_take() {
    uint64_t val = g_bronze_dummy_exception_cell;
    g_bronze_dummy_exception_cell = 0xFFFA000000000000ULL;
    return val;
}
BRONZE_WEAK int32_t bronze_exception_pending() {
    return g_bronze_dummy_exception_cell != 0xFFFA000000000000ULL;
}
BRONZE_WEAK void bronze_uncaught_exception() { std::exit(1); }

struct BronzeDummyGcFrame {
    BronzeDummyGcFrame* prev;
    uint64_t count;
    uint64_t slots[1];
};

struct DummyShadowStack {
    std::vector<uint64_t> storage;
    size_t top = 0;
    BronzeDummyGcFrame* frame_top = nullptr;

    void ensure_init() {
        if (storage.empty()) {
            storage.resize(4 * 1024 * 1024, 0xFFF6000000000000ULL);
        }
    }
};
static thread_local DummyShadowStack g_dummy_shadow_stack;

BRONZE_WEAK void* bronze_gc_frame_push(uint32_t count) {
    g_dummy_shadow_stack.ensure_init();
    size_t cur = g_dummy_shadow_stack.top;
    g_dummy_shadow_stack.top += 2 + count;
    auto* frame = reinterpret_cast<BronzeDummyGcFrame*>(&g_dummy_shadow_stack.storage[cur]);
    frame->prev = g_dummy_shadow_stack.frame_top;
    frame->count = count;
    for (uint32_t i = 0; i < count; ++i) {
        frame->slots[i] = 0xFFF6000000000000ULL;
    }
    g_dummy_shadow_stack.frame_top = frame;
    return frame;
}

BRONZE_WEAK void bronze_gc_frame_pop() {
    if (g_dummy_shadow_stack.frame_top) {
        g_dummy_shadow_stack.top = reinterpret_cast<uint64_t*>(g_dummy_shadow_stack.frame_top) - g_dummy_shadow_stack.storage.data();
        g_dummy_shadow_stack.frame_top = g_dummy_shadow_stack.frame_top->prev;
    }
}

BRONZE_WEAK uint64_t bronze_pin_violation(uint32_t /*key_index*/, uint64_t bits) {
    g_bronze_dummy_exception_cell = bits;
    return 0xFFF6000000000000ULL;
}
BRONZE_WEAK void bronze_pin_check_array(uint32_t /*key_index*/, uint64_t /*bits*/) {}

BRONZE_WEAK int32_t bronze_strict_eq(int64_t a, int64_t b) {
    if (a == b) {
        uint64_t ua = static_cast<uint64_t>(a);
        if (ua >= 0xFFF0000000000000ULL) return 1;
        double da;
        std::memcpy(&da, &a, sizeof(double));
        return std::isnan(da) ? 0 : 1;
    }
    uint64_t ua = static_cast<uint64_t>(a);
    uint64_t ub = static_cast<uint64_t>(b);
    if (ua < 0xFFF0000000000000ULL && ub < 0xFFF0000000000000ULL) {
        double da, db;
        std::memcpy(&da, &a, sizeof(double));
        std::memcpy(&db, &b, sizeof(double));
        return (da == db) ? 1 : 0;
    }
    return 0;
}
BRONZE_WEAK int32_t bronze_loose_eq(int64_t a, int64_t b) { return bronze_strict_eq(a, b); }
static inline double to_dbl(int64_t v) { double d = 0; std::memcpy(&d, &v, sizeof(double)); return d; }
BRONZE_WEAK int32_t bronze_rel_lt(int64_t a, int64_t b) { return to_dbl(a) < to_dbl(b) ? 1 : 0; }
BRONZE_WEAK int32_t bronze_rel_gt(int64_t a, int64_t b) { return to_dbl(a) > to_dbl(b) ? 1 : 0; }
BRONZE_WEAK int32_t bronze_rel_le(int64_t a, int64_t b) { return to_dbl(a) <= to_dbl(b) ? 1 : 0; }
BRONZE_WEAK int32_t bronze_rel_ge(int64_t a, int64_t b) { return to_dbl(a) >= to_dbl(b) ? 1 : 0; }
BRONZE_WEAK double bronze_pow(double base, double exp) { return std::pow(base, exp); }
BRONZE_WEAK uint64_t bronze_dynamic_pow(uint64_t l, uint64_t r) {
    double d = std::pow(to_dbl(static_cast<int64_t>(l)), to_dbl(static_cast<int64_t>(r)));
    uint64_t u = 0; std::memcpy(&u, &d, sizeof(double)); return u;
}
static inline uint64_t to_bits(double d) { uint64_t u = 0; std::memcpy(&u, &d, sizeof(double)); return u; }
BRONZE_WEAK uint64_t bronze_dynamic_bitand(uint64_t l, uint64_t r) {
    int32_t a = static_cast<int32_t>(to_dbl(static_cast<int64_t>(l)));
    int32_t b = static_cast<int32_t>(to_dbl(static_cast<int64_t>(r)));
    return to_bits(static_cast<double>(a & b));
}
BRONZE_WEAK uint64_t bronze_dynamic_bitor(uint64_t l, uint64_t r) {
    int32_t a = static_cast<int32_t>(to_dbl(static_cast<int64_t>(l)));
    int32_t b = static_cast<int32_t>(to_dbl(static_cast<int64_t>(r)));
    return to_bits(static_cast<double>(a | b));
}
BRONZE_WEAK uint64_t bronze_dynamic_bitxor(uint64_t l, uint64_t r) {
    int32_t a = static_cast<int32_t>(to_dbl(static_cast<int64_t>(l)));
    int32_t b = static_cast<int32_t>(to_dbl(static_cast<int64_t>(r)));
    return to_bits(static_cast<double>(a ^ b));
}
BRONZE_WEAK uint64_t bronze_dynamic_shl(uint64_t l, uint64_t r) {
    int32_t a = static_cast<int32_t>(to_dbl(static_cast<int64_t>(l)));
    uint32_t b = static_cast<uint32_t>(to_dbl(static_cast<int64_t>(r))) & 31u;
    return to_bits(static_cast<double>(static_cast<int32_t>(static_cast<uint32_t>(a) << b)));
}
BRONZE_WEAK uint64_t bronze_dynamic_shr(uint64_t l, uint64_t r) {
    int32_t a = static_cast<int32_t>(to_dbl(static_cast<int64_t>(l)));
    uint32_t b = static_cast<uint32_t>(to_dbl(static_cast<int64_t>(r))) & 31u;
    return to_bits(static_cast<double>(a >> b));
}
BRONZE_WEAK uint64_t bronze_dynamic_ushr(uint64_t l, uint64_t r) {
    uint32_t a = static_cast<uint32_t>(to_dbl(static_cast<int64_t>(l)));
    uint32_t b = static_cast<uint32_t>(to_dbl(static_cast<int64_t>(r))) & 31u;
    return to_bits(static_cast<double>(a >> b));
}
BRONZE_WEAK uint64_t bronze_dynamic_sub(uint64_t l, uint64_t r) {
    return to_bits(to_dbl(static_cast<int64_t>(l)) - to_dbl(static_cast<int64_t>(r)));
}
BRONZE_WEAK uint64_t bronze_dynamic_mul(uint64_t l, uint64_t r) {
    return to_bits(to_dbl(static_cast<int64_t>(l)) * to_dbl(static_cast<int64_t>(r)));
}
BRONZE_WEAK uint64_t bronze_dynamic_div(uint64_t l, uint64_t r) {
    return to_bits(to_dbl(static_cast<int64_t>(l)) / to_dbl(static_cast<int64_t>(r)));
}
BRONZE_WEAK uint64_t bronze_dynamic_mod(uint64_t l, uint64_t r) {
    return to_bits(std::fmod(to_dbl(static_cast<int64_t>(l)), to_dbl(static_cast<int64_t>(r))));
}
BRONZE_WEAK uint64_t bronze_dynamic_neg(uint64_t bits) {
    return to_bits(-to_dbl(static_cast<int64_t>(bits)));
}
BRONZE_WEAK uint64_t bronze_dynamic_bitnot(uint64_t bits) {
    int32_t a = static_cast<int32_t>(to_dbl(static_cast<int64_t>(bits)));
    return to_bits(static_cast<double>(~a));
}

BRONZE_WEAK uint64_t bronze_bigint_literal(uint32_t) { return 0; }
BRONZE_WEAK uint64_t bronze_async_machine(uint64_t resumeBits) { return resumeBits; }
BRONZE_WEAK uint64_t bronze_to_string(uint64_t bits) { return bits; }
BRONZE_WEAK int32_t bronze_prop_delete(uint64_t, uint32_t, int32_t) { return 1; }
BRONZE_WEAK int32_t bronze_elem_delete(uint64_t, uint64_t, int32_t) { return 1; }

uint64_t bronze_get_new_target() {
    return static_cast<uint64_t>(kUndefinedTag);
}

BRONZE_WEAK void bronze_print_f64_err(double v) { std::fprintf(stderr, "%s", format_js_number(v).c_str()); }
BRONZE_WEAK void bronze_print_i32_err(int32_t v) { std::fprintf(stderr, "%d", v); }
BRONZE_WEAK void bronze_print_dynamic_err(int64_t v) {
    if (!g_bronze_print_enabled) return;
    uint64_t u = static_cast<uint64_t>(v);
    if (u <= 0xFFF0000000000000ULL) {
        double d = 0;
        std::memcpy(&d, &v, sizeof(double));
        std::cerr << format_js_number(d);
    } else if ((u >> 48) == 0xFFF9 || (u >> 48) == 0xFFF3) {
        int32_t iv = static_cast<int32_t>(u & 0xFFFFFFFFULL);
        std::cerr << iv;
    } else if ((u >> 48) == 0xFFFA || (u >> 48) == 0xFFF5) {
        std::cerr << "null";
    } else if ((u >> 48) == 0xFFFB || (u >> 48) == 0xFFF4) {
        bool b = (u & 1) != 0;
        std::cerr << (b ? "true" : "false");
    } else if ((u >> 48) == 0xFFFC || (u >> 48) == 0xFFF6) {
        std::cerr << "undefined";
    } else if (u == kPrintErrTag) {
        std::cerr << "[Function: printErr]";
    } else {
        std::cerr << "undefined";
    }
}
BRONZE_WEAK void bronze_print_space_err() { std::fputc(' ', stderr); }
BRONZE_WEAK void bronze_print_newline_err() { std::fputc('\n', stderr); std::fflush(stderr); }
BRONZE_WEAK void bronze_print_spread(uint64_t) {}
BRONZE_WEAK void bronze_print_spread_err(uint64_t) {}
BRONZE_WEAK uint64_t bronze_immutable_assign() {
    g_bronze_dummy_exception_cell = 0xFFF6000000000000ULL;
    return 0xFFF6000000000000ULL;
}

uint32_t g_bronze_main_key_constants = 0;
int64_t g_bronze_module_env = kUndefinedTag;
extern "C" uint64_t __bronze_template_cells[1024];

static struct BronzeTemplateCellsInit {
    BronzeTemplateCellsInit() {
        for (size_t i = 0; i < 1024; ++i) {
            __bronze_template_cells[i] = kUndefinedTag;
        }
    }
} g_bronze_template_cells_init;
uint64_t __bronze_template_cells[1024];

extern "C" BRONZE_WEAK uint64_t bronze_template_object(uint64_t cookedBits, uint64_t rawBits, uint64_t* cell) {
    (void)rawBits;
    if (cell) *cell = cookedBits;
    return cookedBits;
}

void set_active_jit(codegen::JitExecutionEngine* jit) {
    g_active_jit = jit;
}

codegen::JitExecutionEngine* get_active_jit() {
    return g_active_jit;
}

} // namespace brass::il

