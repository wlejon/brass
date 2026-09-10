#include "il_runtime.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/shape.hpp>
#include <brass/runtime/object.hpp>
#include <brass/runtime/inline_cache.hpp>
#include <brass/runtime/coroutine.hpp>
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

struct BronzeClosure {
    char fn_name[64];
    void* code_ptr;
    int64_t env_box;
    uint32_t param_count;
};

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

extern "C" {

void bronze_print_f64(double v) {
    if (!g_bronze_print_enabled) return;
    std::cout << format_js_number(v) << " ";
}

void bronze_print_i32(int32_t v) {
    if (!g_bronze_print_enabled) return;
    std::cout << v << " ";
}

void bronze_print_dynamic(int64_t v) {
    if (!g_bronze_print_enabled) return;
    uint64_t u = static_cast<uint64_t>(v);
    if (u < 0xFFF8000000000000ULL) {
        double d;
        std::memcpy(&d, &v, sizeof(double));
        std::cout << format_js_number(d) << " ";
    } else if ((u >> 48) == 0xFFF9) {
        int32_t iv = static_cast<int32_t>(u & 0xFFFFFFFFULL);
        std::cout << iv << " ";
    } else if ((u >> 48) == 0xFFFA) {
        std::cout << "null ";
    } else if ((u >> 48) == 0xFFFB) {
        bool b = (u & 1) != 0;
        std::cout << (b ? "true " : "false ");
    } else if ((u >> 48) == 0xFFFC) {
        std::cout << "undefined ";
    } else if (u == kPrintTag) {
        std::cout << "[Function: print] ";
    } else {
        std::cout << "undefined ";
    }
}

void bronze_print_newline() {
    if (!g_bronze_print_enabled) return;
    std::cout << "\n";
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

int64_t bronze_env_get(int64_t env_box, int32_t depth, int32_t index) {
    if (!env_box) return static_cast<int64_t>(kUndefinedTag);
    if (is_active_coro_frame(static_cast<uintptr_t>(env_box))) {
        auto* frame = reinterpret_cast<const BrassCoroFrame*>(env_box);
        if (static_cast<uint32_t>(index) < frame->slot_count) {
            return static_cast<int64_t>(frame->slots[index]);
        }
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* cur = reinterpret_cast<BronzeEnv*>(env_box);
    for (int32_t d = 0; d < depth && cur; ++d) {
        cur = reinterpret_cast<BronzeEnv*>(cur->parent_box);
    }
    if (!cur || static_cast<uint32_t>(index) >= cur->size) {
        return static_cast<int64_t>(kUndefinedTag);
    }
    return cur->slots[index];
}

void bronze_env_set(int64_t env_box, int32_t depth, int32_t index, int64_t val) {
    auto* cur = reinterpret_cast<BronzeEnv*>(env_box);
    for (int32_t d = 0; d < depth && cur; ++d) {
        cur = reinterpret_cast<BronzeEnv*>(cur->parent_box);
    }
    if (cur && static_cast<uint32_t>(index) < cur->size) {
        cur->slots[index] = val;
    }
}

int64_t bronze_create_func(const char* fn_name, int32_t param_count, int64_t env_box) {
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
    if (fn_name) {
        snprintf(closure->fn_name, sizeof(closure->fn_name), "%s", fn_name);
    }
    closure->code_ptr = bronze_resolve_function(closure->fn_name);
    closure->env_box = env_box;
    closure->param_count = param_count >= 0 ? static_cast<uint32_t>(param_count) : 0;
    return reinterpret_cast<int64_t>(closure);
}

static DynamicObject* unpack_dynamic_object(int64_t obj_box) {
    if (!obj_box) return nullptr;
    uint64_t u = static_cast<uint64_t>(obj_box);
    HostValue hv(u);
    if (hv.is_gcref()) {
        return hv.as_gcref_ptr<DynamicObject>();
    }
    if (u < 0x0000800000000000ULL) {
        return reinterpret_cast<DynamicObject*>(obj_box);
    }
    return nullptr;
}

static int64_t unbox_to_int64(int64_t v) {
    uint64_t u = static_cast<uint64_t>(v);
    if (u < 0xFFF8000000000000ULL) {
        double d;
        std::memcpy(&d, &v, sizeof(double));
        return static_cast<int64_t>(d);
    } else if ((u >> 48) == 0xFFF9) {
        return static_cast<int64_t>(static_cast<int32_t>(u & 0xFFFFFFFFULL));
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

int64_t bronze_prop_get(int64_t obj_box, int32_t key_index) {
    auto* obj = unpack_dynamic_object(obj_box);
    if (!obj) return static_cast<int64_t>(kUndefinedTag);
    HostValue val = obj->get_property(static_cast<uint32_t>(key_index));
    if (val.is_undefined()) return static_cast<int64_t>(kUndefinedTag);
    return static_cast<int64_t>(val.raw());
}

void bronze_prop_set(int64_t obj_box, int32_t key_index, int64_t val, int32_t slot_idx, int32_t /*imm*/) {
    auto* obj = unpack_dynamic_object(obj_box);
    if (!obj) return;
    HostGC* gc = brass::get_active_host_gc();
    obj->set_property(static_cast<uint32_t>(key_index), HostValue(static_cast<uint64_t>(val)), ShapeRegistry::global(), gc);
    if (slot_idx >= 0 && (obj->element_capacity > 0 || slot_idx < static_cast<int32_t>(obj->length()))) {
        obj->set_element(slot_idx, HostValue(static_cast<uint64_t>(val)), gc);
    }
}

int64_t bronze_elem_get(int64_t arr_box, int64_t index_box) {
    auto* arr = unpack_dynamic_object(arr_box);
    if (!arr) return static_cast<int64_t>(kUndefinedTag);
    int64_t idx = unbox_to_int64(index_box);
    HostValue val = arr->get_element(idx);
    if (val.is_undefined()) return static_cast<int64_t>(kUndefinedTag);
    return static_cast<int64_t>(val.raw());
}

void bronze_elem_set(int64_t arr_box, int64_t index_box, int64_t val, int32_t /*ic_slot*/) {
    auto* arr = unpack_dynamic_object(arr_box);
    if (!arr) return;
    int64_t idx = unbox_to_int64(index_box);
    HostGC* gc = brass::get_active_host_gc();
    arr->set_element(idx, HostValue(static_cast<uint64_t>(val)), gc);
}

void bronze_method_def(int64_t obj_box, const char* name, int32_t symbol_id, int64_t closure_box) {
    auto* obj = unpack_dynamic_object(obj_box);
    if (!obj) return;
    HostGC* gc = brass::get_active_host_gc();
    HostValue val(static_cast<uint64_t>(closure_box));
    if (name && name[0] != '\0') {
        obj->set_property(name, val, ShapeRegistry::global(), gc);
    } else {
        obj->set_property(static_cast<uint32_t>(symbol_id), val, ShapeRegistry::global(), gc);
    }
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

static void* get_closure_code(BronzeClosure* closure) {
    if (!closure) return nullptr;
    if (!closure->code_ptr && closure->fn_name[0] != '\0') {
        closure->code_ptr = bronze_resolve_function(closure->fn_name);
    }
    return closure->code_ptr;
}

int64_t bronze_call_dynamic_0(int64_t callee_box, int64_t /*this_box*/) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = reinterpret_cast<BronzeClosure*>(callee_box);
    void* code = get_closure_code(closure);
    if (code) {
        using Fn0 = int64_t(*)(int64_t);
        auto fn = reinterpret_cast<Fn0>(code);
        return fn(closure->env_box);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_1(int64_t callee_box, int64_t /*this_box*/, int64_t arg0) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_dynamic(arg0);
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = reinterpret_cast<BronzeClosure*>(callee_box);
    void* code = get_closure_code(closure);
    if (code) {
        using Fn1 = int64_t(*)(int64_t, int64_t);
        auto fn = reinterpret_cast<Fn1>(code);
        return fn(closure->env_box, arg0);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_2(int64_t callee_box, int64_t /*this_box*/, int64_t arg0, int64_t arg1) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_dynamic(arg0);
        bronze_print_dynamic(arg1);
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = reinterpret_cast<BronzeClosure*>(callee_box);
    void* code = get_closure_code(closure);
    if (code) {
        using Fn2 = int64_t(*)(int64_t, int64_t, int64_t);
        auto fn = reinterpret_cast<Fn2>(code);
        return fn(closure->env_box, arg0, arg1);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_3(int64_t callee_box, int64_t /*this_box*/, int64_t arg0, int64_t arg1, int64_t arg2) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_dynamic(arg0);
        bronze_print_dynamic(arg1);
        bronze_print_dynamic(arg2);
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = reinterpret_cast<BronzeClosure*>(callee_box);
    void* code = get_closure_code(closure);
    if (code) {
        using Fn3 = int64_t(*)(int64_t, int64_t, int64_t, int64_t);
        auto fn = reinterpret_cast<Fn3>(code);
        return fn(closure->env_box, arg0, arg1, arg2);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_4(int64_t callee_box, int64_t /*this_box*/, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_dynamic(arg0);
        bronze_print_dynamic(arg1);
        bronze_print_dynamic(arg2);
        bronze_print_dynamic(arg3);
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = reinterpret_cast<BronzeClosure*>(callee_box);
    void* code = get_closure_code(closure);
    if (code) {
        using Fn4 = int64_t(*)(int64_t, int64_t, int64_t, int64_t, int64_t);
        auto fn = reinterpret_cast<Fn4>(code);
        return fn(closure->env_box, arg0, arg1, arg2, arg3);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_5(int64_t callee_box, int64_t /*this_box*/, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_dynamic(arg0);
        bronze_print_dynamic(arg1);
        bronze_print_dynamic(arg2);
        bronze_print_dynamic(arg3);
        bronze_print_dynamic(arg4);
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = reinterpret_cast<BronzeClosure*>(callee_box);
    void* code = get_closure_code(closure);
    if (code) {
        using Fn5 = int64_t(*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
        auto fn = reinterpret_cast<Fn5>(code);
        return fn(closure->env_box, arg0, arg1, arg2, arg3, arg4);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_6(int64_t callee_box, int64_t /*this_box*/, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_dynamic(arg0);
        bronze_print_dynamic(arg1);
        bronze_print_dynamic(arg2);
        bronze_print_dynamic(arg3);
        bronze_print_dynamic(arg4);
        bronze_print_dynamic(arg5);
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = reinterpret_cast<BronzeClosure*>(callee_box);
    void* code = get_closure_code(closure);
    if (code) {
        using Fn6 = int64_t(*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
        auto fn = reinterpret_cast<Fn6>(code);
        return fn(closure->env_box, arg0, arg1, arg2, arg3, arg4, arg5);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_7(int64_t callee_box, int64_t /*this_box*/, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_dynamic(arg0);
        bronze_print_dynamic(arg1);
        bronze_print_dynamic(arg2);
        bronze_print_dynamic(arg3);
        bronze_print_dynamic(arg4);
        bronze_print_dynamic(arg5);
        bronze_print_dynamic(arg6);
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = reinterpret_cast<BronzeClosure*>(callee_box);
    void* code = get_closure_code(closure);
    if (code) {
        using Fn7 = int64_t(*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
        auto fn = reinterpret_cast<Fn7>(code);
        return fn(closure->env_box, arg0, arg1, arg2, arg3, arg4, arg5, arg6);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_8(int64_t callee_box, int64_t /*this_box*/, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6, int64_t arg7) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_dynamic(arg0);
        bronze_print_dynamic(arg1);
        bronze_print_dynamic(arg2);
        bronze_print_dynamic(arg3);
        bronze_print_dynamic(arg4);
        bronze_print_dynamic(arg5);
        bronze_print_dynamic(arg6);
        bronze_print_dynamic(arg7);
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = reinterpret_cast<BronzeClosure*>(callee_box);
    void* code = get_closure_code(closure);
    if (code) {
        using Fn8 = int64_t(*)(int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
        auto fn = reinterpret_cast<Fn8>(code);
        return fn(closure->env_box, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_n(int64_t callee_box, int64_t this_box, int32_t argc, const int64_t* argv) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        for (int32_t i = 0; i < argc; ++i) {
            bronze_print_dynamic(argv[i]);
        }
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    if (argc == 0) return bronze_call_dynamic_0(callee_box, this_box);
    if (argc == 1) return bronze_call_dynamic_1(callee_box, this_box, argv[0]);
    if (argc == 2) return bronze_call_dynamic_2(callee_box, this_box, argv[0], argv[1]);
    if (argc == 3) return bronze_call_dynamic_3(callee_box, this_box, argv[0], argv[1], argv[2]);
    if (argc == 4) return bronze_call_dynamic_4(callee_box, this_box, argv[0], argv[1], argv[2], argv[3]);
    return static_cast<int64_t>(kUndefinedTag);
}

} // extern "C"

void register_all_runtime_symbols(codegen::JitExecutionEngine& jit) {
    g_active_jit = &jit;

    jit.register_external_symbol("bronze_print_f64", reinterpret_cast<void*>(&bronze_print_f64));
    jit.register_external_symbol("bronze_print_i32", reinterpret_cast<void*>(&bronze_print_i32));
    jit.register_external_symbol("bronze_print_dynamic", reinterpret_cast<void*>(&bronze_print_dynamic));
    jit.register_external_symbol("bronze_print_newline", reinterpret_cast<void*>(&bronze_print_newline));
    jit.register_external_symbol("bronze_f64_mod", reinterpret_cast<void*>(&bronze_f64_mod));

    jit.register_external_symbol("bronze_name_resolve", reinterpret_cast<void*>(&bronze_name_resolve));
    jit.register_external_symbol("bronze_env_create", reinterpret_cast<void*>(&bronze_env_create));
    jit.register_external_symbol("bronze_env_get", reinterpret_cast<void*>(&bronze_env_get));
    jit.register_external_symbol("bronze_env_set", reinterpret_cast<void*>(&bronze_env_set));
    jit.register_external_symbol("bronze_create_func", reinterpret_cast<void*>(&bronze_create_func));
    jit.register_external_symbol("bronze_create_array", reinterpret_cast<void*>(&bronze_create_array));
    jit.register_external_symbol("bronze_create_object", reinterpret_cast<void*>(&bronze_create_object));
    jit.register_external_symbol("bronze_prop_get", reinterpret_cast<void*>(&bronze_prop_get));
    jit.register_external_symbol("bronze_prop_set", reinterpret_cast<void*>(&bronze_prop_set));
    jit.register_external_symbol("bronze_elem_get", reinterpret_cast<void*>(&bronze_elem_get));
    jit.register_external_symbol("bronze_elem_set", reinterpret_cast<void*>(&bronze_elem_set));
    jit.register_external_symbol("bronze_method_def", reinterpret_cast<void*>(&bronze_method_def));
    jit.register_external_symbol("bronze_ic_get", reinterpret_cast<void*>(&bronze_ic_get));
    jit.register_external_symbol("bronze_ic_set", reinterpret_cast<void*>(&bronze_ic_set));
    jit.register_external_symbol("brass_ic_get_prop", reinterpret_cast<void*>(&brass_ic_get_prop));
    jit.register_external_symbol("brass_ic_set_prop", reinterpret_cast<void*>(&brass_ic_set_prop));
    jit.register_external_symbol("brass_dynamic_object_get_prop_str", reinterpret_cast<void*>(&brass_dynamic_object_get_prop_str));
    jit.register_external_symbol("brass_dynamic_object_set_prop_str", reinterpret_cast<void*>(&brass_dynamic_object_set_prop_str));

    jit.register_external_symbol("bronze_call_dynamic_0", reinterpret_cast<void*>(&bronze_call_dynamic_0));
    jit.register_external_symbol("bronze_call_dynamic_1", reinterpret_cast<void*>(&bronze_call_dynamic_1));
    jit.register_external_symbol("bronze_call_dynamic_2", reinterpret_cast<void*>(&bronze_call_dynamic_2));
    jit.register_external_symbol("bronze_call_dynamic_3", reinterpret_cast<void*>(&bronze_call_dynamic_3));
    jit.register_external_symbol("bronze_call_dynamic_4", reinterpret_cast<void*>(&bronze_call_dynamic_4));
    jit.register_external_symbol("bronze_call_dynamic_5", reinterpret_cast<void*>(&bronze_call_dynamic_5));
    jit.register_external_symbol("bronze_call_dynamic_6", reinterpret_cast<void*>(&bronze_call_dynamic_6));
    jit.register_external_symbol("bronze_call_dynamic_7", reinterpret_cast<void*>(&bronze_call_dynamic_7));
    jit.register_external_symbol("bronze_call_dynamic_8", reinterpret_cast<void*>(&bronze_call_dynamic_8));
    jit.register_external_symbol("bronze_call_dynamic_n", reinterpret_cast<void*>(&bronze_call_dynamic_n));

    set_coro_symbol_resolver(&bronze_resolve_function);
    jit.register_external_symbol("brass_gc_write_barrier", reinterpret_cast<void*>(&brass_gc_write_barrier));
    jit.register_external_symbol("brass_gc_card_table_base", reinterpret_cast<void*>(&brass_gc_card_table_base));
    jit.register_external_symbol("brass_gc_heap_base", reinterpret_cast<void*>(&brass_gc_heap_base));
    jit.register_external_symbol("bronze_create_async_machine", reinterpret_cast<void*>(&bronze_create_async_machine));
    jit.register_external_symbol("bronze_async_start", reinterpret_cast<void*>(&bronze_async_start));
    jit.register_external_symbol("bronze_async_await", reinterpret_cast<void*>(&bronze_async_await));
    jit.register_external_symbol("bronze_iter_open", reinterpret_cast<void*>(&bronze_iter_open));
    jit.register_external_symbol("bronze_iter_step", reinterpret_cast<void*>(&bronze_iter_step));
}

void register_bronze_runtime_symbols(void* jit_engine_ptr) {
    if (jit_engine_ptr) {
        auto* jit = reinterpret_cast<codegen::JitExecutionEngine*>(jit_engine_ptr);
        register_all_runtime_symbols(*jit);
    }
}

void register_bronze_interpreter_symbols(void* interp_ptr) {
    if (!interp_ptr) return;
    auto* interp = reinterpret_cast<Interpreter*>(interp_ptr);
    interp->register_external_function("bronze_print_f64", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        if (!args.empty()) bronze_print_f64(args[0].as_f64());
        return RuntimeValue::from_void();
    });
    interp->register_external_function("bronze_print_i32", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        if (!args.empty()) bronze_print_i32(args[0].as_i32());
        return RuntimeValue::from_void();
    });
    interp->register_external_function("bronze_print_dynamic", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        if (!args.empty()) bronze_print_dynamic(args[0].as_i64());
        return RuntimeValue::from_void();
    });
    interp->register_external_function("bronze_print_newline", [](Interpreter&, const std::vector<RuntimeValue>&) {
        bronze_print_newline();
        return RuntimeValue::from_void();
    });
    interp->register_external_function("bronze_f64_mod", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        if (args.size() >= 2) return RuntimeValue::from_f64(bronze_f64_mod(args[0].as_f64(), args[1].as_f64()));
        return RuntimeValue::from_f64(0.0);
    });
    interp->register_external_function("bronze_call_dynamic_0", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        int64_t r = bronze_call_dynamic_0(args.size() > 0 ? args[0].as_i64() : 0, args.size() > 1 ? args[1].as_i64() : 0);
        return RuntimeValue::from_i64(r);
    });
    interp->register_external_function("bronze_call_dynamic_1", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        int64_t r = bronze_call_dynamic_1(
            args.size() > 0 ? args[0].as_i64() : 0,
            args.size() > 1 ? args[1].as_i64() : 0,
            args.size() > 2 ? args[2].as_i64() : 0
        );
        return RuntimeValue::from_i64(r);
    });
    interp->register_external_function("bronze_call_dynamic_2", [](Interpreter&, const std::vector<RuntimeValue>& args) {
        int64_t r = bronze_call_dynamic_2(
            args.size() > 0 ? args[0].as_i64() : 0,
            args.size() > 1 ? args[1].as_i64() : 0,
            args.size() > 2 ? args[2].as_i64() : 0,
            args.size() > 3 ? args[3].as_i64() : 0
        );
        return RuntimeValue::from_i64(r);
    });
}

} // namespace brass::il
