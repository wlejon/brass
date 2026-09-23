#include "il_runtime.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <charconv>
#include <bit>
#include <vector>
#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace brass::il {

#ifndef _WIN32
#define BRONZE_WEAK __attribute__((weak))
#define BRONZE_WEAK_DATA __attribute__((weak))
#else
#define BRONZE_WEAK
#define BRONZE_WEAK_DATA __declspec(selectany)
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

BRONZE_WEAK uint64_t bronze_resolve_name(uint32_t /*key_index*/, int32_t /*soft*/) {
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

static thread_local uint64_t g_bronze_dummy_exception_cell = kBronzeNoExceptionBits;
BRONZE_WEAK uint64_t bronze_exception_get() { return g_bronze_dummy_exception_cell; }
BRONZE_WEAK void bronze_exception_set(uint64_t bits) { g_bronze_dummy_exception_cell = bits; }
BRONZE_WEAK uint64_t bronze_exception_take() {
    uint64_t val = g_bronze_dummy_exception_cell;
    g_bronze_dummy_exception_cell = kBronzeNoExceptionBits;
    return val;
}
BRONZE_WEAK int32_t bronze_exception_pending() {
    return g_bronze_dummy_exception_cell != kBronzeNoExceptionBits;
}
BRONZE_WEAK void bronze_uncaught_exception() { std::exit(1); }

struct BronzeDummyGcFrame {
    BronzeDummyGcFrame* prev;
    uint64_t count;
    uint64_t slots[1];
};

// A fixed reservation: JIT code holds frame pointers (and frames link to
// each other) for as long as the frames are live, so the storage can never
// move. Overflowing it is a hard error, not a reallocation.
struct DummyShadowStack {
    static constexpr size_t kWords = 4 * 1024 * 1024; // 32 MB per thread
    std::vector<uint64_t> storage;
    size_t top = 0;
    BronzeDummyGcFrame* frame_top = nullptr;

    void ensure_init() {
        if (storage.empty()) {
            storage.resize(kWords, 0xFFF6000000000000ULL);
        }
    }
};
static thread_local DummyShadowStack g_dummy_shadow_stack;

BRONZE_WEAK void* bronze_gc_frame_push(uint32_t count) {
    g_dummy_shadow_stack.ensure_init();
    size_t required = sizeof(BronzeDummyGcFrame) / sizeof(uint64_t) + (count > 1 ? count - 1 : 0);
    if (g_dummy_shadow_stack.top + required > g_dummy_shadow_stack.storage.size()) {
        std::cerr << "fatal: bronze mock GC shadow stack overflow (" << DummyShadowStack::kWords
                  << " words; pushing a " << count << "-slot frame)\n";
        std::abort();
    }
    auto* frame = reinterpret_cast<BronzeDummyGcFrame*>(&g_dummy_shadow_stack.storage[g_dummy_shadow_stack.top]);
    g_dummy_shadow_stack.top += required;
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
        auto* prev = g_dummy_shadow_stack.frame_top->prev;
        g_dummy_shadow_stack.top = reinterpret_cast<uint64_t*>(g_dummy_shadow_stack.frame_top) - g_dummy_shadow_stack.storage.data();
        g_dummy_shadow_stack.frame_top = prev;
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
        double d = 0;
        std::memcpy(&d, &a, sizeof(double));
        return std::isnan(d) ? 0 : 1;
    }
    uint64_t ua = static_cast<uint64_t>(a);
    uint64_t ub = static_cast<uint64_t>(b);
    if (ua < 0xFFF0000000000000ULL && ub < 0xFFF0000000000000ULL) {
        double da = 0, db = 0;
        std::memcpy(&da, &a, sizeof(double));
        std::memcpy(&db, &b, sizeof(double));
        return da == db ? 1 : 0;
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

BRONZE_WEAK uint64_t bronze_immutable_assign() {
    g_bronze_dummy_exception_cell = 0xFFF6000000000000ULL;
    return 0xFFF6000000000000ULL;
}

struct BronzeDummyTlsBlock {
    void* frame_top = nullptr;
    uint64_t exception_cell = kBronzeNoExceptionBits;
    uint32_t proto_epoch = 1;
    uint64_t alloc_cursor = 0;
    uint64_t alloc_limit = 0;
    uint64_t plain_shape = 0;
};

extern "C" {
BRONZE_WEAK_DATA uintptr_t brass_tlab_top = 0;
BRONZE_WEAK_DATA uintptr_t brass_tlab_end = 0;
BRONZE_WEAK_DATA void* brass_root_shape = nullptr;
#if defined(_MSC_VER)
void* brass_dummy_bronze_tls_block_addr() {
    static thread_local BronzeDummyTlsBlock s_tls;
    return &s_tls;
}
#pragma comment(linker, "/alternatename:bronze_tls_block_addr=brass_dummy_bronze_tls_block_addr")
#else
BRONZE_WEAK void* bronze_tls_block_addr() {
#ifndef _WIN32
    using Fn = void* (*)();
    static Fn real_fn = []() -> Fn {
        void* sym = dlsym(RTLD_DEFAULT, "bronze_tls_block_addr");
        if (sym && sym != reinterpret_cast<void*>(&bronze_tls_block_addr)) {
            return reinterpret_cast<Fn>(sym);
        }
        return nullptr;
    }();
    if (real_fn) return real_fn();
#endif
    static thread_local BronzeDummyTlsBlock s_tls;
    return &s_tls;
}
#endif
} // extern "C"

BRONZE_WEAK void* bronze_tls_enter() { return bronze_tls_block_addr(); }
BRONZE_WEAK void bronze_stack_overflow() {}

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

#if defined(_MSC_VER)
extern "C" uint64_t brass_dummy_bronze_template_object(uint64_t cookedBits, uint64_t rawBits, uint64_t* cell) {
    (void)rawBits;
    if (cell) *cell = cookedBits;
    return cookedBits;
}
#pragma comment(linker, "/alternatename:bronze_template_object=brass_dummy_bronze_template_object")
#else
extern "C" BRONZE_WEAK uint64_t bronze_template_object(uint64_t cookedBits, uint64_t rawBits, uint64_t* cell) {
    (void)rawBits;
    if (cell) *cell = cookedBits;
    return cookedBits;
}
#endif

} // namespace brass::il
