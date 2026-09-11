#include "il_runtime.hpp"
#include <brass/il_translator/il_translator.hpp>
#include <brass/codegen/jit_exec.hpp>
#include <brass/embedding/host_gc.hpp>
#include <brass/runtime/object.hpp>
#include <iostream>
#include <cstring>

namespace brass::il {

using namespace brass::runtime;

#ifndef _WIN32
#define BRONZE_WEAK __attribute__((weak))
#else
#define BRONZE_WEAK
#endif

static inline BronzeClosure* unpack_closure(int64_t callee_box) {
    if (!callee_box) return nullptr;
    uint64_t u = static_cast<uint64_t>(callee_box);
    if ((u & HostValue::TAG_MASK) == HostValue::TAG_GCREF) {
        return reinterpret_cast<BronzeClosure*>(u & HostValue::PAYLOAD_MASK);
    }
    if (u < 0x0000800000000000ULL && u >= 0x1000ULL) {
        return reinterpret_cast<BronzeClosure*>(callee_box);
    }
    return nullptr;
}

static void* get_closure_code(BronzeClosure* closure) {
    if (!closure) return nullptr;
    uintptr_t ptr = reinterpret_cast<uintptr_t>(closure);
    if (ptr >= 0x0000800000000000ULL || ptr < 0x1000ULL) return nullptr;
    if (!closure->code_ptr && closure->fn_name[0] != '\0') {
        closure->code_ptr = bronze_resolve_function(closure->fn_name);
    }
    return closure->code_ptr;
}

using BronzeFnCode = int64_t(*)(int64_t, int64_t, uint32_t, const int64_t*);

static inline void print_dynamic_helper(const int64_t* argv, size_t argc) {
    for (size_t i = 0; i < argc; ++i) {
        if (i > 0) bronze_print_space();
        bronze_print_dynamic(argv[i]);
    }
    bronze_print_newline();
}

int64_t bronze_call_dynamic_0(int64_t callee_box, int64_t this_box) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 0, nullptr);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_1(int64_t callee_box, int64_t this_box, int64_t arg0) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        bronze_print_dynamic(arg0);
        bronze_print_newline();
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[1] = {arg0};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 1, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_2(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[2] = {arg0, arg1};
        print_dynamic_helper(args, 2);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[2] = {arg0, arg1};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 2, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_3(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[3] = {arg0, arg1, arg2};
        print_dynamic_helper(args, 3);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[3] = {arg0, arg1, arg2};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 3, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_4(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[4] = {arg0, arg1, arg2, arg3};
        print_dynamic_helper(args, 4);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[4] = {arg0, arg1, arg2, arg3};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 4, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_5(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[5] = {arg0, arg1, arg2, arg3, arg4};
        print_dynamic_helper(args, 5);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[5] = {arg0, arg1, arg2, arg3, arg4};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 5, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_6(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[6] = {arg0, arg1, arg2, arg3, arg4, arg5};
        print_dynamic_helper(args, 6);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[6] = {arg0, arg1, arg2, arg3, arg4, arg5};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 6, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_7(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[7] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6};
        print_dynamic_helper(args, 7);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[7] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 7, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_8(int64_t callee_box, int64_t this_box, int64_t arg0, int64_t arg1, int64_t arg2, int64_t arg3, int64_t arg4, int64_t arg5, int64_t arg6, int64_t arg7) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[8] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7};
        print_dynamic_helper(args, 8);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[8] = {arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 8, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_9(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[9] = {a0, a1, a2, a3, a4, a5, a6, a7, a8};
        print_dynamic_helper(args, 9);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[9] = {a0, a1, a2, a3, a4, a5, a6, a7, a8};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 9, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_10(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[10] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9};
        print_dynamic_helper(args, 10);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[10] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 10, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_11(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[11] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10};
        print_dynamic_helper(args, 11);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[11] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 11, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_12(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[12] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11};
        print_dynamic_helper(args, 12);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[12] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 12, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_13(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[13] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12};
        print_dynamic_helper(args, 13);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[13] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 13, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_14(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[14] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13};
        print_dynamic_helper(args, 14);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[14] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 14, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_15(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[15] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14};
        print_dynamic_helper(args, 15);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[15] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 15, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic_16(int64_t callee_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14, int64_t a15) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        const int64_t args[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
        print_dynamic_helper(args, 16);
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        int64_t argv[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, 16, argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

int64_t bronze_call_dynamic(int64_t callee_box, uint32_t argc, const int64_t* argv) {
    return bronze_call_dynamic_n(callee_box, static_cast<int64_t>(kUndefinedTag), static_cast<int32_t>(argc), argv);
}

int64_t bronze_call_dynamic_n(int64_t callee_box, int64_t this_box, int32_t argc, const int64_t* argv) {
    if (callee_box == static_cast<int64_t>(kPrintTag)) {
        if (argc > 0 && argv) {
            print_dynamic_helper(argv, static_cast<size_t>(argc));
        } else {
            bronze_print_newline();
        }
        return static_cast<int64_t>(kUndefinedTag);
    }
    auto* closure = unpack_closure(callee_box);
    void* code = get_closure_code(closure);
    if (code && closure) {
        return reinterpret_cast<BronzeFnCode>(code)(closure->env_box, this_box, static_cast<uint32_t>(argc), argv);
    }
    return static_cast<int64_t>(kUndefinedTag);
}

BRONZE_WEAK int64_t bronze_construct_0(int64_t callee_box) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_0(callee_box, obj);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_1(int64_t callee_box, int64_t a0) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_1(callee_box, obj, a0);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_2(int64_t callee_box, int64_t a0, int64_t a1) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_2(callee_box, obj, a0, a1);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_3(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_3(callee_box, obj, a0, a1, a2);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_4(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_4(callee_box, obj, a0, a1, a2, a3);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_5(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_5(callee_box, obj, a0, a1, a2, a3, a4);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_6(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_6(callee_box, obj, a0, a1, a2, a3, a4, a5);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_7(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_7(callee_box, obj, a0, a1, a2, a3, a4, a5, a6);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_8(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_8(callee_box, obj, a0, a1, a2, a3, a4, a5, a6, a7);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_9(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_9(callee_box, obj, a0, a1, a2, a3, a4, a5, a6, a7, a8);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_10(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_10(callee_box, obj, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_11(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_11(callee_box, obj, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_12(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_12(callee_box, obj, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_13(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_13(callee_box, obj, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_14(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_14(callee_box, obj, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_15(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_15(callee_box, obj, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct_16(int64_t callee_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14, int64_t a15) {
    int64_t obj = bronze_create_object();
    int64_t res = bronze_call_dynamic_16(callee_box, obj, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15);
    return (res && (res & 0xFFF0000000000000ULL) != 0xFFF0000000000000ULL) ? res : obj;
}
BRONZE_WEAK int64_t bronze_construct(int64_t callee_box, uint32_t argc, const int64_t* argv) {
    if (argc == 0) return bronze_construct_0(callee_box);
    if (argc == 1) return bronze_construct_1(callee_box, argv[0]);
    if (argc == 2) return bronze_construct_2(callee_box, argv[0], argv[1]);
    if (argc == 3) return bronze_construct_3(callee_box, argv[0], argv[1], argv[2]);
    if (argc == 4) return bronze_construct_4(callee_box, argv[0], argv[1], argv[2], argv[3]);
    if (argc == 5) return bronze_construct_5(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4]);
    if (argc == 6) return bronze_construct_6(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5]);
    if (argc == 7) return bronze_construct_7(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]);
    if (argc == 8) return bronze_construct_8(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]);
    if (argc == 9) return bronze_construct_9(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8]);
    if (argc == 10) return bronze_construct_10(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9]);
    if (argc == 11) return bronze_construct_11(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10]);
    if (argc == 12) return bronze_construct_12(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11]);
    if (argc == 13) return bronze_construct_13(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12]);
    if (argc == 14) return bronze_construct_14(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13]);
    if (argc == 15) return bronze_construct_15(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14]);
    if (argc >= 16) return bronze_construct_16(callee_box, argv[0], argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9], argv[10], argv[11], argv[12], argv[13], argv[14], argv[15]);
    return bronze_create_object();
}

BRONZE_WEAK void bronze_class_extends(int64_t, int64_t) {}
BRONZE_WEAK int64_t bronze_super_call(int64_t sub_box, int64_t this_box, uint32_t argc, const int64_t* argv) {
    return bronze_call_dynamic_n(sub_box, this_box, argc, argv);
}
BRONZE_WEAK int64_t bronze_super_call_0(int64_t sub_box, int64_t this_box) {
    return bronze_super_call(sub_box, this_box, 0, nullptr);
}
BRONZE_WEAK int64_t bronze_super_call_1(int64_t sub_box, int64_t this_box, int64_t a0) {
    int64_t argv[1] = {a0};
    return bronze_super_call(sub_box, this_box, 1, argv);
}
BRONZE_WEAK int64_t bronze_super_call_2(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1) {
    int64_t argv[2] = {a0, a1};
    return bronze_super_call(sub_box, this_box, 2, argv);
}
BRONZE_WEAK int64_t bronze_super_call_3(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2) {
    int64_t argv[3] = {a0, a1, a2};
    return bronze_super_call(sub_box, this_box, 3, argv);
}
BRONZE_WEAK int64_t bronze_super_call_4(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3) {
    int64_t argv[4] = {a0, a1, a2, a3};
    return bronze_super_call(sub_box, this_box, 4, argv);
}
BRONZE_WEAK int64_t bronze_super_call_5(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4) {
    int64_t argv[5] = {a0, a1, a2, a3, a4};
    return bronze_super_call(sub_box, this_box, 5, argv);
}
BRONZE_WEAK int64_t bronze_super_call_6(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5) {
    int64_t argv[6] = {a0, a1, a2, a3, a4, a5};
    return bronze_super_call(sub_box, this_box, 6, argv);
}
BRONZE_WEAK int64_t bronze_super_call_7(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6) {
    int64_t argv[7] = {a0, a1, a2, a3, a4, a5, a6};
    return bronze_super_call(sub_box, this_box, 7, argv);
}
BRONZE_WEAK int64_t bronze_super_call_8(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7) {
    int64_t argv[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    return bronze_super_call(sub_box, this_box, 8, argv);
}
BRONZE_WEAK int64_t bronze_super_call_9(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8) {
    int64_t argv[9] = {a0, a1, a2, a3, a4, a5, a6, a7, a8};
    return bronze_super_call(sub_box, this_box, 9, argv);
}
BRONZE_WEAK int64_t bronze_super_call_10(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9) {
    int64_t argv[10] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9};
    return bronze_super_call(sub_box, this_box, 10, argv);
}
BRONZE_WEAK int64_t bronze_super_call_11(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10) {
    int64_t argv[11] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10};
    return bronze_super_call(sub_box, this_box, 11, argv);
}
BRONZE_WEAK int64_t bronze_super_call_12(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11) {
    int64_t argv[12] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11};
    return bronze_super_call(sub_box, this_box, 12, argv);
}
BRONZE_WEAK int64_t bronze_super_call_13(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12) {
    int64_t argv[13] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12};
    return bronze_super_call(sub_box, this_box, 13, argv);
}
BRONZE_WEAK int64_t bronze_super_call_14(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13) {
    int64_t argv[14] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13};
    return bronze_super_call(sub_box, this_box, 14, argv);
}
BRONZE_WEAK int64_t bronze_super_call_15(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14) {
    int64_t argv[15] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14};
    return bronze_super_call(sub_box, this_box, 15, argv);
}
BRONZE_WEAK int64_t bronze_super_call_16(int64_t sub_box, int64_t this_box, int64_t a0, int64_t a1, int64_t a2, int64_t a3, int64_t a4, int64_t a5, int64_t a6, int64_t a7, int64_t a8, int64_t a9, int64_t a10, int64_t a11, int64_t a12, int64_t a13, int64_t a14, int64_t a15) {
    int64_t argv[16] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15};
    return bronze_super_call(sub_box, this_box, 16, argv);
}
BRONZE_WEAK int64_t bronze_super_call_n(int64_t sub_box, int64_t this_box, uint32_t argc, const int64_t* argv) {
    return bronze_super_call(sub_box, this_box, argc, argv);
}

} // namespace brass::il
