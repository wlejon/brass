#pragma once

// A front end's native runtime function, registered once for every tier of
// a program: native code (the baseline compiler, tier-2 engines) links it by
// address, and the interpreters call it through a host function that unpacks
// their RuntimeValues into the C++ parameters. The MIR signature comes from
// the C++ type, so the two can never disagree.
//
//   register_native_function(pipeline, "my_rt_add", &my_rt_add);
//   register_native_function(engine, "my_rt_add", &my_rt_add);

#include <brass/codegen/jit_exec.hpp>
#include <brass/interpreter/interpreter.hpp>
#include <brass/interpreter/value.hpp>
#include <brass/mir/types.hpp>
#include <brass/runtime/multi_tier_pipeline.hpp>
#include <brass/vm/fast_interpreter.hpp>

#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace brass::runtime {

namespace native_detail {

template <typename T>
T unpack(const std::vector<RuntimeValue>& args, size_t i) {
    if (i >= args.size()) return T{};
    if constexpr (std::is_same_v<T, double>) {
        return args[i].as_f64();
    } else if constexpr (std::is_same_v<T, float>) {
        return args[i].as_f32();
    } else if constexpr (std::is_same_v<T, bool>) {
        return args[i].as_i32() != 0;
    } else if constexpr (std::is_integral_v<T> && sizeof(T) <= 4) {
        return static_cast<T>(args[i].as_i32());
    } else if constexpr (std::is_pointer_v<T>) {
        return reinterpret_cast<T>(args[i].as_ptr());
    } else {
        return static_cast<T>(args[i].as_u64());
    }
}

template <typename Ret>
RuntimeValue pack(Ret val) {
    if constexpr (std::is_same_v<Ret, double>) {
        return RuntimeValue::from_f64(val);
    } else if constexpr (std::is_same_v<Ret, float>) {
        return RuntimeValue::from_f32(val);
    } else if constexpr (std::is_same_v<Ret, bool>) {
        return RuntimeValue::from_i32(val ? 1 : 0);
    } else if constexpr (std::is_integral_v<Ret> && sizeof(Ret) <= 4) {
        return RuntimeValue::from_i32(static_cast<int32_t>(val));
    } else if constexpr (std::is_pointer_v<Ret>) {
        return RuntimeValue::from_ptr(reinterpret_cast<uintptr_t>(val));
    } else {
        return RuntimeValue::from_u64(static_cast<uint64_t>(val));
    }
}

template <typename Ret, typename... Args, size_t... Is>
RuntimeValue invoke(Ret (*fn)(Args...), const std::vector<RuntimeValue>& args, std::index_sequence<Is...>) {
    if constexpr (std::is_void_v<Ret>) {
        fn(unpack<Args>(args, Is)...);
        return RuntimeValue::from_void();
    } else {
        return pack(fn(unpack<Args>(args, Is)...));
    }
}

} // namespace native_detail

// The MIR type of a C++ parameter or result.
template <typename T>
Type native_type() {
    if constexpr (std::is_void_v<T>) {
        return Type::void_type();
    } else if constexpr (std::is_same_v<T, double>) {
        return Type::f64();
    } else if constexpr (std::is_same_v<T, float>) {
        return Type::f32();
    } else if constexpr (std::is_pointer_v<T>) {
        return Type::ptr();
    } else if constexpr (std::is_integral_v<T> && sizeof(T) <= 4) {
        return Type::i32();
    } else {
        static_assert(std::is_integral_v<T> && sizeof(T) == 8, "a native function's values are scalars");
        return Type::i64();
    }
}

// `fn` as an interpreter's host function: the fast interpreter's (FastHostFn)
// or the reference interpreter's (HostFn).
template <typename Interp, typename Ret, typename... Args>
auto host_function(Ret (*fn)(Args...)) {
    return [fn](Interp&, const std::vector<RuntimeValue>& args) -> RuntimeValue {
        return native_detail::invoke(fn, args, std::index_sequence_for<Args...>{});
    };
}

template <typename Ret, typename... Args>
void register_native_function(MultiTierPipeline& pipeline, std::string_view name, Ret (*fn)(Args...)) {
    pipeline.register_external_symbol(name, reinterpret_cast<void*>(fn));
    pipeline.register_external_function(name, FastHostFn(host_function<FastInterpreter>(fn)));
}

template <typename Ret, typename... Args>
void register_native_function(codegen::JitExecutionEngine& engine, std::string_view name, Ret (*fn)(Args...)) {
    engine.register_external_symbol(name, reinterpret_cast<void*>(fn));
    engine.register_function_signature(name, native_type<Ret>(), {native_type<Args>()...});
}

} // namespace brass::runtime
