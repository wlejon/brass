#pragma once

#include <brass/vm/fast_interpreter.hpp>
#include <brass/gc/generational_gc.hpp>
#include <brass/gc/runtime_gc.hpp>
#include <brass/runtime/coroutine.hpp>
#include <brass/runtime/deopt.hpp>
#include <brass/runtime/tiering.hpp>
#include <cmath>
#include <cstring>
#include <bit>
#include <limits>
#include <algorithm>

#if defined(_MSC_VER)
#include <malloc.h>
#define BRASS_ALLOCA _alloca
#else
#include <alloca.h>
#define BRASS_ALLOCA alloca
#endif

#if defined(__GNUC__) || defined(__clang__)
#define BRASS_LIKELY(x) __builtin_expect(!!(x), 1)
#define BRASS_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define BRASS_LIKELY(x) (x)
#define BRASS_UNLIKELY(x) (x)
#endif

namespace brass {

struct FastCoroSuspendException : public std::exception {
    uint64_t yielded_val = 0;
    uint32_t state_id = 0;

    explicit FastCoroSuspendException(uint64_t yv, uint32_t sid = 0)
        : yielded_val(yv), state_id(sid) {}

    const char* what() const noexcept override {
        return "FastInterpreter coroutine suspended";
    }
};

struct FastCoroState {
    uint32_t state_id = 0;
    bool is_done = false;
    const BytecodeFunction* bfn = nullptr;
    const Function* mir_fn = nullptr;
    uint32_t pc = 0;
    std::vector<uint64_t> registers;
    std::vector<uint8_t> vector_storage;
    uint64_t yielded_val = 0;
    uint64_t resume_arg = 0;
    uint8_t resume_dst_reg = 0;
    runtime::BrassCoroFrame* c_frame = nullptr;
};

struct FastFrame {
    const BytecodeFunction* bfn = nullptr;
    const Function* mir_fn = nullptr;
    uint64_t* registers = nullptr;
    uint32_t num_registers = 0;
    FastFrame* caller = nullptr;
    uint32_t pc = 0;

    struct AllocaBlock {
        std::vector<uint8_t> storage;
    };
    std::vector<AllocaBlock> alloca_storage;

    alignas(32) uint8_t* vector_regs = nullptr;
    std::vector<uint8_t> vector_storage;

    void* allocate_alloca(size_t size, size_t align = 16) {
        if (align == 0) align = 16;
        size_t total = size + align;
        alloca_storage.emplace_back(AllocaBlock{std::vector<uint8_t>(total, 0)});
        uintptr_t addr = reinterpret_cast<uintptr_t>(alloca_storage.back().storage.data());
        uintptr_t aligned_addr = (addr + align - 1) & ~(align - 1);
        return reinterpret_cast<void*>(aligned_addr);
    }

    uint8_t* ensure_vector_regs() {
        if (!vector_regs) {
            size_t count = std::max<size_t>(num_registers, 32);
            size_t size = count * 32 + 32;
            vector_storage.assign(size, 0);
            uintptr_t addr = reinterpret_cast<uintptr_t>(vector_storage.data());
            uintptr_t aligned = (addr + 31) & ~static_cast<uintptr_t>(31);
            vector_regs = reinterpret_cast<uint8_t*>(aligned);
        }
        return vector_regs;
    }
};

struct FrameGuard {
    FastInterpreter& interp;
    FastFrame& frame;

    FrameGuard(FastInterpreter& in, FastFrame& f) : interp(in), frame(f) {
        frame.caller = interp.current_frame();
        interp.set_current_frame(&frame);
        interp.inc_call_depth();
    }
    ~FrameGuard() {
        interp.set_current_frame(frame.caller);
        interp.dec_call_depth();
    }
};

inline float get_f32(uint64_t reg) noexcept {
    float f;
    uint32_t bits = static_cast<uint32_t>(reg);
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

inline uint64_t put_f32(float f) noexcept {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(float));
    return static_cast<uint64_t>(bits);
}

inline double get_f64(uint64_t reg) noexcept {
    double d;
    std::memcpy(&d, &reg, sizeof(double));
    return d;
}

inline uint64_t put_f64(double d) noexcept {
    uint64_t reg = 0;
    std::memcpy(&reg, &d, sizeof(double));
    return reg;
}

inline RuntimeValue marshal_return_value(const BytecodeFunction& fn, uint64_t raw_bits, const uint8_t* vregs, uint8_t ret_reg) {
    if (fn.return_type.is_void()) {
        return RuntimeValue::from_void();
    }
    if (fn.return_type.is_vector()) {
        if (!vregs) return RuntimeValue::from_bits(fn.return_type, raw_bits);
        const uint8_t* src_bytes = vregs + ret_reg * 32;
        if (fn.return_type.is_v128()) {
            return RuntimeValue::from_v128(fn.return_type, src_bytes);
        } else {
            return RuntimeValue::from_v256(fn.return_type, src_bytes);
        }
    }
    return RuntimeValue::from_bits(fn.return_type, raw_bits);
}

} // namespace brass
