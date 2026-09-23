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
#include <type_traits>

#if defined(_WIN32)
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

// Bytes of vector state per register (the widest vector type).
inline constexpr size_t kFastVecBytes = 32;

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
    BcReg resume_dst_reg = kNoReg;
    runtime::BrassCoroFrame* c_frame = nullptr;
};

// A resolved call site. Valid while `epoch` matches the interpreter's
// resolve epoch (module, symbol and patch changes) and `registry_gen`
// matches runtime::registry_generation() (dispatch-table and tiering
// registry changes); otherwise it is resolved again before use.
struct FastCallTarget {
    uint64_t epoch = 0;
    uint64_t registry_gen = 0;
    FastFnInfo* callee = nullptr;            // bytecode callee
    const Function* callee_mir = nullptr;    // its MIR, for native routing
    runtime::FunctionHandle* handle = nullptr;
    const FastHostFn* host = nullptr;
    uintptr_t indirect_ptr = 0;              // call_indirect: the pointer resolved
    std::string name;
};

// Per bytecode function state kept by one interpreter: call-site caches and
// the function's tiering counters, resolved once instead of per call.
struct FastFnInfo {
    const BytecodeFunction* bfn = nullptr;
    const Function* mir_fn = nullptr;
    uint32_t num_regs = 1;
    bool has_vector_params = false;
    runtime::TieringFeedback* feedback = nullptr;
    uint64_t feedback_gen = 0;
    std::vector<FastCallTarget> calls;

    runtime::TieringFeedback& tiering() {
        const uint64_t gen = runtime::registry_generation();
        if (BRASS_UNLIKELY(feedback == nullptr || feedback_gen != gen)) {
            feedback = &runtime::TieringRegistry::instance().get_or_create(bfn->name);
            feedback_gen = gen;
        }
        return *feedback;
    }
};

// Bump allocator for register files and alloca_: a frame takes a mark
// before allocating and releases it on exit, so both cost a pointer bump
// instead of a heap allocation (or a large C-stack alloca).
class FastAllocaArena {
public:
    struct Mark {
        size_t chunk = 0;
        size_t offset = 0;
    };

    Mark mark() const noexcept { return {cur_, off_}; }
    void release(Mark m) noexcept {
        cur_ = m.chunk;
        off_ = m.offset;
    }

    // Zero-filled, `align`-aligned (a power of two, 0 meaning 1).
    void* allocate(size_t size, size_t align) {
        if (align == 0) align = 1;
        if (BRASS_LIKELY(cur_ < chunks_.size())) {
            Chunk& c = chunks_[cur_];
            const uintptr_t base = reinterpret_cast<uintptr_t>(c.data.get());
            const size_t start = ((base + off_ + align - 1) & ~(static_cast<uintptr_t>(align) - 1)) - base;
            if (BRASS_LIKELY(start + size <= c.size)) {
                off_ = start + size;
                uint8_t* p = c.data.get() + start;
                std::memset(p, 0, size);
                return p;
            }
        }
        return allocate_slow(size, align);
    }

private:
    void* allocate_slow(size_t size, size_t align);

    static constexpr size_t kChunkSize = 64 * 1024;
    struct Chunk {
        std::unique_ptr<uint8_t[]> data;
        size_t size = 0;
    };
    std::vector<Chunk> chunks_;
    size_t cur_ = 0;
    size_t off_ = 0;
};

struct FastFrame {
    const BytecodeFunction* bfn = nullptr;
    const Function* mir_fn = nullptr;
    FastFnInfo* info = nullptr;
    uint64_t* registers = nullptr;
    uint32_t num_registers = 0;
    uint32_t pc = 0;
    FastFrame* caller = nullptr;

    // Vector state, kFastVecBytes per register, allocated on first use.
    uint8_t* vector_regs = nullptr;
    std::vector<uint8_t> vector_storage;

    uint8_t* ensure_vector_regs() {
        if (!vector_regs) {
            vector_storage.assign(static_cast<size_t>(std::max<uint32_t>(num_registers, 1)) * kFastVecBytes, 0);
            vector_regs = vector_storage.data();
        }
        return vector_regs;
    }
};

struct FrameGuard {
    FastInterpreter& interp;
    FastFrame& frame;
    FastInterpreter* prev_interp{nullptr};
    FastAllocaArena::Mark alloca_mark;

    // `mark` is the arena position before the frame's register file was
    // allocated; everything the frame allocates is released on exit.
    FrameGuard(FastInterpreter& in, FastFrame& f, FastAllocaArena::Mark mark)
        : interp(in), frame(f), alloca_mark(mark) {
        prev_interp = FastInterpreter::current();
        FastInterpreter::set_current(&in);
        frame.caller = interp.current_frame();
        interp.set_current_frame(&frame);
        interp.inc_call_depth();
    }
    FrameGuard(FastInterpreter& in, FastFrame& f) : FrameGuard(in, f, in.alloca_arena_->mark()) {}
    ~FrameGuard() {
        interp.alloca_arena_->release(alloca_mark);
        FastInterpreter::set_current(prev_interp);
        interp.set_current_frame(frame.caller);
        interp.dec_call_depth();
    }
};

template <typename T>
inline bool mul_overflows(T a, T b, T& res) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, &res);
#else
    res = static_cast<T>(a * b);
    if constexpr (std::is_signed_v<T>) {
        if (a == 0 || b == 0) return false;
        if ((a == -1 && b == std::numeric_limits<T>::min()) || (b == -1 && a == std::numeric_limits<T>::min())) return true;
        return res / b != a;
    } else {
        return a != 0 && res / a != b;
    }
#endif
}

inline bool is_narrow_int_type(Type t) noexcept {
    return t == Type::i32() || t == Type::i16() || t == Type::i8();
}

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

// The RuntimeValue of register `reg` (typed by the function's register
// types), including its vector state.
inline RuntimeValue fast_reg_value(const FastFrame& frame, uint32_t reg) {
    const BytecodeFunction& fn = *frame.bfn;
    Type t = reg < fn.register_types.size() ? fn.register_types[reg] : Type::i64();
    if (t.is_vector()) {
        const uint8_t* bytes = frame.vector_regs ? frame.vector_regs + static_cast<size_t>(reg) * kFastVecBytes : nullptr;
        if (!bytes) return RuntimeValue::from_bits(t, frame.registers[reg]);
        return t.is_v128() ? RuntimeValue::from_v128(t, bytes) : RuntimeValue::from_v256(t, bytes);
    }
    if (t.is_gcref()) return RuntimeValue::from_gcref(static_cast<uintptr_t>(frame.registers[reg]));
    return RuntimeValue::from_bits(t, frame.registers[reg]);
}

// Writes `v` to register `reg` in the register form of its type: narrow
// integers are held zero-extended from 32 bits.
inline void fast_set_reg(FastFrame& frame, uint32_t reg, const RuntimeValue& v) {
    const BytecodeFunction& fn = *frame.bfn;
    Type t = reg < fn.register_types.size() ? fn.register_types[reg] : Type::i64();
    uint64_t bits = v.raw_bits();
    if (is_narrow_int_type(t) || t == Type::f32()) bits &= 0xFFFFFFFFull;
    frame.registers[reg] = bits;
    if (t.is_vector() || v.is_vector()) {
        uint8_t* vregs = frame.ensure_vector_regs();
        std::memcpy(vregs + static_cast<size_t>(reg) * kFastVecBytes, v.vec_bytes(), kFastVecBytes);
    }
}

inline RuntimeValue marshal_return_value(const BytecodeFunction& fn, uint64_t raw_bits, const uint8_t* vregs, uint32_t ret_reg) {
    if (fn.return_type.is_void()) {
        return RuntimeValue::from_void();
    }
    if (fn.return_type.is_vector()) {
        if (!vregs) return RuntimeValue::from_bits(fn.return_type, raw_bits);
        const uint8_t* src_bytes = vregs + static_cast<size_t>(ret_reg) * kFastVecBytes;
        if (fn.return_type.is_v128()) {
            return RuntimeValue::from_v128(fn.return_type, src_bytes);
        }
        return RuntimeValue::from_v256(fn.return_type, src_bytes);
    }
    return RuntimeValue::from_bits(fn.return_type, raw_bits);
}

} // namespace brass
