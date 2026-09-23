#pragma once
// An integer divide fault (#DE: division by zero, or INT_MIN / -1 where the
// target traps) raised by JIT code is reported as an error instead of
// killing the process with an unhandled 0xC0000094 / SIGFPE and no
// diagnostic. MIR leaves integer division by zero undefined, so generated
// code carries no zero check; the fault is caught around the native call.
// AArch64 code, which cannot fault on a division, traps a zero divisor with
// a dedicated brk that is reported the same way.

namespace brass {

using NativeBody = void (*)(void* ctx);

// Runs body(ctx). Returns nullptr when it returns normally, or a description
// of the arithmetic fault it raised ("integer division by zero"). Frames
// between here and the fault are abandoned without running destructors, so
// the caller should report the error and exit.
const char* call_catching_arith_faults(NativeBody body, void* ctx);

// Same, for any callable.
template <typename F>
const char* call_catching_arith_faults(F f) {
    return call_catching_arith_faults(static_cast<NativeBody>([](void* ctx) { (*static_cast<F*>(ctx))(); }), &f);
}

} // namespace brass
