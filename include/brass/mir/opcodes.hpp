#pragma once

#include <cstdint>
#include <string_view>
#include <iosfwd>

namespace brass {

enum class Opcode : uint16_t {
    // Constants
    iconst_i32,
    iconst_i64,
    fconst_f64,
    patchable_const_i32,
    patchable_const_i64,

    // Conversions
    sext_i64,
    zext_i64,
    trunc_i32,
    trunc_i8,
    fptosi_i32,
    fptosi_i64,
    fptosi_i32_f32,
    fptosi_i64_f32,
    sitofp_f64_i32,
    sitofp_f64_i64,
    sitofp_f32_i32,
    sitofp_f32_i64,
    fptrunc_f32_f64,
    fpext_f64_f32,
    bitcast_i64_f64,
    bitcast_f64_i64,
    // The bits of a tagged value (tagged -> i64) and a tagged value from bits
    // (i64 -> tagged). The first reads a snapshot: when the value is a
    // reference, the bits name where the object was, so passes never move it
    // earlier, merge it with another, or hoist it across a GC point
    // (gc_refs.hpp).
    bitcast_i64_tagged,
    bitcast_tagged_i64,

    // Arithmetic / Logic
    add,
    sub,
    mul,
    fma_f32,
    fma_f64,
    sqrt_f32,
    sqrt_f64,
    floor_f32,
    floor_f64,
    ceil_f32,
    ceil_f64,
    round_f32,
    round_f64,
    fabs_f32,
    fabs_f64,
    fmin_f32,
    fmin_f64,
    fmax_f32,
    fmax_f64,
    sdiv,
    udiv,
    smod,
    umod,
    neg,
    and_,
    or_,
    xor_,
    shl,
    lshr,
    ashr,
    not_,
    clz,
    ctz,
    popcnt,

    // Comparison
    eq,
    ne,
    slt,
    ult,
    sle,
    ule,
    sgt,
    ugt,
    sge,
    uge,

    // Overflow-checked Arithmetic
    sadd_overflow,
    ssub_overflow,
    smul_overflow,
    uadd_overflow,
    usub_overflow,
    umul_overflow,

    // Selection
    select,

    // Memory
    alloca_,
    load,
    store,
    load_indexed,
    store_indexed,
    write_barrier,

    // Pinned-register reads and writes. `pinned_tls_read` yields the address
    // held in the register the module pins for its thread-local block (x64:
    // R13, aarch64: X28), `pinned_tls_write` stores one into it — only the
    // module entry does that, after fetching the block from the runtime —
    // and `read_sp` yields the stack pointer after the prologue, for the
    // stack-limit check. See Module::pinned_tls_register().
    pinned_tls_read,
    pinned_tls_write,
    read_sp,

    // Calls & Safepoints
    call,
    call_indirect,
    patchable_call,
    safepoint,
    // keep_alive %v: a use of %v that does nothing. It keeps %v live, and so
    // rooted if it is a gcref or tagged value, up to this point.
    keep_alive,
    func_addr,

    // Speculation
    guard,
    resume_point,

    // Terminators
    br,
    br_if,
    switch_,
    ret,
    unreachable,

    // Exceptions & Unwinding
    throw_,
    invoke,
    landing_pad,
    resume,

    // Vector Opcodes
    vadd,
    vsub,
    vmul,
    vdiv,
    vfma,
    vneg,
    vmin,
    vmax,
    vsqrt,
    vand,
    vor,
    vxor,
    vnot,
    vload,
    vstore,
    vbroadcast,
    vextract_lane,
    vinsert_lane,
    vshuffle,
    vzero,

    // Coroutines
    coro_create,
    coro_suspend,
    coro_resume,
    coro_destroy
};

std::string_view opcode_name(Opcode op) noexcept;
bool is_terminator(Opcode op) noexcept;
bool is_branch(Opcode op) noexcept;
bool is_call(Opcode op) noexcept;
bool is_constant(Opcode op) noexcept;
bool is_conversion(Opcode op) noexcept;
bool is_arithmetic(Opcode op) noexcept;
bool is_bitwise(Opcode op) noexcept;
bool is_comparison(Opcode op) noexcept;
bool is_memory(Opcode op) noexcept;
bool is_write_barrier(Opcode op) noexcept;
bool is_select(Opcode op) noexcept;
bool is_vector_op(Opcode op) noexcept;
bool is_coro_op(Opcode op) noexcept;
bool is_coro_suspend(Opcode op) noexcept;
bool is_coro_resume(Opcode op) noexcept;
bool has_side_effects(Opcode op) noexcept;

std::ostream& operator<<(std::ostream& os, Opcode op);

} // namespace brass
