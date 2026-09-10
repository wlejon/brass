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
    fptosi_i32,
    fptosi_i64,
    sitofp_f64_i32,
    sitofp_f64_i64,
    bitcast_i64_f64,
    bitcast_f64_i64,

    // Arithmetic / Logic
    add,
    sub,
    mul,
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
    load,
    store,
    load_indexed,
    store_indexed,
    write_barrier,

    // Calls & Safepoints
    call,
    call_indirect,
    patchable_call,
    safepoint,

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
