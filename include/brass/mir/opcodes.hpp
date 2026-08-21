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

    // Selection
    select,

    // Memory
    load,
    store,
    load_indexed,
    store_indexed,

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
    ret,
    unreachable
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
bool is_select(Opcode op) noexcept;
bool has_side_effects(Opcode op) noexcept;

std::ostream& operator<<(std::ostream& os, Opcode op);

} // namespace brass
