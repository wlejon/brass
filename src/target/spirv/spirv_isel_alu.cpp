// SpirvISel: arithmetic, bitwise, shifts, comparisons, select, conversions.
// Every rule works per 16-byte part, so the same code lowers a scalar (one
// part) and a MIR vector (one SPIR-V vector per 16 bytes).

#include <brass/target/spirv/spirv_isel.hpp>

#include <spirv/unified1/GLSL.std.450.h>

namespace brass::spirv {

namespace {

Type element_of(Type t) { return t.is_vector() ? t.element_type() : t; }

} // namespace

// int_op for integer (and ptr) elements, float_op for float ones; OpNop
// marks an operation the type does not have.
void SpirvISel::lower_binary(const brass::Instruction& inst, spv::Op int_op, spv::Op float_op) {
    Type t = inst.type();
    spv::Op o = element_of(t).is_float() ? float_op : int_op;
    if (o == spv::OpNop) malformed(inst, "operation not defined for this operand type");
    const std::vector<Id> a = ids_of(inst.operand(0), "operand 0");
    const std::vector<Id> b = ids_of(inst.operand(1), "operand 1");
    if (a.size() != b.size()) malformed(inst, "operand widths differ");
    Id pt = part_type(t);
    std::vector<Id> out;
    for (size_t k = 0; k < a.size(); ++k) out.push_back(op(o, pt, {a[k], b[k]}));
    define(inst.result(), std::move(out));
}

// MIR integer division follows x64, where MIN / -1 wraps (and MIN % -1 is
// 0); SPIR-V leaves both undefined, so a divisor that may be -1 is replaced
// by 1 and the wrapped result selected. Division by zero stays undefined:
// a compute shader has nothing to trap with.
void SpirvISel::lower_int_div_rem(const brass::Instruction& inst, bool is_rem, bool is_unsigned) {
    Type t = inst.type();
    if (!t.is_i32() && !t.is_i64()) malformed(inst, "integer division needs i32 or i64");
    Id ty = scalar_type(t);
    uint32_t w = t.is_i32() ? 32 : 64;
    Id a = id_of(inst.operand(0), "operand 0");
    Id d = id_of(inst.operand(1), "operand 1");
    if (is_unsigned) {
        define(inst.result(), op(is_rem ? spv::OpUMod : spv::OpUDiv, ty, {a, d}));
        return;
    }
    spv::Op o = is_rem ? spv::OpSRem : spv::OpSDiv;
    int64_t c = 0;
    if (const_int(inst.operand(1), &c) && c != -1 && c != 0) {
        define(inst.result(), op(o, ty, {a, d}));
        return;
    }
    Id is_m1 = op(spv::OpIEqual, t_bool(), {d, m_.c_int(w, ~0ull)});
    Id safe = op(spv::OpSelect, ty, {is_m1, m_.c_int(w, 1), d});
    Id q = op(o, ty, {a, safe});
    Id wrapped = is_rem ? m_.c_int(w, 0) : op(spv::OpSNegate, ty, {a});
    define(inst.result(), op(spv::OpSelect, ty, {is_m1, wrapped, q}));
}

void SpirvISel::lower_unary_glsl(const brass::Instruction& inst, uint32_t glsl_inst) {
    const std::vector<Id> a = ids_of(inst.operand(0), "operand 0");
    Id pt = part_type(inst.type());
    std::vector<Id> out;
    for (Id x : a) out.push_back(ext(glsl_inst, pt, {x}));
    define(inst.result(), std::move(out));
}

// The comparison yields a bool kept for br_if/select; the i32 0/1 MIR value
// is only materialized (OpSelect) when something reads it as a value.
void SpirvISel::lower_comparison(const brass::Instruction& inst, spv::Op int_op, spv::Op float_op) {
    const Value* lhs = inst.operand(0);
    const Value* rhs = inst.operand(1);
    if (!lhs || !rhs) malformed(inst, "missing operand");
    if (lhs->type().is_vector()) malformed(inst, "vector comparison");
    spv::Op o = lhs->type().is_float() ? float_op : int_op;
    Id b = op(o, t_bool(), {id_of(lhs, "operand 0"), id_of(rhs, "operand 1")});
    bools_[inst.result()] = b;
    if (has_value_uses(inst.result())) {
        define(inst.result(), op(spv::OpSelect, t_u32(), {b, m_.c_u32(1), m_.c_u32(0)}));
    }
}

void SpirvISel::lower_select(const brass::Instruction& inst) {
    Id c = cond_of(inst.operand(0));
    const std::vector<Id> a = ids_of(inst.operand(1), "true value");
    const std::vector<Id> b = ids_of(inst.operand(2), "false value");
    if (a.size() != b.size()) malformed(inst, "operand widths differ");
    Id pt = part_type(inst.type());
    std::vector<Id> out;
    for (size_t k = 0; k < a.size(); ++k) out.push_back(op(spv::OpSelect, pt, {c, a[k], b[k]}));
    define(inst.result(), std::move(out));
}

// sext/zext/trunc are OpSConvert/OpUConvert between the 32- and 64-bit
// integer types; int<->float and f32<->f64 are the matching OpConvert*;
// fptosi truncates toward zero (out-of-range results are undefined in
// SPIR-V, where PTX's cvt.rzi saturates).
void SpirvISel::lower_convert(const brass::Instruction& inst, spv::Op o, Type dst) {
    const Value* src = inst.operand(0);
    if (!src) malformed(inst, "missing operand");
    if (src->type().is_i8() || src->type().is_i16()) {
        fail(std::string(brass::opcode_name(inst.opcode())) + " of an " + std::string(src->type().name()) +
             " value; SPIR-V kernels have no 8/16-bit integer values (use ptx_load_u8/u16)");
    }
    define(inst.result(), op(o, scalar_type(dst), {id_of(src, "operand 0")}));
}

// `add` with a shared-memory pointer operand: Workgroup arrays have no
// numeric address, so the byte offset is accumulated on the SharedRef.
void SpirvISel::lower_add_ptr(const brass::Instruction& inst) {
    bool lhs_shared = shared_.count(inst.operand(0)) != 0;
    if (lhs_shared && shared_.count(inst.operand(1))) malformed(inst, "adds two shared-memory pointers");
    const Value* p = lhs_shared ? inst.operand(0) : inst.operand(1);
    const Value* off = lhs_shared ? inst.operand(1) : inst.operand(0);
    SharedRef r = shared_.at(p);
    int64_t c = 0;
    if (const_int(off, &c)) {
        r.const_off += c;
    } else {
        Id o = as_u32(off, "shared pointer offset");
        r.dyn_off = r.dyn_off ? op(spv::OpIAdd, t_u32(), {r.dyn_off, o}) : o;
    }
    shared_[inst.result()] = r;
}

} // namespace brass::spirv
