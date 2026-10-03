// SpirvISel: MIR vector ops. A MIR vector is one SPIR-V vector per 16 bytes
// ("part"): f32x4/i32x4 one 4-lane vector, f64x2/i64x2 one 2-lane vector,
// the 256-bit types two parts (SPIR-V shader vectors stop at 4 lanes). Each
// op applies to every part with SPIR-V's native vector arithmetic. The same
// rules lower scalar fmin/fmax (one scalar "part").

#include <brass/target/spirv/spirv_isel.hpp>

#include <spirv/unified1/GLSL.std.450.h>

namespace brass::spirv {

namespace {

Type element_of(Type t) { return t.is_vector() ? t.element_type() : t; }

// The x64 shufps/shufpd mask encoding, per 128-bit half (= part): a 4-lane
// part takes lanes 0..1 from `a` and 2..3 from `b`, 2-bit selectors; a
// 2-lane part takes lane 0 from `a` and lane 1 from `b`, one bit each (the
// bit index also depends on the half). Returns the OpVectorShuffle
// component (b's components follow a's).
uint32_t shuffle_component(uint32_t lanes_per_part, uint32_t half, uint32_t in_half, uint32_t mask) {
    if (lanes_per_part == 4) {
        uint32_t sel = (mask >> (2 * in_half)) & 3;
        return in_half >= 2 ? 4 + sel : sel;
    }
    uint32_t sel = (mask >> (half * 2 + in_half)) & 1;
    return in_half == 1 ? 2 + sel : sel;
}

} // namespace

void SpirvISel::lower_vector_binary(const brass::Instruction& inst, spv::Op int_op, spv::Op float_op) {
    if (!inst.type().is_vector()) malformed(inst, "result is not a vector type");
    lower_binary(inst, int_op, float_op);
}

// fmin/fmax and vmin/vmax: NMin/NMax for floats (the non-NaN operand wins,
// as PTX min/max.f32 and fminf do), SMin/SMax for integer lanes.
void SpirvISel::lower_vector_minmax(const brass::Instruction& inst, bool is_max) {
    bool fl = element_of(inst.type()).is_float();
    uint32_t f = fl ? (is_max ? GLSLstd450NMax : GLSLstd450NMin) : (is_max ? GLSLstd450SMax : GLSLstd450SMin);
    const std::vector<Id> a = ids_of(inst.operand(0), "operand 0");
    const std::vector<Id> b = ids_of(inst.operand(1), "operand 1");
    if (a.size() != b.size()) malformed(inst, "operand widths differ");
    Id pt = part_type(inst.type());
    std::vector<Id> out;
    for (size_t k = 0; k < a.size(); ++k) out.push_back(ext(f, pt, {a[k], b[k]}));
    define(inst.result(), std::move(out));
}

void SpirvISel::lower_vector_fma(const brass::Instruction& inst) {
    bool fl = element_of(inst.type()).is_float();
    const std::vector<Id> a = ids_of(inst.operand(0), "operand 0");
    const std::vector<Id> b = ids_of(inst.operand(1), "operand 1");
    const std::vector<Id> c = ids_of(inst.operand(2), "operand 2");
    if (a.size() != b.size() || a.size() != c.size()) malformed(inst, "operand widths differ");
    Id pt = part_type(inst.type());
    std::vector<Id> out;
    for (size_t k = 0; k < a.size(); ++k) {
        out.push_back(fl ? ext(GLSLstd450Fma, pt, {a[k], b[k], c[k]})
                         : op(spv::OpIAdd, pt, {op(spv::OpIMul, pt, {a[k], b[k]}), c[k]}));
    }
    define(inst.result(), std::move(out));
}

void SpirvISel::lower_vector_unary(const brass::Instruction& inst) {
    bool fl = element_of(inst.type()).is_float();
    bool is_sqrt = inst.opcode() == brass::Opcode::vsqrt;
    if (is_sqrt && !fl) malformed(inst, "vsqrt requires a float vector");
    const std::vector<Id> a = ids_of(inst.operand(0), "operand 0");
    Id pt = part_type(inst.type());
    std::vector<Id> out;
    for (Id x : a) {
        if (is_sqrt) out.push_back(ext(GLSLstd450Sqrt, pt, {x}));
        else out.push_back(op(fl ? spv::OpFNegate : spv::OpSNegate, pt, {x}));
    }
    define(inst.result(), std::move(out));
}

// Bitwise ops on float lanes (the sign-mask idiom) go through an integer
// vector of the same lane width with OpBitcast on both sides.
void SpirvISel::lower_vector_bitwise(const brass::Instruction& inst, spv::Op o) {
    Type t = inst.type();
    if (!t.is_vector()) malformed(inst, "result is not a vector type");
    Type e = t.element_type();
    Id pt = part_type(t);
    uint32_t lanes = static_cast<uint32_t>(16 / e.size_in_bytes());
    Id it = e.is_float() ? m_.t_vector(m_.t_int(static_cast<uint32_t>(e.size_in_bytes() * 8)), lanes) : pt;
    auto to_int = [&](Id x) { return e.is_float() ? op(spv::OpBitcast, it, {x}) : x; };

    const std::vector<Id> a = ids_of(inst.operand(0), "operand 0");
    std::vector<Id> b;
    if (o != spv::OpNot) b = ids_of(inst.operand(1), "operand 1");
    std::vector<Id> out;
    for (size_t k = 0; k < a.size(); ++k) {
        Id r = o == spv::OpNot ? op(o, it, {to_int(a[k])}) : op(o, it, {to_int(a[k]), to_int(b.at(k))});
        out.push_back(e.is_float() ? op(spv::OpBitcast, pt, {r}) : r);
    }
    define(inst.result(), std::move(out));
}

void SpirvISel::lower_vbroadcast(const brass::Instruction& inst) {
    Type t = inst.type();
    if (!t.is_vector()) malformed(inst, "result is not a vector type");
    Id s = id_of(inst.operand(0), "scalar operand");
    Id pt = part_type(t);
    Id r = m_.new_id();
    Inst c(spv::OpCompositeConstruct, pt, r);
    for (uint32_t i = 0; i < 16 / t.element_type().size_in_bytes(); ++i) c.id(s);
    emit(std::move(c));
    define(inst.result(), std::vector<Id>(part_count(t), r));
}

void SpirvISel::lower_vzero(const brass::Instruction& inst) {
    Type t = inst.type();
    if (!t.is_vector()) malformed(inst, "result is not a vector type");
    define(inst.result(), std::vector<Id>(part_count(t), m_.c_null(part_type(t))));
}

void SpirvISel::lower_vextract_lane(const brass::Instruction& inst) {
    const Value* vec = inst.operand(0);
    if (!vec || !vec->type().is_vector()) malformed(inst, "missing vector operand");
    uint32_t per = static_cast<uint32_t>(16 / vec->type().element_type().size_in_bytes());
    uint32_t lane = inst.lane();
    if (lane >= vec->type().vector_lanes()) malformed(inst, "lane index out of range");
    Id part = ids_of(vec, "vector operand").at(lane / per);
    Id r = m_.new_id();
    emit(Inst(spv::OpCompositeExtract, scalar_type(vec->type().element_type()), r).id(part).lit(lane % per));
    define(inst.result(), r);
}

void SpirvISel::lower_vinsert_lane(const brass::Instruction& inst) {
    const Value* vec = inst.operand(0);
    if (!vec || !vec->type().is_vector()) malformed(inst, "missing vector operand");
    uint32_t per = static_cast<uint32_t>(16 / vec->type().element_type().size_in_bytes());
    uint32_t lane = inst.lane();
    if (lane >= vec->type().vector_lanes()) malformed(inst, "lane index out of range");
    std::vector<Id> parts = ids_of(vec, "vector operand");
    Id s = id_of(inst.operand(1), "scalar operand");
    Id r = m_.new_id();
    emit(Inst(spv::OpCompositeInsert, part_type(vec->type()), r).id(s).id(parts[lane / per]).lit(lane % per));
    parts[lane / per] = r;
    define(inst.result(), std::move(parts));
}

void SpirvISel::lower_vshuffle(const brass::Instruction& inst) {
    Type t = inst.type();
    if (!t.is_vector()) malformed(inst, "result is not a vector type");
    const std::vector<Id> a = ids_of(inst.operand(0), "operand 0");
    const std::vector<Id> b = ids_of(inst.operand(1), "operand 1");
    if (a.size() != b.size()) malformed(inst, "operand widths differ");
    uint32_t per = static_cast<uint32_t>(16 / t.element_type().size_in_bytes());
    Id pt = part_type(t);
    std::vector<Id> out;
    for (uint32_t half = 0; half < a.size(); ++half) {
        Id r = m_.new_id();
        Inst s(spv::OpVectorShuffle, pt, r);
        s.id(a[half]).id(b[half]);
        for (uint32_t i = 0; i < per; ++i) s.lit(shuffle_component(per, half, i, inst.shuffle_mask()));
        emit(std::move(s));
        out.push_back(r);
    }
    define(inst.result(), std::move(out));
}

} // namespace brass::spirv
