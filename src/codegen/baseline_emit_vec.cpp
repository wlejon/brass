// x64 baseline tier: 128-bit vectors. A vector lives in a 16-byte slot and
// is operated on with SSE (SSE4.1 for pmulld / pminsd / pmaxsd, as tier 2),
// through XMM0 / XMM1. Lane-wise integer ops SSE lacks (i64x2 mul / min /
// max) go through GPRs; integer vdiv and vfma call a small helper, so their
// edge cases (x / 0, INT_MIN / -1, a single-rounding fma) are the
// interpreter's exactly. 256-bit vectors are rejected by the pre-scan.
#include "baseline_emit_internal.hpp"
#include <cmath>
#include <cstdint>
#include <limits>

namespace brass::codegen {

using namespace brass::x64;

namespace {

// Lane helpers, with the interpreter's semantics (value_vec.cpp).
template <typename T>
T bl_sdiv(T a, T b) {
    if (b == 0) return 0;
    if (a == std::numeric_limits<T>::min() && b == -1) return a;
    return a / b;
}
void bl_vdiv_i32x4(int32_t* d, const int32_t* a, const int32_t* b) {
    for (int i = 0; i < 4; ++i) d[i] = bl_sdiv(a[i], b[i]);
}
void bl_vdiv_i64x2(int64_t* d, const int64_t* a, const int64_t* b) {
    for (int i = 0; i < 2; ++i) d[i] = bl_sdiv(a[i], b[i]);
}
void bl_vfma_f32x4(float* d, const float* a, const float* b, const float* c) {
    for (int i = 0; i < 4; ++i) d[i] = std::fma(a[i], b[i], c[i]);
}
void bl_vfma_f64x2(double* d, const double* a, const double* b, const double* c) {
    for (int i = 0; i < 2; ++i) d[i] = std::fma(a[i], b[i], c[i]);
}

[[noreturn]] void reject(const Instruction& inst, const std::string& why) {
    throw_unsupported(kX64BaselineStage, std::string(opcode_name(inst.opcode())) + ": " + why);
}

// The effective type of a memory access (as baseline_emit_ops.cpp).
Type access_type(const Instruction& inst, Type value_type) {
    Type mt = inst.memory_type();
    return mt.is_void() ? value_type : mt;
}

// The scalar a vector of type `vt` takes as a lane (vbroadcast, vinsert_lane):
// its element type (as the verifier requires); narrow integers are i32s.
bool lane_source_ok(Type vt, Type s) {
    switch (vt.kind()) {
        case TypeKind::F32x4: return s.kind() == TypeKind::F32;
        case TypeKind::F64x2: return s.kind() == TypeKind::F64;
        case TypeKind::I32x4: return bl_is_int32(s);
        case TypeKind::I64x2: return s.kind() == TypeKind::I64;
        default: return false;
    }
}

void check_vector_op(const Instruction& inst) {
    const Opcode op = inst.opcode();
    Type vt;
    if (op == Opcode::vstore) vt = inst.operand(1)->type();
    else if (op == Opcode::vextract_lane) vt = inst.operand(0)->type();
    else vt = inst.type();
    if (!vt.is_v128()) reject(inst, "vector type " + brass::to_string(vt) + " (only 128-bit vectors are compiled)");
    const bool flt = vt.element_type().is_float();

    // Every vector operand has the vector's type.
    for (size_t i = 0; i < inst.operand_count(); ++i) {
        const Value* o = inst.operand(i);
        if (!o) reject(inst, "missing operand");
        if (o->type().is_vector() && o->type() != vt) reject(inst, "mixed vector operand types");
    }
    switch (op) {
        case Opcode::vfma:
        case Opcode::vsqrt:
            if (!flt) reject(inst, "integer " + brass::to_string(vt));
            return;
        case Opcode::vbroadcast:
            if (!lane_source_ok(vt, inst.operand(0)->type())) reject(inst, "scalar type " + brass::to_string(inst.operand(0)->type()));
            return;
        case Opcode::vinsert_lane:
            if (inst.lane() >= vt.vector_lanes()) reject(inst, "lane out of range");
            if (!lane_source_ok(vt, inst.operand(1)->type())) reject(inst, "scalar type " + brass::to_string(inst.operand(1)->type()));
            return;
        case Opcode::vextract_lane:
            if (inst.lane() >= vt.vector_lanes()) reject(inst, "lane out of range");
            if (inst.type() != vt.element_type()) reject(inst, "result type " + brass::to_string(inst.type()));
            return;
        case Opcode::vload:
        case Opcode::vstore:
            if (inst.operand(0)->type().is_vector()) reject(inst, "vector address");
            return;
        default:
            return;
    }
}

// Base in RAX, sign-extended index in RCX (as baseline_emit_ops.cpp).
MemAddress indexed_address(X64BaselineEmitter& em, const Instruction& inst) {
    em.enc.mov(GPR::RAX, em.slot_addr(inst.operand(0)));
    const Value* idx = inst.operand(1);
    if (bl_is_int32(idx->type())) em.enc.movsxd(GPR::RCX, em.slot_addr(idx));
    else em.enc.mov(GPR::RCX, em.slot_addr(idx));
    return MemAddress::base_index(GPR::RAX, GPR::RCX, scale_from_int(inst.scale()), inst.offset());
}

void load_v(X64BaselineEmitter& em, XMM x, const Value* v) { em.enc.movups(x, em.slot_addr(v)); }
void store_v(X64BaselineEmitter& em, const Value* v, XMM x) { em.enc.movups(em.slot_addr(v), x); }

// Calls helper(dst, operands...) with the slot addresses in the C argument
// registers. RSP is 16-byte aligned between instructions of this tier, and
// on Win64 the frame's bottom 32 bytes are the callee's shadow space.
void call_lane_helper(X64BaselineEmitter& em, const void* helper, const Instruction& inst) {
    const auto& gprs = em.cc.arg_gprs();
    em.enc.lea(gprs[0], em.slot_addr(inst.result()));
    for (size_t i = 0; i < inst.operand_count(); ++i) em.enc.lea(gprs[i + 1], em.slot_addr(inst.operand(i)));
    em.call_abs(helper);
}

void emit_binop(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const TypeKind k = inst.type().kind();
    const Opcode op = inst.opcode();
    const Value* a = inst.operand(0);
    const Value* b = inst.operand(1);

    // Lane-wise through GPRs: i64x2 mul / min / max.
    if (k == TypeKind::I64x2 && (op == Opcode::vmul || op == Opcode::vmin || op == Opcode::vmax)) {
        for (int32_t off : {0, 8}) {
            enc.mov(GPR::RAX, em.slot_addr_at(a, off));
            if (op == Opcode::vmul) {
                enc.imul(GPR::RAX, em.slot_addr_at(b, off));
            } else {
                enc.mov(GPR::RCX, em.slot_addr_at(b, off));
                enc.cmp(GPR::RAX, GPR::RCX);
                if (op == Opcode::vmin) enc.cmovg(GPR::RAX, GPR::RCX);
                else enc.cmovl(GPR::RAX, GPR::RCX);
            }
            enc.mov(em.slot_addr_at(inst.result(), off), GPR::RAX);
        }
        return;
    }
    if (op == Opcode::vdiv && (k == TypeKind::I32x4 || k == TypeKind::I64x2)) {
        call_lane_helper(em, k == TypeKind::I32x4 ? reinterpret_cast<const void*>(&bl_vdiv_i32x4)
                                                  : reinterpret_cast<const void*>(&bl_vdiv_i64x2), inst);
        return;
    }

    // Float min / max: the interpreter's (b < a) ? b : a is minps with b as
    // the destination (it returns the source, a, when either is NaN or both
    // are zeros); max likewise.
    const bool flt_minmax = (op == Opcode::vmin || op == Opcode::vmax) &&
                            (k == TypeKind::F32x4 || k == TypeKind::F64x2);
    load_v(em, XMM::XMM0, flt_minmax ? b : a);
    load_v(em, XMM::XMM1, flt_minmax ? a : b);
    const XMM d = XMM::XMM0, s = XMM::XMM1;
    switch (op) {
        case Opcode::vadd:
            if (k == TypeKind::F32x4) enc.addps(d, s); else if (k == TypeKind::F64x2) enc.addpd(d, s);
            else if (k == TypeKind::I32x4) enc.paddd(d, s); else enc.paddq(d, s);
            break;
        case Opcode::vsub:
            if (k == TypeKind::F32x4) enc.subps(d, s); else if (k == TypeKind::F64x2) enc.subpd(d, s);
            else if (k == TypeKind::I32x4) enc.psubd(d, s); else enc.psubq(d, s);
            break;
        case Opcode::vmul:
            if (k == TypeKind::F32x4) enc.mulps(d, s); else if (k == TypeKind::F64x2) enc.mulpd(d, s);
            else enc.pmulld(d, s);
            break;
        case Opcode::vdiv:
            if (k == TypeKind::F32x4) enc.divps(d, s); else enc.divpd(d, s);
            break;
        case Opcode::vmin:
            if (k == TypeKind::F32x4) enc.minps(d, s); else if (k == TypeKind::F64x2) enc.minpd(d, s);
            else enc.pminsd(d, s);
            break;
        case Opcode::vmax:
            if (k == TypeKind::F32x4) enc.maxps(d, s); else if (k == TypeKind::F64x2) enc.maxpd(d, s);
            else enc.pmaxsd(d, s);
            break;
        case Opcode::vand: enc.pand(d, s); break;
        case Opcode::vor: enc.por(d, s); break;
        case Opcode::vxor: enc.pxor(d, s); break;
        default:
            reject(inst, "no binary-op emitter (bug)");
    }
    store_v(em, inst.result(), d);
}

// Writes scalar `s` into lane `lane` of the (already copied) vector slot of
// `dst`, of vector type `vt`.
void store_lane(X64BaselineEmitter& em, const Value* dst, Type vt, uint32_t lane, const Value* s) {
    auto& enc = em.enc;
    const int32_t esz = static_cast<int32_t>(vt.element_type().size_in_bytes());
    const MemAddress at = em.slot_addr_at(dst, static_cast<int32_t>(lane) * esz);
    if (esz == 4) {
        enc.mov32(GPR::RAX, em.slot_addr(s));
        enc.mov32(at, GPR::RAX);
    } else {
        enc.mov(GPR::RAX, em.slot_addr(s));
        enc.mov(at, GPR::RAX);
    }
}

void emit_vector_op(X64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const Opcode op = inst.opcode();
    switch (op) {
        case Opcode::vadd: case Opcode::vsub: case Opcode::vmul: case Opcode::vdiv:
        case Opcode::vmin: case Opcode::vmax: case Opcode::vand: case Opcode::vor: case Opcode::vxor:
            emit_binop(em, inst);
            return;
        case Opcode::vfma:
            call_lane_helper(em, inst.type().kind() == TypeKind::F32x4
                                     ? reinterpret_cast<const void*>(&bl_vfma_f32x4)
                                     : reinterpret_cast<const void*>(&bl_vfma_f64x2), inst);
            return;
        case Opcode::vneg: {
            const TypeKind k = inst.type().kind();
            if (k == TypeKind::F32x4 || k == TypeKind::F64x2) {
                // Flip each sign bit, as -x does (NaNs included).
                load_v(em, XMM::XMM0, inst.operand(0));
                enc.pcmpeqd(XMM::XMM1, XMM::XMM1);
                if (k == TypeKind::F32x4) enc.pslld(XMM::XMM1, 31); else enc.psllq(XMM::XMM1, 63);
                enc.pxor(XMM::XMM0, XMM::XMM1);
            } else {
                enc.pxor(XMM::XMM0, XMM::XMM0);
                load_v(em, XMM::XMM1, inst.operand(0));
                if (k == TypeKind::I32x4) enc.psubd(XMM::XMM0, XMM::XMM1); else enc.psubq(XMM::XMM0, XMM::XMM1);
            }
            store_v(em, inst.result(), XMM::XMM0);
            return;
        }
        case Opcode::vnot:
            load_v(em, XMM::XMM0, inst.operand(0));
            enc.pcmpeqd(XMM::XMM1, XMM::XMM1);
            enc.pxor(XMM::XMM0, XMM::XMM1);
            store_v(em, inst.result(), XMM::XMM0);
            return;
        case Opcode::vsqrt:
            load_v(em, XMM::XMM1, inst.operand(0));
            if (inst.type().kind() == TypeKind::F32x4) enc.sqrtps(XMM::XMM0, XMM::XMM1);
            else enc.sqrtpd(XMM::XMM0, XMM::XMM1);
            store_v(em, inst.result(), XMM::XMM0);
            return;
        case Opcode::vload:
            enc.mov(GPR::RAX, em.slot_addr(inst.operand(0)));
            enc.movups(XMM::XMM0, MemAddress::base_disp(GPR::RAX, inst.offset()));
            store_v(em, inst.result(), XMM::XMM0);
            return;
        case Opcode::vstore:
            enc.mov(GPR::RAX, em.slot_addr(inst.operand(0)));
            load_v(em, XMM::XMM0, inst.operand(1));
            enc.movups(MemAddress::base_disp(GPR::RAX, inst.offset()), XMM::XMM0);
            return;
        case Opcode::vzero:
            enc.xorps(XMM::XMM0, XMM::XMM0);
            store_v(em, inst.result(), XMM::XMM0);
            return;
        case Opcode::vbroadcast: {
            const Type vt = inst.type();
            const Value* s = inst.operand(0);
            switch (vt.kind()) {
                case TypeKind::F32x4:
                    enc.movss(XMM::XMM0, em.slot_addr(s));
                    enc.shufps(XMM::XMM0, XMM::XMM0, 0x00);
                    break;
                case TypeKind::I32x4:
                    enc.movss(XMM::XMM0, em.slot_addr(s));
                    enc.pshufd(XMM::XMM0, XMM::XMM0, 0x00);
                    break;
                default: // F64x2, I64x2: bit copies of the 8-byte scalar
                    enc.movsd(XMM::XMM0, em.slot_addr(s));
                    enc.shufpd(XMM::XMM0, XMM::XMM0, 0x00);
                    break;
            }
            store_v(em, inst.result(), XMM::XMM0);
            return;
        }
        case Opcode::vextract_lane: {
            const Type vt = inst.operand(0)->type();
            const int32_t esz = static_cast<int32_t>(vt.element_type().size_in_bytes());
            const MemAddress src = em.slot_addr_at(inst.operand(0), static_cast<int32_t>(inst.lane()) * esz);
            if (esz == 4) { enc.mov32(GPR::RAX, src); enc.mov32(em.slot_addr(inst.result()), GPR::RAX); }
            else { enc.mov(GPR::RAX, src); enc.mov(em.slot_addr(inst.result()), GPR::RAX); }
            return;
        }
        case Opcode::vinsert_lane:
            load_v(em, XMM::XMM0, inst.operand(0));
            store_v(em, inst.result(), XMM::XMM0);
            store_lane(em, inst.result(), inst.type(), inst.lane(), inst.operand(1));
            return;
        case Opcode::vshuffle: {
            // Lanes 0-1 from operand 0, 2-3 from operand 1, two selector
            // bits each (one each for 2-lane types): shufps / shufpd exactly.
            const TypeKind k = inst.type().kind();
            load_v(em, XMM::XMM0, inst.operand(0));
            load_v(em, XMM::XMM1, inst.operand(1));
            if (k == TypeKind::F32x4 || k == TypeKind::I32x4) {
                enc.shufps(XMM::XMM0, XMM::XMM1, static_cast<uint8_t>(inst.shuffle_mask() & 0xFF));
            } else {
                enc.shufpd(XMM::XMM0, XMM::XMM1, static_cast<uint8_t>(inst.shuffle_mask() & 0x3));
            }
            store_v(em, inst.result(), XMM::XMM0);
            return;
        }
        default:
            reject(inst, "no vector emitter (bug)");
    }
}

bool touches_vector(const Instruction& inst) {
    if (inst.produces_value() && inst.type().is_vector()) return true;
    for (size_t i = 0; i < inst.operand_count(); ++i) {
        if (inst.operand(i) && inst.operand(i)->type().is_vector()) return true;
    }
    return false;
}

} // namespace

bool bl_vectors_in_registers(const Target& target, const CallingConvention& cc, const std::vector<Type>& types) {
    size_t xmm_idx = 0;
    for (size_t i = 0; i < types.size(); ++i) {
        const Type t = types[i];
        if (target.is_windows()) {
            if (t.is_vector() && i >= 4) return false;
        } else if (bl_in_xmm(t)) {
            if (t.is_vector() && xmm_idx >= cc.arg_xmms().size()) return false;
            ++xmm_idx;
        }
    }
    return true;
}

void check_x64_baseline_vector_inst(const Function& fn, const Instruction& inst, const Target& target,
                                    const CallingConvention& cc) {
    (void)fn;
    const Opcode op = inst.opcode();
    for (const auto* v : inst.state_map()) {
        if (v && v->type().is_vector()) reject(inst, "vector value in a state map");
    }
    if (is_vector_op(op)) {
        check_vector_op(inst);
        return;
    }
    if (!touches_vector(inst)) return;
    for (size_t i = 0; i < inst.operand_count(); ++i) {
        const Value* o = inst.operand(i);
        if (o && o->type().is_v256()) reject(inst, "256-bit vector operand");
    }
    if (inst.produces_value() && inst.type().is_v256()) reject(inst, "256-bit vector result");
    switch (op) {
        case Opcode::select:
            if (inst.operand(0)->type().is_vector() || inst.operand(1)->type() != inst.type() ||
                inst.operand(2)->type() != inst.type()) {
                reject(inst, "vector select operands");
            }
            return;
        case Opcode::load:
        case Opcode::load_indexed:
            if (!inst.type().is_vector() || access_type(inst, inst.type()) != inst.type()) {
                reject(inst, "vector load with a different memory type");
            }
            for (size_t i = 0; i < inst.operand_count(); ++i) {
                if (inst.operand(i)->type().is_vector()) reject(inst, "vector address");
            }
            return;
        case Opcode::store:
        case Opcode::store_indexed: {
            const size_t vi = (op == Opcode::store) ? 1 : 2;
            const Type vt = inst.operand(vi)->type();
            if (!vt.is_vector() || access_type(inst, vt) != vt) reject(inst, "vector store with a different memory type");
            for (size_t i = 0; i < vi; ++i) {
                if (inst.operand(i)->type().is_vector()) reject(inst, "vector address");
            }
            return;
        }
        case Opcode::call:
        case Opcode::patchable_call:
        case Opcode::call_indirect: {
            std::vector<Type> types;
            const size_t first = (op == Opcode::call_indirect) ? 1 : 0;
            if (first == 1 && inst.operand(0)->type().is_vector()) reject(inst, "vector callee");
            for (size_t i = first; i < inst.operand_count(); ++i) types.push_back(inst.operand(i)->type());
            if (!bl_vectors_in_registers(target, cc, types)) reject(inst, "vector argument passed on the stack");
            return;
        }
        case Opcode::ret:
            return;
        default:
            reject(inst, "vector-typed operand or result");
    }
}

bool emit_baseline_x64_vec_op(X64BaselineEmitter& em, const Instruction& inst) {
    const Opcode op = inst.opcode();
    if (is_vector_op(op)) {
        emit_vector_op(em, inst);
        return true;
    }
    auto& enc = em.enc;
    switch (op) {
        case Opcode::select: {
            if (!inst.type().is_vector()) return false;
            Label done = em.buffer.create_label();
            em.test_cond(inst.operand(0));
            load_v(em, XMM::XMM0, inst.operand(1)); // movups leaves the flags
            enc.jne(done);
            load_v(em, XMM::XMM0, inst.operand(2));
            em.buffer.bind(done);
            store_v(em, inst.result(), XMM::XMM0);
            return true;
        }
        case Opcode::load:
            if (!inst.type().is_vector()) return false;
            enc.mov(GPR::RAX, em.slot_addr(inst.operand(0)));
            enc.movups(XMM::XMM0, MemAddress::base_disp(GPR::RAX, inst.offset()));
            store_v(em, inst.result(), XMM::XMM0);
            return true;
        case Opcode::load_indexed:
            if (!inst.type().is_vector()) return false;
            enc.movups(XMM::XMM0, indexed_address(em, inst));
            store_v(em, inst.result(), XMM::XMM0);
            return true;
        case Opcode::store:
            if (!inst.operand(1)->type().is_vector()) return false;
            load_v(em, XMM::XMM0, inst.operand(1));
            enc.mov(GPR::RAX, em.slot_addr(inst.operand(0)));
            enc.movups(MemAddress::base_disp(GPR::RAX, inst.offset()), XMM::XMM0);
            return true;
        case Opcode::store_indexed:
            if (!inst.operand(2)->type().is_vector()) return false;
            load_v(em, XMM::XMM0, inst.operand(2));
            enc.movups(indexed_address(em, inst), XMM::XMM0);
            return true;
        default:
            return false;
    }
}

} // namespace brass::codegen
