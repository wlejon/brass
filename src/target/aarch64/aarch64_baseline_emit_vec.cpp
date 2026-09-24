// AArch64 baseline tier: 128-bit vectors. A vector lives in a 16-byte slot
// and is operated on with NEON through V0-V2. What NEON lacks (i64x2 mul),
// or does differently from the interpreter (float min / max, which it
// computes as (b < a) ? b : a), is built from compares and selects or done
// lane-wise through GPRs; integer vdiv calls a small helper so its edge cases
// (x / 0, INT_MIN / -1) are the interpreter's exactly. Lane accesses
// (extract, insert, shuffle) go through the slots in memory. 256-bit vectors
// are rejected by the pre-scan.
#include "aarch64_baseline_emit_internal.hpp"
#include <cstdint>
#include <limits>

namespace brass::aarch64 {

using namespace brass::codegen;

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

[[noreturn]] void reject(const Instruction& inst, const std::string& why) {
    throw_unsupported(kA64BaselineStage, std::string(opcode_name(inst.opcode())) + ": " + why);
}

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

// Calls helper(dst, operands...) with the slot addresses in X0.. .
void call_lane_helper(AArch64BaselineEmitter& em, const void* helper, const Instruction& inst) {
    auto addr_of = [&](GPR dst, const Value* v) {
        const MemAddress m = em.slot_addr(v, 16);
        if (m.offset >= 0) em.enc.add(dst, m.base, static_cast<uint32_t>(m.offset));
        else em.enc.sub(dst, m.base, static_cast<uint32_t>(-m.offset));
    };
    addr_of(GPR::X0, inst.result());
    for (size_t i = 0; i < inst.operand_count(); ++i) addr_of(static_cast<GPR>(i + 1), inst.operand(i));
    em.call_abs(helper);
}

void emit_binop(AArch64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const TypeKind k = inst.type().kind();
    const Opcode op = inst.opcode();
    const Value* a = inst.operand(0);
    const Value* b = inst.operand(1);

    // Lane-wise through GPRs: i64x2 mul.
    if (k == TypeKind::I64x2 && op == Opcode::vmul) {
        for (int32_t off : {0, 8}) {
            enc.ldr(GPR::X0, em.slot_addr_at(a, off, 8));
            enc.ldr(GPR::X1, em.slot_addr_at(b, off, 8));
            enc.mul(GPR::X0, GPR::X0, GPR::X1);
            enc.str(GPR::X0, em.slot_addr_at(inst.result(), off, 8));
        }
        return;
    }
    if (op == Opcode::vdiv && (k == TypeKind::I32x4 || k == TypeKind::I64x2)) {
        call_lane_helper(em, k == TypeKind::I32x4 ? reinterpret_cast<const void*>(&bl_vdiv_i32x4)
                                                  : reinterpret_cast<const void*>(&bl_vdiv_i64x2), inst);
        return;
    }

    em.load_v(FPR::V0, a);
    em.load_v(FPR::V1, b);
    const FPR d = FPR::V0, x = FPR::V0, y = FPR::V1;
    switch (op) {
        case Opcode::vadd:
            if (k == TypeKind::F32x4) enc.vec_fadd_4s(d, x, y); else if (k == TypeKind::F64x2) enc.vec_fadd_2d(d, x, y);
            else if (k == TypeKind::I32x4) enc.vec_add_4s(d, x, y); else enc.vec_add_2d(d, x, y);
            break;
        case Opcode::vsub:
            if (k == TypeKind::F32x4) enc.vec_fsub_4s(d, x, y); else if (k == TypeKind::F64x2) enc.vec_fsub_2d(d, x, y);
            else if (k == TypeKind::I32x4) enc.vec_sub_4s(d, x, y); else enc.vec_sub_2d(d, x, y);
            break;
        case Opcode::vmul:
            if (k == TypeKind::F32x4) enc.vec_fmul_4s(d, x, y); else if (k == TypeKind::F64x2) enc.vec_fmul_2d(d, x, y);
            else enc.vec_mul_4s(d, x, y);
            break;
        case Opcode::vdiv:
            if (k == TypeKind::F32x4) enc.vec_fdiv_4s(d, x, y); else enc.vec_fdiv_2d(d, x, y);
            break;
        case Opcode::vmin:
        case Opcode::vmax:
            if (k == TypeKind::I32x4) {
                if (op == Opcode::vmin) enc.vec_smin_4s(d, x, y); else enc.vec_smax_4s(d, x, y);
                break;
            }
            // The interpreter's min is (b < a) ? b : a and max (b > a) ? b : a:
            // mask = a > b (min) or b > a (max), then b where the mask is
            // set, a elsewhere. A NaN in either lane compares false, so the
            // lane is a; both zeros give a.
            {
                const FPR lhs = (op == Opcode::vmin) ? x : y;
                const FPR rhs = (op == Opcode::vmin) ? y : x;
                switch (k) {
                    case TypeKind::F32x4: enc.vec_fcmgt_4s(FPR::V2, lhs, rhs); break;
                    case TypeKind::F64x2: enc.vec_fcmgt_2d(FPR::V2, lhs, rhs); break;
                    default: enc.vec_cmgt_2d(FPR::V2, lhs, rhs); break; // I64x2
                }
                enc.vec_bsl(FPR::V2, y, x);
                em.store_v(inst.result(), FPR::V2);
                return;
            }
        case Opcode::vand: enc.vec_and(d, x, y); break;
        case Opcode::vor: enc.vec_orr(d, x, y); break;
        case Opcode::vxor: enc.vec_eor(d, x, y); break;
        default:
            reject(inst, "no binary-op emitter (bug)");
    }
    em.store_v(inst.result(), d);
}

void emit_vector_op(AArch64BaselineEmitter& em, const Instruction& inst) {
    auto& enc = em.enc;
    const Opcode op = inst.opcode();
    switch (op) {
        case Opcode::vadd: case Opcode::vsub: case Opcode::vmul: case Opcode::vdiv:
        case Opcode::vmin: case Opcode::vmax: case Opcode::vand: case Opcode::vor: case Opcode::vxor:
            emit_binop(em, inst);
            return;
        case Opcode::vfma: {
            // fmla is fused, one rounding: the interpreter's std::fma.
            em.load_v(FPR::V0, inst.operand(2));
            em.load_v(FPR::V1, inst.operand(0));
            em.load_v(FPR::V2, inst.operand(1));
            if (inst.type().kind() == TypeKind::F32x4) enc.vec_fmla_4s(FPR::V0, FPR::V1, FPR::V2);
            else enc.vec_fmla_2d(FPR::V0, FPR::V1, FPR::V2);
            em.store_v(inst.result(), FPR::V0);
            return;
        }
        case Opcode::vneg: {
            em.load_v(FPR::V0, inst.operand(0));
            switch (inst.type().kind()) {
                case TypeKind::F32x4: enc.vec_fneg_4s(FPR::V0, FPR::V0); break; // flips the sign bit, NaNs too
                case TypeKind::F64x2: enc.vec_fneg_2d(FPR::V0, FPR::V0); break;
                case TypeKind::I32x4: enc.vec_neg_4s(FPR::V0, FPR::V0); break;
                default: enc.vec_neg_2d(FPR::V0, FPR::V0); break;
            }
            em.store_v(inst.result(), FPR::V0);
            return;
        }
        case Opcode::vnot:
            em.load_v(FPR::V0, inst.operand(0));
            enc.vec_not(FPR::V0, FPR::V0);
            em.store_v(inst.result(), FPR::V0);
            return;
        case Opcode::vsqrt:
            em.load_v(FPR::V0, inst.operand(0));
            if (inst.type().kind() == TypeKind::F32x4) enc.vec_fsqrt_4s(FPR::V0, FPR::V0);
            else enc.vec_fsqrt_2d(FPR::V0, FPR::V0);
            em.store_v(inst.result(), FPR::V0);
            return;
        case Opcode::vload:
            enc.ldr(GPR::X0, em.slot_addr(inst.operand(0), 8));
            enc.ldr_q(FPR::V0, em.based(GPR::X0, inst.offset(), 16));
            em.store_v(inst.result(), FPR::V0);
            return;
        case Opcode::vstore:
            enc.ldr(GPR::X0, em.slot_addr(inst.operand(0), 8));
            em.load_v(FPR::V0, inst.operand(1));
            enc.str_q(FPR::V0, em.based(GPR::X0, inst.offset(), 16));
            return;
        case Opcode::vzero:
            enc.str(GPR::XZR, em.slot_addr_at(inst.result(), 0, 8));
            enc.str(GPR::XZR, em.slot_addr_at(inst.result(), 8, 8));
            return;
        case Opcode::vbroadcast: {
            // Bit copies of the scalar into every lane.
            const Value* s = inst.operand(0);
            const TypeKind k = inst.type().kind();
            if (k == TypeKind::F32x4 || k == TypeKind::I32x4) {
                enc.ldr32(GPR::X0, em.slot_addr(s, 4));
                enc.vec_dup_4s(FPR::V0, GPR::X0);
            } else {
                enc.ldr(GPR::X0, em.slot_addr(s, 8));
                enc.vec_dup_2d(FPR::V0, GPR::X0);
            }
            em.store_v(inst.result(), FPR::V0);
            return;
        }
        case Opcode::vextract_lane: {
            const Type vt = inst.operand(0)->type();
            const int32_t esz = static_cast<int32_t>(vt.element_type().size_in_bytes());
            const int32_t byte = static_cast<int32_t>(inst.lane()) * esz;
            if (esz == 4) {
                enc.ldr32(GPR::X0, em.slot_addr_at(inst.operand(0), byte, 4));
                enc.str32(GPR::X0, em.slot_addr(inst.result(), 4));
            } else {
                enc.ldr(GPR::X0, em.slot_addr_at(inst.operand(0), byte, 8));
                enc.str(GPR::X0, em.slot_addr(inst.result(), 8));
            }
            return;
        }
        case Opcode::vinsert_lane: {
            // The result's slot never shares with an operand's (vectors get
            // dedicated slots), so copy then overwrite the lane.
            em.load_v(FPR::V0, inst.operand(0));
            em.store_v(inst.result(), FPR::V0);
            const int32_t esz = static_cast<int32_t>(inst.type().element_type().size_in_bytes());
            const int32_t byte = static_cast<int32_t>(inst.lane()) * esz;
            if (esz == 4) {
                enc.ldr32(GPR::X0, em.slot_addr(inst.operand(1), 4));
                enc.str32(GPR::X0, em.slot_addr_at(inst.result(), byte, 4));
            } else {
                enc.ldr(GPR::X0, em.slot_addr(inst.operand(1), 8));
                enc.str(GPR::X0, em.slot_addr_at(inst.result(), byte, 8));
            }
            return;
        }
        case Opcode::vshuffle: {
            // shufps / shufpd: lanes 0-1 from operand 0, the rest from
            // operand 1, each chosen by its selector bits (two per lane, one
            // for 2-lane types).
            const TypeKind k = inst.type().kind();
            const uint32_t mask = inst.shuffle_mask();
            const bool four = (k == TypeKind::F32x4 || k == TypeKind::I32x4);
            const int lanes = four ? 4 : 2;
            const int32_t esz = four ? 4 : 8;
            for (int i = 0; i < lanes; ++i) {
                const Value* src = (i < lanes / 2) ? inst.operand(0) : inst.operand(1);
                const uint32_t sel = four ? ((mask >> (2 * i)) & 3u) : ((mask >> i) & 1u);
                const int32_t from = static_cast<int32_t>(sel) * esz;
                const int32_t to = i * esz;
                if (four) {
                    enc.ldr32(GPR::X0, em.slot_addr_at(src, from, 4));
                    enc.str32(GPR::X0, em.slot_addr_at(inst.result(), to, 4));
                } else {
                    enc.ldr(GPR::X0, em.slot_addr_at(src, from, 8));
                    enc.str(GPR::X0, em.slot_addr_at(inst.result(), to, 8));
                }
            }
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

// Every vector among `types`, laid out as a call's arguments, is in a V
// register (as the x64 tier requires XMM): a vector on the stack is left to
// the interpreter.
bool vectors_in_registers(const Target& target, const std::vector<Type>& types) {
    const auto locs = a64_assign_args(target, types, nullptr);
    for (size_t i = 0; i < types.size(); ++i) {
        if (types[i].is_vector() && !locs[i].in_reg) return false;
    }
    return true;
}

} // namespace

void check_aarch64_baseline_vector_inst(const Instruction& inst, const Target& target) {
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
            if (!vectors_in_registers(target, types)) reject(inst, "vector argument passed on the stack");
            return;
        }
        case Opcode::ret:
            return;
        default:
            reject(inst, "vector-typed operand or result");
    }
}

bool emit_baseline_aarch64_vec_op(AArch64BaselineEmitter& em, const Instruction& inst) {
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
            em.load_gpr(GPR::X0, inst.operand(0));
            em.load_v(FPR::V0, inst.operand(1));
            if (bl_is_int32(inst.operand(0)->type())) enc.cbnz32(GPR::X0, done); else enc.cbnz(GPR::X0, done);
            em.load_v(FPR::V0, inst.operand(2));
            em.buffer.bind(done);
            em.store_v(inst.result(), FPR::V0);
            return true;
        }
        case Opcode::load:
            if (!inst.type().is_vector()) return false;
            enc.ldr(GPR::X0, em.slot_addr(inst.operand(0), 8));
            enc.ldr_q(FPR::V0, em.based(GPR::X0, inst.offset(), 16));
            em.store_v(inst.result(), FPR::V0);
            return true;
        case Opcode::load_indexed:
        case Opcode::store_indexed: {
            const bool is_load = (op == Opcode::load_indexed);
            if (is_load ? !inst.type().is_vector() : !inst.operand(2)->type().is_vector()) return false;
            enc.ldr(GPR::X0, em.slot_addr(inst.operand(0), 8));
            const Value* idx = inst.operand(1);
            if (bl_is_int32(idx->type())) enc.ldrsw(GPR::X1, em.slot_addr(idx, 4));
            else enc.ldr(GPR::X1, em.slot_addr(idx, 8));
            enc.mov(GPR::X2, static_cast<uint64_t>(static_cast<int64_t>(inst.scale())));
            enc.madd(GPR::X0, GPR::X1, GPR::X2, GPR::X0);
            if (is_load) {
                enc.ldr_q(FPR::V0, em.based(GPR::X0, inst.offset(), 16));
                em.store_v(inst.result(), FPR::V0);
            } else {
                em.load_v(FPR::V0, inst.operand(2));
                enc.str_q(FPR::V0, em.based(GPR::X0, inst.offset(), 16));
            }
            return true;
        }
        case Opcode::store:
            if (!inst.operand(1)->type().is_vector()) return false;
            enc.ldr(GPR::X0, em.slot_addr(inst.operand(0), 8));
            em.load_v(FPR::V0, inst.operand(1));
            enc.str_q(FPR::V0, em.based(GPR::X0, inst.offset(), 16));
            return true;
        default:
            return false;
    }
}

} // namespace brass::aarch64
