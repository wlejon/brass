// PtxISel intrinsic rules: per-kernel .shared arrays and narrow (8/16-bit)
// global loads/stores, plus the operand helpers shared by every rule
// (compile-time constants, [base + offset] and [base + index*size]
// addressing, B32 lane/barrier operands).
//
// Shared memory model: `ptx_shared_alloc_<T>(count)` declares a
// `.shared .align 16 .<T> smem_<n>[count]` array in the current kernel
// (count must be a compile-time constant) and yields its shared-window
// address as a 64-bit `ptr` value (`mov.u64 %rd, smem_<n>`). That pointer is
// only meaningful to the `ptx_shared_load/store_*` intrinsics, which emit
// `ld.shared`/`st.shared`; MIR `load`/`store` stay `.global` and there is no
// address-space tracking through pointer arithmetic. Plain integer
// arithmetic on the pointer (`add`) is fine as long as the result is again
// used only by the shared intrinsics.

#include "ptx_isel_intrinsics.hpp"

#include <brass/mir/opcodes.hpp>

#include <limits>
#include <stdexcept>
#include <string>

namespace brass::ptx {

// ---------------------------------------------------------------------------
// Operand helpers (PtxISel members)
// ---------------------------------------------------------------------------

bool PtxISel::const_int(const Value* v, int64_t* out) const {
    if (!v || !v->is_instruction()) return false;
    const brass::Instruction* def = v->defining_instruction();
    if (!def) return false;
    if (def->opcode() == brass::Opcode::iconst_i32) { *out = def->imm_i32(); return true; }
    if (def->opcode() == brass::Opcode::iconst_i64) { *out = def->imm_i64(); return true; }
    return false;
}

Operand PtxISel::address_operand(const Value* ptr, const Value* offset, const char* what) {
    Reg base = reg_of(ptr, what);
    if (base.cls != RegClass::B64) {
        throw std::runtime_error(std::string("PtxISel: ") + what + " must be a 64-bit pointer value");
    }
    if (!offset) return Operand::addr(base, 0);

    int64_t c = 0;
    if (const_int(offset, &c) && c >= std::numeric_limits<int32_t>::min() && c <= std::numeric_limits<int32_t>::max()) {
        return Operand::addr(base, static_cast<int32_t>(c));
    }
    Reg off = reg_of(offset, "byte offset");
    if (off.cls == RegClass::B32) {
        Reg wide = fn_->new_b64();
        emit(Inst::make(Opcode::cvt, Type::s64).from(Type::s32).dst(wide).src(off));
        off = wide;
    }
    Reg sum = fn_->new_b64();
    emit(Inst::make(Opcode::add, Type::s64).dst(sum).src(base).src(off));
    return Operand::addr(sum, 0);
}

Operand PtxISel::indexed_operand(const Value* ptr, const Value* index, uint32_t elem_size, const char* what) {
    Reg base = reg_of(ptr, what);
    if (base.cls != RegClass::B64) {
        throw std::runtime_error(std::string("PtxISel: ") + what + " must be a 64-bit pointer value");
    }
    int64_t c = 0;
    if (const_int(index, &c)) {
        int64_t disp = c * static_cast<int64_t>(elem_size);
        if (disp >= std::numeric_limits<int32_t>::min() && disp <= std::numeric_limits<int32_t>::max()) {
            return Operand::addr(base, static_cast<int32_t>(disp));
        }
    }
    Reg idx = reg_of(index, "index");
    if (idx.cls == RegClass::B32) {
        Reg wide = fn_->new_b64();
        emit(Inst::make(Opcode::cvt, Type::s64).from(Type::s32).dst(wide).src(idx));
        idx = wide;
    }
    if (elem_size > 1) {
        Reg scaled = fn_->new_b64();
        int shift = (elem_size == 2) ? 1 : (elem_size == 4) ? 2 : (elem_size == 8) ? 3 : -1;
        if (shift >= 0) emit(Inst::make(Opcode::shl, Type::b64).dst(scaled).src(idx).src(Operand::imm(shift)));
        else emit(Inst::make(Opcode::mul, Type::s64).lo().dst(scaled).src(idx).src(Operand::imm(elem_size)));
        idx = scaled;
    }
    Reg sum = fn_->new_b64();
    emit(Inst::make(Opcode::add, Type::s64).dst(sum).src(base).src(idx));
    return Operand::addr(sum, 0);
}

Operand PtxISel::lane_operand(const Value* v, const char* what) {
    int64_t c = 0;
    if (const_int(v, &c)) {
        if (c < 0 || c > std::numeric_limits<uint32_t>::max()) {
            throw std::runtime_error(std::string("PtxISel: ") + what + " constant does not fit in 32 bits");
        }
        return Operand::imm(c);
    }
    Reg r = reg_of(v, what);
    if (r.cls == RegClass::B32) return Operand::reg(r);
    if (r.cls == RegClass::B64) {
        Reg narrow = fn_->new_b32();
        emit(Inst::make(Opcode::cvt, Type::u32).from(Type::u64).dst(narrow).src(r));
        return Operand::reg(narrow);
    }
    throw std::runtime_error(std::string("PtxISel: ") + what + " must be an integer value");
}

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------

namespace {

constexpr uint32_t kSharedAlign = 16; // v4 accesses stay legal at any element offset multiple of 16

uint32_t elem_bytes(Type t) { return bit_width(t) / 8; }

} // namespace

// ptx_shared_alloc_<T>(const count) -> ptr:   .shared .align 16 .<T> smem_<n>[count]; mov.u64 %rd, smem_<n>
template <Type T>
void PtxISel::Intrinsics::shared_alloc(PtxISel& isel, const brass::Instruction& inst) {
    int64_t count = 0;
    if (!isel.const_int(inst.operand(0), &count) || count <= 0 || count > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("PtxISel: " + std::string(inst.symbol()) +
                                 " requires a positive compile-time constant element count (iconst_i32/iconst_i64 operand)");
    }
    Reg dst = isel.result_reg(inst);
    if (dst.cls != RegClass::B64) isel.malformed(inst, "shared array address must be a ptr/i64 result");
    std::string name = "smem_" + std::to_string(isel.fn_->shared.size());
    isel.fn_->add_shared(T, name, static_cast<uint32_t>(count), kSharedAlign);
    isel.emit(Inst::make(Opcode::mov, Type::u64).dst(dst).src(Operand::symbol(name)));
}
template void PtxISel::Intrinsics::shared_alloc<Type::f32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_alloc<Type::u32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_alloc<Type::f64>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_alloc<Type::u64>(PtxISel&, const brass::Instruction&);

// ptx_shared_load_<T>(ptr[, byte_offset]) -> T:   ld.shared.<T> d, [ptr + off]
template <Type T>
void PtxISel::Intrinsics::shared_load(PtxISel& isel, const brass::Instruction& inst) {
    Reg dst = isel.result_reg(inst);
    if (dst.cls != reg_class_for(T)) isel.malformed(inst, "result type does not match the shared element type");
    isel.emit(Inst::make(Opcode::ld, T).space(StateSpace::shared)
                  .dst(dst)
                  .src(isel.address_operand(inst.operand(0), inst.operand(1), "shared pointer")));
}
template void PtxISel::Intrinsics::shared_load<Type::f32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_load<Type::u32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_load<Type::f64>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_load<Type::u64>(PtxISel&, const brass::Instruction&);

// ptx_shared_load_<T>_indexed(ptr, index) -> T:   ld.shared.<T> d, [ptr + index * sizeof(T)]
template <Type T>
void PtxISel::Intrinsics::shared_load_indexed(PtxISel& isel, const brass::Instruction& inst) {
    Reg dst = isel.result_reg(inst);
    if (dst.cls != reg_class_for(T)) isel.malformed(inst, "result type does not match the shared element type");
    if (!inst.operand(1)) isel.malformed(inst, "missing index");
    isel.emit(Inst::make(Opcode::ld, T).space(StateSpace::shared)
                  .dst(dst)
                  .src(isel.indexed_operand(inst.operand(0), inst.operand(1), elem_bytes(T), "shared pointer")));
}
template void PtxISel::Intrinsics::shared_load_indexed<Type::f32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_load_indexed<Type::u32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_load_indexed<Type::f64>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_load_indexed<Type::u64>(PtxISel&, const brass::Instruction&);

// ptx_shared_store_<T>(ptr, value[, byte_offset]):   st.shared.<T> [ptr + off], value
template <Type T>
void PtxISel::Intrinsics::shared_store(PtxISel& isel, const brass::Instruction& inst) {
    Reg value = isel.reg_of(inst.operand(1), "value");
    if (value.cls != reg_class_for(T)) isel.malformed(inst, "value type does not match the shared element type");
    Operand addr = isel.address_operand(inst.operand(0), inst.operand(2), "shared pointer");
    isel.emit(Inst::make(Opcode::st, T).space(StateSpace::shared).src(addr).src(value));
}
template void PtxISel::Intrinsics::shared_store<Type::f32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_store<Type::u32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_store<Type::f64>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_store<Type::u64>(PtxISel&, const brass::Instruction&);

// ptx_shared_store_<T>_indexed(ptr, index, value):   st.shared.<T> [ptr + index * sizeof(T)], value
template <Type T>
void PtxISel::Intrinsics::shared_store_indexed(PtxISel& isel, const brass::Instruction& inst) {
    if (!inst.operand(1)) isel.malformed(inst, "missing index");
    Reg value = isel.reg_of(inst.operand(2), "value");
    if (value.cls != reg_class_for(T)) isel.malformed(inst, "value type does not match the shared element type");
    Operand addr = isel.indexed_operand(inst.operand(0), inst.operand(1), elem_bytes(T), "shared pointer");
    isel.emit(Inst::make(Opcode::st, T).space(StateSpace::shared).src(addr).src(value));
}
template void PtxISel::Intrinsics::shared_store_indexed<Type::f32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_store_indexed<Type::u32>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_store_indexed<Type::f64>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::shared_store_indexed<Type::u64>(PtxISel&, const brass::Instruction&);

// ---------------------------------------------------------------------------
// Narrow global loads/stores. PTX zero-extends .u8/.u16 and sign-extends
// .s8/.s16 into the 32-bit destination register, so the i32 result is the
// extended value.
// ---------------------------------------------------------------------------

// ptx_load_<u8|s8|u16|s16>(ptr[, byte_offset]) -> i32:   ld.global.<T> %r, [ptr + off]
template <Type T>
void PtxISel::Intrinsics::load_narrow(PtxISel& isel, const brass::Instruction& inst) {
    Reg dst = isel.result_reg(inst);
    if (dst.cls != RegClass::B32) isel.malformed(inst, "narrow load result must be i32");
    isel.emit(Inst::make(Opcode::ld, T).space(StateSpace::global)
                  .dst(dst)
                  .src(isel.address_operand(inst.operand(0), inst.operand(1), "pointer")));
}
template void PtxISel::Intrinsics::load_narrow<Type::u8>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::load_narrow<Type::s8>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::load_narrow<Type::u16>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::load_narrow<Type::s16>(PtxISel&, const brass::Instruction&);

// ptx_store_<u8|u16>(ptr, i32 value[, byte_offset]):   st.global.<T> [ptr + off], %r   (low bits stored)
template <Type T>
void PtxISel::Intrinsics::store_narrow(PtxISel& isel, const brass::Instruction& inst) {
    Reg value = isel.reg_of(inst.operand(1), "value");
    if (value.cls != RegClass::B32) isel.malformed(inst, "narrow store value must be i32");
    Operand addr = isel.address_operand(inst.operand(0), inst.operand(2), "pointer");
    isel.emit(Inst::make(Opcode::st, T).space(StateSpace::global).src(addr).src(value));
}
template void PtxISel::Intrinsics::store_narrow<Type::u8>(PtxISel&, const brass::Instruction&);
template void PtxISel::Intrinsics::store_narrow<Type::u16>(PtxISel&, const brass::Instruction&);

} // namespace brass::ptx
