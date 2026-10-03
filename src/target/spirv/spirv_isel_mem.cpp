// SpirvISel: memory. MIR `ptr` values are 64-bit buffer device addresses;
// every access is `OpConvertUToPtr` of the byte address to a
// PhysicalStorageBuffer pointer of the accessed type, then OpLoad/OpStore
// with an Aligned operand (the element size; 16 for vector parts). The same
// pointer can therefore be read as f32 in one place and as i32x4 in another
// with no per-type buffer views.
//
// Shared memory is Workgroup arrays reached through SharedRef provenance
// (the ptx_shared_* intrinsics, spirv_isel_intrinsics_mem.cpp, use the
// helpers at the end of this file).

#include <brass/target/spirv/spirv_isel.hpp>

namespace brass::spirv {

namespace {

uint32_t log2_of(uint32_t v) {
    uint32_t s = 0;
    while ((1u << s) < v) ++s;
    return s;
}

} // namespace

Id SpirvISel::address(Id base_u64, int64_t offset) {
    if (offset == 0) return base_u64;
    return op(spv::OpIAdd, t_u64(), {base_u64, m_.c_int(64, static_cast<uint64_t>(offset))});
}

// base + sext(index) * scale + offset; a constant index folds into the
// displacement.
Id SpirvISel::indexed_address(const brass::Instruction& inst, const Value* base, const Value* index) {
    if (!base || !index) malformed(inst, "missing operand");
    Id b = id_of(base, "base pointer");
    int64_t c = 0;
    if (const_int(index, &c)) return address(b, c * inst.scale() + inst.offset());
    Id idx = as_u64(index, "index");
    if (inst.scale() > 1) idx = op(spv::OpIMul, t_u64(), {idx, m_.c_int(64, inst.scale())});
    return address(op(spv::OpIAdd, t_u64(), {b, idx}), inst.offset());
}

Id SpirvISel::load_global(Id addr, Id type, uint32_t align) {
    Id p = op(spv::OpConvertUToPtr, psb_pointer(type), {addr});
    Id r = m_.new_id();
    emit(Inst(spv::OpLoad, type, r).id(p).lit(spv::MemoryAccessAlignedMask).lit(align));
    return r;
}

void SpirvISel::store_global(Id addr, Id value, Id type, uint32_t align) {
    Id p = op(spv::OpConvertUToPtr, psb_pointer(type), {addr});
    emit(Inst(spv::OpStore).id(p).id(value).lit(spv::MemoryAccessAlignedMask).lit(align));
}

void SpirvISel::lower_load(const brass::Instruction& inst) {
    Type t = inst.type();
    if (t.is_vector()) {
        lower_vload(inst);
        return;
    }
    Id base = id_of(inst.operand(0), "pointer");
    define(inst.result(), load_global(address(base, inst.offset()), scalar_type(t), static_cast<uint32_t>(t.size_in_bytes())));
}

void SpirvISel::lower_store(const brass::Instruction& inst) {
    const Value* value = inst.operand(1);
    if (!inst.operand(0) || !value) malformed(inst, "missing operand");
    if (value->type().is_vector()) {
        lower_vstore(inst);
        return;
    }
    Id base = id_of(inst.operand(0), "pointer");
    store_global(address(base, inst.offset()), id_of(value, "value"), scalar_type(value->type()),
                 static_cast<uint32_t>(value->type().size_in_bytes()));
}

// One 16-byte OpLoad per part (vec4 of 32-bit lanes, vec2 of 64-bit ones);
// 256-bit types are two parts 16 bytes apart, as on PTX.
void SpirvISel::lower_vload(const brass::Instruction& inst) {
    Type t = inst.type();
    if (!t.is_vector()) malformed(inst, "vector load of a non-vector type");
    Id base = id_of(inst.operand(0), "pointer");
    Id pt = part_type(t);
    std::vector<Id> parts;
    for (uint32_t k = 0; k < part_count(t); ++k) {
        parts.push_back(load_global(address(base, inst.offset() + 16 * static_cast<int64_t>(k)), pt, 16));
    }
    define(inst.result(), std::move(parts));
}

void SpirvISel::lower_vstore(const brass::Instruction& inst) {
    const Value* value = inst.operand(1);
    if (!inst.operand(0) || !value) malformed(inst, "missing operand");
    Type t = value->type();
    if (!t.is_vector()) malformed(inst, "vector store of a non-vector value");
    Id base = id_of(inst.operand(0), "pointer");
    const std::vector<Id> parts = ids_of(value, "value");
    Id pt = part_type(t);
    for (uint32_t k = 0; k < parts.size(); ++k) {
        store_global(address(base, inst.offset() + 16 * static_cast<int64_t>(k)), parts[k], pt, 16);
    }
}

void SpirvISel::lower_load_indexed(const brass::Instruction& inst) {
    Type t = inst.type();
    if (t.is_vector()) malformed(inst, "indexed vector load");
    Id addr = indexed_address(inst, inst.operand(0), inst.operand(1));
    define(inst.result(), load_global(addr, scalar_type(t), static_cast<uint32_t>(t.size_in_bytes())));
}

void SpirvISel::lower_store_indexed(const brass::Instruction& inst) {
    const Value* value = inst.operand(2);
    if (!value) malformed(inst, "missing value");
    if (value->type().is_vector()) malformed(inst, "indexed vector store");
    Id addr = indexed_address(inst, inst.operand(0), inst.operand(1));
    store_global(addr, id_of(value, "value"), scalar_type(value->type()),
                 static_cast<uint32_t>(value->type().size_in_bytes()));
}

// ---------------------------------------------------------------------------
// Shared memory
// ---------------------------------------------------------------------------

const SpirvISel::SharedRef& SpirvISel::shared_ref(const Value* v, const char* what) const {
    auto it = shared_.find(v);
    if (it == shared_.end()) {
        fail(std::string(what) + " is not a shared-memory pointer; on SPIR-V the ptx_shared_* intrinsics take the result "
                                 "of ptx_shared_alloc_* (optionally offset with add), traced at compile time");
    }
    return it->second;
}

// Pointer to element (ref + extra) of the Workgroup array. `extra` is a
// byte offset, or an element index when `extra_is_index`; either may be
// null. The access must have the array's element size (an i32 access of an
// f32 array is a bitcast; a 64-bit access of a 32-bit array is refused).
Id SpirvISel::shared_element_pointer(const SharedRef& ref, const Value* extra, bool extra_is_index, Type access) {
    auto esz = static_cast<uint32_t>(ref.elem.size_in_bytes());
    if (access.size_in_bytes() != esz) {
        fail("a " + std::to_string(access.size_in_bytes()) + "-byte shared access of an array of " +
             std::string(ref.elem.name()) + "; the SPIR-V Workgroup array is typed, so accesses must match its element size");
    }
    if (ref.const_off % esz != 0) {
        fail("shared pointer offset " + std::to_string(ref.const_off) + " is not a multiple of the element size " +
             std::to_string(esz));
    }
    uint32_t shift = log2_of(esz);
    int64_t const_index = ref.const_off / esz;
    Id dyn = 0;
    auto add_dyn = [&](Id x) { dyn = dyn ? op(spv::OpIAdd, t_u32(), {dyn, x}) : x; };
    if (ref.dyn_off) add_dyn(op(spv::OpShiftRightLogical, t_u32(), {ref.dyn_off, m_.c_u32(shift)}));
    if (extra) {
        int64_t c = 0;
        if (const_int(extra, &c)) {
            if (!extra_is_index && c % esz != 0) {
                fail("shared byte offset " + std::to_string(c) + " is not a multiple of the element size " + std::to_string(esz));
            }
            const_index += extra_is_index ? c : c / esz;
        } else {
            Id x = as_u32(extra, extra_is_index ? "shared index" : "shared byte offset");
            add_dyn(extra_is_index ? x : op(spv::OpShiftRightLogical, t_u32(), {x, m_.c_u32(shift)}));
        }
    }
    Id index = m_.c_u32(static_cast<uint32_t>(const_index));
    if (dyn) index = const_index ? op(spv::OpIAdd, t_u32(), {dyn, index}) : dyn;
    return op(spv::OpAccessChain, m_.t_pointer(spv::StorageClassWorkgroup, scalar_type(ref.elem)), {ref.var, index});
}

Id SpirvISel::shared_load(const SharedRef& ref, const Value* extra, bool extra_is_index, Type access) {
    Id p = shared_element_pointer(ref, extra, extra_is_index, access);
    Id elem_t = scalar_type(ref.elem);
    Id v = op(spv::OpLoad, elem_t, {p});
    Id want = scalar_type(access);
    return want == elem_t ? v : op(spv::OpBitcast, want, {v});
}

void SpirvISel::shared_store(const SharedRef& ref, const Value* extra, bool extra_is_index, const Value* value) {
    if (!value) fail("shared store without a value");
    Id p = shared_element_pointer(ref, extra, extra_is_index, value->type());
    Id elem_t = scalar_type(ref.elem);
    Id v = id_of(value, "shared store value");
    if (scalar_type(value->type()) != elem_t) v = op(spv::OpBitcast, elem_t, {v});
    emit(Inst(spv::OpStore).id(p).id(v));
}

} // namespace brass::spirv
