// SpirvISel intrinsic rules: barriers, subgroup shuffles, atomics, the wide
// integer multiplies, shared (Workgroup) arrays and narrow global access.
// See spirv_isel_intrinsics.hpp for the contract.

#include "spirv_isel_intrinsics.hpp"

#include <limits>
#include <string>

namespace brass::spirv {

// ---------------------------------------------------------------------------
// Barriers: bar.sync orders shared and global memory among the block's
// threads, so the barrier releases/acquires Workgroup and Uniform
// (StorageBuffer / PhysicalStorageBuffer) memory.
// ---------------------------------------------------------------------------

void SpirvISel::Intrinsics::barrier(SpirvISel& isel) {
    Module& m = isel.m_;
    Id wg = m.c_u32(spv::ScopeWorkgroup);
    Id sem = m.c_u32(spv::MemorySemanticsAcquireReleaseMask | spv::MemorySemanticsWorkgroupMemoryMask |
                     spv::MemorySemanticsUniformMemoryMask);
    isel.emit(Inst(spv::OpControlBarrier).id(wg).id(wg).id(sem));
}

void SpirvISel::Intrinsics::bar_sync(SpirvISel& isel, const brass::Instruction&) {
    barrier(isel);
}

// bar.sync id: every thread of the block waits, whatever the id; SPIR-V has
// the one workgroup barrier, which is the same thing when, as bar.sync
// without a count requires, all threads reach it.
void SpirvISel::Intrinsics::bar_sync_id(SpirvISel& isel, const brass::Instruction& inst) {
    int64_t id = 0;
    if (!isel.const_int(inst.operand(0), &id)) {
        isel.fail("ptx_bar_sync needs a compile-time constant barrier id on SPIR-V (there is one workgroup barrier)");
    }
    barrier(isel);
}

// ---------------------------------------------------------------------------
// Shuffles. PTX semantics on 32-lane segments of the subgroup: the source
// lane is computed from the lane within the segment, and a source outside
// the segment (down past lane 31, up below lane 0) yields the thread's own
// value, as shfl.sync does with the clamp values nvcc uses. One
// OpGroupNonUniformShuffle with a computed absolute index does every mode,
// so only the GroupNonUniformShuffle capability is needed. The member mask
// has no SPIR-V counterpart and is ignored (non-uniform ops act on the
// active invocations).
// ---------------------------------------------------------------------------

Id SpirvISel::Intrinsics::shuffle(SpirvISel& isel, const brass::Instruction& inst, Shfl mode, const Value* value,
                                  const Value* delta) {
    if (!value || value->type().is_vector() || value->type().size_in_bytes() != 4) {
        isel.malformed(inst, "shfl value must be a 32-bit f32/i32");
    }
    Module& m = isel.m_;
    m.add_capability(spv::CapabilityGroupNonUniformShuffle);
    Id u32 = isel.t_u32();
    Id raw = raw_subgroup_lane(isel);
    Id seg = lane_id(isel);
    Id base = isel.prologue_value("lanebase", [&] { return isel.op(spv::OpBitwiseAnd, u32, {raw, m.c_u32(~31u)}); });
    Id d = isel.as_u32(delta, "lane delta");

    Id src = 0;
    switch (mode) {
        case Shfl::down: {
            Id j = isel.op(spv::OpIAdd, u32, {raw, d});
            Id ok = isel.op(spv::OpULessThanEqual, isel.t_bool(), {isel.op(spv::OpIAdd, u32, {seg, d}), m.c_u32(31)});
            src = isel.op(spv::OpSelect, u32, {ok, j, raw});
            break;
        }
        case Shfl::up: {
            Id j = isel.op(spv::OpISub, u32, {raw, d});
            Id ok = isel.op(spv::OpUGreaterThanEqual, isel.t_bool(), {seg, d});
            src = isel.op(spv::OpSelect, u32, {ok, j, raw});
            break;
        }
        case Shfl::bfly: {
            Id x = isel.op(spv::OpBitwiseXor, u32, {seg, isel.op(spv::OpBitwiseAnd, u32, {d, m.c_u32(31)})});
            src = isel.op(spv::OpBitwiseOr, u32, {base, x});
            break;
        }
        case Shfl::idx:
            src = isel.op(spv::OpBitwiseOr, u32, {base, isel.op(spv::OpBitwiseAnd, u32, {d, m.c_u32(31)})});
            break;
    }
    return isel.op(spv::OpGroupNonUniformShuffle, isel.scalar_type(value->type()),
                   {m.c_u32(spv::ScopeSubgroup), isel.id_of(value, "shfl value"), src});
}

template <SpirvISel::Intrinsics::Shfl M>
void SpirvISel::Intrinsics::shfl(SpirvISel& isel, const brass::Instruction& inst) {
    isel.define(inst.result(), shuffle(isel, inst, M, inst.operand(0), inst.operand(1)));
}
template void SpirvISel::Intrinsics::shfl<SpirvISel::Intrinsics::Shfl::down>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::shfl<SpirvISel::Intrinsics::Shfl::up>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::shfl<SpirvISel::Intrinsics::Shfl::bfly>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::shfl<SpirvISel::Intrinsics::Shfl::idx>(SpirvISel&, const brass::Instruction&);

// (mask, value, delta)
void SpirvISel::Intrinsics::shfl_down_sync_f32(SpirvISel& isel, const brass::Instruction& inst) {
    isel.define(inst.result(), shuffle(isel, inst, Shfl::down, inst.operand(1), inst.operand(2)));
}

// ---------------------------------------------------------------------------
// Atomics (relaxed, like PTX atom): device scope through a
// PhysicalStorageBuffer pointer, workgroup scope on a shared element.
// ---------------------------------------------------------------------------

template <SpirvISel::Intrinsics::Atom A, bool Shared>
void SpirvISel::Intrinsics::atom(SpirvISel& isel, const brass::Instruction& inst) {
    const Value* cmp = A == Atom::cas ? inst.operand(1) : nullptr;
    const Value* value = inst.operand(A == Atom::cas ? 2 : 1);
    if (!inst.operand(0) || !value || (A == Atom::cas && !cmp)) isel.malformed(inst, "missing argument");
    Module& m = isel.m_;
    Type vt = value->type();
    Id t = isel.scalar_type(vt);
    if constexpr (A != Atom::add && A != Atom::exch) {
        if (vt.is_float()) isel.malformed(inst, "only atomic add and exchange support float values");
    }
    if (vt.size_in_bytes() == 8) m.add_capability(spv::CapabilityInt64Atomics);

    Id ptr = 0;
    Id elem_t = t;
    spv::Scope scope = spv::ScopeDevice;
    if constexpr (Shared) {
        const SharedRef& ref = isel.shared_ref(inst.operand(0), "atomic pointer");
        ptr = isel.shared_element_pointer(ref, nullptr, false, vt);
        elem_t = isel.scalar_type(ref.elem);
        scope = spv::ScopeWorkgroup;
    } else {
        ptr = isel.op(spv::OpConvertUToPtr, isel.psb_pointer(t), {isel.id_of(inst.operand(0), "atomic pointer")});
    }

    if constexpr (A == Atom::cas) {
        Id val_id = isel.id_of(value, "atomic value");
        Id cmp_id = isel.id_of(cmp, "atomic comparator");
        if (elem_t != t) {
            val_id = isel.op(spv::OpBitcast, elem_t, {val_id});
            cmp_id = isel.op(spv::OpBitcast, elem_t, {cmp_id});
        }
        Id r = isel.op(spv::OpAtomicCompareExchange, elem_t,
                       {ptr, m.c_u32(scope), m.c_u32(spv::MemorySemanticsMaskNone),
                        m.c_u32(spv::MemorySemanticsMaskNone), val_id, cmp_id});
        if (elem_t != t) r = isel.op(spv::OpBitcast, t, {r});
        isel.define(inst.result(), r);
        return;
    }

    spv::Op o = spv::OpAtomicIAdd;
    if constexpr (A == Atom::add) {
        if (vt.is_float()) {
            o = spv::OpAtomicFAddEXT;
            m.add_capability(vt.size_in_bytes() == 8 ? spv::CapabilityAtomicFloat64AddEXT
                                                     : spv::CapabilityAtomicFloat32AddEXT);
            m.add_extension("SPV_EXT_shader_atomic_float_add");
        }
    } else if constexpr (A == Atom::min) {
        o = spv::OpAtomicSMin;
    } else if constexpr (A == Atom::umin) {
        o = spv::OpAtomicUMin;
    } else if constexpr (A == Atom::max) {
        o = spv::OpAtomicSMax;
    } else if constexpr (A == Atom::umax) {
        o = spv::OpAtomicUMax;
    } else if constexpr (A == Atom::exch) {
        o = spv::OpAtomicExchange;
    } else if constexpr (A == Atom::and_) {
        o = spv::OpAtomicAnd;
    } else if constexpr (A == Atom::or_) {
        o = spv::OpAtomicOr;
    } else if constexpr (A == Atom::xor_) {
        o = spv::OpAtomicXor;
    }
    Id val_id = isel.id_of(value, "atomic value");
    if (elem_t != t) val_id = isel.op(spv::OpBitcast, elem_t, {val_id});
    Id r = isel.op(o, elem_t, {ptr, m.c_u32(scope), m.c_u32(spv::MemorySemanticsMaskNone), val_id});
    if (elem_t != t) r = isel.op(spv::OpBitcast, t, {r});
    isel.define(inst.result(), r);
}
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::add, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::min, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::umin, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::max, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::umax, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::exch, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::and_, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::or_, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::xor_, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::cas, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::add, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::min, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::umin, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::max, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::umax, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::exch, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::and_, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::or_, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::xor_, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::atom<SpirvISel::Intrinsics::Atom::cas, true>(SpirvISel&, const brass::Instruction&);

// ---------------------------------------------------------------------------
// Wide and fused integer multiplies
// ---------------------------------------------------------------------------

template <bool Signed>
void SpirvISel::Intrinsics::mul_wide(SpirvISel& isel, const brass::Instruction& inst) {
    spv::Op ext = Signed ? spv::OpSConvert : spv::OpUConvert;
    Id a = isel.op(ext, isel.t_u64(), {isel.id_of(inst.operand(0), "argument 0")});
    Id b = isel.op(ext, isel.t_u64(), {isel.id_of(inst.operand(1), "argument 1")});
    isel.define(inst.result(), isel.op(spv::OpIMul, isel.t_u64(), {a, b}));
}
template void SpirvISel::Intrinsics::mul_wide<false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::mul_wide<true>(SpirvISel&, const brass::Instruction&);

void SpirvISel::Intrinsics::mul_hi_u32(SpirvISel& isel, const brass::Instruction& inst) {
    Id& pair = isel.m_.cached("struct.u32.u32");
    if (!pair) pair = isel.m_.t_struct({isel.t_u32(), isel.t_u32()});
    Id r = isel.op(spv::OpUMulExtended, pair,
                   {isel.id_of(inst.operand(0), "argument 0"), isel.id_of(inst.operand(1), "argument 1")});
    Id hi = isel.m_.new_id();
    isel.emit(Inst(spv::OpCompositeExtract, isel.t_u32(), hi).id(r).lit(1));
    isel.define(inst.result(), hi);
}

void SpirvISel::Intrinsics::mad_lo_u32(SpirvISel& isel, const brass::Instruction& inst) {
    Id p = isel.op(spv::OpIMul, isel.t_u32(), {isel.id_of(inst.operand(0), "argument 0"), isel.id_of(inst.operand(1), "argument 1")});
    isel.define(inst.result(), isel.op(spv::OpIAdd, isel.t_u32(), {p, isel.id_of(inst.operand(2), "argument 2")}));
}

// ---------------------------------------------------------------------------
// Shared memory: one Workgroup array per ptx_shared_alloc_* call
// ---------------------------------------------------------------------------

template <TypeKind K>
void SpirvISel::Intrinsics::shared_alloc(SpirvISel& isel, const brass::Instruction& inst) {
    int64_t count = 0;
    if (!isel.const_int(inst.operand(0), &count) || count <= 0 || count > std::numeric_limits<int32_t>::max()) {
        isel.fail(std::string(inst.symbol()) + " requires a positive compile-time constant element count");
    }
    Module& m = isel.m_;
    Type elem(K);
    Id array = m.t_array(isel.scalar_type(elem), static_cast<uint32_t>(count));
    Id var = m.variable(m.t_pointer(spv::StorageClassWorkgroup, array), spv::StorageClassWorkgroup);
    m.set_name(var, "smem_" + std::to_string(var));
    isel.use_global(var);
    isel.shared_[inst.result()] = SharedRef{var, elem, 0, 0};
    isel.shared_bytes_ += static_cast<uint32_t>(count * static_cast<int64_t>(elem.size_in_bytes()));
}
template void SpirvISel::Intrinsics::shared_alloc<TypeKind::F32>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::shared_alloc<TypeKind::I32>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::shared_alloc<TypeKind::F64>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::shared_alloc<TypeKind::I64>(SpirvISel&, const brass::Instruction&);

// (ptr[, byte offset]) or, indexed, (ptr, element index)
template <TypeKind K, bool Indexed>
void SpirvISel::Intrinsics::shared_load(SpirvISel& isel, const brass::Instruction& inst) {
    if constexpr (Indexed) {
        if (!inst.operand(1)) isel.malformed(inst, "missing index");
    }
    const SharedRef& ref = isel.shared_ref(inst.operand(0), "shared load pointer");
    isel.define(inst.result(), isel.shared_load(ref, inst.operand(1), Indexed, Type(K)));
}

// (ptr, value[, byte offset]) or, indexed, (ptr, element index, value)
template <TypeKind K, bool Indexed>
void SpirvISel::Intrinsics::shared_store(SpirvISel& isel, const brass::Instruction& inst) {
    const Value* value = Indexed ? inst.operand(2) : inst.operand(1);
    const Value* extra = Indexed ? inst.operand(1) : inst.operand(2);
    if (!value || value->type().size_in_bytes() != Type(K).size_in_bytes()) {
        isel.malformed(inst, "value type does not match the intrinsic's element type");
    }
    const SharedRef& ref = isel.shared_ref(inst.operand(0), "shared store pointer");
    isel.shared_store(ref, extra, Indexed, value);
}

#define BRASS_SPIRV_SHARED_ACCESS(K)                                                                        \
    template void SpirvISel::Intrinsics::shared_load<K, false>(SpirvISel&, const brass::Instruction&);    \
    template void SpirvISel::Intrinsics::shared_load<K, true>(SpirvISel&, const brass::Instruction&);     \
    template void SpirvISel::Intrinsics::shared_store<K, false>(SpirvISel&, const brass::Instruction&);   \
    template void SpirvISel::Intrinsics::shared_store<K, true>(SpirvISel&, const brass::Instruction&);
BRASS_SPIRV_SHARED_ACCESS(TypeKind::F32)
BRASS_SPIRV_SHARED_ACCESS(TypeKind::I32)
BRASS_SPIRV_SHARED_ACCESS(TypeKind::F64)
BRASS_SPIRV_SHARED_ACCESS(TypeKind::I64)
#undef BRASS_SPIRV_SHARED_ACCESS

// ---------------------------------------------------------------------------
// Narrow global access through 8/16-bit PhysicalStorageBuffer pointers
// (storageBuffer8BitAccess / storageBuffer16BitAccess, core in Vulkan 1.2).
// ---------------------------------------------------------------------------

Id SpirvISel::Intrinsics::byte_address(SpirvISel& isel, const Value* ptr, const Value* offset) {
    Id base = isel.id_of(ptr, "pointer");
    if (!offset) return base;
    int64_t c = 0;
    if (isel.const_int(offset, &c)) return isel.address(base, c);
    return isel.op(spv::OpIAdd, isel.t_u64(), {base, isel.as_u64(offset, "byte offset")});
}

template <uint32_t Bits, bool Signed>
void SpirvISel::Intrinsics::load_narrow(SpirvISel& isel, const brass::Instruction& inst) {
    isel.m_.add_capability(Bits == 8 ? spv::CapabilityStorageBuffer8BitAccess : spv::CapabilityStorageBuffer16BitAccess);
    Id t = isel.m_.t_int(Bits);
    Id v = isel.load_global(byte_address(isel, inst.operand(0), inst.operand(1)), t, Bits / 8);
    isel.define(inst.result(), isel.op(Signed ? spv::OpSConvert : spv::OpUConvert, isel.t_u32(), {v}));
}
template void SpirvISel::Intrinsics::load_narrow<8, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::load_narrow<8, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::load_narrow<16, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::load_narrow<16, true>(SpirvISel&, const brass::Instruction&);

template <uint32_t Bits>
void SpirvISel::Intrinsics::store_narrow(SpirvISel& isel, const brass::Instruction& inst) {
    isel.m_.add_capability(Bits == 8 ? spv::CapabilityStorageBuffer8BitAccess : spv::CapabilityStorageBuffer16BitAccess);
    Id t = isel.m_.t_int(Bits);
    Id v = isel.op(spv::OpUConvert, t, {isel.as_u32(inst.operand(1), "value")});
    isel.store_global(byte_address(isel, inst.operand(0), inst.operand(2)), v, t, Bits / 8);
}
template void SpirvISel::Intrinsics::store_narrow<8>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::store_narrow<16>(SpirvISel&, const brass::Instruction&);

// ---------------------------------------------------------------------------
// Subgroup operations over the whole subgroup (not the 32-lane segments the
// shuffles keep): reductions of the active invocations' values
// (GroupNonUniformArithmetic), the first active invocation's value
// (GroupNonUniformBallot), the subgroup size and the raw lane.
// ---------------------------------------------------------------------------

template <SpirvISel::Intrinsics::Group G>
void SpirvISel::Intrinsics::subgroup_reduce(SpirvISel& isel, const brass::Instruction& inst) {
    const Value* v = inst.operand(0);
    if (!v || v->type().is_vector() || v->type().size_in_bytes() != 4) {
        isel.malformed(inst, "subgroup value must be a 32-bit f32/i32");
    }
    Module& m = isel.m_;
    m.add_capability(spv::CapabilityGroupNonUniform);
    m.add_capability(spv::CapabilityGroupNonUniformArithmetic);
    const bool f = v->type().is_float();
    spv::Op o = spv::OpGroupNonUniformIAdd;
    switch (G) {
        case Group::add: o = f ? spv::OpGroupNonUniformFAdd : spv::OpGroupNonUniformIAdd; break;
        case Group::min: o = f ? spv::OpGroupNonUniformFMin : spv::OpGroupNonUniformSMin; break;
        case Group::max: o = f ? spv::OpGroupNonUniformFMax : spv::OpGroupNonUniformSMax; break;
    }
    Id r = m.new_id();
    isel.emit(Inst(o, isel.scalar_type(v->type()), r)
                  .id(m.c_u32(spv::ScopeSubgroup))
                  .lit(spv::GroupOperationReduce)
                  .id(isel.id_of(v, "subgroup value")));
    isel.define(inst.result(), r);
}
template void SpirvISel::Intrinsics::subgroup_reduce<SpirvISel::Intrinsics::Group::add>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::subgroup_reduce<SpirvISel::Intrinsics::Group::min>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::subgroup_reduce<SpirvISel::Intrinsics::Group::max>(SpirvISel&, const brass::Instruction&);

void SpirvISel::Intrinsics::subgroup_broadcast_first(SpirvISel& isel, const brass::Instruction& inst) {
    const Value* v = inst.operand(0);
    if (!v || v->type().is_vector() || v->type().size_in_bytes() != 4) {
        isel.malformed(inst, "subgroup value must be a 32-bit f32/i32");
    }
    Module& m = isel.m_;
    m.add_capability(spv::CapabilityGroupNonUniform);
    m.add_capability(spv::CapabilityGroupNonUniformBallot);
    isel.define(inst.result(), isel.op(spv::OpGroupNonUniformBroadcastFirst, isel.scalar_type(v->type()),
                                       {m.c_u32(spv::ScopeSubgroup), isel.id_of(v, "subgroup value")}));
}

// (i32 predicate) -> i32 1 / 0: whether it is nonzero on every / any active
// invocation (GroupNonUniformVote).
template <bool All>
void SpirvISel::Intrinsics::subgroup_vote(SpirvISel& isel, const brass::Instruction& inst) {
    const Value* v = inst.operand(0);
    if (!v || v->type() != Type::i32()) isel.malformed(inst, "subgroup predicate must be i32");
    Module& m = isel.m_;
    m.add_capability(spv::CapabilityGroupNonUniform);
    m.add_capability(spv::CapabilityGroupNonUniformVote);
    Id pred = isel.op(spv::OpINotEqual, isel.t_bool(), {isel.id_of(v, "subgroup predicate"), m.c_u32(0)});
    Id r = isel.op(All ? spv::OpGroupNonUniformAll : spv::OpGroupNonUniformAny, isel.t_bool(),
                   {m.c_u32(spv::ScopeSubgroup), pred});
    isel.define(inst.result(), isel.op(spv::OpSelect, isel.t_u32(), {r, m.c_u32(1), m.c_u32(0)}));
}
template void SpirvISel::Intrinsics::subgroup_vote<true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::subgroup_vote<false>(SpirvISel&, const brass::Instruction&);

void SpirvISel::Intrinsics::subgroup_size(SpirvISel& isel, const brass::Instruction& inst) {
    isel.m_.add_capability(spv::CapabilityGroupNonUniform);
    isel.define(inst.result(), isel.builtin_scalar(spv::BuiltInSubgroupSize));
}

void SpirvISel::Intrinsics::subgroup_local_id(SpirvISel& isel, const brass::Instruction& inst) {
    isel.define(inst.result(), raw_subgroup_lane(isel));
}

} // namespace brass::spirv
