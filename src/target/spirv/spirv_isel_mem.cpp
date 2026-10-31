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
#include <brass/mir/block.hpp>
#include <brass/mir/opcodes.hpp>

#include <cstring>

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

void SpirvISel::lower_alloca(const brass::Instruction& inst) {
    int64_t bytes = inst.imm_i32();
    if (bytes <= 0) fail("alloca requires a positive byte size");

    std::unordered_map<const Value*, std::pair<bool, int64_t>> derived;
    derived[inst.result()] = {true, 0};
    for (bool changed = true; changed;) {
        changed = false;
        for (const BasicBlock* bb : mir_->blocks()) {
            for (const brass::Instruction* u : *bb) {
                if (u->opcode() != brass::Opcode::add || derived.count(u->result())) continue;
                bool lhs_in = derived.count(u->operand(0)) != 0;
                bool rhs_in = derived.count(u->operand(1)) != 0;
                if (!lhs_in && !rhs_in) continue;
                const Value* base_p = lhs_in ? u->operand(0) : u->operand(1);
                const Value* off_v = lhs_in ? u->operand(1) : u->operand(0);
                auto [base_const, base_off] = derived[base_p];
                int64_t c = 0;
                if (base_const && const_int(off_v, &c)) {
                    derived[u->result()] = {true, base_off + c};
                } else {
                    derived[u->result()] = {false, 0};
                }
                changed = true;
            }
        }
    }

    bool has_4 = false;
    bool has_8 = false;
    bool any_float = false;
    bool all_float = true;
    auto record_type = [&](Type t) {
        if (t.is_vector()) t = t.element_type();
        if (t.size_in_bytes() == 8) {
            has_8 = true;
            if (t.kind() == TypeKind::F64) any_float = true;
            else all_float = false;
        } else if (t.size_in_bytes() == 4) {
            has_4 = true;
            if (t.kind() == TypeKind::F32) any_float = true;
            else all_float = false;
        }
    };

    bool can_const_init = (inst.parent() == mir_->entry_block());
    bool seen_load_in_entry = false;
    std::vector<const brass::Instruction*> entry_stores;

    for (const BasicBlock* bb : mir_->blocks()) {
        for (const brass::Instruction* u : *bb) {
            using brass::Opcode;
            const Opcode op = u->opcode();
            if ((op == Opcode::load || op == Opcode::load_indexed || op == Opcode::vload) &&
                derived.count(u->operand(0))) {
                record_type(u->type());
                if (bb == mir_->entry_block()) seen_load_in_entry = true;
            } else if ((op == Opcode::store || op == Opcode::store_indexed || op == Opcode::vstore) &&
                       derived.count(u->operand(0))) {
                const Value* val = op == Opcode::store_indexed ? u->operand(2) : u->operand(1);
                if (val) record_type(val->type());
                int64_t dummy_i = 0;
                double dummy_f = 0.0;
                if (bb != mir_->entry_block() || seen_load_in_entry || op != Opcode::store ||
                    !derived[u->operand(0)].first || !val || val->type().is_vector() ||
                    (!const_int(val, &dummy_i) && !const_float(val, &dummy_f))) {
                    can_const_init = false;
                } else {
                    entry_stores.push_back(u);
                }
            }
        }
    }

    if (has_4 && has_8) fail("alloca mixes 32-bit and 64-bit accesses on SPIR-V");
    Type elem = has_8 ? ((any_float && all_float) ? Type::f64() : Type::i64())
                      : ((any_float && all_float) ? Type::f32() : Type::i32());
    auto esz = static_cast<uint32_t>(elem.size_in_bytes());
    auto count = static_cast<uint32_t>((bytes + esz - 1) / esz);
    if (count == 0) count = 1;
    if (entry_stores.empty() || count > 4096) can_const_init = false;
    if (can_const_init) {
        for (const brass::Instruction* s : entry_stores) {
            int64_t off = derived[s->operand(0)].second + s->offset();
            if (off < 0 || (off % esz) != 0 || static_cast<uint64_t>(off / esz) >= count) {
                can_const_init = false;
                break;
            }
        }
    }

    Id elem_spv_t = scalar_type(elem);
    Id array_t = m_.t_array(elem_spv_t, count);
    Id init_id = 0;
    if (can_const_init) {
        std::vector<Id> elems(count, m_.c_null(elem_spv_t));
        for (const brass::Instruction* s : entry_stores) {
            int64_t off = derived[s->operand(0)].second + s->offset();
            auto idx = static_cast<size_t>(off / esz);
            const Value* val = s->operand(1);
            int64_t ci = 0;
            double cf = 0.0;
            bool is_ci = const_int(val, &ci);
            if (!is_ci) const_float(val, &cf);
            if (elem.kind() == TypeKind::F32) {
                float f = 0.0f;
                if (is_ci) {
                    auto u = static_cast<uint32_t>(ci);
                    std::memcpy(&f, &u, 4);
                } else {
                    f = static_cast<float>(cf);
                }
                elems[idx] = m_.c_f32(f);
            } else if (elem.kind() == TypeKind::I32) {
                uint32_t u = 0;
                if (is_ci) {
                    u = static_cast<uint32_t>(ci);
                } else {
                    float f = static_cast<float>(cf);
                    std::memcpy(&u, &f, 4);
                }
                elems[idx] = m_.c_u32(u);
            } else if (elem.kind() == TypeKind::F64) {
                double d = 0.0;
                if (is_ci) {
                    auto u = static_cast<uint64_t>(ci);
                    std::memcpy(&d, &u, 8);
                } else {
                    d = cf;
                }
                elems[idx] = m_.c_f64(d);
            } else {
                uint64_t u = 0;
                if (is_ci) {
                    u = static_cast<uint64_t>(ci);
                } else {
                    std::memcpy(&u, &cf, 8);
                }
                elems[idx] = m_.c_int(64, u);
            }
        }
        init_id = m_.c_composite(array_t, elems);
    }

    Id ptr_t = m_.t_pointer(spv::StorageClassFunction, array_t);
    Id var_id = m_.new_id();
    m_.register_value_type(var_id, ptr_t);
    m_.set_name(var_id, "local_" + std::to_string(var_id));
    Inst vinst(spv::OpVariable, ptr_t, var_id);
    vinst.lit(spv::StorageClassFunction);
    if (init_id) vinst.id(init_id);
    fn().blocks[0].insts.insert(fn().blocks[0].insts.begin() + static_cast<std::ptrdiff_t>(num_local_vars_++),
                                std::move(vinst));
    local_[inst.result()] = LocalRef{var_id, elem, 0, 0, init_id != 0};
}

Id SpirvISel::local_element_pointer(const LocalRef& ref, int64_t inst_offset, const Value* index, uint32_t scale,
                                    Type access) {
    auto esz = static_cast<uint32_t>(ref.elem.size_in_bytes());
    if (access.size_in_bytes() != esz) {
        fail("a " + std::to_string(access.size_in_bytes()) + "-byte local access of an array of " +
             std::string(ref.elem.name()));
    }
    int64_t const_bytes = ref.const_off + inst_offset;
    uint32_t shift = log2_of(esz);
    Id dyn = 0;
    auto add_dyn = [&](Id x) { dyn = dyn ? op(spv::OpIAdd, t_u32(), {dyn, x}) : x; };
    if (ref.dyn_off) {
        add_dyn(shift > 0 ? op(spv::OpShiftRightLogical, t_u32(), {ref.dyn_off, m_.c_u32(shift)}) : ref.dyn_off);
    }
    if (index) {
        int64_t c = 0;
        if (const_int(index, &c)) {
            const_bytes += c * static_cast<int64_t>(scale);
        } else {
            Id x = as_u32(index, "local index");
            if (scale == esz) {
                add_dyn(x);
            } else if (scale > esz && (scale % esz) == 0) {
                add_dyn(op(spv::OpIMul, t_u32(), {x, m_.c_u32(scale / esz)}));
            } else {
                Id b_off = scale > 1 ? op(spv::OpIMul, t_u32(), {x, m_.c_u32(scale)}) : x;
                add_dyn(shift > 0 ? op(spv::OpShiftRightLogical, t_u32(), {b_off, m_.c_u32(shift)}) : b_off);
            }
        }
    }
    if (const_bytes % esz != 0) {
        fail("local pointer offset " + std::to_string(const_bytes) + " is not a multiple of the element size " +
             std::to_string(esz));
    }
    int64_t const_index = const_bytes / esz;
    Id elem_idx = m_.c_u32(static_cast<uint32_t>(const_index));
    if (dyn) elem_idx = const_index ? op(spv::OpIAdd, t_u32(), {dyn, elem_idx}) : dyn;
    return op(spv::OpAccessChain, m_.t_pointer(spv::StorageClassFunction, scalar_type(ref.elem)), {ref.var, elem_idx});
}

void SpirvISel::lower_load(const brass::Instruction& inst) {
    Type t = inst.type();
    if (t.is_vector()) {
        lower_vload(inst);
        return;
    }
    if (auto it = local_.find(inst.operand(0)); it != local_.end()) {
        Id p = local_element_pointer(it->second, inst.offset(), nullptr, 1, t);
        Id elem_t = scalar_type(it->second.elem);
        Id v = op(spv::OpLoad, elem_t, {p});
        Id want = scalar_type(t);
        define(inst.result(), want == elem_t ? v : op(spv::OpBitcast, want, {v}));
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
    if (auto it = local_.find(inst.operand(0)); it != local_.end()) {
        if (it->second.const_init) return;
        Id p = local_element_pointer(it->second, inst.offset(), nullptr, 1, value->type());
        Id elem_t = scalar_type(it->second.elem);
        Id v = id_of(value, "value");
        if (scalar_type(value->type()) != elem_t) v = op(spv::OpBitcast, elem_t, {v});
        emit(Inst(spv::OpStore).id(p).id(v));
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
    if (auto it = local_.find(inst.operand(0)); it != local_.end()) {
        Id p = local_element_pointer(it->second, inst.offset(), inst.operand(1), inst.scale(), t);
        Id elem_t = scalar_type(it->second.elem);
        Id v = op(spv::OpLoad, elem_t, {p});
        Id want = scalar_type(t);
        define(inst.result(), want == elem_t ? v : op(spv::OpBitcast, want, {v}));
        return;
    }
    Id addr = indexed_address(inst, inst.operand(0), inst.operand(1));
    define(inst.result(), load_global(addr, scalar_type(t), static_cast<uint32_t>(t.size_in_bytes())));
}

void SpirvISel::lower_store_indexed(const brass::Instruction& inst) {
    const Value* value = inst.operand(2);
    if (!value) malformed(inst, "missing value");
    if (value->type().is_vector()) malformed(inst, "indexed vector store");
    if (auto it = local_.find(inst.operand(0)); it != local_.end()) {
        Id p = local_element_pointer(it->second, inst.offset(), inst.operand(1), inst.scale(), value->type());
        Id elem_t = scalar_type(it->second.elem);
        Id v = id_of(value, "value");
        if (scalar_type(value->type()) != elem_t) v = op(spv::OpBitcast, elem_t, {v});
        emit(Inst(spv::OpStore).id(p).id(v));
        return;
    }
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
