// SpirvISel: 16x16x16 subgroup cooperative matrix lowering (SPV_KHR_cooperative_matrix).

#include "spirv_isel_intrinsics.hpp"

#include <string>

namespace brass::spirv {

namespace {

void enable_coopmat(Module& m) {
    m.add_capability(spv::CapabilityFloat16);
    m.add_capability(spv::CapabilityVulkanMemoryModel);
    m.add_capability(spv::CapabilityVulkanMemoryModelDeviceScope);
    m.add_capability(spv::CapabilityCooperativeMatrixKHR);
    m.add_extension("SPV_KHR_cooperative_matrix");
    m.set_memory_model(spv::AddressingModelPhysicalStorageBuffer64, spv::MemoryModelVulkan);
}

} // namespace

const SpirvISel::CoopRef& SpirvISel::Intrinsics::coop_ref(SpirvISel& isel, const Value* v, const char* what) {
    auto it = isel.coop_.find(v);
    if (it == isel.coop_.end()) {
        isel.fail(std::string(what) + " must come from ptx_coop_alloc_{a,b,acc}");
    }
    return it->second;
}

Id SpirvISel::Intrinsics::coop_ptr_and_stride(SpirvISel& isel,
                                              const Value* ptr,
                                              const Value* stride_elems,
                                              bool elem_is_f16,
                                              Id& out_stride,
                                              bool& out_is_psb,
                                              uint32_t& out_align) {
    Module& m = isel.m_;
    if (auto it = isel.shared_.find(ptr); it != isel.shared_.end()) {
        const SpirvISel::SharedRef& ref = it->second;
        uint32_t shift = 0;
        Id p = 0;
        if (ref.elem == Type::i32x4()) {
            p = isel.shared_element_pointer(ref, nullptr, false, Type::i32x4());
            shift = elem_is_f16 ? 3u : 2u;
        } else if (ref.elem == Type::f32() && !elem_is_f16) {
            p = isel.shared_element_pointer(ref, nullptr, false, Type::f32());
            shift = 0u;
        } else {
            isel.fail("coop matrix shared load/store requires ptx_shared_alloc_i32x4 (or f32 for f32 acc)");
        }
        int64_t sc = 0;
        if (isel.const_int(stride_elems, &sc)) {
            out_stride = m.c_u32(static_cast<uint32_t>(sc >> shift));
        } else if (shift == 0u) {
            out_stride = isel.as_u32(stride_elems, "stride");
        } else {
            out_stride = isel.op(spv::OpShiftRightLogical, m.t_int(32),
                                 {isel.as_u32(stride_elems, "stride"), m.c_u32(shift)});
        }
        out_is_psb = false;
        out_align = 0;
        return p;
    }
    Id elem_t = elem_is_f16 ? m.t_float(16) : m.t_float(32);
    Id addr = isel.id_of(ptr, "coop global pointer");
    Id p = isel.op(spv::OpConvertUToPtr, isel.psb_pointer(elem_t), {addr});
    out_stride = isel.as_u32(stride_elems, "stride");
    out_is_psb = true;
    out_align = elem_is_f16 ? 2u : 4u;
    return p;
}

template <uint32_t Use>
void SpirvISel::Intrinsics::coop_alloc(SpirvISel& isel, const brass::Instruction& inst) {
    Module& m = isel.m_;
    enable_coopmat(m);
    const bool is_acc = (Use == spv::CooperativeMatrixUseMatrixAccumulatorKHR);
    Id elem_t = is_acc ? m.t_float(32) : m.t_float(16);
    Id mat_t = m.t_coop_mat(elem_t, spv::ScopeSubgroup, 16, 16, Use);
    Id ptr_t = m.t_pointer(spv::StorageClassFunction, mat_t);
    Id var_id = m.new_id();
    m.register_value_type(var_id, ptr_t);
    m.set_name(var_id, "coop_" + std::to_string(var_id));
    Inst vinst(spv::OpVariable, ptr_t, var_id);
    vinst.lit(spv::StorageClassFunction);
    auto& entry_insts = isel.fn().blocks[0].insts;
    entry_insts.insert(entry_insts.begin() + static_cast<std::ptrdiff_t>(isel.num_local_vars_++),
                       std::move(vinst));
    isel.coop_[inst.result()] = CoopRef{var_id, Use, mat_t};
}
template void SpirvISel::Intrinsics::coop_alloc<spv::CooperativeMatrixUseMatrixAKHR>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::coop_alloc<spv::CooperativeMatrixUseMatrixBKHR>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::coop_alloc<spv::CooperativeMatrixUseMatrixAccumulatorKHR>(SpirvISel&, const brass::Instruction&);

void SpirvISel::Intrinsics::coop_fill_acc(SpirvISel& isel, const brass::Instruction& inst) {
    const CoopRef& ref = coop_ref(isel, inst.operand(0), "coop_fill_acc target");
    if (ref.use != spv::CooperativeMatrixUseMatrixAccumulatorKHR) {
        isel.fail("ptx_coop_fill_acc requires an accumulator slot");
    }
    Id val = isel.id_of(inst.operand(1), "fill value");
    Id mat = isel.op(spv::OpCompositeConstruct, ref.mat_type, {val});
    isel.emit(Inst(spv::OpStore).id(ref.var).id(mat));
}

template <uint32_t Use>
void SpirvISel::Intrinsics::coop_load(SpirvISel& isel, const brass::Instruction& inst) {
    Module& m = isel.m_;
    const CoopRef& ref = coop_ref(isel, inst.operand(0), "coop_load target");
    if (ref.use != Use) isel.fail("coop_load fragment use mismatch");

    int64_t col_major = 0;
    if (!isel.const_int(inst.operand(3), &col_major)) {
        isel.fail("coop_load layout must be a compile-time constant");
    }
    int64_t src_is_f16 = 0;
    if constexpr (Use == spv::CooperativeMatrixUseMatrixAccumulatorKHR) {
        if (inst.operand(4)) {
            if (!isel.const_int(inst.operand(4), &src_is_f16)) {
                isel.fail("coop_load_acc src_is_f16 must be a compile-time constant");
            }
        }
    }
    const bool is_f16_mem = (Use != spv::CooperativeMatrixUseMatrixAccumulatorKHR) || (src_is_f16 != 0);

    Id stride_id = 0;
    bool is_psb = false;
    uint32_t align = 0;
    Id ptr_id = coop_ptr_and_stride(isel, inst.operand(1), inst.operand(2),
                                    is_f16_mem, stride_id, is_psb, align);

    Id layout_id = m.c_u32(col_major ? spv::CooperativeMatrixLayoutColumnMajorKHR
                                     : spv::CooperativeMatrixLayoutRowMajorKHR);
    Id load_mat_t = (Use == spv::CooperativeMatrixUseMatrixAccumulatorKHR && src_is_f16)
                        ? m.t_coop_mat(m.t_float(16), spv::ScopeSubgroup, 16, 16,
                                       spv::CooperativeMatrixUseMatrixAccumulatorKHR)
                        : ref.mat_type;

    Id r = m.new_id();
    Inst ld(spv::OpCooperativeMatrixLoadKHR, load_mat_t, r);
    ld.id(ptr_id).id(layout_id).id(stride_id);
    if (is_psb) ld.lit(spv::MemoryAccessAlignedMask).lit(align);
    isel.emit(std::move(ld));

    if (load_mat_t != ref.mat_type) {
        r = isel.op(spv::OpFConvert, ref.mat_type, {r});
    }
    isel.emit(Inst(spv::OpStore).id(ref.var).id(r));
}
template void SpirvISel::Intrinsics::coop_load<spv::CooperativeMatrixUseMatrixAKHR>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::coop_load<spv::CooperativeMatrixUseMatrixBKHR>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::coop_load<spv::CooperativeMatrixUseMatrixAccumulatorKHR>(SpirvISel&, const brass::Instruction&);

void SpirvISel::Intrinsics::coop_mma(SpirvISel& isel, const brass::Instruction& inst) {
    const CoopRef& acc = coop_ref(isel, inst.operand(0), "coop_mma acc");
    const CoopRef& a   = coop_ref(isel, inst.operand(1), "coop_mma a");
    const CoopRef& b   = coop_ref(isel, inst.operand(2), "coop_mma b");
    if (acc.use != spv::CooperativeMatrixUseMatrixAccumulatorKHR ||
        a.use != spv::CooperativeMatrixUseMatrixAKHR ||
        b.use != spv::CooperativeMatrixUseMatrixBKHR) {
        isel.fail("ptx_coop_mma operands must be (acc, a, b)");
    }
    Id va = isel.op(spv::OpLoad, a.mat_type, {a.var});
    Id vb = isel.op(spv::OpLoad, b.mat_type, {b.var});
    Id vc = isel.op(spv::OpLoad, acc.mat_type, {acc.var});
    Id res = isel.op(spv::OpCooperativeMatrixMulAddKHR, acc.mat_type, {va, vb, vc});
    isel.emit(Inst(spv::OpStore).id(acc.var).id(res));
}

void SpirvISel::Intrinsics::coop_store_acc(SpirvISel& isel, const brass::Instruction& inst) {
    Module& m = isel.m_;
    const CoopRef& ref = coop_ref(isel, inst.operand(0), "coop_store_acc source");
    if (ref.use != spv::CooperativeMatrixUseMatrixAccumulatorKHR) {
        isel.fail("ptx_coop_store_acc requires an accumulator slot");
    }
    int64_t col_major = 0;
    int64_t dst_is_f16 = 0;
    if (!isel.const_int(inst.operand(3), &col_major) ||
        (inst.operand(4) && !isel.const_int(inst.operand(4), &dst_is_f16))) {
        isel.fail("ptx_coop_store_acc layout and dst_is_f16 must be compile-time constants");
    }
    const bool is_f16_mem = (dst_is_f16 != 0);
    Id stride_id = 0;
    bool is_psb = false;
    uint32_t align = 0;
    Id ptr_id = coop_ptr_and_stride(isel, inst.operand(1), inst.operand(2),
                                    is_f16_mem, stride_id, is_psb, align);

    Id val = isel.op(spv::OpLoad, ref.mat_type, {ref.var});
    if (is_f16_mem) {
        Id f16_mat_t = m.t_coop_mat(m.t_float(16), spv::ScopeSubgroup, 16, 16,
                                    spv::CooperativeMatrixUseMatrixAccumulatorKHR);
        val = isel.op(spv::OpFConvert, f16_mat_t, {val});
    }
    Id layout_id = m.c_u32(col_major ? spv::CooperativeMatrixLayoutColumnMajorKHR
                                     : spv::CooperativeMatrixLayoutRowMajorKHR);
    Inst st(spv::OpCooperativeMatrixStoreKHR);
    st.id(ptr_id).id(val).id(layout_id).id(stride_id);
    if (is_psb) st.lit(spv::MemoryAccessAlignedMask).lit(align);
    isel.emit(std::move(st));
}

void SpirvISel::Intrinsics::coop_scale_acc(SpirvISel& isel, const brass::Instruction& inst) {
    const CoopRef& ref = coop_ref(isel, inst.operand(0), "coop_scale_acc acc");
    if (ref.use != spv::CooperativeMatrixUseMatrixAccumulatorKHR) {
        isel.fail("ptx_coop_scale_acc requires an accumulator slot");
    }
    Id val = isel.op(spv::OpLoad, ref.mat_type, {ref.var});
    Id res = isel.op(spv::OpMatrixTimesScalar, ref.mat_type, {val, isel.id_of(inst.operand(1), "scale")});
    isel.emit(Inst(spv::OpStore).id(ref.var).id(res));
}

void SpirvISel::Intrinsics::coop_add_acc(SpirvISel& isel, const brass::Instruction& inst) {
    const CoopRef& dst = coop_ref(isel, inst.operand(0), "coop_add_acc dst");
    const CoopRef& rhs = coop_ref(isel, inst.operand(1), "coop_add_acc rhs");
    if (dst.use != spv::CooperativeMatrixUseMatrixAccumulatorKHR ||
        rhs.use != spv::CooperativeMatrixUseMatrixAccumulatorKHR) {
        isel.fail("ptx_coop_add_acc requires accumulator slots");
    }
    Id a = isel.op(spv::OpLoad, dst.mat_type, {dst.var});
    Id b = isel.op(spv::OpLoad, rhs.mat_type, {rhs.var});
    Id res = isel.op(spv::OpFAdd, dst.mat_type, {a, b});
    isel.emit(Inst(spv::OpStore).id(dst.var).id(res));
}

void SpirvISel::Intrinsics::coop_acc_len(SpirvISel& isel, const brass::Instruction& inst) {
    const CoopRef& ref = coop_ref(isel, inst.operand(0), "coop_acc_len acc");
    if (ref.use != spv::CooperativeMatrixUseMatrixAccumulatorKHR) {
        isel.fail("ptx_coop_acc_len requires an accumulator slot");
    }
    Id len = isel.op(spv::OpCooperativeMatrixLengthKHR, isel.t_u32(), {ref.mat_type});
    isel.define(inst.result(), len);
}

void SpirvISel::Intrinsics::coop_get_acc(SpirvISel& isel, const brass::Instruction& inst) {
    Module& m = isel.m_;
    const CoopRef& ref = coop_ref(isel, inst.operand(0), "coop_get_acc acc");
    if (ref.use != spv::CooperativeMatrixUseMatrixAccumulatorKHR) {
        isel.fail("ptx_coop_get_acc requires an accumulator slot");
    }
    Id idx = isel.as_u32(inst.operand(1), "coop_get_acc idx");
    Id ptr_t = m.t_pointer(spv::StorageClassFunction, m.t_float(32));
    Id ep = isel.op(spv::OpAccessChain, ptr_t, {ref.var, idx});
    Id val = isel.op(spv::OpLoad, m.t_float(32), {ep});
    isel.define(inst.result(), val);
}

void SpirvISel::Intrinsics::coop_set_acc(SpirvISel& isel, const brass::Instruction& inst) {
    Module& m = isel.m_;
    const CoopRef& ref = coop_ref(isel, inst.operand(0), "coop_set_acc acc");
    if (ref.use != spv::CooperativeMatrixUseMatrixAccumulatorKHR) {
        isel.fail("ptx_coop_set_acc requires an accumulator slot");
    }
    Id idx = isel.as_u32(inst.operand(1), "coop_set_acc idx");
    Id val = isel.id_of(inst.operand(2), "coop_set_acc value");
    Id ptr_t = m.t_pointer(spv::StorageClassFunction, m.t_float(32));
    Id ep = isel.op(spv::OpAccessChain, ptr_t, {ref.var, idx});
    isel.emit(Inst(spv::OpStore).id(ep).id(val));
}

} // namespace brass::spirv
