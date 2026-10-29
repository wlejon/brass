// SpirvISel graphics & descriptor-bound image/sampler lowering:
//   - Descriptor sets (`UniformConstant` globals with `DescriptorSet` and
//     `Binding` decorations)
//   - 2D and 3D storage images (`OpTypeImage`, `OpImageRead`, `OpImageWrite`)
//   - 2D and 3D sampled textures (`OpTypeSampledImage`, `OpImageSampleExplicitLod`)
//   - Graphics pipeline stage builtins (`VertexIndex`, `InstanceIndex`,
//     `Position`, `FragCoord`) and `Location`-decorated stage inputs/outputs

#include "spirv_isel_intrinsics.hpp"

#include <algorithm>
#include <limits>
#include <string>

namespace brass::spirv {

// ---------------------------------------------------------------------------
// SpirvISel helpers for graphics builtins, stage I/O, and descriptor bindings
// ---------------------------------------------------------------------------

Id SpirvISel::builtin_fvec4(spv::BuiltIn b) {
    std::string key = std::string("builtin.fvec4.") + std::to_string(static_cast<uint32_t>(b));
    Id& var = m_.cached(key);
    Id v4 = m_.t_vector(t_f32(), 4);
    if (!var) {
        var = m_.variable(m_.t_pointer(spv::StorageClassInput, v4), spv::StorageClassInput);
        m_.decorate(var, spv::DecorationBuiltIn, {static_cast<uint32_t>(b)});
    }
    Id vid = var;
    use_global(vid);
    return prologue_value(key, [&] { return op(spv::OpLoad, v4, {vid}); });
}

Id SpirvISel::builtin_fvec4_component(spv::BuiltIn b, uint32_t component) {
    std::string key = std::string("builtin.fvec4.") + std::to_string(static_cast<uint32_t>(b));
    Id vec = builtin_fvec4(b);
    return prologue_value(key + "." + std::to_string(component), [&] {
        Id r = m_.new_id();
        emit(Inst(spv::OpCompositeExtract, t_f32(), r).id(vec).lit(component));
        return r;
    });
}

Id SpirvISel::descriptor_var(uint32_t set, uint32_t binding, target::SpirvDescriptorKind kind, Id handle_type) {
    std::string key = "desc." + std::to_string(set) + "." + std::to_string(binding);
    std::string kind_key = "desc_kind." + std::to_string(set) + "." + std::to_string(binding);
    Id& var = m_.cached(key);
    Id& cached_kind = m_.cached(kind_key);
    auto kind_tag = static_cast<Id>(kind) + 1;
    if (!var) {
        Id ptr_t = m_.t_pointer(spv::StorageClassUniformConstant, handle_type);
        var = m_.variable(ptr_t, spv::StorageClassUniformConstant);
        m_.decorate(var, spv::DecorationDescriptorSet, {set});
        m_.decorate(var, spv::DecorationBinding, {binding});
        m_.set_name(var, "desc_s" + std::to_string(set) + "_b" + std::to_string(binding));
        cached_kind = kind_tag;
    } else if (cached_kind != kind_tag) {
        fail("descriptor (set=" + std::to_string(set) + ", binding=" + std::to_string(binding) +
             ") used with conflicting descriptor kinds");
    }
    Id vid = var;
    use_global(vid);

    target::SpirvDescriptorBinding db{set, binding, kind};
    if (std::find(descriptors_.begin(), descriptors_.end(), db) == descriptors_.end()) {
        descriptors_.push_back(db);
    }
    return vid;
}

Id SpirvISel::stage_io_var(spv::StorageClass sc, uint32_t location) {
    std::string key = std::string(sc == spv::StorageClassInput ? "stage_in." : "stage_out.") +
                      std::to_string(location);
    Id& var = m_.cached(key);
    Id v4 = m_.t_vector(t_f32(), 4);
    if (!var) {
        var = m_.variable(m_.t_pointer(sc, v4), sc);
        m_.decorate(var, spv::DecorationLocation, {location});
        m_.set_name(var, (sc == spv::StorageClassInput ? "in_loc" : "out_loc") + std::to_string(location));
    }
    Id vid = var;
    use_global(vid);
    return vid;
}

// ---------------------------------------------------------------------------
// 2D and 3D Storage Images (OpTypeImage, OpImageRead, OpImageWrite)
// ---------------------------------------------------------------------------

namespace {

Id storage_image_handle_type(Module& m, spv::Dim dim, bool vec4, Id existing_var) {
    if (existing_var != 0) {
        const TypeInfo* pti = m.type_info(m.type_of_value(existing_var));
        if (pti && pti->op == spv::OpTypePointer && pti->elem != 0) return pti->elem;
    }
    return m.t_image(m.t_float(32), dim, 0, 0, 0, 2,
                     vec4 ? spv::ImageFormatRgba32f : spv::ImageFormatR32f);
}

} // namespace

void SpirvISel::Intrinsics::parse_set_binding(SpirvISel& isel, const brass::Instruction& inst,
                                              uint32_t& out_set, uint32_t& out_binding) {
    int64_t s = 0, b = 0;
    if (!isel.const_int(inst.operand(0), &s) || s < 0 || s > std::numeric_limits<int32_t>::max()) {
        isel.fail(std::string(inst.symbol()) + " requires a non-negative compile-time constant descriptor set");
    }
    if (!isel.const_int(inst.operand(1), &b) || b < 0 || b > std::numeric_limits<int32_t>::max()) {
        isel.fail(std::string(inst.symbol()) + " requires a non-negative compile-time constant binding");
    }
    out_set = static_cast<uint32_t>(s);
    out_binding = static_cast<uint32_t>(b);
}

template <spv::Dim Dim, bool Vec4>
void SpirvISel::Intrinsics::image_load(SpirvISel& isel, const brass::Instruction& inst) {
    constexpr size_t expected_ops = (Dim == spv::Dim3D) ? 5 : 4;
    if (inst.operand_count() < expected_ops) isel.malformed(inst, "missing image_load operand");
    uint32_t set = 0, binding = 0;
    parse_set_binding(isel, inst, set, binding);

    Module& m = isel.m_;
    auto kind = (Dim == spv::Dim3D) ? target::SpirvDescriptorKind::StorageImage3D
                                    : target::SpirvDescriptorKind::StorageImage2D;
    Id existing = m.cached("desc." + std::to_string(set) + "." + std::to_string(binding));
    Id img_t = storage_image_handle_type(m, Dim, Vec4, existing);
    Id var = isel.descriptor_var(set, binding, kind, img_t);
    Id img = isel.op(spv::OpLoad, img_t, {var});

    Id u32 = isel.t_u32();
    Id coord = 0;
    if constexpr (Dim == spv::Dim2D) {
        coord = isel.op(spv::OpCompositeConstruct, m.t_vector(u32, 2),
                        {isel.as_u32(inst.operand(2), "ix"), isel.as_u32(inst.operand(3), "iy")});
    } else {
        coord = isel.op(spv::OpCompositeConstruct, m.t_vector(u32, 3),
                        {isel.as_u32(inst.operand(2), "ix"),
                         isel.as_u32(inst.operand(3), "iy"),
                         isel.as_u32(inst.operand(4), "iz")});
    }

    Id v4f32 = m.t_vector(isel.t_f32(), 4);
    Id texel = isel.op(spv::OpImageRead, v4f32, {img, coord});
    if constexpr (Vec4) {
        isel.define(inst.result(), texel);
    } else {
        Id r = m.new_id();
        isel.emit(Inst(spv::OpCompositeExtract, isel.t_f32(), r).id(texel).lit(0));
        isel.define(inst.result(), r);
    }
}
template void SpirvISel::Intrinsics::image_load<spv::Dim2D, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::image_load<spv::Dim3D, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::image_load<spv::Dim2D, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::image_load<spv::Dim3D, true>(SpirvISel&, const brass::Instruction&);

template <spv::Dim Dim, bool Vec4>
void SpirvISel::Intrinsics::image_store(SpirvISel& isel, const brass::Instruction& inst) {
    constexpr size_t val_idx = (Dim == spv::Dim3D) ? 5 : 4;
    if (inst.operand_count() <= val_idx) isel.malformed(inst, "missing image_store operand");
    uint32_t set = 0, binding = 0;
    parse_set_binding(isel, inst, set, binding);

    Module& m = isel.m_;
    auto kind = (Dim == spv::Dim3D) ? target::SpirvDescriptorKind::StorageImage3D
                                    : target::SpirvDescriptorKind::StorageImage2D;
    Id existing = m.cached("desc." + std::to_string(set) + "." + std::to_string(binding));
    Id img_t = storage_image_handle_type(m, Dim, Vec4, existing);
    Id var = isel.descriptor_var(set, binding, kind, img_t);
    Id img = isel.op(spv::OpLoad, img_t, {var});

    Id u32 = isel.t_u32();
    Id coord = 0;
    if constexpr (Dim == spv::Dim2D) {
        coord = isel.op(spv::OpCompositeConstruct, m.t_vector(u32, 2),
                        {isel.as_u32(inst.operand(2), "ix"), isel.as_u32(inst.operand(3), "iy")});
    } else {
        coord = isel.op(spv::OpCompositeConstruct, m.t_vector(u32, 3),
                        {isel.as_u32(inst.operand(2), "ix"),
                         isel.as_u32(inst.operand(3), "iy"),
                         isel.as_u32(inst.operand(4), "iz")});
    }

    Id v4f32 = m.t_vector(isel.t_f32(), 4);
    Id texel = 0;
    if constexpr (Vec4) {
        texel = isel.id_of(inst.operand(val_idx), "texel");
    } else {
        Id zero = m.c_f32(0.0f);
        texel = isel.op(spv::OpCompositeConstruct, v4f32,
                        {isel.id_of(inst.operand(val_idx), "value"), zero, zero, zero});
    }
    isel.emit(Inst(spv::OpImageWrite).id(img).id(coord).id(texel));
}
template void SpirvISel::Intrinsics::image_store<spv::Dim2D, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::image_store<spv::Dim3D, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::image_store<spv::Dim2D, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::image_store<spv::Dim3D, true>(SpirvISel&, const brass::Instruction&);

// ---------------------------------------------------------------------------
// 2D and 3D Sampled Textures (OpTypeSampledImage, OpImageSampleExplicitLod)
// ---------------------------------------------------------------------------

template <spv::Dim Dim, bool Vec4>
void SpirvISel::Intrinsics::sampler_lod(SpirvISel& isel, const brass::Instruction& inst) {
    constexpr size_t lod_idx = (Dim == spv::Dim3D) ? 5 : 4;
    if (inst.operand_count() <= lod_idx) isel.malformed(inst, "missing sampler_lod operand");
    uint32_t set = 0, binding = 0;
    parse_set_binding(isel, inst, set, binding);

    Module& m = isel.m_;
    Id f32 = isel.t_f32();
    Id img_t = m.t_image(f32, Dim, 0, 0, 0, 1, spv::ImageFormatUnknown);
    Id sampled_img_t = m.t_sampled_image(img_t);
    auto kind = (Dim == spv::Dim3D) ? target::SpirvDescriptorKind::SampledImage3D
                                    : target::SpirvDescriptorKind::SampledImage2D;
    Id var = isel.descriptor_var(set, binding, kind, sampled_img_t);
    Id sampled_img = isel.op(spv::OpLoad, sampled_img_t, {var});

    Id coord = 0;
    if constexpr (Dim == spv::Dim2D) {
        coord = isel.op(spv::OpCompositeConstruct, m.t_vector(f32, 2),
                        {isel.id_of(inst.operand(2), "u"), isel.id_of(inst.operand(3), "v")});
    } else {
        coord = isel.op(spv::OpCompositeConstruct, m.t_vector(f32, 3),
                        {isel.id_of(inst.operand(2), "u"),
                         isel.id_of(inst.operand(3), "v"),
                         isel.id_of(inst.operand(4), "w")});
    }
    Id lod = isel.id_of(inst.operand(lod_idx), "lod");

    Id v4f32 = m.t_vector(f32, 4);
    Id texel = m.new_id();
    isel.emit(Inst(spv::OpImageSampleExplicitLod, v4f32, texel)
                  .id(sampled_img)
                  .id(coord)
                  .lit(static_cast<uint32_t>(spv::ImageOperandsLodMask))
                  .id(lod));

    if constexpr (Vec4) {
        isel.define(inst.result(), texel);
    } else {
        Id r = m.new_id();
        isel.emit(Inst(spv::OpCompositeExtract, f32, r).id(texel).lit(0));
        isel.define(inst.result(), r);
    }
}
template void SpirvISel::Intrinsics::sampler_lod<spv::Dim2D, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::sampler_lod<spv::Dim3D, false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::sampler_lod<spv::Dim2D, true>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::sampler_lod<spv::Dim3D, true>(SpirvISel&, const brass::Instruction&);

// ---------------------------------------------------------------------------
// Graphics Stage Inputs & Outputs (Location-decorated and BuiltIn::Position)
// ---------------------------------------------------------------------------

void SpirvISel::Intrinsics::stage_input_f32(SpirvISel& isel, const brass::Instruction& inst) {
    int64_t loc = 0, comp = 0;
    if (!isel.const_int(inst.operand(0), &loc) || loc < 0) {
        isel.fail("spirv_stage_input_f32 requires a non-negative constant location");
    }
    if (!isel.const_int(inst.operand(1), &comp) || comp < 0 || comp >= 4) {
        isel.fail("spirv_stage_input_f32 requires a constant component in 0..3");
    }
    auto uloc = static_cast<uint32_t>(loc);
    auto ucomp = static_cast<uint32_t>(comp);
    Id var = isel.stage_io_var(spv::StorageClassInput, uloc);
    Id v4 = isel.m_.t_vector(isel.t_f32(), 4);
    std::string key = "stage_in." + std::to_string(uloc);
    Id vec = isel.prologue_value(key, [&] { return isel.op(spv::OpLoad, v4, {var}); });
    Id val = isel.prologue_value(key + "." + std::to_string(ucomp), [&] {
        Id r = isel.m_.new_id();
        isel.emit(Inst(spv::OpCompositeExtract, isel.t_f32(), r).id(vec).lit(ucomp));
        return r;
    });
    isel.define(inst.result(), val);
}

void SpirvISel::Intrinsics::stage_input_vec4(SpirvISel& isel, const brass::Instruction& inst) {
    int64_t loc = 0;
    if (!isel.const_int(inst.operand(0), &loc) || loc < 0) {
        isel.fail("spirv_stage_input_vec4 requires a non-negative constant location");
    }
    auto uloc = static_cast<uint32_t>(loc);
    Id var = isel.stage_io_var(spv::StorageClassInput, uloc);
    Id v4 = isel.m_.t_vector(isel.t_f32(), 4);
    std::string key = "stage_in." + std::to_string(uloc);
    Id vec = isel.prologue_value(key, [&] { return isel.op(spv::OpLoad, v4, {var}); });
    isel.define(inst.result(), vec);
}

void SpirvISel::Intrinsics::stage_output_f32(SpirvISel& isel, const brass::Instruction& inst) {
    int64_t loc = 0, comp = 0;
    if (!isel.const_int(inst.operand(0), &loc) || loc < 0) {
        isel.fail("spirv_stage_output_f32 requires a non-negative constant location");
    }
    if (!isel.const_int(inst.operand(1), &comp) || comp < 0 || comp >= 4) {
        isel.fail("spirv_stage_output_f32 requires a constant component in 0..3");
    }
    auto uloc = static_cast<uint32_t>(loc);
    auto ucomp = static_cast<uint32_t>(comp);
    Id var = isel.stage_io_var(spv::StorageClassOutput, uloc);
    Id ptr_f32 = isel.m_.t_pointer(spv::StorageClassOutput, isel.t_f32());
    Id elem_ptr = isel.op(spv::OpAccessChain, ptr_f32, {var, isel.m_.c_u32(ucomp)});
    isel.emit(Inst(spv::OpStore).id(elem_ptr).id(isel.id_of(inst.operand(2), "stage output value")));
}

void SpirvISel::Intrinsics::stage_output_vec4(SpirvISel& isel, const brass::Instruction& inst) {
    int64_t loc = 0;
    if (!isel.const_int(inst.operand(0), &loc) || loc < 0) {
        isel.fail("spirv_stage_output_vec4 requires a non-negative constant location");
    }
    auto uloc = static_cast<uint32_t>(loc);
    Id var = isel.stage_io_var(spv::StorageClassOutput, uloc);
    Id v4 = isel.m_.t_vector(isel.t_f32(), 4);
    Id vec = 0;
    if (inst.operand_count() == 2) {
        vec = isel.id_of(inst.operand(1), "stage output vec4");
    } else if (inst.operand_count() == 5) {
        vec = isel.op(spv::OpCompositeConstruct, v4,
                      {isel.id_of(inst.operand(1), "x"),
                       isel.id_of(inst.operand(2), "y"),
                       isel.id_of(inst.operand(3), "z"),
                       isel.id_of(inst.operand(4), "w")});
    } else {
        isel.malformed(inst, "spirv_stage_output_vec4 expects (loc, vec4) or (loc, x, y, z, w)");
    }
    isel.emit(Inst(spv::OpStore).id(var).id(vec));
}

void SpirvISel::Intrinsics::set_vertex_position(SpirvISel& isel, const brass::Instruction& inst) {
    Module& m = isel.m_;
    Id v4 = m.t_vector(isel.t_f32(), 4);
    Id& var = m.cached("builtin.position");
    if (!var) {
        var = m.variable(m.t_pointer(spv::StorageClassOutput, v4), spv::StorageClassOutput);
        m.decorate(var, spv::DecorationBuiltIn, {static_cast<uint32_t>(spv::BuiltInPosition)});
        m.set_name(var, "gl_Position");
    }
    Id vid = var;
    isel.use_global(vid);

    Id vec = 0;
    if (inst.operand_count() == 1) {
        vec = isel.id_of(inst.operand(0), "position vec4");
    } else if (inst.operand_count() == 4) {
        vec = isel.op(spv::OpCompositeConstruct, v4,
                      {isel.id_of(inst.operand(0), "x"),
                       isel.id_of(inst.operand(1), "y"),
                       isel.id_of(inst.operand(2), "z"),
                       isel.id_of(inst.operand(3), "w")});
    } else {
        isel.malformed(inst, "spirv_set_vertex_position expects (vec4) or (x, y, z, w)");
    }
    isel.emit(Inst(spv::OpStore).id(vid).id(vec));
}

// ---------------------------------------------------------------------------
// Fragment-only: screen-space derivatives and facing
// ---------------------------------------------------------------------------

void SpirvISel::Intrinsics::require_fragment(SpirvISel& isel, const brass::Instruction& inst) {
    if (isel.opts_.stage != target::SpirvStage::Fragment) {
        isel.fail(std::string(inst.symbol()) + " is only valid in a fragment stage");
    }
}

template <spv::Op O>
void SpirvISel::Intrinsics::derivative(SpirvISel& isel, const brass::Instruction& inst) {
    require_fragment(isel, inst);
    const Value* a = inst.operand(0);
    if (!a || a->type() != Type::f32()) isel.malformed(inst, "argument 0 must be f32");
    isel.define(inst.result(), isel.op(O, isel.t_f32(), {isel.id_of(a, "argument 0")}));
}
template void SpirvISel::Intrinsics::derivative<spv::OpDPdx>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::derivative<spv::OpDPdy>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::derivative<spv::OpFwidth>(SpirvISel&, const brass::Instruction&);

// Discard as OpDemoteToHelperInvocation: the invocation's outputs are dropped
// but it keeps running as a helper, so the call is not a terminator and the
// structured control flow around it stays as written (core in SPIR-V 1.6,
// SPV_EXT_demote_to_helper_invocation before).
void SpirvISel::Intrinsics::demote_to_helper(SpirvISel& isel, const brass::Instruction& inst) {
    require_fragment(isel, inst);
    isel.m_.add_capability(spv::CapabilityDemoteToHelperInvocation);
    if (isel.opts_.spirv_version < 0x00010600) isel.m_.add_extension("SPV_EXT_demote_to_helper_invocation");
    isel.emit(Inst(spv::OpDemoteToHelperInvocation));
}

// BuiltIn FrontFacing is a bool input; the call yields i32 1 (front) or 0.
void SpirvISel::Intrinsics::front_facing(SpirvISel& isel, const brass::Instruction& inst) {
    require_fragment(isel, inst);
    Module& m = isel.m_;
    const std::string key = "builtin.front_facing";
    Id& var = m.cached(key);
    if (!var) {
        var = m.variable(m.t_pointer(spv::StorageClassInput, isel.t_bool()), spv::StorageClassInput);
        m.decorate(var, spv::DecorationBuiltIn, {static_cast<uint32_t>(spv::BuiltInFrontFacing)});
    }
    Id vid = var;
    isel.use_global(vid);
    isel.define(inst.result(), isel.prologue_value(key, [&] {
        Id front = isel.op(spv::OpLoad, isel.t_bool(), {vid});
        return isel.op(spv::OpSelect, isel.t_u32(), {front, m.c_u32(1), m.c_u32(0)});
    }));
}

} // namespace brass::spirv
