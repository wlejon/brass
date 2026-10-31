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

Id SpirvISel::descriptor_var(const target::SpirvDescriptorBinding& db, Id handle_type, spv::StorageClass sc) {
    std::string key = "desc." + std::to_string(db.set) + "." + std::to_string(db.binding);
    Id& var = m_.cached(key);
    if (!var) {
        Id ptr_t = m_.t_pointer(sc, handle_type);
        var = m_.variable(ptr_t, sc);
        m_.decorate(var, spv::DecorationDescriptorSet, {db.set});
        m_.decorate(var, spv::DecorationBinding, {db.binding});
        m_.set_name(var, "desc_s" + std::to_string(db.set) + "_b" + std::to_string(db.binding));
        descriptors_.push_back(db);
    } else {
        auto it = std::find_if(descriptors_.begin(), descriptors_.end(), [&](const target::SpirvDescriptorBinding& d) {
            return d.set == db.set && d.binding == db.binding;
        });
        if (it == descriptors_.end() || !(*it == db)) {
            fail("descriptor (set=" + std::to_string(db.set) + ", binding=" + std::to_string(db.binding) +
                 ") used as two different descriptors");
        }
    }
    Id vid = var;
    use_global(vid);
    return vid;
}

Id SpirvISel::stage_io_var(spv::StorageClass sc, uint32_t location, Id type, uint32_t interp) {
    std::string key = std::string(sc == spv::StorageClassInput ? "stage_in." : "stage_out.") +
                      std::to_string(location);
    Id& var = m_.cached(key);
    Id& var_type = m_.cached(key + ".type");
    if (!var) {
        var = m_.variable(m_.t_pointer(sc, type), sc);
        var_type = type;
        m_.decorate(var, spv::DecorationLocation, {location});
        // Interpolation is the fragment input's: a vertex stage's outputs and
        // inputs (attributes) carry none.
        if (sc == spv::StorageClassInput && opts_.stage == target::SpirvStage::Fragment) {
            if (interp & target::kSpirvStageFlat) m_.decorate(var, spv::DecorationFlat);
            else if (interp & target::kSpirvStageNoPerspective) m_.decorate(var, spv::DecorationNoPerspective);
        }
        m_.set_name(var, (sc == spv::StorageClassInput ? "in_loc" : "out_loc") + std::to_string(location));
    } else if (var_type != type) {
        fail(std::string(sc == spv::StorageClassInput ? "stage input" : "stage output") + " at location " +
             std::to_string(location) + " used as two different types");
    }
    Id vid = var;
    use_global(vid);
    return vid;
}

// ---------------------------------------------------------------------------
// Textures and storage images (OpTypeImage, OpTypeSampledImage, OpTypeSampler)
// ---------------------------------------------------------------------------

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

// A texture or image call's operands open with the image's set and binding,
// its description (KernelBuilder::ImageRef, packed by image_desc: dim in bits
// 0-1, arrayed bit 2, depth bit 3, scalar bits 4-5, a storage image's
// spv::ImageFormat in bits 8-15), and a separate sampler's set and binding
// (-1 for a combined image sampler, -2 for a sampled image read without one:
// a fetch or a query); the call's own operands follow from 5.
SpirvISel::Intrinsics::ImageCall SpirvISel::Intrinsics::image_call(SpirvISel& isel, const brass::Instruction& inst,
                                                                   bool storage, size_t own_operands) {
    if (inst.operand_count() < 5 + own_operands) isel.malformed(inst, "missing image operand");
    ImageCall c;
    parse_set_binding(isel, inst, c.set, c.binding);
    int64_t desc = 0, sset = 0, sbind = 0;
    if (!isel.const_int(inst.operand(2), &desc) || !isel.const_int(inst.operand(3), &sset) ||
        !isel.const_int(inst.operand(4), &sbind)) {
        isel.fail(std::string(inst.symbol()) + " requires a constant image description and sampler binding");
    }
    c.dim = static_cast<target::SpirvImageDim>(desc & 3);
    c.arrayed = (desc & 4) != 0;
    c.depth = (desc & 8) != 0;
    c.scalar = static_cast<target::SpirvImageScalar>((desc >> 4) & 3);
    c.format = static_cast<uint32_t>((desc >> 8) & 0xff);
    if (c.scalar == target::SpirvImageScalar::I32) isel.fail(std::string(inst.symbol()) + ": i32 images are not supported");
    if (storage && c.format == 0) isel.fail(std::string(inst.symbol()) + " requires a storage image format");
    if (storage && (c.depth || c.dim == target::SpirvImageDim::Cube)) {
        isel.fail(std::string(inst.symbol()) + ": a storage image is neither a depth image nor a cube");
    }

    Module& m = isel.m_;
    static constexpr spv::Dim kDims[] = {spv::Dim1D, spv::Dim2D, spv::Dim3D, spv::DimCube};
    spv::Dim dim = kDims[static_cast<int>(c.dim)];
    if (c.dim == target::SpirvImageDim::D1) m.add_capability(storage ? spv::CapabilityImage1D : spv::CapabilitySampled1D);
    if (c.dim == target::SpirvImageDim::Cube && c.arrayed) m.add_capability(spv::CapabilitySampledCubeArray);
    if (storage) {
        switch (static_cast<spv::ImageFormat>(c.format)) {
        case spv::ImageFormatRgba32f: case spv::ImageFormatRgba16f: case spv::ImageFormatR32f:
        case spv::ImageFormatRgba8: case spv::ImageFormatRgba8Snorm: case spv::ImageFormatRgba32ui:
        case spv::ImageFormatRgba16ui: case spv::ImageFormatRgba8ui: case spv::ImageFormatR32ui:
            break;
        default:
            m.add_capability(spv::CapabilityStorageImageExtendedFormats);
        }
    }
    c.texel_scalar = c.scalar == target::SpirvImageScalar::F32 ? isel.t_f32() : isel.t_u32();
    c.texel_type = m.t_vector(c.texel_scalar, 4);
    c.image_type = m.t_image(c.texel_scalar, dim, c.depth ? 1 : 0, c.arrayed ? 1 : 0, 0, storage ? 2 : 1,
                             static_cast<spv::ImageFormat>(storage ? c.format : 0));

    target::SpirvDescriptorBinding db;
    db.set = c.set;
    db.binding = c.binding;
    db.dim = c.dim;
    db.arrayed = c.arrayed;
    db.depth = c.depth;
    db.scalar = c.scalar;
    db.format = storage ? c.format : 0;
    if (storage) {
        db.kind = target::SpirvDescriptorKind::StorageImage;
        c.image = isel.op(spv::OpLoad, c.image_type, {isel.descriptor_var(db, c.image_type)});
        return c;
    }
    Id sampled_t = m.t_sampled_image(c.image_type);
    if (sset == -1) {
        db.kind = target::SpirvDescriptorKind::CombinedImageSampler;
        c.sampled = isel.op(spv::OpLoad, sampled_t, {isel.descriptor_var(db, sampled_t)});
        c.image = isel.op(spv::OpImage, c.image_type, {c.sampled});
        return c;
    }
    db.kind = target::SpirvDescriptorKind::SampledImage;
    c.image = isel.op(spv::OpLoad, c.image_type, {isel.descriptor_var(db, c.image_type)});
    if (sset < 0) return c;
    target::SpirvDescriptorBinding sb;
    sb.set = static_cast<uint32_t>(sset);
    sb.binding = static_cast<uint32_t>(sbind);
    sb.kind = target::SpirvDescriptorKind::Sampler;
    Id sampler_t = m.t_sampler();
    Id sampler = isel.op(spv::OpLoad, sampler_t, {isel.descriptor_var(sb, sampler_t)});
    c.sampled = isel.op(spv::OpSampledImage, sampled_t, {c.image, sampler});
    return c;
}

// The image's coordinate count: its dimensions, a cube's direction three, and
// the layer after them when arrayed.
uint32_t SpirvISel::Intrinsics::ImageCall::coords() const {
    uint32_t n = dim == target::SpirvImageDim::D1 ? 1 : (dim == target::SpirvImageDim::D2 ? 2 : 3);
    return n + (arrayed ? 1 : 0);
}

// Operands [first, first + n) as one coordinate: a scalar or a vector of f32,
// or of u32 (`integer`: i32 / i64 operands, truncated).
Id SpirvISel::Intrinsics::image_coord(SpirvISel& isel, const brass::Instruction& inst, size_t first, uint32_t n,
                                      bool integer) {
    std::vector<Id> parts;
    for (uint32_t i = 0; i < n; ++i) {
        parts.push_back(integer ? isel.as_u32(inst.operand(first + i), "coordinate")
                                : isel.id_of(inst.operand(first + i), "coordinate"));
    }
    if (n == 1) return parts[0];
    return isel.op(spv::OpCompositeConstruct, isel.m_.t_vector(integer ? isel.t_u32() : isel.t_f32(), n), parts);
}

// Constant operands [first, first + n) as a constant integer vector (a
// texel offset).
Id SpirvISel::Intrinsics::image_const_offset(SpirvISel& isel, const brass::Instruction& inst, size_t first, uint32_t n) {
    std::vector<Id> parts;
    for (uint32_t i = 0; i < n; ++i) {
        int64_t v = 0;
        if (!isel.const_int(inst.operand(first + i), &v)) isel.fail(std::string(inst.symbol()) + " requires constant offsets");
        parts.push_back(isel.m_.c_int(32, static_cast<uint32_t>(static_cast<int32_t>(v))));
    }
    if (n == 1) return parts[0];
    return isel.m_.c_composite(isel.m_.t_vector(isel.t_u32(), n), parts);
}

// spirv_tex_sample(img, coords..., lod, grad, offset):
//   Op 0 implicit LOD (fragment), 1 explicit `lod`, 2 gradients (dx then dy, a
//   value each of the image's dimensions), each optionally with a constant
//   offset (OffsetBit) per dimension at the end. A depth image's call takes
//   its reference after the coordinates and gives an f32 (DrefBit).
template <int Op>
void SpirvISel::Intrinsics::tex_sample(SpirvISel& isel, const brass::Instruction& inst) {
    constexpr int kind = Op & 3;
    constexpr bool offset = (Op & kOffsetBit) != 0;
    constexpr bool dref = (Op & kDrefBit) != 0;
    if constexpr (kind == 0) require_fragment(isel, inst);
    ImageCall c = image_call(isel, inst, false, 0);
    if (!c.sampled) isel.fail(std::string(inst.symbol()) + ": a separate image is sampled through a sampler");
    if (dref != c.depth) isel.fail(std::string(inst.symbol()) + ": a reference value is given for, and only for, a depth image");
    uint32_t n = c.coords();
    uint32_t space = n - (c.arrayed ? 1 : 0);   // a gradient's and an offset's components
    size_t at = 5;
    Id coord = image_coord(isel, inst, at, n, false);
    at += n;
    Id ref = 0;
    if (dref) ref = isel.id_of(inst.operand(at++), "reference");
    uint32_t mask = 0;
    std::vector<Id> extra;
    if constexpr (kind == 1) {
        mask |= spv::ImageOperandsLodMask;
        extra.push_back(isel.id_of(inst.operand(at++), "lod"));
    } else if constexpr (kind == 2) {
        mask |= spv::ImageOperandsGradMask;
        extra.push_back(image_coord(isel, inst, at, space, false));
        at += space;
        extra.push_back(image_coord(isel, inst, at, space, false));
        at += space;
    }
    if (offset) {
        if (c.dim == target::SpirvImageDim::Cube) isel.fail(std::string(inst.symbol()) + ": a cube takes no offset");
        mask |= spv::ImageOperandsConstOffsetMask;
        extra.push_back(image_const_offset(isel, inst, at, space));
        at += space;
    }
    if (inst.operand_count() != at) isel.malformed(inst, "wrong operand count for the image's shape");
    spv::Op op = dref ? (kind == 0 ? spv::OpImageSampleDrefImplicitLod : spv::OpImageSampleDrefExplicitLod)
                      : (kind == 0 ? spv::OpImageSampleImplicitLod : spv::OpImageSampleExplicitLod);
    Id r = isel.m_.new_id();
    Inst in = Inst(op, dref ? c.texel_scalar : c.texel_type, r).id(c.sampled).id(coord);
    if (dref) in.id(ref);
    if (mask) {
        in.lit(mask);
        for (Id e : extra) in.id(e);
    }
    isel.emit(std::move(in));
    isel.define(inst.result(), r);
}
template void SpirvISel::Intrinsics::tex_sample<0>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::tex_sample<1>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::tex_sample<2>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::tex_sample<0 | SpirvISel::Intrinsics::kOffsetBit>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::tex_sample<1 | SpirvISel::Intrinsics::kOffsetBit>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::tex_sample<2 | SpirvISel::Intrinsics::kOffsetBit>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::tex_sample<0 | SpirvISel::Intrinsics::kDrefBit>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::tex_sample<1 | SpirvISel::Intrinsics::kDrefBit>(SpirvISel&, const brass::Instruction&);

// spirv_tex_fetch(img, icoords..., lod): one texel, unfiltered.
void SpirvISel::Intrinsics::tex_fetch(SpirvISel& isel, const brass::Instruction& inst) {
    ImageCall c = image_call(isel, inst, false, 0);
    if (c.dim == target::SpirvImageDim::Cube) isel.fail(std::string(inst.symbol()) + ": a cube is not fetched");
    uint32_t n = c.coords();
    if (inst.operand_count() != 5 + n + 1) isel.malformed(inst, "wrong operand count for the image's shape");
    Id coord = image_coord(isel, inst, 5, n, true);
    Id lod = isel.as_u32(inst.operand(5 + n), "lod");
    Id r = isel.m_.new_id();
    isel.emit(Inst(spv::OpImageFetch, c.texel_type, r).id(c.image).id(coord)
                  .lit(static_cast<uint32_t>(spv::ImageOperandsLodMask)).id(lod));
    isel.define(inst.result(), r);
}

// spirv_tex_gather(img, coords..., component): the component of the four
// texels a bilinear sample would blend (a depth image's: its reference
// after the coordinates, compared, in place of the component).
void SpirvISel::Intrinsics::tex_gather(SpirvISel& isel, const brass::Instruction& inst) {
    ImageCall c = image_call(isel, inst, false, 0);
    if (!c.sampled) isel.fail(std::string(inst.symbol()) + ": a separate image is sampled through a sampler");
    if (c.dim == target::SpirvImageDim::D1 || c.dim == target::SpirvImageDim::D3) {
        isel.fail(std::string(inst.symbol()) + ": only 2D and cube images gather");
    }
    uint32_t n = c.coords();
    if (inst.operand_count() != 5 + n + 1) isel.malformed(inst, "wrong operand count for the image's shape");
    Id coord = image_coord(isel, inst, 5, n, false);
    Id r = isel.m_.new_id();
    if (c.depth) {
        isel.emit(Inst(spv::OpImageDrefGather, c.texel_type, r).id(c.sampled).id(coord)
                      .id(isel.id_of(inst.operand(5 + n), "reference")));
    } else {
        int64_t comp = 0;
        if (!isel.const_int(inst.operand(5 + n), &comp) || comp < 0 || comp > 3) {
            isel.fail(std::string(inst.symbol()) + " requires a constant component in 0..3");
        }
        isel.emit(Inst(spv::OpImageGather, c.texel_type, r).id(c.sampled).id(coord)
                      .id(isel.m_.c_u32(static_cast<uint32_t>(comp))));
    }
    isel.define(inst.result(), r);
}

// spirv_tex_size(img, lod) / spirv_img_size(img): the image's size at a mip
// level (a storage image's has one), as an i32x4 (lanes past its coordinate
// count 0); spirv_tex_levels(img): its mip count.
template <bool Storage>
void SpirvISel::Intrinsics::tex_size(SpirvISel& isel, const brass::Instruction& inst) {
    ImageCall c = image_call(isel, inst, Storage, 0);
    isel.m_.add_capability(spv::CapabilityImageQuery);
    uint32_t n = c.dim == target::SpirvImageDim::Cube ? 2 + (c.arrayed ? 1 : 0) : c.coords();
    Id u32 = isel.t_u32();
    Id size_t_ = n == 1 ? u32 : isel.m_.t_vector(u32, n);
    Id size = 0;
    if (Storage) {
        if (inst.operand_count() != 5) isel.malformed(inst, "spirv_img_size takes the image alone");
        size = isel.op(spv::OpImageQuerySize, size_t_, {c.image});
    } else {
        if (inst.operand_count() != 6) isel.malformed(inst, "spirv_tex_size takes the image and a lod");
        size = isel.op(spv::OpImageQuerySizeLod, size_t_, {c.image, isel.as_u32(inst.operand(5), "lod")});
    }
    std::vector<Id> lanes;
    for (uint32_t i = 0; i < 4; ++i) {
        if (i >= n) {
            lanes.push_back(isel.m_.c_u32(0));
        } else if (n == 1) {
            lanes.push_back(size);
        } else {
            Id l = isel.m_.new_id();
            isel.emit(Inst(spv::OpCompositeExtract, u32, l).id(size).lit(i));
            lanes.push_back(l);
        }
    }
    isel.define(inst.result(), isel.op(spv::OpCompositeConstruct, isel.m_.t_vector(u32, 4), lanes));
}
template void SpirvISel::Intrinsics::tex_size<false>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::tex_size<true>(SpirvISel&, const brass::Instruction&);

void SpirvISel::Intrinsics::tex_levels(SpirvISel& isel, const brass::Instruction& inst) {
    ImageCall c = image_call(isel, inst, false, 0);
    isel.m_.add_capability(spv::CapabilityImageQuery);
    if (inst.operand_count() != 5) isel.malformed(inst, "spirv_tex_levels takes the image alone");
    isel.define(inst.result(), isel.op(spv::OpImageQueryLevels, isel.t_u32(), {c.image}));
}

// spirv_img_load(img, icoords...) / spirv_img_store(img, icoords..., texel):
// a storage image's texel, as four lanes of its scalar.
void SpirvISel::Intrinsics::img_load(SpirvISel& isel, const brass::Instruction& inst) {
    ImageCall c = image_call(isel, inst, true, 0);
    uint32_t n = c.coords();
    if (inst.operand_count() != 5 + n) isel.malformed(inst, "wrong operand count for the image's shape");
    isel.define(inst.result(), isel.op(spv::OpImageRead, c.texel_type, {c.image, image_coord(isel, inst, 5, n, true)}));
}

void SpirvISel::Intrinsics::img_store(SpirvISel& isel, const brass::Instruction& inst) {
    ImageCall c = image_call(isel, inst, true, 0);
    uint32_t n = c.coords();
    if (inst.operand_count() != 5 + n + 1) isel.malformed(inst, "wrong operand count for the image's shape");
    Id coord = image_coord(isel, inst, 5, n, true);
    isel.emit(Inst(spv::OpImageWrite).id(c.image).id(coord).id(isel.id_of(inst.operand(5 + n), "texel")));
}

// ---------------------------------------------------------------------------
// Buffers at descriptors: a 32-bit word at a byte offset
// ---------------------------------------------------------------------------
// spirv_buffer_load(set, binding, desc, offset) -> f32 | i32 and
// spirv_buffer_store(set, binding, desc, offset, value). desc's bit 0 makes it
// a storage buffer (otherwise uniform) and its bits 1 up a uniform block's
// size in 16-byte units. The block is words a layout above lays its data on
// (std140 for a uniform buffer, std430 for storage): a uniform buffer is
// uvec4 data[size / 16], a storage buffer uint data[], and the offset (a
// multiple of 4) picks a word, which an f32 reads as its bits.

Id SpirvISel::Intrinsics::buffer_word(SpirvISel& isel, const brass::Instruction& inst, bool store) {
    bool storage = false, vec4_rows = false;
    Id var = buffer_var(isel, inst, store, storage, vec4_rows);
    Module& m = isel.m_;
    const spv::StorageClass sc = storage ? spv::StorageClassStorageBuffer : spv::StorageClassUniform;
    Id u32 = isel.t_u32();
    Id ptr = m.t_pointer(sc, u32);
    int64_t const_off = 0;
    const bool is_const = isel.const_int(inst.operand(3), &const_off) && const_off >= 0;
    if (!vec4_rows) {
        Id word = is_const ? m.c_u32(static_cast<uint32_t>(const_off) >> 2)
                           : isel.op(spv::OpShiftRightLogical, u32,
                                     {isel.as_u32(inst.operand(3), "byte offset"), m.c_u32(2)});
        return isel.op(spv::OpAccessChain, ptr, {var, m.c_u32(0), word});
    }
    Id row = 0, lane = 0;
    if (is_const) {
        row = m.c_u32(static_cast<uint32_t>(const_off) >> 4);
        lane = m.c_u32((static_cast<uint32_t>(const_off) >> 2) & 3);
    } else {
        Id offset = isel.as_u32(inst.operand(3), "byte offset");
        row = isel.op(spv::OpShiftRightLogical, u32, {offset, m.c_u32(4)});
        lane = isel.op(spv::OpBitwiseAnd, u32,
                       {isel.op(spv::OpShiftRightLogical, u32, {offset, m.c_u32(2)}), m.c_u32(3)});
    }
    return isel.op(spv::OpAccessChain, ptr, {var, m.c_u32(0), row, lane});
}

// spirv_buffer_length(set, binding, desc) -> i32: a storage buffer's words.
void SpirvISel::Intrinsics::buffer_length(SpirvISel& isel, const brass::Instruction& inst) {
    bool storage = false, vec4_rows = false;
    Id var = buffer_var(isel, inst, false, storage, vec4_rows);
    if (!storage) isel.fail(std::string(inst.symbol()) + ": a uniform buffer has a fixed size");
    Id r = isel.m_.new_id();
    isel.emit(Inst(spv::OpArrayLength, isel.t_u32(), r).id(var).lit(0));
    if (vec4_rows) r = isel.op(spv::OpShiftLeftLogical, isel.t_u32(), {r, isel.m_.c_u32(2)});
    isel.define(inst.result(), r);
}

// The block variable a buffer call's (set, binding, description) names.
Id SpirvISel::Intrinsics::buffer_var(SpirvISel& isel, const brass::Instruction& inst, bool store, bool& storage_out,
                                     bool& vec4_rows_out) {
    int64_t set = 0, binding = 0, desc = 0;
    if (!isel.const_int(inst.operand(0), &set) || !isel.const_int(inst.operand(1), &binding) ||
        !isel.const_int(inst.operand(2), &desc) || set < 0 || binding < 0) {
        isel.fail(std::string(inst.symbol()) + " requires a constant set, binding and description");
    }
    Module& m = isel.m_;
    const bool storage = (desc & 1) != 0;
    target::SpirvDescriptorBinding db;
    db.set = static_cast<uint32_t>(set);
    db.binding = static_cast<uint32_t>(binding);
    db.kind = storage ? target::SpirvDescriptorKind::StorageBuffer : target::SpirvDescriptorKind::UniformBuffer;
    db.bytes = storage ? 0 : static_cast<uint32_t>(desc >> 1) * 16;
    if (store && !storage) isel.fail(std::string(inst.symbol()) + ": a uniform buffer is read-only");
    if (!storage && db.bytes == 0) isel.fail(std::string(inst.symbol()) + ": a uniform block needs a size");
    const spv::StorageClass sc = storage ? spv::StorageClassStorageBuffer : spv::StorageClassUniform;
    Id u32 = isel.t_u32();
    std::string key = "buffer_block." + std::to_string(db.set) + "." + std::to_string(db.binding);
    Id& block = m.cached(key);
    Id& vec4_flag = m.cached(key + ".vec4");
    if (!block) {
        bool vec4_rows = !storage;
        if (storage && isel.mir_) {
            for (const BasicBlock* bb : isel.mir_->blocks()) {
                for (const Instruction* in = bb->head(); in && !vec4_rows; in = in->next()) {
                    if (in->opcode() == Opcode::call && in->symbol() == "spirv_buffer_load" &&
                        in->type().is_vector() && in->operand_count() >= 2) {
                        int64_t iset = -1, ibind = -1;
                        if (isel.const_int(in->operand(0), &iset) && isel.const_int(in->operand(1), &ibind) &&
                            iset == set && ibind == binding) {
                            vec4_rows = true;
                        }
                    }
                }
            }
        }
        vec4_flag = vec4_rows ? 1 : 0;
        Id arr = 0;
        if (storage && !vec4_rows) {
            arr = m.t_runtime_array(u32);
            m.decorate(arr, spv::DecorationArrayStride, {4});
        } else if (storage) {
            arr = m.t_runtime_array(m.t_vector(u32, 4));
            m.decorate(arr, spv::DecorationArrayStride, {16});
        } else {
            arr = m.t_array_fresh(m.t_vector(u32, 4), db.bytes / 16);
            m.decorate(arr, spv::DecorationArrayStride, {16});
        }
        block = m.t_struct({arr});
        m.decorate(block, spv::DecorationBlock);
        m.member_decorate(block, 0, spv::DecorationOffset, {0});
    }
    storage_out = storage;
    vec4_rows_out = (vec4_flag != 0);
    return isel.descriptor_var(db, block, sc);
}

void SpirvISel::Intrinsics::buffer_load(SpirvISel& isel, const brass::Instruction& inst) {
    if (inst.type() == Type::f32x4() || inst.type() == Type::i32x4()) {
        bool storage = false, vec4_rows = false;
        Id var = buffer_var(isel, inst, false, storage, vec4_rows);
        Module& m = isel.m_;
        const spv::StorageClass sc = storage ? spv::StorageClassStorageBuffer : spv::StorageClassUniform;
        Id u32 = isel.t_u32();
        Id uvec4 = m.t_vector(u32, 4);
        int64_t const_off = 0;
        Id row = (isel.const_int(inst.operand(3), &const_off) && const_off >= 0)
                     ? m.c_u32(static_cast<uint32_t>(const_off) >> 4)
                     : isel.op(spv::OpShiftRightLogical, u32,
                               {isel.as_u32(inst.operand(3), "byte offset"), m.c_u32(4)});
        Id ptr = isel.op(spv::OpAccessChain, m.t_pointer(sc, uvec4), {var, m.c_u32(0), row});
        Id vec = isel.op(spv::OpLoad, uvec4, {ptr});
        if (inst.type() == Type::f32x4()) vec = isel.op(spv::OpBitcast, m.t_vector(isel.t_f32(), 4), {vec});
        isel.define(inst.result(), vec);
        return;
    }
    Id ptr = buffer_word(isel, inst, false);
    Id word = isel.op(spv::OpLoad, isel.t_u32(), {ptr});
    if (inst.type() == Type::f32()) word = isel.op(spv::OpBitcast, isel.t_f32(), {word});
    else if (inst.type() != Type::i32()) isel.malformed(inst, "spirv_buffer_load gives an f32, i32, f32x4 or i32x4");
    isel.define(inst.result(), word);
}

void SpirvISel::Intrinsics::buffer_store(SpirvISel& isel, const brass::Instruction& inst) {
    Id ptr = buffer_word(isel, inst, true);
    const Value* v = inst.operand(4);
    if (!v || (v->type() != Type::f32() && v->type() != Type::i32())) {
        isel.malformed(inst, "spirv_buffer_store stores an f32 or an i32");
    }
    Id word = v->type() == Type::f32() ? isel.op(spv::OpBitcast, isel.t_u32(), {isel.id_of(v, "value")})
                                       : isel.as_u32(v, "value");
    isel.emit(Inst(spv::OpStore).id(ptr).id(word));
}

template <SpirvISel::Intrinsics::Atom A>
void SpirvISel::Intrinsics::buffer_atomic(SpirvISel& isel, const brass::Instruction& inst) {
    Id ptr = buffer_word(isel, inst, true);
    const Value* cmp = A == Atom::cas ? inst.operand(4) : nullptr;
    const Value* v = inst.operand(A == Atom::cas ? 5 : 4);
    if (!v || (A == Atom::cas && !cmp)) isel.malformed(inst, "missing buffer atomic argument");
    Module& m = isel.m_;
    Id u32 = isel.t_u32();
    Id scope = m.c_u32(spv::ScopeDevice);
    Id sem = m.c_u32(spv::MemorySemanticsMaskNone);
    if constexpr (A == Atom::cas) {
        Id r = isel.op(spv::OpAtomicCompareExchange, u32,
                       {ptr, scope, sem, sem, isel.as_u32(v, "value"), isel.as_u32(cmp, "comparator")});
        isel.define(inst.result(), r);
        return;
    }
    if constexpr (A == Atom::exch) {
        Id val = v->type() == Type::f32() ? isel.op(spv::OpBitcast, u32, {isel.id_of(v, "value")})
                                          : isel.as_u32(v, "value");
        Id r = isel.op(spv::OpAtomicExchange, u32, {ptr, scope, sem, val});
        if (inst.type() == Type::f32()) r = isel.op(spv::OpBitcast, isel.t_f32(), {r});
        isel.define(inst.result(), r);
        return;
    }
    spv::Op o = spv::OpAtomicIAdd;
    if constexpr (A == Atom::min) o = spv::OpAtomicSMin;
    else if constexpr (A == Atom::umin) o = spv::OpAtomicUMin;
    else if constexpr (A == Atom::max) o = spv::OpAtomicSMax;
    else if constexpr (A == Atom::umax) o = spv::OpAtomicUMax;
    else if constexpr (A == Atom::and_) o = spv::OpAtomicAnd;
    else if constexpr (A == Atom::or_) o = spv::OpAtomicOr;
    else if constexpr (A == Atom::xor_) o = spv::OpAtomicXor;
    Id r = isel.op(o, u32, {ptr, scope, sem, isel.as_u32(v, "value")});
    isel.define(inst.result(), r);
}
template void SpirvISel::Intrinsics::buffer_atomic<SpirvISel::Intrinsics::Atom::add>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::buffer_atomic<SpirvISel::Intrinsics::Atom::min>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::buffer_atomic<SpirvISel::Intrinsics::Atom::umin>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::buffer_atomic<SpirvISel::Intrinsics::Atom::max>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::buffer_atomic<SpirvISel::Intrinsics::Atom::umax>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::buffer_atomic<SpirvISel::Intrinsics::Atom::exch>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::buffer_atomic<SpirvISel::Intrinsics::Atom::and_>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::buffer_atomic<SpirvISel::Intrinsics::Atom::or_>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::buffer_atomic<SpirvISel::Intrinsics::Atom::xor_>(SpirvISel&, const brass::Instruction&);
template void SpirvISel::Intrinsics::buffer_atomic<SpirvISel::Intrinsics::Atom::cas>(SpirvISel&, const brass::Instruction&);

// ---------------------------------------------------------------------------
// Graphics Stage Inputs & Outputs (Location-decorated and BuiltIn::Position)
// ---------------------------------------------------------------------------

// A stage variable's (location, desc), desc as target::kSpirvStage*: its
// type is the scalar or the vector of its lanes.
SpirvISel::Intrinsics::StageVar SpirvISel::Intrinsics::stage_var(SpirvISel& isel, const brass::Instruction& inst) {
    int64_t loc = 0, desc = 0;
    if (!isel.const_int(inst.operand(0), &loc) || loc < 0 || !isel.const_int(inst.operand(1), &desc)) {
        isel.fail(std::string(inst.symbol()) + " requires a constant location and description");
    }
    StageVar v;
    v.location = static_cast<uint32_t>(loc);
    v.lanes = static_cast<uint32_t>(desc) & 0xf;
    v.interp = static_cast<uint32_t>(desc) & (target::kSpirvStageFlat | target::kSpirvStageNoPerspective);
    v.integer = (desc & target::kSpirvStageInt) != 0;
    if (v.lanes < 1 || v.lanes > 4) isel.fail(std::string(inst.symbol()) + ": a stage variable has 1 to 4 lanes");
    return v;
}

// spirv_stage_in(location, desc, lane): one lane of the stage input, the
// whole variable loaded once in the prologue. A fragment stage's integer
// input is flat, as Vulkan requires.
void SpirvISel::Intrinsics::stage_in(SpirvISel& isel, const brass::Instruction& inst) {
    StageVar v = stage_var(isel, inst);
    int64_t lane = 0;
    if (!isel.const_int(inst.operand(2), &lane) || lane < 0 || lane >= static_cast<int64_t>(v.lanes)) {
        isel.fail("spirv_stage_in requires a constant lane of the variable");
    }
    Id scalar = v.integer ? isel.t_u32() : isel.t_f32();
    Id type = v.lanes == 1 ? scalar : isel.m_.t_vector(scalar, v.lanes);
    if (v.integer) v.interp = target::kSpirvStageFlat;
    Id var = isel.stage_io_var(spv::StorageClassInput, v.location, type, v.interp);
    std::string key = "stage_in." + std::to_string(v.location);
    Id whole = isel.prologue_value(key, [&] { return isel.op(spv::OpLoad, type, {var}); });
    if (v.lanes == 1) {
        isel.define(inst.result(), whole);
        return;
    }
    auto ulane = static_cast<uint32_t>(lane);
    isel.define(inst.result(), isel.prologue_value(key + "." + std::to_string(ulane), [&] {
        Id r = isel.m_.new_id();
        isel.emit(Inst(spv::OpCompositeExtract, scalar, r).id(whole).lit(ulane));
        return r;
    }));
}

// spirv_stage_out(location, desc, lane values...): the whole variable stored.
void SpirvISel::Intrinsics::stage_out(SpirvISel& isel, const brass::Instruction& inst) {
    StageVar v = stage_var(isel, inst);
    if (inst.operand_count() != 2 + v.lanes) isel.malformed(inst, "spirv_stage_out expects one value per lane");
    Id scalar = v.integer ? isel.t_u32() : isel.t_f32();
    Id type = v.lanes == 1 ? scalar : isel.m_.t_vector(scalar, v.lanes);
    Id var = isel.stage_io_var(spv::StorageClassOutput, v.location, type, 0);
    std::vector<Id> parts;
    for (uint32_t i = 0; i < v.lanes; ++i) {
        parts.push_back(v.integer ? isel.as_u32(inst.operand(2 + i), "stage output lane")
                                  : isel.id_of(inst.operand(2 + i), "stage output lane"));
    }
    Id value = v.lanes == 1 ? parts[0] : isel.op(spv::OpCompositeConstruct, type, parts);
    isel.emit(Inst(spv::OpStore).id(var).id(value));
}

// spirv_set_frag_depth(depth): BuiltIn FragDepth, the entry DepthReplacing.
void SpirvISel::Intrinsics::set_frag_depth(SpirvISel& isel, const brass::Instruction& inst) {
    require_fragment(isel, inst);
    Module& m = isel.m_;
    Id& var = m.cached("builtin.frag_depth");
    if (!var) {
        var = m.variable(m.t_pointer(spv::StorageClassOutput, isel.t_f32()), spv::StorageClassOutput);
        m.decorate(var, spv::DecorationBuiltIn, {static_cast<uint32_t>(spv::BuiltInFragDepth)});
        m.set_name(var, "gl_FragDepth");
        m.add_execution_mode(isel.fn().id, spv::ExecutionModeDepthReplacing, {});
    }
    Id vid = var;
    isel.use_global(vid);
    isel.emit(Inst(spv::OpStore).id(vid).id(isel.id_of(inst.operand(0), "depth")));
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
